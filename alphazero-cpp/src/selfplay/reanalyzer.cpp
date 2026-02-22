#include "../../include/selfplay/reanalyzer.hpp"
#include "../../include/selfplay/chess_extensions.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace selfplay {

// ============================================================================
// Constructor / Destructor
// ============================================================================

Reanalyzer::Reanalyzer(training::ReplayBuffer& buffer, const ReanalysisConfig& config)
    : buffer_(buffer)
    , config_(config)
    , eval_queue_(config.gpu_batch_size, config.queue_capacity)
{
}

Reanalyzer::~Reanalyzer() {
    stop();
    wait();
}

// ============================================================================
// Lifecycle
// ============================================================================

void Reanalyzer::set_indices(const std::vector<size_t>& indices) {
    work_indices_ = indices;
}

void Reanalyzer::start() {
    // Guard against double-start: join any existing workers first
    wait();

    shutdown_.store(false, std::memory_order_release);
    work_cursor_.store(0, std::memory_order_release);
    eval_queue_.reset();
    stats_.reset();
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        last_error_.clear();
    }

    worker_threads_.reserve(config_.num_workers);
    for (int i = 0; i < config_.num_workers; ++i) {
        worker_threads_.emplace_back(&Reanalyzer::worker_func, this, i);
    }
}

void Reanalyzer::wait() {
    for (auto& t : worker_threads_) {
        if (t.joinable()) t.join();
    }
    worker_threads_.clear();
    // Signal GPU thread to stop pulling from this queue
    eval_queue_.shutdown();
}

void Reanalyzer::stop() {
    shutdown_.store(true, std::memory_order_release);
    eval_queue_.shutdown();
}

// ============================================================================
// Worker Function
// ============================================================================

void Reanalyzer::worker_func(int worker_id) {
    workers_active_.fetch_add(1, std::memory_order_relaxed);

    try {
        mcts::NodePool node_pool;

        // Pre-allocate buffers — sized for mcts_batch_size leaves per collect_leaves call.
        int bs = std::max(1, config_.mcts_batch_size);
        std::vector<float> obs_buffer(bs * encoding::PositionEncoder::TOTAL_SIZE);
        std::vector<float> mask_buffer(bs * encoding::MoveEncoder::POLICY_SIZE);
        std::vector<float> policy_buffer(bs * encoding::MoveEncoder::POLICY_SIZE);
        std::vector<float> value_buffer(bs);
        std::vector<float> wdl_buffer(bs * 3);

        while (!shutdown_.load(std::memory_order_acquire)) {
            // Atomically claim next work item
            size_t cursor = work_cursor_.fetch_add(1, std::memory_order_relaxed);
            if (cursor >= work_indices_.size()) break;

            size_t index = work_indices_[cursor];

            // Read FEN — skip if empty (FEN storage disabled or legacy sample)
            std::string fen = buffer_.get_fen(index);
            if (fen.empty()) {
                stats_.positions_skipped.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // Reconstruct board from FEN (ReanalysisBoard adds pushHistoryHash)
            selfplay::ReanalysisBoard board;
            board.setFen(fen);

            // Inject Zobrist history hashes for repetition detection
            // Stored [T-1, T-2, ..., T-8], inject oldest-first for chronological prev_states_
            const uint64_t* hashes = buffer_.get_history_hashes(index);
            uint8_t num_hist = buffer_.get_num_history(index);
            if (hashes && num_hist > 0) {
                for (int h = static_cast<int>(num_hist) - 1; h >= 0; --h) {
                    board.pushHistoryHash(hashes[h]);
                }
            }

            // Build position_history from stored history FENs.
            // Stored [T-1, T-2, ..., T-8] -> reverse to chronological (oldest first).
            std::vector<chess::Board> position_history;
            if (num_hist > 0) {
                position_history.reserve(num_hist);
                for (int h = static_cast<int>(num_hist) - 1; h >= 0; --h) {
                    std::string hfen = buffer_.get_history_fen(index, h);
                    if (!hfen.empty()) {
                        position_history.emplace_back(hfen);
                    }
                }
            }

            // Skip terminal positions (checkmate, stalemate, etc.)
            if (board.isGameOver().first != chess::GameResultReason::NONE) {
                stats_.positions_skipped.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // Read stored observation (full history encoding from original game)
            const float* stored_obs = buffer_.get_observation_ptr(index);
            if (!stored_obs) {
                stats_.positions_skipped.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            std::memcpy(obs_buffer.data(), stored_obs,
                        encoding::PositionEncoder::TOTAL_SIZE * sizeof(float));

            // Build legal move mask
            chess::Movelist legal_moves;
            chess::movegen::legalmoves(legal_moves, board);
            std::fill(mask_buffer.begin(), mask_buffer.end(), 0.0f);
            for (const auto& move : legal_moves) {
                int idx = encoding::MoveEncoder::move_to_index(move, board);
                if (idx >= 0 && idx < encoding::MoveEncoder::POLICY_SIZE) {
                    mask_buffer[idx] = 1.0f;
                }
            }

            // Submit for root evaluation
            std::vector<int32_t> request_ids;
            int submitted = eval_queue_.submit_for_evaluation(
                worker_id, obs_buffer.data(), mask_buffer.data(),
                1, request_ids, config_.worker_timeout_ms);

            if (submitted == 0) {
                stats_.positions_skipped.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // Retry loop: get_results returns 0 on timeout, retry until success or shutdown
            int got = 0;
            int timeout_count = 0;
            while (got == 0 && !shutdown_.load(std::memory_order_acquire)) {
                got = eval_queue_.get_results(
                    worker_id, policy_buffer.data(), value_buffer.data(),
                    1, wdl_buffer.data(),
                    config_.worker_timeout_ms);
                if (got == 0 && ++timeout_count % 5 == 0) {
                    fprintf(stderr, "[Reanalysis %d] waiting for NN results (%ds)\n",
                            worker_id, timeout_count * config_.worker_timeout_ms / 1000);
                }
            }

            if (got == 0) {
                // Only returns 0 on shutdown
                break;
            }

            stats_.total_nn_evals.fetch_add(1, std::memory_order_relaxed);

            // Reset node pool and create MCTS search (NO Dirichlet noise)
            node_pool.reset();

            mcts::BatchSearchConfig search_config;
            search_config.num_simulations = config_.num_simulations;
            search_config.batch_size = config_.mcts_batch_size;
            search_config.c_puct = config_.c_puct;
            search_config.dirichlet_alpha = 0.0f;     // No noise for reanalysis
            search_config.dirichlet_epsilon = 0.0f;    // No noise for reanalysis
            search_config.risk_beta = config_.risk_beta;
            search_config.fpu_base = config_.fpu_base;
            search_config.use_gumbel = config_.use_gumbel;
            search_config.gumbel_top_k = config_.gumbel_top_k;
            search_config.gumbel_c_visit = config_.gumbel_c_visit;
            search_config.gumbel_c_scale = config_.gumbel_c_scale;

            mcts::MCTSSearch search(node_pool, search_config);

            // Initialize search with root evaluation and position history
            // History FENs enable full 8-board history encoding for leaf nodes
            std::vector<float> root_policy(
                policy_buffer.data(),
                policy_buffer.data() + encoding::MoveEncoder::POLICY_SIZE);
            search.init_search(board, root_policy, value_buffer[0], position_history);

            // MCTS simulation loop
            while (!search.is_search_complete() &&
                   !shutdown_.load(std::memory_order_acquire))
            {
                int num_leaves = search.collect_leaves(
                    obs_buffer.data(), mask_buffer.data(), config_.mcts_batch_size);

                if (num_leaves == 0) break;

                request_ids.clear();
                int queued = eval_queue_.submit_for_evaluation(
                    worker_id, obs_buffer.data(), mask_buffer.data(),
                    num_leaves, request_ids, config_.worker_timeout_ms);

                if (queued == 0) {
                    search.cancel_pending_evaluations();
                    continue;
                }

                // Retry loop: get_results returns 0 on timeout, retry until success or shutdown
                got = 0;
                timeout_count = 0;
                while (got == 0 && !shutdown_.load(std::memory_order_acquire)) {
                    got = eval_queue_.get_results(
                        worker_id, policy_buffer.data(), value_buffer.data(),
                        queued, wdl_buffer.data(),
                        config_.worker_timeout_ms);
                    if (got == 0 && ++timeout_count % 5 == 0) {
                        fprintf(stderr, "[Reanalysis %d] waiting for NN results (%ds)\n",
                                worker_id, timeout_count * config_.worker_timeout_ms / 1000);
                    }
                }

                if (got == 0) {
                    search.cancel_pending_evaluations();
                    break;  // shutdown
                }

                stats_.total_nn_evals.fetch_add(got, std::memory_order_relaxed);

                std::vector<std::vector<float>> policies(got);
                std::vector<float> values(got);
                for (int i = 0; i < got; ++i) {
                    policies[i].assign(
                        policy_buffer.data() + i * encoding::MoveEncoder::POLICY_SIZE,
                        policy_buffer.data() + (i + 1) * encoding::MoveEncoder::POLICY_SIZE);
                    values[i] = value_buffer[i];
                }
                search.update_leaves(policies, values, wdl_buffer.data());
            }

            // Extract policy and write back to buffer
            std::vector<float> new_policy;
            if (config_.use_gumbel) {
                new_policy = search.get_improved_policy();
            } else {
                // Visit-count policy (normalized)
                std::vector<int32_t> visit_counts = search.get_visit_counts();
                new_policy.resize(encoding::MoveEncoder::POLICY_SIZE, 0.0f);
                float total = 0.0f;
                for (int i = 0; i < encoding::MoveEncoder::POLICY_SIZE; ++i) {
                    new_policy[i] = static_cast<float>(visit_counts[i]);
                    total += new_policy[i];
                }
                if (total > 0.0f) {
                    for (float& p : new_policy) p /= total;
                }
            }

            // Compute KL(old || new) before overwriting
            const float* old_policy = buffer_.get_policy_ptr(index);
            if (old_policy) {
                constexpr float eps = 1e-8f;
                double kl = 0.0;
                for (int i = 0; i < encoding::MoveEncoder::POLICY_SIZE; ++i) {
                    float p = old_policy[i];
                    if (p > eps) {
                        float q = new_policy[i] + eps;
                        kl += static_cast<double>(p) * std::log(static_cast<double>(p) / q);
                    }
                }
                // Clamp negative values from floating-point noise
                if (kl < 0.0) kl = 0.0;
                int64_t kl_fixed = static_cast<int64_t>(kl * ReanalysisStats::KL_SCALE);
                stats_.total_kl_fixed.fetch_add(kl_fixed, std::memory_order_relaxed);
            }

            // Write updated policy back to buffer
            buffer_.update_policy(index, new_policy.data());

            stats_.positions_completed.fetch_add(1, std::memory_order_relaxed);
            stats_.total_simulations.fetch_add(
                search.get_simulations_completed(), std::memory_order_relaxed);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[FATAL] Reanalysis worker %d exception: %s\n",
                worker_id, e.what());
        fflush(stderr);
        {
            std::lock_guard<std::mutex> lock(error_mutex_);
            if (last_error_.empty()) {
                last_error_ = std::string("Reanalysis worker ") +
                    std::to_string(worker_id) + ": " + e.what();
            }
        }
        shutdown_.store(true, std::memory_order_release);
    } catch (...) {
        fprintf(stderr, "[FATAL] Reanalysis worker %d unknown exception\n", worker_id);
        fflush(stderr);
        {
            std::lock_guard<std::mutex> lock(error_mutex_);
            if (last_error_.empty()) {
                last_error_ = std::string("Reanalysis worker ") +
                    std::to_string(worker_id) + ": unknown exception";
            }
        }
        shutdown_.store(true, std::memory_order_release);
    }

    // Last worker signals that reanalysis is done
    // Use acq_rel + return value to avoid TOCTOU race (fetch_sub + separate load)
    int prev = workers_active_.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
        eval_queue_.shutdown();
    }
}

} // namespace selfplay
