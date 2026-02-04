#include "../../include/selfplay/parallel_coordinator.hpp"
#include <algorithm>
#include <cstdio>
#include <random>

namespace selfplay {

// ============================================================================
// Constructor / Destructor
// ============================================================================

ParallelSelfPlayCoordinator::ParallelSelfPlayCoordinator(
    const ParallelSelfPlayConfig& config,
    training::ReplayBuffer* replay_buffer)
    : config_(config)
    , eval_queue_(config.gpu_batch_size, config.queue_capacity)
    , replay_buffer_(replay_buffer)
{
    stats_.reset();
}

ParallelSelfPlayCoordinator::~ParallelSelfPlayCoordinator() {
    stop();
    wait();
}

// ============================================================================
// Main Interface
// ============================================================================

void ParallelSelfPlayCoordinator::generate_games(NeuralEvaluatorFn evaluator) {
    start(std::move(evaluator));
    wait();
}

void ParallelSelfPlayCoordinator::start(NeuralEvaluatorFn evaluator) {
    if (running_.load(std::memory_order_acquire)) {
        return; // Already running
    }

    // Reset state
    stats_.reset();
    eval_queue_.reset();
    shutdown_.store(false, std::memory_order_release);
    running_.store(true, std::memory_order_release);
    workers_active_.store(0, std::memory_order_release);
    games_remaining_.store(config_.num_workers * config_.games_per_worker, std::memory_order_release);

    evaluator_ = std::move(evaluator);

    // Start GPU thread
    gpu_thread_ = std::thread(&ParallelSelfPlayCoordinator::gpu_thread_func, this);

    // Start worker threads
    worker_threads_.clear();
    worker_threads_.reserve(config_.num_workers);

    for (int i = 0; i < config_.num_workers; ++i) {
        worker_threads_.emplace_back(
            &ParallelSelfPlayCoordinator::worker_thread_func, this, i
        );
    }
}

void ParallelSelfPlayCoordinator::wait() {
    // Wait for all worker threads
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();

    // Signal GPU thread to stop (all workers done)
    eval_queue_.shutdown();

    // Wait for GPU thread
    if (gpu_thread_.joinable()) {
        gpu_thread_.join();
    }

    running_.store(false, std::memory_order_release);
}

void ParallelSelfPlayCoordinator::stop() {
    shutdown_.store(true, std::memory_order_release);
    eval_queue_.shutdown();
}

// ============================================================================
// Thread Functions
// ============================================================================

void ParallelSelfPlayCoordinator::worker_thread_func(int worker_id) {
    workers_active_.fetch_add(1, std::memory_order_relaxed);

    try {
        // Thread-local node pool
        mcts::NodePool node_pool;

        while (!shutdown_.load(std::memory_order_acquire)) {
            int remaining = games_remaining_.fetch_sub(1, std::memory_order_acq_rel);
            if (remaining <= 0) {
                games_remaining_.fetch_add(1, std::memory_order_relaxed);
                break;
            }

            // Play one game
            GameTrajectory trajectory = play_single_game(worker_id, node_pool);

            // Add to replay buffer or completed queue
            if (replay_buffer_ != nullptr) {
                for (const auto& state : trajectory.states) {
                    replay_buffer_->add_sample(
                        state.observation,
                        state.policy,
                        state.value
                    );
                }
            } else {
                completed_games_.push(std::move(trajectory));
            }

            // Reset pool for next game
            node_pool.reset();
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[FATAL] Worker %d exception: %s\n", worker_id, e.what());
        fflush(stderr);
        {
            std::lock_guard<std::mutex> lock(error_mutex_);
            if (last_error_.empty()) {
                last_error_ = std::string("Worker ") + std::to_string(worker_id) + ": " + e.what();
            }
        }
        shutdown_.store(true, std::memory_order_release);
    } catch (...) {
        fprintf(stderr, "[FATAL] Worker %d unknown exception\n", worker_id);
        fflush(stderr);
        {
            std::lock_guard<std::mutex> lock(error_mutex_);
            if (last_error_.empty()) {
                last_error_ = std::string("Worker ") + std::to_string(worker_id) + ": unknown exception";
            }
        }
        shutdown_.store(true, std::memory_order_release);
    }

    workers_active_.fetch_sub(1, std::memory_order_relaxed);

    // If this is the last worker, signal GPU thread
    if (workers_active_.load(std::memory_order_acquire) == 0) {
        eval_queue_.shutdown();
    }
}

void ParallelSelfPlayCoordinator::gpu_thread_func() {
    try {
        // Pre-allocate output buffers (reused across batches to avoid heap allocs)
        std::vector<float> policies(config_.gpu_batch_size * POLICY_SIZE);
        std::vector<float> values(config_.gpu_batch_size);

        while (!shutdown_.load(std::memory_order_acquire)) {
            float* obs_ptr = nullptr;
            float* mask_ptr = nullptr;

            // Collect batch (blocks with timeout)
            int batch_size = eval_queue_.collect_batch(
                &obs_ptr, &mask_ptr, config_.gpu_timeout_ms
            );

            if (batch_size == 0) {
                // Check if we should exit
                if (eval_queue_.is_shutdown() ||
                    shutdown_.load(std::memory_order_acquire)) {
                    break;
                }
                continue;
            }

            try {
                // Call neural network evaluator (writes to pre-allocated buffers)
                evaluator_(obs_ptr, mask_ptr, batch_size,
                           policies.data(), values.data());

                // Distribute results to workers
                eval_queue_.submit_results(policies.data(), values.data(), batch_size);
            } catch (const std::exception& e) {
                // Track error for debugging GPU thread issues
                stats_.gpu_errors.fetch_add(1, std::memory_order_relaxed);
                fprintf(stderr, "[ERROR] GPU thread evaluator exception: %s\n", e.what());
                fflush(stderr);
                {
                    std::lock_guard<std::mutex> lock(error_mutex_);
                    if (last_error_.empty()) {
                        last_error_ = std::string("GPU thread: ") + e.what();
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[FATAL] GPU thread exception: %s\n", e.what());
        fflush(stderr);
        {
            std::lock_guard<std::mutex> lock(error_mutex_);
            if (last_error_.empty()) {
                last_error_ = std::string("GPU thread fatal: ") + e.what();
            }
        }
        shutdown_.store(true, std::memory_order_release);
    } catch (...) {
        fprintf(stderr, "[FATAL] GPU thread unknown exception\n");
        fflush(stderr);
        {
            std::lock_guard<std::mutex> lock(error_mutex_);
            if (last_error_.empty()) {
                last_error_ = "GPU thread: unknown exception";
            }
        }
        shutdown_.store(true, std::memory_order_release);
    }
}

// ============================================================================
// Game Playing
// ============================================================================

GameTrajectory ParallelSelfPlayCoordinator::play_single_game(
    int worker_id,
    mcts::NodePool& pool)
{
    auto game_start = std::chrono::steady_clock::now();

    GameTrajectory trajectory;
    trajectory.reserve(80);

    chess::Board board;
    std::deque<chess::Board> position_history;
    int move_count = 0;

    // Pre-allocate buffers for this game
    std::vector<float> obs_buffer(config_.mcts_batch_size * encoding::PositionEncoder::TOTAL_SIZE);
    std::vector<float> mask_buffer(config_.mcts_batch_size * encoding::MoveEncoder::POLICY_SIZE);
    std::vector<float> policy_buffer(config_.mcts_batch_size * encoding::MoveEncoder::POLICY_SIZE);
    std::vector<float> value_buffer(config_.mcts_batch_size);

    // Random number generator for move selection
    std::mt19937 rng(std::random_device{}());

    // Check game over: isGameOver() returns pair<GameResultReason, GameResult>
    // Game is ongoing when reason is NONE
    while (board.isGameOver().first == chess::GameResultReason::NONE &&
           move_count < config_.max_moves_per_game &&
           !shutdown_.load(std::memory_order_acquire))
    {
        pool.reset();

        auto mcts_start = std::chrono::steady_clock::now();

        // Create MCTS search
        mcts::BatchSearchConfig search_config;
        search_config.num_simulations = config_.num_simulations;
        search_config.batch_size = config_.mcts_batch_size;
        search_config.c_puct = config_.c_puct;
        search_config.dirichlet_alpha = config_.dirichlet_alpha;
        search_config.dirichlet_epsilon = config_.dirichlet_epsilon;

        mcts::MCTSSearch search(pool, search_config);

        // Run MCTS
        run_mcts_search(worker_id, search, board, position_history,
                        obs_buffer, mask_buffer, policy_buffer, value_buffer);

        auto mcts_end = std::chrono::steady_clock::now();
        auto mcts_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            mcts_end - mcts_start).count();
        stats_.total_mcts_time_ms.fetch_add(mcts_time, std::memory_order_relaxed);
        stats_.total_simulations.fetch_add(search.get_simulations_completed(),
                                            std::memory_order_relaxed);

        // Get visit counts as policy
        std::vector<int32_t> visit_counts = search.get_visit_counts();

        // Convert to policy distribution
        std::vector<float> policy(encoding::MoveEncoder::POLICY_SIZE, 0.0f);
        int total_visits = 0;
        for (int i = 0; i < encoding::MoveEncoder::POLICY_SIZE; ++i) {
            total_visits += visit_counts[i];
        }

        if (total_visits > 0) {
            // Apply temperature
            float temperature = (move_count < config_.temperature_moves) ? 1.0f : 0.0f;

            if (temperature > 0.0f) {
                // Softmax with temperature
                float sum = 0.0f;
                for (int i = 0; i < encoding::MoveEncoder::POLICY_SIZE; ++i) {
                    if (visit_counts[i] > 0) {
                        policy[i] = std::pow(static_cast<float>(visit_counts[i]), 1.0f / temperature);
                        sum += policy[i];
                    }
                }
                if (sum > 0) {
                    for (int i = 0; i < encoding::MoveEncoder::POLICY_SIZE; ++i) {
                        policy[i] /= sum;
                    }
                }
            } else {
                // Greedy (argmax)
                int best_idx = 0;
                int best_visits = 0;
                for (int i = 0; i < encoding::MoveEncoder::POLICY_SIZE; ++i) {
                    if (visit_counts[i] > best_visits) {
                        best_visits = visit_counts[i];
                        best_idx = i;
                    }
                }
                policy[best_idx] = 1.0f;
            }
        }

        // Encode current position for storage
        std::vector<float> state_obs(encoding::PositionEncoder::TOTAL_SIZE);
        std::vector<chess::Board> history_vec(position_history.begin(), position_history.end());
        encoding::PositionEncoder::encode_to_buffer(board, state_obs.data(), history_vec);

        // Add to trajectory
        trajectory.add_state(state_obs, policy);

        // Select move
        chess::Move selected_move;
        // Get legal moves
        chess::Movelist legal_moves;
        chess::movegen::legalmoves(legal_moves, board);

        // Handle case where MCTS failed or returned no visits
        if (total_visits == 0) {
            // MCTS failed - track and select a random legal move
            stats_.mcts_failures.fetch_add(1, std::memory_order_relaxed);
            if (!legal_moves.empty()) {
                std::uniform_int_distribution<size_t> dist(0, legal_moves.size() - 1);
                selected_move = legal_moves[dist(rng)];
            }
        } else if (move_count < config_.temperature_moves) {
            // Sample from policy using temperature
            std::discrete_distribution<int> dist(policy.begin(), policy.end());
            int move_idx = dist(rng);

            // Convert index to move
            bool found = false;
            for (const auto& move : legal_moves) {
                int idx = encoding::MoveEncoder::move_to_index(move, board);
                if (idx == move_idx) {
                    selected_move = move;
                    found = true;
                    break;
                }
            }

            if (!found && !legal_moves.empty()) {
                // Fallback to best move by visit count
                int best_visits = -1;  // Start at -1 so we always select something
                for (const auto& move : legal_moves) {
                    int idx = encoding::MoveEncoder::move_to_index(move, board);
                    if (idx >= 0 && visit_counts[idx] > best_visits) {
                        best_visits = visit_counts[idx];
                        selected_move = move;
                    }
                }
                // If still not found, pick first legal move
                if (best_visits < 0) {
                    selected_move = legal_moves[0];
                }
            }
        } else {
            // Greedy selection - pick move with most visits
            int best_visits = -1;  // Start at -1 so we always select something
            for (const auto& move : legal_moves) {
                int idx = encoding::MoveEncoder::move_to_index(move, board);
                if (idx >= 0 && visit_counts[idx] > best_visits) {
                    best_visits = visit_counts[idx];
                    selected_move = move;
                }
            }
            // If no visits found, pick first legal move
            if (best_visits < 0 && !legal_moves.empty()) {
                selected_move = legal_moves[0];
            }
        }

        // Safety check: if no move was selected (shouldn't happen), break
        if (legal_moves.empty()) {
            break;  // No legal moves = game over (shouldn't reach here due to isGameOver check)
        }

        // Update position history
        position_history.push_back(board);
        if (position_history.size() > 8) {
            position_history.pop_front();
        }

        // Make move
        board.makeMove(selected_move);
        move_count++;
    }

    // Determine game result
    chess::GameResult result;
    auto game_result = board.isGameOver();

    if (game_result.first == chess::GameResultReason::CHECKMATE) {
        // Side to move is checkmated, so they lost
        result = (board.sideToMove() == chess::Color::WHITE) ?
                 chess::GameResult::LOSE : chess::GameResult::WIN;
    } else if (game_result.first != chess::GameResultReason::NONE) {
        result = chess::GameResult::DRAW;
    } else {
        // Max moves reached
        result = chess::GameResult::DRAW;
    }

    trajectory.set_outcomes(result);

    // Update statistics
    auto game_end = std::chrono::steady_clock::now();
    auto game_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        game_end - game_start).count();

    stats_.games_completed.fetch_add(1, std::memory_order_relaxed);
    stats_.total_moves.fetch_add(move_count, std::memory_order_relaxed);
    stats_.total_game_time_ms.fetch_add(game_time, std::memory_order_relaxed);

    if (result == chess::GameResult::WIN) {
        stats_.white_wins.fetch_add(1, std::memory_order_relaxed);
    } else if (result == chess::GameResult::LOSE) {
        stats_.black_wins.fetch_add(1, std::memory_order_relaxed);
    } else {
        stats_.draws.fetch_add(1, std::memory_order_relaxed);
    }

    return trajectory;
}

void ParallelSelfPlayCoordinator::run_mcts_search(
    int worker_id,
    mcts::MCTSSearch& search,
    const chess::Board& board,
    const std::deque<chess::Board>& position_history,
    std::vector<float>& obs_buffer,
    std::vector<float>& mask_buffer,
    std::vector<float>& policy_buffer,
    std::vector<float>& value_buffer)
{
    // Convert position history
    std::vector<chess::Board> history_vec(position_history.begin(), position_history.end());

    // Encode root position
    encoding::PositionEncoder::encode_to_buffer(board, obs_buffer.data(), history_vec);

    // Get legal moves for root
    chess::Movelist legal_moves;
    chess::movegen::legalmoves(legal_moves, board);

    // Build legal mask
    std::fill(mask_buffer.begin(), mask_buffer.begin() + encoding::MoveEncoder::POLICY_SIZE, 0.0f);
    for (const auto& move : legal_moves) {
        int idx = encoding::MoveEncoder::move_to_index(move, board);
        if (idx >= 0 && idx < encoding::MoveEncoder::POLICY_SIZE) {
            mask_buffer[idx] = 1.0f;
        }
    }

    // Submit root for evaluation with retry on timeout
    int got = 0;
    for (int attempt = 0; attempt <= config_.root_eval_retries; ++attempt) {
        if (attempt > 0) {
            // Flush stale results from previous timed-out attempt
            int flushed = eval_queue_.flush_worker_results(worker_id);
            stats_.root_retries.fetch_add(1, std::memory_order_relaxed);
            if (flushed > 0) {
                stats_.stale_results_flushed.fetch_add(flushed, std::memory_order_relaxed);
            }
        }

        std::vector<int32_t> request_ids;
        eval_queue_.submit_for_evaluation(
            worker_id,
            obs_buffer.data(),
            mask_buffer.data(),
            1,
            request_ids
        );

        got = eval_queue_.get_results(
            worker_id,
            policy_buffer.data(),
            value_buffer.data(),
            1,
            config_.worker_timeout_ms
        );

        if (got > 0 || shutdown_.load(std::memory_order_acquire)) {
            break;
        }
    }

    if (got == 0 || shutdown_.load(std::memory_order_acquire)) {
        // All retries exhausted - caller falls back to uniform random
        return;
    }

    stats_.total_nn_evals.fetch_add(got, std::memory_order_relaxed);

    // Initialize search with root evaluation
    search.init_search(board,
                       std::vector<float>(policy_buffer.begin(),
                                         policy_buffer.begin() + encoding::MoveEncoder::POLICY_SIZE),
                       value_buffer[0],
                       history_vec);

    // =========================================================================
    // MCTS simulation loop
    // =========================================================================
    // Two modes based on mcts_batch_size:
    //
    // search-batch=1 (default, recommended for cross-game batching):
    //   Simple synchronous loop. GPU batching comes from multiple workers
    //   submitting to the shared queue, NOT from per-worker leaf batching.
    //   Minimizes virtual losses → maximizes search quality.
    //
    // search-batch>1 (async pipeline for experimentation):
    //   Uses double-buffering to overlap CPU leaf collection with GPU evaluation.
    //   Higher virtual losses but amortizes per-leaf overhead.

    std::vector<int32_t> sim_request_ids;

    if (config_.mcts_batch_size <= 1) {
        // =====================================================================
        // Synchronous single-leaf loop (optimal for cross-game batching)
        // =====================================================================
        // With 16+ workers each submitting 1 leaf at a time, the shared queue
        // naturally produces GPU batches of ~16+ leaves. Virtual loss is minimal
        // (at most 1 per worker = 16 total) so PUCT scores stay accurate.

        while (!search.is_search_complete() &&
               !shutdown_.load(std::memory_order_acquire))
        {
            int num_leaves = search.collect_leaves(
                obs_buffer.data(), mask_buffer.data(), 1
            );

            if (num_leaves == 0) {
                break;
            }

            sim_request_ids.clear();
            int queued = eval_queue_.submit_for_evaluation(
                worker_id, obs_buffer.data(), mask_buffer.data(),
                num_leaves, sim_request_ids
            );

            if (queued == 0) {
                search.cancel_pending_evaluations();
                continue;
            }

            got = eval_queue_.get_results(
                worker_id, policy_buffer.data(), value_buffer.data(),
                queued, config_.worker_timeout_ms
            );

            if (got == 0) {
                search.cancel_pending_evaluations();
                int flushed = eval_queue_.flush_worker_results(worker_id);
                if (flushed > 0) {
                    stats_.stale_results_flushed.fetch_add(flushed, std::memory_order_relaxed);
                }
                continue;
            }

            stats_.total_nn_evals.fetch_add(got, std::memory_order_relaxed);

            std::vector<std::vector<float>> policies(got);
            std::vector<float> values(got);
            for (int i = 0; i < got; ++i) {
                policies[i].assign(
                    policy_buffer.data() + i * encoding::MoveEncoder::POLICY_SIZE,
                    policy_buffer.data() + (i + 1) * encoding::MoveEncoder::POLICY_SIZE
                );
                values[i] = value_buffer[i];
            }
            search.update_leaves(policies, values);
        }
    } else {
        // =====================================================================
        // Pipelined async loop (for search-batch > 1)
        // =====================================================================
        // Uses double-buffering: while GPU processes batch N, CPU collects N+1.
        // Virtual losses prevent re-selecting the same paths across in-flight batches.

        int prev_queued = 0;

        // Phase 1: Collect first batch
        search.start_next_batch_collection();
        int num_leaves = search.collect_leaves_async(
            obs_buffer.data(), mask_buffer.data(), config_.mcts_batch_size
        );

        if (num_leaves > 0 && !shutdown_.load(std::memory_order_acquire)) {
            sim_request_ids.clear();
            int queued = eval_queue_.submit_for_evaluation(
                worker_id, obs_buffer.data(), mask_buffer.data(),
                num_leaves, sim_request_ids
            );

            if (queued > 0) {
                search.commit_and_swap();
                prev_queued = queued;
            } else {
                search.cancel_collection_pending();
            }
        }

        // Phase 2: Pipelined loop
        while (!search.is_search_complete() && !shutdown_.load(std::memory_order_acquire)) {
            // Step A: Collect NEXT batch (overlaps GPU processing prev batch)
            num_leaves = search.collect_leaves_async(
                obs_buffer.data(), mask_buffer.data(), config_.mcts_batch_size
            );

            // Step B: Get results from PREVIOUS batch
            if (prev_queued > 0) {
                got = eval_queue_.get_results(
                    worker_id, policy_buffer.data(), value_buffer.data(),
                    prev_queued, config_.worker_timeout_ms
                );

                if (got > 0) {
                    stats_.total_nn_evals.fetch_add(got, std::memory_order_relaxed);

                    std::vector<std::vector<float>> policies(got);
                    std::vector<float> values(got);
                    for (int i = 0; i < got; ++i) {
                        policies[i].assign(
                            policy_buffer.data() + i * encoding::MoveEncoder::POLICY_SIZE,
                            policy_buffer.data() + (i + 1) * encoding::MoveEncoder::POLICY_SIZE
                        );
                        values[i] = value_buffer[i];
                    }
                    search.update_prev_leaves(policies, values);
                } else {
                    search.cancel_prev_pending();
                    int flushed = eval_queue_.flush_worker_results(worker_id);
                    if (flushed > 0) {
                        stats_.stale_results_flushed.fetch_add(flushed, std::memory_order_relaxed);
                    }
                }
            }

            // Step C: Submit current batch
            if (num_leaves > 0) {
                sim_request_ids.clear();
                int queued = eval_queue_.submit_for_evaluation(
                    worker_id, obs_buffer.data(), mask_buffer.data(),
                    num_leaves, sim_request_ids
                );

                if (queued > 0) {
                    search.commit_and_swap();
                    prev_queued = queued;
                } else {
                    search.cancel_collection_pending();
                    prev_queued = 0;
                }
            } else {
                prev_queued = 0;
            }
        }

        // Phase 3: Drain final batch
        if (prev_queued > 0) {
            got = eval_queue_.get_results(
                worker_id, policy_buffer.data(), value_buffer.data(),
                prev_queued, config_.worker_timeout_ms
            );

            if (got > 0) {
                stats_.total_nn_evals.fetch_add(got, std::memory_order_relaxed);

                std::vector<std::vector<float>> policies(got);
                std::vector<float> values(got);
                for (int i = 0; i < got; ++i) {
                    policies[i].assign(
                        policy_buffer.data() + i * encoding::MoveEncoder::POLICY_SIZE,
                        policy_buffer.data() + (i + 1) * encoding::MoveEncoder::POLICY_SIZE
                    );
                    values[i] = value_buffer[i];
                }
                search.update_prev_leaves(policies, values);
            } else {
                search.cancel_prev_pending();
                eval_queue_.flush_worker_results(worker_id);
            }
        }
    }
}

// ============================================================================
// Metrics
// ============================================================================

ParallelSelfPlayCoordinator::DetailedMetrics
ParallelSelfPlayCoordinator::get_detailed_metrics() const {
    DetailedMetrics m;

    // From stats
    m.games_completed = stats_.games_completed.load(std::memory_order_relaxed);
    m.total_moves = stats_.total_moves.load(std::memory_order_relaxed);
    m.total_simulations = stats_.total_simulations.load(std::memory_order_relaxed);
    m.total_nn_evals = stats_.total_nn_evals.load(std::memory_order_relaxed);
    m.white_wins = stats_.white_wins.load(std::memory_order_relaxed);
    m.black_wins = stats_.black_wins.load(std::memory_order_relaxed);
    m.draws = stats_.draws.load(std::memory_order_relaxed);
    m.avg_game_length = stats_.avg_game_length();
    m.moves_per_sec = stats_.moves_per_second();
    m.sims_per_sec = stats_.sims_per_second();
    m.nn_evals_per_sec = stats_.nn_evals_per_second();

    // From queue
    const auto& qm = eval_queue_.get_metrics();
    m.batch_fill_ratio = qm.batch_fill_ratio(config_.gpu_batch_size);
    m.avg_batch_size = qm.avg_batch_size();
    m.gpu_idle_time_us = qm.gpu_wait_time_us.load(std::memory_order_relaxed);
    m.worker_wait_time_us = qm.worker_wait_time_us.load(std::memory_order_relaxed);
    m.pending_requests = eval_queue_.pending_count();

    return m;
}

std::vector<GameTrajectory> ParallelSelfPlayCoordinator::get_completed_games() {
    std::vector<GameTrajectory> games;
    GameTrajectory traj;

    while (completed_games_.try_pop(traj)) {
        games.push_back(std::move(traj));
    }

    return games;
}

} // namespace selfplay
