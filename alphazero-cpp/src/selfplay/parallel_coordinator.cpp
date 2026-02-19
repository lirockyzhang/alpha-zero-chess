#include "../../include/selfplay/parallel_coordinator.hpp"
#include <algorithm>
#include <climits>
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
    if (config.num_workers > static_cast<int>(MAX_WORKERS)) {
        fprintf(stderr,
            "[WARNING] num_workers (%d) exceeds MAX_WORKERS (%zu). "
            "Workers %zu-%d will silently fail all evaluations! "
            "Clamping to %zu workers.\n",
            config.num_workers, MAX_WORKERS, MAX_WORKERS,
            config.num_workers - 1, MAX_WORKERS);
        fflush(stderr);
        config_.num_workers = static_cast<int>(MAX_WORKERS);
    }
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

    // Reset state — allocate per-worker move tracking before reset() zeroes it
    stats_.worker_current_moves = std::make_unique<std::atomic<int>[]>(config_.num_workers);
    stats_.num_workers_allocated = config_.num_workers;
    stats_.reset();
    eval_queue_.reset();
    {
        std::lock_guard<std::mutex> lock(sample_game_mutex_);
        has_sample_game_ = false;
    }
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

            // Store sample game (prefer decisive games over draws)
            {
                std::lock_guard<std::mutex> lock(sample_game_mutex_);
                bool is_decisive = (trajectory.result == chess::GameResult::WIN ||
                                    trajectory.result == chess::GameResult::LOSE);
                bool current_is_draw = has_sample_game_ &&
                    sample_game_.result == chess::GameResult::DRAW;
                if (!has_sample_game_ || (current_is_draw && is_decisive)) {
                    sample_game_.moves_uci = trajectory.moves_uci;
                    sample_game_.result = trajectory.result;
                    sample_game_.result_reason = trajectory.result_reason;
                    sample_game_.num_moves = trajectory.num_moves;
                    has_sample_game_ = true;
                }
            }

            // Add to replay buffer or completed queue
            if (replay_buffer_ != nullptr) {
                // Compute game-level metadata fields
                uint8_t game_res = (trajectory.result == chess::GameResult::WIN) ? 0
                                 : (trajectory.result == chess::GameResult::LOSE) ? 2 : 1;
                uint8_t term;
                switch (trajectory.result_reason) {
                    case chess::GameResultReason::CHECKMATE:              term = 0; break;
                    case chess::GameResultReason::STALEMATE:              term = 1; break;
                    case chess::GameResultReason::THREEFOLD_REPETITION:   term = 2; break;
                    case chess::GameResultReason::FIFTY_MOVE_RULE:        term = 3; break;
                    case chess::GameResultReason::INSUFFICIENT_MATERIAL:  term = 4; break;
                    default:                                             term = 255; break;
                }
                // NONE reason with draw result means max_moves reached
                if (trajectory.result_reason == chess::GameResultReason::NONE &&
                    trajectory.result == chess::GameResult::DRAW) {
                    term = 5; // MAX_MOVES
                }
                uint16_t game_len = static_cast<uint16_t>(trajectory.states.size());

                for (size_t i = 0; i < trajectory.states.size(); ++i) {
                    const auto& state = trajectory.states[i];
                    training::SampleMeta meta{
                        replay_buffer_->current_iteration(),
                        game_res, term,
                        static_cast<uint16_t>(i), game_len
                    };
                    replay_buffer_->add_sample(
                        state.observation,
                        state.policy,
                        state.value,
                        state.mcts_wdl.data(),
                        &state.soft_value,
                        &meta
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
        std::vector<float> wdl(config_.gpu_batch_size * 3);    // WDL probabilities from NN
        std::vector<float> values(config_.gpu_batch_size);      // Scalar values after WDL→value conversion

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
                // Call neural network evaluator (writes WDL probs to wdl buffer)
                evaluator_(obs_ptr, mask_ptr, batch_size,
                           policies.data(), wdl.data());

                // Convert WDL probabilities to scalar values
                // Risk adjustment happens at node level (q_value_risk), not here
                for (int i = 0; i < batch_size; ++i) {
                    float pw = wdl[i * 3 + 0];  // P(win)
                    float pd = wdl[i * 3 + 1];  // P(draw)
                    float pl = wdl[i * 3 + 2];  // P(loss)
                    values[i] = mcts::wdl_to_value(pw, pd, pl);
                }

                // Distribute scalar results + WDL probs to workers
                eval_queue_.submit_results(policies.data(), values.data(), batch_size, wdl.data());
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
    std::vector<float> wdl_buffer(config_.mcts_batch_size * 3);  // WDL probs from eval queue

    // Random number generator for move selection
    std::mt19937 rng(std::random_device{}());

    // Per-game asymmetric risk:
    // One side plays with fixed config_.risk_beta, other samples from [min, max].
    // Which side is "opponent" is chosen per-game (50/50) to avoid color bias.
    float risk_white, risk_black;
    bool is_asymmetric = false;
    bool white_is_standard = true; // only meaningful when is_asymmetric=true
    if (config_.opponent_risk_min < config_.opponent_risk_max) {
        is_asymmetric = true;
        std::uniform_real_distribution<float> risk_dist(config_.opponent_risk_min, config_.opponent_risk_max);
        float opponent_risk = risk_dist(rng);
        if (std::uniform_int_distribution<int>(0, 1)(rng)) {
            risk_white = opponent_risk;
            risk_black = config_.risk_beta;
            white_is_standard = false;  // white plays as opponent
        } else {
            risk_white = config_.risk_beta;
            risk_black = opponent_risk;
            white_is_standard = true;   // white plays as standard
        }
    } else {
        risk_white = config_.risk_beta;
        risk_black = config_.risk_beta;
    }

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
        search_config.risk_beta = (board.sideToMove() == chess::Color::WHITE)
                                  ? risk_white : risk_black;
        search_config.fpu_base = config_.fpu_base;

        // Gumbel Top-k Sequential Halving config
        search_config.use_gumbel = config_.use_gumbel;
        search_config.gumbel_top_k = config_.gumbel_top_k;
        search_config.gumbel_c_visit = config_.gumbel_c_visit;
        search_config.gumbel_c_scale = config_.gumbel_c_scale;

        mcts::MCTSSearch search(pool, search_config);

        // Run MCTS
        run_mcts_search(worker_id, search, board, position_history,
                        obs_buffer, mask_buffer, policy_buffer, value_buffer,
                        wdl_buffer);

        auto mcts_end = std::chrono::steady_clock::now();
        auto mcts_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            mcts_end - mcts_start).count();
        stats_.total_mcts_time_ms.fetch_add(mcts_time, std::memory_order_relaxed);
        stats_.total_simulations.fetch_add(search.get_simulations_completed(),
                                            std::memory_order_relaxed);

        // Track tree depth from this move's search
        {
            int64_t depth_max = static_cast<int64_t>(search.get_max_depth());
            int64_t depth_min = static_cast<int64_t>(search.get_min_depth());

            if (depth_max > 0) {
                // Atomic max (CAS strong, explicit int64_t)
                int64_t cur = stats_.max_search_depth.load(std::memory_order_relaxed);
                while (depth_max > cur &&
                       !stats_.max_search_depth.compare_exchange_strong(
                           cur, depth_max, std::memory_order_relaxed));

                // Atomic min (CAS) — only if depth_min is valid (> 0)
                if (depth_min > 0) {
                    int64_t cur_min = stats_.min_search_depth.load(std::memory_order_relaxed);
                    while (depth_min < cur_min &&
                           !stats_.min_search_depth.compare_exchange_strong(
                               cur_min, depth_min, std::memory_order_relaxed));
                }

                // Accumulate for average
                stats_.total_max_depth.fetch_add(depth_max, std::memory_order_relaxed);
                stats_.depth_sample_count.fetch_add(1, std::memory_order_relaxed);
            }
        }

        // Get legal moves (needed for both paths)
        chess::Movelist legal_moves;
        chess::movegen::legalmoves(legal_moves, board);

        // Policy extraction and move selection — Gumbel vs PUCT paths
        std::vector<float> policy(encoding::MoveEncoder::POLICY_SIZE, 0.0f);
        chess::Move selected_move;

        if (config_.use_gumbel) {
            // =========================================================
            // Gumbel path: improved policy + SH winner / argmax
            // =========================================================
            policy = search.get_improved_policy();

            // Check if policy is valid (non-zero sum)
            float policy_sum = 0.0f;
            for (float p : policy) policy_sum += p;

            if (policy_sum < 1e-8f) {
                // Gumbel search failed — fallback to random
                stats_.mcts_failures.fetch_add(1, std::memory_order_relaxed);
                if (!legal_moves.empty()) {
                    std::uniform_int_distribution<size_t> dist(0, legal_moves.size() - 1);
                    selected_move = legal_moves[dist(rng)];
                }
            } else if (move_count < config_.temperature_moves) {
                // Stochastic: SH winner (valid sample from improved policy via Gumbel-Max)
                selected_move = search.get_gumbel_action();
            } else {
                // Deterministic: argmax of improved policy
                int best_idx = static_cast<int>(
                    std::max_element(policy.begin(), policy.end()) - policy.begin());
                bool found = false;
                for (const auto& move : legal_moves) {
                    int idx = encoding::MoveEncoder::move_to_index(move, board);
                    if (idx == best_idx) {
                        selected_move = move;
                        found = true;
                        break;
                    }
                }
                if (!found && !legal_moves.empty()) {
                    selected_move = legal_moves[0];
                }
            }
        } else {
            // =========================================================
            // PUCT path: visit-count policy + temperature sampling
            // =========================================================
            std::vector<int32_t> visit_counts = search.get_visit_counts();
            int total_visits = 0;
            for (int i = 0; i < encoding::MoveEncoder::POLICY_SIZE; ++i) {
                total_visits += visit_counts[i];
            }

            if (total_visits > 0) {
                float temperature = (move_count < config_.temperature_moves) ? 1.0f : 0.0f;

                if (temperature > 0.0f) {
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

            if (total_visits == 0) {
                stats_.mcts_failures.fetch_add(1, std::memory_order_relaxed);
                if (!legal_moves.empty()) {
                    std::uniform_int_distribution<size_t> dist(0, legal_moves.size() - 1);
                    selected_move = legal_moves[dist(rng)];
                }
            } else if (move_count < config_.temperature_moves) {
                std::discrete_distribution<int> dist(policy.begin(), policy.end());
                int move_idx = dist(rng);
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
                    int best_visits = -1;
                    for (const auto& move : legal_moves) {
                        int idx = encoding::MoveEncoder::move_to_index(move, board);
                        if (idx >= 0 && visit_counts[idx] > best_visits) {
                            best_visits = visit_counts[idx];
                            selected_move = move;
                        }
                    }
                    if (best_visits < 0) {
                        selected_move = legal_moves[0];
                    }
                }
            } else {
                int best_visits = -1;
                for (const auto& move : legal_moves) {
                    int idx = encoding::MoveEncoder::move_to_index(move, board);
                    if (idx >= 0 && visit_counts[idx] > best_visits) {
                        best_visits = visit_counts[idx];
                        selected_move = move;
                    }
                }
                if (best_visits < 0 && !legal_moves.empty()) {
                    selected_move = legal_moves[0];
                }
            }
        }

        // Encode current position for storage
        std::vector<float> state_obs(encoding::PositionEncoder::TOTAL_SIZE);
        std::vector<chess::Board> history_vec(position_history.begin(), position_history.end());
        encoding::PositionEncoder::encode_to_buffer(board, state_obs.data(), history_vec);

        // Compute risk-adjusted root value (LogSumExp diagnostic) if risk_beta != 0
        float move_risk_beta = (board.sideToMove() == chess::Color::WHITE)
                               ? risk_white : risk_black;
        float soft_val = (move_risk_beta != 0.0f)
            ? search.get_root_risk_value(move_risk_beta) : 0.0f;

        // Add to trajectory (with MCTS root WDL distribution and soft value)
        trajectory.add_state(state_obs, policy, search.get_root_wdl(), soft_val);

        // Safety check: if no move was selected (shouldn't happen), break
        if (legal_moves.empty()) {
            break;  // No legal moves = game over (shouldn't reach here due to isGameOver check)
        }

        // Update position history
        position_history.push_back(board);
        if (position_history.size() > 8) {
            position_history.pop_front();
        }

        // Record move for PGN export
        trajectory.moves_uci.push_back(chess::uci::moveToUci(selected_move));

        // Make move
        board.makeMove(selected_move);
        move_count++;
        stats_.worker_current_moves[worker_id].store(move_count, std::memory_order_relaxed);
    }

    // Determine game result and reason
    chess::GameResult result;
    auto game_result = board.isGameOver();
    chess::GameResultReason reason = game_result.first;

    if (reason == chess::GameResultReason::CHECKMATE) {
        // Side to move is checkmated, so they lost
        result = (board.sideToMove() == chess::Color::WHITE) ?
                 chess::GameResult::LOSE : chess::GameResult::WIN;
    } else if (reason != chess::GameResultReason::NONE) {
        result = chess::GameResult::DRAW;
    } else {
        // Max moves reached — reason stays NONE to distinguish from library-detected draws
        result = chess::GameResult::DRAW;
    }

    trajectory.set_outcomes(result);
    trajectory.result_reason = reason;

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
        // Draw reason breakdown
        if (reason == chess::GameResultReason::THREEFOLD_REPETITION) {
            stats_.draws_repetition.fetch_add(1, std::memory_order_relaxed);
            if (move_count < 60) {  // 60 plies = 30 full moves
                stats_.draws_early_repetition.fetch_add(1, std::memory_order_relaxed);
            }
        } else if (reason == chess::GameResultReason::STALEMATE) {
            stats_.draws_stalemate.fetch_add(1, std::memory_order_relaxed);
        } else if (reason == chess::GameResultReason::FIFTY_MOVE_RULE) {
            stats_.draws_fifty_move.fetch_add(1, std::memory_order_relaxed);
        } else if (reason == chess::GameResultReason::INSUFFICIENT_MATERIAL) {
            stats_.draws_insufficient.fetch_add(1, std::memory_order_relaxed);
        } else {
            // reason == NONE means max_moves_per_game reached (loop exit condition)
            stats_.draws_max_moves.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Track per-persona outcomes (only for asymmetric games)
    if (is_asymmetric) {
        if (result == chess::GameResult::DRAW) {
            stats_.asymmetric_draws.fetch_add(1, std::memory_order_relaxed);
        } else {
            // WIN = white won, LOSE = black won
            // standard_won when (white won AND white is standard) OR (black won AND black is standard)
            bool standard_won = (result == chess::GameResult::WIN) == white_is_standard;
            if (standard_won) {
                stats_.standard_wins.fetch_add(1, std::memory_order_relaxed);
            } else {
                stats_.opponent_wins.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    // Reset move counter (game finished)
    stats_.worker_current_moves[worker_id].store(0, std::memory_order_relaxed);

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
    std::vector<float>& value_buffer,
    std::vector<float>& wdl_buffer)
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
            request_ids,
            config_.worker_timeout_ms  // Blocking submit (backpressure)
        );

        got = eval_queue_.get_results(
            worker_id,
            policy_buffer.data(),
            value_buffer.data(),
            1,
            config_.worker_timeout_ms,
            wdl_buffer.data()
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
                num_leaves, sim_request_ids,
                config_.worker_timeout_ms  // Blocking submit (backpressure)
            );

            if (queued == 0) {
                search.cancel_pending_evaluations();
                continue;
            }

            got = eval_queue_.get_results(
                worker_id, policy_buffer.data(), value_buffer.data(),
                queued, config_.worker_timeout_ms,
                wdl_buffer.data()
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
            search.update_leaves(policies, values, wdl_buffer.data());
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
                num_leaves, sim_request_ids,
                config_.worker_timeout_ms  // Blocking submit (backpressure)
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
                    prev_queued, config_.worker_timeout_ms,
                    wdl_buffer.data()
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
                    search.update_prev_leaves(policies, values, wdl_buffer.data());
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
                    num_leaves, sim_request_ids,
                    config_.worker_timeout_ms  // Blocking submit (backpressure)
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
                prev_queued, config_.worker_timeout_ms,
                wdl_buffer.data()
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
                search.update_prev_leaves(policies, values, wdl_buffer.data());
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

    // Tree depth
    m.max_search_depth = stats_.max_search_depth.load(std::memory_order_relaxed);
    int64_t raw_min = stats_.min_search_depth.load(std::memory_order_relaxed);
    m.min_search_depth = (raw_min == INT64_MAX) ? 0 : raw_min;
    int64_t depth_samples = stats_.depth_sample_count.load(std::memory_order_relaxed);
    int64_t total_depth = stats_.total_max_depth.load(std::memory_order_relaxed);
    m.avg_search_depth = depth_samples > 0 ? static_cast<double>(total_depth) / depth_samples : 0.0;

    // Active game move counts (scan non-zero entries for min/max)
    int min_moves = INT_MAX, max_moves = 0;
    for (int i = 0; i < stats_.num_workers_allocated; ++i) {
        int mc = stats_.worker_current_moves[i].load(std::memory_order_relaxed);
        if (mc > 0) {
            if (mc < min_moves) min_moves = mc;
            if (mc > max_moves) max_moves = mc;
        }
    }
    m.min_current_moves = (min_moves == INT_MAX) ? 0 : min_moves;
    m.max_current_moves = max_moves;

    // From queue
    const auto& qm = eval_queue_.get_metrics();
    m.batch_fill_ratio = qm.batch_fill_ratio(config_.gpu_batch_size);
    m.avg_batch_size = qm.avg_batch_size();
    m.gpu_idle_time_us = qm.gpu_wait_time_us.load(std::memory_order_relaxed);
    m.worker_wait_time_us = qm.worker_wait_time_us.load(std::memory_order_relaxed);
    m.pending_requests = eval_queue_.pending_count();

    return m;
}

SampleGame ParallelSelfPlayCoordinator::get_sample_game() const {
    std::lock_guard<std::mutex> lock(sample_game_mutex_);
    return sample_game_;
}

bool ParallelSelfPlayCoordinator::has_sample_game() const {
    std::lock_guard<std::mutex> lock(sample_game_mutex_);
    return has_sample_game_;
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
