#pragma once

#include "evaluation_queue.hpp"
#include "game.hpp"
#include "coordinator.hpp"  // For ThreadSafeQueue
#include "../mcts/search.hpp"
#include "../mcts/node_pool.hpp"
#include "../encoding/position_encoder.hpp"
#include "../encoding/move_encoder.hpp"
#include "../training/replay_buffer.hpp"

#include <thread>
#include <vector>
#include <atomic>
#include <functional>
#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <cstdint>

namespace selfplay {

// ============================================================================
// Configuration
// ============================================================================

struct ParallelSelfPlayConfig {
    // Worker configuration
    int num_workers = 16;               // Number of parallel game workers
    int games_per_worker = 4;           // Games each worker plays

    // MCTS configuration
    int num_simulations = 800;          // MCTS simulations per move
    int mcts_batch_size = 1;            // Per-game MCTS leaf batch size (1 = optimal for cross-game batching)
    float c_puct = 1.5f;                // PUCT exploration constant
    float dirichlet_alpha = 0.3f;       // Dirichlet noise alpha
    float dirichlet_epsilon = 0.25f;    // Dirichlet noise weight

    // Move selection
    int temperature_moves = 30;         // Moves with temperature=1.0
    int max_moves_per_game = 10000;     // Safety cap (real chess rules terminate games first)

    // GPU batching configuration
    int gpu_batch_size = 512;           // Maximum GPU batch size
    int gpu_timeout_ms = 20;            // GPU batch timeout (ms) - synchronized with Python default
    int worker_timeout_ms = 2000;       // Worker result timeout (ms) - increased for GIL/GPU latency

    // Queue configuration
    int queue_capacity = 8192;          // Evaluation queue capacity (increased to prevent exhaustion)

    // Retry configuration
    int root_eval_retries = 3;          // Max retries for root NN evaluation before giving up

    // Dynamic FPU
    float fpu_base = 0.3f;              // Dynamic FPU: penalty = fpu_base * sqrt(1 - prior)

    // Entropic Risk Measure (ERM) beta for risk-sensitive MCTS
    // 0.0 = risk-neutral (standard AlphaZero), >0 = risk-seeking, <0 = risk-averse
    // Recommended range: [-3, 3]. Search-time only, not baked into training labels.
    float risk_beta = 0.0f;

    // Per-game asymmetric risk: one side uses risk_beta, other samples from [min, max]
    // If min >= max, disabled (both sides use risk_beta)
    float opponent_risk_min = 0.0f;
    float opponent_risk_max = 0.0f;

    // Gumbel Top-k Sequential Halving (Danihelka et al. 2022)
    bool use_gumbel = false;        // Use Gumbel SH at root instead of PUCT+Dirichlet
    int gumbel_top_k = 16;         // Initial m for Sequential Halving
    float gumbel_c_visit = 50.0f;  // sigma() constant
    float gumbel_c_scale = 1.0f;   // sigma() scale
};

// ============================================================================
// Statistics (Thread-Safe)
// ============================================================================

struct ParallelSelfPlayStats {
    // Game statistics
    std::atomic<int64_t> games_completed{0};
    std::atomic<int64_t> total_moves{0};
    std::atomic<int64_t> total_simulations{0};
    std::atomic<int64_t> total_nn_evals{0};

    // Game results
    std::atomic<int64_t> white_wins{0};
    std::atomic<int64_t> black_wins{0};
    std::atomic<int64_t> draws{0};

    // Draw reason breakdown
    std::atomic<int64_t> draws_repetition{0};       // THREEFOLD_REPETITION
    std::atomic<int64_t> draws_stalemate{0};         // STALEMATE
    std::atomic<int64_t> draws_fifty_move{0};        // FIFTY_MOVE_RULE
    std::atomic<int64_t> draws_insufficient{0};      // INSUFFICIENT_MATERIAL
    std::atomic<int64_t> draws_max_moves{0};         // Hit move limit (max_moves_per_game)
    std::atomic<int64_t> draws_early_repetition{0};  // Repetition AND move_count < 60 plies

    // Per-persona outcome tracking (only incremented when asymmetric risk is active)
    std::atomic<int64_t> standard_wins{0};   // "Standard" persona (risk_beta) won
    std::atomic<int64_t> opponent_wins{0};   // "Opponent" persona (sampled risk) won
    std::atomic<int64_t> asymmetric_draws{0}; // Asymmetric game ended in draw

    // Timing
    std::atomic<int64_t> total_game_time_ms{0};
    std::atomic<int64_t> total_mcts_time_ms{0};

    // Error tracking
    std::atomic<int64_t> mcts_failures{0};  // MCTS returned zero visits (eval timeout/error)
    std::atomic<int64_t> gpu_errors{0};     // GPU thread exceptions

    // Retry/stale result tracking
    std::atomic<int64_t> root_retries{0};           // Times root eval was retried (timeout recovery)
    std::atomic<int64_t> stale_results_flushed{0};  // Total stale results discarded

    // Tree depth tracking (across all moves in this generation)
    std::atomic<int64_t> max_search_depth{0};        // Deepest simulation across all moves
    std::atomic<int64_t> min_search_depth{INT64_MAX}; // Shallowest simulation (sentinel: CAS naturally reduces)
    std::atomic<int64_t> total_max_depth{0};         // Sum of per-move max depths (for avg)
    std::atomic<int64_t> depth_sample_count{0};      // Number of moves contributing to total_max_depth

    // Active game move tracking (per-worker current move count)
    std::unique_ptr<std::atomic<int>[]> worker_current_moves;  // Per-worker move count
    int num_workers_allocated{0};  // Size of the array

    void reset() {
        games_completed = 0;
        total_moves = 0;
        total_simulations = 0;
        total_nn_evals = 0;
        white_wins = 0;
        black_wins = 0;
        draws = 0;
        draws_repetition = 0;
        draws_stalemate = 0;
        draws_fifty_move = 0;
        draws_insufficient = 0;
        draws_max_moves = 0;
        draws_early_repetition = 0;
        standard_wins = 0;
        opponent_wins = 0;
        asymmetric_draws = 0;
        total_game_time_ms = 0;
        total_mcts_time_ms = 0;
        mcts_failures = 0;
        gpu_errors = 0;
        root_retries = 0;
        stale_results_flushed = 0;
        max_search_depth = 0;
        min_search_depth = INT64_MAX;
        total_max_depth = 0;
        depth_sample_count = 0;
        for (int i = 0; i < num_workers_allocated; ++i) {
            worker_current_moves[i].store(0, std::memory_order_relaxed);
        }
    }

    double avg_game_length() const {
        int64_t games = games_completed.load(std::memory_order_relaxed);
        int64_t moves = total_moves.load(std::memory_order_relaxed);
        return games > 0 ? static_cast<double>(moves) / games : 0.0;
    }

    double avg_game_time_ms() const {
        int64_t games = games_completed.load(std::memory_order_relaxed);
        int64_t time = total_game_time_ms.load(std::memory_order_relaxed);
        return games > 0 ? static_cast<double>(time) / games : 0.0;
    }

    double moves_per_second() const {
        int64_t time = total_game_time_ms.load(std::memory_order_relaxed);
        int64_t moves = total_moves.load(std::memory_order_relaxed);
        return time > 0 ? moves * 1000.0 / time : 0.0;
    }

    double sims_per_second() const {
        int64_t time = total_mcts_time_ms.load(std::memory_order_relaxed);
        int64_t sims = total_simulations.load(std::memory_order_relaxed);
        return time > 0 ? sims * 1000.0 / time : 0.0;
    }

    double nn_evals_per_second() const {
        int64_t time = total_game_time_ms.load(std::memory_order_relaxed);
        int64_t evals = total_nn_evals.load(std::memory_order_relaxed);
        return time > 0 ? evals * 1000.0 / time : 0.0;
    }
};

// ============================================================================
// Neural Network Evaluator Callback Type
// ============================================================================

// Callback signature for neural network evaluation
// Called by GPU thread with batched observations
// Observations are in NHWC format (batch_size x 8 x 8 x 123) — Python permutes to NCHW
// Should perform forward pass and write results to policies and WDL probabilities
using NeuralEvaluatorFn = std::function<void(
    const float* observations,      // batch_size x 8 x 8 x 123 (NHWC)
    const float* legal_masks,       // batch_size x 4672
    int batch_size,
    float* out_policies,            // batch_size x 4672
    float* out_values               // batch_size x 3 (WDL probabilities: win, draw, loss)
)>;

// ============================================================================
// Parallel Self-Play Coordinator
// ============================================================================
//
// Orchestrates multi-threaded self-play with cross-game batching:
// - N worker threads play games independently
// - Workers submit leaf evaluations to shared GlobalEvaluationQueue
// - GPU thread collects batches and evaluates
// - Results distributed back to workers via per-worker queues
//
// This achieves high GPU utilization by:
// 1. Parallelizing game execution across many workers
// 2. Batching NN evaluations across all games
// 3. Overlapping CPU (MCTS) and GPU (NN) work
//
// A lightweight record of a completed game's moves and result (for PGN export)
struct SampleGame {
    std::vector<std::string> moves_uci;   // UCI move strings
    chess::GameResult result;              // WIN/LOSE/DRAW from White's perspective
    chess::GameResultReason result_reason{chess::GameResultReason::NONE}; // CHECKMATE, STALEMATE, etc.
    int num_moves;                         // Total number of moves
};

class ParallelSelfPlayCoordinator {
public:
    explicit ParallelSelfPlayCoordinator(
        const ParallelSelfPlayConfig& config,
        training::ReplayBuffer* replay_buffer = nullptr
    );

    ~ParallelSelfPlayCoordinator();

    // =========================================================================
    // Main Interface
    // =========================================================================

    // Generate games using the provided neural network evaluator
    // This is the main entry point - starts workers, runs to completion
    // Blocks until all games are generated
    void generate_games(NeuralEvaluatorFn evaluator);

    // Start generation (non-blocking)
    void start(NeuralEvaluatorFn evaluator);

    // Wait for completion
    void wait();

    // Stop early (graceful shutdown)
    void stop();

    // Get last error from worker/GPU threads (thread-safe, first-error-wins)
    std::string get_last_error() const {
        std::lock_guard<std::mutex> lock(error_mutex_);
        return last_error_;
    }

    // =========================================================================
    // Status and Metrics
    // =========================================================================

    bool is_running() const { return running_.load(std::memory_order_acquire); }

    const ParallelSelfPlayStats& get_stats() const { return stats_; }

    const QueueMetrics& get_queue_metrics() const {
        return eval_queue_.get_metrics();
    }

    const GlobalEvaluationQueue& get_eval_queue() const {
        return eval_queue_;
    }

    // Get detailed metrics for monitoring
    struct DetailedMetrics {
        // From stats
        int64_t games_completed;
        int64_t total_moves;
        int64_t total_simulations;
        int64_t total_nn_evals;
        int64_t white_wins;
        int64_t black_wins;
        int64_t draws;
        double avg_game_length;
        double moves_per_sec;
        double sims_per_sec;
        double nn_evals_per_sec;

        // Tree depth
        int64_t max_search_depth;
        int64_t min_search_depth;
        double avg_search_depth;

        // Active game move counts
        int min_current_moves;
        int max_current_moves;

        // From queue
        double batch_fill_ratio;
        double avg_batch_size;
        uint64_t gpu_idle_time_us;
        uint64_t worker_wait_time_us;
        uint64_t pending_requests;
    };

    DetailedMetrics get_detailed_metrics() const;

    // =========================================================================
    // Sample Game (for PGN export)
    // =========================================================================

    // Get a sample game from this generation run (prefers decisive games)
    // Returns {moves_uci, result, num_moves} or empty if no games completed yet
    SampleGame get_sample_game() const;
    bool has_sample_game() const;

    // =========================================================================
    // Collected Games (if no replay buffer provided)
    // =========================================================================

    // Get completed game trajectories (drains the queue)
    std::vector<GameTrajectory> get_completed_games();


private:
    // Worker thread function
    void worker_thread_func(int worker_id);

    // GPU thread function
    void gpu_thread_func();

    // Play a single game (called by worker)
    GameTrajectory play_single_game(int worker_id, mcts::NodePool& pool);

    // Run MCTS search for one move
    void run_mcts_search(
        int worker_id,
        mcts::MCTSSearch& search,
        const chess::Board& board,
        const std::deque<chess::Board>& position_history,
        std::vector<float>& obs_buffer,
        std::vector<float>& mask_buffer,
        std::vector<float>& policy_buffer,
        std::vector<float>& value_buffer,
        std::vector<float>& wdl_buffer
    );

    // Configuration
    ParallelSelfPlayConfig config_;

    // Evaluation queue
    GlobalEvaluationQueue eval_queue_;

    // Replay buffer (optional, for direct data storage)
    training::ReplayBuffer* replay_buffer_;

    // Game trajectory queue (if no replay buffer)
    ThreadSafeQueue<GameTrajectory> completed_games_;

    // Threads
    std::vector<std::thread> worker_threads_;
    std::thread gpu_thread_;

    // Neural network evaluator
    NeuralEvaluatorFn evaluator_;

    // Lifecycle
    std::atomic<bool> running_{false};
    std::atomic<bool> shutdown_{false};
    std::atomic<int> workers_active_{0};
    std::atomic<int> games_remaining_{0};

    // Statistics
    ParallelSelfPlayStats stats_;

    // Error tracking (thread-safe, first-error-wins)
    mutable std::mutex error_mutex_;
    std::string last_error_;

    // Sample game storage (thread-safe, prefers decisive games)
    mutable std::mutex sample_game_mutex_;
    SampleGame sample_game_;
    bool has_sample_game_ = false;
};

} // namespace selfplay
