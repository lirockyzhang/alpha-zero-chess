#pragma once

#include "evaluation_queue.hpp"
#include "../mcts/search.hpp"
#include "../mcts/node_pool.hpp"
#include "../encoding/position_encoder.hpp"
#include "../encoding/move_encoder.hpp"
#include "../training/replay_buffer.hpp"

#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <string>
#include <cstdint>

namespace selfplay {

// ============================================================================
// Configuration
// ============================================================================

struct ReanalysisConfig {
    int num_workers = 8;
    int num_simulations = 50;         // Typically 25% of self-play sims
    int mcts_batch_size = 1;          // Per-position MCTS leaf batch size
    float c_puct = 1.5f;
    float fpu_base = 0.3f;
    float risk_beta = 0.0f;

    bool use_gumbel = false;
    int gumbel_top_k = 16;
    float gumbel_c_visit = 50.0f;
    float gumbel_c_scale = 1.0f;

    int gpu_batch_size = 512;         // Must match coordinator's eval queue size
    int worker_timeout_ms = 2000;
    int queue_capacity = 4096;

    // NO dirichlet noise, NO temperature — pure MCTS policy for reanalysis
};

// ============================================================================
// Statistics
// ============================================================================

struct ReanalysisStats {
    std::atomic<int64_t> positions_completed{0};
    std::atomic<int64_t> total_simulations{0};
    std::atomic<int64_t> total_nn_evals{0};
    std::atomic<int64_t> positions_skipped{0};  // Terminal or missing FEN

    void reset() {
        positions_completed = 0;
        total_simulations = 0;
        total_nn_evals = 0;
        positions_skipped = 0;
    }
};

// ============================================================================
// Reanalyzer
// ============================================================================
//
// Runs MCTS re-searches on stored replay buffer positions using the current
// (improved) network. Worker threads only — no GPU thread. The coordinator's
// GPU thread pulls from this class's eval_queue_ as a secondary queue.
//
// Usage:
//   Reanalyzer reanalyzer(buffer, config);
//   reanalyzer.set_indices({0, 1, 2, ...});
//   coordinator.set_secondary_queue(&reanalyzer.get_eval_queue());
//   reanalyzer.start();
//   // ... self-play workers run concurrently ...
//   coordinator.wait_for_workers();
//   reanalyzer.wait();
//   coordinator.clear_secondary_queue();
//   coordinator.shutdown_gpu_thread();

class Reanalyzer {
public:
    Reanalyzer(training::ReplayBuffer& buffer, const ReanalysisConfig& config);
    ~Reanalyzer();

    // Set buffer indices to reanalyze (call before start())
    void set_indices(const std::vector<size_t>& indices);

    // Start reanalysis workers (non-blocking)
    void start();

    // Wait for all positions to be processed
    void wait();

    // Graceful shutdown (can be called from another thread)
    void stop();

    // Get the eval queue (connect to coordinator's secondary queue)
    GlobalEvaluationQueue& get_eval_queue() { return eval_queue_; }

    // Statistics
    const ReanalysisStats& get_stats() const { return stats_; }

    // Last error message (empty if no errors)
    std::string get_last_error() const {
        std::lock_guard<std::mutex> lock(error_mutex_);
        return last_error_;
    }

private:
    void worker_func(int worker_id);

    training::ReplayBuffer& buffer_;
    ReanalysisConfig config_;
    GlobalEvaluationQueue eval_queue_;

    // Work distribution: atomic cursor over work_indices_
    std::vector<size_t> work_indices_;
    std::atomic<size_t> work_cursor_{0};

    // Thread management
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> shutdown_{false};
    std::atomic<int> workers_active_{0};

    // Stats and error tracking
    ReanalysisStats stats_;
    mutable std::mutex error_mutex_;
    std::string last_error_;
};

} // namespace selfplay
