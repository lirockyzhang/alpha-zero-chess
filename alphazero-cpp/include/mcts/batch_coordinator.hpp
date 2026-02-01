#pragma once

#include "search.hpp"
#include "../third_party/chess-library/include/chess.hpp"
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace mcts {

// Position to be evaluated by neural network
struct EvalRequest {
    int game_id;           // Which game this position belongs to
    Node* node;            // Node to expand after evaluation
    chess::Board board;    // Board position to evaluate
};

// Result from neural network evaluation
struct EvalResult {
    int game_id;
    std::vector<float> policy;  // Policy vector (1858 moves)
    float value;                 // Value estimate [-1, 1]
};

// Batch coordinator for parallel MCTS across multiple games
// Implements 90% threshold with Hard Sync mechanism
class BatchCoordinator {
public:
    struct Config {
        int batch_size;
        int simulations_per_move;
        float batch_threshold;
        int hard_sync_interval;

        Config()
            : batch_size(256)
            , simulations_per_move(800)
            , batch_threshold(0.9f)
            , hard_sync_interval(10)
        {}
    };

    BatchCoordinator(const Config& config = Config())
        : config_(config)
        , batch_counter_(0)
        , active_games_(0)
    {}

    // Add a game to the batch
    void add_game(int game_id, const chess::Board& initial_position) {
        std::lock_guard<std::mutex> lock(mutex_);

        GameState state;
        state.game_id = game_id;
        state.board = initial_position;
        state.simulations_done = 0;
        state.waiting_for_eval = false;

        games_[game_id] = state;
        active_games_++;
    }

    // Submit a position for neural network evaluation
    void submit_eval_request(const EvalRequest& request) {
        std::lock_guard<std::mutex> lock(mutex_);

        eval_queue_.push(request);
        games_[request.game_id].waiting_for_eval = true;

        // Check if we should send a batch
        if (should_send_batch()) {
            cv_.notify_one();
        }
    }

    // Collect a batch of positions for neural network evaluation
    // Returns when batch_threshold of games are ready, or Hard Sync is triggered
    std::vector<EvalRequest> collect_batch() {
        std::unique_lock<std::mutex> lock(mutex_);

        // Determine if this is a Hard Sync batch
        bool force_hard_sync = (batch_counter_ % config_.hard_sync_interval == 0);

        // Wait until enough positions are ready
        cv_.wait(lock, [this, force_hard_sync]() {
            if (force_hard_sync) {
                // Hard Sync: wait for ALL active games
                return eval_queue_.size() >= active_games_;
            } else {
                // Normal: wait for 90% threshold
                return should_send_batch();
            }
        });

        // Collect batch
        std::vector<EvalRequest> batch;
        size_t batch_size = force_hard_sync ? active_games_.load() :
                        std::min(static_cast<size_t>(config_.batch_size), eval_queue_.size());

        for (int i = 0; i < batch_size && !eval_queue_.empty(); ++i) {
            batch.push_back(eval_queue_.front());
            eval_queue_.pop();
        }

        batch_counter_++;
        return batch;
    }

    // Process neural network results
    void process_results(const std::vector<EvalResult>& results) {
        std::lock_guard<std::mutex> lock(mutex_);

        for (const auto& result : results) {
            auto it = games_.find(result.game_id);
            if (it != games_.end()) {
                it->second.waiting_for_eval = false;
                it->second.simulations_done++;

                // Store result for game to process
                pending_results_[result.game_id].push_back(result);
            }
        }

        cv_.notify_all();
    }

    // Get pending results for a game
    std::vector<EvalResult> get_pending_results(int game_id) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<EvalResult> results;
        auto it = pending_results_.find(game_id);
        if (it != pending_results_.end()) {
            results = std::move(it->second);
            pending_results_.erase(it);
        }

        return results;
    }

    // Check if a game is complete
    bool is_game_complete(int game_id) const {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = games_.find(game_id);
        if (it == games_.end()) return true;

        return it->second.simulations_done >= config_.simulations_per_move;
    }

    // Remove a completed game
    void remove_game(int game_id) {
        std::lock_guard<std::mutex> lock(mutex_);

        games_.erase(game_id);
        pending_results_.erase(game_id);
        active_games_--;
    }

    // Get statistics
    struct Stats {
        int active_games;
        int pending_evals;
        int batch_counter;
        bool next_is_hard_sync;
    };

    Stats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);

        Stats stats;
        stats.active_games = active_games_;
        stats.pending_evals = eval_queue_.size();
        stats.batch_counter = batch_counter_;
        stats.next_is_hard_sync = (batch_counter_ % config_.hard_sync_interval == 0);

        return stats;
    }

private:
    struct GameState {
        int game_id;
        chess::Board board;
        int simulations_done;
        bool waiting_for_eval;
    };

    bool should_send_batch() const {
        if (active_games_ == 0) return false;

        // Check if we have enough positions ready
        float ready_ratio = static_cast<float>(eval_queue_.size()) / active_games_;
        return ready_ratio >= config_.batch_threshold;
    }

    Config config_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;

    std::unordered_map<int, GameState> games_;
    std::queue<EvalRequest> eval_queue_;
    std::unordered_map<int, std::vector<EvalResult>> pending_results_;

    uint32_t batch_counter_;
    std::atomic<int> active_games_;
};

} // namespace mcts
