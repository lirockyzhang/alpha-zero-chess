#pragma once

#include "game.hpp"
#include "../mcts/node_pool.hpp"
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <chrono>

namespace selfplay {

// Thread-safe queue for game trajectories
template<typename T>
class ThreadSafeQueue {
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        cv_.notify_one();
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || done_; });

        if (queue_.empty()) {
            throw std::runtime_error("Queue is empty and done");
        }

        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void set_done() {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
        cv_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<T> queue_;
    std::atomic<bool> done_{false};
};

// Statistics for self-play progress (internal, thread-safe)
struct SelfPlayStats {
    std::atomic<int> games_completed{0};
    std::atomic<int> total_moves{0};
    std::atomic<int> white_wins{0};
    std::atomic<int> black_wins{0};
    std::atomic<int> draws{0};
    std::atomic<double> total_game_time{0.0};

    void reset() {
        games_completed = 0;
        total_moves = 0;
        white_wins = 0;
        black_wins = 0;
        draws = 0;
        total_game_time = 0.0;
    }
};

// Snapshot of statistics (copyable, for returning to caller)
struct SelfPlayStatsSnapshot {
    int games_completed = 0;
    int total_moves = 0;
    int white_wins = 0;
    int black_wins = 0;
    int draws = 0;
    double total_game_time = 0.0;

    double avg_game_length() const {
        return games_completed > 0 ? static_cast<double>(total_moves) / games_completed : 0.0;
    }

    double avg_game_time() const {
        return games_completed > 0 ? total_game_time / games_completed : 0.0;
    }
};

// Configuration for self-play coordinator
struct CoordinatorConfig {
    int num_workers = 4;                    // Number of parallel workers
    int games_per_worker = 25;              // Games per worker before returning
    SelfPlayConfig game_config;             // Configuration for each game
    bool collect_stats = true;              // Whether to collect statistics
};

// Coordinates multi-threaded self-play game generation
// Uses worker threads to generate games in parallel while sharing NN evaluator
class SelfPlayCoordinator {
public:
    explicit SelfPlayCoordinator(const CoordinatorConfig& config = CoordinatorConfig())
        : config_(config)
        , stats_()
        , running_(false)
    {}

    ~SelfPlayCoordinator() {
        stop();
    }

    // Generate games using the provided neural network evaluator
    // Returns when target number of games is reached
    template<typename NeuralEvaluator>
    std::vector<GameTrajectory> generate_games(NeuralEvaluator& neural_evaluator, int num_games);

    // Start async game generation (non-blocking)
    template<typename NeuralEvaluator>
    void start_async(NeuralEvaluator& neural_evaluator, int num_games);

    // Stop async generation
    void stop();

    // Get completed trajectories (non-blocking)
    std::vector<GameTrajectory> get_completed_games();

    // Get current statistics snapshot (thread-safe copy of atomic values)
    SelfPlayStatsSnapshot get_stats() const;

    // Check if async generation is running
    bool is_running() const { return running_.load(); }

private:
    // Worker function for generating games
    template<typename NeuralEvaluator>
    void worker_thread(NeuralEvaluator& neural_evaluator, int games_to_generate);

    CoordinatorConfig config_;
    SelfPlayStats stats_;
    std::atomic<bool> running_;

    // Thread management
    std::vector<std::thread> workers_;
    ThreadSafeQueue<GameTrajectory> completed_games_;
};

// Template implementations

template<typename NeuralEvaluator>
std::vector<GameTrajectory> SelfPlayCoordinator::generate_games(
    NeuralEvaluator& neural_evaluator,
    int num_games)
{
    stats_.reset();
    running_ = true;

    // Calculate games per worker
    int games_per_worker = (num_games + config_.num_workers - 1) / config_.num_workers;

    // Launch worker threads
    std::vector<std::thread> workers;
    for (int i = 0; i < config_.num_workers; ++i) {
        int worker_games = std::min(games_per_worker, num_games - i * games_per_worker);
        if (worker_games > 0) {
            workers.emplace_back(&SelfPlayCoordinator::worker_thread<NeuralEvaluator>,
                                this, std::ref(neural_evaluator), worker_games);
        }
    }

    // Wait for all workers to complete
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    running_ = false;

    // Collect all completed games
    return get_completed_games();
}

template<typename NeuralEvaluator>
void SelfPlayCoordinator::start_async(NeuralEvaluator& neural_evaluator, int num_games) {
    if (running_.load()) {
        throw std::runtime_error("Async generation already running");
    }

    stats_.reset();
    running_ = true;

    // Calculate games per worker
    int games_per_worker = (num_games + config_.num_workers - 1) / config_.num_workers;

    // Launch worker threads
    workers_.clear();
    for (int i = 0; i < config_.num_workers; ++i) {
        int worker_games = std::min(games_per_worker, num_games - i * games_per_worker);
        if (worker_games > 0) {
            workers_.emplace_back(&SelfPlayCoordinator::worker_thread<NeuralEvaluator>,
                                this, std::ref(neural_evaluator), worker_games);
        }
    }
}

template<typename NeuralEvaluator>
void SelfPlayCoordinator::worker_thread(NeuralEvaluator& neural_evaluator, int games_to_generate) {
    // Each worker has its own node pool
    mcts::NodePool pool;

    for (int i = 0; i < games_to_generate && running_.load(); ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Create game and play
        SelfPlayGame game(pool, config_.game_config);
        GameTrajectory trajectory = game.play_game(neural_evaluator);

        auto end_time = std::chrono::high_resolution_clock::now();
        double game_time = std::chrono::duration<double>(end_time - start_time).count();

        // Update statistics
        if (config_.collect_stats) {
            stats_.games_completed.fetch_add(1);
            stats_.total_moves.fetch_add(trajectory.num_moves);
            stats_.total_game_time.fetch_add(game_time);

            if (trajectory.result == chess::GameResult::WIN) {
                stats_.white_wins.fetch_add(1);
            } else if (trajectory.result == chess::GameResult::LOSE) {
                stats_.black_wins.fetch_add(1);
            } else {
                stats_.draws.fetch_add(1);
            }
        }

        // Add to completed queue
        completed_games_.push(std::move(trajectory));

        // Reset pool for next game
        pool.reset();
    }
}

} // namespace selfplay
