#include "selfplay/coordinator.hpp"

namespace selfplay {

void SelfPlayCoordinator::stop() {
    if (!running_.load()) {
        return;
    }

    running_ = false;

    // Wait for all worker threads to finish
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    workers_.clear();
    completed_games_.set_done();
}

std::vector<GameTrajectory> SelfPlayCoordinator::get_completed_games() {
    std::vector<GameTrajectory> games;

    // Collect all available games from queue
    GameTrajectory trajectory;
    while (completed_games_.try_pop(trajectory)) {
        games.push_back(std::move(trajectory));
    }

    return games;
}

SelfPlayStatsSnapshot SelfPlayCoordinator::get_stats() const {
    SelfPlayStatsSnapshot snapshot;
    snapshot.games_completed = stats_.games_completed.load();
    snapshot.total_moves = stats_.total_moves.load();
    snapshot.white_wins = stats_.white_wins.load();
    snapshot.black_wins = stats_.black_wins.load();
    snapshot.draws = stats_.draws.load();
    snapshot.total_game_time = stats_.total_game_time.load();
    return snapshot;
}

} // namespace selfplay
