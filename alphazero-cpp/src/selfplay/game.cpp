#include "selfplay/game.hpp"
#include "encoding/position_encoder.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace selfplay {

void SelfPlayGame::reset() {
    board_ = chess::Board();  // Starting position
    move_count_ = 0;
    position_history_.clear();  // Clear position history
}

chess::Move SelfPlayGame::select_move(const std::vector<int32_t>& visit_counts, int move_number) {
    // Determine temperature based on move number
    float temperature = (move_number < config_.temp_threshold) ? config_.temperature : 0.0f;

    // Get legal moves
    chess::Movelist legal_moves;
    chess::movegen::legalmoves(legal_moves, board_);

    if (legal_moves.empty()) {
        throw std::runtime_error("No legal moves available");
    }

    // Apply temperature to visit counts
    std::vector<float> probabilities = apply_temperature(visit_counts, temperature);

    // Sample move based on probabilities
    std::discrete_distribution<int> dist(probabilities.begin(), probabilities.end());
    int selected_idx = dist(rng_);

    // Find the corresponding move
    int current_idx = 0;
    for (const auto& move : legal_moves) {
        if (visit_counts[current_idx] > 0) {
            if (selected_idx == 0) {
                return move;
            }
            selected_idx--;
        }
        current_idx++;
    }

    // Fallback: return most visited move
    int max_visits = 0;
    chess::Move best_move = legal_moves[0];
    current_idx = 0;
    for (const auto& move : legal_moves) {
        if (visit_counts[current_idx] > max_visits) {
            max_visits = visit_counts[current_idx];
            best_move = move;
        }
        current_idx++;
    }

    return best_move;
}

std::vector<float> SelfPlayGame::visit_counts_to_policy(const std::vector<int32_t>& visit_counts) {
    std::vector<float> policy(encoding::MoveEncoder::POLICY_SIZE, 0.0f);

    // Get legal moves and their indices
    chess::Movelist legal_moves;
    chess::movegen::legalmoves(legal_moves, board_);

    for (const auto& move : legal_moves) {
        // Convert move to policy index
        std::string uci = chess::uci::moveToUci(move);
        int index = encoding::MoveEncoder::move_to_index(uci, board_);

        if (index >= 0 && index < encoding::MoveEncoder::POLICY_SIZE) {
            // Use raw visit counts as policy targets (will be normalized during training)
            policy[index] = static_cast<float>(visit_counts[index]);
        }
    }

    // Normalize to sum to 1
    float total = std::accumulate(policy.begin(), policy.end(), 0.0f);
    if (total > 0.0f) {
        for (float& p : policy) {
            p /= total;
        }
    }

    return policy;
}

std::vector<float> SelfPlayGame::apply_temperature(const std::vector<int32_t>& visit_counts, float temperature) {
    std::vector<float> probabilities(encoding::MoveEncoder::POLICY_SIZE, 0.0f);

    if (temperature < 1e-6f) {
        // Greedy selection: argmax
        int max_visits = *std::max_element(visit_counts.begin(), visit_counts.end());
        for (size_t i = 0; i < visit_counts.size(); ++i) {
            probabilities[i] = (visit_counts[i] == max_visits) ? 1.0f : 0.0f;
        }
    } else {
        // Temperature scaling
        float sum = 0.0f;
        for (size_t i = 0; i < visit_counts.size(); ++i) {
            if (visit_counts[i] > 0) {
                // Apply temperature: p_i = (n_i)^(1/T)
                probabilities[i] = std::pow(static_cast<float>(visit_counts[i]), 1.0f / temperature);
                sum += probabilities[i];
            }
        }

        // Normalize
        if (sum > 0.0f) {
            for (float& p : probabilities) {
                p /= sum;
            }
        }
    }

    return probabilities;
}

bool SelfPlayGame::is_game_over() const {
    // NOTE: With position history encoding implemented, the model can now see
    // the last 8 positions and learn to avoid repetitions. We can safely
    // re-enable threefold repetition and fifty-move rule detection.

    // Use chess-library's full game over detection
    auto [reason, _] = board_.isGameOver();
    return reason != chess::GameResultReason::NONE;
}

chess::GameResult SelfPlayGame::get_game_result() const {
    // Use chess-library's standard result detection
    auto [reason, result] = board_.isGameOver();

    if (reason != chess::GameResultReason::NONE) {
        return result;
    }

    // Check for max moves reached (declared as draw)
    if (move_count_ >= config_.max_moves) {
        return chess::GameResult::DRAW;
    }

    // Game not over yet
    return chess::GameResult::NONE;
}

} // namespace selfplay
