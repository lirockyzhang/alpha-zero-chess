#pragma once

#include "../third_party/chess-library/include/chess.hpp"
#include "../encoding/position_encoder.hpp"
#include "../encoding/move_encoder.hpp"
#include <array>
#include <cstdint>
#include <vector>
#include <string>

namespace selfplay {

// Single game state for self-play trajectory
struct GameState {
    std::vector<float> observation;     // Position encoding (8 x 8 x 123)
    std::vector<float> policy;          // MCTS visit counts as policy (4672)
    float value;                        // Game outcome from this player's perspective
    std::array<float, 3> mcts_wdl;     // MCTS root WDL distribution {P(win), P(draw), P(loss)} from side-to-move
    float soft_value;                   // ERM risk-adjusted root value; 0 if disabled
    std::string fen;                           // Board position as FEN string (for reanalysis)
    std::array<uint64_t, 8> history_hashes{};  // Zobrist hashes [T-1..T-8] for repetition detection
    uint8_t num_history{0};                    // How many hashes are valid
    std::array<std::string, 8> history_fens{}; // FENs [T-1..T-8], most recent first

    GameState() : observation(encoding::PositionEncoder::TOTAL_SIZE, 0.0f),
                  policy(encoding::MoveEncoder::POLICY_SIZE, 0.0f),
                  value(0.0f), mcts_wdl{0.0f, 0.0f, 0.0f}, soft_value(0.0f) {}
};

// Complete self-play game trajectory
struct GameTrajectory {
    std::vector<GameState> states;      // All states in the game
    std::vector<std::string> moves_uci; // UCI move strings (e.g. "e2e4", "g8f6")
    chess::GameResult result;           // Final game result
    chess::GameResultReason result_reason{chess::GameResultReason::NONE}; // Draw/end reason
    int num_moves;                      // Total number of moves

    GameTrajectory() : result(chess::GameResult::NONE), num_moves(0) {}

    // Reserve capacity for typical game length
    void reserve(int capacity = 80) {
        states.reserve(capacity);
        moves_uci.reserve(capacity);
    }

    // Add a state to the trajectory
    void add_state(const std::vector<float>& obs, const std::vector<float>& pol,
                   const std::array<float, 3>& wdl = {0.0f, 0.0f, 0.0f},
                   float sv = 0.0f,
                   const std::string& fen = "",
                   const std::array<uint64_t, 8>& hist_hashes = {},
                   uint8_t num_hist = 0,
                   const std::array<std::string, 8>& hist_fens = {}) {
        states.emplace_back();
        states.back().observation = obs;
        states.back().policy = pol;
        states.back().mcts_wdl = wdl;
        states.back().soft_value = sv;
        states.back().fen = fen;
        states.back().history_hashes = hist_hashes;
        states.back().num_history = num_hist;
        states.back().history_fens = hist_fens;
        num_moves++;
    }

    // Set game outcome for all states (from each player's perspective)
    // result should be WIN/LOSE/DRAW from White's perspective
    // Training labels are always pure: draws = 0.0 (risk_beta is search-time only)
    void set_outcomes(chess::GameResult game_result) {
        result = game_result;

        // Convert game result to values from each player's perspective
        for (size_t i = 0; i < states.size(); ++i) {
            bool white_to_move = (i % 2 == 0);

            if (game_result == chess::GameResult::WIN) {
                // White won
                states[i].value = white_to_move ? 1.0f : -1.0f;
            } else if (game_result == chess::GameResult::LOSE) {
                // White lost (Black won)
                states[i].value = white_to_move ? -1.0f : 1.0f;
            } else {
                // Draw — pure training label (risk_beta not baked into labels)
                states[i].value = 0.0f;
            }
        }
    }
};

} // namespace selfplay
