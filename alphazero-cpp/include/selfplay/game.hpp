#pragma once

#include "../third_party/chess-library/include/chess.hpp"
#include "../mcts/search.hpp"
#include "../mcts/node_pool.hpp"
#include "../encoding/move_encoder.hpp"
#include <array>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <deque>

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
                   uint8_t num_hist = 0) {
        states.emplace_back();
        states.back().observation = obs;
        states.back().policy = pol;
        states.back().mcts_wdl = wdl;
        states.back().soft_value = sv;
        states.back().fen = fen;
        states.back().history_hashes = hist_hashes;
        states.back().num_history = num_hist;
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

// Configuration for self-play game execution
struct SelfPlayConfig {
    int num_simulations = 800;          // MCTS simulations per move
    int batch_size = 256;               // MCTS batch size
    float c_puct = 1.5f;                // PUCT exploration constant
    float temperature = 1.0f;           // Temperature for move selection (early game)
    float temp_threshold = 30.0f;       // Move number to switch to greedy selection
    int max_moves = 10000;              // Safety cap (real chess rules terminate games first)
    bool add_dirichlet_noise = true;    // Add Dirichlet noise to root
    float dirichlet_alpha = 0.3f;       // Dirichlet noise alpha
    float dirichlet_epsilon = 0.25f;    // Dirichlet noise weight
};

// Self-play game executor
// Plays a single game using MCTS and collects training data
class SelfPlayGame {
public:
    SelfPlayGame(mcts::NodePool& pool, const SelfPlayConfig& config = SelfPlayConfig())
        : pool_(pool)
        , config_(config)
        , rng_(std::random_device{}())
    {}

    // Play a game and return the trajectory
    // neural_evaluator is a callback: (obs_batch, num_leaves) -> (policies, values)
    template<typename NeuralEvaluator>
    GameTrajectory play_game(NeuralEvaluator& neural_evaluator);

    // Get the current board state
    const chess::Board& get_board() const { return board_; }

    // Reset for new game
    void reset();

private:
    // Select a move using temperature-based sampling
    chess::Move select_move(const std::vector<int32_t>& visit_counts, int move_number);

    // Convert visit counts to policy distribution
    std::vector<float> visit_counts_to_policy(const std::vector<int32_t>& visit_counts);

    // Apply temperature to visit counts
    std::vector<float> apply_temperature(const std::vector<int32_t>& visit_counts, float temperature);

    // Check if game is over
    bool is_game_over() const;

    // Get game result
    chess::GameResult get_game_result() const;

    // Get position history (for encoding)
    const std::deque<chess::Board>& get_position_history() const { return position_history_; }

    mcts::NodePool& pool_;
    SelfPlayConfig config_;
    chess::Board board_;
    std::mt19937 rng_;
    int move_count_;
    std::deque<chess::Board> position_history_;  // Last 8 positions for history encoding
};

// Template implementation
template<typename NeuralEvaluator>
GameTrajectory SelfPlayGame::play_game(NeuralEvaluator& neural_evaluator) {
    GameTrajectory trajectory;
    trajectory.reserve(80);  // Typical game length

    reset();

    while (!is_game_over() && move_count_ < config_.max_moves) {
        // Create MCTS search for this position
        mcts::BatchSearchConfig search_config;
        search_config.num_simulations = config_.num_simulations;
        search_config.batch_size = config_.batch_size;
        search_config.c_puct = config_.c_puct;
        search_config.dirichlet_alpha = config_.dirichlet_alpha;
        search_config.dirichlet_epsilon = config_.dirichlet_epsilon;

        mcts::MCTSSearch search(pool_, search_config);

        // Get initial evaluation from neural network
        std::vector<float> obs_buffer(encoding::PositionEncoder::TOTAL_SIZE);
        std::vector<float> mask_buffer(encoding::MoveEncoder::POLICY_SIZE);

        // Convert deque to vector for position history
        std::vector<chess::Board> history_vec(position_history_.begin(), position_history_.end());

        encoding::PositionEncoder::encode_to_buffer(board_, obs_buffer.data(), history_vec);

        // Get NN evaluation for root
        std::vector<std::vector<float>> root_policies;
        std::vector<float> root_values;
        neural_evaluator(obs_buffer.data(), 1, root_policies, root_values);

        // Initialize search with position history
        search.init_search(board_, root_policies[0], root_values[0], history_vec);

        // Run MCTS with batched leaf evaluation
        while (!search.is_search_complete()) {
            // Collect leaves
            std::vector<float> leaf_obs(config_.batch_size * encoding::PositionEncoder::TOTAL_SIZE);
            std::vector<float> leaf_masks(config_.batch_size * encoding::MoveEncoder::POLICY_SIZE);

            int num_leaves = search.collect_leaves(leaf_obs.data(), leaf_masks.data(), config_.batch_size);

            if (num_leaves == 0) break;

            // Evaluate leaves with neural network
            std::vector<std::vector<float>> policies;
            std::vector<float> values;
            neural_evaluator(leaf_obs.data(), num_leaves, policies, values);

            // Update search with evaluations
            search.update_leaves(policies, values);
        }

        // Get visit counts
        std::vector<int32_t> visit_counts = search.get_visit_counts();

        // Save state to trajectory
        std::vector<float> policy = visit_counts_to_policy(visit_counts);
        trajectory.add_state(obs_buffer, policy);

        // Select and make move
        chess::Move move = select_move(visit_counts, move_count_);

        // Record move for PGN export
        trajectory.moves_uci.push_back(chess::uci::moveToUci(move));

        // Add current position to history before making move
        position_history_.push_back(board_);
        if (position_history_.size() > 8) {
            position_history_.pop_front();  // Keep only last 8 positions
        }

        board_.makeMove(move);
        move_count_++;
    }

    // Set game outcomes
    chess::GameResult result = get_game_result();
    trajectory.set_outcomes(result);

    return trajectory;
}

} // namespace selfplay
