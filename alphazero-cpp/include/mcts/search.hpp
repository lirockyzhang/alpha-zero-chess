#pragma once

#include "node.hpp"
#include "node_pool.hpp"
#include "../third_party/chess-library/include/chess.hpp"
#include <vector>
#include <cmath>
#include <random>

namespace mcts {

// MCTS search configuration
struct SearchConfig {
    // PUCT exploration constant (typically 1.0-2.0)
    float c_puct = 1.5f;

    // Dirichlet noise for root exploration
    float dirichlet_alpha = 0.3f;
    float dirichlet_epsilon = 0.25f;

    // Temperature for move selection
    float temperature = 1.0f;

    // Number of simulations per search
    int num_simulations = 800;

    // Virtual loss value for parallel MCTS
    int virtual_loss = 3;
};

// MCTS search engine
class MCTSSearch {
public:
    MCTSSearch(NodePool& pool, const SearchConfig& config = SearchConfig())
        : pool_(pool)
        , config_(config)
        , rng_(std::random_device{}())
    {}

    // Run MCTS search from a given position
    // Returns the root node after search completes
    Node* search(const chess::Board& root_position,
                 const std::vector<float>& policy,
                 float value);

    // Select best move from root node using visit counts
    chess::Move select_move(Node* root, float temperature = 1.0f);

    // Get visit count distribution for all children (for training)
    std::vector<float> get_policy_target(Node* root);

private:
    // MCTS phases
    Node* select(Node* node, chess::Board& board);
    void expand(Node* node, const chess::Board& board,
                const std::vector<float>& policy, float value);
    void backpropagate(Node* node, float value);

    // PUCT formula for node selection
    float puct_score(const Node* parent, const Node* child) const;

    // Add Dirichlet noise to root node for exploration
    void add_dirichlet_noise(Node* root);

    NodePool& pool_;
    SearchConfig config_;
    std::mt19937 rng_;
};

// Inline implementation of PUCT score
inline float MCTSSearch::puct_score(const Node* parent, const Node* child) const {
    uint32_t parent_visits = parent->visit_count.load(std::memory_order_relaxed);
    uint32_t child_visits = child->visit_count.load(std::memory_order_relaxed);
    int16_t virtual_loss = child->virtual_loss.load(std::memory_order_relaxed);

    // Q-value with virtual loss
    float parent_q = parent->q_value(0.0f);  // Root has no parent
    float q = child->q_value(parent_q);

    // Prior probability
    float prior = child->prior();

    // PUCT formula: Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
    float exploration = config_.c_puct * prior * std::sqrt(static_cast<float>(parent_visits))
                       / (1.0f + child_visits + virtual_loss);

    return q + exploration;
}

} // namespace mcts
