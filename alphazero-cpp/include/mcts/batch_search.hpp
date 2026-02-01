#pragma once

#include "node.hpp"
#include "node_pool.hpp"
#include "search.hpp"
#include "../third_party/chess-library/include/chess.hpp"
#include "../encoding/position_encoder.hpp"
#include <vector>
#include <string>
#include <atomic>

namespace mcts {

// Pending evaluation request for a leaf node
struct PendingEval {
    Node* node;                    // Leaf node to expand
    chess::Board board;            // Board state at this node
    int simulation_idx;            // Which simulation this belongs to

    PendingEval() : node(nullptr), simulation_idx(-1) {}
    PendingEval(Node* n, const chess::Board& b, int idx)
        : node(n), board(b), simulation_idx(idx) {}
};

// Configuration for batched search
struct BatchSearchConfig {
    int num_simulations = 800;      // Total simulations per search
    int batch_size = 256;           // Max leaves to batch together
    float c_puct = 1.5f;            // PUCT exploration constant
    float dirichlet_alpha = 0.3f;   // Dirichlet noise alpha
    float dirichlet_epsilon = 0.25f; // Dirichlet noise weight
    float batch_threshold = 0.9f;   // Dispatch when this fraction ready
    int batch_timeout_ms = 20;      // Max wait time for batch
};

// Batched MCTS search that collects leaves for batch evaluation
// This implements the proper AlphaZero pattern where every leaf
// gets a neural network evaluation
class BatchedMCTSSearch {
public:
    BatchedMCTSSearch(NodePool& pool, const BatchSearchConfig& config = BatchSearchConfig())
        : pool_(pool)
        , config_(config)
        , rng_(std::random_device{}())
    {}

    // Initialize search with root position and initial NN evaluation
    // Returns the root node
    Node* init_search(const chess::Board& root_position,
                      const std::vector<float>& root_policy,
                      float root_value);

    // Collect pending leaf evaluations into a batch
    // Returns the number of leaves collected
    // Observations are written to obs_buffer (shape: batch_size x 8 x 8 x 119)
    // Legal masks are written to mask_buffer (shape: batch_size x 4672)
    int collect_leaves(float* obs_buffer, float* mask_buffer, int max_batch_size);

    // Update leaves with neural network evaluation results
    // policies: batch_size x 4672 policy vectors
    // values: batch_size values
    void update_leaves(const std::vector<std::vector<float>>& policies,
                       const std::vector<float>& values);

    // Check if search is complete
    bool is_search_complete() const {
        return simulations_completed_ >= config_.num_simulations;
    }

    // Get the root node after search
    Node* get_root() const { return root_; }

    // Get number of simulations completed
    int get_simulations_completed() const { return simulations_completed_; }

    // Get visit counts as policy vector (4672 dimensions)
    std::vector<int32_t> get_visit_counts() const;

private:
    // Run selection phase for multiple simulations
    // Collects leaves that need evaluation
    void run_selection_phase(int num_sims);

    // Selection: traverse tree to leaf
    Node* select(Node* node, chess::Board& board);

    // Expand node with policy and value
    void expand(Node* node, const chess::Board& board,
                const std::vector<float>& policy, float value);

    // Backpropagate value up the tree
    void backpropagate(Node* node, float value);

    // PUCT score calculation
    float puct_score(const Node* parent, const Node* child) const;

    // Add Dirichlet noise to root
    void add_dirichlet_noise(Node* root);

    NodePool& pool_;
    BatchSearchConfig config_;
    std::mt19937 rng_;

    Node* root_ = nullptr;
    chess::Board root_position_;

    // Pending evaluations waiting for NN
    std::vector<PendingEval> pending_evals_;

    // Simulation tracking
    int simulations_completed_ = 0;
    int simulations_in_flight_ = 0;
};

} // namespace mcts
