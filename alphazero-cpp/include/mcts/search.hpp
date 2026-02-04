#pragma once

#include "node.hpp"
#include "node_pool.hpp"
#include "../third_party/chess-library/include/chess.hpp"
#include "../encoding/position_encoder.hpp"
#include <vector>
#include <string>
#include <atomic>
#include <random>

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

// MCTS search that collects leaves for batch evaluation
// This implements the proper AlphaZero pattern where every leaf
// gets a neural network evaluation
class MCTSSearch {
public:
    MCTSSearch(NodePool& pool, const BatchSearchConfig& config = BatchSearchConfig())
        : pool_(pool)
        , config_(config)
        , rng_(std::random_device{}())
    {}

    // Initialize search with root position and initial NN evaluation
    // Returns the root node
    // position_history: Last N positions (up to 8) for history encoding
    Node* init_search(const chess::Board& root_position,
                      const std::vector<float>& root_policy,
                      float root_value,
                      const std::vector<chess::Board>& position_history = {});

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

    // Cancel pending evaluations (call when evaluation times out)
    // This removes virtual losses and clears in-flight tracking so search can continue
    void cancel_pending_evaluations() {
        // Remove virtual losses from all pending paths
        for (const auto& eval : pending_evals_) {
            remove_virtual_losses_for_path(eval.node);
        }
        pending_evals_.clear();
        simulations_in_flight_ = 0;
    }

    // =========================================================================
    // Async Double-Buffer Pipeline
    // =========================================================================
    // Enables CPU leaf collection to overlap with GPU evaluation of previous batch.
    // Workers collect into the "collection buffer" while GPU processes the "evaluation buffer".

    // Collect leaves into the collection buffer (async: doesn't touch pending_evals_)
    // Returns the number of leaves collected
    int collect_leaves_async(float* obs_buffer, float* mask_buffer, int max_batch_size);

    // Get the size of the previous batch (evaluation buffer)
    int get_prev_batch_size() const;

    // Backpropagate results for the evaluation buffer (previous batch)
    void update_prev_leaves(const std::vector<std::vector<float>>& policies,
                           const std::vector<float>& values);

    // Cancel previous batch: remove virtual losses from evaluation buffer leaves
    void cancel_prev_pending();

    // Cancel current collection batch: remove virtual losses from collection buffer leaves
    void cancel_collection_pending();

    // Swap buffers: collection buffer becomes evaluation buffer, old eval buffer is cleared
    void commit_and_swap();

    // Double buffering support (low-level)
    void start_next_batch_collection();
    std::vector<PendingEval>& get_collection_buffer();
    const std::vector<PendingEval>& get_evaluation_buffer() const;
    void swap_buffers();

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

    // Remove virtual losses along path from node to root (for cancelled evaluations)
    void remove_virtual_losses_for_path(Node* node);

    NodePool& pool_;
    BatchSearchConfig config_;
    std::mt19937 rng_;

    Node* root_ = nullptr;
    chess::Board root_position_;
    std::vector<chess::Board> position_history_;  // Position history for encoding

    // Pending evaluations waiting for NN
    std::vector<PendingEval> pending_evals_;

    // Double buffering for GPU/CPU overlap
    // While GPU processes buffer A, CPU collects leaves into buffer B
    std::vector<PendingEval> buffer_a_;
    std::vector<PendingEval> buffer_b_;
    bool using_buffer_a_ = true;  // Which buffer is currently being filled

    // Simulation tracking
    int simulations_completed_ = 0;
    int simulations_in_flight_ = 0;
};

} // namespace mcts
