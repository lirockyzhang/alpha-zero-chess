#pragma once

#include "node.hpp"
#include "node_pool.hpp"
#include "../third_party/chess-library/include/chess.hpp"
#include "../encoding/position_encoder.hpp"
#include <array>
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
    float risk_beta = 0.0f;         // ERM risk sensitivity: >0 risk-seeking, <0 risk-averse, 0 neutral
    float fpu_base = 0.3f;          // Dynamic FPU: penalty = fpu_base * sqrt(1 - prior)

    // Gumbel Top-k Sequential Halving (Danihelka et al. 2022)
    bool use_gumbel = false;        // Use Gumbel SH at root instead of PUCT+Dirichlet
    int gumbel_top_k = 16;         // Initial m for Sequential Halving
    float gumbel_c_visit = 50.0f;  // sigma() constant for value transformation
    float gumbel_c_scale = 1.0f;   // sigma() scale for value transformation
};

// Convert WDL probabilities to a scalar value
// pw = P(win), pd = P(draw), pl = P(loss)
// Returns value in [-1, 1]
// Risk adjustment happens at node level (q_value_risk), not here
inline float wdl_to_value(float pw, float pd, float pl) {
    return pw - pl;
}

// State for Gumbel Top-k Sequential Halving root search
struct GumbelState {
    std::vector<float> gumbel;        // g(a) noise per root child
    std::vector<float> logit;         // log(prior(a)) per root child
    std::vector<Node*> children;      // All root children (indexed)
    std::vector<int> active_indices;  // Active set for current SH phase
    int num_phases = 0;
    int current_phase = 0;
    int sims_per_action = 0;          // Budget per action this phase
    int round_robin_counter = 0;      // Cycles through active set
    int phase_sims_done = 0;          // Total sims completed in current phase
    int total_budget = 0;
    int total_sims_used = 0;
    float c_visit = 50.0f;           // sigma() constant
    float c_scale = 1.0f;            // sigma() scale
    bool all_phases_done = false;
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
    // wdl: optional batch_size x 3 WDL probs (for root WDL accumulation)
    void update_leaves(const std::vector<std::vector<float>>& policies,
                       const std::vector<float>& values,
                       const float* wdl = nullptr);

    // Check if search is complete
    bool is_search_complete() const {
        if (config_.use_gumbel) {
            return gumbel_.all_phases_done || simulations_completed_ >= config_.num_simulations;
        }
        return simulations_completed_ >= config_.num_simulations;
    }

    // Get the root node after search
    Node* get_root() const { return root_; }

    // Get number of simulations completed
    int get_simulations_completed() const { return simulations_completed_; }

    // Tree depth tracking (per-search min/max across all simulations)
    int get_max_depth() const { return max_depth_; }
    int get_min_depth() const { return has_depth_ ? min_depth_ : 0; }

    // Get visit counts as policy vector (4672 dimensions)
    std::vector<int32_t> get_visit_counts() const;

    // Gumbel Top-k Sequential Halving: improved policy training target (4672 dims)
    // pi_improved(a) = softmax(logit(a) + sigma(Q_completed(a))) for all children
    std::vector<float> get_improved_policy() const;

    // Gumbel Top-k Sequential Halving: SH winner move
    chess::Move get_gumbel_action() const;

    // Get root WDL distribution (average of leaf WDL propagated to root)
    // Returns {P(win), P(draw), P(loss)} from root's perspective, or {0,0,0} if no WDL data
    std::array<float, 3> get_root_wdl() const {
        if (root_wdl_count_ == 0) return {0.0f, 0.0f, 0.0f};
        float inv = 1.0f / root_wdl_count_;
        return {root_wdl_sum_[0] * inv, root_wdl_sum_[1] * inv, root_wdl_sum_[2] * inv};
    }

    // Compute risk-adjusted root value using LogSumExp over children Q-values.
    // V_risk = (1/beta) * log( sum_a exp(beta * Q_risk_a) )  for visited children.
    // Uses the "shift trick" for numerical stability. Returns 0 if beta == 0 or no visited children.
    // Children Q-values use q_value_risk() for per-node variance-adjusted scores.
    float get_root_risk_value(float beta) const;

    // Cancel pending evaluations (call when evaluation times out)
    void cancel_pending_evaluations() {
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
    // wdl: optional batch_size x 3 WDL probs (for root WDL accumulation)
    void update_prev_leaves(const std::vector<std::vector<float>>& policies,
                           const std::vector<float>& values,
                           const float* wdl = nullptr);

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

    // Backpropagate value up the tree (scalar only — terminal nodes, already-expanded nodes)
    void backpropagate(Node* node, float value);

    // Backpropagate value + WDL up the tree (NN-evaluated leaves with WDL probs)
    // pw/pd/pl are from the evaluated leaf's perspective; flipped at each level
    void backpropagate(Node* node, float value, float pw, float pd, float pl);

    // PUCT score calculation
    float puct_score(const Node* parent, const Node* child) const;

    // Add Dirichlet noise to root
    void add_dirichlet_noise(Node* root);

    // Gumbel Top-k Sequential Halving internal methods
    void init_gumbel();                           // Setup noise, top-m, SH phases
    Node* get_gumbel_target_child();              // Round-robin root child selection
    void advance_gumbel_sim();                    // Track phase-level progress
    void complete_gumbel_phase();                 // Score, prune, advance phase
    float completed_q(const Node* child) const;   // Q_hat(a) with V(root) interpolation
    float sigma_q(float q, float min_q, float max_q, int max_visits) const; // Value transform

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

    // Root WDL accumulator: average of leaf WDL predictions propagated to root
    // Reset in init_search(), accumulated in backpropagate(node, value, pw, pd, pl)
    float root_wdl_sum_[3] = {0.0f, 0.0f, 0.0f};
    int root_wdl_count_ = 0;

    // Tree depth tracking (per-search min/max across all simulations)
    int max_depth_ = 0;
    int min_depth_ = 0;
    bool has_depth_ = false;

    // Gumbel Top-k Sequential Halving state
    GumbelState gumbel_;
    float root_value_ = 0.0f;  // V(root) from initial NN eval, for completed_q
};

} // namespace mcts
