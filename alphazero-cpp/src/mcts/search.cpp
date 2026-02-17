#include "mcts/search.hpp"
#include "encoding/move_encoder.hpp"
#include "encoding/position_encoder.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>

namespace mcts {

Node* MCTSSearch::init_search(const chess::Board& root_position,
                                      const std::vector<float>& root_policy,
                                      float root_value,
                                      const std::vector<chess::Board>& position_history) {
    // Reset state
    root_position_ = root_position;
    position_history_ = position_history;  // Store position history for encoding
    pending_evals_.clear();
    simulations_completed_ = 0;
    simulations_in_flight_ = 0;
    max_depth_ = 0;
    min_depth_ = 0;
    has_depth_ = false;

    // Reset root WDL accumulator (accumulated per-move, reported after search)
    root_wdl_sum_[0] = root_wdl_sum_[1] = root_wdl_sum_[2] = 0.0f;
    root_wdl_count_ = 0;

    // Store root value for Gumbel completed_q
    root_value_ = root_value;

    // Create and expand root node
    root_ = pool_.allocate();
    expand(root_, root_position_, root_policy, root_value);

    if (config_.use_gumbel) {
        // Gumbel noise replaces Dirichlet noise
        init_gumbel();
    } else {
        // Standard AlphaZero: add Dirichlet noise to root for exploration
        add_dirichlet_noise(root_);
    }

    return root_;
}

int MCTSSearch::collect_leaves(float* obs_buffer, float* mask_buffer, int max_batch_size) {
    // First, run selection to collect leaves that need evaluation
    int leaves_needed = std::min(max_batch_size, config_.num_simulations - simulations_completed_ - simulations_in_flight_);

    if (leaves_needed <= 0) {
        return 0;
    }

    run_selection_phase(leaves_needed);

    // Encode pending evaluations into buffers
    int batch_size = std::min(static_cast<int>(pending_evals_.size()), max_batch_size);

    for (int i = 0; i < batch_size; ++i) {
        const auto& eval = pending_evals_[i];

        // Encode position to observation buffer (NHWC format: 8 x 8 x 122)
        // Pass position history for history plane encoding
        encoding::PositionEncoder::encode_to_buffer(
            eval.board,
            obs_buffer + i * encoding::PositionEncoder::TOTAL_SIZE,
            position_history_
        );

        // Encode legal mask
        float* mask_ptr = mask_buffer + i * encoding::MoveEncoder::POLICY_SIZE;
        std::fill(mask_ptr, mask_ptr + encoding::MoveEncoder::POLICY_SIZE, 0.0f);

        chess::Movelist moves;
        chess::movegen::legalmoves(moves, eval.board);

        for (const auto& move : moves) {
            int idx = encoding::MoveEncoder::move_to_index(move, eval.board);
            if (idx >= 0 && idx < encoding::MoveEncoder::POLICY_SIZE) {
                mask_ptr[idx] = 1.0f;
            }
        }
    }

    simulations_in_flight_ = batch_size;
    return batch_size;
}

void MCTSSearch::update_leaves(const std::vector<std::vector<float>>& policies,
                                       const std::vector<float>& values,
                                       const float* wdl) {
    int batch_size = std::min({
        static_cast<int>(pending_evals_.size()),
        static_cast<int>(policies.size()),
        static_cast<int>(values.size())
    });

    for (int i = 0; i < batch_size; ++i) {
        const auto& eval = pending_evals_[i];
        Node* leaf = eval.node;

        // Expand the leaf with NN policy and value
        expand(leaf, eval.board, policies[i], values[i]);

        // Backpropagate the value (with WDL if available)
        if (wdl != nullptr) {
            backpropagate(leaf, values[i], wdl[i*3+0], wdl[i*3+1], wdl[i*3+2]);
        } else {
            backpropagate(leaf, values[i]);
        }

        simulations_completed_++;
        if (config_.use_gumbel) advance_gumbel_sim();
    }

    // If we received fewer results than pending evaluations,
    // remove virtual losses from the remaining paths to prevent tree corruption
    for (int i = batch_size; i < static_cast<int>(pending_evals_.size()); ++i) {
        remove_virtual_losses_for_path(pending_evals_[i].node);
    }

    // Clear pending evaluations
    pending_evals_.clear();
    simulations_in_flight_ = 0;
}

void MCTSSearch::run_selection_phase(int num_sims) {
    pending_evals_.clear();
    pending_evals_.reserve(num_sims);

    for (int sim = 0; sim < num_sims; ++sim) {
        // Copy board state for this simulation
        chess::Board board = root_position_;

        // Select leaf node
        Node* leaf = select(root_, board);

        // If leaf is terminal, backpropagate immediately with deterministic WDL
        if (leaf->is_terminal()) {
            auto [reason, result] = board.isGameOver();
            float value = 0.0f;
            float pw = 0.0f, pd = 0.0f, pl = 0.0f;

            if (result == chess::GameResult::LOSE) {
                // Current side to move lost (checkmate)
                value = -1.0f;
                pl = 1.0f;  // Loss from current side's perspective
            } else if (result == chess::GameResult::WIN) {
                value = 1.0f;
                pw = 1.0f;
            } else {
                // Draw — value=0, variance=0 → Q_beta=0 regardless of risk_beta
                value = 0.0f;
                pd = 1.0f;
            }

            backpropagate(leaf, value, pw, pd, pl);
            simulations_completed_++;
            if (config_.use_gumbel) advance_gumbel_sim();
        } else if (!leaf->is_expanded()) {
            // Leaf needs NN evaluation - add to pending
            pending_evals_.emplace_back(leaf, board, sim);
        } else {
            // Already expanded (race condition in parallel) - just backpropagate
            // No WDL info available for this path (use scalar-only overload)
            float value = leaf->q_value(0.0f, config_.fpu_base);
            backpropagate(leaf, value);
            simulations_completed_++;
            if (config_.use_gumbel) advance_gumbel_sim();
        }
    }
}

Node* MCTSSearch::select(Node* node, chess::Board& board) {
    int depth = 0;
    while (node->is_expanded() && !node->is_terminal()) {
        // Add virtual loss to prevent other threads from selecting same path
        if (config_.use_virtual_loss) node->add_virtual_loss();

        Node* best_child = nullptr;

        // Gumbel mode: round-robin root child selection (depth 0 only)
        if (depth == 0 && config_.use_gumbel && !gumbel_.all_phases_done) {
            best_child = get_gumbel_target_child();
        } else {
            // Standard PUCT — used for non-root nodes and PUCT mode
            float best_score = -std::numeric_limits<float>::infinity();

            for (Node* child = node->first_child; child != nullptr; child = child->next_sibling) {
                float score = puct_score(node, child);
                if (score > best_score) {
                    best_score = score;
                    best_child = child;
                }
            }
        }

        if (best_child == nullptr) {
            // No children (shouldn't happen for expanded non-terminal node)
            break;
        }

        // Apply move to board
        board.makeMove(best_child->move);
        node = best_child;
        depth++;
    }

    // Add VL to the leaf/terminal node for symmetry with backpropagate()
    // and remove_virtual_losses_for_path(), which both remove VL from
    // every node including the leaf.
    if (config_.use_virtual_loss) node->add_virtual_loss();

    // Update depth tracking
    if (depth > max_depth_) max_depth_ = depth;
    if (!has_depth_ || depth < min_depth_) min_depth_ = depth;
    has_depth_ = true;

    return node;
}

void MCTSSearch::expand(Node* node, const chess::Board& board,
                                const std::vector<float>& policy, float value) {
    // Check if already expanded (race condition)
    if (node->is_expanded()) {
        return;
    }

    // Check for terminal state
    auto [reason, result] = board.isGameOver();
    if (reason != chess::GameResultReason::NONE) {
        node->set_terminal(true);
        node->set_expanded(true);
        return;
    }

    // Generate legal moves
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    if (moves.empty()) {
        node->set_terminal(true);
        node->set_expanded(true);
        return;
    }

    // Create child nodes for each legal move
    Node* prev_child = nullptr;
    float prior_sum = 0.0f;

    for (const auto& move : moves) {
        Node* child = pool_.allocate();
        child->parent = node;
        child->move = move;

        // Look up prior from policy vector using move encoder
        int policy_index = encoding::MoveEncoder::move_to_index(move, board);
        float prior = 0.0f;

        if (policy_index >= 0 && policy_index < static_cast<int>(policy.size())) {
            prior = policy[policy_index];
        }

        prior = std::max(prior, 0.0f);
        prior_sum += prior;

        child->set_prior(prior);

        // Link child into sibling list
        if (prev_child == nullptr) {
            node->first_child = child;
        } else {
            prev_child->next_sibling = child;
        }
        prev_child = child;
    }

    // Normalize priors
    if (prior_sum > 0.0f && std::abs(prior_sum - 1.0f) > 0.01f) {
        for (Node* child = node->first_child; child != nullptr; child = child->next_sibling) {
            float normalized_prior = child->prior() / prior_sum;
            child->set_prior(normalized_prior);
        }
    } else if (prior_sum == 0.0f) {
        float uniform_prior = 1.0f / moves.size();
        for (Node* child = node->first_child; child != nullptr; child = child->next_sibling) {
            child->set_prior(uniform_prior);
        }
    }

    node->set_expanded(true);
}

void MCTSSearch::backpropagate(Node* node, float value) {
    while (node != nullptr) {
        // Remove virtual loss
        if (config_.use_virtual_loss) node->remove_virtual_loss();

        // Update node value
        if (node->parent == nullptr) {
            // Root node - use release semantics
            node->update_root(value);
        } else {
            node->update(value);
        }

        // Move to parent and flip value (opponent's perspective)
        node = node->parent;
        value = -value;
    }
}

void MCTSSearch::backpropagate(Node* node, float value, float pw, float pd, float pl) {
    while (node != nullptr) {
        if (config_.use_virtual_loss) node->remove_virtual_loss();

        if (node->parent == nullptr) {
            node->update_root(value);
            // Accumulate WDL at root only (not at intermediate nodes)
            root_wdl_sum_[0] += pw;
            root_wdl_sum_[1] += pd;
            root_wdl_sum_[2] += pl;
            root_wdl_count_++;
        } else {
            node->update(value);
        }

        node = node->parent;
        value = -value;
        std::swap(pw, pl);  // Flip W/L for opponent's perspective (D stays)
    }
}

float MCTSSearch::puct_score(const Node* parent, const Node* child) const {
    uint32_t parent_visits = parent->visit_count.load(std::memory_order_relaxed);
    uint32_t child_visits = child->visit_count.load(std::memory_order_relaxed);
    int16_t virtual_loss = child->virtual_loss.load(std::memory_order_relaxed);

    // Risk-adjusted Q-value (delegates to q_value() when risk_beta=0)
    float parent_q = parent->q_value_risk(0.0f, config_.fpu_base, config_.risk_beta);
    float q = child->q_value_risk(parent_q, config_.fpu_base, config_.risk_beta);

    // Prior probability
    float prior = child->prior();

    // PUCT formula: Q + c_puct * P * sqrt(N_parent) / (1 + N_child + virtual_loss)
    float exploration = config_.c_puct * prior * std::sqrt(static_cast<float>(parent_visits))
                       / (1.0f + child_visits + virtual_loss);

    return q + exploration;
}

void MCTSSearch::remove_virtual_losses_for_path(Node* node) {
    if (!config_.use_virtual_loss) return;
    // Traverse from leaf to root, removing virtual loss from each node
    // This undoes the virtual losses added during selection
    while (node != nullptr) {
        node->remove_virtual_loss();
        node = node->parent;
    }
}

void MCTSSearch::add_dirichlet_noise(Node* root) {
    // Count children
    int num_children = 0;
    for (Node* child = root->first_child; child != nullptr; child = child->next_sibling) {
        num_children++;
    }

    if (num_children == 0) return;

    // Generate Dirichlet noise
    std::gamma_distribution<float> gamma(config_.dirichlet_alpha, 1.0f);
    std::vector<float> noise(num_children);
    float noise_sum = 0.0f;

    for (int i = 0; i < num_children; ++i) {
        noise[i] = gamma(rng_);
        noise_sum += noise[i];
    }

    // Normalize noise
    for (int i = 0; i < num_children; ++i) {
        noise[i] /= noise_sum;
    }

    // Apply noise to priors
    int i = 0;
    for (Node* child = root->first_child; child != nullptr; child = child->next_sibling) {
        float original_prior = child->prior();
        float noisy_prior = (1.0f - config_.dirichlet_epsilon) * original_prior
                          + config_.dirichlet_epsilon * noise[i];
        child->set_prior(noisy_prior);
        i++;
    }
}

std::vector<int32_t> MCTSSearch::get_visit_counts() const {
    std::vector<int32_t> visit_counts(encoding::MoveEncoder::POLICY_SIZE, 0);

    if (root_ == nullptr) return visit_counts;

    for (Node* child = root_->first_child; child != nullptr; child = child->next_sibling) {
        int idx = encoding::MoveEncoder::move_to_index(child->move, root_position_);
        if (idx >= 0 && idx < encoding::MoveEncoder::POLICY_SIZE) {
            visit_counts[idx] = static_cast<int32_t>(
                child->visit_count.load(std::memory_order_relaxed)
            );
        }
    }

    return visit_counts;
}

// ============================================================================
// Async Double-Buffer Pipeline Implementation
// ============================================================================

int MCTSSearch::collect_leaves_async(float* obs_buffer, float* mask_buffer, int max_batch_size) {
    // Same as collect_leaves but writes to the collection buffer instead of pending_evals_
    int leaves_needed = std::min(max_batch_size, config_.num_simulations - simulations_completed_ - simulations_in_flight_);

    if (leaves_needed <= 0) {
        return 0;
    }

    // Run selection into collection buffer
    auto& collection = get_collection_buffer();
    collection.clear();
    collection.reserve(leaves_needed);

    for (int sim = 0; sim < leaves_needed; ++sim) {
        chess::Board board = root_position_;
        Node* leaf = select(root_, board);

        if (leaf->is_terminal()) {
            auto [reason, result] = board.isGameOver();
            float value = 0.0f;
            float pw = 0.0f, pd = 0.0f, pl = 0.0f;
            if (result == chess::GameResult::LOSE) {
                value = -1.0f;
                pl = 1.0f;
            } else if (result == chess::GameResult::WIN) {
                value = 1.0f;
                pw = 1.0f;
            } else {
                // Draw — value=0, variance=0 → Q_beta=0 regardless of risk_beta
                value = 0.0f;
                pd = 1.0f;
            }
            backpropagate(leaf, value, pw, pd, pl);
            simulations_completed_++;
            if (config_.use_gumbel) advance_gumbel_sim();
        } else if (!leaf->is_expanded()) {
            collection.emplace_back(leaf, board, sim);
        } else {
            float value = leaf->q_value(0.0f, config_.fpu_base);
            backpropagate(leaf, value);
            simulations_completed_++;
            if (config_.use_gumbel) advance_gumbel_sim();
        }
    }

    // Encode collected leaves into buffers
    int batch_size = static_cast<int>(collection.size());

    for (int i = 0; i < batch_size; ++i) {
        const auto& eval = collection[i];

        encoding::PositionEncoder::encode_to_buffer(
            eval.board,
            obs_buffer + i * encoding::PositionEncoder::TOTAL_SIZE,
            position_history_
        );

        float* mask_ptr = mask_buffer + i * encoding::MoveEncoder::POLICY_SIZE;
        std::fill(mask_ptr, mask_ptr + encoding::MoveEncoder::POLICY_SIZE, 0.0f);

        chess::Movelist moves;
        chess::movegen::legalmoves(moves, eval.board);

        for (const auto& move : moves) {
            int idx = encoding::MoveEncoder::move_to_index(move, eval.board);
            if (idx >= 0 && idx < encoding::MoveEncoder::POLICY_SIZE) {
                mask_ptr[idx] = 1.0f;
            }
        }
    }

    simulations_in_flight_ += batch_size;
    return batch_size;
}

int MCTSSearch::get_prev_batch_size() const {
    return static_cast<int>(get_evaluation_buffer().size());
}

void MCTSSearch::update_prev_leaves(const std::vector<std::vector<float>>& policies,
                                     const std::vector<float>& values,
                                     const float* wdl) {
    const auto& eval_buffer = get_evaluation_buffer();
    int batch_size = std::min({
        static_cast<int>(eval_buffer.size()),
        static_cast<int>(policies.size()),
        static_cast<int>(values.size())
    });

    for (int i = 0; i < batch_size; ++i) {
        const auto& eval = eval_buffer[i];
        Node* leaf = eval.node;

        expand(leaf, eval.board, policies[i], values[i]);
        if (wdl != nullptr) {
            backpropagate(leaf, values[i], wdl[i*3+0], wdl[i*3+1], wdl[i*3+2]);
        } else {
            backpropagate(leaf, values[i]);
        }
        simulations_completed_++;
        if (config_.use_gumbel) advance_gumbel_sim();
    }

    // Remove virtual losses from any extra leaves that didn't get results
    for (int i = batch_size; i < static_cast<int>(eval_buffer.size()); ++i) {
        remove_virtual_losses_for_path(eval_buffer[i].node);
    }

    simulations_in_flight_ -= static_cast<int>(eval_buffer.size());
}

void MCTSSearch::cancel_prev_pending() {
    const auto& eval_buffer = get_evaluation_buffer();
    for (const auto& eval : eval_buffer) {
        remove_virtual_losses_for_path(eval.node);
    }
    simulations_in_flight_ -= static_cast<int>(eval_buffer.size());
}

void MCTSSearch::cancel_collection_pending() {
    auto& coll_buffer = get_collection_buffer();
    for (const auto& eval : coll_buffer) {
        remove_virtual_losses_for_path(eval.node);
    }
    simulations_in_flight_ -= static_cast<int>(coll_buffer.size());
    coll_buffer.clear();
}

void MCTSSearch::commit_and_swap() {
    swap_buffers();
    start_next_batch_collection();
}

// Double buffering implementation for GPU/CPU overlap
void MCTSSearch::start_next_batch_collection() {
    // Start collecting into the inactive buffer
    // This allows CPU to prepare the next batch while GPU processes current batch
    auto& collection_buffer = using_buffer_a_ ? buffer_b_ : buffer_a_;
    collection_buffer.clear();
}

std::vector<PendingEval>& MCTSSearch::get_collection_buffer() {
    // Return the buffer currently being filled by CPU
    return using_buffer_a_ ? buffer_b_ : buffer_a_;
}

const std::vector<PendingEval>& MCTSSearch::get_evaluation_buffer() const {
    // Return the buffer currently being processed by GPU
    return using_buffer_a_ ? buffer_a_ : buffer_b_;
}

void MCTSSearch::swap_buffers() {
    // Swap which buffer is active after GPU completes evaluation
    using_buffer_a_ = !using_buffer_a_;
}

// ============================================================================
// Gumbel Top-k Sequential Halving Implementation
// ============================================================================

void MCTSSearch::init_gumbel() {
    // Collect all root children into indexed vector
    gumbel_.children.clear();
    gumbel_.logit.clear();
    gumbel_.gumbel.clear();

    for (Node* child = root_->first_child; child != nullptr; child = child->next_sibling) {
        gumbel_.children.push_back(child);
        // Recover logit from post-softmax prior: logit = log(max(prior, 1e-6))
        // Constant shift from softmax normalization cancels in softmax(logit + sigma)
        float p = std::max(child->prior(), 1e-6f);
        gumbel_.logit.push_back(std::log(p));
    }

    int num_children = static_cast<int>(gumbel_.children.size());
    if (num_children <= 1) {
        // Forced move or no children — skip SH entirely
        gumbel_.all_phases_done = true;
        gumbel_.active_indices.clear();
        if (num_children == 1) {
            gumbel_.active_indices.push_back(0);
        }
        return;
    }

    // Sample Gumbel(0,1) noise: g = -log(-log(U)), U ~ Uniform(0,1)
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    gumbel_.gumbel.resize(num_children);
    for (int i = 0; i < num_children; ++i) {
        float u = std::clamp(uniform(rng_), 1e-10f, 1.0f - 1e-10f);
        gumbel_.gumbel[i] = -std::log(-std::log(u));
    }

    // Top-m selection: keep m actions with highest (logit + gumbel)
    int m = std::min(config_.gumbel_top_k, num_children);

    // Create index array and partial sort by score descending
    std::vector<int> indices(num_children);
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(), indices.begin() + m, indices.end(),
        [this](int a, int b) {
            float score_a = gumbel_.logit[a] + gumbel_.gumbel[a];
            float score_b = gumbel_.logit[b] + gumbel_.gumbel[b];
            return score_a > score_b;
        });

    gumbel_.active_indices.assign(indices.begin(), indices.begin() + m);

    // Setup Sequential Halving phases
    gumbel_.num_phases = static_cast<int>(std::ceil(std::log2(static_cast<float>(m))));
    gumbel_.num_phases = std::max(gumbel_.num_phases, 1);
    gumbel_.current_phase = 0;
    gumbel_.total_budget = config_.num_simulations;
    gumbel_.total_sims_used = 0;
    gumbel_.phase_sims_done = 0;
    gumbel_.round_robin_counter = 0;
    gumbel_.all_phases_done = false;
    gumbel_.c_visit = config_.gumbel_c_visit;
    gumbel_.c_scale = config_.gumbel_c_scale;

    // Budget per action for first phase
    int active_count = static_cast<int>(gumbel_.active_indices.size());
    gumbel_.sims_per_action = std::max(1,
        gumbel_.total_budget / (gumbel_.num_phases * active_count));

    // Gumbel uses round-robin root selection — VL is unnecessary and harmful
    // (biases Q-values when we need clean Q estimates for scoring)
    config_.use_virtual_loss = false;
}

Node* MCTSSearch::get_gumbel_target_child() {
    int active_count = static_cast<int>(gumbel_.active_indices.size());
    if (active_count == 0) return nullptr;

    int active_idx = gumbel_.round_robin_counter % active_count;
    gumbel_.round_robin_counter++;
    return gumbel_.children[gumbel_.active_indices[active_idx]];
}

void MCTSSearch::advance_gumbel_sim() {
    if (gumbel_.all_phases_done) return;

    gumbel_.phase_sims_done++;
    gumbel_.total_sims_used++;

    // Phase complete when total sims = sims_per_action * num_active_actions
    int sims_for_phase = gumbel_.sims_per_action
                       * static_cast<int>(gumbel_.active_indices.size());
    if (gumbel_.phase_sims_done >= sims_for_phase) {
        complete_gumbel_phase();
    }
}

void MCTSSearch::complete_gumbel_phase() {
    int active_count = static_cast<int>(gumbel_.active_indices.size());
    if (active_count <= 1) {
        gumbel_.all_phases_done = true;
        return;
    }

    // Compute completed_q and find min/max for sigma normalization
    std::vector<float> q_vals(active_count);
    float min_q = std::numeric_limits<float>::infinity();
    float max_q = -std::numeric_limits<float>::infinity();
    int max_visits = 0;

    for (int i = 0; i < active_count; ++i) {
        int idx = gumbel_.active_indices[i];
        q_vals[i] = completed_q(gumbel_.children[idx]);
        min_q = std::min(min_q, q_vals[i]);
        max_q = std::max(max_q, q_vals[i]);
        int n = static_cast<int>(
            gumbel_.children[idx]->visit_count.load(std::memory_order_relaxed));
        max_visits = std::max(max_visits, n);
    }

    // Score each active action: logit + gumbel + sigma(q_completed)
    std::vector<std::pair<float, int>> scores(active_count);
    for (int i = 0; i < active_count; ++i) {
        int idx = gumbel_.active_indices[i];
        float sq = sigma_q(q_vals[i], min_q, max_q, max_visits);
        float score = gumbel_.logit[idx] + gumbel_.gumbel[idx] + sq;
        scores[i] = {score, idx};
    }

    // Sort descending by score
    std::sort(scores.begin(), scores.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    // Keep top ceil(active/2)
    int keep = (active_count + 1) / 2;
    gumbel_.active_indices.clear();
    for (int i = 0; i < keep; ++i) {
        gumbel_.active_indices.push_back(scores[i].second);
    }

    gumbel_.current_phase++;

    // Check if all phases done
    if (gumbel_.active_indices.size() <= 1 ||
        gumbel_.current_phase >= gumbel_.num_phases) {
        gumbel_.all_phases_done = true;
        return;
    }

    // Recalculate budget for next phase from remaining budget
    int remaining_budget = gumbel_.total_budget - gumbel_.total_sims_used;
    int remaining_phases = gumbel_.num_phases - gumbel_.current_phase;
    int new_active_count = static_cast<int>(gumbel_.active_indices.size());
    gumbel_.sims_per_action = std::max(1,
        remaining_budget / (remaining_phases * new_active_count));

    // Reset phase counters
    gumbel_.phase_sims_done = 0;
    gumbel_.round_robin_counter = 0;
}

float MCTSSearch::completed_q(const Node* child) const {
    uint32_t n = child->visit_count.load(std::memory_order_relaxed);

    if (n == 0) {
        return root_value_;  // Unvisited: use V(root)
    }

    // Get raw Q-value (risk-adjusted if beta != 0)
    float q;
    if (config_.risk_beta != 0.0f) {
        q = child->q_value_risk(0.0f, config_.fpu_base, config_.risk_beta);
    } else {
        int64_t sum = child->value_sum_fixed.load(std::memory_order_relaxed);
        q = sum / (10000.0f * n);
    }

    // Interpolate: Q_hat = N/(1+N) * Q + 1/(1+N) * V_root
    float fn = static_cast<float>(n);
    return (fn / (1.0f + fn)) * q + (1.0f / (1.0f + fn)) * root_value_;
}

float MCTSSearch::sigma_q(float q, float min_q, float max_q, int max_visits) const {
    float range = max_q - min_q;
    float q_normalized = (range < 1e-8f) ? 0.0f : 2.0f * (q - min_q) / range - 1.0f;
    return (gumbel_.c_visit + static_cast<float>(max_visits)) * gumbel_.c_scale * q_normalized;
}

std::vector<float> MCTSSearch::get_improved_policy() const {
    std::vector<float> policy(encoding::MoveEncoder::POLICY_SIZE, 0.0f);

    if (root_ == nullptr || gumbel_.children.empty()) return policy;

    int num_children = static_cast<int>(gumbel_.children.size());

    // Compute completed_q for all children and find min/max for sigma
    std::vector<float> q_vals(num_children);
    float min_q = std::numeric_limits<float>::infinity();
    float max_q = -std::numeric_limits<float>::infinity();
    int max_visits = 0;

    for (int i = 0; i < num_children; ++i) {
        q_vals[i] = completed_q(gumbel_.children[i]);
        min_q = std::min(min_q, q_vals[i]);
        max_q = std::max(max_q, q_vals[i]);
        int n = static_cast<int>(
            gumbel_.children[i]->visit_count.load(std::memory_order_relaxed));
        max_visits = std::max(max_visits, n);
    }

    // Compute log_pi = logit + sigma(q_completed) for each child
    // Then softmax with log-sum-exp stability trick
    std::vector<float> log_pi(num_children);
    float max_log_pi = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < num_children; ++i) {
        float sq = sigma_q(q_vals[i], min_q, max_q, max_visits);
        log_pi[i] = gumbel_.logit[i] + sq;
        max_log_pi = std::max(max_log_pi, log_pi[i]);
    }

    // Softmax: exp(log_pi - max) / sum(exp(log_pi - max))
    float sum_exp = 0.0f;
    std::vector<float> exp_vals(num_children);
    for (int i = 0; i < num_children; ++i) {
        exp_vals[i] = std::exp(log_pi[i] - max_log_pi);
        sum_exp += exp_vals[i];
    }

    // Write to 4672-dim policy vector
    for (int i = 0; i < num_children; ++i) {
        float prob = exp_vals[i] / sum_exp;
        int idx = encoding::MoveEncoder::move_to_index(
            gumbel_.children[i]->move, root_position_);
        if (idx >= 0 && idx < encoding::MoveEncoder::POLICY_SIZE) {
            policy[idx] = prob;
        }
    }

    return policy;
}

chess::Move MCTSSearch::get_gumbel_action() const {
    if (gumbel_.active_indices.empty()) {
        // Fallback: return first child's move
        if (root_ && root_->first_child) return root_->first_child->move;
        return chess::Move();
    }

    if (gumbel_.active_indices.size() == 1) {
        return gumbel_.children[gumbel_.active_indices[0]]->move;
    }

    // Multiple remaining: pick argmax of logit + gumbel + sigma_q among active set
    int num_children = static_cast<int>(gumbel_.children.size());

    // Need min/max Q across active set for sigma
    std::vector<float> q_vals;
    float min_q = std::numeric_limits<float>::infinity();
    float max_q = -std::numeric_limits<float>::infinity();
    int max_visits = 0;

    for (int idx : gumbel_.active_indices) {
        float qc = completed_q(gumbel_.children[idx]);
        q_vals.push_back(qc);
        min_q = std::min(min_q, qc);
        max_q = std::max(max_q, qc);
        int n = static_cast<int>(
            gumbel_.children[idx]->visit_count.load(std::memory_order_relaxed));
        max_visits = std::max(max_visits, n);
    }

    float best_score = -std::numeric_limits<float>::infinity();
    int best_idx = gumbel_.active_indices[0];

    for (int i = 0; i < static_cast<int>(gumbel_.active_indices.size()); ++i) {
        int idx = gumbel_.active_indices[i];
        float sq = sigma_q(q_vals[i], min_q, max_q, max_visits);
        float score = gumbel_.logit[idx] + gumbel_.gumbel[idx] + sq;
        if (score > best_score) {
            best_score = score;
            best_idx = idx;
        }
    }

    return gumbel_.children[best_idx]->move;
}

float MCTSSearch::get_root_risk_value(float beta) const {
    if (beta == 0.0f || root_ == nullptr) return 0.0f;

    float parent_q = root_->q_value_risk(0.0f, config_.fpu_base, config_.risk_beta);
    float max_q = -std::numeric_limits<float>::infinity();
    int count = 0;

    // First pass: find max Q for numerical stability (shift trick)
    for (Node* child = root_->first_child; child != nullptr; child = child->next_sibling) {
        if (child->visit_count.load(std::memory_order_relaxed) > 0) {
            float q = child->q_value_risk(parent_q, config_.fpu_base, config_.risk_beta);
            if (q > max_q) max_q = q;
            count++;
        }
    }
    if (count == 0) return 0.0f;

    // Second pass: LogSumExp = max_q + (1/beta) * log( sum exp(beta * (q - max_q)) )
    float sum_exp = 0.0f;
    for (Node* child = root_->first_child; child != nullptr; child = child->next_sibling) {
        if (child->visit_count.load(std::memory_order_relaxed) > 0) {
            float q = child->q_value_risk(parent_q, config_.fpu_base, config_.risk_beta);
            sum_exp += std::exp(beta * (q - max_q));
        }
    }

    float lse = max_q + std::log(sum_exp) / beta;
    return std::clamp(lse, -1.0f, 1.0f);
}

} // namespace mcts
