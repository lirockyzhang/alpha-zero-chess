#include "mcts/batch_search.hpp"
#include "encoding/move_encoder.hpp"
#include "encoding/position_encoder.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

namespace mcts {

Node* BatchedMCTSSearch::init_search(const chess::Board& root_position,
                                      const std::vector<float>& root_policy,
                                      float root_value) {
    // Reset state
    root_position_ = root_position;
    pending_evals_.clear();
    simulations_completed_ = 0;
    simulations_in_flight_ = 0;

    // Create and expand root node
    root_ = pool_.allocate();
    expand(root_, root_position_, root_policy, root_value);

    // Add Dirichlet noise to root for exploration
    add_dirichlet_noise(root_);

    return root_;
}

int BatchedMCTSSearch::collect_leaves(float* obs_buffer, float* mask_buffer, int max_batch_size) {
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

        // Encode position to observation buffer (NHWC format: 8 x 8 x 119)
        encoding::PositionEncoder::encode_to_buffer(
            eval.board,
            obs_buffer + i * 8 * 8 * 119
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

void BatchedMCTSSearch::update_leaves(const std::vector<std::vector<float>>& policies,
                                       const std::vector<float>& values) {
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

        // Backpropagate the value
        backpropagate(leaf, values[i]);

        simulations_completed_++;
    }

    // Clear pending evaluations
    pending_evals_.clear();
    simulations_in_flight_ = 0;
}

void BatchedMCTSSearch::run_selection_phase(int num_sims) {
    pending_evals_.clear();
    pending_evals_.reserve(num_sims);

    for (int sim = 0; sim < num_sims; ++sim) {
        // Copy board state for this simulation
        chess::Board board = root_position_;

        // Select leaf node
        Node* leaf = select(root_, board);

        // If leaf is terminal, backpropagate immediately
        if (leaf->is_terminal()) {
            // Get terminal value
            auto [reason, result] = board.isGameOver();
            float value = 0.0f;

            if (result == chess::GameResult::LOSE) {
                // Current side to move lost (checkmate)
                value = -1.0f;
            } else if (result == chess::GameResult::WIN) {
                // Current side to move won (shouldn't happen in standard chess)
                value = 1.0f;
            }
            // Draw = 0.0f

            backpropagate(leaf, value);
            simulations_completed_++;
        } else if (!leaf->is_expanded()) {
            // Leaf needs NN evaluation - add to pending
            pending_evals_.emplace_back(leaf, board, sim);
        } else {
            // Already expanded (race condition in parallel) - just backpropagate
            float value = leaf->q_value(0.0f);
            backpropagate(leaf, value);
            simulations_completed_++;
        }
    }
}

Node* BatchedMCTSSearch::select(Node* node, chess::Board& board) {
    while (node->is_expanded() && !node->is_terminal()) {
        // Add virtual loss to prevent other threads from selecting same path
        node->add_virtual_loss();

        // Find best child using PUCT
        Node* best_child = nullptr;
        float best_score = -std::numeric_limits<float>::infinity();

        float parent_q = node->q_value(0.0f);

        for (Node* child = node->first_child; child != nullptr; child = child->next_sibling) {
            float score = puct_score(node, child);
            if (score > best_score) {
                best_score = score;
                best_child = child;
            }
        }

        if (best_child == nullptr) {
            // No children (shouldn't happen for expanded non-terminal node)
            break;
        }

        // Apply move to board
        board.makeMove(best_child->move);
        node = best_child;
    }

    return node;
}

void BatchedMCTSSearch::expand(Node* node, const chess::Board& board,
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

void BatchedMCTSSearch::backpropagate(Node* node, float value) {
    while (node != nullptr) {
        // Remove virtual loss
        node->remove_virtual_loss();

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

float BatchedMCTSSearch::puct_score(const Node* parent, const Node* child) const {
    uint32_t parent_visits = parent->visit_count.load(std::memory_order_relaxed);
    uint32_t child_visits = child->visit_count.load(std::memory_order_relaxed);
    int16_t virtual_loss = child->virtual_loss.load(std::memory_order_relaxed);

    // Q-value with virtual loss
    float parent_q = parent->q_value(0.0f);
    float q = child->q_value(parent_q);

    // Prior probability
    float prior = child->prior();

    // PUCT formula: Q + c_puct * P * sqrt(N_parent) / (1 + N_child + virtual_loss)
    float exploration = config_.c_puct * prior * std::sqrt(static_cast<float>(parent_visits))
                       / (1.0f + child_visits + virtual_loss);

    return q + exploration;
}

void BatchedMCTSSearch::add_dirichlet_noise(Node* root) {
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

std::vector<int32_t> BatchedMCTSSearch::get_visit_counts() const {
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

} // namespace mcts
