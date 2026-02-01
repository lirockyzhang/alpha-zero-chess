#include "mcts/search.hpp"
#include "encoding/move_encoder.hpp"
#include <algorithm>
#include <numeric>

namespace mcts {

Node* MCTSSearch::search(const chess::Board& root_position,
                         const std::vector<float>& policy,
                         float value) {
    // Create root node
    Node* root = pool_.allocate();

    // Expand root immediately
    expand(root, root_position, policy, value);

    // Add Dirichlet noise to root for exploration
    add_dirichlet_noise(root);

    // Run simulations
    for (int i = 0; i < config_.num_simulations; ++i) {
        // Make a copy of the board for this simulation
        chess::Board board = root_position;

        // Selection phase: traverse tree to leaf
        Node* leaf = select(root, board);

        // If leaf is terminal, backpropagate immediately
        if (leaf->is_terminal()) {
            // Terminal value depends on game outcome
            float terminal_value = 0.0f;  // Draw

            // Check if it's checkmate or stalemate
            auto game_result = board.isGameOver();
            if (game_result.second == chess::GameResult::LOSE) {
                terminal_value = -1.0f;  // Loss for side to move
            }

            backpropagate(leaf, terminal_value);
            continue;
        }

        // If leaf hasn't been expanded yet, we need neural network evaluation
        // For now, use a placeholder (will be replaced with actual NN inference)
        if (!leaf->is_expanded()) {
            // TODO: Queue for batch neural network inference
            // For now, use random policy and value
            std::vector<float> leaf_policy(1858, 1.0f / 1858.0f);  // Uniform policy
            float leaf_value = 0.0f;  // Neutral value

            expand(leaf, board, leaf_policy, leaf_value);
            backpropagate(leaf, leaf_value);
        }
    }

    return root;
}

Node* MCTSSearch::select(Node* node, chess::Board& board) {
    while (node->is_expanded() && !node->is_terminal()) {
        // Find child with highest PUCT score
        Node* best_child = nullptr;
        float best_score = -std::numeric_limits<float>::infinity();

        for (Node* child = node->first_child; child != nullptr; child = child->next_sibling) {
            float score = puct_score(node, child);
            if (score > best_score) {
                best_score = score;
                best_child = child;
            }
        }

        if (best_child == nullptr) {
            break;  // No children (terminal node)
        }

        // Add virtual loss to prevent multiple threads from selecting same path
        best_child->add_virtual_loss();

        // Make move on board
        board.makeMove(best_child->move);

        node = best_child;
    }

    return node;
}

void MCTSSearch::expand(Node* node, const chess::Board& board,
                        const std::vector<float>& policy, float value) {
    // Check if game is over
    auto game_result = board.isGameOver();
    if (game_result.first != chess::GameResultReason::NONE) {
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
    float prior_sum = 0.0f;  // Track sum for normalization

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

        // Ensure non-negative prior (policy should already be non-negative)
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

    // Normalize priors if sum is not close to 1.0 (handles edge cases)
    if (prior_sum > 0.0f && std::abs(prior_sum - 1.0f) > 0.01f) {
        for (Node* child = node->first_child; child != nullptr; child = child->next_sibling) {
            float normalized_prior = child->prior() / prior_sum;
            child->set_prior(normalized_prior);
        }
    } else if (prior_sum == 0.0f) {
        // Fallback to uniform if all priors are zero (shouldn't happen with proper policy)
        float uniform_prior = 1.0f / moves.size();
        for (Node* child = node->first_child; child != nullptr; child = child->next_sibling) {
            child->set_prior(uniform_prior);
        }
    }

    node->set_expanded(true);
}

void MCTSSearch::backpropagate(Node* node, float value) {
    // Backpropagate value up the tree
    // Value is negated at each level (zero-sum game)
    while (node != nullptr) {
        // Remove virtual loss
        if (node->parent != nullptr) {
            node->remove_virtual_loss();
        }

        // Update node with value
        // Use release semantics on root node to prevent race conditions
        if (node->parent == nullptr) {
            node->update_root(value);  // Root node: use memory_order_release
        } else {
            node->update(value);  // Non-root: use memory_order_relaxed
        }

        // Negate value for parent (opponent's perspective)
        value = -value;

        node = node->parent;
    }
}

chess::Move MCTSSearch::select_move(Node* root, float temperature) {
    if (root->first_child == nullptr) {
        return chess::Move();  // No legal moves
    }

    if (temperature < 0.01f) {
        // Greedy selection: pick move with highest visit count
        Node* best_child = nullptr;
        uint32_t best_visits = 0;

        for (Node* child = root->first_child; child != nullptr; child = child->next_sibling) {
            uint32_t visits = child->visit_count.load(std::memory_order_relaxed);
            if (visits > best_visits) {
                best_visits = visits;
                best_child = child;
            }
        }

        return best_child ? best_child->move : chess::Move();
    } else {
        // Stochastic selection: sample proportional to visit_count^(1/temperature)
        std::vector<Node*> children;
        std::vector<float> weights;

        for (Node* child = root->first_child; child != nullptr; child = child->next_sibling) {
            children.push_back(child);
            uint32_t visits = child->visit_count.load(std::memory_order_relaxed);
            float weight = std::pow(static_cast<float>(visits), 1.0f / temperature);
            weights.push_back(weight);
        }

        // Sample from distribution
        std::discrete_distribution<> dist(weights.begin(), weights.end());
        int idx = dist(rng_);

        return children[idx]->move;
    }
}

std::vector<float> MCTSSearch::get_policy_target(Node* root) {
    // Get visit count distribution for training
    // Returns normalized visit counts for all children

    std::vector<float> policy_target;
    uint32_t total_visits = 0;

    // Count total visits
    for (Node* child = root->first_child; child != nullptr; child = child->next_sibling) {
        uint32_t visits = child->visit_count.load(std::memory_order_relaxed);
        total_visits += visits;
    }

    // Normalize visit counts
    for (Node* child = root->first_child; child != nullptr; child = child->next_sibling) {
        uint32_t visits = child->visit_count.load(std::memory_order_relaxed);
        float prob = total_visits > 0 ? static_cast<float>(visits) / total_visits : 0.0f;
        policy_target.push_back(prob);
    }

    return policy_target;
}

void MCTSSearch::add_dirichlet_noise(Node* root) {
    // Add Dirichlet noise to root node for exploration
    // This encourages exploration of different moves during self-play

    if (root->first_child == nullptr) {
        return;
    }

    // Count number of children
    int num_children = 0;
    for (Node* child = root->first_child; child != nullptr; child = child->next_sibling) {
        num_children++;
    }

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

    // Mix noise with prior: P = (1 - epsilon) * P + epsilon * noise
    int idx = 0;
    for (Node* child = root->first_child; child != nullptr; child = child->next_sibling) {
        float prior = child->prior();
        float mixed_prior = (1.0f - config_.dirichlet_epsilon) * prior
                          + config_.dirichlet_epsilon * noise[idx];
        child->set_prior(mixed_prior);
        idx++;
    }
}

} // namespace mcts
