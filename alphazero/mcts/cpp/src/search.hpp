#pragma once
/**
 * C++ MCTS Search implementation for AlphaZero.
 */

#include "node.hpp"
#include <vector>
#include <random>
#include <functional>

namespace alphazero {
namespace mcts {

/**
 * MCTS configuration.
 */
struct MCTSConfig {
    int num_simulations = 800;
    float c_puct = 1.25f;
    float dirichlet_alpha = 0.3f;
    float dirichlet_epsilon = 0.25f;
    float temperature = 1.0f;
    int temperature_threshold = 30;
};

/**
 * Statistics from an MCTS search.
 */
struct MCTSStats {
    int num_simulations = 0;
    int max_depth = 0;
    double root_value = 0.0;
    int nodes_created = 0;
};

/**
 * MCTS Search algorithm.
 */
class MCTS {
public:
    MCTS(const MCTSConfig& config = MCTSConfig())
        : config_(config)
        , num_actions_(4672)
        , rng_(std::random_device{}())
    {}

    /**
     * Get temperature for action selection.
     */
    float get_temperature(int move_number) const {
        if (move_number < config_.temperature_threshold) {
            return config_.temperature;
        }
        return 0.0f;
    }

    /**
     * Apply temperature to visit counts to get policy.
     */
    std::vector<float> apply_temperature(const std::vector<float>& visit_counts,
                                         float temperature) const {
        std::vector<float> policy(visit_counts.size(), 0.0f);

        if (temperature <= 0.01f) {
            auto max_it = std::max_element(visit_counts.begin(), visit_counts.end());
            if (*max_it > 0) {
                policy[std::distance(visit_counts.begin(), max_it)] = 1.0f;
            }
            return policy;
        }

        float exponent = std::min(1.0f / temperature, 10.0f);
        double total = 0.0;

        for (size_t i = 0; i < visit_counts.size(); ++i) {
            policy[i] = std::pow(visit_counts[i], exponent);
            total += policy[i];
        }

        if (total > 0) {
            for (float& p : policy) {
                p /= total;
            }
        }

        return policy;
    }

    /**
     * Add Dirichlet noise to root priors for exploration.
     */
    std::vector<float> add_dirichlet_noise(const std::vector<float>& priors,
                                           const std::vector<float>& legal_mask) {
        std::vector<float> noisy_priors = priors;

        // Count legal actions
        std::vector<int> legal_indices;
        for (size_t i = 0; i < legal_mask.size(); ++i) {
            if (legal_mask[i] > 0) {
                legal_indices.push_back(i);
            }
        }

        if (legal_indices.empty()) return noisy_priors;

        // Generate Dirichlet noise
        std::gamma_distribution<float> gamma(config_.dirichlet_alpha, 1.0f);
        std::vector<float> noise(legal_indices.size());
        float noise_sum = 0.0f;

        for (size_t i = 0; i < legal_indices.size(); ++i) {
            noise[i] = gamma(rng_);
            noise_sum += noise[i];
        }

        // Normalize and blend
        float epsilon = config_.dirichlet_epsilon;
        for (size_t i = 0; i < legal_indices.size(); ++i) {
            int idx = legal_indices[i];
            float n = (noise_sum > 0) ? noise[i] / noise_sum : 1.0f / legal_indices.size();
            noisy_priors[idx] = (1.0f - epsilon) * priors[idx] + epsilon * n;
        }

        return noisy_priors;
    }

    /**
     * Run MCTS search.
     *
     * This is a template method that takes evaluator and state as callbacks
     * to avoid coupling with Python types.
     */
    template<typename EvalFunc, typename ApplyFunc, typename TerminalFunc, typename ValueFunc>
    std::tuple<std::vector<float>, std::shared_ptr<MCTSNode>, MCTSStats>
    search(const std::vector<float>& initial_priors,
           const std::vector<float>& initial_legal_mask,
           float initial_value,
           EvalFunc evaluate,
           ApplyFunc apply_action,
           TerminalFunc is_terminal,
           ValueFunc get_value,
           int move_number = 0,
           bool add_noise = true) {

        MCTSStats stats;

        // Create root node
        auto root = std::make_shared<MCTSNode>(1.0f);

        // Add Dirichlet noise at root
        std::vector<float> priors = add_noise
            ? add_dirichlet_noise(initial_priors, initial_legal_mask)
            : initial_priors;

        // Expand root
        root->expand(priors, initial_legal_mask);
        root->update(initial_value);
        stats.nodes_created = 1;

        // Run simulations
        for (int sim = 0; sim < config_.num_simulations; ++sim) {
            int depth = simulate(root, evaluate, apply_action, is_terminal, get_value, stats);
            stats.max_depth = std::max(stats.max_depth, depth);
        }

        stats.num_simulations = config_.num_simulations;
        stats.root_value = root->q_value();

        // Get policy from visit counts
        float temperature = get_temperature(move_number);
        std::vector<float> policy = root->get_policy(num_actions_, temperature);

        return {policy, root, stats};
    }

private:
    template<typename EvalFunc, typename ApplyFunc, typename TerminalFunc, typename ValueFunc>
    int simulate(std::shared_ptr<MCTSNode> root,
                 EvalFunc& evaluate,
                 ApplyFunc& apply_action,
                 TerminalFunc& is_terminal,
                 ValueFunc& get_value,
                 MCTSStats& stats) {

        auto node = root;
        std::vector<std::pair<std::shared_ptr<MCTSNode>, int>> path;
        int depth = 0;
        int state_id = 0;  // Root state

        // Selection
        while (node->is_expanded() && !node->is_terminal()) {
            auto [action, child] = node->select_child(config_.c_puct);
            path.push_back({node, action});
            node = child;
            state_id = apply_action(state_id, action);
            depth++;
        }

        double value;

        // Check for terminal
        if (is_terminal(state_id)) {
            value = get_value(state_id);
            node->set_terminal(value);
        } else if (!node->is_expanded()) {
            // Expansion and evaluation
            auto [priors, legal_mask, v] = evaluate(state_id);
            value = v;
            node->expand(priors, legal_mask);
            stats.nodes_created++;
        } else {
            value = node->terminal_value();
        }

        // Backpropagation
        node->update(value);
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            value = -value;
            it->first->update(value);
        }

        return depth;
    }

    MCTSConfig config_;
    int num_actions_;
    mutable std::mt19937 rng_;
};

}  // namespace mcts
}  // namespace alphazero
