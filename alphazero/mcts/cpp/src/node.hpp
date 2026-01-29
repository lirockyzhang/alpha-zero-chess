#pragma once
/**
 * C++ MCTS Node implementation for AlphaZero.
 *
 * This provides ~20-50x speedup over pure Python by:
 * - Using native C++ data structures
 * - Avoiding Python object overhead
 * - Inline PUCT calculation
 * - Cache-friendly memory layout
 */

#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <memory>
#include <limits>

namespace alphazero {
namespace mcts {

/**
 * MCTS Node with PUCT selection.
 *
 * Stores statistics for a single state-action pair:
 *   - N(s,a): Visit count
 *   - W(s,a): Total value (sum of backpropagated values)
 *   - Q(s,a): Mean value = W(s,a) / N(s,a)
 *   - P(s,a): Prior probability from neural network
 */
class MCTSNode {
public:
    MCTSNode(float prior = 1.0f)
        : visit_count_(0)
        , value_sum_(0.0)
        , prior_(prior)
        , is_expanded_(false)
        , is_terminal_(false)
        , terminal_value_(0.0)
    {}

    // Getters
    int visit_count() const { return visit_count_; }
    double value_sum() const { return value_sum_; }
    float prior() const { return prior_; }

    // Q(s,a) = W(s,a) / N(s,a)
    double q_value() const {
        if (visit_count_ == 0) return 0.0;
        return value_sum_ / static_cast<double>(visit_count_);
    }

    bool is_expanded() const { return is_expanded_; }
    bool is_terminal() const { return is_terminal_; }
    double terminal_value() const { return terminal_value_; }

    void set_terminal(double value) {
        is_terminal_ = true;
        terminal_value_ = value;
    }

    void set_prior(float prior) { prior_ = prior; }

    /**
     * Expand this node by creating child nodes for legal actions.
     */
    void expand(const std::vector<float>& priors,
                const std::vector<float>& legal_mask) {
        children_.clear();
        legal_actions_.clear();

        // Compute prior sum for normalization
        double prior_sum = 0.0;
        for (size_t i = 0; i < priors.size(); ++i) {
            if (legal_mask[i] > 0) {
                prior_sum += priors[i];
                legal_actions_.push_back(static_cast<int>(i));
            }
        }

        // Create child nodes with normalized priors
        for (int action : legal_actions_) {
            float p = (prior_sum > 0) ? priors[action] / prior_sum
                                      : 1.0f / legal_actions_.size();
            children_[action] = std::make_shared<MCTSNode>(p);
        }

        is_expanded_ = true;
    }

    /**
     * Select the best child using PUCT algorithm.
     *
     * PUCT formula:
     *   a* = argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)) ]
     */
    std::pair<int, std::shared_ptr<MCTSNode>> select_child(float c_puct) const {
        int best_action = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        std::shared_ptr<MCTSNode> best_child = nullptr;

        double sqrt_parent = std::sqrt(static_cast<double>(visit_count_));

        for (const auto& [action, child] : children_) {
            // Q(s,a): exploitation term
            double q = child->q_value();

            // U(s,a): exploration term
            double u = c_puct * child->prior_ * sqrt_parent / (1.0 + child->visit_count_);

            // PUCT score
            double score = q + u;

            if (score > best_score) {
                best_score = score;
                best_action = action;
                best_child = child;
            }
        }

        return {best_action, best_child};
    }

    /**
     * Get child node for a specific action.
     */
    std::shared_ptr<MCTSNode> get_child(int action) const {
        auto it = children_.find(action);
        if (it != children_.end()) {
            return it->second;
        }
        return nullptr;
    }

    /**
     * Get all children.
     */
    const std::unordered_map<int, std::shared_ptr<MCTSNode>>& children() const {
        return children_;
    }

    /**
     * Update node statistics during backpropagation.
     */
    void update(double value) {
        visit_count_++;
        value_sum_ += value;
    }

    /**
     * Get visit counts for all actions as a vector.
     */
    std::vector<float> get_visit_counts(int num_actions) const {
        std::vector<float> counts(num_actions, 0.0f);
        for (const auto& [action, child] : children_) {
            counts[action] = static_cast<float>(child->visit_count_);
        }
        return counts;
    }

    /**
     * Get policy distribution from visit counts.
     */
    std::vector<float> get_policy(int num_actions, float temperature = 1.0f) const {
        std::vector<float> counts = get_visit_counts(num_actions);
        std::vector<float> policy(num_actions, 0.0f);

        if (temperature <= 0.01f) {
            // Greedy selection
            auto max_it = std::max_element(counts.begin(), counts.end());
            if (*max_it > 0) {
                policy[std::distance(counts.begin(), max_it)] = 1.0f;
            }
            return policy;
        }

        // Apply temperature with numerical stability
        float exponent = std::min(1.0f / temperature, 10.0f);
        double total = 0.0;

        for (size_t i = 0; i < counts.size(); ++i) {
            policy[i] = std::pow(counts[i], exponent);
            total += policy[i];
        }

        if (total > 0) {
            for (float& p : policy) {
                p /= total;
            }
        }

        return policy;
    }

private:
    int visit_count_;
    double value_sum_;
    float prior_;
    bool is_expanded_;
    bool is_terminal_;
    double terminal_value_;

    std::unordered_map<int, std::shared_ptr<MCTSNode>> children_;
    std::vector<int> legal_actions_;
};

}  // namespace mcts
}  // namespace alphazero
