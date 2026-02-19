#include "training/sum_tree.hpp"
#include <algorithm>
#include <cmath>

namespace training {

size_t SumTree::next_power_of_2(size_t n) {
    if (n == 0) return 1;
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

SumTree::SumTree(size_t capacity)
    : logical_capacity_(capacity)
    , tree_capacity_(next_power_of_2(capacity))
    , tree_(2 * tree_capacity_, 0.0f)
    , max_priority_(1.0f)
{}

void SumTree::propagate(size_t tree_idx) {
    // Walk up from leaf to root, updating parent sums
    tree_idx >>= 1;  // start at parent
    while (tree_idx >= 1) {
        tree_[tree_idx] = tree_[2 * tree_idx] + tree_[2 * tree_idx + 1];
        tree_idx >>= 1;
    }
}

void SumTree::update(size_t leaf_idx, float priority) {
    if (leaf_idx >= logical_capacity_) return;

    if (priority > max_priority_) {
        max_priority_ = priority;
    }

    size_t tree_idx = tree_capacity_ + leaf_idx;
    tree_[tree_idx] = priority;
    propagate(tree_idx);
}

void SumTree::set_leaf(size_t leaf_idx, float priority) {
    if (leaf_idx >= logical_capacity_) return;

    if (priority > max_priority_) {
        max_priority_ = priority;
    }

    tree_[tree_capacity_ + leaf_idx] = priority;
    // No propagation — caller must call rebuild() before sampling
}

size_t SumTree::sample(float uniform) const {
    // Descend from root, choosing left or right child based on prefix sums
    size_t idx = 1;  // root
    while (idx < tree_capacity_) {
        float left = tree_[2 * idx];
        if (uniform <= left) {
            idx = 2 * idx;
        } else {
            uniform -= left;
            idx = 2 * idx + 1;
        }
    }
    // idx is now in [tree_capacity_, 2*tree_capacity_), convert to leaf index
    size_t leaf = idx - tree_capacity_;
    // Clamp to logical capacity (power-of-2 padding leaves have priority 0)
    return std::min(leaf, logical_capacity_ - 1);
}

float SumTree::get(size_t leaf_idx) const {
    if (leaf_idx >= logical_capacity_) return 0.0f;
    return tree_[tree_capacity_ + leaf_idx];
}

float SumTree::total() const {
    return tree_[1];  // root
}

void SumTree::rebuild() {
    // Bottom-up: recompute all internal nodes from leaves
    for (size_t i = tree_capacity_ - 1; i >= 1; --i) {
        tree_[i] = tree_[2 * i] + tree_[2 * i + 1];
    }
}

void SumTree::clear() {
    std::fill(tree_.begin(), tree_.end(), 0.0f);
    max_priority_ = 1.0f;
}

} // namespace training
