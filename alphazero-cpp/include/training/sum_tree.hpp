#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>

namespace training {

/**
 * Sum-tree (segment tree) for O(log N) proportional sampling.
 *
 * Complete binary tree stored in a flat array:
 *   - tree_[0] is unused
 *   - tree_[1] is the root (total sum)
 *   - Leaves are at indices [tree_capacity_, 2*tree_capacity_)
 *   - Internal node i has children at 2i and 2i+1
 *
 * Each leaf stores priority^alpha for one replay buffer slot.
 * Internal nodes store the sum of their children's values.
 *
 * Supports two write modes:
 *   - update(): writes leaf + propagates sums to root — O(log N)
 *   - set_leaf(): writes leaf only, no propagation — O(1)
 *     Must call rebuild() before sampling after set_leaf() calls.
 */
class SumTree {
public:
    explicit SumTree(size_t capacity);

    /// Write priority to leaf and propagate sums to root. O(log N).
    void update(size_t leaf_idx, float priority);

    /// Write priority to leaf only (no propagation). O(1).
    /// Must call rebuild() before sampling after using this.
    void set_leaf(size_t leaf_idx, float priority);

    /// Proportional sample: returns leaf index. O(log N).
    /// @param uniform Random value in [0, total())
    size_t sample(float uniform) const;

    /// Get priority of a leaf. O(1).
    float get(size_t leaf_idx) const;

    /// Get total sum (root value). O(1).
    float total() const;

    /// Rebuild all internal nodes from leaves. O(N).
    /// Call after one or more set_leaf() calls.
    void rebuild();

    /// Reset all priorities to zero.
    void clear();

    /// Get the tracked maximum priority ever seen.
    float max_priority() const { return max_priority_; }

    /// Get logical capacity (number of leaves).
    size_t capacity() const { return logical_capacity_; }

private:
    size_t logical_capacity_;   // user-requested capacity
    size_t tree_capacity_;      // next power of 2 >= logical_capacity_
    std::vector<float> tree_;   // flat array, 2 * tree_capacity_ elements
    float max_priority_;

    static size_t next_power_of_2(size_t n);
    void propagate(size_t tree_idx);
};

} // namespace training
