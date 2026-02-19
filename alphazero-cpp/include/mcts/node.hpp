#pragma once

#include <atomic>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include "../third_party/chess-library/include/chess.hpp"

namespace mcts {

// Forward declarations
class NodePool;

// Node structure for MCTS tree
// CRITICAL: 64-byte aligned to prevent false sharing between CPU cores
// Each node fits exactly in one cache line for optimal performance
//
// Layout (51 data bytes + 13 explicit padding = 64):
//   [0-7]   parent (8)
//   [8-15]  first_child (8)
//   [16-23] next_sibling (8)
//   [24-31] value_sum_fixed (8)        8-byte aligned, no gap
//   [32-39] value_sum_sq_fixed (8)     8-byte aligned (variance tracking)
//   [40-43] visit_count (4)            4-byte aligned
//   [44-47] move (4)                   chess::Move = uint16_t move_ + int16_t score_
//   [48-49] prior_fixed (2)
//   [50]    flags (1)
//   [51-63] padding[13] (13)
struct alignas(64) Node {
    // Parent node (nullptr for root)
    Node* parent;

    // First child in linked list (nullptr if no children)
    Node* first_child;

    // Next sibling in parent's child list (nullptr if last child)
    Node* next_sibling;

    // Value sum in fixed-point (multiply by 10000 for precision)
    // Using int64_t to prevent overflow: max value = 2^31 visits * 10000 * 1.0 = ~2^45
    std::atomic<int64_t> value_sum_fixed;

    // Sum of squared values in fixed-point (v^2 * 10000)
    // Used for variance computation: Var(v) = E[v^2] - E[v]^2
    // Max value = 2^31 visits * 10000 * 1.0^2 = ~2^45 (well within int64_t)
    std::atomic<int64_t> value_sum_sq_fixed;

    // Visit count (atomic for thread-safe MCTS)
    std::atomic<uint32_t> visit_count;

    // Move that led to this node (from parent's perspective)
    chess::Move move;

    // Policy prior from neural network (stored as uint16_t to save space)
    // Actual value = prior_fixed / 10000.0f
    uint16_t prior_fixed;

    // Flags: terminal (bit 0), expanded (bit 1)
    uint8_t flags;

    // Padding to reach exactly 64 bytes
    uint8_t padding[13];

    // Constructor
    Node()
        : parent(nullptr)
        , first_child(nullptr)
        , next_sibling(nullptr)
        , value_sum_fixed(0)
        , value_sum_sq_fixed(0)
        , move()
        , visit_count(0)
        , prior_fixed(0)
        , flags(0)
    {
        static_assert(sizeof(Node) == 64, "Node must be exactly 64 bytes for cache-line alignment");
    }

    // Q-value calculation with dynamic FPU (First Play Urgency)
    // Dynamic FPU: penalty = fpu_base * sqrt(1 - prior)
    //   high-prior moves (prior=0.9) get small penalty: sqrt(0.1) ≈ 0.32
    //   low-prior moves (prior=0.03) get harsh penalty: sqrt(0.97) ≈ 0.98
    // sqrt compression makes medium-prior moves more explorable than linear (1-prior).
    float q_value(float parent_q, float fpu_base = 0.3f) const {
        uint32_t n = visit_count.load(std::memory_order_relaxed);

        // Unvisited node: use dynamic FPU
        if (n == 0) {
            return parent_q - fpu_base * std::sqrt(1.0f - prior());
        }

        // Visited node: compute average value
        int64_t sum = value_sum_fixed.load(std::memory_order_relaxed);
        return sum / (10000.0f * n);
    }

    // Risk-adjusted Q-value using Entropic Risk Measure (ERM) approximation:
    //   Q_beta = E[v] + (beta/2) * Var(v)
    // beta > 0: risk-seeking (prefers high-variance positions)
    // beta < 0: risk-averse (prefers low-variance positions)
    // beta = 0: delegates to q_value() (zero overhead fast path)
    float q_value_risk(float parent_q, float fpu_base, float beta) const {
        // Fast path: standard AlphaZero when beta = 0
        if (beta == 0.0f) {
            return q_value(parent_q, fpu_base);
        }

        uint32_t n = visit_count.load(std::memory_order_relaxed);

        // Unvisited node: FPU unchanged (no variance data)
        if (n == 0) {
            return parent_q - fpu_base * std::sqrt(1.0f - prior());
        }

        // Visited node: compute risk-adjusted Q
        int64_t sum = value_sum_fixed.load(std::memory_order_relaxed);
        int64_t sum_sq = value_sum_sq_fixed.load(std::memory_order_relaxed);

        float mean = sum / (10000.0f * n);
        float mean_sq = sum_sq / (10000.0f * n);
        // Guard against tiny negative residuals from fixed-point rounding
        float var = std::max(0.0f, mean_sq - mean * mean);

        float q_risk = mean + (beta / 2.0f) * var;

        // Clamp to valid range (extreme beta may saturate)
        return std::clamp(q_risk, -1.0f, 1.0f);
    }

    // Prior probability from neural network
    float prior() const {
        return prior_fixed / 10000.0f;
    }

    // Set prior probability (converts float to fixed-point)
    void set_prior(float p) {
        prior_fixed = static_cast<uint16_t>(p * 10000.0f);
    }

    // Terminal node check
    bool is_terminal() const {
        return flags & 0x01;
    }

    void set_terminal(bool terminal) {
        if (terminal) {
            flags |= 0x01;
        } else {
            flags &= ~0x01;
        }
    }

    // Expanded node check (has children been generated)
    bool is_expanded() const {
        return flags & 0x02;
    }

    void set_expanded(bool expanded) {
        if (expanded) {
            flags |= 0x02;
        } else {
            flags &= ~0x02;
        }
    }

    // Update node with backpropagation value
    void update(float value) {
        // Use std::round() to avoid systematic bias in fixed-point conversion
        int64_t value_fixed = static_cast<int64_t>(std::round(value * 10000.0f));
        value_sum_fixed.fetch_add(value_fixed, std::memory_order_relaxed);
        // Accumulate v^2 for variance tracking (v^2 * 10000, not v^2 * 10000^2)
        int64_t value_sq_fixed = static_cast<int64_t>(std::round(value * value * 10000.0f));
        value_sum_sq_fixed.fetch_add(value_sq_fixed, std::memory_order_relaxed);
        visit_count.fetch_add(1, std::memory_order_relaxed);
    }

    // Update root node with release semantics (prevents race conditions)
    void update_root(float value) {
        int64_t value_fixed = static_cast<int64_t>(std::round(value * 10000.0f));
        value_sum_fixed.fetch_add(value_fixed, std::memory_order_relaxed);
        int64_t value_sq_fixed = static_cast<int64_t>(std::round(value * value * 10000.0f));
        value_sum_sq_fixed.fetch_add(value_sq_fixed, std::memory_order_relaxed);
        // Use release semantics on root to ensure all updates are visible
        visit_count.fetch_add(1, std::memory_order_release);
    }
};

// Verify alignment at compile time
static_assert(sizeof(Node) == 64, "Node must be exactly 64 bytes");
static_assert(alignof(Node) == 64, "Node must be 64-byte aligned");

} // namespace mcts
