#pragma once

#include <atomic>
#include <cstdint>
#include <vector>
#include <cmath>
#include "../third_party/chess-library/include/chess.hpp"

namespace mcts {

// Forward declarations
class NodePool;

// Node structure for MCTS tree
// CRITICAL: 64-byte aligned to prevent false sharing between CPU cores
// Each node fits exactly in one cache line for optimal performance
struct alignas(64) Node {
    // Parent node (nullptr for root)
    Node* parent;

    // First child in linked list (nullptr if no children)
    Node* first_child;

    // Next sibling in parent's child list (nullptr if last child)
    Node* next_sibling;

    // Move that led to this node (from parent's perspective)
    chess::Move move;

    // Visit count (atomic for thread-safe MCTS)
    std::atomic<uint32_t> visit_count;

    // Virtual loss for parallel MCTS (prevents multiple threads from exploring same path)
    std::atomic<int16_t> virtual_loss;

    // Value sum in fixed-point (multiply by 10000 for precision)
    // Using int64_t to prevent overflow: max value = 2^31 visits * 10000 * 1.0 = ~2^45
    std::atomic<int64_t> value_sum_fixed;

    // Policy prior from neural network (stored as uint16_t to save space)
    // Actual value = prior_fixed / 10000.0f
    uint16_t prior_fixed;

    // Flags: terminal (bit 0), expanded (bit 1)
    uint8_t flags;

    // Padding to reach exactly 64 bytes
    uint8_t padding[5];

    // Constructor
    Node()
        : parent(nullptr)
        , first_child(nullptr)
        , next_sibling(nullptr)
        , move()
        , visit_count(0)
        , virtual_loss(0)
        , value_sum_fixed(0)
        , prior_fixed(0)
        , flags(0)
    {
        static_assert(sizeof(Node) == 64, "Node must be exactly 64 bytes for cache-line alignment");
    }

    // Q-value calculation with virtual loss and FPU (First Play Urgency)
    // Uses Leela Chess Zero approach: unvisited nodes use parent Q-value as placeholder
    float q_value(float parent_q) const {
        uint32_t n = visit_count.load(std::memory_order_relaxed);
        int16_t v = virtual_loss.load(std::memory_order_relaxed);

        // Unvisited node with no virtual loss: use FPU (First Play Urgency)
        if (n == 0 && v == 0) {
            return parent_q - 0.2f;  // Slight penalty to encourage exploration
        }

        // Unvisited node with virtual loss: use parent Q-value (Leela approach)
        if (n == 0 && v > 0) {
            return parent_q;
        }

        // Visited node: compute average value including virtual loss
        int64_t sum = value_sum_fixed.load(std::memory_order_relaxed);
        float real_value = sum / 10000.0f;
        float total_value = real_value + v * parent_q;
        return total_value / (n + v);
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

    // Add virtual loss (called when thread starts exploring this path)
    void add_virtual_loss() {
        virtual_loss.fetch_add(1, std::memory_order_relaxed);
    }

    // Remove virtual loss (called when thread finishes exploring this path)
    void remove_virtual_loss() {
        virtual_loss.fetch_sub(1, std::memory_order_relaxed);
    }

    // Update node with backpropagation value
    void update(float value) {
        // Use std::round() to avoid systematic bias in fixed-point conversion
        int64_t value_fixed = static_cast<int64_t>(std::round(value * 10000.0f));
        value_sum_fixed.fetch_add(value_fixed, std::memory_order_relaxed);
        visit_count.fetch_add(1, std::memory_order_relaxed);
    }

    // Update root node with release semantics (prevents race conditions)
    void update_root(float value) {
        int64_t value_fixed = static_cast<int64_t>(std::round(value * 10000.0f));
        value_sum_fixed.fetch_add(value_fixed, std::memory_order_relaxed);
        // Use release semantics on root to ensure all updates are visible
        visit_count.fetch_add(1, std::memory_order_release);
    }
};

// Verify alignment at compile time
static_assert(sizeof(Node) == 64, "Node must be exactly 64 bytes");
static_assert(alignof(Node) == 64, "Node must be 64-byte aligned");

} // namespace mcts
