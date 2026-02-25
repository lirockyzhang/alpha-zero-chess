// Tests for NodePool arena allocator changes:
//
//   1. Block reuse after reset() — allocate() reuses existing blocks instead of
//      pushing new ones, preventing unbounded growth and MAX_BLOCKS exhaustion.
//   2. shrink(keep_blocks) — frees excess blocks beyond keep_blocks, reclaiming
//      memory from outlier searches between games.
//   3. Split metrics — memory_reserved() (total physical) vs memory_used() (active
//      nodes since last reset).
//   4. Node initialization — placement new zeroes all fields on reused memory.
//   5. Stress tests — simulates the parallel_coordinator lifecycle pattern.

#include "mcts/node_pool.hpp"
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cmath>

using namespace mcts;

static constexpr size_t NODES_PER_BLOCK = NodePool::NODES_PER_BLOCK;  // 16384

// ──────────────────────────────────────────────────────────────────────────
// Test 1 — Block reuse: reset() + allocate() reuses blocks, no growth
// ──────────────────────────────────────────────────────────────────────────
bool test_block_reuse_after_reset() {
    std::cout << "=== Test 1: Block reuse after reset ===" << std::endl;

    NodePool pool;

    // Allocate enough nodes to span 3 blocks (2 full + partial third)
    size_t target = NODES_PER_BLOCK * 2 + 100;
    for (size_t i = 0; i < target; ++i) {
        pool.allocate();
    }

    size_t reserved_before = pool.memory_reserved();
    size_t size_before = pool.size();
    std::cout << "  After initial allocation: size=" << size_before
              << ", reserved=" << reserved_before << " bytes ("
              << reserved_before / (1024 * 1024) << " MB)" << std::endl;

    if (size_before != target) {
        std::cout << "  FAIL: expected size " << target << ", got " << size_before << std::endl;
        return false;
    }

    // Reset and allocate again — should reuse existing blocks
    pool.reset();
    for (size_t i = 0; i < target; ++i) {
        pool.allocate();
    }

    size_t reserved_after = pool.memory_reserved();
    size_t size_after = pool.size();
    std::cout << "  After reset + re-allocation: size=" << size_after
              << ", reserved=" << reserved_after << " bytes" << std::endl;

    if (reserved_after != reserved_before) {
        std::cout << "  FAIL: reserved memory grew from " << reserved_before
                  << " to " << reserved_after << " (blocks not reused)" << std::endl;
        return false;
    }

    if (size_after != target) {
        std::cout << "  FAIL: expected size " << target << ", got " << size_after << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 2 — Repeated reset/allocate cycles don't grow block count
// ──────────────────────────────────────────────────────────────────────────
bool test_no_growth_across_many_cycles() {
    std::cout << "=== Test 2: No block growth across 500 reset/allocate cycles ===" << std::endl;

    NodePool pool;

    // Fill 3 blocks initially
    size_t target = NODES_PER_BLOCK * 3;
    for (size_t i = 0; i < target; ++i) {
        pool.allocate();
    }

    size_t reserved_baseline = pool.memory_reserved();
    std::cout << "  Baseline reserved: " << reserved_baseline << " bytes ("
              << reserved_baseline / (1024 * 1024) << " MB, 3 blocks)" << std::endl;

    // 500 cycles of reset + re-allocate (fewer nodes each time to stay within blocks)
    for (int cycle = 0; cycle < 500; ++cycle) {
        pool.reset();
        // Allocate varying amounts, always <= 3 blocks
        size_t count = NODES_PER_BLOCK * ((cycle % 3) + 1);
        for (size_t i = 0; i < count; ++i) {
            pool.allocate();
        }
    }

    size_t reserved_final = pool.memory_reserved();
    std::cout << "  After 500 cycles: reserved=" << reserved_final << " bytes" << std::endl;

    if (reserved_final != reserved_baseline) {
        std::cout << "  FAIL: reserved memory changed from " << reserved_baseline
                  << " to " << reserved_final << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 3 — Block reuse: exceeding previous block count allocates new blocks
// ──────────────────────────────────────────────────────────────────────────
bool test_block_reuse_then_grow() {
    std::cout << "=== Test 3: Reuse blocks, then grow when needed ===" << std::endl;

    NodePool pool;

    // Fill 2 blocks
    for (size_t i = 0; i < NODES_PER_BLOCK * 2; ++i) {
        pool.allocate();
    }
    size_t reserved_2_blocks = pool.memory_reserved();
    std::cout << "  After 2 blocks: reserved=" << reserved_2_blocks << std::endl;

    // Reset and allocate 4 blocks worth — should reuse 2, allocate 2 new
    pool.reset();
    for (size_t i = 0; i < NODES_PER_BLOCK * 4; ++i) {
        pool.allocate();
    }
    size_t reserved_4_blocks = pool.memory_reserved();
    std::cout << "  After reset + 4 blocks: reserved=" << reserved_4_blocks << std::endl;

    if (reserved_4_blocks != reserved_2_blocks * 2) {
        std::cout << "  FAIL: expected " << reserved_2_blocks * 2
                  << " but got " << reserved_4_blocks << std::endl;
        return false;
    }

    if (pool.size() != NODES_PER_BLOCK * 4) {
        std::cout << "  FAIL: expected size " << NODES_PER_BLOCK * 4
                  << " but got " << pool.size() << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 4 — shrink(2) frees blocks beyond 2
// ──────────────────────────────────────────────────────────────────────────
bool test_shrink_keeps_requested_blocks() {
    std::cout << "=== Test 4: shrink(2) frees blocks beyond 2 ===" << std::endl;

    NodePool pool;

    // Allocate 5 blocks
    for (size_t i = 0; i < NODES_PER_BLOCK * 5; ++i) {
        pool.allocate();
    }
    size_t reserved_5 = pool.memory_reserved();
    std::cout << "  Before shrink: reserved=" << reserved_5 << " bytes (5 blocks)" << std::endl;

    pool.shrink(2);

    size_t reserved_after = pool.memory_reserved();
    size_t expected = 2 * NODES_PER_BLOCK * sizeof(Node);
    std::cout << "  After shrink(2): reserved=" << reserved_after << " bytes" << std::endl;

    if (reserved_after != expected) {
        std::cout << "  FAIL: expected reserved=" << expected
                  << " (2 blocks), got " << reserved_after << std::endl;
        return false;
    }

    if (pool.size() != 0) {
        std::cout << "  FAIL: expected size=0 after shrink, got " << pool.size() << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 5 — shrink(1) keeps exactly 1 block
// ──────────────────────────────────────────────────────────────────────────
bool test_shrink_to_one() {
    std::cout << "=== Test 5: shrink(1) keeps exactly 1 block ===" << std::endl;

    NodePool pool;

    for (size_t i = 0; i < NODES_PER_BLOCK * 4; ++i) {
        pool.allocate();
    }
    pool.shrink(1);

    size_t reserved = pool.memory_reserved();
    size_t expected = 1 * NODES_PER_BLOCK * sizeof(Node);

    if (reserved != expected) {
        std::cout << "  FAIL: expected reserved=" << expected
                  << " (1 block), got " << reserved << std::endl;
        return false;
    }

    if (pool.size() != 0) {
        std::cout << "  FAIL: expected size=0 after shrink, got " << pool.size() << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 6 — shrink(0) is safe (keep_blocks >= 1 guard)
// ──────────────────────────────────────────────────────────────────────────
bool test_shrink_zero_safe() {
    std::cout << "=== Test 6: shrink(0) is safe (guard prevents freeing all) ===" << std::endl;

    NodePool pool;

    for (size_t i = 0; i < NODES_PER_BLOCK * 3; ++i) {
        pool.allocate();
    }
    size_t reserved_before = pool.memory_reserved();

    // shrink(0) should be a no-op because the guard requires keep_blocks >= 1
    pool.shrink(0);

    size_t reserved_after = pool.memory_reserved();
    std::cout << "  Before shrink(0): reserved=" << reserved_before
              << ", after: reserved=" << reserved_after << std::endl;

    // The while loop condition is: blocks_.size() > keep_blocks && keep_blocks >= 1
    // With keep_blocks=0, the second condition (keep_blocks >= 1) is false,
    // so no blocks are freed. But reset() is still called.
    if (reserved_after != reserved_before) {
        std::cout << "  FAIL: shrink(0) should not free blocks, but reserved changed" << std::endl;
        return false;
    }

    // But reset() was still called, so size should be 0
    if (pool.size() != 0) {
        std::cout << "  FAIL: expected size=0 after shrink(0) (reset called), got "
                  << pool.size() << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 7 — Allocation works correctly after shrink
// ──────────────────────────────────────────────────────────────────────────
bool test_allocate_after_shrink() {
    std::cout << "=== Test 7: Allocation works after shrink ===" << std::endl;

    NodePool pool;

    // Fill 5 blocks, shrink to 2, then allocate across 3 blocks
    for (size_t i = 0; i < NODES_PER_BLOCK * 5; ++i) {
        pool.allocate();
    }
    pool.shrink(2);

    // Allocate 3 blocks worth — should reuse 2 existing + allocate 1 new
    for (size_t i = 0; i < NODES_PER_BLOCK * 3; ++i) {
        pool.allocate();
    }

    size_t expected_reserved = 3 * NODES_PER_BLOCK * sizeof(Node);
    size_t expected_size = NODES_PER_BLOCK * 3;

    if (pool.memory_reserved() != expected_reserved) {
        std::cout << "  FAIL: expected reserved=" << expected_reserved
                  << ", got " << pool.memory_reserved() << std::endl;
        return false;
    }

    if (pool.size() != expected_size) {
        std::cout << "  FAIL: expected size=" << expected_size
                  << ", got " << pool.size() << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 8 — memory_reserved() vs memory_used() semantics
// ──────────────────────────────────────────────────────────────────────────
bool test_reserved_vs_used() {
    std::cout << "=== Test 8: memory_reserved() vs memory_used() ===" << std::endl;

    NodePool pool;

    // Initially: 1 block allocated by constructor, 0 nodes used
    size_t initial_reserved = pool.memory_reserved();
    size_t initial_used = pool.memory_used();
    std::cout << "  Initial: reserved=" << initial_reserved
              << ", used=" << initial_used << std::endl;

    if (initial_reserved != NODES_PER_BLOCK * sizeof(Node)) {
        std::cout << "  FAIL: expected initial reserved=" << NODES_PER_BLOCK * sizeof(Node)
                  << " (1 block)" << std::endl;
        return false;
    }
    if (initial_used != 0) {
        std::cout << "  FAIL: expected initial used=0, got " << initial_used << std::endl;
        return false;
    }

    // Allocate 100 nodes
    for (int i = 0; i < 100; ++i) {
        pool.allocate();
    }
    size_t used_100 = pool.memory_used();
    size_t reserved_100 = pool.memory_reserved();
    std::cout << "  After 100 nodes: reserved=" << reserved_100
              << ", used=" << used_100 << std::endl;

    if (used_100 != 100 * sizeof(Node)) {
        std::cout << "  FAIL: expected used=" << 100 * sizeof(Node)
                  << ", got " << used_100 << std::endl;
        return false;
    }
    if (reserved_100 < used_100) {
        std::cout << "  FAIL: reserved < used (invariant violation)" << std::endl;
        return false;
    }

    // Fill into 3 blocks, then reset — used drops to 0, reserved stays
    for (size_t i = 0; i < NODES_PER_BLOCK * 2; ++i) {
        pool.allocate();
    }
    size_t reserved_3_blocks = pool.memory_reserved();

    pool.reset();
    if (pool.memory_used() != 0) {
        std::cout << "  FAIL: memory_used() should be 0 after reset, got "
                  << pool.memory_used() << std::endl;
        return false;
    }
    if (pool.memory_reserved() != reserved_3_blocks) {
        std::cout << "  FAIL: memory_reserved() changed after reset" << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 9 — reserved >= used invariant across many operations
// ──────────────────────────────────────────────────────────────────────────
bool test_reserved_ge_used_invariant() {
    std::cout << "=== Test 9: reserved >= used invariant ===" << std::endl;

    NodePool pool;

    for (int cycle = 0; cycle < 50; ++cycle) {
        // Allocate a variable number of nodes
        size_t count = (cycle * 1000 + 500) % (NODES_PER_BLOCK * 3);
        if (count == 0) count = 1;
        for (size_t i = 0; i < count; ++i) {
            pool.allocate();
        }

        if (pool.memory_reserved() < pool.memory_used()) {
            std::cout << "  FAIL at cycle " << cycle
                      << ": reserved=" << pool.memory_reserved()
                      << " < used=" << pool.memory_used() << std::endl;
            return false;
        }

        if (cycle % 3 == 0) {
            pool.reset();
        } else if (cycle % 7 == 0) {
            pool.shrink(2);
        }

        if (pool.memory_reserved() < pool.memory_used()) {
            std::cout << "  FAIL after reset/shrink at cycle " << cycle << std::endl;
            return false;
        }
    }

    std::cout << "  PASS" << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 10 — Node initialization: placement new zeroes all fields on reuse
// ──────────────────────────────────────────────────────────────────────────
bool test_node_initialization_on_reuse() {
    std::cout << "=== Test 10: Node fields zeroed on reused memory ===" << std::endl;

    NodePool pool;

    // Allocate nodes and dirty them
    constexpr size_t N = 1000;
    Node* nodes[N];
    for (size_t i = 0; i < N; ++i) {
        nodes[i] = pool.allocate();
        // Set all fields to non-zero values
        nodes[i]->parent = reinterpret_cast<Node*>(0xDEADBEEF);
        nodes[i]->first_child = reinterpret_cast<Node*>(0xCAFEBABE);
        nodes[i]->next_sibling = reinterpret_cast<Node*>(0xBAADF00D);
        nodes[i]->value_sum_fixed.store(99999, std::memory_order_relaxed);
        nodes[i]->value_sum_sq_fixed.store(88888, std::memory_order_relaxed);
        nodes[i]->visit_count.store(42, std::memory_order_relaxed);
        nodes[i]->prior_fixed = 5000;
        nodes[i]->flags = 0xFF;
    }

    // Reset and reallocate — nodes should land on the same memory
    pool.reset();

    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        Node* node = pool.allocate();
        // Placement new in allocate() should have zeroed everything
        if (node->parent != nullptr) {
            std::cout << "  FAIL: node[" << i << "] parent not null" << std::endl;
            ok = false; break;
        }
        if (node->first_child != nullptr) {
            std::cout << "  FAIL: node[" << i << "] first_child not null" << std::endl;
            ok = false; break;
        }
        if (node->next_sibling != nullptr) {
            std::cout << "  FAIL: node[" << i << "] next_sibling not null" << std::endl;
            ok = false; break;
        }
        if (node->value_sum_fixed.load(std::memory_order_relaxed) != 0) {
            std::cout << "  FAIL: node[" << i << "] value_sum_fixed not 0" << std::endl;
            ok = false; break;
        }
        if (node->value_sum_sq_fixed.load(std::memory_order_relaxed) != 0) {
            std::cout << "  FAIL: node[" << i << "] value_sum_sq_fixed not 0" << std::endl;
            ok = false; break;
        }
        if (node->visit_count.load(std::memory_order_relaxed) != 0) {
            std::cout << "  FAIL: node[" << i << "] visit_count not 0" << std::endl;
            ok = false; break;
        }
        if (node->prior_fixed != 0) {
            std::cout << "  FAIL: node[" << i << "] prior_fixed not 0" << std::endl;
            ok = false; break;
        }
        if (node->flags != 0) {
            std::cout << "  FAIL: node[" << i << "] flags not 0" << std::endl;
            ok = false; break;
        }
    }

    if (!ok) return false;
    std::cout << "  PASS" << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 11 — Node initialization across block boundaries
// ──────────────────────────────────────────────────────────────────────────
bool test_node_initialization_across_blocks() {
    std::cout << "=== Test 11: Node initialization across block boundaries ===" << std::endl;

    NodePool pool;

    // Fill 2 full blocks + a few into the third, and dirty all nodes
    size_t target = NODES_PER_BLOCK * 2 + 50;
    for (size_t i = 0; i < target; ++i) {
        Node* node = pool.allocate();
        node->visit_count.store(999, std::memory_order_relaxed);
        node->value_sum_fixed.store(-77777, std::memory_order_relaxed);
        node->flags = 0xFF;
    }

    pool.reset();

    // Check first node in each block boundary region
    size_t check_positions[] = {
        0,                        // first node of block 0
        NODES_PER_BLOCK - 1,     // last node of block 0
        NODES_PER_BLOCK,         // first node of block 1 (boundary!)
        NODES_PER_BLOCK * 2 - 1, // last node of block 1
        NODES_PER_BLOCK * 2,     // first node of block 2 (boundary!)
        NODES_PER_BLOCK * 2 + 49 // last node we dirtied
    };

    // Allocate up to the last check position
    size_t max_pos = NODES_PER_BLOCK * 2 + 50;
    std::vector<Node*> new_nodes(max_pos);
    for (size_t i = 0; i < max_pos; ++i) {
        new_nodes[i] = pool.allocate();
    }

    bool ok = true;
    for (size_t pos : check_positions) {
        Node* node = new_nodes[pos];
        if (node->visit_count.load(std::memory_order_relaxed) != 0 ||
            node->value_sum_fixed.load(std::memory_order_relaxed) != 0 ||
            node->flags != 0 ||
            node->parent != nullptr) {
            std::cout << "  FAIL: node at position " << pos << " not properly initialized"
                      << std::endl;
            ok = false;
            break;
        }
    }

    if (!ok) return false;
    std::cout << "  PASS" << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 12 — Stress: parallel_coordinator lifecycle (reset per move, shrink per game)
// ──────────────────────────────────────────────────────────────────────────
bool test_parallel_coordinator_lifecycle() {
    std::cout << "=== Test 12: Simulate parallel_coordinator lifecycle ===" << std::endl;

    NodePool pool;

    // Simulate 20 games
    for (int game = 0; game < 20; ++game) {
        // Each game: 30-80 moves (half-moves)
        int moves = 30 + (game * 17) % 51;  // deterministic variation

        for (int move = 0; move < moves; ++move) {
            pool.reset();  // reset per move

            // Each move: 200-1000 MCTS nodes (sims expand the tree)
            size_t nodes = 200 + (move * 37) % 801;
            for (size_t i = 0; i < nodes; ++i) {
                Node* node = pool.allocate();
                // Simulate MCTS: set some fields
                node->visit_count.store(1, std::memory_order_relaxed);
                node->value_sum_fixed.store(5000, std::memory_order_relaxed);
            }

            if (pool.size() != nodes) {
                std::cout << "  FAIL at game " << game << " move " << move
                          << ": expected size=" << nodes << ", got " << pool.size() << std::endl;
                return false;
            }
        }

        // Between games: shrink to reclaim memory
        pool.shrink(2);
    }

    // Final state: max nodes per move is 1000, which fits in 1 block (16384 nodes/block).
    // shrink(2) keeps up to 2 blocks but only 1 was ever allocated, so 1 remains.
    size_t final_reserved = pool.memory_reserved();
    size_t max_expected = 2 * NODES_PER_BLOCK * sizeof(Node);
    if (final_reserved > max_expected) {
        std::cout << "  FAIL: after all games, reserved=" << final_reserved
                  << " exceeds max expected=" << max_expected << std::endl;
        return false;
    }
    if (pool.size() != 0) {
        std::cout << "  FAIL: expected size=0 after final shrink, got " << pool.size() << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 13 — Stress: many small allocations with field verification
// ──────────────────────────────────────────────────────────────────────────
bool test_stress_field_integrity() {
    std::cout << "=== Test 13: Stress field integrity across many cycles ===" << std::endl;

    NodePool pool;

    for (int cycle = 0; cycle < 200; ++cycle) {
        pool.reset();

        // Allocate a block's worth of nodes and tag them
        constexpr size_t N = 500;
        Node* nodes[N];
        for (size_t i = 0; i < N; ++i) {
            nodes[i] = pool.allocate();
            // Every node should start clean
            if (nodes[i]->visit_count.load(std::memory_order_relaxed) != 0 ||
                nodes[i]->parent != nullptr ||
                nodes[i]->first_child != nullptr ||
                nodes[i]->flags != 0) {
                std::cout << "  FAIL at cycle " << cycle << " node " << i
                          << ": not properly initialized" << std::endl;
                return false;
            }
            // Now dirty them for the next cycle
            nodes[i]->visit_count.store(static_cast<uint32_t>(cycle + i),
                                        std::memory_order_relaxed);
            nodes[i]->value_sum_fixed.store(cycle * 10000 + static_cast<int64_t>(i),
                                            std::memory_order_relaxed);
            nodes[i]->flags = 0x03;  // terminal + expanded
            nodes[i]->prior_fixed = 9999;
        }
    }

    std::cout << "  PASS (200 cycles, 500 nodes each, all initialized correctly)" << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 14 — size() correctness across reset and partial fills
// ──────────────────────────────────────────────────────────────────────────
bool test_size_across_operations() {
    std::cout << "=== Test 14: size() correctness across operations ===" << std::endl;

    NodePool pool;

    // Empty pool
    if (pool.size() != 0) {
        std::cout << "  FAIL: initial size should be 0, got " << pool.size() << std::endl;
        return false;
    }

    // Allocate 1 node
    pool.allocate();
    if (pool.size() != 1) {
        std::cout << "  FAIL: expected size 1 after 1 allocation" << std::endl;
        return false;
    }

    // Fill exactly 1 block
    for (size_t i = 1; i < NODES_PER_BLOCK; ++i) {
        pool.allocate();
    }
    if (pool.size() != NODES_PER_BLOCK) {
        std::cout << "  FAIL: expected size " << NODES_PER_BLOCK << " after filling 1 block, got "
                  << pool.size() << std::endl;
        return false;
    }

    // Add 1 more (crosses into block 2)
    pool.allocate();
    if (pool.size() != NODES_PER_BLOCK + 1) {
        std::cout << "  FAIL: expected size " << NODES_PER_BLOCK + 1
                  << " after crossing into block 2" << std::endl;
        return false;
    }

    // Reset
    pool.reset();
    if (pool.size() != 0) {
        std::cout << "  FAIL: expected size 0 after reset" << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 15 — Node alignment: every allocated node is 64-byte aligned
// ──────────────────────────────────────────────────────────────────────────
bool test_node_alignment() {
    std::cout << "=== Test 15: Node 64-byte alignment ===" << std::endl;

    NodePool pool;

    // Check alignment across two blocks
    size_t count = NODES_PER_BLOCK + 100;
    for (size_t i = 0; i < count; ++i) {
        Node* node = pool.allocate();
        uintptr_t addr = reinterpret_cast<uintptr_t>(node);
        if (addr % 64 != 0) {
            std::cout << "  FAIL: node " << i << " at address 0x" << std::hex << addr
                      << " not 64-byte aligned" << std::dec << std::endl;
            return false;
        }
    }

    std::cout << "  PASS (" << count << " nodes all 64-byte aligned)" << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 16 — shrink with keep_blocks larger than current block count
// ──────────────────────────────────────────────────────────────────────────
bool test_shrink_larger_than_current() {
    std::cout << "=== Test 16: shrink(10) with only 2 blocks ===" << std::endl;

    NodePool pool;

    // Allocate into 2 blocks
    for (size_t i = 0; i < NODES_PER_BLOCK + 100; ++i) {
        pool.allocate();
    }
    size_t reserved_before = pool.memory_reserved();

    // shrink(10) should not free anything (only 2 blocks exist)
    pool.shrink(10);

    if (pool.memory_reserved() != reserved_before) {
        std::cout << "  FAIL: shrink(10) freed blocks when only 2 existed" << std::endl;
        return false;
    }

    if (pool.size() != 0) {
        std::cout << "  FAIL: expected size=0 after shrink (reset still called)" << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "\n========== NodePool Unit Tests ==========\n" << std::endl;

    struct TestCase {
        const char* name;
        bool (*func)();
    };

    TestCase tests[] = {
        {"Block reuse after reset",             test_block_reuse_after_reset},
        {"No growth across 500 cycles",         test_no_growth_across_many_cycles},
        {"Reuse blocks then grow",              test_block_reuse_then_grow},
        {"shrink(2) frees excess",              test_shrink_keeps_requested_blocks},
        {"shrink(1) keeps 1 block",             test_shrink_to_one},
        {"shrink(0) safe guard",                test_shrink_zero_safe},
        {"Allocate after shrink",               test_allocate_after_shrink},
        {"reserved vs used semantics",          test_reserved_vs_used},
        {"reserved >= used invariant",          test_reserved_ge_used_invariant},
        {"Node init on reused memory",          test_node_initialization_on_reuse},
        {"Node init across block boundaries",   test_node_initialization_across_blocks},
        {"Parallel coordinator lifecycle",      test_parallel_coordinator_lifecycle},
        {"Stress field integrity",              test_stress_field_integrity},
        {"size() across operations",            test_size_across_operations},
        {"Node 64-byte alignment",              test_node_alignment},
        {"shrink larger than current",          test_shrink_larger_than_current},
    };

    int total = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    int failed = 0;

    for (int i = 0; i < total; ++i) {
        std::cout << std::endl;
        bool ok = tests[i].func();
        if (ok) {
            ++passed;
        } else {
            ++failed;
            std::cout << "  >>> FAILED: " << tests[i].name << std::endl;
        }
    }

    std::cout << "\n=========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " passed";
    if (failed > 0) {
        std::cout << ", " << failed << " FAILED";
    }
    std::cout << std::endl;
    std::cout << "=========================================\n" << std::endl;

    return failed > 0 ? 1 : 0;
}
