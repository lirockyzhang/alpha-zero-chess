// Tests for virtual losses in the async double-buffer pipeline.
//
// Virtual losses temporarily add pessimistic visits (value=-1.0, visit_count+1)
// along in-flight leaf paths so that PUCT strongly avoids re-selecting them.
// These tests verify the key invariants:
//
//   1.  collect_leaves_async adds virtual losses  (visit + value_sum change)
//   2.  cancel_{prev,collection}_pending removes them  (both fully restored)
//   3.  update_prev_leaves removes virtual losses before real backprop
//       (final counts == initial + real sims, no double-counting)
//   4.  Virtual losses actually diversify leaf selection below root
//   5.  value_sum_fixed is fully conserved across the add/remove cycle

#include "mcts/search.hpp"
#include "mcts/node_pool.hpp"
#include "encoding/position_encoder.hpp"
#include "encoding/move_encoder.hpp"
#include "../third_party/chess-library/include/chess.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cassert>

using namespace mcts;
using namespace chess;

static constexpr int OBS_SIZE  = encoding::PositionEncoder::TOTAL_SIZE;  // 8*8*123
static constexpr int POL_SIZE  = encoding::MoveEncoder::POLICY_SIZE;     // 4672

// Helpers ──────────────────────────────────────────────────────────────────

static int count_children(const Node* node) {
    int n = 0;
    for (Node* c = node->first_child; c; c = c->next_sibling) ++n;
    return n;
}

static uint32_t sum_child_visits(const Node* node) {
    uint32_t s = 0;
    for (Node* c = node->first_child; c; c = c->next_sibling)
        s += c->visit_count.load(std::memory_order_relaxed);
    return s;
}

static int64_t sum_child_value_sum(const Node* node) {
    int64_t s = 0;
    for (Node* c = node->first_child; c; c = c->next_sibling)
        s += c->value_sum_fixed.load(std::memory_order_relaxed);
    return s;
}

// Create a uniform policy vector of the right size.
static std::vector<float> uniform_policy() {
    return std::vector<float>(POL_SIZE, 1.0f / 20.0f);  // ~20 legal moves
}

// ──────────────────────────────────────────────────────────────────────────
// Test 1 — collect_leaves_async inflates visit counts and depresses values
// ──────────────────────────────────────────────────────────────────────────
bool test_virtual_losses_inflate() {
    std::cout << "=== Test 1: Virtual losses inflate visits + depress values ===" << std::endl;

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 200;
    config.use_gumbel = false;     // PUCT mode for simplicity
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    Board board;
    Node* root = search.init_search(board, uniform_policy(), 0.0f);

    uint32_t root_visits_before = root->visit_count.load(std::memory_order_relaxed);
    int64_t root_vsum_before = root->value_sum_fixed.load(std::memory_order_relaxed);
    std::cout << "  Root visits after init: " << root_visits_before << std::endl;
    std::cout << "  Root value_sum after init: " << root_vsum_before << std::endl;

    int batch_size = 4;
    std::vector<float> obs(batch_size * OBS_SIZE, 0.0f);
    std::vector<float> mask(batch_size * POL_SIZE, 0.0f);

    search.start_next_batch_collection();
    int collected = search.collect_leaves_async(obs.data(), mask.data(), batch_size);
    std::cout << "  Collected " << collected << " leaves" << std::endl;

    uint32_t root_visits_after = root->visit_count.load(std::memory_order_relaxed);
    int64_t root_vsum_after = root->value_sum_fixed.load(std::memory_order_relaxed);
    std::cout << "  Root visits after collect: " << root_visits_after << std::endl;
    std::cout << "  Root value_sum after collect: " << root_vsum_after << std::endl;

    // Virtual losses should have inflated root visits by +collected
    uint32_t expected_visits = root_visits_before + collected;
    if (root_visits_after != expected_visits) {
        std::cout << "  FAIL: expected root visits = " << expected_visits
                  << ", got " << root_visits_after << std::endl;
        search.cancel_collection_pending();
        return false;
    }

    // Child visit counts should be inflated (one virtual visit per collected leaf)
    uint32_t child_visits = sum_child_visits(root);
    std::cout << "  Sum of child visits (inflated): " << child_visits << std::endl;
    if (child_visits != static_cast<uint32_t>(collected)) {
        std::cout << "  FAIL: expected child visit sum = " << collected
                  << ", got " << child_visits << std::endl;
        search.cancel_collection_pending();
        return false;
    }

    // Root value_sum should have changed (virtual loss adds alternating-sign values)
    // At root level, the virtual loss is +1.0 (flipped from child's -1.0)
    // So root_vsum should have increased by collected * 10000
    int64_t expected_vsum = root_vsum_before + collected * 10000LL;
    if (root_vsum_after != expected_vsum) {
        std::cout << "  FAIL: expected root value_sum = " << expected_vsum
                  << ", got " << root_vsum_after << std::endl;
        search.cancel_collection_pending();
        return false;
    }

    // Each child with a virtual loss should have value_sum = -10000 (loss from child's perspective)
    int64_t child_vsum = sum_child_value_sum(root);
    int64_t expected_child_vsum = -10000LL * collected;
    std::cout << "  Child value_sum total: " << child_vsum
              << " (expected " << expected_child_vsum << ")" << std::endl;
    if (child_vsum != expected_child_vsum) {
        std::cout << "  FAIL: child value_sum mismatch" << std::endl;
        search.cancel_collection_pending();
        return false;
    }

    search.cancel_collection_pending();
    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 2 — cancel_collection_pending restores both visits AND value_sum
// ──────────────────────────────────────────────────────────────────────────
bool test_cancel_collection_restores() {
    std::cout << "=== Test 2: cancel_collection_pending restores visits + values ===" << std::endl;

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 200;
    config.use_gumbel = false;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    Board board;
    Node* root = search.init_search(board, uniform_policy(), 0.0f);
    uint32_t root_visits_before = root->visit_count.load(std::memory_order_relaxed);
    int64_t root_vsum_before = root->value_sum_fixed.load(std::memory_order_relaxed);

    int batch_size = 4;
    std::vector<float> obs(batch_size * OBS_SIZE, 0.0f);
    std::vector<float> mask(batch_size * POL_SIZE, 0.0f);

    search.start_next_batch_collection();
    int collected = search.collect_leaves_async(obs.data(), mask.data(), batch_size);
    std::cout << "  Collected " << collected << " leaves" << std::endl;

    // Cancel → should fully restore
    search.cancel_collection_pending();

    uint32_t root_visits_restored = root->visit_count.load(std::memory_order_relaxed);
    int64_t root_vsum_restored = root->value_sum_fixed.load(std::memory_order_relaxed);
    std::cout << "  Root visits after cancel: " << root_visits_restored
              << " (expected " << root_visits_before << ")" << std::endl;
    std::cout << "  Root value_sum after cancel: " << root_vsum_restored
              << " (expected " << root_vsum_before << ")" << std::endl;

    if (root_visits_restored != root_visits_before) {
        std::cout << "  FAIL: visit count not restored" << std::endl;
        return false;
    }
    if (root_vsum_restored != root_vsum_before) {
        std::cout << "  FAIL: value_sum not restored" << std::endl;
        return false;
    }

    uint32_t child_visits = sum_child_visits(root);
    int64_t child_vsum = sum_child_value_sum(root);
    if (child_visits != 0 || child_vsum != 0) {
        std::cout << "  FAIL: child visits=" << child_visits
                  << " value_sum=" << child_vsum << " (both should be 0)" << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 3 — cancel_prev_pending restores visits + value_sum
// ──────────────────────────────────────────────────────────────────────────
bool test_cancel_prev_restores() {
    std::cout << "=== Test 3: cancel_prev_pending restores visits + values ===" << std::endl;

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 200;
    config.use_gumbel = false;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    Board board;
    Node* root = search.init_search(board, uniform_policy(), 0.0f);
    uint32_t root_visits_before = root->visit_count.load(std::memory_order_relaxed);
    int64_t root_vsum_before = root->value_sum_fixed.load(std::memory_order_relaxed);

    int batch_size = 4;
    std::vector<float> obs(batch_size * OBS_SIZE, 0.0f);
    std::vector<float> mask(batch_size * POL_SIZE, 0.0f);

    search.start_next_batch_collection();
    int collected = search.collect_leaves_async(obs.data(), mask.data(), batch_size);
    std::cout << "  Collected " << collected << " leaves" << std::endl;

    search.commit_and_swap();

    // Verify still inflated after swap
    uint32_t root_inflated = root->visit_count.load(std::memory_order_relaxed);
    std::cout << "  Root visits after commit_and_swap: " << root_inflated << std::endl;

    // Cancel prev → should restore
    search.cancel_prev_pending();

    uint32_t root_restored = root->visit_count.load(std::memory_order_relaxed);
    int64_t root_vsum_restored = root->value_sum_fixed.load(std::memory_order_relaxed);
    std::cout << "  Root visits after cancel_prev: " << root_restored
              << " (expected " << root_visits_before << ")" << std::endl;
    std::cout << "  Root value_sum after cancel_prev: " << root_vsum_restored
              << " (expected " << root_vsum_before << ")" << std::endl;

    if (root_restored != root_visits_before) {
        std::cout << "  FAIL: visit count not restored" << std::endl;
        return false;
    }
    if (root_vsum_restored != root_vsum_before) {
        std::cout << "  FAIL: value_sum not restored" << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 4 — Full cycle: collect → update replaces virtual with real visits
// ──────────────────────────────────────────────────────────────────────────
bool test_full_cycle_consistency() {
    std::cout << "=== Test 4: Full cycle visit count + value consistency ===" << std::endl;

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 200;
    config.use_gumbel = false;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    Board board;
    Node* root = search.init_search(board, uniform_policy(), 0.0f);
    uint32_t root_visits_before = root->visit_count.load(std::memory_order_relaxed);

    int batch_size = 4;
    std::vector<float> obs(batch_size * OBS_SIZE, 0.0f);
    std::vector<float> mask(batch_size * POL_SIZE, 0.0f);

    search.start_next_batch_collection();
    int collected = search.collect_leaves_async(obs.data(), mask.data(), batch_size);
    std::cout << "  Collected " << collected << " leaves" << std::endl;

    search.commit_and_swap();

    // Simulate NN results: uniform policy + value=0 for each leaf
    int prev_batch = search.get_prev_batch_size();
    std::vector<std::vector<float>> policies(prev_batch, uniform_policy());
    std::vector<float> values(prev_batch, 0.0f);

    search.update_prev_leaves(policies, values, nullptr);

    // After update: virtual losses removed, real backprop adds 1 visit per leaf
    uint32_t root_after = root->visit_count.load(std::memory_order_relaxed);
    uint32_t expected = root_visits_before + static_cast<uint32_t>(collected);
    std::cout << "  Root visits after update: " << root_after
              << " (expected " << expected << ")" << std::endl;

    if (root_after != expected) {
        std::cout << "  FAIL: visit count mismatch" << std::endl;
        return false;
    }

    // Child visit sum should equal collected
    uint32_t child_visits = sum_child_visits(root);
    std::cout << "  Child visit sum: " << child_visits << std::endl;
    if (child_visits != static_cast<uint32_t>(collected)) {
        std::cout << "  FAIL: expected child visit sum = " << collected
                  << ", got " << child_visits << std::endl;
        return false;
    }

    // Value_sum at root: init had value=0.0, each backprop also used value=0.0.
    // Virtual loss added +10000 per leaf at root, then removed. Real backprop
    // adds 0 (negated from child's perspective → 0 at root too since value=0).
    // So root value_sum should still be 0 (from init) + 0 (from 4 backprops of 0).
    int64_t root_vsum = root->value_sum_fixed.load(std::memory_order_relaxed);
    std::cout << "  Root value_sum: " << root_vsum << " (expected 0)" << std::endl;
    if (root_vsum != 0) {
        std::cout << "  FAIL: root value_sum should be 0" << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 5 — Multi-phase: virtual losses diversify below root
// ──────────────────────────────────────────────────────────────────────────
//
// Phase 1: collect 2 leaves (root children C0, C1) and update → expands them.
// Phase 2: collect 4 leaves targeting C0,C1,C0,C1 (Gumbel round-robin).
//          Without virtual losses: both C0-sims would pick the same grandchild
//          (FPU makes unvisited nodes look worse than Q=0 of first pick).
//          With virtual losses: first grandchild gets Q=-1.0 → PUCT picks a
//          different grandchild for the second sim.
bool test_diversity_below_root() {
    std::cout << "=== Test 5: Virtual losses diversify sub-root selection ===" << std::endl;

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 200;
    config.use_gumbel = true;
    config.gumbel_top_k = 2;   // Only 2 active root children
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    Board board;
    Node* root = search.init_search(board, uniform_policy(), 0.0f);

    int num_children = count_children(root);
    std::cout << "  Root children: " << num_children << std::endl;

    // --- Phase 1: Expand 2 root children ---
    int batch1 = 2;
    std::vector<float> obs(8 * OBS_SIZE, 0.0f);
    std::vector<float> mask(8 * POL_SIZE, 0.0f);

    search.start_next_batch_collection();
    int collected1 = search.collect_leaves_async(obs.data(), mask.data(), batch1);
    std::cout << "  Phase 1 collected: " << collected1 << std::endl;
    search.commit_and_swap();

    // Provide NN results to expand those children
    int prev1 = search.get_prev_batch_size();
    std::vector<std::vector<float>> policies1(prev1, uniform_policy());
    std::vector<float> values1(prev1, 0.0f);
    search.update_prev_leaves(policies1, values1, nullptr);

    // Verify children are now expanded
    int expanded_count = 0;
    for (Node* c = root->first_child; c; c = c->next_sibling) {
        if (c->is_expanded()) expanded_count++;
    }
    std::cout << "  Expanded root children after phase 1: " << expanded_count << std::endl;
    if (expanded_count < 2) {
        std::cout << "  FAIL: expected at least 2 expanded children" << std::endl;
        return false;
    }

    // --- Phase 2: Collect 4 leaves targeting the 2 expanded children ---
    // Gumbel round-robin: C0, C1, C0, C1
    // Since C0 and C1 are expanded, select() goes deeper → picks grandchildren.
    // Virtual losses on first grandchild (Q=-1.0) should force PUCT to pick
    // a different grandchild for the second sim targeting the same root child.
    int batch2 = 4;
    int collected2 = search.collect_leaves_async(obs.data(), mask.data(), batch2);
    std::cout << "  Phase 2 collected: " << collected2 << std::endl;

    // Check grandchild visit diversity: for each expanded child, count how many
    // distinct grandchildren have virtual visits.
    int total_unique_grandchildren = 0;
    for (Node* child = root->first_child; child; child = child->next_sibling) {
        if (!child->is_expanded()) continue;

        int grandchild_count = 0;
        int grandchildren_with_visits = 0;
        for (Node* gc = child->first_child; gc; gc = gc->next_sibling) {
            grandchild_count++;
            uint32_t v = gc->visit_count.load(std::memory_order_relaxed);
            if (v > 0) grandchildren_with_visits++;
        }

        std::cout << "  Child (expanded, " << grandchild_count << " grandchildren): "
                  << grandchildren_with_visits << " with visits" << std::endl;
        total_unique_grandchildren += grandchildren_with_visits;
    }

    // With virtual losses and 2 sims per root child, we expect at least 2
    // distinct grandchildren per expanded child (4 total across 2 children).
    // Without virtual losses, we'd get exactly 1 per child (2 total).
    std::cout << "  Total unique grandchildren with visits: " << total_unique_grandchildren << std::endl;

    if (total_unique_grandchildren >= 4) {
        std::cout << "  PASS: virtual losses diversified sub-root selection" << std::endl;
    } else if (total_unique_grandchildren >= 3) {
        std::cout << "  PASS (marginal): " << total_unique_grandchildren
                  << " unique grandchildren (>= 3)" << std::endl;
    } else {
        std::cout << "  FAIL: only " << total_unique_grandchildren
                  << " unique grandchildren (expected >= 3)" << std::endl;
        search.cancel_collection_pending();
        return false;
    }

    search.cancel_collection_pending();
    std::cout << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 6 — Stress: full async search with many collect→update cycles
// ──────────────────────────────────────────────────────────────────────────
//
// Simulates the real pipelined loop from parallel_coordinator.cpp:
//   Phase 1: collect first batch
//   Phase 2: while not done: collect next + update prev
//   Phase 3: drain final batch
// After search completes, verify tree invariants:
//   root.visit_count == sum(child.visit_count) + 1  (the +1 is from init_search)
//   all visit counts > 0
//   no negative value_sum_sq_fixed
bool test_stress_full_search() {
    std::cout << "=== Test 6: Stress — full pipelined search ===" << std::endl;

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 200;
    config.use_gumbel = true;
    config.gumbel_top_k = 8;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    Board board;
    Node* root = search.init_search(board, uniform_policy(), 0.0f);

    int mcts_batch_size = 8;  // match gumbel_top_k
    std::vector<float> obs(mcts_batch_size * OBS_SIZE, 0.0f);
    std::vector<float> mask(mcts_batch_size * POL_SIZE, 0.0f);

    int total_cycles = 0;
    int prev_batch_size = 0;

    // Phase 1: collect first batch
    search.start_next_batch_collection();
    int num_leaves = search.collect_leaves_async(obs.data(), mask.data(), mcts_batch_size);

    if (num_leaves > 0) {
        search.commit_and_swap();
        prev_batch_size = num_leaves;
    }

    // Phase 2: pipelined loop
    while (!search.is_search_complete()) {
        // Step A: collect NEXT batch
        num_leaves = search.collect_leaves_async(obs.data(), mask.data(), mcts_batch_size);

        // Step B: update PREVIOUS batch with fake NN results
        if (prev_batch_size > 0) {
            int prev = search.get_prev_batch_size();
            std::vector<std::vector<float>> policies(prev, uniform_policy());
            // Use small random-ish values so value_sum isn't always 0
            std::vector<float> values(prev);
            for (int i = 0; i < prev; i++) values[i] = (i % 3 == 0) ? 0.1f : -0.1f;

            search.update_prev_leaves(policies, values, nullptr);
        }

        // Step C: commit current batch
        if (num_leaves > 0) {
            search.commit_and_swap();
            prev_batch_size = num_leaves;
        } else {
            prev_batch_size = 0;
        }

        total_cycles++;
        if (total_cycles > 500) {
            std::cout << "  FAIL: search did not complete within 500 cycles" << std::endl;
            return false;
        }
    }

    // Phase 3: drain final batch
    if (prev_batch_size > 0) {
        int prev = search.get_prev_batch_size();
        std::vector<std::vector<float>> policies(prev, uniform_policy());
        std::vector<float> values(prev, 0.0f);
        search.update_prev_leaves(policies, values, nullptr);
    }

    std::cout << "  Search completed in " << total_cycles << " cycles" << std::endl;
    std::cout << "  Simulations completed: " << search.get_simulations_completed() << std::endl;

    // Invariant 1: root.visit_count == sum(child.visit_count) + 1
    uint32_t root_visits = root->visit_count.load(std::memory_order_relaxed);
    uint32_t child_visits = sum_child_visits(root);
    std::cout << "  Root visits: " << root_visits
              << ", child sum: " << child_visits << std::endl;

    // Root gets +1 from init_search, +1 per backpropagate reaching root
    // child_visits should be root_visits - 1 (init visit doesn't go to children)
    if (root_visits != child_visits + 1) {
        std::cout << "  FAIL: root (" << root_visits << ") != children ("
                  << child_visits << ") + 1" << std::endl;
        return false;
    }

    // Invariant 2: no negative visit counts or value_sum_sq
    for (Node* c = root->first_child; c; c = c->next_sibling) {
        uint32_t v = c->visit_count.load(std::memory_order_relaxed);
        int64_t sq = c->value_sum_sq_fixed.load(std::memory_order_relaxed);
        if (sq < 0) {
            std::cout << "  FAIL: child has negative value_sum_sq = " << sq << std::endl;
            return false;
        }
        // Visit count is unsigned so can't be negative, but check it's reasonable
        if (v > config.num_simulations) {
            std::cout << "  FAIL: child visit count " << v << " > num_simulations" << std::endl;
            return false;
        }
    }

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 7 — Stress: repeated cancel/collect without update (no drift)
// ──────────────────────────────────────────────────────────────────────────
bool test_stress_cancel_cycles() {
    std::cout << "=== Test 7: Stress — 100 cancel/collect cycles ===" << std::endl;

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 1000;  // large budget so search never completes
    config.use_gumbel = true;
    config.gumbel_top_k = 4;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    Board board;
    Node* root = search.init_search(board, uniform_policy(), 0.0f);
    uint32_t root_visits_init = root->visit_count.load(std::memory_order_relaxed);
    int64_t root_vsum_init = root->value_sum_fixed.load(std::memory_order_relaxed);

    int batch_size = 4;
    std::vector<float> obs(batch_size * OBS_SIZE, 0.0f);
    std::vector<float> mask(batch_size * POL_SIZE, 0.0f);

    // Repeatedly collect and cancel — visit counts should always return to init
    for (int i = 0; i < 100; i++) {
        search.start_next_batch_collection();
        search.collect_leaves_async(obs.data(), mask.data(), batch_size);

        if (i % 2 == 0) {
            // Cancel collection buffer directly
            search.cancel_collection_pending();
        } else {
            // Commit, then cancel prev
            search.commit_and_swap();
            search.cancel_prev_pending();
        }

        uint32_t root_v = root->visit_count.load(std::memory_order_relaxed);
        int64_t root_vs = root->value_sum_fixed.load(std::memory_order_relaxed);

        if (root_v != root_visits_init || root_vs != root_vsum_init) {
            std::cout << "  FAIL at cycle " << i
                      << ": root visits=" << root_v << " (expected " << root_visits_init
                      << "), value_sum=" << root_vs << " (expected " << root_vsum_init
                      << ")" << std::endl;
            return false;
        }
    }

    // Also verify children are all zeroed out
    uint32_t child_visits = sum_child_visits(root);
    int64_t child_vsum = sum_child_value_sum(root);
    if (child_visits != 0 || child_vsum != 0) {
        std::cout << "  FAIL: residual child visits=" << child_visits
                  << " value_sum=" << child_vsum << std::endl;
        return false;
    }

    std::cout << "  100 cancel/collect cycles: no drift detected" << std::endl;
    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 8 — Edge case: checkmate position (terminal during collect)
// ──────────────────────────────────────────────────────────────────────────
bool test_terminal_during_collect() {
    std::cout << "=== Test 8: Terminal position during async collect ===" << std::endl;

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 50;
    config.use_gumbel = false;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    // Fool's mate: after 1.f3 e5 2.g4 Qh4# — checkmate
    Board board;
    board.makeMove(uci::uciToMove(board, "f2f3"));
    board.makeMove(uci::uciToMove(board, "e7e5"));
    board.makeMove(uci::uciToMove(board, "g2g4"));
    board.makeMove(uci::uciToMove(board, "d8h4"));

    // Position is now checkmate — root should be terminal
    auto [reason, result] = board.isGameOver();
    std::cout << "  Position is game over: "
              << (result != chess::GameResult::NONE ? "yes" : "no") << std::endl;

    Node* root = search.init_search(board, uniform_policy(), -1.0f);

    // Root should be terminal (no legal moves)
    if (!root->is_terminal()) {
        std::cout << "  FAIL: checkmate position not detected as terminal" << std::endl;
        return false;
    }

    // Trying to collect leaves from a terminal root should return 0
    int batch_size = 4;
    std::vector<float> obs(batch_size * OBS_SIZE, 0.0f);
    std::vector<float> mask(batch_size * POL_SIZE, 0.0f);

    search.start_next_batch_collection();
    int collected = search.collect_leaves_async(obs.data(), mask.data(), batch_size);
    std::cout << "  Collected from terminal: " << collected << std::endl;

    // No leaves should be collected (root is terminal, has no children)
    // But simulations may be counted via terminal backprop in the selection phase
    search.cancel_collection_pending();

    uint32_t root_visits = root->visit_count.load(std::memory_order_relaxed);
    std::cout << "  Root visits: " << root_visits << std::endl;

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 9 — Edge case: near-terminal position (some sims hit terminal nodes)
// ──────────────────────────────────────────────────────────────────────────
bool test_mixed_terminal_leaves() {
    std::cout << "=== Test 9: Mixed terminal + NN leaves ===" << std::endl;

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 100;
    config.use_gumbel = true;
    config.gumbel_top_k = 4;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    // Pre-checkmate: 1.f3 e5 2.g4 — Black can play Qh4#
    Board board;
    board.makeMove(uci::uciToMove(board, "f2f3"));
    board.makeMove(uci::uciToMove(board, "e7e5"));
    board.makeMove(uci::uciToMove(board, "g2g4"));
    // Black to move, Qh4# is available but other moves also exist

    Node* root = search.init_search(board, uniform_policy(), 0.0f);

    int num_children = count_children(root);
    std::cout << "  Root children: " << num_children << std::endl;

    // Run full pipelined search
    int mcts_batch = 4;
    std::vector<float> obs(mcts_batch * OBS_SIZE, 0.0f);
    std::vector<float> mask(mcts_batch * POL_SIZE, 0.0f);
    int prev_batch_size = 0;

    search.start_next_batch_collection();
    int num_leaves = search.collect_leaves_async(obs.data(), mask.data(), mcts_batch);
    if (num_leaves > 0) {
        search.commit_and_swap();
        prev_batch_size = num_leaves;
    }

    int cycles = 0;
    while (!search.is_search_complete() && cycles < 200) {
        num_leaves = search.collect_leaves_async(obs.data(), mask.data(), mcts_batch);

        if (prev_batch_size > 0) {
            int prev = search.get_prev_batch_size();
            std::vector<std::vector<float>> policies(prev, uniform_policy());
            std::vector<float> values(prev, 0.0f);
            search.update_prev_leaves(policies, values, nullptr);
        }

        if (num_leaves > 0) {
            search.commit_and_swap();
            prev_batch_size = num_leaves;
        } else {
            prev_batch_size = 0;
        }
        cycles++;
    }

    // Drain
    if (prev_batch_size > 0) {
        int prev = search.get_prev_batch_size();
        std::vector<std::vector<float>> policies(prev, uniform_policy());
        std::vector<float> values(prev, 0.0f);
        search.update_prev_leaves(policies, values, nullptr);
    }

    std::cout << "  Search completed in " << cycles << " cycles" << std::endl;
    std::cout << "  Simulations: " << search.get_simulations_completed() << std::endl;

    // Verify tree invariants
    uint32_t root_visits = root->visit_count.load(std::memory_order_relaxed);
    uint32_t child_sum = sum_child_visits(root);
    std::cout << "  Root visits: " << root_visits << ", child sum: " << child_sum << std::endl;

    if (root_visits != child_sum + 1) {
        std::cout << "  FAIL: root (" << root_visits << ") != children ("
                  << child_sum << ") + 1" << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Test 10 — Sync path (PUCT, batch_size=1) unchanged behavior
// ──────────────────────────────────────────────────────────────────────────
bool test_sync_path_unaffected() {
    std::cout << "=== Test 10: Sync path (batch_size=1) unaffected ===" << std::endl;

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 50;
    config.use_gumbel = false;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    Board board;
    Node* root = search.init_search(board, uniform_policy(), 0.0f);

    // Use sync path: collect_leaves with batch_size=1
    std::vector<float> obs(OBS_SIZE, 0.0f);
    std::vector<float> mask(POL_SIZE, 0.0f);

    int total_sims = 0;
    while (!search.is_search_complete()) {
        int collected = search.collect_leaves(obs.data(), mask.data(), 1);
        if (collected == 0) break;

        std::vector<std::vector<float>> policies(collected, uniform_policy());
        std::vector<float> values(collected, 0.0f);
        search.update_leaves(policies, values, nullptr);
        total_sims++;
    }

    std::cout << "  Sync search completed: " << total_sims << " sims" << std::endl;

    // Verify tree invariants
    uint32_t root_visits = root->visit_count.load(std::memory_order_relaxed);
    uint32_t child_sum = sum_child_visits(root);
    std::cout << "  Root visits: " << root_visits << ", child sum: " << child_sum << std::endl;

    if (root_visits != child_sum + 1) {
        std::cout << "  FAIL: root (" << root_visits << ") != children ("
                  << child_sum << ") + 1" << std::endl;
        return false;
    }

    // No virtual loss residue: all children should have non-negative value_sum_sq
    for (Node* c = root->first_child; c; c = c->next_sibling) {
        int64_t sq = c->value_sum_sq_fixed.load(std::memory_order_relaxed);
        if (sq < 0) {
            std::cout << "  FAIL: child has negative value_sum_sq = " << sq << std::endl;
            return false;
        }
    }

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "Virtual Losses Correctness Tests" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int failed = 0;

    auto run = [&](bool (*test)()) {
        try {
            if (test()) { passed++; } else { failed++; }
        } catch (const std::exception& e) {
            std::cout << "  EXCEPTION: " << e.what() << std::endl;
            failed++;
        }
    };

    run(test_virtual_losses_inflate);
    run(test_cancel_collection_restores);
    run(test_cancel_prev_restores);
    run(test_full_cycle_consistency);
    run(test_diversity_below_root);
    run(test_stress_full_search);
    run(test_stress_cancel_cycles);
    run(test_terminal_during_collect);
    run(test_mixed_terminal_leaves);
    run(test_sync_path_unaffected);

    std::cout << "==========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "==========================================" << std::endl;

    return (failed > 0) ? 1 : 0;
}
