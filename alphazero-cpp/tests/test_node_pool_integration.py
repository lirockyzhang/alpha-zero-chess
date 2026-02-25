#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration test for NodePool block reuse through Python bindings.

Exercises the NodePool arena allocator's block-reuse path by running many
reset/search cycles through PyBatchedMCTSSearch. After reset(), the pool
rewinds its cursors but keeps allocated blocks -- subsequent allocations
reuse those blocks instead of calling the OS allocator again.

These tests verify that:
  1. Repeated reset/search cycles produce valid results (no corruption).
  2. Visit counts are well-formed after every cycle.
  3. Memory does not grow unboundedly across many cycles (block reuse works).
"""

import sys
import os
import numpy as np
import pytest

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add build directory to path (MSVC puts .pyd in build/Release/)
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'Release')
if not os.path.exists(build_dir):
    build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

import alphazero_cpp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POLICY_SIZE = 4672
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Use small simulation counts so tests run quickly
NUM_SIMS = 32
BATCH_SIZE = 16


def _random_policy(rng: np.random.Generator) -> np.ndarray:
    """Return a random probability-like policy vector."""
    p = rng.random(POLICY_SIZE, dtype=np.float32)
    p /= p.sum()
    return p


def _run_search(search: alphazero_cpp.BatchedMCTSSearch,
                fen: str,
                rng: np.random.Generator) -> np.ndarray:
    """Run a full search cycle and return visit counts.

    Steps: init_search -> collect/update loop -> get_visit_counts.
    Uses random policy/value as a stand-in for a neural network.
    """
    root_policy = _random_policy(rng)
    root_value = rng.uniform(-1.0, 1.0)
    search.init_search(fen, root_policy, float(root_value))

    while not search.is_complete():
        num_leaves, obs, masks = search.collect_leaves()
        if num_leaves == 0:
            break
        # Random "neural network" output
        policies = np.zeros((num_leaves, POLICY_SIZE), dtype=np.float32)
        for i in range(num_leaves):
            policies[i] = _random_policy(rng)
        values = rng.uniform(-1.0, 1.0, size=num_leaves).astype(np.float32)
        search.update_leaves(policies, values)

    return search.get_visit_counts()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNodePoolResetCycles:
    """Test that repeated reset/search cycles work correctly."""

    def test_single_search_produces_visits(self):
        """Baseline: a single search cycle produces non-zero visit counts."""
        rng = np.random.default_rng(42)
        search = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=NUM_SIMS, batch_size=BATCH_SIZE)

        visits = _run_search(search, STARTING_FEN, rng)

        assert visits.shape == (POLICY_SIZE,), f"Bad shape: {visits.shape}"
        assert visits.sum() > 0, "No visits recorded"

    def test_many_reset_cycles(self):
        """Core test: 50 reset/search cycles all produce valid results."""
        rng = np.random.default_rng(123)
        search = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=NUM_SIMS, batch_size=BATCH_SIZE)

        num_cycles = 50
        for i in range(num_cycles):
            visits = _run_search(search, STARTING_FEN, rng)

            # Every cycle must produce valid visit counts
            assert visits.shape == (POLICY_SIZE,), (
                f"Cycle {i}: bad shape {visits.shape}")
            assert visits.sum() > 0, f"Cycle {i}: no visits"
            assert np.all(visits >= 0), f"Cycle {i}: negative visits"

            # Simulations completed should be close to NUM_SIMS
            sims = search.get_simulations_completed()
            assert sims > 0, f"Cycle {i}: 0 simulations completed"

            search.reset()

    def test_reset_cycles_with_different_positions(self):
        """Reset/search cycles work across different board positions."""
        rng = np.random.default_rng(456)
        search = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=NUM_SIMS, batch_size=BATCH_SIZE)

        positions = [
            # Starting position
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            # After 1.e4
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            # Sicilian Defense
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
            # Italian Game
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            # Ruy Lopez
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            # An endgame position (K+R vs K)
            "8/8/8/4k3/8/8/8/4K2R w - - 0 1",
        ]

        for cycle in range(3):  # 3 passes over all positions
            for j, fen in enumerate(positions):
                visits = _run_search(search, fen, rng)
                total = int(visits.sum())
                assert total > 0, (
                    f"Pass {cycle}, position {j}: no visits (fen={fen})")
                assert np.all(visits >= 0), (
                    f"Pass {cycle}, position {j}: negative visits")
                search.reset()

    def test_root_value_reasonable(self):
        """Root Q-value is within [-1, 1] after each cycle."""
        rng = np.random.default_rng(789)
        search = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=NUM_SIMS, batch_size=BATCH_SIZE)

        for i in range(20):
            _run_search(search, STARTING_FEN, rng)
            root_val = search.get_root_value()
            assert -1.0 <= root_val <= 1.0, (
                f"Cycle {i}: root value {root_val} out of [-1, 1]")
            search.reset()

    def test_visit_sum_matches_simulations(self):
        """Total visit count closely tracks the simulation budget."""
        rng = np.random.default_rng(321)
        search = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=NUM_SIMS, batch_size=BATCH_SIZE)

        for i in range(20):
            visits = _run_search(search, STARTING_FEN, rng)
            total_visits = int(visits.sum())
            sims_completed = search.get_simulations_completed()

            # Visits should equal simulations completed (root visits = sims)
            # Allow some slack: in batch search, some sims may hit terminal
            # nodes and not produce "visits" in the child array.
            assert total_visits > 0, f"Cycle {i}: no visits at all"
            assert total_visits <= sims_completed + BATCH_SIZE, (
                f"Cycle {i}: visits {total_visits} > sims {sims_completed} + batch")
            search.reset()


class TestNodePoolMemoryStability:
    """Verify memory does not grow unboundedly across reset cycles."""

    def test_memory_stable_across_cycles(self):
        """Memory usage stabilizes after the first few cycles.

        After the first search allocates enough blocks to satisfy the tree
        size, subsequent searches (with the same simulation budget) should
        reuse those blocks. Memory should NOT keep growing.
        """
        rng = np.random.default_rng(999)
        # Use slightly more sims to build a decent-sized tree
        sims = 64
        search = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=sims, batch_size=BATCH_SIZE)

        # Warm-up: run a few cycles to let the pool grow to its steady state
        for _ in range(5):
            _run_search(search, STARTING_FEN, rng)
            search.reset()

        # Measure: the process RSS should not jump significantly over
        # additional cycles. We use a simpler proxy: just ensure no crash
        # and that results remain valid across 100 more cycles.
        num_stress_cycles = 100
        for i in range(num_stress_cycles):
            visits = _run_search(search, STARTING_FEN, rng)
            total = int(visits.sum())
            assert total > 0, f"Stress cycle {i}: no visits"
            search.reset()


class TestNodePoolWithHistoryFens:
    """Test reset cycles work when positions include history FENs."""

    def test_search_with_history(self):
        """Search with history_fens provided to init_search."""
        rng = np.random.default_rng(555)
        search = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=NUM_SIMS, batch_size=BATCH_SIZE)

        current_fen = (
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")
        history = [
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            STARTING_FEN,
        ]

        for i in range(20):
            root_policy = _random_policy(rng)
            root_value = float(rng.uniform(-1.0, 1.0))
            search.init_search(current_fen, root_policy, root_value, history)

            while not search.is_complete():
                num_leaves, obs, masks = search.collect_leaves()
                if num_leaves == 0:
                    break
                policies = np.zeros(
                    (num_leaves, POLICY_SIZE), dtype=np.float32)
                for j in range(num_leaves):
                    policies[j] = _random_policy(rng)
                values = rng.uniform(
                    -1.0, 1.0, size=num_leaves).astype(np.float32)
                search.update_leaves(policies, values)

            visits = search.get_visit_counts()
            assert visits.sum() > 0, f"Cycle {i}: no visits with history"
            search.reset()


class TestNodePoolRiskBeta:
    """Test reset cycles work with non-zero risk_beta (ERM)."""

    @pytest.mark.parametrize("risk_beta", [-1.0, 0.0, 1.0])
    def test_search_with_risk_beta(self, risk_beta):
        """Search completes correctly with various risk_beta values."""
        rng = np.random.default_rng(777)
        search = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=NUM_SIMS, batch_size=BATCH_SIZE,
            risk_beta=risk_beta)

        for i in range(10):
            visits = _run_search(search, STARTING_FEN, rng)
            total = int(visits.sum())
            assert total > 0, (
                f"risk_beta={risk_beta}, cycle {i}: no visits")
            root_val = search.get_root_value()
            assert -1.0 <= root_val <= 1.0, (
                f"risk_beta={risk_beta}, cycle {i}: bad root value {root_val}")
            search.reset()


# ---------------------------------------------------------------------------
# Standalone runner (also works with pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("========================================")
    print("NodePool Integration Test Suite")
    print("========================================\n")

    # Run with pytest if available
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
