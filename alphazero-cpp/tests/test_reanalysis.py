#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration tests for MCTS Reanalysis."""

import sys
import os
import tempfile
import random
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'Release')
if not os.path.exists(build_dir):
    build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

import alphazero_cpp

OBS_SIZE = 8 * 8 * 123  # 7872
POLICY_SIZE = 4672


# ============================================================================
# Helpers
# ============================================================================

def make_evaluator():
    """Create a dummy neural network evaluator (uniform policy, even WDL)."""
    def evaluator(obs, masks, batch_size, out_policies, out_values):
        out_policies[:batch_size] = 1.0 / POLICY_SIZE
        out_values[:batch_size, 0] = 0.34  # P(win)
        out_values[:batch_size, 1] = 0.33  # P(draw)
        out_values[:batch_size, 2] = 0.33  # P(loss)
    return evaluator


def make_coordinator(buffer, num_sims=8, gumbel=True, top_k=4):
    """Create a minimal ParallelSelfPlayCoordinator for testing."""
    coord = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=1,
        games_per_worker=1,
        num_simulations=num_sims,
        mcts_batch_size=top_k if gumbel else 1,
        gpu_batch_size=64,
        temperature_moves=2,
        gpu_timeout_ms=50,
        worker_timeout_ms=5000,
        queue_capacity=512,
        use_gumbel=gumbel,
        gumbel_top_k=top_k,
    )
    coord.set_replay_buffer(buffer)
    return coord


# ============================================================================
# Test 1: FEN Storage
# ============================================================================

def test_fen_storage():
    """Enable FEN storage, run self-play, verify FENs are stored."""
    print("=== Test 1: FEN Storage ===")

    buf = alphazero_cpp.ReplayBuffer(1000)
    buf.enable_fen_storage()
    assert buf.fen_storage_enabled(), "FEN storage should be enabled"

    # Run tiny self-play to populate buffer
    coord = make_coordinator(buf)
    stats = coord.generate_games(make_evaluator())
    assert stats['games_completed'] == 1, f"Expected 1 game, got {stats['games_completed']}"
    assert buf.size() > 0, "Buffer should have samples"

    # Verify all samples have FENs
    non_empty = 0
    for i in range(buf.size()):
        fen = buf.get_fen(i)
        if fen:
            non_empty += 1
            # Basic FEN validity check: board + side to move + castling + ep + halfmove + fullmove
            parts = fen.split(' ')
            assert len(parts) >= 2, f"Invalid FEN: {fen}"
            assert parts[1] in ('w', 'b'), f"Invalid side to move in FEN: {fen}"

    assert non_empty == buf.size(), f"All {buf.size()} samples should have FENs, got {non_empty}"
    print(f"  {non_empty} FENs stored in {buf.size()} samples")
    print("PASS\n")


# ============================================================================
# Test 2: FEN Save/Load (RPBF v4)
# ============================================================================

def test_fen_save_load():
    """Save buffer with FENs, reload into new buffer, verify FENs preserved."""
    print("=== Test 2: FEN Save/Load ===")

    buf = alphazero_cpp.ReplayBuffer(1000)
    buf.enable_fen_storage()

    coord = make_coordinator(buf)
    coord.generate_games(make_evaluator())
    original_size = buf.size()
    assert original_size > 0, "Buffer should have samples"

    # Collect original FENs
    original_fens = [buf.get_fen(i) for i in range(original_size)]
    assert all(f for f in original_fens), "All original FENs should be non-empty"

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.rpbf', delete=False) as f:
        tmp_path = f.name

    try:
        assert buf.save(tmp_path), "Save should succeed"

        # Load into new buffer
        buf2 = alphazero_cpp.ReplayBuffer(1000)
        buf2.enable_fen_storage()
        assert buf2.load(tmp_path), "Load should succeed"
        assert buf2.size() == original_size, f"Size mismatch: {buf2.size()} != {original_size}"

        # Verify FENs match
        for i in range(original_size):
            loaded_fen = buf2.get_fen(i)
            assert loaded_fen == original_fens[i], \
                f"FEN mismatch at {i}: '{loaded_fen}' != '{original_fens[i]}'"

        print(f"  Saved and reloaded {original_size} samples with FENs intact")
    finally:
        os.unlink(tmp_path)

    print("PASS\n")


# ============================================================================
# Test 3: FEN Disabled Returns Empty
# ============================================================================

def test_fen_disabled_returns_empty():
    """Buffer without FEN storage: get_fen() returns empty strings."""
    print("=== Test 3: FEN Disabled Returns Empty ===")

    buf = alphazero_cpp.ReplayBuffer(1000)
    assert not buf.fen_storage_enabled(), "FEN storage should be disabled by default"

    coord = make_coordinator(buf)
    coord.generate_games(make_evaluator())
    assert buf.size() > 0, "Buffer should have samples"

    for i in range(buf.size()):
        fen = buf.get_fen(i)
        assert fen == "", f"Expected empty FEN at {i}, got: '{fen}'"

    print(f"  All {buf.size()} samples return empty FEN (correct)")
    print("PASS\n")


# ============================================================================
# Test 4: Backward Compatibility (v3 load into FEN-enabled buffer)
# ============================================================================

def test_backward_compat_v3():
    """Save v3 buffer (no FENs), load into FEN-enabled buffer."""
    print("=== Test 4: Backward Compat v3 ===")

    # Create buffer WITHOUT FEN storage (saves as v3-compatible)
    buf_v3 = alphazero_cpp.ReplayBuffer(1000)
    coord = make_coordinator(buf_v3)
    coord.generate_games(make_evaluator())
    original_size = buf_v3.size()
    assert original_size > 0

    with tempfile.NamedTemporaryFile(suffix='.rpbf', delete=False) as f:
        tmp_path = f.name

    try:
        assert buf_v3.save(tmp_path), "Save should succeed"

        # Load into FEN-enabled buffer
        buf_new = alphazero_cpp.ReplayBuffer(1000)
        buf_new.enable_fen_storage()
        assert buf_new.load(tmp_path), "Load should succeed"
        assert buf_new.size() == original_size

        # FENs should be empty (v3 has no FEN data)
        for i in range(buf_new.size()):
            assert buf_new.get_fen(i) == "", f"v3 load should have empty FEN at {i}"

        print(f"  v3 buffer loaded into FEN-enabled buffer: {original_size} samples, all empty FENs")
    finally:
        os.unlink(tmp_path)

    print("PASS\n")


# ============================================================================
# Test 5: Concurrent Reanalysis
# ============================================================================

def test_reanalyzer_concurrent():
    """Run self-play + reanalysis concurrently, verify policies updated."""
    print("=== Test 5: Concurrent Reanalysis ===")

    buf = alphazero_cpp.ReplayBuffer(2000)
    buf.enable_fen_storage()

    # Phase 1: Generate some games to fill buffer with FEN-tagged positions
    coord1 = make_coordinator(buf, num_sims=8)
    coord1.generate_games(make_evaluator())
    buffer_size_after_selfplay = buf.size()
    assert buffer_size_after_selfplay > 0, "Buffer should have samples"
    print(f"  Phase 1: {buffer_size_after_selfplay} positions from self-play")

    # Record original policies for comparison
    original_policies = []
    for i in range(buffer_size_after_selfplay):
        obs, pol, val, wdl, sv = buf.sample(1)
        # We can't easily get specific policy by index via sample(), so we skip exact comparison
        # Instead, we'll just verify the reanalyzer completes and reports stats

    # Phase 2: Concurrent self-play + reanalysis
    coord2 = make_coordinator(buf, num_sims=8)

    # Select positions to reanalyze (all existing positions)
    num_to_reanalyze = min(buffer_size_after_selfplay, 20)  # Cap for speed
    reanalyze_indices = random.sample(range(buffer_size_after_selfplay), num_to_reanalyze)

    reanalysis_config = alphazero_cpp.ReanalysisConfig()
    reanalysis_config.num_workers = 2
    reanalysis_config.num_simulations = 4  # Very shallow for speed
    reanalysis_config.gpu_batch_size = 64
    reanalysis_config.queue_capacity = 512
    reanalysis_config.worker_timeout_ms = 5000
    reanalysis_config.use_gumbel = True
    reanalysis_config.gumbel_top_k = 4
    reanalysis_config.mcts_batch_size = 1

    reanalyzer = alphazero_cpp.Reanalyzer(buf, reanalysis_config)
    reanalyzer.set_indices(reanalyze_indices)

    # Start concurrent self-play + reanalysis
    coord2.start_generation(make_evaluator())
    coord2.set_secondary_queue(reanalyzer)
    reanalyzer.start()

    # Wait for self-play to finish
    coord2.wait_for_workers()
    selfplay_stats = coord2.get_generation_stats()

    # Wait for reanalysis to finish
    reanalyzer.wait()
    reanalysis_stats = reanalyzer.get_stats()

    # Clean up
    coord2.clear_secondary_queue()
    coord2.shutdown_gpu_thread()

    # Verify self-play completed
    assert selfplay_stats['games_completed'] == 1, \
        f"Expected 1 game, got {selfplay_stats['games_completed']}"

    # Verify reanalysis ran
    completed = reanalysis_stats['positions_completed']
    skipped = reanalysis_stats['positions_skipped']
    total_processed = completed + skipped
    print(f"  Phase 2: self-play={selfplay_stats['games_completed']} games, "
          f"reanalysis={completed} completed, {skipped} skipped")
    print(f"  Reanalysis NN evals: {reanalysis_stats['total_nn_evals']}")

    assert total_processed == num_to_reanalyze, \
        f"Expected {num_to_reanalyze} processed, got {total_processed}"
    assert completed > 0, "At least some positions should complete reanalysis"

    # Check no errors
    error = reanalyzer.get_last_error()
    assert error == "", f"Reanalyzer error: {error}"

    print("PASS\n")


# ============================================================================
# Test 6: generate_games() Backward Compatibility
# ============================================================================

def test_generate_games_backward_compat():
    """Existing generate_games() works without reanalysis."""
    print("=== Test 6: generate_games() Backward Compat ===")

    buf = alphazero_cpp.ReplayBuffer(1000)
    # Intentionally NOT enabling FEN storage

    coord = make_coordinator(buf)
    stats = coord.generate_games(make_evaluator())

    assert stats['games_completed'] == 1, f"Expected 1 game, got {stats['games_completed']}"
    assert buf.size() > 0, f"Buffer should have samples, got {buf.size()}"
    assert stats['gpu_errors'] == 0, f"No GPU errors expected, got {stats['gpu_errors']}"

    print(f"  generate_games() completed: {stats['games_completed']} game, {buf.size()} positions")
    print("PASS\n")


# ============================================================================
# Test 7: ReanalysisConfig Defaults
# ============================================================================

def test_reanalysis_config_defaults():
    """Verify ReanalysisConfig default values."""
    print("=== Test 7: ReanalysisConfig Defaults ===")

    config = alphazero_cpp.ReanalysisConfig()
    assert config.num_workers == 8
    assert config.num_simulations == 50
    assert config.mcts_batch_size == 1
    assert abs(config.c_puct - 1.5) < 1e-6
    assert abs(config.fpu_base - 0.3) < 1e-6
    assert abs(config.risk_beta - 0.0) < 1e-6
    assert config.use_gumbel == False
    assert config.gumbel_top_k == 16
    assert config.gpu_batch_size == 512
    assert config.worker_timeout_ms == 2000
    assert config.queue_capacity == 4096

    # Test readwrite
    config.num_simulations = 100
    assert config.num_simulations == 100
    config.use_gumbel = True
    assert config.use_gumbel == True

    print("  All defaults verified")
    print("PASS\n")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MCTS Reanalysis Integration Tests")
    print("=" * 60)
    print()

    tests = [
        test_fen_storage,
        test_fen_save_load,
        test_fen_disabled_returns_empty,
        test_backward_compat_v3,
        test_reanalyzer_concurrent,
        test_generate_games_backward_compat,
        test_reanalysis_config_defaults,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
