"""Integration test: run self-play games through the full C++ pipeline.

Verifies that virtual losses don't break real self-play in either mode:
  1. Gumbel mode (mcts_batch_size = gumbel_top_k) — uses async pipeline with virtual losses
  2. PUCT mode (mcts_batch_size = 1) — uses sync pipeline (virtual losses not applied)

Uses a dummy evaluator (random policy + zero value) so no GPU or model is needed.
"""

import sys, os, time
import numpy as np

# Add the build directory to path so we can import alphazero_cpp
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'Release')
sys.path.insert(0, build_dir)

import alphazero_cpp


def dummy_evaluator(observations, legal_masks, batch_size, out_policies, out_values):
    """Random policy (masked) + draw WDL, writes directly to C++ output buffers."""
    # observations: (batch_size, 8, 8, 123) NHWC
    # legal_masks: (batch_size, 4672)
    # out_policies: (batch_size, 4672) writable buffer
    # out_values: (batch_size, 3) writable buffer (WDL)
    masks = np.array(legal_masks, dtype=np.float32)

    # Random policy, masked to legal moves, then normalized
    policies = np.random.rand(batch_size, 4672).astype(np.float32)
    policies *= masks
    row_sums = policies.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    policies /= row_sums

    # Write to output buffers (zero-copy)
    out_pol = np.array(out_policies, copy=False)
    out_val = np.array(out_values, copy=False)
    out_pol[:] = policies
    # WDL: pure draw prediction (0, 1, 0)
    out_val[:, 0] = 0.0  # P(win)
    out_val[:, 1] = 1.0  # P(draw)
    out_val[:, 2] = 0.0  # P(loss)


def test_gumbel_selfplay():
    """Test Gumbel mode (async pipeline with virtual losses)."""
    print("=== Integration Test 1: Gumbel self-play ===")

    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=1,
        games_per_worker=1,
        num_simulations=20,        # small for speed
        mcts_batch_size=8,         # async pipeline (virtual losses active)
        gpu_batch_size=32,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        temperature_moves=10,
        gpu_timeout_ms=100,
        worker_timeout_ms=5000,
        queue_capacity=256,

        fpu_base=0.3,
        risk_beta=0.0,
        opponent_risk_min=0.0,
        opponent_risk_max=0.0,
        use_gumbel=True,
        gumbel_top_k=8,
        gumbel_c_visit=50.0,
        gumbel_c_scale=1.0,
    )

    total_games = coordinator.total_games()
    print(f"  Total games to generate: {total_games}")

    t0 = time.time()
    result = coordinator.generate_games(dummy_evaluator)
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.2f}s")

    # Check for C++ errors
    if isinstance(result, dict):
        if 'cpp_error' in result and result['cpp_error']:
            print(f"  FAIL: C++ error: {result['cpp_error']}")
            return False
        games_done = result.get('games_completed', 0)
        print(f"  Games completed: {games_done}")
        if games_done < total_games:
            print(f"  FAIL: expected {total_games} games, got {games_done}")
            return False
    else:
        # Result is a list of game trajectories
        print(f"  Games returned: {len(result)}")
        if len(result) < total_games:
            print(f"  FAIL: expected {total_games} games, got {len(result)}")
            return False

    # Try to get a sample game
    sample = coordinator.get_sample_game()
    if sample.get('has_game', False):
        print(f"  Sample game: {sample['num_moves']} moves, result: {sample['result']}")

    print("  PASS")
    print()
    return True


def test_puct_selfplay():
    """Test PUCT mode (sync pipeline, virtual losses NOT applied)."""
    print("=== Integration Test 2: PUCT self-play ===")

    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=1,
        games_per_worker=1,
        num_simulations=20,
        mcts_batch_size=1,         # sync pipeline (no virtual losses)
        gpu_batch_size=32,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        temperature_moves=10,
        gpu_timeout_ms=100,
        worker_timeout_ms=5000,
        queue_capacity=256,

        fpu_base=0.3,
        risk_beta=0.0,
        opponent_risk_min=0.0,
        opponent_risk_max=0.0,
        use_gumbel=False,
        gumbel_top_k=16,
        gumbel_c_visit=50.0,
        gumbel_c_scale=1.0,
    )

    total_games = coordinator.total_games()
    print(f"  Total games to generate: {total_games}")

    t0 = time.time()
    result = coordinator.generate_games(dummy_evaluator)
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.2f}s")

    if isinstance(result, dict):
        if 'cpp_error' in result and result['cpp_error']:
            print(f"  FAIL: C++ error: {result['cpp_error']}")
            return False
        games_done = result.get('games_completed', 0)
        print(f"  Games completed: {games_done}")
        if games_done < total_games:
            print(f"  FAIL: expected {total_games} games, got {games_done}")
            return False
    else:
        print(f"  Games returned: {len(result)}")
        if len(result) < total_games:
            print(f"  FAIL: expected {total_games} games, got {len(result)}")
            return False

    print("  PASS")
    print()
    return True


def test_gumbel_with_replay_buffer():
    """Test Gumbel mode writing directly to replay buffer."""
    print("=== Integration Test 3: Gumbel with replay buffer ===")

    buffer = alphazero_cpp.ReplayBuffer(10000)

    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=1,
        games_per_worker=1,
        num_simulations=20,
        mcts_batch_size=8,
        gpu_batch_size=32,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        temperature_moves=10,
        gpu_timeout_ms=100,
        worker_timeout_ms=5000,
        queue_capacity=256,

        fpu_base=0.3,
        risk_beta=0.0,
        opponent_risk_min=0.0,
        opponent_risk_max=0.0,
        use_gumbel=True,
        gumbel_top_k=8,
        gumbel_c_visit=50.0,
        gumbel_c_scale=1.0,
    )

    coordinator.set_replay_buffer(buffer)

    t0 = time.time()
    result = coordinator.generate_games(dummy_evaluator)
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.2f}s")

    if isinstance(result, dict) and result.get('cpp_error'):
        print(f"  FAIL: C++ error: {result['cpp_error']}")
        return False

    buf_size = buffer.size()
    print(f"  Replay buffer size: {buf_size}")

    if buf_size == 0:
        print("  FAIL: replay buffer is empty after self-play")
        return False

    # Verify buffer contents are sane
    comp = buffer.get_composition()
    print(f"  Buffer composition: W={comp['wins']}, D={comp['draws']}, L={comp['losses']}")

    print("  PASS")
    print()
    return True


if __name__ == '__main__':
    print("==========================================")
    print("Virtual Losses Integration Tests (Python)")
    print("==========================================")
    print()

    passed = 0
    failed = 0

    for test in [test_gumbel_selfplay, test_puct_selfplay, test_gumbel_with_replay_buffer]:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("==========================================")
    print(f"Results: {passed} passed, {failed} failed")
    print("==========================================")

    sys.exit(1 if failed > 0 else 0)
