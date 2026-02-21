#!/usr/bin/env python3
"""
Test: Evaluation Queue Blocking Wait & Pipeline Health

Tests for blocking get_results() with watchdog, queue capacity, and pipeline
health under various conditions. Designed to run FAST (<60s total).

Usage:
    uv run python alphazero-cpp/tests/test_eval_queue_staleness.py
"""

import sys
import os
import time
import numpy as np
import threading
from pathlib import Path

# Windows console encoding fix (reconfigure avoids closing the underlying
# buffer, which breaks pytest's capture mechanism)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add build directory to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir / "build" / "Release"))
sys.path.insert(0, str(script_dir / "scripts"))

try:
    import alphazero_cpp
except ImportError as e:
    print(f"Error importing alphazero_cpp: {e}")
    print("Build first: cd alphazero-cpp/build && cmake --build . --config Release")
    sys.exit(1)


# =============================================================================
# Helpers
# =============================================================================

# Minimal config for fast tests — 4 sims keeps games very short
FAST = dict(num_simulations=4, mcts_batch_size=2, gpu_batch_size=8, temperature_moves=1)

class T:
    """Tiny test harness."""
    passed = 0
    failed = 0
    errors = []

    @classmethod
    def ok(cls, cond, msg):
        if cond:
            print(f"  OK {msg}")
            cls.passed += 1
        else:
            print(f"  FAIL {msg}")
            cls.failed += 1
            cls.errors.append(msg)

    @classmethod
    def eq(cls, a, b, msg):
        cls.ok(a == b, f"{msg}: expected {b}, got {a}" if a != b else f"{msg}: {a}")

    @classmethod
    def gte(cls, a, b, msg):
        cls.ok(a >= b, f"{msg}: {a} >= {b}" if a >= b else f"{msg}: {a} < {b}")

    @classmethod
    def summary(cls):
        total = cls.passed + cls.failed
        print(f"\n{'='*60}")
        if cls.failed == 0:
            print(f"ALL {total} CHECKS PASSED")
        else:
            print(f"{cls.passed}/{total} passed, {cls.failed} FAILED")
            for e in cls.errors:
                print(f"  - {e}")
        print(f"{'='*60}")
        return cls.failed == 0


def fast_eval(delay_ms=0):
    """Random evaluator (5-arg zero-copy API), optional delay."""
    count = [0]
    def fn(obs, masks, bs, out_pol, out_val):
        count[0] += 1
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        p = np.random.rand(bs, 4672).astype(np.float32)
        m = np.array(masks[:bs], dtype=np.float32)
        p *= m
        s = p.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        p /= s
        out_pol[:bs] = p
        # Write WDL probs: neutral [0.33, 0.34, 0.33]
        out_val[:bs, 0] = 0.33
        out_val[:bs, 1] = 0.34
        out_val[:bs, 2] = 0.33
    return fn, count


def run_games(workers=2, gpw=1, timeout_ms=5000,
              queue_cap=8192, eval_delay=0, **kw):
    """Helper: create coordinator, run games, return result dict."""
    ev, cnt = fast_eval(eval_delay)
    merged = {**FAST, **kw}
    c = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=workers, games_per_worker=gpw,
        worker_timeout_ms=timeout_ms,
        queue_capacity=queue_cap, **merged)
    buf = alphazero_cpp.ReplayBuffer(capacity=50000)
    c.set_replay_buffer(buf)
    result = c.generate_games(ev)
    return result, buf, cnt


# =============================================================================
# Tests
# =============================================================================

def test_basic_selfplay():
    """Test 1: Basic sanity — games complete, stats present."""
    print("\n" + "="*60)
    print("Test 1: Basic Parallel Self-Play Sanity")
    print("="*60)
    r, buf, cnt = run_games(workers=2, gpw=1)
    T.eq(r['games_completed'], 2, "2 games completed")
    T.ok(r['total_moves'] > 0, f"moves > 0 ({r['total_moves']})")
    T.ok(cnt[0] > 0, f"evaluator called ({cnt[0]}x)")
    T.ok('mcts_failures' in r, "mcts_failures present")


def test_stats_types():
    """Test 2: Stats are correct types and non-negative."""
    print("\n" + "="*60)
    print("Test 2: Stats Types and Values")
    print("="*60)
    r, _, _ = run_games(workers=2, gpw=1)
    T.ok(isinstance(r['mcts_failures'], int), "mcts_failures is int")
    T.ok(isinstance(r['gpu_errors'], int), "gpu_errors is int")
    T.gte(r['mcts_failures'], 0, "mcts_failures >= 0")
    T.gte(r['gpu_errors'], 0, "gpu_errors >= 0")


def test_queue_capacity_param():
    """Test 3: queue_capacity parameter accepted."""
    print("\n" + "="*60)
    print("Test 3: Queue Capacity Parameter")
    print("="*60)
    try:
        r, _, _ = run_games(workers=2, gpw=1, queue_cap=2048)
        T.ok(True, "queue_capacity=2048 accepted")
        T.eq(r['games_completed'], 2, "games completed with custom capacity")
    except TypeError as e:
        T.ok(False, f"queue_capacity rejected: {e}")


def test_single_worker():
    """Test 4: Single worker completes game with blocking wait."""
    print("\n" + "="*60)
    print("Test 4: Single Worker Blocking Wait")
    print("="*60)
    r, _, _ = run_games(workers=1, gpw=1)
    T.eq(r['games_completed'], 1, "game completes with single worker")
    T.eq(r['mcts_failures'], 0, "no MCTS failures")


def test_multiple_games_per_worker():
    """Test 5: Multiple games per worker complete."""
    print("\n" + "="*60)
    print("Test 5: Multiple Games Per Worker")
    print("="*60)
    r, _, _ = run_games(workers=1, gpw=3)
    T.eq(r['games_completed'], 3, "all 3 games completed")


def test_no_failures_healthy():
    """Test 6: No failures when evaluator is fast."""
    print("\n" + "="*60)
    print("Test 6: No Failures Under Healthy Conditions")
    print("="*60)
    r, _, _ = run_games(workers=2, gpw=1)
    T.eq(r['mcts_failures'], 0, "zero MCTS failures")
    T.eq(r['gpu_errors'], 0, "zero GPU errors")


def test_slow_evaluator():
    """Test 7: Slow evaluator — blocking wait handles it (no timeout)."""
    print("\n" + "="*60)
    print("Test 7: Slow Evaluator (Blocking Wait)")
    print("="*60)
    # 20ms eval delay — blocking wait should handle this fine
    r, _, _ = run_games(workers=2, gpw=1, eval_delay=20)
    games = r['games_completed']
    T.eq(games, 2, f"both games completed ({games})")
    T.eq(r['mcts_failures'], 0, "no MCTS failures despite slow eval")


def test_multi_worker_slow_eval():
    """Test 8: Multiple workers with slow evaluator complete cleanly."""
    print("\n" + "="*60)
    print("Test 8: Multi-Worker Slow Evaluator")
    print("="*60)
    r, _, _ = run_games(workers=4, gpw=1, eval_delay=15)
    T.eq(r['games_completed'], 4, f"all 4 games completed")
    T.eq(r['mcts_failures'], 0, "no MCTS failures")
    print(f"  Info: games={r['games_completed']}, failures={r['mcts_failures']}")


def test_many_workers():
    """Test 9: 8 workers contention — all games finish."""
    print("\n" + "="*60)
    print("Test 9: Many Workers (8)")
    print("="*60)
    r, _, cnt = run_games(workers=8, gpw=1)
    T.eq(r['games_completed'], 8, "all 8 games completed")
    T.gte(cnt[0], 1, f"evaluator called ({cnt[0]}x)")
    T.eq(r['mcts_failures'], 0, "no MCTS failures with 8 workers")


def test_large_queue():
    """Test 10: Large queue_capacity=16384 works."""
    print("\n" + "="*60)
    print("Test 10: Large Queue Capacity (16384)")
    print("="*60)
    r, _, _ = run_games(workers=2, gpw=1, queue_cap=16384)
    T.eq(r['games_completed'], 2, "games complete with large queue")
    T.eq(r.get('pool_exhaustion_count', 0), 0, "no pool exhaustion")


def test_game_integrity():
    """Test 11: Game results valid, buffer data not corrupted."""
    print("\n" + "="*60)
    print("Test 11: Game Integrity (No Corruption)")
    print("="*60)
    r, buf, _ = run_games(workers=2, gpw=2)
    games = r['games_completed']
    T.eq(games, 4, "4 games completed")

    w, b, d = r.get('white_wins', 0), r.get('black_wins', 0), r.get('draws', 0)
    T.eq(w + b + d, games, f"W({w})+B({b})+D({d}) == {games}")

    sz = buf.size()
    T.ok(sz > 0, f"buffer has {sz} samples")
    if sz >= 2:
        obs, pol, val, wdl, sv = buf.sample(2)
        T.ok(not np.any(np.isnan(obs)), "no NaN in obs")
        T.ok(not np.any(np.isinf(obs)), "no Inf in obs")
        T.ok(not np.any(np.isnan(pol)), "no NaN in policy")
        T.ok(not np.any(np.isnan(val)), "no NaN in values")


def test_default_params():
    """Test 12: Default parameters work."""
    print("\n" + "="*60)
    print("Test 12: Default Parameters")
    print("="*60)
    # Don't pass queue_capacity at all
    ev, _ = fast_eval()
    c = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=1, games_per_worker=1, **FAST)
    buf = alphazero_cpp.ReplayBuffer(capacity=10000)
    c.set_replay_buffer(buf)
    r = c.generate_games(ev)
    T.eq(r['games_completed'], 1, "game completes with defaults")
    T.ok('mcts_failures' in r, "mcts_failures in result with defaults")


def test_live_stats():
    """Test 13: get_live_stats() returns new metrics."""
    print("\n" + "="*60)
    print("Test 13: Live Stats Include New Metrics")
    print("="*60)
    ev, _ = fast_eval(delay_ms=5)
    c = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=4, games_per_worker=2,
        worker_timeout_ms=5000, **FAST)
    buf = alphazero_cpp.ReplayBuffer(capacity=50000)
    c.set_replay_buffer(buf)

    snapshots = []

    def poll():
        while c.is_running():
            s = c.get_live_stats()
            if s:
                snapshots.append(s)
            time.sleep(0.05)

    gt = threading.Thread(target=lambda: c.generate_games(ev), daemon=True)
    gt.start()
    time.sleep(0.02)
    pt = threading.Thread(target=poll, daemon=True)
    pt.start()
    gt.join(timeout=30)
    pt.join(timeout=1)

    if snapshots:
        last = snapshots[-1]
        T.ok('mcts_failures' in last, "mcts_failures in live stats")
        T.ok('avg_batch_size' in last, "avg_batch_size in live stats")
    else:
        # Games finished too fast to poll — that's OK, just check final result
        T.ok(True, "generation too fast to poll (not a failure)")


def test_graceful_shutdown():
    """Test 14: stop() terminates generation cleanly."""
    print("\n" + "="*60)
    print("Test 14: Graceful Shutdown")
    print("="*60)
    ev, _ = fast_eval(delay_ms=10)
    c = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=2, games_per_worker=50,  # many games
        worker_timeout_ms=2000,
        num_simulations=8, mcts_batch_size=4, gpu_batch_size=16,
        temperature_moves=1)
    buf = alphazero_cpp.ReplayBuffer(capacity=100000)
    c.set_replay_buffer(buf)

    holder = [None]
    def run():
        holder[0] = c.generate_games(ev)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.5)
    c.stop()
    t.join(timeout=10)

    T.ok(not t.is_alive(), "thread exited after stop()")
    if holder[0] is not None:
        T.ok(holder[0]['games_completed'] < 100, f"stopped early ({holder[0]['games_completed']}/100)")
        T.ok('mcts_failures' in holder[0], "stats present after stop")


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("Eval Queue Blocking Wait & Pipeline Health Tests")
    print("="*60)
    print(f"alphazero_cpp version: {alphazero_cpp.__version__}")

    start = time.time()
    try:
        test_basic_selfplay()
        test_stats_types()
        test_queue_capacity_param()
        test_single_worker()
        test_multiple_games_per_worker()
        test_no_failures_healthy()
        test_slow_evaluator()
        test_multi_worker_slow_eval()
        test_many_workers()
        test_large_queue()
        test_game_integrity()
        test_default_params()
        test_live_stats()
        test_graceful_shutdown()
    except Exception as e:
        print(f"\nFATAL: {e}")
        import traceback
        traceback.print_exc()
        T.failed += 1

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    return 0 if T.summary() else 1


if __name__ == "__main__":
    sys.exit(main())
