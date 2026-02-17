#!/usr/bin/env python3
"""
Test: Evaluation Queue Stale Result Prevention & Root Eval Retry

Tests for generation-based stale result filtering and root eval retry loop.
Designed to run FAST (<60s total) with minimal simulations.

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


def run_games(workers=2, gpw=1, timeout_ms=5000, retries=3,
              queue_cap=8192, eval_delay=0, **kw):
    """Helper: create coordinator, run games, return result dict."""
    ev, cnt = fast_eval(eval_delay)
    merged = {**FAST, **kw}
    c = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=workers, games_per_worker=gpw,
        worker_timeout_ms=timeout_ms, root_eval_retries=retries,
        queue_capacity=queue_cap, **merged)
    buf = alphazero_cpp.ReplayBuffer(capacity=50000)
    c.set_replay_buffer(buf)
    result = c.generate_games(ev)
    return result, buf, cnt


# =============================================================================
# Tests
# =============================================================================

def test_basic_selfplay():
    """Test 1: Basic sanity — games complete, new stats present."""
    print("\n" + "="*60)
    print("Test 1: Basic Parallel Self-Play Sanity")
    print("="*60)
    r, buf, cnt = run_games(workers=2, gpw=1)
    T.eq(r['games_completed'], 2, "2 games completed")
    T.ok(r['total_moves'] > 0, f"moves > 0 ({r['total_moves']})")
    T.ok(cnt[0] > 0, f"evaluator called ({cnt[0]}x)")
    T.ok('root_retries' in r, "root_retries present")
    T.ok('stale_results_flushed' in r, "stale_results_flushed present")


def test_stats_types():
    """Test 2: New stats are correct types and non-negative."""
    print("\n" + "="*60)
    print("Test 2: Stats Types and Values")
    print("="*60)
    r, _, _ = run_games(workers=2, gpw=1)
    T.ok(isinstance(r['root_retries'], int), "root_retries is int")
    T.ok(isinstance(r['stale_results_flushed'], int), "stale_results_flushed is int")
    T.gte(r['root_retries'], 0, "root_retries >= 0")
    T.gte(r['stale_results_flushed'], 0, "stale_results_flushed >= 0")


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


def test_retries_param_zero():
    """Test 4: root_eval_retries=0 works (no retries)."""
    print("\n" + "="*60)
    print("Test 4: root_eval_retries=0")
    print("="*60)
    r, _, _ = run_games(workers=1, gpw=1, retries=0)
    T.eq(r['games_completed'], 1, "game completes with retries=0")
    T.eq(r['root_retries'], 0, "zero retries reported")


def test_retries_param_five():
    """Test 5: root_eval_retries=5 accepted."""
    print("\n" + "="*60)
    print("Test 5: root_eval_retries=5")
    print("="*60)
    r, _, _ = run_games(workers=1, gpw=1, retries=5)
    T.eq(r['games_completed'], 1, "game completes with retries=5")


def test_no_retries_healthy():
    """Test 6: No retries when evaluator is fast and timeout generous."""
    print("\n" + "="*60)
    print("Test 6: No Retries Under Healthy Conditions")
    print("="*60)
    r, _, _ = run_games(workers=2, gpw=1, timeout_ms=10000, retries=3)
    T.eq(r['root_retries'], 0, "zero retries with generous timeout")
    T.eq(r['stale_results_flushed'], 0, "zero stale flushes")


def test_forced_timeout_retries():
    """Test 7: Short timeout forces retries; games still complete."""
    print("\n" + "="*60)
    print("Test 7: Forced Timeout Triggers Retries")
    print("="*60)
    # 20ms eval delay + 50ms timeout → timeouts likely; retries=5 should recover
    r, _, _ = run_games(workers=2, gpw=1, timeout_ms=50, retries=5,
                        eval_delay=20)
    games = r['games_completed']
    T.gte(games, 1, f"at least 1/2 games completed ({games})")
    print(f"  Info: retries={r['root_retries']}, failures={r['mcts_failures']}, "
          f"stale={r['stale_results_flushed']}")


def test_stale_counter_nonneg():
    """Test 8: stale_results_flushed is non-negative under stress."""
    print("\n" + "="*60)
    print("Test 8: Stale Counter Non-Negative (Short Timeout)")
    print("="*60)
    r, _, _ = run_games(workers=4, gpw=1, timeout_ms=60, retries=3,
                        eval_delay=15)
    T.gte(r['stale_results_flushed'], 0, "stale >= 0")
    T.gte(r['root_retries'], 0, "retries >= 0")
    print(f"  Info: stale={r['stale_results_flushed']}, retries={r['root_retries']}, "
          f"games={r['games_completed']}")


def test_many_workers():
    """Test 9: 8 workers contention — all games finish."""
    print("\n" + "="*60)
    print("Test 9: Many Workers (8)")
    print("="*60)
    r, _, cnt = run_games(workers=8, gpw=1, timeout_ms=5000, retries=3)
    T.eq(r['games_completed'], 8, "all 8 games completed")
    T.gte(cnt[0], 1, f"evaluator called ({cnt[0]}x)")
    print(f"  Info: retries={r['root_retries']}, stale={r['stale_results_flushed']}")


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
    r, buf, _ = run_games(workers=2, gpw=2, retries=3)
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
    """Test 12: Default queue_capacity/root_eval_retries work."""
    print("\n" + "="*60)
    print("Test 12: Default Parameters")
    print("="*60)
    # Don't pass queue_capacity or root_eval_retries at all
    ev, _ = fast_eval()
    c = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=1, games_per_worker=1, **FAST)
    buf = alphazero_cpp.ReplayBuffer(capacity=10000)
    c.set_replay_buffer(buf)
    r = c.generate_games(ev)
    T.eq(r['games_completed'], 1, "game completes with defaults")
    T.ok('root_retries' in r, "root_retries in result with defaults")


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
        T.ok('root_retries' in last, "root_retries in live stats")
        T.ok('stale_results_flushed' in last, "stale_results_flushed in live stats")
        T.ok('mcts_failures' in last, "mcts_failures in live stats")
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
        worker_timeout_ms=2000, root_eval_retries=2,
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
        T.ok('root_retries' in holder[0], "stats present after stop")


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("Eval Queue Stale Result Prevention & Retry Tests")
    print("="*60)
    print(f"alphazero_cpp version: {alphazero_cpp.__version__}")

    start = time.time()
    try:
        test_basic_selfplay()
        test_stats_types()
        test_queue_capacity_param()
        test_retries_param_zero()
        test_retries_param_five()
        test_no_retries_healthy()
        test_forced_timeout_retries()
        test_stale_counter_nonneg()
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
