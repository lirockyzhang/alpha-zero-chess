"""Benchmark ParallelSelfPlayCoordinator with realistic workloads.

Measures end-to-end throughput of the evaluation queue pipeline:
  - Workers submit MCTS leaves via submit_for_evaluation()
  - GPU thread collects batches via collect_batch() (with spin-poll stall detection)
  - Simulated NN evaluation
  - Results distributed back to workers

Tests various configurations to show how batch fill, throughput, and spin-poll
latency change with worker count, search batch size, and GPU latency.

Usage:
    uv run python alphazero-cpp/tests/benchmark_batch_coordinator_realistic.py
"""

import numpy as np
import sys
import os
import time
import threading
import json

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'Release'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import alphazero_cpp


def make_evaluator(latency_ms=0.5):
    """Create a mock NN evaluator with configurable latency.

    Returns (evaluator_fn, call_count_ref) where call_count_ref[0]
    is incremented on each batch evaluation.
    """
    call_count = [0]

    def evaluator(obs, masks, batch_size, out_policies=None, out_values=None):
        call_count[0] += 1
        if latency_ms > 0:
            time.sleep(latency_ms / 1000.0)

        policies = np.full((batch_size, 4672), 1.0 / 4672, dtype=np.float32)
        # WDL: slight white advantage
        wdl = np.zeros((batch_size, 3), dtype=np.float32)
        wdl[:, 0] = 0.35
        wdl[:, 1] = 0.40
        wdl[:, 2] = 0.25

        if out_policies is not None and out_values is not None:
            np.copyto(np.asarray(out_policies), policies)
            np.copyto(np.asarray(out_values), wdl)
            return None
        return policies, wdl

    return evaluator, call_count


def benchmark_config(name, num_workers, games_per_worker, simulations,
                     mcts_batch_size, gpu_batch_size, gpu_timeout_ms=20,
                     queue_capacity=8192, latency_ms=0.5, use_gumbel=True):
    """Run a single benchmark configuration and return metrics."""
    total_games = num_workers * games_per_worker
    evaluator, call_count = make_evaluator(latency_ms)

    buffer = alphazero_cpp.ReplayBuffer(200000)

    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=num_workers,
        games_per_worker=games_per_worker,
        num_simulations=simulations,
        mcts_batch_size=mcts_batch_size,
        gpu_batch_size=gpu_batch_size,
        gpu_timeout_ms=gpu_timeout_ms,
        queue_capacity=queue_capacity,
        use_gumbel=use_gumbel,
        gumbel_top_k=mcts_batch_size if use_gumbel else 16,
    )
    coordinator.set_replay_buffer(buffer)

    # Collect live stats in background to capture spin_poll_avg_us over time
    live_stats_history = []
    stop_monitoring = threading.Event()

    def monitor():
        while not stop_monitoring.is_set():
            try:
                stats = coordinator.get_live_stats()
                if stats:
                    live_stats_history.append(stats)
            except Exception:
                pass
            time.sleep(0.5)

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()

    start = time.perf_counter()
    result = coordinator.generate_games(evaluator)
    elapsed = time.perf_counter() - start

    stop_monitoring.set()
    mon.join(timeout=2)

    # Extract key metrics
    games_completed = result.get('games_completed', 0)
    total_moves = result.get('total_moves', 0)
    total_batches = result.get('total_batches', 0)
    avg_batch = result.get('avg_batch_size', 0)
    total_leaves = result.get('total_leaves', 0)
    drops = result.get('submission_drops', 0)
    waits = result.get('submission_waits', 0)
    pool_exhaust = result.get('pool_exhaustion_count', 0)

    # Spin-poll metric from live stats
    spin_values = [s.get('spin_poll_avg_us', -1) for s in live_stats_history
                   if 'spin_poll_avg_us' in s and s.get('spin_poll_avg_us', -1) >= 0]
    spin_avg = np.mean(spin_values) if spin_values else -1
    spin_last = spin_values[-1] if spin_values else -1

    # Throughput
    batches_per_sec = total_batches / elapsed if elapsed > 0 else 0
    leaves_per_sec = total_leaves / elapsed if elapsed > 0 else 0
    moves_per_sec = total_moves / elapsed if elapsed > 0 else 0
    drop_rate = drops / max(total_leaves + drops, 1)

    return {
        'name': name,
        'num_workers': num_workers,
        'games_per_worker': games_per_worker,
        'simulations': simulations,
        'mcts_batch_size': mcts_batch_size,
        'gpu_batch_size': gpu_batch_size,
        'latency_ms': latency_ms,
        'use_gumbel': use_gumbel,
        'queue_capacity': queue_capacity,
        # Results
        'elapsed_s': elapsed,
        'games_completed': games_completed,
        'total_games': total_games,
        'total_moves': total_moves,
        'total_batches': total_batches,
        'total_leaves': total_leaves,
        'avg_batch_size': avg_batch,
        'batch_fill_pct': (avg_batch / gpu_batch_size * 100) if gpu_batch_size > 0 else 0,
        'batches_per_sec': batches_per_sec,
        'leaves_per_sec': leaves_per_sec,
        'moves_per_sec': moves_per_sec,
        'submission_drops': drops,
        'drop_rate_pct': drop_rate * 100,
        'submission_waits': waits,
        'pool_exhaustion': pool_exhaust,
        'spin_poll_avg_us': spin_avg,
        'spin_poll_last_us': spin_last,
        'nn_eval_calls': call_count[0],
        'buffer_samples': buffer.size(),
        'cpp_error': result.get('cpp_error', ''),
    }


def print_result(r):
    """Pretty-print a single benchmark result."""
    print(f"\n  {'='*56}")
    print(f"  {r['name']}")
    print(f"  {'='*56}")
    print(f"  Config: {r['num_workers']}w x {r['games_per_worker']}g, "
          f"sims={r['simulations']}, batch={r['mcts_batch_size']}, "
          f"gpu_batch={r['gpu_batch_size']}, latency={r['latency_ms']}ms")
    print(f"  Time: {r['elapsed_s']:.1f}s")
    print(f"  Games: {r['games_completed']}/{r['total_games']}")
    print(f"  Throughput:")
    print(f"    Batches/sec:  {r['batches_per_sec']:>8.1f}")
    print(f"    Leaves/sec:   {r['leaves_per_sec']:>8.0f}")
    print(f"    Moves/sec:    {r['moves_per_sec']:>8.1f}")
    print(f"  Batch fill:     {r['avg_batch_size']:.1f} / {r['gpu_batch_size']} "
          f"({r['batch_fill_pct']:.1f}%)")
    print(f"  Spin-poll:      avg={r['spin_poll_avg_us']:.0f}us, "
          f"last={r['spin_poll_last_us']:.0f}us")
    print(f"  Pipeline:")
    print(f"    Drops:        {r['submission_drops']} ({r['drop_rate_pct']:.2f}%)")
    print(f"    Waits:        {r['submission_waits']}")
    print(f"    Pool exhaust: {r['pool_exhaustion']}")
    print(f"    NN calls:     {r['nn_eval_calls']}")
    if r['cpp_error']:
        print(f"  ERROR: {r['cpp_error']}")


def main():
    print("=" * 70)
    print("Benchmark: ParallelSelfPlayCoordinator Pipeline Throughput")
    print("=" * 70)
    print(f"alphazero_cpp version: {alphazero_cpp.__version__}")
    print()
    print("Measures end-to-end throughput of the evaluation queue pipeline")
    print("including the lock-free spin-poll stall detection in collect_batch().")
    print()

    configs = [
        # Baseline: moderate workers, Gumbel search_batch=16
        dict(name="Baseline: 16w x 4g, sb=16, Gumbel",
             num_workers=16, games_per_worker=4, simulations=100,
             mcts_batch_size=16, gpu_batch_size=512, latency_ms=0.5),

        # Scale workers: 32 workers
        dict(name="Scale workers: 32w x 2g, sb=16, Gumbel",
             num_workers=32, games_per_worker=2, simulations=100,
             mcts_batch_size=16, gpu_batch_size=512, latency_ms=0.5),

        # Scale workers: 48 workers
        dict(name="Scale workers: 48w x 2g, sb=16, Gumbel",
             num_workers=48, games_per_worker=2, simulations=80,
             mcts_batch_size=16, gpu_batch_size=512, latency_ms=0.5),

        # PUCT mode: search_batch=1 (many small submissions)
        dict(name="PUCT mode: 32w x 2g, sb=1",
             num_workers=32, games_per_worker=2, simulations=100,
             mcts_batch_size=1, gpu_batch_size=256, latency_ms=0.3,
             use_gumbel=False),

        # Low latency: fast GPU
        dict(name="Low latency: 32w x 2g, sb=16, 0.1ms GPU",
             num_workers=32, games_per_worker=2, simulations=100,
             mcts_batch_size=16, gpu_batch_size=512, latency_ms=0.1),

        # Zero latency: stress spin-poll
        dict(name="Zero latency: 32w x 2g, sb=16, 0ms GPU",
             num_workers=32, games_per_worker=2, simulations=80,
             mcts_batch_size=16, gpu_batch_size=512, latency_ms=0),

        # Small queue: backpressure stress
        dict(name="Small queue: 32w x 2g, sb=16, q=2048",
             num_workers=32, games_per_worker=2, simulations=80,
             mcts_batch_size=16, gpu_batch_size=256, queue_capacity=2048,
             latency_ms=0.5),

        # Max workers: 64 workers
        dict(name="Max workers: 64w x 1g, sb=16, Gumbel",
             num_workers=64, games_per_worker=1, simulations=60,
             mcts_batch_size=16, gpu_batch_size=512, latency_ms=0.5,
             gpu_timeout_ms=30),
    ]

    results = []
    for config in configs:
        print(f"\nRunning: {config['name']}...")
        r = benchmark_config(**config)
        print_result(r)
        results.append(r)

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Config':<45} {'Batch/s':>8} {'Leaf/s':>8} {'Fill%':>6} {'Spin':>6}")
    print(f"{'-'*45} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    for r in results:
        name = r['name'][:44]
        print(f"{name:<45} {r['batches_per_sec']:>8.1f} {r['leaves_per_sec']:>8.0f} "
              f"{r['batch_fill_pct']:>5.1f}% {r['spin_poll_avg_us']:>5.0f}us")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__),
                               'benchmark_results.json')
    # Convert for JSON serialization (remove non-serializable types)
    json_results = []
    for r in results:
        jr = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
              for k, v in r.items()}
        json_results.append(jr)

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Check for failures
    failures = [r for r in results if r['games_completed'] < r['total_games']]
    if failures:
        print(f"\nWARNING: {len(failures)} configs did not complete all games!")
        for r in failures:
            print(f"  {r['name']}: {r['games_completed']}/{r['total_games']}")
        return 1

    print(f"\nAll {len(results)} configurations completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
