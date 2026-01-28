#!/usr/bin/env python3
"""Benchmark MCTS backend performance."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from alphazero import MCTSConfig, MCTSBackend
from alphazero.chess_env import GameState
from alphazero.mcts import create_mcts, get_available_backends
from alphazero.mcts.evaluator import RandomEvaluator


def benchmark_backend(backend: MCTSBackend, num_searches: int, simulations: int) -> dict:
    """Benchmark a single MCTS backend.

    Args:
        backend: Backend to benchmark
        num_searches: Number of MCTS searches to run
        simulations: Simulations per search

    Returns:
        Dictionary of benchmark results
    """
    config = MCTSConfig(num_simulations=simulations)

    try:
        mcts = create_mcts(backend=backend, config=config)
    except ImportError as e:
        return {"error": str(e)}

    evaluator = RandomEvaluator()
    state = GameState()

    # Warmup
    for _ in range(3):
        mcts.search(state, evaluator, move_number=0, add_noise=False)

    # Benchmark
    times = []
    total_nodes = 0

    for i in range(num_searches):
        start = time.perf_counter()
        policy, root, stats = mcts.search(state, evaluator, move_number=0, add_noise=False)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        total_nodes += stats.nodes_created

    times = np.array(times)

    return {
        "backend": backend.value,
        "num_searches": num_searches,
        "simulations": simulations,
        "total_time": times.sum(),
        "mean_time": times.mean(),
        "std_time": times.std(),
        "min_time": times.min(),
        "max_time": times.max(),
        "searches_per_sec": num_searches / times.sum(),
        "sims_per_sec": (num_searches * simulations) / times.sum(),
        "total_nodes": total_nodes,
    }


def print_results(results: list):
    """Print benchmark results as a table."""
    print("\n" + "=" * 80)
    print("MCTS Backend Benchmark Results")
    print("=" * 80)

    # Header
    print(f"\n{'Backend':<12} {'Searches/s':>12} {'Sims/s':>12} {'Mean (ms)':>12} {'Std (ms)':>10}")
    print("-" * 60)

    for r in results:
        if "error" in r:
            print(f"{r.get('backend', 'unknown'):<12} {'ERROR: ' + r['error']}")
        else:
            print(
                f"{r['backend']:<12} "
                f"{r['searches_per_sec']:>12.2f} "
                f"{r['sims_per_sec']:>12.0f} "
                f"{r['mean_time']*1000:>12.2f} "
                f"{r['std_time']*1000:>10.2f}"
            )

    # Speedup comparison
    python_result = next((r for r in results if r.get('backend') == 'python'), None)
    if python_result and "error" not in python_result:
        print("\nSpeedup vs Python:")
        for r in results:
            if "error" not in r and r['backend'] != 'python':
                speedup = r['sims_per_sec'] / python_result['sims_per_sec']
                print(f"  {r['backend']}: {speedup:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MCTS backends")

    parser.add_argument("--searches", type=int, default=100,
                        help="Number of MCTS searches to run")
    parser.add_argument("--simulations", type=int, default=800,
                        help="Simulations per search")
    parser.add_argument("--backends", type=str, nargs="+",
                        default=["python", "cython", "cpp"],
                        help="Backends to benchmark")

    args = parser.parse_args()

    print("MCTS Backend Benchmark")
    print(f"Searches: {args.searches}")
    print(f"Simulations per search: {args.simulations}")

    # Check available backends
    available = get_available_backends()
    print(f"\nAvailable backends: {[b.value for b in available]}")

    # Run benchmarks
    results = []
    for backend_name in args.backends:
        try:
            backend = MCTSBackend(backend_name)
        except ValueError:
            print(f"Unknown backend: {backend_name}")
            continue

        print(f"\nBenchmarking {backend_name}...")
        result = benchmark_backend(backend, args.searches, args.simulations)
        results.append(result)

    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
