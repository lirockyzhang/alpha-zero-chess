"""Benchmark OpenMP parallel batch encoding performance.

This benchmark compares three approaches:
1. Baseline: Python loop calling encode_position_to_buffer() for each position
2. Sequential C++: C++ batch encoding without OpenMP parallelization
3. Parallel C++: C++ batch encoding with OpenMP parallelization

Expected improvement: 1.5-2x speedup (enough to reach <1ms target)
"""

import numpy as np
import sys
import os
import time
import json

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'Release'))

import alphazero_cpp

def benchmark_python_loop(fens, batch_buffer, num_iterations=100):
    """Baseline: Python loop calling encode_position_to_buffer() for each position."""
    times = []

    for iteration in range(num_iterations):
        start = time.perf_counter()

        # Python loop (baseline)
        for i, fen in enumerate(fens):
            alphazero_cpp.encode_position_to_buffer(fen, batch_buffer[i])

        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to milliseconds

    return times

def benchmark_cpp_sequential(fens, batch_buffer, num_iterations=100):
    """C++ batch encoding without OpenMP parallelization."""
    times = []

    for iteration in range(num_iterations):
        start = time.perf_counter()

        # C++ sequential batch encoding
        alphazero_cpp.encode_batch(fens, batch_buffer, use_parallel=False)

        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to milliseconds

    return times

def benchmark_cpp_parallel(fens, batch_buffer, num_iterations=100):
    """C++ batch encoding with OpenMP parallelization."""
    times = []

    for iteration in range(num_iterations):
        start = time.perf_counter()

        # C++ parallel batch encoding
        alphazero_cpp.encode_batch(fens, batch_buffer, use_parallel=True)

        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to milliseconds

    return times

def print_statistics(times, unit="ms"):
    """Print statistics for benchmark times."""
    times = np.array(times)
    print(f"  Mean:   {np.mean(times):.3f} {unit}")
    print(f"  Median: {np.median(times):.3f} {unit}")
    print(f"  Min:    {np.min(times):.3f} {unit}")
    print(f"  Max:    {np.max(times):.3f} {unit}")
    print(f"  Std:    {np.std(times):.3f} {unit}")
    print(f"  P95:    {np.percentile(times, 95):.3f} {unit}")
    print(f"  P99:    {np.percentile(times, 99):.3f} {unit}")

def main():
    print("=" * 70)
    print("Benchmark: OpenMP Parallel Batch Encoding")
    print("=" * 70)
    print()
    print("This benchmark compares three approaches:")
    print("  1. Baseline: Python loop calling encode_position_to_buffer()")
    print("  2. Sequential C++: C++ batch encoding without OpenMP")
    print("  3. Parallel C++: C++ batch encoding with OpenMP")
    print()
    print("Target: <1ms for 256 positions")
    print("Baseline: ~1.456ms for 256 positions")
    print("Expected with OpenMP: <1ms (1.5-2x speedup)")
    print()

    # Prepare test data
    batch_size = 256
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # After e4
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",  # After e4 d5
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",  # After e4 d5 e5
        "rnbqkb1r/ppp1pppp/5n2/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3",  # After e4 d5 e5 Nf6
    ]

    # Repeat FENs to fill batch
    fens = [fens[i % len(fens)] for i in range(batch_size)]

    # Create batch buffer
    batch_buffer = np.zeros((batch_size, 8, 8, 119), dtype=np.float32)

    # Warm-up
    print("Warming up...")
    for _ in range(10):
        alphazero_cpp.encode_batch(fens, batch_buffer, use_parallel=True)
    print()

    # Benchmark 1: Python loop (baseline)
    print("Benchmarking Python loop (baseline)...")
    python_times = benchmark_python_loop(fens, batch_buffer, num_iterations=100)
    print()
    print("Python Loop (Baseline):")
    print_statistics(python_times, unit="ms")
    print()

    # Benchmark 2: C++ sequential
    print("Benchmarking C++ sequential batch encoding...")
    sequential_times = benchmark_cpp_sequential(fens, batch_buffer, num_iterations=100)
    print()
    print("C++ Sequential Batch Encoding:")
    print_statistics(sequential_times, unit="ms")
    print()

    # Benchmark 3: C++ parallel (OpenMP)
    print("Benchmarking C++ parallel batch encoding (OpenMP)...")
    parallel_times = benchmark_cpp_parallel(fens, batch_buffer, num_iterations=100)
    print()
    print("C++ Parallel Batch Encoding (OpenMP):")
    print_statistics(parallel_times, unit="ms")
    print()

    # Calculate speedups
    python_mean = np.mean(python_times)
    sequential_mean = np.mean(sequential_times)
    parallel_mean = np.mean(parallel_times)

    print("=" * 70)
    print("Performance Comparison")
    print("=" * 70)
    print(f"Python loop (baseline):     {python_mean:.3f}ms")
    print(f"C++ sequential:             {sequential_mean:.3f}ms  ({python_mean/sequential_mean:.2f}x speedup)")
    print(f"C++ parallel (OpenMP):      {parallel_mean:.3f}ms  ({python_mean/parallel_mean:.2f}x speedup)")
    print()
    print(f"Sequential vs Parallel:     {sequential_mean/parallel_mean:.2f}x speedup from OpenMP")
    print()

    # Calculate throughput
    throughput_python = batch_size / (python_mean / 1000)
    throughput_sequential = batch_size / (sequential_mean / 1000)
    throughput_parallel = batch_size / (parallel_mean / 1000)

    print(f"Throughput (Python):        {throughput_python:.0f} positions/second")
    print(f"Throughput (Sequential):    {throughput_sequential:.0f} positions/second")
    print(f"Throughput (Parallel):      {throughput_parallel:.0f} positions/second")
    print()

    # Check if target is met
    target_time = 1.0  # ms
    if parallel_mean < target_time:
        print(f"[PASS] Target met: {parallel_mean:.3f}ms < {target_time}ms")
        print()
        print("The OpenMP parallel batch encoding optimization successfully")
        print("achieves the <1ms target for 256 positions!")
    else:
        speedup_needed = parallel_mean / target_time
        print(f"[INFO] Target not met: {parallel_mean:.3f}ms > {target_time}ms")
        print(f"[INFO] Additional speedup needed: {speedup_needed:.2f}x")
    print()

    # Save results
    results = {
        'batch_size': batch_size,
        'python_loop_mean_ms': float(python_mean),
        'cpp_sequential_mean_ms': float(sequential_mean),
        'cpp_parallel_mean_ms': float(parallel_mean),
        'speedup_sequential_vs_python': float(python_mean / sequential_mean),
        'speedup_parallel_vs_python': float(python_mean / parallel_mean),
        'speedup_parallel_vs_sequential': float(sequential_mean / parallel_mean),
        'throughput_python_pos_per_sec': float(throughput_python),
        'throughput_sequential_pos_per_sec': float(throughput_sequential),
        'throughput_parallel_pos_per_sec': float(throughput_parallel),
        'target_met': bool(parallel_mean < target_time),
    }

    with open('benchmark_parallel_traversal_optimized.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to: benchmark_parallel_traversal_optimized.json")
    print()

    # Load baseline results for comparison
    try:
        with open('benchmark_parallel_traversal_baseline.json', 'r') as f:
            baseline = json.load(f)

        print("=" * 70)
        print("Comparison with Baseline")
        print("=" * 70)
        print(f"Baseline (Python loop):     {baseline['batch_256_mean_ms']:.3f}ms")
        print(f"Optimized (C++ parallel):   {parallel_mean:.3f}ms")
        print(f"Overall improvement:        {baseline['batch_256_mean_ms']/parallel_mean:.2f}x speedup")
        print()
    except FileNotFoundError:
        print("Baseline results not found. Run benchmark_parallel_traversal.py first.")
        print()

    return results

if __name__ == "__main__":
    main()
