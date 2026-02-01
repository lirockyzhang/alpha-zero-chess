"""Benchmark parallel tree traversal performance.

This benchmark simulates the leaf collection phase of batched MCTS, where we need to
collect leaves from multiple games in parallel. The target is <1ms for 256 leaves.

Current implementation: Single-threaded leaf collection
Optimization: OpenMP parallel tree traversal

Expected improvement: 3.7x speedup (from 3.67ms to <1ms for 256 positions)
"""

import numpy as np
import sys
import os
import time

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'Release'))

import alphazero_cpp

def benchmark_batch_encoding(batch_size=256, num_iterations=100):
    """Benchmark batch encoding performance (simulates leaf collection + encoding)."""
    # Use a variety of positions to simulate real game states
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # After e4
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",  # After e4 d5
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",  # After e4 d5 e5
        "rnbqkb1r/ppp1pppp/5n2/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3",  # After e4 d5 e5 Nf6
    ]

    # Create batch buffer
    batch_buffer = np.zeros((batch_size, 8, 8, 119), dtype=np.float32)

    # Warm-up
    for i in range(10):
        fen = fens[i % len(fens)]
        alphazero_cpp.encode_position_to_buffer(fen, batch_buffer[0])

    # Benchmark
    times = []
    for iteration in range(num_iterations):
        start = time.perf_counter()

        # Encode batch (simulates leaf collection + encoding)
        for i in range(batch_size):
            fen = fens[i % len(fens)]
            alphazero_cpp.encode_position_to_buffer(fen, batch_buffer[i])

        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to milliseconds

    return times

def benchmark_single_position_encoding(num_iterations=1000):
    """Benchmark single position encoding (baseline)."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    buffer = np.zeros((8, 8, 119), dtype=np.float32)

    # Warm-up
    for _ in range(10):
        alphazero_cpp.encode_position_to_buffer(fen, buffer)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        alphazero_cpp.encode_position_to_buffer(fen, buffer)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000000)  # Convert to microseconds

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
    print("Benchmark: Parallel Tree Traversal (Batch Encoding)")
    print("=" * 70)
    print()
    print("This benchmark measures the time to encode a batch of positions,")
    print("which simulates the leaf collection + encoding phase of batched MCTS.")
    print()
    print("Target: <1ms for 256 positions")
    print("Current (baseline): ~3.67ms for 256 positions")
    print("Expected with OpenMP: <1ms for 256 positions (3.7x speedup)")
    print()

    # Benchmark single position encoding (baseline)
    print("Benchmarking single position encoding (baseline)...")
    single_times = benchmark_single_position_encoding(num_iterations=1000)
    print()
    print("Single Position Encoding:")
    print_statistics(single_times, unit="μs")
    print()

    # Benchmark batch encoding (current implementation)
    print("Benchmarking batch encoding (256 positions)...")
    batch_times = benchmark_batch_encoding(batch_size=256, num_iterations=100)
    print()
    print("Batch Encoding (256 positions):")
    print_statistics(batch_times, unit="ms")
    print()

    # Calculate throughput
    mean_batch_time = np.mean(batch_times)
    throughput = 256 / (mean_batch_time / 1000)  # positions per second
    print(f"Throughput: {throughput:.0f} positions/second")
    print()

    # Check if target is met
    target_time = 1.0  # ms
    if mean_batch_time < target_time:
        print(f"[PASS] Target met: {mean_batch_time:.3f}ms < {target_time}ms")
    else:
        speedup_needed = mean_batch_time / target_time
        print(f"[INFO] Target not met: {mean_batch_time:.3f}ms > {target_time}ms")
        print(f"[INFO] Speedup needed: {speedup_needed:.2f}x")
    print()

    # Estimate parallel speedup potential
    single_mean = np.mean(single_times) / 1000  # Convert to ms
    theoretical_parallel_time = single_mean  # If perfectly parallel
    theoretical_speedup = mean_batch_time / theoretical_parallel_time
    print(f"Theoretical parallel speedup: {theoretical_speedup:.2f}x")
    print(f"Theoretical parallel time: {theoretical_parallel_time:.3f}ms")
    print()

    # Save results for comparison (convert numpy types to native Python types)
    results = {
        'single_position_mean_us': float(np.mean(single_times)),
        'batch_256_mean_ms': float(mean_batch_time),
        'throughput_pos_per_sec': float(throughput),
        'target_met': bool(mean_batch_time < target_time),
        'speedup_needed': float(mean_batch_time / target_time),
        'theoretical_speedup': float(theoretical_speedup),
    }

    import json
    with open('benchmark_parallel_traversal_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to: benchmark_parallel_traversal_baseline.json")
    print()

    return results

if __name__ == "__main__":
    main()
