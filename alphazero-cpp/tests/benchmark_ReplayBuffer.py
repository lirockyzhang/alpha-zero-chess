#!/usr/bin/env python3
"""Performance benchmark for C++ training components."""

import sys
import os
import time
import numpy as np
from pathlib import Path
from typing import Tuple
import threading

# Add build directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "build" / "Release"))

try:
    import alphazero_cpp
except ImportError as e:
    print(f"Error importing alphazero_cpp: {e}")
    sys.exit(1)

def benchmark_add_sample(buffer_size: int = 10000, num_samples: int = 10000):
    """Benchmark single sample addition."""
    print(f"\n{'='*80}")
    print(f"Benchmark 1: Add Single Samples ({num_samples:,} samples)")
    print(f"{'='*80}")

    buffer = alphazero_cpp.ReplayBuffer(capacity=buffer_size)

    # Pre-generate data
    obs = np.random.rand(7808).astype(np.float32)
    pol = np.random.rand(4672).astype(np.float32)
    val = 0.5

    # Warmup
    for _ in range(100):
        buffer.add_sample(obs, pol, val)

    # Benchmark
    buffer.clear()
    start = time.perf_counter()

    for i in range(num_samples):
        buffer.add_sample(obs, pol, val)

    elapsed = time.perf_counter() - start

    print(f"  Samples added: {num_samples:,}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Samples/sec: {num_samples/elapsed:,.0f}")
    print(f"  Time per sample: {elapsed/num_samples*1000:.3f}ms")

    return num_samples / elapsed

def benchmark_add_batch(buffer_size: int = 100000, batch_size: int = 256, num_batches: int = 100):
    """Benchmark batch addition."""
    print(f"\n{'='*80}")
    print(f"Benchmark 2: Add Batches (batch_size={batch_size}, num_batches={num_batches})")
    print(f"{'='*80}")

    buffer = alphazero_cpp.ReplayBuffer(capacity=buffer_size)

    # Pre-generate batches
    obs_batch = np.random.rand(batch_size, 7808).astype(np.float32)
    pol_batch = np.random.rand(batch_size, 4672).astype(np.float32)
    val_batch = np.random.rand(batch_size).astype(np.float32)

    # Warmup
    for _ in range(10):
        buffer.add_batch(obs_batch, pol_batch, val_batch)

    # Benchmark
    buffer.clear()
    start = time.perf_counter()

    for _ in range(num_batches):
        buffer.add_batch(obs_batch, pol_batch, val_batch)

    elapsed = time.perf_counter() - start
    total_samples = batch_size * num_batches

    print(f"  Batches added: {num_batches:,}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Samples/sec: {total_samples/elapsed:,.0f}")
    print(f"  Batches/sec: {num_batches/elapsed:,.0f}")
    print(f"  Time per batch: {elapsed/num_batches*1000:.3f}ms")

    return total_samples / elapsed

def benchmark_sample(buffer_size: int = 100000, batch_size: int = 256, num_samples: int = 1000):
    """Benchmark sampling."""
    print(f"\n{'='*80}")
    print(f"Benchmark 3: Sample Batches (batch_size={batch_size}, num_samples={num_samples})")
    print(f"{'='*80}")

    # Fill buffer
    buffer = alphazero_cpp.ReplayBuffer(capacity=buffer_size)

    print("  Filling buffer...")
    for i in range(buffer_size):
        obs = np.random.rand(7808).astype(np.float32)
        pol = np.random.rand(4672).astype(np.float32)
        buffer.add_sample(obs, pol, float(i % 100) / 100)

    print(f"  Buffer filled: {buffer.size():,} samples")

    # Warmup
    for _ in range(10):
        buffer.sample(batch_size)

    # Benchmark
    start = time.perf_counter()

    for _ in range(num_samples):
        obs, pol, val, wdl, sv = buffer.sample(batch_size)

    elapsed = time.perf_counter() - start
    total_sampled = batch_size * num_samples

    print(f"  Batches sampled: {num_samples:,}")
    print(f"  Total samples: {total_sampled:,}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Samples/sec: {total_sampled/elapsed:,.0f}")
    print(f"  Batches/sec: {num_samples/elapsed:,.0f}")
    print(f"  Time per batch: {elapsed/num_samples*1000:.3f}ms")

    # Data throughput
    bytes_per_sample = (7808 + 4672 + 1) * 4  # floats
    throughput_mbps = (total_sampled * bytes_per_sample / elapsed) / (1024**2)
    print(f"  Data throughput: {throughput_mbps:,.1f} MB/s")

    return num_samples / elapsed

def benchmark_threading(buffer_size: int = 100000, num_threads: int = 4, ops_per_thread: int = 1000):
    """Benchmark multi-threaded performance."""
    print(f"\n{'='*80}")
    print(f"Benchmark 5: Multi-Threading ({num_threads} threads, {ops_per_thread} ops each)")
    print(f"{'='*80}")

    buffer = alphazero_cpp.ReplayBuffer(capacity=buffer_size)

    # Pre-fill buffer for sampling
    print("  Filling buffer...")
    for i in range(buffer_size):
        obs = np.random.rand(7808).astype(np.float32)
        pol = np.random.rand(4672).astype(np.float32)
        buffer.add_sample(obs, pol, 0.5)

    def writer_thread(thread_id: int, num_ops: int):
        """Writer thread that adds samples."""
        for i in range(num_ops):
            obs = np.random.rand(7808).astype(np.float32)
            pol = np.random.rand(4672).astype(np.float32)
            buffer.add_sample(obs, pol, float(thread_id))

    def reader_thread(thread_id: int, num_ops: int):
        """Reader thread that samples batches."""
        for i in range(num_ops):
            buffer.sample(32)

    # Test 1: Concurrent writes
    print(f"\n  Test 5a: Concurrent Writes")
    threads = []
    start = time.perf_counter()

    for i in range(num_threads):
        t = threading.Thread(target=writer_thread, args=(i, ops_per_thread))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    write_time = time.perf_counter() - start
    total_writes = num_threads * ops_per_thread

    print(f"    Total writes: {total_writes:,}")
    print(f"    Total time: {write_time:.3f}s")
    print(f"    Writes/sec: {total_writes/write_time:,.0f}")
    print(f"    Throughput per thread: {ops_per_thread/write_time:,.0f} ops/s")

    # Test 2: Concurrent reads
    print(f"\n  Test 5b: Concurrent Reads")
    threads = []
    start = time.perf_counter()

    for i in range(num_threads):
        t = threading.Thread(target=reader_thread, args=(i, ops_per_thread))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    read_time = time.perf_counter() - start
    total_reads = num_threads * ops_per_thread

    print(f"    Total reads: {total_reads:,}")
    print(f"    Total time: {read_time:.3f}s")
    print(f"    Reads/sec: {total_reads/read_time:,.0f}")
    print(f"    Throughput per thread: {ops_per_thread/read_time:,.0f} ops/s")

    # Test 3: Mixed reads/writes
    print(f"\n  Test 5c: Mixed Reads/Writes")
    threads = []
    start = time.perf_counter()

    for i in range(num_threads // 2):
        t1 = threading.Thread(target=writer_thread, args=(i, ops_per_thread))
        t2 = threading.Thread(target=reader_thread, args=(i, ops_per_thread))
        threads.extend([t1, t2])
        t1.start()
        t2.start()

    for t in threads:
        t.join()

    mixed_time = time.perf_counter() - start
    total_ops = num_threads * ops_per_thread

    print(f"    Total operations: {total_ops:,}")
    print(f"    Total time: {mixed_time:.3f}s")
    print(f"    Ops/sec: {total_ops/mixed_time:,.0f}")

def benchmark_memory_usage(buffer_size: int = 100000):
    """Estimate memory usage."""
    print(f"\n{'='*80}")
    print(f"Benchmark 6: Memory Usage (capacity={buffer_size:,})")
    print(f"{'='*80}")

    obs_size = 7808 * 4  # floats
    pol_size = 4672 * 4
    val_size = 4

    sample_size = obs_size + pol_size + val_size
    total_size = buffer_size * sample_size

    print(f"  Size per sample: {sample_size:,} bytes ({sample_size/1024:.1f} KB)")
    print(f"  Total capacity: {total_size:,} bytes ({total_size/(1024**2):.1f} MB, {total_size/(1024**3):.2f} GB)")
    print(f"\n  Breakdown:")
    print(f"    Observations: {buffer_size * obs_size / (1024**2):.1f} MB")
    print(f"    Policies: {buffer_size * pol_size / (1024**2):.1f} MB")
    print(f"    Values: {buffer_size * val_size / 1024:.1f} KB")

def main():
    print("\n" + "="*80)
    print("C++ TRAINING COMPONENTS - PERFORMANCE BENCHMARK")
    print("="*80)

    print("\nConfiguration:")
    print("  Observation size: 7808 floats (8x8x122)")
    print("  Policy size: 4672 floats")
    print("  Value size: 1 float")
    print("  Sample size: 49.8 KB per sample")

    results = {}

    # Run benchmarks
    try:
        results['add_sample'] = benchmark_add_sample(buffer_size=10000, num_samples=10000)
        results['add_batch'] = benchmark_add_batch(buffer_size=100000, batch_size=256, num_batches=100)
        results['sample'] = benchmark_sample(buffer_size=100000, batch_size=256, num_samples=1000)
        benchmark_threading(buffer_size=100000, num_threads=4, ops_per_thread=1000)
        benchmark_memory_usage(buffer_size=100000)

        # Summary
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        print(f"  Add (single): {results['add_sample']:,.0f} samples/sec")
        print(f"  Add (batch):  {results['add_batch']:,.0f} samples/sec")
        print(f"  Sample:       {results['sample']:,.0f} batches/sec")

        print("\n" + "="*80)
        print("BOTTLENECK ANALYSIS")
        print("="*80)

        # Identify bottlenecks
        if results['add_sample'] < 50000:
            print("  [SLOW] Single sample addition: Consider batching")

        if results['sample'] < 500:
            print("  [SLOW] Sampling: Consider pre-allocation or caching")

        print("\n  See TRAINING_OPTIMIZATION_PROPOSALS.md for detailed optimizations")

        print("\n" + "="*80)

        return 0

    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
