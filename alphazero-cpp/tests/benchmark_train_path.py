#!/usr/bin/env python3
"""
Benchmark the ACTUAL training path used by scripts/train.py

This tests the real CppBatchedMCTS + NetworkEvaluator combination
to verify batch evaluation is working correctly.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alphazero import MCTSConfig, MCTSBackend
from alphazero.mcts import create_mcts
from alphazero.mcts.evaluator import NetworkEvaluator
from alphazero.neural import AlphaZeroNetwork
from alphazero.chess_env import GameState


def benchmark_training_path(device: str, num_searches: int = 10, simulations: int = 800):
    """Benchmark the actual training path."""
    print(f"\n{'='*60}")
    print("Benchmark: ACTUAL Training Path (scripts/train.py)")
    print(f"  Device: {device}")
    print(f"  Simulations: {simulations}")
    print(f"  Backend: CppBatchedMCTS")
    print(f"{'='*60}")

    # Create network (same as scripts/train.py)
    network = AlphaZeroNetwork(
        num_filters=64,
        num_blocks=5,
        input_channels=119,
        num_actions=4672
    )
    network = network.to(device)
    network.eval()

    print(f"Network created: {sum(p.numel() for p in network.parameters()):,} parameters")

    # Create evaluator (same as actor.py)
    evaluator = NetworkEvaluator(network, device, use_amp=True)

    # Check if evaluate_batch exists
    has_batch = hasattr(evaluator, 'evaluate_batch')
    print(f"Evaluator has evaluate_batch: {has_batch}")

    # Create MCTS with C++ backend (same as actor.py)
    config = MCTSConfig(
        num_simulations=simulations,
        backend=MCTSBackend.CPP,
        batch_size=64
    )

    try:
        mcts = create_mcts(backend=MCTSBackend.CPP, config=config, use_batched=True)
        print(f"MCTS created: {type(mcts).__name__}")
    except Exception as e:
        print(f"ERROR creating MCTS: {e}")
        return None

    # Create game state
    state = GameState()

    # Warmup
    print("\nWarming up...")
    for _ in range(2):
        policy, root, stats = mcts.search(state, evaluator, move_number=0, add_noise=True)

    # Benchmark
    print(f"\nRunning {num_searches} searches...")
    times = []
    total_leaves = 0

    for i in range(num_searches):
        state = GameState()  # Fresh state each time

        start = time.perf_counter()
        policy, root, stats = mcts.search(state, evaluator, move_number=0, add_noise=True)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        total_leaves += stats.nodes_created

        if (i + 1) % 5 == 0:
            print(f"  Search {i+1}/{num_searches}: {elapsed*1000:.1f}ms, leaves={stats.nodes_created}")

    times = np.array(times)

    # Results
    print(f"\n{'─'*50}")
    print(f"Results ({num_searches} searches):")
    print(f"{'─'*50}")
    print(f"  Total time:        {times.sum()*1000:.1f} ms")
    print(f"  Mean time:         {times.mean()*1000:.1f} ms")
    print(f"  Std time:          {times.std()*1000:.1f} ms")
    print(f"  Min time:          {times.min()*1000:.1f} ms")
    print(f"  Max time:          {times.max()*1000:.1f} ms")
    print(f"{'─'*50}")
    print(f"  Searches/sec:      {num_searches / times.sum():.2f}")
    print(f"  Simulations/sec:   {(num_searches * simulations) / times.sum():.0f}")
    print(f"  Avg leaves/search: {total_leaves / num_searches:.1f}")
    print(f"{'─'*50}")

    # Get MCTS stats
    if hasattr(mcts, 'get_stats'):
        mcts_stats = mcts.get_stats()
        print(f"\nMCTS Statistics:")
        for k, v in mcts_stats.items():
            print(f"  {k}: {v}")

    return {
        'searches_per_sec': num_searches / times.sum(),
        'sims_per_sec': (num_searches * simulations) / times.sum(),
        'mean_time_ms': times.mean() * 1000,
    }


def compare_single_vs_batch():
    """Compare single evaluation vs batch evaluation performance."""
    print(f"\n{'='*60}")
    print("Comparison: Single vs Batch Evaluation in CppBatchedMCTS")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create network and evaluator
    network = AlphaZeroNetwork(num_filters=64, num_blocks=5, input_channels=119, num_actions=4672)
    network = network.to(device)
    network.eval()

    evaluator = NetworkEvaluator(network, device, use_amp=True)

    # Test single evaluation
    obs = np.random.randn(119, 8, 8).astype(np.float32)
    mask = np.ones(4672, dtype=np.float32)

    print("\nSingle evaluation (evaluator.evaluate):")
    start = time.perf_counter()
    for _ in range(100):
        policy, value = evaluator.evaluate(obs, mask)
    if device == "cuda":
        torch.cuda.synchronize()
    single_time = time.perf_counter() - start
    print(f"  100 evaluations: {single_time*1000:.1f} ms")
    print(f"  Per evaluation:  {single_time*10:.2f} ms")
    print(f"  Throughput:      {100/single_time:.0f} evals/sec")

    # Test batch evaluation
    obs_batch = np.random.randn(64, 119, 8, 8).astype(np.float32)
    mask_batch = np.ones((64, 4672), dtype=np.float32)

    print("\nBatch evaluation (evaluator.evaluate_batch, batch=64):")
    start = time.perf_counter()
    for _ in range(100):
        policies, values = evaluator.evaluate_batch(obs_batch, mask_batch)
    if device == "cuda":
        torch.cuda.synchronize()
    batch_time = time.perf_counter() - start
    print(f"  100 batches:     {batch_time*1000:.1f} ms")
    print(f"  Per batch:       {batch_time*10:.2f} ms")
    print(f"  Throughput:      {100*64/batch_time:.0f} evals/sec")

    print(f"\nSpeedup from batching: {(100/single_time) / (100*64/batch_time) * 64:.1f}x")


def main():
    print("="*60)
    print("Training Path Benchmark")
    print("="*60)

    if torch.cuda.is_available():
        device = "cuda"
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        print(f"Device: CPU")

    print(f"PyTorch version: {torch.__version__}")

    # Compare single vs batch
    compare_single_vs_batch()

    # Benchmark actual training path
    result = benchmark_training_path(device, num_searches=20, simulations=800)

    if result:
        print(f"\n{'='*60}")
        print("FINAL RESULT")
        print(f"{'='*60}")
        print(f"  Training path performance: {result['sims_per_sec']:.0f} sims/sec")
        print(f"  Time per MCTS search:      {result['mean_time_ms']:.1f} ms")
        print(f"  Moves per second:          {result['searches_per_sec']:.2f}")


if __name__ == "__main__":
    main()
