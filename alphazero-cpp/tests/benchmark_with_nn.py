#!/usr/bin/env python3
"""
Benchmark MCTS with ACTUAL neural network evaluation.

This measures REAL AlphaZero performance, not just tree operations.
The previous benchmarks used RandomEvaluator which is instant -
this uses actual neural network inference which is the real bottleneck.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "build" / "Release"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import alphazero_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("WARNING: alphazero_cpp not available")

import chess


# =============================================================================
# Neural Network (same as train.py)
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class AlphaZeroNet(nn.Module):
    def __init__(self, num_filters: int = 64, num_blocks: int = 5):
        super().__init__()
        self.input_conv = ConvBlock(119, num_filters)
        self.res_blocks = nn.ModuleList([ResBlock(num_filters) for _ in range(num_blocks)])
        self.policy_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4672)
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x, mask=None):
        out = self.input_conv(x)
        for block in self.res_blocks:
            out = block(out)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.reshape(policy.size(0), -1)  # Use reshape for non-contiguous tensors
        policy = self.policy_fc(policy)
        if mask is not None:
            policy = policy.masked_fill(mask == 0, -1e9)
        policy = F.softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.reshape(value.size(0), -1)  # Use reshape for non-contiguous tensors
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_single_inference(network, device, num_evals=100):
    """Benchmark single position inference (worst case)."""
    print(f"\n{'='*60}")
    print("Benchmark: Single Position Inference")
    print(f"{'='*60}")

    # Create dummy input
    obs = torch.randn(1, 119, 8, 8, device=device)
    mask = torch.ones(1, 4672, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            network(obs, mask)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_evals):
        with torch.no_grad():
            network(obs, mask)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(f"  Evaluations: {num_evals}")
    print(f"  Total time: {elapsed*1000:.2f} ms")
    print(f"  Per evaluation: {elapsed/num_evals*1000:.3f} ms")
    print(f"  Throughput: {num_evals/elapsed:.1f} evals/sec")

    return num_evals / elapsed


def benchmark_batched_inference(network, device, batch_sizes=[1, 8, 16, 32, 64, 128, 256]):
    """Benchmark batched inference at different batch sizes."""
    print(f"\n{'='*60}")
    print("Benchmark: Batched Inference")
    print(f"{'='*60}")

    results = {}

    for batch_size in batch_sizes:
        obs = torch.randn(batch_size, 119, 8, 8, device=device)
        mask = torch.ones(batch_size, 4672, device=device)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                network(obs, mask)

        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        num_batches = max(10, 100 // batch_size)
        start = time.perf_counter()
        for _ in range(num_batches):
            with torch.no_grad():
                network(obs, mask)

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        positions_per_sec = (num_batches * batch_size) / elapsed

        results[batch_size] = positions_per_sec
        print(f"  Batch size {batch_size:3d}: {positions_per_sec:,.0f} positions/sec "
              f"({elapsed/num_batches*1000:.2f} ms/batch)")

    return results


def benchmark_mcts_with_nn(network, device, num_searches=10, simulations=800, batch_size=64):
    """Benchmark full MCTS search with neural network evaluation."""
    print(f"\n{'='*60}")
    print(f"Benchmark: Full MCTS with Neural Network")
    print(f"  Simulations: {simulations}, Batch size: {batch_size}")
    print(f"{'='*60}")

    if not CPP_AVAILABLE:
        print("  ERROR: alphazero_cpp not available")
        return None

    # Create MCTS engine
    mcts = alphazero_cpp.BatchedMCTSSearch(
        num_simulations=simulations,
        batch_size=batch_size,
        c_puct=1.5
    )

    # Pre-allocate buffers
    obs_buffer = np.zeros((batch_size, 8, 8, 119), dtype=np.float32)
    mask_buffer = np.zeros((batch_size, 4672), dtype=np.float32)

    # Timing breakdown
    total_time = 0
    total_nn_time = 0
    total_mcts_time = 0
    total_leaves = 0
    total_batches = 0

    board = chess.Board()
    fen = board.fen()

    for search_idx in range(num_searches):
        search_start = time.perf_counter()

        # Get initial evaluation
        obs = alphazero_cpp.encode_position(fen)  # (8, 8, 119) NHWC
        obs_chw = np.transpose(obs, (2, 0, 1))  # (119, 8, 8) CHW
        obs_tensor = torch.from_numpy(obs_chw).unsqueeze(0).to(device)

        legal_moves = list(board.legal_moves)
        mask = np.zeros(4672, dtype=np.float32)
        for move in legal_moves:
            idx = alphazero_cpp.move_to_index(move.uci(), fen)
            if 0 <= idx < 4672:
                mask[idx] = 1.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)

        nn_start = time.perf_counter()
        with torch.no_grad():
            policy, value = network(obs_tensor, mask_tensor)
        if device == "cuda":
            torch.cuda.synchronize()
        total_nn_time += time.perf_counter() - nn_start

        root_policy = policy[0].cpu().numpy()
        root_value = float(value[0].item())

        # Initialize search
        mcts_start = time.perf_counter()
        mcts.init_search(fen, root_policy, root_value)
        total_mcts_time += time.perf_counter() - mcts_start

        # Search loop
        while not mcts.is_complete():
            mcts_start = time.perf_counter()
            num_leaves, obs_batch, mask_batch = mcts.collect_leaves()
            total_mcts_time += time.perf_counter() - mcts_start

            if num_leaves == 0:
                break

            total_leaves += num_leaves
            total_batches += 1

            # Convert NHWC to NCHW
            obs_nchw = np.transpose(obs_batch[:num_leaves], (0, 3, 1, 2))
            obs_tensor = torch.from_numpy(obs_nchw).to(device)
            mask_tensor = torch.from_numpy(mask_batch[:num_leaves]).to(device)

            nn_start = time.perf_counter()
            with torch.no_grad():
                policies, values = network(obs_tensor, mask_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
            total_nn_time += time.perf_counter() - nn_start

            policies_np = policies.cpu().numpy()
            values_np = values.squeeze(-1).cpu().numpy()

            mcts_start = time.perf_counter()
            mcts.update_leaves(policies_np, values_np)
            total_mcts_time += time.perf_counter() - mcts_start

        mcts_start = time.perf_counter()
        visit_counts = mcts.get_visit_counts()
        mcts.reset()
        total_mcts_time += time.perf_counter() - mcts_start

        total_time += time.perf_counter() - search_start

    # Results
    searches_per_sec = num_searches / total_time
    sims_per_sec = (num_searches * simulations) / total_time
    avg_leaves_per_search = total_leaves / num_searches
    avg_batches_per_search = total_batches / num_searches

    print(f"\n  Results ({num_searches} searches):")
    print(f"  {'─'*50}")
    print(f"  Total time:        {total_time*1000:.1f} ms")
    print(f"  NN time:           {total_nn_time*1000:.1f} ms ({total_nn_time/total_time*100:.1f}%)")
    print(f"  MCTS time:         {total_mcts_time*1000:.1f} ms ({total_mcts_time/total_time*100:.1f}%)")
    print(f"  {'─'*50}")
    print(f"  Searches/sec:      {searches_per_sec:.2f}")
    print(f"  Simulations/sec:   {sims_per_sec:.0f}")
    print(f"  Time per search:   {total_time/num_searches*1000:.1f} ms")
    print(f"  {'─'*50}")
    print(f"  Avg leaves/search: {avg_leaves_per_search:.1f}")
    print(f"  Avg batches/search:{avg_batches_per_search:.1f}")
    print(f"  Avg batch size:    {total_leaves/total_batches:.1f}" if total_batches > 0 else "")

    return {
        'searches_per_sec': searches_per_sec,
        'sims_per_sec': sims_per_sec,
        'nn_time_pct': total_nn_time / total_time * 100,
        'mcts_time_pct': total_mcts_time / total_time * 100,
    }


def benchmark_random_evaluator(num_searches=10, simulations=800):
    """Benchmark MCTS with random evaluator (for comparison)."""
    print(f"\n{'='*60}")
    print(f"Benchmark: MCTS with Random Evaluator (NO neural network)")
    print(f"  Simulations: {simulations}")
    print(f"{'='*60}")

    if not CPP_AVAILABLE:
        print("  ERROR: alphazero_cpp not available")
        return None

    mcts = alphazero_cpp.MCTSSearch(
        num_simulations=simulations,
        c_puct=1.5
    )

    board = chess.Board()
    fen = board.fen()

    # Random policy
    policy = np.random.rand(4672).astype(np.float32)
    policy = policy / policy.sum()
    value = 0.0

    # Warmup
    for _ in range(3):
        mcts.search(fen, policy, value)

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_searches):
        mcts.search(fen, policy, value)
    elapsed = time.perf_counter() - start

    searches_per_sec = num_searches / elapsed
    sims_per_sec = (num_searches * simulations) / elapsed

    print(f"  Searches/sec:      {searches_per_sec:.2f}")
    print(f"  Simulations/sec:   {sims_per_sec:,.0f}")
    print(f"  Time per search:   {elapsed/num_searches*1000:.2f} ms")

    return sims_per_sec


def main():
    print("="*60)
    print("AlphaZero Performance Benchmark WITH Neural Network")
    print("="*60)

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        print(f"Device: CPU")

    print(f"PyTorch version: {torch.__version__}")
    print(f"C++ extension: {'Available' if CPP_AVAILABLE else 'NOT AVAILABLE'}")

    # Create network
    network = AlphaZeroNet(num_filters=64, num_blocks=5)
    network = network.to(device)
    network.eval()

    num_params = sum(p.numel() for p in network.parameters())
    print(f"Network: 64 filters, 5 blocks ({num_params:,} parameters)")

    # Run benchmarks
    benchmark_single_inference(network, device, num_evals=100)
    benchmark_batched_inference(network, device)

    # Compare random vs NN
    random_sims = benchmark_random_evaluator(num_searches=100, simulations=800)
    nn_result = benchmark_mcts_with_nn(network, device, num_searches=10, simulations=800, batch_size=64)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Random Evaluator vs Neural Network")
    print(f"{'='*60}")
    if random_sims and nn_result:
        print(f"  Random evaluator: {random_sims:,.0f} sims/sec")
        print(f"  Neural network:   {nn_result['sims_per_sec']:,.0f} sims/sec")
        print(f"  Slowdown factor:  {random_sims / nn_result['sims_per_sec']:.1f}x")
        print(f"\n  NN is {nn_result['nn_time_pct']:.1f}% of total time (the bottleneck!)")


if __name__ == "__main__":
    main()
