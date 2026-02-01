#!/usr/bin/env python3
"""
Optimized AlphaZero Training Script

This script maximizes performance by:
1. Using C++ MCTS for fast tree operations (selection, expansion, backprop)
2. Using CUDA for BATCHED neural network inference (not one-by-one!)
3. Using CUDA for training the policy and value networks

The key optimization is batching NN inference - instead of evaluating
one leaf at a time, we collect all leaves and evaluate them in one
GPU batch call.

Usage:
    python train_fast.py --iterations 10 --games-per-iter 100 --simulations 200
"""

import argparse
import sys
import os
import time
import random
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add C++ extension to path
sys.path.insert(0, str(Path(__file__).parent / "build" / "Release"))

try:
    import alphazero_cpp
except ImportError:
    print("ERROR: alphazero_cpp not found. Build it first:")
    print("  cd alphazero-cpp")
    print("  cmake -B build -DCMAKE_BUILD_TYPE=Release")
    print("  cmake --build build --config Release")
    sys.exit(1)

import chess


# =============================================================================
# Neural Network (Minimal AlphaZero Architecture)
# =============================================================================

class ConvBlock(nn.Module):
    """Convolutional block with batch norm and ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block."""
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
    """Minimal AlphaZero neural network."""

    def __init__(self, num_filters: int = 64, num_blocks: int = 5):
        super().__init__()
        self.num_filters = num_filters
        self.num_blocks = num_blocks

        # Input: 119 planes of 8x8
        self.input_conv = ConvBlock(119, num_filters)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResBlock(num_filters) for _ in range(num_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4672)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x, legal_mask=None):
        # Input convolution
        x = self.input_conv(x)

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Apply legal mask
        if legal_mask is not None:
            p = p.masked_fill(legal_mask == 0, -1e9)

        p = F.softmax(p, dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v.squeeze(-1)


# =============================================================================
# Batched Neural Network Evaluator (CUDA optimized)
# =============================================================================

class BatchedEvaluator:
    """
    Batched neural network evaluator for maximum GPU utilization.

    Instead of evaluating one position at a time, this class:
    1. Collects multiple positions into a batch
    2. Runs ONE forward pass on GPU for the entire batch
    3. Returns all results at once

    This is 10-50x faster than single-position evaluation!
    """

    def __init__(self, network: AlphaZeroNet, device: str, max_batch_size: int = 256):
        self.network = network
        self.device = device
        self.max_batch_size = max_batch_size

        # Pre-allocate GPU tensors for zero-copy
        self.obs_buffer = torch.zeros(
            (max_batch_size, 119, 8, 8),
            dtype=torch.float32,
            device=device
        )
        self.mask_buffer = torch.zeros(
            (max_batch_size, 4672),
            dtype=torch.float32,
            device=device
        )

    @torch.no_grad()
    def evaluate_batch(self, obs_batch: np.ndarray, mask_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a batch of positions in ONE GPU call.

        Args:
            obs_batch: (batch_size, 119, 8, 8) observations
            mask_batch: (batch_size, 4672) legal masks

        Returns:
            policies: (batch_size, 4672) policy distributions
            values: (batch_size,) value estimates
        """
        batch_size = obs_batch.shape[0]

        if batch_size == 0:
            return np.array([]), np.array([])

        # Copy to pre-allocated GPU buffers (fast!)
        self.obs_buffer[:batch_size].copy_(torch.from_numpy(obs_batch))
        self.mask_buffer[:batch_size].copy_(torch.from_numpy(mask_batch))

        # Single GPU forward pass for entire batch
        policies, values = self.network(
            self.obs_buffer[:batch_size],
            self.mask_buffer[:batch_size]
        )

        return policies.cpu().numpy(), values.cpu().numpy()

    @torch.no_grad()
    def evaluate_single(self, obs: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Evaluate a single position (for root node)."""
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        policy, value = self.network(obs_tensor, mask_tensor)
        return policy.cpu().numpy()[0], value.cpu().item()


# =============================================================================
# Training Data
# =============================================================================

@dataclass
class TrainingSample:
    """Single training sample."""
    observation: np.ndarray  # (119, 8, 8)
    policy: np.ndarray       # (4672,)
    value: float


class ReplayBuffer:
    """Simple replay buffer for training samples."""

    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, sample: TrainingSample):
        self.buffer.append(sample)

    def add_game(self, samples: List[TrainingSample]):
        for sample in samples:
            self.add(sample)

    def sample(self, batch_size: int) -> List[TrainingSample]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class TrainingDataset(Dataset):
    """PyTorch dataset for training."""

    def __init__(self, samples: List[TrainingSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.FloatTensor(s.observation),
            torch.FloatTensor(s.policy),
            torch.FloatTensor([s.value])
        )


# =============================================================================
# Fast Self-Play with Batched Inference
# =============================================================================

class FastSelfPlay:
    """
    High-performance self-play using:
    - C++ MCTS for fast tree operations
    - CUDA batched inference for neural network evaluation

    The key insight is that C++ collects ALL leaves that need evaluation,
    then we evaluate them in ONE batched GPU call, then C++ updates all
    leaves and continues. This maximizes both CPU and GPU utilization.
    """

    def __init__(self, evaluator: BatchedEvaluator,
                 num_simulations: int = 200, batch_size: int = 64):
        self.evaluator = evaluator
        self.num_simulations = num_simulations
        self.batch_size = batch_size

    def play_game(self, temperature_threshold: int = 30) -> Tuple[List[TrainingSample], str]:
        """
        Play a single self-play game with batched inference.

        Returns:
            samples: List of training samples
            result: Game result string ("1-0", "0-1", "1/2-1/2")
        """
        board = chess.Board()
        samples = []
        move_count = 0

        # Create C++ MCTS search engine
        mcts = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=self.num_simulations,
            batch_size=self.batch_size,
            c_puct=1.5
        )

        while not board.is_game_over() and move_count < 512:
            # Encode current position
            fen = board.fen()
            obs = np.array(alphazero_cpp.encode_position(fen), dtype=np.float32)
            obs = obs.reshape(8, 8, 119).transpose(2, 0, 1)  # NHWC -> CHW

            # Get legal mask
            legal_mask = np.zeros(4672, dtype=np.float32)
            for move in board.legal_moves:
                idx = alphazero_cpp.move_to_index(move.uci(), fen)
                if idx >= 0:
                    legal_mask[idx] = 1.0

            # Get initial policy and value from network (single position)
            root_policy, root_value = self.evaluator.evaluate_single(obs, legal_mask)

            # Initialize MCTS search with root evaluation
            mcts.init_search(fen, root_policy.astype(np.float32), float(root_value))

            # Run MCTS with BATCHED leaf evaluation
            while not mcts.is_complete():
                # C++ collects all leaves that need evaluation
                num_leaves, obs_batch_nhwc, mask_batch = mcts.collect_leaves()

                if num_leaves == 0:
                    break

                # Convert NHWC -> NCHW for PyTorch
                # obs_batch_nhwc shape: (num_leaves, 8, 8, 119)
                obs_batch = np.transpose(obs_batch_nhwc, (0, 3, 1, 2))  # -> (num_leaves, 119, 8, 8)

                # BATCHED GPU inference - evaluate ALL leaves in ONE call!
                policies, values = self.evaluator.evaluate_batch(
                    obs_batch.astype(np.float32),
                    mask_batch.astype(np.float32)
                )

                # C++ updates all leaves and continues search
                mcts.update_leaves(
                    policies.astype(np.float32),
                    values.astype(np.float32)
                )

            # Get visit counts as policy target
            visit_counts = mcts.get_visit_counts().astype(np.float32)
            visit_counts = visit_counts * legal_mask  # Mask illegal moves
            if visit_counts.sum() > 0:
                mcts_policy = visit_counts / visit_counts.sum()
            else:
                mcts_policy = legal_mask / legal_mask.sum()

            # Store sample (value will be updated at game end)
            samples.append(TrainingSample(
                observation=obs.copy(),
                policy=mcts_policy.copy(),
                value=0.0  # Placeholder
            ))

            # Select move with temperature
            temperature = 1.0 if move_count < temperature_threshold else 0.1
            if temperature < 0.5:
                action = int(np.argmax(mcts_policy))
            else:
                probs = np.power(mcts_policy, 1.0 / temperature)
                probs = probs / probs.sum()
                action = int(np.random.choice(len(probs), p=probs))

            # Convert action to move and apply
            move_uci = alphazero_cpp.index_to_move(action, fen)
            if move_uci:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    board.push(random.choice(list(board.legal_moves)))
            else:
                board.push(random.choice(list(board.legal_moves)))

            move_count += 1
            mcts.reset()

        # Determine game result
        result = board.result()
        if result == "1-0":
            final_value = 1.0
        elif result == "0-1":
            final_value = -1.0
        else:
            final_value = 0.0

        # Update sample values (from each player's perspective)
        for i, sample in enumerate(samples):
            sample.value = final_value if i % 2 == 0 else -final_value

        return samples, result


# =============================================================================
# Training Loop (CUDA optimized)
# =============================================================================

def train_network(network: AlphaZeroNet, replay_buffer: ReplayBuffer,
                  device: str, batch_size: int = 256, epochs: int = 1,
                  lr: float = 0.001) -> dict:
    """Train the network on replay buffer data using CUDA."""
    if len(replay_buffer) < batch_size:
        return {"loss": 0, "policy_loss": 0, "value_loss": 0}

    network.train()
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)

    samples = replay_buffer.sample(min(len(replay_buffer), batch_size * epochs * 4))
    dataset = TrainingDataset(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=(device == "cuda"))

    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    num_batches = 0

    for epoch in range(epochs):
        for obs, target_policy, target_value in dataloader:
            obs = obs.to(device, non_blocking=True)
            target_policy = target_policy.to(device, non_blocking=True)
            target_value = target_value.to(device, non_blocking=True).squeeze()

            optimizer.zero_grad()

            # Forward pass (no legal mask during training)
            policy, value = network(obs)

            # Policy loss (cross-entropy)
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8)) / obs.size(0)

            # Value loss (MSE)
            value_loss = F.mse_loss(value, target_value)

            # Total loss
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

    network.eval()

    return {
        "loss": total_loss / max(num_batches, 1),
        "policy_loss": total_policy_loss / max(num_batches, 1),
        "value_loss": total_value_loss / max(num_batches, 1)
    }


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fast AlphaZero Training with Batched Inference")
    parser.add_argument("--iterations", type=int, default=10, help="Training iterations")
    parser.add_argument("--games-per-iter", type=int, default=10, help="Games per iteration")
    parser.add_argument("--simulations", type=int, default=200, help="MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=64, help="MCTS leaf batch size")
    parser.add_argument("--train-batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--filters", type=int, default=64, help="Network filters")
    parser.add_argument("--blocks", type=int, default=5, help="Network residual blocks")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")
    args = parser.parse_args()

    # Handle device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("=" * 70)
            print("WARNING: CUDA requested but not available!")
            print("=" * 70)
            print(f"PyTorch version: {torch.__version__}")
            print()
            print("Your PyTorch is CPU-only. To use CUDA, install CUDA-enabled PyTorch:")
            print()
            print("  # For CUDA 11.8:")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
            print()
            print("  # For CUDA 12.1:")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
            print()
            print("Falling back to CPU...")
            print("=" * 70)
            print()
            device = "cpu"
        else:
            device = "cuda"
    else:
        device = args.device

    print("=" * 70)
    print("Fast AlphaZero Training")
    print("  - C++ MCTS for tree operations")
    print("  - Batched neural network inference")
    print("=" * 70)
    print(f"Device: {device}" + (" (GPU)" if device == "cuda" else " (CPU)"))
    print(f"Network: {args.filters} filters, {args.blocks} blocks")
    print(f"MCTS: {args.simulations} simulations, batch_size={args.batch_size}")
    print(f"Training: {args.iterations} iterations, {args.games_per_iter} games/iter")
    print("=" * 70)
    print()

    # Create network on GPU
    network = AlphaZeroNet(num_filters=args.filters, num_blocks=args.blocks)
    network = network.to(device)
    network.eval()

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        network.load_state_dict(checkpoint["model_state_dict"])
        print("Checkpoint loaded!")

    # Create batched evaluator (uses GPU)
    evaluator = BatchedEvaluator(
        network=network,
        device=args.device,
        max_batch_size=args.batch_size * 2
    )

    # Create replay buffer
    replay_buffer = ReplayBuffer(max_size=100000)

    # Create fast self-play engine
    self_play = FastSelfPlay(
        evaluator=evaluator,
        num_simulations=args.simulations,
        batch_size=args.batch_size
    )

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Training loop
    total_games = 0
    total_moves = 0
    total_time = 0

    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*70}")
        print(f"Iteration {iteration}/{args.iterations}")
        print(f"{'='*70}")

        # Self-play phase
        print(f"\nSelf-play: generating {args.games_per_iter} games...")
        iter_start = time.time()
        iter_moves = 0
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

        for game_idx in range(args.games_per_iter):
            game_start = time.time()
            samples, result = self_play.play_game()
            replay_buffer.add_game(samples)
            iter_moves += len(samples)
            results[result] = results.get(result, 0) + 1

            game_time = time.time() - game_start
            moves_per_sec = len(samples) / game_time
            print(f"  Game {game_idx + 1}/{args.games_per_iter}: "
                  f"{len(samples)} moves in {game_time:.1f}s ({moves_per_sec:.1f} moves/s) [{result}]")

        self_play_time = time.time() - iter_start
        total_games += args.games_per_iter
        total_moves += iter_moves
        total_time += self_play_time

        print(f"\nSelf-play summary:")
        print(f"  Games: {args.games_per_iter} ({results})")
        print(f"  Moves: {iter_moves} in {self_play_time:.1f}s ({iter_moves/self_play_time:.1f} moves/s)")
        print(f"  Replay buffer: {len(replay_buffer)} samples")

        # Training phase
        print(f"\nTraining on {args.device}...")
        train_start = time.time()
        metrics = train_network(
            network=network,
            replay_buffer=replay_buffer,
            device=args.device,
            batch_size=args.train_batch_size,
            epochs=1,
            lr=args.lr
        )
        train_time = time.time() - train_start

        print(f"Training complete in {train_time:.1f}s")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Policy loss: {metrics['policy_loss']:.4f}")
        print(f"  Value loss: {metrics['value_loss']:.4f}")

        # Save checkpoint
        checkpoint_path = save_dir / f"checkpoint_iter{iteration}.pt"
        torch.save({
            "iteration": iteration,
            "model_state_dict": network.state_dict(),
            "metrics": metrics,
            "total_games": total_games,
            "total_moves": total_moves
        }, checkpoint_path)
        print(f"Saved: {checkpoint_path}")

    # Save final model
    final_path = save_dir / f"model_final_f{args.filters}_b{args.blocks}.pt"
    torch.save({
        "model_state_dict": network.state_dict(),
        "num_filters": args.filters,
        "num_blocks": args.blocks,
        "total_games": total_games,
        "total_moves": total_moves
    }, final_path)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Total games: {total_games}")
    print(f"Total moves: {total_moves}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average: {total_moves/total_time:.1f} moves/s")
    print(f"Final model: {final_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
