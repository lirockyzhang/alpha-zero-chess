#!/usr/bin/env python3
"""
AlphaZero Training with C++ MCTS Backend

This script uses:
- C++ MCTS (alphazero-cpp) for fast tree search with proper leaf evaluation
- CUDA for neural network inference and training
- 192x15 network architecture (192 filters, 15 residual blocks)

Usage:
    # Basic training (recommended starting point)
    uv run python scripts/cpptrain.py

    # Custom parameters
    uv run python scripts/cpptrain.py --iterations 50 --games-per-iter 100 --simulations 800

    # Resume from checkpoint
    uv run python scripts/cpptrain.py --resume checkpoints/cpp_iter_10.pt

Parameters you can change:
    --iterations        Number of training iterations (default: 100)
    --games-per-iter    Self-play games per iteration (default: 50)
    --simulations       MCTS simulations per move (default: 800)
    --batch-size        Training batch size (default: 256)
    --mcts-batch-size   MCTS leaf evaluation batch size (default: 64)
    --lr                Learning rate (default: 0.001)
    --filters           Network filters (default: 192)
    --blocks            Residual blocks (default: 15)
    --buffer-size       Replay buffer size (default: 100000)
    --epochs            Training epochs per iteration (default: 5)
    --temperature-moves Moves with temperature=1 (default: 30)
    --c-puct            MCTS exploration constant (default: 1.5)
    --device            Device: cuda or cpu (default: cuda)
    --save-dir          Checkpoint directory (default: checkpoints)
    --resume            Resume from checkpoint path
    --save-interval     Save checkpoint every N iterations (default: 5)
"""

import argparse
import os
import sys
import time
import random
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "alphazero-cpp" / "build" / "Release"))

try:
    import alphazero_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("ERROR: alphazero_cpp not found. Build it first:")
    print("  cd alphazero-cpp")
    print("  cmake -B build -DCMAKE_BUILD_TYPE=Release")
    print("  cmake --build build --config Release")
    sys.exit(1)

import chess


# =============================================================================
# Neural Network (192x15 AlphaZero Architecture)
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
    """AlphaZero neural network with configurable size."""

    def __init__(self, num_filters: int = 192, num_blocks: int = 15):
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

    def forward(self, x, mask=None):
        # Input convolution
        out = self.input_conv(x)

        # Residual tower
        for block in self.res_blocks:
            out = block(out)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.reshape(policy.size(0), -1)
        policy = self.policy_fc(policy)

        if mask is not None:
            # Mask illegal moves
            policy = policy.masked_fill(mask == 0, -1e9)

        policy = F.softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.reshape(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


# =============================================================================
# Replay Buffer
# =============================================================================

@dataclass
class Experience:
    """Single training example."""
    observation: np.ndarray  # (119, 8, 8)
    policy: np.ndarray       # (4672,)
    value: float             # -1 to 1


class ReplayBuffer:
    """Replay buffer for training data."""

    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience: Experience):
        self.buffer.append(experience)

    def add_game(self, observations: List[np.ndarray], policies: List[np.ndarray], result: float):
        """Add a complete game to the buffer."""
        for i, (obs, policy) in enumerate(zip(observations, policies)):
            # Alternate value based on player (result is from white's perspective)
            value = result if i % 2 == 0 else -result
            self.buffer.append(Experience(obs, policy, value))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of experiences."""
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        batch = [self.buffer[i] for i in indices]

        observations = np.array([e.observation for e in batch], dtype=np.float32)
        policies = np.array([e.policy for e in batch], dtype=np.float32)
        values = np.array([e.value for e in batch], dtype=np.float32)

        return observations, policies, values

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# Batched Evaluator (GPU)
# =============================================================================

class BatchedEvaluator:
    """Efficient batched neural network evaluation on GPU."""

    def __init__(self, network: nn.Module, device: str, use_amp: bool = True):
        self.network = network
        self.device = device
        self.use_amp = use_amp and device == "cuda"

    @torch.no_grad()
    def evaluate(self, obs: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Evaluate single position."""
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)

        if self.use_amp:
            with autocast('cuda'):
                policy, value = self.network(obs_tensor, mask_tensor)
        else:
            policy, value = self.network(obs_tensor, mask_tensor)

        return policy[0].cpu().numpy(), float(value[0].item())

    @torch.no_grad()
    def evaluate_batch(self, obs_batch: np.ndarray, mask_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate batch of positions."""
        obs_tensor = torch.from_numpy(obs_batch).float().to(self.device)
        mask_tensor = torch.from_numpy(mask_batch).float().to(self.device)

        if self.use_amp:
            with autocast('cuda'):
                policies, values = self.network(obs_tensor, mask_tensor)
        else:
            policies, values = self.network(obs_tensor, mask_tensor)

        return policies.cpu().numpy(), values.squeeze(-1).cpu().numpy()


# =============================================================================
# Self-Play with C++ MCTS
# =============================================================================

class CppSelfPlay:
    """Self-play using C++ MCTS backend."""

    def __init__(
        self,
        evaluator: BatchedEvaluator,
        num_simulations: int = 800,
        mcts_batch_size: int = 64,
        c_puct: float = 1.5,
        temperature_moves: int = 30
    ):
        self.evaluator = evaluator
        self.num_simulations = num_simulations
        self.mcts_batch_size = mcts_batch_size
        self.c_puct = c_puct
        self.temperature_moves = temperature_moves

        # Create C++ MCTS engine
        self.mcts = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=num_simulations,
            batch_size=mcts_batch_size,
            c_puct=c_puct
        )

    def play_game(self) -> Tuple[List[np.ndarray], List[np.ndarray], float, int]:
        """Play a single self-play game.

        Returns:
            observations: List of board observations
            policies: List of MCTS policies
            result: Game result (1=white wins, -1=black wins, 0=draw)
            num_moves: Number of moves played
        """
        board = chess.Board()
        observations = []
        policies = []
        move_count = 0

        while not board.is_game_over() and move_count < 512:
            fen = board.fen()

            # Encode position
            obs = alphazero_cpp.encode_position(fen)  # (8, 8, 119) NHWC
            obs_chw = np.transpose(obs, (2, 0, 1))    # (119, 8, 8) CHW

            # Get legal mask
            legal_moves = list(board.legal_moves)
            mask = np.zeros(4672, dtype=np.float32)
            for move in legal_moves:
                idx = alphazero_cpp.move_to_index(move.uci(), fen)
                if 0 <= idx < 4672:
                    mask[idx] = 1.0

            # Get root evaluation
            root_policy, root_value = self.evaluator.evaluate(obs_chw, mask)

            # Initialize MCTS search
            self.mcts.init_search(fen, root_policy.astype(np.float32), float(root_value))

            # Run MCTS with batched leaf evaluation
            while not self.mcts.is_complete():
                num_leaves, obs_batch, mask_batch = self.mcts.collect_leaves()
                if num_leaves == 0:
                    break

                # Convert NHWC to NCHW
                obs_nchw = np.transpose(obs_batch[:num_leaves], (0, 3, 1, 2))
                masks = mask_batch[:num_leaves]

                # Batch evaluate
                leaf_policies, leaf_values = self.evaluator.evaluate_batch(obs_nchw, masks)

                # Update leaves
                self.mcts.update_leaves(
                    leaf_policies.astype(np.float32),
                    leaf_values.astype(np.float32)
                )

            # Get visit counts as policy
            visit_counts = self.mcts.get_visit_counts()
            policy = visit_counts.astype(np.float32)
            policy = policy * mask  # Mask illegal moves
            if policy.sum() > 0:
                policy = policy / policy.sum()
            else:
                policy = mask / mask.sum()

            # Store for training
            observations.append(obs_chw.copy())
            policies.append(policy.copy())

            # Select move with temperature
            if move_count < self.temperature_moves:
                # Sample proportionally
                action = np.random.choice(4672, p=policy)
            else:
                # Greedy
                action = np.argmax(policy)

            # Convert action to move
            move_uci = alphazero_cpp.index_to_move(action, fen)
            move = chess.Move.from_uci(move_uci)

            board.push(move)
            move_count += 1
            self.mcts.reset()

        # Get game result
        result = board.result()
        if result == "1-0":
            value = 1.0
        elif result == "0-1":
            value = -1.0
        else:
            value = 0.0

        return observations, policies, value, move_count


# =============================================================================
# Training Loop
# =============================================================================

def train_iteration(
    network: nn.Module,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    epochs: int,
    device: str,
    scaler: GradScaler
) -> dict:
    """Train for one iteration."""
    network.train()

    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    num_batches = 0

    for epoch in range(epochs):
        # Sample batch
        obs, policies, values = replay_buffer.sample(batch_size)

        obs_tensor = torch.from_numpy(obs).to(device)
        policy_target = torch.from_numpy(policies).to(device)
        value_target = torch.from_numpy(values).unsqueeze(1).to(device)

        optimizer.zero_grad()

        with autocast('cuda', enabled=(device == "cuda")):
            # Forward pass (no mask during training - targets already masked)
            policy_pred, value_pred = network(obs_tensor)

            # Policy loss (cross-entropy)
            policy_loss = -torch.sum(policy_target * torch.log(policy_pred + 1e-8)) / policy_pred.size(0)

            # Value loss (MSE)
            value_loss = F.mse_loss(value_pred, value_target)

            # Total loss
            loss = policy_loss + value_loss

        # Backward pass with mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero Training with C++ MCTS Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    uv run python scripts/cpptrain.py

    # Faster iteration (fewer games, more iterations)
    uv run python scripts/cpptrain.py --iterations 200 --games-per-iter 25

    # Higher quality (more simulations)
    uv run python scripts/cpptrain.py --simulations 1600

    # Resume training
    uv run python scripts/cpptrain.py --resume checkpoints/cpp_iter_50.pt
        """
    )

    # Training iterations
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations (default: 100)")
    parser.add_argument("--games-per-iter", type=int, default=50,
                        help="Self-play games per iteration (default: 50)")

    # MCTS parameters
    parser.add_argument("--simulations", type=int, default=800,
                        help="MCTS simulations per move (default: 800)")
    parser.add_argument("--mcts-batch-size", type=int, default=64,
                        help="MCTS leaf evaluation batch size (default: 64)")
    parser.add_argument("--c-puct", type=float, default=1.5,
                        help="MCTS exploration constant (default: 1.5)")
    parser.add_argument("--temperature-moves", type=int, default=30,
                        help="Moves with temperature=1 for exploration (default: 30)")

    # Network parameters
    parser.add_argument("--filters", type=int, default=192,
                        help="Network filters (default: 192)")
    parser.add_argument("--blocks", type=int, default=15,
                        help="Residual blocks (default: 15)")

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs per iteration (default: 5)")
    parser.add_argument("--buffer-size", type=int, default=100000,
                        help="Replay buffer size (default: 100000)")

    # Device and paths
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Checkpoint directory (default: checkpoints)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="Save checkpoint every N iterations (default: 5)")

    args = parser.parse_args()

    # Handle device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        print("To use CUDA, install: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        device = "cpu"
    else:
        device = args.device

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Print configuration
    print("=" * 70)
    print("AlphaZero Training with C++ MCTS Backend")
    print("=" * 70)
    print(f"Device:              {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"Network:             {args.filters} filters × {args.blocks} blocks")
    print(f"MCTS:                {args.simulations} sims, batch={args.mcts_batch_size}, c_puct={args.c_puct}")
    print(f"Training:            {args.iterations} iters × {args.games_per_iter} games")
    print(f"                     batch={args.batch_size}, lr={args.lr}, epochs={args.epochs}")
    print(f"Buffer:              {args.buffer_size} positions")
    print(f"Checkpoints:         {args.save_dir}/cpp_iter_*.pt (every {args.save_interval} iters)")
    print("=" * 70)

    # Create network
    network = AlphaZeroNet(num_filters=args.filters, num_blocks=args.blocks)
    network = network.to(device)

    num_params = sum(p.numel() for p in network.parameters())
    print(f"Network parameters:  {num_params:,}")

    # Create optimizer and scaler
    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler('cuda', enabled=(device == "cuda"))

    # Create replay buffer
    replay_buffer = ReplayBuffer(max_size=args.buffer_size)

    # Resume from checkpoint
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\nLoading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint.get('iteration', 0)
        print(f"Resumed from iteration {start_iter}")

    # Create evaluator and self-play
    evaluator = BatchedEvaluator(network, device, use_amp=(device == "cuda"))
    self_play = CppSelfPlay(
        evaluator=evaluator,
        num_simulations=args.simulations,
        mcts_batch_size=args.mcts_batch_size,
        c_puct=args.c_puct,
        temperature_moves=args.temperature_moves
    )

    print("\nStarting training...\n")

    # Training loop
    for iteration in range(start_iter, args.iterations):
        iter_start = time.time()

        # Self-play phase
        print(f"Iteration {iteration + 1}/{args.iterations}")
        print(f"  Self-play: generating {args.games_per_iter} games...")

        network.eval()
        games_data = []
        total_moves = 0

        for game_idx in range(args.games_per_iter):
            obs, policies, result, num_moves = self_play.play_game()
            replay_buffer.add_game(obs, policies, result)
            total_moves += num_moves

            if (game_idx + 1) % 10 == 0:
                print(f"    Game {game_idx + 1}/{args.games_per_iter}, "
                      f"moves={num_moves}, result={result:+.0f}, "
                      f"buffer={len(replay_buffer)}")

        selfplay_time = time.time() - iter_start
        moves_per_sec = total_moves / selfplay_time

        print(f"  Self-play complete: {total_moves} moves in {selfplay_time:.1f}s "
              f"({moves_per_sec:.1f} moves/sec)")

        # Training phase
        if len(replay_buffer) >= args.batch_size:
            print(f"  Training: {args.epochs} epochs on {len(replay_buffer)} positions...")
            train_start = time.time()

            metrics = train_iteration(
                network, optimizer, replay_buffer,
                args.batch_size, args.epochs, device, scaler
            )

            train_time = time.time() - train_start
            print(f"  Training complete: loss={metrics['loss']:.4f} "
                  f"(policy={metrics['policy_loss']:.4f}, value={metrics['value_loss']:.4f}) "
                  f"in {train_time:.1f}s")
        else:
            print(f"  Skipping training (buffer={len(replay_buffer)} < batch_size={args.batch_size})")

        iter_time = time.time() - iter_start
        print(f"  Iteration time: {iter_time:.1f}s\n")

        # Save checkpoint
        if (iteration + 1) % args.save_interval == 0 or iteration == args.iterations - 1:
            checkpoint_path = os.path.join(args.save_dir, f"cpp_iter_{iteration + 1}.pt")
            torch.save({
                'iteration': iteration + 1,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': {
                    'filters': args.filters,
                    'blocks': args.blocks,
                    'simulations': args.simulations,
                }
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}\n")

    print("=" * 70)
    print("Training complete!")
    print(f"Final checkpoint: {args.save_dir}/cpp_iter_{args.iterations}.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()
