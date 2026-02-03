#!/usr/bin/env python3
"""
AlphaZero Optimal Training Script

This script provides recommended configurations for training AlphaZero chess
using the C++ backend with optimal parameters based on the AlphaZero paper.

Usage:
    # Quick test (small network, ~2 hours)
    uv run python scripts/train_optimal.py --preset test

    # Development (medium network, ~8 hours)
    uv run python scripts/train_optimal.py --preset dev

    # Production (full network, ~24+ hours)
    uv run python scripts/train_optimal.py --preset full

    # Custom parameters
    uv run python scripts/train_optimal.py --iterations 100 --games-per-iter 50

Presets:
    test:  64×5 network,  100 sims, 10 iters,  10 games/iter (~2 hours)
    dev:   128×10 network, 400 sims, 50 iters,  25 games/iter (~8 hours)
    full:  192×15 network, 800 sims, 100 iters, 50 games/iter (~24+ hours)
"""

import subprocess
import sys
import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingPreset:
    """Pre-configured training parameters."""
    name: str
    filters: int
    blocks: int
    simulations: int
    iterations: int
    games_per_iter: int
    batch_size: int
    mcts_batch_size: int
    learning_rate: float
    buffer_size: int
    description: str


# Optimal presets based on AlphaZero paper and empirical testing
PRESETS = {
    "test": TrainingPreset(
        name="test",
        filters=64,
        blocks=5,
        simulations=100,
        iterations=10,
        games_per_iter=10,
        batch_size=256,
        mcts_batch_size=32,
        learning_rate=0.001,
        buffer_size=50000,
        description="Quick test (~2 hours, RTX 4060)"
    ),
    "dev": TrainingPreset(
        name="dev",
        filters=128,
        blocks=10,
        simulations=400,
        iterations=50,
        games_per_iter=25,
        batch_size=512,
        mcts_batch_size=64,
        learning_rate=0.001,
        buffer_size=100000,
        description="Development (~8 hours, RTX 4060)"
    ),
    "full": TrainingPreset(
        name="full",
        filters=192,
        blocks=15,
        simulations=800,
        iterations=100,
        games_per_iter=50,
        batch_size=1024,
        mcts_batch_size=64,
        learning_rate=0.001,
        buffer_size=500000,
        description="Production (~24+ hours, RTX 4090/A100)"
    ),
    "paper": TrainingPreset(
        name="paper",
        filters=256,
        blocks=19,
        simulations=800,
        iterations=500,
        games_per_iter=100,
        batch_size=4096,
        mcts_batch_size=128,
        learning_rate=0.2,
        buffer_size=1000000,
        description="AlphaZero paper settings (requires multi-GPU)"
    ),
}


def print_preset_info(preset: TrainingPreset):
    """Print preset configuration."""
    print(f"\n{'=' * 60}")
    print(f"Training Preset: {preset.name.upper()}")
    print(f"{'=' * 60}")
    print(f"Description: {preset.description}")
    print(f"\nNetwork Architecture:")
    print(f"  Filters: {preset.filters}")
    print(f"  Blocks:  {preset.blocks}")
    print(f"  ~Parameters: {estimate_params(preset.filters, preset.blocks):,}")
    print(f"\nMCTS Configuration:")
    print(f"  Simulations: {preset.simulations}")
    print(f"  Batch size:  {preset.mcts_batch_size}")
    print(f"\nTraining Configuration:")
    print(f"  Iterations:     {preset.iterations}")
    print(f"  Games/iter:     {preset.games_per_iter}")
    print(f"  Batch size:     {preset.batch_size}")
    print(f"  Learning rate:  {preset.learning_rate}")
    print(f"  Buffer size:    {preset.buffer_size:,}")
    print(f"{'=' * 60}\n")


def estimate_params(filters: int, blocks: int) -> int:
    """Estimate network parameters (rough approximation)."""
    # Input conv: 122 * filters * 9 (3x3 kernel)
    input_params = 122 * filters * 9
    # Each residual block: 2 * filters * filters * 9
    block_params = 2 * filters * filters * 9 * blocks
    # Policy head: ~filters * 2 * 1 + 2 * 64 * 4672
    policy_params = filters * 2 + 128 * 4672
    # Value head: ~filters * 1 * 1 + 64 * 256 + 256
    value_params = filters + 64 * 256 + 256
    return input_params + block_params + policy_params + value_params


def build_command(preset: TrainingPreset, args) -> list:
    """Build the training command."""
    cmd = [
        sys.executable, "alphazero-cpp/scripts/train.py",
        f"--iterations={args.iterations or preset.iterations}",
        f"--games-per-iter={args.games_per_iter or preset.games_per_iter}",
        f"--simulations={args.simulations or preset.simulations}",
        f"--mcts-batch-size={preset.mcts_batch_size}",
        f"--filters={preset.filters}",
        f"--blocks={preset.blocks}",
        f"--batch-size={preset.batch_size}",
        f"--lr={preset.learning_rate}",
        f"--buffer-size={preset.buffer_size}",
        f"--device={args.device}",
        f"--save-dir={args.save_dir}",
        f"--save-interval={args.save_interval}",
    ]

    if args.resume:
        cmd.append(f"--resume={args.resume}")

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero Optimal Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Preset selection
    parser.add_argument("--preset", type=str, default="dev",
                        choices=list(PRESETS.keys()),
                        help="Training preset (default: dev)")
    parser.add_argument("--list-presets", action="store_true",
                        help="List all available presets")

    # Override parameters
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override iterations")
    parser.add_argument("--games-per-iter", type=int, default=None,
                        help="Override games per iteration")
    parser.add_argument("--simulations", type=int, default=None,
                        help="Override MCTS simulations")

    # Common options
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="Save every N iterations")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print command without executing")

    args = parser.parse_args()

    # List presets
    if args.list_presets:
        print("\nAvailable Training Presets:")
        print("-" * 60)
        for name, preset in PRESETS.items():
            params = estimate_params(preset.filters, preset.blocks)
            print(f"\n  {name:8s}: {preset.description}")
            print(f"           Network: {preset.filters}×{preset.blocks} (~{params//1000}K params)")
            print(f"           MCTS: {preset.simulations} sims, {preset.iterations} iters, {preset.games_per_iter} games/iter")
        print()
        return

    # Get preset
    preset = PRESETS[args.preset]
    print_preset_info(preset)

    # Build and run command
    cmd = build_command(preset, args)

    print("Command:")
    print(f"  {' '.join(cmd)}\n")

    if args.dry_run:
        print("(Dry run - not executing)")
        return

    # Execute
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
