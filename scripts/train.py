#!/usr/bin/env python3
"""Main training entry point for AlphaZero chess engine."""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphazero import AlphaZeroConfig, MCTSConfig, NetworkConfig, TrainingConfig
from alphazero.neural import AlphaZeroNetwork, count_parameters
from alphazero.selfplay import SelfPlayCoordinator


def setup_logging(log_dir: str, verbose: bool = False):
    """Setup logging configuration."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{log_dir}/training.log")
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero chess engine")

    # Training parameters
    parser.add_argument("--steps", type=int, default=100000,
                        help="Number of training steps")
    parser.add_argument("--actors", type=int, default=4,
                        help="Number of self-play actors")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Training batch size")

    # Network parameters
    parser.add_argument("--filters", type=int, default=192,
                        help="Number of filters in residual tower")
    parser.add_argument("--blocks", type=int, default=15,
                        help="Number of residual blocks")

    # MCTS parameters
    parser.add_argument("--simulations", type=int, default=800,
                        help="MCTS simulations per move")

    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for logs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

    # Other
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_dir, args.verbose)
    logger = logging.getLogger(__name__)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Create configuration
    config = AlphaZeroConfig(
        mcts=MCTSConfig(num_simulations=args.simulations),
        network=NetworkConfig(num_filters=args.filters, num_blocks=args.blocks),
        training=TrainingConfig(batch_size=args.batch_size),
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    config.selfplay.num_actors = args.actors

    logger.info("AlphaZero Chess Training")
    logger.info(f"Device: {args.device}")
    logger.info(f"Network: {args.blocks} blocks, {args.filters} filters")
    logger.info(f"MCTS: {args.simulations} simulations")
    logger.info(f"Actors: {args.actors}")

    # Create network
    network = AlphaZeroNetwork(
        num_filters=args.filters,
        num_blocks=args.blocks
    )
    logger.info(f"Network parameters: {count_parameters(network):,}")

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        state = torch.load(args.resume, map_location=args.device)
        network.load_state_dict(state['network_state_dict'])

    # Create coordinator and run training
    coordinator = SelfPlayCoordinator(network, config)

    try:
        coordinator.run(num_steps=args.steps, num_actors=args.actors)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        # Save final checkpoint
        final_path = f"{args.checkpoint_dir}/checkpoint_final.pt"
        coordinator.learner.save_checkpoint(final_path)
        logger.info(f"Saved final checkpoint to {final_path}")

    logger.info("Training complete!")
    stats = coordinator.get_stats()
    logger.info(f"Total games: {stats['total_games']}")
    logger.info(f"Total positions: {stats['total_positions']}")


if __name__ == "__main__":
    main()
