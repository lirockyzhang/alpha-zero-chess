#!/usr/bin/env python3
"""Main training entry point for AlphaZero chess engine."""

import argparse
import logging
import multiprocessing
import sys
from pathlib import Path

import torch

# IMPORTANT: Set multiprocessing start method to 'spawn' for CUDA compatibility.
# On Linux (including Google Colab), the default is 'fork', which causes:
# "RuntimeError: Cannot re-initialize CUDA in forked subprocess"
# This must be called before any CUDA operations or process creation.
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # Already set (e.g., on Windows where 'spawn' is default)
        pass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphazero import AlphaZeroConfig, MCTSConfig, NetworkConfig, TrainingConfig, ReplayBufferConfig, MCTSBackend, PROFILES, TrainingProfile
from alphazero.neural import AlphaZeroNetwork, count_parameters
from alphazero.selfplay import SelfPlayCoordinator, BatchedSelfPlayCoordinator
from alphazero.mcts import get_available_backends, get_best_backend


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


def run_iterative_training(coordinator, args, logger, num_actors: int):
    """Run iterative training with buffer refresh.

    Each iteration:
    1. Starts actors and fills replay buffer with fresh games
    2. Trains for steps_per_iteration
    3. Clears replay buffer (forces new games to be generated)
    4. Repeats

    This prevents overfitting to weak early games and accelerates learning.
    """
    from tqdm import tqdm

    logger.info("="*60)
    logger.info("ITERATIVE TRAINING MODE")
    logger.info("="*60)
    logger.info(f"Iterations: {args.iterations}")
    logger.info(f"Steps per iteration: {args.steps_per_iteration}")
    logger.info(f"Total steps: {args.iterations * args.steps_per_iteration}")
    logger.info("="*60)

    # Start inference server first (creates response queues)
    coordinator.start_inference_server(num_actors)
    import time
    time.sleep(1)  # Give server time to initialize

    # Then start actors (uses pre-created response queues)
    # Note: BatchedSelfPlayCoordinator doesn't have start_collection() - actors
    # automatically begin collecting trajectories once started
    coordinator.start_actors(num_actors)

    try:
        for iteration in range(args.iterations):
            logger.info("")
            logger.info("="*60)
            logger.info(f"ITERATION {iteration + 1}/{args.iterations}")
            logger.info("="*60)

            # Step 1: Fill replay buffer with fresh games
            min_size = coordinator.config.replay_buffer.min_size_to_train
            logger.info(f"Filling replay buffer (target: {min_size} positions)...")

            with tqdm(total=min_size, desc=f"Iter {iteration+1} - Filling buffer", unit="pos") as pbar:
                while len(coordinator.replay_buffer) < min_size and not coordinator._shutdown_requested:
                    prev_size = len(coordinator.replay_buffer)
                    coordinator.collect_trajectories(timeout=1.0)
                    current_size = len(coordinator.replay_buffer)

                    delta = current_size - prev_size
                    if delta > 0:
                        pbar.update(delta)
                        pbar.set_postfix({
                            'games': coordinator.total_games,
                            'buffer': current_size
                        })

            if coordinator._shutdown_requested:
                logger.info("Shutdown requested during buffer fill")
                break

            logger.info(f"Buffer filled with {len(coordinator.replay_buffer)} positions from {coordinator.total_games} games")

            # Step 2: Train for steps_per_iteration
            logger.info(f"Training for {args.steps_per_iteration} steps...")

            with tqdm(total=args.steps_per_iteration, desc=f"Iter {iteration+1} - Training", unit="step") as pbar:
                for step in range(args.steps_per_iteration):
                    if coordinator._shutdown_requested:
                        logger.info(f"Shutdown requested at step {step}")
                        break

                    try:
                        # Collect any pending trajectories
                        coordinator.collect_trajectories(timeout=0.001)

                        # Training step
                        metrics = coordinator.learner.train_step()

                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({
                            'loss': f"{metrics['loss']:.4f}",
                            'p_loss': f"{metrics['policy_loss']:.4f}",
                            'v_loss': f"{metrics['value_loss']:.4f}",
                            'buffer': len(coordinator.replay_buffer),
                            'games': coordinator.total_games
                        })

                        # Logging
                        if (step + 1) % coordinator.config.training.log_interval == 0:
                            logger.info(
                                f"Iteration {iteration+1}, Step {step + 1}/{args.steps_per_iteration}: "
                                f"loss={metrics['loss']:.4f}, "
                                f"policy_loss={metrics['policy_loss']:.4f}, "
                                f"value_loss={metrics['value_loss']:.4f}, "
                                f"buffer={len(coordinator.replay_buffer)}, "
                                f"games={coordinator.total_games}"
                            )

                        # Broadcast weights to actors
                        if (step + 1) % 100 == 0:
                            coordinator.broadcast_weights()

                        # Checkpoint
                        if (step + 1) % coordinator.config.training.checkpoint_interval == 0:
                            filters = coordinator.config.network.num_filters
                            blocks = coordinator.config.network.num_blocks
                            global_step = coordinator.learner.global_step
                            path = f"{coordinator.config.checkpoint_dir}/checkpoint_iter{iteration+1}_step{global_step}_f{filters}_b{blocks}.pt"
                            coordinator.learner.save_checkpoint(path)

                    except Exception as e:
                        logger.error(f"Error at training step {step + 1}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        continue

            # Step 3: Save checkpoint after iteration
            filters = coordinator.config.network.num_filters
            blocks = coordinator.config.network.num_blocks
            global_step = coordinator.learner.global_step
            path = f"{coordinator.config.checkpoint_dir}/checkpoint_iter{iteration+1}_final_f{filters}_b{blocks}.pt"
            coordinator.learner.save_checkpoint(path)
            logger.info(f"Saved iteration {iteration+1} checkpoint: {path}")

            # Step 4: Clear replay buffer for next iteration (except on last iteration)
            if iteration < args.iterations - 1:
                logger.info("Clearing replay buffer for next iteration...")
                coordinator.replay_buffer.clear()
                logger.info("Buffer cleared - actors will generate fresh games with updated model")

                # Broadcast updated weights to actors
                coordinator.broadcast_weights()

        logger.info("")
        logger.info("="*60)
        logger.info("ITERATIVE TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Total steps: {coordinator.learner.global_step}")
        logger.info(f"Total games: {coordinator.total_games}")
        logger.info("="*60)

    finally:
        # Stop collection and actors
        coordinator.stop_collection()
        coordinator.stop_actors()


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero chess engine")

    # Hardware profile (applies optimized defaults for different GPUs)
    parser.add_argument("--profile", type=str, default=None,
                        choices=["high", "mid", "low"],
                        help="Hardware profile: high (A100/H100), mid (T4/V100), low (RTX 4060). "
                             "Sets optimized defaults for actors, batch sizes, and timeouts.")

    # Training parameters
    parser.add_argument("--steps", type=int, default=100000,
                        help="Number of training steps (total if not using iterations)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of training iterations (enables iterative training)")
    parser.add_argument("--steps-per-iteration", type=int, default=None,
                        help="Steps per iteration (required if --iterations is set)")
    parser.add_argument("--actors", type=int, default=None,
                        help="Number of self-play actors (default: from profile or 4)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Training batch size (default: from profile or 4096)")
    parser.add_argument("--min-buffer", type=int, default=None,
                        help="Minimum replay buffer size before training starts")

    # Network parameters
    parser.add_argument("--filters", type=int, default=None,
                        help="Number of filters in residual tower (default: from profile or 192)")
    parser.add_argument("--blocks", type=int, default=None,
                        help="Number of residual blocks (default: from profile or 15)")

    # MCTS parameters
    parser.add_argument("--simulations", type=int, default=None,
                        help="MCTS simulations per move (default: 800)")
    parser.add_argument("--mcts-backend", type=str, default=None,
                        choices=["python", "cython", "cpp", "auto"],
                        help="MCTS backend (auto=best available)")

    # Inference mode
    parser.add_argument("--batched-inference", action="store_true",
                        help="Use centralized GPU inference server (recommended for multiple actors)")
    parser.add_argument("--inference-batch-size", type=int, default=None,
                        help="Inference server batch size (default: from profile)")
    parser.add_argument("--inference-timeout", type=float, default=None,
                        help="Inference server batch timeout in seconds (default: from profile)")

    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for logs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

    # Mixed precision
    parser.add_argument("--no-amp-training", action="store_true",
                        help="Disable mixed precision (FP16) for training")
    parser.add_argument("--no-amp-inference", action="store_true",
                        help="Disable mixed precision (FP16) for inference")

    # Other
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    # Apply hardware profile defaults
    profile = PROFILES.get(args.profile) if args.profile else None
    if profile:
        print(f"Using hardware profile: {profile.name.upper()}")

    # Resolve settings with priority: CLI args > profile > hardcoded defaults
    num_actors = args.actors or (profile.actors if profile else 4)
    batch_size = args.batch_size or (profile.training_batch_size if profile else 4096)
    min_buffer = args.min_buffer or (profile.min_buffer_size if profile else 10000)
    num_filters = args.filters or (profile.filters if profile else 192)
    num_blocks = args.blocks or (profile.blocks if profile else 15)
    simulations = args.simulations or (profile.simulations if profile else 800)
    inference_batch_size = args.inference_batch_size or (profile.inference_batch_size if profile else 512)
    inference_timeout = args.inference_timeout or (profile.inference_timeout if profile else 0.02)

    # Auto-detect best MCTS backend if not specified
    if args.mcts_backend == "auto" or args.mcts_backend is None:
        if profile and profile.mcts_backend:
            backend_str = profile.mcts_backend
        else:
            best_backend = get_best_backend()
            backend_str = best_backend.value
        print(f"Auto-detected MCTS backend: {backend_str}")
    else:
        backend_str = args.mcts_backend

    mcts_backend = MCTSBackend(backend_str)

    # Validate iterative training arguments
    if args.iterations is not None:
        if args.steps_per_iteration is None:
            parser.error("--steps-per-iteration is required when --iterations is set")
        total_steps = args.iterations * args.steps_per_iteration
        logger_temp = logging.getLogger(__name__)
        logger_temp.info(f"Iterative training enabled: {args.iterations} iterations × {args.steps_per_iteration} steps = {total_steps} total steps")
    elif args.steps_per_iteration is not None:
        parser.error("--iterations is required when --steps-per-iteration is set")

    # Setup logging
    setup_logging(args.log_dir, args.verbose)
    logger = logging.getLogger(__name__)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Check MCTS backend availability
    available_backends = get_available_backends()
    if mcts_backend not in available_backends:
        logger.error(f"MCTS backend '{backend_str}' not available.")
        logger.error(f"Available backends: {[b.value for b in available_backends]}")
        logger.error("Build the backend first. See README.md for instructions.")
        sys.exit(1)

    # Create configuration
    config = AlphaZeroConfig(
        mcts=MCTSConfig(num_simulations=simulations, backend=mcts_backend),
        network=NetworkConfig(num_filters=num_filters, num_blocks=num_blocks),
        training=TrainingConfig(
            batch_size=batch_size,
            log_interval=1000,  # Log every 1000 steps
            use_amp=not args.no_amp_training,
            use_amp_inference=not args.no_amp_inference
        ),
        replay_buffer=ReplayBufferConfig(min_size_to_train=min_buffer),
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    config.selfplay.num_actors = num_actors

    # Store inference settings for batched coordinator
    config.inference_batch_size = inference_batch_size
    config.inference_timeout = inference_timeout

    # Validate batch size doesn't exceed replay buffer capacity
    if config.training.batch_size > config.replay_buffer.capacity:
        logger.warning(
            f"Batch size ({config.training.batch_size}) exceeds replay buffer capacity "
            f"({config.replay_buffer.capacity}). Adjusting batch size to {config.replay_buffer.capacity}."
        )
        config.training.batch_size = config.replay_buffer.capacity

    # Validate batch size doesn't exceed minimum buffer size
    if config.training.batch_size > config.replay_buffer.min_size_to_train:
        logger.warning(
            f"Batch size ({config.training.batch_size}) exceeds minimum buffer size "
            f"({config.replay_buffer.min_size_to_train}). Consider increasing --min-buffer "
            f"or decreasing --batch-size for better training stability."
        )

    logger.info("AlphaZero Chess Training")
    if profile:
        logger.info(f"Hardware profile: {profile.name.upper()}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Network: {num_blocks} blocks, {num_filters} filters")
    logger.info(f"MCTS: {simulations} simulations, backend={backend_str}")
    logger.info(f"Actors: {num_actors}")
    logger.info(f"Batched inference: {args.batched_inference}")
    if args.batched_inference:
        logger.info(f"Inference batch size: {inference_batch_size}, timeout: {inference_timeout*1000:.0f}ms")
    logger.info(f"Mixed precision training: {config.training.use_amp}")
    logger.info(f"Mixed precision inference: {config.training.use_amp_inference}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Replay buffer capacity: {config.replay_buffer.capacity}")
    logger.info(f"Min buffer size: {min_buffer}")

    # Log iterative training info
    if args.iterations is not None:
        logger.info(f"Iterative training: {args.iterations} iterations × {args.steps_per_iteration} steps")
        logger.info(f"Total steps: {args.iterations * args.steps_per_iteration}")
    else:
        logger.info(f"Continuous training: {args.steps} steps")

    # Create network
    network = AlphaZeroNetwork(
        num_filters=num_filters,
        num_blocks=num_blocks
    )
    logger.info(f"Network parameters: {count_parameters(network):,}")

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        state = torch.load(args.resume, map_location=args.device)
        network.load_state_dict(state['network_state_dict'])

    # Create coordinator and run training
    if args.batched_inference:
        logger.info("Using batched GPU inference mode")
        coordinator = BatchedSelfPlayCoordinator(network, config)
    else:
        coordinator = SelfPlayCoordinator(network, config)

    try:
        if args.iterations is not None:
            # Iterative training mode
            run_iterative_training(coordinator, args, logger, num_actors)
        else:
            # Continuous training mode
            coordinator.run(num_steps=args.steps, num_actors=num_actors)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        # Save final checkpoint with architecture in filename
        final_path = f"{args.checkpoint_dir}/checkpoint_final_f{num_filters}_b{num_blocks}.pt"
        coordinator.learner.save_checkpoint(final_path)
        logger.info(f"Saved final checkpoint to {final_path}")

    logger.info("Training complete!")
    stats = coordinator.get_stats()
    logger.info(f"Total games: {stats['total_games']}")
    logger.info(f"Total positions: {stats['total_positions']}")
    logger.info(f"------ End of training ------")



if __name__ == "__main__":
    main()
