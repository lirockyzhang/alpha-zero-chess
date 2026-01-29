"""Multi-process self-play coordinator.

Orchestrates multiple actor processes for parallel self-play.
Supports two modes:
1. Standard: Each actor has its own CPU model copy
2. Batched GPU: Centralized GPU inference server batches requests from actors
"""

import torch
import signal
import sys
import traceback
from multiprocessing import Process, Queue
from queue import Empty
from typing import Optional, List, Dict
import logging
import time
import threading
from tqdm import tqdm

from ..neural.network import AlphaZeroNetwork
from ..training.replay_buffer import ReplayBuffer
from ..training.learner import Learner
from ..config import AlphaZeroConfig
from .actor import Actor, ActorProcess


logger = logging.getLogger(__name__)


class SelfPlayCoordinator:
    """Coordinates self-play actors and training.

    Architecture:
        - Main process: Learner (GPU training)
        - Worker processes: Actors (CPU self-play)
        - Communication: Queues for trajectories and weights
    """

    def __init__(
        self,
        network: AlphaZeroNetwork,
        config: Optional[AlphaZeroConfig] = None
    ):
        """Initialize coordinator.

        Args:
            network: Neural network (will be copied to actors)
            config: Configuration
        """
        self.network = network
        self.config = config or AlphaZeroConfig()

        # Queues for communication
        self.trajectory_queue: Queue = Queue(maxsize=1000)
        self.weight_queues: List[Queue] = []

        # Actor processes
        self.actors: List[ActorProcess] = []

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.replay_buffer.capacity
        )

        # Learner
        self.learner = Learner(
            network=self.network,
            replay_buffer=self.replay_buffer,
            config=self.config.training,
            device=self.config.device
        )

        # Statistics
        self.total_games = 0
        self.total_positions = 0

        # Control
        self._running = False
        self._shutdown_requested = False
        self._collector_thread = None

    def start_actors(self, num_actors: Optional[int] = None) -> None:
        """Start actor processes.

        Args:
            num_actors: Number of actors (default from config)
        """
        num_actors = num_actors or self.config.selfplay.num_actors

        # Get initial weights
        initial_weights = self.learner.get_network_weights()

        for i in range(num_actors):
            weight_queue = Queue(maxsize=1)
            self.weight_queues.append(weight_queue)

            actor = ActorProcess(
                actor_id=i,
                trajectory_queue=self.trajectory_queue,
                weight_queue=weight_queue,
                config=self.config,
                initial_weights=initial_weights
            )
            actor.start()
            self.actors.append(actor)

        logger.info(f"Started {num_actors} actor processes")

    def stop_actors(self) -> None:
        """Stop all actor processes gracefully."""
        logger.info("Stopping actor processes...")

        for actor in self.actors:
            if actor.is_alive():
                actor.terminate()

        # Give processes time to terminate
        for actor in self.actors:
            actor.join(timeout=2)
            if actor.is_alive():
                logger.warning(f"Actor {actor.actor_id} did not terminate, killing...")
                actor.kill()
                actor.join(timeout=1)

        self.actors.clear()
        self.weight_queues.clear()
        logger.info("Stopped all actors")

    def broadcast_weights(self) -> None:
        """Send current network weights to all actors."""
        weights = self.learner.get_network_weights()

        for queue in self.weight_queues:
            # Clear old weights and add new
            while not queue.empty():
                try:
                    queue.get_nowait()
                except Empty:
                    break
            try:
                queue.put_nowait(weights)
            except:
                pass  # Queue full, skip this update

        logger.debug("Broadcast weights to actors")

    def collect_trajectories(self, timeout: float = 0.1) -> int:
        """Collect trajectories from actors.

        Args:
            timeout: How long to wait for trajectories

        Returns:
            Number of trajectories collected
        """
        collected = 0

        while not self._shutdown_requested:
            try:
                trajectory = self.trajectory_queue.get(timeout=timeout)
                self.replay_buffer.add_trajectory(trajectory)
                self.total_games += 1
                self.total_positions += len(trajectory)
                collected += 1
            except Empty:
                break
            except Exception as e:
                logger.error(f"Error collecting trajectory: {e}")
                break

        return collected

    def _collector_loop(self) -> None:
        """Background thread that collects trajectories."""
        while self._running and not self._shutdown_requested:
            try:
                self.collect_trajectories(timeout=0.1)
            except Exception as e:
                logger.error(f"Error in collector loop: {e}")

    def start_collection(self) -> None:
        """Start background trajectory collection."""
        self._running = True
        self._collector_thread = threading.Thread(
            target=self._collector_loop,
            daemon=True
        )
        self._collector_thread.start()

    def stop_collection(self) -> None:
        """Stop background trajectory collection."""
        self._running = False
        if self._collector_thread:
            self._collector_thread.join(timeout=2)

    def train(
        self,
        num_steps: int,
        weight_update_interval: int = 100,
        checkpoint_interval: Optional[int] = None
    ) -> None:
        """Run training loop.

        Args:
            num_steps: Total training steps
            weight_update_interval: Steps between weight broadcasts
            checkpoint_interval: Steps between checkpoints
        """
        checkpoint_interval = checkpoint_interval or self.config.training.checkpoint_interval

        # Wait for enough data
        min_size = self.config.replay_buffer.min_size_to_train
        logger.info(f"Waiting for {min_size} positions in replay buffer...")

        # Progress bar for replay buffer filling
        with tqdm(total=min_size, desc="Filling replay buffer", unit="pos") as pbar:
            while len(self.replay_buffer) < min_size and not self._shutdown_requested:
                prev_size = len(self.replay_buffer)
                self.collect_trajectories(timeout=1.0)
                current_size = len(self.replay_buffer)

                # Update progress bar
                delta = current_size - prev_size
                if delta > 0:
                    pbar.update(delta)
                    pbar.set_postfix({
                        'games': self.total_games,
                        'buffer': current_size
                    })

        if self._shutdown_requested:
            logger.info("Shutdown requested during warmup")
            return

        logger.info("Starting training...")

        # Progress bar for training
        with tqdm(total=num_steps, desc="Training", unit="step") as pbar:
            for step in range(num_steps):
                if self._shutdown_requested:
                    logger.info(f"Shutdown requested at step {step}")
                    break

                try:
                    # Collect any pending trajectories
                    self.collect_trajectories(timeout=0.001)

                    # Training step
                    metrics = self.learner.train_step()

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'p_loss': f"{metrics['policy_loss']:.4f}",
                        'v_loss': f"{metrics['value_loss']:.4f}",
                        'buffer': len(self.replay_buffer),
                        'games': self.total_games
                    })

                    # Logging (less frequent now that we have progress bar)
                    if (step + 1) % self.config.training.log_interval == 0:
                        logger.info(
                            f"Step {step + 1}/{num_steps}: "
                            f"loss={metrics['loss']:.4f}, "
                            f"policy_loss={metrics['policy_loss']:.4f}, "
                            f"value_loss={metrics['value_loss']:.4f}, "
                            f"buffer={len(self.replay_buffer)}, "
                            f"games={self.total_games}"
                        )

                    # Broadcast weights to actors
                    if (step + 1) % weight_update_interval == 0:
                        self.broadcast_weights()

                    # Checkpoint
                    if (step + 1) % checkpoint_interval == 0:
                        # Include network structure in filename
                        filters = self.config.network.num_filters
                        blocks = self.config.network.num_blocks
                        path = f"{self.config.checkpoint_dir}/checkpoint_{step + 1}_f{filters}_b{blocks}.pt"
                        self.learner.save_checkpoint(path)

                except Exception as e:
                    logger.error(f"Error at training step {step + 1}: {e}")
                    logger.error(traceback.format_exc())
                    # Continue training despite errors
                    continue

    def run(
        self,
        num_steps: int,
        num_actors: Optional[int] = None
    ) -> None:
        """Run complete training pipeline.

        Args:
            num_steps: Total training steps
            num_actors: Number of actor processes
        """
        # Setup signal handlers for graceful shutdown
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_requested = True

        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            self.start_actors(num_actors)
            self.train(num_steps)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down...")
            self._shutdown_requested = True

        except Exception as e:
            logger.error(f"Fatal error in training: {e}")
            logger.error(traceback.format_exc())

        finally:
            # Restore original signal handlers
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

            # Clean up
            self.stop_actors()
            logger.info("Training pipeline shut down")

    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            'total_games': self.total_games,
            'total_positions': self.total_positions,
            'replay_buffer_size': len(self.replay_buffer),
            'training_step': self.learner.global_step,
        }


def run_training(
    config: Optional[AlphaZeroConfig] = None,
    num_steps: int = 100000,
    num_actors: Optional[int] = None
) -> SelfPlayCoordinator:
    """Convenience function to run training.

    Args:
        config: Configuration
        num_steps: Training steps
        num_actors: Number of actors

    Returns:
        Coordinator instance
    """
    config = config or AlphaZeroConfig()

    # Create network
    network = AlphaZeroNetwork(
        num_filters=config.network.num_filters,
        num_blocks=config.network.num_blocks
    )

    # Create and run coordinator
    coordinator = SelfPlayCoordinator(network, config)
    coordinator.run(num_steps, num_actors)

    return coordinator


class BatchedSelfPlayCoordinator:
    """Coordinator with centralized GPU inference server.

    This coordinator uses a separate process for GPU inference that batches
    requests from multiple actors. This is more efficient than having each
    actor do its own CPU inference.

    Architecture:
        Actor 1 ──┐                              ┌── Response Queue 1
        Actor 2 ──┼── Request Queue ── GPU ──────┼── Response Queue 2
        Actor 3 ──┤    Server                    ├── Response Queue 3
        Actor 4 ──┘                              └── Response Queue 4
                                                          │
                                                          ▼
                                                   Trajectory Queue
                                                          │
                                                          ▼
                                                      Learner
    """

    def __init__(
        self,
        network: AlphaZeroNetwork,
        config: Optional[AlphaZeroConfig] = None
    ):
        """Initialize batched coordinator.

        Args:
            network: Neural network
            config: Configuration
        """
        self.network = network
        self.config = config or AlphaZeroConfig()

        # Queues for inference server
        self.inference_request_queue: Queue = Queue(maxsize=1000)
        self.inference_response_queues: Dict[int, Queue] = {}

        # Queue for weight updates to inference server
        self.inference_weight_queue: Queue = Queue(maxsize=1)

        # Trajectory queue from actors
        self.trajectory_queue: Queue = Queue(maxsize=1000)

        # Processes
        self.inference_server: Optional[Process] = None
        self.actors: List[Process] = []

        # Replay buffer and learner
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.replay_buffer.capacity
        )
        self.learner = Learner(
            network=self.network,
            replay_buffer=self.replay_buffer,
            config=self.config.training,
            device=self.config.device
        )

        # Statistics
        self.total_games = 0
        self.total_positions = 0

        # Control
        self._shutdown_requested = False

    def start_inference_server(self, num_actors: int) -> None:
        """Start the centralized GPU inference server.

        Args:
            num_actors: Number of actors (needed to pre-create response queues)
        """
        from .inference_server import InferenceServer

        # IMPORTANT: Create response queues BEFORE starting the server
        # because the server process gets a pickled copy of the dict
        for i in range(num_actors):
            self.inference_response_queues[i] = Queue(maxsize=100)

        initial_weights = self.learner.get_network_weights()

        self.inference_server = InferenceServer(
            request_queue=self.inference_request_queue,
            response_queues=self.inference_response_queues,
            network_class=AlphaZeroNetwork,
            network_kwargs={
                'num_filters': self.config.network.num_filters,
                'num_blocks': self.config.network.num_blocks,
            },
            initial_weights=initial_weights,
            device=self.config.device,
            batch_size=32,
            batch_timeout=0.002,
            weight_queue=self.inference_weight_queue,
            use_amp=self.config.training.use_amp_inference,
        )
        self.inference_server.start()
        logger.info(f"Started inference server with {num_actors} response queues")

    def start_actors(self, num_actors: Optional[int] = None) -> None:
        """Start actor processes that use batched inference.

        Args:
            num_actors: Number of actors
        """
        from .batched_actor import BatchedActorProcess

        num_actors = num_actors or self.config.selfplay.num_actors

        for i in range(num_actors):
            # Use the pre-created response queue
            response_queue = self.inference_response_queues[i]

            actor = BatchedActorProcess(
                actor_id=i,
                trajectory_queue=self.trajectory_queue,
                inference_request_queue=self.inference_request_queue,
                inference_response_queue=response_queue,
                config=self.config,
            )
            actor.start()
            self.actors.append(actor)

        logger.info(f"Started {num_actors} batched actor processes")

    def stop_all(self) -> None:
        """Stop all processes."""
        logger.info("Stopping all processes...")

        # Stop actors first
        for actor in self.actors:
            if actor.is_alive():
                actor.terminate()

        for actor in self.actors:
            actor.join(timeout=2)
            if actor.is_alive():
                actor.kill()

        self.actors.clear()

        # Stop inference server
        if self.inference_server and self.inference_server.is_alive():
            self.inference_server.terminate()
            self.inference_server.join(timeout=2)
            if self.inference_server.is_alive():
                self.inference_server.kill()

        self.inference_response_queues.clear()
        logger.info("Stopped all processes")

    def broadcast_weights(self) -> None:
        """Send updated weights to inference server."""
        weights = self.learner.get_network_weights()

        # Clear old weights
        while not self.inference_weight_queue.empty():
            try:
                self.inference_weight_queue.get_nowait()
            except Empty:
                break

        try:
            self.inference_weight_queue.put_nowait(weights)
        except:
            pass

    def collect_trajectories(self, timeout: float = 0.1) -> int:
        """Collect trajectories from actors."""
        collected = 0

        while not self._shutdown_requested:
            try:
                trajectory = self.trajectory_queue.get(timeout=timeout)
                self.replay_buffer.add_trajectory(trajectory)
                self.total_games += 1
                self.total_positions += len(trajectory)
                collected += 1
            except Empty:
                break
            except Exception as e:
                logger.error(f"Error collecting trajectory: {e}")
                break

        return collected

    def train(
        self,
        num_steps: int,
        weight_update_interval: int = 100,
        checkpoint_interval: Optional[int] = None
    ) -> None:
        """Run training loop."""
        checkpoint_interval = checkpoint_interval or self.config.training.checkpoint_interval

        # Wait for enough data
        min_size = self.config.replay_buffer.min_size_to_train
        logger.info(f"Waiting for {min_size} positions in replay buffer...")

        # Progress bar for replay buffer filling
        with tqdm(total=min_size, desc="Filling replay buffer", unit="pos") as pbar:
            while len(self.replay_buffer) < min_size and not self._shutdown_requested:
                prev_size = len(self.replay_buffer)
                self.collect_trajectories(timeout=1.0)
                current_size = len(self.replay_buffer)

                # Update progress bar
                delta = current_size - prev_size
                if delta > 0:
                    pbar.update(delta)
                    pbar.set_postfix({
                        'games': self.total_games,
                        'buffer': current_size
                    })

        if self._shutdown_requested:
            logger.info("Shutdown requested during warmup")
            return

        logger.info("Starting training...")

        # Progress bar for training
        with tqdm(total=num_steps, desc="Training", unit="step") as pbar:
            for step in range(num_steps):
                if self._shutdown_requested:
                    logger.info(f"Shutdown requested at step {step}")
                    break

                try:
                    self.collect_trajectories(timeout=0.001)
                    metrics = self.learner.train_step()

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'p_loss': f"{metrics['policy_loss']:.4f}",
                        'v_loss': f"{metrics['value_loss']:.4f}",
                        'buffer': len(self.replay_buffer),
                        'games': self.total_games
                    })

                    if (step + 1) % self.config.training.log_interval == 0:
                        logger.info(
                            f"Step {step + 1}/{num_steps}: "
                            f"loss={metrics['loss']:.4f}, "
                            f"policy_loss={metrics['policy_loss']:.4f}, "
                            f"value_loss={metrics['value_loss']:.4f}, "
                            f"buffer={len(self.replay_buffer)}, "
                            f"games={self.total_games}"
                        )

                    if (step + 1) % weight_update_interval == 0:
                        self.broadcast_weights()

                    if (step + 1) % checkpoint_interval == 0:
                        # Include network structure in filename
                        filters = self.config.network.num_filters
                        blocks = self.config.network.num_blocks
                        path = f"{self.config.checkpoint_dir}/checkpoint_{step + 1}_f{filters}_b{blocks}.pt"
                        self.learner.save_checkpoint(path)

                except Exception as e:
                    logger.error(f"Error at training step {step + 1}: {e}")
                    logger.error(traceback.format_exc())
                    continue

    def run(
        self,
        num_steps: int,
        num_actors: Optional[int] = None
    ) -> None:
        """Run complete training pipeline with batched inference."""
        num_actors = num_actors or self.config.selfplay.num_actors

        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_requested = True

        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            # Start inference server first (creates response queues)
            self.start_inference_server(num_actors)
            time.sleep(1)  # Give server time to initialize

            # Then start actors (uses pre-created response queues)
            self.start_actors(num_actors)
            self.train(num_steps)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down...")
            self._shutdown_requested = True

        except Exception as e:
            logger.error(f"Fatal error in training: {e}")
            logger.error(traceback.format_exc())

        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            self.stop_all()
            logger.info("Training pipeline shut down")

    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            'total_games': self.total_games,
            'total_positions': self.total_positions,
            'replay_buffer_size': len(self.replay_buffer),
            'training_step': self.learner.global_step,
        }

