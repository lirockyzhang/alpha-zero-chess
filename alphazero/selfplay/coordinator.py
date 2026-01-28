"""Multi-process self-play coordinator.

Orchestrates multiple actor processes for parallel self-play.
"""

import torch
from multiprocessing import Process, Queue
from typing import Optional, List
import logging
import time
import threading

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
        """Stop all actor processes."""
        for actor in self.actors:
            actor.terminate()
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
                except:
                    break
            queue.put(weights)

        logger.debug("Broadcast weights to actors")

    def collect_trajectories(self, timeout: float = 0.1) -> int:
        """Collect trajectories from actors.

        Args:
            timeout: How long to wait for trajectories

        Returns:
            Number of trajectories collected
        """
        collected = 0

        while True:
            try:
                trajectory = self.trajectory_queue.get(timeout=timeout)
                self.replay_buffer.add_trajectory(trajectory)
                self.total_games += 1
                self.total_positions += len(trajectory)
                collected += 1
            except:
                break

        return collected

    def _collector_loop(self) -> None:
        """Background thread that collects trajectories."""
        while self._running:
            self.collect_trajectories(timeout=0.1)

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
            self._collector_thread.join(timeout=1)

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
        
        last_logged = -1
        while len(self.replay_buffer) < min_size:
            self.collect_trajectories(timeout=1.0)
            if len(self.replay_buffer) - last_logged >= 1:
                logger.info(
                    f"Replay buffer: {len(self.replay_buffer)}/{min_size} "
                    f"({self.total_games} games)"
                )
                last_logged = len(self.replay_buffer)

        logger.info("Starting training...")

        for step in range(num_steps):
            # Collect any pending trajectories
            self.collect_trajectories(timeout=0.001)

            # Training step
            metrics = self.learner.train_step()

            # Logging
            if (step + 1) % self.config.training.log_interval == 0:
                logger.info(
                    f"Step {step + 1}/{num_steps}: "
                    f"loss={metrics['loss']:.4f}, "
                    f"buffer={len(self.replay_buffer)}, "
                    f"games={self.total_games}"
                )

            # Broadcast weights to actors
            if (step + 1) % weight_update_interval == 0:
                self.broadcast_weights()

            # Checkpoint
            if (step + 1) % checkpoint_interval == 0:
                path = f"{self.config.checkpoint_dir}/checkpoint_{step + 1}.pt"
                self.learner.save_checkpoint(path)

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
        try:
            self.start_actors(num_actors)
            self.train(num_steps)
        finally:
            self.stop_actors()

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
