"""Self-play actor that runs in a separate process.

Generates training data by playing games against itself.
"""

import torch
import numpy as np
from typing import Optional, Callable
from multiprocessing import Process, Queue
import logging
import time

from ..chess_env import GameState
from ..mcts import create_mcts
from ..mcts.evaluator import NetworkEvaluator
from ..neural.network import AlphaZeroNetwork
from ..training.trajectory import Trajectory
from ..config import AlphaZeroConfig, MCTSConfig, SelfPlayConfig
from .game import SelfPlayGame


logger = logging.getLogger(__name__)


class Actor:
    """Self-play actor that generates training data.

    Runs MCTS with neural network evaluation to play games.
    Can run in the main process or as a subprocess.
    """

    def __init__(
        self,
        actor_id: int,
        network: AlphaZeroNetwork,
        config: Optional[AlphaZeroConfig] = None,
        device: str = "cpu"
    ):
        """Initialize actor.

        Args:
            actor_id: Unique identifier for this actor
            network: Neural network for evaluation
            config: Configuration
            device: Device to run inference on (usually CPU for actors)
        """
        self.actor_id = actor_id
        self.network = network.to(device)
        self.network.eval()
        self.config = config or AlphaZeroConfig()
        self.device = device

        # Create MCTS and evaluator
        self.mcts = create_mcts(
            backend=self.config.mcts.backend,
            config=self.config.mcts
        )
        self.evaluator = NetworkEvaluator(self.network, device)

        # Statistics
        self.games_played = 0
        self.total_moves = 0

    def play_game(self) -> Trajectory:
        """Play a single self-play game.

        Returns:
            Game trajectory with training data
        """
        game = SelfPlayGame(
            self.mcts,
            self.evaluator,
            self.config.selfplay
        )
        trajectory, result_str = game.play()

        self.games_played += 1
        self.total_moves += len(trajectory)

        logger.debug(
            f"Actor {self.actor_id}: Game {self.games_played} "
            f"finished with {result_str} ({len(trajectory)} moves)"
        )

        return trajectory

    def play_games(self, num_games: int) -> list:
        """Play multiple self-play games.

        Args:
            num_games: Number of games to play

        Returns:
            List of trajectories
        """
        trajectories = []
        for i in range(num_games):
            trajectory = self.play_game()
            trajectories.append(trajectory)
        return trajectories

    def update_weights(self, weights: dict) -> None:
        """Update network weights.

        Args:
            weights: State dict with new weights
        """
        self.network.load_state_dict(weights)
        self.network.eval()


class ActorProcess(Process):
    """Actor running in a separate process.

    Communicates via queues for trajectories and weight updates.
    """

    def __init__(
        self,
        actor_id: int,
        trajectory_queue: Queue,
        weight_queue: Queue,
        config: AlphaZeroConfig,
        initial_weights: Optional[dict] = None
    ):
        """Initialize actor process.

        Args:
            actor_id: Unique identifier
            trajectory_queue: Queue to send trajectories to
            weight_queue: Queue to receive weight updates from
            config: Configuration
            initial_weights: Initial network weights
        """
        super().__init__()
        self.actor_id = actor_id
        self.trajectory_queue = trajectory_queue
        self.weight_queue = weight_queue
        self.config = config
        self.initial_weights = initial_weights
        self.daemon = True

    def run(self):
        """Main actor loop."""
        # Create network and actor
        network = AlphaZeroNetwork(
            num_filters=self.config.network.num_filters,
            num_blocks=self.config.network.num_blocks
        )

        if self.initial_weights:
            network.load_state_dict(self.initial_weights)

        actor = Actor(
            self.actor_id,
            network,
            self.config,
            device="cpu"
        )

        logger.info(f"Actor {self.actor_id} started")

        while True:
            # Check for weight updates (non-blocking)
            while not self.weight_queue.empty():
                try:
                    weights = self.weight_queue.get_nowait()
                    actor.update_weights(weights)
                    logger.debug(f"Actor {self.actor_id}: Updated weights")
                except:
                    break

            # Play a game
            trajectory = actor.play_game()

            # Send trajectory to learner
            self.trajectory_queue.put(trajectory)


def run_actor_loop(
    actor: Actor,
    trajectory_callback: Callable[[Trajectory], None],
    weight_provider: Optional[Callable[[], Optional[dict]]] = None,
    num_games: Optional[int] = None
) -> None:
    """Run actor loop in current process.

    Args:
        actor: Actor instance
        trajectory_callback: Called with each completed trajectory
        weight_provider: Optional function that returns new weights
        num_games: Number of games to play (None = infinite)
    """
    games = 0
    while num_games is None or games < num_games:
        # Check for weight updates
        if weight_provider:
            weights = weight_provider()
            if weights is not None:
                actor.update_weights(weights)

        # Play game
        trajectory = actor.play_game()
        trajectory_callback(trajectory)
        games += 1
