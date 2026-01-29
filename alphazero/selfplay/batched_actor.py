"""Batched actor process that uses centralized GPU inference.

This actor sends inference requests to a centralized server instead of
running its own neural network. This allows efficient GPU batching
across multiple actors.
"""

import numpy as np
import logging
import traceback
from multiprocessing import Process, Queue
from queue import Empty
from typing import Optional
import time

from ..config import AlphaZeroConfig
from ..chess_env import GameState
from ..mcts import create_mcts
from ..training.trajectory import Trajectory, TrajectoryState
from .inference_server import BatchedEvaluator

logger = logging.getLogger(__name__)


class BatchedActorProcess(Process):
    """Actor process that uses batched GPU inference.

    Instead of running neural network inference locally, this actor
    sends requests to a centralized inference server and waits for
    responses. This allows the GPU to batch requests from multiple
    actors for better efficiency.
    """

    def __init__(
        self,
        actor_id: int,
        trajectory_queue: Queue,
        inference_request_queue: Queue,
        inference_response_queue: Queue,
        config: Optional[AlphaZeroConfig] = None,
    ):
        """Initialize batched actor.

        Args:
            actor_id: Unique identifier for this actor
            trajectory_queue: Queue to send completed trajectories
            inference_request_queue: Queue to send inference requests
            inference_response_queue: Queue to receive inference responses
            config: Configuration
        """
        super().__init__()
        self.actor_id = actor_id
        self.trajectory_queue = trajectory_queue
        self.inference_request_queue = inference_request_queue
        self.inference_response_queue = inference_response_queue
        self.config = config or AlphaZeroConfig()
        self.daemon = True

    def run(self):
        """Main actor loop."""
        # Setup logging for this process
        logging.basicConfig(
            level=logging.INFO,
            format=f"%(asctime)s - Actor{self.actor_id} - %(levelname)s - %(message)s"
        )

        try:
            logger.info(f"Actor {self.actor_id} starting with batched inference")

            # Create evaluator that uses inference server
            evaluator = BatchedEvaluator(
                actor_id=self.actor_id,
                request_queue=self.inference_request_queue,
                response_queue=self.inference_response_queue,
                timeout=10.0
            )

            # Create MCTS (uses CPU for tree operations only)
            mcts = create_mcts(
                backend=self.config.mcts.backend,
                config=self.config.mcts
            )

            games_played = 0

            while True:
                try:
                    trajectory = self._play_game(mcts, evaluator)

                    if trajectory:
                        self.trajectory_queue.put(trajectory, timeout=5.0)
                        games_played += 1

                        if games_played % 10 == 0:
                            logger.info(f"Actor {self.actor_id}: {games_played} games completed")

                except TimeoutError as e:
                    logger.warning(f"Actor {self.actor_id}: Inference timeout - {e}")
                    time.sleep(0.1)
                    continue

                except Exception as e:
                    logger.error(f"Actor {self.actor_id}: Error in game - {e}")
                    logger.error(traceback.format_exc())
                    time.sleep(0.1)
                    continue

        except Exception as e:
            logger.error(f"Actor {self.actor_id}: Fatal error - {e}")
            logger.error(traceback.format_exc())

    def _play_game(self, mcts, evaluator) -> Optional[Trajectory]:
        """Play a single self-play game.

        Args:
            mcts: MCTS search instance
            evaluator: Batched evaluator

        Returns:
            Completed trajectory or None if game failed
        """
        state = GameState()
        trajectory_states = []
        move_number = 0

        while not state.is_terminal() and move_number < self.config.selfplay.max_moves:
            # Run MCTS search
            add_noise = (move_number == 0)  # Only add noise at root of game
            policy, root, stats = mcts.search(
                state=state,
                evaluator=evaluator,
                move_number=move_number,
                add_noise=add_noise
            )

            # Select action
            temperature = mcts.get_temperature(move_number)
            if temperature <= 0.01:
                action = int(np.argmax(policy))
            else:
                action = int(np.random.choice(len(policy), p=policy))

            # Store trajectory state
            trajectory_states.append(TrajectoryState(
                observation=state.get_observation(),
                legal_mask=state.get_legal_actions(),
                policy=policy,
                value=0.0,  # Will be filled in after game ends
                action=action,
                player=0 if state.turn else 1  # True=White(0), False=Black(1)
            ))

            # Apply action
            state = state.apply_action(action)
            move_number += 1

        if not trajectory_states:
            return None

        # Get game result
        game_result = state.get_result()
        if game_result is None:
            # Game ended without a clear outcome (max moves or edge case)
            result = 0.0  # Treat as draw
        else:
            result = game_result.value_for_white

        # Fill in values from game result
        for ts in trajectory_states:
            # Value from perspective of player who made the move
            ts.value = result if ts.player == 0 else -result

        return Trajectory(states=trajectory_states, result=result)


class BatchedActor:
    """Non-process version of batched actor for testing."""

    def __init__(
        self,
        actor_id: int,
        evaluator: BatchedEvaluator,
        config: Optional[AlphaZeroConfig] = None
    ):
        self.actor_id = actor_id
        self.evaluator = evaluator
        self.config = config or AlphaZeroConfig()
        self.mcts = create_mcts(
            backend=self.config.mcts.backend,
            config=self.config.mcts
        )

    def play_game(self) -> Trajectory:
        """Play a single game and return trajectory."""
        state = GameState()
        trajectory_states = []
        move_number = 0

        while not state.is_terminal() and move_number < self.config.selfplay.max_moves:
            add_noise = (move_number == 0)
            policy, root, stats = self.mcts.search(
                state=state,
                evaluator=self.evaluator,
                move_number=move_number,
                add_noise=add_noise
            )

            temperature = self.mcts.get_temperature(move_number)
            if temperature <= 0.01:
                action = int(np.argmax(policy))
            else:
                action = int(np.random.choice(len(policy), p=policy))

            trajectory_states.append(TrajectoryState(
                observation=state.get_observation(),
                legal_mask=state.get_legal_actions(),
                policy=policy,
                value=0.0,
                action=action,
                player=0 if state.turn else 1  # True=White(0), False=Black(1)
            ))

            state = state.apply_action(action)
            move_number += 1

        game_result = state.get_result()
        if game_result is None:
            # Game ended without a clear outcome (max moves or edge case)
            result = 0.0  # Treat as draw
        else:
            result = game_result.value_for_white

        for ts in trajectory_states:
            ts.value = result if ts.player == 0 else -result

        return Trajectory(states=trajectory_states, result=result)
