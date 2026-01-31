"""Single self-play game execution.

Plays a complete game using MCTS and collects training data.
"""

import numpy as np
from typing import Tuple, Optional
import logging

from ..chess_env import GameState
from ..mcts import MCTSBase, create_mcts
from ..mcts.python.parallel import ParallelMCTS
from ..mcts.evaluator import Evaluator
from ..training.trajectory import Trajectory
from ..config import MCTSConfig, SelfPlayConfig


logger = logging.getLogger(__name__)


class SelfPlayGame:
    """Executes a single self-play game."""

    def __init__(
        self,
        mcts: MCTSBase,
        evaluator: Evaluator,
        config: Optional[SelfPlayConfig] = None
    ):
        """Initialize self-play game.

        Args:
            mcts: MCTS search instance
            evaluator: Neural network evaluator
            config: Self-play configuration
        """
        self.mcts = mcts
        self.evaluator = evaluator
        self.config = config or SelfPlayConfig()

    def play(self) -> Tuple[Trajectory, str]:
        """Play a complete self-play game.

        Returns:
            Tuple of (trajectory, result_string)
        """
        trajectory = Trajectory()
        state = GameState()
        move_number = 0
        resign_counter = 0

        while not state.is_terminal() and move_number < self.config.max_moves:
            # Run MCTS search with batched leaf evaluation if supported
            if isinstance(self.mcts, ParallelMCTS) and hasattr(self.evaluator, 'evaluate_batch'):
                # Use batched search for better GPU utilization
                batch_size = getattr(self.mcts.config, 'batch_size', 16)
                policy, root, stats = self.mcts.search_with_batching(
                    state,
                    self.evaluator,
                    move_number=move_number,
                    add_noise=True,
                    batch_size=batch_size
                )
            else:
                # Fall back to regular search
                policy, root, stats = self.mcts.search(
                    state,
                    self.evaluator,
                    move_number=move_number,
                    add_noise=True
                )

            # Get observation and legal mask before applying action
            observation = state.get_observation()
            legal_mask = state.get_legal_actions()

            # Select action from policy
            action = self._select_action(policy, move_number)

            # Check for resignation
            value = root.q_value
            if value < self.config.resign_threshold:
                resign_counter += 1
                if resign_counter >= self.config.resign_check_moves:
                    # Resign - current player loses
                    result = -1.0 if state.turn else 1.0  # From white's perspective
                    trajectory.set_result(result)
                    result_str = "0-1" if state.turn else "1-0"
                    return trajectory, f"{result_str} (resignation)"
            else:
                resign_counter = 0

            # Store state in trajectory
            player = 0 if state.turn else 1  # 0=white, 1=black
            trajectory.add_state(
                observation=observation,
                legal_mask=legal_mask,
                policy=policy,
                action=action,
                player=player
            )

            # Apply action
            state = state.apply_action(action)
            move_number += 1

        # Game ended
        if state.is_terminal():
            game_result = state.get_result()
            if game_result is None:
                # Terminal but no outcome (edge case)
                result = 0.0
                result_str = "1/2-1/2 (no outcome)"
            else:
                result = game_result.value_for_white
                result_str = self._result_to_string(game_result)
        else:
            # Max moves reached - draw
            result = 0.0
            result_str = "1/2-1/2 (max moves)"

        trajectory.set_result(result)
        return trajectory, result_str

    def _select_action(self, policy: np.ndarray, move_number: int) -> int:
        """Select action from policy distribution.

        Args:
            policy: Probability distribution over actions
            move_number: Current move number

        Returns:
            Selected action index
        """
        temperature = self.mcts.get_temperature(move_number)

        if temperature <= 0.01:
            # Greedy selection
            return int(np.argmax(policy))
        else:
            # Sample from distribution
            # Ensure policy sums to 1 and handle numerical issues
            policy = policy.astype(np.float64)
            policy = np.maximum(policy, 0)  # Ensure non-negative
            total = np.sum(policy)
            if total > 0:
                policy = policy / total
            else:
                # Fallback to uniform over non-zero entries
                policy = (policy > 0).astype(np.float64)
                policy = policy / np.sum(policy)
            return int(np.random.choice(len(policy), p=policy))

    def _result_to_string(self, result) -> str:
        """Convert game result to string."""
        if result.winner is True:
            return f"1-0 ({result.termination})"
        elif result.winner is False:
            return f"0-1 ({result.termination})"
        else:
            return f"1/2-1/2 ({result.termination})"


def play_game(
    evaluator: Evaluator,
    mcts_config: Optional[MCTSConfig] = None,
    selfplay_config: Optional[SelfPlayConfig] = None
) -> Tuple[Trajectory, str]:
    """Convenience function to play a single self-play game.

    Args:
        evaluator: Neural network evaluator
        mcts_config: MCTS configuration
        selfplay_config: Self-play configuration

    Returns:
        Tuple of (trajectory, result_string)
    """
    mcts = create_mcts(config=mcts_config)
    game = SelfPlayGame(mcts, evaluator, selfplay_config)
    return game.play()
