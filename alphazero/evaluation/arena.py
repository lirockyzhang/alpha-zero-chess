"""Arena for match play between agents.

Evaluates model strength through head-to-head matches.
"""

import numpy as np
from typing import Tuple, Optional, List, Protocol
from dataclasses import dataclass
import logging

from ..chess_env import GameState
from ..mcts import create_mcts, MCTSBase
from ..mcts.evaluator import Evaluator
from ..config import MCTSConfig


logger = logging.getLogger(__name__)


class Player(Protocol):
    """Protocol for players in the arena."""

    def select_action(self, state: GameState) -> int:
        """Select an action for the given state."""
        ...


@dataclass
class MatchResult:
    """Result of a single match."""
    white_player: str
    black_player: str
    result: float  # 1.0 white wins, -1.0 black wins, 0.0 draw
    num_moves: int
    termination: str

    @property
    def winner(self) -> Optional[str]:
        if self.result > 0:
            return self.white_player
        elif self.result < 0:
            return self.black_player
        return None


@dataclass
class MatchStats:
    """Statistics from a series of matches."""
    player1: str
    player2: str
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.wins / self.total

    @property
    def score(self) -> float:
        """Score from player1's perspective (win=1, draw=0.5, loss=0)."""
        if self.total == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.total

    def elo_difference(self) -> float:
        """Estimate Elo difference from score."""
        score = self.score
        if score <= 0:
            return -400
        if score >= 1:
            return 400
        return -400 * np.log10(1 / score - 1)


class MCTSPlayer:
    """Player that uses MCTS for move selection."""

    def __init__(
        self,
        name: str,
        evaluator: Evaluator,
        mcts_config: Optional[MCTSConfig] = None,
        temperature: float = 0.0
    ):
        """Initialize MCTS player.

        Args:
            name: Player name
            evaluator: Neural network evaluator
            mcts_config: MCTS configuration
            temperature: Temperature for action selection (0 = greedy)
        """
        self.name = name
        self.evaluator = evaluator
        self.mcts = create_mcts(config=mcts_config)
        self.temperature = temperature

    def select_action(self, state: GameState) -> int:
        """Select action using MCTS."""
        policy, _, _ = self.mcts.search(
            state,
            self.evaluator,
            move_number=state.ply,
            add_noise=False
        )

        if self.temperature < 0.01:
            return int(np.argmax(policy))
        else:
            return int(np.random.choice(len(policy), p=policy))


class RandomPlayer:
    """Player that selects random legal moves."""

    def __init__(self, name: str = "Random"):
        self.name = name

    def select_action(self, state: GameState) -> int:
        """Select a random legal action."""
        legal_actions = state.get_legal_action_indices()
        return int(np.random.choice(legal_actions))


class Arena:
    """Arena for playing matches between agents."""

    def __init__(self, max_moves: int = 512):
        """Initialize arena.

        Args:
            max_moves: Maximum moves before declaring draw
        """
        self.max_moves = max_moves

    def play_match(
        self,
        white: Player,
        black: Player
    ) -> MatchResult:
        """Play a single match.

        Args:
            white: White player
            black: Black player

        Returns:
            Match result
        """
        state = GameState()
        move_count = 0

        while not state.is_terminal() and move_count < self.max_moves:
            # Select player
            player = white if state.turn else black

            # Get action
            action = player.select_action(state)

            # Apply action
            state = state.apply_action(action)
            move_count += 1

        # Determine result
        if state.is_terminal():
            game_result = state.get_result()
            result = game_result.value_for_white
            termination = game_result.termination
        else:
            result = 0.0
            termination = "max_moves"

        return MatchResult(
            white_player=white.name,
            black_player=black.name,
            result=result,
            num_moves=move_count,
            termination=termination
        )

    def play_matches(
        self,
        player1: Player,
        player2: Player,
        num_games: int,
        alternate_colors: bool = True
    ) -> Tuple[MatchStats, List[MatchResult]]:
        """Play a series of matches.

        Args:
            player1: First player
            player2: Second player
            num_games: Number of games to play
            alternate_colors: Whether to alternate colors

        Returns:
            Tuple of (stats, results)
        """
        stats = MatchStats(player1=player1.name, player2=player2.name)
        results = []

        for i in range(num_games):
            # Determine colors
            if alternate_colors and i % 2 == 1:
                white, black = player2, player1
                player1_is_white = False
            else:
                white, black = player1, player2
                player1_is_white = True

            # Play match
            result = self.play_match(white, black)
            results.append(result)

            # Update stats from player1's perspective
            if player1_is_white:
                if result.result > 0:
                    stats.wins += 1
                elif result.result < 0:
                    stats.losses += 1
                else:
                    stats.draws += 1
            else:
                if result.result < 0:
                    stats.wins += 1
                elif result.result > 0:
                    stats.losses += 1
                else:
                    stats.draws += 1

            logger.info(
                f"Game {i + 1}/{num_games}: "
                f"{result.white_player} vs {result.black_player} = "
                f"{self._result_str(result.result)} ({result.termination})"
            )

        logger.info(
            f"Final: {player1.name} vs {player2.name}: "
            f"+{stats.wins} ={stats.draws} -{stats.losses} "
            f"(score: {stats.score:.1%}, Elo diff: {stats.elo_difference():+.0f})"
        )

        return stats, results

    def _result_str(self, result: float) -> str:
        if result > 0:
            return "1-0"
        elif result < 0:
            return "0-1"
        return "1/2-1/2"


def evaluate_against_random(
    player: Player,
    num_games: int = 100
) -> MatchStats:
    """Evaluate a player against random play.

    Args:
        player: Player to evaluate
        num_games: Number of games

    Returns:
        Match statistics
    """
    arena = Arena()
    random_player = RandomPlayer()
    stats, _ = arena.play_matches(player, random_player, num_games)
    return stats
