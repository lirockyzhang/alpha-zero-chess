"""Stockfish integration for Elo estimation.

Uses Stockfish at various skill levels to estimate model strength.
"""

import chess
import chess.engine
import numpy as np
from typing import Optional, Tuple, List, Protocol
from dataclasses import dataclass
import logging
from pathlib import Path

from ..chess_env import GameState
from .arena import Arena, MatchStats, Player


logger = logging.getLogger(__name__)


class StockfishPlayer:
    """Player that uses Stockfish engine."""

    def __init__(
        self,
        stockfish_path: str,
        elo: int = 1500,
        time_limit: float = 0.1,
        name: Optional[str] = None
    ):
        """Initialize Stockfish player.

        Args:
            stockfish_path: Path to Stockfish executable
            elo: Target Elo rating (uses UCI_LimitStrength)
            time_limit: Time limit per move in seconds
            name: Player name
        """
        self.stockfish_path = stockfish_path
        self.elo = elo
        self.time_limit = time_limit
        self.name = name or f"Stockfish-{elo}"

        self._engine = None

    def _get_engine(self) -> chess.engine.SimpleEngine:
        """Get or create engine instance."""
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

            # Configure skill level
            self._engine.configure({
                "UCI_LimitStrength": True,
                "UCI_Elo": self.elo
            })

        return self._engine

    def select_action(self, state: GameState) -> int:
        """Select action using Stockfish."""
        engine = self._get_engine()
        board = state.board

        # Get best move from Stockfish
        result = engine.play(
            board,
            chess.engine.Limit(time=self.time_limit)
        )

        # Convert to action index
        return state.move_to_action(result.move)

    def close(self):
        """Close the engine."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __del__(self):
        self.close()


@dataclass
class EloEstimate:
    """Estimated Elo rating with confidence interval."""
    elo: float
    lower: float
    upper: float
    confidence: float = 0.95

    def __str__(self):
        return f"{self.elo:.0f} ({self.lower:.0f} - {self.upper:.0f})"


class StockfishEvaluator:
    """Evaluates model strength against Stockfish at various levels."""

    # Stockfish Elo levels for calibration
    ELO_LEVELS = [800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]

    def __init__(
        self,
        stockfish_path: str,
        time_limit: float = 0.1
    ):
        """Initialize evaluator.

        Args:
            stockfish_path: Path to Stockfish executable
            time_limit: Time limit per move
        """
        self.stockfish_path = stockfish_path
        self.time_limit = time_limit
        self.arena = Arena()

    def evaluate_at_level(
        self,
        player: Player,
        elo: int,
        num_games: int = 20
    ) -> MatchStats:
        """Evaluate player against Stockfish at a specific Elo.

        Args:
            player: Player to evaluate
            elo: Stockfish Elo level
            num_games: Number of games to play

        Returns:
            Match statistics
        """
        stockfish = StockfishPlayer(
            self.stockfish_path,
            elo=elo,
            time_limit=self.time_limit
        )

        try:
            stats, _ = self.arena.play_matches(
                player, stockfish, num_games,
                alternate_colors=True
            )
            return stats
        finally:
            stockfish.close()

    def estimate_elo(
        self,
        player: Player,
        num_games_per_level: int = 10,
        levels: Optional[List[int]] = None
    ) -> EloEstimate:
        """Estimate player's Elo by playing against multiple levels.

        Uses binary search to find the level where player scores ~50%.

        Args:
            player: Player to evaluate
            num_games_per_level: Games per Stockfish level
            levels: Elo levels to test (default: ELO_LEVELS)

        Returns:
            Estimated Elo with confidence interval
        """
        levels = levels or self.ELO_LEVELS
        results = {}

        for elo in levels:
            logger.info(f"Testing against Stockfish {elo}...")
            stats = self.evaluate_at_level(player, elo, num_games_per_level)
            results[elo] = stats.score
            logger.info(f"Score vs Stockfish {elo}: {stats.score:.1%}")

            # Early termination if clearly stronger/weaker
            if stats.score > 0.9 and elo < levels[-1]:
                continue
            if stats.score < 0.1 and elo > levels[0]:
                break

        # Estimate Elo from results
        return self._estimate_from_results(results)

    def _estimate_from_results(self, results: dict) -> EloEstimate:
        """Estimate Elo from score results at various levels."""
        if not results:
            return EloEstimate(elo=1000, lower=800, upper=1200)

        # Find levels where score crosses 50%
        levels = sorted(results.keys())
        scores = [results[l] for l in levels]

        # Simple linear interpolation to find 50% crossing
        estimated_elo = levels[0]

        for i in range(len(levels) - 1):
            if scores[i] >= 0.5 >= scores[i + 1]:
                # Interpolate
                t = (0.5 - scores[i + 1]) / (scores[i] - scores[i + 1])
                estimated_elo = levels[i + 1] + t * (levels[i] - levels[i + 1])
                break
            elif scores[i] < 0.5:
                estimated_elo = levels[i]
                break
        else:
            if scores[-1] > 0.5:
                estimated_elo = levels[-1] + 200  # Stronger than highest level

        # Confidence interval (rough estimate)
        margin = 100  # +/- 100 Elo

        return EloEstimate(
            elo=estimated_elo,
            lower=estimated_elo - margin,
            upper=estimated_elo + margin
        )


def quick_elo_test(
    player: Player,
    stockfish_path: str,
    elo: int = 1500,
    num_games: int = 10
) -> MatchStats:
    """Quick Elo test against a single Stockfish level.

    Args:
        player: Player to test
        stockfish_path: Path to Stockfish
        elo: Stockfish Elo level
        num_games: Number of games

    Returns:
        Match statistics
    """
    evaluator = StockfishEvaluator(stockfish_path)
    return evaluator.evaluate_at_level(player, elo, num_games)
