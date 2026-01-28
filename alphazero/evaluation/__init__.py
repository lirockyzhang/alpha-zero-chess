"""Evaluation module for AlphaZero."""

from .arena import (
    Arena,
    MatchResult,
    MatchStats,
    MCTSPlayer,
    RandomPlayer,
    evaluate_against_random,
)
from .stockfish import (
    StockfishPlayer,
    StockfishEvaluator,
    EloEstimate,
    quick_elo_test,
)

__all__ = [
    "Arena",
    "MatchResult",
    "MatchStats",
    "MCTSPlayer",
    "RandomPlayer",
    "evaluate_against_random",
    "StockfishPlayer",
    "StockfishEvaluator",
    "EloEstimate",
    "quick_elo_test",
]
