"""Standalone AlphaZero components for the web interface.

This module provides self-contained implementations of:
- GameState: Game state wrapper around python-chess Board
- MCTSConfig: Configuration dataclass

Move decoding uses alphazero_cpp.index_to_move() from the C++ backend
(single source of truth for move encoding/decoding).

The neural network is imported from alphazero-cpp/scripts/network.py
(single source of truth) — see web/app.py for the import.
"""

import chess
from typing import Optional, List
from dataclasses import dataclass


# =============================================================================
# Game State
# =============================================================================

@dataclass(frozen=True)
class GameResult:
    """Result of a completed game."""
    winner: Optional[bool]  # True=White, False=Black, None=Draw
    termination: str


class GameState:
    """Game state wrapper around python-chess Board."""

    def __init__(
        self,
        board: Optional[chess.Board] = None,
        position_history: Optional[List[str]] = None
    ):
        self._board = board.copy() if board else chess.Board()
        self._position_history: List[str] = list(position_history) if position_history else []

    @property
    def board(self) -> chess.Board:
        """Get a copy of the underlying board."""
        return self._board.copy()

    def apply_move(self, move: chess.Move) -> 'GameState':
        """Apply a chess.Move and return a new game state with updated history."""
        current_fen = self._board.fen()
        new_history = self._position_history + [current_fen]
        if len(new_history) > 8:
            new_history = new_history[-8:]
        new_board = self._board.copy()
        new_board.push(move)
        return GameState(board=new_board, position_history=new_history)

    def is_terminal(self) -> bool:
        """Check if the game has ended."""
        return self._board.is_game_over()

    def get_result(self) -> Optional[GameResult]:
        """Get the game result if terminal."""
        if not self.is_terminal():
            return None

        outcome = self._board.outcome()
        if outcome is None:
            return None

        return GameResult(
            winner=outcome.winner,
            termination=outcome.termination.name.lower()
        )

    def fen(self) -> str:
        """Get FEN string of current position."""
        return self._board.fen()

    @property
    def position_history(self) -> List[str]:
        """Get list of previous position FENs (up to 8, oldest first)."""
        return list(self._position_history)

    def __str__(self) -> str:
        return str(self._board)

    def __repr__(self) -> str:
        return f"GameState(fen='{self._board.fen()}')"


# =============================================================================
# MCTS Config (minimal)
# =============================================================================

@dataclass
class MCTSConfig:
    """MCTS configuration."""
    num_simulations: int = 800
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
