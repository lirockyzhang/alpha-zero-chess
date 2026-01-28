"""GameState wrapper around python-chess Board.

Provides an immutable interface for MCTS and self-play.
"""

import chess
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .encoding import encode_board, NUM_HISTORY_STEPS
from .moves import get_encoder, MoveEncoder


@dataclass(frozen=True)
class GameResult:
    """Result of a completed game."""
    winner: Optional[bool]  # True=White, False=Black, None=Draw
    termination: str  # "checkmate", "stalemate", "insufficient", "fifty_move", "repetition", "max_moves"

    @property
    def value_for_white(self) -> float:
        """Get result value from white's perspective."""
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner else -1.0


class GameState:
    """Immutable game state wrapper around python-chess Board.

    This class provides the interface expected by MCTS and self-play:
    - Immutable: apply_action returns a new GameState
    - Observation: get_observation returns the neural network input
    - Legal actions: get_legal_actions returns a mask over the action space
    """

    def __init__(
        self,
        board: Optional[chess.Board] = None,
        history: Optional[List[chess.Board]] = None,
        move_encoder: Optional[MoveEncoder] = None,
        flip_for_black: bool = True
    ):
        """Initialize a game state.

        Args:
            board: Chess board (default: starting position)
            history: List of previous board states
            move_encoder: Move encoder instance
            flip_for_black: Whether to flip observations for black
        """
        self._board = board.copy() if board else chess.Board()
        self._history = list(history) if history else []
        self._move_encoder = move_encoder or get_encoder()
        self._flip_for_black = flip_for_black

        # Cache computed values
        self._observation: Optional[np.ndarray] = None
        self._legal_mask: Optional[np.ndarray] = None

    @property
    def board(self) -> chess.Board:
        """Get a copy of the underlying board."""
        return self._board.copy()

    @property
    def turn(self) -> bool:
        """Current player (True=White, False=Black)."""
        return self._board.turn

    @property
    def fullmove_number(self) -> int:
        """Current full move number."""
        return self._board.fullmove_number

    @property
    def ply(self) -> int:
        """Current ply (half-move) count."""
        return self._board.ply()

    def get_observation(self) -> np.ndarray:
        """Get the neural network input tensor.

        Returns:
            Array of shape (119, 8, 8)
        """
        if self._observation is None:
            self._observation = encode_board(
                self._board,
                self._history,
                self._flip_for_black
            )
        return self._observation

    def get_legal_actions(self) -> np.ndarray:
        """Get a binary mask of legal actions.

        Returns:
            Array of shape (4672,) with 1.0 for legal actions
        """
        if self._legal_mask is None:
            self._legal_mask = self._move_encoder.get_legal_action_mask(self._board)
        return self._legal_mask

    def get_legal_action_indices(self) -> np.ndarray:
        """Get indices of legal actions.

        Returns:
            Array of legal action indices
        """
        mask = self.get_legal_actions()
        return np.where(mask > 0)[0]

    def apply_action(self, action: int) -> 'GameState':
        """Apply an action and return a new game state.

        Args:
            action: Action index in range [0, 4672)

        Returns:
            New GameState after the action
        """
        move = self._move_encoder.decode(action, self._board)

        new_board = self._board.copy()
        new_board.push(move)

        # Update history
        new_history = [self._board.copy()] + self._history[:NUM_HISTORY_STEPS - 1]

        return GameState(
            board=new_board,
            history=new_history,
            move_encoder=self._move_encoder,
            flip_for_black=self._flip_for_black
        )

    def apply_move(self, move: chess.Move) -> 'GameState':
        """Apply a chess.Move and return a new game state.

        Args:
            move: Chess move to apply

        Returns:
            New GameState after the move
        """
        action = self._move_encoder.encode(move, self._board)
        return self.apply_action(action)

    def is_terminal(self) -> bool:
        """Check if the game has ended."""
        return self._board.is_game_over()

    def get_result(self) -> Optional[GameResult]:
        """Get the game result if terminal.

        Returns:
            GameResult if game is over, None otherwise
        """
        if not self.is_terminal():
            return None

        outcome = self._board.outcome()
        if outcome is None:
            return None

        winner = outcome.winner
        termination = outcome.termination.name.lower()

        return GameResult(winner=winner, termination=termination)

    def get_value(self) -> float:
        """Get the game value from current player's perspective.

        Returns:
            1.0 for win, -1.0 for loss, 0.0 for draw or ongoing
        """
        result = self.get_result()
        if result is None:
            return 0.0

        value = result.value_for_white
        if not self.turn:  # Black's turn
            value = -value
        return value

    def action_to_move(self, action: int) -> chess.Move:
        """Convert an action index to a chess.Move."""
        return self._move_encoder.decode(action, self._board)

    def move_to_action(self, move: chess.Move) -> int:
        """Convert a chess.Move to an action index."""
        return self._move_encoder.encode(move, self._board)

    def __str__(self) -> str:
        """String representation of the board."""
        return str(self._board)

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"GameState(fen='{self._board.fen()}')"

    def fen(self) -> str:
        """Get FEN string of current position."""
        return self._board.fen()

    @classmethod
    def from_fen(cls, fen: str, **kwargs) -> 'GameState':
        """Create a GameState from a FEN string."""
        board = chess.Board(fen)
        return cls(board=board, **kwargs)
