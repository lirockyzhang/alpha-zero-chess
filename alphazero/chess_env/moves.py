"""Move encoding/decoding for AlphaZero chess.

Maps between chess.Move objects and action indices (0-4671).

Action space (4672 total):
- Queen-like moves: 56 directions × 64 squares = 3584
  - 7 distances × 8 directions (N, NE, E, SE, S, SW, W, NW)
- Knight moves: 8 × 64 = 512
- Underpromotions: 9 × 64 = 576
  - 3 piece types (rook, bishop, knight) × 3 directions × 64 squares
"""

import chess
import numpy as np
from typing import Optional, Tuple


# Direction vectors for queen-like moves (row_delta, col_delta)
# Using chess board coordinates where row 0 = rank 1, row 7 = rank 8
QUEEN_DIRECTIONS = [
    (1, 0),    # N  (increasing rank)
    (1, 1),    # NE
    (0, 1),    # E
    (-1, 1),   # SE
    (-1, 0),   # S  (decreasing rank)
    (-1, -1),  # SW
    (0, -1),   # W
    (1, -1),   # NW
]

# Knight move deltas
KNIGHT_MOVES = [
    (2, 1), (2, -1), (1, 2), (1, -2),
    (-1, 2), (-1, -2), (-2, 1), (-2, -1),
]

# Underpromotion pieces (queen promotion is encoded as regular move)
UNDERPROMOTION_PIECES = [chess.ROOK, chess.BISHOP, chess.KNIGHT]

# Promotion directions: left-capture, forward, right-capture
# For white (moving from rank 7 to rank 8): row increases by 1
PROMOTION_DIRECTIONS = [(1, -1), (1, 0), (1, 1)]


def _square_to_rowcol(square: int) -> Tuple[int, int]:
    """Convert chess square (0-63) to (row, col) where row=rank, col=file."""
    return chess.square_rank(square), chess.square_file(square)


def _rowcol_to_square(row: int, col: int) -> int:
    """Convert (row, col) to chess square."""
    return chess.square(col, row)


class MoveEncoder:
    """Bidirectional mapping between chess.Move and action indices.

    The encoding scheme:
    - Queen-like moves: from_sq * 56 + dir_idx * 7 + (distance - 1)
    - Knight moves: 3584 + from_sq * 8 + move_idx
    - Underpromotions: 4096 + from_sq * 9 + dir_idx * 3 + piece_idx

    For underpromotions, we always use the same 3 directions regardless of color,
    but we flip the board perspective for black pawns.
    """

    def __init__(self):
        self.queen_offset = 0
        self.knight_offset = 3584
        self.underpromo_offset = 4096
        self.num_actions = 4672
        self._build_tables()

    def _build_tables(self):
        """Build bidirectional lookup tables."""
        # encode_table: (from_sq, to_sq, promotion) -> action
        # decode_table: action -> (from_sq, to_sq, promotion)
        self._encode_table = {}
        self._decode_table = {}

        # Queen-like moves
        for from_sq in range(64):
            from_row, from_col = _square_to_rowcol(from_sq)
            for dir_idx, (dr, dc) in enumerate(QUEEN_DIRECTIONS):
                for dist in range(1, 8):
                    to_row = from_row + dr * dist
                    to_col = from_col + dc * dist
                    if 0 <= to_row < 8 and 0 <= to_col < 8:
                        to_sq = _rowcol_to_square(to_row, to_col)
                        action = self.queen_offset + from_sq * 56 + dir_idx * 7 + (dist - 1)
                        self._encode_table[(from_sq, to_sq, None)] = action
                        self._decode_table[action] = (from_sq, to_sq, None)

        # Knight moves
        for from_sq in range(64):
            from_row, from_col = _square_to_rowcol(from_sq)
            for move_idx, (dr, dc) in enumerate(KNIGHT_MOVES):
                to_row = from_row + dr
                to_col = from_col + dc
                if 0 <= to_row < 8 and 0 <= to_col < 8:
                    to_sq = _rowcol_to_square(to_row, to_col)
                    action = self.knight_offset + from_sq * 8 + move_idx
                    self._encode_table[(from_sq, to_sq, None)] = action
                    self._decode_table[action] = (from_sq, to_sq, None)

        # Underpromotions
        # White pawns promote from rank 6 (row 6) moving to rank 7 (row 7)
        # Black pawns promote from rank 1 (row 1) moving to rank 0 (row 0)
        for from_sq in range(64):
            from_row, from_col = _square_to_rowcol(from_sq)

            if from_row == 6:  # White pawn about to promote
                for dir_idx, (dr, dc) in enumerate(PROMOTION_DIRECTIONS):
                    to_row = from_row + dr  # Will be 7
                    to_col = from_col + dc
                    if 0 <= to_col < 8:
                        to_sq = _rowcol_to_square(to_row, to_col)
                        for piece_idx, piece in enumerate(UNDERPROMOTION_PIECES):
                            action = self.underpromo_offset + from_sq * 9 + dir_idx * 3 + piece_idx
                            self._encode_table[(from_sq, to_sq, piece)] = action
                            self._decode_table[action] = (from_sq, to_sq, piece)

            elif from_row == 1:  # Black pawn about to promote
                # Black moves in opposite direction (decreasing rank)
                for dir_idx, (dr, dc) in enumerate(PROMOTION_DIRECTIONS):
                    to_row = from_row - dr  # Flip direction: will be 0
                    to_col = from_col + dc  # Keep file direction same
                    if 0 <= to_col < 8:
                        to_sq = _rowcol_to_square(to_row, to_col)
                        for piece_idx, piece in enumerate(UNDERPROMOTION_PIECES):
                            action = self.underpromo_offset + from_sq * 9 + dir_idx * 3 + piece_idx
                            self._encode_table[(from_sq, to_sq, piece)] = action
                            self._decode_table[action] = (from_sq, to_sq, piece)

    def encode(self, move: chess.Move, board: Optional[chess.Board] = None) -> int:
        """Convert a chess.Move to an action index.

        Args:
            move: The chess move to encode
            board: Optional board for context (not used, kept for API compatibility)

        Returns:
            Action index in range [0, 4672)
        """
        from_sq = move.from_square
        to_sq = move.to_square
        promotion = move.promotion

        # Queen promotion is encoded as a regular queen-like move
        if promotion == chess.QUEEN:
            promotion = None

        key = (from_sq, to_sq, promotion)
        if key in self._encode_table:
            return self._encode_table[key]

        # This shouldn't happen for legal chess moves
        raise ValueError(f"Cannot encode move {move.uci()}: not in encoding table")

    def decode(self, action: int, board: chess.Board) -> chess.Move:
        """Convert an action index to a chess.Move.

        Args:
            action: Action index in range [0, 4672)
            board: Current board state (needed for queen promotion detection)

        Returns:
            The corresponding chess.Move
        """
        if action not in self._decode_table:
            raise ValueError(f"Invalid action index: {action}")

        from_sq, to_sq, promotion = self._decode_table[action]

        # Check if this is a pawn reaching the back rank without explicit promotion
        # (queen promotion is encoded as regular move)
        if promotion is None:
            piece = board.piece_at(from_sq)
            if piece and piece.piece_type == chess.PAWN:
                to_rank = chess.square_rank(to_sq)
                if (piece.color == chess.WHITE and to_rank == 7) or \
                   (piece.color == chess.BLACK and to_rank == 0):
                    promotion = chess.QUEEN

        return chess.Move(from_sq, to_sq, promotion=promotion)

    def get_legal_action_mask(self, board: chess.Board) -> np.ndarray:
        """Get a binary mask of legal actions for the current position.

        Args:
            board: Current board state

        Returns:
            Binary array of shape (4672,) where 1 indicates legal action
        """
        mask = np.zeros(self.num_actions, dtype=np.float32)
        for move in board.legal_moves:
            try:
                action = self.encode(move, board)
                mask[action] = 1.0
            except ValueError:
                # Skip moves that can't be encoded (shouldn't happen)
                pass
        return mask


# Global encoder instance
_encoder = None


def get_encoder() -> MoveEncoder:
    """Get the global MoveEncoder instance.

    IMPORTANT: This now returns the C++-aligned encoder to ensure
    compatibility with the C++ MCTS backend.
    """
    global _encoder
    if _encoder is None:
        # Use C++-aligned encoder for compatibility with C++ backend
        from .moves_cpp_aligned import get_cpp_aligned_encoder
        _encoder = get_cpp_aligned_encoder()
    return _encoder


def reset_encoder():
    """Reset the global encoder (useful for testing)."""
    global _encoder
    _encoder = None


def encode_move(move: chess.Move, board: Optional[chess.Board] = None) -> int:
    """Convenience function to encode a move."""
    return get_encoder().encode(move, board)


def decode_move(action: int, board: chess.Board) -> chess.Move:
    """Convenience function to decode an action."""
    return get_encoder().decode(action, board)


def get_legal_mask(board: chess.Board) -> np.ndarray:
    """Convenience function to get legal action mask."""
    return get_encoder().get_legal_action_mask(board)
