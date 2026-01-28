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
from typing import Optional


# Direction vectors for queen-like moves (row_delta, col_delta)
QUEEN_DIRECTIONS = [
    (-1, 0),   # N
    (-1, 1),   # NE
    (0, 1),    # E
    (1, 1),    # SE
    (1, 0),    # S
    (1, -1),   # SW
    (0, -1),   # W
    (-1, -1),  # NW
]

# Knight move deltas
KNIGHT_MOVES = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
]

# Underpromotion pieces (queen promotion is encoded as regular move)
UNDERPROMOTION_PIECES = [chess.ROOK, chess.BISHOP, chess.KNIGHT]

# Promotion directions (from white's perspective): left-capture, forward, right-capture
PROMOTION_DIRECTIONS_WHITE = [(-1, -1), (-1, 0), (-1, 1)]
# Promotion directions (from black's perspective): left-capture, forward, right-capture
PROMOTION_DIRECTIONS_BLACK = [(1, 1), (1, 0), (1, -1)]
# Combined for lookup
PROMOTION_DIRECTIONS = PROMOTION_DIRECTIONS_WHITE + PROMOTION_DIRECTIONS_BLACK


class MoveEncoder:
    """Bidirectional mapping between chess.Move and action indices."""

    def __init__(self):
        self._build_encoding_tables()

    def _build_encoding_tables(self):
        """Build lookup tables for fast encoding/decoding."""
        # Action index layout:
        # [0, 3584): Queen-like moves (56 per square × 64 squares)
        # [3584, 4096): Knight moves (8 per square × 64 squares)
        # [4096, 4672): Underpromotions (9 per square × 64 squares)

        self.queen_offset = 0
        self.knight_offset = 3584
        self.underpromo_offset = 4096
        self.num_actions = 4672

        # Build decode table: action_index -> (from_sq, to_sq, promotion)
        self._decode_table = {}

        # Queen-like moves
        for from_sq in range(64):
            from_row, from_col = divmod(from_sq, 8)
            for dir_idx, (dr, dc) in enumerate(QUEEN_DIRECTIONS):
                for dist in range(1, 8):
                    to_row = from_row + dr * dist
                    to_col = from_col + dc * dist
                    if 0 <= to_row < 8 and 0 <= to_col < 8:
                        to_sq = to_row * 8 + to_col
                        action = self.queen_offset + from_sq * 56 + dir_idx * 7 + (dist - 1)
                        self._decode_table[action] = (from_sq, to_sq, None)

        # Knight moves
        for from_sq in range(64):
            from_row, from_col = divmod(from_sq, 8)
            for move_idx, (dr, dc) in enumerate(KNIGHT_MOVES):
                to_row = from_row + dr
                to_col = from_col + dc
                if 0 <= to_row < 8 and 0 <= to_col < 8:
                    to_sq = to_row * 8 + to_col
                    action = self.knight_offset + from_sq * 8 + move_idx
                    self._decode_table[action] = (from_sq, to_sq, None)

        # Underpromotions
        # White pawns promote from rank 6 (row 6) to rank 7 (row 7)
        # Black pawns promote from rank 1 (row 1) to rank 0 (row 0)
        for from_sq in range(64):
            from_row, from_col = divmod(from_sq, 8)
            # Determine which directions are valid based on rank
            if from_row == 6:  # White pawn about to promote
                directions = PROMOTION_DIRECTIONS_WHITE
            elif from_row == 1:  # Black pawn about to promote
                directions = PROMOTION_DIRECTIONS_BLACK
            else:
                continue  # No promotions from other ranks

            for dir_idx, (dr, dc) in enumerate(directions):
                to_row = from_row + dr
                to_col = from_col + dc
                if 0 <= to_row < 8 and 0 <= to_col < 8:
                    to_sq = to_row * 8 + to_col
                    for piece_idx, piece in enumerate(UNDERPROMOTION_PIECES):
                        action = self.underpromo_offset + from_sq * 9 + dir_idx * 3 + piece_idx
                        self._decode_table[action] = (from_sq, to_sq, piece)

        # Build encode table: (from_sq, to_sq, promotion) -> action_index
        self._encode_table = {v: k for k, v in self._decode_table.items()}

    def encode(self, move: chess.Move, board: Optional[chess.Board] = None) -> int:
        """Convert a chess.Move to an action index.

        Args:
            move: The chess move to encode
            board: Optional board for context (used to determine move type)

        Returns:
            Action index in range [0, 4672)
        """
        from_sq = move.from_square
        to_sq = move.to_square
        promotion = move.promotion

        # Handle queen promotion as regular move
        if promotion == chess.QUEEN:
            promotion = None

        # Try direct lookup first
        key = (from_sq, to_sq, promotion)
        if key in self._encode_table:
            return self._encode_table[key]

        # Fallback: compute encoding
        from_row, from_col = divmod(from_sq, 8)
        to_row, to_col = divmod(to_sq, 8)
        dr = to_row - from_row
        dc = to_col - from_col

        # Check if it's a knight move
        if (dr, dc) in KNIGHT_MOVES:
            move_idx = KNIGHT_MOVES.index((dr, dc))
            return self.knight_offset + from_sq * 8 + move_idx

        # Check if it's an underpromotion
        if promotion in UNDERPROMOTION_PIECES:
            # Find direction index (handle both white and black promotions)
            if (dr, dc) in PROMOTION_DIRECTIONS_WHITE:
                dir_idx = PROMOTION_DIRECTIONS_WHITE.index((dr, dc))
            else:
                dir_idx = PROMOTION_DIRECTIONS_BLACK.index((dr, dc))
            piece_idx = UNDERPROMOTION_PIECES.index(promotion)
            return self.underpromo_offset + from_sq * 9 + dir_idx * 3 + piece_idx

        # Must be a queen-like move
        # Normalize direction
        dist = max(abs(dr), abs(dc))
        dir_dr = dr // dist if dr != 0 else 0
        dir_dc = dc // dist if dc != 0 else 0
        dir_idx = QUEEN_DIRECTIONS.index((dir_dr, dir_dc))
        return self.queen_offset + from_sq * 56 + dir_idx * 7 + (dist - 1)

    def decode(self, action: int, board: chess.Board) -> chess.Move:
        """Convert an action index to a chess.Move.

        Args:
            action: Action index in range [0, 4672)
            board: Current board state (needed for promotion context)

        Returns:
            The corresponding chess.Move
        """
        if action not in self._decode_table:
            raise ValueError(f"Invalid action index: {action}")

        from_sq, to_sq, promotion = self._decode_table[action]

        # Check if this is a pawn reaching the back rank (needs queen promotion)
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_sq)
            if (piece.color == chess.WHITE and to_rank == 7) or \
               (piece.color == chess.BLACK and to_rank == 0):
                if promotion is None:
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
            action = self.encode(move, board)
            mask[action] = 1.0
        return mask


# Global encoder instance
_encoder = None


def get_encoder() -> MoveEncoder:
    """Get the global MoveEncoder instance."""
    global _encoder
    if _encoder is None:
        _encoder = MoveEncoder()
    return _encoder


def encode_move(move: chess.Move, board: Optional[chess.Board] = None) -> int:
    """Convenience function to encode a move."""
    return get_encoder().encode(move, board)


def decode_move(action: int, board: chess.Board) -> chess.Move:
    """Convenience function to decode an action."""
    return get_encoder().decode(action, board)


def get_legal_mask(board: chess.Board) -> np.ndarray:
    """Convenience function to get legal action mask."""
    return get_encoder().get_legal_action_mask(board)
