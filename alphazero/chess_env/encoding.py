"""Board state encoding for AlphaZero chess.

Encodes a chess position into a tensor of shape (119, 8, 8).

Plane layout (119 planes total):
- Planes 0-95: Piece positions for 8 history steps (12 planes each)
  - 6 piece types × 2 colors = 12 planes per position
  - Order: P, N, B, R, Q, K (white), p, n, b, r, q, k (black)
- Planes 96-99: Castling rights (4 planes)
  - White kingside, white queenside, black kingside, black queenside
- Plane 100: Side to move (all 1s if white, all 0s if black)
- Planes 101-108: Repetition counters (8 planes, one-hot encoded)
- Planes 109-118: Move clocks
  - 109-113: Halfmove clock (no-progress count, 5 planes)
  - 114-118: Fullmove number (5 planes)
"""

import chess
import numpy as np
from typing import List, Optional


# Piece type to plane offset mapping
PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

NUM_PIECE_PLANES = 12
NUM_HISTORY_STEPS = 8
NUM_PIECE_HISTORY_PLANES = NUM_PIECE_PLANES * NUM_HISTORY_STEPS  # 96
NUM_CASTLING_PLANES = 4
NUM_SIDE_TO_MOVE_PLANES = 1
NUM_REPETITION_PLANES = 8
NUM_HALFMOVE_PLANES = 5
NUM_FULLMOVE_PLANES = 5

TOTAL_PLANES = (
    NUM_PIECE_HISTORY_PLANES +  # 96
    NUM_CASTLING_PLANES +       # 4
    NUM_SIDE_TO_MOVE_PLANES +   # 1
    NUM_REPETITION_PLANES +     # 8
    NUM_HALFMOVE_PLANES +       # 5
    NUM_FULLMOVE_PLANES         # 5
)  # = 119


def encode_piece_planes(board: chess.Board, flip: bool = False) -> np.ndarray:
    """Encode piece positions for a single board state.

    Args:
        board: Chess board to encode
        flip: If True, flip the board (for black's perspective)

    Returns:
        Array of shape (12, 8, 8)
    """
    planes = np.zeros((NUM_PIECE_PLANES, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            plane_idx = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
            row = square // 8
            col = square % 8

            if flip:
                row = 7 - row
                # Swap white and black planes
                if plane_idx < 6:
                    plane_idx += 6
                else:
                    plane_idx -= 6

            planes[plane_idx, row, col] = 1.0

    return planes


def encode_castling(board: chess.Board, flip: bool = False) -> np.ndarray:
    """Encode castling rights as 4 planes.

    Args:
        board: Chess board
        flip: If True, swap white/black castling rights

    Returns:
        Array of shape (4, 8, 8)
    """
    planes = np.zeros((NUM_CASTLING_PLANES, 8, 8), dtype=np.float32)

    if flip:
        rights = [
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
        ]
    else:
        rights = [
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
        ]

    for i, has_right in enumerate(rights):
        if has_right:
            planes[i, :, :] = 1.0

    return planes


def encode_side_to_move(board: chess.Board) -> np.ndarray:
    """Encode side to move as a single plane.

    Args:
        board: Chess board

    Returns:
        Array of shape (1, 8, 8) - all 1s if white to move, all 0s if black
    """
    plane = np.zeros((1, 8, 8), dtype=np.float32)
    if board.turn == chess.WHITE:
        plane[:, :, :] = 1.0
    return plane


def encode_repetition(board: chess.Board) -> np.ndarray:
    """Encode repetition count as one-hot planes.

    Args:
        board: Chess board

    Returns:
        Array of shape (8, 8, 8)
    """
    planes = np.zeros((NUM_REPETITION_PLANES, 8, 8), dtype=np.float32)

    # Count repetitions (0, 1, 2, or 3+)
    rep_count = min(board.is_repetition(2) + board.is_repetition(3), 7)
    if rep_count > 0:
        planes[rep_count, :, :] = 1.0
    else:
        planes[0, :, :] = 1.0

    return planes


def encode_move_clocks(board: chess.Board) -> np.ndarray:
    """Encode halfmove clock and fullmove number.

    Args:
        board: Chess board

    Returns:
        Array of shape (10, 8, 8)
    """
    planes = np.zeros((NUM_HALFMOVE_PLANES + NUM_FULLMOVE_PLANES, 8, 8), dtype=np.float32)

    # Halfmove clock (50-move rule counter) - encode as binary
    halfmove = min(board.halfmove_clock, 31)  # Cap at 31 (5 bits)
    for i in range(NUM_HALFMOVE_PLANES):
        if halfmove & (1 << i):
            planes[i, :, :] = 1.0

    # Fullmove number - encode as binary
    fullmove = min(board.fullmove_number, 31)  # Cap at 31 (5 bits)
    for i in range(NUM_FULLMOVE_PLANES):
        if fullmove & (1 << i):
            planes[NUM_HALFMOVE_PLANES + i, :, :] = 1.0

    return planes


def encode_board(
    board: chess.Board,
    history: Optional[List[chess.Board]] = None,
    flip_for_black: bool = True
) -> np.ndarray:
    """Encode a chess position into the full 119-plane representation.

    Args:
        board: Current chess board
        history: List of previous board states (most recent first)
        flip_for_black: If True, flip board when it's black's turn

    Returns:
        Array of shape (119, 8, 8)
    """
    flip = flip_for_black and board.turn == chess.BLACK

    planes_list = []

    # Encode current position and history
    if history is None:
        history = []

    all_boards = [board] + history[:NUM_HISTORY_STEPS - 1]

    for i in range(NUM_HISTORY_STEPS):
        if i < len(all_boards):
            piece_planes = encode_piece_planes(all_boards[i], flip=flip)
        else:
            piece_planes = np.zeros((NUM_PIECE_PLANES, 8, 8), dtype=np.float32)
        planes_list.append(piece_planes)

    # Castling rights
    planes_list.append(encode_castling(board, flip=flip))

    # Side to move (always from current player's perspective after flip)
    side_plane = np.ones((1, 8, 8), dtype=np.float32)  # Always 1 after flip
    planes_list.append(side_plane)

    # Repetition
    planes_list.append(encode_repetition(board))

    # Move clocks
    planes_list.append(encode_move_clocks(board))

    # Concatenate all planes
    observation = np.concatenate(planes_list, axis=0)

    assert observation.shape == (TOTAL_PLANES, 8, 8), \
        f"Expected shape ({TOTAL_PLANES}, 8, 8), got {observation.shape}"

    return observation


class BoardEncoder:
    """Stateful encoder that maintains history for a game."""

    def __init__(self, flip_for_black: bool = True):
        self.flip_for_black = flip_for_black
        self.history: List[chess.Board] = []

    def reset(self):
        """Reset history for a new game."""
        self.history = []

    def encode(self, board: chess.Board) -> np.ndarray:
        """Encode the current position with history.

        Args:
            board: Current board state

        Returns:
            Array of shape (119, 8, 8)
        """
        observation = encode_board(board, self.history, self.flip_for_black)

        # Update history
        self.history.insert(0, board.copy())
        if len(self.history) > NUM_HISTORY_STEPS:
            self.history = self.history[:NUM_HISTORY_STEPS]

        return observation

    def encode_without_update(self, board: chess.Board) -> np.ndarray:
        """Encode without updating history (for MCTS simulations)."""
        return encode_board(board, self.history, self.flip_for_black)
