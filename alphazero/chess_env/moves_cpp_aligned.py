"""Move encoding/decoding aligned with C++ implementation.

This encoder exactly matches the C++ move encoder in alphazero-cpp,
including perspective flipping and knight move ordering.

Key differences from the original Python encoder:
1. Perspective flipping: Flips board coordinates for black's moves
2. Knight move ordering: Matches C++ clockwise pattern
3. Underpromotion handling: Matches C++ direction mapping
"""

import chess
import numpy as np
from typing import Optional, Tuple


class CppAlignedMoveEncoder:
    """Move encoder that exactly matches the C++ implementation.

    The encoding scheme (matching C++ alphazero-cpp/src/encoding/move_encoder.cpp):
    - Queen-like moves: from_sq * 56 + dir_idx * 7 + (distance - 1)
    - Knight moves: 3584 + from_sq * 8 + knight_idx
    - Underpromotions: 4096 + from_sq * 9 + dir_idx * 3 + piece_idx

    CRITICAL: All moves are encoded from the current player's perspective.
    For black's moves, board coordinates are flipped (square -> 63 - square).
    """

    def __init__(self):
        self.queen_offset = 0
        self.knight_offset = 3584  # 56 * 64
        self.underpromo_offset = 4096  # 56 * 64 + 8 * 64
        self.num_actions = 4672

        # Queen move directions (N, NE, E, SE, S, SW, W, NW)
        self.queen_directions = [
            (1, 0),   # N
            (1, 1),   # NE
            (0, 1),   # E
            (-1, 1),  # SE
            (-1, 0),  # S
            (-1, -1), # SW
            (0, -1),  # W
            (1, -1),  # NW
        ]

        # Knight move ordering (matching C++ lines 137-144)
        # Clockwise pattern starting from top-right
        self.knight_moves = [
            (2, 1),   # index 0: rank_diff=2, file_diff=1
            (1, 2),   # index 1: rank_diff=1, file_diff=2
            (-1, 2),  # index 2: rank_diff=-1, file_diff=2
            (-2, 1),  # index 3: rank_diff=-2, file_diff=1
            (-2, -1), # index 4: rank_diff=-2, file_diff=-1
            (-1, -2), # index 5: rank_diff=-1, file_diff=-2
            (1, -2),  # index 6: rank_diff=1, file_diff=-2
            (2, -1),  # index 7: rank_diff=2, file_diff=-1
        ]

    def _flip_square(self, square: int) -> int:
        """Flip square for black's perspective (square -> 63 - square)."""
        return 63 - square

    def _square_to_rank_file(self, square: int) -> Tuple[int, int]:
        """Convert square to (rank, file)."""
        return square // 8, square % 8

    def _rank_file_to_square(self, rank: int, file: int) -> int:
        """Convert (rank, file) to square."""
        return rank * 8 + file

    def _get_direction_index(self, from_sq: int, to_sq: int) -> int:
        """Get direction index for queen-like moves."""
        from_rank, from_file = self._square_to_rank_file(from_sq)
        to_rank, to_file = self._square_to_rank_file(to_sq)

        rank_diff = to_rank - from_rank
        file_diff = to_file - from_file

        # Normalize to direction
        if rank_diff == 0 and file_diff == 0:
            return -1

        # Determine direction
        dr = 0 if rank_diff == 0 else (1 if rank_diff > 0 else -1)
        dc = 0 if file_diff == 0 else (1 if file_diff > 0 else -1)

        # Find matching direction
        for idx, (d_rank, d_file) in enumerate(self.queen_directions):
            if dr == d_rank and dc == d_file:
                return idx

        return -1

    def _get_distance(self, from_sq: int, to_sq: int) -> int:
        """Get distance for queen-like moves."""
        from_rank, from_file = self._square_to_rank_file(from_sq)
        to_rank, to_file = self._square_to_rank_file(to_sq)

        rank_diff = abs(to_rank - from_rank)
        file_diff = abs(to_file - from_file)

        return max(rank_diff, file_diff)

    def encode(self, move: chess.Move, board: chess.Board) -> int:
        """Encode a chess move to an action index.

        Args:
            move: The chess move to encode
            board: Current board state (needed for perspective flipping)

        Returns:
            Action index in range [0, 4672)
        """
        from_sq = move.from_square
        to_sq = move.to_square
        promotion = move.promotion

        # Apply perspective flipping for black
        flip = (board.turn == chess.BLACK)
        if flip:
            from_sq = self._flip_square(from_sq)
            to_sq = self._flip_square(to_sq)

        # Determine move type
        piece = board.piece_at(move.from_square)
        if piece is None:
            raise ValueError(f"No piece at {chess.square_name(move.from_square)}")

        # Knight moves
        if piece.piece_type == chess.KNIGHT:
            from_rank, from_file = self._square_to_rank_file(from_sq)
            to_rank, to_file = self._square_to_rank_file(to_sq)
            rank_diff = to_rank - from_rank
            file_diff = to_file - from_file

            # Find knight move index
            knight_idx = -1
            for idx, (dr, dc) in enumerate(self.knight_moves):
                if rank_diff == dr and file_diff == dc:
                    knight_idx = idx
                    break

            if knight_idx < 0:
                raise ValueError(f"Invalid knight move: {move.uci()}")

            return self.knight_offset + from_sq * 8 + knight_idx

        # Underpromotions
        if promotion is not None and promotion != chess.QUEEN:
            from_file = from_sq % 8
            to_file = to_sq % 8
            file_diff = to_file - from_file

            # Direction: -1 (left), 0 (straight), 1 (right) -> maps to 0, 1, 2
            direction = file_diff + 1
            if direction < 0 or direction > 2:
                raise ValueError(f"Invalid underpromotion direction: {move.uci()}")

            # Piece index: Knight=0, Bishop=1, Rook=2
            if promotion == chess.KNIGHT:
                piece_idx = 0
            elif promotion == chess.BISHOP:
                piece_idx = 1
            elif promotion == chess.ROOK:
                piece_idx = 2
            else:
                raise ValueError(f"Invalid underpromotion piece: {promotion}")

            return self.underpromo_offset + from_sq * 9 + direction * 3 + piece_idx

        # Queen-like moves (includes queen promotions)
        direction = self._get_direction_index(from_sq, to_sq)
        if direction < 0:
            raise ValueError(f"Invalid queen-like move direction: {move.uci()}")

        distance = self._get_distance(from_sq, to_sq)
        if distance < 1 or distance > 7:
            raise ValueError(f"Invalid queen-like move distance: {move.uci()}")

        return self.queen_offset + from_sq * 56 + direction * 7 + (distance - 1)

    def decode(self, action: int, board: chess.Board) -> chess.Move:
        """Decode an action index to a chess move.

        Args:
            action: Action index in range [0, 4672)
            board: Current board state (needed for perspective flipping)

        Returns:
            The corresponding chess.Move
        """
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Invalid action index: {action}")

        flip = (board.turn == chess.BLACK)

        # Determine move type by offset
        if action < self.knight_offset:
            # Queen-like move
            from_sq = action // 56
            remainder = action % 56
            direction = remainder // 7
            distance = (remainder % 7) + 1

            # Calculate to_square
            from_rank, from_file = self._square_to_rank_file(from_sq)
            dr, dc = self.queen_directions[direction]
            to_rank = from_rank + dr * distance
            to_file = from_file + dc * distance

            if not (0 <= to_rank < 8 and 0 <= to_file < 8):
                raise ValueError(f"Invalid decoded position for action {action}")

            to_sq = self._rank_file_to_square(to_rank, to_file)

            # Unflip if black
            if flip:
                from_sq = self._flip_square(from_sq)
                to_sq = self._flip_square(to_sq)

            # Check for queen promotion
            promotion = None
            piece = board.piece_at(from_sq)
            if piece and piece.piece_type == chess.PAWN:
                to_rank_abs = chess.square_rank(to_sq)
                if (piece.color == chess.WHITE and to_rank_abs == 7) or \
                   (piece.color == chess.BLACK and to_rank_abs == 0):
                    promotion = chess.QUEEN

            return chess.Move(from_sq, to_sq, promotion=promotion)

        elif action < self.underpromo_offset:
            # Knight move
            offset = action - self.knight_offset
            from_sq = offset // 8
            knight_idx = offset % 8

            # Calculate to_square
            from_rank, from_file = self._square_to_rank_file(from_sq)
            dr, dc = self.knight_moves[knight_idx]
            to_rank = from_rank + dr
            to_file = from_file + dc

            if not (0 <= to_rank < 8 and 0 <= to_file < 8):
                raise ValueError(f"Invalid decoded knight position for action {action}")

            to_sq = self._rank_file_to_square(to_rank, to_file)

            # Unflip if black
            if flip:
                from_sq = self._flip_square(from_sq)
                to_sq = self._flip_square(to_sq)

            return chess.Move(from_sq, to_sq)

        else:
            # Underpromotion
            offset = action - self.underpromo_offset
            from_sq = offset // 9
            remainder = offset % 9
            direction = remainder // 3
            piece_idx = remainder % 3

            # Calculate to_square
            from_file = from_sq % 8
            file_diff = direction - 1  # Maps 0,1,2 back to -1,0,1
            to_file = from_file + file_diff

            # Determine to_rank based on perspective
            from_rank = from_sq // 8
            if flip:
                # Black's perspective: moving down (decreasing rank)
                to_rank = from_rank - 1
            else:
                # White's perspective: moving up (increasing rank)
                to_rank = from_rank + 1

            if not (0 <= to_rank < 8 and 0 <= to_file < 8):
                raise ValueError(f"Invalid decoded underpromotion position for action {action}")

            to_sq = self._rank_file_to_square(to_rank, to_file)

            # Unflip if black
            if flip:
                from_sq = self._flip_square(from_sq)
                to_sq = self._flip_square(to_sq)

            # Determine promotion piece
            if piece_idx == 0:
                promotion = chess.KNIGHT
            elif piece_idx == 1:
                promotion = chess.BISHOP
            else:  # piece_idx == 2
                promotion = chess.ROOK

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
_cpp_aligned_encoder = None


def get_cpp_aligned_encoder() -> CppAlignedMoveEncoder:
    """Get the global C++-aligned MoveEncoder instance."""
    global _cpp_aligned_encoder
    if _cpp_aligned_encoder is None:
        _cpp_aligned_encoder = CppAlignedMoveEncoder()
    return _cpp_aligned_encoder
