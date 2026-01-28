"""Chess environment module for AlphaZero."""

from .board import GameState, GameResult
from .encoding import encode_board, BoardEncoder, TOTAL_PLANES
from .moves import (
    MoveEncoder,
    get_encoder,
    encode_move,
    decode_move,
    get_legal_mask,
)

__all__ = [
    "GameState",
    "GameResult",
    "encode_board",
    "BoardEncoder",
    "TOTAL_PLANES",
    "MoveEncoder",
    "get_encoder",
    "encode_move",
    "decode_move",
    "get_legal_mask",
]
