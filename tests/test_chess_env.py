"""Tests for the chess environment module."""

import pytest
import numpy as np
import chess

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphazero.chess_env import (
    GameState,
    encode_board,
    MoveEncoder,
    get_encoder,
    encode_move,
    decode_move,
    get_legal_mask,
    TOTAL_PLANES,
)


class TestMoveEncoder:
    """Tests for move encoding/decoding."""

    def test_encoder_singleton(self):
        """Test that get_encoder returns the same instance."""
        enc1 = get_encoder()
        enc2 = get_encoder()
        assert enc1 is enc2

    def test_encode_decode_roundtrip(self):
        """Test that encoding and decoding are inverses."""
        board = chess.Board()
        encoder = get_encoder()

        for move in board.legal_moves:
            action = encoder.encode(move, board)
            decoded = encoder.decode(action, board)
            assert move == decoded, f"Roundtrip failed for {move}"

    def test_action_range(self):
        """Test that all actions are in valid range."""
        board = chess.Board()
        encoder = get_encoder()

        for move in board.legal_moves:
            action = encoder.encode(move, board)
            assert 0 <= action < 4672, f"Action {action} out of range for {move}"

    def test_legal_mask_shape(self):
        """Test legal mask has correct shape."""
        board = chess.Board()
        mask = get_legal_mask(board)
        assert mask.shape == (4672,)
        assert mask.dtype == np.float32

    def test_legal_mask_count(self):
        """Test legal mask has correct number of legal moves."""
        board = chess.Board()
        mask = get_legal_mask(board)
        assert np.sum(mask) == len(list(board.legal_moves))

    def test_promotion_encoding(self):
        """Test that promotions are encoded correctly."""
        # Position with pawn about to promote
        board = chess.Board("8/P7/8/8/8/8/8/4K2k w - - 0 1")
        encoder = get_encoder()

        for move in board.legal_moves:
            if move.promotion:
                action = encoder.encode(move, board)
                decoded = encoder.decode(action, board)
                assert decoded.promotion == move.promotion


class TestBoardEncoding:
    """Tests for board state encoding."""

    def test_encoding_shape(self):
        """Test that encoding has correct shape."""
        board = chess.Board()
        encoding = encode_board(board)
        assert encoding.shape == (TOTAL_PLANES, 8, 8)
        assert encoding.dtype == np.float32

    def test_encoding_values(self):
        """Test that encoding values are binary."""
        board = chess.Board()
        encoding = encode_board(board)
        assert np.all((encoding == 0) | (encoding == 1))

    def test_piece_planes(self):
        """Test that piece planes are correct for starting position."""
        board = chess.Board()
        encoding = encode_board(board, flip_for_black=False)

        # Check white pawns (plane 0)
        pawn_plane = encoding[0]
        assert pawn_plane[1, :].sum() == 8  # 8 pawns on rank 2

        # Check white king (plane 5)
        king_plane = encoding[5]
        assert king_plane[0, 4] == 1  # King on e1


class TestGameState:
    """Tests for GameState wrapper."""

    def test_initial_state(self):
        """Test initial game state."""
        state = GameState()
        assert state.turn == chess.WHITE
        assert state.fullmove_number == 1
        assert not state.is_terminal()

    def test_apply_action(self):
        """Test applying an action."""
        state = GameState()
        legal_actions = state.get_legal_action_indices()
        action = legal_actions[0]

        new_state = state.apply_action(action)
        assert new_state.turn == chess.BLACK
        assert state.turn == chess.WHITE  # Original unchanged

    def test_observation_shape(self):
        """Test observation shape."""
        state = GameState()
        obs = state.get_observation()
        assert obs.shape == (119, 8, 8)

    def test_legal_actions_shape(self):
        """Test legal actions shape."""
        state = GameState()
        mask = state.get_legal_actions()
        assert mask.shape == (4672,)

    def test_terminal_detection(self):
        """Test terminal state detection."""
        # Fool's mate
        state = GameState()
        moves = ["f2f3", "e7e5", "g2g4", "d8h4"]

        for move_str in moves:
            move = chess.Move.from_uci(move_str)
            state = state.apply_move(move)

        assert state.is_terminal()
        result = state.get_result()
        assert result.winner == chess.BLACK

    def test_from_fen(self):
        """Test creating state from FEN."""
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        state = GameState.from_fen(fen)
        assert state.turn == chess.BLACK


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
