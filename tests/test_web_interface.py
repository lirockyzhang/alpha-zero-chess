"""Tests for web interface functionality.

Tests the Flask web application, API endpoints, and game state management.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if Flask is available
try:
    from flask import Flask
    import chess
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

pytestmark = pytest.mark.skipif(not FLASK_AVAILABLE, reason="Flask not installed")


@pytest.fixture
def mock_checkpoint(tmp_path):
    """Create a mock checkpoint file for testing."""
    import torch
    from alphazero.neural.network import AlphaZeroNetwork

    # Create a small test network
    network = AlphaZeroNetwork(num_filters=32, num_blocks=2)

    checkpoint = {
        'network_state_dict': network.state_dict(),
        'num_filters': 32,
        'num_blocks': 2,
        'step': 1000,
        'iteration': 1
    }

    checkpoint_path = tmp_path / "test_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    return str(checkpoint_path)


@pytest.fixture
def web_app(mock_checkpoint):
    """Create a test web application instance."""
    from web.app import ChessWebInterface

    interface = ChessWebInterface(
        checkpoint_path=mock_checkpoint,
        num_simulations=10,  # Low for fast testing
        device='cpu',
        port=5001  # Different port for testing
    )

    interface.app.config['TESTING'] = True
    return interface.app.test_client(), interface


class TestWebInterface:
    """Test web interface initialization and configuration."""

    def test_app_creation(self, web_app):
        """Test that the Flask app is created successfully."""
        client, interface = web_app
        assert client is not None
        assert interface is not None

    def test_index_route(self, web_app):
        """Test that the index route returns HTML."""
        client, _ = web_app
        response = client.get('/')
        assert response.status_code == 200
        assert b'AlphaZero Chess' in response.data or b'chess' in response.data.lower()


class TestAPIEndpoints:
    """Test REST API endpoints."""

    def test_new_game_white(self, web_app):
        """Test starting a new game as white."""
        client, _ = web_app

        response = client.post('/api/new_game', json={
            'session_id': 'test_session_1',
            'color': 'white'
        })

        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'fen' in data
        assert data['game_over'] is False
        # Starting position FEN
        assert 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR' in data['fen']

    def test_new_game_black(self, web_app):
        """Test starting a new game as black (AI makes first move)."""
        client, _ = web_app

        response = client.post('/api/new_game', json={
            'session_id': 'test_session_2',
            'color': 'black'
        })

        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'fen' in data
        assert 'model_move' in data
        assert data['game_over'] is False
        # FEN should be different from starting position
        assert data['fen'] != 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

    def test_make_move_valid(self, web_app):
        """Test making a valid move."""
        client, _ = web_app

        # Start a new game
        client.post('/api/new_game', json={
            'session_id': 'test_session_3',
            'color': 'white'
        })

        # Make a move (e2e4)
        response = client.post('/api/make_move', json={
            'session_id': 'test_session_3',
            'move': 'e2e4'
        })

        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'fen' in data
        assert 'model_move' in data
        # FEN should reflect the moves made
        assert 'e4' in data['fen'] or data['fen'] != 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

    def test_make_move_invalid(self, web_app):
        """Test making an invalid move."""
        client, _ = web_app

        # Start a new game
        client.post('/api/new_game', json={
            'session_id': 'test_session_4',
            'color': 'white'
        })

        # Try an illegal move (e2e5 - pawn can't move 3 squares)
        response = client.post('/api/make_move', json={
            'session_id': 'test_session_4',
            'move': 'e2e5'
        })

        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
        assert 'error' in data

    def test_make_move_no_session(self, web_app):
        """Test making a move without a valid session."""
        client, _ = web_app

        response = client.post('/api/make_move', json={
            'session_id': 'nonexistent_session',
            'move': 'e2e4'
        })

        assert response.status_code == 404
        data = response.get_json()
        assert data['success'] is False
        assert 'Game not found' in data['error']

    def test_get_legal_moves(self, web_app):
        """Test getting legal moves for current position."""
        client, _ = web_app

        # Start a new game
        client.post('/api/new_game', json={
            'session_id': 'test_session_5',
            'color': 'white'
        })

        # Get legal moves
        response = client.post('/api/get_legal_moves', json={
            'session_id': 'test_session_5'
        })

        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'legal_moves' in data
        assert len(data['legal_moves']) == 20  # Starting position has 20 legal moves
        assert 'e2e4' in data['legal_moves']
        assert 'd2d4' in data['legal_moves']


class TestGameStateManagement:
    """Test game state management and immutability."""

    def test_multiple_sessions(self, web_app):
        """Test that multiple sessions are independent."""
        client, _ = web_app

        # Start two games
        client.post('/api/new_game', json={
            'session_id': 'session_a',
            'color': 'white'
        })
        client.post('/api/new_game', json={
            'session_id': 'session_b',
            'color': 'white'
        })

        # Make different moves in each session
        response_a = client.post('/api/make_move', json={
            'session_id': 'session_a',
            'move': 'e2e4'
        })
        response_b = client.post('/api/make_move', json={
            'session_id': 'session_b',
            'move': 'd2d4'
        })

        # Verify they have different states
        data_a = response_a.get_json()
        data_b = response_b.get_json()
        assert data_a['fen'] != data_b['fen']

    def test_game_state_persistence(self, web_app):
        """Test that game state persists across multiple moves."""
        client, _ = web_app

        # Start a game
        client.post('/api/new_game', json={
            'session_id': 'test_persistence',
            'color': 'white'
        })

        # Make first move
        response1 = client.post('/api/make_move', json={
            'session_id': 'test_persistence',
            'move': 'e2e4'
        })
        fen1 = response1.get_json()['fen']

        # Make second move (after AI responds)
        # The FEN should be different after each move
        response2 = client.post('/api/make_move', json={
            'session_id': 'test_persistence',
            'move': 'd2d4'
        })
        fen2 = response2.get_json()['fen']

        # FENs should be different and reflect move history
        assert fen1 != fen2
        # At least one should show evidence of moves being made
        assert fen1 != 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        assert fen2 != 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_malformed_json(self, web_app):
        """Test handling of malformed JSON requests."""
        client, _ = web_app

        response = client.post('/api/new_game',
                              data='invalid json',
                              content_type='application/json')

        # Should handle gracefully (400 or 500 depending on Flask version)
        assert response.status_code in [400, 500]

    def test_missing_parameters(self, web_app):
        """Test handling of missing required parameters."""
        client, _ = web_app

        # Start a game first
        client.post('/api/new_game', json={
            'session_id': 'test_session',
            'color': 'white'
        })

        # Missing 'move' parameter
        response = client.post('/api/make_move', json={
            'session_id': 'test_session'
        })

        # Should return error (400, 404, or 500 depending on implementation)
        assert response.status_code in [400, 404, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
