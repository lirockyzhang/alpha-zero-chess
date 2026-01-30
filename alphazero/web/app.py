"""Web interface for playing chess against the AlphaZero model.

Flask-based web server with chessboard.js frontend for interactive gameplay.
"""

import os
import json
import chess
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict

try:
    from flask import Flask, render_template, request, jsonify, send_from_directory
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: flask not installed. Install with: pip install flask flask-cors")

from alphazero.neural.network import AlphaZeroNetwork
from alphazero.chess_env import GameState
from alphazero.mcts import create_mcts
from alphazero.mcts.evaluator import NetworkEvaluator
from alphazero.config import MCTSConfig


class ChessWebInterface:
    """Web interface for playing against AlphaZero model."""

    def __init__(self, checkpoint_path: str, num_simulations: int = 400,
                 device: str = "cuda", port: int = 5000):
        """Initialize web interface.

        Args:
            checkpoint_path: Path to model checkpoint
            num_simulations: MCTS simulations per move
            device: Device to run model on
            port: Port to run web server on
        """
        if not FLASK_AVAILABLE:
            raise ImportError("flask is required. Install with: pip install flask flask-cors")

        self.checkpoint_path = checkpoint_path
        self.num_simulations = num_simulations
        self.device = device
        self.port = port

        # Load model
        self.network, self.num_filters, self.num_blocks = self._load_model()

        # Create MCTS and evaluator
        self.mcts_config = MCTSConfig(
            num_simulations=num_simulations,
            c_puct=1.25,
            dirichlet_alpha=0.0,  # No exploration noise
            dirichlet_epsilon=0.0,
            temperature=0.0  # Greedy selection
        )
        self.mcts = create_mcts(config=self.mcts_config)
        self.evaluator = NetworkEvaluator(self.network, device, use_amp=True)

        # Game state storage (in-memory, keyed by session ID)
        self.games: Dict[str, GameState] = {}

        # Create Flask app
        self.app = Flask(__name__,
                        template_folder=str(Path(__file__).parent / 'templates'),
                        static_folder=str(Path(__file__).parent / 'static'))
        CORS(self.app)
        self._setup_routes()

    def _load_model(self):
        """Load model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}...")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Extract network architecture from checkpoint or filename
        num_filters = checkpoint.get('num_filters', 192)
        num_blocks = checkpoint.get('num_blocks', 15)

        # Create and load network
        network = AlphaZeroNetwork(num_filters=num_filters, num_blocks=num_blocks)
        network.load_state_dict(checkpoint['network_state_dict'])
        network = network.to(self.device)
        network.eval()

        print(f"Model loaded: {num_filters} filters, {num_blocks} blocks")
        return network, num_filters, num_blocks

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            """Serve main page."""
            return render_template('chess.html',
                                 num_simulations=self.num_simulations,
                                 num_filters=self.num_filters,
                                 num_blocks=self.num_blocks)

        @self.app.route('/api/new_game', methods=['POST'])
        def new_game():
            """Start a new game."""
            data = request.json
            session_id = data.get('session_id', 'default')
            human_color = data.get('color', 'white')  # 'white' or 'black'

            # Create new game
            self.games[session_id] = GameState()

            # If human plays black, make first move
            if human_color == 'black':
                move_uci = self._get_model_move(session_id)
                return jsonify({
                    'success': True,
                    'fen': self.games[session_id].board.fen(),
                    'model_move': move_uci,
                    'game_over': False
                })

            return jsonify({
                'success': True,
                'fen': self.games[session_id].board.fen(),
                'game_over': False
            })

        @self.app.route('/api/make_move', methods=['POST'])
        def make_move():
            """Make a move (human move + model response)."""
            data = request.json
            session_id = data.get('session_id', 'default')
            move_uci = data.get('move')

            if session_id not in self.games:
                return jsonify({'success': False, 'error': 'Game not found'}), 404

            game = self.games[session_id]

            try:
                # Apply human move
                move = chess.Move.from_uci(move_uci)
                if move not in game.board.legal_moves:
                    return jsonify({'success': False, 'error': 'Illegal move'}), 400

                # Apply move (immutable update)
                self.games[session_id] = game.apply_move(move)
                game = self.games[session_id]

                # Check if game is over
                if game.is_terminal():
                    result = game.get_result()
                    return jsonify({
                        'success': True,
                        'fen': game.board.fen(),
                        'game_over': True,
                        'result': self._format_result(result)
                    })

                # Get model move
                model_move_uci = self._get_model_move(session_id)

                # Reload game state after model move (it was updated in _get_model_move)
                game = self.games[session_id]

                # Check if game is over after model move
                game_over = game.is_terminal()
                result = None
                if game_over:
                    result = self._format_result(game.get_result())

                return jsonify({
                    'success': True,
                    'fen': game.board.fen(),
                    'model_move': model_move_uci,
                    'game_over': game_over,
                    'result': result
                })

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/get_legal_moves', methods=['POST'])
        def get_legal_moves():
            """Get legal moves for current position."""
            data = request.json
            session_id = data.get('session_id', 'default')

            if session_id not in self.games:
                return jsonify({'success': False, 'error': 'Game not found'}), 404

            game = self.games[session_id]
            legal_moves = [move.uci() for move in game.board.legal_moves]

            return jsonify({
                'success': True,
                'legal_moves': legal_moves
            })

    def _get_model_move(self, session_id: str) -> str:
        """Get model's move for current position.

        Args:
            session_id: Game session ID

        Returns:
            Move in UCI format
        """
        game = self.games[session_id]

        # Run MCTS
        policy, root, stats = self.mcts.search(
            game, self.evaluator,
            move_number=len(game.board.move_stack),
            add_noise=False
        )

        # Select best action
        action = int(np.argmax(policy))

        # Convert action to move using GameState method
        move = game.action_to_move(action)

        # Apply move (immutable update)
        self.games[session_id] = game.apply_move(move)

        return move.uci()

    def _format_result(self, result) -> Dict:
        """Format game result for JSON response."""
        if result is None:
            return {'winner': None, 'reason': 'draw'}

        return {
            'winner': 'white' if result.winner is True else 'black' if result.winner is False else None,
            'reason': str(result.termination)
        }

    def run(self, debug: bool = False):
        """Run the web server.

        Args:
            debug: Enable debug mode
        """
        print(f"Starting chess web interface on http://localhost:{self.port}")
        print(f"Model: {self.num_filters} filters, {self.num_blocks} blocks")
        print(f"MCTS simulations: {self.num_simulations}")
        print("Press Ctrl+C to stop")
        self.app.run(debug=debug, port=self.port, host='0.0.0.0')


def main():
    """Run web interface from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="AlphaZero Chess Web Interface")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--simulations", type=int, default=400,
                       help="MCTS simulations per move")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run model on (cuda or cpu)")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port to run web server on")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")

    args = parser.parse_args()

    interface = ChessWebInterface(
        checkpoint_path=args.checkpoint,
        num_simulations=args.simulations,
        device=args.device,
        port=args.port
    )
    interface.run(debug=args.debug)


if __name__ == "__main__":
    main()
