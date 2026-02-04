"""Web interface for playing chess against the AlphaZero model.

Flask-based web server with chessboard.js frontend for interactive gameplay.
"""

import os
import sys
import json
import chess
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from flask import Flask, render_template, request, jsonify, send_from_directory
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: flask not installed. Install with: pip install flask flask-cors")

# Use standalone module (no dependency on alphazero/ package)
try:
    # Try relative import (when running as package)
    from .alphazero_standalone import AlphaZeroNetwork, GameState, MCTSConfig
except ImportError:
    # Fall back to direct import (when running as script)
    from alphazero_standalone import AlphaZeroNetwork, GameState, MCTSConfig

# Try to import C++ backend for 122-channel encoding
try:
    cpp_build_path = str(Path(__file__).parent.parent / "alphazero-cpp" / "build" / "Release")
    if cpp_build_path not in sys.path:
        sys.path.insert(0, cpp_build_path)
    import alphazero_cpp
    CPP_BACKEND_AVAILABLE = True
except ImportError:
    CPP_BACKEND_AVAILABLE = False


class CppEncodingEvaluator:
    """Evaluator that uses C++ 122-channel encoding for C++ checkpoint compatibility."""

    def __init__(self, network, device: str = "cuda", use_amp: bool = True):
        """Initialize evaluator with C++ encoding.

        Args:
            network: AlphaZero network (122 input channels)
            device: Device to run inference on
            use_amp: Use mixed precision inference
        """
        self.network = network
        self.device = device
        self.use_amp = use_amp and device == "cuda"
        self.network.eval()

        if not CPP_BACKEND_AVAILABLE:
            raise ImportError(
                "C++ backend (alphazero_cpp) not available. "
                "Build it: cd alphazero-cpp && cmake -B build && cmake --build build --config Release"
            )

    def evaluate(self, observation: np.ndarray, legal_mask: np.ndarray,
                 fen: str = None) -> tuple:
        """Evaluate position using C++ encoding.

        Args:
            observation: Ignored (we use FEN instead for C++ encoding)
            legal_mask: Legal action mask (4672,)
            fen: FEN string (required for C++ encoding)

        Returns:
            Tuple of (policy, value)
        """
        if fen is None:
            raise ValueError("CppEncodingEvaluator requires FEN string")

        # Use C++ encoding: returns (8, 8, 122) HWC format
        obs_hwc = alphazero_cpp.encode_position(fen)
        # Convert to (122, 8, 8) CHW format
        obs_chw = np.transpose(obs_hwc, (2, 0, 1))

        # Convert to tensors
        obs_tensor = torch.from_numpy(obs_chw).float().unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(legal_mask).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.use_amp:
                from torch.amp import autocast
                with autocast('cuda'):
                    policy, value = self.network.predict(obs_tensor, mask_tensor)
            else:
                policy, value = self.network.predict(obs_tensor, mask_tensor)

        return policy.squeeze(0).cpu().numpy(), value.item()


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

        # Require C++ backend for encoding
        if not CPP_BACKEND_AVAILABLE:
            raise ImportError(
                "C++ backend (alphazero_cpp) is required. "
                "Build it: cd alphazero-cpp && cmake -B build && cmake --build build --config Release"
            )

        print("Using C++ encoding (122 channels)")
        self.evaluator = CppEncodingEvaluator(self.network, device, use_amp=True)

        # MCTS parameters (configurable at runtime via /api/update_settings)
        self.c_puct = 1.25

        # Game state storage (in-memory, keyed by session ID)
        self.games: Dict[str, GameState] = {}

        # Create Flask app with paths relative to this file
        template_folder = str(Path(__file__).parent / 'templates')
        static_folder = str(Path(__file__).parent / 'static')

        self.app = Flask(__name__,
                        template_folder=template_folder,
                        static_folder=static_folder)
        CORS(self.app)
        self._setup_routes()

    def _load_model(self):
        """Load model from checkpoint.

        Supports both Python and C++ checkpoint formats:
        - Python: 'network_state_dict', 'num_filters', 'num_blocks'
        - C++: 'model_state_dict', config dict with all params
        """
        print(f"Loading model from {self.checkpoint_path}...")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # Detect checkpoint format
        is_cpp_checkpoint = 'model_state_dict' in checkpoint

        if is_cpp_checkpoint:
            # C++ checkpoint format
            config = checkpoint.get('config', {})
            num_filters = config.get('num_filters', 192)
            num_blocks = config.get('num_blocks', 15)
            input_channels = config.get('input_channels', 122)
            policy_filters = config.get('policy_filters', 2)
            value_hidden = config.get('value_hidden', 256)
            backend = checkpoint.get('backend', 'cpp')

            print(f"Detected C++ checkpoint (backend={backend}, version={checkpoint.get('version', '?')})")
            print(f"  input_channels={input_channels}, policy_filters={policy_filters}, value_hidden={value_hidden}")

            # Create network with C++ architecture
            network = AlphaZeroNetwork(
                input_channels=input_channels,
                num_filters=num_filters,
                num_blocks=num_blocks,
                policy_filters=policy_filters,
                value_hidden=value_hidden
            )
            network.load_state_dict(checkpoint['model_state_dict'])

            # Store flag to use C++ encoding
            self.use_cpp_encoding = True
            self.input_channels = input_channels
        else:
            # Python checkpoint format
            num_filters = checkpoint.get('num_filters', 192)
            num_blocks = checkpoint.get('num_blocks', 15)

            print("Detected Python checkpoint")

            # Create network with Python architecture (119 channels)
            network = AlphaZeroNetwork(
                input_channels=119,
                num_filters=num_filters,
                num_blocks=num_blocks
            )
            network.load_state_dict(checkpoint['network_state_dict'])

            self.use_cpp_encoding = False
            self.input_channels = 119

        network = network.to(self.device)
        network.eval()

        print(f"Model loaded: {num_filters} filters, {num_blocks} blocks, {self.input_channels} input channels")
        return network, num_filters, num_blocks

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            """Serve main page."""
            return render_template('chess.html',
                                 num_simulations=self.num_simulations,
                                 c_puct=self.c_puct,
                                 num_filters=self.num_filters,
                                 num_blocks=self.num_blocks)

        @self.app.route('/api/update_settings', methods=['POST'])
        def update_settings():
            """Update MCTS settings at runtime."""
            data = request.json
            if 'num_simulations' in data:
                val = int(data['num_simulations'])
                if 1 <= val <= 100000:
                    self.num_simulations = val
            if 'c_puct' in data:
                val = float(data['c_puct'])
                if 0.01 <= val <= 100.0:
                    self.c_puct = val
            return jsonify({
                'success': True,
                'num_simulations': self.num_simulations,
                'c_puct': self.c_puct
            })

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
                move_uci, eval_data = self._get_model_move(session_id)
                return jsonify({
                    'success': True,
                    'fen': self.games[session_id].board.fen(),
                    'model_move': move_uci,
                    'evaluation': eval_data,
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
                model_move_uci, eval_data = self._get_model_move(session_id)

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
                    'evaluation': eval_data,
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

        @self.app.route('/api/self_play_move', methods=['POST'])
        def self_play_move():
            """Make one AI move in self-play mode (AI plays both sides)."""
            data = request.json
            session_id = data.get('session_id', 'default')

            if session_id not in self.games:
                return jsonify({'success': False, 'error': 'Game not found'}), 404

            game = self.games[session_id]

            if game.is_terminal():
                result = game.get_result()
                return jsonify({
                    'success': True,
                    'fen': game.board.fen(),
                    'game_over': True,
                    'result': self._format_result(result)
                })

            # Get model's move for current position (whichever side is to play)
            move_uci, eval_data = self._get_model_move(session_id)

            # Reload game state after model move
            game = self.games[session_id]

            game_over = game.is_terminal()
            result = None
            if game_over:
                result = self._format_result(game.get_result())

            return jsonify({
                'success': True,
                'fen': game.board.fen(),
                'model_move': move_uci,
                'evaluation': eval_data,
                'game_over': game_over,
                'result': result
            })

    def _get_model_move(self, session_id: str) -> tuple:
        """Get model's move for current position.

        Args:
            session_id: Game session ID

        Returns:
            Tuple of (move_uci, evaluation_data) where evaluation_data contains
            value estimate and top move probabilities
        """
        game = self.games[session_id]

        # Run C++ MCTS with batched leaf evaluation
        policy, root_value = self._run_cpp_mcts(game)

        # Select best action
        action = int(np.argmax(policy))

        # Convert action to move using GameState method
        move = game.action_to_move(action)

        # Get evaluation data from policy (already from visit counts)
        evaluation_data = {
            'value': float(root_value),
            'top_moves': self._get_top_moves_from_policy(game, policy, top_k=5)
        }

        # Apply move (immutable update)
        self.games[session_id] = game.apply_move(move)

        return move.uci(), evaluation_data

    def _run_cpp_mcts(self, game: GameState) -> tuple:
        """Run C++ MCTS with batched leaf evaluation.

        Args:
            game: Current game state

        Returns:
            Tuple of (policy, root_value)
        """
        fen = game.fen()
        board = game.board

        # Get initial evaluation
        obs_hwc = alphazero_cpp.encode_position(fen)
        obs_chw = np.transpose(obs_hwc, (2, 0, 1))

        legal_mask = np.zeros(4672, dtype=np.float32)
        for move in board.legal_moves:
            idx = alphazero_cpp.move_to_index(move.uci(), fen)
            if 0 <= idx < 4672:
                legal_mask[idx] = 1.0

        # Get root evaluation
        obs_tensor = torch.from_numpy(obs_chw).float().unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(legal_mask).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.device == "cuda":
                from torch.amp import autocast
                with autocast('cuda'):
                    root_policy, root_value = self.network.predict(obs_tensor, mask_tensor)
            else:
                root_policy, root_value = self.network.predict(obs_tensor, mask_tensor)

        root_policy = root_policy.squeeze(0).cpu().numpy().astype(np.float32)
        root_value_float = float(root_value.item())

        # Create C++ MCTS and run search
        mcts = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=self.num_simulations,
            batch_size=64,
            c_puct=self.c_puct
        )
        mcts.init_search(fen, root_policy, root_value_float)

        # Run MCTS with batched leaf evaluation
        while not mcts.is_complete():
            num_leaves, obs_batch, mask_batch = mcts.collect_leaves()
            if num_leaves == 0:
                break

            # Convert to NCHW and evaluate
            obs_nchw = np.transpose(obs_batch[:num_leaves], (0, 3, 1, 2))
            masks = mask_batch[:num_leaves]

            obs_t = torch.from_numpy(obs_nchw).float().to(self.device)
            mask_t = torch.from_numpy(masks).float().to(self.device)

            with torch.no_grad():
                if self.device == "cuda":
                    from torch.amp import autocast
                    with autocast('cuda'):
                        leaf_policies, leaf_values = self.network.predict(obs_t, mask_t)
                else:
                    leaf_policies, leaf_values = self.network.predict(obs_t, mask_t)

            mcts.update_leaves(
                leaf_policies.cpu().numpy().astype(np.float32),
                leaf_values.cpu().numpy().astype(np.float32)
            )

        # Get final policy from visit counts
        visit_counts = mcts.get_visit_counts()
        policy = visit_counts.astype(np.float32)
        policy = policy * legal_mask
        if policy.sum() > 0:
            policy = policy / policy.sum()
        else:
            policy = legal_mask / legal_mask.sum()

        # Use MCTS-backed root Q-value (averaged over all simulations)
        # instead of raw NN value which doesn't reflect search results
        mcts_value = mcts.get_root_value()

        return policy, float(mcts_value)

    def _get_top_moves_from_policy(self, game: GameState, policy: np.ndarray, top_k: int = 5) -> list:
        """Get top K moves from policy probabilities.

        Args:
            game: Current game state
            policy: Policy probabilities (4672,)
            top_k: Number of top moves to return

        Returns:
            List of dicts with move info
        """
        top_indices = np.argsort(policy)[-top_k:][::-1]

        top_moves = []
        for idx in top_indices:
            if policy[idx] > 0.001:  # Only include moves with meaningful probability
                try:
                    move = game.action_to_move(int(idx))
                    top_moves.append({
                        'move': move.uci(),
                        'move_san': game.board.san(move),
                        'probability': float(policy[idx]),
                        'visits': int(policy[idx] * self.num_simulations)  # Approximate
                    })
                except:
                    continue

        return top_moves

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
