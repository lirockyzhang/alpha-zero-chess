"""Web interface for playing chess against the AlphaZero model.

Flask-based web server with chessboard.js frontend for interactive gameplay.
"""

import io
import os
import sys
import json
import chess
import chess.pgn
import torch
import numpy as np
import webbrowser
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add training scripts to path for network import
sys.path.insert(0, str(Path(__file__).parent.parent / "alphazero-cpp" / "scripts"))

try:
    from flask import Flask, render_template, request, jsonify, send_from_directory
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: flask not installed. Install with: pip install flask flask-cors")

# Import network from training scripts (single source of truth)
from network import AlphaZeroNet

# Use standalone module for game state and config
try:
    from .alphazero_standalone import GameState, MCTSConfig
except ImportError:
    from alphazero_standalone import GameState, MCTSConfig

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

    def __init__(self, network, device: str = "cuda"):
        """Initialize evaluator with C++ encoding.

        Args:
            network: AlphaZero network (122 input channels)
            device: Device to run inference on
        """
        self.network = network
        self.device = device
        self.network.eval()

        if not CPP_BACKEND_AVAILABLE:
            raise ImportError(
                "C++ backend (alphazero_cpp) not available. "
                "Build it: cd alphazero-cpp && cmake -B build && cmake --build build --config Release"
            )

    def evaluate(self, observation: np.ndarray, legal_mask: np.ndarray,
                 fen: str = None, history_fens: list = None) -> tuple:
        """Evaluate position using C++ encoding.

        Args:
            observation: Ignored (we use FEN instead for C++ encoding)
            legal_mask: Legal action mask (4672,)
            fen: FEN string (required for C++ encoding)
            history_fens: Optional list of previous position FENs (up to 8)

        Returns:
            Tuple of (policy, value)
        """
        if fen is None:
            raise ValueError("CppEncodingEvaluator requires FEN string")

        # Use C++ encoding: returns (8, 8, 122) NHWC format
        obs_hwc = alphazero_cpp.encode_position(fen, history_fens or [])

        # Convert to tensors — permute is zero-copy metadata swap for channels_last
        obs_tensor = torch.from_numpy(obs_hwc).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.device)
        mask_tensor = torch.from_numpy(legal_mask).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy, value, _ = self.network.predict(obs_tensor, mask_tensor)

        return policy.squeeze(0).cpu().numpy(), value.item()


class ChessWebInterface:
    """Web interface for playing against AlphaZero model."""

    def __init__(self, checkpoint_path: Optional[str] = None, num_simulations: int = 400,
                 device: str = "cuda", port: int = 5000, model_name: str = "",
                 auto_open_browser: bool = True):
        """Initialize web interface.

        Args:
            checkpoint_path: Path to model checkpoint (optional — can load later via UI)
            num_simulations: MCTS simulations per move
            device: Device to run model on
            port: Port to run web server on
            model_name: Display name for the model (defaults to checkpoint directory name)
            auto_open_browser: Whether to open browser automatically on startup
        """
        if not FLASK_AVAILABLE:
            raise ImportError("flask is required. Install with: pip install flask flask-cors")

        self.checkpoint_path = checkpoint_path
        self.num_simulations = num_simulations
        self.device = device
        self.port = port
        self.auto_open_browser = auto_open_browser

        # Require C++ backend for encoding
        if not CPP_BACKEND_AVAILABLE:
            raise ImportError(
                "C++ backend (alphazero_cpp) is required. "
                "Build it: cd alphazero-cpp && cmake -B build && cmake --build build --config Release"
            )

        # Load model if checkpoint provided, otherwise start without model
        if checkpoint_path:
            self.model_name = model_name or Path(checkpoint_path).parent.name
            self.network, self.num_filters, self.num_blocks = self._load_model()
            self.evaluator = CppEncodingEvaluator(self.network, device)
            print("Using C++ encoding (122 channels)")
        else:
            self.model_name = ""
            self.network = None
            self.evaluator = None
            self.num_filters = 0
            self.num_blocks = 0
            print("Starting without model — load one via the web UI")

        # MCTS parameters (configurable at runtime via /api/update_settings)
        self.c_puct = 1.25
        self.risk_beta = 0.0

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
            se_reduction = config.get('se_reduction', 16)
            backend = checkpoint.get('backend', 'cpp')

            wdl = config.get('wdl', True)  # Default True — all new models use WDL

            print(f"Detected C++ checkpoint (backend={backend}, version={checkpoint.get('version', '?')})")
            print(f"  input_channels={input_channels}, policy_filters={policy_filters}, value_hidden={value_hidden}, wdl={wdl}, se_r={se_reduction}")

            # Create network with C++ architecture
            network = AlphaZeroNet(
                input_channels=input_channels,
                num_filters=num_filters,
                num_blocks=num_blocks,
                policy_filters=policy_filters,
                value_hidden=value_hidden,
                wdl=wdl,
                se_reduction=se_reduction
            )
            network.load_state_dict(checkpoint['model_state_dict'])

            # Store flag to use C++ encoding
            self.use_cpp_encoding = True
            self.input_channels = input_channels
        else:
            # Python checkpoint format
            num_filters = checkpoint.get('num_filters', 192)
            num_blocks = checkpoint.get('num_blocks', 15)

            wdl = checkpoint.get('wdl', True)  # Default True for newer checkpoints

            print(f"Detected Python checkpoint (wdl={wdl})")

            # Create network with Python architecture (119 channels)
            network = AlphaZeroNet(
                input_channels=119,
                num_filters=num_filters,
                num_blocks=num_blocks,
                wdl=wdl
            )
            network.load_state_dict(checkpoint['network_state_dict'])

            self.use_cpp_encoding = False
            self.input_channels = 119

        network = network.to(self.device)
        if self.device == "cuda":
            network = network.to(memory_format=torch.channels_last)
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
                                 risk_beta=self.risk_beta,
                                 num_filters=self.num_filters,
                                 num_blocks=self.num_blocks,
                                 model_name=self.model_name,
                                 model_loaded=self.network is not None)

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
            if 'risk_beta' in data:
                val = float(data['risk_beta'])
                if -3.0 <= val <= 3.0:
                    self.risk_beta = val
            return jsonify({
                'success': True,
                'num_simulations': self.num_simulations,
                'c_puct': self.c_puct,
                'risk_beta': self.risk_beta
            })

        @self.app.route('/api/model_status', methods=['GET'])
        def model_status():
            """Get current model status."""
            return jsonify({
                'loaded': self.network is not None,
                'model_name': self.model_name,
                'num_filters': self.num_filters,
                'num_blocks': self.num_blocks
            })

        @self.app.route('/api/load_model', methods=['POST'])
        def load_model():
            """Load or swap model checkpoint at runtime."""
            data = request.json
            path = data.get('checkpoint_path', '')
            if not path:
                return jsonify({'success': False, 'error': 'No checkpoint_path provided'}), 400
            if not os.path.isfile(path):
                return jsonify({'success': False, 'error': f'File not found: {path}'}), 404
            try:
                self.checkpoint_path = path
                self.model_name = data.get('model_name', '') or Path(path).parent.name
                self.network, self.num_filters, self.num_blocks = self._load_model()
                self.evaluator = CppEncodingEvaluator(self.network, self.device)
                return jsonify({
                    'success': True,
                    'model_name': self.model_name,
                    'num_filters': self.num_filters,
                    'num_blocks': self.num_blocks
                })
            except Exception as e:
                self.network = None
                self.evaluator = None
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/new_game', methods=['POST'])
        def new_game():
            """Start a new game."""
            if self.network is None:
                return jsonify({'success': False, 'error': 'No model loaded. Load a model first.'}), 400

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
            if self.network is None:
                return jsonify({'success': False, 'error': 'No model loaded'}), 400

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
            if self.network is None:
                return jsonify({'success': False, 'error': 'No model loaded'}), 400

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

        @self.app.route('/api/export_pgn', methods=['POST'])
        def export_pgn():
            """Export current game as PGN."""
            data = request.json
            session_id = data.get('session_id', 'default')

            if session_id not in self.games:
                return jsonify({'success': False, 'error': 'Game not found'}), 404

            game = self.games[session_id]
            human_color = data.get('humanColor', 'white')

            # Build PGN from move stack
            pgn_game = chess.pgn.Game()
            pgn_game.headers["Event"] = "AlphaZero Web Game"
            pgn_game.headers["Site"] = "AlphaZero Web Interface"
            pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            pgn_game.headers["White"] = "Human" if human_color == "white" else f"AlphaZero ({self.model_name})"
            pgn_game.headers["Black"] = "Human" if human_color == "black" else f"AlphaZero ({self.model_name})"

            # Determine result
            if game.is_terminal():
                result_obj = game.get_result()
                if result_obj and result_obj.winner is True:
                    pgn_game.headers["Result"] = "1-0"
                elif result_obj and result_obj.winner is False:
                    pgn_game.headers["Result"] = "0-1"
                else:
                    pgn_game.headers["Result"] = "1/2-1/2"
            else:
                pgn_game.headers["Result"] = "*"

            # Replay moves to build the PGN game tree
            node = pgn_game
            for move in game.board.move_stack:
                node = node.add_variation(move)

            # Convert to string
            pgn_str = str(pgn_game)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alphazero_game_{timestamp}.pgn"

            return jsonify({
                'success': True,
                'pgn': pgn_str,
                'filename': filename
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
        policy, mcts_value, nn_value, wdl = self._run_cpp_mcts(game)

        # Select best action
        action = int(np.argmax(policy))

        # Convert action to move using C++ decoder (single source of truth)
        move = chess.Move.from_uci(alphazero_cpp.index_to_move(action, game.fen()))

        # Get evaluation data from policy (already from visit counts)
        evaluation_data = {
            'mcts_value': float(mcts_value),
            'nn_value': float(nn_value),
            'wdl': wdl,  # [win, draw, loss] or None
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
            Tuple of (policy, mcts_value, nn_value)
        """
        fen = game.fen()
        board = game.board
        history = game.position_history

        # Get initial evaluation (with position history for proper 122-channel encoding)
        obs_hwc = alphazero_cpp.encode_position(fen, history)

        legal_mask = np.zeros(4672, dtype=np.float32)
        for move in board.legal_moves:
            idx = alphazero_cpp.move_to_index(move.uci(), fen)
            if 0 <= idx < 4672:
                legal_mask[idx] = 1.0

        # Get root evaluation — permute is zero-copy for channels_last
        obs_tensor = torch.from_numpy(obs_hwc).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.device)
        mask_tensor = torch.from_numpy(legal_mask).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            root_policy, root_value, root_wdl = self.network.predict(obs_tensor, mask_tensor)

        root_policy = root_policy.squeeze(0).cpu().numpy().astype(np.float32)
        root_value_float = float(root_value.item())
        # Extract WDL as [win, draw, loss] list of floats (or None)
        wdl_list = root_wdl.squeeze(0).tolist() if root_wdl is not None else None

        print(f"[EVAL] FEN: {fen[:50]}... NN value: {root_value_float:.6f}")

        # Create C++ MCTS and run search
        mcts = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=self.num_simulations,
            batch_size=64,
            c_puct=self.c_puct,
            risk_beta=self.risk_beta
        )
        mcts.init_search(fen, root_policy, root_value_float, history)

        # Run MCTS with batched leaf evaluation
        while not mcts.is_complete():
            num_leaves, obs_batch, mask_batch = mcts.collect_leaves()
            if num_leaves == 0:
                break

            masks = mask_batch[:num_leaves]

            # permute NHWC → channels_last NCHW (zero-copy metadata swap)
            obs_t = torch.from_numpy(obs_batch[:num_leaves]).permute(0, 3, 1, 2).float().to(self.device)
            mask_t = torch.from_numpy(masks).float().to(self.device)

            with torch.no_grad():
                if self.device == "cuda":
                    from torch.amp import autocast
                    with autocast('cuda'):
                        leaf_policies, leaf_values, _ = self.network.predict(obs_t, mask_t)
                else:
                    leaf_policies, leaf_values, _ = self.network.predict(obs_t, mask_t)

            # Ensure float32 outputs after AMP (float16 tanh loses precision)
            leaf_policies = leaf_policies.float()
            leaf_values = leaf_values.float()

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

        return policy, float(mcts_value), root_value_float, wdl_list

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
                    move = chess.Move.from_uci(alphazero_cpp.index_to_move(int(idx), game.fen()))
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
        url = f"http://localhost:{self.port}"
        print(f"Starting chess web interface on {url}")
        if self.network is not None:
            print(f"Model: {self.num_filters} filters, {self.num_blocks} blocks")
        else:
            print("No model loaded — load one via the web UI")
        print(f"MCTS simulations: {self.num_simulations}")
        print("Press Ctrl+C to stop")

        # Auto-open browser after a short delay (gives Flask time to bind the port)
        if self.auto_open_browser and not debug:
            threading.Timer(1.5, webbrowser.open, args=[url]).start()

        self.app.run(debug=debug, port=self.port, host='0.0.0.0')


def main():
    """Run web interface from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="AlphaZero Chess Web Interface")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint (optional — can load via UI)")
    parser.add_argument("--simulations", type=int, default=400,
                       help="MCTS simulations per move")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run model on (cuda or cpu)")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port to run web server on")
    parser.add_argument("--name", type=str, default="",
                       help="Display name for the model (defaults to checkpoint directory name)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't auto-open browser on startup")

    args = parser.parse_args()

    interface = ChessWebInterface(
        checkpoint_path=args.checkpoint,
        num_simulations=args.simulations,
        device=args.device,
        port=args.port,
        model_name=args.name,
        auto_open_browser=not args.no_browser
    )
    interface.run(debug=args.debug)


if __name__ == "__main__":
    main()
