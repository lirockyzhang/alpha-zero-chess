"""Web interface for playing chess against the AlphaZero model.

Flask-based web server with chessboard.js frontend for interactive gameplay.
"""

import io
import os
import sys
import json
import math
import atexit
import chess
import chess.pgn
import chess.engine
import torch
import numpy as np
import webbrowser
import threading
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List

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

# Import Stockfish utilities from evaluation script
try:
    from evaluation import find_stockfish, StockfishEngine
    STOCKFISH_IMPORTS_AVAILABLE = True
except ImportError:
    STOCKFISH_IMPORTS_AVAILABLE = False
    find_stockfish = None
    StockfishEngine = None


class PlayerType(Enum):
    HUMAN = "human"
    MODEL = "model"
    STOCKFISH = "stockfish"


@dataclass
class GameConfig:
    white_player: PlayerType = PlayerType.HUMAN
    black_player: PlayerType = PlayerType.MODEL
    stockfish_depth: int = 20
    stockfish_elo: Optional[int] = None  # None = full strength


@dataclass
class LoadedModel:
    """A loaded model checkpoint with its network, evaluator, and metadata."""
    network: object  # AlphaZeroNet instance
    evaluator: object  # CppEncodingEvaluator instance
    name: str = ""
    num_filters: int = 0
    num_blocks: int = 0
    checkpoint_path: str = ""

# Use standalone module for game state and config
try:
    from .alphazero_standalone import GameState, MCTSConfig
except ImportError:
    from alphazero_standalone import GameState, MCTSConfig

# Try to import C++ backend for 123-channel encoding
try:
    cpp_build_path = str(Path(__file__).parent.parent / "alphazero-cpp" / "build" / "Release")
    if cpp_build_path not in sys.path:
        sys.path.insert(0, cpp_build_path)
    import alphazero_cpp
    CPP_BACKEND_AVAILABLE = True
except ImportError:
    CPP_BACKEND_AVAILABLE = False


class CppEncodingEvaluator:
    """Evaluator that uses C++ 123-channel encoding for C++ checkpoint compatibility."""

    def __init__(self, network, device: str = "cuda"):
        """Initialize evaluator with C++ encoding.

        Args:
            network: AlphaZero network (123 input channels)
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

        # Use C++ encoding: returns (8, 8, 123) NHWC format
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

        # Per-slot model storage: keyed by "white", "black"
        self.models: Dict[str, LoadedModel] = {}

        # Load model if checkpoint provided, otherwise start without model
        if checkpoint_path:
            name = model_name or Path(checkpoint_path).parent.name
            loaded = self._load_model_from_path(checkpoint_path, name)
            # Load into both slots so Human vs Model works out of the box
            self.models["white"] = loaded
            self.models["black"] = loaded
            print("Using C++ encoding (123 channels)")
        else:
            print("Starting without model — load one via the web UI")

        # MCTS parameters (configurable at runtime via /api/update_settings)
        self.c_puct = 1.25
        self.risk_beta = 0.0

        # Game state storage (in-memory, keyed by session ID)
        self.games: Dict[str, GameState] = {}

        # Game config storage (player types per session)
        self.game_configs: Dict[str, GameConfig] = {}

        # Stockfish engine management
        self.stockfish_path = find_stockfish() if find_stockfish else None
        self.stockfish_engine: Optional[object] = None  # StockfishEngine instance
        self._stockfish_lock = threading.Lock()
        atexit.register(self._shutdown_stockfish)

        # Create Flask app with paths relative to this file
        template_folder = str(Path(__file__).parent / 'templates')
        static_folder = str(Path(__file__).parent / 'static')

        self.app = Flask(__name__,
                        template_folder=template_folder,
                        static_folder=static_folder)
        CORS(self.app)
        self._setup_routes()

    def _load_model_from_path(self, path: str, name: str = "") -> 'LoadedModel':
        """Load model from checkpoint path, returning a LoadedModel.

        Deduplicates: if the same path is already loaded in another slot,
        returns the existing LoadedModel (avoids double GPU memory).

        Supports both Python and C++ checkpoint formats:
        - Python: 'network_state_dict', 'num_filters', 'num_blocks'
        - C++: 'model_state_dict', config dict with all params
        """
        # Deduplication: reuse existing model if same path already loaded
        abs_path = os.path.abspath(path)
        for existing in self.models.values():
            if os.path.abspath(existing.checkpoint_path) == abs_path:
                print(f"Reusing already-loaded model for {path}")
                # Return a new LoadedModel sharing the same network/evaluator but with updated name
                return LoadedModel(
                    network=existing.network,
                    evaluator=existing.evaluator,
                    name=name or existing.name,
                    num_filters=existing.num_filters,
                    num_blocks=existing.num_blocks,
                    checkpoint_path=path,
                )

        print(f"Loading model from {path}...")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Detect checkpoint format
        is_cpp_checkpoint = 'model_state_dict' in checkpoint

        if is_cpp_checkpoint:
            # C++ checkpoint format
            config = checkpoint.get('config', {})
            num_filters = config.get('num_filters', 192)
            num_blocks = config.get('num_blocks', 15)
            input_channels = config.get('input_channels', 123)
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
        else:
            # Python checkpoint format
            num_filters = checkpoint.get('num_filters', 192)
            num_blocks = checkpoint.get('num_blocks', 15)
            input_channels = 119

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

        network = network.to(self.device)
        if self.device == "cuda":
            network = network.to(memory_format=torch.channels_last)
        network.eval()

        evaluator = CppEncodingEvaluator(network, self.device)
        model_name = name or Path(path).parent.name

        print(f"Model loaded: {num_filters} filters, {num_blocks} blocks")
        return LoadedModel(
            network=network,
            evaluator=evaluator,
            name=model_name,
            num_filters=num_filters,
            num_blocks=num_blocks,
            checkpoint_path=path,
        )

    def _model_slot_info(self, slot: str) -> dict:
        """Get JSON-serializable info dict for a model slot."""
        model = self.models.get(slot)
        if model:
            return {
                'loaded': True,
                'name': model.name,
                'num_filters': model.num_filters,
                'num_blocks': model.num_blocks,
                'checkpoint_path': model.checkpoint_path,
            }
        return {'loaded': False, 'name': '', 'num_filters': 0, 'num_blocks': 0, 'checkpoint_path': ''}

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            """Serve main page."""
            any_loaded = len(self.models) > 0
            return render_template('chess.html',
                                 num_simulations=self.num_simulations,
                                 c_puct=self.c_puct,
                                 risk_beta=self.risk_beta,
                                 model_loaded=any_loaded,
                                 white_model=self._model_slot_info("white"),
                                 black_model=self._model_slot_info("black"),
                                 stockfish_available=self.stockfish_path is not None)

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
            """Get per-slot model status."""
            return jsonify({
                'white': self._model_slot_info("white"),
                'black': self._model_slot_info("black"),
                'any_loaded': len(self.models) > 0,
            })

        @self.app.route('/api/load_model', methods=['POST'])
        def load_model():
            """Load or swap model checkpoint at runtime.

            Accepts optional 'slot' param: "white", "black", or "both" (default).
            """
            data = request.json
            path = data.get('checkpoint_path', '')
            slot = data.get('slot', 'both')
            if not path:
                return jsonify({'success': False, 'error': 'No checkpoint_path provided'}), 400
            if not os.path.isfile(path):
                return jsonify({'success': False, 'error': f'File not found: {path}'}), 404
            if slot not in ('white', 'black', 'both'):
                return jsonify({'success': False, 'error': f'Invalid slot: {slot}'}), 400
            try:
                name = data.get('model_name', '') or Path(path).parent.name
                loaded = self._load_model_from_path(path, name)
                if slot == 'both':
                    self.models["white"] = loaded
                    self.models["black"] = loaded
                else:
                    self.models[slot] = loaded
                return jsonify({
                    'success': True,
                    'slot': slot,
                    'model_name': loaded.name,
                    'num_filters': loaded.num_filters,
                    'num_blocks': loaded.num_blocks,
                    'white': self._model_slot_info("white"),
                    'black': self._model_slot_info("black"),
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/new_game', methods=['POST'])
        def new_game():
            """Start a new game with unified player configuration."""
            data = request.json
            session_id = data.get('session_id', 'default')

            # Parse player types (backward compat: 'color' param → human vs model)
            white_type_str = data.get('white_player', None)
            black_type_str = data.get('black_player', None)

            if white_type_str and black_type_str:
                # New unified API
                try:
                    white_player = PlayerType(white_type_str)
                    black_player = PlayerType(black_type_str)
                except ValueError as e:
                    return jsonify({'success': False, 'error': f'Invalid player type: {e}'}), 400
            else:
                # Legacy API: color param
                human_color = data.get('color', 'white')
                white_player = PlayerType.HUMAN if human_color == 'white' else PlayerType.MODEL
                black_player = PlayerType.MODEL if human_color == 'white' else PlayerType.HUMAN

            # Validate: model must be loaded in the correct slot
            if white_player == PlayerType.MODEL and "white" not in self.models:
                return jsonify({'success': False, 'error': 'No model loaded for White. Load a checkpoint first.'}), 400
            if black_player == PlayerType.MODEL and "black" not in self.models:
                return jsonify({'success': False, 'error': 'No model loaded for Black. Load a checkpoint first.'}), 400

            # Validate: stockfish must be available if selected
            if PlayerType.STOCKFISH in (white_player, black_player):
                if not self.stockfish_path or not STOCKFISH_IMPORTS_AVAILABLE:
                    return jsonify({'success': False, 'error': 'Stockfish is not installed or not found.'}), 400

            # Build game config
            config = GameConfig(
                white_player=white_player,
                black_player=black_player,
                stockfish_depth=int(data.get('stockfish_depth', 20)),
                stockfish_elo=int(data['stockfish_elo']) if data.get('stockfish_elo') else None,
            )
            self.game_configs[session_id] = config

            # Initialize Stockfish if needed
            if PlayerType.STOCKFISH in (white_player, black_player):
                try:
                    self._ensure_stockfish(config)
                except Exception as e:
                    return jsonify({'success': False, 'error': f'Stockfish init failed: {e}'}), 500

            # Create new game
            self.games[session_id] = GameState()

            # If White is an AI, auto-make the first move
            if white_player != PlayerType.HUMAN:
                try:
                    move_uci, eval_data = self._get_next_move(session_id)
                    return jsonify({
                        'success': True,
                        'fen': self.games[session_id].board.fen(),
                        'ai_move': move_uci,
                        'evaluation': eval_data,
                        'game_over': False,
                        'config': {'white': white_player.value, 'black': black_player.value}
                    })
                except Exception as e:
                    return jsonify({'success': False, 'error': str(e)}), 500

            return jsonify({
                'success': True,
                'fen': self.games[session_id].board.fen(),
                'game_over': False,
                'config': {'white': white_player.value, 'black': black_player.value}
            })

        @self.app.route('/api/make_move', methods=['POST'])
        def make_move():
            """Make a human move, then auto-respond if opponent is AI."""
            data = request.json
            session_id = data.get('session_id', 'default')
            move_uci = data.get('move')

            if session_id not in self.games:
                return jsonify({'success': False, 'error': 'Game not found'}), 404

            game = self.games[session_id]
            config = self.game_configs.get(session_id, GameConfig())

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

                # Determine opponent type
                is_white_turn = game.board.turn == chess.WHITE
                opponent = config.white_player if is_white_turn else config.black_player

                if opponent == PlayerType.HUMAN:
                    # Human vs Human: no auto-response
                    return jsonify({
                        'success': True,
                        'fen': game.board.fen(),
                        'game_over': False,
                    })

                # Opponent is AI — auto-respond
                ai_move_uci, eval_data = self._get_next_move(session_id)
                game = self.games[session_id]

                game_over = game.is_terminal()
                result = None
                if game_over:
                    result = self._format_result(game.get_result())

                return jsonify({
                    'success': True,
                    'fen': game.board.fen(),
                    'ai_move': ai_move_uci,
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

        @self.app.route('/api/ai_move', methods=['POST'])
        @self.app.route('/api/self_play_move', methods=['POST'])  # backward compat alias
        def ai_move():
            """Make one AI move (for any game with an AI player's turn)."""
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

            try:
                move_uci, eval_data = self._get_next_move(session_id)
            except ValueError:
                # It's a human's turn — nothing to do
                return jsonify({
                    'success': True,
                    'fen': game.board.fen(),
                    'game_over': False,
                    'waiting_for_human': True,
                })

            game = self.games[session_id]

            game_over = game.is_terminal()
            result = None
            if game_over:
                result = self._format_result(game.get_result())

            return jsonify({
                'success': True,
                'fen': game.board.fen(),
                'ai_move': move_uci,
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
            config = self.game_configs.get(session_id, GameConfig())

            # Build PGN from move stack
            pgn_game = chess.pgn.Game()
            pgn_game.headers["Event"] = "AlphaZero Web Game"
            pgn_game.headers["Site"] = "AlphaZero Web Interface"
            pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            pgn_game.headers["White"] = self._player_display_name(config.white_player, config, "white")
            pgn_game.headers["Black"] = self._player_display_name(config.black_player, config, "black")

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

        @self.app.route('/api/list_checkpoints', methods=['GET'])
        def list_checkpoints():
            """Scan checkpoints directory for available model files."""
            checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
            results = []

            if checkpoints_dir.exists():
                for run_dir in sorted(checkpoints_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    run_name = run_dir.name
                    # Find model files in this run
                    for pt_file in sorted(run_dir.glob("*.pt")):
                        results.append({
                            'run': run_name,
                            'filename': pt_file.name,
                            'path': str(pt_file),
                            'display': f"{run_name}/{pt_file.name}"
                        })

            return jsonify({'success': True, 'checkpoints': results})

        @self.app.route('/api/stockfish_status', methods=['GET'])
        def stockfish_status():
            """Check if Stockfish is available."""
            return jsonify({
                'available': self.stockfish_path is not None,
                'path': self.stockfish_path or ''
            })

    def _get_model_move(self, session_id: str) -> tuple:
        """Get model's move for current position.

        Uses the model from the slot matching the current turn (white/black).

        Args:
            session_id: Game session ID

        Returns:
            Tuple of (move_uci, evaluation_data) where evaluation_data contains
            value estimate and top move probabilities
        """
        game = self.games[session_id]
        slot = "white" if game.board.turn == chess.WHITE else "black"
        model = self.models[slot]

        # Run C++ MCTS with batched leaf evaluation
        policy, mcts_value, nn_value, wdl = self._run_cpp_mcts(game, model)

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

    def _run_cpp_mcts(self, game: GameState, model: 'LoadedModel') -> tuple:
        """Run C++ MCTS with batched leaf evaluation.

        Args:
            game: Current game state
            model: LoadedModel to use for evaluation

        Returns:
            Tuple of (policy, mcts_value, nn_value, wdl_list)
        """
        fen = game.fen()
        board = game.board
        history = game.position_history
        network = model.network

        # Get initial evaluation (with position history for proper 123-channel encoding)
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
            root_policy, root_value, root_wdl = network.predict(obs_tensor, mask_tensor)

        root_policy = root_policy.squeeze(0).cpu().numpy().astype(np.float32)
        root_value_float = float(root_value.item())
        # Extract WDL as [win, draw, loss] list of floats (or None)
        wdl_list = root_wdl.squeeze(0).tolist() if root_wdl is not None else None

        print(f"[EVAL] FEN: {fen[:50]}... NN value: {root_value_float:.6f}")

        # Create C++ MCTS and run search
        # batch_size=1: each simulation fully completes before the next,
        # so PUCT always sees the up-to-date tree (no virtual loss needed)
        mcts = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=self.num_simulations,
            batch_size=1,
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
                        leaf_policies, leaf_values, _ = network.predict(obs_t, mask_t)
                else:
                    leaf_policies, leaf_values, _ = network.predict(obs_t, mask_t)

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

    def _ensure_stockfish(self, config: GameConfig):
        """Lazy-start or reconfigure Stockfish engine based on GameConfig."""
        with self._stockfish_lock:
            if self.stockfish_engine is not None:
                # Shut down existing engine to reconfigure
                try:
                    self.stockfish_engine.close()
                except Exception:
                    pass
                self.stockfish_engine = None

            if not self.stockfish_path or not STOCKFISH_IMPORTS_AVAILABLE:
                raise RuntimeError("Stockfish is not available")

            self.stockfish_engine = StockfishEngine(
                path=self.stockfish_path,
                elo=config.stockfish_elo,
                depth=config.stockfish_depth,
            )

    def _get_stockfish_move(self, session_id: str) -> tuple:
        """Get Stockfish's move for current position.

        Returns:
            Tuple of (move_uci, evaluation_data) matching _get_model_move format.
        """
        game = self.games[session_id]
        board = game.board

        with self._stockfish_lock:
            if self.stockfish_engine is None:
                raise RuntimeError("Stockfish engine not initialized")

            # Get best move
            best_move = self.stockfish_engine.get_move(board)

            # Get evaluation (value in [-1, 1])
            value = self.stockfish_engine.evaluate(board)

            # Get top moves via multi-PV analysis
            config = self.game_configs.get(session_id, GameConfig())
            limit = chess.engine.Limit(depth=config.stockfish_depth)
            analysis = self.stockfish_engine.engine.analyse(board, limit, multipv=5)

        # Convert centipawn scores to WDL approximation
        # Using logistic model: win_prob = 1 / (1 + exp(-cp/200))
        best_cp = None
        top_moves = []
        for info in analysis:
            pv = info.get("pv")
            score = info.get("score")
            if not pv or not score:
                continue
            move = pv[0]
            cp = score.white().score(mate_score=10000)
            if cp is None:
                continue
            if best_cp is None:
                best_cp = cp
            win_prob = 1.0 / (1.0 + math.exp(-cp / 200.0))
            top_moves.append({
                'move': move.uci(),
                'move_san': board.san(move),
                'probability': float(cp),  # raw centipawn for Stockfish
                'visits': cp,  # display as centipawns
                'cp': cp,
            })

        # WDL from best centipawn score
        if best_cp is not None:
            wp = 1.0 / (1.0 + math.exp(-best_cp / 200.0))
            # Approximate draw probability using a peaked distribution around cp=0
            draw_factor = math.exp(-(best_cp / 400.0) ** 2)
            wp_adj = wp * (1 - 0.3 * draw_factor)
            lp_adj = (1 - wp) * (1 - 0.3 * draw_factor)
            dp = 1.0 - wp_adj - lp_adj
            wdl_list = [wp_adj, dp, lp_adj]
        else:
            wdl_list = [0.33, 0.34, 0.33]

        evaluation_data = {
            'mcts_value': float(value),
            'nn_value': float(value),
            'wdl': wdl_list,
            'top_moves': top_moves[:5],
            'engine': 'stockfish',
        }

        # Apply move
        self.games[session_id] = game.apply_move(best_move)

        return best_move.uci(), evaluation_data

    def _get_next_move(self, session_id: str) -> tuple:
        """Dispatch to the correct engine based on GameConfig and current turn.

        Returns:
            Tuple of (move_uci, evaluation_data).
        """
        game = self.games[session_id]
        config = self.game_configs.get(session_id, GameConfig())

        # Determine which player is to move
        is_white_turn = game.board.turn == chess.WHITE
        current_player = config.white_player if is_white_turn else config.black_player

        if current_player == PlayerType.MODEL:
            return self._get_model_move(session_id)
        elif current_player == PlayerType.STOCKFISH:
            return self._get_stockfish_move(session_id)
        else:
            raise ValueError("Cannot auto-move for a human player")

    def _shutdown_stockfish(self):
        """Clean shutdown of Stockfish engine (called via atexit)."""
        with self._stockfish_lock:
            if self.stockfish_engine is not None:
                try:
                    self.stockfish_engine.close()
                except Exception:
                    pass
                self.stockfish_engine = None

    def _player_display_name(self, player_type: PlayerType, config: GameConfig, side: str = "white") -> str:
        """Get display name for a player type (for PGN headers).

        Args:
            player_type: Type of player
            config: Game configuration
            side: "white" or "black" — used to look up the correct model slot
        """
        if player_type == PlayerType.HUMAN:
            return "Human"
        elif player_type == PlayerType.MODEL:
            model = self.models.get(side)
            name = model.name if model else ""
            return f"AlphaZero ({name})" if name else "AlphaZero"
        elif player_type == PlayerType.STOCKFISH:
            elo_str = f", ELO {config.stockfish_elo}" if config.stockfish_elo else ""
            return f"Stockfish (depth {config.stockfish_depth}{elo_str})"
        return "Unknown"

    def run(self, debug: bool = False):
        """Run the web server.

        Args:
            debug: Enable debug mode
        """
        url = f"http://localhost:{self.port}"
        print(f"Starting chess web interface on {url}")
        if self.models:
            for slot, m in self.models.items():
                print(f"  {slot.capitalize()} model: {m.name} ({m.num_filters}f/{m.num_blocks}b)")
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
