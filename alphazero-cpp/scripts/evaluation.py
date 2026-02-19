#!/usr/bin/env python3
"""
Standalone Evaluation CLI for AlphaZero Checkpoints

This script runs evaluation independently of training. It uses a registry
pattern for easy addition of new evaluators.

Usage Examples:
    # Evaluate a specific checkpoint
    python evaluation.py --checkpoint checkpoints/f192-b15_2024-02-03/model_iter_010.pt

    # Evaluate latest checkpoint in a run directory
    python evaluation.py --checkpoint checkpoints/f192-b15_2024-02-03/

    # Run specific evaluators only
    python evaluation.py --checkpoint model.pt --evaluators vs_random endgame

    # Evaluate against Stockfish (full strength, depth 20)
    python evaluation.py --checkpoint model.pt --evaluators vs_stockfish

    # Evaluate against Stockfish at limited ELO
    python evaluation.py --checkpoint model.pt --evaluators vs_stockfish --stockfish-elo 1500

    # Evaluate against Stockfish at low depth (weaker)
    python evaluation.py --checkpoint model.pt --evaluators vs_stockfish --stockfish-depth 5

    # Custom MCTS settings
    python evaluation.py --checkpoint model.pt --simulations 1600 --search-batch 64

    # Save results to JSON
    python evaluation.py --checkpoint model.pt --output results.json

    # List available evaluators
    python evaluation.py --list
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
import random
import glob
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
from torch.amp import autocast

# Add paths for alphazero_cpp module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "build" / "Release"))

try:
    import alphazero_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("ERROR: alphazero_cpp not found. Build it first:")
    print("  cd alphazero-cpp")
    print("  cmake -B build -DCMAKE_BUILD_TYPE=Release")
    print("  cmake --build build --config Release")
    sys.exit(1)

import chess
import chess.engine

# Import shared network architecture
from network import AlphaZeroNet, INPUT_CHANNELS, POLICY_SIZE


# =============================================================================
# Evaluation Configuration
# =============================================================================

@dataclass
class EvalConfig:
    """Configuration for running evaluators."""
    simulations: int = 800
    search_batch: int = 32
    c_puct: float = 1.5
    risk_beta: float = 0.0
    device: str = "cuda"
    stockfish_path: Optional[str] = None    # Path to binary (None = auto-detect)
    stockfish_elo: Optional[int] = None     # ELO limit (None = full strength)
    stockfish_depth: Optional[int] = None   # Depth limit (None = use default 20)


# =============================================================================
# Evaluation Result
# =============================================================================

@dataclass
class EvaluationResult:
    """Standard result format for any evaluator."""
    name: str                     # e.g., "vs_random", "endgame_puzzles"
    score: float                  # Primary metric (0.0-1.0)
    details: Dict[str, Any]       # Evaluator-specific details
    display_str: str              # Human-readable summary for console


# =============================================================================
# Batched Evaluator (GPU)
# =============================================================================

class BatchedEvaluator:
    """Efficient batched neural network evaluation on GPU."""

    def __init__(self, network: torch.nn.Module, device: str, use_amp: bool = True):
        self.network = network
        self.device = device
        self.use_amp = use_amp and device == "cuda"

    @torch.inference_mode()
    def evaluate(self, obs: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Evaluate single position. obs is NHWC (8, 8, C)."""
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.device)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)

        if self.use_amp:
            with autocast('cuda'):
                policy, value, _, _ = self.network(obs_tensor, mask_tensor)
        else:
            policy, value, _, _ = self.network(obs_tensor, mask_tensor)

        return policy[0].cpu().numpy(), float(value[0].item())

    @torch.inference_mode()
    def evaluate_batch(self, obs_batch: np.ndarray, mask_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate batch of positions. obs_batch is NHWC (N, 8, 8, C)."""
        obs_tensor = torch.from_numpy(obs_batch).permute(0, 3, 1, 2).float().to(self.device)
        mask_tensor = torch.from_numpy(mask_batch).float().to(self.device)

        if self.use_amp:
            with autocast('cuda'):
                policies, values, _, _ = self.network(obs_tensor, mask_tensor)
        else:
            policies, values, _, _ = self.network(obs_tensor, mask_tensor)

        return policies.cpu().numpy(), values.squeeze(-1).cpu().numpy()


# =============================================================================
# Stockfish Engine Wrapper
# =============================================================================

def find_stockfish() -> Optional[str]:
    """Auto-detect Stockfish binary in common locations.

    Search order:
        1. alphazero-cpp/bin/stockfish[.exe]  (our install location)
        2. System PATH via shutil.which()
    """
    # Check our install directory
    script_dir = Path(__file__).resolve().parent
    bin_dir = script_dir.parent / "bin"
    if sys.platform == "win32":
        local_path = bin_dir / "stockfish.exe"
    else:
        local_path = bin_dir / "stockfish"
    if local_path.exists():
        return str(local_path)

    # Check system PATH
    system_path = shutil.which("stockfish")
    if system_path:
        return system_path

    return None


class StockfishEngine:
    """Reusable Stockfish wrapper for evaluation and future self-play.

    Communicates with a Stockfish binary via UCI protocol using python-chess's
    chess.engine module. Supports ELO-limited play, depth-limited search,
    and multi-PV analysis for policy extraction.
    """

    def __init__(self, path: str, elo: Optional[int] = None,
                 depth: Optional[int] = None, time_limit: Optional[float] = None):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)

        # Configure ELO limitation if requested
        if elo is not None:
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

        self.depth = depth
        self.time_limit = time_limit

    def _make_limit(self) -> chess.engine.Limit:
        """Build a search limit from configured depth/time."""
        if self.depth is not None:
            return chess.engine.Limit(depth=self.depth)
        elif self.time_limit is not None:
            return chess.engine.Limit(time=self.time_limit)
        else:
            return chess.engine.Limit(depth=20)

    def get_move(self, board: chess.Board) -> chess.Move:
        """Get Stockfish's best move for the given position."""
        result = self.engine.play(board, self._make_limit())
        return result.move

    def get_policy(self, board: chess.Board, num_pvs: int = 10) -> np.ndarray:
        """Extract a policy distribution from Stockfish's multi-PV analysis.

        Runs Stockfish with MultiPV, converts centipawn scores to a softmax
        probability distribution over the 4672-dim action space. This allows
        Stockfish's move preferences to be used as training targets.
        """
        analysis = self.engine.analyse(board, self._make_limit(), multipv=num_pvs)

        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        scores = []
        indices = []

        for info in analysis:
            pv = info.get("pv")
            score = info.get("score")
            if not pv or not score:
                continue

            move = pv[0]
            cp = score.white().score(mate_score=10000)
            if cp is None:
                continue

            fen = board.fen()
            idx = alphazero_cpp.move_to_index(move.uci(), fen)
            if 0 <= idx < POLICY_SIZE:
                scores.append(cp / 100.0)  # Scale centipawns for softmax
                indices.append(idx)

        if indices:
            # Softmax over the Stockfish-scored moves
            scores_arr = np.array(scores, dtype=np.float32)
            scores_arr -= scores_arr.max()  # Numerical stability
            exp_scores = np.exp(scores_arr)
            probs = exp_scores / exp_scores.sum()

            for idx, prob in zip(indices, probs):
                policy[idx] = prob

        return policy

    def evaluate(self, board: chess.Board) -> float:
        """Get Stockfish's position evaluation as a value in [-1, +1].

        Converts centipawn scores via tanh(cp / 300) and maps mate scores
        directly to +/-1.0.
        """
        info = self.engine.analyse(board, self._make_limit())
        score = info["score"].white()
        cp = score.score(mate_score=10000)
        if cp is None:
            return 0.0
        if abs(cp) >= 10000:
            return 1.0 if cp > 0 else -1.0
        return math.tanh(cp / 300.0)

    def close(self):
        """Shut down the Stockfish process."""
        self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# Evaluator Registry
# =============================================================================

EVALUATORS: Dict[str, 'Evaluator'] = {}


def register_evaluator(name: str):
    """Decorator to register evaluators."""
    def decorator(cls):
        EVALUATORS[name] = cls()
        return cls
    return decorator


class Evaluator(ABC):
    """Base class for all evaluators."""
    name: str
    description: str

    @abstractmethod
    def run(self, network: torch.nn.Module, device: str, config: EvalConfig) -> EvaluationResult:
        """Run the evaluation and return results."""
        pass


# =============================================================================
# Endgame Puzzles Data
# =============================================================================

# Endgame puzzles for testing model strength
# Each puzzle has a move_sequence: alternating model/opponent moves.
# Indices 0, 2, 4, ... are model moves (verified against the network's prediction).
# Indices 1, 3, 5, ... are opponent responses (played to advance the board).
# Scoring: consecutive correct model moves from the start (prefix match).
ENDGAME_PUZZLES = [
    # Mate in 2: Queen + King coordination (forced line)
    # After Qa1+ the only legal king move is Kb8 (a7/b7 controlled by Kb6), then Qa8#
    {"fen": "k7/8/1K6/8/8/8/8/7Q w - - 0 1",
     "move_sequence": ["Qa1+", "Kb8", "Qa8#"], "type": "KQ_mate_in_2"},

    # Mate in 2: Back rank with two rooks
    # Ra8+ forces king to h8 corner (f7/g7/h7 pawns block), then Rb8# is mate
    {"fen": "6k1/5ppp/8/8/8/8/8/RR4K1 w - - 0 1",
     "move_sequence": ["Ra8+", "Kh8", "Rb8#"], "type": "back_rank_mate_in_2"},

    # Mate in 1: Scholar's mate
    {"fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
     "move_sequence": ["Qxf7#"], "type": "scholars_mate"},

    # Best move: Back rank check
    {"fen": "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1",
     "move_sequence": ["Ra8+"], "type": "back_rank"},

    # Best move: Pawn promotion
    {"fen": "8/4P3/8/8/4k3/8/8/4K3 w - - 0 1",
     "move_sequence": ["e8=Q"], "type": "pawn_promo"},
]


# =============================================================================
# Evaluator Implementations
# =============================================================================

@register_evaluator("vs_random")
class VsRandomEvaluator(Evaluator):
    """Play games against a random opponent to test basic competence."""

    name = "vs_random"
    description = "Play games against random opponent (batched parallel)"

    def run(self, network: torch.nn.Module, device: str, config: EvalConfig) -> EvaluationResult:
        """Run games against random opponent."""
        network.eval()
        evaluator = BatchedEvaluator(network, device, use_amp=(device == "cuda"))

        num_games = 5
        # Cap simulations at 400 - if model can't beat random with 400, more won't help
        eval_sims = min(config.simulations, 400)
        eval_search_batch = min(config.search_batch, 32)

        print(f"    Running {num_games} games (batched)...", end=" ", flush=True)
        start_time = time.time()

        results = self._play_games(
            evaluator, num_games, eval_sims, eval_search_batch,
            config.c_puct, config.risk_beta
        )

        elapsed = time.time() - start_time
        result_chars = []
        for g in sorted(results["games"], key=lambda x: x["game_idx"]):
            if g["result"] == "win":
                result_chars.append("W")
            elif g["result"] == "loss":
                result_chars.append("L")
            else:
                result_chars.append("D")

        print(f"{''.join(result_chars)} ({elapsed:.1f}s)")

        wins = results["wins"]
        losses = results["losses"]
        draws = results["draws"]
        win_rate = wins / num_games

        return EvaluationResult(
            name=self.name,
            score=win_rate,
            details={
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "games": num_games,
                "simulations": eval_sims,
            },
            display_str=f"vs Random: {wins}/{num_games} wins ({win_rate*100:.0f}%)"
        )

    def _play_games(self, evaluator: BatchedEvaluator, num_games: int,
                    simulations: int, search_batch: int, c_puct: float,
                    risk_beta: float = 0.0) -> Dict[str, Any]:
        """Play games with batched evaluation across all games."""

        class GameState:
            def __init__(self, game_idx: int):
                self.game_idx = game_idx
                self.board = chess.Board()
                self.model_plays_white = (game_idx % 2 == 0)
                self.move_count = 0
                self.position_history = []  # FEN history for encoder (up to 8 positions)
                self.mcts = alphazero_cpp.BatchedMCTSSearch(
                    num_simulations=simulations,
                    batch_size=search_batch,
                    c_puct=c_puct,
                    risk_beta=risk_beta
                )
                self.result = None
                self.needs_root_eval = False
                self.in_mcts_search = False
                self.current_fen = None
                self.current_obs = None
                self.current_mask = None
                self.current_idx_to_move = {}

        def _track_history(g):
            """Append current board FEN to history (before making a move)."""
            g.position_history.append(g.board.fen())
            if len(g.position_history) > 8:
                g.position_history.pop(0)

        games = [GameState(i) for i in range(num_games)]
        active_games = games.copy()

        def prepare_move(g: GameState):
            g.current_fen = g.board.fen()
            obs = alphazero_cpp.encode_position(g.current_fen, g.position_history)
            g.current_obs = obs

            legal_moves = list(g.board.legal_moves)
            g.current_mask = np.zeros(POLICY_SIZE, dtype=np.float32)
            g.current_idx_to_move = {}

            for move in legal_moves:
                idx = alphazero_cpp.move_to_index(move.uci(), g.current_fen)
                if 0 <= idx < POLICY_SIZE:
                    g.current_mask[idx] = 1.0
                    g.current_idx_to_move[idx] = move

            g.needs_root_eval = True
            g.in_mcts_search = False

        def finish_move(g: GameState):
            _track_history(g)
            visit_counts = g.mcts.get_visit_counts()
            policy = visit_counts * g.current_mask
            if policy.sum() > 0:
                action = np.argmax(policy)
                if action in g.current_idx_to_move:
                    g.board.push(g.current_idx_to_move[action])
                else:
                    print(f"\n    WARNING: Game {g.game_idx} move {g.move_count}: "
                          f"argmax action {action} not in idx_to_move, playing random", flush=True)
                    g.board.push(random.choice(list(g.board.legal_moves)))
            else:
                legal = list(g.board.legal_moves)
                if legal:
                    print(f"\n    WARNING: Game {g.game_idx} move {g.move_count}: "
                          f"zero visit counts, playing random", flush=True)
                    g.board.push(random.choice(legal))
            g.mcts.reset()
            g.move_count += 1
            g.in_mcts_search = False
            g.needs_root_eval = False

        def check_game_over(g: GameState) -> bool:
            if g.board.is_game_over():
                result = g.board.result()
                if result == "1-0":
                    g.result = "win" if g.model_plays_white else "loss"
                elif result == "0-1":
                    g.result = "loss" if g.model_plays_white else "win"
                else:
                    g.result = "draw"
                return True
            if g.move_count >= 200:
                g.result = "draw"
                return True
            return False

        def random_opponent_move(g: GameState):
            """Play a random legal move for the opponent."""
            _track_history(g)
            legal = list(g.board.legal_moves)
            if legal:
                g.board.push(random.choice(legal))
            g.move_count += 1

        # Initialize: advance to first model move for each game
        for g in active_games:
            while not check_game_over(g):
                is_model_turn = (g.board.turn == chess.WHITE) == g.model_plays_white
                if is_model_turn:
                    prepare_move(g)
                    break
                else:
                    random_opponent_move(g)

        # Main loop: batch evaluations across all active games
        while active_games:
            # 1. Collect games needing root evaluation
            root_eval_games = [g for g in active_games if g.needs_root_eval and g.result is None]
            if root_eval_games:
                obs_batch = np.stack([g.current_obs for g in root_eval_games])
                mask_batch = np.stack([g.current_mask for g in root_eval_games])
                policies, values = evaluator.evaluate_batch(obs_batch, mask_batch)

                for i, g in enumerate(root_eval_games):
                    g.mcts.init_search(g.current_fen, policies[i].astype(np.float32), float(values[i]), g.position_history)
                    g.needs_root_eval = False
                    g.in_mcts_search = True

            # 2. Collect leaves from all games in MCTS search
            games_in_search = [g for g in active_games if g.in_mcts_search and g.result is None]
            if not games_in_search:
                active_games = [g for g in active_games if g.result is None]
                continue

            all_obs = []
            all_masks = []
            leaf_counts = []

            for g in games_in_search:
                if g.mcts.is_complete():
                    leaf_counts.append(0)
                    continue

                num_leaves, obs_batch, mask_batch = g.mcts.collect_leaves()
                if num_leaves == 0:
                    leaf_counts.append(0)
                    continue

                all_obs.append(obs_batch[:num_leaves])
                all_masks.append(mask_batch[:num_leaves])
                leaf_counts.append(num_leaves)

            # 3. Batch evaluate all leaves together
            if all_obs:
                combined_obs = np.concatenate(all_obs, axis=0)
                combined_masks = np.concatenate(all_masks, axis=0)
                all_policies, all_values = evaluator.evaluate_batch(combined_obs, combined_masks)

                # 4. Distribute results back to each game
                offset = 0
                leaf_idx = 0
                for g in games_in_search:
                    count = leaf_counts[leaf_idx]
                    leaf_idx += 1
                    if count == 0:
                        continue
                    game_policies = all_policies[offset:offset + count]
                    game_values = all_values[offset:offset + count]
                    g.mcts.update_leaves(game_policies.astype(np.float32), game_values.astype(np.float32))
                    offset += count

            # 5. Check for completed searches and advance games
            for g in games_in_search:
                if g.mcts.is_complete():
                    finish_move(g)

                    if check_game_over(g):
                        continue

                    while not check_game_over(g):
                        is_model_turn = (g.board.turn == chess.WHITE) == g.model_plays_white
                        if is_model_turn:
                            prepare_move(g)
                            break
                        else:
                            random_opponent_move(g)

            active_games = [g for g in active_games if g.result is None]

        # Tally results
        wins = sum(1 for g in games if g.result == "win")
        losses = sum(1 for g in games if g.result == "loss")
        draws = sum(1 for g in games if g.result == "draw")

        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "games": [{"game_idx": g.game_idx, "result": g.result} for g in games]
        }


@register_evaluator("endgame")
class EndgamePuzzleEvaluator(Evaluator):
    """Test model on endgame puzzle positions with move sequences."""

    name = "endgame"
    description = "Test on endgame puzzle positions (sequence-based scoring)"

    def run(self, network: torch.nn.Module, device: str, config: EvalConfig) -> EvaluationResult:
        """Run endgame puzzle evaluation."""
        network.eval()
        evaluator = BatchedEvaluator(network, device, use_amp=(device == "cuda"))

        puzzles = ENDGAME_PUZZLES
        puzzles_fully_correct = 0
        total_model_moves = 0
        total_consecutive_correct = 0
        details = []

        for puzzle in puzzles:
            fen = puzzle["fen"]
            move_sequence = puzzle["move_sequence"]
            puzzle_type = puzzle["type"]

            board = chess.Board(fen)

            num_model_moves = len(range(0, len(move_sequence), 2))
            consecutive_correct = 0
            sequence_broken = False
            predicted_moves = []

            for i, expected_san in enumerate(move_sequence):
                is_model_turn = (i % 2 == 0)

                if is_model_turn:
                    current_fen = board.fen()
                    obs = alphazero_cpp.encode_position(current_fen)

                    legal_moves = list(board.legal_moves)
                    mask = np.zeros(POLICY_SIZE, dtype=np.float32)
                    idx_to_move = {}

                    for move in legal_moves:
                        idx = alphazero_cpp.move_to_index(move.uci(), current_fen)
                        if 0 <= idx < POLICY_SIZE:
                            mask[idx] = 1.0
                            idx_to_move[idx] = move

                    policy, value = evaluator.evaluate(obs, mask)
                    policy = policy * mask

                    if policy.sum() > 0:
                        top_idx = np.argmax(policy)
                        if top_idx in idx_to_move:
                            top_move = idx_to_move[top_idx]
                            top_move_san = board.san(top_move)
                            predicted_moves.append(top_move_san)

                            if not sequence_broken and top_move_san == expected_san:
                                consecutive_correct += 1
                            else:
                                sequence_broken = True
                        else:
                            predicted_moves.append("invalid")
                            sequence_broken = True
                    else:
                        predicted_moves.append("no_legal_moves")
                        sequence_broken = True

                # Play the expected move to advance the board
                try:
                    expected_move = board.parse_san(expected_san)
                    board.push(expected_move)
                except (ValueError, chess.InvalidMoveError):
                    break

            all_correct = (consecutive_correct == num_model_moves)
            if all_correct:
                puzzles_fully_correct += 1

            total_model_moves += num_model_moves
            total_consecutive_correct += consecutive_correct

            details.append({
                "type": puzzle_type,
                "fen": fen,
                "move_sequence": move_sequence,
                "predicted": predicted_moves,
                "consecutive_correct": consecutive_correct,
                "total_model_moves": num_model_moves,
                "move_accuracy": consecutive_correct / num_model_moves if num_model_moves > 0 else 0,
                "fully_correct": all_correct
            })

        move_accuracy = total_consecutive_correct / total_model_moves if total_model_moves > 0 else 0

        return EvaluationResult(
            name=self.name,
            score=puzzles_fully_correct / len(puzzles) if puzzles else 0,
            details={
                "puzzles_correct": puzzles_fully_correct,
                "total_puzzles": len(puzzles),
                "move_score": total_consecutive_correct,
                "total_moves": total_model_moves,
                "move_accuracy": move_accuracy,
                "puzzle_details": details
            },
            display_str=f"Endgame: {puzzles_fully_correct}/{len(puzzles)} puzzles, "
                        f"{total_consecutive_correct}/{total_model_moves} moves ({move_accuracy*100:.0f}%)"
        )


@register_evaluator("vs_stockfish")
class VsStockfishEvaluator(Evaluator):
    """Play games against Stockfish to measure strength against a calibrated opponent."""

    name = "vs_stockfish"
    description = "Play games against Stockfish engine"

    def run(self, network: torch.nn.Module, device: str, config: EvalConfig) -> EvaluationResult:
        """Run games against Stockfish."""

        # Locate Stockfish binary
        sf_path = config.stockfish_path or find_stockfish()
        if sf_path is None:
            print("    Stockfish not found! Install it with:")
            print("      python alphazero-cpp/scripts/install_stockfish.py")
            return EvaluationResult(
                name=self.name,
                score=0.0,
                details={"error": "Stockfish binary not found"},
                display_str="vs Stockfish: SKIPPED (binary not found)"
            )

        network.eval()
        evaluator = BatchedEvaluator(network, device, use_amp=(device == "cuda"))

        num_games = 5
        eval_sims = min(config.simulations, 400)
        eval_search_batch = min(config.search_batch, 32)

        sf_depth = config.stockfish_depth
        elo_str = f" (ELO {config.stockfish_elo})" if config.stockfish_elo else ""
        depth_str = f" depth={sf_depth}" if sf_depth else " depth=20"
        print(f"    Stockfish:{elo_str}{depth_str}")
        print(f"    Running {num_games} games (batched)...", end=" ", flush=True)
        start_time = time.time()

        with StockfishEngine(sf_path, elo=config.stockfish_elo, depth=sf_depth) as sf_engine:
            results = self._play_games(
                evaluator, sf_engine, num_games, eval_sims, eval_search_batch,
                config.c_puct, config.risk_beta
            )

        elapsed = time.time() - start_time
        result_chars = []
        for g in sorted(results["games"], key=lambda x: x["game_idx"]):
            if g["result"] == "win":
                result_chars.append("W")
            elif g["result"] == "loss":
                result_chars.append("L")
            else:
                result_chars.append("D")

        print(f"{''.join(result_chars)} ({elapsed:.1f}s)")

        wins = results["wins"]
        losses = results["losses"]
        draws = results["draws"]
        win_rate = wins / num_games

        return EvaluationResult(
            name=self.name,
            score=win_rate,
            details={
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "games": num_games,
                "simulations": eval_sims,
                "stockfish_elo": config.stockfish_elo,
                "stockfish_depth": sf_depth,
            },
            display_str=f"vs Stockfish{elo_str}: {wins}/{num_games} wins ({win_rate*100:.0f}%)"
        )

    def _play_games(self, evaluator: BatchedEvaluator, sf_engine: StockfishEngine,
                    num_games: int, simulations: int, search_batch: int,
                    c_puct: float, risk_beta: float = 0.0) -> Dict[str, Any]:
        """Play games with batched MCTS for AlphaZero vs Stockfish.

        Mirrors VsRandomEvaluator._play_games but uses Stockfish for opponent
        moves instead of random.choice().
        """

        class GameState:
            def __init__(self, game_idx: int):
                self.game_idx = game_idx
                self.board = chess.Board()
                self.model_plays_white = (game_idx % 2 == 0)
                self.move_count = 0
                self.position_history = []  # FEN history for encoder (up to 8 positions)
                self.mcts = alphazero_cpp.BatchedMCTSSearch(
                    num_simulations=simulations,
                    batch_size=search_batch,
                    c_puct=c_puct,
                    risk_beta=risk_beta
                )
                self.result = None
                self.needs_root_eval = False
                self.in_mcts_search = False
                self.current_fen = None
                self.current_obs = None
                self.current_mask = None
                self.current_idx_to_move = {}

        def _track_history(g):
            """Append current board FEN to history (before making a move)."""
            g.position_history.append(g.board.fen())
            if len(g.position_history) > 8:
                g.position_history.pop(0)

        games = [GameState(i) for i in range(num_games)]
        active_games = games.copy()

        def prepare_move(g: GameState):
            g.current_fen = g.board.fen()
            obs = alphazero_cpp.encode_position(g.current_fen, g.position_history)
            g.current_obs = obs

            legal_moves = list(g.board.legal_moves)
            g.current_mask = np.zeros(POLICY_SIZE, dtype=np.float32)
            g.current_idx_to_move = {}

            for move in legal_moves:
                idx = alphazero_cpp.move_to_index(move.uci(), g.current_fen)
                if 0 <= idx < POLICY_SIZE:
                    g.current_mask[idx] = 1.0
                    g.current_idx_to_move[idx] = move

            g.needs_root_eval = True
            g.in_mcts_search = False

        def finish_move(g: GameState):
            _track_history(g)
            visit_counts = g.mcts.get_visit_counts()
            policy = visit_counts * g.current_mask
            if policy.sum() > 0:
                action = np.argmax(policy)
                if action in g.current_idx_to_move:
                    g.board.push(g.current_idx_to_move[action])
                else:
                    print(f"\n    WARNING: Game {g.game_idx} move {g.move_count}: "
                          f"argmax action {action} not in idx_to_move, playing random", flush=True)
                    g.board.push(random.choice(list(g.board.legal_moves)))
            else:
                legal = list(g.board.legal_moves)
                if legal:
                    print(f"\n    WARNING: Game {g.game_idx} move {g.move_count}: "
                          f"zero visit counts, playing random", flush=True)
                    g.board.push(random.choice(legal))
            g.mcts.reset()
            g.move_count += 1
            g.in_mcts_search = False
            g.needs_root_eval = False

        def check_game_over(g: GameState) -> bool:
            if g.board.is_game_over():
                result = g.board.result()
                if result == "1-0":
                    g.result = "win" if g.model_plays_white else "loss"
                elif result == "0-1":
                    g.result = "loss" if g.model_plays_white else "win"
                else:
                    g.result = "draw"
                return True
            if g.move_count >= 200:
                g.result = "draw"
                return True
            return False

        def opponent_move(g: GameState):
            """Let Stockfish play the opponent move."""
            _track_history(g)
            try:
                move = sf_engine.get_move(g.board)
                g.board.push(move)
            except Exception as e:
                print(f"\n    WARNING: Stockfish error in game {g.game_idx}: {e}", flush=True)
                legal = list(g.board.legal_moves)
                if legal:
                    g.board.push(random.choice(legal))
            g.move_count += 1

        # Initialize: advance to first model move for each game
        for g in active_games:
            while not check_game_over(g):
                is_model_turn = (g.board.turn == chess.WHITE) == g.model_plays_white
                if is_model_turn:
                    prepare_move(g)
                    break
                else:
                    opponent_move(g)

        # Main loop: batch evaluations across all active games
        while active_games:
            # 1. Collect games needing root evaluation
            root_eval_games = [g for g in active_games if g.needs_root_eval and g.result is None]
            if root_eval_games:
                obs_batch = np.stack([g.current_obs for g in root_eval_games])
                mask_batch = np.stack([g.current_mask for g in root_eval_games])
                policies, values = evaluator.evaluate_batch(obs_batch, mask_batch)

                for i, g in enumerate(root_eval_games):
                    g.mcts.init_search(g.current_fen, policies[i].astype(np.float32), float(values[i]), g.position_history)
                    g.needs_root_eval = False
                    g.in_mcts_search = True

            # 2. Collect leaves from all games in MCTS search
            games_in_search = [g for g in active_games if g.in_mcts_search and g.result is None]
            if not games_in_search:
                active_games = [g for g in active_games if g.result is None]
                continue

            all_obs = []
            all_masks = []
            leaf_counts = []

            for g in games_in_search:
                if g.mcts.is_complete():
                    leaf_counts.append(0)
                    continue

                num_leaves, obs_batch, mask_batch = g.mcts.collect_leaves()
                if num_leaves == 0:
                    leaf_counts.append(0)
                    continue

                all_obs.append(obs_batch[:num_leaves])
                all_masks.append(mask_batch[:num_leaves])
                leaf_counts.append(num_leaves)

            # 3. Batch evaluate all leaves together
            if all_obs:
                combined_obs = np.concatenate(all_obs, axis=0)
                combined_masks = np.concatenate(all_masks, axis=0)
                all_policies, all_values = evaluator.evaluate_batch(combined_obs, combined_masks)

                # 4. Distribute results back to each game
                offset = 0
                leaf_idx = 0
                for g in games_in_search:
                    count = leaf_counts[leaf_idx]
                    leaf_idx += 1
                    if count == 0:
                        continue
                    game_policies = all_policies[offset:offset + count]
                    game_values = all_values[offset:offset + count]
                    g.mcts.update_leaves(game_policies.astype(np.float32), game_values.astype(np.float32))
                    offset += count

            # 5. Check for completed searches and advance games
            for g in games_in_search:
                if g.mcts.is_complete():
                    finish_move(g)

                    if check_game_over(g):
                        continue

                    while not check_game_over(g):
                        is_model_turn = (g.board.turn == chess.WHITE) == g.model_plays_white
                        if is_model_turn:
                            prepare_move(g)
                            break
                        else:
                            opponent_move(g)

            active_games = [g for g in active_games if g.result is None]

        # Tally results
        wins = sum(1 for g in games if g.result == "win")
        losses = sum(1 for g in games if g.result == "loss")
        draws = sum(1 for g in games if g.result == "draw")

        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "games": [{"game_idx": g.game_idx, "result": g.result} for g in games]
        }


# =============================================================================
# Checkpoint Loading Helpers
# =============================================================================

def find_latest_checkpoint(run_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a run directory."""
    pattern = os.path.join(run_dir, "model_iter_*.pt")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        pattern = os.path.join(run_dir, "cpp_iter_*.pt")
        checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None

    def extract_iter(path):
        name = os.path.basename(path)
        try:
            if "model_iter_" in name:
                return int(name.replace("model_iter_", "").replace(".pt", "").replace("_emergency", ""))
            elif "cpp_iter_" in name:
                return int(name.replace("cpp_iter_", "").replace(".pt", "").replace("_emergency", ""))
        except ValueError:
            return 0
        return 0

    checkpoints.sort(key=extract_iter)
    return checkpoints[-1]


def load_checkpoint(checkpoint_path: str, device: str) -> torch.nn.Module:
    """Load a checkpoint and return the network.

    Args:
        checkpoint_path: Path to .pt file or run directory
        device: Device to load model onto

    Returns:
        Loaded AlphaZeroNet model in eval mode
    """
    # Handle directory path (find latest checkpoint)
    if os.path.isdir(checkpoint_path):
        actual_path = find_latest_checkpoint(checkpoint_path)
        if actual_path is None:
            raise ValueError(f"No checkpoint found in directory: {checkpoint_path}")
        checkpoint_path = actual_path

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from checkpoint
    config = checkpoint.get('config', {})
    num_filters = config.get('num_filters', 192)
    num_blocks = config.get('num_blocks', 15)
    input_channels = config.get('input_channels', INPUT_CHANNELS)
    num_actions = config.get('num_actions', POLICY_SIZE)
    policy_filters = config.get('policy_filters', 2)
    value_filters = config.get('value_filters', 1)
    value_hidden = config.get('value_hidden', 256)
    wdl = config.get('wdl', True)
    se_reduction = config.get('se_reduction', 16)

    print(f"  Architecture: {num_filters} filters x {num_blocks} blocks (SE r={se_reduction})")
    print(f"  Value head: {'WDL' if wdl else 'scalar'}")
    print(f"  Iteration: {checkpoint.get('iteration', 'unknown')}")

    # Create network with checkpoint config
    network = AlphaZeroNet(
        input_channels=input_channels,
        num_filters=num_filters,
        num_blocks=num_blocks,
        num_actions=num_actions,
        policy_filters=policy_filters,
        value_filters=value_filters,
        value_hidden=value_hidden,
        wdl=wdl,
        se_reduction=se_reduction
    )

    # Load state dict
    network.load_state_dict(checkpoint['model_state_dict'])
    network = network.to(device)
    network.eval()

    return network


# =============================================================================
# Evaluation Orchestration
# =============================================================================

def run_all_evaluations(
    network: torch.nn.Module,
    device: str,
    config: EvalConfig,
    evaluator_names: Optional[List[str]] = None
) -> List[EvaluationResult]:
    """Run all (or selected) evaluators and return results.

    Args:
        network: Loaded neural network
        device: Device to run on
        config: Evaluation configuration
        evaluator_names: List of evaluator names to run, or None for all

    Returns:
        List of EvaluationResult objects
    """
    results = []

    if evaluator_names is None:
        evaluator_names = list(EVALUATORS.keys())

    for name in evaluator_names:
        if name not in EVALUATORS:
            print(f"  Warning: Unknown evaluator '{name}', skipping")
            continue

        evaluator = EVALUATORS[name]
        print(f"  [{evaluator.name}] {evaluator.description}")
        result = evaluator.run(network, device, config)
        results.append(result)

    return results


def display_results(results: List[EvaluationResult]):
    """Display evaluation results to console."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for result in results:
        print(f"  {result.display_str}")

    print("=" * 60)


def save_results(results: List[EvaluationResult], output_path: str):
    """Save evaluation results to JSON file."""
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": []
    }

    for result in results:
        output["results"].append({
            "name": result.name,
            "score": result.score,
            "display": result.display_str,
            "details": result.details
        })

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AlphaZero checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation.py --checkpoint checkpoints/run_dir/model_iter_010.pt
  python evaluation.py --checkpoint checkpoints/run_dir/  # Latest checkpoint
  python evaluation.py --checkpoint model.pt --evaluators endgame
  python evaluation.py --list
        """
    )

    parser.add_argument("--checkpoint", type=str,
                        help="Checkpoint path (.pt file) or run directory")
    parser.add_argument("--evaluators", nargs="+", default=None,
                        help="Specific evaluators to run (default: all)")
    parser.add_argument("--simulations", type=int, default=800,
                        help="MCTS simulations per move (default: 800)")
    parser.add_argument("--search-batch", type=int, default=32,
                        help="MCTS leaf batch size (default: 32)")
    parser.add_argument("--c-puct", type=float, default=1.5,
                        help="MCTS exploration constant (default: 1.5)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON output file for results")
    parser.add_argument("--list", action="store_true",
                        help="List available evaluators and exit")
    parser.add_argument("--stockfish-path", type=str, default=None,
                        help="Path to Stockfish binary (default: auto-detect)")
    parser.add_argument("--stockfish-elo", type=int, default=None,
                        help="Stockfish ELO rating limit (default: full strength)")
    parser.add_argument("--stockfish-depth", type=int, default=None,
                        help="Stockfish search depth limit (default: 20)")
    parser.add_argument("--risk-beta", type=float, default=0.0,
                        help="ERM risk sensitivity: >0 risk-seeking, <0 risk-averse (default: 0.0)")

    args = parser.parse_args()

    # List evaluators mode
    if args.list:
        print("\nAvailable evaluators:")
        print("-" * 50)
        for name, evaluator in EVALUATORS.items():
            print(f"  {name:15} {evaluator.description}")
        print("-" * 50)
        print(f"\nTotal: {len(EVALUATORS)} evaluators")
        return

    # Check for checkpoint argument
    if not args.checkpoint:
        parser.error("--checkpoint is required (unless using --list)")

    # Handle device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    # Print header
    print("=" * 60)
    print("AlphaZero Checkpoint Evaluation")
    print("=" * 60)

    # Load checkpoint
    try:
        network = load_checkpoint(args.checkpoint, device)
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        sys.exit(1)

    # Create config
    config = EvalConfig(
        simulations=args.simulations,
        search_batch=args.search_batch,
        c_puct=args.c_puct,
        risk_beta=args.risk_beta,
        device=device,
        stockfish_path=args.stockfish_path,
        stockfish_elo=args.stockfish_elo,
        stockfish_depth=args.stockfish_depth,
    )

    risk_str = f", risk_beta={config.risk_beta}" if config.risk_beta != 0.0 else ""
    print(f"  MCTS: {config.simulations} sims, batch={config.search_batch}, c_puct={config.c_puct}{risk_str}")
    print(f"  Device: {device}")
    print("-" * 60)

    # Run evaluations
    print("\nRunning evaluations...")
    results = run_all_evaluations(network, device, config, args.evaluators)

    # Display results
    display_results(results)

    # Save if requested
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
