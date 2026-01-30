"""Endgame position evaluation for AlphaZero.

Tests model performance on curated endgame positions to assess
tactical understanding and endgame technique.
"""

import chess
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from alphazero.chess_env import GameState
from alphazero.mcts import create_mcts
from alphazero.mcts.evaluator import NetworkEvaluator
from alphazero.config import MCTSConfig


@dataclass
class EndgamePosition:
    """Represents an endgame position for evaluation."""
    fen: str
    description: str
    category: str  # 'basic_mate', 'pawn_endgame', 'rook_endgame', 'tactical'
    expected_result: str  # 'win', 'draw', 'loss'
    optimal_first_move: Optional[str] = None  # UCI format
    difficulty: int = 1  # 1-5 scale


# Curated endgame positions
ENDGAME_POSITIONS = [
    # ========================================================================
    # Basic Checkmates (10 positions)
    # ========================================================================
    EndgamePosition(
        fen="4k3/8/8/8/8/8/8/4K2Q w - - 0 1",
        description="Queen vs King - Basic checkmate",
        category="basic_mate",
        expected_result="win",
        optimal_first_move="h1h5",
        difficulty=1
    ),
    EndgamePosition(
        fen="4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",
        description="Rook vs King - Basic checkmate",
        category="basic_mate",
        expected_result="win",
        optimal_first_move="a1a8",
        difficulty=1
    ),
    EndgamePosition(
        fen="6k1/8/6K1/8/8/8/8/6Q1 w - - 0 1",
        description="Queen checkmate in 2",
        category="basic_mate",
        expected_result="win",
        optimal_first_move="g1g7",
        difficulty=2
    ),
    EndgamePosition(
        fen="7k/5K2/8/8/8/8/8/6R1 w - - 0 1",
        description="Rook checkmate in 3",
        category="basic_mate",
        expected_result="win",
        optimal_first_move="g1g8",
        difficulty=2
    ),
    EndgamePosition(
        fen="8/8/8/8/8/3k4/3P4/3K4 w - - 0 1",
        description="King and pawn vs King",
        category="basic_mate",
        expected_result="win",
        optimal_first_move="d1c2",
        difficulty=2
    ),
    EndgamePosition(
        fen="8/8/8/8/8/2k5/2P5/2K5 w - - 0 1",
        description="King and pawn vs King - Opposition",
        category="basic_mate",
        expected_result="win",
        optimal_first_move="c1d2",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/k1K5/1Q6 w - - 0 1",
        description="Queen checkmate - Stalemate trap",
        category="basic_mate",
        expected_result="win",
        optimal_first_move="b1b3",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/k1K5/1R6 w - - 0 1",
        description="Rook checkmate - Stalemate trap",
        category="basic_mate",
        expected_result="win",
        optimal_first_move="b1b3",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/8/8/5k2/5P2/5K2 w - - 0 1",
        description="King and pawn vs King - Key squares",
        category="basic_mate",
        expected_result="win",
        optimal_first_move="f1g2",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/5PPk/6K1 w - - 0 1",
        description="Two pawns vs King",
        category="basic_mate",
        expected_result="win",
        optimal_first_move="g1h1",
        difficulty=2
    ),

    # ========================================================================
    # Pawn Endgames (15 positions)
    # ========================================================================
    EndgamePosition(
        fen="8/8/8/4k3/4P3/4K3/8/8 w - - 0 1",
        description="King and pawn vs King - Opposition critical",
        category="pawn_endgame",
        expected_result="win",
        optimal_first_move="e3d3",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/8/3k4/3P4/3K4/8 w - - 0 1",
        description="King and pawn vs King - Winning",
        category="pawn_endgame",
        expected_result="win",
        optimal_first_move="d2c3",
        difficulty=2
    ),
    EndgamePosition(
        fen="8/8/8/8/8/3k4/3P4/3K4 b - - 0 1",
        description="King and pawn vs King - Draw",
        category="pawn_endgame",
        expected_result="draw",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/5p2/5k2/8/5P2/5K2/8/8 w - - 0 1",
        description="Pawn vs Pawn - Opposition",
        category="pawn_endgame",
        expected_result="draw",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/3k4/3p4/3P4/3K4/8 w - - 0 1",
        description="Blocked pawns - Draw",
        category="pawn_endgame",
        expected_result="draw",
        difficulty=2
    ),
    EndgamePosition(
        fen="8/8/8/8/4Pk2/8/5K2/8 w - - 0 1",
        description="Outside passed pawn",
        category="pawn_endgame",
        expected_result="win",
        optimal_first_move="e4e5",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/5pk1/5p2/8/5P2/5PK1/8/8 w - - 0 1",
        description="Pawn breakthrough",
        category="pawn_endgame",
        expected_result="win",
        optimal_first_move="f4f5",
        difficulty=4
    ),
    EndgamePosition(
        fen="8/8/8/2k5/2p5/2P5/2K5/8 w - - 0 1",
        description="Pawn endgame - Zugzwang",
        category="pawn_endgame",
        expected_result="draw",
        difficulty=4
    ),
    EndgamePosition(
        fen="8/8/8/8/8/1k6/1p6/1K6 w - - 0 1",
        description="Rook pawn draw",
        category="pawn_endgame",
        expected_result="draw",
        difficulty=2
    ),
    EndgamePosition(
        fen="8/8/8/8/8/7k/7p/7K w - - 0 1",
        description="Rook pawn draw - Wrong color bishop",
        category="pawn_endgame",
        expected_result="draw",
        difficulty=2
    ),
    EndgamePosition(
        fen="8/8/8/8/3k4/2pP4/2P5/2K5 w - - 0 1",
        description="Pawn majority",
        category="pawn_endgame",
        expected_result="win",
        optimal_first_move="c1d2",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/8/8/2k5/1pP5/2K5 w - - 0 1",
        description="Pawn race",
        category="pawn_endgame",
        expected_result="draw",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/k1PP4/2K5 w - - 0 1",
        description="Two connected pawns vs King",
        category="pawn_endgame",
        expected_result="win",
        optimal_first_move="c1b1",
        difficulty=2
    ),
    EndgamePosition(
        fen="8/8/8/8/8/2k5/2PP4/2K5 w - - 0 1",
        description="Doubled pawns endgame",
        category="pawn_endgame",
        expected_result="win",
        optimal_first_move="c1d1",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/8/3Pk3/8/3K4/8 w - - 0 1",
        description="Distant opposition",
        category="pawn_endgame",
        expected_result="win",
        optimal_first_move="d2e2",
        difficulty=4
    ),

    # ========================================================================
    # Rook Endgames (15 positions)
    # ========================================================================
    EndgamePosition(
        fen="8/8/8/8/8/3k4/3r4/3K3R w - - 0 1",
        description="Rook vs Rook - Philidor position",
        category="rook_endgame",
        expected_result="draw",
        difficulty=4
    ),
    EndgamePosition(
        fen="6k1/6p1/6P1/8/8/8/r7/6KR w - - 0 1",
        description="Rook endgame - Lucena position",
        category="rook_endgame",
        expected_result="win",
        optimal_first_move="h1h4",
        difficulty=4
    ),
    EndgamePosition(
        fen="8/8/8/8/8/3k4/3p4/3K3R w - - 0 1",
        description="Rook vs pawn - Winning",
        category="rook_endgame",
        expected_result="win",
        optimal_first_move="h1h3",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/1k1p4/1K1R4 w - - 0 1",
        description="Rook vs pawn - Draw (rook pawn)",
        category="rook_endgame",
        expected_result="draw",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/3k4/R3K3 w Q - 0 1",
        description="Rook endgame - Back rank mate",
        category="rook_endgame",
        expected_result="win",
        optimal_first_move="a1a8",
        difficulty=2
    ),
    EndgamePosition(
        fen="6k1/8/6K1/8/8/8/8/7R w - - 0 1",
        description="Rook checkmate technique",
        category="rook_endgame",
        expected_result="win",
        optimal_first_move="h1h8",
        difficulty=2
    ),
    EndgamePosition(
        fen="8/8/8/8/8/5k2/5p2/5K1R w - - 0 1",
        description="Rook vs pawn - Cutting off king",
        category="rook_endgame",
        expected_result="win",
        optimal_first_move="h1h3",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/5pk1/5K1R w - - 0 1",
        description="Rook vs pawn - Stalemate trap",
        category="rook_endgame",
        expected_result="draw",
        difficulty=4
    ),
    EndgamePosition(
        fen="8/8/8/8/8/3k4/3pR3/3K4 w - - 0 1",
        description="Rook behind passed pawn",
        category="rook_endgame",
        expected_result="draw",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/8/8/3k4/R2p4/3K4 w - - 0 1",
        description="Rook in front of passed pawn",
        category="rook_endgame",
        expected_result="win",
        optimal_first_move="a2a3",
        difficulty=3
    ),
    EndgamePosition(
        fen="6k1/6p1/6P1/8/8/8/8/6KR w - - 0 1",
        description="Rook and pawn vs pawn",
        category="rook_endgame",
        expected_result="win",
        optimal_first_move="h1h7",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/1k1R4/1K6 w - - 0 1",
        description="Rook endgame - Opposition",
        category="rook_endgame",
        expected_result="win",
        optimal_first_move="d2d8",
        difficulty=2
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/r7/K6k w - - 0 1",
        description="Rook vs King - Skewer",
        category="rook_endgame",
        expected_result="loss",
        difficulty=2
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/6k1/R5K1 w - - 0 1",
        description="Rook endgame - Cutting off king",
        category="rook_endgame",
        expected_result="win",
        optimal_first_move="a1a7",
        difficulty=2
    ),
    EndgamePosition(
        fen="8/8/8/8/8/5k2/6r1/5K1R w - - 0 1",
        description="Rook vs Rook - Active rook",
        category="rook_endgame",
        expected_result="draw",
        difficulty=3
    ),

    # ========================================================================
    # Tactical Endgames (10 positions)
    # ========================================================================
    EndgamePosition(
        fen="8/8/8/8/8/2k5/1q6/2K5 w - - 0 1",
        description="Queen vs King - Avoid stalemate",
        category="tactical",
        expected_result="loss",
        difficulty=2
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/6pk/6K1 w - - 0 1",
        description="Stalemate trap",
        category="tactical",
        expected_result="draw",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/5PPk/6K1 w - - 0 1",
        description="Two pawns vs King - Promotion",
        category="tactical",
        expected_result="win",
        optimal_first_move="g1h1",
        difficulty=2
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/6Pk/7K w - - 0 1",
        description="Pawn promotion race",
        category="tactical",
        expected_result="win",
        optimal_first_move="g2g3",
        difficulty=2
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/k7/K6Q w - - 0 1",
        description="Queen vs King - Zugzwang",
        category="tactical",
        expected_result="win",
        optimal_first_move="h1h2",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/6k1/6KR w - - 0 1",
        description="Rook checkmate - Tempo",
        category="tactical",
        expected_result="win",
        optimal_first_move="h1h2",
        difficulty=2
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/5k2/5K1Q w - - 0 1",
        description="Queen checkmate - Triangulation",
        category="tactical",
        expected_result="win",
        optimal_first_move="h1h2",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/6pk/7K w - - 0 1",
        description="Stalemate - Forced draw",
        category="tactical",
        expected_result="draw",
        difficulty=2
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/5PPk/6K1 b - - 0 1",
        description="Two pawns vs King - Defense",
        category="tactical",
        expected_result="loss",
        difficulty=3
    ),
    EndgamePosition(
        fen="8/8/8/8/8/8/6k1/6KQ w - - 0 1",
        description="Queen checkmate - Precision",
        category="tactical",
        expected_result="win",
        optimal_first_move="h1h2",
        difficulty=2
    ),
]


class EndgameEvaluator:
    """Evaluates model performance on endgame positions."""

    def __init__(self, network, device: str = "cuda", num_simulations: int = 400,
                 max_moves: int = 100, use_amp: bool = True):
        """Initialize endgame evaluator.

        Args:
            network: Neural network to evaluate
            device: Device to run on
            num_simulations: MCTS simulations per move
            max_moves: Maximum moves per position
            use_amp: Use mixed precision
        """
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.max_moves = max_moves

        # Create MCTS and evaluator
        self.mcts_config = MCTSConfig(
            num_simulations=num_simulations,
            c_puct=1.25,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            temperature=0.0
        )
        self.mcts = create_mcts(config=self.mcts_config)
        self.evaluator = NetworkEvaluator(network, device, use_amp=use_amp)

    def evaluate_position(self, position: EndgamePosition) -> Dict:
        """Evaluate model on a single endgame position.

        Args:
            position: Endgame position to evaluate

        Returns:
            Dictionary with evaluation results
        """
        # Create game state from FEN
        state = GameState.from_fen(position.fen)

        move_count = 0
        moves_played = []
        found_optimal = False

        # Play out the position
        while not state.is_terminal() and move_count < self.max_moves:
            # Run MCTS
            policy, root, stats = self.mcts.search(
                state, self.evaluator,
                move_number=move_count,
                add_noise=False
            )

            # Select best action
            action = int(np.argmax(policy))

            # Convert action to move using GameState method
            move = state.action_to_move(action)

            # Check if first move is optimal
            if move_count == 0 and position.optimal_first_move:
                if move.uci() == position.optimal_first_move:
                    found_optimal = True

            moves_played.append(move.uci())

            # Apply move
            state = state.apply_move(move)
            move_count += 1

        # Determine result
        if state.is_terminal():
            result = state.get_result()
            if result is None:
                actual_result = "draw"
            elif result.winner is True:
                actual_result = "win"
            elif result.winner is False:
                actual_result = "loss"
            else:
                actual_result = "draw"
        else:
            actual_result = "timeout"

        # Check if result matches expected
        correct_result = (actual_result == position.expected_result)

        return {
            'position': position.description,
            'category': position.category,
            'difficulty': position.difficulty,
            'expected_result': position.expected_result,
            'actual_result': actual_result,
            'correct': correct_result,
            'found_optimal': found_optimal,
            'moves_played': move_count,
            'moves': moves_played
        }

    def evaluate_all(self, positions: List[EndgamePosition] = None) -> Dict:
        """Evaluate model on all endgame positions.

        Args:
            positions: List of positions to evaluate (default: all 50)

        Returns:
            Dictionary with aggregate results
        """
        if positions is None:
            positions = ENDGAME_POSITIONS

        results = []
        category_stats = {}

        print(f"\nEvaluating {len(positions)} endgame positions...")
        print(f"MCTS simulations: {self.num_simulations}")
        print(f"Max moves per position: {self.max_moves}\n")

        with tqdm(total=len(positions), desc="Evaluating", unit="pos") as pbar:
            for position in positions:
                result = self.evaluate_position(position)
                results.append(result)

                # Update category stats
                category = position.category
                if category not in category_stats:
                    category_stats[category] = {
                        'total': 0,
                        'correct': 0,
                        'optimal': 0
                    }

                category_stats[category]['total'] += 1
                if result['correct']:
                    category_stats[category]['correct'] += 1
                if result['found_optimal']:
                    category_stats[category]['optimal'] += 1

                pbar.update(1)
                pbar.set_postfix({
                    'correct': f"{sum(r['correct'] for r in results)}/{len(results)}",
                    'accuracy': f"{sum(r['correct'] for r in results) / len(results) * 100:.1f}%"
                })

        # Calculate aggregate statistics
        total_positions = len(results)
        correct_results = sum(r['correct'] for r in results)
        optimal_moves = sum(r['found_optimal'] for r in results if r['found_optimal'] is not False)
        avg_moves = np.mean([r['moves_played'] for r in results])

        # Difficulty breakdown
        difficulty_stats = {}
        for i in range(1, 6):
            diff_results = [r for r in results if r['difficulty'] == i]
            if diff_results:
                difficulty_stats[i] = {
                    'total': len(diff_results),
                    'correct': sum(r['correct'] for r in diff_results),
                    'accuracy': sum(r['correct'] for r in diff_results) / len(diff_results) * 100
                }

        return {
            'total_positions': total_positions,
            'correct_results': correct_results,
            'accuracy': correct_results / total_positions * 100,
            'optimal_moves_found': optimal_moves,
            'avg_moves_per_position': avg_moves,
            'category_stats': category_stats,
            'difficulty_stats': difficulty_stats,
            'detailed_results': results
        }

    def print_summary(self, results: Dict):
        """Print evaluation summary.

        Args:
            results: Results from evaluate_all()
        """
        print("\n" + "="*60)
        print("ENDGAME EVALUATION RESULTS")
        print("="*60)
        print(f"Total positions: {results['total_positions']}")
        print(f"Correct results: {results['correct_results']}/{results['total_positions']}")
        print(f"Overall accuracy: {results['accuracy']:.1f}%")
        print(f"Optimal first moves found: {results['optimal_moves_found']}")
        print(f"Average moves per position: {results['avg_moves_per_position']:.1f}")

        print("\n" + "-"*60)
        print("CATEGORY BREAKDOWN")
        print("-"*60)
        for category, stats in results['category_stats'].items():
            accuracy = stats['correct'] / stats['total'] * 100
            print(f"{category:20s}: {stats['correct']:2d}/{stats['total']:2d} ({accuracy:5.1f}%)")

        print("\n" + "-"*60)
        print("DIFFICULTY BREAKDOWN")
        print("-"*60)
        for difficulty, stats in sorted(results['difficulty_stats'].items()):
            print(f"Difficulty {difficulty}: {stats['correct']:2d}/{stats['total']:2d} ({stats['accuracy']:5.1f}%)")

        print("\n" + "-"*60)
        print("FAILED POSITIONS")
        print("-"*60)
        failed = [r for r in results['detailed_results'] if not r['correct']]
        if failed:
            for r in failed:
                print(f"- {r['position']}")
                print(f"  Expected: {r['expected_result']}, Got: {r['actual_result']}")
                print(f"  Category: {r['category']}, Difficulty: {r['difficulty']}")
        else:
            print("None - Perfect score!")

        print("="*60 + "\n")
