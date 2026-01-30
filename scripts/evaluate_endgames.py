#!/usr/bin/env python3
"""Evaluate model performance on endgame positions.

Tests the model on 50 curated endgame positions covering:
- Basic checkmates
- Pawn endgames
- Rook endgames
- Tactical positions
"""

import sys
import argparse
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphazero.neural.network import AlphaZeroNetwork
from alphazero.evaluation.endgame_eval import EndgameEvaluator, ENDGAME_POSITIONS


def main():
    parser = argparse.ArgumentParser(description="Endgame Position Evaluation")

    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--simulations", type=int, default=400,
                       help="MCTS simulations per move (default: 400)")
    parser.add_argument("--max-moves", type=int, default=100,
                       help="Maximum moves per position (default: 100)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (cuda or cpu)")
    parser.add_argument("--filters", type=int, default=None,
                       help="Network filters (auto-detected from checkpoint)")
    parser.add_argument("--blocks", type=int, default=None,
                       help="Network blocks (auto-detected from checkpoint)")
    parser.add_argument("--category", type=str, default=None,
                       choices=['basic_mate', 'pawn_endgame', 'rook_endgame', 'tactical'],
                       help="Evaluate only specific category")
    parser.add_argument("--difficulty", type=int, default=None,
                       choices=[1, 2, 3, 4, 5],
                       help="Evaluate only specific difficulty level")

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("="*60)
    print("ALPHAZERO ENDGAME EVALUATION")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"MCTS simulations: {args.simulations}")
    print(f"Max moves per position: {args.max_moves}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Extract network architecture
    num_filters = args.filters or checkpoint.get('num_filters', 192)
    num_blocks = args.blocks or checkpoint.get('num_blocks', 15)

    print(f"Network: {num_filters} filters, {num_blocks} blocks")

    # Create network
    network = AlphaZeroNetwork(num_filters=num_filters, num_blocks=num_blocks)
    network.load_state_dict(checkpoint['network_state_dict'])
    network = network.to(args.device)
    network.eval()

    print("Model loaded successfully")

    # Filter positions if requested
    positions = ENDGAME_POSITIONS
    if args.category:
        positions = [p for p in positions if p.category == args.category]
        print(f"\nFiltered to {len(positions)} positions in category: {args.category}")

    if args.difficulty:
        positions = [p for p in positions if p.difficulty == args.difficulty]
        print(f"Filtered to {len(positions)} positions with difficulty: {args.difficulty}")

    # Create evaluator
    evaluator = EndgameEvaluator(
        network=network,
        device=args.device,
        num_simulations=args.simulations,
        max_moves=args.max_moves,
        use_amp=True
    )

    # Run evaluation
    results = evaluator.evaluate_all(positions)

    # Print summary
    evaluator.print_summary(results)

    # Save detailed results
    import json
    output_file = Path(args.checkpoint).parent / "endgame_evaluation.json"
    with open(output_file, 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {
            'checkpoint': args.checkpoint,
            'num_filters': num_filters,
            'num_blocks': num_blocks,
            'simulations': args.simulations,
            'total_positions': results['total_positions'],
            'correct_results': results['correct_results'],
            'accuracy': results['accuracy'],
            'optimal_moves_found': results['optimal_moves_found'],
            'avg_moves_per_position': results['avg_moves_per_position'],
            'category_stats': results['category_stats'],
            'difficulty_stats': results['difficulty_stats'],
            'detailed_results': results['detailed_results']
        }
        json.dump(json_results, f, indent=2)

    print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
