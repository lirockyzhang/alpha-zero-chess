#!/usr/bin/env python3
"""Evaluate model strength against Stockfish or other models."""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alphazero import AlphaZeroConfig, MCTSConfig
from alphazero.neural import AlphaZeroNetwork
from alphazero.mcts.evaluator import NetworkEvaluator
from alphazero.evaluation import (
    Arena, MCTSPlayer, RandomPlayer, MatchStats,
    StockfishEvaluator, quick_elo_test
)
from alphazero.evaluation.endgame_eval import EndgameEvaluator, ENDGAME_POSITIONS
from alphazero.utils import load_checkpoint_with_architecture


def main():
    parser = argparse.ArgumentParser(description="Evaluate AlphaZero model")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--opponent", type=str, default="random",
                        choices=["random", "stockfish", "self", "endgame"],
                        help="Opponent type")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games to play")
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS simulations per move (default: 100 for fast evaluation, use 800 for accurate measurement)")
    parser.add_argument("--filters", type=int, default=192,
                        help="Number of filters in network (must match checkpoint)")
    parser.add_argument("--blocks", type=int, default=15,
                        help="Number of residual blocks (must match checkpoint)")
    parser.add_argument("--stockfish-path", type=str, default=None,
                        help="Path to Stockfish executable")
    parser.add_argument("--stockfish-elo", type=int, default=1500,
                        help="Stockfish Elo level")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")
    parser.add_argument("--max-moves", type=int, default=100,
                        help="Maximum moves per endgame position (only for --opponent endgame)")
    parser.add_argument("--category", type=str, default=None,
                        choices=['basic_mate', 'pawn_endgame', 'rook_endgame', 'tactical'],
                        help="Endgame category filter (only for --opponent endgame)")
    parser.add_argument("--difficulty", type=int, default=None,
                        choices=[1, 2, 3, 4, 5],
                        help="Endgame difficulty filter (only for --opponent endgame)")

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    state, filters_from_ckpt, blocks_from_ckpt = load_checkpoint_with_architecture(
        args.checkpoint, args.device
    )

    # Determine architecture: use checkpoint info if available, otherwise use args
    num_filters = filters_from_ckpt if filters_from_ckpt is not None else args.filters
    num_blocks = blocks_from_ckpt if blocks_from_ckpt is not None else args.blocks

    if filters_from_ckpt is not None:
        print(f"Detected architecture from checkpoint: {num_blocks} blocks, {num_filters} filters")
    else:
        print(f"Using architecture from arguments: {num_blocks} blocks, {num_filters} filters")

    # Create network with matching architecture
    network = AlphaZeroNetwork(
        num_filters=num_filters,
        num_blocks=num_blocks
    )
    network.load_state_dict(state['network_state_dict'])
    network = network.to(args.device)
    network.eval()

    # Create player
    mcts_config = MCTSConfig(num_simulations=args.simulations)
    evaluator = NetworkEvaluator(network, args.device)
    player = MCTSPlayer("AlphaZero", evaluator, mcts_config, temperature=0.0)

    # Run evaluation
    arena = Arena()

    if args.opponent == "random":
        print(f"\nEvaluating against random player ({args.games} games)...")
        opponent = RandomPlayer()

        # Play matches with progress bar
        stats = MatchStats(player.name, opponent.name)
        with tqdm(total=args.games, desc="Playing games", unit="game") as pbar:
            for i in range(args.games):
                # Alternate colors
                if i % 2 == 0:
                    result = arena.play_match(player, opponent)
                    if result.result > 0:
                        stats.wins += 1
                    elif result.result < 0:
                        stats.losses += 1
                    else:
                        stats.draws += 1
                else:
                    result = arena.play_match(opponent, player)
                    if result.result < 0:
                        stats.wins += 1
                    elif result.result > 0:
                        stats.losses += 1
                    else:
                        stats.draws += 1

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'W': stats.wins,
                    'D': stats.draws,
                    'L': stats.losses,
                    'score': f"{stats.score:.1%}"
                })

        print(f"\nResults vs Random:")
        print(f"  Wins: {stats.wins}")
        print(f"  Draws: {stats.draws}")
        print(f"  Losses: {stats.losses}")
        print(f"  Score: {stats.score:.1%}")

    elif args.opponent == "stockfish":
        if not args.stockfish_path:
            print("Error: --stockfish-path required for Stockfish evaluation")
            sys.exit(1)

        print(f"\nEvaluating against Stockfish {args.stockfish_elo} ({args.games} games)...")

        # Use quick_elo_test with progress tracking
        stats = quick_elo_test(
            player,
            args.stockfish_path,
            elo=args.stockfish_elo,
            num_games=args.games,
            show_progress=True
        )

        print(f"\nResults vs Stockfish {args.stockfish_elo}:")
        print(f"  Wins: {stats.wins}")
        print(f"  Draws: {stats.draws}")
        print(f"  Losses: {stats.losses}")
        print(f"  Score: {stats.score:.1%}")
        print(f"  Estimated Elo diff: {stats.elo_difference():+.0f}")

    elif args.opponent == "self":
        print(f"\nSelf-play evaluation ({args.games} games)...")
        # Create second player with same weights
        player2 = MCTSPlayer("AlphaZero-2", evaluator, mcts_config, temperature=0.0)

        # Play matches with progress bar
        stats = MatchStats(player.name, player2.name)
        with tqdm(total=args.games, desc="Playing games", unit="game") as pbar:
            for i in range(args.games):
                # Alternate colors
                if i % 2 == 0:
                    result = arena.play_match(player, player2)
                    if result.result > 0:
                        stats.wins += 1
                    elif result.result < 0:
                        stats.losses += 1
                    else:
                        stats.draws += 1
                else:
                    result = arena.play_match(player2, player)
                    if result.result < 0:
                        stats.wins += 1
                    elif result.result > 0:
                        stats.losses += 1
                    else:
                        stats.draws += 1

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'W': stats.wins,
                    'D': stats.draws,
                    'L': stats.losses
                })

        print(f"\nSelf-play results:")
        print(f"  White wins: {stats.wins}")
        print(f"  Draws: {stats.draws}")
        print(f"  Black wins: {stats.losses}")

    elif args.opponent == "endgame":
        print(f"\nEvaluating on endgame positions...")
        print(f"Max moves per position: {args.max_moves}")

        # Filter positions if requested
        positions = ENDGAME_POSITIONS
        if args.category:
            positions = [p for p in positions if p.category == args.category]
            print(f"Filtered to {len(positions)} positions in category: {args.category}")

        if args.difficulty:
            positions = [p for p in positions if p.difficulty == args.difficulty]
            print(f"Filtered to {len(positions)} positions with difficulty: {args.difficulty}")

        if not positions:
            print("Error: No positions match the specified filters")
            sys.exit(1)

        # Create endgame evaluator
        endgame_eval = EndgameEvaluator(
            network=network,
            device=args.device,
            num_simulations=args.simulations,
            max_moves=args.max_moves,
            use_amp=True
        )

        # Run evaluation
        results = endgame_eval.evaluate_all(positions)

        # Print summary
        endgame_eval.print_summary(results)

        # Save detailed results
        import json
        from pathlib import Path
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

        print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
