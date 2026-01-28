#!/usr/bin/env python3
"""Evaluate model strength against Stockfish or other models."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from alphazero import AlphaZeroConfig, MCTSConfig
from alphazero.neural import AlphaZeroNetwork
from alphazero.mcts.evaluator import NetworkEvaluator
from alphazero.evaluation import (
    Arena, MCTSPlayer, RandomPlayer,
    StockfishEvaluator, quick_elo_test
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate AlphaZero model")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--opponent", type=str, default="random",
                        choices=["random", "stockfish", "self"],
                        help="Opponent type")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games to play")
    parser.add_argument("--simulations", type=int, default=800,
                        help="MCTS simulations per move")
    parser.add_argument("--stockfish-path", type=str, default=None,
                        help="Path to Stockfish executable")
    parser.add_argument("--stockfish-elo", type=int, default=1500,
                        help="Stockfish Elo level")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    state = torch.load(args.checkpoint, map_location=args.device)

    network = AlphaZeroNetwork()
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
        stats, _ = arena.play_matches(player, opponent, args.games)

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
        stats = quick_elo_test(
            player,
            args.stockfish_path,
            elo=args.stockfish_elo,
            num_games=args.games
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
        stats, _ = arena.play_matches(player, player2, args.games)

        print(f"\nSelf-play results:")
        print(f"  White wins: {stats.wins}")
        print(f"  Draws: {stats.draws}")
        print(f"  Black wins: {stats.losses}")


if __name__ == "__main__":
    main()
