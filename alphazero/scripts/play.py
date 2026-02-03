#!/usr/bin/env python3
"""Play against a trained AlphaZero model."""

import argparse
import sys
from pathlib import Path

import torch
import chess

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alphazero import AlphaZeroConfig, MCTSConfig
from alphazero.chess_env import GameState
from alphazero.neural import AlphaZeroNetwork
from alphazero.mcts import create_mcts
from alphazero.mcts.evaluator import NetworkEvaluator
from alphazero.utils import load_checkpoint_with_architecture


def print_board(state: GameState):
    """Print the board with coordinates."""
    board = state.board
    print()
    print("  a b c d e f g h")
    print(" +-----------------+")
    for rank in range(7, -1, -1):
        print(f"{rank + 1}| ", end="")
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece:
                print(f"{piece.symbol()} ", end="")
            else:
                print(". ", end="")
        print(f"|{rank + 1}")
    print(" +-----------------+")
    print("  a b c d e f g h")
    print()


def get_human_move(state: GameState) -> int:
    """Get move from human player."""
    board = state.board

    while True:
        try:
            move_str = input("Your move (e.g., e2e4): ").strip()

            if move_str.lower() in ['quit', 'exit', 'q']:
                return -1

            if move_str.lower() == 'moves':
                print("Legal moves:", [m.uci() for m in board.legal_moves])
                continue

            # Parse move
            move = chess.Move.from_uci(move_str)

            if move not in board.legal_moves:
                print("Illegal move. Type 'moves' to see legal moves.")
                continue

            return state.move_to_action(move)

        except ValueError:
            print("Invalid move format. Use UCI notation (e.g., e2e4, e7e8q)")


def main():
    parser = argparse.ArgumentParser(description="Play against AlphaZero")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--simulations", type=int, default=400,
                        help="MCTS simulations per move (default: 400 for interactive play)")
    parser.add_argument("--filters", type=int, default=192,
                        help="Number of filters in network (must match checkpoint)")
    parser.add_argument("--blocks", type=int, default=15,
                        help="Number of residual blocks (must match checkpoint)")
    parser.add_argument("--color", type=str, default="white",
                        choices=["white", "black"],
                        help="Your color")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")

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

    # Create MCTS
    mcts_config = MCTSConfig(num_simulations=args.simulations)
    mcts = create_mcts(config=mcts_config)
    evaluator = NetworkEvaluator(network, args.device)

    # Setup game
    game_state = GameState()
    human_is_white = args.color == "white"

    print("\nAlphaZero Chess")
    print("===============")
    print(f"You are playing as {'White' if human_is_white else 'Black'}")
    print("Type 'moves' to see legal moves, 'quit' to exit")
    print()

    while not game_state.is_terminal():
        print_board(game_state)

        is_human_turn = (game_state.turn == chess.WHITE) == human_is_white

        if is_human_turn:
            print("Your turn!")
            action = get_human_move(game_state)
            if action == -1:
                print("Game aborted.")
                return
        else:
            print("AlphaZero is thinking...")
            policy, root, stats = mcts.search(
                game_state,
                evaluator,
                move_number=game_state.ply,
                add_noise=False
            )
            action = int(policy.argmax())
            move = game_state.action_to_move(action)
            print(f"AlphaZero plays: {move.uci()}")
            print(f"  (value: {root.q_value:.3f}, depth: {stats.max_depth})")

        game_state = game_state.apply_action(action)

    # Game over
    print_board(game_state)
    result = game_state.get_result()

    if result.winner is None:
        print(f"Draw by {result.termination}!")
    elif result.winner == human_is_white:
        print("Congratulations! You won!")
    else:
        print("AlphaZero wins!")


if __name__ == "__main__":
    main()
