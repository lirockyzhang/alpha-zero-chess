#!/usr/bin/env python3
"""Single game self-play walkthrough.

This demo shows a complete self-play game with detailed output.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from alphazero import MCTSConfig, SelfPlayConfig
from alphazero.chess_env import GameState
from alphazero.neural import AlphaZeroNetwork
from alphazero.mcts import create_mcts
from alphazero.mcts.evaluator import NetworkEvaluator, RandomEvaluator


def print_board(state: GameState, move_num: int):
    """Print board with move number."""
    print(f"\nMove {move_num}:")
    print(state)


def demo_selfplay_random():
    """Demo self-play with random evaluator."""
    print("=" * 60)
    print("Self-Play Demo (Random Evaluator)")
    print("=" * 60)
    print("\nThis demo uses a random evaluator (uniform policy, zero value)")
    print("to show the self-play process without a trained network.\n")

    config = MCTSConfig(num_simulations=100)
    selfplay_config = SelfPlayConfig(max_moves=50)

    mcts = create_mcts(config=config)
    evaluator = RandomEvaluator()

    state = GameState()
    move_num = 0

    print("Starting position:")
    print(state)

    while not state.is_terminal() and move_num < selfplay_config.max_moves:
        # Run MCTS
        policy, root, stats = mcts.search(
            state,
            evaluator,
            move_number=move_num,
            add_noise=True
        )

        # Select action
        temperature = 1.0 if move_num < 30 else 0.01
        if temperature < 0.1:
            action = int(np.argmax(policy))
        else:
            action = int(np.random.choice(len(policy), p=policy))

        move = state.action_to_move(action)

        # Print move info
        player = "White" if state.turn else "Black"
        print(f"\n{player} plays: {move.uci()}")
        print(f"  MCTS: {stats.num_simulations} sims, depth={stats.max_depth}, "
              f"value={root.q_value:.3f}")

        # Top alternatives
        visit_counts = root.get_visit_counts(4672)
        top_actions = np.argsort(-visit_counts)[:3]
        print("  Top moves:", end=" ")
        for a in top_actions:
            if visit_counts[a] > 0:
                m = state.action_to_move(a)
                print(f"{m.uci()}({int(visit_counts[a])})", end=" ")
        print()

        # Apply move
        state = state.apply_action(action)
        move_num += 1

        # Print board every 10 moves
        if move_num % 10 == 0:
            print_board(state, move_num)

    # Game result
    print("\n" + "=" * 60)
    print("Game Over")
    print("=" * 60)
    print_board(state, move_num)

    if state.is_terminal():
        result = state.get_result()
        if result.winner is True:
            print("Result: White wins!")
        elif result.winner is False:
            print("Result: Black wins!")
        else:
            print(f"Result: Draw ({result.termination})")
    else:
        print("Result: Draw (max moves reached)")


def demo_selfplay_network():
    """Demo self-play with neural network (untrained)."""
    print("\n" + "=" * 60)
    print("Self-Play Demo (Neural Network - Untrained)")
    print("=" * 60)
    print("\nThis demo uses an untrained neural network.")
    print("The network outputs random-ish values initially.\n")

    # Create small network for demo
    network = AlphaZeroNetwork(
        num_filters=64,
        num_blocks=5
    )
    network.eval()

    config = MCTSConfig(num_simulations=50)
    mcts = create_mcts(config=config)
    evaluator = NetworkEvaluator(network, device="cpu")

    state = GameState()

    print("Starting position:")
    print(state)

    # Play just a few moves
    for move_num in range(10):
        policy, root, stats = mcts.search(
            state,
            evaluator,
            move_number=move_num,
            add_noise=True
        )

        action = int(np.argmax(policy))
        move = state.action_to_move(action)

        player = "White" if state.turn else "Black"
        print(f"\nMove {move_num + 1}: {player} plays {move.uci()} "
              f"(value={root.q_value:.3f})")

        state = state.apply_action(action)

        if state.is_terminal():
            break

    print("\n(Demo limited to 10 moves)")
    print("\nFinal position:")
    print(state)


def main():
    print("AlphaZero Self-Play Demo")
    print("========================\n")

    # Run random evaluator demo
    demo_selfplay_random()

    # Run network demo
    demo_selfplay_network()

    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)
    print("""
Key concepts demonstrated:
1. MCTS search with neural network evaluation
2. Temperature-based action selection
3. Dirichlet noise for exploration
4. Visit count policy extraction
5. Game trajectory collection

In actual training:
- The network learns from self-play games
- Policy targets come from MCTS visit distributions
- Value targets come from game outcomes
- The network improves, leading to stronger MCTS
""")


if __name__ == "__main__":
    main()
