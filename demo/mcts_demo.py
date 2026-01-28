#!/usr/bin/env python3
"""Step-by-step MCTS visualization demo.

This demo shows how MCTS works by visualizing each step of the algorithm.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from alphazero import MCTSConfig
from alphazero.chess_env import GameState
from alphazero.mcts.python.node import MCTSNode
from alphazero.mcts.evaluator import RandomEvaluator


def print_tree(node: MCTSNode, state: GameState, depth: int = 0, max_depth: int = 2):
    """Print MCTS tree structure."""
    indent = "  " * depth

    if depth == 0:
        print(f"{indent}Root: N={node.visit_count}, Q={node.q_value:.3f}")
    else:
        print(f"{indent}N={node.visit_count}, Q={node.q_value:.3f}, P={node.prior:.3f}")

    if depth < max_depth and node.is_expanded():
        children = node.get_children()
        # Sort by visit count
        sorted_children = sorted(children.items(), key=lambda x: -x[1].visit_count)

        for action, child in sorted_children[:5]:  # Top 5 children
            move = state.action_to_move(action)
            print(f"{indent}  └─ {move.uci()}: ", end="")
            print_tree(child, state.apply_action(action), depth + 1, max_depth)


def demo_mcts_step_by_step():
    """Demonstrate MCTS algorithm step by step."""
    print("=" * 60)
    print("MCTS Step-by-Step Demo")
    print("=" * 60)

    # Setup
    config = MCTSConfig(num_simulations=50, c_puct=1.25)
    evaluator = RandomEvaluator()
    state = GameState()

    print("\nInitial position:")
    print(state)

    # Create root node
    root = MCTSNode(prior=1.0)

    # Get initial evaluation
    observation = state.get_observation()
    legal_mask = state.get_legal_actions()
    priors, value = evaluator.evaluate(observation, legal_mask)

    print(f"\nNeural network evaluation:")
    print(f"  Value: {value:.3f}")
    print(f"  Legal moves: {int(legal_mask.sum())}")

    # Add Dirichlet noise
    alpha = config.dirichlet_alpha
    epsilon = config.dirichlet_epsilon
    num_legal = int(np.sum(legal_mask))
    noise = np.random.dirichlet([alpha] * num_legal)
    full_noise = np.zeros_like(priors)
    full_noise[legal_mask > 0] = noise
    noisy_priors = (1 - epsilon) * priors + epsilon * full_noise

    print(f"\nDirichlet noise added (α={alpha}, ε={epsilon})")

    # Expand root
    root.expand(noisy_priors, legal_mask)
    root.update(value)

    print(f"\nRoot expanded with {len(root.get_children())} children")

    # Run simulations
    print("\n" + "-" * 60)
    print("Running MCTS simulations...")
    print("-" * 60)

    for sim in range(config.num_simulations):
        # Selection
        node = root
        current_state = state
        path = []

        while node.is_expanded() and not current_state.is_terminal():
            action, child = node.select_child(config.c_puct)
            path.append((node, action))
            node = child
            current_state = current_state.apply_action(action)

        # Expansion and evaluation
        if not current_state.is_terminal() and not node.is_expanded():
            obs = current_state.get_observation()
            mask = current_state.get_legal_actions()
            p, v = evaluator.evaluate(obs, mask)
            node.expand(p, mask)
        else:
            v = current_state.get_value() if current_state.is_terminal() else 0.0

        # Backpropagation
        node.update(v)
        for parent, action in reversed(path):
            v = -v
            parent.update(v)

        # Print progress
        if (sim + 1) % 10 == 0:
            print(f"\nAfter {sim + 1} simulations:")
            print(f"  Root visits: {root.visit_count}")
            print(f"  Root value: {root.q_value:.3f}")

            # Top moves
            children = root.get_children()
            sorted_children = sorted(children.items(), key=lambda x: -x[1].visit_count)
            print("  Top moves:")
            for action, child in sorted_children[:3]:
                move = state.action_to_move(action)
                print(f"    {move.uci()}: N={child.visit_count}, Q={child.q_value:.3f}")

    # Final results
    print("\n" + "=" * 60)
    print("Final MCTS Results")
    print("=" * 60)

    print(f"\nTotal simulations: {root.visit_count}")
    print(f"Root value: {root.q_value:.3f}")

    # Get policy
    visit_counts = root.get_visit_counts(4672)
    policy = visit_counts / visit_counts.sum()

    print("\nTop 5 moves by visit count:")
    top_actions = np.argsort(-visit_counts)[:5]
    for action in top_actions:
        if visit_counts[action] > 0:
            move = state.action_to_move(action)
            child = root.get_child(action)
            print(f"  {move.uci()}: visits={int(visit_counts[action])}, "
                  f"policy={policy[action]:.3f}, Q={child.q_value:.3f}")

    # PUCT formula explanation
    print("\n" + "-" * 60)
    print("PUCT Formula Explanation")
    print("-" * 60)
    print("""
The PUCT (Polynomial Upper Confidence Trees) formula:

    a* = argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)) ]

Where:
    Q(s,a)  = Mean value of action a (exploitation)
    P(s,a)  = Prior probability from neural network
    N(s)    = Visit count of parent node
    N(s,a)  = Visit count of action a
    c_puct  = Exploration constant (1.25 for chess)

The formula balances:
    - Exploitation: Prefer actions with high Q values
    - Exploration: Prefer actions with high priors and low visit counts
""")


if __name__ == "__main__":
    demo_mcts_step_by_step()
