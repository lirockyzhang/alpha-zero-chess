#!/usr/bin/env python3
"""Neural network architecture demo.

This demo shows the AlphaZero network architecture and forward pass.
"""

import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from alphazero.chess_env import GameState
from alphazero.neural import AlphaZeroNetwork, count_parameters


def demo_network():
    """Demonstrate the neural network architecture."""
    print("=" * 60)
    print("AlphaZero Neural Network Demo")
    print("=" * 60)

    # Create network
    print("\nCreating network...")
    network = AlphaZeroNetwork(
        input_channels=119,
        num_filters=192,
        num_blocks=15,
        num_actions=4672
    )

    print(f"\nNetwork Architecture:")
    print(f"  Input: 119 planes × 8 × 8")
    print(f"  Initial conv: 3×3, 192 filters")
    print(f"  Residual blocks: 15")
    print(f"  Policy head: 1×1 conv → FC(4672)")
    print(f"  Value head: 1×1 conv → FC(192) → FC(1)")

    print(f"\nTotal parameters: {count_parameters(network):,}")

    # Count parameters by component
    input_params = sum(p.numel() for p in network.input_conv.parameters())
    tower_params = sum(p.numel() for p in network.residual_tower.parameters())
    policy_params = sum(p.numel() for p in network.policy_head.parameters())
    value_params = sum(p.numel() for p in network.value_head.parameters())

    print(f"\nParameters by component:")
    print(f"  Input conv: {input_params:,}")
    print(f"  Residual tower: {tower_params:,}")
    print(f"  Policy head: {policy_params:,}")
    print(f"  Value head: {value_params:,}")

    # Create sample input
    print("\n" + "-" * 60)
    print("Forward Pass Demo")
    print("-" * 60)

    state = GameState()
    observation = state.get_observation()
    legal_mask = state.get_legal_actions()

    print(f"\nInput observation shape: {observation.shape}")
    print(f"Legal moves: {int(legal_mask.sum())}")

    # Convert to tensors
    obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
    mask_tensor = torch.from_numpy(legal_mask).float().unsqueeze(0)

    # Forward pass
    network.eval()
    with torch.no_grad():
        policy_logits, value = network(obs_tensor, mask_tensor)

    print(f"\nOutput shapes:")
    print(f"  Policy logits: {policy_logits.shape}")
    print(f"  Value: {value.shape}")

    # Get probabilities
    policy = torch.softmax(policy_logits, dim=-1).squeeze().numpy()
    value_scalar = value.item()

    print(f"\nNetwork evaluation:")
    print(f"  Value: {value_scalar:.4f}")
    print(f"  Policy entropy: {-np.sum(policy * np.log(policy + 1e-8)):.4f}")

    # Top moves
    print("\nTop 5 moves by policy:")
    top_actions = np.argsort(-policy)[:5]
    for action in top_actions:
        if legal_mask[action] > 0:
            move = state.action_to_move(action)
            print(f"  {move.uci()}: {policy[action]:.4f}")

    # Encoding explanation
    print("\n" + "-" * 60)
    print("Input Encoding (119 planes)")
    print("-" * 60)
    print("""
Plane layout:
  0-11:   Current position (6 piece types × 2 colors)
  12-23:  Position T-1 (one move ago)
  24-35:  Position T-2
  ...     (8 history positions total = 96 planes)
  96-99:  Castling rights (4 planes)
  100:    Side to move
  101-108: Repetition counters
  109-118: Move clocks (halfmove, fullmove)

Each plane is 8×8, representing the chess board.
""")

    # Action space explanation
    print("-" * 60)
    print("Action Space (4672 actions)")
    print("-" * 60)
    print("""
Action encoding:
  0-3583:    Queen-like moves (56 directions × 64 squares)
             - 7 distances × 8 directions (N, NE, E, SE, S, SW, W, NW)
  3584-4095: Knight moves (8 × 64 = 512)
  4096-4671: Underpromotions (9 × 64 = 576)
             - 3 piece types × 3 directions × 64 squares

Queen promotions are encoded as regular pawn moves to the 8th rank.
""")


if __name__ == "__main__":
    demo_network()
