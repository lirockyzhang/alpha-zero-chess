"""Pure Python MCTS search implementation.

This implementation follows the AlphaZero paper closely with detailed comments.
"""

import numpy as np
from typing import Tuple, Optional, Any, List

from ..base import MCTSBase, MCTSStats
from ..evaluator import Evaluator
from .node import MCTSNode
from ...config import MCTSConfig


class PythonMCTS(MCTSBase):
    """Pure Python MCTS implementation.

    Algorithm overview:
    1. Selection: Traverse tree using PUCT until reaching unexpanded node
    2. Expansion: Expand node using neural network priors
    3. Evaluation: Get value estimate from neural network
    4. Backpropagation: Update statistics along the path

    This implementation is optimized for clarity, not speed.
    """

    def __init__(self, config: Optional[MCTSConfig] = None):
        """Initialize MCTS.

        Args:
            config: MCTS configuration
        """
        super().__init__(config)
        self.num_actions = 4672  # Chess action space size

    def search(
        self,
        state: Any,
        evaluator: Evaluator,
        move_number: int = 0,
        add_noise: bool = True
    ) -> Tuple[np.ndarray, MCTSNode, MCTSStats]:
        """Run MCTS search from the given state.

        Args:
            state: Current game state (GameState object)
            evaluator: Neural network evaluator
            move_number: Current move number (for temperature)
            add_noise: Whether to add Dirichlet noise at root

        Returns:
            Tuple of (policy, root_node, stats)
        """
        stats = MCTSStats()

        # Create root node
        root = MCTSNode(prior=1.0)

        # Get initial evaluation
        observation = state.get_observation()
        legal_mask = state.get_legal_actions()
        priors, value = evaluator.evaluate(observation, legal_mask)

        # Add Dirichlet noise at root for exploration
        if add_noise:
            priors = self.add_dirichlet_noise(priors, legal_mask)

        # Expand root
        root.expand(priors, legal_mask)
        root.update(value)
        stats.nodes_created = 1

        # Run simulations
        for sim in range(self.config.num_simulations):
            # Selection + Expansion + Evaluation + Backpropagation
            depth = self._simulate(root, state, evaluator, stats)
            stats.max_depth = max(stats.max_depth, depth)

        stats.num_simulations = self.config.num_simulations
        stats.root_value = root.q_value

        # Get policy from visit counts
        temperature = self.get_temperature(move_number)
        policy = root.get_policy(self.num_actions, temperature)

        return policy, root, stats

    def _simulate(
        self,
        root: MCTSNode,
        root_state: Any,
        evaluator: Evaluator,
        stats: MCTSStats
    ) -> int:
        """Run a single MCTS simulation.

        Args:
            root: Root node of the search tree
            root_state: Game state at root
            evaluator: Neural network evaluator
            stats: Statistics to update

        Returns:
            Depth reached in this simulation
        """
        node = root
        state = root_state
        path: List[Tuple[MCTSNode, int]] = []  # (node, action) pairs
        depth = 0

        # Selection: traverse tree until we reach an unexpanded node
        while node.is_expanded() and not node.is_terminal():
            action, child = node.select_child(self.config.c_puct)
            path.append((node, action))
            node = child
            state = state.apply_action(action)
            depth += 1

        # Check for terminal state
        if state.is_terminal():
            # Terminal value from current player's perspective
            value = state.get_value()
            node.set_terminal(value)
        elif not node.is_expanded():
            # Expansion and evaluation
            observation = state.get_observation()
            legal_mask = state.get_legal_actions()
            priors, value = evaluator.evaluate(observation, legal_mask)

            node.expand(priors, legal_mask)
            stats.nodes_created += 1
        else:
            # Node is terminal
            value = node.get_terminal_value()

        # Backpropagation
        # Value is from the perspective of the player at the leaf
        # We need to flip it as we go up the tree
        self._backpropagate(path, node, value)

        return depth

    def _backpropagate(
        self,
        path: List[Tuple[MCTSNode, int]],
        leaf: MCTSNode,
        value: float
    ) -> None:
        """Backpropagate value through the tree.

        The value alternates sign as we go up because players alternate.

        Args:
            path: List of (node, action) pairs from root to parent of leaf
            leaf: The leaf node
            value: Value at the leaf (from leaf player's perspective)
        """
        # Update leaf node
        leaf.update(value)

        # Backpropagate through path
        # Value flips at each level because players alternate
        for node, action in reversed(path):
            value = -value  # Flip for opponent
            node.update(value)

    def select_action(
        self,
        policy: np.ndarray,
        temperature: float = 1.0
    ) -> int:
        """Select an action from the policy distribution.

        Args:
            policy: Probability distribution over actions
            temperature: Temperature (0 = greedy, 1 = sample)

        Returns:
            Selected action index
        """
        if temperature < 0.01:
            # Greedy selection
            return int(np.argmax(policy))
        else:
            # Sample from distribution
            return int(np.random.choice(len(policy), p=policy))


def create_mcts(config: Optional[MCTSConfig] = None) -> PythonMCTS:
    """Factory function to create Python MCTS."""
    return PythonMCTS(config)
