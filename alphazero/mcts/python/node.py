"""Pure Python MCTS node implementation.

This implementation prioritizes readability and correctness over performance.
Each method includes comments mapping to the AlphaZero paper equations.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List


class MCTSNode:
    """MCTS node with PUCT selection.

    Stores statistics for a single state-action pair:
        - N(s,a): Visit count
        - W(s,a): Total value (sum of backpropagated values)
        - Q(s,a): Mean value = W(s,a) / N(s,a)
        - P(s,a): Prior probability from neural network
    """

    __slots__ = [
        '_visit_count',
        '_value_sum',
        '_prior',
        '_children',
        '_legal_actions',
        '_is_terminal',
        '_terminal_value',
    ]

    def __init__(self, prior: float = 1.0):
        """Initialize a node.

        Args:
            prior: Prior probability P(s,a) from neural network
        """
        self._visit_count: int = 0
        self._value_sum: float = 0.0
        self._prior: float = prior
        self._children: Optional[Dict[int, 'MCTSNode']] = None
        self._legal_actions: Optional[np.ndarray] = None
        self._is_terminal: bool = False
        self._terminal_value: float = 0.0

    @property
    def visit_count(self) -> int:
        """N(s,a): Number of times this node has been visited."""
        return self._visit_count

    @property
    def value_sum(self) -> float:
        """W(s,a): Sum of all backpropagated values."""
        return self._value_sum

    @property
    def prior(self) -> float:
        """P(s,a): Prior probability from neural network."""
        return self._prior

    @prior.setter
    def prior(self, value: float):
        """Set prior (used for adding Dirichlet noise at root)."""
        self._prior = value

    @property
    def q_value(self) -> float:
        """Q(s,a) = W(s,a) / N(s,a): Mean action value."""
        if self._visit_count == 0:
            return 0.0
        return self._value_sum / self._visit_count

    def is_expanded(self) -> bool:
        """Check if this node has been expanded with children."""
        return self._children is not None

    def is_terminal(self) -> bool:
        """Check if this is a terminal node."""
        return self._is_terminal

    def set_terminal(self, value: float):
        """Mark this node as terminal with given value."""
        self._is_terminal = True
        self._terminal_value = value

    def get_terminal_value(self) -> float:
        """Get terminal value (only valid if is_terminal)."""
        return self._terminal_value

    def expand(self, priors: np.ndarray, legal_mask: np.ndarray) -> None:
        """Expand this node by creating child nodes for legal actions.

        Args:
            priors: Prior probabilities from neural network (num_actions,)
            legal_mask: Binary mask of legal actions (num_actions,)
        """
        self._children = {}
        self._legal_actions = np.where(legal_mask > 0)[0]

        # Renormalize priors over legal actions
        legal_priors = priors * legal_mask
        prior_sum = np.sum(legal_priors)
        if prior_sum > 0:
            legal_priors = legal_priors / prior_sum

        # Create child nodes for each legal action
        for action in self._legal_actions:
            self._children[action] = MCTSNode(prior=legal_priors[action])

    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        """Select the best child using PUCT algorithm.

        PUCT formula (from AlphaZero paper):
            a* = argmax_a [ Q(s,a) + U(s,a) ]

        Where the exploration bonus U(s,a) is:
            U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            c_puct: Exploration constant (typically 1.25)

        Returns:
            Tuple of (best_action, best_child)
        """
        if self._children is None:
            raise ValueError("Cannot select child from unexpanded node")

        best_action = -1
        best_score = float('-inf')
        best_child = None

        # sqrt(N(s)) - parent visit count
        sqrt_parent = np.sqrt(self._visit_count)

        for action, child in self._children.items():
            # Q(s,a): exploitation term
            q = child.q_value

            # U(s,a): exploration term
            # U = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            u = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)

            # PUCT score
            score = q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def get_children(self) -> Dict[int, 'MCTSNode']:
        """Get all child nodes."""
        return self._children or {}

    def get_child(self, action: int) -> Optional['MCTSNode']:
        """Get child node for a specific action."""
        if self._children is None:
            return None
        return self._children.get(action)

    def update(self, value: float) -> None:
        """Update node statistics during backpropagation.

        Args:
            value: Value to backpropagate (from current player's perspective)
        """
        self._visit_count += 1
        self._value_sum += value

    def get_visit_counts(self, num_actions: int) -> np.ndarray:
        """Get visit counts for all actions as an array.

        Args:
            num_actions: Total number of actions in action space

        Returns:
            Array of visit counts (num_actions,)
        """
        counts = np.zeros(num_actions, dtype=np.float32)
        if self._children is not None:
            for action, child in self._children.items():
                counts[action] = child.visit_count
        return counts

    def get_policy(self, num_actions: int, temperature: float = 1.0) -> np.ndarray:
        """Get policy distribution from visit counts.

        π(a) ∝ N(s,a)^(1/τ)

        Args:
            num_actions: Total number of actions
            temperature: Temperature parameter (lower = more greedy)

        Returns:
            Probability distribution over actions
        """
        counts = self.get_visit_counts(num_actions)

        if temperature < 0.01:
            # Greedy
            policy = np.zeros(num_actions, dtype=np.float32)
            if np.sum(counts) > 0:
                policy[np.argmax(counts)] = 1.0
            return policy

        # Apply temperature
        counts = np.power(counts, 1.0 / temperature)
        total = np.sum(counts)

        if total > 0:
            return counts / total
        return counts

    def __repr__(self) -> str:
        return (
            f"MCTSNode(N={self._visit_count}, W={self._value_sum:.3f}, "
            f"Q={self.q_value:.3f}, P={self._prior:.3f})"
        )
