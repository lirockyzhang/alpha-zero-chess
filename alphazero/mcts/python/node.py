"""Pure Python MCTS node implementation.

This implementation prioritizes readability and correctness over performance.
Each method includes comments mapping to the AlphaZero paper equations.

Supports virtual loss for parallel MCTS simulations.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import threading


class MCTSNode:
    """MCTS node with PUCT selection and virtual loss support.

    Stores statistics for a single state-action pair:
        - N(s,a): Visit count
        - W(s,a): Total value (sum of backpropagated values)
        - Q(s,a): Mean value = W(s,a) / N(s,a)
        - P(s,a): Prior probability from neural network
        - V(s,a): Virtual loss (for parallel MCTS)

    Virtual Loss:
        When running parallel simulations, multiple threads may select the same
        promising path. Virtual loss temporarily penalizes a node when it's being
        explored, encouraging other threads to explore different paths.

        During selection: Q_effective = (W - V*virtual_loss_value) / (N + V)
        After backprop: Virtual loss is removed and real value is added
    """

    __slots__ = [
        '_visit_count',
        '_value_sum',
        '_prior',
        '_children',
        '_legal_actions',
        '_is_terminal',
        '_terminal_value',
        '_virtual_loss',
        '_lock',
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
        self._virtual_loss: int = 0  # Number of virtual losses applied
        self._lock: Optional[threading.Lock] = None  # Lazy initialization

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

    @property
    def virtual_loss(self) -> int:
        """V(s,a): Number of virtual losses currently applied."""
        return self._virtual_loss

    def q_value_with_virtual_loss(self, virtual_loss_value: float = 1.0) -> float:
        """Q(s,a) adjusted for virtual loss.

        Q_effective = (W - V * virtual_loss_value) / (N + V)

        This makes nodes being explored by other threads appear less attractive,
        encouraging exploration of different paths.

        Args:
            virtual_loss_value: Penalty per virtual loss (typically 1.0)

        Returns:
            Adjusted Q value
        """
        effective_visits = self._visit_count + self._virtual_loss
        if effective_visits == 0:
            return 0.0
        effective_value = self._value_sum - self._virtual_loss * virtual_loss_value
        return effective_value / effective_visits

    def add_virtual_loss(self) -> None:
        """Add a virtual loss to this node.

        Called when a thread starts exploring through this node.
        Thread-safe if lock is initialized.
        """
        if self._lock:
            with self._lock:
                self._virtual_loss += 1
        else:
            self._virtual_loss += 1

    def remove_virtual_loss(self) -> None:
        """Remove a virtual loss from this node.

        Called during backpropagation after the real value is known.
        Thread-safe if lock is initialized.
        """
        if self._lock:
            with self._lock:
                self._virtual_loss = max(0, self._virtual_loss - 1)
        else:
            self._virtual_loss = max(0, self._virtual_loss - 1)

    def enable_threading(self) -> None:
        """Enable thread-safe operations by initializing the lock."""
        if self._lock is None:
            self._lock = threading.Lock()

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

    def select_child(
        self,
        c_puct: float,
        use_virtual_loss: bool = False,
        virtual_loss_value: float = 1.0
    ) -> Tuple[int, 'MCTSNode']:
        """Select the best child using PUCT algorithm.

        PUCT formula (from AlphaZero paper):
            a* = argmax_a [ Q(s,a) + U(s,a) ]

        Where the exploration bonus U(s,a) is:
            U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        With virtual loss enabled:
            Q_effective = (W - V * loss_value) / (N + V)
            U_effective = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a) + V)

        Args:
            c_puct: Exploration constant (typically 1.25)
            use_virtual_loss: Whether to account for virtual losses
            virtual_loss_value: Penalty per virtual loss

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
            if use_virtual_loss:
                # Q(s,a) with virtual loss: penalize nodes being explored
                q = child.q_value_with_virtual_loss(virtual_loss_value)
                # U(s,a) with virtual loss: treat virtual visits as real
                effective_visits = child.visit_count + child.virtual_loss
                u = c_puct * child.prior * sqrt_parent / (1 + effective_visits)
            else:
                # Standard Q(s,a): exploitation term
                q = child.q_value
                # Standard U(s,a): exploration term
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
        if self._lock:
            with self._lock:
                self._visit_count += 1
                self._value_sum += value
        else:
            self._visit_count += 1
            self._value_sum += value

    def update_and_remove_virtual_loss(self, value: float) -> None:
        """Update node and remove virtual loss atomically.

        Used in parallel MCTS during backpropagation.

        Args:
            value: Value to backpropagate
        """
        if self._lock:
            with self._lock:
                self._visit_count += 1
                self._value_sum += value
                self._virtual_loss = max(0, self._virtual_loss - 1)
        else:
            self._visit_count += 1
            self._value_sum += value
            self._virtual_loss = max(0, self._virtual_loss - 1)

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

        if temperature <= 0.01:
            # Greedy selection
            policy = np.zeros(num_actions, dtype=np.float32)
            if np.sum(counts) > 0:
                policy[np.argmax(counts)] = 1.0
            return policy

        # Apply temperature with numerical stability
        counts = counts.astype(np.float64)
        exponent = min(1.0 / temperature, 10.0)  # Cap to prevent overflow
        counts = np.power(counts, exponent)
        total = np.sum(counts)

        if total > 0:
            return (counts / total).astype(np.float32)
        return counts.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"MCTSNode(N={self._visit_count}, W={self._value_sum:.3f}, "
            f"Q={self.q_value:.3f}, P={self._prior:.3f})"
        )
