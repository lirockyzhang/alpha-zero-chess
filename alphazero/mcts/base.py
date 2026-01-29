"""Abstract base classes for MCTS implementations.

Defines the interface that all MCTS backends must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Any
import numpy as np

from ..config import MCTSConfig


@dataclass
class MCTSStats:
    """Statistics from an MCTS search."""
    num_simulations: int = 0
    max_depth: int = 0
    root_value: float = 0.0
    nodes_created: int = 0


class MCTSNodeBase(ABC):
    """Abstract base class for MCTS nodes.

    All MCTS backends must implement this interface.
    """

    @property
    @abstractmethod
    def visit_count(self) -> int:
        """Number of times this node has been visited."""
        pass

    @property
    @abstractmethod
    def value_sum(self) -> float:
        """Sum of all values backpropagated through this node."""
        pass

    @property
    @abstractmethod
    def prior(self) -> float:
        """Prior probability from neural network."""
        pass

    @property
    def q_value(self) -> float:
        """Mean action value Q(s,a) = W(s,a) / N(s,a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @abstractmethod
    def is_expanded(self) -> bool:
        """Check if this node has been expanded."""
        pass

    @abstractmethod
    def expand(
        self,
        priors: np.ndarray,
        legal_mask: np.ndarray
    ) -> None:
        """Expand this node with child nodes.

        Args:
            priors: Prior probabilities from neural network (num_actions,)
            legal_mask: Binary mask of legal actions (num_actions,)
        """
        pass

    @abstractmethod
    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNodeBase']:
        """Select the best child using PUCT.

        PUCT formula:
            a* = argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)) ]

        Args:
            c_puct: Exploration constant

        Returns:
            Tuple of (action, child_node)
        """
        pass

    @abstractmethod
    def get_children(self) -> Dict[int, 'MCTSNodeBase']:
        """Get all child nodes."""
        pass

    @abstractmethod
    def update(self, value: float) -> None:
        """Update node statistics during backpropagation.

        Args:
            value: Value to backpropagate (from current player's perspective)
        """
        pass


class MCTSBase(ABC):
    """Abstract base class for MCTS search.

    All MCTS backends must implement this interface.
    """

    def __init__(self, config: Optional[MCTSConfig] = None):
        """Initialize MCTS.

        Args:
            config: MCTS configuration
        """
        self.config = config or MCTSConfig()

    @abstractmethod
    def search(
        self,
        state: Any,
        evaluator: Any,
        move_number: int = 0,
        add_noise: bool = True
    ) -> Tuple[np.ndarray, MCTSNodeBase, MCTSStats]:
        """Run MCTS search from the given state.

        Args:
            state: Current game state
            evaluator: Neural network evaluator
            move_number: Current move number (for temperature)
            add_noise: Whether to add Dirichlet noise at root

        Returns:
            Tuple of:
                - policy: Visit count distribution (num_actions,)
                - root: Root node of the search tree
                - stats: Search statistics
        """
        pass

    def get_temperature(self, move_number: int) -> float:
        """Get temperature for action selection.

        Args:
            move_number: Current move number

        Returns:
            Temperature value (1.0 for exploration, 0 for greedy exploitation)
        """
        if move_number < self.config.temperature_threshold:
            return self.config.temperature
        return 0.0  # Greedy after threshold

    def apply_temperature(
        self,
        visit_counts: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        """Apply temperature to visit counts to get policy.

        π(a) ∝ N(s,a)^(1/τ)

        Args:
            visit_counts: Raw visit counts (num_actions,)
            temperature: Temperature parameter

        Returns:
            Probability distribution over actions
        """
        if temperature <= 0.01:
            # Greedy selection (temperature near 0)
            policy = np.zeros_like(visit_counts, dtype=np.float32)
            if np.sum(visit_counts) > 0:
                policy[np.argmax(visit_counts)] = 1.0
            return policy

        # Apply temperature with numerical stability
        counts = visit_counts.astype(np.float64)
        # Clip to avoid overflow: 1/temperature can be large
        exponent = min(1.0 / temperature, 10.0)  # Cap at 10 to prevent overflow
        counts = np.power(counts, exponent)

        # Normalize
        total = np.sum(counts)
        if total > 0:
            return (counts / total).astype(np.float32)
        else:
            # Uniform if no visits
            legal = (visit_counts > 0).astype(np.float32)
            return legal / np.sum(legal)

    def add_dirichlet_noise(
        self,
        priors: np.ndarray,
        legal_mask: np.ndarray
    ) -> np.ndarray:
        """Add Dirichlet noise to root priors for exploration.

        P(s,a) = (1 - ε) * p_a + ε * η_a
        Where η ~ Dir(α)

        Args:
            priors: Prior probabilities from network
            legal_mask: Binary mask of legal actions

        Returns:
            Noisy priors
        """
        alpha = self.config.dirichlet_alpha
        epsilon = self.config.dirichlet_epsilon

        # Generate noise only for legal actions
        num_legal = int(np.sum(legal_mask))
        noise = np.random.dirichlet([alpha] * num_legal)

        # Map noise to full action space
        full_noise = np.zeros_like(priors)
        full_noise[legal_mask > 0] = noise

        # Blend with original priors
        noisy_priors = (1 - epsilon) * priors + epsilon * full_noise

        return noisy_priors
