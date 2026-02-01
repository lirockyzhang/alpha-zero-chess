"""C++ MCTS backend integration for AlphaZero training.

This module provides the integration layer between the C++ MCTS engine
(alphazero-cpp) and the Python training infrastructure.

According to batched_mcts.md Phase 4 requirements:
- End-to-end testing with full self-play games
- Performance profiling
- Verification of training data format

Two backends are available:
1. CppMCTS: Fast but only evaluates root node (not true AlphaZero)
2. CppBatchedMCTS: Proper AlphaZero with batch leaf evaluation
"""

import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np

from ..base import MCTSBase, MCTSNodeBase, MCTSStats

# Add C++ extension to path
# Try multiple possible locations for the C++ extension
cpp_search_paths = [
    Path(__file__).parent.parent.parent.parent / "alphazero-cpp" / "build" / "Release",
    Path(__file__).parent.parent.parent / "alphazero-cpp" / "build" / "Release",
    Path("alphazero-cpp") / "build" / "Release",
]

for cpp_path in cpp_search_paths:
    if cpp_path.exists():
        sys.path.insert(0, str(cpp_path))
        break

try:
    import alphazero_cpp
    CPP_AVAILABLE = True
except ImportError as e:
    CPP_AVAILABLE = False
    alphazero_cpp = None
    _import_error = str(e)


class CppMCTSNode(MCTSNodeBase):
    """Wrapper for C++ MCTS root node to match Python interface."""

    def __init__(self, visit_count: int, value_sum: float):
        self._visit_count = visit_count
        self._value_sum = value_sum

    @property
    def visit_count(self) -> int:
        return self._visit_count

    @property
    def value_sum(self) -> float:
        return self._value_sum

    @property
    def prior(self) -> float:
        return 1.0  # Root node has no prior

    def is_expanded(self) -> bool:
        return True

    def expand(self, priors: np.ndarray, legal_mask: np.ndarray) -> None:
        pass  # C++ handles expansion internally

    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNodeBase']:
        raise NotImplementedError("C++ MCTS handles tree traversal internally")

    def get_children(self) -> Dict[int, 'MCTSNodeBase']:
        return {}  # C++ tree not exposed to Python

    def update(self, value: float) -> None:
        pass  # C++ handles updates internally


class CppBatchedMCTS(MCTSBase):
    """C++ Batched MCTS backend - PROPER AlphaZero implementation.

    This class implements true AlphaZero MCTS where every leaf node
    gets a neural network evaluation. The search loop is:

    1. Initialize search with root position and NN evaluation
    2. Loop until all simulations complete:
       a. C++ collects leaves that need evaluation
       b. Python evaluates leaves with neural network (batched)
       c. C++ updates leaves and continues search
    3. Return visit count distribution as policy

    This is the CORRECT implementation for training. The simpler CppMCTS
    only evaluates the root node, which is faster but produces lower
    quality training data.
    """

    def __init__(self, config):
        """Initialize C++ Batched MCTS engine.

        Args:
            config: MCTSConfig object with search parameters
        """
        super().__init__(config)

        if not CPP_AVAILABLE:
            raise RuntimeError(
                "C++ MCTS backend not available. "
                "Build alphazero-cpp extension first."
            )

        self.config = config

        # Batch size for leaf evaluation (can be tuned)
        self.batch_size = getattr(config, 'batch_size', 64)

        # Create C++ batched MCTS search engine
        self.search_engine = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=config.num_simulations,
            batch_size=self.batch_size,
            c_puct=config.c_puct
        )

        # Statistics
        self.total_searches = 0
        self.total_simulations = 0
        self.total_batches = 0
        self.total_leaves_evaluated = 0

    def search(
        self,
        state,
        evaluator,
        move_number: int = 0,
        add_noise: bool = True
    ) -> Tuple[np.ndarray, MCTSNodeBase, MCTSStats]:
        """Run batched MCTS search from the given state.

        This implements the proper AlphaZero search loop where every
        leaf node gets a neural network evaluation.

        Args:
            state: GameState object (chess position)
            evaluator: NetworkEvaluator for position evaluation
            move_number: Current move number (for temperature)
            add_noise: Whether to add Dirichlet noise at root

        Returns:
            Tuple of:
                - policy: np.ndarray of shape (4672,) with visit count distribution
                - root: Root node of the search tree
                - stats: Search statistics
        """
        # Get FEN representation
        fen = state.board.fen()

        # Get observation and legal mask from state
        observation = state.get_observation()
        legal_mask = state.get_legal_actions()

        # Get initial neural network evaluation for root
        root_policy, root_value = evaluator.evaluate(observation, legal_mask)

        # Initialize search with root evaluation
        self.search_engine.init_search(
            fen,
            root_policy.astype(np.float32),
            float(root_value)
        )

        # Main search loop: collect leaves, evaluate, update
        batches_processed = 0
        leaves_evaluated = 0

        # Check if evaluator supports batch evaluation
        has_batch_eval = hasattr(evaluator, 'evaluate_batch')

        while not self.search_engine.is_complete():
            # Collect leaves that need evaluation
            num_leaves, obs_batch, mask_batch = self.search_engine.collect_leaves()

            if num_leaves == 0:
                # No more leaves to evaluate (search complete or all terminal)
                break

            # Evaluate leaves with neural network
            # obs_batch shape from C++: (num_leaves, 8, 8, 119) - NHWC format
            # Network expects: (batch, 119, 8, 8) - NCHW format
            # mask_batch shape: (num_leaves, 4672)

            # Transpose from NHWC to NCHW for the entire batch at once
            obs_nchw = np.transpose(obs_batch[:num_leaves], (0, 3, 1, 2))  # (N, 119, 8, 8)
            masks = mask_batch[:num_leaves]  # (N, 4672)

            if has_batch_eval:
                # Use efficient batch evaluation (single GPU call)
                policies_array, values_array = evaluator.evaluate_batch(obs_nchw, masks)
                values_array = values_array.squeeze(-1) if values_array.ndim > 1 else values_array
            else:
                # Fallback to single evaluation (slower)
                policies = []
                values = []
                for i in range(num_leaves):
                    policy, value = evaluator.evaluate(obs_nchw[i], masks[i])
                    policies.append(policy.astype(np.float32))
                    values.append(float(value))
                policies_array = np.array(policies, dtype=np.float32)
                values_array = np.array(values, dtype=np.float32)

            # Update leaves with evaluation results
            self.search_engine.update_leaves(
                policies_array.astype(np.float32),
                values_array.astype(np.float32)
            )

            batches_processed += 1
            leaves_evaluated += num_leaves

        # Get visit counts
        visit_counts = self.search_engine.get_visit_counts()

        # Update statistics
        self.total_searches += 1
        self.total_simulations += self.search_engine.get_simulations_completed()
        self.total_batches += batches_processed
        self.total_leaves_evaluated += leaves_evaluated

        # Convert visit counts to policy (normalized)
        policy = visit_counts.astype(np.float32)

        # Mask policy with legal moves
        policy = policy * legal_mask

        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # Fallback to uniform over legal moves
            policy = legal_mask / legal_mask.sum()

        # Create root node wrapper
        total_visits = int(visit_counts.sum())
        root = CppMCTSNode(visit_count=total_visits, value_sum=float(root_value) * total_visits)

        # Create stats
        stats = MCTSStats(
            num_simulations=self.search_engine.get_simulations_completed(),
            max_depth=0,
            root_value=float(root_value),
            nodes_created=leaves_evaluated
        )

        # Reset for next search
        self.search_engine.reset()

        return policy, root, stats

    def get_stats(self) -> dict:
        """Get search statistics."""
        return {
            'total_searches': self.total_searches,
            'total_simulations': self.total_simulations,
            'total_batches': self.total_batches,
            'total_leaves_evaluated': self.total_leaves_evaluated,
            'avg_simulations_per_search': (
                self.total_simulations / self.total_searches
                if self.total_searches > 0 else 0
            ),
            'avg_leaves_per_search': (
                self.total_leaves_evaluated / self.total_searches
                if self.total_searches > 0 else 0
            ),
        }


class CppMCTS(MCTSBase):
    """C++ MCTS backend - FAST but simplified (root-only evaluation).

    WARNING: This implementation only evaluates the root node with the
    neural network. All leaf expansions use uniform priors. This is
    FASTER but produces LOWER QUALITY training data.

    For proper AlphaZero training, use CppBatchedMCTS instead.

    Performance targets (from batched_mcts.md):
    - Move generation: 5-10M moves/sec (achieved: 189-422M nps)
    - MCTS simulations: 50K-100K sims/sec (achieved: 362K NPS)
    - Batch encoding: <1ms for 256 positions (achieved: 0.272ms)
    """

    def __init__(self, config):
        """Initialize C++ MCTS engine.

        Args:
            config: MCTSConfig object with search parameters
        """
        super().__init__(config)

        if not CPP_AVAILABLE:
            raise RuntimeError(
                "C++ MCTS backend not available. "
                "Build alphazero-cpp extension first."
            )

        self.config = config

        # Create C++ MCTS search engine
        self.search_engine = alphazero_cpp.MCTSSearch(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct
        )

        # Statistics
        self.total_searches = 0
        self.total_simulations = 0

    def search(
        self,
        state,
        evaluator,
        move_number: int = 0,
        add_noise: bool = True
    ) -> Tuple[np.ndarray, MCTSNodeBase, MCTSStats]:
        """Run MCTS search from the given state.

        NOTE: This only evaluates the root node. For proper AlphaZero,
        use CppBatchedMCTS instead.

        Args:
            state: GameState object (chess position)
            evaluator: NetworkEvaluator for position evaluation
            move_number: Current move number (for temperature)
            add_noise: Whether to add Dirichlet noise at root

        Returns:
            Tuple of:
                - policy: np.ndarray of shape (4672,) with visit count distribution
                - root: Root node of the search tree
                - stats: Search statistics
        """
        # Get FEN representation
        fen = state.board.fen()

        # Get observation and legal mask from state
        observation = state.get_observation()
        legal_mask = state.get_legal_actions()

        # Get neural network evaluation
        policy_probs, value = evaluator.evaluate(observation, legal_mask)

        # Run C++ MCTS search
        visit_counts = self.search_engine.search(
            fen,
            policy_probs.astype(np.float32),
            float(value)
        )

        # Update statistics
        self.total_searches += 1
        self.total_simulations += self.config.num_simulations

        # Convert visit counts to policy (normalized)
        policy = visit_counts.astype(np.float32)

        # CRITICAL: Mask policy with legal moves to prevent illegal move selection
        policy = policy * legal_mask

        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            policy = legal_mask / legal_mask.sum()

        # Create root node wrapper
        total_visits = int(visit_counts.sum())
        root_value = float(value)
        root = CppMCTSNode(visit_count=total_visits, value_sum=root_value * total_visits)

        # Create stats
        stats = MCTSStats(
            num_simulations=self.config.num_simulations,
            max_depth=0,
            root_value=root_value,
            nodes_created=0
        )

        return policy, root, stats

    def select_action(self, policy: np.ndarray, temperature: float = 1.0) -> int:
        """Select action from MCTS policy using temperature."""
        if temperature == 0:
            return int(np.argmax(policy))
        else:
            policy_temp = np.power(policy, 1.0 / temperature)
            policy_temp = policy_temp / policy_temp.sum()
            return int(np.random.choice(len(policy), p=policy_temp))

    def reset(self):
        """Reset the search tree."""
        self.search_engine.reset()

    def get_stats(self) -> dict:
        """Get search statistics."""
        return {
            'total_searches': self.total_searches,
            'total_simulations': self.total_simulations,
            'avg_simulations_per_search': (
                self.total_simulations / self.total_searches
                if self.total_searches > 0 else 0
            ),
        }


def is_cpp_available() -> bool:
    """Check if C++ MCTS backend is available."""
    return CPP_AVAILABLE


def create_cpp_mcts(config, use_batched: bool = True):
    """Factory function to create C++ MCTS instance.

    Args:
        config: MCTSConfig object
        use_batched: If True, use CppBatchedMCTS (proper AlphaZero).
                     If False, use CppMCTS (fast but root-only eval).

    Returns:
        MCTS instance

    Raises:
        RuntimeError: If C++ backend is not available
    """
    if not CPP_AVAILABLE:
        raise RuntimeError(
            "C++ MCTS backend not available. "
            "Build alphazero-cpp extension first:\n"
            "  cd alphazero-cpp\n"
            "  cmake -B build -DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build build --config Release"
        )

    if use_batched:
        return CppBatchedMCTS(config)
    else:
        return CppMCTS(config)


# Export public API
__all__ = [
    'CppMCTS',
    'CppBatchedMCTS',
    'is_cpp_available',
    'create_cpp_mcts',
    'CPP_AVAILABLE',
]
