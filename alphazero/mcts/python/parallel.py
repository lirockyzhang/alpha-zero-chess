"""Parallel MCTS implementation with virtual loss and batched inference.

This module provides a parallel MCTS implementation that can run multiple
simulations concurrently using threading. It uses virtual loss to encourage
exploration of different paths when multiple threads are searching.

Key features:
- Virtual loss for parallel tree search
- Support for batched inference (collect multiple leaf evaluations)
- Thread-safe node operations
"""

import numpy as np
import threading
from typing import Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging

from .node import MCTSNode
from ..base import MCTSStats
from ...config import MCTSConfig

logger = logging.getLogger(__name__)


@dataclass
class PendingEvaluation:
    """A leaf node waiting for neural network evaluation."""
    node: MCTSNode
    path: List[Tuple[MCTSNode, int]]  # (parent, action) pairs
    state: object  # GameState


class ParallelMCTS:
    """Parallel MCTS with virtual loss and optional batched inference.

    This implementation can run multiple simulations in parallel using threads.
    Virtual loss is used to discourage multiple threads from exploring the
    same path, improving search diversity.

    For maximum efficiency, combine with BatchedEvaluator to batch neural
    network evaluations across multiple pending leaf nodes.
    """

    def __init__(self, config: Optional[MCTSConfig] = None):
        """Initialize parallel MCTS.

        Args:
            config: MCTS configuration
        """
        if config is None:
            from ...config import MCTSConfig
            config = MCTSConfig()

        self.config = config
        self.num_actions = 4672  # Chess action space
        self.virtual_loss_value = 1.0  # Penalty per virtual loss

    def get_temperature(self, move_number: int) -> float:
        """Get temperature for action selection."""
        if move_number < self.config.temperature_threshold:
            return self.config.temperature
        return 0.0

    def add_dirichlet_noise(
        self,
        priors: np.ndarray,
        legal_mask: np.ndarray
    ) -> np.ndarray:
        """Add Dirichlet noise to root priors for exploration."""
        alpha = self.config.dirichlet_alpha
        epsilon = self.config.dirichlet_epsilon

        legal_indices = np.where(legal_mask > 0)[0]
        noise = np.random.dirichlet([alpha] * len(legal_indices))

        noisy_priors = priors.copy()
        noisy_priors[legal_indices] = (
            (1 - epsilon) * priors[legal_indices] + epsilon * noise
        )
        return noisy_priors

    def search(
        self,
        state,
        evaluator,
        move_number: int = 0,
        add_noise: bool = True,
        num_threads: int = 1
    ) -> Tuple[np.ndarray, MCTSNode, MCTSStats]:
        """Run parallel MCTS search.

        Args:
            state: Current game state
            evaluator: Neural network evaluator
            move_number: Current move number (for temperature)
            add_noise: Whether to add Dirichlet noise at root
            num_threads: Number of parallel threads (1 = sequential)

        Returns:
            Tuple of (policy, root_node, stats)
        """
        stats = MCTSStats()

        # Create and expand root node
        root = MCTSNode(prior=1.0)

        # Get initial evaluation
        observation = state.get_observation()
        legal_mask = state.get_legal_actions()
        priors, value = evaluator.evaluate(observation, legal_mask)

        # Add Dirichlet noise at root
        if add_noise:
            priors = self.add_dirichlet_noise(priors, legal_mask)

        root.expand(priors, legal_mask)
        root.update(value)
        stats.nodes_created = 1

        if num_threads <= 1:
            # Sequential search (no virtual loss needed)
            for _ in range(self.config.num_simulations):
                depth = self._simulate_sequential(root, state, evaluator, stats)
                stats.max_depth = max(stats.max_depth, depth)
        else:
            # Parallel search with virtual loss
            self._search_parallel(root, state, evaluator, stats, num_threads)

        stats.num_simulations = self.config.num_simulations
        stats.root_value = root.q_value

        # Get policy from visit counts
        temperature = self.get_temperature(move_number)
        policy = root.get_policy(self.num_actions, temperature)

        return policy, root, stats

    def _simulate_sequential(
        self,
        root: MCTSNode,
        root_state,
        evaluator,
        stats: MCTSStats
    ) -> int:
        """Run a single sequential simulation (no virtual loss)."""
        node = root
        state = root_state
        path = []
        depth = 0

        # Selection: traverse tree using PUCT
        while node.is_expanded() and not node.is_terminal():
            action, child = node.select_child(self.config.c_puct)
            path.append((node, action))
            node = child
            state = state.apply_action(action)
            depth += 1

        # Evaluate leaf
        if state.is_terminal():
            value = state.get_value()
            node.set_terminal(value)
        elif not node.is_expanded():
            observation = state.get_observation()
            legal_mask = state.get_legal_actions()
            priors, value = evaluator.evaluate(observation, legal_mask)
            node.expand(priors, legal_mask)
            stats.nodes_created += 1
        else:
            value = node.get_terminal_value()

        # Backpropagation
        node.update(value)
        for parent, action in reversed(path):
            value = -value
            parent.update(value)

        return depth

    def _search_parallel(
        self,
        root: MCTSNode,
        root_state,
        evaluator,
        stats: MCTSStats,
        num_threads: int
    ):
        """Run parallel search with virtual loss."""
        # Enable threading on root (children will inherit during expansion)
        root.enable_threading()

        # Track statistics thread-safely
        stats_lock = threading.Lock()
        nodes_created = [0]
        max_depth = [0]

        def run_simulation():
            """Run a single simulation with virtual loss."""
            node = root
            state = root_state
            path = []
            depth = 0

            # Selection with virtual loss
            while node.is_expanded() and not node.is_terminal():
                action, child = node.select_child(
                    self.config.c_puct,
                    use_virtual_loss=True,
                    virtual_loss_value=self.virtual_loss_value
                )
                child.add_virtual_loss()
                path.append((node, action, child))
                node = child
                state = state.apply_action(action)
                depth += 1

            # Evaluate leaf
            if state.is_terminal():
                value = state.get_value()
                node.set_terminal(value)
            elif not node.is_expanded():
                observation = state.get_observation()
                legal_mask = state.get_legal_actions()
                priors, value = evaluator.evaluate(observation, legal_mask)

                # Thread-safe expansion
                if not node.is_expanded():
                    node.enable_threading()
                    node.expand(priors, legal_mask)
                    with stats_lock:
                        nodes_created[0] += 1
            else:
                value = node.get_terminal_value()

            # Backpropagation with virtual loss removal
            node.update(value)
            for parent, action, child in reversed(path):
                value = -value
                parent.update(value)
                child.remove_virtual_loss()

            with stats_lock:
                max_depth[0] = max(max_depth[0], depth)

        # Run simulations in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(run_simulation)
                for _ in range(self.config.num_simulations)
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Simulation error: {e}")

        stats.nodes_created += nodes_created[0]
        stats.max_depth = max_depth[0]

    def search_with_batching(
        self,
        state,
        evaluator,
        move_number: int = 0,
        add_noise: bool = True,
        batch_size: int = 8
    ) -> Tuple[np.ndarray, MCTSNode, MCTSStats]:
        """Run MCTS with batched leaf evaluation.

        Instead of evaluating leaves one at a time, this collects multiple
        pending leaves and evaluates them in a batch. This is more efficient
        when using GPU inference.

        Args:
            state: Current game state
            evaluator: Neural network evaluator (should support evaluate_batch)
            move_number: Current move number
            add_noise: Whether to add Dirichlet noise
            batch_size: Number of leaves to batch together

        Returns:
            Tuple of (policy, root_node, stats)
        """
        stats = MCTSStats()

        # Create and expand root
        root = MCTSNode(prior=1.0)
        root.enable_threading()

        observation = state.get_observation()
        legal_mask = state.get_legal_actions()
        priors, value = evaluator.evaluate(observation, legal_mask)

        if add_noise:
            priors = self.add_dirichlet_noise(priors, legal_mask)

        root.expand(priors, legal_mask)
        root.update(value)
        stats.nodes_created = 1

        simulations_done = 0

        while simulations_done < self.config.num_simulations:
            # Collect batch of pending evaluations
            pending = []
            batch_count = min(batch_size, self.config.num_simulations - simulations_done)

            for _ in range(batch_count):
                pending_eval = self._select_leaf_with_virtual_loss(root, state)
                if pending_eval:
                    pending.append(pending_eval)

            if not pending:
                break

            # Batch evaluate all pending leaves
            self._batch_evaluate_and_backprop(pending, evaluator, stats)
            simulations_done += len(pending)

        stats.num_simulations = simulations_done
        stats.root_value = root.q_value

        temperature = self.get_temperature(move_number)
        policy = root.get_policy(self.num_actions, temperature)

        return policy, root, stats

    def _select_leaf_with_virtual_loss(
        self,
        root: MCTSNode,
        root_state
    ) -> Optional[PendingEvaluation]:
        """Select a leaf node, applying virtual loss along the path."""
        node = root
        state = root_state
        path = []

        while node.is_expanded() and not node.is_terminal():
            action, child = node.select_child(
                self.config.c_puct,
                use_virtual_loss=True,
                virtual_loss_value=self.virtual_loss_value
            )
            child.add_virtual_loss()
            path.append((node, action))
            node = child
            state = state.apply_action(action)

        if node.is_terminal():
            # Terminal node - backprop immediately
            value = node.get_terminal_value()
            self._backprop_with_virtual_loss_removal(path, node, value)
            return None

        return PendingEvaluation(node=node, path=path, state=state)

    def _batch_evaluate_and_backprop(
        self,
        pending: List[PendingEvaluation],
        evaluator,
        stats: MCTSStats
    ):
        """Evaluate a batch of leaves and backpropagate."""
        # Check for terminal states first
        non_terminal = []
        for p in pending:
            if p.state.is_terminal():
                value = p.state.get_value()
                p.node.set_terminal(value)
                self._backprop_with_virtual_loss_removal(p.path, p.node, value)
            else:
                non_terminal.append(p)

        if not non_terminal:
            return

        # Batch evaluate non-terminal leaves
        observations = np.stack([p.state.get_observation() for p in non_terminal])
        legal_masks = np.stack([p.state.get_legal_actions() for p in non_terminal])

        # Use batch evaluation if available
        if hasattr(evaluator, 'evaluate_batch'):
            policies, values = evaluator.evaluate_batch(observations, legal_masks)
        else:
            # Fall back to individual evaluation
            policies = []
            values = []
            for obs, mask in zip(observations, legal_masks):
                policy, value = evaluator.evaluate(obs, mask)
                policies.append(policy)
                values.append(value)
            policies = np.stack(policies)
            values = np.array(values)

        # Expand and backprop each leaf
        for i, p in enumerate(non_terminal):
            if not p.node.is_expanded():
                p.node.enable_threading()
                p.node.expand(policies[i], legal_masks[i])
                stats.nodes_created += 1

            value = float(values[i])
            self._backprop_with_virtual_loss_removal(p.path, p.node, value)

    def _backprop_with_virtual_loss_removal(
        self,
        path: List[Tuple[MCTSNode, int]],
        leaf: MCTSNode,
        value: float
    ):
        """Backpropagate value and remove virtual losses."""
        leaf.update(value)

        for parent, action in reversed(path):
            value = -value
            parent.update(value)
            # Remove virtual loss from the child we traversed through
            child = parent.get_child(action)
            if child:
                child.remove_virtual_loss()
