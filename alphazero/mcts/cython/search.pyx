# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Cython-optimized MCTS search implementation."""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

from .node cimport CythonMCTSNode
from ..base import MCTSBase, MCTSStats

np.import_array()

ctypedef np.float32_t FLOAT_t


cdef class CythonMCTS:
    """Cython-optimized MCTS implementation.

    Algorithm overview:
    1. Selection: Traverse tree using PUCT until reaching unexpanded node
    2. Expansion: Expand node using neural network priors
    3. Evaluation: Get value estimate from neural network
    4. Backpropagation: Update statistics along the path
    """

    cdef public object config
    cdef public int num_actions

    def __init__(self, config=None):
        """Initialize MCTS.

        Args:
            config: MCTS configuration
        """
        from ...config import MCTSConfig
        self.config = config if config is not None else MCTSConfig()
        self.num_actions = 4672  # Chess action space size

    cpdef double get_temperature(self, int move_number):
        """Get temperature for action selection."""
        if move_number < self.config.temperature_threshold:
            return self.config.temperature
        return 0.0

    cpdef np.ndarray apply_temperature(self, np.ndarray visit_counts, double temperature):
        """Apply temperature to visit counts to get policy."""
        cdef np.ndarray[np.float64_t, ndim=1] counts
        cdef double total, exponent

        if temperature <= 0.01:
            policy = np.zeros_like(visit_counts, dtype=np.float32)
            if np.sum(visit_counts) > 0:
                policy[np.argmax(visit_counts)] = 1.0
            return policy

        counts = visit_counts.astype(np.float64)
        exponent = min(1.0 / temperature, 10.0)
        counts = np.power(counts, exponent)

        total = np.sum(counts)
        if total > 0:
            return (counts / total).astype(np.float32)
        else:
            legal = (visit_counts > 0).astype(np.float32)
            return legal / np.sum(legal)

    cpdef np.ndarray add_dirichlet_noise(self, np.ndarray priors, np.ndarray legal_mask):
        """Add Dirichlet noise to root priors for exploration."""
        cdef double alpha = self.config.dirichlet_alpha
        cdef double epsilon = self.config.dirichlet_epsilon
        cdef int num_legal = int(np.sum(legal_mask))

        noise = np.random.dirichlet([alpha] * num_legal)
        full_noise = np.zeros_like(priors)
        full_noise[legal_mask > 0] = noise

        return (1 - epsilon) * priors + epsilon * full_noise

    def search(self, state, evaluator, int move_number=0, bint add_noise=True):
        """Run MCTS search from the given state.

        Args:
            state: Current game state (GameState object)
            evaluator: Neural network evaluator
            move_number: Current move number (for temperature)
            add_noise: Whether to add Dirichlet noise at root

        Returns:
            Tuple of (policy, root_node, stats)
        """
        cdef int sim, depth
        cdef double value
        cdef CythonMCTSNode root

        stats = MCTSStats()

        # Create root node
        root = CythonMCTSNode(prior=1.0)

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
            depth = self._simulate(root, state, evaluator, stats)
            stats.max_depth = max(stats.max_depth, depth)

        stats.num_simulations = self.config.num_simulations
        stats.root_value = root.q_value

        # Get policy from visit counts
        temperature = self.get_temperature(move_number)
        policy = root.get_policy(self.num_actions, temperature)

        return policy, root, stats

    cdef int _simulate(self, CythonMCTSNode root, object root_state,
                       object evaluator, object stats):
        """Run a single MCTS simulation.

        Returns:
            Depth reached in this simulation
        """
        cdef CythonMCTSNode node = root
        cdef CythonMCTSNode child
        cdef int action, depth = 0
        cdef double value
        cdef list path = []

        state = root_state

        # Selection: traverse tree until we reach an unexpanded node
        while node.is_expanded() and not node.is_terminal():
            action, child = node.select_child(self.config.c_puct)
            path.append((node, action))
            node = child
            state = state.apply_action(action)
            depth += 1

        # Check for terminal state
        if state.is_terminal():
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
        self._backpropagate(path, node, value)

        return depth

    cdef void _backpropagate(self, list path, CythonMCTSNode leaf, double value):
        """Backpropagate value through the tree."""
        cdef CythonMCTSNode node
        cdef int action

        # Update leaf node
        leaf.update(value)

        # Backpropagate through path (value flips at each level)
        for node, action in reversed(path):
            value = -value
            node.update(value)

    def select_action(self, np.ndarray policy, double temperature=1.0):
        """Select an action from the policy distribution."""
        if temperature <= 0.01:
            return int(np.argmax(policy))
        else:
            return int(np.random.choice(len(policy), p=policy))
