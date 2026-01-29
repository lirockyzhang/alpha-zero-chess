# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Cython-optimized MCTS node implementation.

This provides ~5-10x speedup over pure Python by:
- Using typed memoryviews for array operations
- cdef classes to avoid Python object overhead
- Inline PUCT calculation
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free

np.import_array()

# Import type definition from .pxd
from .node cimport FLOAT_t
ctypedef np.int32_t INT_t


cdef class CythonMCTSNode:
    """Cython-optimized MCTS node with PUCT selection.

    Stores statistics for a single state-action pair:
        - N(s,a): Visit count
        - W(s,a): Total value (sum of backpropagated values)
        - Q(s,a): Mean value = W(s,a) / N(s,a)
        - P(s,a): Prior probability from neural network

    Note: Attributes are declared in node.pxd for cimport support.
    """

    def __init__(self, double prior=1.0):
        """Initialize a node.

        Args:
            prior: Prior probability P(s,a) from neural network
        """
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self._is_expanded = False
        self._is_terminal = False
        self._terminal_value = 0.0
        self._children = None
        self._legal_actions = None

    @property
    def q_value(self):
        """Q(s,a) = W(s,a) / N(s,a): Mean action value."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    cpdef bint is_expanded(self):
        """Check if this node has been expanded with children."""
        return self._is_expanded

    cpdef bint is_terminal(self):
        """Check if this is a terminal node."""
        return self._is_terminal

    cpdef void set_terminal(self, double value):
        """Mark this node as terminal with given value."""
        self._is_terminal = True
        self._terminal_value = value

    cpdef double get_terminal_value(self):
        """Get terminal value (only valid if is_terminal)."""
        return self._terminal_value

    cpdef void expand(self, np.ndarray[FLOAT_t, ndim=1] priors,
                      np.ndarray[FLOAT_t, ndim=1] legal_mask):
        """Expand this node by creating child nodes for legal actions.

        Args:
            priors: Prior probabilities from neural network (num_actions,)
            legal_mask: Binary mask of legal actions (num_actions,)
        """
        cdef int num_actions = priors.shape[0]
        cdef int action
        cdef double prior_sum = 0.0
        cdef double p

        self._children = {}

        # Find legal actions and compute prior sum
        legal_indices = []
        for action in range(num_actions):
            if legal_mask[action] > 0:
                legal_indices.append(action)
                prior_sum += priors[action]

        self._legal_actions = np.array(legal_indices, dtype=np.int32)

        # Create child nodes with normalized priors
        if prior_sum > 0:
            for action in legal_indices:
                p = priors[action] / prior_sum
                self._children[action] = CythonMCTSNode(prior=p)
        else:
            # Uniform priors if sum is zero
            p = 1.0 / len(legal_indices) if legal_indices else 0.0
            for action in legal_indices:
                self._children[action] = CythonMCTSNode(prior=p)

        self._is_expanded = True

    cpdef tuple select_child(self, double c_puct):
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
        cdef int best_action = -1
        cdef double best_score = -1e9
        cdef double sqrt_parent = sqrt(<double>self.visit_count)
        cdef double q, u, score
        cdef int action
        cdef CythonMCTSNode child

        for action, child in self._children.items():
            # Q(s,a): exploitation term
            q = child.q_value

            # U(s,a): exploration term
            u = c_puct * child.prior * sqrt_parent / (1.0 + child.visit_count)

            # PUCT score
            score = q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return (best_action, best_child)

    cpdef dict get_children(self):
        """Get all child nodes."""
        return self._children if self._children is not None else {}

    cpdef object get_child(self, int action):
        """Get child node for a specific action."""
        if self._children is None:
            return None
        return self._children.get(action)

    cpdef void update(self, double value):
        """Update node statistics during backpropagation.

        Args:
            value: Value to backpropagate (from current player's perspective)
        """
        self.visit_count += 1
        self.value_sum += value

    cpdef np.ndarray get_visit_counts(self, int num_actions):
        """Get visit counts for all actions as an array.

        Args:
            num_actions: Total number of actions in action space

        Returns:
            Array of visit counts (num_actions,)
        """
        cdef np.ndarray[FLOAT_t, ndim=1] counts = np.zeros(num_actions, dtype=np.float32)
        cdef int action
        cdef CythonMCTSNode child

        if self._children is not None:
            for action, child in self._children.items():
                counts[action] = <float>child.visit_count

        return counts

    cpdef np.ndarray get_policy(self, int num_actions, double temperature=1.0):
        """Get policy distribution from visit counts.

        π(a) ∝ N(s,a)^(1/τ)

        Args:
            num_actions: Total number of actions
            temperature: Temperature parameter (lower = more greedy)

        Returns:
            Probability distribution over actions
        """
        cdef np.ndarray[FLOAT_t, ndim=1] counts = self.get_visit_counts(num_actions)
        cdef np.ndarray[np.float64_t, ndim=1] counts64
        cdef double total, exponent
        cdef int i

        if temperature <= 0.01:
            # Greedy selection
            policy = np.zeros(num_actions, dtype=np.float32)
            if np.sum(counts) > 0:
                policy[np.argmax(counts)] = 1.0
            return policy

        # Apply temperature with numerical stability
        counts64 = counts.astype(np.float64)
        exponent = min(1.0 / temperature, 10.0)
        counts64 = np.power(counts64, exponent)
        total = np.sum(counts64)

        if total > 0:
            return (counts64 / total).astype(np.float32)
        return counts64.astype(np.float32)

    def __repr__(self):
        return (
            f"CythonMCTSNode(N={self.visit_count}, W={self.value_sum:.3f}, "
            f"Q={self.q_value:.3f}, P={self.prior:.3f})"
        )
