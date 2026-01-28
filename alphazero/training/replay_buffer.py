"""Replay buffer for storing and sampling training data.

Implements a fixed-size circular buffer with uniform random sampling.
"""

import numpy as np
import threading
from typing import List, Optional, Tuple
from collections import deque

from .trajectory import TrajectoryState, Trajectory, TrajectoryBatch


class ReplayBuffer:
    """Fixed-size replay buffer with uniform random sampling.

    Thread-safe for concurrent writes from multiple actors.
    """

    def __init__(self, capacity: int = 1_000_000):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of positions to store
        """
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()

        # Statistics
        self._total_added = 0
        self._total_games = 0

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Add all states from a trajectory to the buffer.

        Args:
            trajectory: Game trajectory to add
        """
        with self._lock:
            for state in trajectory.states:
                self._buffer.append(state)
            self._total_added += len(trajectory.states)
            self._total_games += 1

    def add_state(self, state: TrajectoryState) -> None:
        """Add a single state to the buffer.

        Args:
            state: Training state to add
        """
        with self._lock:
            self._buffer.append(state)
            self._total_added += 1

    def sample(self, batch_size: int) -> TrajectoryBatch:
        """Sample a random batch of training data.

        Args:
            batch_size: Number of samples to return

        Returns:
            TrajectoryBatch with sampled data
        """
        with self._lock:
            if len(self._buffer) < batch_size:
                raise ValueError(
                    f"Not enough samples in buffer: {len(self._buffer)} < {batch_size}"
                )

            # Random sampling with replacement
            indices = np.random.randint(0, len(self._buffer), size=batch_size)
            states = [self._buffer[i] for i in indices]

        return TrajectoryBatch.from_states(states)

    def sample_numpy(
        self,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch and return as numpy arrays.

        Args:
            batch_size: Number of samples

        Returns:
            Tuple of (observations, legal_masks, policies, values)
        """
        batch = self.sample(batch_size)
        return (
            batch.observations,
            batch.legal_masks,
            batch.policies,
            batch.values
        )

    def __len__(self) -> int:
        """Current number of samples in buffer."""
        return len(self._buffer)

    @property
    def total_added(self) -> int:
        """Total number of samples ever added."""
        return self._total_added

    @property
    def total_games(self) -> int:
        """Total number of games added."""
        return self._total_games

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self._buffer) >= min_size

    def clear(self) -> None:
        """Clear all samples from buffer."""
        with self._lock:
            self._buffer.clear()

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            'size': len(self._buffer),
            'capacity': self.capacity,
            'total_added': self._total_added,
            'total_games': self._total_games,
            'utilization': len(self._buffer) / self.capacity,
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer with prioritized sampling (optional enhancement).

    Samples positions with higher TD-error more frequently.
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        alpha: float = 0.6,
        beta: float = 0.4
    ):
        """Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of positions
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self._priorities: deque = deque(maxlen=capacity)
        self._max_priority = 1.0

    def add_state(self, state: TrajectoryState, priority: float = None) -> None:
        """Add state with priority."""
        with self._lock:
            self._buffer.append(state)
            self._priorities.append(priority or self._max_priority)
            self._total_added += 1

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Add trajectory with default priorities."""
        with self._lock:
            for state in trajectory.states:
                self._buffer.append(state)
                self._priorities.append(self._max_priority)
            self._total_added += len(trajectory.states)
            self._total_games += 1

    def sample(self, batch_size: int) -> Tuple[TrajectoryBatch, np.ndarray, np.ndarray]:
        """Sample with priorities.

        Returns:
            Tuple of (batch, indices, importance_weights)
        """
        with self._lock:
            n = len(self._buffer)
            if n < batch_size:
                raise ValueError(f"Not enough samples: {n} < {batch_size}")

            # Compute sampling probabilities
            priorities = np.array(self._priorities, dtype=np.float64)
            probs = priorities ** self.alpha
            probs /= probs.sum()

            # Sample indices
            indices = np.random.choice(n, size=batch_size, p=probs, replace=False)
            states = [self._buffer[i] for i in indices]

            # Compute importance sampling weights
            weights = (n * probs[indices]) ** (-self.beta)
            weights /= weights.max()

        batch = TrajectoryBatch.from_states(states)
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled indices."""
        with self._lock:
            for idx, priority in zip(indices, priorities):
                self._priorities[idx] = priority
                self._max_priority = max(self._max_priority, priority)
