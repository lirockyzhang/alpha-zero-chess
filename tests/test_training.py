"""Tests for the training module."""

import pytest
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphazero.training import (
    TrajectoryState,
    Trajectory,
    TrajectoryBatch,
    ReplayBuffer,
)


class TestTrajectoryState:
    """Tests for TrajectoryState."""

    def test_creation(self):
        """Test creating a trajectory state."""
        state = TrajectoryState(
            observation=np.zeros((119, 8, 8)),
            legal_mask=np.ones(4672),
            policy=np.ones(4672) / 4672,
            value=0.5,
            action=100,
            player=0
        )
        assert state.action == 100
        assert state.player == 0
        assert state.value == 0.5


class TestTrajectory:
    """Tests for Trajectory."""

    def test_add_state(self):
        """Test adding states to trajectory."""
        traj = Trajectory()
        traj.add_state(
            observation=np.zeros((119, 8, 8)),
            legal_mask=np.ones(4672),
            policy=np.ones(4672) / 4672,
            action=0,
            player=0
        )
        assert len(traj) == 1

    def test_set_result(self):
        """Test setting game result."""
        traj = Trajectory()

        # Add white move
        traj.add_state(
            observation=np.zeros((119, 8, 8)),
            legal_mask=np.ones(4672),
            policy=np.ones(4672) / 4672,
            action=0,
            player=0  # White
        )

        # Add black move
        traj.add_state(
            observation=np.zeros((119, 8, 8)),
            legal_mask=np.ones(4672),
            policy=np.ones(4672) / 4672,
            action=1,
            player=1  # Black
        )

        # White wins
        traj.set_result(1.0)

        assert traj.result == 1.0
        assert traj.states[0].value == 1.0   # White's perspective
        assert traj.states[1].value == -1.0  # Black's perspective


class TestTrajectoryBatch:
    """Tests for TrajectoryBatch."""

    def test_from_states(self):
        """Test creating batch from states."""
        states = [
            TrajectoryState(
                observation=np.zeros((119, 8, 8)),
                legal_mask=np.ones(4672),
                policy=np.ones(4672) / 4672,
                value=0.5,
                action=i,
                player=i % 2
            )
            for i in range(10)
        ]

        batch = TrajectoryBatch.from_states(states)

        assert batch.observations.shape == (10, 119, 8, 8)
        assert batch.legal_masks.shape == (10, 4672)
        assert batch.policies.shape == (10, 4672)
        assert batch.values.shape == (10,)


class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    def test_add_and_sample(self):
        """Test adding and sampling from buffer."""
        buffer = ReplayBuffer(capacity=100)

        # Add trajectory
        traj = Trajectory()
        for i in range(10):
            traj.add_state(
                observation=np.random.randn(119, 8, 8).astype(np.float32),
                legal_mask=np.ones(4672, dtype=np.float32),
                policy=np.ones(4672, dtype=np.float32) / 4672,
                action=i,
                player=i % 2
            )
        traj.set_result(1.0)

        buffer.add_trajectory(traj)

        assert len(buffer) == 10
        assert buffer.total_games == 1

        # Sample
        batch = buffer.sample(5)
        assert len(batch) == 5

    def test_capacity_limit(self):
        """Test that buffer respects capacity."""
        buffer = ReplayBuffer(capacity=10)

        for i in range(20):
            state = TrajectoryState(
                observation=np.zeros((119, 8, 8)),
                legal_mask=np.ones(4672),
                policy=np.ones(4672) / 4672,
                value=0.0,
                action=i,
                player=0
            )
            buffer.add_state(state)

        assert len(buffer) == 10  # Capped at capacity

    def test_sample_numpy(self):
        """Test sampling as numpy arrays."""
        buffer = ReplayBuffer(capacity=100)

        for i in range(20):
            state = TrajectoryState(
                observation=np.random.randn(119, 8, 8).astype(np.float32),
                legal_mask=np.ones(4672, dtype=np.float32),
                policy=np.ones(4672, dtype=np.float32) / 4672,
                value=float(i % 2),
                action=i,
                player=0
            )
            buffer.add_state(state)

        obs, masks, policies, values = buffer.sample_numpy(10)

        assert obs.shape == (10, 119, 8, 8)
        assert masks.shape == (10, 4672)
        assert policies.shape == (10, 4672)
        assert values.shape == (10,)

    def test_is_ready(self):
        """Test is_ready check."""
        buffer = ReplayBuffer(capacity=100)
        assert not buffer.is_ready(10)

        for i in range(10):
            state = TrajectoryState(
                observation=np.zeros((119, 8, 8)),
                legal_mask=np.ones(4672),
                policy=np.ones(4672) / 4672,
                value=0.0,
                action=i,
                player=0
            )
            buffer.add_state(state)

        assert buffer.is_ready(10)
        assert not buffer.is_ready(20)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
