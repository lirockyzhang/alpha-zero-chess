"""Tests for the MCTS module."""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphazero import MCTSConfig
from alphazero.chess_env import GameState
from alphazero.mcts import create_mcts, get_available_backends, MCTSBackend
from alphazero.mcts.python.node import MCTSNode
from alphazero.mcts.evaluator import RandomEvaluator


class TestMCTSNode:
    """Tests for MCTSNode."""

    def test_initial_state(self):
        """Test initial node state."""
        node = MCTSNode(prior=0.5)
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.prior == 0.5
        assert node.q_value == 0.0
        assert not node.is_expanded()

    def test_update(self):
        """Test node update."""
        node = MCTSNode()
        node.update(0.5)
        assert node.visit_count == 1
        assert node.value_sum == 0.5
        assert node.q_value == 0.5

        node.update(-0.5)
        assert node.visit_count == 2
        assert node.value_sum == 0.0
        assert node.q_value == 0.0

    def test_expand(self):
        """Test node expansion."""
        node = MCTSNode()
        priors = np.array([0.3, 0.5, 0.2, 0.0])
        legal_mask = np.array([1, 1, 1, 0])

        node.expand(priors, legal_mask)

        assert node.is_expanded()
        children = node.get_children()
        assert len(children) == 3  # Only legal actions
        assert 3 not in children  # Illegal action not included

    def test_select_child(self):
        """Test PUCT child selection."""
        node = MCTSNode()
        priors = np.array([0.5, 0.3, 0.2])
        legal_mask = np.array([1, 1, 1])

        node.expand(priors, legal_mask)
        node.update(0.0)  # Need at least one visit

        action, child = node.select_child(c_puct=1.25)
        assert action in [0, 1, 2]
        assert child is not None

    def test_get_visit_counts(self):
        """Test getting visit counts."""
        node = MCTSNode()
        priors = np.array([0.5, 0.5])
        legal_mask = np.array([1, 1])

        node.expand(priors, legal_mask)

        # Update children
        node.get_children()[0].update(0.5)
        node.get_children()[0].update(0.5)
        node.get_children()[1].update(-0.5)

        counts = node.get_visit_counts(2)
        assert counts[0] == 2
        assert counts[1] == 1


class TestMCTS:
    """Tests for MCTS search."""

    def test_available_backends(self):
        """Test that Python backend is always available."""
        backends = get_available_backends()
        assert MCTSBackend.PYTHON in backends

    def test_create_mcts(self):
        """Test MCTS creation."""
        mcts = create_mcts(backend=MCTSBackend.PYTHON)
        assert mcts is not None

    def test_search_returns_policy(self):
        """Test that search returns valid policy."""
        config = MCTSConfig(num_simulations=50)
        mcts = create_mcts(config=config)
        evaluator = RandomEvaluator()
        state = GameState()

        policy, root, stats = mcts.search(state, evaluator)

        assert policy.shape == (4672,)
        assert np.isclose(policy.sum(), 1.0)
        assert np.all(policy >= 0)

    def test_search_respects_legal_moves(self):
        """Test that policy only has mass on legal moves."""
        config = MCTSConfig(num_simulations=50)
        mcts = create_mcts(config=config)
        evaluator = RandomEvaluator()
        state = GameState()

        policy, _, _ = mcts.search(state, evaluator)
        legal_mask = state.get_legal_actions()

        # Policy should be zero for illegal moves
        illegal_mass = np.sum(policy * (1 - legal_mask))
        assert illegal_mass < 1e-6

    def test_search_stats(self):
        """Test search statistics."""
        config = MCTSConfig(num_simulations=100)
        mcts = create_mcts(config=config)
        evaluator = RandomEvaluator()
        state = GameState()

        _, root, stats = mcts.search(state, evaluator)

        assert stats.num_simulations == 100
        assert stats.max_depth > 0
        assert stats.nodes_created > 0

    def test_temperature(self):
        """Test temperature-based action selection."""
        config = MCTSConfig(num_simulations=100, temperature_threshold=30)
        mcts = create_mcts(config=config)

        # Before threshold: temperature = 1.0
        temp = mcts.get_temperature(move_number=10)
        assert temp == 1.0

        # After threshold: temperature -> 0
        temp = mcts.get_temperature(move_number=50)
        assert temp < 0.1

    def test_dirichlet_noise(self):
        """Test Dirichlet noise addition."""
        config = MCTSConfig(dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
        mcts = create_mcts(config=config)

        priors = np.array([0.5, 0.3, 0.2, 0.0])
        legal_mask = np.array([1, 1, 1, 0])

        noisy = mcts.add_dirichlet_noise(priors, legal_mask)

        # Should still sum to ~1 for legal moves
        assert np.isclose(noisy[legal_mask > 0].sum(), 1.0, atol=0.01)

        # Should be different from original
        assert not np.allclose(noisy, priors)


class TestRandomEvaluator:
    """Tests for RandomEvaluator."""

    def test_uniform_policy(self):
        """Test that policy is uniform over legal moves."""
        evaluator = RandomEvaluator()
        observation = np.zeros((119, 8, 8))
        legal_mask = np.array([1, 1, 0, 1, 0])

        policy, value = evaluator.evaluate(observation, legal_mask)

        assert np.isclose(policy.sum(), 1.0)
        assert np.isclose(policy[0], 1/3)
        assert np.isclose(policy[2], 0.0)
        assert value == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
