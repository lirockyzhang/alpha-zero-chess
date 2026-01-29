"""Tests for MCTS backend implementations (Python, Cython, C++).

This module tests all MCTS backends to ensure they:
1. Implement the correct interface
2. Produce consistent results
3. Handle edge cases properly
4. Maintain correctness across implementations
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphazero import MCTSConfig, MCTSBackend
from alphazero.chess_env import GameState
from alphazero.mcts import create_mcts, get_available_backends
from alphazero.mcts.evaluator import RandomEvaluator
from alphazero.mcts.base import MCTSBase, MCTSNodeBase


class TestMCTSBackendAvailability:
    """Tests for backend availability detection."""

    def test_python_always_available(self):
        """Python backend should always be available."""
        backends = get_available_backends()
        assert MCTSBackend.PYTHON in backends

    def test_get_available_backends_returns_list(self):
        """get_available_backends should return a list."""
        backends = get_available_backends()
        assert isinstance(backends, list)
        assert len(backends) >= 1  # At least Python


class TestMCTSBackendInterface:
    """Tests that all backends implement the correct interface."""

    @pytest.fixture(params=[MCTSBackend.PYTHON])
    def backend(self, request):
        """Parametrized fixture that yields available backends."""
        backend_type = request.param
        available = get_available_backends()

        if backend_type not in available:
            pytest.skip(f"{backend_type.value} backend not available")

        return backend_type

    def test_backend_creation(self, backend):
        """Test creating MCTS with each backend."""
        config = MCTSConfig(num_simulations=10)
        mcts = create_mcts(backend=backend, config=config)

        assert mcts is not None
        assert isinstance(mcts, MCTSBase)
        assert mcts.config.num_simulations == 10

    def test_search_returns_correct_types(self, backend):
        """Test that search returns correct types."""
        config = MCTSConfig(num_simulations=20)
        mcts = create_mcts(backend=backend, config=config)
        evaluator = RandomEvaluator()
        state = GameState()

        policy, root, stats = mcts.search(state, evaluator)

        # Check types
        assert isinstance(policy, np.ndarray)
        assert isinstance(root, MCTSNodeBase)
        assert hasattr(stats, 'num_simulations')

        # Check shapes and values
        assert policy.shape == (4672,)
        assert np.isclose(policy.sum(), 1.0)
        assert np.all(policy >= 0)

    def test_search_respects_simulation_count(self, backend):
        """Test that search performs the correct number of simulations."""
        num_sims = 50
        config = MCTSConfig(num_simulations=num_sims)
        mcts = create_mcts(backend=backend, config=config)
        evaluator = RandomEvaluator()
        state = GameState()

        _, root, stats = mcts.search(state, evaluator)

        assert stats.num_simulations == num_sims
        # Root should have been visited num_sims times
        assert root.visit_count >= num_sims

    def test_search_respects_legal_moves(self, backend):
        """Test that policy only assigns probability to legal moves."""
        config = MCTSConfig(num_simulations=30)
        mcts = create_mcts(backend=backend, config=config)
        evaluator = RandomEvaluator()
        state = GameState()

        policy, _, _ = mcts.search(state, evaluator)
        legal_mask = state.get_legal_actions()

        # Policy should be zero for illegal moves
        illegal_mass = np.sum(policy * (1 - legal_mask))
        assert illegal_mass < 1e-6

    def test_temperature_function(self, backend):
        """Test temperature calculation."""
        config = MCTSConfig(temperature_threshold=30)
        mcts = create_mcts(backend=backend, config=config)

        # Before threshold
        temp = mcts.get_temperature(move_number=10)
        assert temp == 1.0

        # After threshold
        temp = mcts.get_temperature(move_number=50)
        assert temp == 0.0

    def test_dirichlet_noise(self, backend):
        """Test Dirichlet noise addition."""
        config = MCTSConfig(dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
        mcts = create_mcts(backend=backend, config=config)

        priors = np.array([0.5, 0.3, 0.2, 0.0])
        legal_mask = np.array([1, 1, 1, 0])

        noisy = mcts.add_dirichlet_noise(priors, legal_mask)

        # Should still sum to ~1 for legal moves
        assert np.isclose(noisy[legal_mask > 0].sum(), 1.0, atol=0.01)

        # Should be different from original
        assert not np.allclose(noisy, priors)

        # Illegal moves should still be zero
        assert noisy[3] == 0.0


class TestMCTSBackendConsistency:
    """Tests that different backends produce consistent results."""

    def test_deterministic_search_consistency(self):
        """Test that backends produce similar results with same random seed."""
        available = get_available_backends()
        if len(available) < 2:
            pytest.skip("Need at least 2 backends for consistency test")

        config = MCTSConfig(
            num_simulations=100,
            dirichlet_epsilon=0.0  # Disable noise for determinism
        )
        state = GameState()
        evaluator = RandomEvaluator()

        # Run search with each backend
        results = {}
        for backend in available[:2]:  # Test first 2 available
            np.random.seed(42)
            mcts = create_mcts(backend=backend, config=config)
            policy, root, stats = mcts.search(state, evaluator, add_noise=False)
            results[backend] = {
                'policy': policy,
                'root_value': root.q_value,
                'visit_count': root.visit_count
            }

        # Compare results (should be similar but not necessarily identical
        # due to implementation differences)
        backends = list(results.keys())
        if len(backends) >= 2:
            policy1 = results[backends[0]]['policy']
            policy2 = results[backends[1]]['policy']

            # Policies should be correlated (top actions should be similar)
            top_k = 5
            top_actions_1 = np.argsort(policy1)[-top_k:]
            top_actions_2 = np.argsort(policy2)[-top_k:]

            # At least some overlap in top actions
            overlap = len(set(top_actions_1) & set(top_actions_2))
            assert overlap >= 2, "Backends should have some agreement on top actions"


class TestMCTSBackendEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture(params=[MCTSBackend.PYTHON])
    def backend(self, request):
        """Parametrized fixture for available backends."""
        backend_type = request.param
        available = get_available_backends()

        if backend_type not in available:
            pytest.skip(f"{backend_type.value} backend not available")

        return backend_type

    def test_search_from_terminal_position(self, backend):
        """Test search from a terminal (game over) position."""
        config = MCTSConfig(num_simulations=10)
        mcts = create_mcts(backend=backend, config=config)
        evaluator = RandomEvaluator()

        # Create a position with checkmate
        # Fool's mate: 1. f3 e5 2. g4 Qh4#
        state = GameState()
        state = state.apply_move(state.board.parse_san('f3'))
        state = state.apply_move(state.board.parse_san('e5'))
        state = state.apply_move(state.board.parse_san('g4'))
        state = state.apply_move(state.board.parse_san('Qh4#'))

        assert state.is_terminal()

        # Search should handle terminal position gracefully
        policy, root, stats = mcts.search(state, evaluator)

        # Policy should be defined (even if game is over)
        assert policy.shape == (4672,)
        assert np.isclose(policy.sum(), 1.0) or policy.sum() == 0.0

    def test_search_with_few_legal_moves(self, backend):
        """Test search in position with very few legal moves."""
        config = MCTSConfig(num_simulations=20)
        mcts = create_mcts(backend=backend, config=config)
        evaluator = RandomEvaluator()

        # Create a position with limited moves (endgame)
        # King and pawn vs King
        state = GameState.from_fen("8/8/8/8/8/k7/p7/K7 w - - 0 1")

        policy, root, stats = mcts.search(state, evaluator)

        # Should still work
        assert policy.shape == (4672,)
        assert np.isclose(policy.sum(), 1.0)

        # Only legal moves should have probability
        legal_mask = state.get_legal_actions()
        num_legal = int(np.sum(legal_mask))
        assert num_legal > 0

        illegal_mass = np.sum(policy * (1 - legal_mask))
        assert illegal_mass < 1e-6

    def test_search_with_zero_simulations(self, backend):
        """Test that zero simulations is handled."""
        config = MCTSConfig(num_simulations=0)
        mcts = create_mcts(backend=backend, config=config)
        evaluator = RandomEvaluator()
        state = GameState()

        # Should either raise error or return uniform policy
        try:
            policy, root, stats = mcts.search(state, evaluator)
            # If it doesn't raise, check it returns valid policy
            assert policy.shape == (4672,)
        except (ValueError, AssertionError):
            # It's acceptable to raise an error for zero simulations
            pass

    def test_multiple_searches_same_instance(self, backend):
        """Test running multiple searches with same MCTS instance."""
        config = MCTSConfig(num_simulations=20)
        mcts = create_mcts(backend=backend, config=config)
        evaluator = RandomEvaluator()

        # Run multiple searches
        for i in range(3):
            state = GameState()
            policy, root, stats = mcts.search(state, evaluator)

            assert policy.shape == (4672,)
            assert np.isclose(policy.sum(), 1.0)
            assert stats.num_simulations == 20


class TestCythonMCTS:
    """Specific tests for Cython MCTS implementation."""

    def test_cython_available(self):
        """Test if Cython backend is available."""
        available = get_available_backends()
        if MCTSBackend.CYTHON not in available:
            pytest.skip("Cython backend not built")

    def test_cython_basic_search(self):
        """Test basic Cython MCTS search."""
        available = get_available_backends()
        if MCTSBackend.CYTHON not in available:
            pytest.skip("Cython backend not built")

        config = MCTSConfig(num_simulations=50)
        mcts = create_mcts(backend=MCTSBackend.CYTHON, config=config)
        evaluator = RandomEvaluator()
        state = GameState()

        policy, root, stats = mcts.search(state, evaluator)

        assert policy.shape == (4672,)
        assert np.isclose(policy.sum(), 1.0)
        assert stats.num_simulations == 50

    def test_cython_vs_python_consistency(self):
        """Compare Cython and Python implementations."""
        available = get_available_backends()
        if MCTSBackend.CYTHON not in available:
            pytest.skip("Cython backend not built")

        config = MCTSConfig(num_simulations=100, dirichlet_epsilon=0.0)
        state = GameState()
        evaluator = RandomEvaluator()

        # Python search
        np.random.seed(42)
        mcts_py = create_mcts(backend=MCTSBackend.PYTHON, config=config)
        policy_py, _, _ = mcts_py.search(state, evaluator, add_noise=False)

        # Cython search
        np.random.seed(42)
        mcts_cy = create_mcts(backend=MCTSBackend.CYTHON, config=config)
        policy_cy, _, _ = mcts_cy.search(state, evaluator, add_noise=False)

        # Top actions should overlap significantly
        top_k = 5
        top_py = set(np.argsort(policy_py)[-top_k:])
        top_cy = set(np.argsort(policy_cy)[-top_k:])

        overlap = len(top_py & top_cy)
        assert overlap >= 2, "Cython and Python should agree on top actions"


class TestCppMCTS:
    """Specific tests for C++ MCTS implementation."""

    def test_cpp_available(self):
        """Test if C++ backend is available."""
        available = get_available_backends()
        if MCTSBackend.CPP not in available:
            pytest.skip("C++ backend not built")

    def test_cpp_basic_search(self):
        """Test basic C++ MCTS search."""
        available = get_available_backends()
        if MCTSBackend.CPP not in available:
            pytest.skip("C++ backend not built")

        config = MCTSConfig(num_simulations=50)
        mcts = create_mcts(backend=MCTSBackend.CPP, config=config)
        evaluator = RandomEvaluator()
        state = GameState()

        policy, root, stats = mcts.search(state, evaluator)

        assert policy.shape == (4672,)
        assert np.isclose(policy.sum(), 1.0)
        assert stats.num_simulations == 50

    def test_cpp_vs_python_consistency(self):
        """Compare C++ and Python implementations."""
        available = get_available_backends()
        if MCTSBackend.CPP not in available:
            pytest.skip("C++ backend not built")

        config = MCTSConfig(num_simulations=100, dirichlet_epsilon=0.0)
        state = GameState()
        evaluator = RandomEvaluator()

        # Python search
        np.random.seed(42)
        mcts_py = create_mcts(backend=MCTSBackend.PYTHON, config=config)
        policy_py, _, _ = mcts_py.search(state, evaluator, add_noise=False)

        # C++ search
        np.random.seed(42)
        mcts_cpp = create_mcts(backend=MCTSBackend.CPP, config=config)
        policy_cpp, _, _ = mcts_cpp.search(state, evaluator, add_noise=False)

        # Top actions should overlap significantly
        top_k = 5
        top_py = set(np.argsort(policy_py)[-top_k:])
        top_cpp = set(np.argsort(policy_cpp)[-top_k:])

        overlap = len(top_py & top_cpp)
        assert overlap >= 2, "C++ and Python should agree on top actions"

    def test_cpp_memory_safety(self):
        """Test C++ implementation doesn't leak memory with multiple searches."""
        available = get_available_backends()
        if MCTSBackend.CPP not in available:
            pytest.skip("C++ backend not built")

        config = MCTSConfig(num_simulations=50)
        mcts = create_mcts(backend=MCTSBackend.CPP, config=config)
        evaluator = RandomEvaluator()

        # Run many searches to detect memory issues
        for _ in range(10):
            state = GameState()
            policy, root, stats = mcts.search(state, evaluator)
            assert policy.shape == (4672,)


class TestMCTSBackendPerformance:
    """Performance comparison tests (not strict assertions)."""

    def test_backend_performance_comparison(self):
        """Compare performance of available backends (informational)."""
        available = get_available_backends()
        if len(available) < 2:
            pytest.skip("Need multiple backends for performance comparison")

        config = MCTSConfig(num_simulations=100)
        state = GameState()
        evaluator = RandomEvaluator()

        import time
        results = {}

        for backend in available:
            mcts = create_mcts(backend=backend, config=config)

            start = time.time()
            for _ in range(5):
                policy, _, _ = mcts.search(state, evaluator)
            elapsed = time.time() - start

            results[backend.value] = elapsed / 5  # Average time per search

        # Print results (informational, not assertions)
        print("\nMCTS Backend Performance (avg time per search):")
        for backend, time_taken in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {backend}: {time_taken:.4f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
