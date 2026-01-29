"""Tests for batched inference components.

This module tests the batched GPU inference system including:
- InferenceServer: Centralized GPU inference server
- BatchedEvaluator: Client-side evaluator that sends requests to server
- BatchedActor: Actor that uses batched inference
- Integration: End-to-end batched inference pipeline
"""

import pytest
import numpy as np
import torch
import time
from multiprocessing import Queue
from queue import Empty

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphazero import AlphaZeroConfig, MCTSConfig
from alphazero.neural.network import AlphaZeroNetwork
from alphazero.chess_env import GameState
from alphazero.selfplay.inference_server import (
    InferenceServer,
    InferenceRequest,
    InferenceResponse,
    BatchedEvaluator,
)
from alphazero.selfplay.batched_actor import BatchedActor
from alphazero.mcts import create_mcts, MCTSBackend


class TestInferenceRequest:
    """Tests for InferenceRequest dataclass."""

    def test_creation(self):
        """Test creating an inference request."""
        obs = np.zeros((119, 8, 8), dtype=np.float32)
        mask = np.ones(4672, dtype=np.float32)

        request = InferenceRequest(
            request_id=1,
            actor_id=0,
            observation=obs,
            legal_mask=mask
        )

        assert request.request_id == 1
        assert request.actor_id == 0
        assert request.observation.shape == (119, 8, 8)
        assert request.legal_mask.shape == (4672,)


class TestInferenceResponse:
    """Tests for InferenceResponse dataclass."""

    def test_creation(self):
        """Test creating an inference response."""
        policy = np.ones(4672, dtype=np.float32) / 4672

        response = InferenceResponse(
            request_id=1,
            policy=policy,
            value=0.5
        )

        assert response.request_id == 1
        assert response.policy.shape == (4672,)
        assert response.value == 0.5


class TestBatchedEvaluator:
    """Tests for BatchedEvaluator."""

    def test_creation(self):
        """Test creating a batched evaluator."""
        request_queue = Queue()
        response_queue = Queue()

        evaluator = BatchedEvaluator(
            actor_id=0,
            request_queue=request_queue,
            response_queue=response_queue,
            timeout=5.0
        )

        assert evaluator.actor_id == 0
        assert evaluator.request_counter == 0

    def test_evaluate_with_mock_server(self):
        """Test evaluation with a mock server response."""
        request_queue = Queue()
        response_queue = Queue()

        evaluator = BatchedEvaluator(
            actor_id=0,
            request_queue=request_queue,
            response_queue=response_queue,
            timeout=1.0
        )

        # Create mock observation and mask
        obs = np.zeros((119, 8, 8), dtype=np.float32)
        mask = np.ones(4672, dtype=np.float32)

        # Start evaluation in a separate thread
        import threading
        result = [None]
        error = [None]

        def evaluate():
            try:
                result[0] = evaluator.evaluate(obs, mask)
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=evaluate)
        thread.start()

        # Simulate server response
        time.sleep(0.1)
        request = request_queue.get(timeout=1.0)
        assert request.actor_id == 0
        assert request.request_id == 0

        # Send mock response
        mock_policy = np.ones(4672, dtype=np.float32) / 4672
        response = InferenceResponse(
            request_id=request.request_id,
            policy=mock_policy,
            value=0.5
        )
        response_queue.put(response)

        thread.join(timeout=2.0)

        assert error[0] is None
        assert result[0] is not None
        policy, value = result[0]
        assert policy.shape == (4672,)
        assert np.isclose(policy.sum(), 1.0)
        assert value == 0.5

    def test_evaluate_timeout(self):
        """Test that evaluation times out if no response."""
        request_queue = Queue()
        response_queue = Queue()

        evaluator = BatchedEvaluator(
            actor_id=0,
            request_queue=request_queue,
            response_queue=response_queue,
            timeout=0.5
        )

        obs = np.zeros((119, 8, 8), dtype=np.float32)
        mask = np.ones(4672, dtype=np.float32)

        with pytest.raises(TimeoutError):
            evaluator.evaluate(obs, mask)


class TestInferenceServer:
    """Tests for InferenceServer."""

    @pytest.fixture
    def small_network(self):
        """Create a small network for testing."""
        return AlphaZeroNetwork(num_filters=32, num_blocks=2)

    def test_server_creation(self, small_network):
        """Test creating an inference server."""
        request_queue = Queue()
        response_queues = {0: Queue(), 1: Queue()}

        server = InferenceServer(
            request_queue=request_queue,
            response_queues=response_queues,
            network_class=AlphaZeroNetwork,
            network_kwargs={'num_filters': 32, 'num_blocks': 2},
            device='cpu',
            batch_size=4,
            batch_timeout=0.01
        )

        assert server.batch_size == 4
        assert server.device == 'cpu'

    def test_server_single_request(self, small_network):
        """Test server handling a single request."""
        request_queue = Queue()
        response_queues = {0: Queue()}

        server = InferenceServer(
            request_queue=request_queue,
            response_queues=response_queues,
            network_class=AlphaZeroNetwork,
            network_kwargs={'num_filters': 32, 'num_blocks': 2},
            initial_weights=small_network.state_dict(),
            device='cpu',
            batch_size=4,
            batch_timeout=0.01
        )

        # Start server
        server.start()
        time.sleep(0.5)  # Give server time to initialize

        try:
            # Create and send request
            state = GameState()
            request = InferenceRequest(
                request_id=0,
                actor_id=0,
                observation=state.get_observation(),
                legal_mask=state.get_legal_actions()
            )
            request_queue.put(request)

            # Get response
            response = response_queues[0].get(timeout=2.0)

            assert response.request_id == 0
            assert response.policy.shape == (4672,)
            assert np.isclose(response.policy.sum(), 1.0)
            assert -1.0 <= response.value <= 1.0

        finally:
            server.terminate()
            server.join(timeout=2.0)

    def test_server_batched_requests(self, small_network):
        """Test server batching multiple requests."""
        request_queue = Queue()
        response_queues = {0: Queue(), 1: Queue(), 2: Queue()}

        server = InferenceServer(
            request_queue=request_queue,
            response_queues=response_queues,
            network_class=AlphaZeroNetwork,
            network_kwargs={'num_filters': 32, 'num_blocks': 2},
            initial_weights=small_network.state_dict(),
            device='cpu',
            batch_size=8,
            batch_timeout=0.05  # Longer timeout to collect batch
        )

        server.start()
        time.sleep(0.5)

        try:
            # Send multiple requests
            state = GameState()
            for actor_id in range(3):
                request = InferenceRequest(
                    request_id=actor_id,
                    actor_id=actor_id,
                    observation=state.get_observation(),
                    legal_mask=state.get_legal_actions()
                )
                request_queue.put(request)

            # Collect responses
            responses = {}
            for actor_id in range(3):
                response = response_queues[actor_id].get(timeout=2.0)
                responses[response.request_id] = response

            # Verify all responses received
            assert len(responses) == 3
            for actor_id in range(3):
                assert actor_id in responses
                assert responses[actor_id].policy.shape == (4672,)

        finally:
            server.terminate()
            server.join(timeout=2.0)

    def test_server_weight_update(self, small_network):
        """Test updating server weights."""
        request_queue = Queue()
        response_queues = {0: Queue()}
        weight_queue = Queue()

        server = InferenceServer(
            request_queue=request_queue,
            response_queues=response_queues,
            network_class=AlphaZeroNetwork,
            network_kwargs={'num_filters': 32, 'num_blocks': 2},
            initial_weights=small_network.state_dict(),
            device='cpu',
            batch_size=4,
            batch_timeout=0.01,
            weight_queue=weight_queue
        )

        server.start()
        time.sleep(0.5)

        try:
            # Send initial request
            state = GameState()
            request1 = InferenceRequest(
                request_id=0,
                actor_id=0,
                observation=state.get_observation(),
                legal_mask=state.get_legal_actions()
            )
            request_queue.put(request1)
            response1 = response_queues[0].get(timeout=2.0)

            # Update weights
            new_network = AlphaZeroNetwork(num_filters=32, num_blocks=2)
            weight_queue.put(new_network.state_dict())
            time.sleep(0.2)  # Give server time to update

            # Send another request
            request2 = InferenceRequest(
                request_id=1,
                actor_id=0,
                observation=state.get_observation(),
                legal_mask=state.get_legal_actions()
            )
            request_queue.put(request2)
            response2 = response_queues[0].get(timeout=2.0)

            # Both should succeed (values may differ due to different weights)
            assert response1.policy.shape == (4672,)
            assert response2.policy.shape == (4672,)

        finally:
            server.terminate()
            server.join(timeout=2.0)


class TestBatchedActor:
    """Tests for BatchedActor (non-process version)."""

    @pytest.fixture
    def small_network(self):
        """Create a small network for testing."""
        return AlphaZeroNetwork(num_filters=32, num_blocks=2)

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = AlphaZeroConfig()
        config.mcts.num_simulations = 10  # Small for testing
        config.selfplay.max_moves = 20
        return config

    def test_actor_creation(self, config):
        """Test creating a batched actor."""
        request_queue = Queue()
        response_queue = Queue()

        evaluator = BatchedEvaluator(
            actor_id=0,
            request_queue=request_queue,
            response_queue=response_queue
        )

        actor = BatchedActor(
            actor_id=0,
            evaluator=evaluator,
            config=config
        )

        assert actor.actor_id == 0
        assert actor.evaluator is evaluator

    def test_actor_play_game_with_mock_server(self, config, small_network):
        """Test actor playing a game with mock server."""
        request_queue = Queue()
        response_queues = {0: Queue()}

        # Start inference server
        server = InferenceServer(
            request_queue=request_queue,
            response_queues=response_queues,
            network_class=AlphaZeroNetwork,
            network_kwargs={'num_filters': 32, 'num_blocks': 2},
            initial_weights=small_network.state_dict(),
            device='cpu',
            batch_size=4,
            batch_timeout=0.01
        )
        server.start()
        time.sleep(0.5)

        try:
            # Create actor
            evaluator = BatchedEvaluator(
                actor_id=0,
                request_queue=request_queue,
                response_queue=response_queues[0],
                timeout=5.0
            )

            actor = BatchedActor(
                actor_id=0,
                evaluator=evaluator,
                config=config
            )

            # Play a game
            trajectory = actor.play_game()

            # Verify trajectory
            assert trajectory is not None
            assert len(trajectory) > 0
            assert trajectory.result is not None
            assert isinstance(trajectory.result, float)
            assert trajectory.result in [-1.0, 0.0, 1.0]

            # Verify trajectory states
            for state in trajectory.states:
                assert state.observation.shape == (119, 8, 8)
                assert state.legal_mask.shape == (4672,)
                assert state.policy.shape == (4672,)
                assert np.isclose(state.policy.sum(), 1.0)
                assert state.player in [0, 1]
                assert isinstance(state.value, float)

        finally:
            server.terminate()
            server.join(timeout=2.0)


class TestBatchedInferenceIntegration:
    """Integration tests for the complete batched inference system."""

    @pytest.fixture
    def small_network(self):
        """Create a small network for testing."""
        return AlphaZeroNetwork(num_filters=32, num_blocks=2)

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = AlphaZeroConfig()
        config.mcts.num_simulations = 10
        config.selfplay.max_moves = 20
        return config

    def test_multiple_actors_single_server(self, config, small_network):
        """Test multiple actors using a single inference server."""
        num_actors = 3
        request_queue = Queue()
        response_queues = {i: Queue() for i in range(num_actors)}

        # Start server
        server = InferenceServer(
            request_queue=request_queue,
            response_queues=response_queues,
            network_class=AlphaZeroNetwork,
            network_kwargs={'num_filters': 32, 'num_blocks': 2},
            initial_weights=small_network.state_dict(),
            device='cpu',
            batch_size=8,
            batch_timeout=0.02
        )
        server.start()
        time.sleep(0.5)

        try:
            # Create actors
            actors = []
            for i in range(num_actors):
                evaluator = BatchedEvaluator(
                    actor_id=i,
                    request_queue=request_queue,
                    response_queue=response_queues[i],
                    timeout=10.0
                )
                actor = BatchedActor(
                    actor_id=i,
                    evaluator=evaluator,
                    config=config
                )
                actors.append(actor)

            # Play games in parallel (using threads to simulate)
            import threading
            trajectories = [None] * num_actors
            errors = [None] * num_actors

            def play_game(actor_idx):
                try:
                    trajectories[actor_idx] = actors[actor_idx].play_game()
                except Exception as e:
                    errors[actor_idx] = e

            threads = [threading.Thread(target=play_game, args=(i,)) for i in range(num_actors)]
            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join(timeout=30.0)

            # Verify all games completed
            for i in range(num_actors):
                assert errors[i] is None, f"Actor {i} error: {errors[i]}"
                assert trajectories[i] is not None
                assert len(trajectories[i]) > 0

        finally:
            server.terminate()
            server.join(timeout=2.0)

    def test_server_handles_missing_response_queue(self, config, small_network):
        """Test that server handles requests for non-existent actors gracefully."""
        request_queue = Queue()
        response_queues = {0: Queue()}  # Only queue for actor 0

        server = InferenceServer(
            request_queue=request_queue,
            response_queues=response_queues,
            network_class=AlphaZeroNetwork,
            network_kwargs={'num_filters': 32, 'num_blocks': 2},
            initial_weights=small_network.state_dict(),
            device='cpu',
            batch_size=4,
            batch_timeout=0.01
        )
        server.start()
        time.sleep(0.5)

        try:
            # Send request from non-existent actor
            state = GameState()
            request = InferenceRequest(
                request_id=0,
                actor_id=999,  # No response queue for this actor
                observation=state.get_observation(),
                legal_mask=state.get_legal_actions()
            )
            request_queue.put(request)

            # Server should log error but not crash
            time.sleep(0.5)
            assert server.is_alive()

        finally:
            server.terminate()
            server.join(timeout=2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
