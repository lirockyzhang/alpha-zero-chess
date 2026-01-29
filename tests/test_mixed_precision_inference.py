"""Tests for mixed precision inference implementation."""

import pytest
import torch
import numpy as np

from alphazero.neural.network import AlphaZeroNetwork
from alphazero.mcts.evaluator import NetworkEvaluator
from alphazero.selfplay.inference_server import InferenceServer, InferenceRequest, BatchedEvaluator
from alphazero.config import TrainingConfig
from multiprocessing import Queue


class TestMixedPrecisionInference:
    """Test mixed precision inference functionality."""

    @pytest.fixture
    def network(self):
        """Create a small test network."""
        return AlphaZeroNetwork(
            num_filters=64,
            num_blocks=2
        )

    @pytest.fixture
    def observation(self):
        """Create a test observation."""
        return np.random.randn(119, 8, 8).astype(np.float32)

    @pytest.fixture
    def legal_mask(self):
        """Create a test legal mask."""
        mask = np.zeros(4672, dtype=np.float32)
        # Set some random moves as legal
        legal_indices = np.random.choice(4672, size=50, replace=False)
        mask[legal_indices] = 1.0
        return mask

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_network_evaluator_with_amp(self, network, observation, legal_mask):
        """Test NetworkEvaluator with mixed precision."""
        network = network.to("cuda")

        # Test with AMP enabled
        evaluator_amp = NetworkEvaluator(network, device="cuda", use_amp=True)
        policy_amp, value_amp = evaluator_amp.evaluate(observation, legal_mask)

        # Test with AMP disabled
        evaluator_fp32 = NetworkEvaluator(network, device="cuda", use_amp=False)
        policy_fp32, value_fp32 = evaluator_fp32.evaluate(observation, legal_mask)

        # Check outputs are valid
        assert policy_amp.shape == (4672,)
        assert np.isclose(policy_amp.sum(), 1.0, atol=1e-5)
        assert -1.0 <= value_amp <= 1.0

        assert policy_fp32.shape == (4672,)
        assert np.isclose(policy_fp32.sum(), 1.0, atol=1e-5)
        assert -1.0 <= value_fp32 <= 1.0

        # Outputs should be similar (not exact due to FP16 precision)
        assert np.allclose(policy_amp, policy_fp32, atol=1e-3)
        assert np.isclose(value_amp, value_fp32, atol=1e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_network_evaluator_batch_with_amp(self, network, observation, legal_mask):
        """Test NetworkEvaluator batch evaluation with mixed precision."""
        network = network.to("cuda")
        batch_size = 4

        observations = np.stack([observation] * batch_size)
        legal_masks = np.stack([legal_mask] * batch_size)

        # Test with AMP
        evaluator_amp = NetworkEvaluator(network, device="cuda", use_amp=True)
        policies_amp, values_amp = evaluator_amp.evaluate_batch(observations, legal_masks)

        # Test without AMP
        evaluator_fp32 = NetworkEvaluator(network, device="cuda", use_amp=False)
        policies_fp32, values_fp32 = evaluator_fp32.evaluate_batch(observations, legal_masks)

        # Check shapes
        assert policies_amp.shape == (batch_size, 4672)
        assert values_amp.shape == (batch_size,)

        # Check validity
        for i in range(batch_size):
            assert np.isclose(policies_amp[i].sum(), 1.0, atol=1e-5)
            assert -1.0 <= values_amp[i] <= 1.0

        # Compare with FP32
        assert np.allclose(policies_amp, policies_fp32, atol=1e-3)
        assert np.allclose(values_amp, values_fp32, atol=1e-2)

    def test_network_evaluator_cpu_amp_disabled(self, network, observation, legal_mask):
        """Test that AMP is automatically disabled on CPU."""
        network = network.to("cpu")

        # Even with use_amp=True, it should work on CPU (AMP disabled automatically)
        evaluator = NetworkEvaluator(network, device="cpu", use_amp=True)
        policy, value = evaluator.evaluate(observation, legal_mask)

        assert policy.shape == (4672,)
        assert np.isclose(policy.sum(), 1.0, atol=1e-5)
        assert -1.0 <= value <= 1.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_inference_server_with_amp(self, network):
        """Test InferenceServer with mixed precision."""
        # Create queues
        request_queue = Queue()
        response_queues = {0: Queue()}

        # Create server with AMP
        server = InferenceServer(
            request_queue=request_queue,
            response_queues=response_queues,
            network_class=AlphaZeroNetwork,
            network_kwargs={'num_filters': 64, 'num_blocks': 2},
            device="cuda",
            use_amp=True,
            batch_size=4,
            batch_timeout=0.01
        )

        # Verify server configuration
        assert server.use_amp is True
        assert server.device == "cuda"

    def test_config_mixed_precision_flags(self):
        """Test TrainingConfig mixed precision flags."""
        # Default config should have AMP enabled
        config = TrainingConfig()
        assert config.use_amp is True
        assert config.use_amp_inference is True

        # Test disabling AMP
        config_no_amp = TrainingConfig(use_amp=False, use_amp_inference=False)
        assert config_no_amp.use_amp is False
        assert config_no_amp.use_amp_inference is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_numerical_stability(self, network, observation, legal_mask):
        """Test that mixed precision maintains numerical stability."""
        network = network.to("cuda")
        evaluator = NetworkEvaluator(network, device="cuda", use_amp=True)

        # Run multiple evaluations
        policies = []
        values = []
        for _ in range(10):
            policy, value = evaluator.evaluate(observation, legal_mask)
            policies.append(policy)
            values.append(value)

        # Check consistency (should be identical for same input)
        for i in range(1, len(policies)):
            assert np.allclose(policies[0], policies[i], atol=1e-6)
            assert np.isclose(values[0], values[i], atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_policy_validity(self, network, observation, legal_mask):
        """Test that mixed precision produces valid probability distributions."""
        network = network.to("cuda")
        evaluator = NetworkEvaluator(network, device="cuda", use_amp=True)

        policy, value = evaluator.evaluate(observation, legal_mask)

        # Check policy is a valid probability distribution
        assert np.all(policy >= 0), "Policy contains negative values"
        assert np.isclose(policy.sum(), 1.0, atol=1e-5), "Policy doesn't sum to 1"

        # Check only legal moves have non-zero probability
        illegal_moves = legal_mask == 0
        assert np.allclose(policy[illegal_moves], 0.0, atol=1e-6), "Illegal moves have non-zero probability"

        # Check value is in valid range
        assert -1.0 <= value <= 1.0, f"Value {value} out of range [-1, 1]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
