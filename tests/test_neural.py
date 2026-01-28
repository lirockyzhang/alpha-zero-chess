"""Tests for the neural network module."""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphazero.neural import (
    AlphaZeroNetwork,
    ConvBlock,
    ResidualBlock,
    PolicyHead,
    ValueHead,
    AlphaZeroLoss,
    count_parameters,
)


class TestConvBlock:
    """Tests for ConvBlock."""

    def test_output_shape(self):
        """Test output shape."""
        block = ConvBlock(119, 192)
        x = torch.randn(2, 119, 8, 8)
        y = block(x)
        assert y.shape == (2, 192, 8, 8)

    def test_different_kernel(self):
        """Test with different kernel size."""
        block = ConvBlock(64, 64, kernel_size=1, padding=0)
        x = torch.randn(2, 64, 8, 8)
        y = block(x)
        assert y.shape == (2, 64, 8, 8)


class TestResidualBlock:
    """Tests for ResidualBlock."""

    def test_output_shape(self):
        """Test output shape matches input."""
        block = ResidualBlock(192)
        x = torch.randn(2, 192, 8, 8)
        y = block(x)
        assert y.shape == x.shape

    def test_skip_connection(self):
        """Test that skip connection works."""
        block = ResidualBlock(64)
        # With zero weights, output should equal input (after ReLU)
        x = torch.ones(1, 64, 8, 8)
        y = block(x)
        # Output should be non-zero due to skip connection
        assert y.sum() > 0


class TestPolicyHead:
    """Tests for PolicyHead."""

    def test_output_shape(self):
        """Test output shape."""
        head = PolicyHead(192, num_filters=2, num_actions=4672)
        x = torch.randn(2, 192, 8, 8)
        y = head(x)
        assert y.shape == (2, 4672)


class TestValueHead:
    """Tests for ValueHead."""

    def test_output_shape(self):
        """Test output shape."""
        head = ValueHead(192, num_filters=1, hidden_size=192)
        x = torch.randn(2, 192, 8, 8)
        y = head(x)
        assert y.shape == (2, 1)

    def test_output_range(self):
        """Test output is in [-1, 1] due to tanh."""
        head = ValueHead(192)
        x = torch.randn(10, 192, 8, 8)
        y = head(x)
        assert torch.all(y >= -1)
        assert torch.all(y <= 1)


class TestAlphaZeroNetwork:
    """Tests for the full network."""

    def test_output_shapes(self):
        """Test output shapes."""
        network = AlphaZeroNetwork(
            input_channels=119,
            num_filters=64,
            num_blocks=5,
            num_actions=4672
        )
        x = torch.randn(2, 119, 8, 8)
        mask = torch.ones(2, 4672)

        policy, value = network(x, mask)
        assert policy.shape == (2, 4672)
        assert value.shape == (2, 1)

    def test_legal_mask(self):
        """Test that illegal moves are masked."""
        network = AlphaZeroNetwork(num_filters=64, num_blocks=2)
        x = torch.randn(1, 119, 8, 8)

        # Mask all but first action
        mask = torch.zeros(1, 4672)
        mask[0, 0] = 1

        policy, _ = network(x, mask)

        # After softmax, only first action should have probability
        probs = torch.softmax(policy, dim=-1)
        assert probs[0, 0] > 0.99

    def test_predict_method(self):
        """Test predict method returns probabilities."""
        network = AlphaZeroNetwork(num_filters=64, num_blocks=2)
        network.eval()

        x = torch.randn(2, 119, 8, 8)
        mask = torch.ones(2, 4672)

        policy, value = network.predict(x, mask)

        # Policy should sum to 1
        assert torch.allclose(policy.sum(dim=-1), torch.ones(2), atol=1e-5)

        # Value should be 1D
        assert value.shape == (2,)

    def test_parameter_count(self):
        """Test parameter counting."""
        network = AlphaZeroNetwork(num_filters=64, num_blocks=5)
        params = count_parameters(network)
        assert params > 0


class TestAlphaZeroLoss:
    """Tests for the loss function."""

    def test_loss_computation(self):
        """Test loss computation."""
        loss_fn = AlphaZeroLoss()

        batch_size = 4
        num_actions = 4672

        policy_logits = torch.randn(batch_size, num_actions)
        target_policy = torch.softmax(torch.randn(batch_size, num_actions), dim=-1)
        value = torch.randn(batch_size, 1)
        target_value = torch.tensor([1.0, -1.0, 0.0, 1.0])
        legal_mask = torch.ones(batch_size, num_actions)

        loss, metrics = loss_fn(
            policy_logits, target_policy,
            value, target_value,
            legal_mask
        )

        assert loss.shape == ()
        assert 'loss' in metrics
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics

    def test_loss_decreases(self):
        """Test that loss decreases with matching targets."""
        loss_fn = AlphaZeroLoss()

        # Create matching policy
        target_policy = torch.zeros(1, 100)
        target_policy[0, 42] = 1.0

        # Logits that match target
        good_logits = torch.zeros(1, 100)
        good_logits[0, 42] = 10.0

        # Random logits
        bad_logits = torch.randn(1, 100)

        mask = torch.ones(1, 100)
        value = torch.zeros(1, 1)
        target_value = torch.zeros(1)

        good_loss, _ = loss_fn(good_logits, target_policy, value, target_value, mask)
        bad_loss, _ = loss_fn(bad_logits, target_policy, value, target_value, mask)

        assert good_loss < bad_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
