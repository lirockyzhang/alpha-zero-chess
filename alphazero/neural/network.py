"""AlphaZero neural network architecture.

Implements the dual-headed network from the AlphaZero paper:
- Shared residual tower
- Policy head (action probabilities)
- Value head (position evaluation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .blocks import ConvBlock, ResidualBlock, PolicyHead, ValueHead


class AlphaZeroNetwork(nn.Module):
    """AlphaZero neural network with policy and value heads.

    Note: Input tensors should be in NCHW format (batch, 122, 8, 8) but stored
    with channels_last memory layout for optimal Conv2d performance. The model
    should be converted to channels_last format during initialization.

    Architecture:
        Input (122, 8, 8) NCHW shape, NHWC memory
            |
        ConvBlock (3x3, num_filters)
            |
        ResidualBlock × num_blocks
            |
        +-------+-------+
        |               |
    PolicyHead      ValueHead
        |               |
    (4672,)          (1,)
    """

    def __init__(
        self,
        input_channels: int = 122,
        num_filters: int = 192,
        num_blocks: int = 15,
        num_actions: int = 4672,
        policy_filters: int = 2,
        value_filters: int = 1,
        value_hidden: int = 256
    ):
        """Initialize AlphaZeroNetwork.

        Args:
            input_channels: Number of input planes (122 for chess)
            num_filters: Number of filters in residual tower
            num_blocks: Number of residual blocks
            num_actions: Size of action space (4672 for chess)
            policy_filters: Filters in policy head conv layer (AlphaZero paper: 2)
            value_filters: Filters in value head conv layer (AlphaZero paper: 1)
            value_hidden: Hidden layer size in value head (AlphaZero paper: 256)
        """
        super().__init__()

        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.num_actions = num_actions

        # Initial convolution
        self.input_conv = ConvBlock(input_channels, num_filters)

        # Residual tower
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_blocks)]
        )

        # Output heads
        self.policy_head = PolicyHead(num_filters, policy_filters, num_actions)
        self.value_head = ValueHead(num_filters, value_filters, value_hidden)

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 122, 8, 8) in NCHW format
               Should be stored with channels_last memory layout
            legal_mask: Optional binary mask of shape (batch, 4672)
                       If provided, illegal actions are masked to -inf

        Returns:
            Tuple of:
                - policy_logits: Log probabilities of shape (batch, 4672)
                - value: Position evaluation of shape (batch, 1)
        """
        # Shared trunk
        x = self.input_conv(x)
        x = self.residual_tower(x)

        # Policy head
        policy_logits = self.policy_head(x)

        # Apply legal move mask
        if legal_mask is not None:
            # Set illegal moves to very negative value before softmax
            # Use -1e4 instead of -inf for FP16 numerical stability (FP16 max ~65k)
            policy_logits = policy_logits.masked_fill(
                legal_mask == 0,
                -1e4
            )

        # Value head
        value = self.value_head(x)

        return policy_logits, value

    def predict(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get policy probabilities and value for inference.

        Args:
            x: Input tensor of shape (batch, 122, 8, 8) in NCHW format
            legal_mask: Binary mask of shape (batch, 4672)

        Returns:
            Tuple of:
                - policy: Probability distribution of shape (batch, 4672)
                - value: Position evaluation of shape (batch,)
        """
        policy_logits, value = self.forward(x, legal_mask)
        policy = F.softmax(policy_logits, dim=-1)
        return policy, value.squeeze(-1)

    def get_policy_and_value(
        self,
        observation: torch.Tensor,
        legal_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Convenience method for single position evaluation.

        Args:
            observation: Single observation of shape (122, 8, 8) in NCHW format
            legal_mask: Legal action mask of shape (4672,)

        Returns:
            Tuple of:
                - policy: Probability distribution of shape (4672,)
                - value: Scalar position evaluation
        """
        # Add batch dimension
        x = observation.unsqueeze(0)
        mask = legal_mask.unsqueeze(0)

        with torch.no_grad():
            policy, value = self.predict(x, mask)

        return policy.squeeze(0), value.item()


def create_network(
    num_filters: int = 192,
    num_blocks: int = 15,
    device: str = "cuda"
) -> AlphaZeroNetwork:
    """Factory function to create an AlphaZero network.

    Args:
        num_filters: Number of filters in residual tower
        num_blocks: Number of residual blocks
        device: Device to place the network on

    Returns:
        Initialized AlphaZeroNetwork
    """
    network = AlphaZeroNetwork(
        num_filters=num_filters,
        num_blocks=num_blocks
    )
    return network.to(device)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
