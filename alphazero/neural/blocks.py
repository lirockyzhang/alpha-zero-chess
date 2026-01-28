"""Neural network building blocks for AlphaZero.

Implements:
- ConvBlock: Convolution + BatchNorm + ReLU
- ResidualBlock: Two ConvBlocks with skip connection
"""

import torch
import torch.nn as nn
from typing import Optional


class ConvBlock(nn.Module):
    """Convolution block: Conv2d -> BatchNorm -> ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False
    ):
        """Initialize ConvBlock.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            padding: Padding size
            bias: Whether to use bias (typically False with BatchNorm)
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with skip connection.

    Architecture:
        x -> Conv -> BN -> ReLU -> Conv -> BN -> (+x) -> ReLU
    """

    def __init__(self, num_filters: int):
        """Initialize ResidualBlock.

        Args:
            num_filters: Number of filters (same for input and output)
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_filters, num_filters,
            kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(
            num_filters, num_filters,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class PolicyHead(nn.Module):
    """Policy head: outputs action probabilities.

    Architecture:
        x -> Conv(1x1, 2 filters) -> BN -> ReLU -> Flatten -> FC(4672)
    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int = 2,
        num_actions: int = 4672
    ):
        """Initialize PolicyHead.

        Args:
            in_channels: Number of input channels from residual tower
            num_filters: Number of filters in policy conv layer
            num_actions: Size of action space
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(num_filters * 8 * 8, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, 8, 8)

        Returns:
            Raw logits of shape (batch, num_actions)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ValueHead(nn.Module):
    """Value head: outputs position evaluation.

    Architecture:
        x -> Conv(1x1, 1 filter) -> BN -> ReLU -> Flatten -> FC(hidden) -> ReLU -> FC(1) -> Tanh
    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int = 1,
        hidden_size: int = 192
    ):
        """Initialize ValueHead.

        Args:
            in_channels: Number of input channels from residual tower
            num_filters: Number of filters in value conv layer
            hidden_size: Size of hidden fully-connected layer
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(num_filters * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, 8, 8)

        Returns:
            Value estimate of shape (batch, 1) in range [-1, 1]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x
