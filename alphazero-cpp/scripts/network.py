#!/usr/bin/env python3
"""
Neural Network Architecture for AlphaZero

This module contains the shared network architecture used by both training
and evaluation. Extracting it ensures checkpoint compatibility between
train.py and evaluation.py.

Architecture matches AlphaZero paper standard settings:
- Configurable input channels (default: 122 for extended encoding)
- 2 policy filters (paper standard)
- 256 value hidden units (paper standard)
- Residual tower with configurable depth/width
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Constants
# =============================================================================

# 122-channel encoding (8 history * 14 piece planes + 8 repetition + 2 color/castling)
INPUT_CHANNELS = 122
POLICY_SIZE = 4672


# =============================================================================
# Network Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """Convolutional block with batch norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: global channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation."""

    def __init__(self, num_filters: int, se_reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(num_filters, se_reduction)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + identity)


class PolicyHead(nn.Module):
    """Policy head: outputs action probabilities (AlphaZero paper: 2 filters)."""

    def __init__(self, in_channels: int, num_filters: int = 2, num_actions: int = POLICY_SIZE):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(num_filters * 8 * 8, num_actions)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class ValueHead(nn.Module):
    """Value head: outputs position evaluation (AlphaZero paper: 256 hidden).

    WDL mode (default): fc2 outputs 3 logits (win/draw/loss), no tanh.
    Legacy scalar mode (wdl=False): fc2 outputs 1 scalar, passed through tanh → [-1, 1].
    """

    def __init__(self, in_channels: int, num_filters: int = 1, hidden_size: int = 256, wdl: bool = True):
        super().__init__()
        self.wdl = wdl
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(num_filters * 8 * 8, hidden_size)
        if wdl:
            self.fc2 = nn.Linear(hidden_size, 3)
        else:
            self.fc2 = nn.Linear(hidden_size, 1)
            self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        if self.wdl:
            return self.fc2(x)
        else:
            return self.tanh(self.fc2(x))


# =============================================================================
# Horizontal Mirror Equivariance
# =============================================================================

def _build_policy_mirror() -> torch.LongTensor:
    """Build a 4672-element permutation mapping each policy index to its
    horizontally mirrored counterpart.

    Move encoding (from move_encoder.cpp):
      Queen moves [0-3583]:  from_sq*56 + direction*7 + (distance-1)
      Knight moves [3584-4095]: 3584 + from_sq*8 + knight_index
      Underpromotions [4096-4671]: 4096 + from_sq*9 + direction*3 + piece_index

    Horizontal mirror negates the file component of every square and
    reverses left/right in direction/knight indices.
    """
    mirror = torch.zeros(POLICY_SIZE, dtype=torch.long)

    def mirror_sq(sq):
        rank, file = divmod(sq, 8)
        return rank * 8 + (7 - file)

    # Queen direction mirror: N=0,NE=1,E=2,SE=3,S=4,SW=5,W=6,NW=7
    # Mirror negates file component: N↔N, NE↔NW, E↔W, SE↔SW, S↔S
    queen_dir_mirror = [0, 7, 6, 5, 4, 3, 2, 1]

    # Knight index mirror: negate file component → reverse order
    knight_mirror = [7, 6, 5, 4, 3, 2, 1, 0]

    # Underpromotion direction mirror: left(0)↔right(2), straight(1)↔straight(1)
    upromo_dir_mirror = [2, 1, 0]

    # Queen moves
    for from_sq in range(64):
        new_from = mirror_sq(from_sq)
        for direction in range(8):
            new_dir = queen_dir_mirror[direction]
            for dist_m1 in range(7):
                old_idx = from_sq * 56 + direction * 7 + dist_m1
                new_idx = new_from * 56 + new_dir * 7 + dist_m1
                mirror[old_idx] = new_idx

    # Knight moves
    for from_sq in range(64):
        new_from = mirror_sq(from_sq)
        for ki in range(8):
            new_ki = knight_mirror[ki]
            old_idx = 3584 + from_sq * 8 + ki
            new_idx = 3584 + new_from * 8 + new_ki
            mirror[old_idx] = new_idx

    # Underpromotions
    for from_sq in range(64):
        new_from = mirror_sq(from_sq)
        for direction in range(3):
            new_dir = upromo_dir_mirror[direction]
            for piece in range(3):
                old_idx = 4096 + from_sq * 9 + direction * 3 + piece
                new_idx = 4096 + new_from * 9 + new_dir * 3 + piece
                mirror[old_idx] = new_idx

    return mirror


# =============================================================================
# Main Network
# =============================================================================

class AlphaZeroNet(nn.Module):
    """AlphaZero neural network - compatible with Python backend checkpoints.

    Architecture matches alphazero/neural/network.py exactly for checkpoint compatibility.
    Uses AlphaZero paper standard settings:
    - 122 input channels (extended encoding)
    - 2 policy filters
    - 256 value hidden units

    Returns 4 values: (policy, value, policy_logits, wdl_logits)
    - policy: softmax probabilities over actions
    - value: scalar in [-1, 1] (WDL: P(win)-P(loss), scalar: tanh output)
    - policy_logits: raw logits before softmax (for log_softmax in training)
    - wdl_logits: raw WDL logits (None if wdl=False)
    """

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_filters: int = 192,
        num_blocks: int = 15,
        num_actions: int = POLICY_SIZE,
        policy_filters: int = 2,
        value_filters: int = 1,
        value_hidden: int = 256,
        wdl: bool = True,
        se_reduction: int = 16
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.num_actions = num_actions
        self.wdl = wdl
        self.se_reduction = se_reduction

        # Input convolution
        self.input_conv = ConvBlock(input_channels, num_filters)

        # Residual tower with SE blocks
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_filters, se_reduction=se_reduction) for _ in range(num_blocks)]
        )

        # Output heads
        self.policy_head = PolicyHead(num_filters, policy_filters, num_actions)
        self.value_head = ValueHead(num_filters, value_filters, value_hidden, wdl=wdl)

        # Policy mirror permutation for horizontal equivariance
        self.register_buffer('policy_mirror', _build_policy_mirror())

    def _trunk(self, x):
        """Shared trunk: input conv + residual tower."""
        return self.residual_tower(self.input_conv(x))

    def _mirror_input(self, x):
        """Horizontally flip input and fix castling channel encoding.

        Channel 16 encodes castling rights as:
          0.0 (none), 0.33 (queenside only), 0.67 (kingside only), 1.0 (both)
        After h-flip, kingside and queenside swap, so 0.67↔0.33.
        Values 0.0 and 1.0 are symmetric and unchanged.
        """
        x_f = torch.flip(x, [3])  # Flip file axis (dim 3 in NCHW)
        ch = x_f[:, 16:17, :, :]
        needs_swap = ((ch > 0.1) & (ch < 0.9)).float()
        fixed = needs_swap * (1.0 - ch) + (1.0 - needs_swap) * ch
        return torch.cat([x_f[:, :16], fixed, x_f[:, 17:]], dim=1)

    def forward(self, x, mask=None):
        B = x.shape[0]

        # Batch original + mirrored for a single efficient GPU pass
        x_flip = self._mirror_input(x)
        x_both = torch.cat([x, x_flip], dim=0)  # (2B, C, 8, 8)

        trunk = self._trunk(x_both)  # (2B, F, 8, 8)

        # Both heads on doubled batch
        policy_logits_both = self.policy_head(trunk)  # (2B, 4672)
        value_raw_both = self.value_head(trunk)        # (2B, 3)

        # Split original / flipped
        policy_orig, policy_flip = policy_logits_both[:B], policy_logits_both[B:]
        value_orig, value_flip = value_raw_both[:B], value_raw_both[B:]

        # Un-mirror flipped policy using precomputed permutation
        policy_flip_unmirrored = policy_flip[:, self.policy_mirror]

        # Average for equivariant output
        policy_logits = (policy_orig + policy_flip_unmirrored) * 0.5
        value_raw = (value_orig + value_flip) * 0.5

        # Standard masking and output
        if mask is not None:
            policy_logits = policy_logits.masked_fill(mask == 0, -1e4)

        policy = F.softmax(policy_logits, dim=1)

        if self.wdl:
            wdl_logits = value_raw
            wdl_probs = F.softmax(wdl_logits, dim=1)
            value = wdl_probs[:, 0:1] - wdl_probs[:, 2:3]
        else:
            wdl_logits = None
            value = value_raw

        return policy, value, policy_logits, wdl_logits

    def predict(self, x, legal_mask):
        """Inference shortcut: returns (softmax_policy, scalar_value, wdl_probs).

        wdl_probs is a (batch, 3) tensor of [win, draw, loss] probabilities,
        or None for legacy scalar-head models.
        """
        policy, value, _, wdl_logits = self.forward(x, legal_mask)
        wdl_probs = F.softmax(wdl_logits, dim=1) if wdl_logits is not None else None
        return policy, value.squeeze(-1), wdl_probs
