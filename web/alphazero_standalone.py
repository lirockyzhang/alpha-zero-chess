"""Standalone AlphaZero components for the web interface.

This module provides self-contained implementations of:
- AlphaZeroNetwork: Neural network with policy and value heads
- GameState: Game state wrapper around python-chess Board
- MCTSConfig: Configuration dataclass

These are decoupled from the main alphazero/ package to allow the web
interface to work independently.
"""

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


# =============================================================================
# Neural Network Building Blocks
# =============================================================================

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
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, num_filters: int):
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
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class PolicyHead(nn.Module):
    """Policy head: outputs action probabilities."""

    def __init__(
        self,
        in_channels: int,
        num_filters: int = 2,
        num_actions: int = 4672
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(num_filters * 8 * 8, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class ValueHead(nn.Module):
    """Value head: outputs position evaluation."""

    def __init__(
        self,
        in_channels: int,
        num_filters: int = 1,
        hidden_size: int = 256
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(num_filters * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.tanh(self.fc2(x))


# =============================================================================
# AlphaZero Neural Network
# =============================================================================

class AlphaZeroNetwork(nn.Module):
    """AlphaZero neural network with policy and value heads."""

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
        """Forward pass through the network."""
        # Shared trunk
        x = self.input_conv(x)
        x = self.residual_tower(x)

        # Policy head
        policy_logits = self.policy_head(x)

        # Apply legal move mask
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(legal_mask == 0, -1e4)

        # Value head
        value = self.value_head(x)

        return policy_logits, value

    def predict(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get policy probabilities and value for inference."""
        policy_logits, value = self.forward(x, legal_mask)
        policy = F.softmax(policy_logits, dim=-1)
        return policy, value.squeeze(-1)


# =============================================================================
# Move Encoder (C++-aligned)
# =============================================================================

class CppAlignedMoveEncoder:
    """Move encoder aligned with C++ implementation."""

    def __init__(self):
        self.queen_offset = 0
        self.knight_offset = 3584  # 56 * 64
        self.underpromo_offset = 4096  # 56 * 64 + 8 * 64
        self.num_actions = 4672

        # Queen move directions (N, NE, E, SE, S, SW, W, NW)
        self.queen_directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1),
        ]

        # Knight moves (clockwise from top-right)
        self.knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1),
        ]

    def _flip_square(self, square: int) -> int:
        return 63 - square

    def decode(self, action: int, board: chess.Board) -> chess.Move:
        """Decode an action index to a chess move."""
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Invalid action index: {action}")

        flip = (board.turn == chess.BLACK)

        if action < self.knight_offset:
            # Queen-like move
            from_sq = action // 56
            remainder = action % 56
            direction = remainder // 7
            distance = (remainder % 7) + 1

            from_rank, from_file = from_sq // 8, from_sq % 8
            dr, dc = self.queen_directions[direction]
            to_rank = from_rank + dr * distance
            to_file = from_file + dc * distance

            if not (0 <= to_rank < 8 and 0 <= to_file < 8):
                raise ValueError(f"Invalid decoded position for action {action}")

            to_sq = to_rank * 8 + to_file

            if flip:
                from_sq = self._flip_square(from_sq)
                to_sq = self._flip_square(to_sq)

            # Check for queen promotion
            promotion = None
            piece = board.piece_at(from_sq)
            if piece and piece.piece_type == chess.PAWN:
                to_rank_abs = chess.square_rank(to_sq)
                if (piece.color == chess.WHITE and to_rank_abs == 7) or \
                   (piece.color == chess.BLACK and to_rank_abs == 0):
                    promotion = chess.QUEEN

            return chess.Move(from_sq, to_sq, promotion=promotion)

        elif action < self.underpromo_offset:
            # Knight move
            offset = action - self.knight_offset
            from_sq = offset // 8
            knight_idx = offset % 8

            from_rank, from_file = from_sq // 8, from_sq % 8
            dr, dc = self.knight_moves[knight_idx]
            to_rank = from_rank + dr
            to_file = from_file + dc

            if not (0 <= to_rank < 8 and 0 <= to_file < 8):
                raise ValueError(f"Invalid decoded knight position for action {action}")

            to_sq = to_rank * 8 + to_file

            if flip:
                from_sq = self._flip_square(from_sq)
                to_sq = self._flip_square(to_sq)

            return chess.Move(from_sq, to_sq)

        else:
            # Underpromotion
            offset = action - self.underpromo_offset
            from_sq = offset // 9
            remainder = offset % 9
            direction = remainder // 3
            piece_idx = remainder % 3

            from_file = from_sq % 8
            file_diff = direction - 1
            to_file = from_file + file_diff

            from_rank = from_sq // 8
            to_rank = from_rank - 1 if flip else from_rank + 1

            if not (0 <= to_rank < 8 and 0 <= to_file < 8):
                raise ValueError(f"Invalid decoded underpromotion position for action {action}")

            to_sq = to_rank * 8 + to_file

            if flip:
                from_sq = self._flip_square(from_sq)
                to_sq = self._flip_square(to_sq)

            promotion = [chess.KNIGHT, chess.BISHOP, chess.ROOK][piece_idx]
            return chess.Move(from_sq, to_sq, promotion=promotion)


# =============================================================================
# Game State
# =============================================================================

@dataclass(frozen=True)
class GameResult:
    """Result of a completed game."""
    winner: Optional[bool]  # True=White, False=Black, None=Draw
    termination: str


class GameState:
    """Game state wrapper around python-chess Board."""

    def __init__(
        self,
        board: Optional[chess.Board] = None,
        move_encoder: Optional[CppAlignedMoveEncoder] = None
    ):
        self._board = board.copy() if board else chess.Board()
        self._move_encoder = move_encoder or CppAlignedMoveEncoder()

    @property
    def board(self) -> chess.Board:
        """Get a copy of the underlying board."""
        return self._board.copy()

    def apply_move(self, move: chess.Move) -> 'GameState':
        """Apply a chess.Move and return a new game state."""
        new_board = self._board.copy()
        new_board.push(move)
        return GameState(board=new_board, move_encoder=self._move_encoder)

    def is_terminal(self) -> bool:
        """Check if the game has ended."""
        return self._board.is_game_over()

    def get_result(self) -> Optional[GameResult]:
        """Get the game result if terminal."""
        if not self.is_terminal():
            return None

        outcome = self._board.outcome()
        if outcome is None:
            return None

        return GameResult(
            winner=outcome.winner,
            termination=outcome.termination.name.lower()
        )

    def action_to_move(self, action: int) -> chess.Move:
        """Convert an action index to a chess.Move."""
        return self._move_encoder.decode(action, self._board)

    def fen(self) -> str:
        """Get FEN string of current position."""
        return self._board.fen()

    def __str__(self) -> str:
        return str(self._board)

    def __repr__(self) -> str:
        return f"GameState(fen='{self._board.fen()}')"


# =============================================================================
# MCTS Config (minimal)
# =============================================================================

@dataclass
class MCTSConfig:
    """MCTS configuration."""
    num_simulations: int = 800
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
