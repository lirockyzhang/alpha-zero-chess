"""Loss functions for AlphaZero training.

Implements:
- Policy loss: Cross-entropy between MCTS policy and network output
- Value loss: MSE between game outcome and network value prediction
- Combined loss: L = L_policy + L_value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class AlphaZeroLoss(nn.Module):
    """Combined policy and value loss for AlphaZero.

    L = L_policy + c_value * L_value

    Where:
        L_policy = -π^T · log(p)  (cross-entropy with MCTS policy)
        L_value = (z - v)²        (MSE with game outcome)
    """

    def __init__(self, value_weight: float = 1.0):
        """Initialize loss function.

        Args:
            value_weight: Weight for value loss (default 1.0)
        """
        super().__init__()
        self.value_weight = value_weight

    def forward(
        self,
        policy_logits: torch.Tensor,
        target_policy: torch.Tensor,
        value: torch.Tensor,
        target_value: torch.Tensor,
        legal_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.

        Args:
            policy_logits: Network policy output (batch, num_actions)
            target_policy: MCTS visit distribution (batch, num_actions)
            value: Network value output (batch, 1)
            target_value: Game outcome (batch,) in {-1, 0, 1}
            legal_mask: Legal action mask (batch, num_actions)

        Returns:
            Tuple of:
                - total_loss: Combined loss scalar
                - metrics: Dict with individual loss components
        """
        # Policy loss: cross-entropy with MCTS policy
        # Apply mask to logits before log_softmax
        # Use -1e4 instead of -inf for FP16 numerical stability (FP16 max ~65k)
        masked_logits = policy_logits.masked_fill(legal_mask == 0, -1e4)
        log_probs = F.log_softmax(masked_logits, dim=-1)

        # Cross-entropy: -sum(target * log(pred))
        policy_loss = -torch.sum(target_policy * log_probs, dim=-1).mean()

        # Value loss: MSE
        value = value.squeeze(-1)
        value_loss = F.mse_loss(value, target_value)

        # Combined loss
        total_loss = policy_loss + self.value_weight * value_loss

        metrics = {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
        }

        return total_loss, metrics


def policy_loss(
    policy_logits: torch.Tensor,
    target_policy: torch.Tensor,
    legal_mask: torch.Tensor
) -> torch.Tensor:
    """Compute policy cross-entropy loss.

    Args:
        policy_logits: Network output logits (batch, num_actions)
        target_policy: MCTS visit distribution (batch, num_actions)
        legal_mask: Legal action mask (batch, num_actions)

    Returns:
        Policy loss scalar
    """
    # Use -1e4 instead of -inf for FP16 numerical stability (FP16 max ~65k)
    masked_logits = policy_logits.masked_fill(legal_mask == 0, -1e4)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    return -torch.sum(target_policy * log_probs, dim=-1).mean()


def value_loss(
    value: torch.Tensor,
    target_value: torch.Tensor
) -> torch.Tensor:
    """Compute value MSE loss.

    Args:
        value: Network value output (batch, 1) or (batch,)
        target_value: Game outcome (batch,)

    Returns:
        Value loss scalar
    """
    if value.dim() == 2:
        value = value.squeeze(-1)
    return F.mse_loss(value, target_value)


def compute_policy_accuracy(
    policy_logits: torch.Tensor,
    target_policy: torch.Tensor,
    legal_mask: torch.Tensor
) -> float:
    """Compute accuracy of policy predictions.

    Accuracy is the fraction of positions where the network's
    top choice matches the MCTS top choice.

    Args:
        policy_logits: Network output logits (batch, num_actions)
        target_policy: MCTS visit distribution (batch, num_actions)
        legal_mask: Legal action mask (batch, num_actions)

    Returns:
        Accuracy as a float in [0, 1]
    """
    # Get network's top choice
    # Use -1e4 instead of -inf for FP16 numerical stability (FP16 max ~65k)
    masked_logits = policy_logits.masked_fill(legal_mask == 0, -1e4)
    pred_actions = masked_logits.argmax(dim=-1)

    # Get MCTS top choice
    target_actions = target_policy.argmax(dim=-1)

    # Compute accuracy
    correct = (pred_actions == target_actions).float().mean()
    return correct.item()


def compute_value_accuracy(
    value: torch.Tensor,
    target_value: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """Compute accuracy of value predictions.

    A prediction is correct if it has the same sign as the target
    (or both are within threshold of zero for draws).

    Args:
        value: Network value output (batch, 1) or (batch,)
        target_value: Game outcome (batch,)
        threshold: Threshold for considering a value as "draw"

    Returns:
        Accuracy as a float in [0, 1]
    """
    if value.dim() == 2:
        value = value.squeeze(-1)

    # Classify predictions
    pred_sign = torch.sign(value)
    target_sign = torch.sign(target_value)

    # Handle draws (values close to 0)
    pred_sign = torch.where(
        torch.abs(value) < threshold,
        torch.zeros_like(pred_sign),
        pred_sign
    )

    correct = (pred_sign == target_sign).float().mean()
    return correct.item()
