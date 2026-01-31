"""Training loop for AlphaZero.

Implements the learner that trains the neural network from replay buffer data.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

from .replay_buffer import ReplayBuffer
from ..neural.network import AlphaZeroNetwork
from ..neural.loss import AlphaZeroLoss, compute_policy_accuracy, compute_value_accuracy
from ..config import TrainingConfig, AlphaZeroConfig


logger = logging.getLogger(__name__)


class Learner:
    """Training loop for AlphaZero neural network.

    Handles:
    - Sampling from replay buffer
    - Forward/backward passes
    - Learning rate scheduling
    - Checkpointing
    - Logging
    """

    def __init__(
        self,
        network: AlphaZeroNetwork,
        replay_buffer: ReplayBuffer,
        config: Optional[TrainingConfig] = None,
        device: str = "cuda"
    ):
        """Initialize learner.

        Args:
            network: Neural network to train
            replay_buffer: Replay buffer to sample from
            config: Training configuration
            device: Device to train on
        """
        self.network = network.to(device)
        self.replay_buffer = replay_buffer
        self.config = config or TrainingConfig()
        self.device = device

        # Optimizer: SGD with momentum (as in AlphaZero paper)
        self.optimizer = optim.SGD(
            self.network.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.config.lr_schedule_steps,
            gamma=self.config.lr_schedule_gamma
        )

        # Loss function
        self.loss_fn = AlphaZeroLoss()

        # Mixed precision training
        self.scaler = GradScaler() if self.config.use_amp else None

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Metrics history
        self.metrics_history = []

    def train_step(self) -> Dict[str, float]:
        """Execute a single training step.

        Returns:
            Dictionary of metrics
        """
        self.network.train()

        # Sample batch from replay buffer
        observations, legal_masks, policies, values = self.replay_buffer.sample_numpy(
            self.config.batch_size
        )

        # Convert to tensors with non_blocking transfers for better GPU utilization
        # non_blocking=True allows CPU→GPU transfer to overlap with computation
        obs_tensor = torch.from_numpy(observations).float().to(self.device, non_blocking=True)
        mask_tensor = torch.from_numpy(legal_masks).float().to(self.device, non_blocking=True)
        policy_tensor = torch.from_numpy(policies).float().to(self.device, non_blocking=True)
        value_tensor = torch.from_numpy(values).float().to(self.device, non_blocking=True)

        # Forward pass with optional mixed precision
        self.optimizer.zero_grad()

        if self.config.use_amp and self.scaler is not None:
            with autocast('cuda'):
                policy_logits, value_pred = self.network(obs_tensor, mask_tensor)
                loss, metrics = self.loss_fn(
                    policy_logits, policy_tensor,
                    value_pred, value_tensor,
                    mask_tensor
                )

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.config.max_grad_norm
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            policy_logits, value_pred = self.network(obs_tensor, mask_tensor)
            loss, metrics = self.loss_fn(
                policy_logits, policy_tensor,
                value_pred, value_tensor,
                mask_tensor
            )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.config.max_grad_norm
            )

            self.optimizer.step()

        # Compute additional metrics
        with torch.no_grad():
            metrics['policy_accuracy'] = compute_policy_accuracy(
                policy_logits, policy_tensor, mask_tensor
            )
            metrics['value_accuracy'] = compute_value_accuracy(
                value_pred, value_tensor
            )
            metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

        self.global_step += 1
        self.metrics_history.append(metrics)

        return metrics

    def train_epoch(self, steps_per_epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            steps_per_epoch: Number of training steps per epoch

        Returns:
            Average metrics for the epoch
        """
        epoch_metrics = []

        for step in range(steps_per_epoch):
            metrics = self.train_step()
            epoch_metrics.append(metrics)

            if (step + 1) % self.config.log_interval == 0:
                avg_metrics = self._average_metrics(epoch_metrics[-self.config.log_interval:])
                logger.info(
                    f"Step {self.global_step}: "
                    f"loss={avg_metrics['loss']:.4f}, "
                    f"policy_loss={avg_metrics['policy_loss']:.4f}, "
                    f"value_loss={avg_metrics['value_loss']:.4f}, "
                    f"policy_acc={avg_metrics['policy_accuracy']:.3f}"
                )

        # Update learning rate
        self.scheduler.step()
        self.epoch += 1

        return self._average_metrics(epoch_metrics)

    def _average_metrics(self, metrics_list) -> Dict[str, float]:
        """Average a list of metric dictionaries."""
        if not metrics_list:
            return {}

        avg = {}
        for key in metrics_list[0].keys():
            avg[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
        return avg

    def save_checkpoint(self, path: str, extra_state: dict = None) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint
            extra_state: Additional state to save
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Get network architecture info
        num_filters = self.network.num_filters
        num_blocks = self.network.num_blocks

        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'num_filters': num_filters,
            'num_blocks': num_blocks,
        }

        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()

        if extra_state:
            state.update(extra_state)

        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> dict:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Extra state from checkpoint
        """
        state = torch.load(path, map_location=self.device)

        self.global_step = state['global_step']
        self.epoch = state['epoch']
        self.network.load_state_dict(state['network_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in state:
            self.scaler.load_state_dict(state['scaler_state_dict'])

        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")

        # Return any extra state
        known_keys = {
            'global_step', 'epoch', 'network_state_dict',
            'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict'
        }
        return {k: v for k, v in state.items() if k not in known_keys}

    def get_network_weights(self) -> dict:
        """Get network weights for distribution to actors."""
        return {k: v.cpu() for k, v in self.network.state_dict().items()}

    def set_network_weights(self, weights: dict) -> None:
        """Set network weights (e.g., from another process)."""
        self.network.load_state_dict(weights)
