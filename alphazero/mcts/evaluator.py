"""Neural network evaluator interface for MCTS.

Provides a clean interface between MCTS and the neural network.
"""

import torch
import numpy as np
from torch.amp import autocast
from typing import Tuple, Optional, Protocol
from dataclasses import dataclass


class Evaluator(Protocol):
    """Protocol for neural network evaluators."""

    def evaluate(
        self,
        observation: np.ndarray,
        legal_mask: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Evaluate a position.

        Args:
            observation: Board observation (119, 8, 8)
            legal_mask: Legal action mask (4672,)

        Returns:
            Tuple of:
                - policy: Probability distribution (4672,)
                - value: Position evaluation in [-1, 1]
        """
        ...


class NetworkEvaluator:
    """Evaluator that uses an AlphaZero neural network."""

    def __init__(
        self,
        network: torch.nn.Module,
        device: str = "cuda",
        use_amp: bool = True
    ):
        """Initialize evaluator.

        Args:
            network: AlphaZero neural network
            device: Device to run inference on
            use_amp: Use mixed precision (FP16) for inference
        """
        self.network = network
        self.device = device
        self.use_amp = use_amp and device == "cuda"  # Only use AMP on CUDA
        self.network.eval()

    def evaluate(
        self,
        observation: np.ndarray,
        legal_mask: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Evaluate a single position.

        Args:
            observation: Board observation (119, 8, 8)
            legal_mask: Legal action mask (4672,)

        Returns:
            Tuple of (policy, value)
        """
        # Convert to tensors
        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(legal_mask).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.use_amp:
                with autocast('cuda'):
                    policy, value = self.network.predict(obs_tensor, mask_tensor)
            else:
                policy, value = self.network.predict(obs_tensor, mask_tensor)

        # Convert back to numpy
        policy_np = policy.squeeze(0).cpu().numpy()
        value_np = value.item()

        return policy_np, value_np

    def evaluate_batch(
        self,
        observations: np.ndarray,
        legal_masks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of positions.

        Args:
            observations: Batch of observations (batch, 119, 8, 8)
            legal_masks: Batch of legal masks (batch, 4672)

        Returns:
            Tuple of (policies, values)
        """
        obs_tensor = torch.from_numpy(observations).float().to(self.device)
        mask_tensor = torch.from_numpy(legal_masks).float().to(self.device)

        with torch.no_grad():
            if self.use_amp:
                with autocast('cuda'):
                    policies, values = self.network.predict(obs_tensor, mask_tensor)
            else:
                policies, values = self.network.predict(obs_tensor, mask_tensor)

        return policies.cpu().numpy(), values.cpu().numpy()


class RandomEvaluator:
    """Random evaluator for testing (uniform policy, zero value)."""

    def __init__(self, num_actions: int = 4672):
        self.num_actions = num_actions

    def evaluate(
        self,
        observation: np.ndarray,
        legal_mask: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Return uniform policy over legal moves and zero value."""
        policy = legal_mask.astype(np.float32)
        policy = policy / np.sum(policy)
        return policy, 0.0


class CachedEvaluator:
    """Evaluator with position caching for repeated evaluations."""

    def __init__(
        self,
        evaluator: Evaluator,
        cache_size: int = 10000
    ):
        """Initialize cached evaluator.

        Args:
            evaluator: Underlying evaluator
            cache_size: Maximum cache size
        """
        self.evaluator = evaluator
        self.cache_size = cache_size
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def evaluate(
        self,
        observation: np.ndarray,
        legal_mask: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Evaluate with caching."""
        # Create cache key from observation
        key = observation.tobytes()

        if key in self.cache:
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        result = self.evaluator.evaluate(observation, legal_mask)

        # Add to cache (simple LRU: just clear when full)
        if len(self.cache) >= self.cache_size:
            self.cache.clear()
        self.cache[key] = result

        return result

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
