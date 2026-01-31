"""Batched GPU inference server for efficient multi-actor self-play.

This module provides a centralized inference server that batches requests
from multiple actors and runs them on the GPU together. This amortizes
GPU kernel launch overhead and allows actors to focus on MCTS.

Architecture:
    Actor 1 ──┐
    Actor 2 ──┼──► Request Queue ──► InferenceServer (GPU) ──► Response Queues
    Actor 3 ──┤                              │
    Actor 4 ──┘                              ▼
                                      Batched forward pass
"""

import torch
import numpy as np
import threading
import time
import logging
from torch.amp import autocast
from typing import Dict, Tuple, Optional, List
from multiprocessing import Process, Queue, Event
from queue import Empty
from dataclasses import dataclass
import traceback

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request for neural network inference."""
    request_id: int
    actor_id: int
    observation: np.ndarray  # (119, 8, 8)
    legal_mask: np.ndarray   # (4672,)


@dataclass
class InferenceResponse:
    """Response from neural network inference."""
    request_id: int
    policy: np.ndarray  # (4672,)
    value: float


class InferenceServer(Process):
    """Centralized GPU inference server that batches requests from actors.

    The server collects inference requests from multiple actors, batches them
    together, runs a single forward pass on the GPU, and distributes results
    back to the actors.

    Benefits:
    - Amortizes GPU kernel launch overhead across multiple requests
    - Frees actor CPUs for MCTS tree operations
    - Better GPU utilization with larger batch sizes
    """

    def __init__(
        self,
        request_queue: Queue,
        response_queues: Dict[int, Queue],
        network_class,
        network_kwargs: dict,
        initial_weights: Optional[dict] = None,
        device: str = "cuda",
        batch_size: int = 512,
        batch_timeout: float = 0.02,  # 20ms timeout to collect batch (was 1ms)
        weight_queue: Optional[Queue] = None,
        use_amp: bool = True,
        shutdown_event: Optional[Event] = None,
    ):
        """Initialize inference server.

        Args:
            request_queue: Queue to receive inference requests
            response_queues: Dict mapping actor_id to response queue
            network_class: Neural network class to instantiate
            network_kwargs: Kwargs for network constructor
            initial_weights: Initial network weights
            device: Device to run inference on
            batch_size: Maximum batch size
            batch_timeout: Time to wait for batch to fill (seconds)
            weight_queue: Queue to receive weight updates
            use_amp: Use mixed precision (FP16) for inference
            shutdown_event: Event to signal server shutdown
        """
        super().__init__()
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.network_class = network_class
        self.network_kwargs = network_kwargs
        self.initial_weights = initial_weights
        self.device = device
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.weight_queue = weight_queue
        self.use_amp = use_amp and device == "cuda"  # Only use AMP on CUDA
        self.shutdown_event = shutdown_event
        self.daemon = True

        # Statistics (updated in run())
        self.total_requests = 0
        self.total_batches = 0

    def run(self):
        """Main server loop."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - InferenceServer - %(levelname)s - %(message)s"
        )

        try:
            # Create network on GPU
            logger.info(f"InferenceServer starting on {self.device}")
            logger.info(f"Mixed precision (AMP) inference: {self.use_amp}")
            logger.info(f"Response queues available for actors: {list(self.response_queues.keys())}")

            network = self.network_class(**self.network_kwargs)
            network = network.to(self.device)
            network.eval()

            if self.initial_weights:
                network.load_state_dict(self.initial_weights)
                logger.info("Loaded initial weights")

            total_requests = 0
            total_batches = 0
            last_log_time = time.time()

            # Check shutdown_event if provided, otherwise run forever (daemon will be killed)
            def should_run():
                if self.shutdown_event is not None:
                    return not self.shutdown_event.is_set()
                return True

            while should_run():
                # Check for weight updates
                if self.weight_queue:
                    try:
                        while not self.weight_queue.empty():
                            weights = self.weight_queue.get_nowait()
                            network.load_state_dict(weights)
                            # logger.info("Updated network weights")
                    except Empty:
                        pass

                # Collect batch of requests
                batch = self._collect_batch()

                if not batch:
                    continue

                # Run batched inference
                try:
                    policies, values = self._run_inference(network, batch)

                    # Send responses
                    for i, request in enumerate(batch):
                        response = InferenceResponse(
                            request_id=request.request_id,
                            policy=policies[i],
                            value=float(values[i])
                        )

                        if request.actor_id in self.response_queues:
                            try:
                                self.response_queues[request.actor_id].put(
                                    response, timeout=1.0
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to send response to actor {request.actor_id}: {e}"
                                )
                        else:
                            logger.error(
                                f"No response queue for actor {request.actor_id}! "
                                f"Available queues: {list(self.response_queues.keys())}"
                            )

                    total_requests += len(batch)
                    total_batches += 1

                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    logger.error(traceback.format_exc())

                # Periodic logging
                now = time.time()
                if now - last_log_time > 30:
                    avg_batch = total_requests / total_batches if total_batches > 0 else 0
                    logger.info(
                        f"Inference stats: {total_requests} req."
                        f"{total_batches} batches, avg batch size: {avg_batch:.1f}"
                    )
                    last_log_time = now

            # Graceful shutdown
            logger.info("InferenceServer shutting down gracefully...")

        except Exception as e:
            logger.error(f"InferenceServer fatal error: {e}")
            logger.error(traceback.format_exc())

    def _collect_batch(self) -> List[InferenceRequest]:
        """Collect a batch of requests with adaptive timeout.

        Uses adaptive batching strategy:
        - Waits for at least 25% of batch_size before timing out
        - Hard timeout at 2x configured timeout to prevent starvation
        - This improves GPU utilization by collecting larger batches
        """
        batch = []
        start_time = time.time()
        min_batch = max(1, self.batch_size // 4)  # At least 25% full

        # Get first request (blocking with longer timeout)
        try:
            request = self.request_queue.get(timeout=0.1)
            batch.append(request)
        except Empty:
            return batch

        # Collect more requests with adaptive timeout
        while len(batch) < self.batch_size:
            elapsed = time.time() - start_time

            # Exit if timeout AND we have minimum batch
            if elapsed > self.batch_timeout and len(batch) >= min_batch:
                break

            # Hard timeout at 2x configured timeout to prevent starvation
            if elapsed > self.batch_timeout * 2:
                break

            try:
                request = self.request_queue.get(timeout=0.001)
                batch.append(request)
            except Empty:
                if len(batch) >= min_batch:
                    break

        return batch

    def _run_inference(
        self,
        network: torch.nn.Module,
        batch: List[InferenceRequest]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run batched inference on GPU with optional mixed precision."""
        # Stack observations and masks
        observations = np.stack([r.observation for r in batch])
        legal_masks = np.stack([r.legal_mask for r in batch])

        # Convert to tensors
        obs_tensor = torch.from_numpy(observations).float().to(self.device)
        mask_tensor = torch.from_numpy(legal_masks).float().to(self.device)

        # Run inference with optional mixed precision
        with torch.no_grad():
            if self.use_amp:
                with autocast('cuda'):
                    policies, values = network.predict(obs_tensor, mask_tensor)
            else:
                policies, values = network.predict(obs_tensor, mask_tensor)

        return policies.cpu().numpy(), values.cpu().numpy().flatten()


class BatchedEvaluator:
    """Evaluator that sends requests to a centralized inference server.

    This evaluator is used by actors when batched GPU inference is enabled.
    Instead of running inference locally, it sends requests to the inference
    server and waits for responses.
    """

    def __init__(
        self,
        actor_id: int,
        request_queue: Queue,
        response_queue: Queue,
        timeout: float = 5.0
    ):
        """Initialize batched evaluator.

        Args:
            actor_id: ID of the actor using this evaluator
            request_queue: Queue to send requests to inference server
            response_queue: Queue to receive responses from server
            timeout: Timeout for waiting for response (seconds)
        """
        self.actor_id = actor_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.timeout = timeout
        self.request_counter = 0
        self._pending_requests: Dict[int, InferenceRequest] = {}

    def evaluate(
        self,
        observation: np.ndarray,
        legal_mask: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Evaluate a position via the inference server.

        Args:
            observation: Board observation (119, 8, 8)
            legal_mask: Legal action mask (4672,)

        Returns:
            Tuple of (policy, value)
        """
        # Create request
        request_id = self.request_counter
        self.request_counter += 1

        request = InferenceRequest(
            request_id=request_id,
            actor_id=self.actor_id,
            observation=observation.copy(),
            legal_mask=legal_mask.copy()
        )

        # Send request
        self.request_queue.put(request)

        # Wait for response
        try:
            response = self.response_queue.get(timeout=self.timeout)
            return response.policy, response.value
        except Empty:
            raise TimeoutError(
                f"Inference request {request_id} timed out after {self.timeout}s"
            )


class InferenceClient:
    """Convenience class for managing batched inference from actor side."""

    def __init__(
        self,
        actor_id: int,
        request_queue: Queue,
        response_queue: Queue
    ):
        self.evaluator = BatchedEvaluator(
            actor_id=actor_id,
            request_queue=request_queue,
            response_queue=response_queue
        )

    def get_evaluator(self) -> BatchedEvaluator:
        """Get the evaluator for use with MCTS."""
        return self.evaluator
