"""Self-play module for AlphaZero."""

from .game import SelfPlayGame, play_game
from .actor import Actor, ActorProcess, run_actor_loop
from .coordinator import SelfPlayCoordinator, BatchedSelfPlayCoordinator, run_training
from .inference_server import InferenceServer, BatchedEvaluator
from .batched_actor import BatchedActorProcess, BatchedActor

__all__ = [
    "SelfPlayGame",
    "play_game",
    "Actor",
    "ActorProcess",
    "run_actor_loop",
    "SelfPlayCoordinator",
    "BatchedSelfPlayCoordinator",
    "run_training",
    "InferenceServer",
    "BatchedEvaluator",
    "BatchedActorProcess",
    "BatchedActor",
]
