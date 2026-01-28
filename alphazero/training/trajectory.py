"""Trajectory data structures for AlphaZero training.

Stores game trajectories for replay buffer and training.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrajectoryState:
    """A single state in a game trajectory.

    Stores all information needed for training:
    - observation: Neural network input
    - legal_mask: Legal action mask
    - policy: MCTS visit distribution (training target)
    - value: Game outcome from this player's view (training target)
    - action: Action taken
    - player: Which player (0=white, 1=black)
    """
    observation: np.ndarray   # (119, 8, 8)
    legal_mask: np.ndarray    # (4672,)
    policy: np.ndarray        # (4672,) MCTS visit distribution
    value: float              # Game outcome from this player's view
    action: int               # Action taken
    player: int               # 0=white, 1=black

    def to_training_sample(self):
        """Convert to training sample tuple."""
        return (
            self.observation,
            self.legal_mask,
            self.policy,
            self.value
        )


@dataclass
class Trajectory:
    """A complete game trajectory.

    Contains all states from a single self-play game.
    """
    states: List[TrajectoryState] = field(default_factory=list)
    result: Optional[float] = None  # 1.0 white wins, -1.0 black wins, 0.0 draw

    def add_state(
        self,
        observation: np.ndarray,
        legal_mask: np.ndarray,
        policy: np.ndarray,
        action: int,
        player: int
    ) -> None:
        """Add a state to the trajectory.

        Note: value is set later when game ends.
        """
        state = TrajectoryState(
            observation=observation,
            legal_mask=legal_mask,
            policy=policy,
            value=0.0,  # Will be set when game ends
            action=action,
            player=player
        )
        self.states.append(state)

    def set_result(self, result: float) -> None:
        """Set the game result and update all state values.

        Args:
            result: Game result (1.0 white wins, -1.0 black wins, 0.0 draw)
        """
        self.result = result

        # Update values for each state
        for state in self.states:
            if state.player == 0:  # White
                state.value = result
            else:  # Black
                state.value = -result

    def __len__(self) -> int:
        return len(self.states)

    def get_training_samples(self) -> List[TrajectoryState]:
        """Get all states as training samples."""
        return self.states


@dataclass
class TrajectoryBatch:
    """A batch of training samples."""
    observations: np.ndarray   # (batch, 119, 8, 8)
    legal_masks: np.ndarray    # (batch, 4672)
    policies: np.ndarray       # (batch, 4672)
    values: np.ndarray         # (batch,)

    @classmethod
    def from_states(cls, states: List[TrajectoryState]) -> 'TrajectoryBatch':
        """Create a batch from a list of trajectory states."""
        return cls(
            observations=np.stack([s.observation for s in states]),
            legal_masks=np.stack([s.legal_mask for s in states]),
            policies=np.stack([s.policy for s in states]),
            values=np.array([s.value for s in states], dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.values)
