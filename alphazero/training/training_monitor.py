"""Training metrics monitor for tracking performance and estimating completion time."""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Deque
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Tracks training metrics and provides ETA estimates."""

    # Configuration
    window_size: int = 60  # Rolling window in seconds for rate calculations

    # Tracking state
    start_time: float = field(default_factory=time.time)

    # Games and positions
    _games_history: Deque = field(default_factory=lambda: deque(maxlen=120))
    _positions_history: Deque = field(default_factory=lambda: deque(maxlen=120))
    _steps_history: Deque = field(default_factory=lambda: deque(maxlen=120))

    # Last known values
    _last_games: int = 0
    _last_positions: int = 0
    _last_steps: int = 0
    _last_update: float = field(default_factory=time.time)

    def update(self, games: int, positions: int, steps: int = 0) -> None:
        """Update metrics with current values."""
        now = time.time()

        # Record history for rate calculations
        self._games_history.append((now, games))
        self._positions_history.append((now, positions))
        self._steps_history.append((now, steps))

        self._last_games = games
        self._last_positions = positions
        self._last_steps = steps
        self._last_update = now

    def _calculate_rate(self, history: Deque, window_seconds: Optional[int] = None) -> float:
        """Calculate rate from history within window."""
        if len(history) < 2:
            return 0.0

        window = window_seconds or self.window_size
        now = time.time()
        cutoff = now - window

        # Find oldest entry within window
        recent = [(t, v) for t, v in history if t >= cutoff]
        if len(recent) < 2:
            # Fall back to all available data
            recent = list(history)

        if len(recent) < 2:
            return 0.0

        oldest_time, oldest_val = recent[0]
        newest_time, newest_val = recent[-1]

        time_diff = newest_time - oldest_time
        if time_diff <= 0:
            return 0.0

        return (newest_val - oldest_val) / time_diff

    @property
    def games_per_hour(self) -> float:
        """Current games per hour rate."""
        return self._calculate_rate(self._games_history) * 3600

    @property
    def positions_per_hour(self) -> float:
        """Current positions per hour rate."""
        return self._calculate_rate(self._positions_history) * 3600

    @property
    def positions_per_second(self) -> float:
        """Current positions per second rate."""
        return self._calculate_rate(self._positions_history)

    @property
    def steps_per_second(self) -> float:
        """Current training steps per second rate."""
        return self._calculate_rate(self._steps_history)

    @property
    def steps_per_hour(self) -> float:
        """Current training steps per hour rate."""
        return self._calculate_rate(self._steps_history) * 3600

    @property
    def elapsed_time(self) -> float:
        """Total elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def elapsed_time_str(self) -> str:
        """Elapsed time as human-readable string."""
        return self._format_duration(self.elapsed_time)

    def eta_to_positions(self, target_positions: int) -> Optional[float]:
        """Estimate seconds until target positions reached."""
        rate = self._calculate_rate(self._positions_history)
        if rate <= 0:
            return None

        remaining = target_positions - self._last_positions
        if remaining <= 0:
            return 0.0

        return remaining / rate

    def eta_to_steps(self, target_steps: int) -> Optional[float]:
        """Estimate seconds until target training steps reached."""
        rate = self._calculate_rate(self._steps_history)
        if rate <= 0:
            return None

        remaining = target_steps - self._last_steps
        if remaining <= 0:
            return 0.0

        return remaining / rate

    def eta_to_positions_str(self, target_positions: int) -> str:
        """ETA to target positions as human-readable string."""
        eta = self.eta_to_positions(target_positions)
        if eta is None:
            return "calculating..."
        return self._format_duration(eta)

    def eta_to_steps_str(self, target_steps: int) -> str:
        """ETA to target steps as human-readable string."""
        eta = self.eta_to_steps(target_steps)
        if eta is None:
            return "calculating..."
        return self._format_duration(eta)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in seconds to human-readable string."""
        if seconds < 0:
            return "0s"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    def get_buffer_fill_summary(self, target: int) -> dict:
        """Get summary dict for buffer filling phase."""
        return {
            'games': self._last_games,
            'positions': self._last_positions,
            'games_per_hour': self.games_per_hour,
            'positions_per_hour': self.positions_per_hour,
            'elapsed': self.elapsed_time_str,
            'eta': self.eta_to_positions_str(target),
            'progress_pct': min(100.0, (self._last_positions / target) * 100) if target > 0 else 0,
        }

    def get_training_summary(self, target_steps: int) -> dict:
        """Get summary dict for training phase."""
        return {
            'steps': self._last_steps,
            'games': self._last_games,
            'positions': self._last_positions,
            'steps_per_hour': self.steps_per_hour,
            'games_per_hour': self.games_per_hour,
            'elapsed': self.elapsed_time_str,
            'eta': self.eta_to_steps_str(target_steps),
            'progress_pct': min(100.0, (self._last_steps / target_steps) * 100) if target_steps > 0 else 0,
        }

    def format_buffer_status(self, target: int) -> str:
        """Format buffer filling status for display."""
        summary = self.get_buffer_fill_summary(target)
        return (
            f"games={summary['games']}, "
            f"pos/hr={summary['positions_per_hour']:.0f}, "
            f"games/hr={summary['games_per_hour']:.1f}, "
            f"ETA={summary['eta']}"
        )

    def format_training_status(self, target_steps: int) -> str:
        """Format training status for display."""
        summary = self.get_training_summary(target_steps)
        return (
            f"steps/hr={summary['steps_per_hour']:.0f}, "
            f"games/hr={summary['games_per_hour']:.1f}, "
            f"ETA={summary['eta']}"
        )

    def log_buffer_progress(self, target: int) -> None:
        """Log buffer filling progress."""
        summary = self.get_buffer_fill_summary(target)
        logger.info(
            f"Buffer progress: {summary['positions']}/{target} ({summary['progress_pct']:.1f}%) | "
            f"Games: {summary['games']} | "
            f"Rate: {summary['positions_per_hour']:.0f} pos/hr, {summary['games_per_hour']:.1f} games/hr | "
            f"Elapsed: {summary['elapsed']} | "
            f"ETA: {summary['eta']}"
        )

    def log_training_progress(self, current_step: int, target_steps: int, loss: float) -> None:
        """Log training progress."""
        summary = self.get_training_summary(target_steps)
        logger.info(
            f"Training: {current_step}/{target_steps} ({summary['progress_pct']:.1f}%) | "
            f"Loss: {loss:.4f} | "
            f"Rate: {summary['steps_per_hour']:.0f} steps/hr | "
            f"Games: {summary['games']} ({summary['games_per_hour']:.1f}/hr) | "
            f"Elapsed: {summary['elapsed']} | "
            f"ETA: {summary['eta']}"
        )
