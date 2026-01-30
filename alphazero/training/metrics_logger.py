"""Metrics logger for training visualization.

Logs training metrics to JSON files for visualization dashboard.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from collections import deque


class MetricsLogger:
    """Logs training metrics to JSON files for visualization."""

    def __init__(self, log_dir: str = "logs/metrics", max_history: int = 10000):
        """Initialize metrics logger.

        Args:
            log_dir: Directory to save metrics
            max_history: Maximum number of metrics to keep in memory
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.log_dir / "training_metrics.jsonl"
        self.summary_file = self.log_dir / "training_summary.json"

        # In-memory history (for dashboard)
        self.metrics_history: deque = deque(maxlen=max_history)

        # Summary statistics
        self.summary = {
            'start_time': time.time(),
            'total_steps': 0,
            'total_games': 0,
            'best_loss': float('inf'),
            'current_lr': 0.0,
        }

    def log_step(self, step: int, metrics: Dict[str, float],
                 games: int = 0, buffer_size: int = 0):
        """Log metrics for a training step.

        Args:
            step: Training step number
            metrics: Dictionary of metrics (loss, policy_loss, value_loss, etc.)
            games: Total games played so far
            buffer_size: Current replay buffer size
        """
        # Add timestamp and step info
        log_entry = {
            'timestamp': time.time(),
            'step': step,
            'games': games,
            'buffer_size': buffer_size,
            **metrics
        }

        # Add to history
        self.metrics_history.append(log_entry)

        # Append to JSONL file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Update summary
        self.summary['total_steps'] = step
        self.summary['total_games'] = games
        if 'loss' in metrics:
            self.summary['best_loss'] = min(self.summary['best_loss'], metrics['loss'])
        if 'learning_rate' in metrics:
            self.summary['current_lr'] = metrics['learning_rate']

        # Save summary periodically (every 100 steps)
        if step % 100 == 0:
            self._save_summary()

    def log_iteration(self, iteration: int, metrics: Dict[str, float]):
        """Log metrics for an iteration (for iterative training).

        Args:
            iteration: Iteration number
            metrics: Dictionary of iteration-level metrics
        """
        iteration_file = self.log_dir / "iterations.jsonl"
        log_entry = {
            'timestamp': time.time(),
            'iteration': iteration,
            **metrics
        }

        with open(iteration_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def get_recent_metrics(self, n: int = 1000) -> List[Dict]:
        """Get recent metrics for visualization.

        Args:
            n: Number of recent metrics to return

        Returns:
            List of recent metric dictionaries
        """
        return list(self.metrics_history)[-n:]

    def get_summary(self) -> Dict:
        """Get training summary statistics."""
        elapsed_time = time.time() - self.summary['start_time']
        self.summary['elapsed_time'] = elapsed_time
        self.summary['steps_per_second'] = self.summary['total_steps'] / elapsed_time if elapsed_time > 0 else 0
        return self.summary.copy()

    def _save_summary(self):
        """Save summary to JSON file."""
        with open(self.summary_file, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)

    def load_history(self) -> List[Dict]:
        """Load full metrics history from JSONL file.

        Returns:
            List of all logged metrics
        """
        if not self.metrics_file.exists():
            return []

        history = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line))
        return history

    def clear(self):
        """Clear all logged metrics."""
        self.metrics_history.clear()
        if self.metrics_file.exists():
            self.metrics_file.unlink()
        if self.summary_file.exists():
            self.summary_file.unlink()
