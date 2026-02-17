"""Claude Code agent integration for AlphaZero training.

Writes a JSONL log file (claude_log.jsonl) that Claude Code can tail to
monitor training progress and detect issues autonomously.

Usage:
    Enabled by passing --claude to train.py.
    Zero overhead when not enabled (all hooks guarded by None check).
"""

import json
import math
import os
from collections import deque
from datetime import datetime


class ClaudeInterface:
    """Manages JSONL log for Claude Code agent integration."""

    def __init__(self, run_dir: str, resume: bool = False, trend_window: int = 20):
        self.run_dir = run_dir
        self.log_path = os.path.join(run_dir, "claude_log.jsonl")
        self.resume = resume
        self.trend_window = trend_window
        # Rolling window for alert detection
        self._recent: deque = deque(maxlen=trend_window)

    def write_header(self, args, config: dict, start_iter: int):
        """Write header/resume line as first entry."""
        cli_args = {}
        for key in ['lr', 'train_batch', 'epochs', 'simulations', 'c_explore',
                     'risk_beta', 'temperature_moves', 'dirichlet_alpha',
                     'dirichlet_epsilon', 'fpu_base', 'games_per_iter', 'workers',
                     'filters', 'blocks', 'iterations', 'eval_batch',
                     'buffer_capacity']:
            if hasattr(args, key):
                cli_args[key] = getattr(args, key)

        header = {
            "type": "resume" if self.resume else "header",
            "run_dir": self.run_dir,
            "model": {
                "filters": config.get("num_filters", 0),
                "blocks": config.get("num_blocks", 0),
            },
            "total_iterations": getattr(args, 'iterations', 0),
            "start_iteration": start_iter,
            "timestamp": datetime.now().isoformat(),
            "cli_args": cli_args,
        }
        if self.resume:
            header["resumed_from_iteration"] = start_iter

        # Resume appends, new run overwrites
        mode = "a" if self.resume else "w"
        with open(self.log_path, mode) as f:
            f.write(json.dumps(header, separators=(",", ":")) + "\n")

    def log_iteration(self, iteration: int, total_iterations: int,
                      params: dict, metrics, sample_game=None):
        """Append one iteration line to the JSONL log."""
        # Build sample game summary
        sg = None
        if sample_game is not None and sample_game.get("has_game", False):
            moves = sample_game.get("moves", [])
            if isinstance(moves, list):
                move_str = " ".join(moves)
            else:
                move_str = str(moves)
            sg = {
                "moves": move_str,
                "result": sample_game.get("result", "?"),
                "reason": sample_game.get("result_reason", ""),
                "length": sample_game.get("num_moves", 0),
            }

        total_games = metrics.white_wins + metrics.black_wins + metrics.draws

        entry = {
            "type": "iteration",
            "iteration": iteration,
            "total_iterations": total_iterations,
            "timestamp": datetime.now().isoformat(),
            # Game results
            "games": total_games,
            "white_wins": metrics.white_wins,
            "black_wins": metrics.black_wins,
            "draws": metrics.draws,
            # Draw breakdown
            "draws_repetition": metrics.draws_repetition,
            "draws_early_repetition": metrics.draws_early_repetition,
            "draws_stalemate": metrics.draws_stalemate,
            "draws_fifty_move": metrics.draws_fifty_move,
            "draws_insufficient": metrics.draws_insufficient,
            "draws_max_moves": metrics.draws_max_moves,
            # Game quality
            "avg_game_length": round(metrics.avg_game_length, 1),
            "total_moves": metrics.total_moves,
            # Training
            "loss": _safe_round(metrics.loss, 4),
            "policy_loss": _safe_round(metrics.policy_loss, 4),
            "value_loss": _safe_round(metrics.value_loss, 4),
            "grad_norm_avg": _safe_round(metrics.grad_norm_avg, 2),
            "grad_norm_max": _safe_round(metrics.grad_norm_max, 2),
            # Timing
            "selfplay_time": round(metrics.selfplay_time, 1),
            "train_time": round(metrics.train_time, 1),
            "total_time": round(metrics.total_time, 1),
            # Buffer
            "buffer_size": metrics.buffer_size,
            # Current params
            "lr": params.get("lr", 0),
            "risk_beta": params.get("risk_beta", 0),
            # Sample game
            "sample_game": sg,
            # Alerts
            "alerts": [],
        }

        # Detect alerts
        self._recent.append(entry)
        alerts = self._detect_alerts(entry)
        entry["alerts"] = alerts

        self._append_jsonl(entry)

    def _detect_alerts(self, current: dict) -> list:
        """Check thresholds and return alert strings."""
        alerts = []
        recent = list(self._recent)

        # NaN detection
        for key in ("loss", "policy_loss", "value_loss"):
            v = current.get(key, 0)
            if v is not None and (math.isnan(v) or math.isinf(v)):
                alerts.append("nan_detected")
                break

        # Loss plateau: < 2% change over last 5 iterations
        if len(recent) >= 5:
            losses = [r["loss"] for r in recent[-5:] if r.get("loss", 0) > 0]
            if len(losses) >= 5:
                max_l, min_l = max(losses), min(losses)
                if max_l > 0 and (max_l - min_l) / max_l < 0.02:
                    alerts.append("loss_plateau")

        # Loss spike: > 20% increase from previous
        if len(recent) >= 2:
            prev_loss = recent[-2].get("loss", 0)
            curr_loss = current.get("loss", 0)
            if prev_loss > 0 and curr_loss > 0:
                if (curr_loss - prev_loss) / prev_loss > 0.20:
                    alerts.append("loss_spike")

        # Gradient spike
        if current.get("grad_norm_max", 0) > 10:
            alerts.append("gradient_spike")

        # High draw rate (last 3 iterations)
        if len(recent) >= 3:
            last3 = recent[-3:]
            total_games = sum(r.get("games", 0) for r in last3)
            total_draws = sum(r.get("draws", 0) for r in last3)
            if total_games > 0 and total_draws / total_games > 0.60:
                alerts.append("high_draw_rate")

            # High repetition rate
            total_rep = sum(r.get("draws_repetition", 0) for r in last3)
            if total_games > 0 and total_rep / total_games > 0.30:
                alerts.append("high_repetition_rate")

            # Short games
            avg_lengths = [r.get("avg_game_length", 100) for r in last3]
            if sum(avg_lengths) / len(avg_lengths) < 40:
                alerts.append("short_games")

        return alerts

    def _append_jsonl(self, data: dict):
        """Append one JSON line to the log file."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(data, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _get_recent_alerts(self):
        """Return alerts from the most recent iteration."""
        if not self._recent:
            return []
        return list(self._recent)[-1].get("alerts", [])

    def _get_loss_trend(self, n=5):
        """Return last N loss values for trend analysis."""
        recent = list(self._recent)[-n:]
        return [r.get("loss", 0) for r in recent]

    def _get_draw_rate_trend(self, n=5):
        """Return last N draw rates for trend analysis."""
        recent = list(self._recent)[-n:]
        return [round(r.get("draws", 0) / max(r.get("games", 1), 1), 3) for r in recent]

    def write_resume_summary(self, args, iteration: int, total_iterations: int,
                             buffer_stats: dict, shutdown_reason: str):
        """Write claude_resume.json with full context for restart.

        This file gives Claude Code all the context it needs to pick up
        where it left off: effective hyperparameters (after any hot-reloads),
        buffer state, recent alerts, and loss/draw trends.
        """
        summary = {
            "shutdown_reason": shutdown_reason,
            "shutdown_iteration": iteration,
            "total_iterations": total_iterations,
            "shutdown_timestamp": datetime.now().isoformat(),
            "effective_params": {
                key: getattr(args, key) for key in [
                    'lr', 'train_batch', 'epochs', 'simulations', 'c_explore',
                    'risk_beta', 'temperature_moves', 'dirichlet_alpha',
                    'dirichlet_epsilon', 'fpu_base', 'opponent_risk_min',
                    'opponent_risk_max', 'games_per_iter', 'max_fillup_factor',
                    'save_interval',
                ] if hasattr(args, key)
            },
            "buffer_stats": buffer_stats,
            "recent_alerts": self._get_recent_alerts(),
            "loss_trend_last_5": self._get_loss_trend(5),
            "draw_rate_trend_last_5": self._get_draw_rate_trend(5),
        }
        path = os.path.join(self.run_dir, "claude_resume.json")
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        return path


def _safe_round(v, n):
    """Round a value, returning 0 for NaN/None."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return 0.0
    return round(v, n)
