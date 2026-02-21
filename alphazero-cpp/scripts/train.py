#!/usr/bin/env python3
"""
AlphaZero Training with C++ Backend

This script uses:
- C++ MCTS (alphazero-cpp) for fast tree search with proper leaf evaluation
- C++ ReplayBuffer for high-performance data storage
- CUDA for neural network inference and training
- 192x15 network architecture (192 filters, 15 residual blocks)
- 123-channel position encoding

Output Directory Structure:
    Each training run creates an organized directory:
    checkpoints/
    └── f192-b15_2024-02-03_14-30-00/    # {filters}-{blocks}_{timestamp}
        ├── model_iter_001.pt            # Checkpoints (every N iterations)
        ├── model_iter_005.pt
        ├── model_final.pt               # Final checkpoint
        ├── training_log.jsonl           # Append-only JSONL: config + per-iteration metrics
        ├── summary.html                 # Training summary (default, unless --no-visualization)
        └── evaluation_results.json      # Evaluation metrics per checkpoint

Usage:
    # Basic training (recommended starting point)
    uv run python alphazero-cpp/scripts/train.py

    # Custom parameters
    uv run python alphazero-cpp/scripts/train.py --iterations 50 --games-per-iter 100 --simulations 800

    # Resume from checkpoint (continues in same run directory)
    uv run python alphazero-cpp/scripts/train.py --resume checkpoints/f192-b15_2024-02-03_14-30-00/model_iter_005.pt

    # Resume from run directory (finds latest checkpoint)
    uv run python alphazero-cpp/scripts/train.py --resume checkpoints/f192-b15_2024-02-03_14-30-00

Parameters:
    --iterations        Number of training iterations (default: 100)
    --games-per-iter    Self-play games per iteration (default: 50)
    --simulations       MCTS simulations per move (default: 800)
    --search-algorithm  Root search: gumbel (default) or puct
    --c-explore         MCTS exploration constant (default: 1.5)
    --train-batch       Samples per training gradient step (default: 256)
    --lr                Learning rate (default: 0.001)
    --filters           Network filters (default: 192)
    --blocks            Residual blocks (default: 15)
    --buffer-size       Replay buffer size (default: 100000)
    --epochs            Training epochs per iteration (default: 5)
    --temperature-moves Moves with temperature=1 (default: 30)
    --device            Device: cuda or cpu (default: cuda)
    --save-dir          Base checkpoint directory (default: checkpoints)
    --resume            Resume from checkpoint path or run directory
    --save-interval     Save checkpoint every N iterations (default: 1)
    --no-visualization  Disable summary.html generation (default: enabled)

    Parallel Self-Play (enabled automatically when --workers > 1):
    --workers              Self-play workers. 1=sequential, >1=parallel (default: 1)
    --gpu-batch-timeout-ms GPU batch collection timeout in ms (default: 20)
    --worker-timeout-ms    Worker wait time for NN results in ms (default: 2000)

    search_batch is auto-derived: gumbel→gumbel_top_k, puct→1.
    eval_batch (GPU-effective) is auto-computed as workers*search_batch*mirror_factor.
    input_batch_size = eval_batch / mirror_factor (used for C++ queue and CUDA graphs).
"""

import argparse
import json
import math
import os
import sys
import time
import random
import signal
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta

# Force line-buffered UTF-8 output (prevents silent buffering when piped through tee)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
elif sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
import chess
import chess.pgn

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "build" / "Release"))

try:
    import alphazero_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("ERROR: alphazero_cpp not found. Build it first:")
    print("  cd alphazero-cpp")
    print("  cmake -B build -DCMAKE_BUILD_TYPE=Release")
    print("  cmake --build build --config Release")
    sys.exit(1)

import chess

from generate_summary import generate_summary_html
from network import AlphaZeroNet, INPUT_CHANNELS, POLICY_SIZE


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Only shows non-zero units:
    - 15 seconds -> "15 sec"
    - 59 min 15 sec -> "59 min 15 sec"
    - 2 hr 30 min -> "2 hr 30 min"
    - 1 hr 0 min 5 sec -> "1 hr 5 sec"
    """
    if seconds < 0:
        return "0 sec"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours} hr")
    if minutes > 0:
        parts.append(f"{minutes} min")
    if secs > 0 or not parts:  # Always show seconds if nothing else
        parts.append(f"{secs} sec")

    return " ".join(parts)


# =============================================================================
# Graceful Shutdown Handler
# =============================================================================

class GracefulShutdown:
    """Handles graceful shutdown on Ctrl+C (SIGINT) or stop file.

    When shutdown is requested:
    1. Finishes current game (if in self-play)
    2. Saves the replay buffer
    3. Saves an emergency checkpoint
    4. Exits cleanly
    """

    def __init__(self):
        self.shutdown_requested = False
        self._lock = threading.Lock()
        self._original_handler = None
        self.stop_file_path = None  # Set after run_dir is known

    def request_shutdown(self, signum=None, frame=None, source="sigint"):
        """Request graceful shutdown."""
        with self._lock:
            if not self.shutdown_requested:
                self.shutdown_requested = True
                if source == "sigint":
                    print("\n\n" + "!" * 70)
                    print("! SHUTDOWN REQUESTED - Finishing current game and saving...")
                    print("! Press Ctrl+C again to force quit (may lose data)")
                    print("!" * 70 + "\n")
                elif source == "stop_file":
                    print("\n" + "!" * 70)
                    print("! [LLM] Stop file detected — stopping immediately")
                    print("! Press Ctrl+C to force quit (may lose data)")
                    print("!" * 70 + "\n")

                # Restore original handler for force quit
                if self._original_handler:
                    signal.signal(signal.SIGINT, self._original_handler)

    def should_stop(self) -> bool:
        """Check if shutdown was requested (SIGINT or stop file)."""
        with self._lock:
            if self.shutdown_requested:
                return True
        # Check stop file (if configured)
        if self.stop_file_path and os.path.exists(self.stop_file_path):
            try:
                os.remove(self.stop_file_path)
            except OSError:
                pass
            self.request_shutdown(source="stop_file")
            return True
        return False

    def install_handler(self):
        """Install the graceful shutdown signal handler."""
        self._original_handler = signal.signal(signal.SIGINT, self.request_shutdown)

    def uninstall_handler(self):
        """Restore the original signal handler."""
        if self._original_handler:
            signal.signal(signal.SIGINT, self._original_handler)


# Global shutdown handler
shutdown_handler = GracefulShutdown()


# =============================================================================
# Periodic Progress Reporter
# =============================================================================

class ProgressReporter:
    """Reports progress statistics at regular intervals during self-play.

    Prints performance metrics every `interval` seconds without interrupting
    the training loop.
    """

    def __init__(self, interval: float = 30.0):
        self.interval = interval
        self.reset()

    def reset(self):
        """Reset for a new iteration."""
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.games_completed = 0
        self.total_moves = 0
        self.total_sims = 0
        self.total_evals = 0
        self.last_games = 0
        self.last_moves = 0
        self.last_sims = 0
        self.last_evals = 0

    def update(self, moves: int, sims: int, evals: int):
        """Update counters after a game completes."""
        self.games_completed += 1
        self.total_moves += moves
        self.total_sims += sims
        self.total_evals += evals

    def should_report(self) -> bool:
        """Check if it's time for a progress report."""
        return time.time() - self.last_report_time >= self.interval

    def report(self, total_games: int, buffer_size: int):
        """Print progress report with performance statistics."""
        now = time.time()
        elapsed_total = now - self.start_time
        elapsed_interval = now - self.last_report_time

        # Calculate interval rates (recent performance)
        interval_games = self.games_completed - self.last_games
        interval_moves = self.total_moves - self.last_moves
        interval_sims = self.total_sims - self.last_sims
        interval_evals = self.total_evals - self.last_evals

        # Calculate rates
        moves_per_sec = interval_moves / elapsed_interval if elapsed_interval > 0 else 0
        sims_per_sec = interval_sims / elapsed_interval if elapsed_interval > 0 else 0
        evals_per_sec = interval_evals / elapsed_interval if elapsed_interval > 0 else 0
        games_per_hour = interval_games / elapsed_interval * 3600 if elapsed_interval > 0 else 0

        # Calculate overall rates
        overall_moves_per_sec = self.total_moves / elapsed_total if elapsed_total > 0 else 0

        # ETA calculation
        remaining_games = total_games - self.games_completed
        if interval_games > 0:
            eta_seconds = remaining_games / (interval_games / elapsed_interval)
            eta_str = format_duration(eta_seconds)
        else:
            eta_str = "calculating..."

        # Print compact progress line
        print(f"    ⏱ {format_duration(elapsed_total)} | "
              f"Games: {self.games_completed}/{total_games} | "
              f"Moves: {moves_per_sec:.1f}/s | "
              f"Sims: {sims_per_sec:,.0f}/s | "
              f"NN: {evals_per_sec:,.0f}/s | "
              f"Buffer: {buffer_size:,} | "
              f"ETA: {eta_str}")

        # Update last values for next interval
        self.last_report_time = now
        self.last_games = self.games_completed
        self.last_moves = self.total_moves
        self.last_sims = self.total_sims
        self.last_evals = self.total_evals


# =============================================================================
# Refutation Elo
# =============================================================================

MIN_REFUTATION_SAMPLES = 20  # Minimum games before Elo is statistically meaningful

def compute_refutation_elo(standard_wins, opponent_wins, draws):
    """Compute Refutation Elo from per-persona outcomes.

    Returns Elo difference (positive = standard is stronger).
    Returns None if insufficient data (< MIN_REFUTATION_SAMPLES).
    """
    total = standard_wins + opponent_wins + draws
    if total < MIN_REFUTATION_SAMPLES:
        return None
    score = (standard_wins + 0.5 * draws) / total
    score = max(0.01, min(0.99, score))  # clamp to avoid log(0)
    return -400 * math.log10((1 - score) / score)


# =============================================================================
# Performance Metrics Tracker
# =============================================================================

@dataclass
class IterationMetrics:
    """Metrics for a training iteration."""
    iteration: int
    # Self-play metrics
    num_games: int = 0
    total_moves: int = 0
    selfplay_time: float = 0.0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    # Draw reason breakdown
    draws_repetition: int = 0
    draws_stalemate: int = 0
    draws_fifty_move: int = 0
    draws_insufficient: int = 0
    draws_max_moves: int = 0
    draws_early_repetition: int = 0
    # Per-persona outcome tracking (asymmetric risk games only)
    standard_wins: int = 0
    opponent_wins: int = 0
    asymmetric_draws: int = 0
    avg_game_length: float = 0.0
    total_simulations: int = 0
    total_nn_evals: int = 0
    # Training metrics
    train_time: float = 0.0
    loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    num_train_batches: int = 0
    risk_beta: float = 0.0
    grad_norm_avg: float = 0.0
    grad_norm_max: float = 0.0
    # PER metrics
    per_beta: float = 0.0
    # Buffer metrics
    buffer_size: int = 0
    # Reanalysis metrics
    reanalysis_positions: int = 0
    reanalysis_time_s: float = 0.0
    reanalysis_mean_kl: float = 0.0
    # Timing
    total_time: float = 0.0


class MetricsTracker:
    """Track and display training metrics."""

    def __init__(self):
        self.history: List[IterationMetrics] = []
        self.start_time = time.time()
        self.total_games = 0
        self.total_moves = 0
        self.total_simulations = 0
        self.total_nn_evals = 0

    def add_iteration(self, metrics: IterationMetrics):
        self.history.append(metrics)
        self.total_games += metrics.num_games
        self.total_moves += metrics.total_moves
        self.total_simulations += metrics.total_simulations
        self.total_nn_evals += metrics.total_nn_evals

    def get_summary(self) -> Dict:
        """Get overall training summary."""
        elapsed = time.time() - self.start_time
        return {
            'total_iterations': len(self.history),
            'total_games': self.total_games,
            'total_moves': self.total_moves,
            'total_simulations': self.total_simulations,
            'total_nn_evals': self.total_nn_evals,
            'elapsed_time': elapsed,
            'games_per_hour': self.total_games / elapsed * 3600 if elapsed > 0 else 0,
            'moves_per_sec': self.total_moves / elapsed if elapsed > 0 else 0,
            'sims_per_sec': self.total_simulations / elapsed if elapsed > 0 else 0,
        }

    def print_iteration_summary(self, m: IterationMetrics, args):
        """Print detailed iteration summary."""
        print(f"\n  {'─' * 66}")
        print(f"  ITERATION {m.iteration} SUMMARY")
        print(f"  {'─' * 66}")

        # Self-play metrics
        moves_per_sec = m.total_moves / m.selfplay_time if m.selfplay_time > 0 else 0
        sims_per_sec = m.total_simulations / m.selfplay_time if m.selfplay_time > 0 else 0
        nn_evals_per_sec = m.total_nn_evals / m.selfplay_time if m.selfplay_time > 0 else 0

        print(f"  Self-Play:")
        print(f"    Games:           {m.num_games} ({m.white_wins}W / {m.draws}D / {m.black_wins}L)")
        total_g = m.white_wins + m.black_wins + m.draws
        if total_g > 0:
            print(f"    Rates:           W={m.white_wins/total_g*100:.0f}% "
                  f"D={m.draws/total_g*100:.0f}% L={m.black_wins/total_g*100:.0f}%")
        if m.draws > 0:
            parts = []
            if m.draws_repetition > 0:
                early_str = f" ({m.draws_early_repetition} early)" if m.draws_early_repetition > 0 else ""
                parts.append(f"{m.draws_repetition} repetition{early_str}")
            if m.draws_stalemate > 0:
                parts.append(f"{m.draws_stalemate} stalemate")
            if m.draws_fifty_move > 0:
                parts.append(f"{m.draws_fifty_move} fifty-move")
            if m.draws_insufficient > 0:
                parts.append(f"{m.draws_insufficient} material")
            if m.draws_max_moves > 0:
                parts.append(f"{m.draws_max_moves} max-moves")
            if parts:
                print(f"    Draw breakdown:  {', '.join(parts)}")
        total_asym = m.standard_wins + m.opponent_wins + m.asymmetric_draws
        if total_asym > 0:
            refutation_elo = compute_refutation_elo(m.standard_wins, m.opponent_wins, m.asymmetric_draws)
            if refutation_elo is not None:
                print(f"    Refutation:      Elo {refutation_elo:+.0f} "
                      f"({m.standard_wins}S / {m.asymmetric_draws}D / {m.opponent_wins}O, n={total_asym})")
            else:
                print(f"    Refutation:      {m.standard_wins}S / {m.asymmetric_draws}D / {m.opponent_wins}O "
                      f"(low sample, n={total_asym})")
        print(f"    Moves:           {m.total_moves} total, {m.avg_game_length:.1f} avg/game")
        print(f"    Time:            {format_duration(m.selfplay_time)} ({moves_per_sec:.1f} moves/sec)")
        print(f"    MCTS Sims:       {m.total_simulations:,} ({sims_per_sec:,.0f}/sec)")
        print(f"    NN Evals:        {m.total_nn_evals:,} ({nn_evals_per_sec:,.0f}/sec)")

        # Training metrics
        if m.train_time > 0:
            samples_per_sec = (m.num_train_batches * args.train_batch) / m.train_time
            sm_str = f", risk_β={m.risk_beta:.2f}" if m.risk_beta != 0 else ""
            print(f"  Training:")
            print(f"    Loss:            {m.loss:.4f} (policy={m.policy_loss:.4f}, value={m.value_loss:.4f}{sm_str})")
            if m.grad_norm_avg > 0:
                print(f"    Grad Norm:       avg={m.grad_norm_avg:.2f}, max={m.grad_norm_max:.2f}")
            print(f"    Batches:         {m.num_train_batches} ({samples_per_sec:.0f} samples/sec)")
            print(f"    Time:            {format_duration(m.train_time)}")

        # Buffer and timing
        print(f"  Buffer:            {m.buffer_size:,} positions")
        print(f"  Iteration Time:    {format_duration(m.total_time)}")

        # ETA calculation
        elapsed = time.time() - self.start_time
        iters_done = len(self.history)
        if iters_done > 0:
            avg_iter_time = elapsed / iters_done
            remaining_iters = args.iterations - m.iteration
            eta_seconds = avg_iter_time * remaining_iters
            print(f"  ETA:               {format_duration(eta_seconds)} ({remaining_iters} iterations remaining)")

        print(f"  {'─' * 66}\n")

    def print_final_summary(self, args):
        """Print final training summary."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE - FINAL SUMMARY")
        print("=" * 70)

        print(f"\n  Duration:          {format_duration(summary['elapsed_time'])}")
        print(f"  Iterations:        {summary['total_iterations']}")
        print(f"  Games Played:      {summary['total_games']:,}")
        print(f"  Total Moves:       {summary['total_moves']:,}")
        print(f"  Total MCTS Sims:   {summary['total_simulations']:,}")
        print(f"  Total NN Evals:    {summary['total_nn_evals']:,}")

        print(f"\n  Performance:")
        print(f"    Games/hour:      {summary['games_per_hour']:.1f}")
        print(f"    Moves/sec:       {summary['moves_per_sec']:.1f}")
        print(f"    Sims/sec:        {summary['sims_per_sec']:,.0f}")

        # Loss progression
        if len(self.history) >= 2:
            first_loss = self.history[0].loss
            last_loss = self.history[-1].loss
            if first_loss > 0:
                print(f"\n  Loss Progression:")
                print(f"    First iteration: {first_loss:.4f}")
                print(f"    Last iteration:  {last_loss:.4f}")
                print(f"    Improvement:     {(first_loss - last_loss) / first_loss * 100:.1f}%")

        # Win rate progression
        total_w = sum(m.white_wins for m in self.history)
        total_d = sum(m.draws for m in self.history)
        total_b = sum(m.black_wins for m in self.history)
        total = total_w + total_d + total_b
        if total > 0:
            print(f"\n  Game Results:")
            print(f"    White wins:      {total_w} ({total_w/total*100:.1f}%)")
            print(f"    Draws:           {total_d} ({total_d/total*100:.1f}%)")
            print(f"    Black wins:      {total_b} ({total_b/total*100:.1f}%)")

        print("\n" + "=" * 70)


# =============================================================================
# Run Directory and Resume Helpers
# =============================================================================

def create_run_directory(base_dir: str, filters: int, blocks: int, se_reduction: int = 16) -> str:
    """Create a new organized run directory with timestamp.

    Args:
        base_dir: Base checkpoint directory (e.g., "checkpoints")
        filters: Number of network filters
        blocks: Number of residual blocks
        se_reduction: SE block reduction ratio

    Returns:
        Path to the created run directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"f{filters}-b{blocks}-se{se_reduction}_{timestamp}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def find_latest_checkpoint(run_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a run directory.

    Args:
        run_dir: Path to the run directory

    Returns:
        Path to the latest checkpoint, or None if not found
    """
    import glob

    # Look for model_iter_*.pt files
    pattern = os.path.join(run_dir, "model_iter_*.pt")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        # Also check for old-style cpp_iter_*.pt files
        pattern = os.path.join(run_dir, "cpp_iter_*.pt")
        checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None

    # Sort by (iteration_number, is_regular) so non-emergency wins at same iter
    def extract_iter(path):
        name = os.path.basename(path)
        is_emergency = '_emergency' in name
        # Handle both model_iter_001.pt and cpp_iter_1.pt formats
        try:
            if "model_iter_" in name:
                iter_num = int(name.replace("model_iter_", "").replace(".pt", "").replace("_emergency", ""))
            elif "cpp_iter_" in name:
                iter_num = int(name.replace("cpp_iter_", "").replace(".pt", "").replace("_emergency", ""))
            else:
                return (0, 0)
            # Regular checkpoints (1) sort after emergency (0), so [-1] picks regular
            return (iter_num, 0 if is_emergency else 1)
        except ValueError:
            return (0, 0)

    checkpoints.sort(key=extract_iter)
    return checkpoints[-1]


def parse_resume_path(resume_arg: str) -> Tuple[str, Optional[str]]:
    """Parse the resume argument to get run directory and checkpoint path.

    Args:
        resume_arg: Either a path to a .pt file or a run directory

    Returns:
        Tuple of (run_dir, checkpoint_path)
    """
    if resume_arg.endswith('.pt'):
        # It's a checkpoint file path
        run_dir = os.path.dirname(resume_arg)
        checkpoint_path = resume_arg
    else:
        # It's a run directory
        run_dir = resume_arg
        checkpoint_path = find_latest_checkpoint(run_dir)

    return run_dir, checkpoint_path


def append_training_log(run_dir: str, record: Dict[str, Any]):
    """Append one JSON record to training_log.jsonl (crash-safe).

    Each call writes a single compact JSON line followed by newline,
    then flushes and fsyncs so the record survives process crashes.
    """
    log_path = os.path.join(run_dir, "training_log.jsonl")
    with open(log_path, 'a') as f:
        f.write(json.dumps(record, separators=(',', ':')) + '\n')
        f.flush()
        os.fsync(f.fileno())


def load_training_log(run_dir: str) -> tuple:
    """Load training log from JSONL (or legacy JSON fallback).

    Returns:
        (config_record, iteration_list) — config may be {} if absent.
    """
    log_path = os.path.join(run_dir, "training_log.jsonl")
    if os.path.exists(log_path):
        config_record = {}
        iterations = []
        with open(log_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("type") == "config":
                        config_record = rec
                    elif rec.get("type") == "iteration":
                        iterations.append(rec)
                except (json.JSONDecodeError, ValueError):
                    print(f"  Warning: Skipping malformed line {line_num} in training_log.jsonl")
        return config_record, iterations

    # Fallback: read legacy training_metrics.json
    legacy_path = os.path.join(run_dir, "training_metrics.json")
    if os.path.exists(legacy_path):
        try:
            with open(legacy_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    config_record = {"type": "config"}
                    config_record.update(data.get("config", {}))
                    iterations = data.get("iterations", [])
                    # Tag legacy records with type for consistency
                    for rec in iterations:
                        rec.setdefault("type", "iteration")
                    return config_record, iterations
        except (json.JSONDecodeError, ValueError):
            print(f"  Warning: Could not parse {legacy_path}")

    return {}, []


def truncate_training_log(run_dir: str, start_iter: int) -> int:
    """Remove iteration records >= start_iter from training_log.jsonl.

    Rewrites via temp file + os.replace() for atomicity.
    Preserves the config record.

    Returns:
        Number of iteration records removed.
    """
    log_path = os.path.join(run_dir, "training_log.jsonl")
    if not os.path.exists(log_path):
        return 0

    kept_lines = []
    removed = 0
    with open(log_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
                if rec.get("type") == "iteration" and rec.get("iteration", 0) >= start_iter:
                    removed += 1
                    continue
            except (json.JSONDecodeError, ValueError):
                pass  # Keep malformed lines (don't silently discard)
            kept_lines.append(stripped + '\n')

    if removed > 0:
        tmp_path = log_path + ".tmp"
        with open(tmp_path, 'w') as f:
            f.writelines(kept_lines)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, log_path)

    return removed


def write_config_record(run_dir: str, config: Dict[str, Any]):
    """Write a config record as the first line of training_log.jsonl (new runs only)."""
    record = {"type": "config"}
    record.update(config)
    append_training_log(run_dir, record)


def safe_torch_save(obj: dict, path: str):
    """Save a PyTorch checkpoint atomically via temp file + rename.

    OneDrive (and similar cloud-sync tools) can corrupt torch.save by
    reading/locking the file mid-write. Writing to a .tmp file first
    and then renaming avoids this race condition.
    """
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    # On Windows, os.replace is atomic if src and dst are on the same volume
    os.replace(tmp_path, path)


# =============================================================================
# Batched Evaluator (GPU)
# =============================================================================

class BatchedEvaluator:
    """Efficient batched neural network evaluation on GPU."""

    def __init__(self, network: nn.Module, device: str, use_amp: bool = True):
        self.network = network
        self.device = device
        self.use_amp = use_amp and device == "cuda"

    @torch.inference_mode()
    def evaluate(self, obs: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Evaluate single position. obs is NHWC (8, 8, 123)."""
        # permute: (8,8,123) → unsqueeze → (1,8,8,123) → permute → (1,123,8,8) channels_last
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.device)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)

        if self.use_amp:
            with autocast('cuda'):
                policy, value, _, _ = self.network(obs_tensor, mask_tensor)
        else:
            policy, value, _, _ = self.network(obs_tensor, mask_tensor)

        return policy[0].cpu().numpy(), float(value[0].item())

    @torch.inference_mode()
    def evaluate_batch(self, obs_batch: np.ndarray, mask_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate batch of positions. obs_batch is NHWC (N, 8, 8, 123)."""
        obs_tensor = torch.from_numpy(obs_batch).permute(0, 3, 1, 2).float().to(self.device)
        mask_tensor = torch.from_numpy(mask_batch).float().to(self.device)

        if self.use_amp:
            with autocast('cuda'):
                policies, values, _, _ = self.network(obs_tensor, mask_tensor)
        else:
            policies, values, _, _ = self.network(obs_tensor, mask_tensor)

        return policies.cpu().numpy(), values.squeeze(-1).cpu().numpy()


# =============================================================================
# Self-Play with C++ MCTS
# =============================================================================

class CppSelfPlay:
    """Self-play using C++ MCTS backend."""

    def __init__(
        self,
        evaluator: BatchedEvaluator,
        num_simulations: int = 800,
        mcts_batch_size: int = 64,
        c_explore: float = 1.5,
        temperature_moves: int = 30,
    ):
        self.evaluator = evaluator
        self.num_simulations = num_simulations
        self.mcts_batch_size = mcts_batch_size
        self.c_explore = c_explore
        self.temperature_moves = temperature_moves

        # Create C++ MCTS engine (BatchedMCTSSearch is the Python binding name)
        self.mcts = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=num_simulations,
            batch_size=mcts_batch_size,
            c_puct=c_explore  # C++ binding kwarg stays c_puct
        )

    def play_game(self) -> Tuple[List[np.ndarray], List[np.ndarray], float, int, int, int, List[str], str]:
        """Play a single self-play game.

        Returns:
            observations: List of board observations (8, 8, 123) NHWC format
            policies: List of MCTS policies
            result: Game result (1=white wins, -1=black wins, 0=draw)
            num_moves: Number of moves played
            total_sims: Total MCTS simulations
            total_evals: Total NN evaluations
            moves_uci: List of UCI move strings
            result_reason: Draw reason string (repetition/stalemate/fifty_move/insufficient/max_moves/checkmate/"")
        """
        board = chess.Board()
        observations = []
        policies = []
        move_count = 0
        total_sims = 0
        total_evals = 0

        while not board.is_game_over() and move_count < 10000:
            fen = board.fen()

            # Encode position (returns 8, 8, 123 in NHWC format)
            obs = alphazero_cpp.encode_position(fen)

            # Get legal moves and build mask + mapping
            legal_moves = list(board.legal_moves)
            mask = np.zeros(POLICY_SIZE, dtype=np.float32)
            move_to_idx = {}
            idx_to_move = {}

            for move in legal_moves:
                idx = alphazero_cpp.move_to_index(move.uci(), fen)
                if 0 <= idx < POLICY_SIZE:
                    mask[idx] = 1.0
                    move_to_idx[move.uci()] = idx
                    idx_to_move[idx] = move

            # Get root evaluation
            root_policy, root_value = self.evaluator.evaluate(obs, mask)
            total_evals += 1

            # Initialize MCTS search
            self.mcts.init_search(fen, root_policy.astype(np.float32), float(root_value))

            # Run MCTS with batched leaf evaluation
            while not self.mcts.is_complete():
                num_leaves, obs_batch, mask_batch = self.mcts.collect_leaves()
                if num_leaves == 0:
                    break

                masks = mask_batch[:num_leaves]

                # Batch evaluate (evaluator handles NHWC → permute internally)
                leaf_policies, leaf_values = self.evaluator.evaluate_batch(obs_batch[:num_leaves], masks)
                total_evals += num_leaves

                # Update leaves
                self.mcts.update_leaves(
                    leaf_policies.astype(np.float32),
                    leaf_values.astype(np.float32)
                )

            total_sims += self.mcts.get_simulations_completed()

            # Get visit counts as policy
            visit_counts = self.mcts.get_visit_counts()
            policy = visit_counts.astype(np.float32)
            policy = policy * mask
            if policy.sum() > 0:
                policy = policy / policy.sum()
            else:
                policy = mask / mask.sum()

            # Store for training
            observations.append(obs.copy())
            policies.append(policy.copy())

            # Select move with temperature
            if move_count < self.temperature_moves:
                action = np.random.choice(POLICY_SIZE, p=policy)
            else:
                action = np.argmax(policy)

            # Get move from mapping
            if action in idx_to_move:
                move = idx_to_move[action]
            else:
                best_idx = max(idx_to_move.keys(), key=lambda i: policy[i])
                move = idx_to_move[best_idx]

            board.push(move)
            move_count += 1
            self.mcts.reset()

        # Get game result
        result = board.result()
        if result == "1-0":
            value = 1.0
        elif result == "0-1":
            value = -1.0
        else:
            value = 0.0  # Pure training label (risk_beta is search-time only)

        # Determine result reason for draw breakdown tracking
        if value == 0.0:
            if move_count >= 10000:
                result_reason = "max_moves"
            elif board.is_repetition():
                result_reason = "repetition"
            elif board.is_stalemate():
                result_reason = "stalemate"
            elif board.is_fifty_moves():
                result_reason = "fifty_move"
            elif board.is_insufficient_material():
                result_reason = "insufficient"
            else:
                result_reason = "max_moves"  # fallback
        elif board.is_checkmate():
            result_reason = "checkmate"
        else:
            result_reason = ""

        # Extract UCI move history
        moves_uci = [m.uci() for m in board.move_stack]

        return observations, policies, value, move_count, total_sims, total_evals, moves_uci, result_reason


# =============================================================================
# Sample Game PGN Export
# =============================================================================

def save_sample_game_pgn(run_dir: str, iteration: int, moves_uci: list,
                          result_str: str, num_moves: int) -> Optional[str]:
    """Save a sample self-play game as PGN file.

    Args:
        run_dir: Training run directory
        iteration: Current iteration number (1-indexed)
        moves_uci: List of UCI move strings (e.g. ["e2e4", "e7e5", ...])
        result_str: PGN result string ("1-0", "0-1", "1/2-1/2")
        num_moves: Total number of moves

    Returns:
        Path to saved PGN file, or None if save failed
    """
    try:
        # Create sample_games directory
        pgn_dir = os.path.join(run_dir, "sample_games")
        os.makedirs(pgn_dir, exist_ok=True)

        # Build PGN game from UCI moves
        game = chess.pgn.Game()
        game.headers["Event"] = "AlphaZero Self-Play"
        game.headers["Site"] = "Training"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = str(iteration)
        game.headers["White"] = "AlphaZero"
        game.headers["Black"] = "AlphaZero"
        game.headers["Result"] = result_str

        # Replay moves on a board to build the PGN game tree
        node = game
        board = chess.Board()
        for uci_str in moves_uci:
            move = chess.Move.from_uci(uci_str)
            if move in board.legal_moves:
                node = node.add_variation(move)
                board.push(move)
            else:
                # Shouldn't happen, but be defensive
                break

        # Save to file
        pgn_path = os.path.join(pgn_dir, f"iter_{iteration:04d}.pgn")
        with open(pgn_path, "w") as f:
            print(game, file=f)

        print(f"  Sample game: {result_str} in {num_moves} moves -> {pgn_path}")
        return pgn_path

    except Exception as e:
        print(f"  Warning: Failed to save sample game PGN: {e}")
        return None


# =============================================================================
# Parallel Self-Play with Cross-Game Batching
# =============================================================================

def run_parallel_selfplay_with_interrupt(
    coordinator,
    evaluator,
    shutdown_handler: GracefulShutdown,
    progress_callback=None,
    progress_interval: float = 5.0
) -> dict:
    """Run self-play in a thread, allowing Ctrl+C to interrupt.

    This wrapper runs generate_games() in a background thread while the main
    thread monitors for shutdown signals and pushes periodic progress updates.
    When Ctrl+C is detected, it calls coordinator.stop() to gracefully
    terminate C++ execution.

    Args:
        coordinator: ParallelSelfPlayCoordinator instance
        evaluator: Neural network evaluator callback
        shutdown_handler: GracefulShutdown instance to check for Ctrl+C
        progress_callback: Optional callable(live_stats_dict) called every
            progress_interval seconds with live stats from C++ atomics
        progress_interval: Seconds between progress updates (default 5.0)

    Returns:
        Result dict from generate_games() or partial results if interrupted
    """
    result = [None]
    exception = [None]

    def worker():
        try:
            result[0] = coordinator.generate_games(evaluator)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    last_progress_time = time.time()

    # Wait with periodic checks for shutdown and progress updates
    while thread.is_alive():
        thread.join(timeout=0.5)

        # Push periodic progress updates by polling C++ atomic stats
        now = time.time()
        if progress_callback and (now - last_progress_time) >= progress_interval:
            try:
                live_stats = coordinator.get_live_stats()
                if live_stats:
                    progress_callback(live_stats)
            except Exception:
                pass  # Don't crash on monitoring errors
            last_progress_time = now

        if shutdown_handler.should_stop():
            print("\n  Stopping self-play (Ctrl+C detected)...")
            coordinator.stop()  # Signal C++ to stop
            thread.join(timeout=5.0)  # Wait up to 5s for graceful stop
            if thread.is_alive():
                print("  Warning: Self-play thread did not stop gracefully")
            break

    if exception[0]:
        raise exception[0]

    return result[0]


def run_selfplay_with_reanalysis(
    coordinator,
    evaluator,
    reanalyzer,
    shutdown_handler,
    progress_callback=None,
    progress_interval: float = 5.0
):
    """Run self-play + reanalysis concurrently using coordinator lifecycle split.

    Self-play workers and reanalysis workers share the same GPU thread.
    The GPU thread fills spare batch capacity from the reanalysis queue.
    After self-play finishes, the GPU thread continues serving reanalysis only.

    Returns:
        (selfplay_result, reanalysis_stats, reanalysis_tail_time)
    """
    # Start generation (non-blocking) and connect reanalysis queue
    coordinator.start_generation(evaluator)
    coordinator.set_secondary_queue(reanalyzer)
    reanalyzer.start()

    # Monitor progress: wait for self-play in a thread so main thread
    # can handle Ctrl+C and push progress updates
    selfplay_done = threading.Event()
    selfplay_exception = [None]

    def selfplay_waiter():
        try:
            coordinator.wait_for_workers()
        except Exception as e:
            selfplay_exception[0] = e
        selfplay_done.set()

    thread = threading.Thread(target=selfplay_waiter, daemon=True)
    thread.start()

    last_progress_time = time.time()

    while not selfplay_done.is_set():
        selfplay_done.wait(timeout=0.5)

        now = time.time()
        if progress_callback and (now - last_progress_time) >= progress_interval:
            try:
                live_stats = coordinator.get_live_stats()
                if live_stats:
                    progress_callback(live_stats)
            except Exception:
                pass
            last_progress_time = now

        if shutdown_handler.should_stop():
            print("\n  Stopping (Ctrl+C detected)...")
            coordinator.stop()
            reanalyzer.stop()
            selfplay_done.wait(timeout=5.0)
            break

    if selfplay_exception[0]:
        reanalyzer.stop()
        raise selfplay_exception[0]

    # Self-play done. Wait for reanalysis tail (GPU thread still serving secondary queue)
    reanalysis_tail_start = time.time()
    reanalyzer.wait()
    reanalysis_tail_time = time.time() - reanalysis_tail_start

    # Get stats before shutdown
    selfplay_result = coordinator.get_generation_stats()
    reanalysis_stats = reanalyzer.get_stats()

    # Clean up
    coordinator.clear_secondary_queue()
    coordinator.shutdown_gpu_thread()

    return selfplay_result, reanalysis_stats, reanalysis_tail_time


def calibrate_cuda_graphs(
    network: nn.Module,
    device: str,
    eval_batch: int,
    warmup_iters: int = 5,
    measure_iters: int = 20
) -> dict:
    """
    Measure eager vs CUDA graph performance to determine optimal thresholds.

    The crossover point where CUDA graph beats eager execution depends on:
        Graph wins when: graph_replay_time < eager_overhead + per_sample × batch_size
        Crossover batch: B = (graph_time - eager_overhead) / per_sample

    For a graph of size G with actual batch B (B ≤ G), we pay for G samples but
    only needed B — this is "padding waste". The calibration finds where graph
    replay + padding is still cheaper than eager overhead.

    Returns dict with:
    - mini_graph_size: Size for mini graph (graph always wins at any fill)
    - small_graph_size: Size for small graph (graph needs some fill to win)
    - large_graph_threshold: Minimum batch to use large graph
    - eager_overhead_ms: Fixed eager overhead
    - eager_per_sample_ms: Per-sample eager cost
    - measurements: Raw timing data for debugging
    """
    import statistics

    if device != "cuda":
        mini_graph_size = 64
        small_graph_size = mini_graph_size * 2
        medium_graph_size = max(small_graph_size + 1, eval_batch // 2)
        return {
            'mini_graph_size': mini_graph_size,
            'small_graph_size': small_graph_size,
            'medium_graph_size': medium_graph_size,
            'large_graph_threshold': eval_batch * 7 // 8,
            'mini_threshold': 0,
            'small_threshold': mini_graph_size,
            'medium_threshold': small_graph_size,
            'eager_overhead_ms': 0.0,
            'eager_per_sample_ms': 0.0,
            'measurements': {},
            'skipped': True
        }

    network.eval()
    torch.cuda.synchronize()

    # Test sizes: powers of 2, capped at eval_batch // 2.
    # Eager is only used for batches below the large graph threshold,
    # so testing at full eval_batch wastes calibration time.
    half_batch = eval_batch // 2
    test_sizes = [s for s in [8, 16, 32, 64, 128, 256, 512, 1024] if s <= half_batch]
    if half_batch > 0 and half_batch not in test_sizes:
        test_sizes.append(half_batch)
    test_sizes = sorted(test_sizes)

    print("  GPU warmup...", end=" ", flush=True)
    # Extended warmup to reach thermal steady-state (~2 seconds of sustained load)
    # This is critical: calibration on a "cold" GPU gives optimistic timings
    warmup_obs = torch.zeros(eval_batch, INPUT_CHANNELS, 8, 8, device='cuda'
                             ).to(memory_format=torch.channels_last)
    warmup_mask = torch.zeros(eval_batch, POLICY_SIZE, device='cuda')
    warmup_start = time.time()
    warmup_count = 0
    while time.time() - warmup_start < 2.0:  # 2 seconds of sustained inference
        with torch.no_grad(), autocast('cuda'):
            network(warmup_obs, warmup_mask)
        warmup_count += 1
    torch.cuda.synchronize()
    print(f"done ({warmup_count} iters)")

    # --- Measure eager execution times ---
    # IMPORTANT: Start from numpy to match the real runtime path in neural_evaluator
    # (lines ~1341-1342: torch.from_numpy(arr).float().to(device))
    # This captures the CPU→GPU transfer cost that dominates for large batches.
    print(f"  Measuring eager: ", end="", flush=True)
    eager_times = {}  # size -> list of times in ms

    for size in test_sizes:
        obs_np = np.zeros((size, 8, 8, INPUT_CHANNELS), dtype=np.float32)  # NHWC
        mask_np = np.zeros((size, POLICY_SIZE), dtype=np.float32)

        # Warmup for this size (full pipeline: NHWC numpy → permute → channels_last GPU)
        for _ in range(warmup_iters):
            obs_t = torch.from_numpy(obs_np).permute(0, 3, 1, 2).float().to('cuda')
            mask_t = torch.from_numpy(mask_np).float().to('cuda')
            with torch.no_grad(), autocast('cuda'):
                network(obs_t, mask_t)
            torch.cuda.synchronize()

        # Measure full pipeline (matches runtime eager path)
        times = []
        for _ in range(measure_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            obs_t = torch.from_numpy(obs_np).permute(0, 3, 1, 2).float().to('cuda')
            mask_t = torch.from_numpy(mask_np).float().to('cuda')
            with torch.no_grad(), autocast('cuda'):
                network(obs_t, mask_t)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        eager_times[size] = times
        print(f"{size}", end="→" if size != test_sizes[-1] else "\n", flush=True)

    # --- Fit linear model: eager_time = overhead + per_sample * batch_size ---
    # Using least squares fit on median times for robustness
    sizes = np.array(list(eager_times.keys()), dtype=np.float64)
    medians = np.array([statistics.median(eager_times[s]) for s in eager_times.keys()], dtype=np.float64)

    # Linear regression: y = a + b*x
    # b = Cov(x,y) / Var(x), a = mean(y) - b * mean(x)
    x_mean, y_mean = sizes.mean(), medians.mean()
    cov_xy = ((sizes - x_mean) * (medians - y_mean)).sum()
    var_x = ((sizes - x_mean) ** 2).sum()

    if var_x > 0:
        per_sample_ms = cov_xy / var_x
        overhead_ms = y_mean - per_sample_ms * x_mean
    else:
        per_sample_ms = 0.0
        overhead_ms = y_mean

    # Calculate R² for fit quality
    y_pred = overhead_ms + per_sample_ms * sizes
    ss_res = ((medians - y_pred) ** 2).sum()
    ss_tot = ((medians - y_mean) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Ensure non-negative values (sanity check)
    overhead_ms = max(0.0, overhead_ms)
    per_sample_ms = max(0.0, per_sample_ms)

    # --- Measure CUDA graph times ---
    # Benchmark from 16 upward to find the largest "always wins" size for mini graph.
    # Also include eval_batch//2 (medium candidate) and eval_batch (large).
    medium_graph_size_candidate = eval_batch // 2
    graph_sizes = [s for s in [16, 32, 64, 128, medium_graph_size_candidate, eval_batch]
                   if s <= eval_batch]
    graph_sizes = sorted(set(graph_sizes))

    print(f"  Measuring graphs: ", end="", flush=True)
    graph_times = {}  # size -> median time in ms (including copy-in)

    for size in graph_sizes:
        try:
            obs = torch.zeros(size, INPUT_CHANNELS, 8, 8, device='cuda'
                             ).to(memory_format=torch.channels_last)
            mask = torch.zeros(size, POLICY_SIZE, device='cuda')

            # Warmup pass
            with torch.no_grad(), autocast('cuda'):
                network(obs, mask)

            # Capture graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                with torch.no_grad(), autocast('cuda'):
                    network(obs, mask)

            # Prepare NHWC numpy source for copy-in measurement
            # (matches runtime: permute NHWC→channels_last then copy_)
            obs_np = np.zeros((size, 8, 8, INPUT_CHANNELS), dtype=np.float32)  # NHWC
            mask_np = np.zeros((size, POLICY_SIZE), dtype=np.float32)

            # Warmup replays with copy-in (permute is zero-cost metadata swap)
            for _ in range(warmup_iters):
                obs.copy_(torch.from_numpy(obs_np).permute(0, 3, 1, 2))
                mask.copy_(torch.from_numpy(mask_np))
                graph.replay()
            torch.cuda.synchronize()

            # Measure copy-in + replay (matches runtime graph paths)
            times = []
            for _ in range(measure_iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                obs.copy_(torch.from_numpy(obs_np).permute(0, 3, 1, 2))
                mask.copy_(torch.from_numpy(mask_np))
                graph.replay()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

            graph_times[size] = statistics.median(times)
            print(f"{size}", end="→" if size != graph_sizes[-1] else "\n", flush=True)

            # Clean up graph to free memory
            del graph, obs, mask
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n  Warning: Graph capture failed for size {size}: {e}")
            continue

    if not graph_times:
        print("  Warning: No CUDA graphs captured, using defaults")
        mini_graph_size = 64
        small_graph_size = mini_graph_size * 2
        medium_graph_size = max(small_graph_size + 1, eval_batch // 2)
        return {
            'mini_graph_size': mini_graph_size,
            'small_graph_size': small_graph_size,
            'medium_graph_size': medium_graph_size,
            'large_graph_threshold': eval_batch * 7 // 8,
            'mini_threshold': 0,
            'small_threshold': mini_graph_size,
            'medium_threshold': small_graph_size,
            'eager_overhead_ms': overhead_ms,
            'eager_per_sample_ms': per_sample_ms,
            'measurements': {'eager': eager_times, 'graph': {}},
            'failed': True
        }

    # --- Calculate crossovers and select optimal parameters ---
    # Report measurement variance as coefficient of variation (CV = std/mean)
    avg_cv = sum(statistics.stdev(t) / statistics.mean(t) for t in eager_times.values() if len(t) > 1) / len(eager_times)
    reliability = "high" if avg_cv < 0.05 else "moderate" if avg_cv < 0.15 else "low"

    print(f"\n  Eager: {overhead_ms:.1f}ms overhead + {per_sample_ms:.3f}ms/sample (R2={r_squared:.3f})")
    print(f"  Measurement variance: {avg_cv*100:.1f}% CV ({reliability} reliability)")
    print("  Graph crossovers:")

    crossovers = {}  # size -> crossover batch
    for size, graph_time in sorted(graph_times.items()):
        if per_sample_ms > 0:
            crossover = (graph_time - overhead_ms) / per_sample_ms
        else:
            crossover = 0 if graph_time < overhead_ms else float('inf')

        crossovers[size] = crossover
        fill_pct = (crossover / size * 100) if size > 0 else 0

        # Explain what crossover means in practical terms
        if crossover <= 0:
            verdict = "graph always wins"
        elif crossover >= size:
            verdict = "graph never wins"
        else:
            verdict = f"graph wins at {fill_pct:.0f}%+ fill"

        print(f"    Size {size:4d}: graph={graph_time:.1f}ms, crossover={crossover:.0f}, {verdict}")

    # Select mini graph: largest measured size where graph always wins (crossover ≤ 0)
    always_wins = [s for s, c in crossovers.items() if c <= 0]
    if always_wins:
        mini_graph_size = max(always_wins)
        mini_quality = "graph always wins"
    else:
        # No size where graph always wins — use smallest measured size
        mini_graph_size = min(graph_times.keys())
        mini_quality = "no always-wins size, using smallest"

    # Small = 2× mini (geometric doubling), capped at eval_batch
    small_graph_size = min(mini_graph_size * 2, eval_batch)

    # Medium = eval_batch // 2 (skip if it would overlap with small)
    medium_graph_size = eval_batch // 2

    # Compute crossover threshold for each tier
    # crossover = minimum batch where graph(S) beats eager(B)
    def get_threshold(size):
        if size in crossovers:
            return max(0, int(crossovers[size]))  # clamp to 0 (0 = graph always wins)
        # If size wasn't benchmarked, interpolate from linear model
        graph_est = overhead_ms + per_sample_ms * size  # rough estimate
        return 0  # conservative: assume graph wins

    mini_threshold = get_threshold(mini_graph_size)

    # Small lower bound: max(mini_graph_size, crossover)
    # Prevents batches <= mini_graph_size that fail mini_threshold from leaking into small
    small_threshold = max(mini_graph_size, get_threshold(small_graph_size))

    # Medium: min(7/8 * medium_graph_size, crossover) caps padding waste at 12.5%
    # max(small_graph_size, ...) prevents batches <= small_graph_size from leaking in
    if medium_graph_size > small_graph_size:
        medium_crossover = get_threshold(medium_graph_size)
        medium_threshold = max(small_graph_size, min(medium_graph_size * 7 // 8, medium_crossover))
    else:
        medium_threshold = 0

    # Large: min(7/8 * eval_batch, crossover) — matches medium's pattern
    # When medium is skipped, large must cover everything above small (no gap)
    large_crossover = get_threshold(eval_batch)
    if medium_graph_size > small_graph_size:
        large_graph_threshold = max(medium_graph_size, min(eval_batch * 7 // 8, large_crossover))
    else:
        large_graph_threshold = max(small_graph_size, min(eval_batch * 7 // 8, large_crossover))

    print(f"\n  Selected: mini={mini_graph_size} ({mini_quality})")
    print(f"           small={small_graph_size} (2x mini, threshold={small_threshold})")
    if medium_graph_size > small_graph_size:
        print(f"           medium={medium_graph_size} (eval_batch//2, threshold={medium_threshold})")
    print(f"           large_threshold={large_graph_threshold} (7/8 fill)")

    return {
        'mini_graph_size': mini_graph_size,
        'small_graph_size': small_graph_size,
        'medium_graph_size': medium_graph_size,
        'large_graph_threshold': large_graph_threshold,
        'mini_threshold': mini_threshold,
        'small_threshold': small_threshold,
        'medium_threshold': medium_threshold,
        'eager_overhead_ms': overhead_ms,
        'eager_per_sample_ms': per_sample_ms,
        'r_squared': r_squared,
        'avg_cv': avg_cv,
        'measurements': {
            'eager_medians': {s: statistics.median(t) for s, t in eager_times.items()},
            'graph_medians': graph_times,
            'crossovers': crossovers
        }
    }


def collect_hardware_stats(coordinator, batch_size_histogram, cuda_graph_stats, device):
    """Merge C++ live stats + Python-side metrics into a unified hardware dict.

    Aggregates two sources that are otherwise inaccessible at the metrics
    append site: C++ atomic counters (via coordinator.get_live_stats()) and
    Python-side closure variables (batch histogram, CUDA graph routing, GPU mem).
    """
    hw = {}

    # --- From coordinator.get_live_stats() (C++ side) ---
    try:
        stats = coordinator.get_live_stats()
    except Exception:
        stats = {}

    hw["gpu_wait_ms"]           = stats.get("gpu_wait_ms", 0.0)
    hw["worker_wait_ms"]        = stats.get("worker_wait_ms", 0.0)
    hw["batch_fill_ratio"]      = stats.get("batch_fill_ratio", 0.0)
    hw["pool_exhaustion_count"] = stats.get("pool_exhaustion_count", 0)
    hw["submission_drops"]      = stats.get("submission_drops", 0)
    hw["avg_search_depth"]      = stats.get("avg_search_depth", 0.0)
    hw["max_search_depth"]      = stats.get("max_search_depth", 0)
    hw["nn_evals_per_sec"]      = stats.get("nn_evals_per_sec", 0.0)
    hw["positions_per_sec"]     = stats.get("moves_per_sec", 0.0)

    # --- From batch_size_histogram (Python side) ---
    counts = batch_size_histogram.get("counts", {})
    if counts:
        all_sizes = []
        for sz, cnt in counts.items():
            all_sizes.extend([int(sz)] * cnt)
        all_sizes.sort()
        hw["batch_count"]    = len(all_sizes)
        hw["batch_size_min"] = all_sizes[0]
        hw["batch_size_max"] = all_sizes[-1]
        hw["batch_fill_p50"] = all_sizes[len(all_sizes) // 2]
        hw["batch_fill_p90"] = all_sizes[int(len(all_sizes) * 0.9)]

    # --- From cuda_graph_stats (Python side) ---
    total_fires = sum(cuda_graph_stats.get(k, 0) for k in
        ["large_graph_fires", "medium_graph_fires", "small_graph_fires",
         "mini_graph_fires", "eager_fires"])
    if total_fires > 0:
        hw["cuda_large_fires"]  = cuda_graph_stats.get("large_graph_fires", 0)
        hw["cuda_small_fires"]  = (cuda_graph_stats.get("small_graph_fires", 0)
                                   + cuda_graph_stats.get("mini_graph_fires", 0))
        hw["cuda_eager_fires"]  = cuda_graph_stats.get("eager_fires", 0)
        graph_fires = total_fires - hw["cuda_eager_fires"]
        hw["cuda_graph_pct"]    = round(100.0 * graph_fires / total_fires, 1)
        total_ms = cuda_graph_stats.get("total_infer_time_ms", 0.0)
        hw["avg_inference_ms"]  = round(total_ms / total_fires, 2)
        total_cap = cuda_graph_stats.get("total_pad_capacity", 0)
        if total_cap > 0:
            hw["cuda_pad_waste_pct"] = round(100.0 * cuda_graph_stats.get("total_pad_waste", 0) / total_cap, 1)

    # --- GPU memory (Python side, torch.cuda) ---
    if device == "cuda":
        hw["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024*1024), 1)
        hw["gpu_memory_reserved_mb"]  = round(torch.cuda.memory_reserved() / (1024*1024), 1)

    return hw


def run_parallel_selfplay(
    network: nn.Module,
    replay_buffer,
    device: str,
    args,
    iteration: int,
    progress_callback=None,
    live_dashboard=None,
    reanalyzer=None,
):
    """Run parallel self-play using cross-game batching.

    This achieves 2-5x higher GPU utilization by:
    1. Running multiple games concurrently across worker threads
    2. Collecting MCTS leaves from ALL games into a shared queue
    3. Batching leaves across games for efficient GPU inference

    Args:
        network: Neural network for evaluation
        replay_buffer: C++ ReplayBuffer for storing game data
        device: CUDA or CPU device
        args: Command-line arguments
        progress_callback: Optional callback for progress updates
        live_dashboard: Optional live dashboard for real-time updates

    Returns:
        Tuple of (IterationMetrics, sample_game_dict, hw_stats_dict)
    """
    network.eval()
    metrics = IterationMetrics(iteration=iteration)
    selfplay_start = time.time()

    def total_buffer_size():
        return replay_buffer.size()

    # Calculate how many games per worker to match games_per_iter
    games_per_worker = max(1, args.games_per_iter // args.workers)
    actual_total_games = games_per_worker * args.workers

    # Auto-calculate optimal queue capacity if not specified
    # Formula: max(8192, workers * search_batch * 16)
    # The *16 factor accounts for mirror averaging (2x GPU workload per batch)
    # and ensures sufficient buffer for backpressure to absorb burst submissions.
    if args.queue_capacity > 0:
        queue_capacity = args.queue_capacity
    else:
        queue_capacity = max(8192, args.workers * args.search_batch * 16)

    risk_beta = args.risk_beta

    # Create parallel coordinator
    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=args.workers,
        games_per_worker=games_per_worker,
        num_simulations=args.simulations,
        mcts_batch_size=args.search_batch,
        gpu_batch_size=args.input_batch_size,
        c_puct=args.c_explore,  # C++ binding kwarg stays c_puct
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        temperature_moves=args.temperature_moves,
        gpu_timeout_ms=args.gpu_batch_timeout_ms,
        stall_detection_us=args.stall_detection_us,
        worker_timeout_ms=args.worker_timeout_ms,
        queue_capacity=queue_capacity,
        fpu_base=args.fpu_base,
        risk_beta=risk_beta,
        opponent_risk_min=args.opponent_risk_min,
        opponent_risk_max=args.opponent_risk_max,
        use_gumbel=(args.search_algorithm == "gumbel"),
        gumbel_top_k=args.gumbel_top_k,
        gumbel_c_visit=args.gumbel_c_visit,
        gumbel_c_scale=args.gumbel_c_scale,
    )

    # Set replay buffer so data is stored directly
    coordinator.set_replay_buffer(replay_buffer)

    print(f"  Parallel: {args.workers} workers × {games_per_worker} games = {actual_total_games} games")
    print(f"  eval_batch={args.eval_batch} GPU-effective "
          f"(input={args.input_batch_size}), gpu_timeout={args.gpu_batch_timeout_ms}ms, "
          f"stall={args.stall_detection_us}µs, queue_capacity={queue_capacity}")
    if args.search_algorithm == "gumbel":
        print(f"  Search: Gumbel Top-k SH (top_k={args.gumbel_top_k}, "
              f"c_visit={args.gumbel_c_visit}, c_scale={args.gumbel_c_scale})")
    else:
        print(f"  Search: PUCT (Dirichlet alpha={args.dirichlet_alpha}, eps={args.dirichlet_epsilon})")

    # Push initial progress to live dashboard (show we're in self-play phase)
    if live_dashboard is not None:
        live_dashboard.push_progress(
            iteration=metrics.iteration,
            games_completed=0,
            total_games=actual_total_games,
            moves=0, sims=0, evals=0,
            elapsed_time=0.1,
            buffer_size=total_buffer_size(),
            phase="selfplay"
        )

    # =========================================================================
    # CUDA Graph Optimization: Two-Sided Graph with Eager Fallback
    # =========================================================================
    # CUDA Graphs capture a sequence of GPU operations and replay them with
    # minimal CPU overhead. For fixed-batch-size inference, this eliminates
    # kernel launch overhead (~5-10ms savings per batch).
    #
    # Strategy: Four CUDA graphs at geometric sizes + eager fallback:
    # - Large graph at input_batch_size for high-throughput when batches are full
    # - Medium graph at input_batch_size//2 for batches in the upper-mid range
    # - Small graph at 128 for mid-range batches
    # - Mini graph at 64 for small batches (graph always wins)
    # - Eager fallback when batch is below crossover threshold for its tier
    #
    # All sizes are in input terms (before mirror doubling in forward()).
    # Calibration measures eager vs graph performance to find optimal crossovers.

    gpu_batch_size = args.input_batch_size

    # Run calibration to determine optimal graph sizes and thresholds
    def _use_default_calibration(gpu_batch_size):
        """Return default CUDA graph tier sizes when calibration is skipped/failed."""
        if gpu_batch_size <= 64:
            mini = max(8, gpu_batch_size // 4)
            small = max(mini + 1, gpu_batch_size // 2)
            return mini, small, 0, small, 0, mini + 1, 0
        else:
            mini = 64
            small = mini * 2
            medium = max(small + 1, gpu_batch_size // 2)
            return mini, small, medium, gpu_batch_size * 7 // 8, 0, mini, small

    if device == "cuda" and args.workers > 1:
        print("\nCalibrating CUDA graphs...")
        calibration_start = time.time()
        calibration = calibrate_cuda_graphs(network, device, args.input_batch_size)
        calibration_time = time.time() - calibration_start

        r_sq = calibration.get('r_squared', 0)
        avg_cv = calibration.get('avg_cv', 1.0)

        if r_sq < 0.5 or avg_cv > 0.15:
            print(f"  WARNING: Low reliability (R²={r_sq:.3f}, CV={avg_cv*100:.1f}%), retrying in 15s...")
            time.sleep(15)
            calibration = calibrate_cuda_graphs(network, device, args.input_batch_size)
            r_sq = calibration.get('r_squared', 0)
            avg_cv = calibration.get('avg_cv', 1.0)

        if r_sq < 0.5:
            print(f"  Calibration unreliable after retry (R²={r_sq:.3f}), using defaults")
            (mini_graph_size, small_graph_size, medium_graph_size,
             *_unused) = _use_default_calibration(gpu_batch_size)
        else:
            mini_graph_size = calibration['mini_graph_size']
            small_graph_size = calibration['small_graph_size']
            medium_graph_size = calibration['medium_graph_size']
            print(f"  Calibration completed in {calibration_time:.1f}s\n")
    else:
        (mini_graph_size, small_graph_size, medium_graph_size,
         *_unused) = _use_default_calibration(gpu_batch_size)

    # Large graph fires only when batch fills ≥75% of expected worker output.
    # During end-of-iteration wind-down, batch sizes shrink below this
    # threshold and fall through to the eager path (less padding waste).
    large_graph_threshold = (args.workers * args.search_batch * 3) // 4
    mini_threshold = 0
    small_threshold = 0
    medium_threshold = 0

    use_cuda_graph = (device == "cuda")

    # Large graph (full eval_batch)
    cuda_graph_large = None
    static_obs_large = None
    static_mask_large = None
    static_policy_large = None
    static_value_large = None
    static_wdl_large = None

    # Medium graph (eval_batch // 2)
    cuda_graph_medium = None
    static_obs_medium = None
    static_mask_medium = None
    static_policy_medium = None
    static_value_medium = None
    static_wdl_medium = None

    # Small graph (128)
    cuda_graph_small = None
    static_obs_small = None
    static_mask_small = None
    static_policy_small = None
    static_value_small = None
    static_wdl_small = None

    # Mini graph (64 - smallest tier)
    cuda_graph_mini = None
    static_obs_mini = None
    static_mask_mini = None
    static_policy_mini = None
    static_value_mini = None
    static_wdl_mini = None

    # Free fragmented GPU memory before allocating static CUDA Graph buffers
    if device == "cuda":
        torch.cuda.empty_cache()

    if use_cuda_graph:
        try:
            # ===== Capture LARGE graph (full eval_batch) =====
            static_obs_large = torch.zeros(gpu_batch_size, INPUT_CHANNELS, 8, 8, device='cuda'
                                           ).to(memory_format=torch.channels_last)
            static_mask_large = torch.zeros(gpu_batch_size, POLICY_SIZE, device='cuda')

            # Warm-up pass (required before graph capture)
            with autocast('cuda'):
                static_policy_large, static_value_large, _spl, static_wdl_large = network(static_obs_large, static_mask_large)

            # Capture CUDA graph at full batch size
            cuda_graph_large = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph_large):
                with autocast('cuda'):
                    static_policy_large, static_value_large, _spl, static_wdl_large = network(static_obs_large, static_mask_large)

            # ===== Capture MEDIUM graph (eval_batch // 2) =====
            # Skip if medium would overlap with small (e.g. eval_batch <= 256)
            if medium_graph_size > small_graph_size:
                static_obs_medium = torch.zeros(medium_graph_size, INPUT_CHANNELS, 8, 8, device='cuda'
                                                ).to(memory_format=torch.channels_last)
                static_mask_medium = torch.zeros(medium_graph_size, POLICY_SIZE, device='cuda')

                with autocast('cuda'):
                    static_policy_medium, static_value_medium, _spm2, static_wdl_medium = network(static_obs_medium, static_mask_medium)

                cuda_graph_medium = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cuda_graph_medium):
                    with autocast('cuda'):
                        static_policy_medium, static_value_medium, _spm2, static_wdl_medium = network(static_obs_medium, static_mask_medium)

            # ===== Capture SMALL graph =====
            static_obs_small = torch.zeros(small_graph_size, INPUT_CHANNELS, 8, 8, device='cuda'
                                           ).to(memory_format=torch.channels_last)
            static_mask_small = torch.zeros(small_graph_size, POLICY_SIZE, device='cuda')

            # Warm-up pass
            with autocast('cuda'):
                static_policy_small, static_value_small, _sps, static_wdl_small = network(static_obs_small, static_mask_small)

            # Capture CUDA graph at small batch size
            cuda_graph_small = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph_small):
                with autocast('cuda'):
                    static_policy_small, static_value_small, _sps, static_wdl_small = network(static_obs_small, static_mask_small)

            # ===== Capture MINI graph =====
            static_obs_mini = torch.zeros(mini_graph_size, INPUT_CHANNELS, 8, 8, device='cuda'
                                          ).to(memory_format=torch.channels_last)
            static_mask_mini = torch.zeros(mini_graph_size, POLICY_SIZE, device='cuda')

            # Warm-up pass
            with autocast('cuda'):
                static_policy_mini, static_value_mini, _spm, static_wdl_mini = network(static_obs_mini, static_mask_mini)

            # Capture CUDA graph at mini batch size
            cuda_graph_mini = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph_mini):
                with autocast('cuda'):
                    static_policy_mini, static_value_mini, _spm, static_wdl_mini = network(static_obs_mini, static_mask_mini)

            # Calculate buffer memory usage
            bytes_per_pos = (INPUT_CHANNELS * 8 * 8 + POLICY_SIZE * 2 + 1) * 4  # obs + mask + policy + value
            graph_sizes_total = gpu_batch_size + small_graph_size + mini_graph_size
            if cuda_graph_medium is not None:
                graph_sizes_total += medium_graph_size
            total_mem_mb = graph_sizes_total * bytes_per_pos / (1024 * 1024)

            medium_str = f", medium={medium_graph_size}" if cuda_graph_medium is not None else ""
            print(f"  CUDA Graphs captured: large={gpu_batch_size}{medium_str}, small={small_graph_size}, mini={mini_graph_size}")
            medium_route = f", <={medium_graph_size}->medium" if cuda_graph_medium is not None else ""
            print(f"  Routing: <={mini_graph_size}->mini, <={small_graph_size}->small{medium_route}, >={large_graph_threshold}->large, else->eager")
            print(f"  Total buffer memory: {total_mem_mb:.1f} MB")
        except Exception as e:
            print(f"  CUDA Graph capture failed ({e}), falling back to eager mode")
            use_cuda_graph = False
            cuda_graph_large = None
            cuda_graph_medium = None
            cuda_graph_small = None
            cuda_graph_mini = None

    # CUDA graph fire tracking (mutable dict for closure capture)
    cuda_graph_stats = {
        'large_graph_fires': 0,   # Times large CUDA graph was used
        'medium_graph_fires': 0,  # Times medium CUDA graph was used
        'small_graph_fires': 0,   # Times small CUDA graph was used
        'mini_graph_fires': 0,    # Times mini CUDA graph was used
        'eager_fires': 0,         # Times eager fallback was used
        'total_infer_time_ms': 0.0,  # Cumulative inference time
        # Per-path time tracking for time-distribution pie chart
        'large_graph_time_ms': 0.0,
        'medium_graph_time_ms': 0.0,
        'small_graph_time_ms': 0.0,
        'mini_graph_time_ms': 0.0,
        'eager_time_ms': 0.0,
        # Padding waste tracking (graph_size - actual_batch_size)
        'total_pad_waste': 0,
        'total_pad_capacity': 0,
    }

    # Batch size histogram for distribution analysis
    # This helps determine optimal CUDA graph sizes for future optimization
    batch_size_histogram = {
        'counts': {},           # batch_size -> count
        'samples': [],          # List of (timestamp, batch_size) for time analysis
        'sample_limit': 10000,  # Keep last N samples to avoid memory bloat
    }

    # Create evaluator callback that will be called from C++ GPU thread
    # Signature: (observations, legal_masks, batch_size, out_policies, out_values) -> None
    # Observations arrive in NHWC format (batch, 8, 8, 123) — no C++ transpose
    # torch.permute(0,3,1,2) gives channels_last layout (zero-copy metadata swap)
    # Output buffers are writable numpy views over C++ memory (zero-copy)
    @torch.inference_mode()
    def neural_evaluator(obs_array: np.ndarray, mask_array: np.ndarray, batch_size: int,
                         out_policies: np.ndarray = None, out_values: np.ndarray = None):
        """Neural network evaluator callback for C++ coordinator."""
        infer_start = time.perf_counter()

        # Track batch size distribution for analysis
        batch_size_histogram['counts'][batch_size] = batch_size_histogram['counts'].get(batch_size, 0) + 1
        if len(batch_size_histogram['samples']) < batch_size_histogram['sample_limit']:
            batch_size_histogram['samples'].append((time.time(), batch_size))

        # Route batch to smallest CUDA graph that fits (no eager gap on CUDA).
        # CUDA graph replay (~0.5ms) beats eager (~3-5ms) even with padding waste.
        path_taken = 'eager'
        graph_size_used = 0

        if use_cuda_graph and batch_size <= mini_graph_size and cuda_graph_mini is not None:
            # MINI GRAPH PATH: tightest fit for tiny batches
            path_taken = 'mini'
            graph_size_used = mini_graph_size
            cuda_graph_stats['mini_graph_fires'] += 1

            # permute is zero-copy metadata swap: NHWC → channels_last NCHW
            static_obs_mini[:batch_size].copy_(torch.from_numpy(obs_array[:batch_size]).permute(0, 3, 1, 2))
            static_mask_mini[:batch_size].copy_(torch.from_numpy(mask_array[:batch_size]))
            cuda_graph_mini.replay()
            policies = static_policy_mini[:batch_size]
            wdl_logits = static_wdl_mini[:batch_size]

        elif use_cuda_graph and batch_size <= small_graph_size and cuda_graph_small is not None:
            # SMALL GRAPH PATH: batch fits in small graph
            path_taken = 'small'
            graph_size_used = small_graph_size
            cuda_graph_stats['small_graph_fires'] += 1

            static_obs_small[:batch_size].copy_(torch.from_numpy(obs_array[:batch_size]).permute(0, 3, 1, 2))
            static_mask_small[:batch_size].copy_(torch.from_numpy(mask_array[:batch_size]))
            cuda_graph_small.replay()
            policies = static_policy_small[:batch_size]
            wdl_logits = static_wdl_small[:batch_size]

        elif use_cuda_graph and batch_size <= medium_graph_size and cuda_graph_medium is not None:
            # MEDIUM GRAPH PATH: mid-range batches
            path_taken = 'medium'
            graph_size_used = medium_graph_size
            cuda_graph_stats['medium_graph_fires'] += 1

            static_obs_medium[:batch_size].copy_(torch.from_numpy(obs_array[:batch_size]).permute(0, 3, 1, 2))
            static_mask_medium[:batch_size].copy_(torch.from_numpy(mask_array[:batch_size]))
            cuda_graph_medium.replay()
            policies = static_policy_medium[:batch_size]
            wdl_logits = static_wdl_medium[:batch_size]

        elif use_cuda_graph and cuda_graph_large is not None and batch_size >= large_graph_threshold:
            # LARGE GRAPH PATH: only when batch fills ≥75% of expected worker output.
            # C++ guarantees batch_size <= eval_batch (= gpu_batch_size)
            path_taken = 'large'
            graph_size_used = gpu_batch_size
            cuda_graph_stats['large_graph_fires'] += 1

            static_obs_large[:batch_size].copy_(torch.from_numpy(obs_array[:batch_size]).permute(0, 3, 1, 2))
            static_mask_large[:batch_size].copy_(torch.from_numpy(mask_array[:batch_size]))
            cuda_graph_large.replay()
            policies = static_policy_large[:batch_size]
            wdl_logits = static_wdl_large[:batch_size]

        else:
            # EAGER PATH: batch > medium but < large threshold (wind-down only),
            # or CPU, or graph capture failed
            cuda_graph_stats['eager_fires'] += 1
            obs_tensor = torch.from_numpy(obs_array[:batch_size]).permute(0, 3, 1, 2).float().to(device)
            mask_tensor = torch.from_numpy(mask_array[:batch_size]).float().to(device)

            if device == "cuda":
                with autocast('cuda'):
                    policies, _, _, wdl_logits = network(obs_tensor, mask_tensor)
            else:
                policies, _, _, wdl_logits = network(obs_tensor, mask_tensor)

        # Track padding waste for CUDA graph tiers
        if graph_size_used > 0:
            cuda_graph_stats['total_pad_waste'] += graph_size_used - batch_size
            cuda_graph_stats['total_pad_capacity'] += graph_size_used

        # Convert raw WDL logits to probabilities (softmax exactly once — no double softmax!)
        wdl_probs = F.softmax(wdl_logits.float(), dim=1)

        # Per-call inference timing: synchronize to get accurate GPU time
        # Cost is ~5µs/call, negligible at 50-200 calls/sec
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - infer_start) * 1000
        cuda_graph_stats['total_infer_time_ms'] += elapsed_ms
        cuda_graph_stats['total_calls'] = cuda_graph_stats.get('total_calls', 0) + 1
        time_key = f'{path_taken}_graph_time_ms' if path_taken != 'eager' else 'eager_time_ms'
        cuda_graph_stats[time_key] = cuda_graph_stats.get(time_key, 0) + elapsed_ms

        # Write results directly to C++ output buffers (zero-copy)
        # out_values is now (batch_size, 3) for WDL probabilities
        policies_np = policies[:batch_size].cpu().numpy().astype(np.float32)
        wdl_np = wdl_probs[:batch_size].cpu().numpy().astype(np.float32)

        if out_policies is not None and out_values is not None:
            np.copyto(out_policies[:batch_size], policies_np)
            np.copyto(out_values[:batch_size], wdl_np)
            return None
        else:
            # Legacy fallback
            return policies_np, wdl_np

    # Create progress callback that prints to console and optionally pushes to dashboard
    last_console_report = [selfplay_start]
    console_interval = getattr(args, 'progress_interval', 30.0)

    def progress_cb(live_stats):
        """Print live C++ stats to console and optionally push to dashboard."""
        now = time.time()
        elapsed = now - selfplay_start

        # Console progress printing (always active)
        if (now - last_console_report[0]) >= console_interval:
            games = live_stats.get('games_completed', 0)
            moves = live_stats.get('total_moves', 0)
            sims = live_stats.get('total_simulations', 0)
            evals = live_stats.get('total_nn_evals', 0)
            failures = live_stats.get('mcts_failures', 0)
            avg_batch = live_stats.get('avg_batch_size', 0.0)

            moves_per_sec = moves / elapsed if elapsed > 0 else 0
            sims_per_sec = sims / elapsed if elapsed > 0 else 0
            evals_per_sec = evals / elapsed if elapsed > 0 else 0

            # ETA calculation based on simulation throughput (NN-bounded)
            # This is more stable than game-based ETA since game lengths vary widely
            if evals > 0 and moves > 0 and games > 0:
                avg_moves_per_game = moves / games
                remaining_games = actual_total_games - games
                estimated_remaining_moves = remaining_games * avg_moves_per_game
                estimated_remaining_sims = estimated_remaining_moves * args.simulations
                if sims_per_sec > 0:
                    eta_sec = estimated_remaining_sims / sims_per_sec
                else:
                    eta_sec = remaining_games * (elapsed / games)  # fallback
                eta_str = format_duration(eta_sec)
            else:
                eta_str = "calculating..."

            comp = replay_buffer.get_composition()
            buf_str = f"Buffer: {replay_buffer.size():,} (W={comp['wins']} D={comp['draws']} L={comp['losses']})"
            print(f"    \u23f1 {format_duration(elapsed)} | "
                  f"Games: {games}/{actual_total_games} | "
                  f"Moves: {moves_per_sec:.1f}/s | "
                  f"Sims: {sims_per_sec:,.0f}/s | "
                  f"NN: {evals_per_sec:,.0f}/s | "
                  f"Batch: {avg_batch:.0f} | "
                  f"{buf_str} | "
                  f"ETA: {eta_str}")

            if failures > 0:
                print(f"      Failures: {failures} MCTS timeouts")

            last_console_report[0] = now

        # Batch size distribution logging (every 5 minutes)
        batch_percentiles = {'p25': 0, 'p50': 0, 'p75': 0, 'p90': 0, 'min': 0, 'max': 0}
        counts = batch_size_histogram['counts']
        if counts:
            total = sum(counts.values())
            sorted_sizes = sorted(counts.keys())

            # Calculate percentiles by walking through sorted sizes
            cumsum = 0
            p25 = p50 = p75 = p90 = sorted_sizes[-1]
            for size in sorted_sizes:
                cumsum += counts[size]
                pct = cumsum / total
                if pct >= 0.25 and p25 == sorted_sizes[-1]:
                    p25 = size
                if pct >= 0.50 and p50 == sorted_sizes[-1]:
                    p50 = size
                if pct >= 0.75 and p75 == sorted_sizes[-1]:
                    p75 = size
                if pct >= 0.90 and p90 == sorted_sizes[-1]:
                    p90 = size

            batch_percentiles = {
                'p25': p25, 'p50': p50, 'p75': p75, 'p90': p90,
                'min': sorted_sizes[0], 'max': sorted_sizes[-1]
            }

        # Generate binned histogram for dashboard (20 bins)
        batch_histogram_data = []
        if counts and len(counts) > 0:
            min_size = min(counts.keys())
            max_size = max(counts.keys())
            if max_size > min_size:
                num_bins = min(20, len(counts))
                bin_width = max(1, (max_size - min_size) // num_bins)
                bins = {}
                for size, count in counts.items():
                    bin_center = min(max_size, min_size + ((size - min_size) // bin_width) * bin_width + bin_width // 2)
                    bins[bin_center] = bins.get(bin_center, 0) + count
                # Convert to sorted list of [bin_center, count]
                batch_histogram_data = sorted([[k, v] for k, v in bins.items()])

        # Reanalysis live stats helper (closure over reanalyzer + selfplay_start)
        # Compute target count once from args (same formula as main loop line ~3278)
        _reanalysis_target = int(replay_buffer.size() * args.reanalyze_fraction) if (reanalyzer is not None and args.reanalyze_fraction > 0) else 0

        def _get_reanalysis_live_stats():
            if reanalyzer is None:
                return {}
            try:
                rs = reanalyzer.get_stats()
                return {
                    'reanalysis_completed': rs.get('positions_completed', 0),
                    'reanalysis_skipped': rs.get('positions_skipped', 0),
                    'reanalysis_total': _reanalysis_target,
                    'reanalysis_nn_evals': rs.get('total_nn_evals', 0),
                    'reanalysis_mean_kl': rs.get('mean_kl', 0.0),
                    'reanalysis_elapsed_s': elapsed,
                }
            except Exception:
                return {}

        # Live dashboard push (if enabled)
        if live_dashboard is not None:
            # Calculate GPU stats for dashboard (combining all graph fires)
            graph_fires = (cuda_graph_stats['large_graph_fires'] + cuda_graph_stats['medium_graph_fires'] +
                           cuda_graph_stats['small_graph_fires'] + cuda_graph_stats['mini_graph_fires'])
            total_fires = graph_fires + cuda_graph_stats['eager_fires']
            graph_fire_rate = graph_fires / total_fires if total_fires > 0 else 0.0
            avg_infer_ms = cuda_graph_stats['total_infer_time_ms'] / total_fires if total_fires > 0 else 0.0
            gpu_mem_mb = torch.cuda.memory_allocated() / (1024*1024) if device == "cuda" else 0.0

            live_dashboard.push_progress(
                iteration=metrics.iteration,
                games_completed=live_stats.get('games_completed', 0),
                total_games=actual_total_games,
                moves=live_stats.get('total_moves', 0),
                sims=live_stats.get('total_simulations', 0),
                evals=live_stats.get('total_nn_evals', 0),
                elapsed_time=max(elapsed, 0.1),
                buffer_size=total_buffer_size(),
                phase="selfplay",
                white_wins=live_stats.get('white_wins', 0),
                black_wins=live_stats.get('black_wins', 0),
                draws=live_stats.get('draws', 0),
                standard_wins=live_stats.get('standard_wins', 0),
                opponent_wins=live_stats.get('opponent_wins', 0),
                asymmetric_draws=live_stats.get('asymmetric_draws', 0),
                timeout_evals=live_stats.get('mcts_failures', 0),
                pool_exhaustion=live_stats.get('pool_exhaustion_count', 0),
                submission_drops=live_stats.get('submission_drops', 0),
                partial_subs=live_stats.get('partial_submissions', 0),
                pool_resets=live_stats.get('pool_resets', 0),
                submission_waits=live_stats.get('submission_waits', 0),
                pool_load=live_stats.get('pool_load', 0.0),
                avg_batch_size=live_stats.get('avg_batch_size', 0.0),
                batch_fill_ratio=live_stats.get('batch_fill_ratio', 0.0),
                # GPU metrics (separate counts for pie chart)
                cuda_graph_fires=graph_fires,
                large_graph_fires=cuda_graph_stats['large_graph_fires'],
                medium_graph_fires=cuda_graph_stats['medium_graph_fires'],
                small_graph_fires=cuda_graph_stats['small_graph_fires'],
                mini_graph_fires=cuda_graph_stats['mini_graph_fires'],
                eager_fires=cuda_graph_stats['eager_fires'],
                graph_fire_rate=graph_fire_rate,
                avg_infer_time_ms=avg_infer_ms,
                gpu_memory_used_mb=gpu_mem_mb,
                cuda_graph_enabled=(use_cuda_graph and (cuda_graph_large is not None or cuda_graph_medium is not None or cuda_graph_small is not None or cuda_graph_mini is not None)),
                # Tree depth metrics
                max_search_depth=live_stats.get('max_search_depth', 0),
                min_search_depth=live_stats.get('min_search_depth', 0),
                avg_search_depth=live_stats.get('avg_search_depth', 0.0),
                # Active game move counts
                min_current_moves=live_stats.get('min_current_moves', 0),
                max_current_moves=live_stats.get('max_current_moves', 0),
                # Queue status metrics
                queue_fill_pct=live_stats.get('queue_fill_pct', 0.0),
                gpu_wait_ms=live_stats.get('gpu_wait_ms', 0.0),
                worker_wait_ms=live_stats.get('worker_wait_ms', 0.0),
                buffer_swaps=live_stats.get('buffer_swaps', 0),
                # Batch size distribution percentiles
                batch_p25=batch_percentiles.get('p25', 0),
                batch_p50=batch_percentiles.get('p50', 0),
                batch_p75=batch_percentiles.get('p75', 0),
                batch_p90=batch_percentiles.get('p90', 0),
                batch_min=batch_percentiles.get('min', 0),
                batch_max=batch_percentiles.get('max', 0),
                # Batch histogram for visualization
                batch_histogram=batch_histogram_data,
                large_graph_threshold=large_graph_threshold,
                medium_graph_size=medium_graph_size,
                small_graph_size=small_graph_size,
                mini_graph_size=mini_graph_size,
                # Crossover thresholds for each tier
                medium_threshold=medium_threshold,
                small_threshold=small_threshold,
                mini_threshold=mini_threshold,
                # Per-path inference time (for time-distribution pie chart)
                large_graph_time_ms=cuda_graph_stats['large_graph_time_ms'],
                medium_graph_time_ms=cuda_graph_stats['medium_graph_time_ms'],
                small_graph_time_ms=cuda_graph_stats['small_graph_time_ms'],
                mini_graph_time_ms=cuda_graph_stats['mini_graph_time_ms'],
                eager_time_ms=cuda_graph_stats['eager_time_ms'],
                # Padding waste tracking
                cuda_pad_waste_pct=round(100.0 * cuda_graph_stats['total_pad_waste'] / max(1, cuda_graph_stats['total_pad_capacity']), 1),
                # Reanalysis live stats (if reanalyzer is active)
                **(_get_reanalysis_live_stats()),
            )

    # Run generation with interrupt support (allows Ctrl+C to stop)
    # Poll C++ stats every 5 seconds, console prints at progress_interval
    reanalysis_stats = None
    if reanalyzer is not None:
        # Concurrent reanalysis: use lifecycle split
        result, reanalysis_stats, reanalysis_tail = run_selfplay_with_reanalysis(
            coordinator, neural_evaluator, reanalyzer, shutdown_handler,
            progress_callback=progress_cb,
            progress_interval=5.0
        )
    else:
        result = run_parallel_selfplay_with_interrupt(
            coordinator, neural_evaluator, shutdown_handler,
            progress_callback=progress_cb,
            progress_interval=5.0
        )

    # Extract stats from result dict
    if result is None:
        # Interrupted before any results - return minimal metrics
        metrics.selfplay_time = time.time() - selfplay_start
        return metrics, None, {}
    elif isinstance(result, dict):
        # Check for C++ thread errors surfaced from the coordinator
        if result.get('cpp_error'):
            print(f"\n  WARNING: C++ error during self-play: {result['cpp_error']}")

        metrics.num_games = result.get('games_completed', actual_total_games)
        metrics.total_moves = result.get('total_moves', 0)
        metrics.white_wins = result.get('white_wins', 0)
        metrics.black_wins = result.get('black_wins', 0)
        metrics.draws = result.get('draws', 0)
        metrics.standard_wins = result.get('standard_wins', 0)
        metrics.opponent_wins = result.get('opponent_wins', 0)
        metrics.asymmetric_draws = result.get('asymmetric_draws', 0)
        metrics.total_simulations = result.get('total_simulations', 0)
        metrics.total_nn_evals = result.get('total_nn_evals', 0)
        metrics.draws_repetition = result.get('draws_repetition', 0)
        metrics.draws_stalemate = result.get('draws_stalemate', 0)
        metrics.draws_fifty_move = result.get('draws_fifty_move', 0)
        metrics.draws_insufficient = result.get('draws_insufficient', 0)
        metrics.draws_max_moves = result.get('draws_max_moves', 0)
        metrics.draws_early_repetition = result.get('draws_early_repetition', 0)

        # Display diagnostic metrics for parallel pipeline health
        mcts_failures = result.get('mcts_failures', 0)
        pool_exhaustion = result.get('pool_exhaustion_count', 0)
        partial_subs = result.get('partial_submissions', 0)
        submission_drops = result.get('submission_drops', 0)
        pool_resets = result.get('pool_resets', 0)
        submission_waits = result.get('submission_waits', 0)
        avg_batch = result.get('avg_batch_size', 0)
        total_batches = result.get('total_batches', 0)

        # Calculate NN evals per move (should be ~51 for 800 sims, batch 64)
        nn_evals_per_move = metrics.total_nn_evals / max(metrics.total_moves, 1)
        failure_rate = mcts_failures / max(metrics.total_moves, 1) * 100

        print(f"  Parallel stats: {metrics.total_nn_evals:,} NN evals ({nn_evals_per_move:.1f}/move), "
              f"avg_batch={avg_batch:.1f}, batches={total_batches:,}")

        # Print final batch size distribution summary
        counts = batch_size_histogram['counts']
        if counts:
            total = sum(counts.values())
            sorted_sizes = sorted(counts.keys())
            cumsum = 0
            p25 = p50 = p75 = p90 = sorted_sizes[-1]
            for size in sorted_sizes:
                cumsum += counts[size]
                pct = cumsum / total
                if pct >= 0.25 and p25 == sorted_sizes[-1]: p25 = size
                if pct >= 0.50 and p50 == sorted_sizes[-1]: p50 = size
                if pct >= 0.75 and p75 == sorted_sizes[-1]: p75 = size
                if pct >= 0.90 and p90 == sorted_sizes[-1]: p90 = size
            top5 = sorted(counts.items(), key=lambda x: -x[1])[:5]
            print(f"  📊 Batch distribution (n={total}): P25={p25}, P50={p50}, P75={p75}, P90={p90}, "
                  f"range=[{sorted_sizes[0]}, {sorted_sizes[-1]}]")
            print(f"     Top 5: {', '.join(f'{s}({c})' for s, c in top5)}")

        if mcts_failures > 0 or pool_exhaustion > 0 or partial_subs > 0 or submission_waits > 0:
            print(f"  Pipeline issues: {mcts_failures} MCTS failures ({failure_rate:.1f}%), "
                  f"pool_exhaustion={pool_exhaustion}, partial_subs={partial_subs}, "
                  f"drops={submission_drops}, resets={pool_resets}")
            if submission_waits > 0:
                print(f"  Backpressure: {submission_waits} submission waits (workers waited for queue space)")
            if failure_rate > 10:
                print(f"  HIGH FAILURE RATE - Consider increasing --queue-capacity")
    elif isinstance(result, list):
        # result is a list of game trajectories (if no replay buffer set)
        metrics.num_games = len(result)
        for traj in result:
            metrics.total_moves += traj.get('num_moves', 0)
            game_result = traj.get('result', 0)
            if game_result > 0:
                metrics.white_wins += 1
            elif game_result < 0:
                metrics.black_wins += 1
            else:
                metrics.draws += 1

    metrics.selfplay_time = time.time() - selfplay_start
    metrics.avg_game_length = metrics.total_moves / max(metrics.num_games, 1)
    metrics.risk_beta = risk_beta

    # Only estimate if we don't have real stats from the C++ coordinator
    # (parallel mode provides real stats, sequential mode needs estimates)
    if metrics.total_simulations == 0:
        metrics.total_simulations = metrics.total_moves * args.simulations
    if metrics.total_nn_evals == 0:
        # Sequential mode: each simulation = one NN eval (search_batch=1 for PUCT)
        metrics.total_nn_evals = metrics.total_simulations

    # Log and record reanalysis stats
    if reanalysis_stats is not None:
        completed = reanalysis_stats.get('positions_completed', 0)
        skipped = reanalysis_stats.get('positions_skipped', 0)
        nn_evals = reanalysis_stats.get('total_nn_evals', 0)
        mean_kl = reanalysis_stats.get('mean_kl', 0.0)
        metrics.reanalysis_positions = completed
        metrics.reanalysis_time_s = reanalysis_tail
        metrics.reanalysis_mean_kl = mean_kl
        print(f"  Reanalysis: {completed} positions updated, {skipped} skipped, "
              f"{nn_evals} NN evals, KL={mean_kl:.4f}, tail={reanalysis_tail:.1f}s")

    # Extract sample game for PGN export (if available)
    sample_game = None
    try:
        sg = coordinator.get_sample_game()
        if sg.get("has_game", False):
            sample_game = sg
    except Exception:
        pass

    # Collect hardware/performance stats from both C++ and Python sources
    hw_stats = collect_hardware_stats(coordinator, batch_size_histogram, cuda_graph_stats, device)

    return metrics, sample_game, hw_stats


# =============================================================================
# Training Loop with C++ ReplayBuffer
# =============================================================================

def train_iteration(
    network: nn.Module,
    optimizer: optim.Optimizer,
    replay_buffer,  # alphazero_cpp.ReplayBuffer
    batch_size: int,
    epochs: int,
    device: str,
    scaler: GradScaler,
    per_beta: float = 0.0,
) -> dict:
    """Train for one iteration using C++ ReplayBuffer.

    When PER is enabled (replay_buffer.per_enabled() and per_beta > 0),
    uses prioritized sampling with IS weight correction and updates
    priorities after each gradient step.
    """
    network.train()

    # Check model weights for NaN/inf before training (catches corrupted checkpoints)
    for name, p in network.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            return {
                'loss': float('nan'), 'policy_loss': float('nan'),
                'value_loss': float('nan'), 'num_batches': 0,
                'error': f'NaN/inf in parameter: {name}'
            }

    use_per = replay_buffer.per_enabled() and per_beta > 0

    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    num_batches = 0
    total_grad_norm = 0.0
    max_grad_norm = 0.0
    nan_skip_count = 0

    for epoch in range(epochs):
        # Sample batch from C++ ReplayBuffer
        indices = None
        if use_per:
            obs, policies, values, wdl_targets, soft_values, indices, is_weights = \
                replay_buffer.sample_prioritized(batch_size, per_beta)
            is_weights_tensor = torch.from_numpy(is_weights).to(device)
        else:
            obs, policies, values, wdl_targets, soft_values = replay_buffer.sample(batch_size)

        # Convert to PyTorch tensors
        # obs is (batch, 7872) flat, need to reshape to (batch, 123, 8, 8)
        obs = obs.reshape(-1, 8, 8, INPUT_CHANNELS)  # (batch, 8, 8, 123) NHWC
        # permute is zero-copy metadata swap; .to(device) preserves channels_last
        obs_tensor = torch.from_numpy(obs).permute(0, 3, 1, 2).to(device)
        policy_target = torch.from_numpy(policies).to(device)
        value_target = torch.from_numpy(values).to(device)  # (batch,)
        mcts_wdl_target = torch.from_numpy(wdl_targets).to(device)  # (batch, 3)

        optimizer.zero_grad()

        with autocast('cuda', enabled=(device == "cuda")):
            # Forward pass (no mask during training - targets already masked)
            policy_pred, value_pred, policy_logits, wdl_logits = network(obs_tensor)

            # Per-sample policy loss: NaN-safe cross-entropy — shape (batch,)
            per_sample_policy = -torch.sum(
                torch.nan_to_num(policy_target * F.log_softmax(policy_logits, dim=1), nan=0.0),
                dim=1
            )

            # Value loss: soft WDL cross-entropy with pure game outcome targets
            # Build outcome WDL from scalar game result: win→[1,0,0], draw→[0,1,0], loss→[0,0,1]
            outcome_wdl = torch.zeros_like(mcts_wdl_target)  # (batch, 3)
            outcome_wdl[:, 0] = (value_target > 0.5).float()    # win
            outcome_wdl[:, 1] = ((value_target >= -0.5) & (value_target <= 0.5)).float()  # draw
            outcome_wdl[:, 2] = (value_target < -0.5).float()   # loss

            # Per-sample value loss: NaN-safe — shape (batch,)
            per_sample_value = -torch.sum(
                torch.nan_to_num(outcome_wdl * F.log_softmax(wdl_logits, dim=1), nan=0.0),
                dim=1
            )

            # Per-sample total loss — shape (batch,)
            per_sample_total = per_sample_policy + per_sample_value

            # Compute weighted or unweighted loss for backprop
            if use_per:
                loss = torch.mean(is_weights_tensor * per_sample_total)
            else:
                loss = torch.mean(per_sample_total)

            # Unweighted means for logging (comparable regardless of PER)
            policy_loss = per_sample_policy.mean()
            value_loss = per_sample_value.mean()

        # Skip batch if loss is NaN/inf (corrupted weights or bad data)
        if torch.isnan(loss) or torch.isinf(loss):
            nan_skip_count += 1
            if nan_skip_count <= 3:
                print(f"  WARNING: NaN/inf loss in batch {num_batches + nan_skip_count}")
                if torch.isnan(policy_logits).any().item():
                    print(f"    Model producing NaN logits (weights corrupted)")
            optimizer.zero_grad()
            continue

        # Backward pass with mixed precision (GradScaler handles inf detection)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Compute grad norm for monitoring (no clipping — avoids inf/inf = NaN trap)
        with torch.no_grad():
            grad_norms = [p.grad.norm(2).item() for p in network.parameters() if p.grad is not None]
            grad_norm_val = (sum(g**2 for g in grad_norms)) ** 0.5
        total_grad_norm += grad_norm_val
        max_grad_norm = max(max_grad_norm, grad_norm_val)

        scaler.step(optimizer)   # internally skips if unscale_ found inf
        scaler.update()

        # PER: update priorities with fresh per-sample losses
        if use_per and indices is not None:
            with torch.no_grad():
                new_priorities = per_sample_total.detach().cpu().numpy() + 1e-6
            replay_buffer.update_priorities(indices, new_priorities)

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1

    if num_batches == 0:
        return {
            'loss': float('nan'), 'policy_loss': float('nan'),
            'value_loss': float('nan'), 'num_batches': 0,
            'nan_skip_count': nan_skip_count,
            'error': f'All {nan_skip_count} batches produced NaN'
        }

    result = {
        'loss': total_loss / num_batches,
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches,
        'num_batches': num_batches,
        'grad_norm_avg': total_grad_norm / num_batches,
        'grad_norm_max': max_grad_norm,
        'nan_skip_count': nan_skip_count,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero Training with C++ Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training (creates organized output directory)
    uv run python alphazero-cpp/scripts/train.py

    # Faster iteration (fewer games, more iterations)
    uv run python alphazero-cpp/scripts/train.py --iterations 200 --games-per-iter 25

    # Higher quality (more simulations)
    uv run python alphazero-cpp/scripts/train.py --simulations 1600

    # Resume from checkpoint file
    uv run python alphazero-cpp/scripts/train.py --resume checkpoints/f192-b15_2024-02-03_14-30-00/model_iter_005.pt

    # Resume from run directory (finds latest checkpoint)
    uv run python alphazero-cpp/scripts/train.py --resume checkpoints/f192-b15_2024-02-03_14-30-00

    # Disable visualization
    uv run python alphazero-cpp/scripts/train.py --no-visualization
        """
    )

    # Training iterations
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations (default: 100)")
    parser.add_argument("--games-per-iter", type=int, default=50,
                        help="Self-play games per iteration (default: 50)")

    # MCTS parameters (shared by both search algorithms)
    parser.add_argument("--simulations", type=int, default=800,
                        help="MCTS simulations per move (default: 800)")
    parser.add_argument("--search-algorithm", type=str, default="gumbel",
                        choices=["gumbel", "puct"],
                        help="Root search algorithm: 'gumbel' (Gumbel Top-k Sequential Halving, default) "
                             "or 'puct' (standard AlphaZero PUCT+Dirichlet). "
                             "search_batch is auto-derived: gumbel→gumbel_top_k, puct→1")
    parser.add_argument("--c-explore", type=float, default=1.5,
                        help="MCTS exploration constant for tree traversal (default: 1.5)")
    parser.add_argument("--fpu-base", type=float, default=0.3,
                        help="Dynamic FPU reduction. penalty = fpu_base * sqrt(1 - prior) (default: 0.3)")
    parser.add_argument("--temperature-moves", type=int, default=30,
                        help="Moves with temperature=1 for exploration (default: 30)")

    # PUCT-specific (ignored when --search-algorithm gumbel)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3,
                        help="Dirichlet noise alpha for root exploration (default: 0.3)")
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25,
                        help="Dirichlet noise weight for root exploration (default: 0.25)")

    # Gumbel-specific (ignored when --search-algorithm puct)
    parser.add_argument("--gumbel-top-k", type=int, default=16,
                        help="Initial m for Gumbel Sequential Halving; also sets search_batch (default: 16)")
    parser.add_argument("--gumbel-c-visit", type=float, default=50.0,
                        help="Gumbel sigma() visit constant (default: 50.0)")
    parser.add_argument("--gumbel-c-scale", type=float, default=1.0,
                        help="Gumbel sigma() scale factor (default: 1.0)")

    # Parallel self-play
    parser.add_argument("--workers", type=int, default=1,
                        help="Self-play workers. 1=sequential, >1=parallel with cross-game batching (default: 1)")
    parser.add_argument("--gpu-batch-timeout-ms", type=int, default=20,
                        help="GPU batch collection timeout in ms (default: 20)")
    parser.add_argument("--stall-detection-us", type=int, default=500,
                        help="GPU spin-poll stall detection in µs (default: 500)")
    parser.add_argument("--worker-timeout-ms", type=int, default=2000,
                        help="Worker wait time for NN results in ms (default: 2000)")
    parser.add_argument("--queue-capacity", type=int, default=0,
                        help="Eval queue capacity. 0=auto-calculate (default: 0)")

    # Risk / ERM
    parser.add_argument("--risk-beta", type=float, default=0.0,
                        help="ERM risk sensitivity. >0 risk-seeking, <0 risk-averse, 0=neutral. Range [-3,3] (default: 0.0)")
    parser.add_argument("--opponent-risk", type=str, default=None,
                        help="Per-game opponent risk range MIN:MAX (e.g. '0.5:2.0'). "
                             "MIN can equal MAX for a fixed opponent risk. "
                             "Each game: one side uses --risk-beta, the other samples from U(min,max).")

    # Network parameters
    parser.add_argument("--filters", type=int, default=192,
                        help="Network filters (default: 192)")
    parser.add_argument("--blocks", type=int, default=15,
                        help="Residual blocks (default: 15)")
    parser.add_argument("--se-reduction", type=int, default=16,
                        help="SE block reduction ratio (default: 16)")

    # Training parameters
    parser.add_argument("--train-batch", type=int, default=256,
                        help="Samples per training gradient step (default: 256)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs per iteration (default: 5)")
    # --no-wdl removed: WDL is always enabled
    parser.add_argument("--buffer-size", type=int, default=100000,
                        help="Replay buffer size (default: 100000)")
    parser.add_argument("--max-fillup-factor", type=int, default=3,
                        help="Max multiplier for extra fill-up games (default: 3). "
                             "Fill-up plays at most N * games_per_iter extra games. "
                             "Set to 0 to disable fill-up entirely.")

    # Prioritized Experience Replay (PER)
    parser.add_argument("--priority-exponent", type=float, default=0.0,
                        help="PER priority exponent alpha (default: 0.0 = uniform sampling). "
                             "Recommended: 0.6. Higher = more aggressive prioritization.")
    parser.add_argument("--per-beta", type=float, default=0.4,
                        help="PER IS correction beta (default: 0.4). "
                             "0=no correction, 1=full correction. See operation manual Section 10q.")

    # Device and paths
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Base checkpoint directory (default: checkpoints)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path (.pt file) or run directory")
    parser.add_argument("--save-interval", type=int, default=1,
                        help="Save checkpoint every N iterations (default: 1)")
    parser.add_argument("--progress-interval", type=float, default=30.0,
                        help="Print progress statistics every N seconds (default: 30)")

    # Visualization and output
    parser.add_argument("--no-visualization", action="store_true",
                        help="Disable summary.html generation (default: enabled)")
    parser.add_argument("--no-sample-games", action="store_true",
                        help="Disable saving sample game PGN after each iteration (default: enabled)")
    parser.add_argument("--live", action="store_true",
                        help="Enable LIVE web dashboard with real-time updates (requires flask)")
    parser.add_argument("--dashboard-port", type=int, default=5000,
                        help="Port for live dashboard server (default: 5000)")
    parser.add_argument("--claude", action="store_true",
                        help="Enable Claude Code agent integration (writes claude_log.jsonl to run_dir)")
    parser.add_argument("--claude-timeout", type=int, default=600,
                        help="Seconds to wait for Claude review before auto-continuing (default: 600). "
                             "Set to 0 to disable blocking (async mode).")

    # Reanalysis
    parser.add_argument("--reanalyze-fraction", type=float, default=0.0,
                        help="Fraction of buffer to reanalyze each iteration (0=disabled, 0.25=recommended)")
    parser.add_argument("--reanalyze-simulations-ratio", type=float, default=0.25,
                        help="Reanalysis sims as fraction of self-play sims (default: 0.25)")
    parser.add_argument("--reanalyze-workers", type=int, default=0,
                        help="Number of reanalysis workers (0=auto: half of self-play workers)")

    args = parser.parse_args()

    # Derive search_batch from search algorithm (no longer user-facing)
    if args.search_algorithm == "gumbel":
        args.search_batch = args.gumbel_top_k
    else:  # puct
        args.search_batch = 1

    # Install last-resort crash hook to ensure errors are visible
    def _crash_hook(exc_type, exc_value, exc_tb):
        import traceback
        msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        sys.stderr.write(f"\n[FATAL CRASH]\n{msg}\n")
        sys.stderr.flush()
    sys.excepthook = _crash_hook

    # Parse --opponent-risk MIN:MAX
    if args.opponent_risk:
        parts = args.opponent_risk.split(":")
        if len(parts) != 2:
            parser.error("--opponent-risk must be MIN:MAX (e.g. '0.5:2.0' or '1.0:1.0')")
        args.opponent_risk_min = float(parts[0])
        args.opponent_risk_max = float(parts[1])
        if args.opponent_risk_min > args.opponent_risk_max:
            parser.error("--opponent-risk MIN must be <= MAX")
        if args.opponent_risk_min == args.opponent_risk_max:
            # Add epsilon so C++ min < max check passes; sampled value is effectively MIN
            args.opponent_risk_max = args.opponent_risk_min + 1e-8
    else:
        args.opponent_risk_min = 0.0
        args.opponent_risk_max = 0.0

    # Force save_interval=1 in Claude mode (every iteration gets a checkpoint)
    if args.claude and args.save_interval != 1:
        print(f"  [Claude] Overriding save_interval={args.save_interval} -> 1 (checkpoint every iteration)")
        args.save_interval = 1

    # Auto-compute eval_batch (GPU-effective batch size)
    # AlphaZeroNet.forward() doubles via horizontal mirror equivariance,
    # so effective GPU batch = input_positions × mirror_factor.
    mirror_factor = 2 if hasattr(AlphaZeroNet, '_mirror_input') else 1
    input_batch_size = ((args.workers * args.search_batch + 31) // 32) * 32
    input_batch_size = max(input_batch_size, 32)
    args.eval_batch = input_batch_size * mirror_factor  # GPU-effective
    args.input_batch_size = input_batch_size             # for C++ queue / CUDA graphs
    if mirror_factor > 1:
        print(f"  eval_batch={args.eval_batch} GPU-effective "
              f"({input_batch_size} input × {mirror_factor} mirror)")
    else:
        print(f"  eval_batch={args.eval_batch} (workers × search_batch, rounded to 32)")

    # Validate parameter relationships for parallel mode
    if args.workers > 1:
        if args.queue_capacity == 0:
            auto_cap = max(8192, args.workers * args.search_batch * 8)
            print(f"  Auto queue_capacity: {auto_cap} (from {args.workers} workers * {args.search_batch} search_batch * 8)")

    # Handle device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    # Enable cuDNN and TF32 optimizations for faster convolutions/matmuls
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # ==========================================================================
    # Run Directory Setup
    # ==========================================================================
    # Determine run directory: either resume from existing or create new
    start_iter = 0

    if args.resume:
        # Resume from existing run
        run_dir, checkpoint_path = parse_resume_path(args.resume)
        if not os.path.exists(run_dir):
            print(f"ERROR: Run directory does not exist: {run_dir}")
            sys.exit(1)
        if checkpoint_path is None:
            print(f"ERROR: No checkpoint found in run directory: {run_dir}")
            sys.exit(1)
        print(f"Resuming from: {run_dir}", flush=True)

        # Load training log (JSONL or legacy JSON fallback)
        print("  Loading training log...", end=" ", flush=True)
        loaded_config, loaded_iterations = load_training_log(run_dir)
        print(f"done ({len(loaded_iterations)} iterations)", flush=True)

        # One-time migration: if only legacy JSON exists, convert to JSONL
        jsonl_path = os.path.join(run_dir, "training_log.jsonl")
        legacy_path = os.path.join(run_dir, "training_metrics.json")
        if not os.path.exists(jsonl_path) and os.path.exists(legacy_path):
            print("  Migrating training_metrics.json -> training_log.jsonl...", end=" ", flush=True)
            # Batch-write all records in a single open/flush/fsync (avoid per-record fsync)
            with open(jsonl_path, 'w') as f:
                if loaded_config:
                    f.write(json.dumps(loaded_config, separators=(',', ':')) + '\n')
                for rec in loaded_iterations:
                    rec.setdefault("type", "iteration")
                    f.write(json.dumps(rec, separators=(',', ':')) + '\n')
                f.flush()
                os.fsync(f.fileno())
            print(f"done ({len(loaded_iterations)} records migrated)")
    else:
        # Create new run directory
        os.makedirs(args.save_dir, exist_ok=True)
        run_dir = create_run_directory(args.save_dir, args.filters, args.blocks, args.se_reduction)
        checkpoint_path = None
        print(f"Created run directory: {run_dir}")

    # Config dict (used for display, dashboard, and JSONL config record)
    config = {
        "filters": args.filters,
        "blocks": args.blocks,
        "simulations": args.simulations,
        "games_per_iter": args.games_per_iter,
        "buffer_size": args.buffer_size,
        "lr": args.lr,
        "workers": args.workers,
        "train_batch": args.train_batch,
        "epochs": args.epochs,
    }

    # Write config record for new runs (resume already has one)
    if not args.resume:
        write_config_record(run_dir, config)

    # Print configuration
    print("=" * 70)
    print("AlphaZero Training with C++ Backend")
    print("=" * 70)
    print(f"Run directory:       {run_dir}")
    print(f"Device:              {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"Network:             {args.filters} filters x {args.blocks} blocks")
    print(f"Input channels:      {INPUT_CHANNELS}")
    print(f"MCTS:                {args.simulations} sims, c_explore={args.c_explore}, fpu_base={args.fpu_base}")
    if args.search_algorithm == "gumbel":
        print(f"Root search:         Gumbel Top-k SH (top_k={args.gumbel_top_k}, c_visit={args.gumbel_c_visit}, c_scale={args.gumbel_c_scale})")
    else:
        print(f"Root search:         PUCT + Dirichlet (alpha={args.dirichlet_alpha}, eps={args.dirichlet_epsilon})")
    if args.workers > 1:
        print(f"Self-play:           Parallel ({args.workers} workers, "
              f"eval_batch={args.eval_batch} GPU-effective, input={args.input_batch_size})")
    else:
        print(f"Self-play:           Sequential")
    print(f"Training:            {args.iterations} iters x {args.games_per_iter} games")
    print(f"                     train_batch={args.train_batch}, lr={args.lr}, epochs={args.epochs}")
    print(f"Value head:          WDL (soft cross-entropy), risk_beta={args.risk_beta}")
    if args.opponent_risk:
        print(f"Opponent risk:       {args.opponent_risk}")
    print(f"Buffer:              {args.buffer_size} positions")
    if args.priority_exponent > 0:
        print(f"PER:                 alpha={args.priority_exponent:g}, beta={args.per_beta:g}")
    print(f"Checkpoints:         model_iter_*.pt (every {args.save_interval} iters)")
    print(f"Visualization:       {'disabled' if args.no_visualization else 'summary.html'}")
    print(f"Progress reports:    Every {args.progress_interval:.0f} seconds")
    print("=" * 70)

    # Create network
    network = AlphaZeroNet(num_filters=args.filters, num_blocks=args.blocks,
                           input_channels=INPUT_CHANNELS, wdl=True,
                           se_reduction=args.se_reduction)
    network = network.to(device)
    if device == "cuda":
        network = network.to(memory_format=torch.channels_last)

    num_params = sum(p.numel() for p in network.parameters())
    print(f"Network parameters:  {num_params:,}")

    # Log initial WDL behavior (sanity check for fresh networks)
    with torch.no_grad():
        dummy = torch.randn(1, INPUT_CHANNELS, 8, 8, device=device)
        if device == "cuda":
            dummy = dummy.to(memory_format=torch.channels_last)
        _, _, _, wdl = network(dummy)
        wdl_p = F.softmax(wdl, dim=1)
        print(f"  Initial WDL distribution: {wdl_p.cpu().numpy().round(3)}")

    # Create optimizer with weight decay excluded from BN and bias parameters.
    # BN gamma/beta and all bias params are 1D and shouldn't be L2-penalized —
    # penalizing the 1-filter BN bottleneck in the value head caused WDL collapse.
    no_decay_keywords = ['bn', 'bias']
    decay_params = [p for n, p in network.named_parameters()
                    if not any(kw in n for kw in no_decay_keywords)]
    no_decay_params = [p for n, p in network.named_parameters()
                       if any(kw in n for kw in no_decay_keywords)]
    optimizer = optim.Adam([
        {'params': decay_params, 'weight_decay': 1e-4},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=args.lr)
    scaler = GradScaler('cuda', enabled=(device == "cuda"))

    # Create C++ ReplayBuffer
    replay_buffer = alphazero_cpp.ReplayBuffer(capacity=args.buffer_size)

    # Enable PER if requested (must be before buffer load so priorities are loaded)
    if args.priority_exponent > 0:
        replay_buffer.enable_per(args.priority_exponent)

    # Enable FEN storage for reanalysis (must be before buffer load so FENs are loaded)
    if args.reanalyze_fraction > 0:
        replay_buffer.enable_fen_storage()

    # Resume from checkpoint (load model weights)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"\nLoading checkpoint: {checkpoint_path} ({ckpt_size_mb:.1f} MB)...", flush=True)
        print("  Reading checkpoint file...", end=" ", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print("done", flush=True)
        print("  Loading model weights...", end=" ", flush=True)
        state_dict = checkpoint['model_state_dict']
        ckpt_wdl = checkpoint.get('config', {}).get('wdl', False)

        if not ckpt_wdl:
            # Migrate scalar value head → WDL: win=old, draw=zeros, loss=-old
            old_w = state_dict['value_head.fc2.weight']  # (1, 256)
            old_b = state_dict['value_head.fc2.bias']    # (1,)
            new_w = torch.zeros(3, old_w.shape[1])
            new_w[0] = old_w[0]       # win
            new_w[2] = -old_w[0]      # loss (negated)
            new_b = torch.zeros(3)
            new_b[0] = old_b[0]
            new_b[2] = -old_b[0]
            state_dict['value_head.fc2.weight'] = new_w
            state_dict['value_head.fc2.bias'] = new_b
            print("migrated scalar→WDL...", end=" ", flush=True)

        network.load_state_dict(state_dict)
        print("done", flush=True)
        if not ckpt_wdl:
            print("  Skipping optimizer state (incompatible after scalar→WDL migration)")
        else:
            print("  Loading optimizer state...", end=" ", flush=True)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("done", flush=True)
        start_iter = checkpoint.get('iteration', 0)

        # Emergency checkpoints saved mid-iteration — re-run that iteration
        is_emergency = checkpoint.get('emergency_save', False) or '_emergency' in os.path.basename(checkpoint_path)
        if is_emergency:
            start_iter = max(0, start_iter - 1)
            print(f"Emergency checkpoint detected — will re-train iteration {start_iter + 1}")
        print(f"Resumed from iteration {start_iter}")

        # Try to load replay buffer — new fixed name first, fall back to old per-iteration files
        buffer_loaded = False
        buffer_path = os.path.join(run_dir, "replay_buffer.rpbf")
        if os.path.exists(buffer_path):
            print(f"  Loading replay buffer: {buffer_path}...", end=" ", flush=True)
            if replay_buffer.load(buffer_path):
                comp = replay_buffer.get_composition()
                print(f"done ({replay_buffer.size()} samples, "
                      f"W={comp['wins']} D={comp['draws']} L={comp['losses']})")
                buffer_loaded = True
            else:
                print("FAILED")
        if not buffer_loaded:
            # Fall back to old per-iteration naming (buffer_iter_NNN.rpbf)
            import glob as glob_mod
            rpbf_files = sorted(glob_mod.glob(os.path.join(run_dir, "buffer_iter_*.rpbf")))
            for rpbf_candidate in reversed(rpbf_files):
                print(f"  Loading replay buffer: {rpbf_candidate}...", end=" ", flush=True)
                if replay_buffer.load(rpbf_candidate):
                    comp = replay_buffer.get_composition()
                    print(f"done ({replay_buffer.size()} samples, "
                          f"W={comp['wins']} D={comp['draws']} L={comp['losses']})")
                    buffer_loaded = True
                    break
                else:
                    print("FAILED, trying older buffer...")
        if not buffer_loaded:
            print("  No replay buffer file found (starting with empty buffer)")

        # Truncate training log to discard records from reverted iterations
        metrics_removed = truncate_training_log(run_dir, start_iter)
        if metrics_removed > 0:
            print(f"  Truncated training log: removed {metrics_removed} records >= iteration {start_iter}")

    # Create evaluator and self-play (only needed for sequential mode)
    evaluator = None
    self_play = None
    if args.workers <= 1:
        evaluator = BatchedEvaluator(network, device, use_amp=(device == "cuda"))
        self_play = CppSelfPlay(
            evaluator=evaluator,
            num_simulations=args.simulations,
            mcts_batch_size=args.search_batch,
            c_puct=args.c_explore,  # C++ binding kwarg stays c_puct
            temperature_moves=args.temperature_moves,
        )

    # Metrics tracker (for console output)
    print("Initializing metrics tracker...", end=" ", flush=True)
    metrics_tracker = MetricsTracker()
    print("done", flush=True)

    # Live dashboard (optional - real-time web server)
    def collect_dashboard_params():
        """Collect current training parameters for dashboard display."""
        return {
            'lr': args.lr, 'train_batch': args.train_batch, 'epochs': args.epochs,
            'simulations': args.simulations, 'c_explore': args.c_explore,
            'risk_beta': args.risk_beta, 'temperature_moves': args.temperature_moves,
            'dirichlet_alpha': args.dirichlet_alpha,
            'dirichlet_epsilon': args.dirichlet_epsilon, 'fpu_base': args.fpu_base,
            'opponent_risk_min': args.opponent_risk_min,
            'opponent_risk_max': args.opponent_risk_max,
            'games_per_iter': args.games_per_iter,
            'max_fillup_factor': args.max_fillup_factor,
            'save_interval': args.save_interval,
            'search_algorithm': args.search_algorithm,
            'gumbel_top_k': args.gumbel_top_k,
            'gumbel_c_visit': args.gumbel_c_visit,
            'gumbel_c_scale': args.gumbel_c_scale,
        }

    # Parameter type/range spec for control file validation (shared with dashboard)
    PARAM_SPEC = {
        'lr': (float, 1e-6, 1.0),
        'train_batch': (int, 16, 8192),
        'epochs': (int, 1, 100),
        'simulations': (int, 50, 10000),
        'c_explore': (float, 0.1, 10.0),
        'risk_beta': (float, -3.0, 3.0),
        'temperature_moves': (int, 0, 200),
        'dirichlet_alpha': (float, 0.01, 2.0),
        'dirichlet_epsilon': (float, 0.0, 1.0),
        'fpu_base': (float, 0.0, 2.0),
        'opponent_risk_min': (float, -3.0, 3.0),
        'opponent_risk_max': (float, -3.0, 3.0),
        'games_per_iter': (int, 1, 10000),
        'max_fillup_factor': (int, 0, 100),
        'save_interval': (int, 1, 1000),
        'gumbel_top_k': (int, 1, 64),
        'gumbel_c_visit': (float, 1.0, 1000.0),
        'gumbel_c_scale': (float, 0.01, 10.0),
    }

    def log_current_settings():
        """Print all current training settings in structured format for LLM consumption."""
        print(f"\n{'='*70}")
        print(f"SETTINGS (iteration {iteration + 1}/{args.iterations})")
        print(f"{'='*70}")
        print(f"  Search:    simulations={args.simulations}, c_explore={args.c_explore:g}, "
              f"fpu_base={args.fpu_base:g}")
        print(f"  Explore:   dirichlet_alpha={args.dirichlet_alpha:g}, "
              f"dirichlet_epsilon={args.dirichlet_epsilon:g}, temperature_moves={args.temperature_moves}")
        print(f"  Risk:      risk_beta={args.risk_beta:g}")
        print(f"  Training:  lr={args.lr:.2e}, train_batch={args.train_batch}, epochs={args.epochs}")
        if args.priority_exponent > 0:
            print(f"  PER:       alpha={args.priority_exponent:g}, beta={args.per_beta:g}")
        print(f"  Self-play: games_per_iter={args.games_per_iter}, workers={args.workers}, "
              f"max_fillup_factor={args.max_fillup_factor}")
        if args.opponent_risk_min != 0.0 or args.opponent_risk_max != 0.0:
            print(f"  Opponent:  risk={args.opponent_risk or f'{args.opponent_risk_min:g}:{args.opponent_risk_max:g}'}")
        print(f"{'='*70}")

    def poll_control_file(phase_label):
        """Check for parameter updates from external LLM tuner via control file.

        Protocol: Claude Code writes to param_updates.json.tmp then renames to
        param_updates.json for atomic delivery (prevents partial-read race).
        """
        control_path = os.path.join(run_dir, "param_updates.json")
        if not os.path.exists(control_path):
            return
        try:
            with open(control_path, 'r') as f:
                data = json.load(f)
            # Delete after successful parse to prevent re-processing
            os.remove(control_path)
        except json.JSONDecodeError as e:
            # File might be mid-write (non-atomic); skip and retry next poll
            print(f"  [LLM@{phase_label}] Skipping malformed control file: {e}")
            return
        except OSError as e:
            print(f"  [LLM@{phase_label}] Error reading control file: {e}")
            return

        reason = data.pop('_reason', '')
        # Skip all keys starting with '_' (metadata)
        param_data = {k: v for k, v in data.items() if not k.startswith('_')}
        if not param_data:
            return

        print(f"\n  [LLM@{phase_label}] Parameter update from control file:")
        if reason:
            print(f"  [LLM@{phase_label}] Reason: {reason}")
        for key, value in param_data.items():
            if key not in PARAM_SPEC:
                print(f"  [LLM@{phase_label}] WARNING: Unknown parameter '{key}', skipped")
                continue
            # Type-cast and clamp to valid range
            ptype, pmin, pmax = PARAM_SPEC[key]
            try:
                typed_value = ptype(value)
                clamped = max(pmin, min(pmax, typed_value))
            except (ValueError, TypeError) as e:
                print(f"  [LLM@{phase_label}] WARNING: {key}={value} invalid ({e}), skipped")
                continue
            old = getattr(args, key)
            setattr(args, key, clamped)
            if key == 'lr':
                for pg in optimizer.param_groups:
                    pg['lr'] = clamped
            if clamped != typed_value:
                print(f"  [LLM@{phase_label}] {key}: {old} -> {clamped} (clamped from {typed_value})")
            else:
                print(f"  [LLM@{phase_label}] {key}: {old} -> {clamped}")

        # Update dashboard if running
        if live_dashboard is not None:
            live_dashboard.set_current_params(collect_dashboard_params())

        # Log parameter change to Claude JSONL
        if claude_iface is not None:
            claude_iface._append_jsonl({
                "type": "param_change",
                "changes": {k: getattr(args, k) for k in param_data if k in PARAM_SPEC},
                "reason": reason,
            })

    live_dashboard = None
    if args.live:
        print("\nInitializing LIVE dashboard server...", flush=True)
        try:
            from live_dashboard import LiveDashboardServer
            live_dashboard = LiveDashboardServer(port=args.dashboard_port)
            if live_dashboard.start(total_iterations=args.iterations, open_browser=True):
                print(f"  Real-time updates via WebSocket", flush=True)
                live_dashboard.set_current_params(collect_dashboard_params())
            else:
                live_dashboard = None
        except ImportError as e:
            print(f"  WARNING: Could not import live dashboard: {e}", flush=True)
            print(f"  Install requirements: pip install flask flask-socketio", flush=True)
            live_dashboard = None

    claude_iface = None
    if args.claude:
        try:
            from claude_interface import ClaudeInterface
            claude_iface = ClaudeInterface(run_dir, resume=bool(args.resume))
            claude_iface.write_header(args, {
                'input_channels': INPUT_CHANNELS,
                'num_filters': args.filters,
                'num_blocks': args.blocks,
                'num_actions': POLICY_SIZE,
                'se_reduction': args.se_reduction,
            }, start_iter)
            print(f"  Claude Code integration: {os.path.join(run_dir, 'claude_log.jsonl')}")
        except ImportError as e:
            print(f"  WARNING: Could not import claude_interface: {e}")
            claude_iface = None

    print("\n" + "=" * 60, flush=True)
    print("INITIALIZATION COMPLETE - Starting training...", flush=True)
    print("=" * 60 + "\n", flush=True)

    # Print run directory and control file paths for LLM tuner
    print(f"\n{'*'*70}")
    print(f"Run directory: {run_dir}")
    print(f"Control file:  {os.path.join(run_dir, 'param_updates.json')}")
    print(f"Stop file:     {os.path.join(run_dir, 'stop')}")
    if claude_iface is not None:
        print(f"Review file:   {os.path.join(run_dir, 'awaiting_review')}")
    print(f"{'*'*70}\n")

    # Install graceful shutdown handler
    shutdown_handler.install_handler()
    shutdown_handler.stop_file_path = os.path.join(run_dir, "stop")

    # Emergency save function
    def emergency_save(iteration_num: int, reason: str):
        """Save checkpoint and replay buffer on shutdown."""
        print(f"\n{'=' * 70}")
        print(f"EMERGENCY SAVE ({reason})")
        print(f"{'=' * 70}")

        # Save checkpoint
        emergency_path = os.path.join(run_dir, f"model_iter_{iteration_num:03d}_emergency.pt")
        safe_torch_save({
            'iteration': iteration_num,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': {
                'input_channels': INPUT_CHANNELS,
                'num_filters': args.filters,
                'num_blocks': args.blocks,
                'num_actions': POLICY_SIZE,
                'policy_filters': 2,
                'value_filters': 1,
                'value_hidden': 256,
                'simulations': args.simulations,
                'wdl': True,
                'se_reduction': args.se_reduction,
            },
            'backend': 'cpp',
            'version': '2.0',
            'emergency_save': True
        }, emergency_path)
        print(f"  Saved checkpoint: {emergency_path}")

        # Save replay buffer
        save_replay_buffer()

        # Note: training_log.jsonl is already flushed+fsynced per append — no extra save needed
        print(f"  Training log already persisted (JSONL append-only)")

        # Save Claude resume summary (if enabled)
        if claude_iface is not None:
            try:
                resume_path = claude_iface.write_resume_summary(
                    args, iteration_num, args.iterations,
                    buffer_stats=get_buffer_stats(),
                    shutdown_reason=reason,
                )
                print(f"  Saved Claude resume summary: {resume_path}")
                claude_iface._append_jsonl({
                    "type": "shutdown",
                    "iteration": iteration_num,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                print(f"  WARNING: Could not save Claude resume state: {e}")

        print(f"{'=' * 70}\n")

    def get_total_buffer_size():
        return replay_buffer.size()

    def get_buffer_stats():
        """Get buffer statistics for Claude resume summary."""
        comp = replay_buffer.get_composition()
        return {
            "total_size": replay_buffer.size(),
            "capacity": replay_buffer.capacity(),
            "wins": comp["wins"],
            "draws": comp["draws"],
            "losses": comp["losses"],
        }

    def save_replay_buffer() -> bool:
        """Save replay buffer to {run_dir}/replay_buffer.rpbf atomically.

        Writes to system temp dir first, then moves to final path to avoid
        OneDrive interference. Returns True on success, False on failure.
        """
        import tempfile
        import shutil
        buffer_path = os.path.join(run_dir, "replay_buffer.rpbf")
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".rpbf", prefix="buffer_")
        os.close(tmp_fd)
        try:
            if replay_buffer.save(tmp_path):
                shutil.move(tmp_path, buffer_path)
                comp = replay_buffer.get_composition()
                buf_mb = os.path.getsize(buffer_path) / (1024 * 1024)
                print(f"  Saved replay buffer: {buffer_path} ({buf_mb:.1f} MB, "
                      f"W={comp['wins']} D={comp['draws']} L={comp['losses']})")
                return True
            else:
                print(f"  WARNING: Failed to save replay buffer to {buffer_path}")
                return False
        except Exception as e:
            print(f"  WARNING: Replay buffer save error: {e}")
            return False
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def check_training_ready():
        """Check if buffer has enough samples for training."""
        return replay_buffer.size() > 0

    def fillup_reason():
        """Return a human-readable reason why check_training_ready() is False."""
        return f"buffer empty ({replay_buffer.size()} samples)"

    # Save initial (random) model as iter0 for fresh runs
    if start_iter == 0:
        iter0_path = os.path.join(run_dir, "model_iter_000.pt")
        safe_torch_save({
            'iteration': 0,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': {
                'input_channels': INPUT_CHANNELS,
                'num_filters': args.filters,
                'num_blocks': args.blocks,
                'num_actions': POLICY_SIZE,
                'policy_filters': 2,
                'value_filters': 1,
                'value_hidden': 256,
                'simulations': args.simulations,
                'wdl': True,
                'se_reduction': args.se_reduction,
            },
            'backend': 'cpp',
            'version': '2.0'
        }, iter0_path)
        print(f"Saved initial model: {iter0_path}")

    # Training loop (wrapped in try/except to catch and report any crash)
    iteration = start_iter  # Track for emergency save on crash
    try:
     for iteration in range(start_iter, args.iterations):
        iter_start = time.time()
        metrics = IterationMetrics(iteration=iteration + 1)

        # Check for shutdown before starting iteration
        if shutdown_handler.should_stop():
            emergency_save(iteration, "Shutdown requested before iteration")
            break

        # Apply any parameter changes from the live dashboard
        def apply_dashboard_updates(phase_label):
            if live_dashboard is None:
                return
            updates, meta = live_dashboard.poll_updates()
            if not updates:
                return
            for key, value in updates.items():
                old = getattr(args, key, None)
                setattr(args, key, value)
                # LR special case: update optimizer param groups directly
                if key == 'lr':
                    for pg in optimizer.param_groups:
                        pg['lr'] = value
                key_meta = meta.get(key, {})
                reason = key_meta.get('reason', '')
                reason_str = f" | {reason}" if reason else ""
                print(f"    [Dashboard@{phase_label}] {key}: {old} -> {value}{reason_str}")
            live_dashboard.set_current_params(collect_dashboard_params())
            live_dashboard.socketio.emit('params_applied', {
                'iteration': iteration + 1,
                'applied': list(updates.keys()),
                'phase': phase_label,
            })

        def check_export_request():
            if live_dashboard is None:
                return
            if not live_dashboard.poll_export_request():
                return
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"model_export_iter{iteration + 1:03d}_{timestamp}.pt"
            export_path = os.path.join(run_dir, filename)
            try:
                safe_torch_save({
                    'iteration': iteration + 1,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': {
                        'input_channels': INPUT_CHANNELS,
                        'num_filters': args.filters,
                        'num_blocks': args.blocks,
                        'num_actions': POLICY_SIZE,
                        'policy_filters': 2,
                        'value_filters': 1,
                        'value_hidden': 256,
                        'simulations': args.simulations,
                        'wdl': True,
                        'se_reduction': args.se_reduction,
                    },
                    'backend': 'cpp',
                    'version': '2.0'
                }, export_path)
                print(f"  [Export] Saved: {export_path}")
                live_dashboard.socketio.emit('export_complete', {
                    'status': 'saved', 'filename': filename
                })
            except Exception as e:
                print(f"  [Export] FAILED: {e}")
                live_dashboard.socketio.emit('export_complete', {
                    'status': 'error', 'error': str(e)
                })

        log_current_settings()
        poll_control_file('self-play')
        apply_dashboard_updates('self-play')
        check_export_request()

        # Re-derive search_batch and downstream sizes (gumbel_top_k may have changed)
        old_search_batch = args.search_batch
        if args.search_algorithm == "gumbel":
            args.search_batch = args.gumbel_top_k
        else:
            args.search_batch = 1
        if args.search_batch != old_search_batch:
            mirror_factor = 2 if hasattr(AlphaZeroNet, '_mirror_input') else 1
            input_batch_size = ((args.workers * args.search_batch + 31) // 32) * 32
            input_batch_size = max(input_batch_size, 32)
            args.eval_batch = input_batch_size * mirror_factor
            args.input_batch_size = input_batch_size
            print(f"    [Derived] search_batch={args.search_batch}, "
                  f"input_batch={args.input_batch_size}, "
                  f"eval_batch={args.eval_batch}")

        # Self-play phase
        replay_buffer.set_iteration(iteration + 1)  # Tag samples with current iteration
        print(f"Iteration {iteration + 1}/{args.iterations}")
        print(f"  Self-play: generating {args.games_per_iter} games...")

        if args.workers > 1:
            # ================================================================
            # PARALLEL SELF-PLAY (Cross-Game Batching)
            # ================================================================
            # Uses ParallelSelfPlayCoordinator for high GPU utilization
            # All games run concurrently, NN evals batched across games

            # Set up concurrent reanalysis if enabled and buffer has data
            reanalyzer = None
            reanalyze_this_iter = (
                args.reanalyze_fraction > 0
                and replay_buffer.size() > args.train_batch
            )
            if reanalyze_this_iter:
                import random as _rng
                reanalyze_sims = max(1, int(args.simulations * args.reanalyze_simulations_ratio))
                num_positions = int(replay_buffer.size() * args.reanalyze_fraction)
                reanalyze_workers = args.reanalyze_workers or max(1, args.workers // 2)

                reanalysis_config = alphazero_cpp.ReanalysisConfig()
                reanalysis_config.num_simulations = reanalyze_sims
                reanalysis_config.num_workers = reanalyze_workers
                reanalysis_config.gpu_batch_size = args.input_batch_size
                reanalysis_config.c_puct = args.c_explore
                reanalysis_config.fpu_base = args.fpu_base
                reanalysis_config.risk_beta = args.risk_beta
                reanalysis_config.use_gumbel = (args.search_algorithm == "gumbel")
                reanalysis_config.gumbel_top_k = args.gumbel_top_k
                reanalysis_config.gumbel_c_visit = args.gumbel_c_visit
                reanalysis_config.gumbel_c_scale = args.gumbel_c_scale
                # Match primary queue capacity so collect_batch has consistent limits
                if args.queue_capacity > 0:
                    reanalysis_config.queue_capacity = args.queue_capacity
                else:
                    reanalysis_config.queue_capacity = max(8192, args.workers * args.search_batch * 16)

                reanalyzer = alphazero_cpp.Reanalyzer(replay_buffer, reanalysis_config)
                indices = _rng.sample(range(replay_buffer.size()), num_positions)
                reanalyzer.set_indices(indices)
                print(f"  Reanalysis: {num_positions} positions, {reanalyze_sims} sims, "
                      f"{reanalyze_workers} workers (concurrent with self-play)")

            parallel_metrics, sample_game, hw_stats = run_parallel_selfplay(
                network=network,
                replay_buffer=replay_buffer,
                device=device,
                args=args,
                iteration=iteration + 1,  # 1-indexed for display
                live_dashboard=live_dashboard,
                reanalyzer=reanalyzer,
            )
            metrics.reanalysis_positions = parallel_metrics.reanalysis_positions
            metrics.reanalysis_time_s = parallel_metrics.reanalysis_time_s
            metrics.reanalysis_mean_kl = parallel_metrics.reanalysis_mean_kl
            metrics.num_games = parallel_metrics.num_games
            metrics.total_moves = parallel_metrics.total_moves
            metrics.total_simulations = parallel_metrics.total_simulations
            metrics.total_nn_evals = parallel_metrics.total_nn_evals
            metrics.white_wins = parallel_metrics.white_wins
            metrics.black_wins = parallel_metrics.black_wins
            metrics.draws = parallel_metrics.draws
            metrics.draws_repetition = parallel_metrics.draws_repetition
            metrics.draws_stalemate = parallel_metrics.draws_stalemate
            metrics.draws_fifty_move = parallel_metrics.draws_fifty_move
            metrics.draws_insufficient = parallel_metrics.draws_insufficient
            metrics.draws_max_moves = parallel_metrics.draws_max_moves
            metrics.draws_early_repetition = parallel_metrics.draws_early_repetition
            metrics.standard_wins = parallel_metrics.standard_wins
            metrics.opponent_wins = parallel_metrics.opponent_wins
            metrics.asymmetric_draws = parallel_metrics.asymmetric_draws
            metrics.selfplay_time = parallel_metrics.selfplay_time
            metrics.risk_beta = parallel_metrics.risk_beta

            # Fill-up: play additional batches if buffer not ready (capped)
            fillup_rounds = 0
            while not check_training_ready() and not shutdown_handler.should_stop():
                if fillup_rounds >= args.max_fillup_factor:
                    print(f"  WARNING: Buffer not ready after {fillup_rounds} extra rounds "
                          f"({fillup_rounds * args.games_per_iter} games), training with available data")
                    break
                print(f"  Buffer fill-up: {fillup_reason()}, playing {args.games_per_iter} more games...")
                extra_metrics, _, _ = run_parallel_selfplay(
                    network=network,
                    replay_buffer=replay_buffer,
                    device=device,
                    args=args,
                    iteration=iteration + 1,
                    live_dashboard=live_dashboard,
                )
                metrics.num_games += extra_metrics.num_games
                metrics.total_moves += extra_metrics.total_moves
                metrics.total_simulations += extra_metrics.total_simulations
                metrics.total_nn_evals += extra_metrics.total_nn_evals
                metrics.white_wins += extra_metrics.white_wins
                metrics.black_wins += extra_metrics.black_wins
                metrics.draws += extra_metrics.draws
                metrics.draws_repetition += extra_metrics.draws_repetition
                metrics.draws_stalemate += extra_metrics.draws_stalemate
                metrics.draws_fifty_move += extra_metrics.draws_fifty_move
                metrics.draws_insufficient += extra_metrics.draws_insufficient
                metrics.draws_max_moves += extra_metrics.draws_max_moves
                metrics.draws_early_repetition += extra_metrics.draws_early_repetition
                metrics.standard_wins += extra_metrics.standard_wins
                metrics.opponent_wins += extra_metrics.opponent_wins
                metrics.asymmetric_draws += extra_metrics.asymmetric_draws
                metrics.selfplay_time += extra_metrics.selfplay_time
                fillup_rounds += 1

            metrics.avg_game_length = metrics.total_moves / max(metrics.num_games, 1)

            # Save sample game as PGN
            if not args.no_sample_games and sample_game is not None:
                save_sample_game_pgn(
                    run_dir, iteration + 1,
                    sample_game["moves"],
                    sample_game["result"],
                    sample_game["num_moves"]
                )

            # Print sample game to console for LLM consumption
            if sample_game is not None:
                result = sample_game.get("result", "?")
                n_moves = sample_game.get("num_moves", 0)
                reason = sample_game.get("result_reason", "")
                reason_str = f" ({reason})" if reason else ""
                moves = sample_game.get("moves", [])
                if len(moves) <= 30:
                    move_str = " ".join(moves)
                else:
                    move_str = " ".join(moves[:20]) + f" ... ({len(moves)-25} moves) ... " + " ".join(moves[-5:])
                print(f"  Sample Game: {result} in {n_moves} moves{reason_str}")
                print(f"    Moves: {move_str}")
        else:
            # ================================================================
            # SEQUENTIAL SELF-PLAY (Original)
            # ================================================================
            # Games played one-by-one, each game makes its own NN calls
            network.eval()
            selfplay_start = time.time()
            games_completed = 0
            seq_sample_game = None  # Track a sample game for PGN export

            # Initialize progress reporter for this iteration
            progress = ProgressReporter(interval=args.progress_interval)

            # Track time for live dashboard updates (every 5 seconds)
            last_live_update = time.time()
            live_update_interval = 5.0  # seconds

            game_idx = 0
            filling_buffer = False
            max_fillup_games = args.games_per_iter * args.max_fillup_factor
            while True:
                if game_idx >= args.games_per_iter:
                    if check_training_ready():
                        break
                    if game_idx >= args.games_per_iter + max_fillup_games:
                        print(f"  WARNING: Buffer not ready after {game_idx} games "
                              f"(cap: {max_fillup_games} extra), training with available data")
                        break
                    if not filling_buffer:
                        filling_buffer = True
                        print(f"  Buffer fill-up: {fillup_reason()}, playing additional games...")
                # Check for shutdown between games
                if shutdown_handler.should_stop():
                    print(f"  Stopping after {game_idx} games (shutdown requested)")
                    break

                obs_list, policy_list, result, num_moves, total_sims, total_evals, moves_uci, result_reason = self_play.play_game()
                games_completed += 1

                # Track sample game (prefer decisive games)
                is_decisive = (result != 0)
                if seq_sample_game is None or (not seq_sample_game["is_decisive"] and is_decisive):
                    result_str = "1-0" if result > 0 else "0-1" if result < 0 else "1/2-1/2"
                    seq_sample_game = {
                        "moves": moves_uci,
                        "result": result_str,
                        "result_reason": result_reason,
                        "num_moves": num_moves,
                        "is_decisive": is_decisive
                    }

                # Add game data to C++ ReplayBuffer
                # Flatten observations for storage: (123, 8, 8) -> (7872,)
                # Value must be from side-to-move perspective (matches C++ game.hpp::set_outcomes)
                for i, (obs, policy) in enumerate(zip(obs_list, policy_list)):
                    obs_flat = obs.flatten().astype(np.float32)  # (7872,)
                    white_to_move = (i % 2 == 0)
                    if result == 1.0:  # White won
                        value = 1.0 if white_to_move else -1.0
                    elif result == -1.0:  # Black won
                        value = -1.0 if white_to_move else 1.0
                    else:  # Draw — pure training label (risk_beta is search-time only)
                        value = 0.0
                    replay_buffer.add_sample(obs_flat, policy.astype(np.float32), float(value))

                metrics.total_moves += num_moves
                metrics.total_simulations += total_sims
                metrics.total_nn_evals += total_evals

                if result == 1.0:
                    metrics.white_wins += 1
                elif result == -1.0:
                    metrics.black_wins += 1
                else:
                    metrics.draws += 1
                    # Draw breakdown (sequential path)
                    if result_reason == "repetition":
                        metrics.draws_repetition += 1
                        if num_moves < 60:
                            metrics.draws_early_repetition += 1
                    elif result_reason == "stalemate":
                        metrics.draws_stalemate += 1
                    elif result_reason == "fifty_move":
                        metrics.draws_fifty_move += 1
                    elif result_reason == "insufficient":
                        metrics.draws_insufficient += 1
                    elif result_reason == "max_moves":
                        metrics.draws_max_moves += 1

                # Update progress tracker
                progress.update(num_moves, total_sims, total_evals)

                # Print periodic progress report (every 30 seconds)
                if progress.should_report():
                    progress.report(max(args.games_per_iter, game_idx + 1), get_total_buffer_size())

                # Push live dashboard update every 5 seconds
                now = time.time()
                if live_dashboard is not None and (now - last_live_update) >= live_update_interval:
                    elapsed_selfplay = now - selfplay_start
                    live_dashboard.push_progress(
                        iteration=iteration + 1,
                        games_completed=games_completed,
                        total_games=max(args.games_per_iter, game_idx + 1),
                        moves=progress.total_moves,
                        sims=progress.total_sims,
                        evals=progress.total_evals,
                        elapsed_time=elapsed_selfplay,
                        buffer_size=get_total_buffer_size(),
                        phase="selfplay"
                    )
                    last_live_update = now

                game_idx += 1

            metrics.num_games = games_completed
            metrics.selfplay_time = time.time() - selfplay_start
            metrics.avg_game_length = metrics.total_moves / max(metrics.num_games, 1)

            # Save sample game as PGN (sequential path)
            if not args.no_sample_games and seq_sample_game is not None:
                save_sample_game_pgn(
                    run_dir, iteration + 1,
                    seq_sample_game["moves"],
                    seq_sample_game["result"],
                    seq_sample_game["num_moves"]
                )

            # Print sample game to console for LLM consumption
            if seq_sample_game is not None:
                result = seq_sample_game.get("result", "?")
                n_moves = seq_sample_game.get("num_moves", 0)
                moves = seq_sample_game.get("moves", [])
                if len(moves) <= 30:
                    move_str = " ".join(moves)
                else:
                    move_str = " ".join(moves[:20]) + f" ... ({len(moves)-25} moves) ... " + " ".join(moves[-5:])
                print(f"  Sample Game: {result} in {n_moves} moves")
                print(f"    Moves: {move_str}")

            # Make sample_game available for claude logging
            sample_game = seq_sample_game
            hw_stats = {}  # No hardware metrics in sequential mode

        # Create selfplay_done safety lock (before training begins)
        selfplay_done_path = None
        if claude_iface is not None and args.claude_timeout > 0:
            selfplay_done_path = os.path.join(run_dir, "selfplay_done")
            with open(selfplay_done_path, 'w') as f:
                f.write(str(iteration + 1))
                f.flush()
                os.fsync(f.fileno())

        # Persist replay buffer after self-play (before training begins)
        save_replay_buffer()

        # Handle shutdown after self-play
        if shutdown_handler.should_stop():
            emergency_save(iteration + 1, f"Shutdown after self-play ({metrics.num_games} games)")
            break

        # Training phase — fill-up may have been capped, so check for empty buffer
        if not shutdown_handler.should_stop():
            total_buf = get_total_buffer_size()
            if total_buf == 0:
                print("  Skipping training (buffer empty)")
            else:
                comp = replay_buffer.get_composition()
                print(f"  Training: {args.epochs} epochs on {total_buf} positions "
                      f"(W={comp['wins']} D={comp['draws']} L={comp['losses']})...")

                # Update live dashboard to show training phase
                if live_dashboard is not None:
                    live_dashboard.push_progress(
                        iteration=iteration + 1,
                        games_completed=metrics.num_games,
                        total_games=metrics.num_games,
                        moves=metrics.total_moves,
                        sims=metrics.total_simulations,
                        evals=metrics.total_nn_evals,
                        elapsed_time=metrics.selfplay_time,
                        buffer_size=get_total_buffer_size(),
                        phase="training"
                    )

                poll_control_file('training')
                apply_dashboard_updates('training')
                check_export_request()

                train_start = time.time()

                # PER beta (fixed, not annealed)
                current_per_beta = args.per_beta if args.priority_exponent > 0 else 0.0

                train_metrics = train_iteration(
                    network, optimizer, replay_buffer,
                    args.train_batch, args.epochs, device, scaler,
                    per_beta=current_per_beta,
                )

                metrics.train_time = time.time() - train_start
                metrics.loss = train_metrics['loss']
                metrics.policy_loss = train_metrics['policy_loss']
                metrics.value_loss = train_metrics['value_loss']
                metrics.num_train_batches = train_metrics['num_batches']
                metrics.grad_norm_avg = train_metrics.get('grad_norm_avg', 0.0)
                metrics.grad_norm_max = train_metrics.get('grad_norm_max', 0.0)
                metrics.per_beta = current_per_beta

                if 'error' in train_metrics:
                    print(f"  Training ERROR: {train_metrics['error']}")
                else:
                    grad_str = ""
                    if metrics.grad_norm_avg > 0:
                        grad_str = f", grad_norm={metrics.grad_norm_avg:.2f}/{metrics.grad_norm_max:.2f}"
                    nan_skip = train_metrics.get('nan_skip_count', 0)
                    nan_str = f" [NaN skipped {nan_skip}]" if nan_skip > 0 else ""
                    print(f"  Training complete: loss={metrics.loss:.4f} "
                          f"(policy={metrics.policy_loss:.4f}, value={metrics.value_loss:.4f}{grad_str}{nan_str}) "
                          f"in {metrics.train_time:.1f}s")

        metrics.buffer_size = get_total_buffer_size()
        metrics.total_time = time.time() - iter_start
        check_export_request()

        # Track metrics for console output
        metrics_tracker.add_iteration(metrics)
        metrics_tracker.print_iteration_summary(metrics, args)

        # Append iteration record to JSONL (crash-safe, no in-memory dict)
        record = {
            "type": "iteration",
            "iteration": iteration + 1,
            "timestamp": datetime.now().isoformat(),
            # Training losses
            "loss": metrics.loss,
            "policy_loss": metrics.policy_loss,
            "value_loss": metrics.value_loss,
            # Self-play game stats
            "games": metrics.num_games,
            "moves": metrics.total_moves,
            "simulations": metrics.total_simulations,
            "nn_evals": metrics.total_nn_evals,
            "white_wins": metrics.white_wins,
            "black_wins": metrics.black_wins,
            "draws": metrics.draws,
            "draws_repetition": metrics.draws_repetition,
            "draws_early_repetition": metrics.draws_early_repetition,
            "draws_stalemate": metrics.draws_stalemate,
            "draws_fifty_move": metrics.draws_fifty_move,
            "draws_insufficient": metrics.draws_insufficient,
            "draws_max_moves": metrics.draws_max_moves,
            "avg_game_length": metrics.avg_game_length,
            "buffer_size": metrics.buffer_size,
            "buffer_composition": replay_buffer.get_composition(),
            "selfplay_time": metrics.selfplay_time,
            "train_time": metrics.train_time,
            "total_time": metrics.total_time,
            "per_beta": metrics.per_beta,
            # Training diagnostics
            "risk_beta": metrics.risk_beta,
            "grad_norm_avg": metrics.grad_norm_avg,
            "grad_norm_max": metrics.grad_norm_max,
            "learning_rate": optimizer.param_groups[0]['lr'],
            # Reanalysis metrics
            "reanalysis_positions": metrics.reanalysis_positions,
            "reanalysis_time_s": metrics.reanalysis_time_s,
            "reanalysis_mean_kl": metrics.reanalysis_mean_kl,
        }
        # Merge hardware/performance stats (empty dict in sequential mode)
        record.update(hw_stats)
        append_training_log(run_dir, record)

        # Log iteration to Claude JSONL (if enabled)
        if claude_iface is not None:
            claude_iface.log_iteration(
                iteration=iteration + 1,
                total_iterations=args.iterations,
                params=collect_dashboard_params(),
                metrics=metrics,
                sample_game=sample_game,
            )

        # Synchronous Claude review handshake (--claude with non-zero timeout)
        # Dual-signal: selfplay_done (created earlier) + awaiting_review (created now)
        # Training blocks until BOTH are deleted by the reviewing agent.
        if claude_iface is not None and args.claude_timeout > 0:
            awaiting_path = os.path.join(run_dir, "awaiting_review")
            with open(awaiting_path, 'w') as f:
                f.write(str(iteration + 1))
                f.flush()
                os.fsync(f.fileno())

            print(f"  Awaiting Claude review... (delete both selfplay_done AND awaiting_review to continue)")
            wait_start = time.time()
            timeout = args.claude_timeout
            while True:
                time.sleep(1.0)
                selfplay_exists = selfplay_done_path is not None and os.path.exists(selfplay_done_path)
                review_exists = os.path.exists(awaiting_path)
                if not selfplay_exists and not review_exists:
                    break
                if shutdown_handler.should_stop():
                    for p in [selfplay_done_path, awaiting_path]:
                        if p:
                            try:
                                os.remove(p)
                            except OSError:
                                pass
                    break
                if time.time() - wait_start > timeout:
                    print(f"  Claude review timeout ({timeout}s) — auto-continuing")
                    for p in [selfplay_done_path, awaiting_path]:
                        if p:
                            try:
                                os.remove(p)
                            except OSError:
                                pass
                    break

            # Apply any parameter changes Claude wrote during review
            poll_control_file('claude-review')

        # Save checkpoint and training metrics BEFORE evaluation (at save_interval)
        save_checkpoint_now = (iteration + 1) % args.save_interval == 0 or iteration == args.iterations - 1
        if save_checkpoint_now:
            # Save model checkpoint
            checkpoint_name = f"model_iter_{iteration + 1:03d}.pt"
            checkpoint_save_path = os.path.join(run_dir, checkpoint_name)
            safe_torch_save({
                'iteration': iteration + 1,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': {
                    'input_channels': INPUT_CHANNELS,
                    'num_filters': args.filters,
                    'num_blocks': args.blocks,
                    'num_actions': POLICY_SIZE,
                    'policy_filters': 2,
                    'value_filters': 1,
                    'value_hidden': 256,
                    'simulations': args.simulations,
                    'wdl': True,
                    'se_reduction': args.se_reduction,
                },
                'backend': 'cpp',
                'version': '2.0'
            }, checkpoint_save_path)
            print(f"  Saved checkpoint: {checkpoint_save_path}")

            # Note: training_log.jsonl already flushed per-iteration (no batch save needed)

            # Generate summary.html (at save_interval)
            if not args.no_visualization:
                summary_path = generate_summary_html(run_dir, config)
                print(f"  Updated summary: {summary_path}")

        # Update live dashboard (WebSocket)
        if live_dashboard is not None:
            live_dashboard.push_metrics(metrics)

        # Save final checkpoint (at save_interval)
        if save_checkpoint_now:
            is_final = (iteration == args.iterations - 1)

            # Save final checkpoint alias
            if is_final:
                final_path = os.path.join(run_dir, "model_final.pt")
                safe_torch_save({
                    'iteration': iteration + 1,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': {
                        'input_channels': INPUT_CHANNELS,
                        'num_filters': args.filters,
                        'num_blocks': args.blocks,
                        'num_actions': POLICY_SIZE,
                        'policy_filters': 2,
                        'value_filters': 1,
                        'value_hidden': 256,
                        'simulations': args.simulations,
                        'wdl': True,
                        'se_reduction': args.se_reduction,
                    },
                    'backend': 'cpp',
                    'version': '2.0'
                }, final_path)
                print(f"  Saved final checkpoint: {final_path}")

            print()

        # Check for shutdown after iteration complete
        if shutdown_handler.should_stop():
            emergency_save(iteration + 1, "Shutdown after training")
            break
    except Exception as e:
        import traceback
        print(f"\n{'=' * 70}")
        print(f"CRASH DETECTED: {type(e).__name__}: {e}")
        print(f"{'=' * 70}")
        traceback.print_exc()
        emergency_save(iteration + 1, f"Crash: {e}")
        raise

    # Uninstall handler
    shutdown_handler.uninstall_handler()

    if live_dashboard is not None:
        live_dashboard.complete()
        print(f"\n  Live dashboard still running at http://127.0.0.1:{args.dashboard_port}")
        print(f"  Press Ctrl+C to exit completely")

    # Final summary
    if shutdown_handler.should_stop():
        print("\n" + "=" * 70)
        print("TRAINING INTERRUPTED - Graceful shutdown complete")
        print("=" * 70)
        print(f"  Run directory: {run_dir}")
        print(f"  Resume with: --resume {run_dir}")
    else:
        metrics_tracker.print_final_summary(args)

    if not shutdown_handler.should_stop():
        # Write Claude resume summary on normal completion
        if claude_iface is not None:
            try:
                resume_path = claude_iface.write_resume_summary(
                    args, args.iterations, args.iterations,
                    buffer_stats=get_buffer_stats(),
                    shutdown_reason="Training complete",
                )
                print(f"  Saved Claude resume summary: {resume_path}")
                claude_iface._append_jsonl({
                    "type": "shutdown",
                    "iteration": args.iterations,
                    "reason": "Training complete",
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                print(f"  WARNING: Could not save Claude resume state: {e}")

        print(f"\n{'=' * 70}")
        print(f"TRAINING COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Run directory:     {run_dir}")
        print(f"  Final checkpoint:  {os.path.join(run_dir, 'model_final.pt')}")
        if not args.no_visualization:
            print(f"  Training summary:  {os.path.join(run_dir, 'summary.html')}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
