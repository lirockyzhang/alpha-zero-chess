#!/usr/bin/env python3
"""
AlphaZero Training with C++ Backend

This script uses:
- C++ MCTS (alphazero-cpp) for fast tree search with proper leaf evaluation
- C++ ReplayBuffer for high-performance data storage with optional persistence
- CUDA for neural network inference and training
- 192x15 network architecture (192 filters, 15 residual blocks)
- 122-channel position encoding

Output Directory Structure:
    Each training run creates an organized directory:
    checkpoints/
    └── f192-b15_2024-02-03_14-30-00/    # {filters}-{blocks}_{timestamp}
        ├── model_iter_001.pt            # Checkpoints (every N iterations)
        ├── model_iter_005.pt
        ├── model_final.pt               # Final checkpoint
        ├── training_metrics.json        # Loss, games, moves per iteration
        ├── summary.html                 # Training summary (default, unless --no-visualization)
        ├── evaluation_results.json      # Evaluation metrics per checkpoint
        └── replay_buffer.rpbf           # Always saved; use --load-buffer to load on resume

Usage:
    # Basic training (recommended starting point)
    uv run python alphazero-cpp/scripts/train.py

    # Custom parameters
    uv run python alphazero-cpp/scripts/train.py --iterations 50 --games-per-iter 100 --simulations 800

    # Resume from checkpoint (continues in same run directory)
    uv run python alphazero-cpp/scripts/train.py --resume checkpoints/f192-b15_2024-02-03_14-30-00/model_iter_005.pt

    # Resume from run directory (finds latest checkpoint)
    uv run python alphazero-cpp/scripts/train.py --resume checkpoints/f192-b15_2024-02-03_14-30-00

    # With buffer loading (buffer always saved; --load-buffer loads on startup)
    uv run python alphazero-cpp/scripts/train.py --load-buffer

Parameters:
    --iterations        Number of training iterations (default: 100)
    --games-per-iter    Self-play games per iteration (default: 50)
    --simulations       MCTS simulations per move (default: 800)
    --search-batch      Leaves to evaluate per MCTS iteration (default: 32)
    --train-batch       Samples per training gradient step (default: 256)
    --lr                Learning rate (default: 0.001)
    --filters           Network filters (default: 192)
    --blocks            Residual blocks (default: 15)
    --buffer-size       Replay buffer size (default: 100000)
    --epochs            Training epochs per iteration (default: 5)
    --temperature-moves Moves with temperature=1 (default: 30)
    --c-puct            MCTS exploration constant (default: 1.5)
    --device            Device: cuda or cpu (default: cuda)
    --save-dir          Base checkpoint directory (default: checkpoints)
    --resume            Resume from checkpoint path or run directory
    --save-interval     Save checkpoint every N iterations (default: 5)
    --load-buffer        Load replay buffer on startup (always saved regardless)
    --no-visualization  Disable summary.html generation (default: enabled)
    --no-eval           Skip evaluation before checkpoint (default: enabled)

    Parallel Self-Play (enabled automatically when --workers > 1):
    --workers              Self-play workers. 1=sequential, >1=parallel (default: 1)
    --eval-batch           Max positions per GPU call in parallel mode (default: 512)
    --gpu-batch-timeout-ms GPU batch collection timeout in ms (default: 20)
    --worker-timeout-ms    Worker wait time for NN results in ms (default: 2000)
    --dirichlet-alpha      Dirichlet noise alpha for root exploration (default: 0.3)
    --dirichlet-epsilon    Dirichlet noise weight for root exploration (default: 0.25)

    Parameter Relationships (for parallel mode):
    --eval-batch should be >= --search-batch * --workers for optimal GPU utilization.
    --search-batch is per-game MCTS leaf batch size; --eval-batch is the cross-game GPU batch.
"""

import argparse
import json
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

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


# =============================================================================
# Constants
# =============================================================================

# 122-channel encoding (8 history * 14 piece planes + 8 repetition + 2 color/castling)
INPUT_CHANNELS = 122
POLICY_SIZE = 4672


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
    """Handles graceful shutdown on Ctrl+C (SIGINT).

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

    def request_shutdown(self, signum=None, frame=None):
        """Request graceful shutdown."""
        with self._lock:
            if not self.shutdown_requested:
                self.shutdown_requested = True
                print("\n\n" + "!" * 70)
                print("! SHUTDOWN REQUESTED - Finishing current game and saving...")
                print("! Press Ctrl+C again to force quit (may lose data)")
                print("!" * 70 + "\n")

                # Restore original handler for force quit
                if self._original_handler:
                    signal.signal(signal.SIGINT, self._original_handler)

    def should_stop(self) -> bool:
        """Check if shutdown was requested."""
        with self._lock:
            return self.shutdown_requested

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
# Visual Training Dashboard (Plotly)
# =============================================================================

class TrainingDashboard:
    """Interactive visual dashboard for monitoring training progress.

    Creates an HTML dashboard with live-updating charts showing:
    - Loss curves (total, policy, value)
    - Performance metrics (moves/s, sims/s, NN evals/s)
    - Game statistics (win/draw/loss, game length)
    - Resource utilization (buffer size, iteration times)

    The dashboard auto-refreshes and can be viewed in a browser during training.
    """

    def __init__(self, output_dir: str = "training_dashboard", update_interval: int = 1):
        """Initialize the dashboard.

        Args:
            output_dir: Directory to save dashboard files
            update_interval: Update dashboard every N iterations
        """
        self.output_dir = Path(output_dir)
        self.update_interval = update_interval
        self.enabled = False

        # Data storage for all metrics
        self.data = {
            # Iteration tracking
            'iterations': [],
            'timestamps': [],
            'elapsed_minutes': [],

            # Loss metrics
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],

            # Self-play performance
            'moves_per_sec': [],
            'sims_per_sec': [],
            'nn_evals_per_sec': [],
            'games_per_hour': [],

            # Game statistics
            'white_wins': [],
            'black_wins': [],
            'draws': [],
            'avg_game_length': [],
            'total_games_cumulative': [],

            # Buffer and timing
            'buffer_size': [],
            'selfplay_time': [],
            'train_time': [],
            'iteration_time': [],

            # Cumulative metrics
            'total_moves_cumulative': [],
            'total_sims_cumulative': [],
        }

        self.start_time = None
        self.total_games = 0
        self.total_moves = 0
        self.total_sims = 0

    def enable(self):
        """Enable the dashboard and create output directory."""
        try:
            import plotly
            self.enabled = True
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.start_time = time.time()
            print(f"  Dashboard enabled: {self.output_dir}/dashboard.html")
            return True
        except ImportError:
            print("  WARNING: plotly not installed. Install with: pip install plotly")
            print("  Dashboard disabled.")
            self.enabled = False
            return False

    def add_iteration(self, metrics: 'IterationMetrics'):
        """Add metrics from a completed iteration."""
        if not self.enabled:
            return

        # Update cumulative counters
        self.total_games += metrics.num_games
        self.total_moves += metrics.total_moves
        self.total_sims += metrics.total_simulations

        # Calculate rates
        moves_per_sec = metrics.total_moves / metrics.selfplay_time if metrics.selfplay_time > 0 else 0
        sims_per_sec = metrics.total_simulations / metrics.selfplay_time if metrics.selfplay_time > 0 else 0
        nn_evals_per_sec = metrics.total_nn_evals / metrics.selfplay_time if metrics.selfplay_time > 0 else 0
        games_per_hour = metrics.num_games / metrics.selfplay_time * 3600 if metrics.selfplay_time > 0 else 0

        elapsed = time.time() - self.start_time

        # Store all metrics
        self.data['iterations'].append(metrics.iteration)
        self.data['timestamps'].append(datetime.now().strftime('%H:%M:%S'))
        self.data['elapsed_minutes'].append(elapsed / 60)

        self.data['total_loss'].append(metrics.loss if metrics.loss > 0 else None)
        self.data['policy_loss'].append(metrics.policy_loss if metrics.policy_loss > 0 else None)
        self.data['value_loss'].append(metrics.value_loss if metrics.value_loss > 0 else None)

        self.data['moves_per_sec'].append(moves_per_sec)
        self.data['sims_per_sec'].append(sims_per_sec)
        self.data['nn_evals_per_sec'].append(nn_evals_per_sec)
        self.data['games_per_hour'].append(games_per_hour)

        self.data['white_wins'].append(metrics.white_wins)
        self.data['black_wins'].append(metrics.black_wins)
        self.data['draws'].append(metrics.draws)
        self.data['avg_game_length'].append(metrics.avg_game_length)
        self.data['total_games_cumulative'].append(self.total_games)

        self.data['buffer_size'].append(metrics.buffer_size)
        self.data['selfplay_time'].append(metrics.selfplay_time)
        self.data['train_time'].append(metrics.train_time)
        self.data['iteration_time'].append(metrics.total_time)

        self.data['total_moves_cumulative'].append(self.total_moves)
        self.data['total_sims_cumulative'].append(self.total_sims)

        # Update dashboard every N iterations
        if metrics.iteration % self.update_interval == 0:
            self._generate_dashboard()

    def _generate_dashboard(self):
        """Generate the HTML dashboard with all charts."""
        if not self.enabled or len(self.data['iterations']) == 0:
            return

        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            import plotly.io as pio
        except ImportError:
            return

        # Create subplot grid: 3 rows x 3 cols
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                '📉 Training Loss', '⚡ Performance (moves/s)', '🎮 Games per Hour',
                '🎯 Policy vs Value Loss', '🔬 MCTS Simulations/s', '📊 Win/Draw/Loss Distribution',
                '💾 Replay Buffer Size', '⏱️ Iteration Time Breakdown', '📈 Cumulative Progress'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )

        iterations = self.data['iterations']
        colors = {
            'primary': '#3498db',
            'secondary': '#2ecc71',
            'tertiary': '#e74c3c',
            'quaternary': '#9b59b6',
            'white': '#ecf0f1',
            'draw': '#95a5a6',
            'black': '#2c3e50'
        }

        # Row 1, Col 1: Training Loss
        if any(l is not None for l in self.data['total_loss']):
            fig.add_trace(go.Scatter(
                x=iterations, y=self.data['total_loss'],
                mode='lines+markers', name='Total Loss',
                line=dict(color=colors['primary'], width=2),
                marker=dict(size=6)
            ), row=1, col=1)

        # Row 1, Col 2: Moves per second
        fig.add_trace(go.Scatter(
            x=iterations, y=self.data['moves_per_sec'],
            mode='lines+markers', name='Moves/s',
            line=dict(color=colors['secondary'], width=2),
            marker=dict(size=6),
            fill='tozeroy', fillcolor='rgba(46, 204, 113, 0.2)'
        ), row=1, col=2)

        # Row 1, Col 3: Games per hour
        fig.add_trace(go.Scatter(
            x=iterations, y=self.data['games_per_hour'],
            mode='lines+markers', name='Games/hour',
            line=dict(color=colors['quaternary'], width=2),
            marker=dict(size=6)
        ), row=1, col=3)

        # Row 2, Col 1: Policy vs Value Loss
        if any(l is not None for l in self.data['policy_loss']):
            fig.add_trace(go.Scatter(
                x=iterations, y=self.data['policy_loss'],
                mode='lines+markers', name='Policy Loss',
                line=dict(color=colors['primary'], width=2)
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=iterations, y=self.data['value_loss'],
                mode='lines+markers', name='Value Loss',
                line=dict(color=colors['tertiary'], width=2)
            ), row=2, col=1)

        # Row 2, Col 2: MCTS Simulations per second
        fig.add_trace(go.Scatter(
            x=iterations, y=self.data['sims_per_sec'],
            mode='lines+markers', name='Sims/s',
            line=dict(color=colors['primary'], width=2),
            fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.2)'
        ), row=2, col=2)

        # Row 2, Col 3: Win/Draw/Loss distribution (stacked bar for last 10 iterations)
        last_n = min(10, len(iterations))
        if last_n > 0:
            recent_iters = iterations[-last_n:]
            fig.add_trace(go.Bar(
                x=recent_iters, y=self.data['white_wins'][-last_n:],
                name='White Wins', marker_color=colors['white'],
                text=self.data['white_wins'][-last_n:], textposition='inside'
            ), row=2, col=3)
            fig.add_trace(go.Bar(
                x=recent_iters, y=self.data['draws'][-last_n:],
                name='Draws', marker_color=colors['draw']
            ), row=2, col=3)
            fig.add_trace(go.Bar(
                x=recent_iters, y=self.data['black_wins'][-last_n:],
                name='Black Wins', marker_color=colors['black']
            ), row=2, col=3)

        # Row 3, Col 1: Replay Buffer Size
        fig.add_trace(go.Scatter(
            x=iterations, y=self.data['buffer_size'],
            mode='lines+markers', name='Buffer Size',
            line=dict(color=colors['secondary'], width=2),
            fill='tozeroy', fillcolor='rgba(46, 204, 113, 0.2)'
        ), row=3, col=1)

        # Row 3, Col 2: Iteration Time Breakdown (stacked bar for last 10)
        if last_n > 0:
            fig.add_trace(go.Bar(
                x=recent_iters, y=self.data['selfplay_time'][-last_n:],
                name='Self-Play Time', marker_color=colors['primary']
            ), row=3, col=2)
            fig.add_trace(go.Bar(
                x=recent_iters, y=self.data['train_time'][-last_n:],
                name='Train Time', marker_color=colors['tertiary']
            ), row=3, col=2)

        # Row 3, Col 3: Cumulative Progress (games and moves)
        fig.add_trace(go.Scatter(
            x=iterations, y=self.data['total_games_cumulative'],
            mode='lines', name='Total Games',
            line=dict(color=colors['primary'], width=2)
        ), row=3, col=3)

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'🎯 AlphaZero Training Dashboard - Iteration {iterations[-1] if iterations else 0}',
                font=dict(size=24, color='#2c3e50'),
                x=0.5
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            ),
            height=1000,
            template='plotly_white',
            font=dict(family="Arial, sans-serif"),
            margin=dict(t=80, b=100, l=60, r=60)
        )

        # Update axes labels
        fig.update_xaxes(title_text="Iteration", row=3, col=1)
        fig.update_xaxes(title_text="Iteration", row=3, col=2)
        fig.update_xaxes(title_text="Iteration", row=3, col=3)

        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Moves/sec", row=1, col=2)
        fig.update_yaxes(title_text="Games/hour", row=1, col=3)
        fig.update_yaxes(title_text="Loss", row=2, col=1)
        fig.update_yaxes(title_text="Sims/sec", row=2, col=2)
        fig.update_yaxes(title_text="Games", row=2, col=3)
        fig.update_yaxes(title_text="Positions", row=3, col=1)
        fig.update_yaxes(title_text="Seconds", row=3, col=2)
        fig.update_yaxes(title_text="Count", row=3, col=3)

        # Stack bars for win/loss and time breakdown
        fig.update_layout(barmode='stack')

        # Add auto-refresh meta tag
        html_content = pio.to_html(fig, include_plotlyjs=True, full_html=True)

        # Inject auto-refresh (every 30 seconds)
        refresh_script = """
        <script>
            setTimeout(function() {
                location.reload();
            }, 30000);
        </script>
        <style>
            body { background-color: #f5f6fa; }
            .plotly-graph-div { background-color: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        </style>
        """
        html_content = html_content.replace('</head>', f'{refresh_script}</head>')

        # Add header with stats
        elapsed = time.time() - self.start_time if self.start_time else 0
        header_html = f"""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin-bottom: 20px; border-radius: 10px;">
            <h1 style="margin: 0;">🧠 AlphaZero Training Monitor</h1>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">
                Elapsed: {timedelta(seconds=int(elapsed))} |
                Games: {self.total_games:,} |
                Moves: {self.total_moves:,} |
                Simulations: {self.total_sims:,}
            </p>
            <p style="margin: 5px 0 0 0; font-size: 12px; opacity: 0.7;">Auto-refreshes every 30 seconds</p>
        </div>
        """
        html_content = html_content.replace('<body>', f'<body>{header_html}')

        # Save dashboard
        dashboard_path = self.output_dir / 'dashboard.html'
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Also save raw data as JSON for external analysis
        self._save_data_json()

    def _save_data_json(self):
        """Save raw metrics data as JSON."""
        import json
        data_path = self.output_dir / 'metrics.json'
        with open(data_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)

    def finalize(self):
        """Generate final dashboard and summary."""
        if not self.enabled:
            return

        self._generate_dashboard()
        self._generate_summary_report()
        print(f"\n  📊 Final dashboard saved: {self.output_dir}/dashboard.html")
        print(f"  📄 Training summary saved: {self.output_dir}/summary.html")
        print(f"  📁 Raw metrics saved: {self.output_dir}/metrics.json")

    def _generate_summary_report(self):
        """Generate a final summary report."""
        if not self.enabled or len(self.data['iterations']) == 0:
            return

        elapsed = time.time() - self.start_time if self.start_time else 0

        # Calculate summary statistics
        avg_moves_per_sec = sum(self.data['moves_per_sec']) / len(self.data['moves_per_sec']) if self.data['moves_per_sec'] else 0
        avg_sims_per_sec = sum(self.data['sims_per_sec']) / len(self.data['sims_per_sec']) if self.data['sims_per_sec'] else 0
        avg_games_per_hour = sum(self.data['games_per_hour']) / len(self.data['games_per_hour']) if self.data['games_per_hour'] else 0

        total_white = sum(self.data['white_wins'])
        total_draws = sum(self.data['draws'])
        total_black = sum(self.data['black_wins'])
        total_games = total_white + total_draws + total_black

        # Loss improvement
        valid_losses = [l for l in self.data['total_loss'] if l is not None and l > 0]
        loss_improvement = ((valid_losses[0] - valid_losses[-1]) / valid_losses[0] * 100) if len(valid_losses) >= 2 else 0

        summary_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AlphaZero Training Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #f5f6fa; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 20px; }}
                .card {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .card h2 {{ color: #2c3e50; margin-top: 0; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .stat {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                .stat-label {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; }}
                .progress-bar {{ height: 20px; background: #ecf0f1; border-radius: 10px; overflow: hidden; margin-top: 10px; }}
                .progress-fill {{ height: 100%; background: linear-gradient(90deg, #3498db, #2ecc71); }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 AlphaZero Training Complete</h1>
                    <p>Training Duration: {timedelta(seconds=int(elapsed))}</p>
                </div>

                <div class="card">
                    <h2>📊 Overall Statistics</h2>
                    <div class="stat-grid">
                        <div class="stat">
                            <div class="stat-value">{len(self.data['iterations'])}</div>
                            <div class="stat-label">Iterations Completed</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{self.total_games:,}</div>
                            <div class="stat-label">Total Games Played</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{self.total_moves:,}</div>
                            <div class="stat-label">Total Moves</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{self.total_sims:,}</div>
                            <div class="stat-label">Total MCTS Simulations</div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h2>⚡ Performance Metrics</h2>
                    <div class="stat-grid">
                        <div class="stat">
                            <div class="stat-value">{avg_moves_per_sec:.1f}</div>
                            <div class="stat-label">Avg Moves/Second</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{avg_sims_per_sec:,.0f}</div>
                            <div class="stat-label">Avg Simulations/Second</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{avg_games_per_hour:.1f}</div>
                            <div class="stat-label">Avg Games/Hour</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{self.data['buffer_size'][-1] if self.data['buffer_size'] else 0:,}</div>
                            <div class="stat-label">Final Buffer Size</div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h2>🎮 Game Results</h2>
                    <div class="stat-grid">
                        <div class="stat">
                            <div class="stat-value">{total_white}</div>
                            <div class="stat-label">White Wins ({total_white/total_games*100:.1f}%)</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{total_draws}</div>
                            <div class="stat-label">Draws ({total_draws/total_games*100:.1f}%)</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{total_black}</div>
                            <div class="stat-label">Black Wins ({total_black/total_games*100:.1f}%)</div>
                        </div>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {total_white/total_games*100:.1f}%; background: #ecf0f1;"></div>
                    </div>
                </div>

                <div class="card">
                    <h2>📉 Training Progress</h2>
                    <div class="stat-grid">
                        <div class="stat">
                            <div class="stat-value">{valid_losses[0]:.4f}</div>
                            <div class="stat-label">Initial Loss</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{valid_losses[-1]:.4f}</div>
                            <div class="stat-label">Final Loss</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{loss_improvement:.1f}%</div>
                            <div class="stat-label">Loss Improvement</div>
                        </div>
                    </div>
                </div>

                <p style="text-align: center; color: #7f8c8d; margin-top: 30px;">
                    Generated by AlphaZero Training Script | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>
        </body>
        </html>
        """

        summary_path = self.output_dir / 'summary.html'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_html)


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
    avg_game_length: float = 0.0
    total_simulations: int = 0
    total_nn_evals: int = 0
    # Training metrics
    train_time: float = 0.0
    loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    num_train_batches: int = 0
    # Buffer metrics
    buffer_size: int = 0
    # Evaluation metrics
    eval_win_rate: Optional[float] = None      # vs random win rate (0.0-1.0)
    eval_endgame_score: Optional[int] = None   # puzzles fully correct (0-5)
    eval_endgame_total: int = 5                # total endgame puzzles
    eval_endgame_move_accuracy: Optional[float] = None  # consecutive move accuracy (0.0-1.0)
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
        print(f"    Moves:           {m.total_moves} total, {m.avg_game_length:.1f} avg/game")
        print(f"    Time:            {format_duration(m.selfplay_time)} ({moves_per_sec:.1f} moves/sec)")
        print(f"    MCTS Sims:       {m.total_simulations:,} ({sims_per_sec:,.0f}/sec)")
        print(f"    NN Evals:        {m.total_nn_evals:,} ({nn_evals_per_sec:,.0f}/sec)")

        # Training metrics
        if m.train_time > 0:
            samples_per_sec = (m.num_train_batches * args.train_batch) / m.train_time
            print(f"  Training:")
            print(f"    Loss:            {m.loss:.4f} (policy={m.policy_loss:.4f}, value={m.value_loss:.4f})")
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
# Evaluation Functions (vs Random + Endgame Puzzles)
# =============================================================================

# Endgame puzzles for testing model strength
# Each puzzle has a move_sequence: alternating model/opponent moves.
# Indices 0, 2, 4, ... are model moves (verified against the network's prediction).
# Indices 1, 3, 5, ... are opponent responses (played to advance the board).
# Scoring: consecutive correct model moves from the start (prefix match).
ENDGAME_PUZZLES = [
    # Mate in 2: Queen + King coordination (forced line)
    # After Qa1+ the only legal king move is Kb8 (a7/b7 controlled by Kb6), then Qa8#
    {"fen": "k7/8/1K6/8/8/8/8/7Q w - - 0 1",
     "move_sequence": ["Qa1+", "Kb8", "Qa8#"], "type": "KQ_mate_in_2"},

    # Mate in 2: Back rank with two rooks
    # Ra8+ forces king to h8 corner (f7/g7/h7 pawns block), then Rb8# is mate
    {"fen": "6k1/5ppp/8/8/8/8/8/RR4K1 w - - 0 1",
     "move_sequence": ["Ra8+", "Kh8", "Rb8#"], "type": "back_rank_mate_in_2"},

    # Mate in 1: Scholar's mate
    {"fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
     "move_sequence": ["Qxf7#"], "type": "scholars_mate"},

    # Best move: Back rank check
    {"fen": "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1",
     "move_sequence": ["Ra8+"], "type": "back_rank"},

    # Best move: Pawn promotion
    {"fen": "8/4P3/8/8/4k3/8/8/4K3 w - - 0 1",
     "move_sequence": ["e8=Q"], "type": "pawn_promo"},
]


def play_vs_random(network: 'AlphaZeroNet', device: str, num_games: int = 5,
                   simulations: int = 800, search_batch: int = 32,
                   c_puct: float = 1.5) -> Dict[str, Any]:
    """Play games against a random opponent to test basic competence (batched parallel).

    Runs multiple games concurrently with batched GPU evaluation. All games share
    a single evaluation loop that batches leaves across games for efficient GPU use.

    Args:
        network: The neural network to evaluate
        device: CUDA or CPU device
        num_games: Number of games to play
        simulations: MCTS simulations per move for the model
        search_batch: Leaf batch size for MCTS
        c_puct: MCTS exploration constant

    Returns:
        Dictionary with wins, losses, draws counts
    """
    network.eval()
    evaluator = BatchedEvaluator(network, device, use_amp=(device == "cuda"))

    print(f"      vs_random: {num_games} games batched...", end=" ", flush=True)
    start_time = time.time()

    # Game state for each concurrent game
    class GameState:
        def __init__(self, game_idx: int):
            self.game_idx = game_idx
            self.board = chess.Board()
            self.model_plays_white = (game_idx % 2 == 0)
            self.move_count = 0
            self.mcts = alphazero_cpp.BatchedMCTSSearch(
                num_simulations=simulations,
                batch_size=search_batch,
                c_puct=c_puct
            )
            self.result = None  # "win", "loss", "draw" when game ends
            self.needs_root_eval = False
            self.in_mcts_search = False
            # Cached data for current move
            self.current_fen = None
            self.current_obs_chw = None
            self.current_mask = None
            self.current_idx_to_move = {}

    games = [GameState(i) for i in range(num_games)]
    active_games = games.copy()

    def prepare_move(g: GameState):
        """Prepare a game for its next model move (encode position, build mask)."""
        g.current_fen = g.board.fen()
        obs = alphazero_cpp.encode_position(g.current_fen)
        g.current_obs_chw = np.transpose(obs, (2, 0, 1))

        legal_moves = list(g.board.legal_moves)
        g.current_mask = np.zeros(POLICY_SIZE, dtype=np.float32)
        g.current_idx_to_move = {}

        for move in legal_moves:
            idx = alphazero_cpp.move_to_index(move.uci(), g.current_fen)
            if 0 <= idx < POLICY_SIZE:
                g.current_mask[idx] = 1.0
                g.current_idx_to_move[idx] = move

        g.needs_root_eval = True
        g.in_mcts_search = False

    def finish_move(g: GameState):
        """Select and play the best move after MCTS search completes."""
        visit_counts = g.mcts.get_visit_counts()
        policy = visit_counts * g.current_mask
        if policy.sum() > 0:
            action = np.argmax(policy)
            if action in g.current_idx_to_move:
                g.board.push(g.current_idx_to_move[action])
            else:
                g.board.push(random.choice(list(g.board.legal_moves)))
        else:
            legal = list(g.board.legal_moves)
            if legal:
                g.board.push(random.choice(legal))
        g.mcts.reset()
        g.move_count += 1
        g.in_mcts_search = False
        g.needs_root_eval = False

    def check_game_over(g: GameState) -> bool:
        """Check if game is over and set result."""
        if g.board.is_game_over() or g.move_count >= 200:
            result = g.board.result()
            if result == "1-0":
                g.result = "win" if g.model_plays_white else "loss"
            elif result == "0-1":
                g.result = "loss" if g.model_plays_white else "win"
            else:
                g.result = "draw"
            return True
        return False

    # Initialize: advance to first model move for each game
    for g in active_games:
        # Play random moves until it's model's turn
        while not check_game_over(g):
            is_model_turn = (g.board.turn == chess.WHITE) == g.model_plays_white
            if is_model_turn:
                prepare_move(g)
                break
            else:
                legal = list(g.board.legal_moves)
                if legal:
                    g.board.push(random.choice(legal))
                g.move_count += 1

    # Main loop: batch evaluations across all active games
    while active_games:
        # 1. Collect games needing root evaluation
        root_eval_games = [g for g in active_games if g.needs_root_eval and g.result is None]
        if root_eval_games:
            obs_batch = np.stack([g.current_obs_chw for g in root_eval_games])
            mask_batch = np.stack([g.current_mask for g in root_eval_games])
            policies, values = evaluator.evaluate_batch(obs_batch, mask_batch)

            for i, g in enumerate(root_eval_games):
                g.mcts.init_search(g.current_fen, policies[i].astype(np.float32), float(values[i]))
                g.needs_root_eval = False
                g.in_mcts_search = True

        # 2. Collect leaves from all games in MCTS search
        games_in_search = [g for g in active_games if g.in_mcts_search and g.result is None]
        if not games_in_search:
            # Remove finished games
            active_games = [g for g in active_games if g.result is None]
            continue

        all_obs = []
        all_masks = []
        leaf_counts = []  # Track how many leaves each game contributed

        for g in games_in_search:
            if g.mcts.is_complete():
                leaf_counts.append(0)
                continue

            num_leaves, obs_batch, mask_batch = g.mcts.collect_leaves()
            if num_leaves == 0:
                leaf_counts.append(0)
                continue

            obs_nchw = np.transpose(obs_batch[:num_leaves], (0, 3, 1, 2))
            all_obs.append(obs_nchw)
            all_masks.append(mask_batch[:num_leaves])
            leaf_counts.append(num_leaves)

        # 3. Batch evaluate all leaves together
        if all_obs:
            combined_obs = np.concatenate(all_obs, axis=0)
            combined_masks = np.concatenate(all_masks, axis=0)
            all_policies, all_values = evaluator.evaluate_batch(combined_obs, combined_masks)

            # 4. Distribute results back to each game
            offset = 0
            leaf_idx = 0
            for g in games_in_search:
                count = leaf_counts[leaf_idx]
                leaf_idx += 1
                if count == 0:
                    continue
                game_policies = all_policies[offset:offset + count]
                game_values = all_values[offset:offset + count]
                g.mcts.update_leaves(game_policies.astype(np.float32), game_values.astype(np.float32))
                offset += count

        # 5. Check for completed searches and advance games
        for g in games_in_search:
            if g.mcts.is_complete():
                finish_move(g)

                # Check game over
                if check_game_over(g):
                    continue

                # Play random opponent move(s) until model's turn again
                while not check_game_over(g):
                    is_model_turn = (g.board.turn == chess.WHITE) == g.model_plays_white
                    if is_model_turn:
                        prepare_move(g)
                        break
                    else:
                        legal = list(g.board.legal_moves)
                        if legal:
                            g.board.push(random.choice(legal))
                        g.move_count += 1

        # Remove finished games
        active_games = [g for g in active_games if g.result is None]

    # Tally results
    results = {"wins": 0, "losses": 0, "draws": 0}
    result_chars = []

    for g in sorted(games, key=lambda x: x.game_idx):
        if g.result == "win":
            results["wins"] += 1
            result_chars.append("W")
        elif g.result == "loss":
            results["losses"] += 1
            result_chars.append("L")
        else:
            results["draws"] += 1
            result_chars.append("D")

    elapsed = time.time() - start_time
    print(f"{''.join(result_chars)} ({elapsed:.1f}s)")

    return results


def test_endgame_positions(network: 'AlphaZeroNet', device: str, puzzles: List[Dict] = None) -> Dict[str, Any]:
    """Test the model on endgame positions with move sequences.

    For each puzzle, plays through the move_sequence checking the model's
    predictions at each model turn (indices 0, 2, 4, ...). Opponent moves
    (indices 1, 3, 5, ...) are played automatically to advance the board.

    Scoring is prefix-based: count consecutive correct model moves from the
    start. If the answer is ABCDE and the model predicts ABDEC, only AB count
    (2 out of 5 model moves correct).

    Args:
        network: The neural network to evaluate
        device: CUDA or CPU device
        puzzles: List of puzzle dictionaries (uses ENDGAME_PUZZLES if None)

    Returns:
        Dictionary with score, move_accuracy, and details
    """
    if puzzles is None:
        puzzles = ENDGAME_PUZZLES

    network.eval()
    evaluator = BatchedEvaluator(network, device, use_amp=(device == "cuda"))

    puzzles_fully_correct = 0
    total_model_moves = 0
    total_consecutive_correct = 0
    details = []

    for puzzle in puzzles:
        fen = puzzle["fen"]
        move_sequence = puzzle["move_sequence"]
        puzzle_type = puzzle["type"]

        board = chess.Board(fen)

        # Count model moves in this sequence (indices 0, 2, 4, ...)
        num_model_moves = len(range(0, len(move_sequence), 2))
        consecutive_correct = 0
        sequence_broken = False
        predicted_moves = []

        for i, expected_san in enumerate(move_sequence):
            is_model_turn = (i % 2 == 0)

            if is_model_turn:
                # Get model's prediction for this position
                current_fen = board.fen()
                obs = alphazero_cpp.encode_position(current_fen)
                obs_chw = np.transpose(obs, (2, 0, 1))

                legal_moves = list(board.legal_moves)
                mask = np.zeros(POLICY_SIZE, dtype=np.float32)
                idx_to_move = {}

                for move in legal_moves:
                    idx = alphazero_cpp.move_to_index(move.uci(), current_fen)
                    if 0 <= idx < POLICY_SIZE:
                        mask[idx] = 1.0
                        idx_to_move[idx] = move

                policy, value = evaluator.evaluate(obs_chw, mask)
                policy = policy * mask

                if policy.sum() > 0:
                    top_idx = np.argmax(policy)
                    if top_idx in idx_to_move:
                        top_move = idx_to_move[top_idx]
                        top_move_san = board.san(top_move)
                        predicted_moves.append(top_move_san)

                        if not sequence_broken and top_move_san == expected_san:
                            consecutive_correct += 1
                        else:
                            sequence_broken = True
                    else:
                        predicted_moves.append("invalid")
                        sequence_broken = True
                else:
                    predicted_moves.append("no_legal_moves")
                    sequence_broken = True

            # Play the expected move to advance the board
            try:
                expected_move = board.parse_san(expected_san)
                board.push(expected_move)
            except (ValueError, chess.InvalidMoveError):
                break

        all_correct = (consecutive_correct == num_model_moves)
        if all_correct:
            puzzles_fully_correct += 1

        total_model_moves += num_model_moves
        total_consecutive_correct += consecutive_correct

        details.append({
            "type": puzzle_type,
            "fen": fen,
            "move_sequence": move_sequence,
            "predicted": predicted_moves,
            "consecutive_correct": consecutive_correct,
            "total_model_moves": num_model_moves,
            "move_accuracy": consecutive_correct / num_model_moves if num_model_moves > 0 else 0,
            "fully_correct": all_correct
        })

    return {
        "score": puzzles_fully_correct,
        "total": len(puzzles),
        "accuracy": puzzles_fully_correct / len(puzzles) if puzzles else 0,
        "move_score": total_consecutive_correct,
        "total_moves": total_model_moves,
        "move_accuracy": total_consecutive_correct / total_model_moves if total_model_moves > 0 else 0,
        "details": details
    }


def evaluate_checkpoint(network: 'AlphaZeroNet', device: str,
                        simulations: int = 800, search_batch: int = 32,
                        c_puct: float = 1.5) -> Dict[str, Any]:
    """Run full evaluation suite before saving checkpoint.

    Args:
        network: The neural network to evaluate
        device: CUDA or CPU device
        simulations: MCTS simulations per move (used for reference, capped for vs_random)
        search_batch: Leaf batch size for MCTS
        c_puct: MCTS exploration constant

    Returns:
        Dictionary with all evaluation results
    """
    results = {}

    # 1. vs Random Agent (5 games)
    # Cap simulations at 400 for vs_random - if model can't beat random with 400 sims,
    # 1600 sims won't help. This keeps evaluation fast (~15-30s instead of 2+ min).
    eval_sims = min(simulations, 400)
    eval_search_batch = min(search_batch, 32)  # Ensure enough MCTS rounds
    random_results = play_vs_random(network, device, num_games=5,
                                    simulations=eval_sims,
                                    search_batch=eval_search_batch,
                                    c_puct=c_puct)
    results["vs_random"] = {
        "wins": random_results["wins"],
        "losses": random_results["losses"],
        "draws": random_results["draws"],
        "win_rate": random_results["wins"] / 5.0
    }

    # 2. Endgame Puzzles (sequence-based evaluation)
    endgame_results = test_endgame_positions(network, device)
    results["endgame"] = {
        "score": endgame_results["score"],
        "total": endgame_results["total"],
        "accuracy": endgame_results["accuracy"],
        "move_score": endgame_results["move_score"],
        "total_moves": endgame_results["total_moves"],
        "move_accuracy": endgame_results["move_accuracy"],
    }

    return results


# =============================================================================
# Summary HTML Generation
# =============================================================================

def generate_summary_html(run_dir: str, config: Dict[str, Any]) -> str:
    """Generate a summary HTML file with training metrics and evaluation results.

    The HTML uses fetch() to load data from training_metrics.json and
    evaluation_results.json at view time, with embedded JSON fallback for
    file:// protocol (where CORS blocks fetch).

    Args:
        run_dir: The run directory to save summary.html
        config: Training configuration

    Returns:
        Path to the generated summary.html file
    """
    # Read JSON files from disk for the embedded fallback
    metrics_history = load_metrics_history(run_dir)
    eval_history = load_eval_history(run_dir)

    # Serialize full JSON objects for the fallback <script> block
    metrics_json = json.dumps(metrics_history)
    eval_json = json.dumps(eval_history)

    # Config table rows (static, small — embedded directly by Python)
    config_rows = (
        f'<tr><td>Network</td><td>{config.get("filters", 192)} filters '
        f'&times; {config.get("blocks", 15)} blocks</td></tr>\n'
        f'                <tr><td>Simulations</td><td>{config.get("simulations", 800)} per move</td></tr>\n'
        f'                <tr><td>Games/Iteration</td><td>{config.get("games_per_iter", 50)}</td></tr>\n'
        f'                <tr><td>Buffer Size</td><td>{config.get("buffer_size", 100000):,}</td></tr>\n'
        f'                <tr><td>Learning Rate</td><td>{config.get("lr", 0.001)}</td></tr>\n'
        f'                <tr><td>Workers</td><td>{config.get("workers", 1)}</td></tr>'
    )

    # Build the HTML with static shell + dynamic JS
    # Note: The f-string only interpolates: run_dir basename, timestamp, config_rows,
    # metrics_json, eval_json. All data extraction happens in JavaScript.
    run_name = os.path.basename(run_dir)
    generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    html_content = (
        '<!DOCTYPE html>\n'
        '<html>\n'
        '<head>\n'
        '    <title>AlphaZero Training Summary</title>\n'
        '    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>\n'
        '    <style>\n'
        '        body { font-family: Arial, sans-serif; background: #f5f6fa; padding: 20px; margin: 0; }\n'
        '        .container { max-width: 1200px; margin: 0 auto; }\n'
        '        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 20px; }\n'
        '        .card { background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }\n'
        '        .card h2 { color: #2c3e50; margin-top: 0; border-bottom: 2px solid #3498db; padding-bottom: 10px; }\n'
        '        .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }\n'
        '        .stat { background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }\n'
        '        .stat-value { font-size: 24px; font-weight: bold; color: #3498db; }\n'
        '        .stat-label { font-size: 12px; color: #7f8c8d; margin-top: 5px; }\n'
        '        .chart-container { height: 300px; margin-top: 20px; }\n'
        '        .config-table { width: 100%; border-collapse: collapse; }\n'
        '        .config-table td { padding: 8px; border-bottom: 1px solid #eee; }\n'
        '        .config-table td:first-child { font-weight: bold; color: #7f8c8d; width: 40%; }\n'
        '        .hidden { display: none; }\n'
        '    </style>\n'
        '</head>\n'
        '<body>\n'
        '    <div class="container">\n'
        '        <div class="header">\n'
        '            <h1>&#127919; AlphaZero Training Summary</h1>\n'
        f'            <p>Run: {run_name}</p>\n'
        f'            <p id="generatedTime">Generated: {generated_time}</p>\n'
        '        </div>\n'
        '\n'
        '        <div class="card">\n'
        '            <h2>&#128202; Overall Statistics</h2>\n'
        '            <div class="stat-grid">\n'
        '                <div class="stat">\n'
        '                    <div class="stat-value" id="statIterations">-</div>\n'
        '                    <div class="stat-label">Iterations Completed</div>\n'
        '                </div>\n'
        '                <div class="stat">\n'
        '                    <div class="stat-value" id="statGames">-</div>\n'
        '                    <div class="stat-label">Total Games Played</div>\n'
        '                </div>\n'
        '                <div class="stat">\n'
        '                    <div class="stat-value" id="statMoves">-</div>\n'
        '                    <div class="stat-label">Total Moves</div>\n'
        '                </div>\n'
        '                <div class="stat">\n'
        '                    <div class="stat-value" id="statImprovement">-</div>\n'
        '                    <div class="stat-label">Loss Improvement</div>\n'
        '                </div>\n'
        '            </div>\n'
        '        </div>\n'
        '\n'
        '        <div class="card hidden" id="evalCard">\n'
        '            <h2>&#127919; Latest Evaluation (Iteration <span id="evalIter">-</span>)</h2>\n'
        '            <div class="stat-grid">\n'
        '                <div class="stat">\n'
        '                    <div class="stat-value" id="evalWinRate">-</div>\n'
        '                    <div class="stat-label" id="evalWinRateLabel">vs Random Win Rate</div>\n'
        '                </div>\n'
        '                <div class="stat">\n'
        '                    <div class="stat-value" id="evalEndgame">-</div>\n'
        '                    <div class="stat-label">Endgame Puzzles Fully Correct</div>\n'
        '                </div>\n'
        '                <div class="stat">\n'
        '                    <div class="stat-value" id="evalMoveAcc">-</div>\n'
        '                    <div class="stat-label" id="evalMoveAccLabel">Move Accuracy</div>\n'
        '                </div>\n'
        '            </div>\n'
        '        </div>\n'
        '\n'
        '        <div class="card">\n'
        '            <h2>&#128201; Training Loss</h2>\n'
        '            <div class="chart-container">\n'
        '                <canvas id="lossChart"></canvas>\n'
        '            </div>\n'
        '        </div>\n'
        '\n'
        '        <div class="card hidden" id="evalChartCard">\n'
        '            <h2>&#127919; Evaluation Progress</h2>\n'
        '            <div class="chart-container">\n'
        '                <canvas id="evalChart"></canvas>\n'
        '            </div>\n'
        '        </div>\n'
        '\n'
        '        <div class="card">\n'
        '            <h2>&#9881;&#65039; Configuration</h2>\n'
        '            <table class="config-table">\n'
        f'                {config_rows}\n'
        '            </table>\n'
        '        </div>\n'
        '\n'
        '        <p style="text-align: center; color: #7f8c8d; margin-top: 30px;">\n'
        '            Generated by AlphaZero Training Script\n'
        '        </p>\n'
        '    </div>\n'
        '\n'
        '    <script>\n'
        '        // Embedded fallback data (used when fetch() fails, e.g. file:// protocol)\n'
        f'        const FALLBACK_METRICS = {metrics_json};\n'
        f'        const FALLBACK_EVAL = {eval_json};\n'
        '\n'
        '        async function loadJSON(path, fallback) {\n'
        '            try {\n'
        '                const resp = await fetch(path);\n'
        '                if (!resp.ok) throw new Error(resp.statusText);\n'
        '                return await resp.json();\n'
        '            } catch (e) {\n'
        "                console.warn('fetch(' + path + ') failed, using embedded fallback:', e.message);\n"
        '                return fallback;\n'
        '            }\n'
        '        }\n'
        '\n'
        '        function formatNumber(n) {\n'
        '            return n.toLocaleString();\n'
        '        }\n'
        '\n'
        '        function buildDashboard(metricsData, evalData) {\n'
        '            const iterations = metricsData.iterations || [];\n'
        '            const evaluations = evalData.evaluations || [];\n'
        '\n'
        '            // --- Overall statistics ---\n'
        "            document.getElementById('statIterations').textContent = iterations.length;\n"
        '\n'
        '            if (iterations.length > 0) {\n'
        '                const totalGames = iterations.reduce((s, m) => s + (m.games || 0), 0);\n'
        '                const totalMoves = iterations.reduce((s, m) => s + (m.moves || 0), 0);\n'
        "                document.getElementById('statGames').textContent = formatNumber(totalGames);\n"
        "                document.getElementById('statMoves').textContent = formatNumber(totalMoves);\n"
        '\n'
        '                if (iterations.length >= 2 && iterations[0].loss > 0) {\n'
        '                    const first = iterations[0].loss;\n'
        '                    const last = iterations[iterations.length - 1].loss;\n'
        '                    const improvement = (first - last) / first * 100;\n'
        "                    document.getElementById('statImprovement').textContent = improvement.toFixed(1) + '%';\n"
        '                } else {\n'
        "                    document.getElementById('statImprovement').textContent = '0.0%';\n"
        '                }\n'
        '            } else {\n'
        "                document.getElementById('statGames').textContent = '0';\n"
        "                document.getElementById('statMoves').textContent = '0';\n"
        "                document.getElementById('statImprovement').textContent = '0.0%';\n"
        '            }\n'
        '\n'
        '            // --- Evaluation card ---\n'
        '            if (evaluations.length > 0) {\n'
        '                const latest = evaluations[evaluations.length - 1];\n'
        '                const vsRandom = latest.vs_random || {};\n'
        '                const endgame = latest.endgame || {};\n'
        '\n'
        "                document.getElementById('evalCard').classList.remove('hidden');\n"
        "                document.getElementById('evalIter').textContent = latest.iteration || 'N/A';\n"
        "                document.getElementById('evalWinRate').textContent =\n"
        "                    ((vsRandom.win_rate || 0) * 100).toFixed(0) + '%';\n"
        "                document.getElementById('evalWinRateLabel').textContent =\n"
        "                    'vs Random Win Rate (' + (vsRandom.wins || 0) + '/5 games)';\n"
        "                document.getElementById('evalEndgame').textContent =\n"
        "                    (endgame.score || 0) + '/' + (endgame.total || 5);\n"
        "                document.getElementById('evalMoveAcc').textContent =\n"
        "                    ((endgame.move_accuracy || 0) * 100).toFixed(0) + '%';\n"
        "                document.getElementById('evalMoveAccLabel').textContent =\n"
        "                    'Move Accuracy (' + (endgame.move_score || 0) + '/' + (endgame.total_moves || 0) + ' moves)';\n"
        '            }\n'
        '\n'
        '            // --- Loss chart ---\n'
        '            const lossIters = iterations.map(m => m.iteration || 0);\n'
        '            const losses = iterations.map(m => m.loss || 0);\n'
        '            const policyLosses = iterations.map(m => m.policy_loss || 0);\n'
        '            const valueLosses = iterations.map(m => m.value_loss || 0);\n'
        '\n'
        "            new Chart(document.getElementById('lossChart'), {\n"
        "                type: 'line',\n"
        '                data: {\n'
        '                    labels: lossIters,\n'
        '                    datasets: [\n'
        '                        {\n'
        "                            label: 'Total Loss',\n"
        '                            data: losses,\n'
        "                            borderColor: '#3498db',\n"
        '                            fill: false,\n'
        '                            tension: 0.1,\n'
        "                            yAxisID: 'y'\n"
        '                        },\n'
        '                        {\n'
        "                            label: 'Policy Loss',\n"
        '                            data: policyLosses,\n'
        "                            borderColor: '#2ecc71',\n"
        '                            fill: false,\n'
        '                            tension: 0.1,\n'
        "                            yAxisID: 'y'\n"
        '                        },\n'
        '                        {\n'
        "                            label: 'Value Loss',\n"
        '                            data: valueLosses,\n'
        "                            borderColor: '#e74c3c',\n"
        '                            fill: false,\n'
        '                            tension: 0.1,\n'
        "                            yAxisID: 'y1'\n"
        '                        }\n'
        '                    ]\n'
        '                },\n'
        '                options: {\n'
        '                    responsive: true,\n'
        '                    maintainAspectRatio: false,\n'
        '                    scales: {\n'
        '                        y: {\n'
        "                            type: 'linear',\n"
        "                            position: 'left',\n"
        '                            beginAtZero: false,\n'
        "                            title: { display: true, text: 'Total/Policy Loss' }\n"
        '                        },\n'
        '                        y1: {\n'
        "                            type: 'linear',\n"
        "                            position: 'right',\n"
        '                            beginAtZero: false,\n'
        "                            title: { display: true, text: 'Value Loss' },\n"
        '                            grid: { drawOnChartArea: false }\n'
        '                        }\n'
        '                    }\n'
        '                }\n'
        '            });\n'
        '\n'
        '            // --- Evaluation chart ---\n'
        '            if (evaluations.length > 0) {\n'
        "                document.getElementById('evalChartCard').classList.remove('hidden');\n"
        '\n'
        '                const evalIters = evaluations.map(e => e.iteration || 0);\n'
        '                const winRates = evaluations.map(e => ((e.vs_random || {}).win_rate || 0) * 100);\n'
        '                const moveAccs = evaluations.map(e => ((e.endgame || {}).move_accuracy || 0) * 100);\n'
        '\n'
        "                new Chart(document.getElementById('evalChart'), {\n"
        "                    type: 'line',\n"
        '                    data: {\n'
        '                        labels: evalIters,\n'
        '                        datasets: [\n'
        '                            {\n'
        "                                label: 'vs Random Win Rate (%)',\n"
        '                                data: winRates,\n'
        "                                borderColor: '#3498db',\n"
        '                                fill: false,\n'
        '                                tension: 0.1,\n'
        "                                yAxisID: 'y'\n"
        '                            },\n'
        '                            {\n'
        "                                label: 'Endgame Move Accuracy (%)',\n"
        '                                data: moveAccs,\n'
        "                                borderColor: '#2ecc71',\n"
        '                                fill: false,\n'
        '                                tension: 0.1,\n'
        "                                yAxisID: 'y1'\n"
        '                            }\n'
        '                        ]\n'
        '                    },\n'
        '                    options: {\n'
        '                        responsive: true,\n'
        '                        maintainAspectRatio: false,\n'
        '                        scales: {\n'
        '                            y: {\n'
        "                                type: 'linear',\n"
        "                                position: 'left',\n"
        '                                min: 0,\n'
        '                                max: 100,\n'
        "                                title: { display: true, text: 'Win Rate (%)' }\n"
        '                            },\n'
        '                            y1: {\n'
        "                                type: 'linear',\n"
        "                                position: 'right',\n"
        '                                min: 0,\n'
        '                                max: 100,\n'
        "                                title: { display: true, text: 'Endgame Move Accuracy (%)' },\n"
        '                                grid: { drawOnChartArea: false }\n'
        '                            }\n'
        '                        }\n'
        '                    }\n'
        '                });\n'
        '            }\n'
        '        }\n'
        '\n'
        '        // Load data and build dashboard\n'
        '        (async function() {\n'
        '            const [metricsData, evalData] = await Promise.all([\n'
        "                loadJSON('./training_metrics.json', FALLBACK_METRICS),\n"
        "                loadJSON('./evaluation_results.json', FALLBACK_EVAL)\n"
        '            ]);\n'
        '            buildDashboard(metricsData, evalData);\n'
        '        })();\n'
        '    </script>\n'
        '</body>\n'
        '</html>\n'
    )

    summary_path = os.path.join(run_dir, "summary.html")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return summary_path


# =============================================================================
# Run Directory and Resume Helpers
# =============================================================================

def create_run_directory(base_dir: str, filters: int, blocks: int) -> str:
    """Create a new organized run directory with timestamp.

    Args:
        base_dir: Base checkpoint directory (e.g., "checkpoints")
        filters: Number of network filters
        blocks: Number of residual blocks

    Returns:
        Path to the created run directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"f{filters}-b{blocks}_{timestamp}"
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

    # Sort by iteration number
    def extract_iter(path):
        name = os.path.basename(path)
        # Handle both model_iter_001.pt and cpp_iter_1.pt formats
        try:
            if "model_iter_" in name:
                return int(name.replace("model_iter_", "").replace(".pt", "").replace("_emergency", ""))
            elif "cpp_iter_" in name:
                return int(name.replace("cpp_iter_", "").replace(".pt", "").replace("_emergency", ""))
        except ValueError:
            return 0
        return 0

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


def load_metrics_history(run_dir: str) -> Dict[str, Any]:
    """Load training metrics history from a run directory.

    Args:
        run_dir: Path to the run directory

    Returns:
        Metrics history dictionary
    """
    metrics_path = os.path.join(run_dir, "training_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, ValueError):
            print(f"  Warning: Could not parse {metrics_path}, starting fresh")
    return {"iterations": [], "config": {}}


def save_metrics_history(run_dir: str, metrics_history: Dict[str, Any]):
    """Save training metrics history to a run directory.

    Args:
        run_dir: Path to the run directory
        metrics_history: Metrics history dictionary
    """
    metrics_path = os.path.join(run_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)


def load_eval_history(run_dir: str) -> Dict[str, Any]:
    """Load evaluation history from a run directory.

    Args:
        run_dir: Path to the run directory

    Returns:
        Evaluation history dictionary
    """
    eval_path = os.path.join(run_dir, "evaluation_results.json")
    if os.path.exists(eval_path):
        try:
            with open(eval_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, ValueError):
            print(f"  Warning: Could not parse {eval_path}, starting fresh")
    return {"evaluations": []}


def save_eval_history(run_dir: str, eval_history: Dict[str, Any]):
    """Save evaluation history to a run directory.

    Args:
        run_dir: Path to the run directory
        eval_history: Evaluation history dictionary
    """
    eval_path = os.path.join(run_dir, "evaluation_results.json")
    with open(eval_path, 'w') as f:
        json.dump(eval_history, f, indent=2)


def truncate_history_to_iteration(
    metrics_history: Dict[str, Any],
    eval_history: Dict[str, Any],
    start_iter: int
) -> tuple:
    """Truncate metrics and eval history to discard records from reverted iterations.

    When resuming from an earlier checkpoint (e.g., after model collapse),
    records at or after start_iter are stale and should be removed so new
    training data replaces them cleanly.

    Args:
        metrics_history: Training metrics dictionary (modified in-place)
        eval_history: Evaluation history dictionary (modified in-place)
        start_iter: First iteration that will be (re-)trained; records with
                    iteration >= start_iter are discarded

    Returns:
        (metrics_removed_count, eval_removed_count)
    """
    old_metrics = metrics_history.get("iterations", [])
    new_metrics = [r for r in old_metrics if r.get("iteration", 0) < start_iter]
    metrics_removed = len(old_metrics) - len(new_metrics)
    metrics_history["iterations"] = new_metrics

    old_evals = eval_history.get("evaluations", [])
    new_evals = [r for r in old_evals if r.get("iteration", 0) < start_iter]
    eval_removed = len(old_evals) - len(new_evals)
    eval_history["evaluations"] = new_evals

    return metrics_removed, eval_removed


# =============================================================================
# Neural Network (AlphaZero Paper Architecture - Compatible with Python Backend)
# =============================================================================
# Architecture matches alphazero/neural/network.py for checkpoint compatibility:
# - Input: 122 channels (extended encoding)
# - Policy head: 2 filters (paper standard)
# - Value head: 256 hidden units (paper standard)
# - Layer naming: residual_tower, policy_head, value_head

class ConvBlock(nn.Module):
    """Convolutional block with batch norm and ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block matching Python backend."""
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class PolicyHead(nn.Module):
    """Policy head: outputs action probabilities (AlphaZero paper: 2 filters)."""
    def __init__(self, in_channels: int, num_filters: int = 2, num_actions: int = POLICY_SIZE):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(num_filters * 8 * 8, num_actions)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class ValueHead(nn.Module):
    """Value head: outputs position evaluation (AlphaZero paper: 256 hidden)."""
    def __init__(self, in_channels: int, num_filters: int = 1, hidden_size: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(num_filters * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.tanh(self.fc2(x))


class AlphaZeroNet(nn.Module):
    """AlphaZero neural network - compatible with Python backend checkpoints.

    Architecture matches alphazero/neural/network.py exactly for checkpoint compatibility.
    Uses AlphaZero paper standard settings:
    - 122 input channels (extended encoding)
    - 2 policy filters
    - 256 value hidden units
    """

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_filters: int = 192,
        num_blocks: int = 15,
        num_actions: int = POLICY_SIZE,
        policy_filters: int = 2,
        value_filters: int = 1,
        value_hidden: int = 256
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.num_actions = num_actions

        # Input convolution
        self.input_conv = ConvBlock(input_channels, num_filters)

        # Residual tower (nn.Sequential for Python compatibility)
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_blocks)]
        )

        # Output heads (with Python-compatible naming)
        self.policy_head = PolicyHead(num_filters, policy_filters, num_actions)
        self.value_head = ValueHead(num_filters, value_filters, value_hidden)

    def forward(self, x, mask=None):
        # Shared trunk
        x = self.input_conv(x)
        x = self.residual_tower(x)

        # Policy head
        policy_logits = self.policy_head(x)

        if mask is not None:
            # Mask illegal moves (use -1e4 for FP16 compatibility)
            policy_logits = policy_logits.masked_fill(mask == 0, -1e4)

        policy = F.softmax(policy_logits, dim=1)

        # Value head
        value = self.value_head(x)

        return policy, value


# =============================================================================
# Batched Evaluator (GPU)
# =============================================================================

class BatchedEvaluator:
    """Efficient batched neural network evaluation on GPU."""

    def __init__(self, network: nn.Module, device: str, use_amp: bool = True):
        self.network = network
        self.device = device
        self.use_amp = use_amp and device == "cuda"

    @torch.no_grad()
    def evaluate(self, obs: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Evaluate single position."""
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)

        if self.use_amp:
            with autocast('cuda'):
                policy, value = self.network(obs_tensor, mask_tensor)
        else:
            policy, value = self.network(obs_tensor, mask_tensor)

        return policy[0].cpu().numpy(), float(value[0].item())

    @torch.no_grad()
    def evaluate_batch(self, obs_batch: np.ndarray, mask_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate batch of positions."""
        obs_tensor = torch.from_numpy(obs_batch).float().to(self.device)
        mask_tensor = torch.from_numpy(mask_batch).float().to(self.device)

        if self.use_amp:
            with autocast('cuda'):
                policies, values = self.network(obs_tensor, mask_tensor)
        else:
            policies, values = self.network(obs_tensor, mask_tensor)

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
        c_puct: float = 1.5,
        temperature_moves: int = 30,
        draw_score: float = 0.0
    ):
        self.evaluator = evaluator
        self.num_simulations = num_simulations
        self.mcts_batch_size = mcts_batch_size
        self.c_puct = c_puct
        self.temperature_moves = temperature_moves
        self.draw_score = draw_score

        # Create C++ MCTS engine (BatchedMCTSSearch is the Python binding name)
        self.mcts = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=num_simulations,
            batch_size=mcts_batch_size,
            c_puct=c_puct
        )

    def play_game(self) -> Tuple[List[np.ndarray], List[np.ndarray], float, int, int, int]:
        """Play a single self-play game.

        Returns:
            observations: List of board observations (122, 8, 8) CHW format
            policies: List of MCTS policies
            result: Game result (1=white wins, -1=black wins, 0=draw)
            num_moves: Number of moves played
            total_sims: Total MCTS simulations
            total_evals: Total NN evaluations
        """
        board = chess.Board()
        observations = []
        policies = []
        move_count = 0
        total_sims = 0
        total_evals = 0

        while not board.is_game_over() and move_count < 512:
            fen = board.fen()

            # Encode position (returns 8, 8, 122 in NHWC format)
            obs = alphazero_cpp.encode_position(fen)
            obs_chw = np.transpose(obs, (2, 0, 1))  # Convert to (122, 8, 8) CHW

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
            root_policy, root_value = self.evaluator.evaluate(obs_chw, mask)
            total_evals += 1

            # Initialize MCTS search
            self.mcts.init_search(fen, root_policy.astype(np.float32), float(root_value))

            # Run MCTS with batched leaf evaluation
            while not self.mcts.is_complete():
                num_leaves, obs_batch, mask_batch = self.mcts.collect_leaves()
                if num_leaves == 0:
                    break

                # Convert NHWC to NCHW for neural network
                obs_nchw = np.transpose(obs_batch[:num_leaves], (0, 3, 1, 2))
                masks = mask_batch[:num_leaves]

                # Batch evaluate
                leaf_policies, leaf_values = self.evaluator.evaluate_batch(obs_nchw, masks)
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
            observations.append(obs_chw.copy())
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
            value = self.draw_score  # Asymmetric draw value from White's perspective

        return observations, policies, value, move_count, total_sims, total_evals


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


def run_parallel_selfplay(
    network: nn.Module,
    replay_buffer,
    device: str,
    args,
    iteration: int,
    progress_callback=None,
    live_dashboard=None
) -> IterationMetrics:
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
        IterationMetrics with self-play statistics
    """
    network.eval()
    metrics = IterationMetrics(iteration=iteration)
    selfplay_start = time.time()

    # Calculate how many games per worker to match games_per_iter
    games_per_worker = max(1, args.games_per_iter // args.workers)
    actual_total_games = games_per_worker * args.workers

    # Auto-calculate optimal queue capacity if not specified
    # Formula: max(8192, workers * search_batch * 8)
    # The *8 factor ensures capacity/4 > 2 * workers * search_batch,
    # preventing pool reset starvation under heavy concurrent submission.
    if args.queue_capacity > 0:
        queue_capacity = args.queue_capacity
    else:
        queue_capacity = max(8192, args.workers * args.search_batch * 8)

    # Create parallel coordinator
    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=args.workers,
        games_per_worker=games_per_worker,
        num_simulations=args.simulations,
        mcts_batch_size=args.search_batch,
        gpu_batch_size=args.eval_batch,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        temperature_moves=args.temperature_moves,
        gpu_timeout_ms=args.gpu_batch_timeout_ms,
        worker_timeout_ms=args.worker_timeout_ms,
        queue_capacity=queue_capacity,
        root_eval_retries=args.root_eval_retries,
        draw_score=args.draw_score
    )

    # Set replay buffer so data is stored directly
    coordinator.set_replay_buffer(replay_buffer)

    print(f"  Parallel: {args.workers} workers × {games_per_worker} games = {actual_total_games} games")
    print(f"  eval_batch={args.eval_batch}, gpu_timeout={args.gpu_batch_timeout_ms}ms, "
          f"queue_capacity={queue_capacity}, root_eval_retries={args.root_eval_retries}")

    # Push initial progress to live dashboard (show we're in self-play phase)
    if live_dashboard is not None:
        live_dashboard.push_progress(
            iteration=metrics.iteration,
            games_completed=0,
            total_games=actual_total_games,
            moves=0, sims=0, evals=0,
            elapsed_time=0.1,
            buffer_size=replay_buffer.size(),
            phase="selfplay"
        )

    # =========================================================================
    # CUDA Graph Optimization for Inference
    # =========================================================================
    # CUDA Graphs capture a sequence of GPU operations and replay them with
    # minimal CPU overhead. For fixed-batch-size inference, this eliminates
    # kernel launch overhead (~5-10ms savings per batch).

    gpu_batch_size = args.eval_batch
    use_cuda_graph = (device == "cuda")
    cuda_graph = None
    static_obs = None
    static_mask = None
    static_policy_out = None
    static_value_out = None

    # Free fragmented GPU memory before allocating static CUDA Graph buffers
    if device == "cuda":
        torch.cuda.empty_cache()

    if use_cuda_graph:
        try:
            # Pre-allocate static GPU buffers for CUDA graph capture
            static_obs = torch.zeros(gpu_batch_size, INPUT_CHANNELS, 8, 8, device='cuda')
            static_mask = torch.zeros(gpu_batch_size, POLICY_SIZE, device='cuda')

            # Warm-up pass (required before graph capture)
            with autocast('cuda'):
                static_policy_out, static_value_out = network(static_obs, static_mask)

            # Capture CUDA graph
            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph):
                with autocast('cuda'):
                    static_policy_out, static_value_out = network(static_obs, static_mask)

            print(f"  CUDA Graph captured for batch_size={gpu_batch_size}")
        except Exception as e:
            print(f"  CUDA Graph capture failed ({e}), falling back to eager mode")
            use_cuda_graph = False
            cuda_graph = None

    # CUDA graph fire tracking (mutable dict for closure capture)
    cuda_graph_stats = {
        'graph_fires': 0,       # Times CUDA graph fast path was used
        'eager_fires': 0,       # Times eager fallback was used
        'total_infer_time_ms': 0.0,  # Cumulative inference time
    }

    # Create evaluator callback that will be called from C++ GPU thread
    # Signature: (observations, legal_masks, batch_size, out_policies, out_values) -> None
    # Observations arrive in NCHW format (C++ does the transpose now)
    # Output buffers are writable numpy views over C++ memory (zero-copy)
    @torch.no_grad()
    def neural_evaluator(obs_array: np.ndarray, mask_array: np.ndarray, batch_size: int,
                         out_policies: np.ndarray = None, out_values: np.ndarray = None):
        """Neural network evaluator callback for C++ coordinator."""
        infer_start = time.perf_counter()

        if use_cuda_graph and cuda_graph is not None and batch_size == gpu_batch_size:
            # Fast path: replay CUDA graph (fixed batch size only)
            cuda_graph_stats['graph_fires'] += 1
            static_obs.copy_(torch.from_numpy(obs_array[:batch_size]))
            static_mask.copy_(torch.from_numpy(mask_array[:batch_size]))
            cuda_graph.replay()
            policies = static_policy_out
            values = static_value_out
        else:
            # Standard path: variable batch size or no CUDA graph
            cuda_graph_stats['eager_fires'] += 1
            obs_tensor = torch.from_numpy(obs_array[:batch_size]).float().to(device)
            mask_tensor = torch.from_numpy(mask_array[:batch_size]).float().to(device)

            if device == "cuda":
                with autocast('cuda'):
                    policies, values = network(obs_tensor, mask_tensor)
            else:
                policies, values = network(obs_tensor, mask_tensor)

        # Track inference time (synchronize for accurate timing on GPU)
        if device == "cuda":
            torch.cuda.synchronize()
        cuda_graph_stats['total_infer_time_ms'] += (time.perf_counter() - infer_start) * 1000

        # Write results directly to C++ output buffers (zero-copy)
        policies_np = policies[:batch_size].cpu().numpy().astype(np.float32)
        values_np = values[:batch_size].squeeze(-1).cpu().numpy().astype(np.float32)

        if out_policies is not None and out_values is not None:
            np.copyto(out_policies[:batch_size], policies_np)
            np.copyto(out_values[:batch_size], values_np)
            return None
        else:
            # Legacy fallback
            return policies_np, values_np

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

            print(f"    \u23f1 {format_duration(elapsed)} | "
                  f"Games: {games}/{actual_total_games} | "
                  f"Moves: {moves_per_sec:.1f}/s | "
                  f"Sims: {sims_per_sec:,.0f}/s | "
                  f"NN: {evals_per_sec:,.0f}/s | "
                  f"Batch: {avg_batch:.0f} | "
                  f"Buffer: {replay_buffer.size():,} | "
                  f"ETA: {eta_str}")

            if failures > 0:
                print(f"      Failures: {failures} MCTS timeouts")

            last_console_report[0] = now

        # Live dashboard push (if enabled)
        if live_dashboard is not None:
            # Calculate GPU stats for dashboard
            total_fires = cuda_graph_stats['graph_fires'] + cuda_graph_stats['eager_fires']
            graph_fire_rate = cuda_graph_stats['graph_fires'] / total_fires if total_fires > 0 else 0.0
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
                buffer_size=replay_buffer.size(),
                phase="selfplay",
                white_wins=live_stats.get('white_wins', 0),
                black_wins=live_stats.get('black_wins', 0),
                draws=live_stats.get('draws', 0),
                timeout_evals=live_stats.get('mcts_failures', 0),
                pool_exhaustion=live_stats.get('pool_exhaustion_count', 0),
                submission_drops=live_stats.get('submission_drops', 0),
                partial_subs=live_stats.get('partial_submissions', 0),
                pool_resets=live_stats.get('pool_resets', 0),
                pool_load=live_stats.get('pool_load', 0.0),
                avg_batch_size=live_stats.get('avg_batch_size', 0.0),
                batch_fill_ratio=live_stats.get('batch_fill_ratio', 0.0),
                root_retries=live_stats.get('root_retries', 0),
                stale_flushed=live_stats.get('stale_results_flushed', 0),
                # GPU metrics
                cuda_graph_fires=cuda_graph_stats['graph_fires'],
                eager_fires=cuda_graph_stats['eager_fires'],
                graph_fire_rate=graph_fire_rate,
                avg_infer_time_ms=avg_infer_ms,
                gpu_memory_used_mb=gpu_mem_mb,
                cuda_graph_enabled=(use_cuda_graph and cuda_graph is not None),
                # Queue status metrics
                queue_fill_pct=live_stats.get('queue_fill_pct', 0.0),
                gpu_wait_ms=live_stats.get('gpu_wait_ms', 0.0),
                worker_wait_ms=live_stats.get('worker_wait_ms', 0.0),
                buffer_swaps=live_stats.get('buffer_swaps', 0),
            )

    # Run generation with interrupt support (allows Ctrl+C to stop)
    # Poll C++ stats every 5 seconds, console prints at progress_interval
    result = run_parallel_selfplay_with_interrupt(
        coordinator, neural_evaluator, shutdown_handler,
        progress_callback=progress_cb,
        progress_interval=5.0
    )

    # Extract stats from result dict
    if result is None:
        # Interrupted before any results - return minimal metrics
        metrics.selfplay_time = time.time() - selfplay_start
        return metrics
    elif isinstance(result, dict):
        # Check for C++ thread errors surfaced from the coordinator
        if result.get('cpp_error'):
            print(f"\n  WARNING: C++ error during self-play: {result['cpp_error']}")

        metrics.num_games = result.get('games_completed', actual_total_games)
        metrics.total_moves = result.get('total_moves', 0)
        metrics.white_wins = result.get('white_wins', 0)
        metrics.black_wins = result.get('black_wins', 0)
        metrics.draws = result.get('draws', 0)
        metrics.total_simulations = result.get('total_simulations', 0)
        metrics.total_nn_evals = result.get('total_nn_evals', 0)

        # Display diagnostic metrics for parallel pipeline health
        mcts_failures = result.get('mcts_failures', 0)
        pool_exhaustion = result.get('pool_exhaustion_count', 0)
        partial_subs = result.get('partial_submissions', 0)
        submission_drops = result.get('submission_drops', 0)
        pool_resets = result.get('pool_resets', 0)
        avg_batch = result.get('avg_batch_size', 0)
        total_batches = result.get('total_batches', 0)
        root_retries = result.get('root_retries', 0)
        stale_flushed = result.get('stale_results_flushed', 0)

        # Calculate NN evals per move (should be ~51 for 800 sims, batch 64)
        nn_evals_per_move = metrics.total_nn_evals / max(metrics.total_moves, 1)
        failure_rate = mcts_failures / max(metrics.total_moves, 1) * 100

        print(f"  Parallel stats: {metrics.total_nn_evals:,} NN evals ({nn_evals_per_move:.1f}/move), "
              f"avg_batch={avg_batch:.1f}, batches={total_batches:,}")

        if mcts_failures > 0 or pool_exhaustion > 0 or partial_subs > 0 or root_retries > 0:
            print(f"  Pipeline issues: {mcts_failures} MCTS failures ({failure_rate:.1f}%), "
                  f"pool_exhaustion={pool_exhaustion}, partial_subs={partial_subs}, "
                  f"drops={submission_drops}, resets={pool_resets}")
            if root_retries > 0 or stale_flushed > 0:
                print(f"  Retry stats: {root_retries} root retries, {stale_flushed} stale results flushed")
            if failure_rate > 10:
                print(f"  HIGH FAILURE RATE - Consider increasing --worker-timeout-ms or --queue-capacity")
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

    # Only estimate if we don't have real stats from the C++ coordinator
    # (parallel mode provides real stats, sequential mode needs estimates)
    if metrics.total_simulations == 0:
        metrics.total_simulations = metrics.total_moves * args.simulations
    if metrics.total_nn_evals == 0:
        # Rough estimate for sequential mode
        metrics.total_nn_evals = metrics.total_simulations // (args.search_batch // 2)

    return metrics


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
    scaler: GradScaler
) -> dict:
    """Train for one iteration using C++ ReplayBuffer."""
    network.train()

    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    num_batches = 0

    for epoch in range(epochs):
        # Sample batch from C++ ReplayBuffer
        obs, policies, values = replay_buffer.sample(batch_size)

        # Convert to PyTorch tensors
        # obs is (batch, 7808) flat, need to reshape to (batch, 122, 8, 8)
        obs = obs.reshape(-1, 8, 8, INPUT_CHANNELS)  # (batch, 8, 8, 122)
        obs = np.transpose(obs, (0, 3, 1, 2))  # (batch, 122, 8, 8)

        obs_tensor = torch.from_numpy(obs).to(device)
        policy_target = torch.from_numpy(policies).to(device)
        value_target = torch.from_numpy(values).unsqueeze(1).to(device)

        optimizer.zero_grad()

        with autocast('cuda', enabled=(device == "cuda")):
            # Forward pass (no mask during training - targets already masked)
            policy_pred, value_pred = network(obs_tensor)

            # Policy loss (cross-entropy)
            policy_loss = -torch.sum(policy_target * torch.log(policy_pred + 1e-8)) / policy_pred.size(0)

            # Value loss (MSE)
            value_loss = F.mse_loss(value_pred, value_target)

            # Total loss
            loss = policy_loss + value_loss

        # Backward pass with mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches,
        'num_batches': num_batches,
    }


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

    # Resume with buffer loading (buffer always saved; --load-buffer loads on startup)
    uv run python alphazero-cpp/scripts/train.py --resume checkpoints/f192-b15_2024-02-03_14-30-00 --load-buffer

    # Disable visualization and evaluation
    uv run python alphazero-cpp/scripts/train.py --no-visualization --no-eval
        """
    )

    # Training iterations
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations (default: 100)")
    parser.add_argument("--games-per-iter", type=int, default=50,
                        help="Self-play games per iteration (default: 50)")

    # MCTS parameters
    parser.add_argument("--simulations", type=int, default=800,
                        help="MCTS simulations per move (default: 800)")
    parser.add_argument("--search-batch", type=int, default=1,
                        help="Leaves per MCTS iteration per worker (default: 1, optimal for cross-game batching)")
    parser.add_argument("--c-puct", type=float, default=1.5,
                        help="MCTS exploration constant (default: 1.5)")
    parser.add_argument("--temperature-moves", type=int, default=30,
                        help="Moves with temperature=1 for exploration (default: 30)")

    # Parallel self-play (automatically enabled when workers > 1)
    parser.add_argument("--workers", type=int, default=1,
                        help="Self-play workers. 1=sequential, >1=parallel with cross-game batching (default: 1)")
    parser.add_argument("--eval-batch", type=int, default=None,
                        help="Max positions per GPU call (default: workers × search-batch, rounded to 32)")
    parser.add_argument("--gpu-batch-timeout-ms", type=int, default=20,
                        help="GPU batch collection timeout in ms (default: 20)")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3,
                        help="Dirichlet noise alpha for root exploration (default: 0.3)")
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25,
                        help="Dirichlet noise weight for root exploration (default: 0.25)")
    parser.add_argument("--worker-timeout-ms", type=int, default=2000,
                        help="Worker wait time for NN results in ms (default: 2000)")
    parser.add_argument("--queue-capacity", type=int, default=0,
                        help="Eval queue capacity. 0=auto-calculate from workers*search_batch*8 (default: 0)")
    parser.add_argument("--root-eval-retries", type=int, default=3,
                        help="Max retries for root NN evaluation before falling back to uniform (default: 3)")
    parser.add_argument("--draw-score", type=float, default=0.0,
                        help="Draw value from White's perspective. -0.5 = penalize White draws (default: 0.0)")

    # Network parameters
    parser.add_argument("--filters", type=int, default=192,
                        help="Network filters (default: 192)")
    parser.add_argument("--blocks", type=int, default=15,
                        help="Residual blocks (default: 15)")

    # Training parameters
    parser.add_argument("--train-batch", type=int, default=256,
                        help="Samples per training gradient step (default: 256)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs per iteration (default: 5)")
    parser.add_argument("--buffer-size", type=int, default=100000,
                        help="Replay buffer size (default: 100000)")

    # Device and paths
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Base checkpoint directory (default: checkpoints)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path (.pt file) or run directory")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="Save checkpoint every N iterations (default: 5)")
    parser.add_argument("--progress-interval", type=float, default=30.0,
                        help="Print progress statistics every N seconds (default: 30)")

    # Buffer loading (buffer is always saved; this controls loading)
    parser.add_argument("--load-buffer", nargs='?', const='', default=None,
                        help="Load replay buffer on startup. No arg: check run dir. With path: load from path.")

    # Visualization and evaluation
    parser.add_argument("--no-visualization", action="store_true",
                        help="Disable summary.html generation (default: enabled)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation before checkpoint save (default: enabled)")
    parser.add_argument("--live", action="store_true",
                        help="Enable LIVE web dashboard with real-time updates (requires flask)")
    parser.add_argument("--dashboard-port", type=int, default=5000,
                        help="Port for live dashboard server (default: 5000)")
    args = parser.parse_args()

    # Install last-resort crash hook to ensure errors are visible
    def _crash_hook(exc_type, exc_value, exc_tb):
        import traceback
        msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        sys.stderr.write(f"\n[FATAL CRASH]\n{msg}\n")
        sys.stderr.flush()
    sys.excepthook = _crash_hook

    # Auto-compute eval-batch if not specified
    if args.eval_batch is None:
        args.eval_batch = ((args.workers * args.search_batch + 31) // 32) * 32
        args.eval_batch = max(args.eval_batch, 32)  # Minimum 32
        print(f"  Auto-setting eval-batch={args.eval_batch} (workers × search_batch, rounded to 32)")

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

    # Enable cuDNN optimizations for faster convolutions
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # ==========================================================================
    # Run Directory Setup
    # ==========================================================================
    # Determine run directory: either resume from existing or create new
    start_iter = 0
    metrics_history = {"iterations": [], "config": {}}
    eval_history = {"evaluations": []}

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

        # Load existing metrics and evaluation history
        print("  Loading metrics history...", end=" ", flush=True)
        metrics_history = load_metrics_history(run_dir)
        print(f"done ({len(metrics_history['iterations'])} iterations)", flush=True)

        print("  Loading eval history...", end=" ", flush=True)
        eval_history = load_eval_history(run_dir)
        print(f"done ({len(eval_history['evaluations'])} evaluations)", flush=True)
    else:
        # Create new run directory
        os.makedirs(args.save_dir, exist_ok=True)
        run_dir = create_run_directory(args.save_dir, args.filters, args.blocks)
        checkpoint_path = None
        print(f"Created run directory: {run_dir}")

    # Buffer path is inside run directory (if persistence enabled)
    buffer_path = os.path.join(run_dir, "replay_buffer.rpbf")

    # Store config in metrics history
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
    metrics_history["config"] = config

    # Print configuration
    print("=" * 70)
    print("AlphaZero Training with C++ Backend")
    print("=" * 70)
    print(f"Run directory:       {run_dir}")
    print(f"Device:              {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"Network:             {args.filters} filters x {args.blocks} blocks")
    print(f"Input channels:      {INPUT_CHANNELS}")
    print(f"MCTS:                {args.simulations} sims, search_batch={args.search_batch}, c_puct={args.c_puct}")
    if args.workers > 1:
        print(f"Self-play:           Parallel ({args.workers} workers, eval_batch={args.eval_batch})")
    else:
        print(f"Self-play:           Sequential")
    print(f"Training:            {args.iterations} iters x {args.games_per_iter} games")
    print(f"                     train_batch={args.train_batch}, lr={args.lr}, epochs={args.epochs}")
    print(f"Buffer:              {args.buffer_size} positions (always saved" + (", loading enabled)" if args.load_buffer is not None else ")"))
    print(f"Checkpoints:         model_iter_*.pt (every {args.save_interval} iters)")
    print(f"Visualization:       {'disabled' if args.no_visualization else 'summary.html'}")
    print(f"Evaluation:          {'disabled' if args.no_eval else 'enabled (vs_random + endgame)'}")
    print(f"Progress reports:    Every {args.progress_interval:.0f} seconds")
    print("=" * 70)

    # Create network
    network = AlphaZeroNet(num_filters=args.filters, num_blocks=args.blocks, input_channels=INPUT_CHANNELS)
    network = network.to(device)

    num_params = sum(p.numel() for p in network.parameters())
    print(f"Network parameters:  {num_params:,}")

    # Create optimizer and scaler
    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler('cuda', enabled=(device == "cuda"))

    # Create C++ ReplayBuffer
    replay_buffer = alphazero_cpp.ReplayBuffer(capacity=args.buffer_size)

    # Load existing replay buffer if --load-buffer specified
    if args.load_buffer is not None:
        # Determine load path
        if args.load_buffer == '':
            # No path specified, use run directory
            load_path = buffer_path
        elif os.path.isdir(args.load_buffer):
            # Directory specified, look for buffer file inside
            load_path = os.path.join(args.load_buffer, "replay_buffer.rpbf")
        else:
            # Direct file path specified
            load_path = args.load_buffer

        if os.path.exists(load_path):
            file_size_mb = os.path.getsize(load_path) / (1024 * 1024)
            print(f"\nLoading replay buffer: {load_path} ({file_size_mb:.1f} MB)...", flush=True)
            import time as _time
            _load_start = _time.time()
            if replay_buffer.load(load_path):
                stats = replay_buffer.get_stats()
                _load_elapsed = _time.time() - _load_start
                print(f"  Loaded {stats['size']:,} samples in {_load_elapsed:.1f}s", flush=True)
            else:
                print("  Failed to load replay buffer, starting fresh", flush=True)
        else:
            print(f"\nWARNING: Replay buffer not found at {load_path}, starting fresh", flush=True)

    # Resume from checkpoint (load model weights)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"\nLoading checkpoint: {checkpoint_path} ({ckpt_size_mb:.1f} MB)...", flush=True)
        print("  Reading checkpoint file...", end=" ", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print("done", flush=True)
        print("  Loading model weights...", end=" ", flush=True)
        network.load_state_dict(checkpoint['model_state_dict'])
        print("done", flush=True)
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

        # Truncate metrics/eval history to discard records from reverted iterations
        metrics_removed, eval_removed = truncate_history_to_iteration(
            metrics_history, eval_history, start_iter
        )
        if metrics_removed > 0 or eval_removed > 0:
            print(f"  Truncated metrics: kept {len(metrics_history['iterations'])} of "
                  f"{len(metrics_history['iterations']) + metrics_removed} records")
            print(f"  Truncated evals:   kept {len(eval_history['evaluations'])} of "
                  f"{len(eval_history['evaluations']) + eval_removed} records")
            save_metrics_history(run_dir, metrics_history)
            save_eval_history(run_dir, eval_history)

    # Create evaluator and self-play (only needed for sequential mode)
    evaluator = None
    self_play = None
    if args.workers <= 1:
        evaluator = BatchedEvaluator(network, device, use_amp=(device == "cuda"))
        self_play = CppSelfPlay(
            evaluator=evaluator,
            num_simulations=args.simulations,
            mcts_batch_size=args.search_batch,
            c_puct=args.c_puct,
            temperature_moves=args.temperature_moves,
            draw_score=args.draw_score
        )

    # Metrics tracker (for console output)
    print("Initializing metrics tracker...", end=" ", flush=True)
    metrics_tracker = MetricsTracker()
    print("done", flush=True)

    # Visual dashboard (optional - saves to HTML files in run_dir)
    # Note: The old dashboard is deprecated; we now generate summary.html directly
    print("Initializing dashboard...", end=" ", flush=True)
    dashboard = TrainingDashboard(
        output_dir=run_dir,  # Dashboard files go in run directory
        update_interval=1
    )
    print("done", flush=True)

    # Live dashboard (optional - real-time web server)
    live_dashboard = None
    if args.live:
        print("\nInitializing LIVE dashboard server...", flush=True)
        try:
            from live_dashboard import LiveDashboardServer
            live_dashboard = LiveDashboardServer(port=args.dashboard_port)
            if live_dashboard.start(total_iterations=args.iterations, open_browser=True):
                print(f"  Real-time updates via WebSocket", flush=True)
            else:
                live_dashboard = None
        except ImportError as e:
            print(f"  WARNING: Could not import live dashboard: {e}", flush=True)
            print(f"  Install requirements: pip install flask flask-socketio", flush=True)
            live_dashboard = None

    print("\n" + "=" * 60, flush=True)
    print("INITIALIZATION COMPLETE - Starting training...", flush=True)
    print("=" * 60 + "\n", flush=True)

    # Install graceful shutdown handler
    shutdown_handler.install_handler()

    # Emergency save function
    def emergency_save(iteration_num: int, reason: str):
        """Save checkpoint and replay buffer on shutdown."""
        print(f"\n{'=' * 70}")
        print(f"EMERGENCY SAVE ({reason})")
        print(f"{'=' * 70}")

        # Save checkpoint
        emergency_path = os.path.join(run_dir, f"model_iter_{iteration_num:03d}_emergency.pt")
        torch.save({
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
            },
            'backend': 'cpp',
            'version': '2.0',
            'emergency_save': True
        }, emergency_path)
        print(f"  Saved checkpoint: {emergency_path}")

        # Save replay buffer (always save for emergency recovery)
        if replay_buffer.save(buffer_path):
            stats = replay_buffer.get_stats()
            print(f"  Saved replay buffer: {buffer_path} ({stats['size']:,} samples)")

        # Save metrics history
        save_metrics_history(run_dir, metrics_history)
        save_eval_history(run_dir, eval_history)
        print(f"  Saved metrics to: {run_dir}")

        print(f"{'=' * 70}\n")

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

        # Self-play phase
        print(f"Iteration {iteration + 1}/{args.iterations}")
        print(f"  Self-play: generating {args.games_per_iter} games...")

        if args.workers > 1:
            # ================================================================
            # PARALLEL SELF-PLAY (Cross-Game Batching)
            # ================================================================
            # Uses ParallelSelfPlayCoordinator for high GPU utilization
            # All games run concurrently, NN evals batched across games
            parallel_metrics = run_parallel_selfplay(
                network=network,
                replay_buffer=replay_buffer,
                device=device,
                args=args,
                iteration=iteration + 1,  # 1-indexed for display
                live_dashboard=live_dashboard
            )
            metrics.num_games = parallel_metrics.num_games
            metrics.total_moves = parallel_metrics.total_moves
            metrics.total_simulations = parallel_metrics.total_simulations
            metrics.total_nn_evals = parallel_metrics.total_nn_evals
            metrics.white_wins = parallel_metrics.white_wins
            metrics.black_wins = parallel_metrics.black_wins
            metrics.draws = parallel_metrics.draws
            metrics.selfplay_time = parallel_metrics.selfplay_time
            metrics.avg_game_length = parallel_metrics.avg_game_length
        else:
            # ================================================================
            # SEQUENTIAL SELF-PLAY (Original)
            # ================================================================
            # Games played one-by-one, each game makes its own NN calls
            network.eval()
            selfplay_start = time.time()
            games_completed = 0

            # Initialize progress reporter for this iteration
            progress = ProgressReporter(interval=args.progress_interval)

            # Track time for live dashboard updates (every 5 seconds)
            last_live_update = time.time()
            live_update_interval = 5.0  # seconds

            for game_idx in range(args.games_per_iter):
                # Check for shutdown between games
                if shutdown_handler.should_stop():
                    print(f"  Stopping after {game_idx} games (shutdown requested)")
                    break

                obs_list, policy_list, result, num_moves, total_sims, total_evals = self_play.play_game()
                games_completed += 1

                # Add game data to C++ ReplayBuffer
                # Flatten observations for storage: (122, 8, 8) -> (7808,)
                # Value must be from side-to-move perspective (matches C++ game.hpp::set_outcomes)
                for i, (obs, policy) in enumerate(zip(obs_list, policy_list)):
                    obs_flat = obs.flatten().astype(np.float32)  # (7808,)
                    white_to_move = (i % 2 == 0)
                    if result > 0:  # White won
                        value = 1.0 if white_to_move else -1.0
                    elif result < 0:  # Black won
                        value = -1.0 if white_to_move else 1.0
                    else:  # Draw
                        value = args.draw_score if white_to_move else -args.draw_score
                    replay_buffer.add_sample(obs_flat, policy.astype(np.float32), float(value))

                metrics.total_moves += num_moves
                metrics.total_simulations += total_sims
                metrics.total_nn_evals += total_evals

                if result > 0:
                    metrics.white_wins += 1
                elif result < 0:
                    metrics.black_wins += 1
                else:
                    metrics.draws += 1

                # Update progress tracker
                progress.update(num_moves, total_sims, total_evals)

                # Print periodic progress report (every 30 seconds)
                if progress.should_report():
                    progress.report(args.games_per_iter, replay_buffer.size())

                # Push live dashboard update every 5 seconds
                now = time.time()
                if live_dashboard is not None and (now - last_live_update) >= live_update_interval:
                    elapsed_selfplay = now - selfplay_start
                    live_dashboard.push_progress(
                        iteration=iteration + 1,
                        games_completed=games_completed,
                        total_games=args.games_per_iter,
                        moves=progress.total_moves,
                        sims=progress.total_sims,
                        evals=progress.total_evals,
                        elapsed_time=elapsed_selfplay,
                        buffer_size=replay_buffer.size(),
                        phase="selfplay"
                    )
                    last_live_update = now

            metrics.num_games = games_completed
            metrics.selfplay_time = time.time() - selfplay_start
            metrics.avg_game_length = metrics.total_moves / max(metrics.num_games, 1)

        # Handle shutdown after self-play
        if shutdown_handler.should_stop():
            emergency_save(iteration + 1, f"Shutdown after self-play ({metrics.num_games} games)")
            break

        # Training phase
        min_samples = args.train_batch
        if replay_buffer.size() >= min_samples:
            print(f"  Training: {args.epochs} epochs on {replay_buffer.size()} positions...")

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
                    buffer_size=replay_buffer.size(),
                    phase="training"
                )

            train_start = time.time()

            train_metrics = train_iteration(
                network, optimizer, replay_buffer,
                args.train_batch, args.epochs, device, scaler
            )

            metrics.train_time = time.time() - train_start
            metrics.loss = train_metrics['loss']
            metrics.policy_loss = train_metrics['policy_loss']
            metrics.value_loss = train_metrics['value_loss']
            metrics.num_train_batches = train_metrics['num_batches']

            print(f"  Training complete: loss={metrics.loss:.4f} "
                  f"(policy={metrics.policy_loss:.4f}, value={metrics.value_loss:.4f}) "
                  f"in {metrics.train_time:.1f}s")
        else:
            print(f"  Skipping training (buffer={replay_buffer.size()} < min={min_samples})")

        metrics.buffer_size = replay_buffer.size()
        metrics.total_time = time.time() - iter_start

        # Track metrics for console output
        metrics_tracker.add_iteration(metrics)
        metrics_tracker.print_iteration_summary(metrics, args)

        # Append to metrics history (for JSON persistence)
        metrics_history["iterations"].append({
            "iteration": iteration + 1,
            "timestamp": datetime.now().isoformat(),
            "loss": metrics.loss,
            "policy_loss": metrics.policy_loss,
            "value_loss": metrics.value_loss,
            "games": metrics.num_games,
            "moves": metrics.total_moves,
            "simulations": metrics.total_simulations,
            "nn_evals": metrics.total_nn_evals,
            "white_wins": metrics.white_wins,
            "black_wins": metrics.black_wins,
            "draws": metrics.draws,
            "avg_game_length": metrics.avg_game_length,
            "buffer_size": metrics.buffer_size,
            "selfplay_time": metrics.selfplay_time,
            "train_time": metrics.train_time,
            "total_time": metrics.total_time,
        })

        # Run evaluation every iteration (unless disabled)
        if not args.no_eval:
            print(f"  Evaluating...")
            eval_start = time.time()
            eval_results = evaluate_checkpoint(
                network, device,
                simulations=args.simulations,
                search_batch=max(args.search_batch, 32),
                c_puct=args.c_puct
            )
            eval_time = time.time() - eval_start

            # Display evaluation results
            vs_random = eval_results.get("vs_random", {})
            endgame = eval_results.get("endgame", {})
            print(f"    vs Random: {vs_random.get('wins', 0)}/5 wins ({vs_random.get('win_rate', 0)*100:.0f}%)")
            print(f"    Endgame:   {endgame.get('score', 0)}/{endgame.get('total', 5)} puzzles, "
                  f"{endgame.get('move_score', 0)}/{endgame.get('total_moves', 7)} moves "
                  f"({endgame.get('move_accuracy', 0)*100:.0f}%)")
            print(f"    Eval time: {eval_time:.1f}s")

            # Store in metrics for dashboard
            metrics.eval_win_rate = vs_random.get('win_rate', 0.0)
            metrics.eval_endgame_score = endgame.get('score', 0)
            metrics.eval_endgame_total = endgame.get('total', 5)
            metrics.eval_endgame_move_accuracy = endgame.get('move_accuracy', 0.0)

            # Append to evaluation history
            eval_history["evaluations"].append({
                "iteration": iteration + 1,
                "timestamp": datetime.now().isoformat(),
                **eval_results
            })

            # Save evaluation history
            save_eval_history(run_dir, eval_history)

        # Update live dashboard (WebSocket)
        if live_dashboard is not None:
            live_dashboard.push_metrics(metrics)

        # Save checkpoint and replay buffer
        if (iteration + 1) % args.save_interval == 0 or iteration == args.iterations - 1:
            is_final = (iteration == args.iterations - 1)

            # Save model checkpoint
            checkpoint_name = f"model_iter_{iteration + 1:03d}.pt"
            checkpoint_save_path = os.path.join(run_dir, checkpoint_name)
            torch.save({
                'iteration': iteration + 1,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': {
                    'input_channels': INPUT_CHANNELS,
                    'num_filters': args.filters,
                    'num_blocks': args.blocks,
                    'num_actions': POLICY_SIZE,
                    'policy_filters': 2,  # AlphaZero paper standard
                    'value_filters': 1,
                    'value_hidden': 256,  # AlphaZero paper standard
                    'simulations': args.simulations,
                },
                'backend': 'cpp',
                'version': '2.0'  # Updated version for new architecture
            }, checkpoint_save_path)
            print(f"  Saved checkpoint: {checkpoint_save_path}")

            # 3. Save final checkpoint alias
            if is_final:
                final_path = os.path.join(run_dir, "model_final.pt")
                torch.save({
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
                    },
                    'backend': 'cpp',
                    'version': '2.0'
                }, final_path)
                print(f"  Saved final checkpoint: {final_path}")

            # 4. Save replay buffer (always save for potential resume)
            if replay_buffer.save(buffer_path):
                stats = replay_buffer.get_stats()
                print(f"  Saved replay buffer: {buffer_path} ({stats['size']:,} samples)")

            # 5. Save metrics history
            save_metrics_history(run_dir, metrics_history)

            # 6. Generate summary.html (unless disabled)
            if not args.no_visualization:
                summary_path = generate_summary_html(run_dir, config)
                print(f"  Updated summary: {summary_path}")

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
        print(f"  Replay buffer saved: {buffer_path}")
    else:
        metrics_tracker.print_final_summary(args)

    if not shutdown_handler.should_stop():
        print(f"\n{'=' * 70}")
        print(f"TRAINING COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Run directory:     {run_dir}")
        print(f"  Final checkpoint:  {os.path.join(run_dir, 'model_final.pt')}")
        if not args.no_visualization:
            print(f"  Training summary:  {os.path.join(run_dir, 'summary.html')}")
        print(f"  Replay buffer:     {buffer_path}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
