#!/usr/bin/env python3
"""
AlphaZero Training with C++ Backend

This script uses:
- C++ MCTS (alphazero-cpp) for fast tree search with proper leaf evaluation
- C++ ReplayBuffer for high-performance data storage with persistence
- CUDA for neural network inference and training
- 192x15 network architecture (192 filters, 15 residual blocks)
- 122-channel position encoding

Usage:
    # Basic training (recommended starting point)
    uv run python alphazero-cpp/scripts/train.py

    # Custom parameters
    uv run python alphazero-cpp/scripts/train.py --iterations 50 --games-per-iter 100 --simulations 800

    # Resume from checkpoint (includes replay buffer)
    uv run python alphazero-cpp/scripts/train.py --resume checkpoints/cpp_iter_10.pt

Parameters:
    --iterations        Number of training iterations (default: 100)
    --games-per-iter    Self-play games per iteration (default: 50)
    --simulations       MCTS simulations per move (default: 800)
    --search-batch      Leaves to evaluate per MCTS iteration (default: 64)
    --train-batch       Samples per training gradient step (default: 256)
    --lr                Learning rate (default: 0.001)
    --filters           Network filters (default: 192)
    --blocks            Residual blocks (default: 15)
    --buffer-size       Replay buffer size (default: 100000)
    --epochs            Training epochs per iteration (default: 5)
    --temperature-moves Moves with temperature=1 (default: 30)
    --c-puct            MCTS exploration constant (default: 1.5)
    --device            Device: cuda or cpu (default: cuda)
    --save-dir          Checkpoint directory (default: checkpoints)
    --resume            Resume from checkpoint path
    --save-interval     Save checkpoint every N iterations (default: 5)
    --buffer-path       Replay buffer persistence path (default: replay_buffer/latest.rpbf)

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
import os
import sys
import time
import random
import signal
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
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
        temperature_moves: int = 30
    ):
        self.evaluator = evaluator
        self.num_simulations = num_simulations
        self.mcts_batch_size = mcts_batch_size
        self.c_puct = c_puct
        self.temperature_moves = temperature_moves

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
            value = 0.0

        return observations, policies, value, move_count, total_sims, total_evals


# =============================================================================
# Parallel Self-Play with Cross-Game Batching
# =============================================================================

def run_parallel_selfplay(
    network: nn.Module,
    replay_buffer,
    device: str,
    args,
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
    metrics = IterationMetrics(iteration=0)  # Will be set by caller
    selfplay_start = time.time()

    # Calculate how many games per worker to match games_per_iter
    games_per_worker = max(1, args.games_per_iter // args.workers)
    actual_total_games = games_per_worker * args.workers

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
        worker_timeout_ms=args.worker_timeout_ms
    )

    # Set replay buffer so data is stored directly
    coordinator.set_replay_buffer(replay_buffer)

    print(f"  Parallel: {args.workers} workers × {games_per_worker} games = {actual_total_games} games")
    print(f"  eval_batch={args.eval_batch}, gpu_timeout={args.gpu_batch_timeout_ms}ms")

    # Create evaluator callback that will be called from C++ GPU thread
    # Signature: (observations, legal_masks, batch_size) -> (policies, values)
    @torch.no_grad()
    def neural_evaluator(obs_array: np.ndarray, mask_array: np.ndarray, batch_size: int):
        """Neural network evaluator callback for C++ coordinator."""
        # Convert NHWC (batch, 8, 8, 122) to NCHW (batch, 122, 8, 8)
        obs_nchw = np.transpose(obs_array, (0, 3, 1, 2))

        # Move to GPU
        obs_tensor = torch.from_numpy(obs_nchw).float().to(device)
        mask_tensor = torch.from_numpy(mask_array).float().to(device)

        # Forward pass with AMP if available
        if device == "cuda":
            with autocast('cuda'):
                policies, values = network(obs_tensor, mask_tensor)
        else:
            policies, values = network(obs_tensor, mask_tensor)

        # Convert back to numpy for C++
        policies_np = policies.cpu().numpy().astype(np.float32)
        values_np = values.squeeze(-1).cpu().numpy().astype(np.float32)

        return policies_np, values_np

    # Run generation (blocking - all games complete)
    result = coordinator.generate_games(neural_evaluator)

    # Extract stats from result dict
    if isinstance(result, dict):
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

        # Calculate NN evals per move (should be ~51 for 800 sims, batch 64)
        nn_evals_per_move = metrics.total_nn_evals / max(metrics.total_moves, 1)
        failure_rate = mcts_failures / max(metrics.total_moves, 1) * 100

        print(f"  Parallel stats: {metrics.total_nn_evals:,} NN evals ({nn_evals_per_move:.1f}/move), "
              f"avg_batch={avg_batch:.1f}, batches={total_batches:,}")

        if mcts_failures > 0 or pool_exhaustion > 0 or partial_subs > 0:
            print(f"  ⚠️  Pipeline issues: {mcts_failures} MCTS failures ({failure_rate:.1f}%), "
                  f"pool_exhaustion={pool_exhaustion}, partial_subs={partial_subs}, "
                  f"drops={submission_drops}, resets={pool_resets}")
            if failure_rate > 10:
                print(f"  ⚠️  HIGH FAILURE RATE - Consider increasing timeouts or queue capacity")
    else:
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

    # Estimate NN evals (approximately sims_per_move * moves_per_game / batch_efficiency)
    # This is approximate since we don't track exact evals in parallel mode
    metrics.total_simulations = metrics.total_moves * args.simulations
    metrics.total_nn_evals = metrics.total_simulations // (args.search_batch // 2)  # Rough estimate

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
    # Basic training
    uv run python alphazero-cpp/scripts/train.py

    # Faster iteration (fewer games, more iterations)
    uv run python alphazero-cpp/scripts/train.py --iterations 200 --games-per-iter 25

    # Higher quality (more simulations)
    uv run python alphazero-cpp/scripts/train.py --simulations 1600

    # Resume training
    uv run python alphazero-cpp/scripts/train.py --resume checkpoints/cpp_iter_50.pt
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
    parser.add_argument("--search-batch", type=int, default=64,
                        help="Leaves to evaluate per MCTS iteration (default: 64)")
    parser.add_argument("--c-puct", type=float, default=1.5,
                        help="MCTS exploration constant (default: 1.5)")
    parser.add_argument("--temperature-moves", type=int, default=30,
                        help="Moves with temperature=1 for exploration (default: 30)")

    # Parallel self-play (automatically enabled when workers > 1)
    parser.add_argument("--workers", type=int, default=1,
                        help="Self-play workers. 1=sequential, >1=parallel with cross-game batching (default: 1)")
    parser.add_argument("--eval-batch", type=int, default=512,
                        help="Max positions per GPU call in parallel mode (default: 512)")
    parser.add_argument("--gpu-batch-timeout-ms", type=int, default=20,
                        help="GPU batch collection timeout in ms (default: 20)")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3,
                        help="Dirichlet noise alpha for root exploration (default: 0.3)")
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25,
                        help="Dirichlet noise weight for root exploration (default: 0.25)")
    parser.add_argument("--worker-timeout-ms", type=int, default=2000,
                        help="Worker wait time for NN results in ms (default: 2000)")

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
                        help="Checkpoint directory (default: checkpoints)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="Save checkpoint every N iterations (default: 5)")
    parser.add_argument("--buffer-path", type=str, default="replay_buffer/latest.rpbf",
                        help="Replay buffer persistence path (default: replay_buffer/latest.rpbf)")
    parser.add_argument("--progress-interval", type=float, default=30.0,
                        help="Print progress statistics every N seconds (default: 30)")

    # Visual dashboard
    parser.add_argument("--visual", action="store_true",
                        help="Enable visual training dashboard (saves to HTML, requires plotly)")
    parser.add_argument("--live", action="store_true",
                        help="Enable LIVE web dashboard with real-time updates (requires flask)")
    parser.add_argument("--dashboard-dir", type=str, default="training_dashboard",
                        help="Directory for dashboard files (default: training_dashboard)")
    parser.add_argument("--dashboard-interval", type=int, default=1,
                        help="Update dashboard every N iterations (default: 1)")
    parser.add_argument("--dashboard-port", type=int, default=5000,
                        help="Port for live dashboard server (default: 5000)")

    args = parser.parse_args()

    # Validate parameter relationships for parallel mode
    if args.workers > 1:
        expected_batch = args.search_batch * args.workers
        if args.eval_batch < expected_batch:
            print(f"  WARNING: eval-batch ({args.eval_batch}) < search-batch*workers ({expected_batch})")
            print(f"           Consider increasing --eval-batch for better GPU utilization")

    # Handle device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.buffer_path) or ".", exist_ok=True)

    # Print configuration
    print("=" * 70)
    print("AlphaZero Training with C++ Backend")
    print("=" * 70)
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
    print(f"Buffer:              {args.buffer_size} positions (path: {args.buffer_path})")
    print(f"Checkpoints:         {args.save_dir}/cpp_iter_*.pt (every {args.save_interval} iters)")
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

    # Create C++ ReplayBuffer with persistence
    replay_buffer = alphazero_cpp.ReplayBuffer(capacity=args.buffer_size)

    # Load existing replay buffer if available
    if os.path.exists(args.buffer_path):
        print(f"\nLoading replay buffer: {args.buffer_path}")
        if replay_buffer.load(args.buffer_path):
            stats = replay_buffer.get_stats()
            print(f"Loaded {stats['size']:,} samples from previous runs")
        else:
            print("Failed to load replay buffer, starting fresh")

    # Resume from checkpoint
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\nLoading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint.get('iteration', 0)
        print(f"Resumed from iteration {start_iter}")

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
            temperature_moves=args.temperature_moves
        )

    # Metrics tracker
    metrics_tracker = MetricsTracker()

    # Visual dashboard (optional - saves to HTML files)
    dashboard = TrainingDashboard(
        output_dir=args.dashboard_dir,
        update_interval=args.dashboard_interval
    )
    if args.visual:
        print("\nInitializing visual dashboard (HTML)...")
        if dashboard.enable():
            print(f"  Open in browser: file://{Path(args.dashboard_dir).absolute()}/dashboard.html")

    # Live dashboard (optional - real-time web server)
    live_dashboard = None
    if args.live:
        print("\nInitializing LIVE dashboard server...")
        try:
            from live_dashboard import LiveDashboardServer
            live_dashboard = LiveDashboardServer(port=args.dashboard_port)
            if live_dashboard.start(total_iterations=args.iterations, open_browser=True):
                print(f"  Real-time updates via WebSocket")
            else:
                live_dashboard = None
        except ImportError as e:
            print(f"  WARNING: Could not import live dashboard: {e}")
            print(f"  Install requirements: pip install flask flask-socketio")
            live_dashboard = None

    print("\nStarting training...\n")

    # Install graceful shutdown handler
    shutdown_handler.install_handler()

    # Emergency save function
    def emergency_save(iteration_num: int, reason: str):
        """Save checkpoint and replay buffer on shutdown."""
        print(f"\n{'=' * 70}")
        print(f"EMERGENCY SAVE ({reason})")
        print(f"{'=' * 70}")

        # Save checkpoint
        emergency_path = os.path.join(args.save_dir, f"cpp_iter_{iteration_num}_emergency.pt")
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

        # Save replay buffer
        if replay_buffer.save(args.buffer_path):
            stats = replay_buffer.get_stats()
            print(f"  Saved replay buffer: {args.buffer_path} ({stats['size']:,} samples)")

        print(f"{'=' * 70}\n")

    # Training loop
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
                for obs, policy in zip(obs_list, policy_list):
                    obs_flat = obs.flatten().astype(np.float32)  # (7808,)
                    value = result if len(obs_list) % 2 == 0 else -result
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
            emergency_save(iteration + 1, f"Shutdown after self-play ({games_completed} games)")
            break

        # Training phase
        min_samples = args.train_batch
        if replay_buffer.size() >= min_samples:
            print(f"  Training: {args.epochs} epochs on {replay_buffer.size()} positions...")
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

        # Track metrics
        metrics_tracker.add_iteration(metrics)
        metrics_tracker.print_iteration_summary(metrics, args)

        # Update visual dashboard (HTML)
        if args.visual:
            dashboard.add_iteration(metrics)

        # Update live dashboard (WebSocket)
        if live_dashboard is not None:
            live_dashboard.push_metrics(metrics)

        # Save checkpoint and replay buffer
        if (iteration + 1) % args.save_interval == 0 or iteration == args.iterations - 1:
            # Save model checkpoint
            checkpoint_path = os.path.join(args.save_dir, f"cpp_iter_{iteration + 1}.pt")
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
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

            # Save replay buffer
            if replay_buffer.save(args.buffer_path):
                stats = replay_buffer.get_stats()
                print(f"  Saved replay buffer: {args.buffer_path} ({stats['size']:,} samples)")
            print()

        # Check for shutdown after iteration complete
        if shutdown_handler.should_stop():
            emergency_save(iteration + 1, "Shutdown after training")
            break

    # Uninstall handler
    shutdown_handler.uninstall_handler()

    # Finalize dashboards
    if args.visual:
        dashboard.finalize()

    if live_dashboard is not None:
        live_dashboard.complete()
        print(f"\n  Live dashboard still running at http://127.0.0.1:{args.dashboard_port}")
        print(f"  Press Ctrl+C to exit completely")

    # Final summary
    if shutdown_handler.should_stop():
        print("\n" + "=" * 70)
        print("TRAINING INTERRUPTED - Graceful shutdown complete")
        print("=" * 70)
        print(f"  Resume with: --resume {args.save_dir}/cpp_iter_*_emergency.pt")
        print(f"  Replay buffer saved: {args.buffer_path}")
    else:
        metrics_tracker.print_final_summary(args)

    if not shutdown_handler.should_stop():
        print(f"\nFinal checkpoint: {args.save_dir}/cpp_iter_{args.iterations}.pt")
        print(f"Replay buffer: {args.buffer_path}")
        if args.visual:
            print(f"Training dashboard: {Path(args.dashboard_dir).absolute()}/dashboard.html")
        print("=" * 70)


if __name__ == "__main__":
    main()
