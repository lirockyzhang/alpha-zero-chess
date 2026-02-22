"""
Live Training Dashboard Server

Real-time web dashboard for monitoring AlphaZero training progress.
Uses Flask + WebSockets for live updates to the browser.

Features:
- Real-time chart updates via WebSocket
- Interactive Plotly.js charts
- Auto-reconnect on connection loss
- Responsive design

Usage:
    # Standalone (for testing)
    python live_dashboard.py

    # Integrated with training (via --visual flag)
    python train.py --visual
"""

import json
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Try to import Flask and dependencies
try:
    from flask import Flask, render_template_string, jsonify
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


@dataclass
class DashboardMetrics:
    """Metrics for a single iteration."""
    iteration: int
    timestamp: str
    elapsed_minutes: float

    # Loss metrics
    total_loss: Optional[float] = None
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None

    # Performance metrics
    moves_per_sec: float = 0.0
    sims_per_sec: float = 0.0
    nn_evals_per_sec: float = 0.0
    games_per_hour: float = 0.0

    # Game statistics
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    avg_game_length: float = 0.0
    num_games: int = 0

    # Buffer and timing
    buffer_size: int = 0
    selfplay_time: float = 0.0
    train_time: float = 0.0
    iteration_time: float = 0.0


# HTML template with embedded Plotly.js for real-time charts
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlphaZero Training Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #eee;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .header h1 {
            font-size: 2em;
            margin-bottom: 10px;
        }
        .header .stats {
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
        }
        .header .stat {
            background: rgba(255,255,255,0.1);
            padding: 10px 20px;
            border-radius: 10px;
        }
        .header .stat-value {
            font-size: 1.5em;
            font-weight: bold;
        }
        .header .stat-label {
            font-size: 0.8em;
            opacity: 0.8;
        }
        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            margin-top: 10px;
        }
        .status.connected {
            background: #2ecc71;
            color: white;
        }
        .status.disconnected {
            background: #e74c3c;
            color: white;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .card h3 {
            margin-bottom: 10px;
            color: #a8d8ea;
            font-size: 1em;
        }
        .chart {
            width: 100%;
            height: 250px;
        }
        .card-wide {
            grid-column: span 2;
        }
        .card-wide .chart {
            height: 300px;
        }
        .footer {
            text-align: center;
            padding: 20px;
            opacity: 0.5;
            font-size: 0.9em;
        }
        .system-monitoring {
            background: rgba(0,0,0,0.3);
            padding: 15px 20px;
            margin: 0;
        }
        .monitoring-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .monitoring-phase {
            font-size: 1.1em;
            font-weight: bold;
            color: #a8d8ea;
        }
        .monitoring-phase .phase-icon {
            margin-right: 8px;
        }
        .monitoring-eta {
            font-size: 0.9em;
            color: #95a5a6;
        }
        .monitoring-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }
        .monitoring-card {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 12px;
        }
        .monitoring-card h4 {
            font-size: 0.9em;
            color: #a8d8ea;
            margin: 0 0 10px 0;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.85em;
            padding: 3px 0;
        }
        .metric-row span:first-child {
            color: #95a5a6;
        }
        .metric-row span:last-child {
            font-weight: bold;
            color: #3498db;
        }
        .bottleneck-indicator {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            font-size: 0.95em;
        }
        .bottleneck-value {
            font-weight: bold;
            padding: 4px 12px;
            border-radius: 4px;
            background: rgba(255,255,255,0.1);
        }
        .bottleneck-value.underfilled {
            color: #e67e22;
            background: rgba(230, 126, 34, 0.2);
        }
        .bottleneck-value.gpu-saturated {
            color: #3498db;
            background: rgba(52, 152, 219, 0.2);
        }
        .bottleneck-value.balanced {
            color: #2ecc71;
            background: rgba(46, 204, 113, 0.2);
        }
        .metric-row span.highlight {
            color: #2ecc71;
            animation: pulse 1s ease-in-out;
        }
        .metric-row span.warn {
            color: #e67e22;
            font-weight: bold;
        }
        .metric-row span.ok {
            color: #2ecc71;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        @media (max-width: 600px) {
            .grid {
                grid-template-columns: 1fr;
            }
            .card-wide {
                grid-column: span 1;
            }
            .header .stats {
                flex-direction: column;
                gap: 10px;
            }
            .live-stats {
                gap: 10px;
            }
            .live-stat {
                min-width: 80px;
            }
        }
        /* Parameter Controls */
        .param-controls-section {
            background: rgba(0,0,0,0.25);
            border-top: 1px solid rgba(255,255,255,0.1);
            padding: 0;
        }
        .param-controls-header {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 20px;
            cursor: pointer;
            user-select: none;
            transition: background 0.2s;
        }
        .param-controls-header:hover {
            background: rgba(255,255,255,0.05);
        }
        .param-controls-header .gear-icon {
            font-size: 1.1em;
        }
        .param-controls-header .toggle-arrow {
            transition: transform 0.3s;
            font-size: 0.8em;
            opacity: 0.6;
        }
        .param-controls-header .toggle-arrow.open {
            transform: rotate(90deg);
        }
        .param-controls-header h3 {
            flex: 1;
            font-size: 1em;
            color: #a8d8ea;
            margin: 0;
        }
        .param-status {
            font-size: 0.8em;
            padding: 3px 10px;
            border-radius: 12px;
            display: none;
        }
        .param-status.pending {
            display: inline-block;
            background: rgba(241, 196, 15, 0.25);
            color: #f1c40f;
        }
        .param-status.applied {
            display: inline-block;
            background: rgba(46, 204, 113, 0.25);
            color: #2ecc71;
        }
        .param-controls-body {
            padding: 0 20px 15px;
        }
        .param-groups {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }
        .param-group {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 12px;
        }
        .param-group h4 {
            font-size: 0.9em;
            color: #a8d8ea;
            margin: 0 0 10px 0;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .param-row {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 4px 0;
            font-size: 0.85em;
        }
        .param-row label {
            flex: 1;
            color: #95a5a6;
            min-width: 0;
        }
        .param-row input[type="number"] {
            width: 100px;
            padding: 4px 6px;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 4px;
            color: #eee;
            font-size: 0.9em;
            text-align: right;
        }
        .param-row input[type="number"]:focus {
            outline: none;
            border-color: #667eea;
        }
        .param-row input[type="number"].modified {
            border-color: #f1c40f;
            box-shadow: 0 0 4px rgba(241, 196, 15, 0.3);
        }
        .param-row .param-current {
            font-size: 0.8em;
            color: #666;
            min-width: 70px;
            text-align: right;
        }
        .param-actions {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .param-apply-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 8px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: bold;
            transition: opacity 0.2s;
        }
        .param-apply-btn:hover {
            opacity: 0.9;
        }
        .param-apply-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .param-reset-btn {
            background: rgba(255,255,255,0.1);
            color: #ccc;
            border: 1px solid rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85em;
            transition: background 0.2s;
        }
        .param-reset-btn:hover {
            background: rgba(255,255,255,0.15);
        }
        .param-feedback {
            font-size: 0.85em;
            color: #95a5a6;
        }
        .export-section {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 20px;
            margin: 0 20px 10px;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
        }
        .export-btn {
            background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
            color: white;
            border: none;
            padding: 8px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: bold;
            transition: opacity 0.2s;
        }
        .export-btn:hover {
            opacity: 0.9;
        }
        .export-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .export-feedback {
            font-size: 0.85em;
            color: #95a5a6;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>AlphaZero Live Training Dashboard</h1>
        <div class="stats">
            <div class="stat">
                <div class="stat-value" id="elapsed">00:00:00</div>
                <div class="stat-label">Elapsed Time</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="iteration">0</div>
                <div class="stat-label">Iteration</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="total-games">0</div>
                <div class="stat-label">Total Games</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="total-moves">0</div>
                <div class="stat-label">Total Moves</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="buffer-size">0</div>
                <div class="stat-label">Buffer Size</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="current-loss">-</div>
                <div class="stat-label">Loss</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="avg-iter-time">-</div>
                <div class="stat-label">Avg Iter Time</div>
            </div>
        </div>
        <div class="status disconnected" id="status">Connecting...</div>
    </div>

    <!-- System Monitoring Section -->
    <div class="system-monitoring" id="system-monitoring">
        <div class="monitoring-header">
            <div class="monitoring-phase">
                <span class="phase-icon" id="phase-icon">🎮</span>
                <span id="phase-text">Waiting to start...</span>
            </div>
            <div class="monitoring-eta">
                ETA: <span id="phase-eta">--</span>
            </div>
        </div>
        <div class="monitoring-grid">
            <!-- MCTS Performance Card -->
            <div class="monitoring-card">
                <h4>🔄 MCTS Performance</h4>
                <div class="metric-row"><span>Sims/sec:</span><span id="sys-sims-sec">0</span></div>
                <div class="metric-row"><span>Moves/sec:</span><span id="sys-moves-sec">0</span></div>
                <div class="metric-row"><span>Queue Fill:</span><span id="queue-fill-pct">0%</span></div>
                <div class="metric-row"><span>Worker Wait:</span><span id="worker-wait-ms">0ms</span></div>
                <div class="metric-row"><span>Drop %:</span><span id="pool-load">0%</span></div>
                <div class="metric-row"><span>Pipeline:</span><span id="pipeline-status">✅ Healthy</span></div>
                <div class="metric-row"><span>Tree Depth:</span><span id="live-tree-depth">--</span></div>
            </div>

            <!-- GPU Performance Card -->
            <div class="monitoring-card">
                <h4>🎮 GPU Performance</h4>
                <div class="metric-row"><span>Avg Batch Size:</span><span id="avg-batch-size">0</span></div>
                <div class="metric-row"><span>NN Evals/sec:</span><span id="sys-evals-sec">0</span></div>
                <div class="metric-row"><span>Batch Fill:</span><span id="batch-fill-ratio">0%</span></div>
                <div class="metric-row"><span>Batch Fire:</span><span id="batch-fire-reason">--</span></div>
                <div class="metric-row"><span>GPU Wait:</span><span id="gpu-wait-ms">0ms</span></div>
                <div class="metric-row"><span>Avg Infer:</span><span id="avg-infer-time">0ms</span></div>
                <div class="metric-row"><span>GPU Memory:</span><span id="gpu-memory">0 MB</span></div>
                <div class="metric-row"><span>CUDA Graph:</span><span id="cuda-graph-status">--</span></div>
            </div>

            <!-- Reanalysis Performance Card (hidden until reanalysis data arrives) -->
            <div class="monitoring-card" id="reanalysis-perf-card" style="display:none;">
                <h4>🔄 Reanalysis Performance</h4>
                <div class="metric-row"><span>Speed:</span><span id="reanalysis-speed">0 pos/s</span></div>
                <div class="metric-row"><span>Completed:</span><span id="reanalysis-completed">0</span></div>
                <div class="metric-row"><span>Skipped:</span><span id="reanalysis-skipped">0</span></div>
                <div class="metric-row"><span>NN Evals:</span><span id="reanalysis-nn-evals">0</span></div>
                <div class="metric-row"><span>Mean KL:</span><span id="reanalysis-kl">0.000</span></div>
                <div style="margin-top: 6px;">
                    <div style="background: rgba(255,255,255,0.1); border-radius: 4px; height: 8px; overflow: hidden;">
                        <div id="reanalysis-progress-bar" style="background: linear-gradient(90deg, #1abc9c, #2ecc71); height: 100%; width: 0%; transition: width 0.5s;"></div>
                    </div>
                    <div style="font-size: 0.7em; color: #95a5a6; margin-top: 2px; text-align: center;" id="reanalysis-progress-text">0%</div>
                </div>
            </div>

            <!-- Training Progress Card (hidden until training phase) -->
            <div class="monitoring-card" id="training-progress-card" style="display:none;">
                <h4>🧠 Training Progress</h4>
                <div class="metric-row"><span>Epoch:</span><span id="train-epoch">--</span></div>
                <div class="metric-row"><span>Loss:</span><span id="train-loss">--</span></div>
                <div class="metric-row"><span>Policy Loss:</span><span id="train-policy-loss">--</span></div>
                <div class="metric-row"><span>Value Loss:</span><span id="train-value-loss">--</span></div>
                <div class="metric-row"><span>Grad Norm:</span><span id="train-grad-norm">--</span></div>
                <div style="margin-top: 6px;">
                    <div style="background: rgba(255,255,255,0.1); border-radius: 4px; height: 8px; overflow: hidden;">
                        <div id="train-progress-bar" style="background: linear-gradient(90deg, #9b59b6, #3498db); height: 100%; width: 0%; transition: width 0.5s;"></div>
                    </div>
                    <div style="font-size: 0.7em; color: #95a5a6; margin-top: 2px; text-align: center;" id="train-progress-text">0%</div>
                </div>
            </div>

            <!-- Iteration Progress Card -->
            <div class="monitoring-card">
                <h4>📊 Iteration Progress</h4>
                <div class="metric-row"><span>Run Time:</span><span id="live-run-time">--</span></div>
                <div class="metric-row"><span>Games:</span><span id="live-games">0/0</span></div>
                <div class="metric-row"><span>W/D/L:</span><span id="live-wdl">0 / 0 / 0</span></div>
                <div class="metric-row"><span>Moves:</span><span id="live-moves">0</span></div>
                <div class="metric-row"><span>Games/min:</span><span id="live-gph">0</span></div>
                <div class="metric-row"><span>Game Moves:</span><span id="live-current-moves">--</span></div>
                <div class="metric-row"><span>Risk β:</span><span id="live-risk-beta">--</span></div>
            </div>

            <!-- Refutation Status Card (hidden until asymmetric risk data arrives) -->
            <div class="monitoring-card" id="refutation-card" style="display:none;">
                <h4>Refutation Status</h4>
                <div class="metric-row"><span>Standard Wins:</span><span id="ref-std-wins">0</span></div>
                <div class="metric-row"><span>Opponent Wins:</span><span id="ref-opp-wins">0</span></div>
                <div class="metric-row"><span>Draws:</span><span id="ref-draws">0</span></div>
                <div class="metric-row"><span>Refutation Elo:</span><span id="ref-elo" style="font-size:1.1em;">--</span></div>
            </div>

        </div>

        <!-- Bottleneck Indicator -->
        <div class="bottleneck-indicator">
            <span>System Status:</span>
            <span id="bottleneck-type" class="bottleneck-value">--</span>
        </div>

        <!-- Batch Analysis Section (below System Status) -->
        <div style="display: flex; gap: 15px; margin-top: 15px;">
            <!-- Batch Size Distribution -->
            <div class="monitoring-card" style="flex: 2;">
                <h4>📈 Batch Size Distribution</h4>
                <div id="batch-histogram" style="width: 100%; height: 180px;"></div>
                <div style="display: flex; justify-content: center; gap: 15px; font-size: 0.75em; margin-top: 4px;">
                    <span><span style="color: #2ecc71;">■</span> Large Graph</span>
                    <span><span style="color: #1abc9c;">■</span> Medium Graph</span>
                    <span><span style="color: #3498db;">■</span> Small Graph</span>
                    <span><span style="color: #9b59b6;">■</span> Mini Graph</span>
                    <span><span style="color: #e67e22;">■</span> Eager</span>
                </div>
                <div class="metric-row" style="margin-top: 4px;">
                    <span>Thresholds:</span>
                    <span id="batch-thresholds">mini≤--, small≤--, medium≤--, large>--</span>
                </div>
                <div class="metric-row">
                    <span>P25/P50/P75/P90:</span>
                    <span id="batch-percentiles">-- / -- / -- / --</span>
                </div>
            </div>

            <!-- Execution Path Pie Charts -->
            <div class="monitoring-card" style="flex: 1;">
                <h4>🎯 Execution Paths</h4>
                <div style="display: flex; gap: 4px;">
                    <div style="flex: 1; text-align: center;">
                        <div style="font-size: 0.7em; color: #95a5a6; margin-bottom: 2px;">Batch Count</div>
                        <div id="exec-path-pie" style="width: 100%; height: 160px;"></div>
                    </div>
                    <div style="flex: 1; text-align: center;">
                        <div style="font-size: 0.7em; color: #95a5a6; margin-bottom: 2px;">Time Spent</div>
                        <div id="exec-time-pie" style="width: 100%; height: 160px;"></div>
                    </div>
                </div>
                <div class="metric-row" style="margin-top: 4px;">
                    <span>Total Batches:</span>
                    <span id="total-batches">0</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Export Model Section -->
    <div class="export-section">
        <button class="export-btn" id="export-btn" onclick="exportModel()">Export Model Weights</button>
        <span class="export-feedback" id="export-feedback"></span>
    </div>

    <!-- Parameter Controls Section -->
    <div class="param-controls-section">
        <div class="param-controls-header" onclick="toggleParamControls()">
            <span class="gear-icon">&#9881;</span>
            <h3>Parameter Controls</h3>
            <span class="param-status" id="param-status"></span>
            <span class="toggle-arrow" id="param-toggle-arrow">&#9654;</span>
        </div>
        <div class="param-controls-body" id="param-controls-body" style="display: none;">
            <div class="param-groups">
                <!-- Training Group -->
                <div class="param-group">
                    <h4>Training</h4>
                    <div class="param-row">
                        <label>Learning Rate</label>
                        <input type="number" data-param="lr" step="0.0001" min="0.000001" max="1">
                        <span class="param-current" id="cur-lr"></span>
                    </div>
                    <div class="param-row">
                        <label>Batch Size</label>
                        <input type="number" data-param="train_batch" step="1" min="16" max="8192">
                        <span class="param-current" id="cur-train_batch"></span>
                    </div>
                    <div class="param-row">
                        <label>Epochs</label>
                        <input type="number" data-param="epochs" step="1" min="1" max="100">
                        <span class="param-current" id="cur-epochs"></span>
                    </div>
                </div>

                <!-- MCTS / Search Group -->
                <div class="param-group">
                    <h4>MCTS / Search</h4>
                    <div class="param-row">
                        <label>Simulations</label>
                        <input type="number" data-param="simulations" step="50" min="50" max="10000">
                        <span class="param-current" id="cur-simulations"></span>
                    </div>
                    <div class="param-row">
                        <label>C_Explore</label>
                        <input type="number" data-param="c_explore" step="0.1" min="0.1" max="10">
                        <span class="param-current" id="cur-c_explore"></span>
                    </div>
                    <div class="param-row">
                        <label>Risk Beta</label>
                        <input type="number" data-param="risk_beta" step="0.1" min="-3" max="3">
                        <span class="param-current" id="cur-risk_beta"></span>
                    </div>
                    <div class="param-row">
                        <label>Opponent Min</label>
                        <input type="number" data-param="opponent_risk_min" step="0.1" min="-3" max="3">
                        <span class="param-current" id="cur-opponent_risk_min"></span>
                    </div>
                    <div class="param-row">
                        <label>Opponent Max</label>
                        <input type="number" data-param="opponent_risk_max" step="0.1" min="-3" max="3">
                        <span class="param-current" id="cur-opponent_risk_max"></span>
                    </div>
                    <div class="param-row">
                        <label>Temp Moves</label>
                        <input type="number" data-param="temperature_moves" step="1" min="0" max="200">
                        <span class="param-current" id="cur-temperature_moves"></span>
                    </div>
                    <div class="param-row">
                        <label>Dir. Alpha</label>
                        <input type="number" data-param="dirichlet_alpha" step="0.01" min="0.01" max="2">
                        <span class="param-current" id="cur-dirichlet_alpha"></span>
                    </div>
                    <div class="param-row">
                        <label>Dir. Epsilon</label>
                        <input type="number" data-param="dirichlet_epsilon" step="0.05" min="0" max="1">
                        <span class="param-current" id="cur-dirichlet_epsilon"></span>
                    </div>
                    <div class="param-row">
                        <label>FPU Base</label>
                        <input type="number" data-param="fpu_base" step="0.05" min="0" max="2">
                        <span class="param-current" id="cur-fpu_base"></span>
                    </div>
                </div>

                <!-- Self-Play Group -->
                <div class="param-group">
                    <h4>Self-Play</h4>
                    <div class="param-row">
                        <label>Games/Iter</label>
                        <input type="number" data-param="games_per_iter" step="1" min="1" max="10000">
                        <span class="param-current" id="cur-games_per_iter"></span>
                    </div>
                    <div class="param-row">
                        <label>Max Fillup</label>
                        <input type="number" data-param="max_fillup_factor" step="1" min="0" max="100">
                        <span class="param-current" id="cur-max_fillup_factor"></span>
                    </div>
                    <div class="param-row">
                        <label>Save Interval</label>
                        <input type="number" data-param="save_interval" step="1" min="1" max="1000">
                        <span class="param-current" id="cur-save_interval"></span>
                    </div>
                </div>
            </div>

            <div class="param-actions">
                <button class="param-apply-btn" id="param-apply-btn" onclick="applyParams()">Apply for Next Phase</button>
                <button class="param-reset-btn" onclick="resetParams()">Reset to Current</button>
                <span class="param-feedback" id="param-feedback"></span>
            </div>
        </div>
    </div>

    <div class="grid">
        <!-- Policy & Value Loss -->
        <div class="card">
            <h3>📈 Policy vs Value Loss</h3>
            <div id="pv-loss-chart" class="chart"></div>
        </div>

        <!-- Performance Chart -->
        <div class="card">
            <h3>⚡ Performance (Moves/sec)</h3>
            <div id="perf-chart" class="chart"></div>
        </div>

        <!-- Win/Draw/Loss -->
        <div class="card">
            <h3>🎯 Game Results</h3>
            <div id="wdl-chart" class="chart"></div>
        </div>

        <!-- Draw Breakdown -->
        <div class="card">
            <h3>📊 Draw Breakdown</h3>
            <div id="draw-breakdown-chart" class="chart"></div>
        </div>

        <!-- MCTS Simulations -->
        <div class="card">
            <h3>🔬 MCTS Simulations/sec</h3>
            <div id="sims-chart" class="chart"></div>
        </div>

        <!-- Time Breakdown -->
        <div class="card">
            <h3>⏱️ Time Breakdown</h3>
            <div id="time-chart" class="chart"></div>
        </div>

        <!-- Average Game Length -->
        <div class="card">
            <h3>📊 Average Game Length</h3>
            <div id="game-length-chart" class="chart"></div>
        </div>

        <!-- Reanalysis -->
        <div class="card card-wide" id="reanalysis-card" style="display:none;">
            <h3>🔄 Reanalysis</h3>
            <div id="reanalysis-chart" class="chart"></div>
        </div>

    </div>

    <div class="footer">
        AlphaZero Training Dashboard | Real-time WebSocket Updates
    </div>

    <script>
        // Data storage
        const data = {
            iterations: [],
            policyLoss: [],
            valueLoss: [],
            movesPerSec: [],
            simsPerSec: [],
            whiteWins: [],
            blackWins: [],
            draws: [],
            drawsRepetition: [],
            drawsStalemate: [],
            drawsFiftyMove: [],
            drawsInsufficient: [],
            drawsMaxMoves: [],
            selfplayTime: [],
            trainTime: [],
            avgGameLength: [],
            reanalysisPositions: [],
            reanalysisKL: [],
        };

        let totalGames = 0;
        let totalMoves = 0;
        let startTime = Date.now();
        let totalIterations = 100;

        // Parameter controls state
        let socket = null;
        let currentParams = {};
        let pendingApplied = false;

        // Chart colors
        const colors = {
            primary: '#3498db',
            secondary: '#2ecc71',
            tertiary: '#e74c3c',
            quaternary: '#9b59b6',
            white: '#ecf0f1',
            draw: '#95a5a6',
            black: '#34495e'
        };

        // Chart layout defaults
        const layoutDefaults = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#ccc', size: 10 },
            margin: { t: 10, b: 40, l: 50, r: 20 },
            xaxis: {
                gridcolor: 'rgba(255,255,255,0.1)',
                title: { text: 'Iteration', font: { size: 10 } }
            },
            yaxis: {
                gridcolor: 'rgba(255,255,255,0.1)'
            },
            showlegend: true,
            legend: {
                orientation: 'h',
                y: -0.2,
                font: { size: 9 }
            }
        };

        // Initialize charts
        function initCharts() {
            // Performance chart
            Plotly.newPlot('perf-chart', [{
                x: [], y: [], type: 'scatter', mode: 'lines+markers',
                name: 'Moves/sec', line: { color: colors.secondary, width: 2 },
                fill: 'tozeroy', fillcolor: 'rgba(46, 204, 113, 0.2)',
                marker: { size: 4 }
            }], {...layoutDefaults, yaxis: {...layoutDefaults.yaxis, title: 'Moves/sec'}});

            // Policy vs Value Loss (Dual Y-Axis)
            Plotly.newPlot('pv-loss-chart', [
                { x: [], y: [], type: 'scatter', mode: 'lines+markers',
                  name: 'Policy Loss', line: { color: colors.primary, width: 2 }, marker: { size: 4 },
                  yaxis: 'y' },
                { x: [], y: [], type: 'scatter', mode: 'lines+markers',
                  name: 'Value Loss', line: { color: colors.tertiary, width: 2 }, marker: { size: 4 },
                  yaxis: 'y2' }
            ], {
                ...layoutDefaults,
                yaxis: { ...layoutDefaults.yaxis, title: 'Policy Loss', side: 'left' },
                yaxis2: {
                    title: 'Value Loss',
                    overlaying: 'y',
                    side: 'right',
                    gridcolor: 'rgba(255,255,255,0.05)'
                }
            });

            // Simulations chart
            Plotly.newPlot('sims-chart', [{
                x: [], y: [], type: 'scatter', mode: 'lines+markers',
                name: 'Sims/sec', line: { color: colors.quaternary, width: 2 },
                fill: 'tozeroy', fillcolor: 'rgba(155, 89, 182, 0.2)',
                marker: { size: 4 }
            }], {...layoutDefaults, yaxis: {...layoutDefaults.yaxis, title: 'Sims/sec'}});

            // Win/Draw/Loss chart
            Plotly.newPlot('wdl-chart', [
                { x: [], y: [], type: 'bar', name: 'White', marker: { color: colors.white } },
                { x: [], y: [], type: 'bar', name: 'Draw', marker: { color: colors.draw } },
                { x: [], y: [], type: 'bar', name: 'Black', marker: { color: colors.black } }
            ], {...layoutDefaults, barmode: 'stack', yaxis: {...layoutDefaults.yaxis, title: 'Games'}});

            // Draw Breakdown chart
            Plotly.newPlot('draw-breakdown-chart', [
                { x: [], y: [], type: 'bar', name: 'Repetition', marker: { color: '#e74c3c' } },
                { x: [], y: [], type: 'bar', name: 'Stalemate', marker: { color: '#f39c12' } },
                { x: [], y: [], type: 'bar', name: 'Fifty-Move', marker: { color: '#9b59b6' } },
                { x: [], y: [], type: 'bar', name: 'Material', marker: { color: '#1abc9c' } },
                { x: [], y: [], type: 'bar', name: 'Max-Moves', marker: { color: '#95a5a6' } }
            ], {...layoutDefaults, barmode: 'stack', yaxis: {...layoutDefaults.yaxis, title: 'Draws'}});

            // Time breakdown chart
            Plotly.newPlot('time-chart', [
                { x: [], y: [], type: 'bar', name: 'Self-Play', marker: { color: colors.primary } },
                { x: [], y: [], type: 'bar', name: 'Training', marker: { color: colors.tertiary } }
            ], {...layoutDefaults, barmode: 'stack', yaxis: {...layoutDefaults.yaxis, title: 'Seconds'}});

            // Average game length chart
            Plotly.newPlot('game-length-chart', [{
                x: [], y: [], type: 'scatter', mode: 'lines+markers',
                name: 'Avg Moves/Game', line: { color: colors.secondary, width: 2 },
                fill: 'tozeroy', fillcolor: 'rgba(46, 204, 113, 0.2)',
                marker: { size: 4 }
            }], {...layoutDefaults, yaxis: {...layoutDefaults.yaxis, title: 'Moves per Game'}});

            // Reanalysis chart (dual y-axis: positions + KL divergence)
            Plotly.newPlot('reanalysis-chart', [
                { x: [], y: [], type: 'scatter', mode: 'lines+markers',
                  name: 'Positions Updated', line: { color: colors.primary, width: 2 },
                  marker: { size: 4 } },
                { x: [], y: [], type: 'scatter', mode: 'lines+markers',
                  name: 'Mean KL', line: { color: colors.tertiary, width: 2, dash: 'dot' },
                  marker: { size: 4 }, yaxis: 'y2' }
            ], {...layoutDefaults,
                yaxis: {...layoutDefaults.yaxis, title: 'Positions'},
                yaxis2: { title: 'KL Divergence', overlaying: 'y', side: 'right',
                          gridcolor: 'rgba(255,255,255,0.05)',
                          titlefont: { color: colors.tertiary },
                          tickfont: { color: colors.tertiary } }
            });

            // Batch size histogram
            Plotly.newPlot('batch-histogram', [{
                x: [], y: [], type: 'bar',
                marker: {
                    color: [],  // Will be set dynamically based on routing
                    line: { color: 'rgba(255,255,255,0.3)', width: 1 }
                },
                hovertemplate: 'Batch %{x}: %{y} calls<extra></extra>'
            }], {
                ...layoutDefaults,
                margin: { t: 5, b: 30, l: 45, r: 10 },
                xaxis: {
                    ...layoutDefaults.xaxis,
                    title: { text: 'Batch Size', font: { size: 9 } },
                    tickmode: 'array',
                    tickvals: [],
                    ticktext: []
                },
                yaxis: { ...layoutDefaults.yaxis, title: { text: 'Count', font: { size: 9 } } },
                showlegend: false,
                bargap: 0.1
            });

            // Execution path pie chart (batch count)
            Plotly.newPlot('exec-path-pie', [{
                values: [0, 0, 0, 0, 0],
                labels: ['Large', 'Medium', 'Small', 'Mini', 'Eager'],
                type: 'pie',
                marker: {
                    colors: ['#2ecc71', '#1abc9c', '#3498db', '#9b59b6', '#e67e22']
                },
                textinfo: 'percent',
                textfont: { color: '#fff', size: 10 },
                hovertemplate: '%{label}: %{value} batches (%{percent})<extra></extra>'
            }], {
                ...layoutDefaults,
                margin: { t: 5, b: 5, l: 5, r: 5 },
                showlegend: false
            });

            // Execution time distribution pie chart
            Plotly.newPlot('exec-time-pie', [{
                values: [0, 0, 0, 0, 0],
                labels: ['Large', 'Medium', 'Small', 'Mini', 'Eager'],
                type: 'pie',
                marker: {
                    colors: ['#2ecc71', '#1abc9c', '#3498db', '#9b59b6', '#e67e22']
                },
                textinfo: 'percent',
                textfont: { color: '#fff', size: 10 },
                hovertemplate: '%{label}: %{value:.0f}ms (%{percent})<extra></extra>'
            }], {
                ...layoutDefaults,
                margin: { t: 5, b: 5, l: 5, r: 5 },
                showlegend: false
            });
        }

        // Update charts with new data
        function updateCharts(metrics) {
            const iter = metrics.iteration;

            // Store data
            data.iterations.push(iter);

            data.policyLoss.push(metrics.policy_loss);
            data.valueLoss.push(metrics.value_loss);
            data.movesPerSec.push(metrics.moves_per_sec);
            data.simsPerSec.push(metrics.sims_per_sec);
            data.whiteWins.push(metrics.white_wins);
            data.blackWins.push(metrics.black_wins);
            data.draws.push(metrics.draws);
            data.drawsRepetition.push(metrics.draws_repetition || 0);
            data.drawsStalemate.push(metrics.draws_stalemate || 0);
            data.drawsFiftyMove.push(metrics.draws_fifty_move || 0);
            data.drawsInsufficient.push(metrics.draws_insufficient || 0);
            data.drawsMaxMoves.push(metrics.draws_max_moves || 0);
            data.selfplayTime.push(metrics.selfplay_time);
            data.trainTime.push(metrics.train_time);
            data.avgGameLength.push(metrics.avg_game_length);
            data.reanalysisPositions.push(metrics.reanalysis_positions || 0);
            data.reanalysisKL.push(metrics.reanalysis_mean_kl || 0);
            // Update cumulative stats
            totalGames += metrics.num_games;
            totalMoves += Math.round(metrics.moves_per_sec * metrics.selfplay_time);

            // Update performance chart
            Plotly.extendTraces('perf-chart', {
                x: [[iter]], y: [[metrics.moves_per_sec]]
            }, [0]);

            // Update policy/value loss chart
            Plotly.extendTraces('pv-loss-chart', {
                x: [[iter], [iter]],
                y: [[metrics.policy_loss], [metrics.value_loss]]
            }, [0, 1]);

            // Update sims chart
            Plotly.extendTraces('sims-chart', {
                x: [[iter]], y: [[metrics.sims_per_sec]]
            }, [0]);

            // Update game length chart
            Plotly.extendTraces('game-length-chart', {
                x: [[iter]], y: [[metrics.avg_game_length]]
            }, [0]);

            // Update WDL chart (all iterations)
            // Use .slice() to create fresh array references so Plotly detects the change
            Plotly.react('wdl-chart', [
                { x: data.iterations.slice(), y: data.whiteWins.slice(), type: 'bar', name: 'White', marker: { color: colors.white } },
                { x: data.iterations.slice(), y: data.draws.slice(), type: 'bar', name: 'Draw', marker: { color: colors.draw } },
                { x: data.iterations.slice(), y: data.blackWins.slice(), type: 'bar', name: 'Black', marker: { color: colors.black } }
            ], {...layoutDefaults, barmode: 'stack', yaxis: {...layoutDefaults.yaxis, title: 'Games'}});

            // Update draw breakdown chart
            Plotly.react('draw-breakdown-chart', [
                { x: data.iterations.slice(), y: data.drawsRepetition.slice(), type: 'bar', name: 'Repetition', marker: { color: '#e74c3c' } },
                { x: data.iterations.slice(), y: data.drawsStalemate.slice(), type: 'bar', name: 'Stalemate', marker: { color: '#f39c12' } },
                { x: data.iterations.slice(), y: data.drawsFiftyMove.slice(), type: 'bar', name: 'Fifty-Move', marker: { color: '#9b59b6' } },
                { x: data.iterations.slice(), y: data.drawsInsufficient.slice(), type: 'bar', name: 'Material', marker: { color: '#1abc9c' } },
                { x: data.iterations.slice(), y: data.drawsMaxMoves.slice(), type: 'bar', name: 'Max-Moves', marker: { color: '#95a5a6' } }
            ], {...layoutDefaults, barmode: 'stack', yaxis: {...layoutDefaults.yaxis, title: 'Draws'}});

            // Update time chart (all iterations)
            Plotly.react('time-chart', [
                { x: data.iterations.slice(), y: data.selfplayTime.slice(), type: 'bar', name: 'Self-Play', marker: { color: colors.primary } },
                { x: data.iterations.slice(), y: data.trainTime.slice(), type: 'bar', name: 'Training', marker: { color: colors.tertiary } }
            ], {...layoutDefaults, barmode: 'stack', yaxis: {...layoutDefaults.yaxis, title: 'Seconds'}});

            // Update reanalysis chart (show card only if data exists)
            const hasReanalysis = data.reanalysisPositions.some(v => v > 0);
            if (hasReanalysis) {
                document.getElementById('reanalysis-card').style.display = '';
                Plotly.react('reanalysis-chart', [
                    { x: data.iterations.slice(), y: data.reanalysisPositions.slice(), type: 'scatter', mode: 'lines+markers',
                      name: 'Positions Updated', line: { color: colors.primary, width: 2 }, marker: { size: 4 } },
                    { x: data.iterations.slice(), y: data.reanalysisKL.slice(), type: 'scatter', mode: 'lines+markers',
                      name: 'Mean KL', line: { color: colors.tertiary, width: 2, dash: 'dot' },
                      marker: { size: 4 }, yaxis: 'y2' }
                ], {...layoutDefaults,
                    yaxis: {...layoutDefaults.yaxis, title: 'Positions'},
                    yaxis2: { title: 'KL Divergence', overlaying: 'y', side: 'right',
                              gridcolor: 'rgba(255,255,255,0.05)',
                              titlefont: { color: colors.tertiary },
                              tickfont: { color: colors.tertiary } }
                });
            }

            // Update header stats
            document.getElementById('iteration').textContent = iter;
            document.getElementById('total-games').textContent = totalGames.toLocaleString();
            document.getElementById('total-moves').textContent = totalMoves.toLocaleString();
            document.getElementById('buffer-size').textContent = metrics.buffer_size.toLocaleString();

            // Update current metrics
            document.getElementById('current-loss').textContent = metrics.total_loss ? metrics.total_loss.toFixed(4) : '-';

            // Calculate and update avg iteration time
            const avgIterTime = data.iterations.length > 0 ?
                (data.selfplayTime.reduce((a,b) => a+b, 0) + data.trainTime.reduce((a,b) => a+b, 0)) / data.iterations.length : 0;
            document.getElementById('avg-iter-time').textContent = formatTime(avgIterTime);
        }

        // Format seconds to HH:MM:SS
        function formatTime(seconds) {
            if (seconds <= 0 || !isFinite(seconds)) return '-';
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
        }

        // Update elapsed time
        function updateElapsed() {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            document.getElementById('elapsed').textContent = formatTime(elapsed);
        }

        // WebSocket connection
        function connectWebSocket() {
            socket = io();

            socket.on('connect', () => {
                console.log('Connected to dashboard server');
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').className = 'status connected';
                // Reset export button in case a prior request was lost
                document.getElementById('export-btn').disabled = false;
                document.getElementById('export-feedback').textContent = '';
            });

            socket.on('disconnect', () => {
                console.log('Disconnected from dashboard server');
                document.getElementById('status').textContent = 'Disconnected - Reconnecting...';
                document.getElementById('status').className = 'status disconnected';
            });

            socket.on('init', (initData) => {
                console.log('Received init data:', initData);
                totalIterations = initData.total_iterations || 100;
                startTime = Date.now() - (initData.elapsed_seconds || 0) * 1000;

                // Load historical data
                if (initData.history) {
                    initData.history.forEach(metrics => updateCharts(metrics));
                }
            });

            socket.on('metrics', (metrics) => {
                console.log('Received metrics:', metrics);
                updateCharts(metrics);
            });

            socket.on('complete', (summary) => {
                console.log('Training complete:', summary);
                document.getElementById('status').textContent = 'Training Complete!';
                document.getElementById('status').style.background = '#27ae60';
            });

            // Handle real-time progress updates (every 5 seconds during self-play)
            socket.on('progress', (data) => {
                console.log('Progress update:', data);

                // Update phase indicator
                const phaseIcon = document.getElementById('phase-icon');
                const phaseText = document.getElementById('phase-text');
                if (data.phase === 'selfplay') {
                    phaseIcon.textContent = '🎮';
                    phaseText.textContent = `Iteration ${data.iteration} - Self-Play (${data.games_completed}/${data.total_games} games)`;
                } else if (data.phase === 'reanalysis') {
                    phaseIcon.textContent = '🔄';
                    const c = data.reanalysis_completed || 0;
                    const t = data.reanalysis_total || 0;
                    phaseText.textContent = `Iteration ${data.iteration} - Reanalysis Tail (${c.toLocaleString()}/${t.toLocaleString()})`;
                } else if (data.phase === 'training') {
                    phaseIcon.textContent = '🧠';
                    const ep = data.train_epoch || 0;
                    const total = data.train_total_epochs || 0;
                    phaseText.textContent = ep > 0
                        ? `Iteration ${data.iteration} - Training (epoch ${ep}/${total})`
                        : `Iteration ${data.iteration} - Training`;
                }

                // Update ETA
                const phaseEta = document.getElementById('phase-eta');
                phaseEta.textContent = formatTime(data.phase_eta);

                // MCTS Performance metrics
                updateMetricWithHighlight('sys-sims-sec', Math.round(data.sims_per_sec).toLocaleString());
                updateMetricWithHighlight('sys-moves-sec', data.moves_per_sec.toFixed(1));
                updateMetricWithHighlight('pool-load', ((data.pool_load || 0) * 100).toFixed(1) + '%');

                // Pipeline health status
                const pipelineEl = document.getElementById('pipeline-status');
                const drops = data.submission_drops || 0;
                const exhaustions = data.pool_exhaustion || 0;
                const partials = data.partial_subs || 0;
                const failures = data.timeout_evals || 0;
                const totalIssues = drops + exhaustions + failures;
                if (totalIssues === 0) {
                    pipelineEl.textContent = '✅ Healthy';
                    pipelineEl.style.color = '#2ecc71';
                } else {
                    const parts = [];
                    if (failures > 0) parts.push(`${failures} timeouts`);
                    if (exhaustions > 0) parts.push(`${exhaustions.toLocaleString()} exhaustions`);
                    if (drops > 0) parts.push(`${drops.toLocaleString()} drops`);
                    pipelineEl.textContent = '⚠️ ' + parts.join(', ');
                    pipelineEl.style.color = '#e74c3c';
                }

                // GPU Performance metrics
                updateMetricWithHighlight('avg-batch-size', (data.avg_batch_size || 0).toFixed(1));
                updateMetricWithHighlight('sys-evals-sec', Math.round(data.evals_per_sec).toLocaleString());
                updateMetricWithHighlight('batch-fill-ratio', ((data.batch_fill_ratio || 0) * 100).toFixed(1) + '%');

                // Batch fire reason breakdown (full/stall/timeout)
                const bFull = data.batches_fired_full || 0;
                const bStall = data.batches_fired_stall || 0;
                const bTimeout = data.batches_fired_timeout || 0;
                const bTotal = bFull + bStall + bTimeout;
                if (bTotal > 0) {
                    const pFull = ((bFull / bTotal) * 100).toFixed(0);
                    const pStall = ((bStall / bTotal) * 100).toFixed(0);
                    const pTimeout = ((bTimeout / bTotal) * 100).toFixed(0);
                    updateMetricWithHighlight('batch-fire-reason', `F${pFull} S${pStall} T${pTimeout}%`);
                }

                updateMetricWithHighlight('gpu-wait-ms', (data.gpu_wait_ms || 0).toFixed(1) + 'ms');
                updateMetricWithHighlight('avg-infer-time', (data.avg_infer_time_ms || 0).toFixed(2) + 'ms');
                updateMetricWithHighlight('gpu-memory', (data.gpu_memory_used_mb || 0).toFixed(0) + ' MB');

                // Queue status metrics (MCTS card)
                updateMetricWithHighlight('queue-fill-pct', (data.queue_fill_pct || 0).toFixed(1) + '%');
                updateMetricWithHighlight('worker-wait-ms', (data.worker_wait_ms || 0).toFixed(1) + 'ms');

                // CUDA Graph status (aggregated like Pipeline)
                const graphStatusEl = document.getElementById('cuda-graph-status');
                if (data.cuda_graph_enabled) {
                    const fires = (data.cuda_graph_fires || 0).toLocaleString();
                    const rate = ((data.graph_fire_rate || 0) * 100).toFixed(1);
                    graphStatusEl.textContent = `✅ ${fires} fires (${rate}% hit)`;
                    graphStatusEl.style.color = '#2ecc71';
                } else {
                    graphStatusEl.textContent = '❌ Disabled';
                    graphStatusEl.style.color = '#95a5a6';
                }

                // Iteration progress
                const phaseElapsed = data.phase_elapsed || 0;
                updateMetricWithHighlight('live-run-time', formatTime(phaseElapsed));
                updateMetricWithHighlight('live-games', `${data.games_completed}/${data.total_games}`);
                updateMetricWithHighlight('live-wdl', `${data.white_wins || 0} / ${data.draws || 0} / ${data.black_wins || 0}`);
                updateMetricWithHighlight('live-moves', (data.moves || 0).toLocaleString());
                updateMetricWithHighlight('live-gph', (data.games_per_hour / 60).toFixed(2));

                // Tree depth display
                const maxDepth = data.max_search_depth || 0;
                const minDepth = data.min_search_depth || 0;
                const avgDepth = data.avg_search_depth || 0;
                if (maxDepth > 0) {
                    updateMetricWithHighlight('live-tree-depth',
                        `${minDepth}\u2013${maxDepth} (avg ${avgDepth.toFixed(1)})`);
                } else {
                    updateMetricWithHighlight('live-tree-depth', '--');
                }

                // Current game moves (min–max across active workers)
                const minMoves = data.min_current_moves || 0;
                const maxMoves = data.max_current_moves || 0;
                if (maxMoves > 0) {
                    updateMetricWithHighlight('live-current-moves', `${minMoves}\u2013${maxMoves}`);
                } else {
                    updateMetricWithHighlight('live-current-moves', '--');
                }

                // Risk beta (show sampled value, highlight non-zero)
                const rb = data.risk_beta || 0;
                const rbText = rb === 0 ? '0' : (rb > 0 ? '+' : '') + rb.toFixed(3);
                updateMetricWithHighlight('live-risk-beta', rbText);

                // Refutation Status (only show when asymmetric risk is active)
                const stdWins = data.standard_wins || 0;
                const oppWins = data.opponent_wins || 0;
                const asymDraws = data.asymmetric_draws || 0;
                const totalAsym = stdWins + oppWins + asymDraws;
                const refCard = document.getElementById('refutation-card');
                if (totalAsym > 0) {
                    refCard.style.display = '';
                    updateMetricWithHighlight('ref-std-wins', stdWins);
                    updateMetricWithHighlight('ref-opp-wins', oppWins);
                    updateMetricWithHighlight('ref-draws', asymDraws);
                    const eloEl = document.getElementById('ref-elo');
                    if (totalAsym >= 20) {
                        const score = Math.max(0.01, Math.min(0.99, (stdWins + 0.5 * asymDraws) / totalAsym));
                        const elo = -400 * Math.log10((1 - score) / score);
                        eloEl.textContent = (elo >= 0 ? '+' : '') + Math.round(elo);
                        eloEl.style.color = elo > 0 ? '#2ecc71' : elo < 0 ? '#e74c3c' : '#3498db';
                    } else {
                        eloEl.textContent = '\u23F3 n=' + totalAsym;
                        eloEl.style.color = '#95a5a6';
                    }
                } else {
                    refCard.style.display = 'none';
                }

                // Reanalysis Performance card (show when reanalysis is active)
                const reanalCompleted = data.reanalysis_completed || 0;
                const reanalSkipped = data.reanalysis_skipped || 0;
                const reanalTarget = data.reanalysis_total || 0;
                const reanalNNEvals = data.reanalysis_nn_evals || 0;
                const reanalKL = data.reanalysis_mean_kl || 0;
                const reanalElapsed = data.reanalysis_elapsed_s || 0;
                const reanalCard = document.getElementById('reanalysis-perf-card');
                if (reanalTarget > 0) {
                    reanalCard.style.display = '';
                    const speed = reanalElapsed > 0 ? (reanalCompleted / reanalElapsed).toFixed(1) : '0';
                    updateMetricWithHighlight('reanalysis-speed', speed + ' pos/s');
                    updateMetricWithHighlight('reanalysis-completed', reanalCompleted.toLocaleString() + ' / ' + reanalTarget.toLocaleString());
                    updateMetricWithHighlight('reanalysis-skipped', reanalSkipped.toLocaleString());
                    updateMetricWithHighlight('reanalysis-nn-evals', reanalNNEvals.toLocaleString());
                    updateMetricWithHighlight('reanalysis-kl', reanalKL > 0 ? reanalKL.toFixed(3) : '--');
                    // Progress bar based on processed / target
                    const processed = reanalCompleted + reanalSkipped;
                    const pct = reanalTarget > 0 ? Math.min(100, (processed / reanalTarget) * 100) : 0;
                    document.getElementById('reanalysis-progress-bar').style.width = pct.toFixed(1) + '%';
                    document.getElementById('reanalysis-progress-text').textContent =
                        pct.toFixed(0) + '% (' + processed.toLocaleString() + ' / ' + reanalTarget.toLocaleString() + ')';
                } else {
                    reanalCard.style.display = 'none';
                }

                // Training Progress card (show during training phase with epoch data)
                const trainCard = document.getElementById('training-progress-card');
                const trainEpoch = data.train_epoch || 0;
                const trainTotalEpochs = data.train_total_epochs || 0;
                if (data.phase === 'training' && trainEpoch > 0) {
                    trainCard.style.display = '';
                    updateMetricWithHighlight('train-epoch', `${trainEpoch} / ${trainTotalEpochs}`);
                    updateMetricWithHighlight('train-loss', (data.train_loss || 0).toFixed(4));
                    updateMetricWithHighlight('train-policy-loss', (data.train_policy_loss || 0).toFixed(4));
                    updateMetricWithHighlight('train-value-loss', (data.train_value_loss || 0).toFixed(4));
                    updateMetricWithHighlight('train-grad-norm', (data.train_grad_norm || 0).toFixed(2));
                    const trainPct = trainTotalEpochs > 0 ? (trainEpoch / trainTotalEpochs * 100) : 0;
                    document.getElementById('train-progress-bar').style.width = trainPct.toFixed(1) + '%';
                    document.getElementById('train-progress-text').textContent =
                        `${trainPct.toFixed(0)}% (epoch ${trainEpoch}/${trainTotalEpochs})`;
                } else {
                    trainCard.style.display = 'none';
                }

                // Update buffer size in header
                document.getElementById('buffer-size').textContent = data.buffer_size.toLocaleString();

                // Bottleneck indicator
                // High fill ratio = batches full = GPU saturated (optimal utilization)
                // Low fill ratio = batches sparse = eval_batch oversized for workers × search_batch
                const bottleneckEl = document.getElementById('bottleneck-type');
                const fillRatio = data.batch_fill_ratio || 0;
                bottleneckEl.classList.remove('underfilled', 'gpu-saturated', 'balanced');
                if (fillRatio > 0.8) {
                    bottleneckEl.textContent = 'GPU saturated (optimal utilization)';
                    bottleneckEl.classList.add('gpu-saturated');
                } else if (fillRatio < 0.3 && fillRatio > 0) {
                    bottleneckEl.textContent = 'Batch underfilled (reduce eval-batch or add workers)';
                    bottleneckEl.classList.add('underfilled');
                } else if (fillRatio > 0) {
                    bottleneckEl.textContent = 'Balanced';
                    bottleneckEl.classList.add('balanced');
                } else {
                    bottleneckEl.textContent = 'Initializing...';
                }

                // Update elapsed time from server
                startTime = Date.now() - (data.total_elapsed * 1000);

                // Routing thresholds (used by both histogram and thresholds display)
                const largeThreshold = data.large_graph_threshold || 0;
                const mediumGraphSize = data.medium_graph_size || 0;
                const smallGraphSize = data.small_graph_size || 0;
                const miniGraphSize = data.mini_graph_size || 0;

                // Update batch size histogram
                if (data.batch_histogram && data.batch_histogram.length > 0) {
                    const bins = data.batch_histogram.map(d => d[0]);
                    const counts = data.batch_histogram.map(d => d[1]);

                    // Color bars based on actual routing thresholds (not just size zones)
                    const miniThreshold = data.mini_threshold || 0;
                    const smallThreshold = data.small_threshold || 0;
                    const mediumThreshold = data.medium_threshold || 0;

                    const barColors = bins.map(bin => {
                        if (bin > largeThreshold) return '#2ecc71';                                           // Large graph - green
                        if (bin <= miniGraphSize && bin >= miniThreshold) return '#9b59b6';                    // Mini graph - purple
                        if (bin <= smallGraphSize && bin >= smallThreshold) return '#3498db';                  // Small graph - blue
                        if (bin <= mediumGraphSize && bin >= mediumThreshold) return '#1abc9c';                // Medium graph - teal
                        return '#e67e22';                                                                      // Eager - orange
                    });

                    Plotly.react('batch-histogram', [{
                        x: bins,
                        y: counts,
                        type: 'bar',
                        marker: {
                            color: barColors,
                            line: { color: 'rgba(255,255,255,0.3)', width: 1 }
                        },
                        hovertemplate: 'Batch %{x}: %{y} calls<extra></extra>'
                    }], {
                        ...layoutDefaults,
                        margin: { t: 5, b: 30, l: 45, r: 10 },
                        xaxis: {
                            ...layoutDefaults.xaxis,
                            title: { text: 'Batch Size', font: { size: 9 } },
                            tickmode: 'array',
                            tickvals: bins,
                            ticktext: bins.map(String),
                            range: [0, Math.max(...bins) + 1]
                        },
                        yaxis: { ...layoutDefaults.yaxis, title: { text: 'Count', font: { size: 9 } } },
                        showlegend: false,
                        bargap: 0.1
                    });
                }

                // Update batch percentiles display
                if (data.batch_p50 > 0) {
                    document.getElementById('batch-percentiles').textContent =
                        `${data.batch_p25} / ${data.batch_p50} / ${data.batch_p75} / ${data.batch_p90}`;
                }

                // Update batch thresholds display (show all 4 tiers with crossover thresholds)
                if (smallGraphSize > 0 || largeThreshold > 0) {
                    const mediumThreshold = data.medium_threshold || 0;
                    const smallThreshold = data.small_threshold || 0;
                    const miniThreshold = data.mini_threshold || 0;
                    const largeText = largeThreshold >= (data.batch_max || 9999) ? 'disabled' : `>${largeThreshold}`;
                    let thresholdText = `mini\u2264${miniGraphSize}(\u2265${miniThreshold}), small\u2264${smallGraphSize}(\u2265${smallThreshold})`;
                    if (mediumGraphSize > smallGraphSize) {
                        thresholdText += `, medium\u2264${mediumGraphSize}(\u2265${mediumThreshold})`;
                    }
                    thresholdText += `, large ${largeText}`;
                    document.getElementById('batch-thresholds').textContent = thresholdText;
                }

                // Update execution path pie charts (count + time)
                const largeGraphFires = data.large_graph_fires || 0;
                const mediumGraphFires = data.medium_graph_fires || 0;
                const smallGraphFires = data.small_graph_fires || 0;
                const miniGraphFires = data.mini_graph_fires || 0;
                const eagerFires = data.eager_fires || 0;
                const totalBatches = largeGraphFires + mediumGraphFires + smallGraphFires + miniGraphFires + eagerFires;

                if (totalBatches > 0) {
                    // Count distribution pie
                    Plotly.react('exec-path-pie', [{
                        values: [largeGraphFires, mediumGraphFires, smallGraphFires, miniGraphFires, eagerFires],
                        labels: ['Large', 'Medium', 'Small', 'Mini', 'Eager'],
                        type: 'pie',
                        marker: { colors: ['#2ecc71', '#1abc9c', '#3498db', '#9b59b6', '#e67e22'] },
                        textinfo: 'percent',
                        textfont: { color: '#fff', size: 10 },
                        hovertemplate: '%{label}: %{value} batches (%{percent})<extra></extra>'
                    }], {
                        ...layoutDefaults,
                        margin: { t: 5, b: 5, l: 5, r: 5 },
                        showlegend: false
                    });

                    // Time distribution pie
                    const largeTime = data.large_graph_time_ms || 0;
                    const mediumTime = data.medium_graph_time_ms || 0;
                    const smallTime = data.small_graph_time_ms || 0;
                    const miniTime = data.mini_graph_time_ms || 0;
                    const eagerTime = data.eager_time_ms || 0;

                    Plotly.react('exec-time-pie', [{
                        values: [largeTime, mediumTime, smallTime, miniTime, eagerTime],
                        labels: ['Large', 'Medium', 'Small', 'Mini', 'Eager'],
                        type: 'pie',
                        marker: { colors: ['#2ecc71', '#1abc9c', '#3498db', '#9b59b6', '#e67e22'] },
                        textinfo: 'percent',
                        textfont: { color: '#fff', size: 10 },
                        hovertemplate: '%{label}: %{value:.0f}ms (%{percent})<extra></extra>'
                    }], {
                        ...layoutDefaults,
                        margin: { t: 5, b: 5, l: 5, r: 5 },
                        showlegend: false
                    });

                    document.getElementById('total-batches').textContent = totalBatches.toLocaleString();
                }
            });

            // --- Parameter Controls SocketIO handlers ---
            socket.on('params_current', (params) => {
                populateParams(params);
                if (pendingApplied) {
                    const statusEl = document.getElementById('param-status');
                    statusEl.textContent = 'Applied';
                    statusEl.className = 'param-status applied';
                    setTimeout(() => { statusEl.className = 'param-status'; }, 5000);
                    pendingApplied = false;
                }
            });

            socket.on('params_pending_ack', (resp) => {
                const statusEl = document.getElementById('param-status');
                const count = Object.keys(resp.accepted || {}).length;
                statusEl.textContent = `Pending (${count})`;
                statusEl.className = 'param-status pending';
                document.getElementById('param-apply-btn').disabled = false;
                pendingApplied = true;

                const feedbackEl = document.getElementById('param-feedback');
                if (resp.errors && resp.errors.length > 0) {
                    feedbackEl.textContent = 'Errors: ' + resp.errors.join(', ');
                    feedbackEl.style.color = '#e74c3c';
                } else {
                    feedbackEl.textContent = `${count} change(s) queued`;
                    feedbackEl.style.color = '#f1c40f';
                }
            });

            socket.on('params_applied', (resp) => {
                const statusEl = document.getElementById('param-status');
                const phase = resp.phase ? ` (${resp.phase})` : '';
                statusEl.textContent = `Applied at iter ${resp.iteration}${phase}`;
                statusEl.className = 'param-status applied';
                setTimeout(() => { statusEl.className = 'param-status'; }, 5000);

                const feedbackEl = document.getElementById('param-feedback');
                feedbackEl.textContent = `Applied${phase}: ${(resp.applied || []).join(', ')}`;
                feedbackEl.style.color = '#2ecc71';
            });

            socket.on('export_ack', (resp) => {
                const feedbackEl = document.getElementById('export-feedback');
                feedbackEl.textContent = 'Queued, waiting for training loop...';
                feedbackEl.style.color = '#f1c40f';
            });

            socket.on('export_complete', (resp) => {
                const btn = document.getElementById('export-btn');
                const feedbackEl = document.getElementById('export-feedback');
                btn.disabled = false;
                if (resp.status === 'saved') {
                    feedbackEl.textContent = 'Saved: ' + resp.filename;
                    feedbackEl.style.color = '#2ecc71';
                } else {
                    feedbackEl.textContent = 'Error: ' + (resp.error || 'unknown');
                    feedbackEl.style.color = '#e74c3c';
                }
            });
        }

        // --- Parameter Controls Functions ---
        function toggleParamControls() {
            const body = document.getElementById('param-controls-body');
            const arrow = document.getElementById('param-toggle-arrow');
            if (body.style.display === 'none') {
                body.style.display = 'block';
                arrow.classList.add('open');
            } else {
                body.style.display = 'none';
                arrow.classList.remove('open');
            }
        }

        function populateParams(params) {
            currentParams = Object.assign({}, params);
            document.querySelectorAll('[data-param]').forEach(input => {
                const key = input.dataset.param;
                if (key in params) {
                    input.value = params[key];
                    input.classList.remove('modified');
                    const curEl = document.getElementById('cur-' + key);
                    if (curEl) curEl.textContent = '(active: ' + params[key] + ')';
                }
            });
        }

        function applyParams() {
            const changes = {};
            document.querySelectorAll('[data-param]').forEach(input => {
                const key = input.dataset.param;
                const val = parseFloat(input.value);
                if (key in currentParams && !isNaN(val) && val !== currentParams[key]) {
                    changes[key] = val;
                }
            });
            if (Object.keys(changes).length === 0) {
                const feedbackEl = document.getElementById('param-feedback');
                feedbackEl.textContent = 'No changes to apply';
                feedbackEl.style.color = '#95a5a6';
                return;
            }
            document.getElementById('param-apply-btn').disabled = true;
            socket.emit('apply_params', changes);
        }

        function resetParams() {
            populateParams(currentParams);
            document.getElementById('param-feedback').textContent = '';
        }

        function exportModel() {
            const btn = document.getElementById('export-btn');
            btn.disabled = true;
            document.getElementById('export-feedback').textContent = 'Requesting...';
            socket.emit('export_model');
        }

        // Input change listeners for modified highlighting
        document.querySelectorAll('[data-param]').forEach(input => {
            input.addEventListener('input', () => {
                const key = input.dataset.param;
                const val = parseFloat(input.value);
                if (key in currentParams && !isNaN(val) && val !== currentParams[key]) {
                    input.classList.add('modified');
                } else {
                    input.classList.remove('modified');
                }
            });
        });

        // Helper function to update metric with highlight animation
        function updateMetricWithHighlight(elementId, value) {
            const element = document.getElementById(elementId);
            if (!element) return;
            const oldValue = element.textContent;
            const newValue = String(value);
            element.textContent = newValue;

            // Add highlight animation if value changed
            if (oldValue !== newValue) {
                element.classList.add('highlight');
                setTimeout(() => element.classList.remove('highlight'), 1000);
            }
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initCharts();
            connectWebSocket();
            setInterval(updateElapsed, 1000);
        });
    </script>
</body>
</html>
"""


class LiveDashboardServer:
    """Live dashboard server with WebSocket support."""

    def __init__(self, host: str = '127.0.0.1', port: int = 5000):
        self.host = host
        self.port = port
        self.app = None
        self.socketio = None
        self.thread = None
        self.running = False
        self.history: List[Dict] = []
        self.start_time = None
        self.total_iterations = 100

        # Parameter controls: browser → training loop
        self._pending_updates: Dict[str, any] = {}
        self._pending_meta: Dict[str, Dict] = {}
        self._pending_lock = threading.Lock()
        self._current_params: Dict[str, any] = {}

        # Export model: browser → training loop
        self._export_requested: bool = False

    def set_current_params(self, params: dict):
        """Push authoritative parameter values. Broadcasts to all connected clients."""
        self._current_params = params.copy()
        if self.socketio is not None and self.running:
            self.socketio.emit('params_current', self._current_params)

    def poll_updates(self):
        """Atomically drain pending parameter updates + source metadata."""
        with self._pending_lock:
            updates = self._pending_updates.copy()
            self._pending_updates.clear()
            meta = self._pending_meta.copy()
            self._pending_meta.clear()
        return updates, meta

    def poll_export_request(self) -> bool:
        """Atomically read and clear export request flag."""
        with self._pending_lock:
            requested = self._export_requested
            self._export_requested = False
        return requested

    def _create_app(self):
        """Create Flask app with SocketIO."""
        if not FLASK_AVAILABLE:
            raise ImportError("Flask and flask-socketio are required for live dashboard")

        app = Flask(__name__)
        app.config['SECRET_KEY'] = 'alphazero-dashboard-secret'
        socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

        @app.route('/')
        def index():
            return render_template_string(DASHBOARD_HTML)

        @app.route('/api/metrics')
        def get_metrics():
            return jsonify({
                'history': self.history,
                'total_iterations': self.total_iterations,
                'elapsed_seconds': time.time() - self.start_time if self.start_time else 0
            })

        @socketio.on('connect')
        def handle_connect():
            # Send initialization data to new clients
            emit('init', {
                'history': self.history,
                'total_iterations': self.total_iterations,
                'elapsed_seconds': time.time() - self.start_time if self.start_time else 0
            })
            # Send current parameter values so controls populate on connect
            if self._current_params:
                emit('params_current', self._current_params)

        # Parameter control spec: (type, min, max)
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
        }

        @socketio.on('apply_params')
        def handle_apply_params(data):
            validated = {}
            errors = []
            for key, value in data.items():
                if key not in PARAM_SPEC:
                    errors.append(f"Unknown: {key}")
                    continue
                ptype, pmin, pmax = PARAM_SPEC[key]
                try:
                    typed = ptype(value)
                    validated[key] = max(pmin, min(pmax, typed))
                except (ValueError, TypeError) as e:
                    errors.append(f"{key}: {e}")
            with self._pending_lock:
                self._pending_updates.update(validated)
                for key in validated:
                    self._pending_meta[key] = {'source': 'dashboard', 'reason': ''}
            socketio.emit('params_pending_ack', {
                'accepted': validated, 'errors': errors
            })

        @socketio.on('export_model')
        def handle_export_model():
            with self._pending_lock:
                self._export_requested = True
            emit('export_ack', {'status': 'queued'})

        self.app = app
        self.socketio = socketio

    def start(self, total_iterations: int = 100, open_browser: bool = True):
        """Start the dashboard server in a background thread."""
        if not FLASK_AVAILABLE:
            print("  WARNING: Flask not installed. Install with: pip install flask flask-socketio")
            print("  Live dashboard disabled.")
            return False

        self._create_app()
        self.total_iterations = total_iterations
        self.start_time = time.time()
        self.running = True

        def run_server():
            # Suppress Flask startup messages
            import logging
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)

            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
                log_output=False
            )

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()

        # Give server time to start
        time.sleep(1)

        url = f"http://{self.host}:{self.port}"
        print(f"  Live dashboard started: {url}")

        if open_browser:
            try:
                webbrowser.open(url)
                print(f"  Browser opened automatically")
            except Exception:
                print(f"  Open in browser: {url}")

        return True

    def push_metrics(self, metrics: 'IterationMetrics'):
        """Push new metrics to connected clients."""
        if not self.running or self.socketio is None:
            return

        # Convert metrics to dashboard format
        elapsed = time.time() - self.start_time if self.start_time else 0

        # Calculate rates
        moves_per_sec = metrics.total_moves / metrics.selfplay_time if metrics.selfplay_time > 0 else 0
        sims_per_sec = metrics.total_simulations / metrics.selfplay_time if metrics.selfplay_time > 0 else 0
        nn_evals_per_sec = metrics.total_nn_evals / metrics.selfplay_time if metrics.selfplay_time > 0 else 0
        games_per_hour = metrics.num_games / metrics.selfplay_time * 3600 if metrics.selfplay_time > 0 else 0

        dashboard_metrics = {
            'iteration': metrics.iteration,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'elapsed_minutes': elapsed / 60,
            'total_loss': metrics.loss if metrics.loss > 0 else None,
            'policy_loss': metrics.policy_loss if metrics.policy_loss > 0 else None,
            'value_loss': metrics.value_loss if metrics.value_loss > 0 else None,
            'moves_per_sec': moves_per_sec,
            'sims_per_sec': sims_per_sec,
            'nn_evals_per_sec': nn_evals_per_sec,
            'games_per_hour': games_per_hour,
            'white_wins': metrics.white_wins,
            'black_wins': metrics.black_wins,
            'draws': metrics.draws,
            'standard_wins': getattr(metrics, 'standard_wins', 0),
            'opponent_wins': getattr(metrics, 'opponent_wins', 0),
            'asymmetric_draws': getattr(metrics, 'asymmetric_draws', 0),
            'avg_game_length': metrics.avg_game_length,
            'num_games': metrics.num_games,
            'buffer_size': metrics.buffer_size,
            'selfplay_time': metrics.selfplay_time,
            'train_time': metrics.train_time,
            'iteration_time': metrics.total_time,
            'grad_norm_avg': getattr(metrics, 'grad_norm_avg', 0.0),
            'grad_norm_max': getattr(metrics, 'grad_norm_max', 0.0),
            'wdl_alpha': getattr(metrics, 'wdl_alpha', 0.0),
            'risk_beta': getattr(metrics, 'risk_beta', 0.0),
            'num_train_batches': getattr(metrics, 'num_train_batches', 0),
            'draws_repetition': getattr(metrics, 'draws_repetition', 0),
            'draws_early_repetition': getattr(metrics, 'draws_early_repetition', 0),
            'draws_stalemate': getattr(metrics, 'draws_stalemate', 0),
            'draws_fifty_move': getattr(metrics, 'draws_fifty_move', 0),
            'draws_insufficient': getattr(metrics, 'draws_insufficient', 0),
            'draws_max_moves': getattr(metrics, 'draws_max_moves', 0),
            'reanalysis_positions': getattr(metrics, 'reanalysis_positions', 0),
            'reanalysis_time_s': getattr(metrics, 'reanalysis_time_s', 0.0),
            'reanalysis_mean_kl': getattr(metrics, 'reanalysis_mean_kl', 0.0),
        }

        self.history.append(dashboard_metrics)
        self.socketio.emit('metrics', dashboard_metrics)

    def push_progress(self, iteration: int, games_completed: int, total_games: int,
                       moves: int, sims: int, evals: int, elapsed_time: float,
                       buffer_size: int, phase: str = "selfplay",
                       risk_beta: float = 0.0,
                       # Game results (live W/D/L)
                       white_wins: int = 0,
                       black_wins: int = 0,
                       draws: int = 0,
                       # Per-persona outcome tracking (asymmetric risk)
                       standard_wins: int = 0,
                       opponent_wins: int = 0,
                       asymmetric_draws: int = 0,
                       # System monitoring metrics
                       timeout_evals: int = 0,
                       pool_exhaustion: int = 0,
                       submission_drops: int = 0,
                       partial_subs: int = 0,
                       pool_resets: int = 0,
                       submission_waits: int = 0,
                       pool_load: float = 0.0,
                       avg_batch_size: float = 0.0,
                       batch_fill_ratio: float = 0.0,
                       # Batch fire reason breakdown
                       batches_fired_full: int = 0,
                       batches_fired_stall: int = 0,
                       batches_fired_timeout: int = 0,
                       # GPU metrics
                       cuda_graph_fires: int = 0,  # Deprecated: sum of all graph fires
                       large_graph_fires: int = 0,
                       medium_graph_fires: int = 0,
                       small_graph_fires: int = 0,
                       mini_graph_fires: int = 0,
                       eager_fires: int = 0,
                       graph_fire_rate: float = 0.0,
                       avg_infer_time_ms: float = 0.0,
                       gpu_memory_used_mb: float = 0.0,
                       cuda_graph_enabled: bool = False,
                       # Tree depth metrics
                       max_search_depth: int = 0,
                       min_search_depth: int = 0,
                       avg_search_depth: float = 0.0,
                       # Active game move counts
                       min_current_moves: int = 0,
                       max_current_moves: int = 0,
                       # Queue status metrics
                       queue_fill_pct: float = 0.0,
                       gpu_wait_ms: float = 0.0,
                       worker_wait_ms: float = 0.0,
                       buffer_swaps: int = 0,
                       # Batch size distribution percentiles
                       batch_p25: int = 0,
                       batch_p50: int = 0,
                       batch_p75: int = 0,
                       batch_p90: int = 0,
                       batch_min: int = 0,
                       batch_max: int = 0,
                       # Batch histogram data (list of [bin_center, count])
                       batch_histogram: list = None,
                       large_graph_threshold: int = 0,
                       medium_graph_size: int = 0,
                       small_graph_size: int = 0,
                       mini_graph_size: int = 0,
                       # Crossover thresholds for each tier
                       medium_threshold: int = 0,
                       small_threshold: int = 0,
                       mini_threshold: int = 0,
                       # Per-path inference time (ms)
                       large_graph_time_ms: float = 0.0,
                       medium_graph_time_ms: float = 0.0,
                       small_graph_time_ms: float = 0.0,
                       mini_graph_time_ms: float = 0.0,
                       eager_time_ms: float = 0.0,
                       # Padding waste
                       cuda_pad_waste_pct: float = 0.0,
                       # Reanalysis live stats
                       reanalysis_completed: int = 0,
                       reanalysis_skipped: int = 0,
                       reanalysis_total: int = 0,
                       reanalysis_nn_evals: int = 0,
                       reanalysis_mean_kl: float = 0.0,
                       reanalysis_elapsed_s: float = 0.0,
                       # Training progress (per-epoch)
                       train_epoch: int = 0,
                       train_total_epochs: int = 0,
                       train_loss: float = 0.0,
                       train_policy_loss: float = 0.0,
                       train_value_loss: float = 0.0,
                       train_grad_norm: float = 0.0):
        """Push real-time progress updates during self-play, reanalysis, or training.

        Args:
            iteration: Current iteration number
            games_completed: Games completed so far this iteration
            total_games: Total games for this iteration
            moves: Total moves so far
            sims: Total MCTS simulations so far
            evals: Total NN evaluations so far
            elapsed_time: Seconds elapsed in this phase
            buffer_size: Current replay buffer size
            phase: Current phase ("selfplay", "reanalysis", or "training")
            timeout_evals: Number of MCTS evaluation timeouts (mcts_failures)
            pool_exhaustion: Times observation pool ran out of slots
            submission_drops: Total leaves dropped due to pool exhaustion
            partial_subs: Times queued fewer leaves than requested
            pool_resets: Times pool was reset
            pool_load: Drop rate (0.0 = healthy, >0 = drops occurring)
            avg_batch_size: Average GPU batch size
            batch_fill_ratio: GPU batch fill ratio (>0.8 = GPU saturated, <0.3 = batch underfilled, 0.3-0.8 = balanced)
        """
        if not self.running or self.socketio is None:
            return

        # Calculate rates
        moves_per_sec = moves / elapsed_time if elapsed_time > 0 else 0
        sims_per_sec = sims / elapsed_time if elapsed_time > 0 else 0
        evals_per_sec = evals / elapsed_time if elapsed_time > 0 else 0
        games_per_hour = games_completed / elapsed_time * 3600 if elapsed_time > 0 else 0

        # Calculate ETA for this iteration
        if games_completed > 0:
            eta_seconds = (total_games - games_completed) / (games_completed / elapsed_time)
        else:
            eta_seconds = 0

        total_elapsed = time.time() - self.start_time if self.start_time else 0

        progress_data = {
            'type': 'progress',
            'iteration': iteration,
            'phase': phase,
            'games_completed': games_completed,
            'total_games': total_games,
            'progress_percent': (games_completed / total_games * 100) if total_games > 0 else 0,
            'moves': moves,
            'moves_per_sec': moves_per_sec,
            'sims': sims,
            'sims_per_sec': sims_per_sec,
            'evals': evals,
            'evals_per_sec': evals_per_sec,
            'games_per_hour': games_per_hour,
            'risk_beta': risk_beta,
            'white_wins': white_wins,
            'black_wins': black_wins,
            'draws': draws,
            'standard_wins': standard_wins,
            'opponent_wins': opponent_wins,
            'asymmetric_draws': asymmetric_draws,
            'buffer_size': buffer_size,
            'phase_elapsed': elapsed_time,
            'phase_eta': eta_seconds,
            'total_elapsed': total_elapsed,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            # System monitoring metrics
            'timeout_evals': timeout_evals,
            'pool_exhaustion': pool_exhaustion,
            'submission_drops': submission_drops,
            'partial_subs': partial_subs,
            'pool_resets': pool_resets,
            'submission_waits': submission_waits,
            'pool_load': pool_load,
            'avg_batch_size': avg_batch_size,
            'batch_fill_ratio': batch_fill_ratio,
            # Batch fire reason breakdown
            'batches_fired_full': batches_fired_full,
            'batches_fired_stall': batches_fired_stall,
            'batches_fired_timeout': batches_fired_timeout,
            # GPU metrics
            'cuda_graph_fires': cuda_graph_fires,
            'large_graph_fires': large_graph_fires,
            'medium_graph_fires': medium_graph_fires,
            'small_graph_fires': small_graph_fires,
            'mini_graph_fires': mini_graph_fires,
            'eager_fires': eager_fires,
            'graph_fire_rate': graph_fire_rate,
            'avg_infer_time_ms': avg_infer_time_ms,
            'gpu_memory_used_mb': gpu_memory_used_mb,
            'cuda_graph_enabled': cuda_graph_enabled,
            # Tree depth
            'max_search_depth': max_search_depth,
            'min_search_depth': min_search_depth,
            'avg_search_depth': avg_search_depth,
            # Active game move counts
            'min_current_moves': min_current_moves,
            'max_current_moves': max_current_moves,
            # Queue status metrics
            'queue_fill_pct': queue_fill_pct,
            'gpu_wait_ms': gpu_wait_ms,
            'worker_wait_ms': worker_wait_ms,
            'buffer_swaps': buffer_swaps,
            # Batch size distribution percentiles
            'batch_p25': batch_p25,
            'batch_p50': batch_p50,
            'batch_p75': batch_p75,
            'batch_p90': batch_p90,
            'batch_min': batch_min,
            'batch_max': batch_max,
            # Batch histogram data and routing thresholds
            'batch_histogram': batch_histogram or [],
            'large_graph_threshold': large_graph_threshold,
            'medium_graph_size': medium_graph_size,
            'small_graph_size': small_graph_size,
            'mini_graph_size': mini_graph_size,
            # Crossover thresholds for each tier
            'medium_threshold': medium_threshold,
            'small_threshold': small_threshold,
            'mini_threshold': mini_threshold,
            # Per-path inference time
            'large_graph_time_ms': large_graph_time_ms,
            'medium_graph_time_ms': medium_graph_time_ms,
            'small_graph_time_ms': small_graph_time_ms,
            'mini_graph_time_ms': mini_graph_time_ms,
            'eager_time_ms': eager_time_ms,
            # Padding waste
            'cuda_pad_waste_pct': cuda_pad_waste_pct,
            # Reanalysis live stats
            'reanalysis_completed': reanalysis_completed,
            'reanalysis_skipped': reanalysis_skipped,
            'reanalysis_total': reanalysis_total,
            'reanalysis_nn_evals': reanalysis_nn_evals,
            'reanalysis_mean_kl': reanalysis_mean_kl,
            'reanalysis_elapsed_s': reanalysis_elapsed_s,
            # Training progress
            'train_epoch': train_epoch,
            'train_total_epochs': train_total_epochs,
            'train_loss': train_loss,
            'train_policy_loss': train_policy_loss,
            'train_value_loss': train_value_loss,
            'train_grad_norm': train_grad_norm,
        }

        self.socketio.emit('progress', progress_data)

    def complete(self):
        """Signal training completion."""
        if not self.running or self.socketio is None:
            return

        summary = {
            'total_iterations': len(self.history),
            'elapsed_seconds': time.time() - self.start_time if self.start_time else 0,
        }
        self.socketio.emit('complete', summary)

    def stop(self):
        """Stop the dashboard server."""
        self.running = False
        # Note: Flask-SocketIO doesn't have a clean shutdown method
        # The daemon thread will exit when the main program exits


# For standalone testing
if __name__ == '__main__':
    import random

    print("=" * 60)
    print("Live Dashboard Server - Standalone Test Mode")
    print("=" * 60)

    server = LiveDashboardServer(port=5000)
    if server.start(total_iterations=20, open_browser=True):
        print("\nSimulating training iterations...")

        # Simulate training iterations
        for i in range(1, 21):
            # Create fake metrics
            class FakeMetrics:
                iteration = i
                num_games = 25
                total_moves = random.randint(1000, 1500)
                selfplay_time = random.uniform(30, 60)
                white_wins = random.randint(8, 12)
                black_wins = random.randint(8, 12)
                draws = 25 - white_wins - black_wins
                avg_game_length = total_moves / num_games
                total_simulations = total_moves * 800
                total_nn_evals = total_simulations // 10
                train_time = random.uniform(5, 15)
                loss = max(0.5, 3.0 - i * 0.1 + random.uniform(-0.1, 0.1))
                policy_loss = loss * 0.7
                value_loss = loss * 0.3
                buffer_size = i * 1500
                total_time = selfplay_time + train_time

            server.push_metrics(FakeMetrics())
            print(f"  Iteration {i}/20 - Loss: {FakeMetrics.loss:.4f}")
            time.sleep(2)  # Simulate iteration time

        server.complete()
        print("\nTraining simulation complete!")
        print("Press Ctrl+C to exit...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
    else:
        print("Failed to start dashboard server")
