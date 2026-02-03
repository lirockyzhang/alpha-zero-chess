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
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .metric {
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #3498db;
        }
        .metric-label {
            font-size: 0.8em;
            opacity: 0.7;
            margin-top: 5px;
        }
        .footer {
            text-align: center;
            padding: 20px;
            opacity: 0.5;
            font-size: 0.9em;
        }
        .live-progress {
            background: rgba(0,0,0,0.3);
            padding: 15px 20px;
            margin: 0;
        }
        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .progress-phase {
            font-size: 1.1em;
            font-weight: bold;
            color: #a8d8ea;
        }
        .progress-phase .phase-icon {
            margin-right: 8px;
        }
        .progress-eta {
            font-size: 0.9em;
            color: #95a5a6;
        }
        .progress-bar-container {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            height: 24px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            border-radius: 10px;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8em;
            font-weight: bold;
            color: white;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }
        .live-stats {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 15px;
        }
        .live-stat {
            text-align: center;
            min-width: 100px;
        }
        .live-stat-value {
            font-size: 1.3em;
            font-weight: bold;
            color: #3498db;
        }
        .live-stat-label {
            font-size: 0.75em;
            opacity: 0.7;
        }
        .live-stat-value.highlight {
            color: #2ecc71;
            animation: pulse 1s ease-in-out;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        @media (max-width: 600px) {
            .grid {
                grid-template-columns: 1fr;
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
    </style>
</head>
<body>
    <!-- Live Progress Bar -->
    <div class="live-progress" id="live-progress">
        <div class="progress-header">
            <div class="progress-phase">
                <span class="phase-icon" id="phase-icon">🎮</span>
                <span id="phase-text">Waiting to start...</span>
            </div>
            <div class="progress-eta">
                ETA: <span id="phase-eta">--</span>
            </div>
        </div>
        <div class="progress-bar-container">
            <div class="progress-bar" id="progress-bar" style="width: 0%">0%</div>
        </div>
        <div class="live-stats">
            <div class="live-stat">
                <div class="live-stat-value" id="live-games">0/0</div>
                <div class="live-stat-label">Games</div>
            </div>
            <div class="live-stat">
                <div class="live-stat-value" id="live-moves">0</div>
                <div class="live-stat-label">Moves/sec</div>
            </div>
            <div class="live-stat">
                <div class="live-stat-value" id="live-sims">0</div>
                <div class="live-stat-label">Sims/sec</div>
            </div>
            <div class="live-stat">
                <div class="live-stat-value" id="live-evals">0</div>
                <div class="live-stat-label">NN/sec</div>
            </div>
            <div class="live-stat">
                <div class="live-stat-value" id="live-gph">0</div>
                <div class="live-stat-label">Games/hr</div>
            </div>
        </div>
    </div>

    <div class="header">
        <h1>🧠 AlphaZero Live Training Dashboard</h1>
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
        </div>
        <div class="status disconnected" id="status">Connecting...</div>
    </div>

    <div class="grid">
        <!-- Loss Chart -->
        <div class="card">
            <h3>📉 Training Loss</h3>
            <div id="loss-chart" class="chart"></div>
        </div>

        <!-- Performance Chart -->
        <div class="card">
            <h3>⚡ Performance (Moves/sec)</h3>
            <div id="perf-chart" class="chart"></div>
        </div>

        <!-- Policy vs Value Loss -->
        <div class="card">
            <h3>🎯 Policy vs Value Loss</h3>
            <div id="pv-loss-chart" class="chart"></div>
        </div>

        <!-- MCTS Simulations -->
        <div class="card">
            <h3>🔬 MCTS Simulations/sec</h3>
            <div id="sims-chart" class="chart"></div>
        </div>

        <!-- Win/Draw/Loss -->
        <div class="card">
            <h3>📊 Game Results (Last 10 Iterations)</h3>
            <div id="wdl-chart" class="chart"></div>
        </div>

        <!-- Time Breakdown -->
        <div class="card">
            <h3>⏱️ Time Breakdown (Last 10 Iterations)</h3>
            <div id="time-chart" class="chart"></div>
        </div>

        <!-- Current Metrics -->
        <div class="card">
            <h3>📈 Current Iteration Metrics</h3>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value" id="current-loss">-</div>
                    <div class="metric-label">Loss</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="current-moves">-</div>
                    <div class="metric-label">Moves/sec</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="current-sims">-</div>
                    <div class="metric-label">Sims/sec</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="current-games">-</div>
                    <div class="metric-label">Games/hour</div>
                </div>
            </div>
        </div>

        <!-- ETA -->
        <div class="card">
            <h3>🕐 Estimated Time</h3>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value" id="eta">-</div>
                    <div class="metric-label">Time Remaining</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="avg-iter-time">-</div>
                    <div class="metric-label">Avg Iteration Time</div>
                </div>
            </div>
        </div>
    </div>

    <div class="footer">
        AlphaZero Training Dashboard | Real-time WebSocket Updates
    </div>

    <script>
        // Data storage
        const data = {
            iterations: [],
            totalLoss: [],
            policyLoss: [],
            valueLoss: [],
            movesPerSec: [],
            simsPerSec: [],
            whiteWins: [],
            blackWins: [],
            draws: [],
            selfplayTime: [],
            trainTime: [],
        };

        let totalGames = 0;
        let totalMoves = 0;
        let startTime = Date.now();
        let totalIterations = 100;

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
            // Loss chart
            Plotly.newPlot('loss-chart', [{
                x: [], y: [], type: 'scatter', mode: 'lines+markers',
                name: 'Total Loss', line: { color: colors.primary, width: 2 },
                marker: { size: 4 }
            }], {...layoutDefaults, yaxis: {...layoutDefaults.yaxis, title: 'Loss'}});

            // Performance chart
            Plotly.newPlot('perf-chart', [{
                x: [], y: [], type: 'scatter', mode: 'lines+markers',
                name: 'Moves/sec', line: { color: colors.secondary, width: 2 },
                fill: 'tozeroy', fillcolor: 'rgba(46, 204, 113, 0.2)',
                marker: { size: 4 }
            }], {...layoutDefaults, yaxis: {...layoutDefaults.yaxis, title: 'Moves/sec'}});

            // Policy vs Value Loss
            Plotly.newPlot('pv-loss-chart', [
                { x: [], y: [], type: 'scatter', mode: 'lines+markers',
                  name: 'Policy Loss', line: { color: colors.primary, width: 2 }, marker: { size: 4 } },
                { x: [], y: [], type: 'scatter', mode: 'lines+markers',
                  name: 'Value Loss', line: { color: colors.tertiary, width: 2 }, marker: { size: 4 } }
            ], {...layoutDefaults, yaxis: {...layoutDefaults.yaxis, title: 'Loss'}});

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

            // Time breakdown chart
            Plotly.newPlot('time-chart', [
                { x: [], y: [], type: 'bar', name: 'Self-Play', marker: { color: colors.primary } },
                { x: [], y: [], type: 'bar', name: 'Training', marker: { color: colors.tertiary } }
            ], {...layoutDefaults, barmode: 'stack', yaxis: {...layoutDefaults.yaxis, title: 'Seconds'}});
        }

        // Update charts with new data
        function updateCharts(metrics) {
            const iter = metrics.iteration;

            // Store data
            data.iterations.push(iter);
            data.totalLoss.push(metrics.total_loss);
            data.policyLoss.push(metrics.policy_loss);
            data.valueLoss.push(metrics.value_loss);
            data.movesPerSec.push(metrics.moves_per_sec);
            data.simsPerSec.push(metrics.sims_per_sec);
            data.whiteWins.push(metrics.white_wins);
            data.blackWins.push(metrics.black_wins);
            data.draws.push(metrics.draws);
            data.selfplayTime.push(metrics.selfplay_time);
            data.trainTime.push(metrics.train_time);

            // Update cumulative stats
            totalGames += metrics.num_games;
            totalMoves += Math.round(metrics.moves_per_sec * metrics.selfplay_time);

            // Update loss chart
            Plotly.extendTraces('loss-chart', {
                x: [[iter]], y: [[metrics.total_loss]]
            }, [0]);

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

            // Update WDL chart (last 10)
            const last10 = Math.max(0, data.iterations.length - 10);
            Plotly.react('wdl-chart', [
                { x: data.iterations.slice(last10), y: data.whiteWins.slice(last10), type: 'bar', name: 'White', marker: { color: colors.white } },
                { x: data.iterations.slice(last10), y: data.draws.slice(last10), type: 'bar', name: 'Draw', marker: { color: colors.draw } },
                { x: data.iterations.slice(last10), y: data.blackWins.slice(last10), type: 'bar', name: 'Black', marker: { color: colors.black } }
            ], {...layoutDefaults, barmode: 'stack', yaxis: {...layoutDefaults.yaxis, title: 'Games'}});

            // Update time chart (last 10)
            Plotly.react('time-chart', [
                { x: data.iterations.slice(last10), y: data.selfplayTime.slice(last10), type: 'bar', name: 'Self-Play', marker: { color: colors.primary } },
                { x: data.iterations.slice(last10), y: data.trainTime.slice(last10), type: 'bar', name: 'Training', marker: { color: colors.tertiary } }
            ], {...layoutDefaults, barmode: 'stack', yaxis: {...layoutDefaults.yaxis, title: 'Seconds'}});

            // Update header stats
            document.getElementById('iteration').textContent = iter;
            document.getElementById('total-games').textContent = totalGames.toLocaleString();
            document.getElementById('total-moves').textContent = totalMoves.toLocaleString();
            document.getElementById('buffer-size').textContent = metrics.buffer_size.toLocaleString();

            // Update current metrics
            document.getElementById('current-loss').textContent = metrics.total_loss ? metrics.total_loss.toFixed(4) : '-';
            document.getElementById('current-moves').textContent = metrics.moves_per_sec.toFixed(1);
            document.getElementById('current-sims').textContent = Math.round(metrics.sims_per_sec).toLocaleString();
            document.getElementById('current-games').textContent = metrics.games_per_hour.toFixed(1);

            // Calculate and update ETA
            const avgIterTime = data.iterations.length > 0 ?
                (data.selfplayTime.reduce((a,b) => a+b, 0) + data.trainTime.reduce((a,b) => a+b, 0)) / data.iterations.length : 0;
            const remainingIters = totalIterations - iter;
            const etaSeconds = avgIterTime * remainingIters;

            document.getElementById('avg-iter-time').textContent = formatTime(avgIterTime);
            document.getElementById('eta').textContent = formatTime(etaSeconds);
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
            const socket = io();

            socket.on('connect', () => {
                console.log('Connected to dashboard server');
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').className = 'status connected';
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

                // Update progress bar
                const progressBar = document.getElementById('progress-bar');
                const progressPercent = data.progress_percent.toFixed(1);
                progressBar.style.width = progressPercent + '%';
                progressBar.textContent = progressPercent + '%';

                // Update phase indicator
                const phaseIcon = document.getElementById('phase-icon');
                const phaseText = document.getElementById('phase-text');
                if (data.phase === 'selfplay') {
                    phaseIcon.textContent = '🎮';
                    phaseText.textContent = `Iteration ${data.iteration} - Self-Play`;
                } else if (data.phase === 'training') {
                    phaseIcon.textContent = '🧠';
                    phaseText.textContent = `Iteration ${data.iteration} - Training`;
                }

                // Update ETA
                const phaseEta = document.getElementById('phase-eta');
                phaseEta.textContent = formatTime(data.phase_eta);

                // Update live stats with animation
                updateLiveStat('live-games', `${data.games_completed}/${data.total_games}`);
                updateLiveStat('live-moves', data.moves_per_sec.toFixed(1));
                updateLiveStat('live-sims', Math.round(data.sims_per_sec).toLocaleString());
                updateLiveStat('live-evals', Math.round(data.evals_per_sec).toLocaleString());
                updateLiveStat('live-gph', data.games_per_hour.toFixed(1));

                // Update buffer size in header
                document.getElementById('buffer-size').textContent = data.buffer_size.toLocaleString();

                // Update elapsed time from server
                startTime = Date.now() - (data.total_elapsed * 1000);
            });
        }

        // Helper function to update live stat with highlight animation
        function updateLiveStat(elementId, value) {
            const element = document.getElementById(elementId);
            const oldValue = element.textContent;
            element.textContent = value;

            // Add highlight animation if value changed
            if (oldValue !== value) {
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
            'avg_game_length': metrics.avg_game_length,
            'num_games': metrics.num_games,
            'buffer_size': metrics.buffer_size,
            'selfplay_time': metrics.selfplay_time,
            'train_time': metrics.train_time,
            'iteration_time': metrics.total_time,
        }

        self.history.append(dashboard_metrics)
        self.socketio.emit('metrics', dashboard_metrics)

    def push_progress(self, iteration: int, games_completed: int, total_games: int,
                       moves: int, sims: int, evals: int, elapsed_time: float,
                       buffer_size: int, phase: str = "selfplay"):
        """Push real-time progress updates during self-play (every few seconds).

        Args:
            iteration: Current iteration number
            games_completed: Games completed so far this iteration
            total_games: Total games for this iteration
            moves: Total moves so far
            sims: Total MCTS simulations so far
            evals: Total NN evaluations so far
            elapsed_time: Seconds elapsed in this phase
            buffer_size: Current replay buffer size
            phase: Current phase ("selfplay" or "training")
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
            'buffer_size': buffer_size,
            'phase_elapsed': elapsed_time,
            'phase_eta': eta_seconds,
            'total_elapsed': total_elapsed,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
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
