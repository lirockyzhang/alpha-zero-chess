#!/usr/bin/env python3
"""
Generate summary.html from training_log.jsonl and evaluation_results.json.

This script creates a standalone HTML dashboard that visualizes:
- Overall training statistics (iterations, games, moves, loss improvement)
- Evaluation results (win rate vs random, endgame puzzle accuracy)
- Training loss charts (total, policy, value loss over time)
- Evaluation progress charts (win rate and move accuracy over time)
- GPU performance (CUDA graph distribution, inference time)
- Batch pipeline (fill percentiles, batch count)
- Throughput (NN evals/sec, positions/sec)
- GPU memory (allocated vs reserved)
- Queue health (GPU wait, worker wait)
- Game quality (draw breakdown, decisive rate)
- Training diagnostics (gradient norms, learning rate)
- Search depth (avg/max)

The HTML uses fetch() to load data from JSON files at view time, with an
embedded JSON fallback for file:// protocol (where CORS blocks fetch).

Usage:
    python generate_summary.py <run_directory>
    python generate_summary.py checkpoints/run_192x15_20250205_123456

Can also be imported and used programmatically:
    from generate_summary import generate_summary_html
    summary_path = generate_summary_html(run_dir, config)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict


def load_metrics_history(run_dir: str) -> Dict[str, Any]:
    """Load training metrics from JSONL or legacy JSON.

    Tries training_log.jsonl first, falls back to training_metrics.json.
    Returns the same shape as before: {"iterations": [...], "config": {...}}
    so JavaScript code works identically.
    """
    # Try JSONL first
    jsonl_path = os.path.join(run_dir, "training_log.jsonl")
    if os.path.exists(jsonl_path):
        config = {}
        iterations = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("type") == "config":
                        config = {k: v for k, v in rec.items() if k != "type"}
                    elif rec.get("type") == "iteration":
                        iterations.append(rec)
                except (json.JSONDecodeError, ValueError):
                    continue
        return {"iterations": iterations, "config": config}

    # Fallback: legacy JSON
    legacy_path = os.path.join(run_dir, "training_metrics.json")
    if os.path.exists(legacy_path):
        try:
            with open(legacy_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, ValueError):
            print(f"  Warning: Could not parse {legacy_path}, using defaults")

    return {"iterations": [], "config": {}}


def load_eval_history(run_dir: str) -> Dict[str, Any]:
    """Load evaluation history from a run directory.

    Args:
        run_dir: Path to the run directory

    Returns:
        Evaluation history dictionary with 'evaluations' key
    """
    eval_path = os.path.join(run_dir, "evaluation_results.json")
    if os.path.exists(eval_path):
        try:
            with open(eval_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, ValueError):
            print(f"  Warning: Could not parse {eval_path}, using defaults")
    return {"evaluations": []}


def generate_summary_html(run_dir: str, config: Dict[str, Any] = None) -> str:
    """Generate a summary HTML file with training metrics and evaluation results.

    The HTML embeds all data for offline viewing (file:// protocol),
    with fetch() as a progressive enhancement when served via HTTP.

    Args:
        run_dir: The run directory to save summary.html
        config: Training configuration (optional, will load from metrics if not provided)

    Returns:
        Path to the generated summary.html file
    """
    if config is None:
        config = {}

    # Read JSON files from disk for the embedded fallback
    metrics_history = load_metrics_history(run_dir)
    eval_history = load_eval_history(run_dir)

    # Use config from metrics file if not provided
    if not config:
        config = metrics_history.get("config", {})

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
        '        .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }\n'
        '        .chart-row .chart-container { height: 250px; }\n'
        '        .config-table { width: 100%; border-collapse: collapse; }\n'
        '        .config-table td { padding: 8px; border-bottom: 1px solid #eee; }\n'
        '        .config-table td:first-child { font-weight: bold; color: #7f8c8d; width: 40%; }\n'
        '        .hidden { display: none; }\n'
        '        @media (max-width: 768px) { .chart-row { grid-template-columns: 1fr; } }\n'
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
        # New chart cards (hidden by default, shown if data exists)
        '        <div class="card hidden" id="gameQualityCard">\n'
        '            <h2>&#9823; Game Quality</h2>\n'
        '            <div class="chart-container">\n'
        '                <canvas id="gameQualityChart"></canvas>\n'
        '            </div>\n'
        '        </div>\n'
        '\n'
        '        <div class="card hidden" id="throughputCard">\n'
        '            <h2>&#9889; Throughput</h2>\n'
        '            <div class="chart-row">\n'
        '                <div class="chart-container"><canvas id="throughputChart"></canvas></div>\n'
        '                <div class="chart-container"><canvas id="batchFillChart"></canvas></div>\n'
        '            </div>\n'
        '        </div>\n'
        '\n'
        '        <div class="card hidden" id="gpuCard">\n'
        '            <h2>&#128187; GPU Performance</h2>\n'
        '            <div class="chart-row">\n'
        '                <div class="chart-container"><canvas id="cudaGraphChart"></canvas></div>\n'
        '                <div class="chart-container"><canvas id="gpuMemChart"></canvas></div>\n'
        '            </div>\n'
        '        </div>\n'
        '\n'
        '        <div class="card hidden" id="queueHealthCard">\n'
        '            <h2>&#128295; Queue Health &amp; Search Depth</h2>\n'
        '            <div class="chart-row">\n'
        '                <div class="chart-container"><canvas id="queueWaitChart"></canvas></div>\n'
        '                <div class="chart-container"><canvas id="searchDepthChart"></canvas></div>\n'
        '            </div>\n'
        '        </div>\n'
        '\n'
        '        <div class="card hidden" id="trainingDiagCard">\n'
        '            <h2>&#128269; Training Diagnostics</h2>\n'
        '            <div class="chart-container">\n'
        '                <canvas id="gradNormChart"></canvas>\n'
        '            </div>\n'
        '        </div>\n'
        '\n'
        '        <div class="card hidden" id="reanalysisCard">\n'
        '            <h2>&#128260; Reanalysis</h2>\n'
        '            <div class="chart-row">\n'
        '                <div class="chart-container"><canvas id="reanalysisKLChart"></canvas></div>\n'
        '                <div class="chart-container"><canvas id="reanalysisVolumeChart"></canvas></div>\n'
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
        '        async function loadJSONL(path, fallback) {\n'
        '            try {\n'
        '                const resp = await fetch(path);\n'
        '                if (!resp.ok) throw new Error(resp.statusText);\n'
        '                const text = await resp.text();\n'
        '                const config = {};\n'
        '                const iterations = [];\n'
        '                for (const line of text.split("\\n")) {\n'
        '                    if (!line.trim()) continue;\n'
        '                    try {\n'
        '                        const rec = JSON.parse(line);\n'
        '                        if (rec.type === "config") Object.assign(config, rec);\n'
        '                        else if (rec.type === "iteration") iterations.push(rec);\n'
        '                    } catch(e) {}\n'
        '                }\n'
        '                return {config, iterations};\n'
        '            } catch(e) {\n'
        '                return fallback;\n'
        '            }\n'
        '        }\n'
        '\n'
        '        function formatNumber(n) {\n'
        '            return n.toLocaleString();\n'
        '        }\n'
        '\n'
        # Helper: check if any iteration has a given field
        '        function hasField(iterations, field) {\n'
        '            return iterations.some(m => m[field] != null && m[field] !== 0);\n'
        '        }\n'
        '\n'
        '        function makeLineChart(canvasId, labels, datasets, opts) {\n'
        '            const scales = {};\n'
        '            if (opts.yTitle) scales.y = { type: "linear", position: "left", beginAtZero: opts.yZero || false, title: { display: true, text: opts.yTitle } };\n'
        '            if (opts.y1Title) scales.y1 = { type: "linear", position: "right", beginAtZero: opts.y1Zero || false, title: { display: true, text: opts.y1Title }, grid: { drawOnChartArea: false } };\n'
        '            return new Chart(document.getElementById(canvasId), {\n'
        '                type: "line",\n'
        '                data: { labels, datasets },\n'
        '                options: { responsive: true, maintainAspectRatio: false, scales,\n'
        '                    plugins: { legend: { position: "top" } } }\n'
        '            });\n'
        '        }\n'
        '\n'
        '        const COLORS = { blue: "#3498db", green: "#2ecc71", red: "#e74c3c",\n'
        '            orange: "#e67e22", purple: "#9b59b6", teal: "#1abc9c",\n'
        '            yellow: "#f1c40f", pink: "#e84393", grey: "#95a5a6" };\n'
        '\n'
        '        function ds(label, data, color, axisId, extra) {\n'
        '            return Object.assign({ label, data, borderColor: color, fill: false, tension: 0.1, yAxisID: axisId || "y", pointRadius: 2 }, extra || {});\n'
        '        }\n'
        '\n'
        '        function buildDashboard(metricsData, evalData) {\n'
        '            const iterations = metricsData.iterations || [];\n'
        '            const evaluations = evalData.evaluations || [];\n'
        '            const iters = iterations.map(m => m.iteration || 0);\n'
        '\n'
        '            // --- Overall statistics ---\n'
        "            document.getElementById('statIterations').textContent = iterations.length;\n"
        '            if (iterations.length > 0) {\n'
        '                const totalGames = iterations.reduce((s, m) => s + (m.games || 0), 0);\n'
        '                const totalMoves = iterations.reduce((s, m) => s + (m.moves || 0), 0);\n'
        "                document.getElementById('statGames').textContent = formatNumber(totalGames);\n"
        "                document.getElementById('statMoves').textContent = formatNumber(totalMoves);\n"
        '                if (iterations.length >= 2 && iterations[0].loss > 0) {\n'
        '                    const first = iterations[0].loss, last = iterations[iterations.length - 1].loss;\n'
        "                    document.getElementById('statImprovement').textContent = ((first - last) / first * 100).toFixed(1) + '%';\n"
        '                } else {\n'
        "                    document.getElementById('statImprovement').textContent = '0.0%';\n"
        '                }\n'
        '            } else {\n'
        "                ['statGames','statMoves','statImprovement'].forEach(id => document.getElementById(id).textContent = '0');\n"
        '            }\n'
        '\n'
        '            // --- Evaluation card ---\n'
        '            if (evaluations.length > 0) {\n'
        '                const latest = evaluations[evaluations.length - 1];\n'
        '                const vsRandom = latest.vs_random || {}, endgame = latest.endgame || {};\n'
        "                document.getElementById('evalCard').classList.remove('hidden');\n"
        "                document.getElementById('evalIter').textContent = latest.iteration || 'N/A';\n"
        "                document.getElementById('evalWinRate').textContent = ((vsRandom.win_rate || 0) * 100).toFixed(0) + '%';\n"
        "                document.getElementById('evalWinRateLabel').textContent = 'vs Random Win Rate (' + (vsRandom.wins || 0) + '/5 games)';\n"
        "                document.getElementById('evalEndgame').textContent = (endgame.score || 0) + '/' + (endgame.total || 5);\n"
        "                document.getElementById('evalMoveAcc').textContent = ((endgame.move_accuracy || 0) * 100).toFixed(0) + '%';\n"
        "                document.getElementById('evalMoveAccLabel').textContent = 'Move Accuracy (' + (endgame.move_score || 0) + '/' + (endgame.total_moves || 0) + ' moves)';\n"
        '            }\n'
        '\n'
        '            // --- Loss chart (always shown) ---\n'
        "            makeLineChart('lossChart', iters, [\n"
        "                ds('Total Loss', iterations.map(m => m.loss || 0), COLORS.blue, 'y'),\n"
        "                ds('Policy Loss', iterations.map(m => m.policy_loss || 0), COLORS.green, 'y'),\n"
        "                ds('Value Loss', iterations.map(m => m.value_loss || 0), COLORS.red, 'y1'),\n"
        "            ], { yTitle: 'Total/Policy Loss', y1Title: 'Value Loss' });\n"
        '\n'
        '            // --- Evaluation chart ---\n'
        '            if (evaluations.length > 0) {\n'
        "                document.getElementById('evalChartCard').classList.remove('hidden');\n"
        '                const evalIters = evaluations.map(e => e.iteration || 0);\n'
        "                makeLineChart('evalChart', evalIters, [\n"
        "                    ds('vs Random Win Rate (%)', evaluations.map(e => ((e.vs_random || {}).win_rate || 0) * 100), COLORS.blue, 'y'),\n"
        "                    ds('Endgame Move Accuracy (%)', evaluations.map(e => ((e.endgame || {}).move_accuracy || 0) * 100), COLORS.green, 'y1'),\n"
        "                ], { yTitle: 'Win Rate (%)', y1Title: 'Move Accuracy (%)', yZero: true, y1Zero: true });\n"
        '            }\n'
        '\n'
        '            // =====================================================\n'
        '            // NEW CHARTS — conditionally shown if data exists\n'
        '            // =====================================================\n'
        '\n'
        '            // --- Game Quality: draw breakdown stacked area + decisive rate ---\n'
        "            if (hasField(iterations, 'draws') && iterations.length > 1) {\n"
        "                document.getElementById('gameQualityCard').classList.remove('hidden');\n"
        "                new Chart(document.getElementById('gameQualityChart'), {\n"
        "                    type: 'line',\n"
        '                    data: { labels: iters, datasets: [\n'
        "                        { label: 'Repetition', data: iterations.map(m => m.draws_repetition || 0), backgroundColor: 'rgba(231,76,60,0.3)', borderColor: COLORS.red, fill: true, tension: 0.1, pointRadius: 1 },\n"
        "                        { label: 'Fifty-move', data: iterations.map(m => m.draws_fifty_move || 0), backgroundColor: 'rgba(230,126,34,0.3)', borderColor: COLORS.orange, fill: true, tension: 0.1, pointRadius: 1 },\n"
        "                        { label: 'Stalemate', data: iterations.map(m => m.draws_stalemate || 0), backgroundColor: 'rgba(155,89,182,0.3)', borderColor: COLORS.purple, fill: true, tension: 0.1, pointRadius: 1 },\n"
        "                        { label: 'Insufficient', data: iterations.map(m => m.draws_insufficient || 0), backgroundColor: 'rgba(149,165,166,0.3)', borderColor: COLORS.grey, fill: true, tension: 0.1, pointRadius: 1 },\n"
        "                        { label: 'Max moves', data: iterations.map(m => m.draws_max_moves || 0), backgroundColor: 'rgba(241,196,15,0.3)', borderColor: COLORS.yellow, fill: true, tension: 0.1, pointRadius: 1 },\n"
        "                        { label: 'Decisive Rate %', data: iterations.map(m => { const g = m.games || 1; return ((m.white_wins || 0) + (m.black_wins || 0)) / g * 100; }), borderColor: COLORS.blue, fill: false, tension: 0.1, pointRadius: 2, borderWidth: 2, yAxisID: 'y1' },\n"
        '                    ]},\n'
        '                    options: { responsive: true, maintainAspectRatio: false,\n'
        "                        scales: { y: { stacked: true, title: { display: true, text: 'Draw Count' }, beginAtZero: true },\n"
        "                            y1: { position: 'right', min: 0, max: 100, title: { display: true, text: 'Decisive Rate (%)' }, grid: { drawOnChartArea: false } } },\n"
        '                        plugins: { legend: { position: "top" } } }\n'
        '                });\n'
        '            }\n'
        '\n'
        '            // --- Throughput: NN evals/sec + positions/sec ---\n'
        "            if (hasField(iterations, 'nn_evals_per_sec')) {\n"
        "                document.getElementById('throughputCard').classList.remove('hidden');\n"
        "                makeLineChart('throughputChart', iters, [\n"
        "                    ds('NN Evals/sec', iterations.map(m => m.nn_evals_per_sec || 0), COLORS.blue, 'y'),\n"
        "                    ds('Positions/sec', iterations.map(m => m.positions_per_sec || 0), COLORS.green, 'y1'),\n"
        "                ], { yTitle: 'NN Evals/sec', y1Title: 'Positions/sec', yZero: true, y1Zero: true });\n"
        '\n'
        '                // --- Batch Fill p50/p90 ---\n'
        "                makeLineChart('batchFillChart', iters, [\n"
        "                    ds('Batch P50', iterations.map(m => m.batch_fill_p50 || 0), COLORS.blue, 'y'),\n"
        "                    ds('Batch P90', iterations.map(m => m.batch_fill_p90 || 0), COLORS.orange, 'y'),\n"
        "                    ds('Batch Min', iterations.map(m => m.batch_size_min || 0), COLORS.grey, 'y', { borderDash: [4, 4] }),\n"
        "                    ds('Batch Max', iterations.map(m => m.batch_size_max || 0), COLORS.purple, 'y', { borderDash: [4, 4] }),\n"
        "                ], { yTitle: 'Batch Size' });\n"
        '            }\n'
        '\n'
        '            // --- GPU Performance: CUDA graph distribution + memory ---\n'
        "            if (hasField(iterations, 'cuda_large_fires')) {\n"
        "                document.getElementById('gpuCard').classList.remove('hidden');\n"
        "                new Chart(document.getElementById('cudaGraphChart'), {\n"
        "                    type: 'bar',\n"
        '                    data: { labels: iters, datasets: [\n'
        "                        { label: 'Large Graph', data: iterations.map(m => m.cuda_large_fires || 0), backgroundColor: COLORS.blue },\n"
        "                        { label: 'Small/Mini Graph', data: iterations.map(m => m.cuda_small_fires || 0), backgroundColor: COLORS.green },\n"
        "                        { label: 'Eager Fallback', data: iterations.map(m => m.cuda_eager_fires || 0), backgroundColor: COLORS.red },\n"
        '                    ]},\n'
        '                    options: { responsive: true, maintainAspectRatio: false,\n'
        "                        scales: { x: { stacked: true }, y: { stacked: true, title: { display: true, text: 'Batch Count' } } },\n"
        '                        plugins: { legend: { position: "top" } } }\n'
        '                });\n'
        '            }\n'
        '\n'
        '            // --- GPU Memory ---\n'
        "            if (hasField(iterations, 'gpu_memory_allocated_mb')) {\n"
        "                if (!hasField(iterations, 'cuda_large_fires')) document.getElementById('gpuCard').classList.remove('hidden');\n"
        "                makeLineChart('gpuMemChart', iters, [\n"
        "                    ds('Allocated (MB)', iterations.map(m => m.gpu_memory_allocated_mb || 0), COLORS.blue, 'y'),\n"
        "                    ds('Reserved (MB)', iterations.map(m => m.gpu_memory_reserved_mb || 0), COLORS.orange, 'y'),\n"
        "                ], { yTitle: 'GPU Memory (MB)', yZero: true });\n"
        '            }\n'
        '\n'
        '            // --- Queue Health + Search Depth ---\n'
        "            if (hasField(iterations, 'gpu_wait_ms') || hasField(iterations, 'avg_search_depth')) {\n"
        "                document.getElementById('queueHealthCard').classList.remove('hidden');\n"
        "                if (hasField(iterations, 'gpu_wait_ms')) {\n"
        "                    makeLineChart('queueWaitChart', iters, [\n"
        "                        ds('GPU Wait (ms)', iterations.map(m => m.gpu_wait_ms || 0), COLORS.red, 'y'),\n"
        "                        ds('Worker Wait (ms)', iterations.map(m => m.worker_wait_ms || 0), COLORS.blue, 'y'),\n"
        "                    ], { yTitle: 'Wait Time (ms)', yZero: true });\n"
        '                }\n'
        "                if (hasField(iterations, 'avg_search_depth')) {\n"
        "                    makeLineChart('searchDepthChart', iters, [\n"
        "                        ds('Avg Depth', iterations.map(m => m.avg_search_depth || 0), COLORS.blue, 'y'),\n"
        "                        ds('Max Depth', iterations.map(m => m.max_search_depth || 0), COLORS.orange, 'y'),\n"
        "                    ], { yTitle: 'Search Depth', yZero: true });\n"
        '                }\n'
        '            }\n'
        '\n'
        '            // --- Training Diagnostics: grad norms + LR ---\n'
        "            if (hasField(iterations, 'grad_norm_avg')) {\n"
        "                document.getElementById('trainingDiagCard').classList.remove('hidden');\n"
        "                makeLineChart('gradNormChart', iters, [\n"
        "                    ds('Grad Norm (avg)', iterations.map(m => m.grad_norm_avg || 0), COLORS.blue, 'y'),\n"
        "                    ds('Grad Norm (max)', iterations.map(m => m.grad_norm_max || 0), COLORS.red, 'y'),\n"
        "                    ds('Learning Rate', iterations.map(m => m.learning_rate || 0), COLORS.green, 'y1'),\n"
        "                ], { yTitle: 'Gradient Norm', y1Title: 'Learning Rate' });\n"
        '            }\n'
        '\n'
        '            // --- Reanalysis: KL divergence + volume ---\n'
        "            if (hasField(iterations, 'reanalysis_positions')) {\n"
        "                document.getElementById('reanalysisCard').classList.remove('hidden');\n"
        "                makeLineChart('reanalysisKLChart', iters, [\n"
        "                    ds('Mean KL Divergence', iterations.map(m => m.reanalysis_mean_kl || 0), COLORS.purple, 'y'),\n"
        "                ], { yTitle: 'KL(old || new)', yZero: true });\n"
        "                makeLineChart('reanalysisVolumeChart', iters, [\n"
        "                    ds('Positions Reanalyzed', iterations.map(m => m.reanalysis_positions || 0), COLORS.blue, 'y'),\n"
        "                    ds('Tail Time (s)', iterations.map(m => m.reanalysis_time_s || 0), COLORS.orange, 'y1'),\n"
        "                ], { yTitle: 'Positions', y1Title: 'Time (s)', yZero: true, y1Zero: true });\n"
        '            }\n'
        '        }\n'
        '\n'
        '        // Load data and build dashboard\n'
        '        (async function() {\n'
        '            // Try JSONL first, then fall back to legacy JSON\n'
        '            let metricsData = await loadJSONL("./training_log.jsonl", null);\n'
        '            if (!metricsData) {\n'
        '                metricsData = await loadJSON("./training_metrics.json", FALLBACK_METRICS);\n'
        '            }\n'
        "            const evalData = await loadJSON('./evaluation_results.json', FALLBACK_EVAL);\n"
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


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate summary.html from training log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_summary.py checkpoints/run_192x15_20250205_123456
    python generate_summary.py ./my_training_run

The script reads training_log.jsonl (or legacy training_metrics.json) and
evaluation_results.json from the specified directory and generates a
summary.html file in the same directory.
        """
    )
    parser.add_argument(
        "run_dir",
        help="Path to run directory containing training_log.jsonl and evaluation_results.json"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"Error: {args.run_dir} is not a directory", file=sys.stderr)
        return 1

    # Load config from metrics if available
    metrics = load_metrics_history(args.run_dir)
    config = metrics.get("config", {})

    summary_path = generate_summary_html(args.run_dir, config)
    print(f"Generated: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
