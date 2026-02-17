#!/usr/bin/env python3
"""
Generate summary.html from training_metrics.json and evaluation_results.json.

This script creates a standalone HTML dashboard that visualizes:
- Overall training statistics (iterations, games, moves, loss improvement)
- Evaluation results (win rate vs random, endgame puzzle accuracy)
- Training loss charts (total, policy, value loss over time)
- Evaluation progress charts (win rate and move accuracy over time)

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
    """Load training metrics history from a run directory.

    Args:
        run_dir: Path to the run directory

    Returns:
        Metrics history dictionary with 'iterations' and 'config' keys
    """
    metrics_path = os.path.join(run_dir, "training_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, ValueError):
            print(f"  Warning: Could not parse {metrics_path}, using defaults")
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

    The HTML uses fetch() to load data from training_metrics.json and
    evaluation_results.json at view time, with embedded JSON fallback for
    file:// protocol (where CORS blocks fetch).

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


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate summary.html from training JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_summary.py checkpoints/run_192x15_20250205_123456
    python generate_summary.py ./my_training_run

The script reads training_metrics.json and evaluation_results.json from the
specified directory and generates a summary.html file in the same directory.
        """
    )
    parser.add_argument(
        "run_dir",
        help="Path to run directory containing training_metrics.json and evaluation_results.json"
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
