"""Training visualization dashboard using Plotly Dash.

Real-time visualization of training metrics including loss, accuracy,
games/hour, and replay buffer statistics.
"""

import json
from pathlib import Path
from typing import List, Dict
import plotly.graph_objs as go
from plotly.subplots import make_subplots


try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Warning: dash not installed. Install with: pip install dash plotly")


class TrainingDashboard:
    """Interactive dashboard for visualizing training metrics."""

    def __init__(self, log_dir: str = "logs/metrics", port: int = 8050):
        """Initialize dashboard.

        Args:
            log_dir: Directory containing metrics logs
            port: Port to run dashboard on
        """
        if not DASH_AVAILABLE:
            raise ImportError("dash is required. Install with: pip install dash plotly")

        self.log_dir = Path(log_dir)
        self.port = port
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    def _load_metrics(self) -> List[Dict]:
        """Load metrics from JSONL file."""
        metrics_file = self.log_dir / "training_metrics.jsonl"
        if not metrics_file.exists():
            return []

        metrics = []
        with open(metrics_file, 'r') as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
        return metrics

    def _load_summary(self) -> Dict:
        """Load training summary."""
        summary_file = self.log_dir / "training_summary.json"
        if not summary_file.exists():
            return {}

        with open(summary_file, 'r') as f:
            return json.load(f)

    def _setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            html.H1("AlphaZero Training Dashboard",
                   style={'textAlign': 'center', 'color': '#2c3e50'}),

            # Summary statistics
            html.Div(id='summary-stats', style={
                'display': 'flex',
                'justifyContent': 'space-around',
                'padding': '20px',
                'backgroundColor': '#ecf0f1',
                'borderRadius': '10px',
                'margin': '20px'
            }),

            # Graphs
            dcc.Graph(id='loss-graph'),
            dcc.Graph(id='accuracy-graph'),
            dcc.Graph(id='buffer-games-graph'),
            dcc.Graph(id='learning-rate-graph'),

            # Auto-refresh interval (every 5 seconds)
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # in milliseconds
                n_intervals=0
            )
        ])

    def _setup_callbacks(self):
        """Setup dashboard callbacks for real-time updates."""

        @self.app.callback(
            [Output('summary-stats', 'children'),
             Output('loss-graph', 'figure'),
             Output('accuracy-graph', 'figure'),
             Output('buffer-games-graph', 'figure'),
             Output('learning-rate-graph', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            metrics = self._load_metrics()
            summary = self._load_summary()

            # Summary statistics
            summary_div = self._create_summary_stats(summary)

            # Create figures
            loss_fig = self._create_loss_figure(metrics)
            accuracy_fig = self._create_accuracy_figure(metrics)
            buffer_games_fig = self._create_buffer_games_figure(metrics)
            lr_fig = self._create_lr_figure(metrics)

            return summary_div, loss_fig, accuracy_fig, buffer_games_fig, lr_fig

    def _create_summary_stats(self, summary: Dict) -> List:
        """Create summary statistics cards."""
        def stat_card(title, value, color):
            return html.Div([
                html.H3(title, style={'color': '#7f8c8d', 'margin': '0'}),
                html.H2(value, style={'color': color, 'margin': '10px 0'})
            ], style={'textAlign': 'center', 'padding': '10px'})

        total_steps = summary.get('total_steps', 0)
        total_games = summary.get('total_games', 0)
        best_loss = summary.get('best_loss', 0)
        steps_per_sec = summary.get('steps_per_second', 0)
        elapsed_time = summary.get('elapsed_time', 0)

        return [
            stat_card('Total Steps', f"{total_steps:,}", '#3498db'),
            stat_card('Total Games', f"{total_games:,}", '#2ecc71'),
            stat_card('Best Loss', f"{best_loss:.4f}", '#e74c3c'),
            stat_card('Steps/Sec', f"{steps_per_sec:.2f}", '#9b59b6'),
            stat_card('Training Time', f"{elapsed_time/3600:.1f}h", '#f39c12')
        ]

    def _create_loss_figure(self, metrics: List[Dict]) -> go.Figure:
        """Create loss plot."""
        if not metrics:
            return go.Figure()

        steps = [m['step'] for m in metrics]
        total_loss = [m.get('loss', 0) for m in metrics]
        policy_loss = [m.get('policy_loss', 0) for m in metrics]
        value_loss = [m.get('value_loss', 0) for m in metrics]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=total_loss, name='Total Loss',
                                line=dict(color='#e74c3c', width=2)))
        fig.add_trace(go.Scatter(x=steps, y=policy_loss, name='Policy Loss',
                                line=dict(color='#3498db', width=2)))
        fig.add_trace(go.Scatter(x=steps, y=value_loss, name='Value Loss',
                                line=dict(color='#2ecc71', width=2)))

        fig.update_layout(
            title='Training Loss',
            xaxis_title='Training Steps',
            yaxis_title='Loss',
            hovermode='x unified',
            template='plotly_white'
        )
        return fig

    def _create_accuracy_figure(self, metrics: List[Dict]) -> go.Figure:
        """Create accuracy plot."""
        if not metrics:
            return go.Figure()

        steps = [m['step'] for m in metrics]
        policy_acc = [m.get('policy_accuracy', 0) for m in metrics]
        value_acc = [m.get('value_accuracy', 0) for m in metrics]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=policy_acc, name='Policy Accuracy',
                                line=dict(color='#3498db', width=2)))
        fig.add_trace(go.Scatter(x=steps, y=value_acc, name='Value Accuracy',
                                line=dict(color='#2ecc71', width=2)))

        fig.update_layout(
            title='Training Accuracy',
            xaxis_title='Training Steps',
            yaxis_title='Accuracy',
            hovermode='x unified',
            template='plotly_white'
        )
        return fig

    def _create_buffer_games_figure(self, metrics: List[Dict]) -> go.Figure:
        """Create buffer size and games plot."""
        if not metrics:
            return go.Figure()

        steps = [m['step'] for m in metrics]
        buffer_size = [m.get('buffer_size', 0) for m in metrics]
        games = [m.get('games', 0) for m in metrics]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=steps, y=buffer_size, name='Buffer Size',
                      line=dict(color='#9b59b6', width=2)),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=steps, y=games, name='Total Games',
                      line=dict(color='#f39c12', width=2)),
            secondary_y=True
        )

        fig.update_xaxes(title_text='Training Steps')
        fig.update_yaxes(title_text='Buffer Size', secondary_y=False)
        fig.update_yaxes(title_text='Total Games', secondary_y=True)

        fig.update_layout(
            title='Replay Buffer & Games',
            hovermode='x unified',
            template='plotly_white'
        )
        return fig

    def _create_lr_figure(self, metrics: List[Dict]) -> go.Figure:
        """Create learning rate plot."""
        if not metrics:
            return go.Figure()

        steps = [m['step'] for m in metrics]
        lr = [m.get('learning_rate', 0) for m in metrics]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=lr, name='Learning Rate',
                                line=dict(color='#e67e22', width=2),
                                fill='tozeroy'))

        fig.update_layout(
            title='Learning Rate Schedule',
            xaxis_title='Training Steps',
            yaxis_title='Learning Rate',
            yaxis_type='log',
            hovermode='x unified',
            template='plotly_white'
        )
        return fig

    def run(self, debug: bool = False):
        """Run the dashboard server.

        Args:
            debug: Enable debug mode
        """
        print(f"Starting training dashboard on http://localhost:{self.port}")
        print(f"Monitoring metrics from: {self.log_dir}")
        print("Press Ctrl+C to stop")
        self.app.run_server(debug=debug, port=self.port)


def main():
    """Run dashboard from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="AlphaZero Training Dashboard")
    parser.add_argument("--log-dir", type=str, default="logs/metrics",
                       help="Directory containing training metrics")
    parser.add_argument("--port", type=int, default=8050,
                       help="Port to run dashboard on")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")

    args = parser.parse_args()

    dashboard = TrainingDashboard(log_dir=args.log_dir, port=args.port)
    dashboard.run(debug=args.debug)


if __name__ == "__main__":
    main()
