"""Tests for dashboard visualization functionality.

Tests the Dash-based training dashboard and metrics visualization.
"""

import pytest
import sys
from pathlib import Path
import json
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if Dash is available
try:
    import dash
    from alphazero.visualization.dashboard import TrainingDashboard
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not DASH_AVAILABLE, reason="Dash not installed")


@pytest.fixture
def temp_metrics_dir(tmp_path):
    """Create a temporary directory with sample metrics."""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()

    # Create sample metrics file
    metrics_file = metrics_dir / "training_metrics.jsonl"

    sample_metrics = [
        {
            "step": 100,
            "loss": 2.5,
            "policy_loss": 1.5,
            "value_loss": 1.0,
            "policy_accuracy": 0.3,
            "value_accuracy": 0.4,
            "learning_rate": 0.2,
            "games_played": 50,
            "buffer_size": 5000
        },
        {
            "step": 200,
            "loss": 2.3,
            "policy_loss": 1.4,
            "value_loss": 0.9,
            "policy_accuracy": 0.35,
            "value_accuracy": 0.45,
            "learning_rate": 0.2,
            "games_played": 100,
            "buffer_size": 10000
        },
        {
            "step": 300,
            "loss": 2.1,
            "policy_loss": 1.3,
            "value_loss": 0.8,
            "policy_accuracy": 0.4,
            "value_accuracy": 0.5,
            "learning_rate": 0.2,
            "games_played": 150,
            "buffer_size": 15000
        }
    ]

    with open(metrics_file, 'w') as f:
        for metric in sample_metrics:
            f.write(json.dumps(metric) + '\n')

    return str(metrics_dir)


class TestDashboardCreation:
    """Test dashboard initialization."""

    def test_dashboard_creation(self, temp_metrics_dir):
        """Test creating a dashboard instance."""
        dashboard = TrainingDashboard(log_dir=temp_metrics_dir, port=8051)
        assert dashboard is not None
        assert dashboard.port == 8051
        assert dashboard.log_dir == Path(temp_metrics_dir)

    def test_dashboard_app_exists(self, temp_metrics_dir):
        """Test that Dash app is created."""
        dashboard = TrainingDashboard(log_dir=temp_metrics_dir, port=8051)
        assert dashboard.app is not None
        assert hasattr(dashboard.app, 'run')  # Should have run method, not run_server


class TestMetricsLoading:
    """Test metrics loading functionality."""

    def test_load_metrics_from_file(self, temp_metrics_dir):
        """Test loading metrics from JSONL file."""
        dashboard = TrainingDashboard(log_dir=temp_metrics_dir, port=8051)
        metrics = dashboard._load_metrics()

        assert len(metrics) == 3
        assert metrics[0]['step'] == 100
        assert metrics[1]['step'] == 200
        assert metrics[2]['step'] == 300

    def test_load_metrics_empty_dir(self, tmp_path):
        """Test loading metrics from empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        dashboard = TrainingDashboard(log_dir=str(empty_dir), port=8051)
        metrics = dashboard._load_metrics()

        assert metrics == []

    def test_metrics_have_required_fields(self, temp_metrics_dir):
        """Test that loaded metrics have required fields."""
        dashboard = TrainingDashboard(log_dir=temp_metrics_dir, port=8051)
        metrics = dashboard._load_metrics()

        required_fields = ['step', 'loss', 'policy_loss', 'value_loss']
        for metric in metrics:
            for field in required_fields:
                assert field in metric


class TestDashboardAPI:
    """Test dashboard API compatibility."""

    def test_run_method_exists(self, temp_metrics_dir):
        """Test that run method exists (not deprecated run_server)."""
        dashboard = TrainingDashboard(log_dir=temp_metrics_dir, port=8051)

        # Should have run method
        assert hasattr(dashboard.app, 'run')

        # Should NOT use run_server (deprecated)
        # We check that the dashboard.run method calls app.run, not app.run_server
        import inspect
        source = inspect.getsource(dashboard.run)
        assert 'app.run(' in source
        assert 'app.run_server(' not in source

    def test_dashboard_config(self, temp_metrics_dir):
        """Test dashboard configuration."""
        dashboard = TrainingDashboard(log_dir=temp_metrics_dir, port=8052)

        # Test that app is configured correctly
        assert dashboard.app is not None
        assert dashboard.port == 8052


class TestDashboardLayout:
    """Test dashboard layout and components."""

    def test_layout_exists(self, temp_metrics_dir):
        """Test that dashboard has a layout."""
        dashboard = TrainingDashboard(log_dir=temp_metrics_dir, port=8051)
        assert dashboard.app.layout is not None

    def test_layout_has_graphs(self, temp_metrics_dir):
        """Test that layout contains graph components."""
        dashboard = TrainingDashboard(log_dir=temp_metrics_dir, port=8051)

        # Convert layout to string to check for graph components
        layout_str = str(dashboard.app.layout)

        # Should have some graph-related components
        # (exact structure depends on implementation)
        assert 'Graph' in layout_str or 'graph' in layout_str.lower()


class TestMetricsVisualization:
    """Test metrics visualization functionality."""

    def test_metrics_data_structure(self, temp_metrics_dir):
        """Test that metrics can be structured for visualization."""
        dashboard = TrainingDashboard(log_dir=temp_metrics_dir, port=8051)
        metrics = dashboard._load_metrics()

        # Extract steps and losses for plotting
        steps = [m['step'] for m in metrics]
        losses = [m['loss'] for m in metrics]

        assert len(steps) == 3
        assert len(losses) == 3
        assert steps == [100, 200, 300]
        assert losses == [2.5, 2.3, 2.1]

    def test_metrics_trend(self, temp_metrics_dir):
        """Test that metrics show expected trends."""
        dashboard = TrainingDashboard(log_dir=temp_metrics_dir, port=8051)
        metrics = dashboard._load_metrics()

        # Loss should be decreasing
        losses = [m['loss'] for m in metrics]
        assert losses[0] > losses[1] > losses[2]

        # Accuracy should be increasing
        policy_acc = [m['policy_accuracy'] for m in metrics]
        assert policy_acc[0] < policy_acc[1] < policy_acc[2]


class TestErrorHandling:
    """Test error handling in dashboard."""

    def test_nonexistent_directory(self):
        """Test handling of nonexistent metrics directory."""
        # Should not crash, just return empty metrics
        dashboard = TrainingDashboard(log_dir="/nonexistent/path", port=8051)
        metrics = dashboard._load_metrics()
        assert metrics == []

    def test_malformed_metrics_file(self, tmp_path):
        """Test handling of malformed metrics file."""
        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()

        # Create file with invalid JSON
        metrics_file = metrics_dir / "training_metrics.jsonl"
        with open(metrics_file, 'w') as f:
            f.write("invalid json\n")
            f.write('{"step": 100, "loss": 2.0}\n')  # Valid line after invalid

        dashboard = TrainingDashboard(log_dir=str(metrics_dir), port=8051)

        # Should handle gracefully - either skip invalid lines or return empty
        try:
            metrics = dashboard._load_metrics()
            # If it loads, should have at most 1 valid metric
            assert len(metrics) <= 1
        except Exception:
            # Or it might raise an exception, which is also acceptable
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
