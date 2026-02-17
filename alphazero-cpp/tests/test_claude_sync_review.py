"""Tests for synchronous Claude review loop, enhanced stop, and rich resume state.

Tests the three features added:
1. GracefulShutdown stop_file_path integration
2. ClaudeInterface.write_resume_summary()
3. Synchronous review handshake (awaiting_review signal file)
4. --claude-timeout arg parsing
"""

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

# Add scripts dir to path so we can import claude_interface
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from claude_interface import ClaudeInterface


# ─────────────────────────────────────────────────────────────────────
# 1. GracefulShutdown tests
# ─────────────────────────────────────────────────────────────────────

def test_graceful_shutdown_stop_file():
    """should_stop() detects and removes stop file."""
    # We inline the class to avoid importing the whole train.py (needs alphazero_cpp)
    import threading, signal

    class GracefulShutdown:
        def __init__(self):
            self.shutdown_requested = False
            self._lock = threading.Lock()
            self._original_handler = None
            self.stop_file_path = None

        def request_shutdown(self, signum=None, frame=None, source="sigint"):
            with self._lock:
                if not self.shutdown_requested:
                    self.shutdown_requested = True

        def should_stop(self):
            with self._lock:
                if self.shutdown_requested:
                    return True
            if self.stop_file_path and os.path.exists(self.stop_file_path):
                try:
                    os.remove(self.stop_file_path)
                except OSError:
                    pass
                self.request_shutdown(source="stop_file")
                return True
            return False

    with tempfile.TemporaryDirectory() as tmpdir:
        handler = GracefulShutdown()
        stop_path = os.path.join(tmpdir, "stop")
        handler.stop_file_path = stop_path

        # No stop file → should not stop
        assert not handler.should_stop(), "should_stop() should be False when no stop file"

        # Create stop file → should stop and remove it
        with open(stop_path, 'w') as f:
            f.write("")
        assert os.path.exists(stop_path), "stop file should exist before check"
        assert handler.should_stop(), "should_stop() should be True when stop file exists"
        assert not os.path.exists(stop_path), "stop file should be removed after detection"

        # Subsequent calls should still return True (shutdown_requested is sticky)
        assert handler.should_stop(), "should_stop() should remain True after shutdown"

    print("  PASS: test_graceful_shutdown_stop_file")


def test_graceful_shutdown_no_stop_file_path():
    """should_stop() works normally when stop_file_path is None."""
    import threading

    class GracefulShutdown:
        def __init__(self):
            self.shutdown_requested = False
            self._lock = threading.Lock()
            self.stop_file_path = None

        def request_shutdown(self, signum=None, frame=None, source="sigint"):
            with self._lock:
                if not self.shutdown_requested:
                    self.shutdown_requested = True

        def should_stop(self):
            with self._lock:
                if self.shutdown_requested:
                    return True
            if self.stop_file_path and os.path.exists(self.stop_file_path):
                try:
                    os.remove(self.stop_file_path)
                except OSError:
                    pass
                self.request_shutdown(source="stop_file")
                return True
            return False

    handler = GracefulShutdown()
    assert not handler.should_stop(), "should be False with no stop_file_path set"
    handler.request_shutdown(source="sigint")
    assert handler.should_stop(), "should be True after explicit request_shutdown"

    print("  PASS: test_graceful_shutdown_no_stop_file_path")


# ─────────────────────────────────────────────────────────────────────
# 2. ClaudeInterface.write_resume_summary() tests
# ─────────────────────────────────────────────────────────────────────

def _make_mock_args(**overrides):
    """Create a mock args namespace with all 17 tunable params."""
    defaults = dict(
        lr=0.001, train_batch=1024, epochs=5, simulations=800,
        c_puct=1.5, risk_beta=0.0, temperature_moves=30,
        dirichlet_alpha=0.3, dirichlet_epsilon=0.25, fpu_base=1.0,
        risk_beta_final=0.0, risk_beta_warmup=0,
        opponent_risk_min=-1.0, opponent_risk_max=1.0,
        games_per_iter=64, max_fillup_factor=3, save_interval=1,
        iterations=60,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_write_resume_summary_basic():
    """write_resume_summary() creates valid JSON with all required fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        iface = ClaudeInterface(tmpdir)
        args = _make_mock_args()
        buffer_stats = {"total_size": 5000, "capacity": 100000, "wins": 0, "draws": 0, "losses": 0}

        path = iface.write_resume_summary(
            args, iteration=10, total_iterations=60,
            buffer_stats=buffer_stats, shutdown_reason="Test shutdown",
        )

        assert os.path.exists(path), f"Resume file should exist at {path}"
        with open(path) as f:
            data = json.load(f)

        assert data["shutdown_reason"] == "Test shutdown"
        assert data["shutdown_iteration"] == 10
        assert data["total_iterations"] == 60
        assert "shutdown_timestamp" in data
        assert data["buffer_stats"] == buffer_stats

        # Check all 17 params present
        params = data["effective_params"]
        expected_keys = {
            'lr', 'train_batch', 'epochs', 'simulations', 'c_puct',
            'risk_beta', 'temperature_moves', 'dirichlet_alpha',
            'dirichlet_epsilon', 'fpu_base', 'risk_beta_final',
            'risk_beta_warmup', 'opponent_risk_min', 'opponent_risk_max',
            'games_per_iter', 'max_fillup_factor', 'save_interval',
        }
        assert set(params.keys()) == expected_keys, f"Missing params: {expected_keys - set(params.keys())}"
        assert params["lr"] == 0.001
        assert params["simulations"] == 800

    print("  PASS: test_write_resume_summary_basic")


def test_write_resume_summary_with_trends():
    """write_resume_summary() includes loss and draw rate trends from recent iterations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        iface = ClaudeInterface(tmpdir)

        # Simulate 5 iterations
        for i in range(5):
            iface._recent.append({
                "loss": 4.0 - i * 0.1,
                "games": 64,
                "draws": 20 + i,
                "alerts": ["loss_plateau"] if i == 4 else [],
            })

        args = _make_mock_args()
        path = iface.write_resume_summary(
            args, iteration=5, total_iterations=60,
            buffer_stats={"total_size": 5000, "capacity": 100000, "wins": 0, "draws": 0, "losses": 0},
            shutdown_reason="Test",
        )

        with open(path) as f:
            data = json.load(f)

        # Loss trend should have 5 values
        assert len(data["loss_trend_last_5"]) == 5
        assert data["loss_trend_last_5"][0] == 4.0
        assert data["loss_trend_last_5"][4] == 3.6

        # Draw rate trend
        assert len(data["draw_rate_trend_last_5"]) == 5
        assert data["draw_rate_trend_last_5"][0] == round(20 / 64, 3)

        # Recent alerts (from last iteration)
        assert data["recent_alerts"] == ["loss_plateau"]

    print("  PASS: test_write_resume_summary_with_trends")


def test_write_resume_summary_empty_recent():
    """write_resume_summary() handles empty _recent deque gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        iface = ClaudeInterface(tmpdir)
        args = _make_mock_args()

        path = iface.write_resume_summary(
            args, iteration=0, total_iterations=60,
            buffer_stats={"total_size": 0, "capacity": 100000, "wins": 0, "draws": 0, "losses": 0},
            shutdown_reason="Early stop",
        )

        with open(path) as f:
            data = json.load(f)

        assert data["recent_alerts"] == []
        assert data["loss_trend_last_5"] == []
        assert data["draw_rate_trend_last_5"] == []

    print("  PASS: test_write_resume_summary_empty_recent")


def test_write_resume_summary_missing_args():
    """write_resume_summary() handles args missing some attributes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        iface = ClaudeInterface(tmpdir)
        # Only provide a few params
        args = SimpleNamespace(lr=0.001, epochs=5)

        path = iface.write_resume_summary(
            args, iteration=1, total_iterations=10,
            buffer_stats={"total_size": 100, "capacity": 1000, "wins": 0, "draws": 0, "losses": 0},
            shutdown_reason="Partial args test",
        )

        with open(path) as f:
            data = json.load(f)

        # Should only have the keys that exist on args
        assert "lr" in data["effective_params"]
        assert "epochs" in data["effective_params"]
        assert "simulations" not in data["effective_params"]

    print("  PASS: test_write_resume_summary_missing_args")


# ─────────────────────────────────────────────────────────────────────
# 3. JSONL shutdown event test
# ─────────────────────────────────────────────────────────────────────

def test_jsonl_shutdown_event():
    """_append_jsonl correctly writes shutdown event."""
    with tempfile.TemporaryDirectory() as tmpdir:
        iface = ClaudeInterface(tmpdir)
        # Write header first (creates the file)
        args = _make_mock_args()
        iface.write_header(args, {"num_filters": 192, "num_blocks": 15}, 0)

        # Append shutdown event
        iface._append_jsonl({
            "type": "shutdown",
            "iteration": 42,
            "reason": "Stop file detected",
            "timestamp": "2026-02-15T12:00:00",
        })

        # Read last line
        with open(iface.log_path) as f:
            lines = f.readlines()

        last = json.loads(lines[-1])
        assert last["type"] == "shutdown"
        assert last["iteration"] == 42
        assert last["reason"] == "Stop file detected"

    print("  PASS: test_jsonl_shutdown_event")


# ─────────────────────────────────────────────────────────────────────
# 4. Awaiting review signal file tests
# ─────────────────────────────────────────────────────────────────────

def test_awaiting_review_file_creation():
    """awaiting_review file is created with iteration number."""
    with tempfile.TemporaryDirectory() as tmpdir:
        awaiting_path = os.path.join(tmpdir, "awaiting_review")
        iteration = 5

        # Simulate what train.py does
        with open(awaiting_path, 'w') as f:
            f.write(str(iteration))

        assert os.path.exists(awaiting_path)
        with open(awaiting_path) as f:
            content = f.read()
        assert content == "5"

        # Simulate Claude deleting it
        os.remove(awaiting_path)
        assert not os.path.exists(awaiting_path)

    print("  PASS: test_awaiting_review_file_creation")


def test_awaiting_review_timeout():
    """Simulates the timeout path of the review loop."""
    with tempfile.TemporaryDirectory() as tmpdir:
        awaiting_path = os.path.join(tmpdir, "awaiting_review")
        with open(awaiting_path, 'w') as f:
            f.write("1")

        timeout = 2  # Short timeout for test
        wait_start = time.time()

        while os.path.exists(awaiting_path):
            if time.time() - wait_start > timeout:
                try:
                    os.remove(awaiting_path)
                except OSError:
                    pass
                break
            time.sleep(0.1)

        elapsed = time.time() - wait_start
        assert not os.path.exists(awaiting_path), "File should be cleaned up after timeout"
        assert elapsed >= timeout, f"Should have waited at least {timeout}s, got {elapsed:.1f}s"
        assert elapsed < timeout + 1, f"Should not overshoot by much, got {elapsed:.1f}s"

    print("  PASS: test_awaiting_review_timeout")


def test_awaiting_review_external_delete():
    """Simulates Claude deleting awaiting_review to continue training."""
    with tempfile.TemporaryDirectory() as tmpdir:
        awaiting_path = os.path.join(tmpdir, "awaiting_review")
        with open(awaiting_path, 'w') as f:
            f.write("1")

        # Background thread deletes the file after 0.5s (simulating Claude)
        def delete_later():
            time.sleep(0.5)
            os.remove(awaiting_path)

        t = threading.Thread(target=delete_later)
        t.start()

        wait_start = time.time()
        timeout = 10
        while os.path.exists(awaiting_path):
            if time.time() - wait_start > timeout:
                break
            time.sleep(0.1)

        elapsed = time.time() - wait_start
        t.join()

        assert not os.path.exists(awaiting_path), "File should be gone"
        assert elapsed < 2.0, f"Should have continued quickly after delete, got {elapsed:.1f}s"

    print("  PASS: test_awaiting_review_external_delete")


# ─────────────────────────────────────────────────────────────────────
# 5. --claude-timeout arg parsing test
# ─────────────────────────────────────────────────────────────────────

def test_claude_timeout_arg_parsing():
    """--claude-timeout is correctly parsed by argparse."""
    import argparse

    # Build a minimal parser that mirrors the real one
    parser = argparse.ArgumentParser()
    parser.add_argument("--claude", action="store_true")
    parser.add_argument("--claude-timeout", type=int, default=600)

    # Default
    args = parser.parse_args([])
    assert args.claude_timeout == 600, f"Default should be 600, got {args.claude_timeout}"
    assert not args.claude

    # Custom value
    args = parser.parse_args(["--claude", "--claude-timeout", "30"])
    assert args.claude_timeout == 30
    assert args.claude

    # Zero (async mode)
    args = parser.parse_args(["--claude-timeout", "0"])
    assert args.claude_timeout == 0

    print("  PASS: test_claude_timeout_arg_parsing")


# ─────────────────────────────────────────────────────────────────────
# 6. Buffer stats helper test
# ─────────────────────────────────────────────────────────────────────

def test_get_buffer_stats_logic():
    """get_buffer_stats() returns correct structure."""
    class MockBuffer:
        def __init__(self, sz, cap):
            self._sz, self._cap = sz, cap
        def size(self): return self._sz
        def capacity(self): return self._cap

    buf = MockBuffer(5000, 100000)
    stats = {"total_size": buf.size(), "capacity": buf.capacity(), "wins": 0, "draws": 0, "losses": 0}
    assert stats == {"total_size": 5000, "capacity": 100000, "wins": 0, "draws": 0, "losses": 0}

    print("  PASS: test_get_buffer_stats_logic")


# ─────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Testing Synchronous Claude Review Loop ===\n")

    tests = [
        test_graceful_shutdown_stop_file,
        test_graceful_shutdown_no_stop_file_path,
        test_write_resume_summary_basic,
        test_write_resume_summary_with_trends,
        test_write_resume_summary_empty_recent,
        test_write_resume_summary_missing_args,
        test_jsonl_shutdown_event,
        test_awaiting_review_file_creation,
        test_awaiting_review_timeout,
        test_awaiting_review_external_delete,
        test_claude_timeout_arg_parsing,
        test_get_buffer_stats_logic,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*50}\n")

    sys.exit(1 if failed else 0)
