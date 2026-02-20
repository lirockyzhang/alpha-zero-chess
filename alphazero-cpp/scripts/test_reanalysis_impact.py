#!/usr/bin/env python3
"""
Reanalysis Impact Test -- Empirical A/B comparison of MCTS reanalysis.

Runs two matched training sessions (baseline vs reanalysis), evaluates
checkpoints at regular intervals, and produces a comparison report showing
whether reanalysis improves training quality enough to justify its GPU overhead.

Usage:
    # Full test (train both, evaluate, compare)
    uv run python alphazero-cpp/scripts/test_reanalysis_impact.py

    # Custom settings
    uv run python alphazero-cpp/scripts/test_reanalysis_impact.py \
        --iterations 30 --games-per-iter 20 --simulations 200 --workers 8

    # Compare existing runs (skip training)
    uv run python alphazero-cpp/scripts/test_reanalysis_impact.py \
        --compare-only --baseline-dir checkpoints/run_a --reanalysis-dir checkpoints/run_b

    # Skip evaluation (just compare training metrics)
    uv run python alphazero-cpp/scripts/test_reanalysis_impact.py --skip-eval

Requires:
    - alphazero_cpp built and importable
    - --reanalyze-fraction support in train.py (for reanalysis run)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
SEPARATOR = "=" * 70


def log(msg: str, label: str = "", flush: bool = True) -> None:
    """Print a timestamped log line. Always flushes for real-time visibility."""
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{ts}]"
    if label:
        prefix += f" [{label}]"
    print(f"{prefix} {msg}", flush=flush)


class Heartbeat:
    """Background thread that prints a heartbeat if no output for N seconds.

    Prevents the user from thinking the script is frozen during long
    subprocess runs where stdout may be buffered.
    """

    def __init__(self, interval: float = 30.0, label: str = ""):
        self._interval = interval
        self._label = label
        self._stop = threading.Event()
        self._last_activity = time.monotonic()
        self._thread: Optional[threading.Thread] = None
        self._lines = 0

    def pulse(self) -> None:
        """Call whenever there is output activity."""
        self._last_activity = time.monotonic()
        self._lines += 1

    def start(self) -> "Heartbeat":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            idle = time.monotonic() - self._last_activity
            if idle >= self._interval:
                log(
                    f"still running... ({self._lines} lines so far, "
                    f"idle {idle:.0f}s)",
                    label=self._label,
                )


# Default avg game length for estimation when no empirical data available.
# With Gumbel search, games are ~300-400 moves; 300 is conservative.
DEFAULT_AVG_GAME_LENGTH = 300


# =============================================================================
# Worker Allocation & Batch Size Recommender
# =============================================================================

def recommend_worker_allocation(
    total_workers: int,
    reanalyze_fraction: float,
    search_batch: int,
    buffer_size: int,
    games_per_iter: int,
    simulations: int,
    avg_game_length: float = DEFAULT_AVG_GAME_LENGTH,
) -> Dict[str, Any]:
    """Compute optimal self-play vs reanalysis worker split and eval_batch.

    MODEL SUMMARY
    =============
    All workers (self-play and reanalysis) share one GPU via the
    GlobalEvaluationQueue.  The GPU is the throughput bottleneck, so the
    worker split is a pure load-balancing problem: both groups should finish
    at the same time.

    Per-move, a self-play worker does N MCTS simulations (each ~1 NN eval),
    plus ~1ms of CPU game logic.  A reanalysis worker does the same N sims
    on a buffer-sampled position, with ~0.3ms of CPU overhead.  Since the
    GPU dominates (N * t_eval >> t_cpu), the CPU difference is <1% and we
    can ignore it.

    The balanced-completion condition gives:

        W_ra / W_sp = f / (1-f)       =>    W_ra = f * W,  W_sp = (1-f) * W

    BATCH SIZE
    ==========
    train.py auto-computes:  eval_batch = ceil(W * S / 32) * 32

    The CUDA graph large-threshold is 87.5% of eval_batch.  If eval_batch
    is much larger than W*S, real batches never reach the large graph tier,
    so oversizing HURTS.  Conversely, undersizing forces multiple GPU passes
    per MCTS step, adding worker wait time.

    Recommendation:  eval_batch = ceil(W_total * S / 32) * 32
    (This is what train.py already does when --workers is set correctly.)

    BUFFER FRESHNESS
    ================
    Reanalysis value comes from the KL divergence between the old network
    (that generated the data) and the current one.  If the buffer only holds
    ~1 iteration of data, the "old" network is almost identical to the
    current one => near-zero KL => wasted GPU cycles.

    buffer_iterations = buffer_size / (games_per_iter * avg_game_length)

    Reanalysis becomes worthwhile at buffer_iterations >= 3 (enough staleness
    to produce meaningful policy improvement).

    GPU OVERHEAD
    ============
    Total NN evals per iteration scale as 1/(1-f):
      - f=0.00:  1.00x  (baseline)
      - f=0.10:  1.11x
      - f=0.25:  1.33x
      - f=0.50:  2.00x

    This is pure GPU overhead.  Wall-clock increase may be less if the GPU
    was not 100% utilized (workers had CPU gaps).

    Args:
        total_workers: Total worker count (W_sp + W_ra).
        reanalyze_fraction: Target fraction of training batch from reanalysis.
        search_batch: Leaves per MCTS step per worker (gumbel_top_k for Gumbel).
        buffer_size: Replay buffer capacity in positions.
        games_per_iter: Self-play games per training iteration.
        simulations: MCTS simulations per move.
        avg_game_length: Estimated average game length in moves.

    Returns:
        Dict with keys:
            w_selfplay, w_reanalysis: Recommended worker counts.
            eval_batch: Recommended GPU batch size.
            gpu_overhead: Multiplicative GPU overhead factor (1/(1-f)).
            buffer_iterations: How many iterations of data the buffer holds.
            buffer_sufficient: Whether the buffer is large enough for reanalysis.
            nn_evals_selfplay: NN evals from self-play per iteration.
            nn_evals_reanalysis: NN evals from reanalysis per iteration.
            nn_evals_total: Total NN evals per iteration.
            notes: List of advisory strings.
    """
    f = reanalyze_fraction
    W = total_workers
    S = search_batch
    notes: List[str] = []

    # -- Worker split (load-balanced) --
    if f <= 0:
        w_sp, w_ra = W, 0
    elif f >= 1.0:
        # Degenerate: all reanalysis, no new games.  Not useful.
        w_sp, w_ra = 1, max(0, W - 1)
        notes.append("WARNING: fraction >= 1.0 means no new self-play data.")
    else:
        w_sp = max(1, round((1 - f) * W))
        w_ra = W - w_sp

    # Ensure at least 1 self-play worker
    if w_sp < 1:
        w_sp = 1
        w_ra = W - 1

    # -- Eval batch size --
    w_total = w_sp + w_ra
    eval_batch = ((w_total * S + 31) // 32) * 32
    eval_batch = max(eval_batch, 32)

    # -- Buffer freshness --
    positions_per_iter = games_per_iter * avg_game_length
    buffer_iters = buffer_size / positions_per_iter if positions_per_iter > 0 else 0

    buffer_sufficient = buffer_iters >= 3
    if w_ra > 0 and not buffer_sufficient:
        notes.append(
            f"Buffer holds only {buffer_iters:.1f} iterations of data. "
            f"Reanalysis needs >= 3 iterations for meaningful KL divergence. "
            f"Consider buffer_size >= {int(3 * positions_per_iter)}."
        )

    # -- NN eval budget --
    nn_selfplay = games_per_iter * avg_game_length * simulations
    nn_reanalysis = nn_selfplay * f / (1 - f) if f < 1 else 0
    nn_total = nn_selfplay + nn_reanalysis
    gpu_overhead = 1.0 / (1.0 - f) if f < 1 else float("inf")

    # -- Batch fill analysis --
    leaves_per_step = w_total * S
    if leaves_per_step > eval_batch:
        notes.append(
            f"Workers generate {leaves_per_step} leaves/step but eval_batch={eval_batch}. "
            f"Some workers will queue across 2+ GPU passes."
        )
    fill_pct = leaves_per_step / eval_batch * 100 if eval_batch > 0 else 0
    large_threshold_pct = 87.5  # CUDA graph large-graph fill requirement
    if fill_pct < large_threshold_pct:
        notes.append(
            f"Estimated batch fill {fill_pct:.0f}% < {large_threshold_pct:.0f}% "
            f"large-graph threshold. Batches may fall to smaller CUDA graph tier."
        )

    return {
        "w_selfplay": w_sp,
        "w_reanalysis": w_ra,
        "eval_batch": eval_batch,
        "gpu_overhead": gpu_overhead,
        "buffer_iterations": buffer_iters,
        "buffer_sufficient": buffer_sufficient,
        "nn_evals_selfplay": int(nn_selfplay),
        "nn_evals_reanalysis": int(nn_reanalysis),
        "nn_evals_total": int(nn_total),
        "leaves_per_step": leaves_per_step,
        "batch_fill_pct": fill_pct,
        "notes": notes,
    }


def print_recommendation(rec: Dict[str, Any], total_workers: int, f: float) -> None:
    """Print a formatted worker allocation recommendation."""
    print(f"\n-- Worker Allocation Recommendation {'-' * 34}")
    print(f"""
  Inputs:
    Total workers:       {total_workers}
    Reanalyze fraction:  {f}
    Search batch:        {rec['leaves_per_step'] // total_workers if total_workers else 0}

  Worker Split (balanced-completion):
    Self-play workers:   {rec['w_selfplay']:>4}   ({rec['w_selfplay']/total_workers*100:.0f}%)
    Reanalysis workers:  {rec['w_reanalysis']:>4}   ({rec['w_reanalysis']/total_workers*100:.0f}%)

  GPU Batch:
    Recommended eval_batch: {rec['eval_batch']}
    Leaves per MCTS step:   {rec['leaves_per_step']}
    Expected batch fill:    {rec['batch_fill_pct']:.0f}%

  NN Eval Budget (per iteration):
    Self-play evals:     {rec['nn_evals_selfplay']:>12,}
    Reanalysis evals:    {rec['nn_evals_reanalysis']:>12,}
    Total:               {rec['nn_evals_total']:>12,}
    GPU overhead:        {rec['gpu_overhead']:.2f}x

  Buffer Freshness:
    Buffer holds:        {rec['buffer_iterations']:.1f} iterations of data
    Sufficient:          {'Yes' if rec['buffer_sufficient'] else 'NO -- need >= 3 iterations'}""")

    if rec["notes"]:
        print("\n  Advisories:")
        for note in rec["notes"]:
            print(f"    * {note}")

    # Sensitivity table: show allocation across fraction values
    print(f"""
  Sensitivity (workers={total_workers}):
    Fraction   W_sp   W_ra   GPU overhead   Eval budget ratio
    --------   ----   ----   ------------   -----------------""")
    for test_f in [0.0, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50]:
        if test_f >= 1.0:
            continue
        tsp = max(1, round((1 - test_f) * total_workers))
        tra = total_workers - tsp
        overhead = 1.0 / (1.0 - test_f)
        marker = "  <--" if abs(test_f - f) < 0.001 else ""
        print(f"    {test_f:<10.2f} {tsp:>4}   {tra:>4}   {overhead:>11.2f}x   {overhead:>16.2f}x{marker}")

    print()


# =============================================================================
# Training Subprocess
# =============================================================================

def run_training(args: argparse.Namespace, fraction: float, label: str) -> str:
    """Run train.py as a subprocess and return the run directory path.

    Streams stdout with a [label] prefix so both runs are distinguishable.
    Parses the "Run directory:" line from train.py output to locate artifacts.

    Args:
        args: Parsed CLI arguments (iterations, games_per_iter, etc.)
        fraction: Reanalyze fraction (0 = baseline, >0 = reanalysis)
        label: Display label for log prefix (e.g., "Baseline", "Reanalysis")

    Returns:
        Absolute path to the run directory created by train.py.

    Raises:
        RuntimeError: If train.py fails or run directory cannot be detected.
    """
    # Use -u to force unbuffered stdout in the child process.
    # Without this, Windows pipes cause full buffering and we see nothing
    # until the child's 4KB buffer fills or the process exits.
    cmd = [
        sys.executable, "-u", str(SCRIPTS_DIR / "train.py"),
        "--iterations", str(args.iterations),
        "--games-per-iter", str(args.games_per_iter),
        "--simulations", str(args.simulations),
        "--workers", str(args.workers),
        "--no-visualization",
        "--no-sample-games",
    ]

    # Only pass --reanalyze-fraction for non-zero values.
    # fraction=0 is the default behavior (no reanalysis), so omitting the arg
    # lets the baseline run work even before the feature is added to train.py.
    if fraction > 0:
        cmd.extend(["--reanalyze-fraction", str(fraction)])

    log(f"Starting training: fraction={fraction}", label=label)
    log(f"Command: {' '.join(cmd)}", label=label)
    print(flush=True)

    run_dir = None
    start_time = time.monotonic()
    hb = Heartbeat(interval=30.0, label=label).start()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line-buffered (works with -u on child)
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    try:
        # Use readline() instead of iteration.  On Windows, Python's
        # ``for line in proc.stdout`` uses internal read-ahead buffering
        # (~8 KB) regardless of bufsize, which freezes visible output
        # until the buffer fills.  readline() returns immediately when a
        # full line is available.
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.rstrip("\n")
            hb.pulse()
            print(f"  [{label}] {line}", flush=True)

            # Parse run directory from train.py output.
            # train.py prints it in several formats; match liberally.
            if "Run directory:" in line:
                # Extract path after "Run directory:" with variable whitespace
                path = line.split("Run directory:")[-1].strip()
                if path and os.path.isdir(path):
                    run_dir = path
    finally:
        hb.stop()
        proc.wait()

    elapsed = time.monotonic() - start_time

    if proc.returncode != 0:
        log(f"FAILED after {elapsed:.0f}s (exit code {proc.returncode})", label=label)
        raise RuntimeError(
            f"[{label}] train.py exited with code {proc.returncode}"
        )

    if run_dir is None:
        log(f"FAILED: could not detect run directory from output", label=label)
        raise RuntimeError(
            f"[{label}] Could not detect run directory from train.py output"
        )

    log(f"Completed in {elapsed:.0f}s. Run dir: {run_dir}", label=label)
    return run_dir


# =============================================================================
# Metrics Loading
# =============================================================================

def load_metrics(run_dir: str) -> List[Dict[str, Any]]:
    """Load per-iteration metrics from training_log.jsonl (or legacy JSON fallback).

    Args:
        run_dir: Path to a training run directory.

    Returns:
        List of per-iteration metric dicts, sorted by iteration number.

    Raises:
        FileNotFoundError: If neither JSONL nor JSON file is found.
    """
    # Try JSONL first
    jsonl_path = os.path.join(run_dir, "training_log.jsonl")
    if os.path.exists(jsonl_path):
        iterations = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("type") == "iteration":
                        iterations.append(rec)
                except (json.JSONDecodeError, ValueError):
                    continue
        iterations.sort(key=lambda m: m.get("iteration", 0))
        return iterations

    # Fallback: legacy JSON
    metrics_path = os.path.join(run_dir, "training_metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found in: {run_dir}")

    with open(metrics_path, "r") as f:
        data = json.load(f)

    iterations = data.get("iterations", [])
    iterations.sort(key=lambda m: m.get("iteration", 0))
    return iterations


# =============================================================================
# Checkpoint Evaluation
# =============================================================================

def evaluate_checkpoints(
    run_dir: str,
    iterations: List[int],
    eval_sims: int,
) -> Dict[int, Dict[str, Any]]:
    """Evaluate checkpoints at specified iterations using evaluation.py.

    Runs evaluation.py --evaluators vs_random for each checkpoint and parses
    the JSON output.

    Args:
        run_dir: Path to the training run directory.
        iterations: Which iteration checkpoints to evaluate.
        eval_sims: MCTS simulations for evaluation games.

    Returns:
        Dict mapping iteration number -> evaluation result dict.
        Each result has keys: score, wins, losses, draws, games.
    """
    results = {}

    for it in iterations:
        checkpoint_path = os.path.join(run_dir, f"model_iter_{it:03d}.pt")
        if not os.path.exists(checkpoint_path):
            log(f"WARNING: Checkpoint not found: {checkpoint_path}, skipping")
            continue

        # Write results to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                sys.executable, "-u", str(SCRIPTS_DIR / "evaluation.py"),
                "--checkpoint", checkpoint_path,
                "--evaluators", "vs_random",
                "--simulations", str(eval_sims),
                "--output", tmp_path,
            ]

            log(f"Evaluating iter {it:03d}...", label="Eval")
            eval_start = time.monotonic()
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per checkpoint
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            eval_elapsed = time.monotonic() - eval_start

            if proc.returncode != 0:
                log(f"  iter {it:03d} FAILED ({eval_elapsed:.0f}s, exit code {proc.returncode})", label="Eval")
                if proc.stderr:
                    for line in proc.stderr.strip().split("\n")[:5]:
                        print(f"    {line}", flush=True)
                continue

            # Parse the evaluation output JSON
            with open(tmp_path, "r") as f:
                eval_data = json.load(f)

            eval_results = eval_data.get("results", [])
            if eval_results:
                r = eval_results[0]
                details = r.get("details", {})
                results[it] = {
                    "score": r.get("score", 0.0),
                    "wins": details.get("wins", 0),
                    "losses": details.get("losses", 0),
                    "draws": details.get("draws", 0),
                    "games": details.get("games", 0),
                }
                w, g = results[it]["wins"], results[it]["games"]
                log(f"  iter {it:03d}: {r.get('score', 0) * 100:.0f}% ({w}/{g}) [{eval_elapsed:.0f}s]", label="Eval")
            else:
                log(f"  iter {it:03d}: no results [{eval_elapsed:.0f}s]", label="Eval")

        except subprocess.TimeoutExpired:
            log(f"  iter {it:03d}: TIMEOUT (600s)", label="Eval")
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            log(f"  iter {it:03d}: parse error: {e}", label="Eval")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return results


# =============================================================================
# Comparison Computation
# =============================================================================

def _safe_get(metrics: List[Dict], key: str, default: float = 0.0) -> List[float]:
    """Extract a numeric field from all iterations, using default for missing."""
    return [m.get(key, default) for m in metrics]


def _decisive_rate(metrics: List[Dict]) -> float:
    """Compute overall decisive game rate across all iterations."""
    total_games = sum(m.get("games", 0) for m in metrics)
    if total_games == 0:
        return 0.0
    decisive = sum(m.get("white_wins", 0) + m.get("black_wins", 0) for m in metrics)
    return decisive / total_games * 100


def _fifty_move_rate(metrics: List[Dict]) -> float:
    """Compute overall fifty-move draw rate across all iterations."""
    total_games = sum(m.get("games", 0) for m in metrics)
    if total_games == 0:
        return 0.0
    fifty = sum(m.get("draws_fifty_move", 0) for m in metrics)
    return fifty / total_games * 100


def _avg_game_length(metrics: List[Dict]) -> float:
    """Compute weighted average game length across all iterations."""
    total_moves = sum(
        m.get("avg_game_length", 0) * m.get("games", 1) for m in metrics
    )
    total_games = sum(m.get("games", 0) for m in metrics)
    if total_games == 0:
        return 0.0
    return total_moves / total_games


def _total_wall_clock(metrics: List[Dict]) -> float:
    """Total wall-clock time in seconds across all iterations."""
    return sum(m.get("total_time", 0) for m in metrics)


def _loss_per_minute(metrics: List[Dict]) -> float:
    """Loss reduction per minute: (first_loss - final_loss) / total_minutes."""
    if len(metrics) < 2:
        return 0.0
    first_loss = metrics[0].get("loss", 0)
    final_loss = metrics[-1].get("loss", 0)
    total_min = _total_wall_clock(metrics) / 60.0
    if total_min <= 0:
        return 0.0
    return (first_loss - final_loss) / total_min


def _delta_pct(baseline: float, reanalysis: float) -> str:
    """Compute percentage delta string: '+12.3%' or '-5.1%'."""
    if baseline == 0:
        return "N/A"
    delta = (reanalysis - baseline) / abs(baseline) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def _delta_pp(baseline_pct: float, reanalysis_pct: float) -> str:
    """Compute percentage-point delta string: '+10.0pp' or '-5.0pp'."""
    delta = reanalysis_pct - baseline_pct
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}pp"


def compute_comparison(
    baseline_metrics: List[Dict],
    rean_metrics: List[Dict],
) -> Dict[str, Any]:
    """Compute all comparison values between baseline and reanalysis runs.

    Returns a dict with sections: loss_curves, game_quality, compute, reanalysis_diag.
    """
    comparison: Dict[str, Any] = {}

    # --- Loss curves at matching iterations ---
    base_by_iter = {m["iteration"]: m for m in baseline_metrics}
    rean_by_iter = {m["iteration"]: m for m in rean_metrics}
    common_iters = sorted(set(base_by_iter) & set(rean_by_iter))

    loss_curves = []
    for it in common_iters:
        b, r = base_by_iter[it], rean_by_iter[it]
        bl, rl = b.get("loss", 0), r.get("loss", 0)
        loss_curves.append({
            "iteration": it,
            "baseline_loss": bl,
            "baseline_policy": b.get("policy_loss", 0),
            "baseline_value": b.get("value_loss", 0),
            "reanalysis_loss": rl,
            "reanalysis_policy": r.get("policy_loss", 0),
            "reanalysis_value": r.get("value_loss", 0),
            "delta_pct": _delta_pct(bl, rl),
        })
    comparison["loss_curves"] = loss_curves

    # --- Game quality ---
    b_decisive = _decisive_rate(baseline_metrics)
    r_decisive = _decisive_rate(rean_metrics)
    b_fifty = _fifty_move_rate(baseline_metrics)
    r_fifty = _fifty_move_rate(rean_metrics)
    b_avg_len = _avg_game_length(baseline_metrics)
    r_avg_len = _avg_game_length(rean_metrics)

    comparison["game_quality"] = {
        "decisive_rate": {"baseline": b_decisive, "reanalysis": r_decisive,
                          "delta": _delta_pp(b_decisive, r_decisive)},
        "avg_game_length": {"baseline": b_avg_len, "reanalysis": r_avg_len,
                            "delta": _delta_pct(b_avg_len, r_avg_len)},
        "fifty_move_rate": {"baseline": b_fifty, "reanalysis": r_fifty,
                            "delta": _delta_pp(b_fifty, r_fifty)},
    }

    # --- Compute efficiency ---
    b_wall = _total_wall_clock(baseline_metrics)
    r_wall = _total_wall_clock(rean_metrics)
    b_lpm = _loss_per_minute(baseline_metrics)
    r_lpm = _loss_per_minute(rean_metrics)
    n_base = len(baseline_metrics)
    n_rean = len(rean_metrics)

    comparison["compute"] = {
        "total_wall_s": {"baseline": b_wall, "reanalysis": r_wall,
                         "delta": _delta_pct(b_wall, r_wall)},
        "avg_iter_time": {
            "baseline": b_wall / n_base if n_base else 0,
            "reanalysis": r_wall / n_rean if n_rean else 0,
            "delta": _delta_pct(
                b_wall / n_base if n_base else 1,
                r_wall / n_rean if n_rean else 1,
            ),
        },
        "loss_per_minute": {"baseline": b_lpm, "reanalysis": r_lpm,
                            "delta": _delta_pct(b_lpm, r_lpm)},
    }

    # --- Final losses ---
    comparison["final_loss"] = {
        "baseline": baseline_metrics[-1].get("loss", 0) if baseline_metrics else 0,
        "reanalysis": rean_metrics[-1].get("loss", 0) if rean_metrics else 0,
    }
    fl_b = comparison["final_loss"]["baseline"]
    fl_r = comparison["final_loss"]["reanalysis"]
    comparison["final_loss"]["delta"] = _delta_pct(fl_b, fl_r)

    # --- Reanalysis diagnostics (fields may not exist yet) ---
    rean_diag = []
    for m in rean_metrics:
        positions = m.get("reanalysis_positions")
        if positions is not None:
            rean_diag.append({
                "iteration": m["iteration"],
                "positions": positions,
                "mean_kl": m.get("reanalysis_mean_kl", 0),
                "tail_time_s": m.get("reanalysis_tail_time_s", 0),
                "overhead_pct": (
                    m.get("reanalysis_tail_time_s", 0) / m.get("selfplay_time", 1) * 100
                    if m.get("selfplay_time", 0) > 0 else 0
                ),
            })
    comparison["reanalysis_diag"] = rean_diag

    return comparison


# =============================================================================
# Report Printing
# =============================================================================

def _fmt_time(seconds: float) -> str:
    """Format seconds as 'X.Y min' or 'X.Ys'."""
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f}s"


def print_report(
    comparison: Dict[str, Any],
    baseline_evals: Dict[int, Dict],
    rean_evals: Dict[int, Dict],
    args: argparse.Namespace,
    baseline_dir: str,
    reanalysis_dir: str,
) -> None:
    """Print the formatted comparison report to stdout."""
    print(f"\n{SEPARATOR}")
    print("REANALYSIS IMPACT TEST -- COMPARISON")
    print(SEPARATOR)

    # Configuration
    fraction = getattr(args, "reanalyze_fraction", 0.25)
    print(f"""
Configuration:
  Iterations: {args.iterations}, Games/iter: {args.games_per_iter}, Sims: {args.simulations}, Workers: {args.workers}
  Reanalyze fraction: {fraction}
  Baseline:    {baseline_dir}
  Reanalysis:  {reanalysis_dir}""")

    # Loss curves
    loss_curves = comparison.get("loss_curves", [])
    if loss_curves:
        print(f"\n-- Loss Curves {'-' * 55}")
        print(f"  {'Iter':>6}    {'Baseline':>24}  {'Reanalysis':>24}  {'Delta':>8}")
        print(f"  {'----':>6}    {'--------':>24}  {'----------':>24}  {'-----':>8}")
        for lc in loss_curves:
            bl = lc["baseline_loss"]
            bp = lc["baseline_policy"]
            bv = lc["baseline_value"]
            rl = lc["reanalysis_loss"]
            rp = lc["reanalysis_policy"]
            rv = lc["reanalysis_value"]
            b_str = f"{bl:.2f} (p={bp:.2f} v={bv:.2f})"
            r_str = f"{rl:.2f} (p={rp:.2f} v={rv:.2f})"
            print(f"  {lc['iteration']:>6}    {b_str:>24}  {r_str:>24}  {lc['delta_pct']:>8}")

    # Game quality
    gq = comparison.get("game_quality", {})
    if gq:
        print(f"\n-- Game Quality {'-' * 54}")
        print(f"  {'Metric':<22}{'Baseline':>12}{'Reanalysis':>14}{'Delta':>10}")
        print(f"  {'------':<22}{'--------':>12}{'----------':>14}{'-----':>10}")

        dr = gq.get("decisive_rate", {})
        print(f"  {'Decisive rate':<22}{dr.get('baseline', 0):>11.1f}%{dr.get('reanalysis', 0):>13.1f}%{dr.get('delta', 'N/A'):>10}")

        al = gq.get("avg_game_length", {})
        print(f"  {'Avg game length':<22}{al.get('baseline', 0):>12.1f}{al.get('reanalysis', 0):>14.1f}{al.get('delta', 'N/A'):>10}")

        fm = gq.get("fifty_move_rate", {})
        print(f"  {'Fifty-move draws':<22}{fm.get('baseline', 0):>11.1f}%{fm.get('reanalysis', 0):>13.1f}%{fm.get('delta', 'N/A'):>10}")

    # Compute efficiency
    comp = comparison.get("compute", {})
    if comp:
        print(f"\n-- Compute Efficiency {'-' * 48}")
        print(f"  {'Metric':<22}{'Baseline':>12}{'Reanalysis':>14}{'Delta':>10}")
        print(f"  {'------':<22}{'--------':>12}{'----------':>14}{'-----':>10}")

        tw = comp.get("total_wall_s", {})
        b_wall_str = _fmt_time(tw.get("baseline", 0))
        r_wall_str = _fmt_time(tw.get("reanalysis", 0))
        print(f"  {'Total wall-clock':<22}{b_wall_str:>12}{r_wall_str:>14}{tw.get('delta', 'N/A'):>10}")

        ai = comp.get("avg_iter_time", {})
        b_ai_str = f"{ai.get('baseline', 0):.1f}s"
        r_ai_str = f"{ai.get('reanalysis', 0):.1f}s"
        print(f"  {'Avg iter time':<22}{b_ai_str:>12}{r_ai_str:>14}{ai.get('delta', 'N/A'):>10}")

        lpm = comp.get("loss_per_minute", {})
        print(f"  {'Final loss/minute':<22}{lpm.get('baseline', 0):>12.4f}{lpm.get('reanalysis', 0):>14.4f}{lpm.get('delta', 'N/A'):>10}")

    # Evaluation results
    if baseline_evals or rean_evals:
        print(f"\n-- Evaluation: vs_random {'-' * 46}")
        all_iters = sorted(set(baseline_evals.keys()) | set(rean_evals.keys()))
        print(f"  {'Checkpoint':<14}{'Baseline':>18}{'Reanalysis':>18}")
        print(f"  {'----------':<14}{'--------':>18}{'----------':>18}")

        for it in all_iters:
            b_eval = baseline_evals.get(it)
            r_eval = rean_evals.get(it)
            if b_eval:
                b_str = f"{b_eval['score'] * 100:.0f}% ({b_eval['wins']}/{b_eval['games']})"
            else:
                b_str = "--"
            if r_eval:
                r_str = f"{r_eval['score'] * 100:.0f}% ({r_eval['wins']}/{r_eval['games']})"
            else:
                r_str = "--"
            print(f"  iter_{it:03d}     {b_str:>18}{r_str:>18}")

    # Reanalysis diagnostics
    rean_diag = comparison.get("reanalysis_diag", [])
    if rean_diag:
        print(f"\n-- Reanalysis Diagnostics {'-' * 44}")
        print(f"  {'Iter':>6}    {'Positions':>10}    {'Mean KL':>8}    {'Tail (s)':>9}    {'Overhead':>9}")
        print(f"  {'----':>6}    {'---------':>10}    {'-------':>8}    {'--------':>9}    {'--------':>9}")
        for d in rean_diag:
            print(
                f"  {d['iteration']:>6}    {d['positions']:>10}    "
                f"{d['mean_kl']:>8.2f}    {d['tail_time_s']:>9.1f}    "
                f"{d['overhead_pct']:>8.1f}%"
            )

    # Verdict
    fl = comparison.get("final_loss", {})
    fl_b = fl.get("baseline", 0)
    fl_r = fl.get("reanalysis", 0)
    fl_delta = fl.get("delta", "N/A")

    lpm = comp.get("loss_per_minute", {}) if comp else {}
    lpm_delta = lpm.get("delta", "N/A")

    tw = comp.get("total_wall_s", {}) if comp else {}
    tw_delta = tw.get("delta", "N/A")

    # Determine recommendation
    loss_better = fl_r < fl_b if fl_b > 0 else False
    efficiency_str = lpm_delta

    print(f"\n-- Verdict {'-' * 59}")
    print(f"  Sample efficiency:  Reanalysis achieves {fl_delta} final loss")
    print(f"  Compute efficiency: Reanalysis achieves {efficiency_str} loss reduction per minute")

    if loss_better:
        print(f"  Recommendation:     Reanalysis provides net benefit despite {tw_delta} time overhead")
    else:
        print(f"  Recommendation:     Reanalysis does NOT provide net benefit ({tw_delta} time overhead)")

    print()


# =============================================================================
# Results Persistence
# =============================================================================

def save_results(
    comparison: Dict[str, Any],
    baseline_evals: Dict[int, Dict],
    rean_evals: Dict[int, Dict],
    args: argparse.Namespace,
    baseline_dir: str,
    reanalysis_dir: str,
    output_path: str,
) -> None:
    """Save the full comparison as a JSON file."""
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "iterations": args.iterations,
            "games_per_iter": args.games_per_iter,
            "simulations": args.simulations,
            "workers": args.workers,
            "reanalyze_fraction": args.reanalyze_fraction,
            "eval_interval": args.eval_interval,
            "eval_sims": args.eval_sims,
        },
        "baseline_dir": baseline_dir,
        "reanalysis_dir": reanalysis_dir,
        "comparison": comparison,
        "baseline_evals": {str(k): v for k, v in baseline_evals.items()},
        "reanalysis_evals": {str(k): v for k, v in rean_evals.items()},
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Full results saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Empirical A/B test: baseline vs. MCTS reanalysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test (~5 min)
  uv run python alphazero-cpp/scripts/test_reanalysis_impact.py \\
      --iterations 3 --games-per-iter 10 --simulations 50 --workers 4 --skip-eval

  # Compare existing runs
  uv run python alphazero-cpp/scripts/test_reanalysis_impact.py \\
      --compare-only --baseline-dir checkpoints/run_a --reanalysis-dir checkpoints/run_b
        """,
    )

    # Training parameters
    parser.add_argument("--iterations", type=int, default=20,
                        help="Training iterations per run (default: 20)")
    parser.add_argument("--games-per-iter", type=int, default=20,
                        help="Self-play games per iteration (default: 20)")
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS sims per move (default: 100)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel self-play workers (default: 4)")
    parser.add_argument("--reanalyze-fraction", type=float, default=0.25,
                        help="Fraction for the reanalysis run (default: 0.25)")

    # Evaluation parameters
    parser.add_argument("--eval-interval", type=int, default=5,
                        help="Evaluate checkpoints every N iterations (default: 5)")
    parser.add_argument("--eval-sims", type=int, default=200,
                        help="MCTS sims for evaluation games (default: 200)")

    # Mode flags
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip training, just compare existing runs")
    parser.add_argument("--baseline-dir", type=str, default=None,
                        help="Existing baseline run directory (for --compare-only)")
    parser.add_argument("--reanalysis-dir", type=str, default=None,
                        help="Existing reanalysis run directory (for --compare-only)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip checkpoint evaluation")
    parser.add_argument("--output", type=str, default=None,
                        help="Save full comparison JSON to this path")

    # Recommender mode
    parser.add_argument("--recommend", action="store_true",
                        help="Print optimal worker allocation and exit")
    parser.add_argument("--search-batch", type=int, default=16,
                        help="Leaves per MCTS step per worker (default: 16)")
    parser.add_argument("--buffer-size", type=int, default=100000,
                        help="Replay buffer capacity (default: 100000)")
    parser.add_argument("--avg-game-length", type=float, default=DEFAULT_AVG_GAME_LENGTH,
                        help=f"Estimated avg game length (default: {DEFAULT_AVG_GAME_LENGTH})")

    args = parser.parse_args()

    # Validate --compare-only requirements
    if args.compare_only:
        if not args.baseline_dir or not args.reanalysis_dir:
            parser.error("--compare-only requires both --baseline-dir and --reanalysis-dir")
        if not os.path.isdir(args.baseline_dir):
            parser.error(f"Baseline directory not found: {args.baseline_dir}")
        if not os.path.isdir(args.reanalysis_dir):
            parser.error(f"Reanalysis directory not found: {args.reanalysis_dir}")

    # --recommend mode: print allocation table and exit
    if args.recommend:
        print(SEPARATOR)
        print("REANALYSIS WORKER ALLOCATION RECOMMENDER")
        print(SEPARATOR)
        rec = recommend_worker_allocation(
            total_workers=args.workers,
            reanalyze_fraction=args.reanalyze_fraction,
            search_batch=args.search_batch,
            buffer_size=args.buffer_size,
            games_per_iter=args.games_per_iter,
            simulations=args.simulations,
            avg_game_length=args.avg_game_length,
        )
        print_recommendation(rec, args.workers, args.reanalyze_fraction)
        return

    print(SEPARATOR)
    print("REANALYSIS IMPACT TEST")
    print(SEPARATOR)
    print(f"  Iterations:         {args.iterations}")
    print(f"  Games/iter:         {args.games_per_iter}")
    print(f"  Simulations:        {args.simulations}")
    print(f"  Workers:            {args.workers}")
    print(f"  Reanalyze fraction: {args.reanalyze_fraction}")
    print(f"  Eval interval:      every {args.eval_interval} iters")
    print(f"  Eval sims:          {args.eval_sims}")
    print(f"  Compare only:       {args.compare_only}")
    print(f"  Skip eval:          {args.skip_eval}")
    print()

    # -- Step 1: Training --------------------------------------------------
    if args.compare_only:
        baseline_dir = args.baseline_dir
        reanalysis_dir = args.reanalysis_dir
        log("Using existing runs:")
        log(f"  Baseline:    {baseline_dir}")
        log(f"  Reanalysis:  {reanalysis_dir}")
    else:
        log("Step 1/5: Training both runs sequentially")
        overall_start = time.monotonic()

        # Run baseline (no reanalysis)
        log("Starting BASELINE run (fraction=0)...")
        baseline_dir = run_training(args, fraction=0.0, label="Baseline")

        # Run reanalysis
        log(f"Starting REANALYSIS run (fraction={args.reanalyze_fraction})...")
        reanalysis_dir = run_training(
            args, fraction=args.reanalyze_fraction, label="Reanalysis"
        )

        elapsed = time.monotonic() - overall_start
        log(f"Both training runs completed in {_fmt_time(elapsed)}")

    # -- Step 2: Load metrics ----------------------------------------------
    log("Step 2/5: Loading training metrics")

    try:
        baseline_metrics = load_metrics(baseline_dir)
        log(f"  Baseline:    {len(baseline_metrics)} iterations loaded")
    except FileNotFoundError as e:
        log(f"  ERROR: {e}")
        sys.exit(1)

    try:
        rean_metrics = load_metrics(reanalysis_dir)
        log(f"  Reanalysis:  {len(rean_metrics)} iterations loaded")
    except FileNotFoundError as e:
        log(f"  ERROR: {e}")
        sys.exit(1)

    # -- Step 3: Evaluate checkpoints --------------------------------------
    baseline_evals: Dict[int, Dict] = {}
    rean_evals: Dict[int, Dict] = {}

    if not args.skip_eval:
        # Determine which iterations to evaluate
        eval_iters = list(range(args.eval_interval, args.iterations + 1, args.eval_interval))
        # Always include the final iteration
        if args.iterations not in eval_iters:
            eval_iters.append(args.iterations)

        log(f"Step 3/5: Evaluating checkpoints at iterations: {eval_iters}")

        log(f"  Baseline evaluations ({baseline_dir}):")
        baseline_evals = evaluate_checkpoints(
            baseline_dir, eval_iters, args.eval_sims
        )

        log(f"  Reanalysis evaluations ({reanalysis_dir}):")
        rean_evals = evaluate_checkpoints(
            reanalysis_dir, eval_iters, args.eval_sims
        )
    else:
        log("Step 3/5: Skipping checkpoint evaluation (--skip-eval)")

    # -- Step 4: Compute comparison ----------------------------------------
    log("Step 4/5: Computing comparison metrics")
    comparison = compute_comparison(baseline_metrics, rean_metrics)

    # -- Step 5: Print report ----------------------------------------------
    log("Step 5/5: Generating report")
    print_report(
        comparison, baseline_evals, rean_evals,
        args, baseline_dir, reanalysis_dir,
    )

    # -- Save results ------------------------------------------------------
    if args.output:
        save_results(
            comparison, baseline_evals, rean_evals,
            args, baseline_dir, reanalysis_dir, args.output,
        )

    log("Done.")


if __name__ == "__main__":
    main()
