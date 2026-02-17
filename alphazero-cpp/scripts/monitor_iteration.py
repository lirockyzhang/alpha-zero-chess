#!/usr/bin/env python3
"""Monitor script for Claude Code training integration.

Polls for `awaiting_review` in a run directory, then reads `claude_log.jsonl`
and outputs a formatted metrics summary. Exits after one iteration so Claude
Code can read the output via TaskOutput and decide what to do next.

Exit codes:
    0 - Success (metrics printed)
    1 - Error (missing dir, missing log, etc.)
    2 - Timeout (awaiting_review did not appear in time)

Usage:
    uv run python monitor_iteration.py <run_dir> [--timeout 300]
    uv run python monitor_iteration.py --latest [--timeout 300]

Workflow:
    1. Claude Code launches train.py with --claude --claude-timeout 3600
    2. Claude Code launches this script in background (Bash run_in_background)
    3. This script blocks until awaiting_review appears (or timeout)
    4. Reads claude_log.jsonl, prints metrics summary, exits
    5. Claude Code reads output, analyzes, writes param_updates.json if needed
    6. Claude Code deletes BOTH selfplay_done AND awaiting_review to resume
    7. Goto step 2
"""

import argparse
import json
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Monitor AlphaZero training for Claude Code review"
    )
    parser.add_argument(
        "run_dir", nargs="?", default=None,
        help="Path to the training run directory"
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="Auto-detect the most recent run in checkpoints/"
    )
    parser.add_argument(
        "--timeout", type=int, default=0,
        help="Max seconds to wait for awaiting_review (0=infinite)"
    )
    return parser.parse_args()


def find_latest_run(base_dir: str = "checkpoints") -> str | None:
    """Find the most recently modified run directory under base_dir."""
    if not os.path.isdir(base_dir):
        return None
    candidates = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            candidates.append((os.path.getmtime(path), path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def wait_for_review(run_dir: str, timeout: int) -> bool:
    """Poll for awaiting_review file. Returns True if found, False on timeout."""
    awaiting_path = os.path.join(run_dir, "awaiting_review")
    print(f"Monitoring {run_dir} for awaiting_review...", file=sys.stderr)
    start = time.time()
    while not os.path.exists(awaiting_path):
        time.sleep(2.0)
        if timeout > 0 and time.time() - start > timeout:
            return False
    return True


def load_jsonl(path: str) -> list[dict]:
    """Load all lines from a JSONL file, skipping malformed lines."""
    entries = []
    with open(path, "r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {i} in {path}: {e}",
                      file=sys.stderr)
    return entries


def format_summary(entries: list[dict], run_dir: str) -> str:
    """Format a metrics summary from JSONL entries."""
    lines = []

    # Separate headers/resumes from iterations
    iterations = [e for e in entries if e.get("type") == "iteration"]
    headers = [e for e in entries if e.get("type") in ("header", "resume")]

    if not iterations:
        return "No iteration data found in claude_log.jsonl"

    current = iterations[-1]
    iter_num = current["iteration"]
    total_iters = current.get("total_iterations", "?")

    lines.append(f"{'='*70}")
    lines.append(f"ITERATION {iter_num}/{total_iters} COMPLETE — AWAITING REVIEW")
    lines.append(f"{'='*70}")

    # Resume context
    resume_path = os.path.join(run_dir, "claude_resume.json")
    if headers and headers[-1].get("type") == "resume" and os.path.exists(resume_path):
        try:
            with open(resume_path) as f:
                resume = json.load(f)
            lines.append(f"\n[RESUMED] from iteration {resume.get('shutdown_iteration', '?')}"
                         f" (reason: {resume.get('shutdown_reason', 'unknown')})")
        except (json.JSONDecodeError, OSError):
            pass

    # Loss breakdown
    lines.append(f"\n--- Losses ---")
    lines.append(f"  Total:  {current.get('loss', 0):.4f}")
    lines.append(f"  Policy: {current.get('policy_loss', 0):.4f}")
    lines.append(f"  Value:  {current.get('value_loss', 0):.4f}")
    lines.append(f"  Grad norm avg/max: {current.get('grad_norm_avg', 0):.2f} / {current.get('grad_norm_max', 0):.2f}")

    # Game outcomes
    games = current.get("games", 0)
    w = current.get("white_wins", 0)
    b = current.get("black_wins", 0)
    d = current.get("draws", 0)
    draw_rate = d / games * 100 if games > 0 else 0

    lines.append(f"\n--- Game Outcomes ({games} games) ---")
    lines.append(f"  White wins: {w}  Black wins: {b}  Draws: {d}  ({draw_rate:.1f}% draw rate)")

    # Draw type breakdown
    dr = current.get("draws_repetition", 0)
    der = current.get("draws_early_repetition", 0)
    ds = current.get("draws_stalemate", 0)
    df = current.get("draws_fifty_move", 0)
    di = current.get("draws_insufficient", 0)
    dm = current.get("draws_max_moves", 0)
    if d > 0:
        lines.append(f"  Draw breakdown: repetition={dr} (early={der}), stalemate={ds}, "
                     f"50-move={df}, insufficient={di}, max-moves={dm}")

    # Game length and buffer
    lines.append(f"\n--- Quality ---")
    lines.append(f"  Avg game length: {current.get('avg_game_length', 0):.1f} plies")
    lines.append(f"  Buffer size:     {current.get('buffer_size', 0):,}")
    lines.append(f"  LR:              {current.get('lr', 0)}")
    lines.append(f"  Risk beta:       {current.get('risk_beta', 0)}")

    # Timing
    lines.append(f"\n--- Timing ---")
    lines.append(f"  Self-play: {current.get('selfplay_time', 0):.1f}s"
                 f"  Training: {current.get('train_time', 0):.1f}s"
                 f"  Total: {current.get('total_time', 0):.1f}s")

    # Alerts
    alerts = current.get("alerts", [])
    if alerts:
        lines.append(f"\n--- ALERTS ---")
        for a in alerts:
            lines.append(f"  !! {a}")
    else:
        lines.append(f"\n--- Alerts: none ---")

    # Trends (last 5 iterations)
    if len(iterations) >= 2:
        recent = iterations[-5:]
        lines.append(f"\n--- Trends (last {len(recent)} iterations) ---")

        losses = [f"{e.get('loss', 0):.4f}" for e in recent]
        lines.append(f"  Loss:      {' -> '.join(losses)}")

        draw_rates = [f"{e.get('draws', 0) / max(e.get('games', 1), 1) * 100:.0f}%" for e in recent]
        lines.append(f"  Draw rate: {' -> '.join(draw_rates)}")

        avg_lens = [f"{e.get('avg_game_length', 0):.0f}" for e in recent]
        lines.append(f"  Game len:  {' -> '.join(avg_lens)}")

    # Sample game
    sg = current.get("sample_game")
    if sg:
        lines.append(f"\n--- Sample Game ---")
        lines.append(f"  Result: {sg.get('result', '?')} ({sg.get('reason', '?')})")
        lines.append(f"  Length: {sg.get('length', '?')} plies")
        moves = sg.get("moves", "")
        if len(moves) > 200:
            moves = moves[:200] + "..."
        lines.append(f"  Moves:  {moves}")

    # Instructions
    selfplay_done_path = os.path.join(run_dir, "selfplay_done")
    awaiting_path = os.path.join(run_dir, "awaiting_review")
    lines.append(f"\n{'='*70}")
    lines.append(f"ACTION REQUIRED:")
    lines.append(f"  1. Analyze metrics above (see llm_operation_manual.md for decision guide)")
    lines.append(f"  2. If changes needed: write {os.path.join(run_dir, 'param_updates.json')}")
    lines.append(f"  3. Delete BOTH signal files to resume training:")
    lines.append(f"     rm \"{selfplay_done_path}\" \"{awaiting_path}\"")
    lines.append(f"{'='*70}")

    return "\n".join(lines)


def main():
    args = parse_args()

    # Resolve run directory
    run_dir = args.run_dir
    if args.latest:
        run_dir = find_latest_run()
        if run_dir is None:
            print("Error: no run directories found in checkpoints/", file=sys.stderr)
            sys.exit(1)
        print(f"Using latest run: {run_dir}", file=sys.stderr)
    elif run_dir is None:
        print("Error: run_dir is required (or use --latest)", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(run_dir):
        print(f"Error: run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Poll for awaiting_review
    if not wait_for_review(run_dir, args.timeout):
        print(f"Timeout: awaiting_review not found after {args.timeout}s", file=sys.stderr)
        sys.exit(2)

    # Belt-and-suspenders delay (with fsync this is mostly redundant, but safe)
    time.sleep(1.0)

    log_path = os.path.join(run_dir, "claude_log.jsonl")
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found", file=sys.stderr)
        sys.exit(1)

    entries = load_jsonl(log_path)
    print(format_summary(entries, run_dir))


if __name__ == "__main__":
    main()
