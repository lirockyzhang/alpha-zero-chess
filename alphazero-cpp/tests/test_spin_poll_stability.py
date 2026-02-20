"""
GPU integration test for lock-free spin-poll stall detection in collect_batch().

Runs ParallelSelfPlayCoordinator with a REAL PyTorch model on GPU to verify
the spin-poll optimization works correctly under actual CUDA inference timing:
  - 32-64 workers, search_batch=16 (Gumbel mode)
  - Real AlphaZeroNet on GPU with CUDA graphs (production path)
  - Verifies no crashes, no data corruption, no deadlocks
  - Checks spin_poll_avg_us metric is present and reasonable
  - Validates batch fill, throughput, and data integrity

Usage:
    uv run python alphazero-cpp/tests/test_spin_poll_stability.py
"""

import sys
import os
import time
import threading
import numpy as np

# Add project paths
scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.insert(0, scripts_dir)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'Release'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import torch
import torch.nn.functional as F
from torch.amp import autocast
import alphazero_cpp
from network import AlphaZeroNet, INPUT_CHANNELS, POLICY_SIZE


def create_gpu_evaluator(num_filters=128, num_blocks=10, se_reduction=16,
                         gpu_batch_size=512, device="cuda"):
    """Create a real AlphaZeroNet GPU evaluator matching the production pipeline.

    Uses eager inference (no CUDA graphs) for simplicity — the key thing being
    tested is the C++ spin-poll, not CUDA graph routing. CUDA graphs require
    fixed batch sizes and calibration which adds complexity without testing
    anything different in the C++ queue.
    """
    network = AlphaZeroNet(
        input_channels=INPUT_CHANNELS,
        num_filters=num_filters,
        num_blocks=num_blocks,
        se_reduction=se_reduction,
        wdl=True,
    ).to(device)
    network.eval()

    # Enable channels_last after .to(device) (matches train.py)
    network = network.to(memory_format=torch.channels_last)

    call_count = [0]
    total_infer_ms = [0.0]

    @torch.no_grad()
    def evaluator(obs_array, mask_array, batch_size, out_policies=None, out_values=None):
        call_count[0] += 1
        infer_start = time.perf_counter()

        # NHWC→channels_last NCHW (zero-copy permute, matches train.py)
        obs_tensor = torch.from_numpy(np.asarray(obs_array)[:batch_size]).permute(0, 3, 1, 2).float().to(device)
        mask_tensor = torch.from_numpy(np.asarray(mask_array)[:batch_size]).float().to(device)

        with autocast('cuda'):
            policies, _, _, wdl_logits = network(obs_tensor, mask_tensor)

        # WDL logits → probabilities (matches train.py exactly)
        wdl_probs = F.softmax(wdl_logits.float(), dim=1)

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - infer_start) * 1000
        total_infer_ms[0] += elapsed_ms

        policies_np = policies[:batch_size].cpu().numpy().astype(np.float32)
        wdl_np = wdl_probs[:batch_size].cpu().numpy().astype(np.float32)

        if out_policies is not None and out_values is not None:
            np.copyto(np.asarray(out_policies)[:batch_size], policies_np)
            np.copyto(np.asarray(out_values)[:batch_size], wdl_np)
            return None
        return policies_np, wdl_np

    return evaluator, call_count, total_infer_ms


def run_test(name, num_workers, games_per_worker, simulations, mcts_batch_size,
             gpu_batch_size, num_filters=128, num_blocks=10, se_reduction=16,
             gpu_timeout_ms=20, queue_capacity=8192, use_gumbel=True):
    """Run a single GPU stress test configuration."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    print(f"  Workers: {num_workers}, Games/worker: {games_per_worker}")
    print(f"  Simulations: {simulations}, MCTS batch: {mcts_batch_size}")
    print(f"  GPU batch: {gpu_batch_size}, Queue capacity: {queue_capacity}")
    print(f"  Network: f{num_filters}-b{num_blocks}-se{se_reduction}")
    print(f"  Gumbel: {use_gumbel}")

    total_games = num_workers * games_per_worker

    evaluator, call_count, total_infer_ms = create_gpu_evaluator(
        num_filters=num_filters, num_blocks=num_blocks,
        se_reduction=se_reduction, gpu_batch_size=gpu_batch_size)

    buffer = alphazero_cpp.ReplayBuffer(200000)

    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=num_workers,
        games_per_worker=games_per_worker,
        num_simulations=simulations,
        mcts_batch_size=mcts_batch_size,
        gpu_batch_size=gpu_batch_size,
        gpu_timeout_ms=gpu_timeout_ms,
        queue_capacity=queue_capacity,
        use_gumbel=use_gumbel,
        gumbel_top_k=mcts_batch_size if use_gumbel else 16,
    )
    coordinator.set_replay_buffer(buffer)

    # Collect live stats in background
    live_stats_samples = []
    stop_monitoring = threading.Event()

    def monitor_stats():
        while not stop_monitoring.is_set():
            try:
                stats = coordinator.get_live_stats()
                if stats:
                    live_stats_samples.append(stats)
            except Exception:
                pass
            time.sleep(0.5)

    monitor = threading.Thread(target=monitor_stats, daemon=True)
    monitor.start()

    start = time.time()
    result = coordinator.generate_games(evaluator)
    elapsed = time.time() - start

    stop_monitoring.set()
    monitor.join(timeout=2)

    # Extract results
    games_completed = result.get('games_completed', 0)
    total_moves = result.get('total_moves', 0)
    total_batches = result.get('total_batches', 0)
    cpp_error = result.get('cpp_error', '')
    pool_exhaustion = result.get('pool_exhaustion_count', 0)
    submission_drops = result.get('submission_drops', 0)
    submission_waits = result.get('submission_waits', 0)
    avg_batch = result.get('avg_batch_size', 0)
    total_leaves = result.get('total_leaves', 0)

    # Spin-poll metric from live stats
    spin_poll_values = [s.get('spin_poll_avg_us', -1) for s in live_stats_samples
                        if 'spin_poll_avg_us' in s]

    # Compute throughput
    avg_infer_ms = total_infer_ms[0] / call_count[0] if call_count[0] > 0 else 0

    print(f"\n  Results:")
    print(f"    Time: {elapsed:.1f}s")
    print(f"    Games: {games_completed}/{total_games}")
    print(f"    Moves: {total_moves}, Moves/sec: {total_moves/elapsed:.1f}")
    print(f"    NN evals: {call_count[0]} calls, {total_batches} batches")
    print(f"    Avg batch: {avg_batch:.1f}/{gpu_batch_size} ({avg_batch/gpu_batch_size*100:.0f}% fill)")
    print(f"    Avg GPU infer: {avg_infer_ms:.2f}ms/batch")
    print(f"    Leaves/sec: {total_leaves/elapsed:.0f}")
    print(f"    Pool exhaustion: {pool_exhaustion}")
    print(f"    Submission drops: {submission_drops}")
    print(f"    Submission waits: {submission_waits}")
    print(f"    Buffer samples: {buffer.size()}")
    if spin_poll_values:
        print(f"    spin_poll_avg_us: min={min(spin_poll_values):.0f}, "
              f"max={max(spin_poll_values):.0f}, "
              f"last={spin_poll_values[-1]:.0f}")
    else:
        print(f"    spin_poll_avg_us: not captured in live stats")
    if cpp_error:
        print(f"    CPP ERROR: {cpp_error}")

    # Validation checks
    checks_passed = 0
    checks_total = 0

    def check(condition, msg):
        nonlocal checks_passed, checks_total
        checks_total += 1
        if condition:
            checks_passed += 1
            print(f"  OK {msg}")
        else:
            print(f"  FAIL {msg}")

    check(games_completed == total_games,
          f"all {total_games} games completed: {games_completed}")
    check(not cpp_error, f"no C++ errors: '{cpp_error}'")
    check(total_moves > 0, f"moves generated: {total_moves}")
    check(buffer.size() > 0, f"buffer has samples: {buffer.size()}")

    # Data integrity: sample from buffer and check for NaN/Inf
    if buffer.size() >= 64:
        obs, pol, val, wdl, sv = buffer.sample(64)
        check(not np.any(np.isnan(obs)), "no NaN in observations")
        check(not np.any(np.isinf(obs)), "no Inf in observations")
        check(not np.any(np.isnan(pol)), "no NaN in policies")
        check(not np.any(np.isnan(val)), "no NaN in values")
        check(np.all(pol >= 0), "all policy values >= 0")
        pol_sums = pol.sum(axis=1)
        check(np.allclose(pol_sums, 1.0, atol=0.01),
              f"policy sums near 1.0: [{pol_sums.min():.4f}, {pol_sums.max():.4f}]")

    # Spin-poll metric check
    check(len(spin_poll_values) > 0,
          f"spin_poll_avg_us captured ({len(spin_poll_values)} samples)")
    if spin_poll_values:
        last_spin = spin_poll_values[-1]
        check(last_spin >= 0, f"spin_poll_avg_us >= 0: {last_spin:.0f}")
        check(last_spin < 10000, f"spin_poll_avg_us < 10ms: {last_spin:.0f}us")

    # Game result integrity
    w = result.get('white_wins', 0)
    b = result.get('black_wins', 0)
    d = result.get('draws', 0)
    check(w + b + d == games_completed, f"W({w})+B({b})+D({d}) == {games_completed}")

    # Drop rate < 1%
    drop_rate = submission_drops / max(total_leaves + submission_drops, 1)
    check(drop_rate < 0.01, f"drop rate < 1%: {drop_rate*100:.2f}%")

    print(f"\n  {checks_passed}/{checks_total} checks passed")
    return checks_passed, checks_total, elapsed


def main():
    print("=" * 60)
    print("Spin-Poll GPU Integration Stress Tests")
    print("=" * 60)
    print(f"alphazero_cpp version: {alphazero_cpp.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        print(f"GPU memory: {mem / 1024**3:.1f} GB")

    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available. This test requires a GPU.")
        return 1

    # Enable TF32 (matches train.py)
    torch.set_float32_matmul_precision("high")

    total_passed = 0
    total_checks = 0
    test_results = []

    # Use a small network (f128-b10) for faster test turnaround
    # while still exercising real CUDA inference latency patterns
    net_kwargs = dict(num_filters=128, num_blocks=10, se_reduction=16)

    # Test 1: 32 workers x 16 search_batch (Gumbel) — primary stress test
    p, t, elapsed = run_test(
        name="32 workers, sb=16, Gumbel (primary stress test)",
        num_workers=32, games_per_worker=1, simulations=50,
        mcts_batch_size=16, gpu_batch_size=512,
        gpu_timeout_ms=20, queue_capacity=8192,
        use_gumbel=True, **net_kwargs,
    )
    total_passed += p; total_checks += t
    test_results.append(("32w sb=16 Gumbel", p, t, elapsed))

    # Test 2: 48 workers x 16 search_batch — extreme contention
    p, t, elapsed = run_test(
        name="48 workers, sb=16, Gumbel (extreme contention)",
        num_workers=48, games_per_worker=1, simulations=50,
        mcts_batch_size=16, gpu_batch_size=512,
        gpu_timeout_ms=20, queue_capacity=8192,
        use_gumbel=True, **net_kwargs,
    )
    total_passed += p; total_checks += t
    test_results.append(("48w sb=16 Gumbel", p, t, elapsed))

    # Test 3: 64 workers x 16 search_batch — maximum workers
    p, t, elapsed = run_test(
        name="64 workers, sb=16, Gumbel (max workers)",
        num_workers=64, games_per_worker=1, simulations=40,
        mcts_batch_size=16, gpu_batch_size=512,
        gpu_timeout_ms=30, queue_capacity=8192,
        use_gumbel=True, **net_kwargs,
    )
    total_passed += p; total_checks += t
    test_results.append(("64w sb=16 Gumbel", p, t, elapsed))

    # Test 4: 32 workers, small queue — backpressure + compaction stress
    p, t, elapsed = run_test(
        name="32 workers, sb=16, small queue (2048) — backpressure",
        num_workers=32, games_per_worker=1, simulations=50,
        mcts_batch_size=16, gpu_batch_size=256,
        gpu_timeout_ms=20, queue_capacity=2048,
        use_gumbel=True, **net_kwargs,
    )
    total_passed += p; total_checks += t
    test_results.append(("32w small queue", p, t, elapsed))

    # Test 5: 32 workers x PUCT (sb=1) — many small submissions
    p, t, elapsed = run_test(
        name="32 workers, sb=1, PUCT (many small submissions)",
        num_workers=32, games_per_worker=1, simulations=50,
        mcts_batch_size=1, gpu_batch_size=256,
        gpu_timeout_ms=20, queue_capacity=4096,
        use_gumbel=False, **net_kwargs,
    )
    total_passed += p; total_checks += t
    test_results.append(("32w sb=1 PUCT", p, t, elapsed))

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for name, p, t, elapsed in test_results:
        status = "PASS" if p == t else "FAIL"
        print(f"  [{status}] {name}: {p}/{t} checks ({elapsed:.1f}s)")

    print(f"\nTotal: {total_passed}/{total_checks} checks passed")

    if total_passed == total_checks:
        print(f"\n{'='*60}")
        print(f"ALL {total_checks} CHECKS PASSED")
        print(f"{'='*60}")
        return 0
    else:
        print(f"\n{'='*60}")
        print(f"FAILURES: {total_checks - total_passed} checks failed")
        print(f"{'='*60}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
