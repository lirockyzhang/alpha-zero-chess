#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extended edge-case tests for Prioritized Experience Replay (PER)."""

import sys
import os
import tempfile
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'Release')
if not os.path.exists(build_dir):
    build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

import alphazero_cpp

OBS_SIZE = 8 * 8 * 123  # 7872
POLICY_SIZE = 4672


def make_dummy_sample(idx=0):
    """Create a dummy sample with unique observation based on idx."""
    obs = np.full(OBS_SIZE, float(idx % 256), dtype=np.float32)
    pol = np.zeros(POLICY_SIZE, dtype=np.float32)
    pol[idx % POLICY_SIZE] = 1.0
    val = float(idx % 3 - 1)  # cycle through -1, 0, 1
    wdl = np.array([0.4, 0.3, 0.3], dtype=np.float32)
    return obs, pol, val, wdl


def test_buffer_wrap_with_per():
    """Test PER behavior when circular buffer wraps around."""
    print("=== Test E1: Buffer Wrapping with PER ===")

    buf = alphazero_cpp.ReplayBuffer(50)
    buf.enable_per(0.6)

    # Add 80 samples → wraps around (overwrites first 30)
    for i in range(80):
        obs, pol, val, wdl = make_dummy_sample(i)
        buf.add_sample(obs, pol, val, wdl)

    assert buf.size() == 50, f"Expected 50 (capacity), got {buf.size()}"

    # All 50 live slots should be sampeable
    result = buf.sample_prioritized(32, beta=0.4)
    obs, pol, val, wdl, sv, indices, weights = result
    assert obs.shape == (32, OBS_SIZE), f"obs shape: {obs.shape}"
    assert np.all(indices < 50), f"Indices should be < capacity: max={indices.max()}"
    assert np.all(weights > 0), "All weights should be positive"

    # Update priorities on the sampled indices — should not crash
    new_priorities = np.full(32, 2.0, dtype=np.float32)
    buf.update_priorities(indices, new_priorities)

    # Sample again after priority update — should still work
    result2 = buf.sample_prioritized(32, beta=0.4)
    _, _, _, _, _, indices2, weights2 = result2
    assert np.all(indices2 < 50), f"Post-update indices should be < capacity"

    print("  Buffer wrapped 80 samples into 50 capacity — PER works correctly")
    print("PASS\n")


def test_per_tiny_buffer():
    """PER with very small buffer (edge case: buffer smaller than batch)."""
    print("=== Test E2: PER with Tiny Buffer ===")

    buf = alphazero_cpp.ReplayBuffer(10)
    buf.enable_per(0.6)

    # Add exactly 10 samples
    for i in range(10):
        obs, pol, val, wdl = make_dummy_sample(i)
        buf.add_sample(obs, pol, val, wdl)

    # Sample batch of 10 (equals capacity)
    result = buf.sample_prioritized(10, beta=0.5)
    obs, pol, val, wdl, sv, indices, weights = result
    assert obs.shape == (10, OBS_SIZE), f"obs shape: {obs.shape}"
    assert weights.shape == (10,), f"weights shape: {weights.shape}"
    assert np.all(weights > 0) and np.all(weights <= 1.0 + 1e-6), \
        f"Weights should be in (0, 1]: [{weights.min():.6f}, {weights.max():.6f}]"

    print(f"  Sampled batch=10 from capacity=10, weights range: "
          f"[{weights.min():.4f}, {weights.max():.4f}]")
    print("PASS\n")


def test_per_all_same_priority():
    """When all priorities are equal, sampling should be uniform-like."""
    print("=== Test E3: PER with Uniform Priorities ===")

    buf = alphazero_cpp.ReplayBuffer(100)
    buf.enable_per(0.6)

    for i in range(100):
        obs, pol, val, wdl = make_dummy_sample(i)
        buf.add_sample(obs, pol, val, wdl)

    # All enter at max_priority (1.0 initial) — priorities should be equal
    # With equal priorities, IS weights should all be ~1.0 at any beta
    result = buf.sample_prioritized(50, beta=1.0)
    _, _, _, _, _, _, weights = result
    assert np.allclose(weights, 1.0, atol=0.01), \
        f"Equal priorities should give weights ~1.0, got [{weights.min():.4f}, {weights.max():.4f}]"

    print(f"  Equal-priority weights: [{weights.min():.4f}, {weights.max():.4f}] (all ~1.0)")
    print("PASS\n")


def test_per_extreme_priority_ratio():
    """Test with extreme priority ratios (1e-8 vs 1e+3)."""
    print("=== Test E4: Extreme Priority Ratios ===")

    buf = alphazero_cpp.ReplayBuffer(100)
    buf.enable_per(0.6)

    for i in range(100):
        obs, pol, val, wdl = make_dummy_sample(i)
        buf.add_sample(obs, pol, val, wdl)

    # Set one sample to extremely high priority, rest to near-zero
    high_idx = np.array([0], dtype=np.uint32)
    low_idx = np.arange(1, 100, dtype=np.uint32)
    buf.update_priorities(high_idx, np.array([1000.0], dtype=np.float32))
    buf.update_priorities(low_idx, np.full(99, 1e-8, dtype=np.float32))

    # With such extreme ratio, sample 0 should appear in almost every batch
    sample_0_count = 0
    total = 0
    for _ in range(100):
        result = buf.sample_prioritized(16, beta=0.4)
        _, _, _, _, _, indices, weights = result
        sample_0_count += np.sum(indices == 0)
        total += 16
        # Weights should be valid (no NaN, no inf, no negative)
        assert np.all(np.isfinite(weights)), f"Non-finite weights found"
        assert np.all(weights > 0), f"Non-positive weights found"

    frac = sample_0_count / total
    print(f"  Sample 0 fraction: {frac:.3f} (expect ~1.0 with extreme ratio)")
    assert frac > 0.5, f"High-priority sample 0 should dominate, got {frac:.3f}"
    print("PASS\n")


def test_per_priority_update_correctness():
    """Verify that updated priorities actually change sampling distribution."""
    print("=== Test E5: Priority Update Changes Distribution ===")

    buf = alphazero_cpp.ReplayBuffer(100)
    buf.enable_per(0.6)

    for i in range(100):
        obs, pol, val, wdl = make_dummy_sample(i)
        buf.add_sample(obs, pol, val, wdl)

    # Phase 1: Set samples 0-9 high, measure their frequency
    high_idx = np.arange(10, dtype=np.uint32)
    low_idx = np.arange(10, 100, dtype=np.uint32)
    buf.update_priorities(high_idx, np.full(10, 10.0, dtype=np.float32))
    buf.update_priorities(low_idx, np.full(90, 0.01, dtype=np.float32))

    count_phase1 = 0
    for _ in range(50):
        _, _, _, _, _, indices, _ = buf.sample_prioritized(32, beta=0.4)
        count_phase1 += np.sum(indices < 10)

    # Phase 2: Swap — now 90-99 are high priority
    buf.update_priorities(high_idx, np.full(10, 0.01, dtype=np.float32))
    new_high = np.arange(90, 100, dtype=np.uint32)
    buf.update_priorities(new_high, np.full(10, 10.0, dtype=np.float32))

    count_phase2_old_high = 0
    count_phase2_new_high = 0
    for _ in range(50):
        _, _, _, _, _, indices, _ = buf.sample_prioritized(32, beta=0.4)
        count_phase2_old_high += np.sum(indices < 10)
        count_phase2_new_high += np.sum(indices >= 90)

    print(f"  Phase 1: idx 0-9 count = {count_phase1} / 1600")
    print(f"  Phase 2: idx 0-9 count = {count_phase2_old_high} / 1600 (should drop)")
    print(f"  Phase 2: idx 90-99 count = {count_phase2_new_high} / 1600 (should rise)")

    assert count_phase1 > count_phase2_old_high * 3, \
        f"After lowering 0-9 priority, they should appear less often"
    assert count_phase2_new_high > count_phase2_old_high * 3, \
        f"90-99 should dominate after raising priority"
    print("PASS\n")


def test_per_indices_dtype():
    """Verify indices from sample_prioritized are uint32 (matches update_priorities input)."""
    print("=== Test E6: Index Dtype Roundtrip ===")

    buf = alphazero_cpp.ReplayBuffer(50)
    buf.enable_per(0.6)

    for i in range(50):
        obs, pol, val, wdl = make_dummy_sample(i)
        buf.add_sample(obs, pol, val, wdl)

    result = buf.sample_prioritized(16, beta=0.4)
    _, _, _, _, _, indices, weights = result

    assert indices.dtype == np.uint32, f"indices dtype should be uint32, got {indices.dtype}"
    assert weights.dtype == np.float32, f"weights dtype should be float32, got {weights.dtype}"

    # Roundtrip: pass indices directly to update_priorities
    new_priorities = np.full(16, 5.0, dtype=np.float32)
    buf.update_priorities(indices, new_priorities)  # should not throw

    print(f"  indices dtype={indices.dtype}, weights dtype={weights.dtype}")
    print("  update_priorities roundtrip succeeded")
    print("PASS\n")


def test_per_save_load_roundtrip_no_per():
    """Save buffer WITHOUT PER, load into PER-enabled buffer → uniform priorities."""
    print("=== Test E7: Load Non-PER File into PER Buffer ===")

    # Save without PER
    buf1 = alphazero_cpp.ReplayBuffer(50)
    for i in range(30):
        obs, pol, val, wdl = make_dummy_sample(i)
        buf1.add_sample(obs, pol, val, wdl)

    with tempfile.NamedTemporaryFile(suffix='.rpbf', delete=False) as f:
        tmppath = f.name

    try:
        assert buf1.save(tmppath), "Save failed"

        # Load into PER-enabled buffer
        buf2 = alphazero_cpp.ReplayBuffer(50)
        buf2.enable_per(0.6)
        assert buf2.load(tmppath), "Load failed"
        assert buf2.size() == 30, f"Loaded size: {buf2.size()}"
        assert buf2.per_enabled(), "PER should still be enabled"

        # Should be able to sample (priorities initialized uniformly)
        result = buf2.sample_prioritized(16, beta=0.4)
        obs, pol, val, wdl, sv, indices, weights = result
        assert obs.shape == (16, OBS_SIZE), f"obs shape: {obs.shape}"
        # With uniform priorities, all weights should be ~1.0
        assert np.allclose(weights, 1.0, atol=0.05), \
            f"Uniform init weights should be ~1.0, got [{weights.min():.4f}, {weights.max():.4f}]"

        print(f"  Loaded 30 samples, weights: [{weights.min():.4f}, {weights.max():.4f}]")
    finally:
        os.unlink(tmppath)

    print("PASS\n")


def test_per_multiple_epochs_simulation():
    """Simulate what train_iteration does: sample, compute 'loss', update priorities, repeat."""
    print("=== Test E8: Multi-Epoch Training Simulation ===")

    buf = alphazero_cpp.ReplayBuffer(200)
    buf.enable_per(0.6)

    # Add 100 samples
    for i in range(100):
        obs, pol, val, wdl = make_dummy_sample(i)
        buf.add_sample(obs, pol, val, wdl)

    # Simulate 5 training epochs (like train_iteration with epochs=5)
    batch_size = 32
    for epoch in range(5):
        result = buf.sample_prioritized(batch_size, beta=0.4)
        obs, pol, val, wdl, sv, indices, is_weights = result

        # Simulate per-sample loss computation (random for testing)
        fake_loss = np.random.uniform(0.1, 5.0, size=batch_size).astype(np.float32)

        # Weight the loss (mimicking IS weighting)
        weighted_loss = np.mean(is_weights * fake_loss)
        assert np.isfinite(weighted_loss), f"Weighted loss should be finite, got {weighted_loss}"

        # Update priorities (as train_iteration would do)
        new_priorities = fake_loss + 1e-6
        buf.update_priorities(indices, new_priorities)

    # After 5 epochs of updates, sampling should still work
    result = buf.sample_prioritized(32, beta=0.8)
    _, _, _, _, _, indices_final, weights_final = result
    assert np.all(np.isfinite(weights_final)), "Final weights should be finite"
    assert np.all(weights_final > 0), "Final weights should be positive"

    print(f"  5 epochs simulated, final weights: [{weights_final.min():.4f}, {weights_final.max():.4f}]")
    print("PASS\n")


def test_per_beta_annealing_behavior():
    """Verify IS weight behavior across beta values (simulating annealing)."""
    print("=== Test E9: Beta Annealing Verification ===")

    buf = alphazero_cpp.ReplayBuffer(100)
    buf.enable_per(0.6)

    for i in range(100):
        obs, pol, val, wdl = make_dummy_sample(i)
        buf.add_sample(obs, pol, val, wdl)

    # Set varied priorities
    indices = np.arange(100, dtype=np.uint32)
    priorities = np.linspace(0.1, 10.0, 100, dtype=np.float32)
    buf.update_priorities(indices, priorities)

    # Sample at various beta values (simulating annealing from 0.4 to 1.0)
    betas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    weight_ranges = []
    for beta in betas:
        result = buf.sample_prioritized(50, beta=beta)
        _, _, _, _, _, _, weights = result
        weight_ranges.append((weights.min(), weights.max(), weights.std()))
        assert np.all(np.isfinite(weights)), f"beta={beta}: non-finite weights"
        assert np.all(weights > 0), f"beta={beta}: non-positive weights"
        assert weights.max() <= 1.0 + 1e-5, f"beta={beta}: max weight > 1.0: {weights.max()}"

    print("  Beta | Min Weight | Max Weight | Std Dev")
    print("  -----+------------+------------+--------")
    for beta, (wmin, wmax, wstd) in zip(betas, weight_ranges):
        print(f"  {beta:.1f}  | {wmin:.6f}  | {wmax:.6f}  | {wstd:.6f}")

    # beta=0 should give all 1.0
    assert weight_ranges[0][2] < 1e-5, "beta=0 should have zero variance"
    # As beta increases, variance should generally increase (more correction)
    assert weight_ranges[-1][2] > weight_ranges[0][2], \
        "beta=1.0 should have more weight variance than beta=0.0"

    print("PASS\n")


def test_regular_sample_still_works_with_per():
    """Calling regular sample() should still work even when PER is enabled."""
    print("=== Test E10: Regular sample() with PER Enabled ===")

    buf = alphazero_cpp.ReplayBuffer(100)
    buf.enable_per(0.6)

    for i in range(50):
        obs, pol, val, wdl = make_dummy_sample(i)
        buf.add_sample(obs, pol, val, wdl)

    # Regular sample should still return 5-tuple (no indices/weights)
    result = buf.sample(32)
    assert len(result) == 5, f"Regular sample should return 5-tuple, got {len(result)}"
    obs, pol, val, wdl, sv = result
    assert obs.shape == (32, OBS_SIZE), f"obs shape: {obs.shape}"

    print(f"  Regular sample() works with PER enabled, shape: {obs.shape}")
    print("PASS\n")


if __name__ == '__main__':
    test_buffer_wrap_with_per()
    test_per_tiny_buffer()
    test_per_all_same_priority()
    test_per_extreme_priority_ratio()
    test_per_priority_update_correctness()
    test_per_indices_dtype()
    test_per_save_load_roundtrip_no_per()
    test_per_multiple_epochs_simulation()
    test_per_beta_annealing_behavior()
    test_regular_sample_still_works_with_per()
    print("=" * 50)
    print("All extended PER tests passed!")
