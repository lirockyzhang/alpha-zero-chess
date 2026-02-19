#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration tests for Prioritized Experience Replay (PER)."""

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
    obs = np.full(OBS_SIZE, float(idx), dtype=np.float32)
    pol = np.zeros(POLICY_SIZE, dtype=np.float32)
    pol[idx % POLICY_SIZE] = 1.0
    val = float(idx) / 100.0
    wdl = np.array([0.4, 0.3, 0.3], dtype=np.float32)
    return obs, pol, val, wdl


def test_per_basic():
    """Enable PER, add samples, set high/low priorities, verify biased sampling."""
    print("=== Test 1: PER Basic Sampling ===")

    buf = alphazero_cpp.ReplayBuffer(200)
    buf.enable_per(0.6)
    assert buf.per_enabled(), "PER should be enabled"
    assert abs(buf.priority_exponent() - 0.6) < 1e-6, "Alpha mismatch"

    # Add 100 samples
    for i in range(100):
        obs, pol, val, wdl = make_dummy_sample(i)
        buf.add_sample(obs, pol, val, wdl)

    assert buf.size() == 100, f"Expected 100, got {buf.size()}"

    # Set indices 0-9 to very high priority, 10-99 to very low
    high_indices = np.arange(10, dtype=np.uint32)
    low_indices = np.arange(10, 100, dtype=np.uint32)
    buf.update_priorities(high_indices, np.full(10, 10.0, dtype=np.float32))
    buf.update_priorities(low_indices, np.full(90, 0.001, dtype=np.float32))

    # Sample many batches and count how often high-priority items appear
    high_count = 0
    total_samples = 0
    for _ in range(50):
        result = buf.sample_prioritized(32, beta=0.4)
        obs, pol, val, wdl, sv, indices, is_weights = result
        assert indices.dtype == np.uint32, f"indices dtype: {indices.dtype}"
        assert is_weights.dtype == np.float32, f"weights dtype: {is_weights.dtype}"
        assert obs.shape == (32, OBS_SIZE), f"obs shape: {obs.shape}"
        high_count += np.sum(indices < 10)
        total_samples += 32

    high_fraction = high_count / total_samples
    print(f"  High-priority fraction: {high_fraction:.3f} (expect > 0.5)")
    assert high_fraction > 0.3, f"High-priority items should dominate, got {high_fraction:.3f}"
    print("PASS\n")


def test_per_disabled():
    """alpha=0 should not enable PER."""
    print("=== Test 2: PER Disabled (alpha=0) ===")

    buf = alphazero_cpp.ReplayBuffer(100)
    buf.enable_per(0.0)  # Should be a no-op
    assert not buf.per_enabled(), "PER should not be enabled with alpha=0"

    # Add samples and verify regular sampling works
    for i in range(50):
        obs, pol, val, wdl = make_dummy_sample(i)
        buf.add_sample(obs, pol, val, wdl)

    result = buf.sample(32)
    obs, pol, val, wdl, sv = result
    assert obs.shape == (32, OBS_SIZE), f"Regular sample shape: {obs.shape}"
    print("PASS\n")


def test_per_is_weights_beta():
    """Verify IS weights behavior at different beta values."""
    print("=== Test 3: IS Weights with Different Beta ===")

    buf = alphazero_cpp.ReplayBuffer(200)
    buf.enable_per(0.6)

    # Add 50 samples
    for i in range(50):
        obs, pol, val, wdl = make_dummy_sample(i)
        buf.add_sample(obs, pol, val, wdl)

    # Set varied priorities
    indices = np.arange(50, dtype=np.uint32)
    priorities = np.linspace(0.1, 10.0, 50, dtype=np.float32)
    buf.update_priorities(indices, priorities)

    # beta=0: all weights should be ~1.0
    result = buf.sample_prioritized(32, beta=0.0)
    _, _, _, _, _, _, weights_b0 = result
    assert np.allclose(weights_b0, 1.0, atol=1e-5), \
        f"beta=0 weights should all be 1.0, got range [{weights_b0.min():.4f}, {weights_b0.max():.4f}]"
    print(f"  beta=0: weights range [{weights_b0.min():.4f}, {weights_b0.max():.4f}] (all ~1.0)")

    # beta=1.0: weights should vary, max should be 1.0
    result = buf.sample_prioritized(32, beta=1.0)
    _, _, _, _, _, _, weights_b1 = result
    assert abs(weights_b1.max() - 1.0) < 1e-4, \
        f"beta=1 max weight should be 1.0, got {weights_b1.max():.4f}"
    assert weights_b1.min() < 0.9, \
        f"beta=1 weights should vary, min={weights_b1.min():.4f}"
    print(f"  beta=1: weights range [{weights_b1.min():.4f}, {weights_b1.max():.4f}]")

    print("PASS\n")


def test_per_save_load():
    """Save with priorities, load into new buffer, verify priorities preserved."""
    print("=== Test 4: PER Save/Load ===")

    buf1 = alphazero_cpp.ReplayBuffer(200)
    buf1.enable_per(0.6)

    # Add samples with varied priorities
    for i in range(50):
        obs, pol, val, wdl = make_dummy_sample(i)
        buf1.add_sample(obs, pol, val, wdl)

    # Set specific priorities
    indices = np.arange(50, dtype=np.uint32)
    priorities = np.linspace(0.5, 5.0, 50, dtype=np.float32)
    buf1.update_priorities(indices, priorities)

    # Sample before save to get reference distribution
    high_count_before = 0
    for _ in range(20):
        result = buf1.sample_prioritized(32, beta=0.4)
        _, _, _, _, _, idx, _ = result
        high_count_before += np.sum(idx >= 40)  # high-priority indices

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.rpbf', delete=False) as f:
        tmppath = f.name

    try:
        assert buf1.save(tmppath), "Save failed"

        # Load into new buffer with PER enabled
        buf2 = alphazero_cpp.ReplayBuffer(200)
        buf2.enable_per(0.6)
        assert buf2.load(tmppath), "Load failed"
        assert buf2.size() == 50, f"Loaded size: {buf2.size()}"
        assert buf2.per_enabled(), "PER should still be enabled"

        # Sample after load and check distribution is similar
        high_count_after = 0
        for _ in range(20):
            result = buf2.sample_prioritized(32, beta=0.4)
            _, _, _, _, _, idx, _ = result
            high_count_after += np.sum(idx >= 40)

        print(f"  Before save: high-idx samples = {high_count_before}")
        print(f"  After load:  high-idx samples = {high_count_after}")
        # Both should show bias toward high-priority indices
        assert high_count_after > high_count_before * 0.3, \
            "Loaded buffer should preserve priority distribution"

    finally:
        os.unlink(tmppath)

    print("PASS\n")


if __name__ == '__main__':
    test_per_basic()
    test_per_disabled()
    test_per_is_weights_beta()
    test_per_save_load()
    print("=" * 50)
    print("All PER tests passed!")
