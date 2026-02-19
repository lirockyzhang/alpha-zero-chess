#!/usr/bin/env python3
"""Test C++ training components (ReplayBuffer and Trainer)."""

import sys
import os
import numpy as np
from pathlib import Path

# Add build directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "build" / "Release"))

try:
    import alphazero_cpp
except ImportError as e:
    print(f"Error importing alphazero_cpp: {e}")
    print("Make sure to build the C++ extension first:")
    print("  cd alphazero-cpp")
    print("  cmake --build build --config Release")
    sys.exit(1)

def test_replay_buffer_basic():
    """Test basic ReplayBuffer operations."""
    print("\n" + "="*80)
    print("Test 1: ReplayBuffer Basic Operations")
    print("="*80)

    # Create buffer
    buffer = alphazero_cpp.ReplayBuffer(capacity=1000)
    print(f"OK Created buffer with capacity 1000")

    # Check initial state
    assert buffer.size() == 0
    assert buffer.capacity() == 1000
    print(f"OK Initial size: {buffer.size()}, capacity: {buffer.capacity()}")

    # Add a single sample
    obs = np.random.rand(7872).astype(np.float32)  # 8*8*123
    pol = np.random.rand(4672).astype(np.float32)
    val = 0.5

    buffer.add_sample(obs, pol, val)
    print(f"OK Added 1 sample, buffer size: {buffer.size()}")

    # Add a batch
    batch_size = 10
    obs_batch = np.random.rand(batch_size, 7872).astype(np.float32)
    pol_batch = np.random.rand(batch_size, 4672).astype(np.float32)
    val_batch = np.random.rand(batch_size).astype(np.float32)

    buffer.add_batch(obs_batch, pol_batch, val_batch)
    print(f"OK Added batch of {batch_size} samples, buffer size: {buffer.size()}")

    # Check stats
    stats = buffer.get_stats()
    print(f"OK Buffer stats: {stats}")

    assert stats['size'] == 11
    assert stats['total_added'] == 11
    assert stats['total_games'] == 1

    print("[PASS] Basic operations test")

def test_replay_buffer_sampling():
    """Test ReplayBuffer sampling."""
    print("\n" + "="*80)
    print("Test 2: ReplayBuffer Sampling")
    print("="*80)

    # Create and populate a buffer for sampling
    buffer = alphazero_cpp.ReplayBuffer(capacity=1000)
    for _ in range(20):
        obs = np.random.rand(7872).astype(np.float32)
        pol = np.random.rand(4672).astype(np.float32)
        buffer.add_sample(obs, pol, 0.5)

    # Sample a batch
    batch_size = 5
    obs, pol, val, wdl, sv = buffer.sample(batch_size)

    print(f"OK Sampled batch of {batch_size}")
    print(f"  Observations shape: {obs.shape}")
    print(f"  Policies shape: {pol.shape}")
    print(f"  Values shape: {val.shape}")
    print(f"  WDL targets shape: {wdl.shape}")
    print(f"  Soft values shape: {sv.shape}")

    assert obs.shape == (batch_size, 7872)
    assert pol.shape == (batch_size, 4672)
    assert val.shape == (batch_size,)
    assert wdl.shape == (batch_size, 3)
    assert sv.shape == (batch_size,)

    print("[PASS] Sampling test")

def test_replay_buffer_overflow():
    """Test ReplayBuffer circular buffer overflow."""
    print("\n" + "="*80)
    print("Test 3: ReplayBuffer Overflow (Circular)")
    print("="*80)

    # Create small buffer
    buffer = alphazero_cpp.ReplayBuffer(capacity=5)

    # Add more samples than capacity
    for i in range(10):
        obs = np.random.rand(7872).astype(np.float32)
        pol = np.random.rand(4672).astype(np.float32)
        buffer.add_sample(obs, pol, float(i))

    # Buffer should be at capacity
    assert buffer.size() == 5
    assert buffer.total_added() == 10

    print(f"OK Added 10 samples to capacity-5 buffer")
    print(f"  Current size: {buffer.size()} (capacity: 5)")
    print(f"  Total added: {buffer.total_added()}")

    print("[PASS] Overflow test")

def test_trainer():
    """Test Trainer coordinator."""
    print("\n" + "="*80)
    print("Test 5: Trainer Coordinator")
    print("="*80)

    # Create trainer with custom config
    config = alphazero_cpp.TrainerConfig()
    config.batch_size = 256
    config.min_buffer_size = 1000
    config.learning_rate = 0.001

    trainer = alphazero_cpp.Trainer(config)
    print(f"OK Created trainer with config")

    # Get config
    cfg = trainer.get_config()
    assert cfg.batch_size == 256
    print(f"OK Config verified: batch_size={cfg.batch_size}")

    # Check buffer readiness
    buffer = alphazero_cpp.ReplayBuffer(capacity=2000)
    assert not trainer.is_ready(buffer)
    print(f"OK Buffer not ready (size={buffer.size()} < min={config.min_buffer_size})")

    # Fill buffer
    for i in range(1200):
        obs = np.random.rand(7872).astype(np.float32)
        pol = np.random.rand(4672).astype(np.float32)
        buffer.add_sample(obs, pol, 0.5)

    assert trainer.is_ready(buffer)
    print(f"OK Buffer ready (size={buffer.size()} >= min={config.min_buffer_size})")

    # Record training steps
    trainer.record_step(batch_size=256, loss=1.5, policy_loss=0.8, value_loss=0.7)
    trainer.record_step(batch_size=256, loss=1.2, policy_loss=0.6, value_loss=0.6)

    # Get stats
    stats = trainer.get_stats()
    print(f"OK Trainer stats: {stats}")

    assert stats['total_steps'] == 2
    assert stats['total_samples_trained'] == 512
    assert abs(stats['last_loss'] - 1.2) < 0.001

    # Reset stats
    trainer.reset_stats()
    stats = trainer.get_stats()
    assert stats['total_steps'] == 0
    print(f"OK Stats reset successfully")

    print("[PASS] Trainer test")

def main():
    print("\n" + "="*80)
    print("C++ TRAINING COMPONENTS TEST SUITE")
    print("="*80)

    try:
        # Run tests
        test_replay_buffer_basic()
        test_replay_buffer_sampling()
        test_replay_buffer_overflow()
        test_trainer()

        # Summary
        print("\n" + "="*80)
        print("ALL TESTS PASSED OK")
        print("="*80)
        print("\nC++ training components are working correctly:")
        print("  OK ReplayBuffer - create, add, sample")
        print("  OK ReplayBuffer - circular overflow")
        print("  OK Trainer - configuration and statistics")
        print("\nReady for integration with training loop!")
        print("="*80)

        return 0

    except Exception as e:
        print("\n" + "="*80)
        print(f"TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
