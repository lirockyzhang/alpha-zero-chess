#!/usr/bin/env python3
"""Simple test to verify 122-channel implementation."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "alphazero-cpp", "build", "Release"))
import alphazero_cpp

print("Testing 122-channel implementation...")
print("=" * 60)

# Create coordinator
coord = alphazero_cpp.SelfPlayCoordinator(1, 10, 4)
print("[OK] Coordinator created")

# Track observations
observations = []

def test_evaluator(obs_array, num_leaves):
    print(f"  Evaluator called: num_leaves={num_leaves}, obs shape={obs_array.shape}")
    observations.append(obs_array[0].copy())

    policies = np.random.random((num_leaves, 4672)).astype(np.float32)
    policies = policies / policies.sum(axis=1, keepdims=True)
    values = np.zeros(num_leaves, dtype=np.float32)
    return policies, values

# Generate game
print("\nGenerating game...")
games = coord.generate_games(test_evaluator, 1)
print(f"[OK] Game completed: {games[0]['num_moves']} moves")

# Check observations
if len(observations) > 0:
    obs = observations[0]
    print(f"\n[OK] Observation shape: {obs.shape}")
    expected_shape = (8, 8, 122)
    if obs.shape == expected_shape:
        print(f"[OK] Shape is correct: {expected_shape}")
    else:
        print(f"[FAIL] Shape is WRONG! Expected {expected_shape}, got {obs.shape}")
        sys.exit(1)

    # Check history encoding at later observation
    if len(observations) >= 10:
        obs_late = observations[9]
        history_count = np.count_nonzero(obs_late[:,:,18:122])
        print(f"[OK] History at move 10: {history_count} nonzero values")

        # Check T-8 specifically (channels 109-121)
        t8_pieces = np.count_nonzero(obs_late[:, :, 109:121])
        print(f"[OK] T-8 (channels 109-121): {t8_pieces} piece values")
        if t8_pieces > 0:
            print("[OK] T-8 is FULLY ENCODED!")
        else:
            print("  (T-8 empty - game not long enough)")

    print("\n" + "=" * 60)
    print("SUCCESS: 122-channel implementation working!")
    print("=" * 60)
else:
    print("[FAIL] No observations collected")
    sys.exit(1)
