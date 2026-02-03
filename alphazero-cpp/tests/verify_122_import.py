#!/usr/bin/env python3
"""Simple verification that alphazero_cpp loads with 122 channels."""

import sys
import os

# Force load from build directory
build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build', 'Release')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

print(f"Loading alphazero_cpp from: {build_dir}")
print(f".pyd file exists: {os.path.exists(os.path.join(build_dir, 'alphazero_cpp.cp313-win_amd64.pyd'))}")

import alphazero_cpp
print(f"Module loaded from: {alphazero_cpp.__file__}")

import numpy as np

# Test 1: Check CHANNELS constant
print("\nTest 1: Checking CHANNELS constant")
print(f"  alphazero_cpp.CHANNELS = {alphazero_cpp.CHANNELS if hasattr(alphazero_cpp, 'CHANNELS') else 'NOT FOUND'}")

# Test 2: Create a simple evaluator and check observation shape
print("\nTest 2: Checking observation shape from SelfPlayCoordinator")

def check_and_evaluate(observations, num_leaves):
    """Simple evaluator that checks observation shape and returns dummy results."""
    obs_array = np.array(observations)
    print(f"  Observation shape: {obs_array.shape}")

    # Check if shape is (batch, 8, 8, 122)
    if len(obs_array.shape) == 4 and obs_array.shape[3] == 122:
        print("  [SUCCESS] Observations have 122 channels!")
        success = True
    elif len(obs_array.shape) == 4 and obs_array.shape[3] == 119:
        print("  [FAILURE] Observations still have 119 channels (old version)")
        success = False
    else:
        print(f"  [ERROR] Unexpected shape: {obs_array.shape}")
        success = False

    # Return dummy policy and value for continuation
    policy = np.random.rand(num_leaves, 4672).astype(np.float32)
    value = np.random.rand(num_leaves).astype(np.float32)

    # Store success result globally
    global verification_success
    verification_success = success

    return (policy, value)

verification_success = False

try:
    # Create coordinator with minimal settings
    print("  Creating SelfPlayCoordinator...")
    coordinator = alphazero_cpp.SelfPlayCoordinator(
        num_workers=1,
        num_simulations=10,  # Minimal for fast test
        batch_size=4
    )

    print("  Generating one game to capture observation shape...")

    # Generate one game using the evaluator
    games = coordinator.generate_games(check_and_evaluate, num_games=1)

    if verification_success:
        print("\n[VERIFICATION COMPLETE] 122 channels confirmed!")
    else:
        print("\n[VERIFICATION FAILED] Still using 119 channels")

except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\nDone.")
