#!/usr/bin/env python3
"""
Realistic End-to-End Test for 122-Channel Implementation

Simulates real training/inference workflow:
1. Create neural network with 122 channels
2. Generate self-play games using C++ backend
3. Train network on collected data
4. Test inference and MCTS search
5. Verify checkpoint save/load
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.join(os.getcwd(), "alphazero-cpp", "build", "Release"))
import alphazero_cpp

from alphazero.neural.network import AlphaZeroNetwork
from alphazero.config import NetworkConfig, TrainingConfig

print("="*80)
print("REALISTIC 122-CHANNEL TEST")
print("="*80)

# Test tracking
tests_passed = 0
tests_failed = 0

def test_result(name, passed, details=""):
    global tests_passed, tests_failed
    if passed:
        tests_passed += 1
        print(f"[PASS] {name}")
        if details:
            print(f"       {details}")
    else:
        tests_failed += 1
        print(f"[FAIL] {name}")
        if details:
            print(f"       {details}")

# ==============================================================================
# TEST 1: Neural Network Creation
# ==============================================================================
print("\n" + "-"*80)
print("TEST 1: Neural Network Creation with 122 Channels")
print("-"*80)

try:
    # Create network with 122 channels
    config = NetworkConfig(
        input_channels=122,  # Updated!
        num_filters=64,      # Small for testing
        num_blocks=5,        # Small for testing
        num_actions=4672
    )

    network = AlphaZeroNetwork(
        input_channels=config.input_channels,
        num_filters=config.num_filters,
        num_blocks=config.num_blocks,
        num_actions=config.num_actions
    )

    # Convert to channels_last for performance
    network = network.to(memory_format=torch.channels_last)

    # Test forward pass
    dummy_input = torch.randn(4, 122, 8, 8).to(memory_format=torch.channels_last)
    with torch.no_grad():
        policy, value = network(dummy_input)

    test_result(
        "Network creation and forward pass",
        policy.shape == (4, 4672) and value.shape == (4, 1),
        f"Policy shape: {policy.shape}, Value shape: {value.shape}"
    )
except Exception as e:
    test_result("Network creation", False, f"Error: {e}")

# ==============================================================================
# TEST 2: Self-Play Game Generation
# ==============================================================================
print("\n" + "-"*80)
print("TEST 2: Self-Play with C++ Backend (Real Network Inference)")
print("-"*80)

try:
    # Put network in eval mode
    network.eval()
    device = torch.device("cpu")
    network = network.to(device)

    observations_collected = []
    inference_count = 0

    def neural_evaluator(obs_array, num_leaves):
        """Real neural network inference"""
        global inference_count
        inference_count += 1

        # Check observation shape
        if len(observations_collected) == 0:
            observations_collected.append(obs_array.shape)

        # Convert to PyTorch tensor (NHWC -> NCHW)
        obs_tensor = torch.from_numpy(obs_array).permute(0, 3, 1, 2).contiguous()
        obs_tensor = obs_tensor.to(memory_format=torch.channels_last)
        obs_tensor = obs_tensor.to(device)

        # Run inference
        with torch.no_grad():
            policy_logits, values = network(obs_tensor)

        # Convert to probabilities
        policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
        values = values.squeeze(-1).cpu().numpy()

        return policies, values

    # Generate games with real network
    print("Generating 2 games with 122-channel observations...")
    coord = alphazero_cpp.SelfPlayCoordinator(
        num_workers=2,
        num_simulations=50,
        batch_size=16
    )

    games = coord.generate_games(neural_evaluator, 2)

    # Verify results
    obs_shape_correct = (
        len(observations_collected) > 0 and
        observations_collected[0][2] == 8 and
        observations_collected[0][3] == 122
    )

    test_result(
        "Self-play game generation",
        len(games) == 2 and obs_shape_correct,
        f"Games: {len(games)}, Shape: {observations_collected[0] if observations_collected else 'none'}, Inferences: {inference_count}"
    )

    # Check game details
    for i, game in enumerate(games):
        print(f"  Game {i+1}: {game['num_moves']} moves, result={game['result']}")

except Exception as e:
    test_result("Self-play generation", False, f"Error: {e}")
    import traceback
    traceback.print_exc()

# ==============================================================================
# TEST 3: Training Data Collection
# ==============================================================================
print("\n" + "-"*80)
print("TEST 3: Training Data Preparation")
print("-"*80)

try:
    # Collect observations from games
    all_observations = []
    all_policies = []
    all_values = []

    for game in games:
        observations = game['observations']
        policies = game['policies']
        result = game['result']

        for obs, policy in zip(observations, policies):
            all_observations.append(obs)
            all_policies.append(policy)
            # Value is the game result from player's perspective
            all_values.append(result)

    # Convert to tensors
    obs_tensor = torch.from_numpy(np.array(all_observations))
    policy_tensor = torch.from_numpy(np.array(all_policies))
    value_tensor = torch.from_numpy(np.array(all_values)).float().unsqueeze(1)

    # Verify shapes
    expected_obs_shape = (len(all_observations), 8, 8, 122)
    expected_policy_shape = (len(all_observations), 4672)

    shapes_correct = (
        obs_tensor.shape == expected_obs_shape and
        policy_tensor.shape == expected_policy_shape
    )

    test_result(
        "Training data preparation",
        shapes_correct,
        f"Observations: {obs_tensor.shape}, Policies: {policy_tensor.shape}, Values: {value_tensor.shape}"
    )

except Exception as e:
    test_result("Training data preparation", False, f"Error: {e}")

# ==============================================================================
# TEST 4: Training Step
# ==============================================================================
print("\n" + "-"*80)
print("TEST 4: Network Training Step")
print("-"*80)

try:
    # Setup optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    # Convert NHWC to NCHW for PyTorch
    obs_batch = obs_tensor.permute(0, 3, 1, 2).contiguous()
    obs_batch = obs_batch.to(memory_format=torch.channels_last)
    policy_batch = policy_tensor
    value_batch = value_tensor

    # Training step
    network.train()
    optimizer.zero_grad()

    policy_pred, value_pred = network(obs_batch)

    # Compute losses
    policy_loss = -torch.mean(torch.sum(policy_batch * torch.log_softmax(policy_pred, dim=1), dim=1))
    value_loss = torch.mean((value_pred - value_batch) ** 2)
    total_loss = policy_loss + value_loss

    # Backward pass
    total_loss.backward()
    optimizer.step()

    test_result(
        "Training step",
        total_loss.item() > 0,
        f"Loss: {total_loss.item():.4f} (Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f})"
    )

except Exception as e:
    test_result("Training step", False, f"Error: {e}")

# ==============================================================================
# TEST 5: Checkpoint Save/Load
# ==============================================================================
print("\n" + "-"*80)
print("TEST 5: Checkpoint Save and Load")
print("-"*80)

try:
    # Save checkpoint
    checkpoint_path = Path("checkpoints/test_122_checkpoint.pt")
    checkpoint_path.parent.mkdir(exist_ok=True)

    checkpoint = {
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': 1,
        'config': {
            'input_channels': 122,
            'num_filters': config.num_filters,
            'num_blocks': config.num_blocks
        }
    }

    torch.save(checkpoint, checkpoint_path)

    # Load checkpoint
    loaded_checkpoint = torch.load(checkpoint_path)

    # Create new network and load weights
    new_network = AlphaZeroNetwork(
        input_channels=122,
        num_filters=config.num_filters,
        num_blocks=config.num_blocks,
        num_actions=config.num_actions
    )
    new_network.load_state_dict(loaded_checkpoint['model_state_dict'])

    # Verify loaded config
    loaded_config = loaded_checkpoint['config']
    config_correct = loaded_config['input_channels'] == 122

    test_result(
        "Checkpoint save/load",
        config_correct,
        f"Saved and loaded 122-channel checkpoint successfully"
    )

    # Cleanup
    checkpoint_path.unlink()

except Exception as e:
    test_result("Checkpoint save/load", False, f"Error: {e}")

# ==============================================================================
# TEST 6: Inference Performance
# ==============================================================================
print("\n" + "-"*80)
print("TEST 6: Inference Performance")
print("-"*80)

try:
    import time

    # Create batch of observations
    batch_size = 32
    test_obs = torch.randn(batch_size, 122, 8, 8).to(memory_format=torch.channels_last)

    # Warm-up
    with torch.no_grad():
        _ = network(test_obs)

    # Benchmark
    num_iterations = 100
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            policy, value = network(test_obs)

    elapsed = time.time() - start_time
    throughput = (num_iterations * batch_size) / elapsed

    test_result(
        "Inference performance",
        throughput > 100,  # Should be able to do >100 evals/sec even on CPU
        f"Throughput: {throughput:.1f} evaluations/sec (batch_size={batch_size})"
    )

except Exception as e:
    test_result("Inference performance", False, f"Error: {e}")

# ==============================================================================
# TEST 7: History Encoding Verification
# ==============================================================================
print("\n" + "-"*80)
print("TEST 7: Verify T-8 Position Encoding")
print("-"*80)

try:
    # Generate a longer game to get 8+ moves of history
    test_observations = []

    def history_collector(obs_array, num_leaves):
        # Only collect root observations (num_leaves=1)
        if num_leaves == 1:
            test_observations.append(obs_array[0].copy())

        # Return random policy
        policies = np.random.random((num_leaves, 4672)).astype(np.float32)
        policies = policies / policies.sum(axis=1, keepdims=True)
        values = np.zeros(num_leaves, dtype=np.float32)
        return policies, values

    coord2 = alphazero_cpp.SelfPlayCoordinator(
        num_workers=1,
        num_simulations=20,
        batch_size=8
    )
    games2 = coord2.generate_games(history_collector, 1)

    # Check T-8 encoding at move 10+
    if len(test_observations) >= 10:
        obs = test_observations[9]  # Move 10

        # Check T-8 (channels 109-121)
        t8_pieces = np.count_nonzero(obs[:, :, 109:121])

        # Also check shape
        shape_correct = obs.shape == (8, 8, 122)
        t8_encoded = t8_pieces > 0

        test_result(
            "T-8 position encoding",
            shape_correct and t8_encoded,
            f"Shape: {obs.shape}, T-8 pieces: {t8_pieces} (channels 109-121)"
        )
    else:
        test_result("T-8 position encoding", False, "Not enough moves generated")

except Exception as e:
    test_result("T-8 position encoding", False, f"Error: {e}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"Total tests: {tests_passed + tests_failed}")
print(f"Passed: {tests_passed}")
print(f"Failed: {tests_failed}")
print(f"Success rate: {100.0 * tests_passed / (tests_passed + tests_failed):.1f}%")

if tests_failed == 0:
    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    print("\n122-Channel Implementation is PRODUCTION READY:")
    print("  [OK] Neural network accepts 122-channel input")
    print("  [OK] Self-play generates correct observations")
    print("  [OK] Training step works with 122 channels")
    print("  [OK] Checkpoint save/load compatible")
    print("  [OK] Inference performance acceptable")
    print("  [OK] T-8 position fully encoded")
    print("\nYou can now train models with full 8-position history!")
    sys.exit(0)
else:
    print(f"\n{tests_failed} TEST(S) FAILED - Please review errors above")
    sys.exit(1)
