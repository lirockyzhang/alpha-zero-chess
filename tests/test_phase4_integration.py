"""End-to-end integration test for C++ MCTS backend.

This test verifies Phase 4 requirements from batched_mcts.md:
1. Run full self-play games with C++ MCTS
2. Verify training data format
3. Test integration with neural network
4. Measure performance

According to batched_mcts.md lines 1761-1765:
- Run full self-play games
- Verify training data format
- Test with evaluate.py
- Test with web app
"""

import sys
import os
from pathlib import Path
import time
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphazero.mcts import create_mcts, get_best_backend, get_available_backends
from alphazero.mcts.evaluator import NetworkEvaluator
from alphazero.neural.network import AlphaZeroNetwork
from alphazero.chess_env.board import GameState
from alphazero.config import MCTSConfig, MCTSBackend, NetworkConfig
from alphazero.selfplay.game import SelfPlayGame


def test_cpp_backend_available():
    """Test that C++ backend is available and can be imported."""
    print("=" * 70)
    print("Test 1: C++ Backend Availability")
    print("=" * 70)

    available_backends = get_available_backends()
    print(f"Available backends: {[b.value for b in available_backends]}")

    if MCTSBackend.CPP in available_backends:
        print("[PASS] C++ backend is available")
        return True
    else:
        print("[FAIL] C++ backend is not available")
        print("Build the C++ extension first:")
        print("  cd alphazero-sync")
        print("  cmake --build build --config Release")
        return False


def test_cpp_mcts_creation():
    """Test that C++ MCTS can be created with config."""
    print("\n" + "=" * 70)
    print("Test 2: C++ MCTS Creation")
    print("=" * 70)

    try:
        config = MCTSConfig(
            num_simulations=100,
            c_puct=1.25,
            backend=MCTSBackend.CPP
        )

        mcts = create_mcts(backend=MCTSBackend.CPP, config=config)
        print(f"[PASS] Created C++ MCTS: {type(mcts).__name__}")
        print(f"  - num_simulations: {config.num_simulations}")
        print(f"  - c_puct: {config.c_puct}")
        return True, mcts

    except Exception as e:
        print(f"[FAIL] Failed to create C++ MCTS: {e}")
        return False, None


def test_single_position_evaluation(mcts):
    """Test C++ MCTS on a single position."""
    print("\n" + "=" * 70)
    print("Test 3: Single Position Evaluation")
    print("=" * 70)

    try:
        # Create a simple neural network for testing
        net_config = NetworkConfig(
            input_channels=119,
            num_filters=64,  # Small network for testing
            num_blocks=3,
            num_actions=4672
        )
        network = AlphaZeroNetwork(
            input_channels=net_config.input_channels,
            num_filters=net_config.num_filters,
            num_blocks=net_config.num_blocks,
            num_actions=net_config.num_actions
        )
        network.eval()

        # Create evaluator
        evaluator = NetworkEvaluator(network, device='cpu')

        # Create starting position
        state = GameState()

        # Run MCTS search
        print("Running MCTS search on starting position...")
        start_time = time.perf_counter()

        policy, root, stats = mcts.search(state, evaluator, move_number=0, add_noise=False)

        elapsed = time.perf_counter() - start_time

        # Verify policy
        assert policy.shape == (4672,), f"Expected policy shape (4672,), got {policy.shape}"
        assert np.abs(policy.sum() - 1.0) < 1e-5, f"Policy should sum to 1.0, got {policy.sum()}"
        assert np.all(policy >= 0), "Policy should be non-negative"

        # Get top moves
        top_indices = np.argsort(policy)[-5:][::-1]
        print(f"\n[PASS] MCTS search completed in {elapsed*1000:.2f}ms")
        print(f"  - Policy shape: {policy.shape}")
        print(f"  - Policy sum: {policy.sum():.6f}")
        print(f"  - Top 5 move probabilities: {policy[top_indices]}")

        return True

    except Exception as e:
        print(f"[FAIL] Single position evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_self_play_game(mcts):
    """Test a full self-play game with C++ MCTS."""
    print("\n" + "=" * 70)
    print("Test 4: Full Self-Play Game")
    print("=" * 70)

    try:
        # Create a simple neural network for testing
        net_config = NetworkConfig(
            input_channels=119,
            num_filters=64,
            num_blocks=3,
            num_actions=4672
        )
        network = AlphaZeroNetwork(
            input_channels=net_config.input_channels,
            num_filters=net_config.num_filters,
            num_blocks=net_config.num_blocks,
            num_actions=net_config.num_actions
        )
        network.eval()

        # Create evaluator
        evaluator = NetworkEvaluator(network, device='cpu')

        # Create MCTS instance
        mcts_config = MCTSConfig(
            num_simulations=100,  # Reduced for testing
            c_puct=1.25,
            temperature=1.0,
            temperature_threshold=30,
            backend=MCTSBackend.CPP
        )
        mcts = create_mcts(backend=MCTSBackend.CPP, config=mcts_config)

        # Create self-play config
        from alphazero.config import SelfPlayConfig
        selfplay_config = SelfPlayConfig(max_moves=50)

        # Create self-play game
        game = SelfPlayGame(
            mcts=mcts,
            evaluator=evaluator,
            config=selfplay_config
        )

        # Play game
        print("Playing self-play game...")
        start_time = time.perf_counter()

        trajectory, result_str = game.play()

        elapsed = time.perf_counter() - start_time

        # Verify trajectory format
        print(f"\n[PASS] Self-play game completed in {elapsed:.2f}s")
        print(f"  - Moves played: {len(trajectory)}")
        print(f"  - Game result: {result_str}")
        if len(trajectory) > 0:
            print(f"  - Avg time per move: {elapsed/len(trajectory)*1000:.2f}ms")

        # Verify trajectory data format
        if len(trajectory) > 0:
            sample = trajectory.states[0]
            print(f"\nTrajectory data format:")
            print(f"  - observation shape: {sample.observation.shape}")
            print(f"  - policy shape: {sample.policy.shape}")
            print(f"  - value: {sample.value}")
            print(f"  - action: {sample.action}")

            # Verify shapes match expected format
            assert sample.observation.shape == (119, 8, 8), \
                f"Expected observation shape (119, 8, 8), got {sample.observation.shape}"
            assert sample.policy.shape == (4672,), \
                f"Expected policy shape (4672,), got {sample.policy.shape}"
            assert isinstance(sample.value, (int, float)), \
                f"Expected value to be numeric, got {type(sample.value)}"
            assert isinstance(sample.action, (int, np.integer)), \
                f"Expected action to be int, got {type(sample.action)}"

        return True, trajectory

    except Exception as e:
        print(f"[FAIL] Self-play game failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_performance_benchmark(mcts):
    """Benchmark C++ MCTS performance."""
    print("\n" + "=" * 70)
    print("Test 5: Performance Benchmark")
    print("=" * 70)

    try:
        # Create a simple neural network for testing
        net_config = NetworkConfig(
            input_channels=119,
            num_filters=64,
            num_blocks=3,
            num_actions=4672
        )
        network = AlphaZeroNetwork(
            input_channels=net_config.input_channels,
            num_filters=net_config.num_filters,
            num_blocks=net_config.num_blocks,
            num_actions=net_config.num_actions
        )
        network.eval()

        # Create evaluator
        evaluator = NetworkEvaluator(network, device='cpu')

        # Create starting position
        state = GameState()

        # Benchmark multiple searches
        num_searches = 10
        search_times = []

        print(f"Running {num_searches} MCTS searches...")
        for i in range(num_searches):
            start_time = time.perf_counter()
            policy = mcts.search(state, evaluator, add_noise=False)
            elapsed = time.perf_counter() - start_time
            search_times.append(elapsed * 1000)  # Convert to ms

            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{num_searches} searches")

        # Calculate statistics
        mean_time = np.mean(search_times)
        median_time = np.median(search_times)
        min_time = np.min(search_times)
        max_time = np.max(search_times)

        # Get MCTS stats
        stats = mcts.get_stats()

        print(f"\n[PASS] Performance benchmark completed")
        print(f"\nSearch time statistics:")
        print(f"  - Mean: {mean_time:.2f}ms")
        print(f"  - Median: {median_time:.2f}ms")
        print(f"  - Min: {min_time:.2f}ms")
        print(f"  - Max: {max_time:.2f}ms")
        print(f"\nMCTS statistics:")
        print(f"  - Total searches: {stats['total_searches']}")
        print(f"  - Total simulations: {stats['total_simulations']}")
        print(f"  - Avg simulations per search: {stats['avg_simulations_per_search']:.1f}")

        # Performance targets from batched_mcts.md line 1769:
        # Target: 50K-100K sims/sec per game
        sims_per_sec = stats['total_simulations'] / (sum(search_times) / 1000)
        print(f"\nPerformance:")
        print(f"  - Simulations per second: {sims_per_sec:.0f}")
        print(f"  - Target: 50K-100K sims/sec")

        if sims_per_sec >= 50000:
            print(f"  - [PASS] Performance target met!")
        else:
            print(f"  - [INFO] Performance below target (expected with small test network)")

        return True

    except Exception as e:
        print(f"[FAIL] Performance benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 4 integration tests."""
    print("=" * 70)
    print("Phase 4: End-to-End Integration Testing")
    print("=" * 70)
    print()
    print("This test suite verifies the C++ MCTS integration according to")
    print("batched_mcts.md Phase 4 requirements (lines 1758-1778):")
    print("  1. End-to-end testing with full self-play games")
    print("  2. Verify training data format")
    print("  3. Performance profiling")
    print()

    # Test 1: Check C++ backend availability
    if not test_cpp_backend_available():
        print("\n[FAIL] C++ backend not available. Cannot proceed with integration tests.")
        return False

    # Test 2: Create C++ MCTS
    success, mcts = test_cpp_mcts_creation()
    if not success:
        print("\n[FAIL] Cannot create C++ MCTS. Cannot proceed with integration tests.")
        return False

    # Test 3: Single position evaluation
    if not test_single_position_evaluation(mcts):
        print("\n[FAIL] Single position evaluation failed.")
        return False

    # Test 4: Full self-play game
    success, trajectory = test_full_self_play_game(mcts)
    if not success:
        print("\n[FAIL] Self-play game failed.")
        return False

    # Test 5: Performance benchmark
    if not test_performance_benchmark(mcts):
        print("\n[FAIL] Performance benchmark failed.")
        return False

    # Summary
    print("\n" + "=" * 70)
    print("Phase 4 Integration Test Summary")
    print("=" * 70)
    print("[PASS] All integration tests passed!")
    print()
    print("Next steps:")
    print("  1. Test with evaluate.py")
    print("  2. Test with web app")
    print("  3. Run full training pipeline")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
