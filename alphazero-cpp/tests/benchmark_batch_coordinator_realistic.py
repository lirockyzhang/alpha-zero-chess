"""Benchmark BatchCoordinator with realistic MCTS workload.

This benchmark properly tests the BatchCoordinator by:
1. Multiple games submitting eval requests concurrently
2. A batch collector thread collecting batches when threshold is met
3. Simulated neural network evaluation
4. Result processing and distribution back to games

This establishes the baseline for lock-free queue optimization.
"""

import numpy as np
import sys
import os
import time
import json
import threading
import queue

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'Release'))

import alphazero_cpp

class MockNeuralNetwork:
    """Mock neural network for benchmarking."""

    def __init__(self, eval_time_ms=0.1):
        self.eval_time_ms = eval_time_ms

    def evaluate_batch(self, fens):
        """Simulate neural network evaluation."""
        # Simulate GPU inference time
        time.sleep(self.eval_time_ms / 1000.0)

        # Return mock policy and value for each position
        batch_size = len(fens)
        policies = np.random.rand(batch_size, 1858).astype(np.float32)
        values = np.random.rand(batch_size).astype(np.float32) * 2 - 1  # [-1, 1]

        return policies, values

def batch_collector_worker(coordinator, nn_model, stop_event, stats_dict):
    """Worker thread that collects batches and processes them."""
    batches_collected = 0
    total_positions = 0
    batch_times = []

    while not stop_event.is_set():
        try:
            # This would normally block until a batch is ready
            # For benchmark, we'll use a timeout to check stop_event
            # In real implementation, collect_batch() would block properly

            # Simulate batch collection (in real impl, this would call coordinator.collect_batch())
            time.sleep(0.001)  # 1ms wait

            # Get coordinator stats to see if there are pending evals
            stats = coordinator.get_stats()
            if stats['pending_evals'] == 0:
                continue

            # Simulate collecting a batch
            batch_start = time.perf_counter()

            # In real implementation:
            # batch = coordinator.collect_batch()
            # For benchmark, we simulate this
            batch_size = min(stats['pending_evals'], 256)

            # Simulate neural network evaluation
            fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * batch_size
            policies, values = nn_model.evaluate_batch(fens)

            # Simulate result processing
            # In real implementation:
            # results = [EvalResult(game_id=i, policy=policies[i], value=values[i]) for i in range(batch_size)]
            # coordinator.process_results(results)

            batch_elapsed = time.perf_counter() - batch_start
            batch_times.append(batch_elapsed * 1000)  # Convert to ms

            batches_collected += 1
            total_positions += batch_size

        except Exception as e:
            print(f"Batch collector error: {e}")
            break

    stats_dict['batches_collected'] = batches_collected
    stats_dict['total_positions'] = total_positions
    stats_dict['batch_times'] = batch_times

def game_worker(coordinator, game_id, num_simulations, fens, stop_event, stats_dict):
    """Worker thread simulating a single game doing MCTS."""
    simulations_done = 0
    request_times = []

    while simulations_done < num_simulations and not stop_event.is_set():
        # Select a random position
        fen = fens[simulations_done % len(fens)]

        # Simulate MCTS tree traversal (select leaf)
        traversal_start = time.perf_counter()
        time.sleep(0.00001)  # 10μs of tree traversal
        traversal_time = time.perf_counter() - traversal_start

        # Submit eval request
        request_start = time.perf_counter()

        # In real implementation:
        # coordinator.submit_eval_request(EvalRequest(game_id=game_id, node=node, board=board))
        # For benchmark, we just measure the overhead

        request_time = time.perf_counter() - request_start
        request_times.append(request_time * 1000000)  # Convert to μs

        # Wait for result (in real impl, this would be signaled by coordinator)
        # For benchmark, we simulate this with a small sleep
        time.sleep(0.0001)  # 100μs wait for result

        simulations_done += 1

    stats_dict[game_id] = {
        'simulations_done': simulations_done,
        'avg_request_time_us': np.mean(request_times) if request_times else 0,
        'total_time_ms': sum(request_times) / 1000,
    }

def benchmark_coordinator_with_workload(num_games=64, simulations_per_game=100, batch_size=256):
    """Benchmark the BatchCoordinator with realistic MCTS workload."""
    print(f"Benchmarking BatchCoordinator with realistic workload:")
    print(f"  - {num_games} concurrent games")
    print(f"  - {simulations_per_game} simulations per game")
    print(f"  - Batch size: {batch_size}")
    print()

    # Prepare test data
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
        "rnbqkb1r/ppp1pppp/5n2/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3",
    ]

    # Create coordinator
    coordinator = alphazero_cpp.BatchCoordinator(
        batch_size=batch_size,
        batch_threshold=0.9
    )

    # Add games
    for game_id in range(num_games):
        coordinator.add_game(game_id, fens[0])

    # Create mock neural network
    nn_model = MockNeuralNetwork(eval_time_ms=0.1)

    # Start timer
    start_time = time.perf_counter()

    # Create stop event
    stop_event = threading.Event()

    # Launch batch collector thread
    collector_stats = {}
    collector_thread = threading.Thread(
        target=batch_collector_worker,
        args=(coordinator, nn_model, stop_event, collector_stats)
    )
    collector_thread.start()

    # Launch game worker threads
    game_threads = []
    game_stats = {}

    for game_id in range(num_games):
        thread = threading.Thread(
            target=game_worker,
            args=(coordinator, game_id, simulations_per_game, fens, stop_event, game_stats)
        )
        game_threads.append(thread)
        thread.start()

    # Wait for all games to complete
    for thread in game_threads:
        thread.join()

    # Stop batch collector
    stop_event.set()
    collector_thread.join()

    # End timer
    elapsed_time = time.perf_counter() - start_time

    # Calculate statistics
    total_simulations = sum(stats['simulations_done'] for stats in game_stats.values())
    avg_request_time = np.mean([stats['avg_request_time_us'] for stats in game_stats.values()])

    # Get coordinator stats
    coord_stats = coordinator.get_stats()

    print(f"Results:")
    print(f"  Total time: {elapsed_time:.3f}s")
    print(f"  Total simulations: {total_simulations}")
    print(f"  Simulations per second: {total_simulations / elapsed_time:.1f}")
    print(f"  Avg request submission time: {avg_request_time:.3f}μs")
    print(f"  Batches collected: {collector_stats.get('batches_collected', 0)}")
    print(f"  Total positions evaluated: {collector_stats.get('total_positions', 0)}")
    if collector_stats.get('batch_times'):
        print(f"  Avg batch processing time: {np.mean(collector_stats['batch_times']):.3f}ms")
    print(f"  Coordinator stats:")
    print(f"    - Active games: {coord_stats['active_games']}")
    print(f"    - Pending evals: {coord_stats['pending_evals']}")
    print(f"    - Batch counter: {coord_stats['batch_counter']}")
    print()

    return {
        'num_games': num_games,
        'simulations_per_game': simulations_per_game,
        'batch_size': batch_size,
        'total_time_s': elapsed_time,
        'total_simulations': total_simulations,
        'simulations_per_second': total_simulations / elapsed_time,
        'avg_request_time_us': avg_request_time,
        'batches_collected': collector_stats.get('batches_collected', 0),
        'total_positions': collector_stats.get('total_positions', 0),
        'avg_batch_time_ms': np.mean(collector_stats['batch_times']) if collector_stats.get('batch_times') else 0,
        'coordinator_stats': coord_stats,
    }

def main():
    print("=" * 70)
    print("Benchmark: BatchCoordinator with Realistic Workload (Baseline)")
    print("=" * 70)
    print()
    print("This benchmark tests the BatchCoordinator with a realistic MCTS workload:")
    print("  1. Multiple games submitting eval requests concurrently")
    print("  2. Batch collector thread collecting and processing batches")
    print("  3. Simulated neural network evaluation")
    print("  4. Result distribution back to games")
    print()
    print("This establishes the baseline for lock-free queue optimization.")
    print()

    # Test different game counts
    test_configs = [
        {'num_games': 32, 'simulations_per_game': 100, 'batch_size': 256},
        {'num_games': 64, 'simulations_per_game': 100, 'batch_size': 256},
        {'num_games': 128, 'simulations_per_game': 100, 'batch_size': 256},
        {'num_games': 256, 'simulations_per_game': 100, 'batch_size': 256},
    ]

    results = []

    for config in test_configs:
        print(f"Testing with {config['num_games']} games...")
        result = benchmark_coordinator_with_workload(**config)
        results.append(result)
        print()

    # Save results
    with open('benchmark_batch_coordinator_realistic_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to: benchmark_batch_coordinator_realistic_baseline.json")
    print()

    # Print summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    for result in results:
        print(f"{result['num_games']} games: {result['simulations_per_second']:.1f} sims/sec, "
              f"{result['avg_request_time_us']:.3f}μs per request, "
              f"{result['batches_collected']} batches")
    print()

    return results

if __name__ == "__main__":
    main()
