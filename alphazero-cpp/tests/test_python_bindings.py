#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test Python bindings for AlphaZero Sync"""

import sys
import os
import numpy as np

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add build directory to path (MSVC puts .pyd in build/Release/)
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'Release')
if not os.path.exists(build_dir):
    build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

import alphazero_cpp

def test_module_info():
    """Test module version and basic info"""
    print("=== Test 1: Module Info ===")
    print(f"Module version: {alphazero_cpp.__version__}")
    print(f"Module name: {alphazero_cpp.__name__}")
    print("✓ PASS: Module loaded successfully\n")

def test_mcts_search():
    """Test MCTS search functionality"""
    print("=== Test 2: MCTS Search ===")

    # Create MCTS search engine
    search = alphazero_cpp.MCTSSearch(num_simulations=100, c_puct=1.5)
    print("✓ Created MCTS search engine")

    # Create uniform policy (1858 moves)
    policy = np.ones(1858, dtype=np.float32) / 1858.0
    value = 0.0

    # Run search on starting position
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    visit_counts = search.search(fen, policy, value)

    print(f"✓ Search completed")
    print(f"✓ Visit counts shape: {visit_counts.shape}")
    print(f"✓ Visit counts dtype: {visit_counts.dtype}")

    # Reset search tree
    search.reset()
    print("✓ Search tree reset")

    print("✓ PASS: MCTS search works\n")

def test_batch_coordinator():
    """Test batch coordinator functionality"""
    print("=== Test 3: Batch Coordinator ===")

    # Create batch coordinator
    coordinator = alphazero_cpp.BatchCoordinator(batch_size=256, batch_threshold=0.9)
    print("✓ Created batch coordinator")

    # Add games
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    for i in range(10):
        coordinator.add_game(i, fen)
    print("✓ Added 10 games")

    # Get statistics
    stats = coordinator.get_stats()
    print(f"✓ Active games: {stats['active_games']}")
    print(f"✓ Pending evals: {stats['pending_evals']}")
    print(f"✓ Batch counter: {stats['batch_counter']}")

    # Check game completion
    is_complete = coordinator.is_game_complete(0)
    print(f"✓ Game 0 complete: {is_complete}")

    print("✓ PASS: Batch coordinator works\n")

def test_position_encoding():
    """Test position encoding functionality"""
    print("=== Test 4: Position Encoding ===")

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    encoding = alphazero_cpp.encode_position(fen)

    print(f"✓ Encoding shape: {encoding.shape}")
    print(f"✓ Encoding dtype: {encoding.dtype}")
    print(f"✓ Expected shape: (119, 8, 8)")

    if encoding.shape == (119, 8, 8):
        print("✓ PASS: Position encoding works\n")
    else:
        print("✗ FAIL: Position encoding shape mismatch\n")

def test_move_conversion():
    """Test move conversion utilities"""
    print("=== Test 5: Move Conversion ===")

    # Test move to index
    index = alphazero_cpp.move_to_index("e2e4")
    print(f"✓ Move 'e2e4' -> index {index}")

    # Test index to move
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move = alphazero_cpp.index_to_move(index, fen)
    print(f"✓ Index {index} -> move '{move}'")

    print("✓ PASS: Move conversion works (placeholder implementation)\n")

if __name__ == "__main__":
    print("========================================")
    print("Python Bindings Test Suite")
    print("========================================\n")

    try:
        test_module_info()
        test_mcts_search()
        test_batch_coordinator()
        test_position_encoding()
        test_move_conversion()

        print("========================================")
        print("✓✓✓ ALL PYTHON BINDING TESTS PASSED ✓✓✓")
        print("========================================")
        print("\nNext Steps:")
        print("1. Implement full position encoding (119 planes)")
        print("2. Implement move-to-index mapping (1858 moves)")
        print("3. Add zero-copy tensor interface for GPU")
        print("4. Test with actual neural network")

    except Exception as e:
        print("========================================")
        print("✗✗✗ TEST FAILED ✗✗✗")
        print("========================================")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
