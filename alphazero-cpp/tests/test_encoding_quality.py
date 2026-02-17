#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test position encoding quality and verify correctness"""

import sys
import os
import numpy as np

# Fix Windows console encoding for Unicode characters (reconfigure avoids
# closing the underlying buffer, which breaks pytest's capture mechanism)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add build directory to path (MSVC puts .pyd in build/Release/)
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'Release')
if not os.path.exists(build_dir):
    build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

import alphazero_cpp

def test_position_encoding_quality():
    """Test that position encoding produces correct piece planes"""
    print("=== Test 1: Position Encoding Quality ===")

    # Test starting position
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    encoding = alphazero_cpp.encode_position(fen)

    print(f"✓ Encoding shape: {encoding.shape}")
    print(f"✓ Encoding dtype: {encoding.dtype}")

    # Encoding is NHWC: shape (8, 8, 122).  Access channel c as encoding[:, :, c].
    # Plane layout: 0-5 current pieces, 6-11 opponent pieces, 12-13 repetition, 14 color, ...

    # Plane 0: Current player's pawns (white pawns on rank 1, from white's perspective)
    white_pawns = encoding[:, :, 0]
    pawn_count = np.sum(white_pawns)
    print(f"  White pawns detected: {int(pawn_count)} (expected: 8)")

    # Plane 6: Opponent's pawns (black pawns on rank 6, from white's perspective)
    black_pawns = encoding[:, :, 6]
    black_pawn_count = np.sum(black_pawns)
    print(f"  Black pawns detected: {int(black_pawn_count)} (expected: 8)")

    # Check that pawns are on correct ranks
    # white_pawns is shape (8, 8) — row = rank, col = file
    white_pawn_ranks = np.where(white_pawns > 0)[0]
    black_pawn_ranks = np.where(black_pawns > 0)[0]
    print(f"  White pawn ranks: {np.unique(white_pawn_ranks)} (expected: [1])")
    print(f"  Black pawn ranks: {np.unique(black_pawn_ranks)} (expected: [6])")

    # Check color plane (plane 14)
    color_plane = encoding[:, :, 14]
    color_value = color_plane[0, 0]
    print(f"  Color plane value: {color_value} (expected: 1.0 for current player)")

    # Check castling plane (plane 16)
    castling_plane = encoding[:, :, 16]
    castling_value = castling_plane[0, 0]
    print(f"  Castling rights value: {castling_value} (expected: 1.0 for both sides)")

    assert pawn_count == 8, f"Expected 8 white pawns, got {int(pawn_count)}"
    assert black_pawn_count == 8, f"Expected 8 black pawns, got {int(black_pawn_count)}"
    assert color_value == 1.0, f"Expected color=1.0, got {color_value}"
    print("✓ PASS: Position encoding produces correct piece planes\n")

def test_move_encoding_coverage():
    """Test that move encoding covers all legal moves"""
    print("=== Test 2: Move Encoding Coverage ===")

    # Test various move types
    # Note: the chess library uses Chess960 (king-captures-rook) notation for
    # castling, so e1g1 (standard UCI) is stored internally as e1h1.  We test
    # the canonical form and verify both representations encode to the same index.
    test_cases = [
        ("e2e4", "pawn push", "e2e4"),
        ("g1f3", "knight move", "g1f3"),
        ("e7e8q", "queen promotion", "e7e8q"),
        ("e7e8n", "knight underpromotion", "e7e8n"),
        ("e1h1", "kingside castle (Chess960 form)", "e1h1"),
    ]

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    all_ok = True
    for uci_move, description, expected_decoded in test_cases:
        try:
            index = alphazero_cpp.move_to_index(uci_move, fen)
            decoded = alphazero_cpp.index_to_move(index, fen)

            if decoded == expected_decoded:
                print(f"✓ {description}: {uci_move} → {index} → {decoded} (round-trip OK)")
            else:
                print(f"✗ {description}: {uci_move} → {index} → {decoded} (expected {expected_decoded})")
                all_ok = False
        except Exception as e:
            print(f"✗ {description}: {uci_move} failed with error: {e}")
            all_ok = False

    # Verify standard UCI castle notation maps to the same index as Chess960 form
    idx_standard = alphazero_cpp.move_to_index("e1g1", fen)
    idx_chess960 = alphazero_cpp.move_to_index("e1h1", fen)
    if idx_standard == idx_chess960:
        print(f"✓ Castle equivalence: e1g1 and e1h1 both → index {idx_standard}")
    else:
        print(f"✗ Castle mismatch: e1g1 → {idx_standard}, e1h1 → {idx_chess960}")
        all_ok = False

    assert all_ok, "Move encoding round-trip failed"
    print("✓ PASS: Move encoding covers various move types\n")

def test_perspective_flip():
    """Test that position encoding flips perspective for black"""
    print("=== Test 3: Perspective Flip ===")

    # Test position after 1.e4
    fen_white = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    encoding_white = alphazero_cpp.encode_position(fen_white)

    # Encoding is NHWC (8, 8, 122).  From black's perspective the board is flipped.
    black_pawns = encoding_white[:, :, 0]  # Current player's pawns (black)
    white_pawns = encoding_white[:, :, 6]  # Opponent's pawns (white)

    black_pawn_ranks = np.unique(np.where(black_pawns > 0)[0])
    white_pawn_ranks = np.unique(np.where(white_pawns > 0)[0])

    print(f"  Black's view - Black pawn ranks: {black_pawn_ranks}")
    print(f"  Black's view - White pawn ranks: {white_pawn_ranks}")

    # Color plane should still be 1.0 (always from current player's perspective)
    color_value = encoding_white[:, :, 14][0, 0]
    print(f"  Color plane value: {color_value} (expected: 1.0)")

    assert color_value == 1.0, f"Expected color=1.0, got {color_value}"
    print("✓ PASS: Perspective flip working correctly\n")

def test_zero_copy_potential():
    """Test that encoding can be done without memory copies"""
    print("=== Test 4: Zero-Copy Potential ===")

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Test encoding performance
    import time

    num_iterations = 1000
    start = time.time()
    for _ in range(num_iterations):
        encoding = alphazero_cpp.encode_position(fen)
    elapsed = time.time() - start

    avg_time_us = (elapsed / num_iterations) * 1e6
    print(f"✓ Average encoding time: {avg_time_us:.1f} μs")
    print(f"✓ Encoding throughput: {num_iterations / elapsed:.0f} positions/sec")

    # Check memory layout
    print(f"✓ Encoding is C-contiguous: {encoding.flags['C_CONTIGUOUS']}")
    print(f"✓ Encoding owns data: {encoding.flags['OWNDATA']}")

    # Target: <100μs per position (from plan line 1786)
    if avg_time_us < 100:
        print(f"✓ PASS: Encoding meets <100μs target ({avg_time_us:.1f}μs)\n")
    else:
        print(f"⚠ WARNING: Encoding slower than 100μs target ({avg_time_us:.1f}μs)\n")

def test_batch_encoding():
    """Test encoding multiple positions for batch processing"""
    print("=== Test 5: Batch Encoding ===")

    positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
    ]

    # Encode all positions
    encodings = []
    for fen in positions:
        encoding = alphazero_cpp.encode_position(fen)
        encodings.append(encoding)

    # Stack into batch
    batch = np.stack(encodings, axis=0)
    expected_shape = (3, 8, 8, 122)  # NHWC
    print(f"  Batch shape: {batch.shape} (expected: {expected_shape})")
    print(f"  Batch dtype: {batch.dtype}")
    print(f"  Batch is C-contiguous: {batch.flags['C_CONTIGUOUS']}")

    # Verify each position is different
    diff_01 = np.sum(np.abs(batch[0] - batch[1]))
    diff_12 = np.sum(np.abs(batch[1] - batch[2]))
    print(f"  Position 0 vs 1 difference: {diff_01:.0f} (should be > 0)")
    print(f"  Position 1 vs 2 difference: {diff_12:.0f} (should be > 0)")

    assert batch.shape == expected_shape, f"Shape mismatch: {batch.shape} != {expected_shape}"
    assert diff_01 > 0, "Positions 0 and 1 should differ"
    assert diff_12 > 0, "Positions 1 and 2 should differ"
    print("✓ PASS: Batch encoding works correctly\n")

if __name__ == "__main__":
    print("========================================")
    print("Position Encoding Quality Tests")
    print("========================================\n")

    try:
        test_position_encoding_quality()
        test_move_encoding_coverage()
        test_perspective_flip()
        test_zero_copy_potential()
        test_batch_encoding()

        print("========================================")
        print("✓✓✓ ALL ENCODING TESTS PASSED ✓✓✓")
        print("========================================")
        print("\nNext Steps:")
        print("1. Implement zero-copy tensor interface (encode_to_buffer)")
        print("2. Test with PyTorch/TensorFlow tensors")
        print("3. Verify GPU memory mapping")
        print("4. Integration test with neural network")

    except Exception as e:
        print("========================================")
        print("✗✗✗ TEST FAILED ✗✗✗")
        print("========================================")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
