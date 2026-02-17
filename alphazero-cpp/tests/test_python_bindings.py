#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test Python bindings for AlphaZero chess engine."""

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

def test_module_info():
    """Test module version and basic info"""
    print("=== Test 1: Module Info ===")
    print(f"Module version: {alphazero_cpp.__version__}")
    print(f"Module name: {alphazero_cpp.__name__}")
    print("✓ PASS: Module loaded successfully\n")

def test_position_encoding():
    """Test position encoding functionality"""
    print("=== Test 2: Position Encoding ===")

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Test single-arg form
    encoding = alphazero_cpp.encode_position(fen)
    print(f"✓ Encoding shape: {encoding.shape}")
    print(f"✓ Encoding dtype: {encoding.dtype}")
    print(f"✓ Expected shape: (8, 8, 122) [NHWC]")

    assert encoding.shape == (8, 8, 122), f"Shape mismatch: {encoding.shape} != (8, 8, 122)"
    assert encoding.dtype == np.float32, f"Dtype mismatch: {encoding.dtype}"

    # Test two-arg form with history FENs
    history = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ]
    encoding2 = alphazero_cpp.encode_position(fen, history)
    assert encoding2.shape == (8, 8, 122), f"Two-arg shape mismatch: {encoding2.shape}"
    print("✓ Two-arg encode_position(fen, history_fens) works")

    print("✓ PASS: Position encoding works\n")

def test_move_conversion():
    """Test move conversion utilities"""
    print("=== Test 3: Move Conversion ===")

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Test move to index (requires FEN for perspective handling)
    index = alphazero_cpp.move_to_index("e2e4", fen)
    print(f"✓ Move 'e2e4' -> index {index}")
    assert 0 <= index < 4672, f"Index {index} out of range"

    # Test index to move (round-trip)
    move = alphazero_cpp.index_to_move(index, fen)
    print(f"✓ Index {index} -> move '{move}'")
    assert move == "e2e4", f"Round-trip failed: expected 'e2e4', got '{move}'"

    print("✓ PASS: Move conversion works\n")

def test_wdl_to_value():
    """Test WDL to scalar value conversion"""
    print("=== Test 4: WDL to Value ===")

    # Pure win: pw=1 → value = 1-0 = 1
    v = alphazero_cpp.wdl_to_value(1.0, 0.0, 0.0)
    assert abs(v - 1.0) < 1e-6, f"Pure win: expected 1.0, got {v}"

    # Pure loss: pl=1 → value = 0-1 = -1
    v = alphazero_cpp.wdl_to_value(0.0, 0.0, 1.0)
    assert abs(v - (-1.0)) < 1e-6, f"Pure loss: expected -1.0, got {v}"

    # Pure draw: → value = 0-0 = 0 (risk adjustment at node level, not here)
    v = alphazero_cpp.wdl_to_value(0.0, 1.0, 0.0)
    assert abs(v) < 1e-6, f"Draw: expected 0.0, got {v}"

    # Mixed WDL: pw=0.7, pd=0.2, pl=0.1 → value = 0.7-0.1 = 0.6
    v = alphazero_cpp.wdl_to_value(0.7, 0.2, 0.1)
    assert abs(v - 0.6) < 1e-6, f"Mixed WDL: expected 0.6, got {v}"

    print("✓ PASS: WDL to value works\n")

if __name__ == "__main__":
    print("========================================")
    print("Python Bindings Test Suite")
    print("========================================\n")

    try:
        test_module_info()
        test_position_encoding()
        test_move_conversion()
        test_wdl_to_value()

        print("========================================")
        print("✓✓✓ ALL PYTHON BINDING TESTS PASSED ✓✓✓")
        print("========================================")

    except Exception as e:
        print("========================================")
        print("✗✗✗ TEST FAILED ✗✗✗")
        print("========================================")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
