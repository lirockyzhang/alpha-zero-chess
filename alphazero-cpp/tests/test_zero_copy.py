#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test zero-copy tensor interface and verify no memory copies"""

import sys
import os
import numpy as np
import time

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

def test_zero_copy_interface():
    """Test that encode_position_to_buffer writes directly without copying"""
    print("=== Test 1: Zero-Copy Interface ===")

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Pre-allocate buffer
    buffer = np.zeros((8, 8, 122), dtype=np.float32)

    # Get buffer address before encoding
    buffer_id_before = id(buffer)
    buffer_ptr_before = buffer.__array_interface__['data'][0]

    # Encode directly to buffer (zero-copy)
    alphazero_cpp.encode_position_to_buffer(fen, buffer)

    # Get buffer address after encoding
    buffer_id_after = id(buffer)
    buffer_ptr_after = buffer.__array_interface__['data'][0]

    print(f"  Buffer ID before: {buffer_id_before}")
    print(f"  Buffer ID after: {buffer_id_after}")
    print(f"  Buffer pointer before: {hex(buffer_ptr_before)}")
    print(f"  Buffer pointer after: {hex(buffer_ptr_after)}")

    # Verify no copy occurred
    assert buffer_id_before == buffer_id_after, "Buffer object ID changed"
    assert buffer_ptr_before == buffer_ptr_after, "Buffer data pointer changed — not zero-copy"

    # Verify encoding is correct (NHWC: buffer[rank, file, channel])
    white_pawns = buffer[:, :, 0]  # Channel 0 = current player's pawns
    pawn_count = np.sum(white_pawns)
    assert pawn_count == 8, f"Expected 8 white pawns, got {int(pawn_count)}"
    print("  PASS: Zero-copy verified, encoding correct")

def test_batch_zero_copy_performance():
    """Test zero-copy performance for batch processing (256 positions)"""
    print("=== Test 2: Batch Zero-Copy Performance ===")

    batch_size = 256
    positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
    ] * (batch_size // 3 + 1)
    positions = positions[:batch_size]

    # Pre-allocate batch buffer
    batch_buffer = np.zeros((batch_size, 8, 8, 122), dtype=np.float32)

    # Benchmark zero-copy encoding
    start = time.perf_counter()
    for i, fen in enumerate(positions):
        alphazero_cpp.encode_position_to_buffer(fen, batch_buffer[i])
    elapsed = time.perf_counter() - start

    elapsed_ms = elapsed * 1000
    per_position_us = (elapsed / batch_size) * 1e6

    print(f"  Batch size: {batch_size} positions")
    print(f"  Total time: {elapsed_ms:.2f} ms")
    print(f"  Per-position time: {per_position_us:.1f} us")
    print(f"  Throughput: {batch_size / elapsed:.0f} positions/sec")

def test_memory_copy_comparison():
    """Compare zero-copy vs copy performance"""
    print("=== Test 3: Zero-Copy vs Copy Comparison ===")

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    num_iterations = 10000

    # Test 1: With copy (encode_position returns new array)
    start = time.perf_counter()
    for _ in range(num_iterations):
        encoding = alphazero_cpp.encode_position(fen)
    elapsed_copy = time.perf_counter() - start

    # Test 2: Zero-copy (encode_position_to_buffer writes to existing buffer)
    buffer = np.zeros((8, 8, 122), dtype=np.float32)
    start = time.perf_counter()
    for _ in range(num_iterations):
        alphazero_cpp.encode_position_to_buffer(fen, buffer)
    elapsed_zero_copy = time.perf_counter() - start

    copy_us = (elapsed_copy / num_iterations) * 1e6
    zero_copy_us = (elapsed_zero_copy / num_iterations) * 1e6
    speedup = elapsed_copy / elapsed_zero_copy

    print(f"  With copy: {copy_us:.1f} us per position")
    print(f"  Zero-copy: {zero_copy_us:.1f} us per position")
    print(f"  Speedup: {speedup:.2f}x faster")

def test_gpu_memory_compatibility():
    """Test that encoding is compatible with GPU tensor libraries"""
    print("=== Test 4: GPU Memory Compatibility ===")

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Create buffer with specific memory layout (C-contiguous, float32)
    buffer = np.zeros((8, 8, 122), dtype=np.float32, order='C')

    # Encode to buffer
    alphazero_cpp.encode_position_to_buffer(fen, buffer)

    # Verify memory layout is GPU-compatible
    print(f"  Buffer dtype: {buffer.dtype} (expected: float32)")
    print(f"  Buffer shape: {buffer.shape} (expected: (8, 8, 122))")
    print(f"  Buffer is C-contiguous: {buffer.flags['C_CONTIGUOUS']}")
    print(f"  Buffer is aligned: {buffer.flags['ALIGNED']}")

    # Check strides for (8, 8, 122) C-contiguous float32
    expected_strides = (8 * 122 * 4, 122 * 4, 4)  # (3904, 488, 4) bytes
    print(f"  Buffer strides: {buffer.strides} (expected: {expected_strides})")

    assert buffer.dtype == np.float32, f"Expected float32, got {buffer.dtype}"
    assert buffer.flags['C_CONTIGUOUS'], "Buffer is not C-contiguous"
    assert buffer.flags['ALIGNED'], "Buffer is not aligned"
    print("  PASS: Buffer is GPU-compatible")

def test_concurrent_encoding():
    """Test that multiple buffers can be encoded concurrently"""
    print("=== Test 5: Concurrent Encoding ===")

    positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 3",
    ]

    # Create separate buffers for each position
    buffers = [np.zeros((8, 8, 122), dtype=np.float32) for _ in positions]

    # Encode all positions
    for i, fen in enumerate(positions):
        alphazero_cpp.encode_position_to_buffer(fen, buffers[i])

    # Verify each buffer is different
    for i in range(len(buffers) - 1):
        diff = np.sum(np.abs(buffers[i] - buffers[i+1]))
        print(f"  Buffer {i} vs {i+1} difference: {diff:.0f}")
        assert diff > 0, f"Buffer {i} and {i+1} are identical"
    print("  PASS: Concurrent encoding produces different results")

if __name__ == "__main__":
    print("========================================")
    print("Zero-Copy Tensor Interface Tests")
    print("========================================\n")

    tests = [
        test_zero_copy_interface,
        test_batch_zero_copy_performance,
        test_memory_copy_comparison,
        test_gpu_memory_compatibility,
        test_concurrent_encoding,
    ]
    for t in tests:
        try:
            t()
            print()
        except Exception as e:
            print(f"FAILED: {e}\n")
            sys.exit(1)
    print("ALL ZERO-COPY TESTS PASSED")
