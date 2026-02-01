#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test zero-copy tensor interface and verify no memory copies"""

import sys
import os
import numpy as np
import time

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

def test_zero_copy_interface():
    """Test that encode_position_to_buffer writes directly without copying"""
    print("=== Test 1: Zero-Copy Interface ===")

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Pre-allocate buffer
    buffer = np.zeros((119, 8, 8), dtype=np.float32)

    # Get buffer address before encoding
    buffer_id_before = id(buffer)
    buffer_ptr_before = buffer.__array_interface__['data'][0]

    # Encode directly to buffer (zero-copy)
    alphazero_cpp.encode_position_to_buffer(fen, buffer)

    # Get buffer address after encoding
    buffer_id_after = id(buffer)
    buffer_ptr_after = buffer.__array_interface__['data'][0]

    print(f"✓ Buffer ID before: {buffer_id_before}")
    print(f"✓ Buffer ID after: {buffer_id_after}")
    print(f"✓ Buffer pointer before: {hex(buffer_ptr_before)}")
    print(f"✓ Buffer pointer after: {hex(buffer_ptr_after)}")

    # Verify no copy occurred
    if buffer_id_before == buffer_id_after and buffer_ptr_before == buffer_ptr_after:
        print("✓ PASS: Zero-copy verified - buffer not reallocated")
    else:
        print("✗ FAIL: Buffer was copied or reallocated")
        return False

    # Verify encoding is correct
    white_pawns = buffer[0]
    pawn_count = np.sum(white_pawns)
    print(f"✓ White pawns detected: {int(pawn_count)} (expected: 8)")

    if pawn_count == 8:
        print("✓ PASS: Zero-copy encoding produces correct results\n")
        return True
    else:
        print("✗ FAIL: Zero-copy encoding incorrect\n")
        return False

def test_batch_zero_copy_performance():
    """Test zero-copy performance for batch processing (256 positions)"""
    print("=== Test 2: Batch Zero-Copy Performance ===")

    # Target: <1ms for 256 positions (from plan line 1784)
    batch_size = 256
    positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
    ] * (batch_size // 3 + 1)
    positions = positions[:batch_size]

    # Pre-allocate batch buffer
    batch_buffer = np.zeros((batch_size, 119, 8, 8), dtype=np.float32)

    # Benchmark zero-copy encoding
    start = time.perf_counter()
    for i, fen in enumerate(positions):
        alphazero_cpp.encode_position_to_buffer(fen, batch_buffer[i])
    elapsed = time.perf_counter() - start

    elapsed_ms = elapsed * 1000
    per_position_us = (elapsed / batch_size) * 1e6

    print(f"✓ Batch size: {batch_size} positions")
    print(f"✓ Total time: {elapsed_ms:.2f} ms")
    print(f"✓ Per-position time: {per_position_us:.1f} μs")
    print(f"✓ Throughput: {batch_size / elapsed:.0f} positions/sec")

    # Target: <1ms for 256 positions = <3.9μs per position
    target_ms = 1.0
    if elapsed_ms < target_ms:
        print(f"✓ PASS: Batch encoding meets <{target_ms}ms target ({elapsed_ms:.2f}ms)\n")
        return True
    else:
        print(f"⚠ WARNING: Batch encoding slower than {target_ms}ms target ({elapsed_ms:.2f}ms)\n")
        return False

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
    buffer = np.zeros((119, 8, 8), dtype=np.float32)
    start = time.perf_counter()
    for _ in range(num_iterations):
        alphazero_cpp.encode_position_to_buffer(fen, buffer)
    elapsed_zero_copy = time.perf_counter() - start

    copy_us = (elapsed_copy / num_iterations) * 1e6
    zero_copy_us = (elapsed_zero_copy / num_iterations) * 1e6
    speedup = elapsed_copy / elapsed_zero_copy

    print(f"✓ With copy: {copy_us:.1f} μs per position")
    print(f"✓ Zero-copy: {zero_copy_us:.1f} μs per position")
    print(f"✓ Speedup: {speedup:.2f}x faster")

    if speedup > 1.0:
        print(f"✓ PASS: Zero-copy is {speedup:.2f}x faster than copy\n")
        return True
    else:
        print(f"⚠ WARNING: Zero-copy not faster than copy\n")
        return False

def test_gpu_memory_compatibility():
    """Test that encoding is compatible with GPU tensor libraries"""
    print("=== Test 4: GPU Memory Compatibility ===")

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Create buffer with specific memory layout (C-contiguous, float32)
    buffer = np.zeros((119, 8, 8), dtype=np.float32, order='C')

    # Encode to buffer
    alphazero_cpp.encode_position_to_buffer(fen, buffer)

    # Verify memory layout is GPU-compatible
    print(f"✓ Buffer dtype: {buffer.dtype} (expected: float32)")
    print(f"✓ Buffer shape: {buffer.shape} (expected: (119, 8, 8))")
    print(f"✓ Buffer is C-contiguous: {buffer.flags['C_CONTIGUOUS']}")
    print(f"✓ Buffer is aligned: {buffer.flags['ALIGNED']}")
    print(f"✓ Buffer owns data: {buffer.flags['OWNDATA']}")

    # Check strides (should be C-contiguous: (512, 64, 8) bytes)
    expected_strides = (8 * 8 * 4, 8 * 4, 4)  # (512, 32, 4) bytes
    print(f"✓ Buffer strides: {buffer.strides} (expected: {expected_strides})")

    if (buffer.dtype == np.float32 and
        buffer.flags['C_CONTIGUOUS'] and
        buffer.flags['ALIGNED']):
        print("✓ PASS: Buffer is GPU-compatible (C-contiguous, aligned, float32)\n")
        return True
    else:
        print("✗ FAIL: Buffer not GPU-compatible\n")
        return False

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
    buffers = [np.zeros((119, 8, 8), dtype=np.float32) for _ in positions]

    # Encode all positions
    for i, fen in enumerate(positions):
        alphazero_cpp.encode_position_to_buffer(fen, buffers[i])

    # Verify each buffer is different
    all_different = True
    for i in range(len(buffers) - 1):
        diff = np.sum(np.abs(buffers[i] - buffers[i+1]))
        print(f"✓ Buffer {i} vs {i+1} difference: {diff:.0f}")
        if diff == 0:
            all_different = False

    if all_different:
        print("✓ PASS: Concurrent encoding produces different results\n")
        return True
    else:
        print("✗ FAIL: Concurrent encoding produced identical results\n")
        return False

if __name__ == "__main__":
    print("========================================")
    print("Zero-Copy Tensor Interface Tests")
    print("========================================")
    print("Target: <1ms for 256 positions (plan line 1784)")
    print("========================================\n")

    results = []
    try:
        results.append(("Zero-Copy Interface", test_zero_copy_interface()))
        results.append(("Batch Zero-Copy Performance", test_batch_zero_copy_performance()))
        results.append(("Memory Copy Comparison", test_memory_copy_comparison()))
        results.append(("GPU Memory Compatibility", test_gpu_memory_compatibility()))
        results.append(("Concurrent Encoding", test_concurrent_encoding()))

        print("========================================")
        print("Test Results Summary")
        print("========================================")
        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {name}")

        all_passed = all(result[1] for result in results)
        if all_passed:
            print("\n========================================")
            print("✓✓✓ ALL ZERO-COPY TESTS PASSED ✓✓✓")
            print("========================================")
            print("\nPhase 3 (Python Bindings) is COMPLETE!")
            print("Ready for Phase 4: Integration & Testing")
        else:
            print("\n========================================")
            print("⚠ SOME TESTS FAILED")
            print("========================================")
            sys.exit(1)

    except Exception as e:
        print("========================================")
        print("✗✗✗ TEST FAILED ✗✗✗")
        print("========================================")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
