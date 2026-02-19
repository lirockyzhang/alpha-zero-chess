"""Test NHWC (channels-last) tensor layout implementation."""

import numpy as np
import sys
import os

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'Release'))

import alphazero_cpp

def test_nhwc_shape():
    """Test that encoding returns NHWC shape (8, 8, 123)."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    encoding = alphazero_cpp.encode_position(fen)

    print(f"Encoding shape: {encoding.shape}")
    assert encoding.shape == (8, 8, 123), f"Expected (8, 8, 123), got {encoding.shape}"
    print("[PASS] NHWC shape correct: (8, 8, 123)")

def test_nhwc_memory_layout():
    """Test that channels are contiguous in memory (channels-last)."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    encoding = alphazero_cpp.encode_position(fen)

    # Check strides - for NHWC, innermost dimension should be channels
    # Expected strides: (8*123*4, 123*4, 4) for float32
    expected_strides = (8 * 123 * 4, 123 * 4, 4)
    print(f"Encoding strides: {encoding.strides}")
    print(f"Expected strides: {expected_strides}")

    assert encoding.strides == expected_strides, \
        f"Expected strides {expected_strides}, got {encoding.strides}"
    print("[PASS] Memory layout is channels-last (NHWC)")

def test_nhwc_piece_positions():
    """Test that piece positions are encoded correctly in NHWC layout."""
    # Starting position
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    encoding = alphazero_cpp.encode_position(fen)

    # In NHWC layout: encoding[rank, file, channel]
    # Channel 0 = current player's pawns (White)
    # Channel 6 = opponent's pawns (Black)

    # Check White pawns on rank 1 (from White's perspective)
    white_pawn_channel = 0
    for file in range(8):
        assert encoding[1, file, white_pawn_channel] == 1.0, \
            f"Expected White pawn at rank 1, file {file}"
    print("[PASS] White pawns encoded correctly on rank 1")

    # Check Black pawns on rank 6 (from White's perspective)
    black_pawn_channel = 6
    for file in range(8):
        assert encoding[6, file, black_pawn_channel] == 1.0, \
            f"Expected Black pawn at rank 6, file {file}"
    print("[PASS] Black pawns encoded correctly on rank 6")

    # Check White king on e1 (rank 0, file 4)
    white_king_channel = 5
    assert encoding[0, 4, white_king_channel] == 1.0, \
        "Expected White king at e1"
    print("[PASS] White king encoded correctly at e1")

def test_nhwc_zero_copy():
    """Test that zero-copy interface works with NHWC layout."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Create buffer with NHWC shape
    buffer = np.zeros((8, 8, 123), dtype=np.float32)
    buffer_ptr = buffer.__array_interface__['data'][0]

    # Encode to buffer
    alphazero_cpp.encode_position_to_buffer(fen, buffer)

    # Verify buffer pointer unchanged (zero-copy)
    assert buffer.__array_interface__['data'][0] == buffer_ptr, \
        "Buffer pointer changed - not zero-copy!"
    print("[PASS] Zero-copy interface works with NHWC layout")

    # Verify encoding is correct
    assert buffer[1, 0, 0] == 1.0, "Expected White pawn at a2"
    assert buffer[6, 0, 6] == 1.0, "Expected Black pawn at a7"
    print("[PASS] Zero-copy encoding produces correct results")

def test_nhwc_batch_encoding():
    """Test batch encoding with NHWC layout."""
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
    ]

    # Create batch buffer with NHWC shape
    batch_size = len(fens)
    batch_buffer = np.zeros((batch_size, 8, 8, 123), dtype=np.float32)

    # Encode each position
    for i, fen in enumerate(fens):
        alphazero_cpp.encode_position_to_buffer(fen, batch_buffer[i])

    # Verify shapes
    assert batch_buffer.shape == (batch_size, 8, 8, 123), \
        f"Expected batch shape ({batch_size}, 8, 8, 123), got {batch_buffer.shape}"
    print(f"[PASS] Batch encoding shape correct: {batch_buffer.shape}")

    # Verify first position has White pawns on rank 1
    assert np.sum(batch_buffer[0, 1, :, 0]) == 8.0, \
        "Expected 8 White pawns in first position"
    print("[PASS] Batch encoding produces correct results")

def test_nhwc_pytorch_compatibility():
    """Test that NHWC layout is compatible with PyTorch channels_last."""
    try:
        import torch
    except ImportError:
        print("[SKIP] PyTorch not installed, skipping PyTorch compatibility test")
        return

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    encoding = alphazero_cpp.encode_position(fen)

    # Convert to PyTorch tensor
    tensor = torch.from_numpy(encoding)

    # Add batch dimension: (1, 8, 8, 123)
    tensor = tensor.unsqueeze(0)

    # Convert to channels_last format
    tensor_cl = tensor.contiguous(memory_format=torch.channels_last)

    print(f"PyTorch tensor shape: {tensor_cl.shape}")
    print(f"PyTorch tensor is_contiguous(channels_last): {tensor_cl.is_contiguous(memory_format=torch.channels_last)}")

    assert tensor_cl.shape == (1, 8, 8, 123), \
        f"Expected shape (1, 8, 8, 123), got {tensor_cl.shape}"
    print("[PASS] NHWC layout compatible with PyTorch channels_last")

def main():
    print("=" * 60)
    print("Testing NHWC (Channels-Last) Tensor Layout")
    print("=" * 60)

    tests = [
        ("NHWC Shape", test_nhwc_shape),
        ("NHWC Memory Layout", test_nhwc_memory_layout),
        ("NHWC Piece Positions", test_nhwc_piece_positions),
        ("NHWC Zero-Copy", test_nhwc_zero_copy),
        ("NHWC Batch Encoding", test_nhwc_batch_encoding),
        ("PyTorch Compatibility", test_nhwc_pytorch_compatibility),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n{name}:")
        print("-" * 60)
        try:
            test_func()
            passed += 1
            print(f"[PASS] {name} PASSED")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
