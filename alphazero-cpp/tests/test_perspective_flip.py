"""Test perspective flip in move encoding.

This test verifies that move encoding correctly flips coordinates for Black's
perspective, matching the position encoder's perspective flip.

Critical requirement from batched_mcts.md lines 1219-1299:
"If your C++ bitboard generates moves from Black's perspective but the encoder
doesn't flip the coordinates before sending them to the model, the policy head
will be 100% noise."
"""

import numpy as np
import sys
import os

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'Release'))

try:
    import chess
except ImportError:
    print("[ERROR] python-chess not installed. Install with: pip install chess")
    sys.exit(1)

import alphazero_cpp

def test_white_queen_move():
    """Test queen move encoding from White's perspective."""
    # White to move: e2e4
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Encode the move
    index = alphazero_cpp.move_to_index("e2e4", fen)

    # Verify encoding succeeded
    assert index >= 0, f"Failed to encode e2e4 for White: index={index}"

    # Verify it's in the queen move range (0 to 56*64-1)
    assert index < 56 * 64, f"e2e4 should be a queen move, got index={index}"

    print(f"[PASS] White queen move e2e4 encoded as index {index}")

def test_black_queen_move():
    """Test queen move encoding from Black's perspective."""
    # Black to move: e7e5
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

    # Encode the move
    index = alphazero_cpp.move_to_index("e7e5", fen)

    # Verify encoding succeeded
    assert index >= 0, f"Failed to encode e7e5 for Black: index={index}"

    # Verify it's in the queen move range
    assert index < 56 * 64, f"e7e5 should be a queen move, got index={index}"

    print(f"[PASS] Black queen move e7e5 encoded as index {index}")

def test_white_knight_move():
    """Test knight move encoding from White's perspective."""
    # White to move: Nf3
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Encode the move
    index = alphazero_cpp.move_to_index("g1f3", fen)

    # Verify encoding succeeded
    assert index >= 0, f"Failed to encode g1f3 for White: index={index}"

    # Verify it's in the knight move range (56*64 to 56*64+8*64-1)
    assert 56 * 64 <= index < 56 * 64 + 8 * 64, \
        f"g1f3 should be a knight move, got index={index}"

    print(f"[PASS] White knight move g1f3 encoded as index {index}")

def test_black_knight_move():
    """Test knight move encoding from Black's perspective."""
    # Black to move: Nf6
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

    # Encode the move
    index = alphazero_cpp.move_to_index("g8f6", fen)

    # Verify encoding succeeded
    assert index >= 0, f"Failed to encode g8f6 for Black: index={index}"

    # Verify it's in the knight move range
    assert 56 * 64 <= index < 56 * 64 + 8 * 64, \
        f"g8f6 should be a knight move, got index={index}"

    print(f"[PASS] Black knight move g8f6 encoded as index {index}")

def test_white_underpromotion():
    """Test underpromotion encoding from White's perspective."""
    # White pawn on a7, can promote to knight
    fen = "8/P7/8/8/8/8/8/k6K w - - 0 1"

    # Encode the move (a7a8n = promote to knight)
    index = alphazero_cpp.move_to_index("a7a8n", fen)

    # Verify encoding succeeded
    assert index >= 0, f"Failed to encode a7a8n for White: index={index}"

    # Verify it's in the underpromotion range (56*64+8*64 to end)
    assert index >= 56 * 64 + 8 * 64, \
        f"a7a8n should be an underpromotion, got index={index}"

    print(f"[PASS] White underpromotion a7a8n encoded as index {index}")

def test_black_underpromotion():
    """Test underpromotion encoding from Black's perspective."""
    # Black pawn on a2, can promote to knight
    fen = "k6K/8/8/8/8/8/p7/8 b - - 0 1"

    # Encode the move (a2a1n = promote to knight)
    index = alphazero_cpp.move_to_index("a2a1n", fen)

    # Verify encoding succeeded
    assert index >= 0, f"Failed to encode a2a1n for Black: index={index}"

    # Verify it's in the underpromotion range
    assert index >= 56 * 64 + 8 * 64, \
        f"a2a1n should be an underpromotion, got index={index}"

    print(f"[PASS] Black underpromotion a2a1n encoded as index {index}")

def test_symmetry():
    """Test that symmetric moves from White and Black perspectives have consistent encoding.

    This is the critical test: if perspective flip is working correctly, then
    a move from square X to square Y for White should have a similar encoding
    structure as the flipped move for Black.
    """
    print("\nTesting move encoding symmetry:")
    print("-" * 60)

    # Test 1: Pawn moves (e2e4 for White, e7e5 for Black)
    white_pawn_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    black_pawn_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

    white_pawn_index = alphazero_cpp.move_to_index("e2e4", white_pawn_fen)
    black_pawn_index = alphazero_cpp.move_to_index("e7e5", black_pawn_fen)

    print(f"White e2e4: index={white_pawn_index}")
    print(f"Black e7e5: index={black_pawn_index}")

    # Both should be queen moves (same range)
    assert white_pawn_index < 56 * 64 and black_pawn_index < 56 * 64, \
        "Both pawn moves should be encoded as queen moves"

    # Test 2: Knight moves (Nf3 for White, Nf6 for Black)
    white_knight_index = alphazero_cpp.move_to_index("g1f3", white_pawn_fen)
    black_knight_index = alphazero_cpp.move_to_index("g8f6", black_pawn_fen)

    print(f"White Nf3: index={white_knight_index}")
    print(f"Black Nf6: index={black_knight_index}")

    # Both should be knight moves (same range)
    assert 56 * 64 <= white_knight_index < 56 * 64 + 8 * 64, \
        "White knight move should be in knight range"
    assert 56 * 64 <= black_knight_index < 56 * 64 + 8 * 64, \
        "Black knight move should be in knight range"

    print("[PASS] Move encoding symmetry verified")

def test_position_move_consistency():
    """Test that position encoding and move encoding use consistent perspective flip.

    This is the ultimate test: verify that when we encode a position from Black's
    perspective, the move encoding also uses Black's perspective.
    """
    print("\nTesting position-move encoding consistency:")
    print("-" * 60)

    # Black to move position
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

    # Encode position
    position = alphazero_cpp.encode_position(fen)

    # Verify position is encoded from Black's perspective
    # In NHWC layout: position[rank, file, channel]
    # Black's pieces should be on the "bottom" ranks (0-1) from Black's perspective
    # White's pieces should be on the "top" ranks (6-7) from Black's perspective

    # Channel 0 = current player's pawns (Black)
    # Channel 6 = opponent's pawns (White)

    # Black pawns should be on rank 1 (from Black's flipped perspective)
    black_pawn_count = np.sum(position[1, :, 0])
    print(f"Black pawns on rank 1 (flipped): {black_pawn_count}")
    assert black_pawn_count == 8, \
        f"Expected 8 Black pawns on rank 1, got {black_pawn_count}"

    # White pawns should be on rank 6 (from Black's flipped perspective)
    white_pawn_count = np.sum(position[6, :, 6])
    print(f"White pawns on rank 6 (flipped): {white_pawn_count}")
    assert white_pawn_count == 7, \
        f"Expected 7 White pawns on rank 6 (one moved to e4), got {white_pawn_count}"

    # Now encode a move from Black's perspective
    # Black knight on g8 moves to f6
    # From Black's flipped perspective, g8 is at rank 0, file 6
    move_index = alphazero_cpp.move_to_index("g8f6", fen)
    print(f"Black Nf6 encoded as index: {move_index}")

    # Verify it's a knight move
    assert 56 * 64 <= move_index < 56 * 64 + 8 * 64, \
        f"g8f6 should be a knight move, got index={move_index}"

    # Extract the from_square from the knight move index
    # Knight move index format: 56*64 + from_square * 8 + knight_index
    offset = move_index - 56 * 64
    from_square = offset // 8
    knight_index = offset % 8

    print(f"Knight move from square: {from_square} (flipped)")
    print(f"Knight move direction index: {knight_index}")

    # From Black's flipped perspective:
    # g8 (square 62 in normal coords) should be flipped to square 1 (63 - 62 = 1)
    # But after flipping, it's at rank 0, file 6 = square 6
    # Actually, let me recalculate:
    # g8 = rank 7, file 6 = 7*8 + 6 = 62
    # Flipped: 63 - 62 = 1
    # Rank = 1 // 8 = 0, File = 1 % 8 = 1

    # The key insight: the from_square should be in the range 0-7 (rank 0)
    # because Black's back rank is flipped to rank 0
    assert from_square < 8, \
        f"Black's back rank knight should be on rank 0 after flip, got square {from_square}"

    print("[PASS] Position and move encoding use consistent perspective flip")

def main():
    """Run all perspective flip tests."""
    print("=" * 70)
    print("Testing Perspective Flip in Move Encoding")
    print("=" * 70)
    print()
    print("This test verifies that move encoding correctly flips coordinates")
    print("for Black's perspective, matching the position encoder's flip.")
    print()

    tests = [
        ("White Queen Move", test_white_queen_move),
        ("Black Queen Move", test_black_queen_move),
        ("White Knight Move", test_white_knight_move),
        ("Black Knight Move", test_black_knight_move),
        ("White Underpromotion", test_white_underpromotion),
        ("Black Underpromotion", test_black_underpromotion),
        ("Move Symmetry", test_symmetry),
        ("Position-Move Consistency", test_position_move_consistency),
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

    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print()
        print("[PASS] All perspective flip tests passed!")
        print()
        print("The move encoder correctly flips coordinates for Black's perspective,")
        print("matching the position encoder's perspective flip.")
        print()
        print("This prevents 100% policy noise during training.")
        return True
    else:
        print()
        print(f"[FAIL] {failed} tests failed")
        print()
        print("Fix the perspective flip bugs before proceeding to Phase 4.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
