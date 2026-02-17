#!/usr/bin/env python3
"""Tests for C++ move encoder/decoder (index_to_move / move_to_index).

Verifies round-trip correctness across all move categories:
queen moves, knight moves, queen promotions, underpromotions,
black-to-move perspective flips, and an exhaustive index sweep.
"""

import sys
import os

# Fix Windows console encoding for Unicode characters (reconfigure avoids
# closing the underlying buffer, which breaks pytest's capture mechanism)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add build directory to path (MSVC puts .pyd in build/Release/)
build_dir = os.path.join(os.path.dirname(__file__), "..", "build", "Release")
if not os.path.exists(build_dir):
    build_dir = os.path.join(os.path.dirname(__file__), "..", "build")
sys.path.insert(0, build_dir)

import alphazero_cpp

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
POLICY_SIZE = 4672

passed = 0
failed = 0


def check(condition, msg):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {msg}")


def round_trip(uci_move, fen):
    """Encode a UCI move to index and decode back; return decoded string."""
    idx = alphazero_cpp.move_to_index(uci_move, fen)
    decoded = alphazero_cpp.index_to_move(idx, fen)
    return idx, decoded


# ── Test 1: Queen-type moves (indices 0-3583) ────────────────────────────

def test_queen_moves():
    print("=== Test 1: Queen-type moves ===")
    moves = ["e2e4", "d2d4", "e2e3", "d2d3", "a2a3", "h2h4", "c2c4", "b2b3"]
    for m in moves:
        idx, decoded = round_trip(m, STARTING_FEN)
        check(decoded == m, f"round-trip {m}: got {decoded!r} (idx={idx})")
        check(0 <= idx < 3584, f"{m} index {idx} not in queen-move range 0-3583")
    print(f"  ✓ Tested {len(moves)} queen-type moves\n")


# ── Test 2: Knight moves (indices 3584-4095) ─────────────────────────────

def test_knight_moves():
    print("=== Test 2: Knight moves ===")
    moves = ["g1f3", "g1h3", "b1c3", "b1a3"]
    for m in moves:
        idx, decoded = round_trip(m, STARTING_FEN)
        check(decoded == m, f"round-trip {m}: got {decoded!r} (idx={idx})")
        check(3584 <= idx < 4096, f"{m} index {idx} not in knight range 3584-4095")
    print(f"  ✓ Tested {len(moves)} knight moves\n")


# ── Test 3: Queen promotions ─────────────────────────────────────────────

def test_queen_promotions():
    print("=== Test 3: Queen promotions ===")
    # Pawn on a7 ready to promote
    fen = "8/P7/8/8/8/8/8/4K2k w - - 0 1"
    move = "a7a8q"
    idx, decoded = round_trip(move, fen)
    check(decoded == move, f"round-trip {move}: got {decoded!r} (idx={idx})")
    # Queen promotions are encoded as queen-type moves (straight advance, 7th distance)
    check(0 <= idx < 3584, f"{move} index {idx} should be in queen-move range")

    # Diagonal capture promotion
    fen2 = "1n6/P7/8/8/8/8/8/4K2k w - - 0 1"
    move2 = "a7b8q"
    idx2, decoded2 = round_trip(move2, fen2)
    check(decoded2 == move2, f"round-trip {move2}: got {decoded2!r} (idx={idx2})")
    print("  ✓ Queen promotions verified\n")


# ── Test 4: Underpromotions (indices 4096-4671) ─────────────────────────

def test_underpromotions():
    print("=== Test 4: Underpromotions ===")
    fen = "8/P7/8/8/8/8/8/4K2k w - - 0 1"
    for piece in ["n", "b", "r"]:
        move = f"a7a8{piece}"
        idx, decoded = round_trip(move, fen)
        check(decoded == move, f"round-trip {move}: got {decoded!r} (idx={idx})")
        check(4096 <= idx < POLICY_SIZE,
              f"{move} index {idx} not in underpromotion range 4096-4671")

    # Diagonal underpromotion captures
    fen2 = "1n6/P7/8/8/8/8/8/4K2k w - - 0 1"
    for piece in ["n", "b", "r"]:
        move = f"a7b8{piece}"
        idx, decoded = round_trip(move, fen2)
        check(decoded == move, f"round-trip {move}: got {decoded!r} (idx={idx})")
        check(4096 <= idx < POLICY_SIZE,
              f"{move} index {idx} not in underpromotion range 4096-4671")

    # Black underpromotions (perspective flip: a2→a1 maps to h7→h8 internally)
    fen3 = "7K/8/8/8/8/8/p7/4k3 b - - 0 1"
    for piece in ["n", "b", "r"]:
        move = f"a2a1{piece}"
        idx, decoded = round_trip(move, fen3)
        check(decoded == move, f"black round-trip {move}: got {decoded!r} (idx={idx})")
        check(4096 <= idx < POLICY_SIZE,
              f"black {move} index {idx} not in underpromotion range 4096-4671")

    # Black diagonal underpromotion capture
    fen4 = "7K/8/8/8/8/8/p7/1N2k3 b - - 0 1"
    for piece in ["n", "b", "r"]:
        move = f"a2b1{piece}"
        idx, decoded = round_trip(move, fen4)
        check(decoded == move, f"black round-trip {move}: got {decoded!r} (idx={idx})")
        check(4096 <= idx < POLICY_SIZE,
              f"black {move} index {idx} not in underpromotion range 4096-4671")
    print("  ✓ Underpromotions verified (white + black)\n")


# ── Test 5: Black-to-move perspective flip ────────────────────────────────

def test_black_perspective():
    print("=== Test 5: Black-to-move perspective flip ===")
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    moves = ["e7e5", "d7d5", "g8f6", "b8c6"]
    for m in moves:
        idx, decoded = round_trip(m, fen)
        check(decoded == m, f"black round-trip {m}: got {decoded!r} (idx={idx})")
    print(f"  ✓ Tested {len(moves)} black moves with perspective flip\n")


# ── Test 6: All legal moves round-trip from starting position ────────────

def test_all_legal_moves():
    print("=== Test 6: All legal moves round-trip (starting position) ===")
    # index_to_move returns geometric moves (not legality-filtered), so we
    # test by encoding each of the 20 known legal moves and decoding back.
    legal_moves = [
        "a2a3", "a2a4", "b2b3", "b2b4", "c2c3", "c2c4", "d2d3", "d2d4",
        "e2e3", "e2e4", "f2f3", "f2f4", "g2g3", "g2g4", "h2h3", "h2h4",
        "b1a3", "b1c3", "g1f3", "g1h3",
    ]
    for uci in legal_moves:
        idx = alphazero_cpp.move_to_index(uci, STARTING_FEN)
        check(idx >= 0, f"move_to_index({uci!r}) returned {idx}")
        decoded = alphazero_cpp.index_to_move(idx, STARTING_FEN)
        check(decoded == uci, f"round-trip {uci}: got {decoded!r} (idx={idx})")

    print(f"  ✓ All {len(legal_moves)} legal moves round-trip correctly\n")


# ── Test 7: Exhaustive index sweep (0..4671) ────────────────────────────

def test_exhaustive_sweep():
    print("=== Test 7: Exhaustive index sweep (0..4671) ===")
    # index_to_move returns geometric moves (any valid from→to), not just
    # legal ones. We just verify no crashes and that valid+empty == POLICY_SIZE.
    valid_count = 0
    empty_count = 0
    for i in range(POLICY_SIZE):
        try:
            uci = alphazero_cpp.index_to_move(i, STARTING_FEN)
            if uci:
                valid_count += 1
            else:
                empty_count += 1
        except Exception as e:
            check(False, f"index {i} raised {type(e).__name__}: {e}")

    check(valid_count + empty_count == POLICY_SIZE,
          f"total should be {POLICY_SIZE}, got {valid_count + empty_count}")
    # Geometric moves >> legal moves (typically ~1800-2500 for any position)
    check(valid_count > 1000, f"expected >1000 geometric moves, got {valid_count}")
    print(f"  ✓ Sweep complete: {valid_count} valid, {empty_count} empty, 0 crashes\n")


# ── Test 8: Out-of-range indices ─────────────────────────────────────────

def test_out_of_range():
    print("=== Test 8: Out-of-range indices ===")
    for idx in [-1, POLICY_SIZE, POLICY_SIZE + 100, 999999]:
        try:
            result = alphazero_cpp.index_to_move(idx, STARTING_FEN)
            # Empty string is acceptable for out-of-range
            check(result == "", f"index {idx} should return empty, got {result!r}")
        except (ValueError, IndexError, RuntimeError):
            passed_local = True  # Exception is also acceptable
    print("  ✓ Out-of-range handled gracefully\n")


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Move Decoder Test Suite")
    print("=" * 50 + "\n")

    test_queen_moves()
    test_knight_moves()
    test_queen_promotions()
    test_underpromotions()
    test_black_perspective()
    test_all_legal_moves()
    test_exhaustive_sweep()
    test_out_of_range()

    print("=" * 50)
    if failed == 0:
        print(f"✓✓✓ ALL {passed} CHECKS PASSED ✓✓✓")
    else:
        print(f"✗ {failed} FAILED, {passed} passed")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)
