"""Test script to verify C++-aligned Python encoder matches C++ exactly."""

import sys
sys.path.insert(0, 'alphazero-cpp/build/Release')

import chess
import alphazero_cpp
from alphazero.chess_env.moves_cpp_aligned import get_cpp_aligned_encoder

# Test if the new encoder matches C++ exactly
fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
board = chess.Board(fen)
encoder = get_cpp_aligned_encoder()

print('Testing C++-aligned Python encoder:')
print('Move      | Python | C++    | Match')
print('----------|--------|--------|------')

test_moves = ['e2e4', 'e2e3', 'g1f3', 'g1h3', 'b1a3', 'b1c3', 'g1e2', 'b1d2']
all_match = True
for uci_str in test_moves:
    move = chess.Move.from_uci(uci_str)
    python_idx = encoder.encode(move, board)
    cpp_idx = alphazero_cpp.move_to_index(uci_str, fen)
    match = 'YES' if python_idx == cpp_idx else 'NO'
    if python_idx != cpp_idx:
        all_match = False
    print(f'{uci_str:9} | {python_idx:6} | {cpp_idx:6} | {match}')

print()
if all_match:
    print('SUCCESS: All moves match!')
else:
    print('FAILURE: Some moves do not match')
