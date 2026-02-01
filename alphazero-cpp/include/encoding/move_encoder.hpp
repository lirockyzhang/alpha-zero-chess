#pragma once

#include "../third_party/chess-library/include/chess.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace encoding {

// AlphaZero move encoding
// 4672 possible moves total:
// - Queen moves: 56 per square (8 directions × 7 distances) = 56 × 64 = 3584 moves
// - Knight moves: 8 per square = 8 × 64 = 512 moves
// - Underpromotions: 9 per square (3 directions × 3 piece types) = 9 × 64 = 576 moves
// Total: 3584 + 512 + 576 = 4672 moves

class MoveEncoder {
public:
    static constexpr int POLICY_SIZE = 4672;

    // Convert UCI move to policy index
    // Returns -1 if move is invalid
    static int move_to_index(const chess::Move& move, const chess::Board& board);
    static int move_to_index(const std::string& uci_move, const chess::Board& board);

    // Convert policy index to UCI move
    // Returns empty move if index is invalid
    static chess::Move index_to_move(int index, const chess::Board& board);
    static std::string index_to_uci(int index, const chess::Board& board);

    // Get all legal moves as policy indices
    static std::vector<int> get_legal_move_indices(const chess::Board& board);

private:
    // Move type classification
    enum class MoveType {
        QUEEN_MOVE,      // Includes rook, bishop, queen moves
        KNIGHT_MOVE,
        UNDERPROMOTION   // Promotions to knight, bishop, rook
    };

    // Helper functions
    static MoveType classify_move(const chess::Move& move, const chess::Board& board);
    static int encode_queen_move(const chess::Move& move, const chess::Board& board);
    static int encode_knight_move(const chess::Move& move, const chess::Board& board);
    static int encode_underpromotion(const chess::Move& move, const chess::Board& board);

    // Decode functions
    static chess::Move decode_queen_move(int index, const chess::Board& board);
    static chess::Move decode_knight_move(int index, const chess::Board& board);
    static chess::Move decode_underpromotion(int index, const chess::Board& board);

    // Direction helpers
    static int get_direction_index(int from_square, int to_square);
    static int get_distance(int from_square, int to_square, int direction);
};

} // namespace encoding
