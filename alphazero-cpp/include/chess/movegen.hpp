#pragma once

#include "chess/bitboard.hpp"
#include "chess/position.hpp"
#include "chess/move.hpp"
#include <vector>

namespace chess {

// Attack tables for non-sliding pieces (precomputed)
extern Bitboard PseudoAttacks[PIECE_TYPE_COUNT][SQUARE_COUNT];
extern Bitboard PawnAttacks[COLOR_COUNT][SQUARE_COUNT];

class MoveGenerator {
public:
    // Initialize all attack tables
    // Must be called once at program startup
    static void init();

    // Generate all pseudo-legal moves for a position
    // Pseudo-legal means they may leave the king in check
    template<bool OnlyCaptures = false>
    static void generate_moves(const Position& pos, std::vector<Move>& moves);

    // Generate only legal moves (filters out moves that leave king in check)
    static void generate_legal_moves(const Position& pos, std::vector<Move>& moves);

    // Check if a move is legal in the given position
    static bool is_legal(const Position& pos, const Move& move);

    // Get attack bitboard for a piece type on a square
    static Bitboard attacks(PieceType pt, Square sq, Bitboard occupied);

    // Specific attack functions
    static Bitboard pawn_attacks(Color c, Square sq);
    static Bitboard knight_attacks(Square sq);
    static Bitboard bishop_attacks(Square sq, Bitboard occupied);
    static Bitboard rook_attacks(Square sq, Bitboard occupied);
    static Bitboard queen_attacks(Square sq, Bitboard occupied);
    static Bitboard king_attacks(Square sq);

private:
    static bool initialized_;
    static void init_non_sliding_attacks();
};

// Inline implementations for performance-critical functions

inline Bitboard MoveGenerator::pawn_attacks(Color c, Square sq) {
    return PawnAttacks[c][sq];
}

inline Bitboard MoveGenerator::knight_attacks(Square sq) {
    return PseudoAttacks[KNIGHT][sq];
}

inline Bitboard MoveGenerator::king_attacks(Square sq) {
    return PseudoAttacks[KING][sq];
}

inline Bitboard MoveGenerator::attacks(PieceType pt, Square sq, Bitboard occupied) {
    switch (pt) {
        case PAWN:   return 0;  // Pawn attacks are color-dependent
        case KNIGHT: return knight_attacks(sq);
        case BISHOP: return bishop_attacks(sq, occupied);
        case ROOK:   return rook_attacks(sq, occupied);
        case QUEEN:  return queen_attacks(sq, occupied);
        case KING:   return king_attacks(sq);
        default:     return 0;
    }
}

} // namespace chess
