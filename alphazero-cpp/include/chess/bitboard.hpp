#pragma once

#include <cstdint>
#include <string>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

namespace chess {

// Bitboard type: 64-bit integer representing the 8x8 board
using Bitboard = uint64_t;

// Square indices (0-63, A1=0, H8=63)
enum Square : int {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    SQUARE_NONE = 64,
    SQUARE_COUNT = 64
};

// Piece types
enum PieceType : int {
    PAWN = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK = 3,
    QUEEN = 4,
    KING = 5,
    PIECE_TYPE_COUNT = 6,
    NO_PIECE_TYPE = 7
};

// Colors
enum Color : int {
    WHITE = 0,
    BLACK = 1,
    COLOR_COUNT = 2
};

// Piece = PieceType + Color
enum Piece : int {
    W_PAWN = 0, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN = 6, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    NO_PIECE = 12,
    PIECE_COUNT = 12
};

// Castling rights
enum CastlingRight : int {
    NO_CASTLING = 0,
    WHITE_OO = 1,      // White kingside
    WHITE_OOO = 2,     // White queenside
    BLACK_OO = 4,      // Black kingside
    BLACK_OOO = 8,     // Black queenside
    WHITE_CASTLING = WHITE_OO | WHITE_OOO,
    BLACK_CASTLING = BLACK_OO | BLACK_OOO,
    ANY_CASTLING = WHITE_CASTLING | BLACK_CASTLING
};

// Directions
enum Direction : int {
    NORTH = 8,
    SOUTH = -8,
    EAST = 1,
    WEST = -1,
    NORTH_EAST = 9,
    NORTH_WEST = 7,
    SOUTH_EAST = -7,
    SOUTH_WEST = -9
};

// Files and Ranks
enum File : int { FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H };
enum Rank : int { RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8 };

// Bitboard constants
constexpr Bitboard FILE_A_BB = 0x0101010101010101ULL;
constexpr Bitboard FILE_B_BB = FILE_A_BB << 1;
constexpr Bitboard FILE_C_BB = FILE_A_BB << 2;
constexpr Bitboard FILE_D_BB = FILE_A_BB << 3;
constexpr Bitboard FILE_E_BB = FILE_A_BB << 4;
constexpr Bitboard FILE_F_BB = FILE_A_BB << 5;
constexpr Bitboard FILE_G_BB = FILE_A_BB << 6;
constexpr Bitboard FILE_H_BB = FILE_A_BB << 7;

constexpr Bitboard RANK_1_BB = 0xFFULL;
constexpr Bitboard RANK_2_BB = RANK_1_BB << 8;
constexpr Bitboard RANK_3_BB = RANK_1_BB << 16;
constexpr Bitboard RANK_4_BB = RANK_1_BB << 24;
constexpr Bitboard RANK_5_BB = RANK_1_BB << 32;
constexpr Bitboard RANK_6_BB = RANK_1_BB << 40;
constexpr Bitboard RANK_7_BB = RANK_1_BB << 48;
constexpr Bitboard RANK_8_BB = RANK_1_BB << 56;

// Hardware intrinsics for bitboard operations
// These provide 2-3x speedup over software implementations

// Count number of set bits (population count)
inline int popcount(Bitboard bb) {
#ifdef _MSC_VER
    return static_cast<int>(__popcnt64(bb));
#else
    return __builtin_popcountll(bb);
#endif
}

// Get index of least significant bit (trailing zero count)
inline Square lsb(Bitboard bb) {
#ifdef _MSC_VER
    unsigned long idx;
    _BitScanForward64(&idx, bb);
    return static_cast<Square>(idx);
#else
    return static_cast<Square>(__builtin_ctzll(bb));
#endif
}

// Get index of most significant bit
inline Square msb(Bitboard bb) {
#ifdef _MSC_VER
    unsigned long idx;
    _BitScanReverse64(&idx, bb);
    return static_cast<Square>(idx);
#else
    return static_cast<Square>(63 - __builtin_clzll(bb));
#endif
}

// Clear least significant bit and return its index
inline Square pop_lsb(Bitboard& bb) {
    Square sq = lsb(bb);
#ifdef _MSC_VER
    bb &= bb - 1;  // Clear lowest set bit
#else
    bb = _blsr_u64(bb);  // BMI1 instruction: clear lowest set bit
#endif
    return sq;
}

// Helper functions
inline constexpr Bitboard square_bb(Square sq) {
    return 1ULL << sq;
}

inline constexpr File file_of(Square sq) {
    return static_cast<File>(sq & 7);
}

inline constexpr Rank rank_of(Square sq) {
    return static_cast<Rank>(sq >> 3);
}

inline constexpr Square make_square(File f, Rank r) {
    return static_cast<Square>((r << 3) | f);
}

inline constexpr Bitboard file_bb(File f) {
    return FILE_A_BB << f;
}

inline constexpr Bitboard rank_bb(Rank r) {
    return RANK_1_BB << (8 * r);
}

inline constexpr Color operator~(Color c) {
    return static_cast<Color>(c ^ 1);
}

inline constexpr Piece make_piece(Color c, PieceType pt) {
    return static_cast<Piece>((c << 3) | pt);
}

inline constexpr PieceType type_of(Piece pc) {
    return static_cast<PieceType>(pc & 7);
}

inline constexpr Color color_of(Piece pc) {
    return static_cast<Color>(pc >> 3);
}

// Shift bitboard in a direction
template<Direction D>
constexpr Bitboard shift(Bitboard bb) {
    if constexpr (D == NORTH) return bb << 8;
    if constexpr (D == SOUTH) return bb >> 8;
    if constexpr (D == EAST) return (bb & ~FILE_H_BB) << 1;
    if constexpr (D == WEST) return (bb & ~FILE_A_BB) >> 1;
    if constexpr (D == NORTH_EAST) return (bb & ~FILE_H_BB) << 9;
    if constexpr (D == NORTH_WEST) return (bb & ~FILE_A_BB) << 7;
    if constexpr (D == SOUTH_EAST) return (bb & ~FILE_H_BB) >> 7;
    if constexpr (D == SOUTH_WEST) return (bb & ~FILE_A_BB) >> 9;
    return 0;
}

// String conversion for debugging
std::string square_to_string(Square sq);
Square string_to_square(const std::string& str);
std::string bitboard_to_string(Bitboard bb);

} // namespace chess
