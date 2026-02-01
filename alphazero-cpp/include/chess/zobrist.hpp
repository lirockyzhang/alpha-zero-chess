#pragma once

#include "chess/bitboard.hpp"
#include <cstdint>

namespace chess {

// Dual Zobrist hashing for virtually collision-free position identification
// Primary: 64-bit hash for main identification
// Secondary: 32-bit hash for collision detection
// Combined probability of collision: ~1 in 2^96 (virtually impossible)

struct PositionHash {
    uint64_t primary;
    uint32_t secondary;

    bool operator==(const PositionHash& other) const {
        return primary == other.primary && secondary == other.secondary;
    }

    bool operator!=(const PositionHash& other) const {
        return !(*this == other);
    }
};

class Zobrist {
public:
    // Initialize Zobrist random numbers with fixed seed for determinism
    static void init();

    // Get hash for piece on square
    static PositionHash piece_hash(Piece pc, Square sq);

    // Get hash for en passant file
    static PositionHash en_passant_hash(File f);

    // Get hash for castling rights
    static PositionHash castling_hash(int castling_rights);

    // Get hash for side to move (black)
    static PositionHash side_hash();

private:
    // Zobrist tables (initialized once at startup)
    static uint64_t piece_primary_[PIECE_COUNT][SQUARE_COUNT];
    static uint32_t piece_secondary_[PIECE_COUNT][SQUARE_COUNT];

    static uint64_t en_passant_primary_[8];  // One per file
    static uint32_t en_passant_secondary_[8];

    static uint64_t castling_primary_[16];  // One per castling rights combination
    static uint32_t castling_secondary_[16];

    static uint64_t side_primary_;  // XOR if black to move
    static uint32_t side_secondary_;

    static bool initialized_;
};

// Helper functions for incremental hash updates
inline PositionHash operator^(const PositionHash& a, const PositionHash& b) {
    return PositionHash{a.primary ^ b.primary, a.secondary ^ b.secondary};
}

inline PositionHash& operator^=(PositionHash& a, const PositionHash& b) {
    a.primary ^= b.primary;
    a.secondary ^= b.secondary;
    return a;
}

} // namespace chess
