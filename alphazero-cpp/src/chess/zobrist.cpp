#include "chess/zobrist.hpp"
#include <random>

namespace chess {

// Static member initialization
uint64_t Zobrist::piece_primary_[PIECE_COUNT][SQUARE_COUNT];
uint32_t Zobrist::piece_secondary_[PIECE_COUNT][SQUARE_COUNT];

uint64_t Zobrist::en_passant_primary_[8];
uint32_t Zobrist::en_passant_secondary_[8];

uint64_t Zobrist::castling_primary_[16];
uint32_t Zobrist::castling_secondary_[16];

uint64_t Zobrist::side_primary_;
uint32_t Zobrist::side_secondary_;

bool Zobrist::initialized_ = false;

void Zobrist::init() {
    if (initialized_) return;

    // Use fixed seed for deterministic Zobrist numbers across runs/machines
    // This ensures consistent hashing behavior for testing and debugging
    std::mt19937_64 rng_64(0x123456789ABCDEF0ULL);
    std::mt19937 rng_32(0x12345678U);

    // Initialize piece hashes
    for (int pc = 0; pc < PIECE_COUNT; ++pc) {
        for (int sq = 0; sq < SQUARE_COUNT; ++sq) {
            piece_primary_[pc][sq] = rng_64();
            piece_secondary_[pc][sq] = rng_32();
        }
    }

    // Initialize en passant hashes (one per file)
    for (int f = 0; f < 8; ++f) {
        en_passant_primary_[f] = rng_64();
        en_passant_secondary_[f] = rng_32();
    }

    // Initialize castling rights hashes (one per combination)
    for (int cr = 0; cr < 16; ++cr) {
        castling_primary_[cr] = rng_64();
        castling_secondary_[cr] = rng_32();
    }

    // Initialize side to move hash
    side_primary_ = rng_64();
    side_secondary_ = rng_32();

    initialized_ = true;
}

PositionHash Zobrist::piece_hash(Piece pc, Square sq) {
    return PositionHash{piece_primary_[pc][sq], piece_secondary_[pc][sq]};
}

PositionHash Zobrist::en_passant_hash(File f) {
    return PositionHash{en_passant_primary_[f], en_passant_secondary_[f]};
}

PositionHash Zobrist::castling_hash(int castling_rights) {
    return PositionHash{castling_primary_[castling_rights], castling_secondary_[castling_rights]};
}

PositionHash Zobrist::side_hash() {
    return PositionHash{side_primary_, side_secondary_};
}

} // namespace chess
