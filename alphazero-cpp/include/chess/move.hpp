#pragma once

#include "chess/bitboard.hpp"
#include <string>

namespace chess {

// Move encoding: 16 bits
// bits 0-5: from square (0-63)
// bits 6-11: to square (0-63)
// bits 12-13: promotion piece type (0=knight, 1=bishop, 2=rook, 3=queen)
// bits 14-15: special move flags (0=normal, 1=promotion, 2=en passant, 3=castling)

enum MoveType : int {
    NORMAL = 0,
    PROMOTION = 1,
    EN_PASSANT = 2,
    CASTLING = 3
};

class Move {
public:
    constexpr Move() : data_(0) {}

    constexpr Move(Square from, Square to, MoveType type = NORMAL, PieceType promotion = KNIGHT)
        : data_(from | (to << 6) | (promotion << 12) | (type << 14)) {}

    Square from() const { return static_cast<Square>(data_ & 0x3F); }
    Square to() const { return static_cast<Square>((data_ >> 6) & 0x3F); }
    MoveType type() const { return static_cast<MoveType>((data_ >> 14) & 0x3); }
    PieceType promotion() const { return static_cast<PieceType>((data_ >> 12) & 0x3); }

    bool is_promotion() const { return type() == PROMOTION; }
    bool is_en_passant() const { return type() == EN_PASSANT; }
    bool is_castling() const { return type() == CASTLING; }

    bool operator==(const Move& other) const { return data_ == other.data_; }
    bool operator!=(const Move& other) const { return data_ != other.data_; }

    bool is_ok() const { return data_ != 0; }

    uint16_t raw() const { return data_; }

    std::string to_uci() const;
    static Move from_uci(const std::string& uci);

private:
    uint16_t data_;
};

constexpr Move MOVE_NONE = Move();
constexpr Move MOVE_NULL = Move();

} // namespace chess
