#pragma once

#include "chess/bitboard.hpp"
#include "chess/move.hpp"
#include "chess/zobrist.hpp"
#include <vector>
#include <string>

namespace chess {

// State info that needs to be restored when unmaking a move
struct StateInfo {
    Piece captured_piece;
    Square en_passant_square;
    int castling_rights;
    int halfmove_clock;
    PositionHash hash;

    StateInfo()
        : captured_piece(NO_PIECE)
        , en_passant_square(SQUARE_NONE)
        , castling_rights(NO_CASTLING)
        , halfmove_clock(0)
        , hash{0, 0} {}
};

class Position {
public:
    Position();

    // Initialize from FEN string
    void set_fen(const std::string& fen);
    std::string fen() const;

    // Board state queries
    Bitboard pieces() const { return occupied_; }
    Bitboard pieces(Color c) const { return by_color_[c]; }
    Bitboard pieces(PieceType pt) const { return by_type_[pt]; }
    Bitboard pieces(PieceType pt, Color c) const { return by_type_[pt] & by_color_[c]; }
    Bitboard pieces(PieceType pt1, PieceType pt2) const { return by_type_[pt1] | by_type_[pt2]; }

    Piece piece_on(Square sq) const { return board_[sq]; }
    bool empty(Square sq) const { return board_[sq] == NO_PIECE; }

    // Game state
    Color side_to_move() const { return side_to_move_; }
    Square en_passant_square() const { return en_passant_square_; }
    int castling_rights() const { return castling_rights_; }
    int halfmove_clock() const { return halfmove_clock_; }
    int fullmove_number() const { return fullmove_number_; }

    // Hash
    PositionHash hash() const { return hash_; }

    // Move operations
    void make_move(const Move& move);
    void unmake_move(const Move& move);

    // Game status
    bool is_check() const;
    bool is_checkmate() const;
    bool is_stalemate() const;
    bool is_draw() const;
    bool is_repetition() const;
    bool is_fifty_move_rule() const;
    bool is_insufficient_material() const;

    // Attacks and checks
    Bitboard attackers_to(Square sq, Bitboard occupied) const;
    bool is_attacked_by(Square sq, Color c) const;

    // Utility
    void clear();
    void put_piece(Piece pc, Square sq);
    void remove_piece(Square sq);
    void move_piece(Square from, Square to);

private:
    // Board representation
    Bitboard by_type_[PIECE_TYPE_COUNT];  // Bitboards for each piece type
    Bitboard by_color_[COLOR_COUNT];       // Bitboards for each color
    Bitboard occupied_;                     // All pieces
    Piece board_[SQUARE_COUNT];            // Piece on each square

    // Game state
    Color side_to_move_;
    Square en_passant_square_;
    int castling_rights_;
    int halfmove_clock_;
    int fullmove_number_;

    // Zobrist hash (dual hash for collision detection)
    PositionHash hash_;

    // History for unmake and repetition detection
    std::vector<StateInfo> state_history_;
    std::vector<PositionHash> position_history_;  // For 3-fold repetition

    // Helper functions
    void compute_hash();
    void update_castling_rights(Square from, Square to);
};

} // namespace chess
