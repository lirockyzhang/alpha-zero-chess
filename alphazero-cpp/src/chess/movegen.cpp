#include "chess/movegen.hpp"
#include <algorithm>

namespace chess {

// Global attack tables
Bitboard PseudoAttacks[PIECE_TYPE_COUNT][SQUARE_COUNT];
Bitboard PawnAttacks[COLOR_COUNT][SQUARE_COUNT];

bool MoveGenerator::initialized_ = false;

// Classical sliding attack generation (simple and reliable)
Bitboard sliding_attacks(Square sq, Bitboard occupied, const int directions[][2], int num_dirs) {
    Bitboard attacks = 0;
    int rank = rank_of(sq);
    int file = file_of(sq);

    for (int d = 0; d < num_dirs; ++d) {
        int dr = directions[d][0];
        int df = directions[d][1];

        int r = rank + dr;
        int f = file + df;

        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
            Square target = make_square(static_cast<File>(f), static_cast<Rank>(r));
            attacks |= square_bb(target);

            // Stop if we hit a piece
            if (occupied & square_bb(target)) {
                break;
            }

            r += dr;
            f += df;
        }
    }

    return attacks;
}

Bitboard MoveGenerator::bishop_attacks(Square sq, Bitboard occupied) {
    static const int directions[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    return sliding_attacks(sq, occupied, directions, 4);
}

Bitboard MoveGenerator::rook_attacks(Square sq, Bitboard occupied) {
    static const int directions[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    return sliding_attacks(sq, occupied, directions, 4);
}

Bitboard MoveGenerator::queen_attacks(Square sq, Bitboard occupied) {
    return bishop_attacks(sq, occupied) | rook_attacks(sq, occupied);
}

void MoveGenerator::init_non_sliding_attacks() {
    // Initialize pawn attacks
    for (Square sq = A1; sq < SQUARE_COUNT; sq = static_cast<Square>(sq + 1)) {
        int rank = rank_of(sq);
        int file = file_of(sq);

        // White pawn attacks
        PawnAttacks[WHITE][sq] = 0;
        if (rank < 7) {
            if (file > 0) PawnAttacks[WHITE][sq] |= square_bb(static_cast<Square>(sq + 7));
            if (file < 7) PawnAttacks[WHITE][sq] |= square_bb(static_cast<Square>(sq + 9));
        }

        // Black pawn attacks
        PawnAttacks[BLACK][sq] = 0;
        if (rank > 0) {
            if (file > 0) PawnAttacks[BLACK][sq] |= square_bb(static_cast<Square>(sq - 9));
            if (file < 7) PawnAttacks[BLACK][sq] |= square_bb(static_cast<Square>(sq - 7));
        }
    }

    // Initialize knight attacks
    const int knight_offsets[8] = {-17, -15, -10, -6, 6, 10, 15, 17};
    for (Square sq = A1; sq < SQUARE_COUNT; sq = static_cast<Square>(sq + 1)) {
        PseudoAttacks[KNIGHT][sq] = 0;
        int rank = rank_of(sq);
        int file = file_of(sq);

        for (int offset : knight_offsets) {
            int target = sq + offset;
            if (target >= 0 && target < 64) {
                int target_rank = rank_of(static_cast<Square>(target));
                int target_file = file_of(static_cast<Square>(target));

                // Check if move is valid (not wrapping around board)
                if (abs(target_rank - rank) <= 2 && abs(target_file - file) <= 2) {
                    PseudoAttacks[KNIGHT][sq] |= square_bb(static_cast<Square>(target));
                }
            }
        }
    }

    // Initialize king attacks
    const int king_offsets[8] = {-9, -8, -7, -1, 1, 7, 8, 9};
    for (Square sq = A1; sq < SQUARE_COUNT; sq = static_cast<Square>(sq + 1)) {
        PseudoAttacks[KING][sq] = 0;
        int rank = rank_of(sq);
        int file = file_of(sq);

        for (int offset : king_offsets) {
            int target = sq + offset;
            if (target >= 0 && target < 64) {
                int target_rank = rank_of(static_cast<Square>(target));
                int target_file = file_of(static_cast<Square>(target));

                // Check if move is valid (not wrapping around board)
                if (abs(target_rank - rank) <= 1 && abs(target_file - file) <= 1) {
                    PseudoAttacks[KING][sq] |= square_bb(static_cast<Square>(target));
                }
            }
        }
    }
}

void MoveGenerator::init() {
    if (initialized_) return;

    // Initialize Zobrist hashing first
    Zobrist::init();

    // Initialize attack tables
    init_non_sliding_attacks();

    initialized_ = true;
}

// Move generation implementation
template<bool OnlyCaptures>
void MoveGenerator::generate_moves(const Position& pos, std::vector<Move>& moves) {
    Color us = pos.side_to_move();
    Color them = ~us;
    Bitboard our_pieces = pos.pieces(us);
    Bitboard their_pieces = pos.pieces(them);
    Bitboard occupied = pos.pieces();
    Bitboard empty = ~occupied;

    // Target squares: captures or all squares
    Bitboard targets = OnlyCaptures ? their_pieces : ~our_pieces;

    // Generate pawn moves
    Bitboard pawns = pos.pieces(PAWN, us);
    while (pawns) {
        Square from = pop_lsb(pawns);
        int rank = rank_of(from);

        // Pawn pushes
        if (!OnlyCaptures) {
            Square push = static_cast<Square>(static_cast<int>(from) + (us == WHITE ? NORTH : SOUTH));
            if (empty & square_bb(push)) {
                // Check for promotion
                if ((us == WHITE && rank == 6) || (us == BLACK && rank == 1)) {
                    moves.push_back(Move(from, push, PROMOTION, QUEEN));
                    moves.push_back(Move(from, push, PROMOTION, ROOK));
                    moves.push_back(Move(from, push, PROMOTION, BISHOP));
                    moves.push_back(Move(from, push, PROMOTION, KNIGHT));
                } else {
                    moves.push_back(Move(from, push));

                    // Double push
                    if ((us == WHITE && rank == 1) || (us == BLACK && rank == 6)) {
                        Square double_push = static_cast<Square>(static_cast<int>(push) + (us == WHITE ? NORTH : SOUTH));
                        if (empty & square_bb(double_push)) {
                            moves.push_back(Move(from, double_push));
                        }
                    }
                }
            }
        }

        // Pawn captures
        Bitboard attacks = pawn_attacks(us, from) & their_pieces;
        while (attacks) {
            Square to = pop_lsb(attacks);
            // Check for promotion
            if ((us == WHITE && rank == 6) || (us == BLACK && rank == 1)) {
                moves.push_back(Move(from, to, PROMOTION, QUEEN));
                moves.push_back(Move(from, to, PROMOTION, ROOK));
                moves.push_back(Move(from, to, PROMOTION, BISHOP));
                moves.push_back(Move(from, to, PROMOTION, KNIGHT));
            } else {
                moves.push_back(Move(from, to));
            }
        }

        // En passant
        if (pos.en_passant_square() != SQUARE_NONE) {
            if (pawn_attacks(us, from) & square_bb(pos.en_passant_square())) {
                moves.push_back(Move(from, pos.en_passant_square(), EN_PASSANT));
            }
        }
    }

    // Generate knight moves
    Bitboard knights = pos.pieces(KNIGHT, us);
    while (knights) {
        Square from = pop_lsb(knights);
        Bitboard attacks = knight_attacks(from) & targets;
        while (attacks) {
            Square to = pop_lsb(attacks);
            moves.push_back(Move(from, to));
        }
    }

    // Generate bishop moves
    Bitboard bishops = pos.pieces(BISHOP, us);
    while (bishops) {
        Square from = pop_lsb(bishops);
        Bitboard attacks = bishop_attacks(from, occupied) & targets;
        while (attacks) {
            Square to = pop_lsb(attacks);
            moves.push_back(Move(from, to));
        }
    }

    // Generate rook moves
    Bitboard rooks = pos.pieces(ROOK, us);
    while (rooks) {
        Square from = pop_lsb(rooks);
        Bitboard attacks = rook_attacks(from, occupied) & targets;
        while (attacks) {
            Square to = pop_lsb(attacks);
            moves.push_back(Move(from, to));
        }
    }

    // Generate queen moves
    Bitboard queens = pos.pieces(QUEEN, us);
    while (queens) {
        Square from = pop_lsb(queens);
        Bitboard attacks = queen_attacks(from, occupied) & targets;
        while (attacks) {
            Square to = pop_lsb(attacks);
            moves.push_back(Move(from, to));
        }
    }

    // Generate king moves
    Bitboard kings = pos.pieces(KING, us);
    if (kings) {
        Square from = lsb(kings);
        Bitboard attacks = king_attacks(from) & targets;
        while (attacks) {
            Square to = pop_lsb(attacks);
            moves.push_back(Move(from, to));
        }

        // Castling - will be validated for legality (king not in/through check) by is_legal()
        if (!OnlyCaptures) {
            int rights = pos.castling_rights();

            // Kingside castling
            if (us == WHITE && (rights & WHITE_OO)) {
                if ((empty & square_bb(F1)) && (empty & square_bb(G1))) {
                    moves.push_back(Move(E1, G1, CASTLING));
                }
            } else if (us == BLACK && (rights & BLACK_OO)) {
                if ((empty & square_bb(F8)) && (empty & square_bb(G8))) {
                    moves.push_back(Move(E8, G8, CASTLING));
                }
            }

            // Queenside castling
            if (us == WHITE && (rights & WHITE_OOO)) {
                if ((empty & square_bb(D1)) && (empty & square_bb(C1)) && (empty & square_bb(B1))) {
                    moves.push_back(Move(E1, C1, CASTLING));
                }
            } else if (us == BLACK && (rights & BLACK_OOO)) {
                if ((empty & square_bb(D8)) && (empty & square_bb(C8)) && (empty & square_bb(B8))) {
                    moves.push_back(Move(E8, C8, CASTLING));
                }
            }
        }
    }
}

// Explicit template instantiations
template void MoveGenerator::generate_moves<false>(const Position&, std::vector<Move>&);
template void MoveGenerator::generate_moves<true>(const Position&, std::vector<Move>&);

void MoveGenerator::generate_legal_moves(const Position& pos, std::vector<Move>& moves) {
    std::vector<Move> pseudo_legal;
    generate_moves<false>(pos, pseudo_legal);

    for (const Move& move : pseudo_legal) {
        if (is_legal(pos, move)) {
            moves.push_back(move);
        }
    }
}

bool MoveGenerator::is_legal(const Position& pos, const Move& move) {
    Color us = pos.side_to_move();
    Color them = ~us;

    // Special handling for castling - must check king doesn't pass through check
    if (move.is_castling()) {
        Square from = move.from();
        Square to = move.to();

        // King must not be in check currently
        if (pos.is_check()) {
            return false;
        }

        // King must not pass through check
        Square middle;
        if (to > from) {  // Kingside
            middle = static_cast<Square>(static_cast<int>(from) + 1);
        } else {  // Queenside
            middle = static_cast<Square>(static_cast<int>(from) - 1);
        }

        if (pos.is_attacked_by(middle, them)) {
            return false;
        }

        // King must not end in check (checked below with normal logic)
    }

    // Make a copy and try the move
    Position temp = pos;
    temp.make_move(move);

    // After make_move, side_to_move has switched
    // We need to check if OUR king (the side that just moved) is in check
    Bitboard kings = temp.pieces(KING, us);
    if (!kings) return false;

    Square king_sq = lsb(kings);
    // Check if our king is attacked by the opponent (now the side to move)
    return !temp.is_attacked_by(king_sq, temp.side_to_move());
}

} // namespace chess
