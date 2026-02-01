#include "chess/position.hpp"
#include "chess/movegen.hpp"
#include <sstream>
#include <algorithm>

namespace chess {

Position::Position() {
    clear();
}

void Position::clear() {
    for (int i = 0; i < PIECE_TYPE_COUNT; ++i) {
        by_type_[i] = 0;
    }
    for (int i = 0; i < COLOR_COUNT; ++i) {
        by_color_[i] = 0;
    }
    occupied_ = 0;

    for (int i = 0; i < SQUARE_COUNT; ++i) {
        board_[i] = NO_PIECE;
    }

    side_to_move_ = WHITE;
    en_passant_square_ = SQUARE_NONE;
    castling_rights_ = NO_CASTLING;
    halfmove_clock_ = 0;
    fullmove_number_ = 1;
    hash_ = {0, 0};

    state_history_.clear();
    position_history_.clear();
}

void Position::put_piece(Piece pc, Square sq) {
    board_[sq] = pc;
    Bitboard bb = square_bb(sq);

    PieceType pt = type_of(pc);
    Color c = color_of(pc);

    by_type_[pt] |= bb;
    by_color_[c] |= bb;
    occupied_ |= bb;
}

void Position::remove_piece(Square sq) {
    Piece pc = board_[sq];
    if (pc == NO_PIECE) return;

    board_[sq] = NO_PIECE;
    Bitboard bb = square_bb(sq);

    PieceType pt = type_of(pc);
    Color c = color_of(pc);

    by_type_[pt] &= ~bb;
    by_color_[c] &= ~bb;
    occupied_ &= ~bb;
}

void Position::move_piece(Square from, Square to) {
    Piece pc = board_[from];
    remove_piece(from);
    put_piece(pc, to);
}

void Position::set_fen(const std::string& fen) {
    clear();

    std::istringstream iss(fen);
    std::string board, side, castling, ep, halfmove, fullmove;

    iss >> board >> side >> castling >> ep >> halfmove >> fullmove;

    // Parse board
    Square sq = A8;
    for (char c : board) {
        if (c == '/') {
            sq = static_cast<Square>(sq - 16);  // Move to next rank
        } else if (c >= '1' && c <= '8') {
            sq = static_cast<Square>(sq + (c - '0'));  // Skip empty squares
        } else {
            // Parse piece
            Color color = (c >= 'A' && c <= 'Z') ? WHITE : BLACK;
            char piece_char = (color == WHITE) ? c : (c - 'a' + 'A');

            PieceType pt = NO_PIECE_TYPE;
            switch (piece_char) {
                case 'P': pt = PAWN; break;
                case 'N': pt = KNIGHT; break;
                case 'B': pt = BISHOP; break;
                case 'R': pt = ROOK; break;
                case 'Q': pt = QUEEN; break;
                case 'K': pt = KING; break;
            }

            if (pt != NO_PIECE_TYPE) {
                put_piece(make_piece(color, pt), sq);
            }
            sq = static_cast<Square>(sq + 1);
        }
    }

    // Parse side to move
    side_to_move_ = (side == "w") ? WHITE : BLACK;

    // Parse castling rights
    castling_rights_ = NO_CASTLING;
    if (castling != "-") {
        for (char c : castling) {
            switch (c) {
                case 'K': castling_rights_ |= WHITE_OO; break;
                case 'Q': castling_rights_ |= WHITE_OOO; break;
                case 'k': castling_rights_ |= BLACK_OO; break;
                case 'q': castling_rights_ |= BLACK_OOO; break;
            }
        }
    }

    // Parse en passant square
    en_passant_square_ = (ep == "-") ? SQUARE_NONE : string_to_square(ep);

    // Parse halfmove clock and fullmove number
    halfmove_clock_ = halfmove.empty() ? 0 : std::stoi(halfmove);
    fullmove_number_ = fullmove.empty() ? 1 : std::stoi(fullmove);

    // Compute initial hash
    compute_hash();
    position_history_.push_back(hash_);
}

std::string Position::fen() const {
    std::ostringstream oss;

    // Board
    for (int r = 7; r >= 0; --r) {
        int empty_count = 0;
        for (int f = 0; f < 8; ++f) {
            Square sq = make_square(static_cast<File>(f), static_cast<Rank>(r));
            Piece pc = board_[sq];

            if (pc == NO_PIECE) {
                empty_count++;
            } else {
                if (empty_count > 0) {
                    oss << empty_count;
                    empty_count = 0;
                }

                PieceType pt = type_of(pc);
                Color c = color_of(pc);

                char piece_char = ' ';
                switch (pt) {
                    case PAWN: piece_char = 'P'; break;
                    case KNIGHT: piece_char = 'N'; break;
                    case BISHOP: piece_char = 'B'; break;
                    case ROOK: piece_char = 'R'; break;
                    case QUEEN: piece_char = 'Q'; break;
                    case KING: piece_char = 'K'; break;
                    default: break;
                }

                if (c == BLACK) {
                    piece_char = piece_char - 'A' + 'a';
                }

                oss << piece_char;
            }
        }

        if (empty_count > 0) {
            oss << empty_count;
        }

        if (r > 0) {
            oss << '/';
        }
    }

    // Side to move
    oss << ' ' << (side_to_move_ == WHITE ? 'w' : 'b');

    // Castling rights
    oss << ' ';
    if (castling_rights_ == NO_CASTLING) {
        oss << '-';
    } else {
        if (castling_rights_ & WHITE_OO) oss << 'K';
        if (castling_rights_ & WHITE_OOO) oss << 'Q';
        if (castling_rights_ & BLACK_OO) oss << 'k';
        if (castling_rights_ & BLACK_OOO) oss << 'q';
    }

    // En passant square
    oss << ' ' << (en_passant_square_ == SQUARE_NONE ? "-" : square_to_string(en_passant_square_));

    // Halfmove clock and fullmove number
    oss << ' ' << halfmove_clock_ << ' ' << fullmove_number_;

    return oss.str();
}

void Position::compute_hash() {
    hash_ = {0, 0};

    // Hash all pieces
    for (Square sq = A1; sq < SQUARE_COUNT; sq = static_cast<Square>(sq + 1)) {
        Piece pc = board_[sq];
        if (pc != NO_PIECE) {
            hash_ ^= Zobrist::piece_hash(pc, sq);
        }
    }

    // Hash castling rights
    hash_ ^= Zobrist::castling_hash(castling_rights_);

    // Hash en passant square
    if (en_passant_square_ != SQUARE_NONE) {
        hash_ ^= Zobrist::en_passant_hash(file_of(en_passant_square_));
    }

    // Hash side to move (if black)
    if (side_to_move_ == BLACK) {
        hash_ ^= Zobrist::side_hash();
    }
}

void Position::make_move(const Move& move) {
    // Save current state for unmake
    StateInfo state;
    state.captured_piece = NO_PIECE;
    state.en_passant_square = en_passant_square_;
    state.castling_rights = castling_rights_;
    state.halfmove_clock = halfmove_clock_;
    state.hash = hash_;
    state_history_.push_back(state);

    Square from = move.from();
    Square to = move.to();
    Piece moving_piece = board_[from];
    PieceType pt = type_of(moving_piece);
    Color us = side_to_move_;
    Color them = ~us;

    // Remove old en passant from hash
    if (en_passant_square_ != SQUARE_NONE) {
        hash_ ^= Zobrist::en_passant_hash(file_of(en_passant_square_));
        en_passant_square_ = SQUARE_NONE;
    }

    // Remove old castling rights from hash
    hash_ ^= Zobrist::castling_hash(castling_rights_);

    // Increment halfmove clock
    halfmove_clock_++;

    // Handle captures
    if (!empty(to)) {
        state.captured_piece = board_[to];
        hash_ ^= Zobrist::piece_hash(state.captured_piece, to);
        remove_piece(to);
        halfmove_clock_ = 0;  // Reset on capture
    }

    // Handle special moves
    if (move.is_en_passant()) {
        // Remove captured pawn
        Square captured_sq = static_cast<Square>(static_cast<int>(to) + (us == WHITE ? SOUTH : NORTH));
        state.captured_piece = board_[captured_sq];
        hash_ ^= Zobrist::piece_hash(state.captured_piece, captured_sq);
        remove_piece(captured_sq);
        halfmove_clock_ = 0;
    } else if (move.is_castling()) {
        // Move rook
        Square rook_from, rook_to;
        if (to > from) {  // Kingside
            rook_from = static_cast<Square>(from + 3);
            rook_to = static_cast<Square>(from + 1);
        } else {  // Queenside
            rook_from = static_cast<Square>(from - 4);
            rook_to = static_cast<Square>(from - 1);
        }

        Piece rook = board_[rook_from];
        hash_ ^= Zobrist::piece_hash(rook, rook_from);
        hash_ ^= Zobrist::piece_hash(rook, rook_to);
        move_piece(rook_from, rook_to);
    }

    // Move piece
    hash_ ^= Zobrist::piece_hash(moving_piece, from);

    if (move.is_promotion()) {
        // Remove pawn, add promoted piece
        remove_piece(from);
        Piece promoted = make_piece(us, static_cast<PieceType>(move.promotion()));
        put_piece(promoted, to);
        hash_ ^= Zobrist::piece_hash(promoted, to);
        halfmove_clock_ = 0;  // Reset on pawn move
    } else {
        move_piece(from, to);
        hash_ ^= Zobrist::piece_hash(moving_piece, to);

        if (pt == PAWN) {
            halfmove_clock_ = 0;  // Reset on pawn move

            // Check for double pawn push
            if (abs(to - from) == 16) {
                en_passant_square_ = static_cast<Square>(static_cast<int>(from) + (us == WHITE ? NORTH : SOUTH));
                hash_ ^= Zobrist::en_passant_hash(file_of(en_passant_square_));
            }
        }
    }

    // Update castling rights
    update_castling_rights(from, to);

    // Add new castling rights to hash
    hash_ ^= Zobrist::castling_hash(castling_rights_);

    // Switch side to move
    hash_ ^= Zobrist::side_hash();
    side_to_move_ = them;

    // Update fullmove number
    if (us == BLACK) {
        fullmove_number_++;
    }

    // Add position to history for repetition detection
    position_history_.push_back(hash_);
}

void Position::unmake_move(const Move& move) {
    if (state_history_.empty()) return;

    // Restore state
    StateInfo state = state_history_.back();
    state_history_.pop_back();
    position_history_.pop_back();

    Square from = move.from();
    Square to = move.to();
    Color us = ~side_to_move_;  // We just switched, so switch back

    // Switch side back
    side_to_move_ = us;

    // Update fullmove number
    if (us == BLACK) {
        fullmove_number_--;
    }

    // Unmove piece
    if (move.is_promotion()) {
        remove_piece(to);
        put_piece(make_piece(us, PAWN), from);
    } else {
        move_piece(to, from);
    }

    // Restore captured piece
    if (state.captured_piece != NO_PIECE) {
        if (move.is_en_passant()) {
            Square captured_sq = static_cast<Square>(static_cast<int>(to) + (us == WHITE ? SOUTH : NORTH));
            put_piece(state.captured_piece, captured_sq);
        } else {
            put_piece(state.captured_piece, to);
        }
    }

    // Unmove rook for castling
    if (move.is_castling()) {
        Square rook_from, rook_to;
        if (to > from) {  // Kingside
            rook_from = static_cast<Square>(from + 3);
            rook_to = static_cast<Square>(from + 1);
        } else {  // Queenside
            rook_from = static_cast<Square>(from - 4);
            rook_to = static_cast<Square>(from - 1);
        }
        move_piece(rook_to, rook_from);
    }

    // Restore state
    en_passant_square_ = state.en_passant_square;
    castling_rights_ = state.castling_rights;
    halfmove_clock_ = state.halfmove_clock;
    hash_ = state.hash;
}

void Position::update_castling_rights(Square from, Square to) {
    // Remove castling rights if king or rook moves
    if (type_of(board_[from]) == KING) {
        if (side_to_move_ == WHITE) {
            castling_rights_ &= ~WHITE_CASTLING;
        } else {
            castling_rights_ &= ~BLACK_CASTLING;
        }
    }

    // Remove castling rights if rook moves or is captured
    if (from == A1 || to == A1) castling_rights_ &= ~WHITE_OOO;
    if (from == H1 || to == H1) castling_rights_ &= ~WHITE_OO;
    if (from == A8 || to == A8) castling_rights_ &= ~BLACK_OOO;
    if (from == H8 || to == H8) castling_rights_ &= ~BLACK_OO;
}

bool Position::is_repetition() const {
    if (position_history_.size() < 4) return false;

    int count = 0;
    // Check backwards from current position
    // Only need to check positions with same side to move (every 2 plies)
    for (int i = static_cast<int>(position_history_.size()) - 1; i >= 0; i -= 2) {
        if (position_history_[i] == hash_) {
            count++;
            if (count >= 3) return true;
        }

        // Stop at irreversible move (capture or pawn move)
        if (i < static_cast<int>(position_history_.size()) - halfmove_clock_) {
            break;
        }
    }

    return false;
}

bool Position::is_fifty_move_rule() const {
    return halfmove_clock_ >= 100;
}

bool Position::is_insufficient_material() const {
    // King vs King
    if (popcount(occupied_) == 2) return true;

    // King + minor piece vs King
    if (popcount(occupied_) == 3) {
        if (pieces(KNIGHT) || pieces(BISHOP)) return true;
    }

    // King + Bishop vs King + Bishop (same color squares)
    if (popcount(occupied_) == 4 && popcount(pieces(BISHOP)) == 2) {
        Bitboard white_bishops = pieces(BISHOP, WHITE);
        Bitboard black_bishops = pieces(BISHOP, BLACK);

        // Check if bishops are on same color squares
        bool white_on_light = white_bishops & 0xAA55AA55AA55AA55ULL;
        bool black_on_light = black_bishops & 0xAA55AA55AA55AA55ULL;

        if (white_on_light == black_on_light) return true;
    }

    return false;
}

bool Position::is_draw() const {
    return is_repetition() || is_fifty_move_rule() || is_insufficient_material();
}

bool Position::is_check() const {
    // Find our king
    Bitboard kings = pieces(KING, side_to_move_);
    if (!kings) return false;
    Square king_sq = lsb(kings);
    return is_attacked_by(king_sq, ~side_to_move_);
}

bool Position::is_attacked_by(Square sq, Color c) const {
    // Check pawn attacks
    if (MoveGenerator::pawn_attacks(~c, sq) & pieces(PAWN, c)) {
        return true;
    }

    // Check knight attacks
    if (MoveGenerator::knight_attacks(sq) & pieces(KNIGHT, c)) {
        return true;
    }

    // Check king attacks
    if (MoveGenerator::king_attacks(sq) & pieces(KING, c)) {
        return true;
    }

    // Check sliding piece attacks
    Bitboard bishops_queens = pieces(BISHOP, c) | pieces(QUEEN, c);
    if (MoveGenerator::bishop_attacks(sq, occupied_) & bishops_queens) {
        return true;
    }

    Bitboard rooks_queens = pieces(ROOK, c) | pieces(QUEEN, c);
    if (MoveGenerator::rook_attacks(sq, occupied_) & rooks_queens) {
        return true;
    }

    return false;
}

Bitboard Position::attackers_to(Square sq, Bitboard occupied) const {
    Bitboard attackers = 0;

    // Pawn attackers
    attackers |= MoveGenerator::pawn_attacks(WHITE, sq) & pieces(PAWN, BLACK);
    attackers |= MoveGenerator::pawn_attacks(BLACK, sq) & pieces(PAWN, WHITE);

    // Knight attackers
    attackers |= MoveGenerator::knight_attacks(sq) & pieces(KNIGHT);

    // King attackers
    attackers |= MoveGenerator::king_attacks(sq) & pieces(KING);

    // Bishop and queen attackers
    Bitboard bishops_queens = pieces(BISHOP) | pieces(QUEEN);
    attackers |= MoveGenerator::bishop_attacks(sq, occupied) & bishops_queens;

    // Rook and queen attackers
    Bitboard rooks_queens = pieces(ROOK) | pieces(QUEEN);
    attackers |= MoveGenerator::rook_attacks(sq, occupied) & rooks_queens;

    return attackers;
}

bool Position::is_checkmate() const {
    if (!is_check()) return false;

    // Generate all legal moves
    std::vector<Move> moves;
    MoveGenerator::generate_legal_moves(*this, moves);

    return moves.empty();
}

bool Position::is_stalemate() const {
    if (is_check()) return false;

    // Generate all legal moves
    std::vector<Move> moves;
    MoveGenerator::generate_legal_moves(*this, moves);

    return moves.empty();
}

} // namespace chess
