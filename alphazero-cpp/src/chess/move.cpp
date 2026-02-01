#include "chess/move.hpp"

namespace chess {

std::string Move::to_uci() const {
    if (!is_ok()) return "0000";

    std::string uci = square_to_string(from()) + square_to_string(to());

    if (is_promotion()) {
        const char promotion_chars[] = "nbrq";
        uci += promotion_chars[promotion()];
    }

    return uci;
}

Move Move::from_uci(const std::string& uci) {
    if (uci.length() < 4) return MOVE_NONE;

    Square from = string_to_square(uci.substr(0, 2));
    Square to = string_to_square(uci.substr(2, 2));

    if (from == SQUARE_NONE || to == SQUARE_NONE) return MOVE_NONE;

    // Check for promotion
    if (uci.length() == 5) {
        PieceType promotion = NO_PIECE_TYPE;
        switch (uci[4]) {
            case 'n': promotion = KNIGHT; break;
            case 'b': promotion = BISHOP; break;
            case 'r': promotion = ROOK; break;
            case 'q': promotion = QUEEN; break;
            default: return MOVE_NONE;
        }
        return Move(from, to, PROMOTION, promotion);
    }

    return Move(from, to);
}

} // namespace chess
