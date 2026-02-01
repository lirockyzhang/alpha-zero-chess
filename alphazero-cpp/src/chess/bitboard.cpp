#include "chess/bitboard.hpp"
#include <sstream>
#include <iomanip>

namespace chess {

std::string square_to_string(Square sq) {
    if (sq == SQUARE_NONE) return "-";

    char file = 'a' + file_of(sq);
    char rank = '1' + rank_of(sq);
    return std::string{file, rank};
}

Square string_to_square(const std::string& str) {
    if (str.length() != 2) return SQUARE_NONE;

    char file = str[0];
    char rank = str[1];

    if (file < 'a' || file > 'h') return SQUARE_NONE;
    if (rank < '1' || rank > '8') return SQUARE_NONE;

    return make_square(static_cast<File>(file - 'a'), static_cast<Rank>(rank - '1'));
}

std::string bitboard_to_string(Bitboard bb) {
    std::ostringstream oss;
    oss << "\n";

    // Print from rank 8 to rank 1 (top to bottom)
    for (int r = 7; r >= 0; --r) {
        oss << (r + 1) << " ";
        for (int f = 0; f < 8; ++f) {
            Square sq = make_square(static_cast<File>(f), static_cast<Rank>(r));
            oss << (bb & square_bb(sq) ? "X " : ". ");
        }
        oss << "\n";
    }

    oss << "  a b c d e f g h\n";
    oss << "Bitboard: 0x" << std::hex << std::setw(16) << std::setfill('0') << bb << std::dec << "\n";
    oss << "Popcount: " << popcount(bb) << "\n";

    return oss.str();
}

} // namespace chess
