#include "encoding/move_encoder.hpp"
#include <cmath>
#include <algorithm>

namespace encoding {

// AlphaZero move encoding follows this structure:
// - Planes 0-55: Queen moves (8 directions × 7 distances)
// - Planes 56-63: Knight moves (8 possible knight moves)
// - Planes 64-72: Underpromotions (3 directions × 3 piece types)
// Total: 73 planes × 64 squares = 4672 possible moves
// But AlphaZero uses only 1858 moves (exact mapping varies by implementation)

int MoveEncoder::move_to_index(const chess::Move& move, const chess::Board& board) {
    MoveType type = classify_move(move, board);

    switch (type) {
        case MoveType::QUEEN_MOVE:
            return encode_queen_move(move, board);
        case MoveType::KNIGHT_MOVE:
            return encode_knight_move(move, board);
        case MoveType::UNDERPROMOTION:
            return encode_underpromotion(move, board);
        default:
            return -1;
    }
}

int MoveEncoder::move_to_index(const std::string& uci_move, const chess::Board& board) {
    chess::Move move = chess::uci::uciToMove(board, uci_move);
    if (move == chess::Move::NO_MOVE) {
        return -1;
    }
    return move_to_index(move, board);
}

chess::Move MoveEncoder::index_to_move(int index, const chess::Board& board) {
    if (index < 0 || index >= POLICY_SIZE) {
        return chess::Move::NO_MOVE;
    }

    // Determine move type based on index range
    // This is a simplified implementation - full AlphaZero uses a more complex mapping
    if (index < 56 * 64) {
        return decode_queen_move(index, board);
    } else if (index < 56 * 64 + 8 * 64) {
        return decode_knight_move(index, board);
    } else {
        return decode_underpromotion(index, board);
    }
}

std::string MoveEncoder::index_to_uci(int index, const chess::Board& board) {
    chess::Move move = index_to_move(index, board);
    if (move == chess::Move::NO_MOVE) {
        return "";
    }
    return chess::uci::moveToUci(move);
}

std::vector<int> MoveEncoder::get_legal_move_indices(const chess::Board& board) {
    std::vector<int> indices;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    for (const auto& move : moves) {
        int index = move_to_index(move, board);
        if (index >= 0) {
            indices.push_back(index);
        }
    }

    return indices;
}

MoveEncoder::MoveType MoveEncoder::classify_move(const chess::Move& move, const chess::Board& board) {
    // Check if it's a knight move
    chess::PieceType piece = board.at(move.from()).type();
    if (piece == chess::PieceType::KNIGHT) {
        return MoveType::KNIGHT_MOVE;
    }

    // Check if it's an underpromotion
    if (move.typeOf() == chess::Move::PROMOTION) {
        chess::PieceType promo = move.promotionType();
        if (promo != chess::PieceType::QUEEN) {
            return MoveType::UNDERPROMOTION;
        }
    }

    // Everything else is a queen move (includes rook, bishop, queen, king, pawn moves)
    return MoveType::QUEEN_MOVE;
}

int MoveEncoder::encode_queen_move(const chess::Move& move, const chess::Board& board) {
    int from = static_cast<int>(move.from().index());
    int to = static_cast<int>(move.to().index());

    // Flip for black's perspective
    bool flip = (board.sideToMove() == chess::Color::BLACK);
    if (flip) {
        from = 63 - from;
        to = 63 - to;
    }

    int direction = get_direction_index(from, to);
    if (direction < 0) return -1;

    int distance = get_distance(from, to, direction);
    if (distance < 1 || distance > 7) return -1;

    // Index: from_square * 56 + direction * 7 + (distance - 1)
    return from * 56 + direction * 7 + (distance - 1);
}

int MoveEncoder::encode_knight_move(const chess::Move& move, const chess::Board& board) {
    int from = static_cast<int>(move.from().index());
    int to = static_cast<int>(move.to().index());

    // Flip for black's perspective (must match position encoder)
    bool flip = (board.sideToMove() == chess::Color::BLACK);
    if (flip) {
        from = 63 - from;
        to = 63 - to;
    }

    int from_rank = from / 8;
    int from_file = from % 8;
    int to_rank = to / 8;
    int to_file = to % 8;

    int rank_diff = to_rank - from_rank;
    int file_diff = to_file - from_file;

    // Map knight move to index (0-7)
    int knight_index = -1;
    if (rank_diff == 2 && file_diff == 1) knight_index = 0;
    else if (rank_diff == 1 && file_diff == 2) knight_index = 1;
    else if (rank_diff == -1 && file_diff == 2) knight_index = 2;
    else if (rank_diff == -2 && file_diff == 1) knight_index = 3;
    else if (rank_diff == -2 && file_diff == -1) knight_index = 4;
    else if (rank_diff == -1 && file_diff == -2) knight_index = 5;
    else if (rank_diff == 1 && file_diff == -2) knight_index = 6;
    else if (rank_diff == 2 && file_diff == -1) knight_index = 7;

    if (knight_index < 0) return -1;

    // Index: 56*64 + from_square * 8 + knight_index
    return 56 * 64 + from * 8 + knight_index;
}

int MoveEncoder::encode_underpromotion(const chess::Move& move, const chess::Board& board) {
    int from = static_cast<int>(move.from().index());
    int to = static_cast<int>(move.to().index());

    // Flip for black's perspective (must match position encoder)
    bool flip = (board.sideToMove() == chess::Color::BLACK);
    if (flip) {
        from = 63 - from;
        to = 63 - to;
    }

    int from_file = from % 8;
    int to_file = to % 8;
    int file_diff = to_file - from_file;

    // Direction: -1 (left), 0 (straight), 1 (right)
    int direction = file_diff + 1;  // Maps to 0, 1, 2
    if (direction < 0 || direction > 2) return -1;

    // Piece type: Knight=0, Bishop=1, Rook=2
    chess::PieceType promo = move.promotionType();
    int piece_index = -1;
    if (promo == chess::PieceType::KNIGHT) piece_index = 0;
    else if (promo == chess::PieceType::BISHOP) piece_index = 1;
    else if (promo == chess::PieceType::ROOK) piece_index = 2;

    if (piece_index < 0) return -1;

    // Index: 56*64 + 8*64 + from_square * 9 + direction * 3 + piece_index
    return 56 * 64 + 8 * 64 + from * 9 + direction * 3 + piece_index;
}

chess::Move MoveEncoder::decode_queen_move(int index, const chess::Board& board) {
    // Extract from_square and move parameters from index
    int from = index / 56;
    int remainder = index % 56;
    int direction = remainder / 7;
    int distance = (remainder % 7) + 1;

    // Flip for black's perspective
    bool flip = (board.sideToMove() == chess::Color::BLACK);
    if (flip) {
        from = 63 - from;
    }

    // Calculate to_square based on direction and distance
    int from_rank = from / 8;
    int from_file = from % 8;

    // Direction vectors: N, NE, E, SE, S, SW, W, NW
    const int rank_dirs[] = {1, 1, 0, -1, -1, -1, 0, 1};
    const int file_dirs[] = {0, 1, 1, 1, 0, -1, -1, -1};

    int to_rank = from_rank + rank_dirs[direction] * distance;
    int to_file = from_file + file_dirs[direction] * distance;

    // Check bounds
    if (to_rank < 0 || to_rank > 7 || to_file < 0 || to_file > 7) {
        return chess::Move::NO_MOVE;
    }

    int to = to_rank * 8 + to_file;

    // Flip back if needed
    if (flip) {
        from = 63 - from;
        to = 63 - to;
    }

    // Create move
    chess::Square from_sq = static_cast<chess::Square>(from);
    chess::Square to_sq = static_cast<chess::Square>(to);

    // Check if it's a promotion (pawn reaching last rank)
    chess::PieceType piece = board.at(from_sq).type();
    if (piece == chess::PieceType::PAWN) {
        int to_rank_abs = to / 8;
        if (to_rank_abs == 0 || to_rank_abs == 7) {
            // Queen promotion
            return chess::Move::make<chess::Move::PROMOTION>(from_sq, to_sq, chess::PieceType::QUEEN);
        }
    }

    return chess::Move::make(from_sq, to_sq);
}

chess::Move MoveEncoder::decode_knight_move(int index, const chess::Board& board) {
    // Extract from_square and knight move index
    int offset = index - 56 * 64;
    int from = offset / 8;
    int knight_index = offset % 8;

    // Knight move offsets
    const int rank_offsets[] = {2, 1, -1, -2, -2, -1, 1, 2};
    const int file_offsets[] = {1, 2, 2, 1, -1, -2, -2, -1};

    int from_rank = from / 8;
    int from_file = from % 8;

    int to_rank = from_rank + rank_offsets[knight_index];
    int to_file = from_file + file_offsets[knight_index];

    // Check bounds
    if (to_rank < 0 || to_rank > 7 || to_file < 0 || to_file > 7) {
        return chess::Move::NO_MOVE;
    }

    int to = to_rank * 8 + to_file;

    chess::Square from_sq = static_cast<chess::Square>(from);
    chess::Square to_sq = static_cast<chess::Square>(to);

    return chess::Move::make(from_sq, to_sq);
}

chess::Move MoveEncoder::decode_underpromotion(int index, const chess::Board& board) {
    // Extract from_square and promotion parameters
    int offset = index - 56 * 64 - 8 * 64;
    int from = offset / 9;
    int remainder = offset % 9;
    int direction = remainder / 3;  // 0=left, 1=straight, 2=right
    int piece_index = remainder % 3;  // 0=knight, 1=bishop, 2=rook

    int from_file = from % 8;
    int to_file = from_file + (direction - 1);  // -1, 0, +1

    // Check bounds
    if (to_file < 0 || to_file > 7) {
        return chess::Move::NO_MOVE;
    }

    // Determine to_rank based on color
    int from_rank = from / 8;
    int to_rank;
    if (board.sideToMove() == chess::Color::WHITE) {
        to_rank = 7;  // White promotes to rank 7
    } else {
        to_rank = 0;  // Black promotes to rank 0
    }

    int to = to_rank * 8 + to_file;

    chess::Square from_sq = static_cast<chess::Square>(from);
    chess::Square to_sq = static_cast<chess::Square>(to);

    // Map piece index to piece type
    chess::PieceType promo_type;
    if (piece_index == 0) promo_type = chess::PieceType::KNIGHT;
    else if (piece_index == 1) promo_type = chess::PieceType::BISHOP;
    else promo_type = chess::PieceType::ROOK;

    return chess::Move::make<chess::Move::PROMOTION>(from_sq, to_sq, promo_type);
}

int MoveEncoder::get_direction_index(int from_square, int to_square) {
    int from_rank = from_square / 8;
    int from_file = from_square % 8;
    int to_rank = to_square / 8;
    int to_file = to_square % 8;

    int rank_diff = to_rank - from_rank;
    int file_diff = to_file - from_file;

    // Normalize to direction
    int rank_dir = (rank_diff == 0) ? 0 : (rank_diff > 0 ? 1 : -1);
    int file_dir = (file_diff == 0) ? 0 : (file_diff > 0 ? 1 : -1);

    // Map to direction index (0-7)
    // N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
    if (rank_dir == 1 && file_dir == 0) return 0;  // N
    if (rank_dir == 1 && file_dir == 1) return 1;  // NE
    if (rank_dir == 0 && file_dir == 1) return 2;  // E
    if (rank_dir == -1 && file_dir == 1) return 3; // SE
    if (rank_dir == -1 && file_dir == 0) return 4; // S
    if (rank_dir == -1 && file_dir == -1) return 5; // SW
    if (rank_dir == 0 && file_dir == -1) return 6; // W
    if (rank_dir == 1 && file_dir == -1) return 7; // NW

    return -1;
}

int MoveEncoder::get_distance(int from_square, int to_square, int direction) {
    int from_rank = from_square / 8;
    int from_file = from_square % 8;
    int to_rank = to_square / 8;
    int to_file = to_square % 8;

    int rank_diff = std::abs(to_rank - from_rank);
    int file_diff = std::abs(to_file - from_file);

    return std::max(rank_diff, file_diff);
}

} // namespace encoding
