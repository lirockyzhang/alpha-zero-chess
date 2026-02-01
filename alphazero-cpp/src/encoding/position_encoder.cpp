#include "encoding/position_encoder.hpp"
#include <cstring>
#include <algorithm>

namespace encoding {

std::vector<float> PositionEncoder::encode(const chess::Board& board) {
    std::vector<float> buffer(TOTAL_SIZE, 0.0f);
    encode_to_buffer(board, buffer.data());
    return buffer;
}

void PositionEncoder::encode_to_buffer(const chess::Board& board, float* buffer) {
    // Clear buffer
    std::memset(buffer, 0, TOTAL_SIZE * sizeof(float));

    // Determine if we need to flip (always encode from current player's perspective)
    bool flip = (board.sideToMove() == chess::Color::BLACK);

    // Encode piece planes (12 channels: 6 piece types × 2 colors)
    encode_piece_planes(board, buffer, flip);

    // Encode auxiliary planes in NHWC layout
    // For each square, write to channels 12-18
    for (int rank = 0; rank < HEIGHT; ++rank) {
        for (int file = 0; file < WIDTH; ++file) {
            float* square_channels = buffer + rank * WIDTH * CHANNELS + file * CHANNELS;

            // Channels 12-13: Repetition counts (placeholder - zeros)
            // Channels 14: Color to move (always 1.0 from current player's perspective)
            square_channels[14] = 1.0f;

            // Channel 15: Move count
            int full_moves = board.fullMoveNumber();
            square_channels[15] = std::min(1.0f, full_moves / 100.0f);

            // Channel 16: Castling rights
            chess::Color current_color = board.sideToMove();
            bool kingside = board.castlingRights().has(current_color, chess::Board::CastlingRights::Side::KING_SIDE);
            bool queenside = board.castlingRights().has(current_color, chess::Board::CastlingRights::Side::QUEEN_SIDE);
            float castling_value = 0.0f;
            if (kingside && queenside) castling_value = 1.0f;
            else if (kingside) castling_value = 0.67f;
            else if (queenside) castling_value = 0.33f;
            square_channels[16] = castling_value;

            // Channel 17: No-progress count (50-move rule)
            int halfmoves = board.halfMoveClock();
            square_channels[17] = std::min(1.0f, halfmoves / 50.0f);

            // Channels 18-118: History planes (placeholder - zeros)
        }
    }
}

void PositionEncoder::encode_piece_planes(const chess::Board& board, float* buffer, bool flip) {
    // NHWC layout: (height, width, channels) = (8, 8, 119)
    // For each square (rank, file), all 119 channels are contiguous in memory
    // This is optimal for Tensor Cores and GPU memory coalescing

    // Piece type order: Pawn, Knight, Bishop, Rook, Queen, King
    const chess::PieceType piece_types[] = {
        chess::PieceType::PAWN,
        chess::PieceType::KNIGHT,
        chess::PieceType::BISHOP,
        chess::PieceType::ROOK,
        chess::PieceType::QUEEN,
        chess::PieceType::KING
    };

    // Encode current player's pieces (channels 0-5)
    chess::Color current_color = board.sideToMove();
    for (int piece_idx = 0; piece_idx < 6; ++piece_idx) {
        chess::Bitboard bb = board.pieces(piece_types[piece_idx], current_color);

        while (bb) {
            chess::Square sq = bb.pop();
            int square_idx = static_cast<int>(sq.index());
            int flipped_idx = flip_square(square_idx, flip);

            // NHWC: Calculate position in channels-last layout
            int rank = flipped_idx / 8;
            int file = flipped_idx % 8;
            float* square_channels = buffer + rank * WIDTH * CHANNELS + file * CHANNELS;
            square_channels[piece_idx] = 1.0f;
        }
    }

    // Encode opponent's pieces (channels 6-11)
    chess::Color opponent_color = (current_color == chess::Color::WHITE)
                                   ? chess::Color::BLACK
                                   : chess::Color::WHITE;
    for (int piece_idx = 0; piece_idx < 6; ++piece_idx) {
        chess::Bitboard bb = board.pieces(piece_types[piece_idx], opponent_color);

        while (bb) {
            chess::Square sq = bb.pop();
            int square_idx = static_cast<int>(sq.index());
            int flipped_idx = flip_square(square_idx, flip);

            // NHWC: Calculate position in channels-last layout
            int rank = flipped_idx / 8;
            int file = flipped_idx % 8;
            float* square_channels = buffer + rank * WIDTH * CHANNELS + file * CHANNELS;
            square_channels[6 + piece_idx] = 1.0f;
        }
    }
}

// These functions are now integrated into encode_to_buffer for NHWC layout
void PositionEncoder::encode_repetition_planes(const chess::Board& board, float* buffer) {
    // Integrated into encode_to_buffer
}

void PositionEncoder::encode_color_plane(const chess::Board& board, float* buffer) {
    // Integrated into encode_to_buffer
}

void PositionEncoder::encode_move_count_plane(const chess::Board& board, float* buffer) {
    // Integrated into encode_to_buffer
}

void PositionEncoder::encode_castling_plane(const chess::Board& board, float* buffer) {
    // Integrated into encode_to_buffer
}

void PositionEncoder::encode_no_progress_plane(const chess::Board& board, float* buffer) {
    // Integrated into encode_to_buffer
}

int PositionEncoder::encode_batch(const std::vector<std::string>& fens, float* buffer, bool use_parallel) {
    int batch_size = static_cast<int>(fens.size());
    int success_count = 0;

    if (use_parallel) {
        // OpenMP parallel encoding
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < batch_size; ++i) {
            try {
                chess::Board board(fens[i]);
                float* position_buffer = buffer + i * TOTAL_SIZE;
                encode_to_buffer(board, position_buffer);
                #pragma omp atomic
                success_count++;
            } catch (...) {
                // Skip invalid FEN strings
            }
        }
    } else {
        // Sequential encoding (for comparison)
        for (int i = 0; i < batch_size; ++i) {
            try {
                chess::Board board(fens[i]);
                float* position_buffer = buffer + i * TOTAL_SIZE;
                encode_to_buffer(board, position_buffer);
                success_count++;
            } catch (...) {
                // Skip invalid FEN strings
            }
        }
    }

    return success_count;
}

} // namespace encoding
