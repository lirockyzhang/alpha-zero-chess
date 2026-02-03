#include "encoding/position_encoder.hpp"
#include <cstring>
#include <algorithm>

namespace encoding {

std::vector<float> PositionEncoder::encode(const chess::Board& board,
                                           const std::vector<chess::Board>& position_history) {
    std::vector<float> buffer(TOTAL_SIZE, 0.0f);
    encode_to_buffer(board, buffer.data(), position_history);
    return buffer;
}

void PositionEncoder::encode_to_buffer(const chess::Board& board, float* buffer,
                                       const std::vector<chess::Board>& position_history) {
    // Clear buffer
    std::memset(buffer, 0, TOTAL_SIZE * sizeof(float));

    // Determine if we need to flip (always encode from current player's perspective)
    bool flip = (board.sideToMove() == chess::Color::BLACK);

    // Encode piece planes (12 channels: 6 piece types × 2 colors)
    encode_piece_planes(board, buffer, flip);

    // Encode auxiliary planes in NHWC layout
    // For each square, write to channels 12-17
    for (int rank = 0; rank < HEIGHT; ++rank) {
        for (int file = 0; file < WIDTH; ++file) {
            float* square_channels = buffer + rank * WIDTH * CHANNELS + file * CHANNELS;

            // Channels 12-13: Repetition counts (encoded in history planes now)
            // Channel 14: Color to move (always 1.0 from current player's perspective)
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
        }
    }

    // Channels 18-118: History planes (8 positions × 12 piece planes + 8 repetition planes)
    encode_history_planes(position_history, board, buffer, flip);
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

void PositionEncoder::encode_history_planes(const std::vector<chess::Board>& position_history,
                                            const chess::Board& current_board,
                                            float* buffer, bool flip) {
    // Encode last 8 positions (or fewer if history is shorter)
    // Each historical position gets:
    // - 12 piece planes (6 piece types × 2 colors)
    // - 1 repetition plane (1.0 if position matches current position)
    // Total: 13 planes per historical position × 8 = 104 planes
    // Channels 18-121 (all 8 positions fully encoded with 122 total channels)

    // Note: We encode up to 8 most recent positions (FULL encoding for all 8)
    // Channels 18-30: Position T-1 (12 pieces + 1 repetition)
    // Channels 31-43: Position T-2 (12 pieces + 1 repetition)
    // Channels 44-56: Position T-3 (12 pieces + 1 repetition)
    // Channels 57-69: Position T-4 (12 pieces + 1 repetition)
    // Channels 70-82: Position T-5 (12 pieces + 1 repetition)
    // Channels 83-95: Position T-6 (12 pieces + 1 repetition)
    // Channels 96-108: Position T-7 (12 pieces + 1 repetition)
    // Channels 109-121: Position T-8 (12 pieces + 1 repetition) - NOW COMPLETE!

    int history_size = std::min(8, static_cast<int>(position_history.size()));

    // Get current position's hash for repetition detection
    uint64_t current_hash = current_board.hash();

    for (int hist_idx = 0; hist_idx < history_size; ++hist_idx) {
        // Access history from most recent to oldest
        // position_history[size-1] = most recent (T-1)
        // position_history[size-2] = second most recent (T-2)
        int pos_idx = position_history.size() - 1 - hist_idx;
        const chess::Board& hist_board = position_history[pos_idx];

        // Starting channel for this historical position
        // Each position needs 13 channels (12 pieces + 1 repetition)
        int base_channel = 18 + hist_idx * 13;

        // Encode piece planes for this historical position (12 channels)
        const chess::PieceType piece_types[] = {
            chess::PieceType::PAWN,
            chess::PieceType::KNIGHT,
            chess::PieceType::BISHOP,
            chess::PieceType::ROOK,
            chess::PieceType::QUEEN,
            chess::PieceType::KING
        };

        // Encode current player's pieces (relative to CURRENT board's perspective)
        chess::Color current_color = current_board.sideToMove();
        for (int piece_idx = 0; piece_idx < 6; ++piece_idx) {
            chess::Bitboard bb = hist_board.pieces(piece_types[piece_idx], current_color);

            while (bb) {
                chess::Square sq = bb.pop();
                int square_idx = static_cast<int>(sq.index());
                int flipped_idx = flip_square(square_idx, flip);

                int rank = flipped_idx / 8;
                int file = flipped_idx % 8;
                float* square_channels = buffer + rank * WIDTH * CHANNELS + file * CHANNELS;
                square_channels[base_channel + piece_idx] = 1.0f;
            }
        }

        // Encode opponent's pieces (relative to CURRENT board's perspective)
        chess::Color opponent_color = (current_color == chess::Color::WHITE)
                                       ? chess::Color::BLACK
                                       : chess::Color::WHITE;
        for (int piece_idx = 0; piece_idx < 6; ++piece_idx) {
            chess::Bitboard bb = hist_board.pieces(piece_types[piece_idx], opponent_color);

            while (bb) {
                chess::Square sq = bb.pop();
                int square_idx = static_cast<int>(sq.index());
                int flipped_idx = flip_square(square_idx, flip);

                int rank = flipped_idx / 8;
                int file = flipped_idx % 8;
                float* square_channels = buffer + rank * WIDTH * CHANNELS + file * CHANNELS;
                square_channels[base_channel + 6 + piece_idx] = 1.0f;
            }
        }

        // Encode repetition marker (1.0 if this position matches current position)
        int rep_channel = base_channel + 12;
        bool is_repetition = (hist_board.hash() == current_hash);

        if (is_repetition) {
            // Mark all squares with 1.0 to indicate this position is a repetition
            for (int rank = 0; rank < HEIGHT; ++rank) {
                for (int file = 0; file < WIDTH; ++file) {
                    float* square_channels = buffer + rank * WIDTH * CHANNELS + file * CHANNELS;
                    square_channels[rep_channel] = 1.0f;
                }
            }
        }
    }
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
            } catch (const std::exception& e) {
                // Invalid FEN string - skip this position
                (void)e;  // Suppress unused warning
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
            } catch (const std::exception& e) {
                // Invalid FEN string - skip this position
                (void)e;  // Suppress unused warning
            }
        }
    }

    return success_count;
}

} // namespace encoding
