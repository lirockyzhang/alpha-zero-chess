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
    // Pre-compute scalar values outside the loop
    chess::Color current_color = board.sideToMove();
    chess::Color opponent_color = (current_color == chess::Color::WHITE)
                                    ? chess::Color::BLACK
                                    : chess::Color::WHITE;

    // Channel 13: Move count
    float move_count_val = std::min(1.0f, board.fullMoveNumber() / 100.0f);

    // Channels 14-17: Castling rights (4 binary planes)
    float p1_kingside  = board.castlingRights().has(current_color, chess::Board::CastlingRights::Side::KING_SIDE)  ? 1.0f : 0.0f;
    float p1_queenside = board.castlingRights().has(current_color, chess::Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0f : 0.0f;
    float p2_kingside  = board.castlingRights().has(opponent_color, chess::Board::CastlingRights::Side::KING_SIDE)  ? 1.0f : 0.0f;
    float p2_queenside = board.castlingRights().has(opponent_color, chess::Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0f : 0.0f;

    // Channel 18: No-progress count
    float no_progress_val = std::min(1.0f, board.halfMoveClock() / 100.0f);

    for (int rank = 0; rank < HEIGHT; ++rank) {
        for (int file = 0; file < WIDTH; ++file) {
            float* square_channels = buffer + rank * WIDTH * CHANNELS + file * CHANNELS;

            // Channel 12: Color to move (always 1.0 from current player's perspective)
            square_channels[12] = 1.0f;

            // Channel 13: Move count
            square_channels[13] = move_count_val;

            // Channels 14-17: Castling rights (binary planes)
            square_channels[14] = p1_kingside;
            square_channels[15] = p1_queenside;
            square_channels[16] = p2_kingside;
            square_channels[17] = p2_queenside;

            // Channel 18: No-progress count (fifty-move rule: draw at halfmoves >= 100)
            square_channels[18] = no_progress_val;
        }
    }

    // Channels 19-122: History planes (8 positions × 13 planes each)
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
    // Channels 19-122 (all 8 positions fully encoded with 123 total channels)

    // Channels 19-31:  Position T-1 (12 pieces + 1 repetition)
    // Channels 32-44:  Position T-2
    // Channels 45-57:  Position T-3
    // Channels 58-70:  Position T-4
    // Channels 71-83:  Position T-5
    // Channels 84-96:  Position T-6
    // Channels 97-109: Position T-7
    // Channels 110-122: Position T-8

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
        int base_channel = 19 + hist_idx * 13;

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
