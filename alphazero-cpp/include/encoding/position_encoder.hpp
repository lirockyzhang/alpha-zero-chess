#pragma once

#include "../third_party/chess-library/include/chess.hpp"
#include <vector>
#include <cstdint>

namespace encoding {

// AlphaZero position encoding
// 123 planes total:
// - 12 planes (ch 0-11): piece positions (6 piece types × 2 colors)
// - 1 plane (ch 12): color to move (always 1.0)
// - 1 plane (ch 13): total move count (fullmoves / 100, clamped to 1.0)
// - 4 planes (ch 14-17): castling rights (binary)
//   - ch 14: current player kingside
//   - ch 15: current player queenside
//   - ch 16: opponent kingside
//   - ch 17: opponent queenside
// - 1 plane (ch 18): no-progress count (halfmoves / 100, clamped to 1.0)
// - 104 planes (ch 19-122): history (8 previous positions × 13 planes each)
//   - Each historical position: 12 piece planes + 1 repetition marker

class PositionEncoder {
public:
    // NHWC (channels-last) layout for Tensor Core performance
    // Shape: (height, width, channels) = (8, 8, 123)
    static constexpr int HEIGHT = 8;
    static constexpr int WIDTH = 8;
    static constexpr int CHANNELS = 123;
    static constexpr int TOTAL_SIZE = HEIGHT * WIDTH * CHANNELS;

    // Legacy constants for compatibility
    static constexpr int NUM_PLANES = CHANNELS;
    static constexpr int BOARD_SIZE = HEIGHT;

    // Encode position from current player's perspective
    // Returns flat array of size 123 × 8 × 8 = 7872
    // position_history: Last N positions (up to 8) for history encoding
    static std::vector<float> encode(const chess::Board& board,
                                     const std::vector<chess::Board>& position_history = {});

    // Encode position with zero-copy (writes directly to provided buffer)
    // Buffer must have size >= TOTAL_SIZE
    // position_history: Last N positions (up to 8) for history encoding
    static void encode_to_buffer(const chess::Board& board, float* buffer,
                                 const std::vector<chess::Board>& position_history = {});

    // Batch encoding with optional OpenMP parallelization
    // Encodes multiple positions in parallel to reduce Python call overhead
    // fens: vector of FEN strings to encode
    // buffer: output buffer of size batch_size * TOTAL_SIZE
    // use_parallel: if true, use OpenMP parallelization (default: true)
    // Returns number of positions successfully encoded
    static int encode_batch(const std::vector<std::string>& fens, float* buffer, bool use_parallel = true);

private:
    // Helper functions
    static void encode_piece_planes(const chess::Board& board, float* buffer, bool flip);
    static void encode_repetition_planes(const chess::Board& board, float* buffer);
    static void encode_color_plane(const chess::Board& board, float* buffer);
    static void encode_move_count_plane(const chess::Board& board, float* buffer);
    static void encode_castling_plane(const chess::Board& board, float* buffer);
    static void encode_no_progress_plane(const chess::Board& board, float* buffer);
    static void encode_history_planes(const std::vector<chess::Board>& position_history,
                                      const chess::Board& current_board,
                                      float* buffer, bool flip);

    // Convert square index based on perspective (flip for black)
    static inline int flip_square(int square, bool flip) {
        if (!flip) return square;
        int rank = square / 8;
        int file = square % 8;
        return (7 - rank) * 8 + file;
    }
};

} // namespace encoding
