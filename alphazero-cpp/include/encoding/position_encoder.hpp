#pragma once

#include "../third_party/chess-library/include/chess.hpp"
#include <vector>
#include <cstdint>

namespace encoding {

// AlphaZero position encoding
// 119 planes total:
// - 12 planes: piece positions (6 piece types × 2 colors)
// - 2 planes: repetition counts (1-fold, 2-fold)
// - 1 plane: color to move
// - 1 plane: total move count
// - 1 plane: castling rights (encoded as 4 bits)
// - 1 plane: no-progress count (50-move rule)
// - 102 planes: history (8 previous positions × 12 piece planes + 8 × repetition)
//
// Note: The original AlphaZero paper uses 119 planes, but the exact breakdown
// varies by implementation. This follows the Leela Chess Zero approach.

class PositionEncoder {
public:
    // NHWC (channels-last) layout for Tensor Core performance
    // Shape: (height, width, channels) = (8, 8, 119)
    static constexpr int HEIGHT = 8;
    static constexpr int WIDTH = 8;
    static constexpr int CHANNELS = 119;
    static constexpr int TOTAL_SIZE = HEIGHT * WIDTH * CHANNELS;

    // Legacy constants for compatibility
    static constexpr int NUM_PLANES = CHANNELS;
    static constexpr int BOARD_SIZE = HEIGHT;

    // Encode position from current player's perspective
    // Returns flat array of size 119 × 8 × 8 = 7616
    static std::vector<float> encode(const chess::Board& board);

    // Encode position with zero-copy (writes directly to provided buffer)
    // Buffer must have size >= TOTAL_SIZE
    static void encode_to_buffer(const chess::Board& board, float* buffer);

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

    // Convert square index based on perspective (flip for black)
    static inline int flip_square(int square, bool flip) {
        if (!flip) return square;
        int rank = square / 8;
        int file = square % 8;
        return (7 - rank) * 8 + file;
    }
};

} // namespace encoding
