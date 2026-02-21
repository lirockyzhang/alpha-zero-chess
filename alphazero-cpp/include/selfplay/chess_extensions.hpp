#pragma once

#include "../third_party/chess-library/include/chess.hpp"

namespace selfplay {

/// Thin extension of chess::Board that exposes pushHistoryHash() for reanalysis.
/// This avoids modifying the chess-library submodule directly.
/// prev_states_ is protected in Board, so the subclass can access it.
class ReanalysisBoard : public chess::Board {
   public:
    using chess::Board::Board;
    using chess::Board::operator=;

    /// Inject a historical Zobrist hash into prev_states_ for repetition detection.
    /// Only the hash matters for repetition checks; other State fields are placeholders.
    void pushHistoryHash(std::uint64_t hash) {
        prev_states_.emplace_back(hash, CastlingRights{}, chess::Square::underlying::NO_SQ, 0,
                                  chess::Piece::NONE);
    }
};

}  // namespace selfplay
