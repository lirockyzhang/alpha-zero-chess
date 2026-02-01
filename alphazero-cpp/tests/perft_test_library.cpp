#include "../third_party/chess-library/include/chess.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace chess;

// Perft: Performance test for move generation
uint64_t perft(Board& board, int depth) {
    if (depth == 0) return 1;

    Movelist moves;
    movegen::legalmoves(moves, board);

    if (depth == 1) return moves.size();

    uint64_t nodes = 0;
    for (const auto& move : moves) {
        board.makeMove(move);
        nodes += perft(board, depth - 1);
        board.unmakeMove(move);
    }

    return nodes;
}

// Divide: Show perft count for each root move (useful for debugging)
void divide(Board& board, int depth) {
    Movelist moves;
    movegen::legalmoves(moves, board);

    uint64_t total = 0;
    for (const auto& move : moves) {
        board.makeMove(move);
        uint64_t count = perft(board, depth - 1);
        board.unmakeMove(move);

        std::cout << uci::moveToUci(move) << ": " << count << std::endl;
        total += count;
    }

    std::cout << "\nTotal: " << total << std::endl;
}

struct PerftTest {
    std::string fen;
    int depth;
    uint64_t expected;
    std::string description;
};

int main() {
    std::cout << "=== AlphaZero Sync Chess Engine - Perft Validation (chess-library) ===" << std::endl;
    std::cout << std::endl;

    // Known perft results for validation
    // Source: https://www.chessprogramming.org/Perft_Results
    PerftTest tests[] = {
        // Starting position
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 20, "Starting position - depth 1"},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2, 400, "Starting position - depth 2"},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3, 8902, "Starting position - depth 3"},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281, "Starting position - depth 4"},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609, "Starting position - depth 5"},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324, "Starting position - depth 6 (CRITICAL)"},

        // Kiwipete position (tests complex scenarios)
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1, 48, "Kiwipete - depth 1"},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2039, "Kiwipete - depth 2"},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862, "Kiwipete - depth 3"},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603, "Kiwipete - depth 4"},

        // Position 3 (tests en passant and castling)
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1, 14, "Position 3 - depth 1"},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 2, 191, "Position 3 - depth 2"},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2812, "Position 3 - depth 3"},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238, "Position 3 - depth 4"},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624, "Position 3 - depth 5"},

        // Position 4 (tests promotions and discovered checks)
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 1, 6, "Position 4 - depth 1"},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2, 264, "Position 4 - depth 2"},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467, "Position 4 - depth 3"},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333, "Position 4 - depth 4"},

        // Position 5 (tests castling rights)
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 1, 44, "Position 5 - depth 1"},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 2, 1486, "Position 5 - depth 2"},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379, "Position 5 - depth 3"},

        // Position 6 (tests en passant)
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 1, 46, "Position 6 - depth 1"},
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 2, 2079, "Position 6 - depth 2"},
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3, 89890, "Position 6 - depth 3"},
    };

    int passed = 0;
    int failed = 0;
    bool critical_test_passed = false;

    for (const auto& test : tests) {
        Board board(test.fen);

        std::cout << "Testing: " << test.description << std::endl;
        std::cout << "FEN: " << test.fen << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        uint64_t result = perft(board, test.depth);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double nps = (duration.count() > 0) ? (result * 1000.0 / duration.count()) : 0;

        bool success = (result == test.expected);
        if (success) {
            passed++;
            std::cout << "✓ PASS: " << result << " nodes";
            std::cout << " (" << duration.count() << "ms, "
                      << std::fixed << std::setprecision(0) << nps << " nps)" << std::endl;

            // Check if this is the critical Perft(6) test
            if (test.depth == 6 && test.expected == 119060324) {
                critical_test_passed = true;
                std::cout << "*** CRITICAL TEST PASSED: Perft(6) = 119,060,324 ***" << std::endl;
            }
        } else {
            failed++;
            std::cout << "✗ FAIL: Expected " << test.expected << ", got " << result << std::endl;
            std::cout << "  Difference: " << static_cast<int64_t>(result - test.expected) << std::endl;

            // Show divide for debugging
            if (test.depth <= 3) {
                std::cout << "\n  Divide output for debugging:" << std::endl;
                Board debug_board(test.fen);
                divide(debug_board, test.depth);
            }
        }
        std::cout << std::endl;
    }

    // Summary
    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;

    if (failed == 0) {
        std::cout << "\n✓✓✓ ALL TESTS PASSED ✓✓✓" << std::endl;
        if (critical_test_passed) {
            std::cout << "\n*** CRITICAL VALIDATION COMPLETE ***" << std::endl;
            std::cout << "Perft(6) = 119,060,324 nodes verified!" << std::endl;
            std::cout << "\nYou may now proceed to Phase 2: MCTS Implementation" << std::endl;
        }
        return 0;
    } else {
        std::cout << "\n✗✗✗ SOME TESTS FAILED ✗✗✗" << std::endl;
        std::cout << "\nDO NOT PROCEED TO MCTS UNTIL ALL PERFT TESTS PASS!" << std::endl;
        std::cout << "Fix the move generation bugs first." << std::endl;
        return 1;
    }
}
