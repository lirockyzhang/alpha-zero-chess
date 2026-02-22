// Tests for position history reconstruction in MCTS leaf encoding.
//
// When collect_leaves() / collect_leaves_async() encode leaf observations,
// they must reconstruct per-leaf history (channels 19-122) by walking parent
// pointers and replaying moves, rather than using the static position_history_.
//
// These tests verify:
//   1. Correct number of non-zero history slots at various depths
//   2. Byte-for-byte match against reference encodings
//   3. Proper windowing when total history > 8
//   4. Sync and async pipelines produce identical encodings
//   5. Correct behavior with pre-root position_history

#include "mcts/search.hpp"
#include "mcts/node_pool.hpp"
#include "encoding/position_encoder.hpp"
#include "encoding/move_encoder.hpp"
#include "../third_party/chess-library/include/chess.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>

using namespace mcts;
using namespace chess;

static constexpr int OBS_SIZE  = encoding::PositionEncoder::TOTAL_SIZE;  // 8*8*123 = 7872
static constexpr int POL_SIZE  = encoding::MoveEncoder::POLICY_SIZE;     // 4672
static constexpr int CHANNELS  = encoding::PositionEncoder::CHANNELS;    // 123

// ============================================================================
// Helper Functions
// ============================================================================

// Check if all 13 channels for a history slot are zero across all 64 squares.
// hist_idx: 0 = T-1 (channels 19-31), 1 = T-2 (channels 32-44), etc.
static bool history_planes_all_zero(const float* obs, int hist_idx) {
    int base_ch = 19 + hist_idx * 13;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            for (int c = 0; c < 13; ++c) {
                int offset = rank * 8 * CHANNELS + file * CHANNELS + (base_ch + c);
                if (obs[offset] != 0.0f) return false;
            }
        }
    }
    return true;
}

// Check if a history slot has any non-zero data.
static bool history_planes_nonzero(const float* obs, int hist_idx) {
    return !history_planes_all_zero(obs, hist_idx);
}

// Count how many of the 8 history slots have non-zero data.
static int count_nonzero_history_slots(const float* obs) {
    int count = 0;
    for (int i = 0; i < 8; ++i) {
        if (history_planes_nonzero(obs, i)) count++;
    }
    return count;
}

// Compare two observation buffers element-by-element.
// Returns true if identical. On mismatch, prints the first difference.
static bool compare_obs(const float* actual, const float* expected, int size, const char* label) {
    for (int i = 0; i < size; ++i) {
        if (actual[i] != expected[i]) {
            int square = i / CHANNELS;
            int channel = i % CHANNELS;
            int rank = square / 8;
            int file = square % 8;
            std::cout << "  MISMATCH [" << label << "] at index " << i
                      << " (rank=" << rank << " file=" << file << " ch=" << channel
                      << "): actual=" << actual[i] << " expected=" << expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Create a policy vector with 100% prior on one specific move.
// This forces deterministic PUCT selection of that move.
static std::vector<float> make_focused_policy(const Board& board, const std::string& uci_move) {
    std::vector<float> policy(POL_SIZE, 0.0f);
    Move move = uci::uciToMove(board, uci_move);
    int idx = encoding::MoveEncoder::move_to_index(move, board);
    if (idx >= 0 && idx < POL_SIZE) {
        policy[idx] = 1.0f;
    }
    return policy;
}

// Create a uniform policy vector.
static std::vector<float> uniform_policy() {
    return std::vector<float>(POL_SIZE, 1.0f / 20.0f);
}

// ============================================================================
// Test 1 — Empty history, depth-1 leaf has root in T-1
// ============================================================================
bool test_reanalysis_depth1() {
    std::cout << "=== Test 1: Depth-1 leaf, empty history ===" << std::endl;

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 10;
    config.use_gumbel = false;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    Board board;  // Starting position
    auto focused_e2e4 = make_focused_policy(board, "e2e4");
    search.init_search(board, focused_e2e4, 0.0f, {});

    std::vector<float> obs(OBS_SIZE, 0.0f);
    std::vector<float> mask(POL_SIZE, 0.0f);
    int collected = search.collect_leaves(obs.data(), mask.data(), 1);

    if (collected != 1) {
        std::cout << "  FAIL: expected 1 leaf, got " << collected << std::endl;
        return false;
    }

    // T-1 (hist_idx=0) should contain root position (starting pos) -> non-zero
    if (!history_planes_nonzero(obs.data(), 0)) {
        std::cout << "  FAIL: T-1 (channels 19-31) should be non-zero" << std::endl;
        return false;
    }

    // T-2 through T-8 should be zero (no more history)
    for (int i = 1; i < 8; ++i) {
        if (!history_planes_all_zero(obs.data(), i)) {
            std::cout << "  FAIL: T-" << (i+1) << " should be zero" << std::endl;
            return false;
        }
    }

    // Exact match: encode after-e4 position with history=[starting_board]
    Board board_after_e4 = board;
    board_after_e4.makeMove(uci::uciToMove(board, "e2e4"));
    std::vector<Board> ref_history = {board};
    std::vector<float> ref(OBS_SIZE, 0.0f);
    encoding::PositionEncoder::encode_to_buffer(board_after_e4, ref.data(), ref_history);

    if (!compare_obs(obs.data(), ref.data(), OBS_SIZE, "depth1-exact")) {
        std::cout << "  FAIL: obs doesn't match reference encoding" << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ============================================================================
// Test 2 — Empty history, depth-2 leaf has 2 in-tree boards
// ============================================================================
bool test_reanalysis_depth2() {
    std::cout << "=== Test 2: Depth-2 leaf, empty history ===" << std::endl;

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 20;
    config.use_gumbel = false;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    Board board;  // Starting position
    auto focused_e2e4 = make_focused_policy(board, "e2e4");
    search.init_search(board, focused_e2e4, 0.0f, {});

    // Depth 1: collect leaf (after e4)
    std::vector<float> obs(OBS_SIZE, 0.0f);
    std::vector<float> mask(POL_SIZE, 0.0f);
    int collected = search.collect_leaves(obs.data(), mask.data(), 1);
    if (collected != 1) {
        std::cout << "  FAIL: depth-1 collect got " << collected << std::endl;
        return false;
    }

    // Expand depth-1 leaf with uniform policy
    std::vector<std::vector<float>> policies = {uniform_policy()};
    std::vector<float> values = {0.0f};
    search.update_leaves(policies, values, nullptr);

    // Depth 2: collect leaf (after e4 + some response)
    std::fill(obs.begin(), obs.end(), 0.0f);
    collected = search.collect_leaves(obs.data(), mask.data(), 1);
    if (collected != 1) {
        std::cout << "  FAIL: depth-2 collect got " << collected << std::endl;
        return false;
    }

    // T-1 and T-2 should be non-zero, T-3..T-8 zero
    int nonzero = count_nonzero_history_slots(obs.data());
    if (nonzero != 2) {
        std::cout << "  FAIL: expected 2 non-zero history slots, got " << nonzero << std::endl;
        return false;
    }

    std::cout << "  Non-zero history slots: " << nonzero << " (expected 2)" << std::endl;
    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ============================================================================
// Test 3 — Empty history, depth-3 leaf has 3 in-tree boards
// ============================================================================
bool test_reanalysis_depth3() {
    std::cout << "=== Test 3: Depth-3 leaf, empty history ===" << std::endl;

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 30;
    config.use_gumbel = false;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    Board board;  // Starting position
    auto focused_e2e4 = make_focused_policy(board, "e2e4");
    search.init_search(board, focused_e2e4, 0.0f, {});

    std::vector<float> obs(OBS_SIZE, 0.0f);
    std::vector<float> mask(POL_SIZE, 0.0f);

    // Depth 1: e2e4
    int collected = search.collect_leaves(obs.data(), mask.data(), 1);
    if (collected != 1) { std::cout << "  FAIL: d1 collect" << std::endl; return false; }

    // Build focused policy for e7e5 from the after-e4 position
    Board after_e4 = board;
    after_e4.makeMove(uci::uciToMove(board, "e2e4"));
    auto focused_e7e5 = make_focused_policy(after_e4, "e7e5");

    search.update_leaves({focused_e7e5}, {0.0f}, nullptr);

    // Depth 2: e7e5
    std::fill(obs.begin(), obs.end(), 0.0f);
    collected = search.collect_leaves(obs.data(), mask.data(), 1);
    if (collected != 1) { std::cout << "  FAIL: d2 collect" << std::endl; return false; }

    search.update_leaves({uniform_policy()}, {0.0f}, nullptr);

    // Depth 3: some move after e4 e5
    std::fill(obs.begin(), obs.end(), 0.0f);
    collected = search.collect_leaves(obs.data(), mask.data(), 1);
    if (collected != 1) { std::cout << "  FAIL: d3 collect" << std::endl; return false; }

    // 3 non-zero history slots, 5 zero
    int nonzero = count_nonzero_history_slots(obs.data());
    if (nonzero != 3) {
        std::cout << "  FAIL: expected 3 non-zero history slots, got " << nonzero << std::endl;
        return false;
    }

    std::cout << "  Non-zero history slots: " << nonzero << " (expected 3)" << std::endl;
    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ============================================================================
// Test 4 — Pre-root history + depth-1 leaf = combined history
// ============================================================================
bool test_selfplay_depth1_combined() {
    std::cout << "=== Test 4: Pre-root + depth-1, combined history ===" << std::endl;

    // Board after 1.e4 e5 (white to move)
    Board starting;
    Board after_e4 = starting;
    after_e4.makeMove(uci::uciToMove(starting, "e2e4"));
    Board after_e4_e5 = after_e4;
    after_e4_e5.makeMove(uci::uciToMove(after_e4, "e7e5"));

    // Pre-root history: [starting, after_e4]
    std::vector<Board> position_history = {starting, after_e4};

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 10;
    config.use_gumbel = false;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    auto focused_nf3 = make_focused_policy(after_e4_e5, "g1f3");
    search.init_search(after_e4_e5, focused_nf3, 0.0f, position_history);

    std::vector<float> obs(OBS_SIZE, 0.0f);
    std::vector<float> mask(POL_SIZE, 0.0f);
    int collected = search.collect_leaves(obs.data(), mask.data(), 1);

    if (collected != 1) {
        std::cout << "  FAIL: expected 1 leaf, got " << collected << std::endl;
        return false;
    }

    // 3 non-zero slots: 2 pre-root + 1 in-tree (root = after_e4_e5)
    int nonzero = count_nonzero_history_slots(obs.data());
    if (nonzero != 3) {
        std::cout << "  FAIL: expected 3 non-zero history slots, got " << nonzero << std::endl;
        return false;
    }

    // Exact match: leaf = after Nf3, history = [starting, after_e4, after_e4_e5]
    Board after_nf3 = after_e4_e5;
    after_nf3.makeMove(uci::uciToMove(after_e4_e5, "g1f3"));
    std::vector<Board> ref_history = {starting, after_e4, after_e4_e5};
    std::vector<float> ref(OBS_SIZE, 0.0f);
    encoding::PositionEncoder::encode_to_buffer(after_nf3, ref.data(), ref_history);

    if (!compare_obs(obs.data(), ref.data(), OBS_SIZE, "selfplay-d1")) {
        std::cout << "  FAIL: obs doesn't match reference encoding" << std::endl;
        return false;
    }

    std::cout << "  Non-zero history slots: " << nonzero << " (expected 3)" << std::endl;
    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ============================================================================
// Test 5 — Pre-root + depth-2 = 4 history boards
// ============================================================================
bool test_selfplay_depth2_combined() {
    std::cout << "=== Test 5: Pre-root + depth-2, combined history ===" << std::endl;

    Board starting;
    Board after_e4 = starting;
    after_e4.makeMove(uci::uciToMove(starting, "e2e4"));
    Board after_e4_e5 = after_e4;
    after_e4_e5.makeMove(uci::uciToMove(after_e4, "e7e5"));

    std::vector<Board> position_history = {starting, after_e4};

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 20;
    config.use_gumbel = false;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    auto focused_nf3 = make_focused_policy(after_e4_e5, "g1f3");
    search.init_search(after_e4_e5, focused_nf3, 0.0f, position_history);

    std::vector<float> obs(OBS_SIZE, 0.0f);
    std::vector<float> mask(POL_SIZE, 0.0f);

    // Depth 1: Nf3
    int collected = search.collect_leaves(obs.data(), mask.data(), 1);
    if (collected != 1) { std::cout << "  FAIL: d1 collect" << std::endl; return false; }

    search.update_leaves({uniform_policy()}, {0.0f}, nullptr);

    // Depth 2: some response to Nf3
    std::fill(obs.begin(), obs.end(), 0.0f);
    collected = search.collect_leaves(obs.data(), mask.data(), 1);
    if (collected != 1) { std::cout << "  FAIL: d2 collect" << std::endl; return false; }

    // 4 non-zero slots: 2 pre-root + 2 in-tree
    int nonzero = count_nonzero_history_slots(obs.data());
    if (nonzero != 4) {
        std::cout << "  FAIL: expected 4 non-zero history slots, got " << nonzero << std::endl;
        return false;
    }

    std::cout << "  Non-zero history slots: " << nonzero << " (expected 4)" << std::endl;
    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ============================================================================
// Test 6 — Windowing caps at 8 history slots
// ============================================================================
bool test_windowing_caps_at_8() {
    std::cout << "=== Test 6: Windowing caps at 8 ===" << std::endl;

    // Play 7-move Ruy Lopez opening to get 7 pre-root boards
    Board b;
    std::vector<Board> game_boards;
    game_boards.push_back(b);  // starting pos

    const char* moves[] = {"e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4"};
    for (const char* m : moves) {
        b.makeMove(uci::uciToMove(b, m));
        game_boards.push_back(b);
    }

    // Root = after Ba4 (position 7), pre-root history = first 7 boards
    Board root_board = game_boards.back();
    std::vector<Board> position_history(game_boards.begin(), game_boards.end() - 1);
    // position_history has 7 entries

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 30;
    config.use_gumbel = false;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    // Use focused policies to force a specific 3-move line
    auto focused_nf6 = make_focused_policy(root_board, "g8f6");
    search.init_search(root_board, focused_nf6, 0.0f, position_history);

    std::vector<float> obs(OBS_SIZE, 0.0f);
    std::vector<float> mask(POL_SIZE, 0.0f);

    // Depth 1
    int collected = search.collect_leaves(obs.data(), mask.data(), 1);
    if (collected != 1) { std::cout << "  FAIL: d1 collect" << std::endl; return false; }
    search.update_leaves({uniform_policy()}, {0.0f}, nullptr);

    // Depth 2
    std::fill(obs.begin(), obs.end(), 0.0f);
    collected = search.collect_leaves(obs.data(), mask.data(), 1);
    if (collected != 1) { std::cout << "  FAIL: d2 collect" << std::endl; return false; }
    search.update_leaves({uniform_policy()}, {0.0f}, nullptr);

    // Depth 3: total = 7 pre-root + 3 in-tree = 10, but capped at 8
    std::fill(obs.begin(), obs.end(), 0.0f);
    collected = search.collect_leaves(obs.data(), mask.data(), 1);
    if (collected != 1) { std::cout << "  FAIL: d3 collect" << std::endl; return false; }

    int nonzero = count_nonzero_history_slots(obs.data());
    if (nonzero != 8) {
        std::cout << "  FAIL: expected all 8 history slots non-zero, got " << nonzero << std::endl;
        return false;
    }

    std::cout << "  Non-zero history slots: " << nonzero << " (expected 8, capped)" << std::endl;
    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ============================================================================
// Test 7 — Async pipeline produces same encoding as sync
// ============================================================================
bool test_async_matches_sync() {
    std::cout << "=== Test 7: Async matches sync encoding ===" << std::endl;

    Board starting;
    Board some_prior = starting;
    some_prior.makeMove(uci::uciToMove(starting, "d2d4"));
    // Undo to get back to starting, but use the prior as history
    std::vector<Board> position_history = {some_prior};

    Board board;  // Starting position
    auto focused_d2d4 = make_focused_policy(board, "d2d4");

    // --- Sync path ---
    NodePool pool_sync;
    BatchSearchConfig config;
    config.num_simulations = 10;
    config.use_gumbel = false;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search_sync(pool_sync, config);

    search_sync.init_search(board, focused_d2d4, 0.0f, position_history);

    std::vector<float> obs_sync(OBS_SIZE, 0.0f);
    std::vector<float> mask_sync(POL_SIZE, 0.0f);
    int collected_sync = search_sync.collect_leaves(obs_sync.data(), mask_sync.data(), 1);

    // --- Async path ---
    NodePool pool_async;
    MCTSSearch search_async(pool_async, config);

    search_async.init_search(board, focused_d2d4, 0.0f, position_history);

    std::vector<float> obs_async(OBS_SIZE, 0.0f);
    std::vector<float> mask_async(POL_SIZE, 0.0f);
    search_async.start_next_batch_collection();
    int collected_async = search_async.collect_leaves_async(obs_async.data(), mask_async.data(), 1);

    if (collected_sync != collected_async) {
        std::cout << "  FAIL: sync collected " << collected_sync
                  << " vs async collected " << collected_async << std::endl;
        search_async.cancel_collection_pending();
        return false;
    }

    if (collected_sync == 0) {
        std::cout << "  FAIL: no leaves collected" << std::endl;
        return false;
    }

    // Compare observation buffers
    if (!compare_obs(obs_async.data(), obs_sync.data(), OBS_SIZE, "sync-vs-async")) {
        std::cout << "  FAIL: async obs differs from sync obs" << std::endl;
        search_async.cancel_collection_pending();
        return false;
    }

    search_async.cancel_collection_pending();
    std::cout << "  Sync and async observations are byte-identical" << std::endl;
    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ============================================================================
// Test 8 — Multiple leaves in one batch each get correct history
// ============================================================================
bool test_batch_multiple_leaves() {
    std::cout << "=== Test 8: Multiple leaves in one batch ===" << std::endl;

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 50;
    config.use_gumbel = false;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    Board board;  // Starting position
    search.init_search(board, uniform_policy(), 0.0f, {});

    int batch_size = 4;
    std::vector<float> obs(batch_size * OBS_SIZE, 0.0f);
    std::vector<float> mask(batch_size * POL_SIZE, 0.0f);
    int collected = search.collect_leaves(obs.data(), mask.data(), batch_size);

    std::cout << "  Collected " << collected << " leaves in batch" << std::endl;

    if (collected < 1) {
        std::cout << "  FAIL: no leaves collected" << std::endl;
        return false;
    }

    // Each depth-1 leaf should have T-1 non-zero (root position in history)
    // and T-2..T-8 zero
    for (int i = 0; i < collected; ++i) {
        const float* leaf_obs = obs.data() + i * OBS_SIZE;
        if (!history_planes_nonzero(leaf_obs, 0)) {
            std::cout << "  FAIL: leaf " << i << " has zero T-1 history" << std::endl;
            return false;
        }

        int nonzero = count_nonzero_history_slots(leaf_obs);
        if (nonzero != 1) {
            std::cout << "  FAIL: leaf " << i << " has " << nonzero
                      << " non-zero slots (expected 1 for depth-1)" << std::endl;
            return false;
        }
    }

    // All T-1 planes should be identical across leaves (same root as history)
    if (collected >= 2) {
        const float* leaf0 = obs.data();
        for (int i = 1; i < collected; ++i) {
            const float* leaf_i = obs.data() + i * OBS_SIZE;
            // Compare only history channels 19-31 (T-1 slot)
            bool t1_match = true;
            for (int idx = 0; idx < OBS_SIZE; ++idx) {
                int ch = idx % CHANNELS;
                if (ch >= 19 && ch <= 31) {
                    if (leaf0[idx] != leaf_i[idx]) {
                        t1_match = false;
                        break;
                    }
                }
            }
            if (!t1_match) {
                std::cout << "  FAIL: leaf " << i << " T-1 history differs from leaf 0" << std::endl;
                return false;
            }
        }
        std::cout << "  All " << collected << " leaves have identical T-1 history planes" << std::endl;
    }

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ============================================================================
// Test 9 — Byte-for-byte match on all 123 channels (known position)
// ============================================================================
bool test_exact_encoding_known_position() {
    std::cout << "=== Test 9: Exact encoding match, all 123 channels ===" << std::endl;

    // Board after 1.e4 (black to move in FEN, but we set up the board manually)
    Board starting;
    Board board_e4 = starting;
    board_e4.makeMove(uci::uciToMove(starting, "e2e4"));

    std::vector<Board> position_history = {starting};

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 10;
    config.use_gumbel = false;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    auto focused_e7e5 = make_focused_policy(board_e4, "e7e5");
    search.init_search(board_e4, focused_e7e5, 0.0f, position_history);

    std::vector<float> obs(OBS_SIZE, 0.0f);
    std::vector<float> mask(POL_SIZE, 0.0f);
    int collected = search.collect_leaves(obs.data(), mask.data(), 1);

    if (collected != 1) {
        std::cout << "  FAIL: expected 1 leaf, got " << collected << std::endl;
        return false;
    }

    // Build reference: leaf = after e4 e5 (white to move), history = [starting, board_e4]
    Board board_e4_e5 = board_e4;
    board_e4_e5.makeMove(uci::uciToMove(board_e4, "e7e5"));
    std::vector<Board> ref_history = {starting, board_e4};
    std::vector<float> ref(OBS_SIZE, 0.0f);
    encoding::PositionEncoder::encode_to_buffer(board_e4_e5, ref.data(), ref_history);

    if (!compare_obs(obs.data(), ref.data(), OBS_SIZE, "exact-all-123")) {
        std::cout << "  FAIL: full 123-channel mismatch" << std::endl;
        return false;
    }

    std::cout << "  All 123 channels match reference encoding" << std::endl;
    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ============================================================================
// Test 10 — Mid-game FEN with modified castling/pieces
// ============================================================================
bool test_midgame_position() {
    std::cout << "=== Test 10: Mid-game FEN position ===" << std::endl;

    // After 1.e4 Nf6 2.e5 Nc6 — non-standard opening with modified piece positions
    Board board("r1bqkb1r/pppppppp/2n2n2/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3");

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 10;
    config.use_gumbel = false;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    auto focused_d2d4 = make_focused_policy(board, "d2d4");
    search.init_search(board, focused_d2d4, 0.0f, {});

    std::vector<float> obs(OBS_SIZE, 0.0f);
    std::vector<float> mask(POL_SIZE, 0.0f);
    int collected = search.collect_leaves(obs.data(), mask.data(), 1);

    if (collected != 1) {
        std::cout << "  FAIL: expected 1 leaf, got " << collected << std::endl;
        return false;
    }

    // T-1 should contain mid-game root (not starting position)
    if (!history_planes_nonzero(obs.data(), 0)) {
        std::cout << "  FAIL: T-1 should be non-zero" << std::endl;
        return false;
    }

    // Build reference: leaf = after d4 from this position, history = [root_board]
    Board after_d4 = board;
    after_d4.makeMove(uci::uciToMove(board, "d2d4"));
    std::vector<Board> ref_history = {board};
    std::vector<float> ref(OBS_SIZE, 0.0f);
    encoding::PositionEncoder::encode_to_buffer(after_d4, ref.data(), ref_history);

    if (!compare_obs(obs.data(), ref.data(), OBS_SIZE, "midgame-exact")) {
        std::cout << "  FAIL: mid-game obs doesn't match reference" << std::endl;
        return false;
    }

    std::cout << "  Mid-game position correctly encoded with history" << std::endl;
    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ============================================================================
// Test 11 — Black to move at root (perspective flipping)
// ============================================================================
bool test_black_to_move_perspective() {
    std::cout << "=== Test 11: Black to move, perspective flipping ===" << std::endl;

    // After 1.e4, black to move
    Board starting;
    Board board_e4 = starting;
    board_e4.makeMove(uci::uciToMove(starting, "e2e4"));

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 10;
    config.use_gumbel = false;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    auto focused_e7e5 = make_focused_policy(board_e4, "e7e5");
    search.init_search(board_e4, focused_e7e5, 0.0f, {});

    std::vector<float> obs(OBS_SIZE, 0.0f);
    std::vector<float> mask(POL_SIZE, 0.0f);
    int collected = search.collect_leaves(obs.data(), mask.data(), 1);

    if (collected != 1) {
        std::cout << "  FAIL: expected 1 leaf, got " << collected << std::endl;
        return false;
    }

    // Leaf = after e4 e5 (white to move), T-1 = after_e4 (black-to-move root)
    Board board_e4_e5 = board_e4;
    board_e4_e5.makeMove(uci::uciToMove(board_e4, "e7e5"));
    std::vector<Board> ref_history = {board_e4};
    std::vector<float> ref(OBS_SIZE, 0.0f);
    encoding::PositionEncoder::encode_to_buffer(board_e4_e5, ref.data(), ref_history);

    if (!compare_obs(obs.data(), ref.data(), OBS_SIZE, "black-perspective")) {
        std::cout << "  FAIL: black-to-move perspective encoding mismatch" << std::endl;
        return false;
    }

    std::cout << "  Black-to-move root correctly encoded with perspective flip" << std::endl;
    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ============================================================================
// Test 12 — Full async pipeline at depth 2 with pre-root history
// ============================================================================
bool test_async_pipeline_multidepth() {
    std::cout << "=== Test 12: Async pipeline multi-depth with pre-root history ===" << std::endl;

    // Board after 1.d4 d5
    Board starting;
    Board after_d4 = starting;
    after_d4.makeMove(uci::uciToMove(starting, "d2d4"));
    Board after_d4_d5 = after_d4;
    after_d4_d5.makeMove(uci::uciToMove(after_d4, "d7d5"));

    std::vector<Board> position_history = {starting, after_d4};

    NodePool pool;
    BatchSearchConfig config;
    config.num_simulations = 20;
    config.use_gumbel = false;
    config.dirichlet_epsilon = 0.0f;
    MCTSSearch search(pool, config);

    auto focused_c2c4 = make_focused_policy(after_d4_d5, "c2c4");
    search.init_search(after_d4_d5, focused_c2c4, 0.0f, position_history);

    // Step 1: Collect depth-1 leaf via async
    std::vector<float> obs1(OBS_SIZE, 0.0f);
    std::vector<float> mask(POL_SIZE, 0.0f);
    search.start_next_batch_collection();
    int collected1 = search.collect_leaves_async(obs1.data(), mask.data(), 1);

    if (collected1 != 1) {
        std::cout << "  FAIL: async d1 collect got " << collected1 << std::endl;
        search.cancel_collection_pending();
        return false;
    }

    // Assert obs1: 3 non-zero slots (2 pre-root + 1 in-tree root)
    int nonzero1 = count_nonzero_history_slots(obs1.data());
    if (nonzero1 != 3) {
        std::cout << "  FAIL: obs1 expected 3 non-zero slots, got " << nonzero1 << std::endl;
        search.cancel_collection_pending();
        return false;
    }

    // Swap to eval buffer, then update to expand c4 child
    search.commit_and_swap();

    // Build focused policy for black's response (e7e6) from the after-c4 position
    Board after_c4 = after_d4_d5;
    after_c4.makeMove(uci::uciToMove(after_d4_d5, "c2c4"));
    auto focused_e7e6 = make_focused_policy(after_c4, "e7e6");

    search.update_prev_leaves({focused_e7e6}, {0.0f}, nullptr);

    // Step 2: Collect depth-2 leaf via async
    std::vector<float> obs3(OBS_SIZE, 0.0f);
    search.start_next_batch_collection();
    int collected3 = search.collect_leaves_async(obs3.data(), mask.data(), 1);

    if (collected3 != 1) {
        std::cout << "  FAIL: async d2 collect got " << collected3 << std::endl;
        search.cancel_collection_pending();
        return false;
    }

    // Assert obs3: 4 non-zero slots (2 pre-root + 2 in-tree)
    int nonzero3 = count_nonzero_history_slots(obs3.data());
    if (nonzero3 != 4) {
        std::cout << "  FAIL: obs3 expected 4 non-zero slots, got " << nonzero3 << std::endl;
        search.cancel_collection_pending();
        return false;
    }

    // Verify obs1 exact match: leaf = after c4, history = [starting, after_d4, after_d4_d5]
    std::vector<Board> ref_history1 = {starting, after_d4, after_d4_d5};
    std::vector<float> ref1(OBS_SIZE, 0.0f);
    encoding::PositionEncoder::encode_to_buffer(after_c4, ref1.data(), ref_history1);

    if (!compare_obs(obs1.data(), ref1.data(), OBS_SIZE, "async-d1-exact")) {
        std::cout << "  FAIL: async depth-1 obs doesn't match reference" << std::endl;
        search.cancel_collection_pending();
        return false;
    }

    search.cancel_collection_pending();
    std::cout << "  obs1: " << nonzero1 << " non-zero slots (expected 3)" << std::endl;
    std::cout << "  obs3: " << nonzero3 << " non-zero slots (expected 4)" << std::endl;
    std::cout << "  Async depth-1 exact match verified" << std::endl;
    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "Position History Encoding Tests" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int failed = 0;

    auto run = [&](bool (*test)()) {
        try {
            if (test()) { passed++; } else { failed++; }
        } catch (const std::exception& e) {
            std::cout << "  EXCEPTION: " << e.what() << std::endl;
            failed++;
        }
    };

    run(test_reanalysis_depth1);
    run(test_reanalysis_depth2);
    run(test_reanalysis_depth3);
    run(test_selfplay_depth1_combined);
    run(test_selfplay_depth2_combined);
    run(test_windowing_caps_at_8);
    run(test_async_matches_sync);
    run(test_batch_multiple_leaves);
    run(test_exact_encoding_known_position);
    run(test_midgame_position);
    run(test_black_to_move_perspective);
    run(test_async_pipeline_multidepth);

    std::cout << "==========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "==========================================" << std::endl;

    return (failed > 0) ? 1 : 0;
}
