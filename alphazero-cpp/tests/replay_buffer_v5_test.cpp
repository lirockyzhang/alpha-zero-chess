// Comprehensive tests for ReplayBuffer v5 format (RPBF with FEN-based compression).
//
// Tests verify:
//   1. FEN storage and retrieval
//   2. Sparse policy encoding roundtrip
//   3. v5 save/load of all fields
//   4. Observation reconstruction from FENs (byte-exact match)
//   5. Reanalyzer-style history reconstruction
//   6. Metadata, composition, clear, truncation, circular overwrite

#include "training/replay_buffer.hpp"
#include "encoding/position_encoder.hpp"
#include "encoding/move_encoder.hpp"
#include "chess.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <array>
#include <cstring>
#include <filesystem>

using namespace training;
using namespace encoding;

static constexpr size_t OBS_SIZE   = PositionEncoder::TOTAL_SIZE;  // 7872
static constexpr size_t POL_SIZE   = 4672;
static constexpr size_t CHANNELS   = PositionEncoder::CHANNELS;    // 123
static constexpr size_t HIST_DEPTH = 8;

// ============================================================================
// Helper: GameState (builds positions via makeMove for full fidelity)
// ============================================================================

struct GameState {
    chess::Board board;
    std::vector<chess::Board> history;  // chronological [oldest..newest]

    GameState() = default;

    void make_move(const std::string& uci) {
        history.push_back(board);
        auto move = chess::uci::uciToMove(board, uci);
        board.makeMove(move);
    }
};

// ============================================================================
// Helper: Common positions built via makeMove
// ============================================================================

static GameState make_starting() {
    return GameState{};
}

static GameState make_after_e4() {
    GameState gs;
    gs.make_move("e2e4");
    return gs;
}

static GameState make_after_e4_e5_Nf3() {
    GameState gs;
    gs.make_move("e2e4");
    gs.make_move("e7e5");
    gs.make_move("g1f3");
    return gs;
}

static GameState make_ruy_lopez() {
    GameState gs;
    gs.make_move("e2e4");   // 1. e4
    gs.make_move("e7e5");   // 1... e5
    gs.make_move("g1f3");   // 2. Nf3
    gs.make_move("b8c6");   // 2... Nc6
    gs.make_move("f1b5");   // 3. Bb5
    gs.make_move("a7a6");   // 3... a6
    gs.make_move("b5a4");   // 4. Ba4
    gs.make_move("g8f6");   // 4... Nf6
    gs.make_move("e1g1");   // 5. O-O  (9 half-moves → 9 history boards, capped to 8)
    return gs;
}

// ============================================================================
// Helper: Float array comparison
// ============================================================================

static bool compare_float_arrays(const float* a, const float* b, size_t size,
                                 const char* label) {
    for (size_t i = 0; i < size; ++i) {
        if (a[i] != b[i]) {
            std::cout << "  MISMATCH [" << label << "] at index " << i
                      << ": a=" << a[i] << " b=" << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Observation-specific comparison with channel decomposition
static bool compare_obs(const float* actual, const float* expected,
                        const char* label) {
    for (size_t i = 0; i < OBS_SIZE; ++i) {
        if (actual[i] != expected[i]) {
            size_t square  = i / CHANNELS;
            size_t channel = i % CHANNELS;
            size_t rank    = square / 8;
            size_t file    = square % 8;
            std::cout << "  MISMATCH [" << label << "] at index " << i
                      << " (rank=" << rank << " file=" << file
                      << " ch=" << channel << "): actual=" << actual[i]
                      << " expected=" << expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// Helper: RAII temp file cleanup
// ============================================================================

struct ScopedFile {
    std::string path;
    ~ScopedFile() { std::filesystem::remove(path); }
};

static std::string get_temp_path(const std::string& name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

// ============================================================================
// Helper: Replicate self-play data flow (parallel_coordinator.cpp:687-716)
// ============================================================================

static std::vector<float> encode_and_store(
    ReplayBuffer& buf,
    const chess::Board& board,
    const std::vector<chess::Board>& history,  // chronological [oldest..newest]
    const std::vector<float>& policy,
    float value,
    const float wdl[3],
    const SampleMeta& meta)
{
    // 1. Encode observation
    std::vector<float> obs(OBS_SIZE);
    PositionEncoder::encode_to_buffer(board, obs.data(), history);

    // 2. Root FEN
    std::string fen = board.getFen();

    // 3. History FENs + hashes in most-recent-first order
    uint8_t num_hist = static_cast<uint8_t>(
        std::min(size_t(HIST_DEPTH), history.size()));
    std::array<uint64_t, 8> hashes{};
    std::array<std::string, 8> fen_strs;
    std::array<const char*, 8> fen_ptrs{};
    for (uint8_t h = 0; h < num_hist; ++h) {
        size_t idx = history.size() - 1 - h;
        hashes[h]   = history[idx].hash();
        fen_strs[h] = history[idx].getFen();
        fen_ptrs[h] = fen_strs[h].c_str();
    }

    // 4. add_sample
    buf.add_sample(obs, policy, value, wdl, nullptr, &meta,
                   fen.c_str(), hashes.data(), num_hist, fen_ptrs.data());

    return obs;  // caller keeps for comparison after load
}

// ============================================================================
// Group A: FEN Storage Basics
// ============================================================================

static bool test_basic_fen_storage() {
    std::cout << "=== Test 1: Basic FEN Storage ===" << std::endl;

    ReplayBuffer buf(100);
    buf.enable_fen_storage();

    auto gs = make_after_e4_e5_Nf3();  // 3 history boards
    std::vector<float> policy(POL_SIZE, 0.0f);
    policy[100] = 1.0f;
    float wdl[3] = {0.5f, 0.3f, 0.2f};
    SampleMeta meta{1, 0, 0, 5, 40};

    encode_and_store(buf, gs.board, gs.history, policy, 0.3f, wdl, meta);

    // Verify root FEN
    std::string root_fen = buf.get_fen(0);
    std::string expected_fen = gs.board.getFen();
    if (root_fen != expected_fen) {
        std::cout << "  FAIL: root FEN mismatch" << std::endl;
        std::cout << "    got:      " << root_fen << std::endl;
        std::cout << "    expected: " << expected_fen << std::endl;
        return false;
    }

    // Verify num_history
    uint8_t nh = buf.get_num_history(0);
    if (nh != 3) {
        std::cout << "  FAIL: num_history = " << (int)nh
                  << ", expected 3" << std::endl;
        return false;
    }

    // Verify history FENs (stored most-recent-first)
    for (int h = 0; h < 3; ++h) {
        std::string hfen = buf.get_history_fen(0, h);
        size_t idx = gs.history.size() - 1 - h;
        std::string expected_hfen = gs.history[idx].getFen();
        if (hfen != expected_hfen) {
            std::cout << "  FAIL: history_fen[" << h << "] mismatch" << std::endl;
            std::cout << "    got:      " << hfen << std::endl;
            std::cout << "    expected: " << expected_hfen << std::endl;
            return false;
        }
    }

    // Out-of-bounds history slots should be empty
    for (int h = 3; h < 8; ++h) {
        if (!buf.get_history_fen(0, h).empty()) {
            std::cout << "  FAIL: history_fen[" << h
                      << "] should be empty" << std::endl;
            return false;
        }
    }

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

static bool test_fen_storage_disabled() {
    std::cout << "=== Test 2: FEN Storage Disabled ===" << std::endl;

    ReplayBuffer buf(100);
    // NOTE: do NOT call enable_fen_storage()

    std::vector<float> obs(OBS_SIZE, 0.0f);
    obs[0] = 1.0f;
    obs[100] = 0.5f;
    std::vector<float> policy(POL_SIZE, 0.0f);
    policy[42] = 1.0f;
    float wdl[3] = {1.0f, 0.0f, 0.0f};
    SampleMeta meta{1, 0, 0, 0, 10};

    std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    buf.add_sample(obs, policy, 1.0f, wdl, nullptr, &meta,
                   fen.c_str(), nullptr, 0, nullptr);

    // FEN accessors should return empty/0
    if (!buf.get_fen(0).empty()) {
        std::cout << "  FAIL: get_fen should be empty when disabled" << std::endl;
        return false;
    }
    if (buf.get_num_history(0) != 0) {
        std::cout << "  FAIL: get_num_history should be 0" << std::endl;
        return false;
    }
    if (!buf.get_history_fen(0, 0).empty()) {
        std::cout << "  FAIL: get_history_fen should be empty" << std::endl;
        return false;
    }

    // Observation and policy should still be stored
    const float* stored_obs = buf.get_observation_ptr(0);
    if (!stored_obs || stored_obs[0] != 1.0f || stored_obs[100] != 0.5f) {
        std::cout << "  FAIL: observation data mismatch" << std::endl;
        return false;
    }
    const float* stored_pol = buf.get_policy_ptr(0);
    if (!stored_pol || stored_pol[42] != 1.0f) {
        std::cout << "  FAIL: policy data mismatch" << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ============================================================================
// Group B: Save/Load Mechanics
// ============================================================================

static bool test_sparse_policy_roundtrip() {
    std::cout << "=== Test 3: Sparse Policy Roundtrip ===" << std::endl;

    std::string path = get_temp_path("test_sparse_policy.rpbf");
    ScopedFile cleanup{path};

    ReplayBuffer buf(100);
    buf.enable_fen_storage();

    auto gs = make_starting();

    // Policy A: 1 non-zero entry
    std::vector<float> pol_a(POL_SIZE, 0.0f);
    pol_a[42] = 1.0f;

    // Policy B: 20 non-zero entries
    std::vector<float> pol_b(POL_SIZE, 0.0f);
    for (int i = 0; i < 20; ++i) pol_b[i * 200] = 0.05f;

    // Policy C: all-zero
    std::vector<float> pol_c(POL_SIZE, 0.0f);

    float wdl[3] = {0.0f, 1.0f, 0.0f};
    SampleMeta meta{1, 1, 1, 0, 10};

    encode_and_store(buf, gs.board, gs.history, pol_a, 0.0f, wdl, meta);
    encode_and_store(buf, gs.board, gs.history, pol_b, 0.0f, wdl, meta);
    encode_and_store(buf, gs.board, gs.history, pol_c, 0.0f, wdl, meta);

    if (!buf.save(path)) {
        std::cout << "  FAIL: save failed" << std::endl;
        return false;
    }

    ReplayBuffer loaded(100);
    loaded.enable_fen_storage();
    if (!loaded.load(path)) {
        std::cout << "  FAIL: load failed" << std::endl;
        return false;
    }

    const float* lp0 = loaded.get_policy_ptr(0);
    const float* lp1 = loaded.get_policy_ptr(1);
    const float* lp2 = loaded.get_policy_ptr(2);
    if (!lp0 || !lp1 || !lp2) {
        std::cout << "  FAIL: null policy pointer" << std::endl;
        return false;
    }

    if (!compare_float_arrays(lp0, pol_a.data(), POL_SIZE, "pol_1entry"))
        return false;
    if (!compare_float_arrays(lp1, pol_b.data(), POL_SIZE, "pol_20entry"))
        return false;
    if (!compare_float_arrays(lp2, pol_c.data(), POL_SIZE, "pol_zero"))
        return false;

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

static bool test_v5_save_load_all_fields() {
    std::cout << "=== Test 4: v5 Save/Load All Fields ===" << std::endl;

    std::string path = get_temp_path("test_all_fields.rpbf");
    ScopedFile cleanup{path};

    ReplayBuffer buf(100);
    buf.enable_fen_storage();

    auto gs = make_after_e4();  // 1 history board

    float values[5]    = {1.0f, -1.0f, 0.0f, 0.5f, -0.5f};
    float wdls[5][3]   = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {0.0f, 1.0f, 0.0f},
        {0.6f, 0.3f, 0.1f},
        {0.1f, 0.2f, 0.7f},
    };
    SampleMeta metas[5] = {
        {1, 0, 0, 0, 20},
        {1, 2, 0, 1, 20},
        {2, 1, 1, 5, 30},
        {3, 0, 0, 10, 40},
        {3, 1, 3, 15, 50},
    };

    std::vector<std::vector<float>> orig_pols(5);
    for (int i = 0; i < 5; ++i) {
        orig_pols[i].resize(POL_SIZE, 0.0f);
        orig_pols[i][i * 100] = 1.0f;
        encode_and_store(buf, gs.board, gs.history,
                         orig_pols[i], values[i], wdls[i], metas[i]);
    }

    if (!buf.save(path)) {
        std::cout << "  FAIL: save failed" << std::endl;
        return false;
    }

    ReplayBuffer loaded(100);
    loaded.enable_fen_storage();
    if (!loaded.load(path)) {
        std::cout << "  FAIL: load failed" << std::endl;
        return false;
    }

    if (loaded.size() != 5) {
        std::cout << "  FAIL: size = " << loaded.size() << std::endl;
        return false;
    }
    if (loaded.total_added() != 5) {
        std::cout << "  FAIL: total_added = " << loaded.total_added() << std::endl;
        return false;
    }

    for (int i = 0; i < 5; ++i) {
        const float* pol = loaded.get_policy_ptr(i);
        if (!pol || pol[i * 100] != 1.0f) {
            std::cout << "  FAIL: policy[" << i << "]" << std::endl;
            return false;
        }
        std::string fen = loaded.get_fen(i);
        if (fen != gs.board.getFen()) {
            std::cout << "  FAIL: FEN[" << i << "]" << std::endl;
            return false;
        }
    }

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

static bool test_save_load_empty_buffer() {
    std::cout << "=== Test 5: Save/Load Empty Buffer ===" << std::endl;

    std::string path = get_temp_path("test_empty.rpbf");
    ScopedFile cleanup{path};

    ReplayBuffer buf(100);
    buf.enable_fen_storage();

    if (!buf.save(path)) {
        std::cout << "  FAIL: save failed" << std::endl;
        return false;
    }

    ReplayBuffer loaded(100);
    loaded.enable_fen_storage();
    if (!loaded.load(path)) {
        std::cout << "  FAIL: load failed" << std::endl;
        return false;
    }

    if (loaded.size() != 0) {
        std::cout << "  FAIL: size = " << loaded.size() << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ============================================================================
// Group C: Observation Reconstruction (CRITICAL)
// ============================================================================

static bool test_obs_reconstruction_no_history() {
    std::cout << "=== Test 6: Obs Reconstruction (No History) ===" << std::endl;

    std::string path = get_temp_path("test_obs_no_hist.rpbf");
    ScopedFile cleanup{path};

    ReplayBuffer buf(100);
    buf.enable_fen_storage();

    auto gs = make_starting();
    std::vector<float> policy(POL_SIZE, 0.0f);
    policy[0] = 1.0f;
    float wdl[3] = {0.0f, 1.0f, 0.0f};
    SampleMeta meta{1, 1, 1, 0, 1};

    auto orig = encode_and_store(buf, gs.board, gs.history,
                                 policy, 0.0f, wdl, meta);

    if (!buf.save(path)) {
        std::cout << "  FAIL: save" << std::endl;
        return false;
    }

    ReplayBuffer loaded(100);
    loaded.enable_fen_storage();
    if (!loaded.load(path)) {
        std::cout << "  FAIL: load" << std::endl;
        return false;
    }

    const float* recon = loaded.get_observation_ptr(0);
    if (!recon) {
        std::cout << "  FAIL: null obs ptr" << std::endl;
        return false;
    }
    if (!compare_obs(recon, orig.data(), "no_history")) return false;

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

static bool test_obs_reconstruction_3_history() {
    std::cout << "=== Test 7: Obs Reconstruction (3 History) ===" << std::endl;

    std::string path = get_temp_path("test_obs_3hist.rpbf");
    ScopedFile cleanup{path};

    ReplayBuffer buf(100);
    buf.enable_fen_storage();

    auto gs = make_after_e4_e5_Nf3();
    std::vector<float> policy(POL_SIZE, 0.0f);
    policy[0] = 1.0f;
    float wdl[3] = {0.5f, 0.3f, 0.2f};
    SampleMeta meta{1, 0, 0, 3, 40};

    auto orig = encode_and_store(buf, gs.board, gs.history,
                                 policy, 0.3f, wdl, meta);

    if (!buf.save(path)) {
        std::cout << "  FAIL: save" << std::endl;
        return false;
    }

    ReplayBuffer loaded(100);
    loaded.enable_fen_storage();
    if (!loaded.load(path)) {
        std::cout << "  FAIL: load" << std::endl;
        return false;
    }

    const float* recon = loaded.get_observation_ptr(0);
    if (!recon) {
        std::cout << "  FAIL: null obs ptr" << std::endl;
        return false;
    }
    if (!compare_obs(recon, orig.data(), "3_history")) return false;

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

static bool test_obs_reconstruction_full_8_history() {
    std::cout << "=== Test 8: Obs Reconstruction (Full 8 History) ===" << std::endl;

    std::string path = get_temp_path("test_obs_8hist.rpbf");
    ScopedFile cleanup{path};

    ReplayBuffer buf(100);
    buf.enable_fen_storage();

    auto gs = make_ruy_lopez();  // 9 history boards, capped to 8
    std::vector<float> policy(POL_SIZE, 0.0f);
    policy[0] = 1.0f;
    float wdl[3] = {0.4f, 0.4f, 0.2f};
    SampleMeta meta{1, 1, 0, 9, 50};

    auto orig = encode_and_store(buf, gs.board, gs.history,
                                 policy, 0.0f, wdl, meta);

    if (!buf.save(path)) {
        std::cout << "  FAIL: save" << std::endl;
        return false;
    }

    ReplayBuffer loaded(100);
    loaded.enable_fen_storage();
    if (!loaded.load(path)) {
        std::cout << "  FAIL: load" << std::endl;
        return false;
    }

    const float* recon = loaded.get_observation_ptr(0);
    if (!recon) {
        std::cout << "  FAIL: null obs ptr" << std::endl;
        return false;
    }
    if (!compare_obs(recon, orig.data(), "full_8_history")) return false;

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

static bool test_obs_reconstruction_black_to_move() {
    std::cout << "=== Test 9: Obs Reconstruction (Black to Move) ===" << std::endl;

    std::string path = get_temp_path("test_obs_black.rpbf");
    ScopedFile cleanup{path};

    ReplayBuffer buf(100);
    buf.enable_fen_storage();

    auto gs = make_after_e4();  // black to move, 1 history
    std::vector<float> policy(POL_SIZE, 0.0f);
    policy[0] = 1.0f;
    float wdl[3] = {0.0f, 0.0f, 1.0f};
    SampleMeta meta{1, 2, 0, 1, 30};

    auto orig = encode_and_store(buf, gs.board, gs.history,
                                 policy, -1.0f, wdl, meta);

    if (!buf.save(path)) {
        std::cout << "  FAIL: save" << std::endl;
        return false;
    }

    ReplayBuffer loaded(100);
    loaded.enable_fen_storage();
    if (!loaded.load(path)) {
        std::cout << "  FAIL: load" << std::endl;
        return false;
    }

    const float* recon = loaded.get_observation_ptr(0);
    if (!recon) {
        std::cout << "  FAIL: null obs ptr" << std::endl;
        return false;
    }
    if (!compare_obs(recon, orig.data(), "black_to_move")) return false;

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

static bool test_obs_reconstruction_multiple_depths() {
    std::cout << "=== Test 10: Obs Reconstruction (Multiple Depths) ===" << std::endl;

    std::string path = get_temp_path("test_obs_multi.rpbf");
    ScopedFile cleanup{path};

    ReplayBuffer buf(100);
    buf.enable_fen_storage();

    std::vector<float> policy(POL_SIZE, 0.0f);
    policy[0] = 1.0f;
    float wdl[3] = {0.0f, 1.0f, 0.0f};
    SampleMeta meta{1, 1, 1, 0, 10};

    auto gs0 = make_starting();         // 0 history
    auto gs1 = make_after_e4();         // 1 history
    auto gs3 = make_after_e4_e5_Nf3();  // 3 history
    auto gs8 = make_ruy_lopez();        // 9 history (capped to 8)

    auto orig0 = encode_and_store(buf, gs0.board, gs0.history,
                                  policy, 0.0f, wdl, meta);
    auto orig1 = encode_and_store(buf, gs1.board, gs1.history,
                                  policy, 0.0f, wdl, meta);
    auto orig3 = encode_and_store(buf, gs3.board, gs3.history,
                                  policy, 0.0f, wdl, meta);
    auto orig8 = encode_and_store(buf, gs8.board, gs8.history,
                                  policy, 0.0f, wdl, meta);

    if (!buf.save(path)) {
        std::cout << "  FAIL: save" << std::endl;
        return false;
    }

    ReplayBuffer loaded(100);
    loaded.enable_fen_storage();
    if (!loaded.load(path)) {
        std::cout << "  FAIL: load" << std::endl;
        return false;
    }

    if (loaded.size() != 4) {
        std::cout << "  FAIL: size = " << loaded.size() << std::endl;
        return false;
    }

    const float* r0 = loaded.get_observation_ptr(0);
    const float* r1 = loaded.get_observation_ptr(1);
    const float* r2 = loaded.get_observation_ptr(2);
    const float* r3 = loaded.get_observation_ptr(3);
    if (!r0 || !r1 || !r2 || !r3) {
        std::cout << "  FAIL: null obs ptr" << std::endl;
        return false;
    }

    if (!compare_obs(r0, orig0.data(), "depth_0")) return false;
    if (!compare_obs(r1, orig1.data(), "depth_1")) return false;
    if (!compare_obs(r2, orig3.data(), "depth_3")) return false;
    if (!compare_obs(r3, orig8.data(), "depth_8+")) return false;

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

static bool test_obs_reconstruction_castling_and_clocks() {
    std::cout << "=== Test 11: Obs Reconstruction (Castling & Clocks) ===" << std::endl;

    std::string path = get_temp_path("test_obs_castle.rpbf");
    ScopedFile cleanup{path};

    ReplayBuffer buf(100);
    buf.enable_fen_storage();

    // Ruy Lopez after 5.O-O: white castled, halfmove=3, fullmove=5, black to move
    auto gs = make_ruy_lopez();
    std::vector<float> policy(POL_SIZE, 0.0f);
    policy[0] = 1.0f;
    float wdl[3] = {0.3f, 0.4f, 0.3f};
    SampleMeta meta{1, 1, 0, 9, 60};

    auto orig = encode_and_store(buf, gs.board, gs.history,
                                 policy, 0.0f, wdl, meta);

    // Spot-check specific channels in the original encoding
    // NHWC layout: index = rank*8*123 + file*123 + channel
    // Uniform planes → check first square (rank=0, file=0)
    float move_count = orig[13];  // ch 13: fullmove/100
    float halfmove   = orig[18];  // ch 18: halfmove/100
    float cur_k      = orig[14];  // ch 14: current player kingside
    float cur_q      = orig[15];  // ch 15: current player queenside
    float opp_k      = orig[16];  // ch 16: opponent kingside
    float opp_q      = orig[17];  // ch 17: opponent queenside

    // After O-O, black to move:
    //   current (black): both castling available → ch14=1, ch15=1
    //   opponent (white): king moved → ch16=0, ch17=0
    if (cur_k != 1.0f || cur_q != 1.0f) {
        std::cout << "  FAIL: black castling should be available: k="
                  << cur_k << " q=" << cur_q << std::endl;
        return false;
    }
    if (opp_k != 0.0f || opp_q != 0.0f) {
        std::cout << "  FAIL: white castling should be gone: k="
                  << opp_k << " q=" << opp_q << std::endl;
        return false;
    }

    // Save and reload
    if (!buf.save(path)) {
        std::cout << "  FAIL: save" << std::endl;
        return false;
    }

    ReplayBuffer loaded(100);
    loaded.enable_fen_storage();
    if (!loaded.load(path)) {
        std::cout << "  FAIL: load" << std::endl;
        return false;
    }

    const float* recon = loaded.get_observation_ptr(0);
    if (!recon) {
        std::cout << "  FAIL: null obs ptr" << std::endl;
        return false;
    }

    // Byte-exact match
    if (!compare_obs(recon, orig.data(), "castling_clocks")) return false;

    // Double-check specific channels survived roundtrip
    if (recon[13] != move_count || recon[18] != halfmove) {
        std::cout << "  FAIL: clock channels changed" << std::endl;
        return false;
    }
    if (recon[14] != cur_k || recon[15] != cur_q ||
        recon[16] != opp_k || recon[17] != opp_q) {
        std::cout << "  FAIL: castling channels changed" << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ============================================================================
// Group D: Reanalyzer-Style Reconstruction
// ============================================================================

static bool test_reanalyzer_reconstruction() {
    std::cout << "=== Test 12: Reanalyzer-Style Reconstruction ===" << std::endl;

    std::string path = get_temp_path("test_reanalyzer.rpbf");
    ScopedFile cleanup{path};

    ReplayBuffer buf(100);
    buf.enable_fen_storage();

    auto gs = make_after_e4_e5_Nf3();
    std::vector<float> policy(POL_SIZE, 0.0f);
    policy[0] = 1.0f;
    float wdl[3] = {0.5f, 0.3f, 0.2f};
    SampleMeta meta{1, 0, 0, 3, 40};

    auto orig = encode_and_store(buf, gs.board, gs.history,
                                 policy, 0.3f, wdl, meta);

    if (!buf.save(path)) {
        std::cout << "  FAIL: save" << std::endl;
        return false;
    }

    ReplayBuffer loaded(100);
    loaded.enable_fen_storage();
    if (!loaded.load(path)) {
        std::cout << "  FAIL: load" << std::endl;
        return false;
    }

    // Simulate reanalyzer code path (reanalyzer.cpp:113-124):
    // Read FENs, reverse to chronological, build Board vector, encode
    std::string root_fen = loaded.get_fen(0);
    uint8_t nh = loaded.get_num_history(0);

    std::vector<chess::Board> recon_history;
    for (int h = static_cast<int>(nh) - 1; h >= 0; --h) {
        std::string hfen = loaded.get_history_fen(0, h);
        if (!hfen.empty()) {
            recon_history.emplace_back(hfen);
        }
    }

    chess::Board recon_board(root_fen);
    std::vector<float> recon_obs(OBS_SIZE);
    PositionEncoder::encode_to_buffer(recon_board, recon_obs.data(),
                                      recon_history);

    if (!compare_obs(recon_obs.data(), orig.data(), "reanalyzer")) return false;

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

static bool test_history_fen_ordering_invariant() {
    std::cout << "=== Test 13: History FEN Ordering Invariant ===" << std::endl;

    ReplayBuffer buf(100);
    buf.enable_fen_storage();

    // 1.e4 e5 2.Nf3 → history = [start, after_e4, after_e5]
    auto gs = make_after_e4_e5_Nf3();
    std::vector<float> policy(POL_SIZE, 0.0f);
    policy[0] = 1.0f;
    float wdl[3] = {0.0f, 1.0f, 0.0f};
    SampleMeta meta{1, 1, 1, 3, 20};

    auto orig = encode_and_store(buf, gs.board, gs.history,
                                 policy, 0.0f, wdl, meta);

    // Stored order: [T-1, T-2, T-3] at indices 0,1,2
    std::string h0 = buf.get_history_fen(0, 0);  // T-1 = after e5
    std::string h1 = buf.get_history_fen(0, 1);  // T-2 = after e4
    std::string h2 = buf.get_history_fen(0, 2);  // T-3 = starting

    std::string exp0 = gs.history[2].getFen();  // after e5
    std::string exp1 = gs.history[1].getFen();  // after e4
    std::string exp2 = gs.history[0].getFen();  // starting

    if (h0 != exp0) {
        std::cout << "  FAIL: stored[0] != T-1 (after e5)" << std::endl;
        std::cout << "    got:      " << h0 << std::endl;
        std::cout << "    expected: " << exp0 << std::endl;
        return false;
    }
    if (h1 != exp1) {
        std::cout << "  FAIL: stored[1] != T-2 (after e4)" << std::endl;
        return false;
    }
    if (h2 != exp2) {
        std::cout << "  FAIL: stored[2] != T-3 (starting)" << std::endl;
        return false;
    }

    // Reverse to chronological: [T-3, T-2, T-1] → encode → should match
    std::vector<chess::Board> chronological;
    for (int h = 2; h >= 0; --h) {
        std::string hfen = buf.get_history_fen(0, h);
        chronological.emplace_back(hfen);
    }

    chess::Board root(buf.get_fen(0));
    std::vector<float> recon(OBS_SIZE);
    PositionEncoder::encode_to_buffer(root, recon.data(), chronological);

    if (!compare_obs(recon.data(), orig.data(), "ordering_invariant"))
        return false;

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ============================================================================
// Group E: Metadata and Edge Cases
// ============================================================================

static bool test_metadata_composition_roundtrip() {
    std::cout << "=== Test 14: Metadata Composition Roundtrip ===" << std::endl;

    std::string path = get_temp_path("test_composition.rpbf");
    ScopedFile cleanup{path};

    ReplayBuffer buf(100);
    buf.enable_fen_storage();

    auto gs = make_starting();
    std::vector<float> policy(POL_SIZE, 0.0f);
    policy[0] = 1.0f;

    // 2 white wins, 2 draws, 2 black wins
    float wdl_w[3] = {1.0f, 0.0f, 0.0f};
    float wdl_d[3] = {0.0f, 1.0f, 0.0f};
    float wdl_l[3] = {0.0f, 0.0f, 1.0f};
    SampleMeta meta_w{1, 0, 0, 0, 10};   // game_result=0 → white win
    SampleMeta meta_d{1, 1, 1, 0, 10};   // game_result=1 → draw
    SampleMeta meta_l{1, 2, 0, 0, 10};   // game_result=2 → black win

    encode_and_store(buf, gs.board, gs.history, policy,  1.0f, wdl_w, meta_w);
    encode_and_store(buf, gs.board, gs.history, policy,  1.0f, wdl_w, meta_w);
    encode_and_store(buf, gs.board, gs.history, policy,  0.0f, wdl_d, meta_d);
    encode_and_store(buf, gs.board, gs.history, policy,  0.0f, wdl_d, meta_d);
    encode_and_store(buf, gs.board, gs.history, policy, -1.0f, wdl_l, meta_l);
    encode_and_store(buf, gs.board, gs.history, policy, -1.0f, wdl_l, meta_l);

    if (!buf.save(path)) {
        std::cout << "  FAIL: save" << std::endl;
        return false;
    }

    ReplayBuffer loaded(100);
    loaded.enable_fen_storage();
    if (!loaded.load(path)) {
        std::cout << "  FAIL: load" << std::endl;
        return false;
    }

    auto comp = loaded.get_composition();
    if (comp.wins != 2 || comp.draws != 2 || comp.losses != 2) {
        std::cout << "  FAIL: composition = {" << comp.wins << ", "
                  << comp.draws << ", " << comp.losses
                  << "}, expected {2, 2, 2}" << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

static bool test_clear_resets_fens() {
    std::cout << "=== Test 15: Clear Resets FENs ===" << std::endl;

    ReplayBuffer buf(100);
    buf.enable_fen_storage();

    auto gs = make_after_e4();
    std::vector<float> policy(POL_SIZE, 0.0f);
    policy[0] = 1.0f;
    float wdl[3] = {0.0f, 1.0f, 0.0f};
    SampleMeta meta{1, 1, 1, 0, 10};

    for (int i = 0; i < 5; ++i) {
        encode_and_store(buf, gs.board, gs.history, policy, 0.0f, wdl, meta);
    }

    if (buf.size() != 5) {
        std::cout << "  FAIL: pre-clear size = " << buf.size() << std::endl;
        return false;
    }

    buf.clear();

    if (buf.size() != 0) {
        std::cout << "  FAIL: post-clear size = " << buf.size() << std::endl;
        return false;
    }

    // FEN accessors should return empty (index >= size() which is 0)
    if (!buf.get_fen(0).empty()) {
        std::cout << "  FAIL: get_fen not empty after clear" << std::endl;
        return false;
    }
    if (buf.get_num_history(0) != 0) {
        std::cout << "  FAIL: get_num_history not 0 after clear" << std::endl;
        return false;
    }

    auto comp = buf.get_composition();
    if (comp.wins != 0 || comp.draws != 0 || comp.losses != 0) {
        std::cout << "  FAIL: composition not zeroed" << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

static bool test_truncation_on_load() {
    std::cout << "=== Test 16: Truncation on Load ===" << std::endl;

    std::string path = get_temp_path("test_truncation.rpbf");
    ScopedFile cleanup{path};

    // Save 10 samples, each with a different first move
    ReplayBuffer big_buf(100);
    big_buf.enable_fen_storage();

    std::string moves[] = {"e2e4", "d2d4", "g1f3", "b1c3", "c2c4",
                           "f2f4", "b2b3", "g2g3", "e2e3", "d2d3"};
    std::vector<std::vector<float>> orig_obs(10);
    std::vector<std::vector<float>> orig_pol(10);
    std::vector<std::string> orig_fens(10);

    for (int i = 0; i < 10; ++i) {
        GameState gs;
        gs.make_move(moves[i]);
        orig_fens[i] = gs.board.getFen();

        orig_pol[i].resize(POL_SIZE, 0.0f);
        orig_pol[i][i * 10] = 1.0f;

        float wdl[3] = {0.0f, 1.0f, 0.0f};
        SampleMeta meta{1, 1, 1, 0, 10};
        orig_obs[i] = encode_and_store(big_buf, gs.board, gs.history,
                                       orig_pol[i],
                                       static_cast<float>(i) * 0.1f,
                                       wdl, meta);
    }

    if (!big_buf.save(path)) {
        std::cout << "  FAIL: save" << std::endl;
        return false;
    }

    // Load into capacity-5 buffer → truncates to first 5
    ReplayBuffer small_buf(5);
    small_buf.enable_fen_storage();
    if (!small_buf.load(path)) {
        std::cout << "  FAIL: load" << std::endl;
        return false;
    }

    if (small_buf.size() != 5) {
        std::cout << "  FAIL: size = " << small_buf.size() << std::endl;
        return false;
    }

    for (int i = 0; i < 5; ++i) {
        const float* ro = small_buf.get_observation_ptr(i);
        const float* rp = small_buf.get_policy_ptr(i);
        if (!ro || !rp) {
            std::cout << "  FAIL: null ptr at " << i << std::endl;
            return false;
        }

        std::string lbl_o = "trunc_obs_" + std::to_string(i);
        std::string lbl_p = "trunc_pol_" + std::to_string(i);
        if (!compare_obs(ro, orig_obs[i].data(), lbl_o.c_str())) return false;
        if (!compare_float_arrays(rp, orig_pol[i].data(), POL_SIZE,
                                  lbl_p.c_str()))
            return false;

        std::string fen = small_buf.get_fen(i);
        if (fen != orig_fens[i]) {
            std::cout << "  FAIL: FEN[" << i << "] after truncation" << std::endl;
            return false;
        }
    }

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

static bool test_circular_overwrite() {
    std::cout << "=== Test 17: Circular Overwrite ===" << std::endl;

    ReplayBuffer buf(3);  // capacity = 3
    buf.enable_fen_storage();

    // Add 5 samples → wraps around twice
    // write_pos: 0, 1, 2, 0(3%3), 1(4%3)
    // Final: pos0=sample3, pos1=sample4, pos2=sample2
    std::string first_moves[] = {"e2e4", "d2d4", "g1f3", "c2c4", "f2f4"};
    std::vector<std::string> expected_fens(5);

    for (int i = 0; i < 5; ++i) {
        GameState gs;
        gs.make_move(first_moves[i]);
        expected_fens[i] = gs.board.getFen();

        std::vector<float> policy(POL_SIZE, 0.0f);
        policy[i] = 1.0f;
        float wdl[3] = {0.0f, 1.0f, 0.0f};
        SampleMeta meta{1, 1, 1, static_cast<uint16_t>(i), 10};
        encode_and_store(buf, gs.board, gs.history, policy, 0.0f, wdl, meta);
    }

    if (buf.size() != 3) {
        std::cout << "  FAIL: size = " << buf.size() << std::endl;
        return false;
    }

    // pos 0 → sample 3 (c2c4)
    std::string fen0 = buf.get_fen(0);
    if (fen0 != expected_fens[3]) {
        std::cout << "  FAIL: pos 0 should be sample 3 (c4)" << std::endl;
        std::cout << "    got:      " << fen0 << std::endl;
        std::cout << "    expected: " << expected_fens[3] << std::endl;
        return false;
    }

    // pos 1 → sample 4 (f2f4)
    std::string fen1 = buf.get_fen(1);
    if (fen1 != expected_fens[4]) {
        std::cout << "  FAIL: pos 1 should be sample 4 (f4)" << std::endl;
        std::cout << "    got:      " << fen1 << std::endl;
        std::cout << "    expected: " << expected_fens[4] << std::endl;
        return false;
    }

    // pos 2 → sample 2 (g1f3)
    std::string fen2 = buf.get_fen(2);
    if (fen2 != expected_fens[2]) {
        std::cout << "  FAIL: pos 2 should be sample 2 (Nf3)" << std::endl;
        std::cout << "    got:      " << fen2 << std::endl;
        std::cout << "    expected: " << expected_fens[2] << std::endl;
        return false;
    }

    std::cout << "  PASS" << std::endl << std::endl;
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "Replay Buffer v5 Comprehensive Tests" << std::endl;
    std::cout << "==========================================" << std::endl;

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

    // Group A: FEN Storage Basics
    run(test_basic_fen_storage);
    run(test_fen_storage_disabled);

    // Group B: Save/Load Mechanics
    run(test_sparse_policy_roundtrip);
    run(test_v5_save_load_all_fields);
    run(test_save_load_empty_buffer);

    // Group C: Observation Reconstruction (CRITICAL)
    run(test_obs_reconstruction_no_history);
    run(test_obs_reconstruction_3_history);
    run(test_obs_reconstruction_full_8_history);
    run(test_obs_reconstruction_black_to_move);
    run(test_obs_reconstruction_multiple_depths);
    run(test_obs_reconstruction_castling_and_clocks);

    // Group D: Reanalyzer-Style Reconstruction
    run(test_reanalyzer_reconstruction);
    run(test_history_fen_ordering_invariant);

    // Group E: Metadata and Edge Cases
    run(test_metadata_composition_roundtrip);
    run(test_clear_resets_fens);
    run(test_truncation_on_load);
    run(test_circular_overwrite);

    std::cout << "==========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed"
              << std::endl;
    std::cout << "==========================================" << std::endl;

    return (failed > 0) ? 1 : 0;
}
