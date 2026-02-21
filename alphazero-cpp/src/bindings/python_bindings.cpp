#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <cstdio>
#include "mcts/search.hpp"
#include "mcts/node_pool.hpp"
#include "mcts/batch_coordinator.hpp"
#include "encoding/position_encoder.hpp"
#include "encoding/move_encoder.hpp"
#include "selfplay/game.hpp"
#include "selfplay/coordinator.hpp"
#include "selfplay/parallel_coordinator.hpp"
#include "selfplay/evaluation_queue.hpp"
#include "selfplay/reanalyzer.hpp"
#include "training/replay_buffer.hpp"
#include "training/trainer.hpp"
#include "../third_party/chess-library/include/chess.hpp"

namespace py = pybind11;

// NOTE: Old PyMCTSSearch removed - use PyBatchedMCTSSearch instead
// The old synchronous API is no longer supported. Use the batched API for proper AlphaZero.

// Python wrapper for batch coordinator
class PyBatchCoordinator {
public:
    PyBatchCoordinator(int batch_size = 256, float batch_threshold = 0.9f)
        : coordinator_(create_config(batch_size, batch_threshold))
    {}

    void add_game(int game_id, const std::string& fen) {
        chess::Board board(fen);
        coordinator_.add_game(game_id, board);
    }

    bool is_game_complete(int game_id) const {
        return coordinator_.is_game_complete(game_id);
    }

    void remove_game(int game_id) {
        coordinator_.remove_game(game_id);
    }

    py::dict get_stats() const {
        auto stats = coordinator_.get_stats();
        py::dict result;
        result["active_games"] = stats.active_games;
        result["pending_evals"] = stats.pending_evals;
        result["batch_counter"] = stats.batch_counter;
        result["next_is_hard_sync"] = stats.next_is_hard_sync;
        return result;
    }

private:
    static mcts::BatchCoordinator::Config create_config(int batch_size, float threshold) {
        mcts::BatchCoordinator::Config config;
        config.batch_size = batch_size;
        config.batch_threshold = threshold;
        config.simulations_per_move = 800;
        config.hard_sync_interval = 10;
        return config;
    }

    mcts::BatchCoordinator coordinator_;
};

// Position encoding for neural network input
// NHWC layout: (height, width, channels) = (8, 8, 123)
py::array_t<float> encode_position(const std::string& fen,
                                    const std::vector<std::string>& history_fens = {}) {
    chess::Board board(fen);

    // Convert history FENs to Board objects
    std::vector<chess::Board> position_history;
    position_history.reserve(history_fens.size());
    for (const auto& h_fen : history_fens) {
        position_history.emplace_back(h_fen);
    }

    // Use the real position encoder with history
    std::vector<float> encoding = encoding::PositionEncoder::encode(board, position_history);

    // Return as numpy array with NHWC shape (8, 8, 123) - channels last
    return py::array_t<float>(
        {encoding::PositionEncoder::HEIGHT,
         encoding::PositionEncoder::WIDTH,
         encoding::PositionEncoder::CHANNELS},
        encoding.data()
    );
}

// Zero-copy position encoding - writes directly to provided buffer
// NHWC layout: (height, width, channels) = (8, 8, 123)
void encode_position_to_buffer(const std::string& fen, py::array_t<float> buffer) {
    chess::Board board(fen);

    // Verify buffer shape and size for NHWC layout
    auto buf_info = buffer.request();
    if (buf_info.ndim != 3 ||
        buf_info.shape[0] != encoding::PositionEncoder::HEIGHT ||
        buf_info.shape[1] != encoding::PositionEncoder::WIDTH ||
        buf_info.shape[2] != encoding::PositionEncoder::CHANNELS) {
        throw std::runtime_error("Buffer must have NHWC shape (8, 8, 123)");
    }

    // Write directly to buffer (zero-copy)
    float* ptr = static_cast<float*>(buf_info.ptr);
    encoding::PositionEncoder::encode_to_buffer(board, ptr);
}

// Batch encoding with optional OpenMP parallelization
// Encodes multiple positions in parallel to reduce Python call overhead
int encode_batch(const std::vector<std::string>& fens, py::array_t<float> buffer, bool use_parallel = true) {
    // Verify buffer shape and size for NHWC layout
    auto buf_info = buffer.request();
    int batch_size = static_cast<int>(fens.size());

    if (buf_info.ndim != 4 ||
        buf_info.shape[0] != batch_size ||
        buf_info.shape[1] != encoding::PositionEncoder::HEIGHT ||
        buf_info.shape[2] != encoding::PositionEncoder::WIDTH ||
        buf_info.shape[3] != encoding::PositionEncoder::CHANNELS) {
        throw std::runtime_error("Buffer must have NHWC shape (batch_size, 8, 8, 123)");
    }

    // Encode batch
    float* ptr = static_cast<float*>(buf_info.ptr);
    return encoding::PositionEncoder::encode_batch(fens, ptr, use_parallel);
}

// Move encoding/decoding utilities
int move_to_index(const std::string& uci_move, const std::string& fen) {
    chess::Board board(fen);
    return encoding::MoveEncoder::move_to_index(uci_move, board);
}

std::string index_to_move(int index, const std::string& fen) {
    chess::Board board(fen);
    return encoding::MoveEncoder::index_to_uci(index, board);
}

// Python wrapper for batched MCTS search
// This implements proper AlphaZero where every leaf gets NN evaluation
class PyBatchedMCTSSearch {
public:
    PyBatchedMCTSSearch(int num_simulations = 800, int batch_size = 256, float c_puct = 1.5f, float risk_beta = 0.0f)
        : pool_()
        , search_(pool_, create_config(num_simulations, batch_size, c_puct, risk_beta))
        , batch_size_(batch_size)
    {}

    // Initialize search with root position and initial NN evaluation
    void init_search(const std::string& fen,
                     py::array_t<float> root_policy,
                     float root_value,
                     const std::vector<std::string>& history_fens = {}) {
        chess::Board board(fen);
        root_fen_ = fen;

        // Convert numpy array to vector
        auto policy_buf = root_policy.request();
        if (policy_buf.ndim != 1 || policy_buf.shape[0] != encoding::MoveEncoder::POLICY_SIZE) {
            throw std::runtime_error("Policy must be 1D array of size " + std::to_string(encoding::MoveEncoder::POLICY_SIZE));
        }
        std::vector<float> policy_vec(static_cast<float*>(policy_buf.ptr),
                                      static_cast<float*>(policy_buf.ptr) + encoding::MoveEncoder::POLICY_SIZE);

        // Convert history FENs to Board objects
        std::vector<chess::Board> position_history;
        position_history.reserve(history_fens.size());
        for (const auto& h_fen : history_fens) {
            position_history.emplace_back(h_fen);
        }

        search_.init_search(board, policy_vec, root_value, position_history);
    }

    // Collect leaves that need NN evaluation
    // Returns (num_leaves, observations, legal_masks)
    py::tuple collect_leaves() {
        // Allocate buffers for observations and masks
        std::vector<float> obs_buffer(batch_size_ * encoding::PositionEncoder::TOTAL_SIZE, 0.0f);
        std::vector<float> mask_buffer(batch_size_ * encoding::MoveEncoder::POLICY_SIZE, 0.0f);

        int num_leaves = search_.collect_leaves(obs_buffer.data(), mask_buffer.data(), batch_size_);

        if (num_leaves == 0) {
            // Return empty arrays
            return py::make_tuple(
                0,
                py::array_t<float>(std::vector<py::ssize_t>{0, 8, 8, encoding::PositionEncoder::CHANNELS}),
                py::array_t<float>(std::vector<py::ssize_t>{0, encoding::MoveEncoder::POLICY_SIZE})
            );
        }

        // Create numpy arrays with actual size
        py::array_t<float> obs_array(std::vector<py::ssize_t>{num_leaves, 8, 8, encoding::PositionEncoder::CHANNELS});
        py::array_t<float> mask_array(std::vector<py::ssize_t>{num_leaves, encoding::MoveEncoder::POLICY_SIZE});

        // Copy data
        std::memcpy(obs_array.mutable_data(), obs_buffer.data(),
                    num_leaves * encoding::PositionEncoder::TOTAL_SIZE * sizeof(float));
        std::memcpy(mask_array.mutable_data(), mask_buffer.data(),
                    num_leaves * encoding::MoveEncoder::POLICY_SIZE * sizeof(float));

        return py::make_tuple(num_leaves, obs_array, mask_array);
    }

    // Update leaves with NN evaluation results
    void update_leaves(py::array_t<float> policies, py::array_t<float> values) {
        auto policy_buf = policies.request();
        auto value_buf = values.request();

        int batch_size = static_cast<int>(value_buf.shape[0]);

        // Convert to vectors
        std::vector<std::vector<float>> policy_vecs(batch_size);
        std::vector<float> value_vec(batch_size);

        float* policy_ptr = static_cast<float*>(policy_buf.ptr);
        float* value_ptr = static_cast<float*>(value_buf.ptr);

        for (int i = 0; i < batch_size; ++i) {
            policy_vecs[i].assign(
                policy_ptr + i * encoding::MoveEncoder::POLICY_SIZE,
                policy_ptr + (i + 1) * encoding::MoveEncoder::POLICY_SIZE
            );
            value_vec[i] = value_ptr[i];
        }

        search_.update_leaves(policy_vecs, value_vec);
    }

    // Check if search is complete
    bool is_complete() const {
        return search_.is_search_complete();
    }

    // Get number of simulations completed
    int get_simulations_completed() const {
        return search_.get_simulations_completed();
    }

    // Get visit counts as policy vector
    py::array_t<int32_t> get_visit_counts() const {
        auto counts = search_.get_visit_counts();
        return py::array_t<int32_t>(counts.size(), counts.data());
    }

    // Get root Q-value after search (MCTS-backed position evaluation)
    float get_root_value() const {
        auto* root = search_.get_root();
        if (!root) return 0.0f;
        // Root has no parent, so parent_q = 0.0f (no FPU needed for visited root)
        return root->q_value(0.0f);
    }

    // Reset for new search
    void reset() {
        pool_.reset();
    }

private:
    static mcts::BatchSearchConfig create_config(int num_simulations, int batch_size, float c_puct, float risk_beta) {
        mcts::BatchSearchConfig config;
        config.num_simulations = num_simulations;
        config.batch_size = batch_size;
        config.c_puct = c_puct;
        config.risk_beta = risk_beta;
        config.dirichlet_alpha = 0.3f;
        config.dirichlet_epsilon = 0.25f;
        return config;
    }

    mcts::NodePool pool_;
    mcts::MCTSSearch search_;
    int batch_size_;
    std::string root_fen_;
};

// Python wrapper for self-play coordinator
// Uses Python callable for neural network evaluation
class PySelfPlayCoordinator {
public:
    PySelfPlayCoordinator(int num_workers = 4, int num_simulations = 800, int batch_size = 256)
        : config_()
    {
        config_.num_workers = num_workers;
        config_.game_config.num_simulations = num_simulations;
        config_.game_config.batch_size = batch_size;
        coordinator_ = std::make_unique<selfplay::SelfPlayCoordinator>(config_);
    }

    // Generate games using Python neural network evaluator
    // evaluator: callable that takes (observations_flat: np.array, num_leaves: int)
    //            and returns (policies: List[np.array], values: np.array)
    py::list generate_games(py::object evaluator, int num_games) {
        // Create C++ evaluator that calls Python
        auto cpp_evaluator = [&evaluator](float* obs_data, int num_leaves,
                                          std::vector<std::vector<float>>& policies,
                                          std::vector<float>& values) {
            // Acquire GIL before calling Python from C++ thread
            py::gil_scoped_acquire acquire;

            // Convert C++ buffer to numpy array
            py::array_t<float> obs_array(std::vector<py::ssize_t>{num_leaves, 8, 8, encoding::PositionEncoder::CHANNELS}, obs_data);

            // Call Python evaluator
            py::object result = evaluator(obs_array, num_leaves);

            // Extract results (expect tuple of (policies, values))
            py::tuple result_tuple = result.cast<py::tuple>();
            py::array_t<float> py_policies = result_tuple[0].cast<py::array_t<float>>();
            py::array_t<float> py_values = result_tuple[1].cast<py::array_t<float>>();

            // Convert back to C++ vectors
            auto policies_buf = py_policies.request();
            auto values_buf = py_values.request();

            float* policies_ptr = static_cast<float*>(policies_buf.ptr);
            float* values_ptr = static_cast<float*>(values_buf.ptr);

            policies.resize(num_leaves);
            values.resize(num_leaves);

            for (int i = 0; i < num_leaves; ++i) {
                policies[i].assign(
                    policies_ptr + i * encoding::MoveEncoder::POLICY_SIZE,
                    policies_ptr + (i + 1) * encoding::MoveEncoder::POLICY_SIZE
                );
                values[i] = values_ptr[i];
            }
        };

        // Release GIL before calling C++ code (worker threads will reacquire as needed)
        std::vector<selfplay::GameTrajectory> trajectories;
        {
            py::gil_scoped_release release;
            trajectories = coordinator_->generate_games(cpp_evaluator, num_games);
        }

        // Convert to Python list of dictionaries
        py::list result;
        for (const auto& traj : trajectories) {
            py::dict game_dict;

            // Convert observations
            py::list obs_list;
            py::list policy_list;
            py::list value_list;

            for (const auto& state : traj.states) {
                // Observation: shape (8, 8, 123)
                py::array_t<float> obs_array(std::vector<py::ssize_t>{8, 8, encoding::PositionEncoder::CHANNELS});
                std::memcpy(obs_array.mutable_data(), state.observation.data(),
                           state.observation.size() * sizeof(float));
                obs_list.append(obs_array);

                // Policy: shape (4672,)
                py::array_t<float> policy_array(std::vector<py::ssize_t>{encoding::MoveEncoder::POLICY_SIZE});
                std::memcpy(policy_array.mutable_data(), state.policy.data(),
                           state.policy.size() * sizeof(float));
                policy_list.append(policy_array);

                // Value: scalar
                value_list.append(state.value);
            }

            game_dict["observations"] = obs_list;
            game_dict["policies"] = policy_list;
            game_dict["values"] = value_list;
            game_dict["num_moves"] = traj.num_moves;

            // Convert result
            int result_value = 0;
            if (traj.result == chess::GameResult::WIN) result_value = 1;
            else if (traj.result == chess::GameResult::LOSE) result_value = -1;
            game_dict["result"] = result_value;

            result.append(game_dict);
        }

        return result;
    }

    // Get statistics
    py::dict get_stats() const {
        auto stats = coordinator_->get_stats();
        py::dict result;
        result["games_completed"] = stats.games_completed;
        result["total_moves"] = stats.total_moves;
        result["white_wins"] = stats.white_wins;
        result["black_wins"] = stats.black_wins;
        result["draws"] = stats.draws;
        result["avg_game_length"] = stats.avg_game_length();
        result["avg_game_time"] = stats.avg_game_time();
        return result;
    }

private:
    selfplay::CoordinatorConfig config_;
    std::unique_ptr<selfplay::SelfPlayCoordinator> coordinator_;
};

// Python wrapper for PARALLEL self-play coordinator (cross-game batching)
// This achieves high GPU utilization by batching NN evaluations across multiple games
class PyParallelSelfPlayCoordinator {
public:
    PyParallelSelfPlayCoordinator(
        int num_workers = 16,
        int games_per_worker = 4,
        int num_simulations = 800,
        int mcts_batch_size = 1,
        int gpu_batch_size = 512,
        float c_puct = 1.5f,
        float dirichlet_alpha = 0.3f,
        float dirichlet_epsilon = 0.25f,
        int temperature_moves = 30,
        int gpu_timeout_ms = 20,
        int stall_detection_us = 500,
        int worker_timeout_ms = 2000,
        int queue_capacity = 8192,
        float fpu_base = 1.0f,
        float risk_beta = 0.0f,
        float opponent_risk_min = 0.0f,
        float opponent_risk_max = 0.0f,
        bool use_gumbel = false,
        int gumbel_top_k = 16,
        float gumbel_c_visit = 50.0f,
        float gumbel_c_scale = 1.0f)
        : replay_buffer_(nullptr)
        , active_coordinator_(nullptr)
    {
        config_.num_workers = num_workers;
        config_.games_per_worker = games_per_worker;
        config_.num_simulations = num_simulations;
        config_.mcts_batch_size = mcts_batch_size;
        config_.gpu_batch_size = gpu_batch_size;
        config_.c_puct = c_puct;
        config_.dirichlet_alpha = dirichlet_alpha;
        config_.dirichlet_epsilon = dirichlet_epsilon;
        config_.temperature_moves = temperature_moves;
        config_.gpu_timeout_ms = gpu_timeout_ms;
        config_.stall_detection_us = stall_detection_us;
        config_.worker_timeout_ms = worker_timeout_ms;
        config_.queue_capacity = queue_capacity;
        config_.fpu_base = fpu_base;
        config_.risk_beta = risk_beta;
        config_.opponent_risk_min = opponent_risk_min;
        config_.opponent_risk_max = opponent_risk_max;
        config_.use_gumbel = use_gumbel;
        config_.gumbel_top_k = gumbel_top_k;
        config_.gumbel_c_visit = gumbel_c_visit;
        config_.gumbel_c_scale = gumbel_c_scale;
    }

    // Set replay buffer (optional - if not set, games are returned instead)
    void set_replay_buffer(training::ReplayBuffer* buffer) {
        replay_buffer_ = buffer;
    }

    // Stop the currently running self-play generation (for graceful shutdown)
    void stop() {
        if (active_coordinator_) {
            active_coordinator_->stop();
        }
    }

    // Check if self-play is currently running
    bool is_running() const {
        return active_coordinator_ && active_coordinator_->is_running();
    }

    // Get live statistics while self-play is running (thread-safe)
    // Reads atomic counters from the active coordinator without blocking
    py::dict get_live_stats() {
        py::dict result;
        if (!active_coordinator_) {
            return result;
        }
        auto m = active_coordinator_->get_detailed_metrics();
        const auto& stats = active_coordinator_->get_stats();
        const auto& qm = active_coordinator_->get_queue_metrics();

        // Game stats
        result["games_completed"] = m.games_completed;
        result["total_moves"] = m.total_moves;
        result["total_simulations"] = m.total_simulations;
        result["total_nn_evals"] = m.total_nn_evals;
        result["white_wins"] = m.white_wins;
        result["black_wins"] = m.black_wins;
        result["draws"] = m.draws;
        result["draws_repetition"] = stats.draws_repetition.load(std::memory_order_relaxed);
        result["draws_stalemate"] = stats.draws_stalemate.load(std::memory_order_relaxed);
        result["draws_fifty_move"] = stats.draws_fifty_move.load(std::memory_order_relaxed);
        result["draws_insufficient"] = stats.draws_insufficient.load(std::memory_order_relaxed);
        result["draws_max_moves"] = stats.draws_max_moves.load(std::memory_order_relaxed);
        result["draws_early_repetition"] = stats.draws_early_repetition.load(std::memory_order_relaxed);
        result["standard_wins"] = stats.standard_wins.load(std::memory_order_relaxed);
        result["opponent_wins"] = stats.opponent_wins.load(std::memory_order_relaxed);
        result["asymmetric_draws"] = stats.asymmetric_draws.load(std::memory_order_relaxed);
        result["avg_game_length"] = m.avg_game_length;

        // Throughput rates
        result["moves_per_sec"] = m.moves_per_sec;
        result["sims_per_sec"] = m.sims_per_sec;
        result["nn_evals_per_sec"] = m.nn_evals_per_sec;

        // GPU/batch metrics
        result["mcts_failures"] = stats.mcts_failures.load(std::memory_order_relaxed);
        result["avg_batch_size"] = m.avg_batch_size;
        result["batch_fill_ratio"] = m.batch_fill_ratio;
        result["pending_requests"] = m.pending_requests;

        // Pipeline health diagnostics (cumulative counters)
        uint64_t total_leaves = qm.total_leaves.load(std::memory_order_relaxed);
        uint64_t drops = qm.submission_drops.load(std::memory_order_relaxed);
        result["pool_exhaustion_count"] = qm.pool_exhaustion_count.load(std::memory_order_relaxed);
        result["partial_submissions"] = qm.partial_submissions.load(std::memory_order_relaxed);
        result["submission_drops"] = drops;
        result["pool_resets"] = qm.pool_resets.load(std::memory_order_relaxed);
        result["submission_waits"] = qm.submission_waits.load(std::memory_order_relaxed);
        result["total_leaves"] = total_leaves;

        // Pool load: fraction of leaves dropped (0.0 = healthy, >0 = pool bottleneck)
        result["pool_load"] = (total_leaves + drops) > 0
            ? static_cast<double>(drops) / static_cast<double>(total_leaves + drops)
            : 0.0;

        // Queue status metrics for dashboard monitoring
        const auto& eq = active_coordinator_->get_eval_queue();
        size_t queue_cap = eq.get_queue_capacity();
        size_t pending = eq.pending_count();  // Actual queue size (not write cursor)
        result["queue_fill_pct"] = queue_cap > 0
            ? (100.0 * pending) / static_cast<double>(queue_cap)
            : 0.0;

        // Per-unit averages (not cumulative totals)
        uint64_t total_batches = qm.total_batches.load(std::memory_order_relaxed);
        uint64_t total_submissions = qm.total_requests_submitted.load(std::memory_order_relaxed);
        uint64_t gpu_wait_us = qm.gpu_wait_time_us.load(std::memory_order_relaxed);
        uint64_t worker_wait_us = qm.worker_wait_time_us.load(std::memory_order_relaxed);

        // Average GPU wait per batch (ms)
        result["gpu_wait_ms"] = total_batches > 0
            ? (gpu_wait_us / 1000.0) / static_cast<double>(total_batches)
            : 0.0;
        // Average worker wait per submission (ms)
        result["worker_wait_ms"] = total_submissions > 0
            ? (worker_wait_us / 1000.0) / static_cast<double>(total_submissions)
            : 0.0;
        result["buffer_swaps"] = total_batches;

        // Spin-poll stall detection metric (average spin time per batch)
        uint64_t spin_polls = qm.spin_poll_count.load(std::memory_order_relaxed);
        uint64_t spin_total_us = qm.spin_poll_total_us.load(std::memory_order_relaxed);
        result["spin_poll_avg_us"] = spin_polls > 0
            ? static_cast<double>(spin_total_us) / static_cast<double>(spin_polls) : 0.0;

        // Tree depth metrics
        result["max_search_depth"] = m.max_search_depth;
        result["min_search_depth"] = m.min_search_depth;
        result["avg_search_depth"] = m.avg_search_depth;

        // Active game move counts
        result["min_current_moves"] = m.min_current_moves;
        result["max_current_moves"] = m.max_current_moves;

        result["is_running"] = active_coordinator_->is_running();
        return result;
    }

    // Generate games using Python neural network evaluator
    // evaluator: callable that takes (observations: np.array, legal_masks: np.array, batch_size: int)
    //            and writes results to (policies: np.array, values: np.array)
    // Returns list of game dictionaries if no replay buffer is set
    py::object generate_games(py::object evaluator) {
        // Create coordinator and store reference for stop()
        active_coordinator_ = std::make_unique<selfplay::ParallelSelfPlayCoordinator>(
            config_, replay_buffer_
        );
        auto& coordinator = active_coordinator_;

        // Create C++ evaluator callback that wraps Python callable
        selfplay::NeuralEvaluatorFn cpp_evaluator = [&evaluator](
            const float* observations,
            const float* legal_masks,
            int batch_size,
            float* out_policies,
            float* out_values)
        {
            // Acquire GIL before calling Python from C++ thread
            py::gil_scoped_acquire acquire;

            // Zero-copy numpy wrapping: observations are NHWC from C++ (no transpose)
            // Python uses torch.permute(0,3,1,2) for zero-cost channels_last layout
            // Use capsule with no-op destructor to prevent Python from freeing C++ memory
            constexpr int CH = encoding::PositionEncoder::CHANNELS;
            py::array_t<float> obs_array(
                {batch_size, 8, 8, CH},
                {8 * 8 * CH * (int)sizeof(float), 8 * CH * (int)sizeof(float),
                 CH * (int)sizeof(float), (int)sizeof(float)},
                const_cast<float*>(observations),
                py::capsule(observations, [](void*) {})
            );

            py::array_t<float> mask_array(
                {batch_size, 4672},
                {4672 * (int)sizeof(float), (int)sizeof(float)},
                const_cast<float*>(legal_masks),
                py::capsule(legal_masks, [](void*) {})
            );

            // Create writable numpy views over C++ output buffers (zero-copy output)
            py::array_t<float> out_policy_array(
                {batch_size, 4672},
                {4672 * (int)sizeof(float), (int)sizeof(float)},
                out_policies,
                py::capsule(out_policies, [](void*) {})
            );
            py::array_t<float> out_value_array(
                {batch_size, 3},
                {3 * (int)sizeof(float), (int)sizeof(float)},
                out_values,
                py::capsule(out_values, [](void*) {})
            );

            // Call Python evaluator with output buffers
            // Signature: evaluator(obs, masks, batch_size, out_policies, out_values) -> None
            // Or legacy: evaluator(obs, masks, batch_size) -> (policies, values)
            try {
                py::object result = evaluator(obs_array, mask_array, batch_size,
                                             out_policy_array, out_value_array);
                // If evaluator returns None, it wrote directly to output buffers
                if (!result.is_none()) {
                    // Legacy path: evaluator returned (policies, values) tuple
                    py::tuple result_tuple = result.cast<py::tuple>();
                    py::array_t<float> py_policies = result_tuple[0].cast<py::array_t<float>>();
                    py::array_t<float> py_values = result_tuple[1].cast<py::array_t<float>>();

                    auto policies_buf = py_policies.request();
                    auto values_buf = py_values.request();

                    std::memcpy(out_policies, policies_buf.ptr,
                               batch_size * 4672 * sizeof(float));
                    std::memcpy(out_values, values_buf.ptr,
                               batch_size * 3 * sizeof(float));
                }
            } catch (py::error_already_set& e) {
                // If the new 5-arg signature fails, fall back to 3-arg legacy
                fprintf(stderr, "[WARNING] Evaluator 5-arg call failed: %s\n", e.what());
                fflush(stderr);
                PyErr_Clear();
                py::object result = evaluator(obs_array, mask_array, batch_size);

                py::tuple result_tuple = result.cast<py::tuple>();
                py::array_t<float> py_policies = result_tuple[0].cast<py::array_t<float>>();
                py::array_t<float> py_values = result_tuple[1].cast<py::array_t<float>>();

                auto policies_buf = py_policies.request();
                auto values_buf = py_values.request();

                std::memcpy(out_policies, policies_buf.ptr,
                           batch_size * 4672 * sizeof(float));
                std::memcpy(out_values, values_buf.ptr,
                           batch_size * 3 * sizeof(float));
            }
        };

        // Run generation (releases GIL during C++ execution)
        {
            py::gil_scoped_release release;
            coordinator->generate_games(cpp_evaluator);
        }

        // Get stats for logging (use reference since stats contains atomics)
        const auto& stats = coordinator->get_stats();
        const auto& queue_metrics = coordinator->get_queue_metrics();

        // Return completed games if no replay buffer
        if (replay_buffer_ == nullptr) {
            auto trajectories = coordinator->get_completed_games();

            py::list result;
            for (const auto& traj : trajectories) {
                py::dict game_dict;

                py::list obs_list;
                py::list policy_list;
                py::list value_list;

                for (const auto& state : traj.states) {
                    py::array_t<float> obs_array(std::vector<py::ssize_t>{8, 8, encoding::PositionEncoder::CHANNELS});
                    std::memcpy(obs_array.mutable_data(), state.observation.data(),
                               state.observation.size() * sizeof(float));
                    obs_list.append(obs_array);

                    py::array_t<float> policy_array(std::vector<py::ssize_t>{4672});
                    std::memcpy(policy_array.mutable_data(), state.policy.data(),
                               state.policy.size() * sizeof(float));
                    policy_list.append(policy_array);

                    value_list.append(state.value);
                }

                game_dict["observations"] = obs_list;
                game_dict["policies"] = policy_list;
                game_dict["values"] = value_list;
                game_dict["num_moves"] = traj.num_moves;

                int result_value = 0;
                if (traj.result == chess::GameResult::WIN) result_value = 1;
                else if (traj.result == chess::GameResult::LOSE) result_value = -1;
                game_dict["result"] = result_value;

                result.append(game_dict);
            }

            return result;
        }

        // Return stats dictionary if using replay buffer
        py::dict result;
        result["games_completed"] = stats.games_completed.load();
        result["total_moves"] = stats.total_moves.load();
        result["white_wins"] = stats.white_wins.load();
        result["black_wins"] = stats.black_wins.load();
        result["draws"] = stats.draws.load();
        result["standard_wins"] = stats.standard_wins.load();
        result["opponent_wins"] = stats.opponent_wins.load();
        result["asymmetric_draws"] = stats.asymmetric_draws.load();
        result["mcts_failures"] = stats.mcts_failures.load();  // Track evaluation failures
        result["gpu_errors"] = stats.gpu_errors.load();  // Track GPU thread exceptions
        result["total_simulations"] = stats.total_simulations.load();
        result["total_nn_evals"] = stats.total_nn_evals.load();

        // Draw reason breakdown
        result["draws_repetition"] = stats.draws_repetition.load();
        result["draws_stalemate"] = stats.draws_stalemate.load();
        result["draws_fifty_move"] = stats.draws_fifty_move.load();
        result["draws_insufficient"] = stats.draws_insufficient.load();
        result["draws_max_moves"] = stats.draws_max_moves.load();
        result["draws_early_repetition"] = stats.draws_early_repetition.load();

        // Surface C++ thread errors to Python
        result["cpp_error"] = coordinator->get_last_error();

        // Queue diagnostic metrics (for debugging parallel pipeline health)
        result["pool_exhaustion_count"] = queue_metrics.pool_exhaustion_count.load();
        result["partial_submissions"] = queue_metrics.partial_submissions.load();
        result["submission_drops"] = queue_metrics.submission_drops.load();
        result["pool_resets"] = queue_metrics.pool_resets.load();
        result["submission_waits"] = queue_metrics.submission_waits.load();
        result["avg_batch_size"] = queue_metrics.avg_batch_size();
        result["total_batches"] = queue_metrics.total_batches.load();
        result["total_leaves"] = queue_metrics.total_leaves.load();

        // Tree depth metrics
        result["max_search_depth"] = stats.max_search_depth.load();
        result["min_search_depth"] = stats.min_search_depth.load() == INT64_MAX
            ? 0 : stats.min_search_depth.load();
        int64_t total_depth = stats.total_max_depth.load();
        int64_t depth_samples = stats.depth_sample_count.load();
        result["avg_search_depth"] = depth_samples > 0 ? static_cast<double>(total_depth) / depth_samples : 0.0;

        return result;
    }

    // Get configuration as dict
    py::dict get_config() const {
        py::dict d;
        d["num_workers"] = config_.num_workers;
        d["games_per_worker"] = config_.games_per_worker;
        d["num_simulations"] = config_.num_simulations;
        d["mcts_batch_size"] = config_.mcts_batch_size;
        d["gpu_batch_size"] = config_.gpu_batch_size;
        d["c_puct"] = config_.c_puct;
        d["dirichlet_alpha"] = config_.dirichlet_alpha;
        d["dirichlet_epsilon"] = config_.dirichlet_epsilon;
        d["temperature_moves"] = config_.temperature_moves;
        d["gpu_timeout_ms"] = config_.gpu_timeout_ms;
        d["fpu_base"] = config_.fpu_base;
        d["risk_beta"] = config_.risk_beta;
        d["opponent_risk_min"] = config_.opponent_risk_min;
        d["opponent_risk_max"] = config_.opponent_risk_max;
        d["use_gumbel"] = config_.use_gumbel;
        d["gumbel_top_k"] = config_.gumbel_top_k;
        d["gumbel_c_visit"] = config_.gumbel_c_visit;
        d["gumbel_c_scale"] = config_.gumbel_c_scale;
        return d;
    }

    // Get a sample game from the last generation run (prefers decisive games)
    py::dict get_sample_game() const {
        py::dict result;
        if (!active_coordinator_ || !active_coordinator_->has_sample_game()) {
            result["has_game"] = false;
            return result;
        }

        auto sample = active_coordinator_->get_sample_game();
        result["has_game"] = true;
        result["moves"] = sample.moves_uci;
        result["num_moves"] = sample.num_moves;

        // Convert GameResult to PGN result string
        if (sample.result == chess::GameResult::WIN) {
            result["result"] = "1-0";
        } else if (sample.result == chess::GameResult::LOSE) {
            result["result"] = "0-1";
        } else {
            result["result"] = "1/2-1/2";
        }

        // Convert GameResultReason to string
        const char* reason_str = "unknown";
        switch (sample.result_reason) {
            case chess::GameResultReason::CHECKMATE: reason_str = "checkmate"; break;
            case chess::GameResultReason::STALEMATE: reason_str = "stalemate"; break;
            case chess::GameResultReason::INSUFFICIENT_MATERIAL: reason_str = "insufficient_material"; break;
            case chess::GameResultReason::FIFTY_MOVE_RULE: reason_str = "fifty_move"; break;
            case chess::GameResultReason::THREEFOLD_REPETITION: reason_str = "repetition"; break;
            case chess::GameResultReason::NONE: reason_str = "max_moves"; break;
        }
        result["result_reason"] = reason_str;

        return result;
    }

    // Total games that will be generated
    int total_games() const {
        return config_.num_workers * config_.games_per_worker;
    }

    // =========================================================================
    // Lifecycle Methods (for concurrent reanalysis)
    // =========================================================================

    // Start generation (non-blocking) — use with set_secondary_queue for reanalysis
    void start_generation(py::object evaluator) {
        active_coordinator_ = std::make_unique<selfplay::ParallelSelfPlayCoordinator>(
            config_, replay_buffer_
        );

        // Store evaluator to prevent garbage collection during async operation
        stored_evaluator_ = evaluator;

        // Create C++ evaluator callback that wraps stored Python callable
        cpp_evaluator_ = [this](
            const float* observations,
            const float* legal_masks,
            int batch_size,
            float* out_policies,
            float* out_values)
        {
            py::gil_scoped_acquire acquire;

            constexpr int CH = encoding::PositionEncoder::CHANNELS;
            py::array_t<float> obs_array(
                {batch_size, 8, 8, CH},
                {8 * 8 * CH * (int)sizeof(float), 8 * CH * (int)sizeof(float),
                 CH * (int)sizeof(float), (int)sizeof(float)},
                const_cast<float*>(observations),
                py::capsule(observations, [](void*) {})
            );

            py::array_t<float> mask_array(
                {batch_size, 4672},
                {4672 * (int)sizeof(float), (int)sizeof(float)},
                const_cast<float*>(legal_masks),
                py::capsule(legal_masks, [](void*) {})
            );

            py::array_t<float> out_policy_array(
                {batch_size, 4672},
                {4672 * (int)sizeof(float), (int)sizeof(float)},
                out_policies,
                py::capsule(out_policies, [](void*) {})
            );
            py::array_t<float> out_value_array(
                {batch_size, 3},
                {3 * (int)sizeof(float), (int)sizeof(float)},
                out_values,
                py::capsule(out_values, [](void*) {})
            );

            try {
                py::object result = stored_evaluator_(obs_array, mask_array, batch_size,
                                                      out_policy_array, out_value_array);
                if (!result.is_none()) {
                    py::tuple result_tuple = result.cast<py::tuple>();
                    py::array_t<float> py_policies = result_tuple[0].cast<py::array_t<float>>();
                    py::array_t<float> py_values = result_tuple[1].cast<py::array_t<float>>();

                    auto policies_buf = py_policies.request();
                    auto values_buf = py_values.request();

                    std::memcpy(out_policies, policies_buf.ptr,
                               batch_size * 4672 * sizeof(float));
                    std::memcpy(out_values, values_buf.ptr,
                               batch_size * 3 * sizeof(float));
                }
            } catch (py::error_already_set& e) {
                fprintf(stderr, "[WARNING] Evaluator 5-arg call failed: %s\n", e.what());
                fflush(stderr);
                PyErr_Clear();
                py::object result = stored_evaluator_(obs_array, mask_array, batch_size);

                py::tuple result_tuple = result.cast<py::tuple>();
                py::array_t<float> py_policies = result_tuple[0].cast<py::array_t<float>>();
                py::array_t<float> py_values = result_tuple[1].cast<py::array_t<float>>();

                auto policies_buf = py_policies.request();
                auto values_buf = py_values.request();

                std::memcpy(out_policies, policies_buf.ptr,
                           batch_size * 4672 * sizeof(float));
                std::memcpy(out_values, values_buf.ptr,
                           batch_size * 3 * sizeof(float));
            }
        };

        // Start generation (non-blocking) — release GIL so GPU thread can acquire it
        py::gil_scoped_release release;
        active_coordinator_->start(cpp_evaluator_);
    }

    void set_secondary_queue_from_reanalyzer(selfplay::Reanalyzer& reanalyzer) {
        if (active_coordinator_) {
            active_coordinator_->set_secondary_queue(&reanalyzer.get_eval_queue());
        }
    }

    void clear_secondary_queue() {
        if (active_coordinator_) {
            active_coordinator_->clear_secondary_queue();
        }
    }

    void wait_for_workers_impl() {
        if (active_coordinator_) {
            active_coordinator_->wait_for_workers();
        }
    }

    void shutdown_gpu_thread_impl() {
        if (active_coordinator_) {
            active_coordinator_->shutdown_gpu_thread();
        }
    }

    void clear_stored_evaluator() {
        stored_evaluator_ = py::none();
        cpp_evaluator_ = nullptr;
    }

    py::dict get_generation_stats() {
        py::dict result;
        if (!active_coordinator_) return result;

        const auto& stats = active_coordinator_->get_stats();
        const auto& queue_metrics = active_coordinator_->get_queue_metrics();

        result["games_completed"] = stats.games_completed.load();
        result["total_moves"] = stats.total_moves.load();
        result["white_wins"] = stats.white_wins.load();
        result["black_wins"] = stats.black_wins.load();
        result["draws"] = stats.draws.load();
        result["standard_wins"] = stats.standard_wins.load();
        result["opponent_wins"] = stats.opponent_wins.load();
        result["asymmetric_draws"] = stats.asymmetric_draws.load();
        result["mcts_failures"] = stats.mcts_failures.load();
        result["gpu_errors"] = stats.gpu_errors.load();
        result["total_simulations"] = stats.total_simulations.load();
        result["total_nn_evals"] = stats.total_nn_evals.load();

        result["draws_repetition"] = stats.draws_repetition.load();
        result["draws_stalemate"] = stats.draws_stalemate.load();
        result["draws_fifty_move"] = stats.draws_fifty_move.load();
        result["draws_insufficient"] = stats.draws_insufficient.load();
        result["draws_max_moves"] = stats.draws_max_moves.load();
        result["draws_early_repetition"] = stats.draws_early_repetition.load();

        result["cpp_error"] = active_coordinator_->get_last_error();

        result["pool_exhaustion_count"] = queue_metrics.pool_exhaustion_count.load();
        result["partial_submissions"] = queue_metrics.partial_submissions.load();
        result["submission_drops"] = queue_metrics.submission_drops.load();
        result["pool_resets"] = queue_metrics.pool_resets.load();
        result["submission_waits"] = queue_metrics.submission_waits.load();
        result["avg_batch_size"] = queue_metrics.avg_batch_size();
        result["total_batches"] = queue_metrics.total_batches.load();
        result["total_leaves"] = queue_metrics.total_leaves.load();

        // Batch fire reason breakdown
        result["batches_fired_full"] = queue_metrics.batches_fired_full.load();
        result["batches_fired_stall"] = queue_metrics.batches_fired_stall.load();
        result["batches_fired_timeout"] = queue_metrics.batches_fired_timeout.load();

        result["max_search_depth"] = stats.max_search_depth.load();
        result["min_search_depth"] = stats.min_search_depth.load() == INT64_MAX
            ? 0 : stats.min_search_depth.load();
        int64_t total_depth = stats.total_max_depth.load();
        int64_t depth_samples = stats.depth_sample_count.load();
        result["avg_search_depth"] = depth_samples > 0
            ? static_cast<double>(total_depth) / depth_samples : 0.0;

        return result;
    }

private:
    selfplay::ParallelSelfPlayConfig config_;
    training::ReplayBuffer* replay_buffer_;
    std::unique_ptr<selfplay::ParallelSelfPlayCoordinator> active_coordinator_;
    py::object stored_evaluator_;
    selfplay::NeuralEvaluatorFn cpp_evaluator_;
};

PYBIND11_MODULE(alphazero_cpp, m) {
    m.doc() = "AlphaZero C++ - High-performance MCTS for chess";

    // MCTS Search class (batched leaf evaluation - proper AlphaZero)
    py::class_<PyBatchedMCTSSearch>(m, "BatchedMCTSSearch")
        .def(py::init<int, int, float, float>(),
             py::arg("num_simulations") = 800,
             py::arg("batch_size") = 256,
             py::arg("c_puct") = 1.5f,
             py::arg("risk_beta") = 0.0f,
             "Create MCTS search engine with batched leaf evaluation (proper AlphaZero)\n"
             "\nParameters:\n"
             "  risk_beta: ERM risk sensitivity (default 0.0). >0 risk-seeking, <0 risk-averse")
        .def("init_search", &PyBatchedMCTSSearch::init_search,
             py::arg("fen"),
             py::arg("root_policy"),
             py::arg("root_value"),
             py::arg("history_fens") = std::vector<std::string>{},
             "Initialize search with root position and initial NN evaluation")
        .def("collect_leaves", &PyBatchedMCTSSearch::collect_leaves,
             "Collect leaves that need NN evaluation. Returns (num_leaves, observations, legal_masks)")
        .def("update_leaves", &PyBatchedMCTSSearch::update_leaves,
             py::arg("policies"),
             py::arg("values"),
             "Update leaves with NN evaluation results")
        .def("is_complete", &PyBatchedMCTSSearch::is_complete,
             "Check if search is complete")
        .def("get_simulations_completed", &PyBatchedMCTSSearch::get_simulations_completed,
             "Get number of simulations completed")
        .def("get_visit_counts", &PyBatchedMCTSSearch::get_visit_counts,
             "Get visit counts as policy vector (4672 dimensions)")
        .def("get_root_value", &PyBatchedMCTSSearch::get_root_value,
             "Get root Q-value after search (MCTS-backed position evaluation)")
        .def("reset", &PyBatchedMCTSSearch::reset,
             "Reset search tree for new search");

    // Utility functions
    m.def("encode_position", &encode_position,
          py::arg("fen"),
          py::arg("history_fens") = std::vector<std::string>{},
          "Encode chess position for neural network input.\n"
          "history_fens: Optional list of FEN strings for position history (up to 8)");

    m.def("encode_position_to_buffer", &encode_position_to_buffer,
          py::arg("fen"),
          py::arg("buffer"),
          "Encode chess position directly to provided buffer (zero-copy)");

    m.def("encode_batch", &encode_batch,
          py::arg("fens"),
          py::arg("buffer"),
          py::arg("use_parallel") = true,
          "Encode multiple positions in parallel (batch encoding with OpenMP)");

    m.def("wdl_to_value", &mcts::wdl_to_value,
          py::arg("pw"), py::arg("pd"), py::arg("pl"),
          "Convert WDL probabilities to scalar value.\n"
          "value = pw - pl\n"
          "pw: P(win), pd: P(draw), pl: P(loss)");

    m.def("move_to_index", &move_to_index,
          py::arg("uci_move"),
          py::arg("fen"),
          "Convert UCI move to policy index");

    m.def("index_to_move", &index_to_move,
          py::arg("index"),
          py::arg("fen"),
          "Convert policy index to UCI move");

    // Self-play coordinator (original - per-game NN calls)
    py::class_<PySelfPlayCoordinator>(m, "SelfPlayCoordinator")
        .def(py::init<int, int, int>(),
             py::arg("num_workers") = 4,
             py::arg("num_simulations") = 800,
             py::arg("batch_size") = 256,
             "Create self-play coordinator for multi-threaded game generation")
        .def("generate_games", &PySelfPlayCoordinator::generate_games,
             py::arg("evaluator"),
             py::arg("num_games"),
             "Generate self-play games using neural network evaluator.\n"
             "evaluator: callable(observations, num_leaves) -> (policies, values)")
        .def("get_stats", &PySelfPlayCoordinator::get_stats,
             "Get self-play statistics");

    // Parallel self-play coordinator (cross-game batching for high GPU utilization)
    py::class_<PyParallelSelfPlayCoordinator>(m, "ParallelSelfPlayCoordinator")
        .def(py::init<int, int, int, int, int, float, float, float, int, int, int, int, int, float, float, float, float, bool, int, float, float>(),
             py::arg("num_workers") = 16,
             py::arg("games_per_worker") = 4,
             py::arg("num_simulations") = 800,
             py::arg("mcts_batch_size") = 1,
             py::arg("gpu_batch_size") = 512,
             py::arg("c_puct") = 1.5f,
             py::arg("dirichlet_alpha") = 0.3f,
             py::arg("dirichlet_epsilon") = 0.25f,
             py::arg("temperature_moves") = 30,
             py::arg("gpu_timeout_ms") = 20,
             py::arg("stall_detection_us") = 500,
             py::arg("worker_timeout_ms") = 2000,
             py::arg("queue_capacity") = 8192,
             py::arg("fpu_base") = 1.0f,
             py::arg("risk_beta") = 0.0f,
             py::arg("opponent_risk_min") = 0.0f,
             py::arg("opponent_risk_max") = 0.0f,
             py::arg("use_gumbel") = true,
             py::arg("gumbel_top_k") = 16,
             py::arg("gumbel_c_visit") = 50.0f,
             py::arg("gumbel_c_scale") = 1.0f,
             "Create parallel self-play coordinator with cross-game batching.\n"
             "This achieves high GPU utilization by batching NN evaluations across multiple games.\n"
             "\nParameters:\n"
             "  num_workers: Number of worker threads (default 16)\n"
             "  games_per_worker: Games each worker plays (default 4)\n"
             "  num_simulations: MCTS simulations per move (default 800)\n"
             "  mcts_batch_size: Leaves collected per MCTS iteration (default 1)\n"
             "  gpu_batch_size: Maximum GPU batch size (default 512)\n"
             "  c_puct: PUCT exploration constant (default 1.5)\n"
             "  dirichlet_alpha: Dirichlet noise alpha (default 0.3)\n"
             "  dirichlet_epsilon: Dirichlet noise weight (default 0.25)\n"
             "  temperature_moves: Moves with temperature=1.0 (default 30)\n"
             "  gpu_timeout_ms: GPU batch collection timeout (default 20)\n"
             "  stall_detection_us: Spin-poll stall detection in microseconds (default 500)\n"
             "  worker_timeout_ms: Worker submit backpressure timeout (default 2000)\n"
             "  queue_capacity: Evaluation queue capacity (default 8192)\n"
             "  fpu_base: Dynamic FPU base penalty (default 1.0). penalty = fpu_base * (1 - prior)\n"
             "  risk_beta: ERM risk sensitivity (default 0.0). >0 risk-seeking, <0 risk-averse. Range [-3, 3]\n"
             "  opponent_risk_min: Min opponent risk beta for asymmetric games (default 0.0)\n"
             "  opponent_risk_max: Max opponent risk beta for asymmetric games (default 0.0)\n"
             "  use_gumbel: Use Gumbel Top-k Sequential Halving at root (default true)\n"
             "  gumbel_top_k: Initial m for Sequential Halving (default 16)\n"
             "  gumbel_c_visit: sigma() visit constant (default 50.0)\n"
             "  gumbel_c_scale: sigma() scale factor (default 1.0)")
        .def("set_replay_buffer", &PyParallelSelfPlayCoordinator::set_replay_buffer,
             py::arg("buffer"),
             "Set replay buffer for direct data storage (optional).\n"
             "If not set, completed games are returned from generate_games().")
        .def("generate_games", &PyParallelSelfPlayCoordinator::generate_games,
             py::arg("evaluator"),
             "Generate self-play games using cross-game batching.\n"
             "\nevaluator: callable(observations, legal_masks, batch_size) -> (policies, values)\n"
             "  observations: np.array shape (batch_size, 8, 8, 123) NHWC layout\n"
             "  legal_masks: np.array shape (batch_size, 4672)\n"
             "  batch_size: int, number of positions to evaluate\n"
             "  Returns: (policies np.array (batch_size, 4672), values np.array (batch_size,))\n"
             "\nReturns: If replay_buffer is set, returns stats dict.\n"
             "         Otherwise, returns list of game trajectories.")
        .def("get_config", &PyParallelSelfPlayCoordinator::get_config,
             "Get configuration as dictionary")
        .def("total_games", &PyParallelSelfPlayCoordinator::total_games,
             "Get total number of games that will be generated")
        .def("stop", &PyParallelSelfPlayCoordinator::stop,
             "Stop self-play generation (for graceful shutdown). Can be called from another thread.")
        .def("is_running", &PyParallelSelfPlayCoordinator::is_running,
             "Check if self-play is currently running")
        .def("get_live_stats", &PyParallelSelfPlayCoordinator::get_live_stats,
             "Get live statistics while self-play is running.\n"
             "Thread-safe: reads atomic counters without blocking.\n"
             "Returns empty dict if no coordinator is active.")
        .def("get_sample_game", &PyParallelSelfPlayCoordinator::get_sample_game,
             "Get a sample game from the last generation run.\n"
             "Returns dict with keys: has_game (bool), moves (list[str]), result (str), num_moves (int).\n"
             "Prefers decisive games (1-0 or 0-1) over draws.")
        .def("get_last_error", [](PyParallelSelfPlayCoordinator& self) -> std::string {
            // Error is primarily surfaced via cpp_error key in generate_games() result dict
            return "";
        },
        "Get last C++ error (use cpp_error key in generate_games result instead)")
        // Lifecycle methods for concurrent reanalysis
        .def("start_generation", &PyParallelSelfPlayCoordinator::start_generation,
             py::arg("evaluator"),
             "Start self-play generation (non-blocking).\n"
             "Use with set_secondary_queue + reanalyzer for concurrent reanalysis.\n"
             "evaluator: same callable as generate_games()")
        .def("set_secondary_queue", &PyParallelSelfPlayCoordinator::set_secondary_queue_from_reanalyzer,
             py::arg("reanalyzer"),
             "Connect reanalyzer's eval queue as secondary GPU queue.\n"
             "GPU thread fills spare batch capacity from reanalysis workers.")
        .def("clear_secondary_queue", &PyParallelSelfPlayCoordinator::clear_secondary_queue,
             "Disconnect the secondary (reanalysis) eval queue")
        .def("wait_for_workers", [](PyParallelSelfPlayCoordinator& self) {
                 py::gil_scoped_release release;
                 self.wait_for_workers_impl();
             },
             "Wait for self-play workers to finish (GPU thread keeps running for reanalysis)")
        .def("shutdown_gpu_thread", [](PyParallelSelfPlayCoordinator& self) {
                 {
                     py::gil_scoped_release release;
                     self.shutdown_gpu_thread_impl();
                 }
                 // GIL held: safe to clear Python objects
                 self.clear_stored_evaluator();
             },
             "Shut down GPU thread after reanalysis completes.\n"
             "Call after reanalyzer.wait() and clear_secondary_queue().")
        .def("get_generation_stats", &PyParallelSelfPlayCoordinator::get_generation_stats,
             "Get self-play stats after start_generation workflow.\n"
             "Returns same dict as generate_games() when using replay buffer.");

    // Reanalysis configuration
    py::class_<selfplay::ReanalysisConfig>(m, "ReanalysisConfig")
        .def(py::init<>())
        .def_readwrite("num_workers", &selfplay::ReanalysisConfig::num_workers)
        .def_readwrite("num_simulations", &selfplay::ReanalysisConfig::num_simulations)
        .def_readwrite("mcts_batch_size", &selfplay::ReanalysisConfig::mcts_batch_size)
        .def_readwrite("c_puct", &selfplay::ReanalysisConfig::c_puct)
        .def_readwrite("fpu_base", &selfplay::ReanalysisConfig::fpu_base)
        .def_readwrite("risk_beta", &selfplay::ReanalysisConfig::risk_beta)
        .def_readwrite("use_gumbel", &selfplay::ReanalysisConfig::use_gumbel)
        .def_readwrite("gumbel_top_k", &selfplay::ReanalysisConfig::gumbel_top_k)
        .def_readwrite("gumbel_c_visit", &selfplay::ReanalysisConfig::gumbel_c_visit)
        .def_readwrite("gumbel_c_scale", &selfplay::ReanalysisConfig::gumbel_c_scale)
        .def_readwrite("gpu_batch_size", &selfplay::ReanalysisConfig::gpu_batch_size)
        .def_readwrite("worker_timeout_ms", &selfplay::ReanalysisConfig::worker_timeout_ms)
        .def_readwrite("queue_capacity", &selfplay::ReanalysisConfig::queue_capacity);

    // Reanalyzer — runs MCTS re-searches on replay buffer positions
    py::class_<selfplay::Reanalyzer>(m, "Reanalyzer")
        .def(py::init<training::ReplayBuffer&, const selfplay::ReanalysisConfig&>(),
             py::arg("buffer"), py::arg("config"),
             "Create reanalyzer for MCTS re-search of replay buffer positions.\n"
             "Uses coordinator's GPU thread via secondary eval queue (no own GPU thread).")
        .def("set_indices", &selfplay::Reanalyzer::set_indices,
             py::arg("indices"),
             "Set buffer indices to reanalyze (call before start)")
        .def("start", &selfplay::Reanalyzer::start,
             "Start reanalysis workers (non-blocking)")
        .def("wait", [](selfplay::Reanalyzer& self) {
                 py::gil_scoped_release release;
                 self.wait();
             },
             "Wait for all reanalysis positions to be processed")
        .def("stop", [](selfplay::Reanalyzer& self) {
                 py::gil_scoped_release release;
                 self.stop();
             },
             "Graceful shutdown (can be called from another thread)")
        .def("get_stats", [](selfplay::Reanalyzer& self) {
                 const auto& s = self.get_stats();
                 py::dict d;
                 d["positions_completed"] = s.positions_completed.load();
                 d["total_simulations"] = s.total_simulations.load();
                 d["total_nn_evals"] = s.total_nn_evals.load();
                 d["positions_skipped"] = s.positions_skipped.load();
                 d["mean_kl"] = s.mean_kl();
                 return d;
             },
             "Get reanalysis statistics")
        .def("get_last_error", &selfplay::Reanalyzer::get_last_error,
             "Get last error message (empty if no errors)");

    // Replay Buffer
    py::class_<training::ReplayBuffer>(m, "ReplayBuffer")
        .def(py::init<size_t>(),
             py::arg("capacity"),
             "Create replay buffer with fixed capacity")
        .def("add_sample", [](training::ReplayBuffer& self,
                             py::array_t<float> observation,
                             py::array_t<float> policy,
                             float value,
                             py::object wdl,
                             py::object soft_value) {
            auto obs_buf = observation.request();
            auto pol_buf = policy.request();

            std::vector<float> obs_vec(static_cast<float*>(obs_buf.ptr),
                                       static_cast<float*>(obs_buf.ptr) + obs_buf.size);
            std::vector<float> pol_vec(static_cast<float*>(pol_buf.ptr),
                                       static_cast<float*>(pol_buf.ptr) + pol_buf.size);

            const float* wdl_ptr = nullptr;
            if (!wdl.is_none()) {
                auto wdl_arr = wdl.cast<py::array_t<float>>();
                wdl_ptr = static_cast<const float*>(wdl_arr.request().ptr);
            }
            const float* sv_ptr = nullptr;
            float sv_val = 0.0f;
            if (!soft_value.is_none()) {
                sv_val = soft_value.cast<float>();
                sv_ptr = &sv_val;
            }
            self.add_sample(obs_vec, pol_vec, value, wdl_ptr, sv_ptr);
        },
        py::arg("observation"),
        py::arg("policy"),
        py::arg("value"),
        py::arg("wdl") = py::none(),
        py::arg("soft_value") = py::none(),
        "Add a single training sample to buffer (optional WDL target and ERM risk value)")
        .def("add_batch", [](training::ReplayBuffer& self,
                            py::array_t<float> observations,
                            py::array_t<float> policies,
                            py::array_t<float> values,
                            py::object wdl_targets) {
            // Zero-copy implementation: direct pointer access
            auto obs_buf = observations.request();
            auto pol_buf = policies.request();
            auto val_buf = values.request();

            size_t batch_size = obs_buf.shape[0];

            // Direct pointer access (zero-copy)
            float* obs_ptr = static_cast<float*>(obs_buf.ptr);
            float* pol_ptr = static_cast<float*>(pol_buf.ptr);
            float* val_ptr = static_cast<float*>(val_buf.ptr);

            const float* wdl_ptr = nullptr;
            if (!wdl_targets.is_none()) {
                auto wdl_arr = wdl_targets.cast<py::array_t<float>>();
                wdl_ptr = static_cast<const float*>(wdl_arr.request().ptr);
            }

            // Call zero-copy add_batch_raw
            self.add_batch_raw(batch_size, obs_ptr, pol_ptr, val_ptr, wdl_ptr);
        },
        py::arg("observations"),
        py::arg("policies"),
        py::arg("values"),
        py::arg("wdl_targets") = py::none(),
        "Add a batch of training samples to buffer (optional WDL targets)")
        .def("sample", [](training::ReplayBuffer& self, size_t batch_size) {
            std::vector<float> observations, policies, values, wdl_targets, soft_values;

            bool success = self.sample(batch_size, observations, policies, values,
                                       &wdl_targets, &soft_values);
            if (!success) {
                throw std::runtime_error("Not enough samples in buffer");
            }

            // Return as numpy arrays
            constexpr size_t OBS_FLAT = encoding::PositionEncoder::TOTAL_SIZE;  // 7872
            py::array_t<float> obs_array(std::vector<size_t>{batch_size, OBS_FLAT});
            py::array_t<float> pol_array(std::vector<size_t>{batch_size, 4672UL});
            py::array_t<float> val_array(std::vector<size_t>{batch_size});
            py::array_t<float> wdl_array(std::vector<size_t>{batch_size, 3UL});
            py::array_t<float> sv_array(std::vector<size_t>{batch_size});

            std::memcpy(obs_array.mutable_data(), observations.data(),
                       batch_size * OBS_FLAT * sizeof(float));
            std::memcpy(pol_array.mutable_data(), policies.data(),
                       batch_size * 4672 * sizeof(float));
            std::memcpy(val_array.mutable_data(), values.data(),
                       batch_size * sizeof(float));
            std::memcpy(wdl_array.mutable_data(), wdl_targets.data(),
                       batch_size * 3 * sizeof(float));
            std::memcpy(sv_array.mutable_data(), soft_values.data(),
                       batch_size * sizeof(float));

            return py::make_tuple(obs_array, pol_array, val_array, wdl_array, sv_array);
        },
        py::arg("batch_size"),
        "Sample a random batch of training data.\n"
        "Returns: (observations, policies, values, wdl_targets, soft_values)")
        .def("size", &training::ReplayBuffer::size,
             "Get current number of samples in buffer")
        .def("capacity", &training::ReplayBuffer::capacity,
             "Get buffer capacity")
        .def("total_added", &training::ReplayBuffer::total_added,
             "Get total number of samples ever added")
        .def("total_games", &training::ReplayBuffer::total_games,
             "Get total number of games added")
        .def("is_ready", &training::ReplayBuffer::is_ready,
             py::arg("min_size"),
             "Check if buffer has enough samples for training")
        .def("clear", &training::ReplayBuffer::clear,
             "Clear all samples from buffer")
        .def("get_stats", [](const training::ReplayBuffer& self) {
            auto stats = self.get_stats();
            py::dict d;
            d["size"] = stats.size;
            d["capacity"] = stats.capacity;
            d["total_added"] = stats.total_added;
            d["total_games"] = stats.total_games;
            d["utilization"] = stats.utilization;
            d["wins"] = stats.wins;
            d["draws"] = stats.draws;
            d["losses"] = stats.losses;
            return d;
        },
        "Get buffer statistics (includes W/D/L composition)")
        .def("save", &training::ReplayBuffer::save,
             py::arg("path"),
             "Save buffer contents to binary file (.rpbf format).\n"
             "Returns True if successful.")
        .def("load", &training::ReplayBuffer::load,
             py::arg("path"),
             "Load buffer contents from binary file (.rpbf format).\n"
             "Returns True if successful. Truncates to buffer capacity if needed.")
        .def("set_iteration", &training::ReplayBuffer::set_iteration,
             py::arg("iteration"),
             "Set the current training iteration.\n"
             "Called before self-play so metadata tracks which iteration generated each sample.")
        .def("current_iteration", &training::ReplayBuffer::current_iteration,
             "Get the current training iteration.")
        .def("get_composition", [](const training::ReplayBuffer& self) {
            auto comp = self.get_composition();
            py::dict d;
            d["wins"] = comp.wins;
            d["draws"] = comp.draws;
            d["losses"] = comp.losses;
            return d;
        },
        "Get W/D/L composition of buffer contents.\n"
        "Returns dict with keys: wins, draws, losses.")
        // Prioritized Experience Replay (PER)
        .def("enable_per", &training::ReplayBuffer::enable_per,
             py::arg("alpha"),
             "Enable Prioritized Experience Replay.\n"
             "alpha: priority exponent (0=uniform, 0.6=recommended)")
        .def("per_enabled", &training::ReplayBuffer::per_enabled,
             "Check if PER is enabled")
        .def("priority_exponent", &training::ReplayBuffer::priority_exponent,
             "Get priority exponent (alpha)")
        .def("sample_prioritized", [](training::ReplayBuffer& self,
                                       size_t batch_size, float beta) {
            std::vector<float> observations, policies, values, wdl_targets, soft_values;
            std::vector<uint32_t> indices;
            std::vector<float> is_weights;

            bool success = self.sample_prioritized(
                batch_size, beta, observations, policies, values,
                &wdl_targets, &soft_values, indices, is_weights);
            if (!success) {
                throw std::runtime_error("PER sampling failed (not enabled or not enough samples)");
            }

            constexpr size_t OBS_FLAT = encoding::PositionEncoder::TOTAL_SIZE;  // 7872
            py::array_t<float> obs_array(std::vector<size_t>{batch_size, OBS_FLAT});
            py::array_t<float> pol_array(std::vector<size_t>{batch_size, 4672UL});
            py::array_t<float> val_array(std::vector<size_t>{batch_size});
            py::array_t<float> wdl_array(std::vector<size_t>{batch_size, 3UL});
            py::array_t<float> sv_array(std::vector<size_t>{batch_size});
            py::array_t<uint32_t> idx_array(std::vector<size_t>{batch_size});
            py::array_t<float> wt_array(std::vector<size_t>{batch_size});

            std::memcpy(obs_array.mutable_data(), observations.data(),
                       batch_size * OBS_FLAT * sizeof(float));
            std::memcpy(pol_array.mutable_data(), policies.data(),
                       batch_size * 4672 * sizeof(float));
            std::memcpy(val_array.mutable_data(), values.data(),
                       batch_size * sizeof(float));
            std::memcpy(wdl_array.mutable_data(), wdl_targets.data(),
                       batch_size * 3 * sizeof(float));
            std::memcpy(sv_array.mutable_data(), soft_values.data(),
                       batch_size * sizeof(float));
            std::memcpy(idx_array.mutable_data(), indices.data(),
                       batch_size * sizeof(uint32_t));
            std::memcpy(wt_array.mutable_data(), is_weights.data(),
                       batch_size * sizeof(float));

            return py::make_tuple(obs_array, pol_array, val_array, wdl_array,
                                  sv_array, idx_array, wt_array);
        },
        py::arg("batch_size"), py::arg("beta"),
        "Sample a prioritized batch with IS weights.\n"
        "Returns: (observations, policies, values, wdl_targets, soft_values, indices, is_weights)")
        .def("update_priorities", [](training::ReplayBuffer& self,
                                      py::array_t<uint32_t> indices,
                                      py::array_t<float> priorities) {
            auto idx_buf = indices.request();
            auto pri_buf = priorities.request();
            size_t n = idx_buf.shape[0];
            std::vector<uint32_t> idx_vec(static_cast<uint32_t*>(idx_buf.ptr),
                                           static_cast<uint32_t*>(idx_buf.ptr) + n);
            std::vector<float> pri_vec(static_cast<float*>(pri_buf.ptr),
                                        static_cast<float*>(pri_buf.ptr) + n);
            self.update_priorities(idx_vec, pri_vec);
        },
        py::arg("indices"), py::arg("priorities"),
        "Update priorities for sampled indices.\n"
        "indices: uint32 array from sample_prioritized\n"
        "priorities: float32 array of new priorities (typically loss + epsilon)")
        // FEN storage for MCTS reanalysis
        .def("enable_fen_storage", &training::ReplayBuffer::enable_fen_storage,
             "Enable FEN string storage for reanalysis.\n"
             "Allocates FEN + Zobrist hash buffers. Call before adding samples.")
        .def("fen_storage_enabled", &training::ReplayBuffer::fen_storage_enabled,
             "Check if FEN storage is enabled")
        .def("get_fen", &training::ReplayBuffer::get_fen,
             py::arg("index"),
             "Get FEN string at buffer index (empty if FEN storage disabled or index invalid)");

    // Trainer
    py::class_<training::Trainer::Config>(m, "TrainerConfig")
        .def(py::init<>())
        .def_readwrite("batch_size", &training::Trainer::Config::batch_size)
        .def_readwrite("min_buffer_size", &training::Trainer::Config::min_buffer_size)
        .def_readwrite("learning_rate", &training::Trainer::Config::learning_rate)
        .def_readwrite("num_epochs_per_iteration", &training::Trainer::Config::num_epochs_per_iteration)
        .def_readwrite("batches_per_epoch", &training::Trainer::Config::batches_per_epoch);

    py::class_<training::Trainer>(m, "Trainer")
        .def(py::init<const training::Trainer::Config&>(),
             py::arg("config") = training::Trainer::Config{},
             "Create trainer with configuration")
        .def("is_ready", &training::Trainer::is_ready,
             py::arg("buffer"),
             "Check if buffer is ready for training")
        .def("get_config", &training::Trainer::get_config,
             py::return_value_policy::reference,
             "Get training configuration")
        .def("get_stats", [](const training::Trainer& self) {
            auto stats = self.get_stats();
            py::dict d;
            d["total_steps"] = stats.total_steps;
            d["total_samples_trained"] = stats.total_samples_trained;
            d["last_loss"] = stats.last_loss;
            d["last_policy_loss"] = stats.last_policy_loss;
            d["last_value_loss"] = stats.last_value_loss;
            return d;
        },
        "Get training statistics")
        .def("record_step", &training::Trainer::record_step,
             py::arg("batch_size"),
             py::arg("loss"),
             py::arg("policy_loss"),
             py::arg("value_loss"),
             "Record training step statistics")
        .def("reset_stats", &training::Trainer::reset_stats,
             "Reset training statistics");

    // Version info
    m.attr("__version__") = "1.0.0";
}
