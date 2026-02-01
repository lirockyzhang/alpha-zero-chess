#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "mcts/search.hpp"
#include "mcts/batch_search.hpp"
#include "mcts/node_pool.hpp"
#include "mcts/batch_coordinator.hpp"
#include "encoding/position_encoder.hpp"
#include "encoding/move_encoder.hpp"
#include "../third_party/chess-library/include/chess.hpp"

namespace py = pybind11;

// Python wrapper for MCTS search
class PyMCTSSearch {
public:
    PyMCTSSearch(int num_simulations = 800, float c_puct = 1.5f)
        : pool_()
        , search_(pool_, create_config(num_simulations, c_puct))
    {}

    // Run MCTS search and return visit counts
    py::array_t<int32_t> search(const std::string& fen,
                                 py::array_t<float> policy,
                                 float value) {
        // Parse FEN position
        chess::Board board(fen);

        // Convert numpy array to vector
        auto policy_buf = policy.request();
        if (policy_buf.ndim != 1 || policy_buf.shape[0] != encoding::MoveEncoder::POLICY_SIZE) {
            throw std::runtime_error("Policy must be 1D array of size " + std::to_string(encoding::MoveEncoder::POLICY_SIZE));
        }
        std::vector<float> policy_vec(static_cast<float*>(policy_buf.ptr),
                                      static_cast<float*>(policy_buf.ptr) + encoding::MoveEncoder::POLICY_SIZE);

        // Run MCTS search
        mcts::Node* root = search_.search(board, policy_vec, value);

        // Extract visit counts for all legal moves
        std::vector<int32_t> visit_counts(encoding::MoveEncoder::POLICY_SIZE, 0);

        // Map each child move to policy index and store visit count
        for (mcts::Node* child = root->first_child; child != nullptr;
             child = child->next_sibling) {
            // Get visit count from child node
            uint32_t visits = child->visit_count.load(std::memory_order_relaxed);

            // Convert move to policy index
            int move_index = encoding::MoveEncoder::move_to_index(child->move, board);

            // Store visit count at the corresponding policy index
            if (move_index >= 0 && move_index < encoding::MoveEncoder::POLICY_SIZE) {
                visit_counts[move_index] = static_cast<int32_t>(visits);
            }
        }

        // Return as numpy array
        return py::array_t<int32_t>(visit_counts.size(), visit_counts.data());
    }

    // Select best move using temperature
    std::string select_move(float temperature = 0.0f) {
        // TODO: Implement move selection
        return "e2e4";
    }

    // Reset search tree
    void reset() {
        pool_.reset();
    }

private:
    static mcts::SearchConfig create_config(int num_simulations, float c_puct) {
        mcts::SearchConfig config;
        config.num_simulations = num_simulations;
        config.c_puct = c_puct;
        config.dirichlet_alpha = 0.3f;
        config.dirichlet_epsilon = 0.25f;
        return config;
    }

    mcts::NodePool pool_;
    mcts::MCTSSearch search_;
};

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
// NHWC layout: (height, width, channels) = (8, 8, 119)
py::array_t<float> encode_position(const std::string& fen) {
    chess::Board board(fen);

    // Use the real position encoder
    std::vector<float> encoding = encoding::PositionEncoder::encode(board);

    // Return as numpy array with NHWC shape (8, 8, 119) - channels last
    return py::array_t<float>(
        {encoding::PositionEncoder::HEIGHT,
         encoding::PositionEncoder::WIDTH,
         encoding::PositionEncoder::CHANNELS},
        encoding.data()
    );
}

// Zero-copy position encoding - writes directly to provided buffer
// NHWC layout: (height, width, channels) = (8, 8, 119)
void encode_position_to_buffer(const std::string& fen, py::array_t<float> buffer) {
    chess::Board board(fen);

    // Verify buffer shape and size for NHWC layout
    auto buf_info = buffer.request();
    if (buf_info.ndim != 3 ||
        buf_info.shape[0] != encoding::PositionEncoder::HEIGHT ||
        buf_info.shape[1] != encoding::PositionEncoder::WIDTH ||
        buf_info.shape[2] != encoding::PositionEncoder::CHANNELS) {
        throw std::runtime_error("Buffer must have NHWC shape (8, 8, 119)");
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
        throw std::runtime_error("Buffer must have NHWC shape (batch_size, 8, 8, 119)");
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
    PyBatchedMCTSSearch(int num_simulations = 800, int batch_size = 256, float c_puct = 1.5f)
        : pool_()
        , search_(pool_, create_config(num_simulations, batch_size, c_puct))
        , batch_size_(batch_size)
    {}

    // Initialize search with root position and initial NN evaluation
    void init_search(const std::string& fen,
                     py::array_t<float> root_policy,
                     float root_value) {
        chess::Board board(fen);
        root_fen_ = fen;

        // Convert numpy array to vector
        auto policy_buf = root_policy.request();
        if (policy_buf.ndim != 1 || policy_buf.shape[0] != encoding::MoveEncoder::POLICY_SIZE) {
            throw std::runtime_error("Policy must be 1D array of size " + std::to_string(encoding::MoveEncoder::POLICY_SIZE));
        }
        std::vector<float> policy_vec(static_cast<float*>(policy_buf.ptr),
                                      static_cast<float*>(policy_buf.ptr) + encoding::MoveEncoder::POLICY_SIZE);

        search_.init_search(board, policy_vec, root_value);
    }

    // Collect leaves that need NN evaluation
    // Returns (num_leaves, observations, legal_masks)
    py::tuple collect_leaves() {
        // Allocate buffers for observations and masks
        std::vector<float> obs_buffer(batch_size_ * 8 * 8 * 119, 0.0f);
        std::vector<float> mask_buffer(batch_size_ * encoding::MoveEncoder::POLICY_SIZE, 0.0f);

        int num_leaves = search_.collect_leaves(obs_buffer.data(), mask_buffer.data(), batch_size_);

        if (num_leaves == 0) {
            // Return empty arrays
            return py::make_tuple(
                0,
                py::array_t<float>({0, 8, 8, 119}),
                py::array_t<float>({0, encoding::MoveEncoder::POLICY_SIZE})
            );
        }

        // Create numpy arrays with actual size
        py::array_t<float> obs_array({num_leaves, 8, 8, 119});
        py::array_t<float> mask_array({num_leaves, encoding::MoveEncoder::POLICY_SIZE});

        // Copy data
        std::memcpy(obs_array.mutable_data(), obs_buffer.data(),
                    num_leaves * 8 * 8 * 119 * sizeof(float));
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

    // Reset for new search
    void reset() {
        pool_.reset();
    }

private:
    static mcts::BatchSearchConfig create_config(int num_simulations, int batch_size, float c_puct) {
        mcts::BatchSearchConfig config;
        config.num_simulations = num_simulations;
        config.batch_size = batch_size;
        config.c_puct = c_puct;
        config.dirichlet_alpha = 0.3f;
        config.dirichlet_epsilon = 0.25f;
        return config;
    }

    mcts::NodePool pool_;
    mcts::BatchedMCTSSearch search_;
    int batch_size_;
    std::string root_fen_;
};

PYBIND11_MODULE(alphazero_cpp, m) {
    m.doc() = "AlphaZero C++ - High-performance MCTS for chess";

    // MCTS Search class
    py::class_<PyMCTSSearch>(m, "MCTSSearch")
        .def(py::init<int, float>(),
             py::arg("num_simulations") = 800,
             py::arg("c_puct") = 1.5f,
             "Create MCTS search engine")
        .def("search", &PyMCTSSearch::search,
             py::arg("fen"),
             py::arg("policy"),
             py::arg("value"),
             "Run MCTS search and return visit counts")
        .def("select_move", &PyMCTSSearch::select_move,
             py::arg("temperature") = 0.0f,
             "Select best move using temperature")
        .def("reset", &PyMCTSSearch::reset,
             "Reset search tree");

    // Batch Coordinator class
    py::class_<PyBatchCoordinator>(m, "BatchCoordinator")
        .def(py::init<int, float>(),
             py::arg("batch_size") = 256,
             py::arg("batch_threshold") = 0.9f,
             "Create batch coordinator")
        .def("add_game", &PyBatchCoordinator::add_game,
             py::arg("game_id"),
             py::arg("fen"),
             "Add game to batch")
        .def("is_game_complete", &PyBatchCoordinator::is_game_complete,
             py::arg("game_id"),
             "Check if game is complete")
        .def("remove_game", &PyBatchCoordinator::remove_game,
             py::arg("game_id"),
             "Remove completed game")
        .def("get_stats", &PyBatchCoordinator::get_stats,
             "Get coordinator statistics");

    // Batched MCTS Search class (proper AlphaZero with batch leaf evaluation)
    py::class_<PyBatchedMCTSSearch>(m, "BatchedMCTSSearch")
        .def(py::init<int, int, float>(),
             py::arg("num_simulations") = 800,
             py::arg("batch_size") = 256,
             py::arg("c_puct") = 1.5f,
             "Create batched MCTS search engine with proper leaf evaluation")
        .def("init_search", &PyBatchedMCTSSearch::init_search,
             py::arg("fen"),
             py::arg("root_policy"),
             py::arg("root_value"),
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
        .def("reset", &PyBatchedMCTSSearch::reset,
             "Reset search tree for new search");

    // Utility functions
    m.def("encode_position", &encode_position,
          py::arg("fen"),
          "Encode chess position for neural network input");

    m.def("encode_position_to_buffer", &encode_position_to_buffer,
          py::arg("fen"),
          py::arg("buffer"),
          "Encode chess position directly to provided buffer (zero-copy)");

    m.def("encode_batch", &encode_batch,
          py::arg("fens"),
          py::arg("buffer"),
          py::arg("use_parallel") = true,
          "Encode multiple positions in parallel (batch encoding with OpenMP)");

    m.def("move_to_index", &move_to_index,
          py::arg("uci_move"),
          py::arg("fen"),
          "Convert UCI move to policy index");

    m.def("index_to_move", &index_to_move,
          py::arg("index"),
          py::arg("fen"),
          "Convert policy index to UCI move");

    // Version info
    m.attr("__version__") = "1.0.0";
}
