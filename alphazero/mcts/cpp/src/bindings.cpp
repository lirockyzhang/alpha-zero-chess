/**
 * pybind11 bindings for C++ MCTS implementation.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "node.hpp"
#include "search.hpp"

namespace py = pybind11;
using namespace alphazero::mcts;

/**
 * Python-friendly MCTS wrapper that handles numpy arrays and Python callbacks.
 */
class PyMCTS {
public:
    PyMCTS(py::object config = py::none()) {
        if (!config.is_none()) {
            config_.num_simulations = config.attr("num_simulations").cast<int>();
            config_.c_puct = config.attr("c_puct").cast<float>();
            config_.dirichlet_alpha = config.attr("dirichlet_alpha").cast<float>();
            config_.dirichlet_epsilon = config.attr("dirichlet_epsilon").cast<float>();
            config_.temperature = config.attr("temperature").cast<float>();
            config_.temperature_threshold = config.attr("temperature_threshold").cast<int>();
        }
        mcts_ = std::make_unique<MCTS>(config_);
    }

    float get_temperature(int move_number) const {
        return mcts_->get_temperature(move_number);
    }

    py::array_t<float> apply_temperature(py::array_t<float> visit_counts, float temperature) const {
        auto buf = visit_counts.request();
        std::vector<float> counts(static_cast<float*>(buf.ptr),
                                  static_cast<float*>(buf.ptr) + buf.size);

        auto result = mcts_->apply_temperature(counts, temperature);

        return py::array_t<float>(result.size(), result.data());
    }

    py::array_t<float> add_dirichlet_noise(py::array_t<float> priors,
                                           py::array_t<float> legal_mask) {
        auto priors_buf = priors.request();
        auto mask_buf = legal_mask.request();

        std::vector<float> priors_vec(static_cast<float*>(priors_buf.ptr),
                                      static_cast<float*>(priors_buf.ptr) + priors_buf.size);
        std::vector<float> mask_vec(static_cast<float*>(mask_buf.ptr),
                                    static_cast<float*>(mask_buf.ptr) + mask_buf.size);

        auto result = mcts_->add_dirichlet_noise(priors_vec, mask_vec);

        return py::array_t<float>(result.size(), result.data());
    }

    /**
     * Run MCTS search with Python state and evaluator.
     */
    py::tuple search(py::object state, py::object evaluator,
                     int move_number = 0, bool add_noise = true) {
        // Get initial evaluation
        py::array_t<float> obs = state.attr("get_observation")().cast<py::array_t<float>>();
        py::array_t<float> legal = state.attr("get_legal_actions")().cast<py::array_t<float>>();

        py::tuple eval_result = evaluator.attr("evaluate")(obs, legal).cast<py::tuple>();
        py::array_t<float> priors_arr = eval_result[0].cast<py::array_t<float>>();
        float initial_value = eval_result[1].cast<float>();

        auto priors_buf = priors_arr.request();
        auto legal_buf = legal.request();

        std::vector<float> priors(static_cast<float*>(priors_buf.ptr),
                                  static_cast<float*>(priors_buf.ptr) + priors_buf.size);
        std::vector<float> legal_mask(static_cast<float*>(legal_buf.ptr),
                                      static_cast<float*>(legal_buf.ptr) + legal_buf.size);

        // Create root node
        auto root = std::make_shared<MCTSNode>(1.0f);

        // Add Dirichlet noise
        if (add_noise) {
            priors = mcts_->add_dirichlet_noise(priors, legal_mask);
        }

        // Expand root
        root->expand(priors, legal_mask);
        root->update(initial_value);

        MCTSStats stats;
        stats.nodes_created = 1;

        // Run simulations
        for (int sim = 0; sim < config_.num_simulations; ++sim) {
            int depth = simulate(root, state, evaluator, stats);
            stats.max_depth = std::max(stats.max_depth, depth);
        }

        stats.num_simulations = config_.num_simulations;
        stats.root_value = root->q_value();

        // Get policy
        float temperature = mcts_->get_temperature(move_number);
        std::vector<float> policy = root->get_policy(4672, temperature);

        // Convert to numpy array
        py::array_t<float> policy_arr(policy.size(), policy.data());

        // Create stats object
        py::object stats_class = py::module::import("alphazero.mcts.base").attr("MCTSStats");
        py::object py_stats = stats_class();
        py_stats.attr("num_simulations") = stats.num_simulations;
        py_stats.attr("max_depth") = stats.max_depth;
        py_stats.attr("root_value") = stats.root_value;
        py_stats.attr("nodes_created") = stats.nodes_created;

        return py::make_tuple(policy_arr, root, py_stats);
    }

private:
    int simulate(std::shared_ptr<MCTSNode> root, py::object root_state,
                 py::object evaluator, MCTSStats& stats) {
        auto node = root;
        py::object state = root_state;
        std::vector<std::pair<std::shared_ptr<MCTSNode>, int>> path;
        int depth = 0;

        // Selection
        while (node->is_expanded() && !node->is_terminal()) {
            auto [action, child] = node->select_child(config_.c_puct);
            path.push_back({node, action});
            node = child;
            state = state.attr("apply_action")(action);
            depth++;
        }

        double value;

        // Check for terminal
        if (state.attr("is_terminal")().cast<bool>()) {
            value = state.attr("get_value")().cast<double>();
            node->set_terminal(value);
        } else if (!node->is_expanded()) {
            // Expansion and evaluation
            py::array_t<float> obs = state.attr("get_observation")().cast<py::array_t<float>>();
            py::array_t<float> legal = state.attr("get_legal_actions")().cast<py::array_t<float>>();

            py::tuple eval_result = evaluator.attr("evaluate")(obs, legal).cast<py::tuple>();
            py::array_t<float> priors_arr = eval_result[0].cast<py::array_t<float>>();
            value = eval_result[1].cast<double>();

            auto priors_buf = priors_arr.request();
            auto legal_buf = legal.request();

            std::vector<float> priors(static_cast<float*>(priors_buf.ptr),
                                      static_cast<float*>(priors_buf.ptr) + priors_buf.size);
            std::vector<float> legal_mask(static_cast<float*>(legal_buf.ptr),
                                          static_cast<float*>(legal_buf.ptr) + legal_buf.size);

            node->expand(priors, legal_mask);
            stats.nodes_created++;
        } else {
            value = node->terminal_value();
        }

        // Backpropagation
        node->update(value);
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            value = -value;
            it->first->update(value);
        }

        return depth;
    }

    MCTSConfig config_;
    std::unique_ptr<MCTS> mcts_;
};


PYBIND11_MODULE(mcts_cpp, m) {
    m.doc() = "C++ MCTS implementation for AlphaZero";

    py::class_<MCTSNode, std::shared_ptr<MCTSNode>>(m, "CppMCTSNode")
        .def(py::init<float>(), py::arg("prior") = 1.0f)
        .def_property_readonly("visit_count", &MCTSNode::visit_count)
        .def_property_readonly("value_sum", &MCTSNode::value_sum)
        .def_property_readonly("prior", &MCTSNode::prior)
        .def_property_readonly("q_value", &MCTSNode::q_value)
        .def("is_expanded", &MCTSNode::is_expanded)
        .def("is_terminal", &MCTSNode::is_terminal)
        .def("get_terminal_value", &MCTSNode::terminal_value)
        .def("set_terminal", &MCTSNode::set_terminal)
        .def("expand", [](MCTSNode& self, py::array_t<float> priors, py::array_t<float> legal_mask) {
            auto priors_buf = priors.request();
            auto mask_buf = legal_mask.request();
            std::vector<float> priors_vec(static_cast<float*>(priors_buf.ptr),
                                          static_cast<float*>(priors_buf.ptr) + priors_buf.size);
            std::vector<float> mask_vec(static_cast<float*>(mask_buf.ptr),
                                        static_cast<float*>(mask_buf.ptr) + mask_buf.size);
            self.expand(priors_vec, mask_vec);
        })
        .def("select_child", &MCTSNode::select_child)
        .def("get_child", &MCTSNode::get_child)
        .def("get_children", &MCTSNode::children)
        .def("update", &MCTSNode::update)
        .def("get_visit_counts", [](const MCTSNode& self, int num_actions) {
            auto counts = self.get_visit_counts(num_actions);
            return py::array_t<float>(counts.size(), counts.data());
        })
        .def("get_policy", [](const MCTSNode& self, int num_actions, float temperature) {
            auto policy = self.get_policy(num_actions, temperature);
            return py::array_t<float>(policy.size(), policy.data());
        }, py::arg("num_actions"), py::arg("temperature") = 1.0f);

    py::class_<PyMCTS>(m, "CppMCTS")
        .def(py::init<py::object>(), py::arg("config") = py::none())
        .def("get_temperature", &PyMCTS::get_temperature)
        .def("apply_temperature", &PyMCTS::apply_temperature)
        .def("add_dirichlet_noise", &PyMCTS::add_dirichlet_noise)
        .def("search", &PyMCTS::search,
             py::arg("state"), py::arg("evaluator"),
             py::arg("move_number") = 0, py::arg("add_noise") = true);
}
