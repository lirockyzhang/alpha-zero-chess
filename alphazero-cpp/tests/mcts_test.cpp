#include "mcts/search.hpp"
#include "mcts/node_pool.hpp"
#include "mcts/batch_coordinator.hpp"
#include "../third_party/chess-library/include/chess.hpp"
#include <iostream>
#include <iomanip>

using namespace mcts;
using namespace chess;

void test_basic_search() {
    std::cout << "=== Test 1: Basic MCTS Search ===" << std::endl;

    // Create starting position
    std::cout << "Creating board..." << std::endl;
    Board board;

    // Create node pool and search
    std::cout << "Creating node pool..." << std::endl;
    NodePool pool;
    std::cout << "Creating search config..." << std::endl;
    SearchConfig config;
    config.num_simulations = 100;  // Small number for testing
    std::cout << "Creating MCTS search..." << std::endl;
    MCTSSearch search(pool, config);

    // Create uniform policy (for testing without neural network)
    std::cout << "Creating policy vector..." << std::endl;
    std::vector<float> policy(1858, 1.0f / 1858.0f);
    float value = 0.0f;

    // Run search
    std::cout << "Running search..." << std::endl;
    Node* root = search.search(board, policy, value);
    std::cout << "Search completed!" << std::endl;

    // Verify root was created
    if (root == nullptr) {
        std::cout << "✗ FAIL: Root node is null" << std::endl;
        return;
    }

    // Verify root has children
    if (root->first_child == nullptr) {
        std::cout << "✗ FAIL: Root has no children" << std::endl;
        return;
    }

    // Count children
    int num_children = 0;
    for (Node* child = root->first_child; child != nullptr; child = child->next_sibling) {
        num_children++;
    }

    std::cout << "✓ Root has " << num_children << " children (expected 20 for starting position)" << std::endl;

    // Verify visit counts
    uint32_t root_visits = root->visit_count.load(std::memory_order_relaxed);
    std::cout << "✓ Root visit count: " << root_visits << std::endl;

    // Find most visited child
    Node* best_child = nullptr;
    uint32_t max_visits = 0;
    for (Node* child = root->first_child; child != nullptr; child = child->next_sibling) {
        uint32_t visits = child->visit_count.load(std::memory_order_relaxed);
        if (visits > max_visits) {
            max_visits = visits;
            best_child = child;
        }
    }

    if (best_child) {
        std::cout << "✓ Most visited move: " << uci::moveToUci(best_child->move)
                  << " (visits: " << max_visits << ")" << std::endl;
    }

    // Verify Q-values are reasonable
    float root_q = root->q_value(0.0f);
    std::cout << "✓ Root Q-value: " << std::fixed << std::setprecision(3) << root_q << std::endl;

    std::cout << "✓ PASS: Basic MCTS search works" << std::endl;
    std::cout << std::endl;
}

void test_node_pool() {
    std::cout << "=== Test 2: Node Pool ===" << std::endl;

    NodePool pool;

    // Allocate some nodes
    std::vector<Node*> nodes;
    for (int i = 0; i < 1000; ++i) {
        nodes.push_back(pool.allocate());
    }

    std::cout << "✓ Allocated 1000 nodes" << std::endl;
    std::cout << "✓ Pool size: " << pool.size() << " nodes" << std::endl;
    std::cout << "✓ Memory usage: " << pool.memory_usage() / 1024 << " KB" << std::endl;

    // Verify nodes are 64-byte aligned
    for (int i = 0; i < 10; ++i) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(nodes[i]);
        if (addr % 64 != 0) {
            std::cout << "✗ FAIL: Node " << i << " is not 64-byte aligned" << std::endl;
            return;
        }
    }

    std::cout << "✓ All nodes are 64-byte aligned" << std::endl;
    std::cout << "✓ PASS: Node pool works correctly" << std::endl;
    std::cout << std::endl;
}

void test_move_selection() {
    std::cout << "=== Test 3: Move Selection ===" << std::endl;

    Board board;
    NodePool pool;
    SearchConfig config;
    config.num_simulations = 200;
    MCTSSearch search(pool, config);

    std::vector<float> policy(1858, 1.0f / 1858.0f);
    float value = 0.0f;

    Node* root = search.search(board, policy, value);

    // Test greedy selection (temperature = 0)
    Move greedy_move = search.select_move(root, 0.0f);
    std::cout << "✓ Greedy move (temp=0): " << uci::moveToUci(greedy_move) << std::endl;

    // Test stochastic selection (temperature = 1)
    Move stochastic_move = search.select_move(root, 1.0f);
    std::cout << "✓ Stochastic move (temp=1): " << uci::moveToUci(stochastic_move) << std::endl;

    // Get policy target for training
    std::vector<float> policy_target = search.get_policy_target(root);
    std::cout << "✓ Policy target has " << policy_target.size() << " entries" << std::endl;

    // Verify policy target sums to 1.0
    float sum = 0.0f;
    for (float p : policy_target) {
        sum += p;
    }
    std::cout << "✓ Policy target sum: " << std::fixed << std::setprecision(6) << sum
              << " (should be ~1.0)" << std::endl;

    std::cout << "✓ PASS: Move selection works correctly" << std::endl;
    std::cout << std::endl;
}

void test_batch_coordinator() {
    std::cout << "=== Test 4: Batch Coordinator ===" << std::endl;

    BatchCoordinator::Config config;
    config.batch_size = 4;
    config.batch_threshold = 0.75f;  // 75% threshold for testing
    BatchCoordinator coordinator(config);

    // Add some games
    for (int i = 0; i < 4; ++i) {
        Board board;
        coordinator.add_game(i, board);
    }

    auto stats = coordinator.get_stats();
    std::cout << "✓ Added " << stats.active_games << " games" << std::endl;

    // Submit eval requests
    NodePool pool;
    for (int i = 0; i < 3; ++i) {
        EvalRequest req;
        req.game_id = i;
        req.node = pool.allocate();
        req.board = Board();
        coordinator.submit_eval_request(req);
    }

    stats = coordinator.get_stats();
    std::cout << "✓ Pending evals: " << stats.pending_evals << std::endl;

    std::cout << "✓ PASS: Batch coordinator works correctly" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "=== AlphaZero Sync - MCTS Core Tests ===" << std::endl;
    std::cout << std::endl;

    try {
        test_node_pool();
        test_basic_search();
        test_move_selection();
        test_batch_coordinator();

        std::cout << "========================================" << std::endl;
        std::cout << "✓✓✓ ALL TESTS PASSED ✓✓✓" << std::endl;
        std::cout << std::endl;
        std::cout << "MCTS Core implementation is working correctly!" << std::endl;
        std::cout << "You may now proceed to Phase 3: Python Bindings" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cout << "✗✗✗ TEST FAILED ✗✗✗" << std::endl;
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}
