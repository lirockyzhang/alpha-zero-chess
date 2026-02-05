#pragma once

#include "node.hpp"
#include <memory>
#include <vector>
#include <cstdlib>

namespace mcts {

// NodePool: Chained arena allocator for MCTS nodes
// Provides O(1) allocation without fragmentation
// Uses 64-byte aligned allocation to prevent false sharing
class NodePool {
public:
    // Configuration
    static constexpr size_t NODES_PER_BLOCK = 16384;  // 16K nodes per block = 1MB
    static constexpr size_t MAX_BLOCKS = 512;          // Max 512 blocks = 512MB per worker

    // Safety checks at compile time
    static_assert(NODES_PER_BLOCK * sizeof(Node) < (1ULL << 32),
                  "Block size must fit in 32-bit address space");
    static_assert(MAX_BLOCKS * NODES_PER_BLOCK <= (1ULL << 32),
                  "Total pool size must be addressable with 32-bit indices");

    NodePool() : current_block_(0), current_offset_(0) {
        allocate_block();
    }

    ~NodePool() {
        // Free all allocated blocks
        for (auto* block : blocks_) {
#ifdef _WIN32
            _aligned_free(block);  // Windows: use _aligned_free for _aligned_malloc
#else
            std::free(block);      // POSIX: use free for posix_memalign
#endif
        }
    }

    // Allocate a new node (O(1) operation)
    Node* allocate() {
        // Check if current block is full
        if (current_offset_ >= NODES_PER_BLOCK) {
            allocate_block();
        }

        // Get node from current block
        Node* node = &blocks_[current_block_][current_offset_++];

        // Initialize node (placement new to call constructor)
        new (node) Node();

        return node;
    }

    // Reset pool (clears all nodes, keeps allocated memory)
    void reset() {
        current_block_ = 0;
        current_offset_ = 0;
    }

    // Get total number of allocated nodes
    size_t size() const {
        return current_block_ * NODES_PER_BLOCK + current_offset_;
    }

    // Get memory usage in bytes
    size_t memory_usage() const {
        return blocks_.size() * NODES_PER_BLOCK * sizeof(Node);
    }

private:
    void allocate_block() {
        if (blocks_.size() >= MAX_BLOCKS) {
            throw std::runtime_error("NodePool: Maximum number of blocks reached");
        }

        // Allocate 64-byte aligned block
        void* ptr = nullptr;

#ifdef _WIN32
        // Windows: use _aligned_malloc
        ptr = _aligned_malloc(NODES_PER_BLOCK * sizeof(Node), 64);
        if (!ptr) {
            throw std::bad_alloc();
        }
#else
        // POSIX: use posix_memalign
        if (posix_memalign(&ptr, 64, NODES_PER_BLOCK * sizeof(Node)) != 0) {
            throw std::bad_alloc();
        }
#endif

        blocks_.push_back(static_cast<Node*>(ptr));
        current_block_ = blocks_.size() - 1;
        current_offset_ = 0;
    }

    std::vector<Node*> blocks_;
    size_t current_block_;
    size_t current_offset_;
};

} // namespace mcts
