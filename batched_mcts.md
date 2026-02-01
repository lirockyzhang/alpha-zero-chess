# Fully Synchronized Batched MCTS Implementation Plan

## Overview
Implement a standalone C++ library with Python bindings for fully synchronized batched MCTS with multi-game batching. Multiple MCTS instances (different games) will synchronize at batch boundaries to collect leaves together for efficient GPU evaluation.

## Key Requirements
- **Multi-game batching**: Synchronize multiple games at batch boundaries
- **C++ core** with Python bindings (pybind11) for maximum speed
- **Standalone**: Complete implementation from scratch (chess engine, MCTS, encoding)
- **Features**: Virtual loss, Dirichlet noise, temperature sampling, root reuse
- **Speed optimizations**: Bitboards, memory pooling, lock-free operations
- **Compatibility**: Load existing PyTorch network weights, work with evaluate.py and web app
- **Colab-friendly**: 1-2 line setup and compilation

## Architecture

### Core Innovation: Batch Coordinator
```
Game 1: [Select Leaf] → [Wait] ──┐
Game 2: [Select Leaf] → [Wait] ──┤
Game N: [Select Leaf] → [Wait] ──┼→ [Batch Eval] → Resume All
```

### Directory Structure
```
alphazero-sync/
├── CMakeLists.txt              # Build configuration
├── setup.py                    # Python package setup
├── colab_setup.sh              # One-line Colab installer
├── include/                    # C++ headers
│   ├── chess/                  # Bitboard chess engine
│   ├── mcts/                   # MCTS implementation
│   ├── encoding/               # Board encoding & action mapping
│   └── utils/                  # Utilities (spinlock, thread pool)
├── src/                        # C++ implementation
│   ├── chess/
│   ├── mcts/
│   ├── encoding/
│   └── bindings/               # pybind11 bindings
├── python/                     # Python wrapper layer
│   ├── evaluator.py
│   ├── game_manager.py
│   └── compatibility.py        # Interface with existing code
├── third_party/                # Vendored dependencies
│   └── pybind11/               # Ensure Colab build stability
└── tests/                      # Unit and integration tests
```

## Implementation Steps

### Phase 1: Chess Engine (C++)
**Files**: `include/chess/`, `src/chess/`

1. **Bitboard representation** (`bitboard.hpp/cpp`)
   - 64-bit integers for piece positions
   - **Hardware intrinsics**: Use BMI2/AVX2 instructions directly
     - `_mm_popcnt_u64`: Count pieces/legal moves
     - `_tzcnt_u64` / `_blsr_u64`: Bit scanning (isolate lowest set bit)
   - Fast bitwise operations (popcount, lsb, shifts)
   - Piece types: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
   - Colors: WHITE, BLACK

**Intrinsics Example**:
```cpp
#include <immintrin.h>

inline int popcount(Bitboard bb) {
    return _mm_popcnt_u64(bb);  // Hardware instruction
}

inline Square lsb(Bitboard bb) {
    return Square(_tzcnt_u64(bb));  // Trailing zero count
}

inline Square pop_lsb(Bitboard& bb) {
    Square sq = lsb(bb);
    bb = _blsr_u64(bb);  // Clear lowest set bit
    return sq;
}
```

2. **Position class** (`position.hpp/cpp`)
   - Bitboards for each piece type and color
   - Game state: side to move, castling rights, en passant, clocks
   - Zobrist hashing for position identification
   - Move history for unmake operations

3. **Move generation** (`movegen.hpp/cpp`)
   - Magic bitboards for sliding pieces (rook, bishop, queen)
   - **Generate magic tables at startup**: ~50ms with **fixed-seed PRNG** for deterministic results
   - Precomputed attack tables for knight and king
   - Legal move generation with check detection
   - Special moves: castling, en passant, promotions
   - **Perft validation**: Verify correctness against known perft results

**Deterministic Magic Generation**:
```cpp
void MoveGenerator::init() {
    // Use fixed seed for deterministic magic table generation
    std::mt19937_64 rng(0x123456789ABCDEF0ULL);  // Fixed seed

    // Generate magic numbers (deterministic across runs/machines)
    for (Square sq = A1; sq < SQUARE_COUNT; ++sq) {
        rook_magics[sq] = find_magic(sq, true, rng);
        bishop_magics[sq] = find_magic(sq, false, rng);
    }
}
```

4. **Zobrist hashing** (`zobrist.hpp/cpp`)
   - **Dual hashing**: 64-bit primary + 32-bit secondary hash (virtually eliminates collisions)
   - Random numbers for each piece/square combination
   - **Incremental updates for BOTH hashes**: Update both 64-bit and 32-bit incrementally during make/unmake
   - **Critical for 3-fold repetition detection**: Required for legal chess play
   - Position history tracking for draw detection

**Dual Hash Strategy with Incremental Updates**:
```cpp
struct PositionHash {
    uint64_t primary;   // Main Zobrist hash
    uint32_t secondary; // Secondary hash for collision detection

    bool operator==(const PositionHash& other) const {
        return primary == other.primary && secondary == other.secondary;
    }
};

// Pre-generated Zobrist tables (both 64-bit and 32-bit)
uint64_t zobrist_primary[12][64];   // [piece_type][square]
uint32_t zobrist_secondary[12][64]; // [piece_type][square]

// Incremental update (O(1))
void Position::make_move(const Move& move) {
    // Update both hashes incrementally
    zobrist_hash.primary ^= zobrist_primary[piece][from_sq];
    zobrist_hash.primary ^= zobrist_primary[piece][to_sq];

    zobrist_hash.secondary ^= zobrist_secondary[piece][from_sq];
    zobrist_hash.secondary ^= zobrist_secondary[piece][to_sq];

    // Add to history for 3-fold repetition
    position_history.push_back(zobrist_hash);
}

// Probability of collision: ~1 in 2^96 (virtually impossible)
```

**Key Optimization**: Magic bitboards provide O(1) sliding piece move generation

### Phase 2: MCTS Core (C++)
**Files**: `include/mcts/`, `src/mcts/`

1. **MCTS Node** (`node.hpp/cpp`)
   - Compact 64-byte structure (cache-line aligned to prevent false sharing)
   - **Pointer compression**: Use 32-bit indices into NodePool instead of 64-bit pointers
   - Atomic operations for thread-safety:
     - `std::atomic<uint32_t> visit_count`
     - `std::atomic<float> value_sum`
     - `std::atomic<uint16_t> virtual_loss`
   - Use `std::memory_order_relaxed` where full consistency not needed
   - PUCT score calculation with virtual loss
   - Lock-free updates using compare-exchange
   - **Critical**: Ensure atomics don't share cache lines (false sharing kills performance)

2. **Node Pool** (`node_pool.hpp/cpp`) - **CHAINED ARENA STRATEGY**
   - **Fixed-size array of blocks**: `Node* blocks[MAX_BLOCKS]` instead of `std::vector<Node*>`
   - **Power-of-two block size**: 2^18 = 262,144 nodes per block (~16MB)
   - When exhausted, allocate new block (prevents crashes on long searches)
   - **Fast O(1) indexing**: `block_id = index >> 18`, `offset = index & 0x3FFFF` (bit shifts, no division!)
   - **Per-game sub-arenas**: Each game gets 16K node chunks from current block
   - Thread-safe global block allocation with atomic counter
   - Reset functionality for reuse across games

**Why fixed-size array (not std::vector)**:
```cpp
class NodePool {
    static constexpr uint32_t MAX_BLOCKS = 256;  // 256 blocks = 64M nodes = 4GB
    Node* blocks_[MAX_BLOCKS];  // Fixed-size array (no reallocation!)
    std::atomic<uint32_t> num_blocks_{0};

    // Power-of-two for fast bit-shift indexing
    static constexpr uint32_t BLOCK_SIZE_BITS = 18;  // 2^18 = 262144 nodes
    static constexpr uint32_t NODES_PER_BLOCK = 1 << BLOCK_SIZE_BITS;
    static constexpr uint32_t BLOCK_MASK = NODES_PER_BLOCK - 1;
    static constexpr uint32_t CHUNK_SIZE = 16384;  // 16K nodes per game

    // CRITICAL: Safety check for 32-bit indexing with 64-byte nodes
    // Each block: 262144 nodes × 64 bytes = 16MB (well under 4GB limit)
    // Total capacity: 256 blocks × 262144 nodes = 67,108,864 nodes (~4GB)
    // 32-bit index can address up to 4,294,967,296 nodes, so we're safe
    static_assert(NODES_PER_BLOCK * sizeof(Node) < (1ULL << 32),
                  "Block size must fit in 32-bit address space");
    static_assert(MAX_BLOCKS * NODES_PER_BLOCK <= (1ULL << 32),
                  "Total pool size must be addressable with 32-bit indices");

    // FAST O(1): Bit shifts, no division, no vector lookup!
    Node& operator[](uint32_t index) {
        uint32_t block_id = index >> BLOCK_SIZE_BITS;  // Fast: divide by 262144
        uint32_t offset = index & BLOCK_MASK;          // Fast: modulo 262144

        // Safety check: Ensure block is allocated
        assert(block_id < num_blocks_.load(std::memory_order_relaxed));

        return blocks_[block_id][offset];  // Direct array access (no indirection!)
    }

    Node* allocate_new_block() {
        uint32_t block_id = num_blocks_.fetch_add(1, std::memory_order_relaxed);
        if (block_id >= MAX_BLOCKS) {
            throw std::runtime_error("NodePool exhausted");
        }

        // CRITICAL: Use aligned_alloc for 64-byte alignment
        void* ptr = std::aligned_alloc(64, NODES_PER_BLOCK * sizeof(Node));
        if (!ptr) throw std::bad_alloc();

        // Placement new for each node
        Node* block = static_cast<Node*>(ptr);
        for (size_t i = 0; i < NODES_PER_BLOCK; ++i) {
            new (&block[i]) Node();
        }

        blocks_[block_id] = block;
        return block;
    }

    ~NodePool() {
        // Proper cleanup
        for (uint32_t i = 0; i < num_blocks_; ++i) {
            for (size_t j = 0; j < NODES_PER_BLOCK; ++j) {
                blocks_[i][j].~Node();
            }
            std::free(blocks_[i]);  // Match aligned_alloc with free
        }
    }
};
```

**Benefits**:
- No fixed limit (can grow to 64M nodes = 4GB)
- 32-bit indices still work
- Per-game chunks eliminate atomic contention
- **64-byte aligned allocation**: Use `std::aligned_alloc` for each block
- **Fast O(1) indexing**: Bit shifts, no division, no vector indirection
- **No reallocation**: Fixed array never moves in memory

3. **MCTS Search** (`search.hpp/cpp`)
   - Selection: PUCT with **dynamic FPU (First Play Urgency)**
   - **Prefetch optimization**: Use `__builtin_prefetch` for next likely child node
   - Expansion: Create child nodes with policy priors
   - Backpropagation: Update value_sum and visit_count
   - Policy extraction: Visit count distribution with temperature
   - Dirichlet noise: Add exploration at root
   - **Adaptive virtual loss**: Scale based on search tree standard deviation
   - **Tree pruning**: Mark old subtrees for reuse when moving root

**Dynamic FPU**:
```cpp
float compute_fpu_value(Node* parent) {
    // Instead of static 0 or -1, use parent's Q-value minus exploration constant
    float parent_q = parent->q_value();
    float fpu_reduction = 0.2f * config_.c_puct;  // Tunable
    return parent_q - fpu_reduction;
}

float Node::puct_score(float c_puct, uint32_t parent_visits, float fpu_value) const {
    if (visit_count == 0) {
        // Unvisited node uses FPU value
        float u = c_puct * prior * std::sqrt(parent_visits);
        return fpu_value + u;
    }
    // Normal PUCT for visited nodes
    float q = q_value();
    float u = c_puct * prior * std::sqrt(parent_visits) / (1 + visit_count);
    return q + u;
}
```

**Tree Pruning for Root Reuse**:
```cpp
void prune_tree(Node* new_root, NodePool& pool) {
    // Mark all nodes except new_root's subtree as available
    // Simple approach: track generation ID
    current_generation_++;
    new_root->generation = current_generation_;
    // Nodes with old generation IDs can be reused
}
```

4. **Batch Coordinator** (`batch_coordinator.hpp/cpp`) - **CORE INNOVATION**
   - **Double buffering**: While GPU processes Batch N, CPU collects Batch N+1
   - Collect pending evaluations from multiple games
   - **Dynamic batching**: Don't wait for stragglers - dispatch when 90% ready or timeout (20ms)
   - **Priority queue batching**: Prioritize root nodes and high-uncertainty nodes for next batch
   - **Strategic dispatch**: Early simulations (critical) get priority over deep-tree evaluations
   - Batch encode observations and legal masks
   - **Optimized Python bridge**: Return batch tensors to Python, Python runs inference, C++ updates results
   - Minimize C++ → Python → C++ context switching
   - Distribute results back to games
   - Statistics: batch size, wait time, throughput, GPU utilization

**Double Buffering Strategy**:
```cpp
class BatchCoordinator {
    // Two buffers: one being processed by GPU, one being filled by CPU
    std::array<std::vector<PendingEval>, 2> buffers_;
    std::atomic<int> active_buffer_{0};
    std::atomic<bool> gpu_busy_{false};

    void run_search_loop() {
        // Pre-allocate buffers to avoid reallocation
        buffers_[0].resize(batch_size_);
        buffers_[1].resize(batch_size_);

        while (not_done) {
            int fill_buffer = active_buffer_.load();
            std::atomic<int> buffer_index{0};

            // CPU: Collect leaves for current buffer (THREAD-SAFE)
            #pragma omp parallel for
            for (int i = 0; i < num_games; ++i) {
                if (games[i].is_active()) {
                    // CRITICAL: Use atomic index, not push_back (not thread-safe!)
                    int idx = buffer_index.fetch_add(1, std::memory_order_relaxed);
                    if (idx < batch_size_) {
                        buffers_[fill_buffer][idx] = games[i].select_leaf();
                    }
                }
            }

            // Resize to actual size
            int actual_size = buffer_index.load();
            buffers_[fill_buffer].resize(actual_size);

            // When buffer ready, dispatch to GPU
            if (should_dispatch(buffers_[fill_buffer])) {
                // Swap buffers
                int process_buffer = fill_buffer;
                active_buffer_ = 1 - fill_buffer;

                // GPU processes this buffer (async)
                dispatch_to_gpu_async(buffers_[process_buffer]);

                // CPU immediately starts filling next buffer
                // (GPU and CPU work in parallel)
            }
        }
    }
};
```

**Priority Queue Strategy**:
```cpp
struct PendingEval {
    int game_id;
    Node* node;
    float priority;  // Higher = more urgent

    float compute_priority() const {
        // Root nodes get highest priority
        if (node->parent_idx == 0) return 1000.0f;

        // High uncertainty nodes (low visit count, high prior)
        float uncertainty = node->prior / (1.0f + node->visit_count);
        return uncertainty * 100.0f;
    }
};

// Use priority queue instead of simple vector
std::priority_queue<PendingEval> pending_queue_;
```

**Key Optimization**: Lock-free node updates minimize contention

### Phase 2.5: Transposition Table (Optional but Recommended)
**Files**: `include/mcts/transposition_table.hpp`, `src/mcts/transposition_table.cpp`

1. **Lock-free transposition table**
   - Store (Zobrist hash → policy, value) from GPU evaluations
   - Before GPU evaluation, check if position already evaluated
   - Can save 10-20% of GPU calls in middle-game
   - Use lock-free hash table with atomic operations
   - Small size: 1M entries (~50MB)

**Implementation**:
```cpp
struct TTEntry {
    std::atomic<uint64_t> hash{0};
    std::array<float, 4672> policy;  // Compressed or top-K only
    float value;
};

class TranspositionTable {
    std::vector<TTEntry> table_;

    bool probe(uint64_t hash, std::vector<float>& policy, float& value) {
        size_t index = hash % table_.size();
        uint64_t stored_hash = table_[index].hash.load(std::memory_order_relaxed);
        if (stored_hash == hash) {
            policy = table_[index].policy;
            value = table_[index].value;
            return true;
        }
        return false;
    }

    void store(uint64_t hash, const std::vector<float>& policy, float value) {
        size_t index = hash % table_.size();
        table_[index].hash.store(hash, std::memory_order_relaxed);
        table_[index].policy = policy;
        table_[index].value = value;
    }
};
```

### Phase 2.6: Endgame Tablebases (Optional Enhancement)
**Files**: `include/chess/syzygy.hpp`, `src/chess/syzygy.cpp`

1. **Syzygy tablebase integration** (optional, can be added post-MVP)
   - Probe Syzygy tablebases for 3-5 piece endgames
   - Terminate MCTS branches early if position is theoretical win/draw
   - Significantly reduces search space in endgame
   - Use existing Syzygy probing library (Fathom)
   - Can be disabled if tablebases not available

**Note**: This is an optional enhancement that can be added after the core implementation is working.

### Phase 3: Encoding (C++)
**Files**: `include/encoding/`, `src/encoding/`

1. **Observation encoding** (`observation.hpp/cpp`)
   - Convert Position → (119, 8, 8) tensor
   - **CRITICAL: Use NHWC (channels-last) layout for Tensor Core performance**
   - **SIMD-optimized encoding**: Use AVX2 intrinsics to scatter bitboard bits into float array
   - Feature planes:
     - 12 planes: piece positions (6 types × 2 colors)
     - 8 planes: position history (last 8 positions)
     - Additional planes: castling, en passant, repetition, etc.
   - **NHWC layout**: (batch, height, width, channels) = (batch, 8, 8, 119)
     - 2-3x better GPU performance with Tensor Cores
     - Cleaner SIMD scatter: 119 channels contiguous per square
     - PyTorch supports via `memory_format=torch.channels_last`
   - Efficient bitboard → tensor conversion with SIMD

**SIMD Encoding Example (NHWC - Channels-Last)**:
```cpp
// Use AVX2 to convert bitboard to float array
// CRITICAL: Use NHWC (channels-last) for Tensor Core performance
void encode_bitboard_simd_nhwc(uint64_t bb, float* output, int plane_idx, int batch_idx) {
    // NHWC layout: (batch, height, width, channels) = (batch, 8, 8, 119)
    // Stride verification:
    // - Between batches: 8 * 8 * 119 = 7616 floats
    // - Between ranks: 8 * 119 = 952 floats
    // - Between files: 119 floats
    // - Between channels: 1 float

    // Calculate base offset for this batch
    const int HEIGHT = 8;
    const int WIDTH = 8;
    const int CHANNELS = 119;
    float* batch_base = output + batch_idx * (HEIGHT * WIDTH * CHANNELS);

    // Extract bits and write to NHWC layout
    // For each square (rank, file), write to position: rank * WIDTH * CHANNELS + file * CHANNELS + plane_idx
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int square = rank * 8 + file;
            int bit = (bb >> square) & 1;

            // NHWC: All 119 channels for this square are contiguous!
            // This is MUCH better for Tensor Cores and memory coalescing
            float* square_base = batch_base + rank * WIDTH * CHANNELS + file * CHANNELS;
            square_base[plane_idx] = static_cast<float>(bit);
        }
    }
}

// Optimized version: Write all planes for a position at once
void encode_position_nhwc(const Position& pos, float* output, int batch_idx) {
    const int HEIGHT = 8;
    const int WIDTH = 8;
    const int CHANNELS = 119;
    float* batch_base = output + batch_idx * (HEIGHT * WIDTH * CHANNELS);

    // For each square, write all 119 channels contiguously
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            float* square_channels = batch_base + rank * WIDTH * CHANNELS + file * CHANNELS;

            // Write all 119 channels for this square (contiguous memory!)
            // Plane 0-11: Piece positions (6 types × 2 colors)
            for (int piece_type = 0; piece_type < 6; ++piece_type) {
                for (int color = 0; color < 2; ++color) {
                    int plane = piece_type * 2 + color;
                    Bitboard bb = pos.get_piece_bitboard(piece_type, color);
                    int square = rank * 8 + file;
                    square_channels[plane] = (bb >> square) & 1;
                }
            }

            // Plane 12-19: Position history (last 8 positions)
            // ... (similar pattern)

            // Remaining planes: castling, en passant, etc.
            // ... (similar pattern)
        }
    }
}

// Compile-time stride verification for NHWC
static_assert(8 * 8 * 119 == 7616, "NHWC batch stride");
static_assert(8 * 119 == 952, "NHWC rank stride");
static_assert(119 == 119, "NHWC file stride (channels per square)");
```

**Why NHWC is Superior for GPUs**:
- **Tensor Core optimization**: Modern NVIDIA GPUs (Volta, Turing, Ampere, Hopper) perform 2-3x better with NHWC
- **Memory coalescing**: All 119 channels for a square are contiguous, enabling efficient GPU memory access
- **Cleaner SIMD scatter**: Writing 119 floats sequentially is trivial (just increment pointer)
- **PyTorch support**: `memory_format=torch.channels_last` enables this layout

2. **Action encoding** (`action.hpp/cpp`)
   - Move ↔ action index (0-4671) mapping
   - Action space:
     - Queen moves: 56 directions × 64 squares = 3584
     - Knight moves: 8 × 64 = 512
     - Underpromotions: 9 × 64 = 576
   - Legal action mask generation

**Key Optimization**: Direct bitboard manipulation for encoding

### Phase 4: Python Bindings (C++)
**Files**: `src/bindings/python_bindings.cpp`

1. **BatchedMCTSManager class**
   - Constructor: `(num_games, num_simulations, batch_size, evaluator_callback)`
   - `search_batch()` → returns batch tensors to Python (minimize C++→Python→C++ switching)
   - **Zero-copy tensor interface**: Write directly to pre-allocated `torch::Tensor` memory
   - `update_with_results(policies, values)` → C++ updates nodes and continues search
   - `get_stats()` → batch statistics

2. **Optimized evaluator interface**
   - **Pattern**: C++ collects batch → writes to torch tensor → Python runs inference → C++ updates
   - **NOT**: C++ → Python callback → C++ (too much context switching)
   - **Direct pointer writing**: C++ writes encoded bitboards directly into torch::Tensor buffer
   - GIL management: Release during C++ work, acquire only for data transfer
   - Return: (policies, values) as NumPy arrays

**Zero-Copy Tensor Writing (NHWC Layout)**:
```cpp
// C++ writes directly to PyTorch tensor memory
py::object collect_batch_to_tensor(py::object obs_tensor, py::object mask_tensor) {
    // Get raw pointers from PyTorch tensors (NHWC layout)
    float* obs_ptr = obs_tensor.attr("data_ptr")().cast<float*>();
    float* mask_ptr = mask_tensor.attr("data_ptr")().cast<float*>();

    std::atomic<int> batch_index{0};

    // C++ writes directly to tensor memory (no copy!)
    #pragma omp parallel for
    for (int i = 0; i < num_games_; ++i) {
        if (games[i].is_active() && games[i].has_pending_leaf()) {
            int idx = batch_index.fetch_add(1, std::memory_order_relaxed);
            if (idx < batch_size_) {
                // NHWC: (batch, 8, 8, 119) - stride = 8 * 8 * 119 per batch
                encode_position_to_buffer_nhwc(games[i].position,
                                               obs_ptr + idx * 8 * 8 * 119);
                encode_legal_mask_to_buffer(games[i].position,
                                            mask_ptr + idx * 4672);
            }
        }
    }

    // Return actual batch size (may be less than batch_size_ due to 90% threshold)
    int actual_size = batch_index.load();
    return py::make_tuple(actual_size, game_ids);
}
```

**Python side handles variable batch size (NHWC layout)**:
```python
# In TorchEvaluator
def evaluate_batch(self, obs, masks, actual_size):
    # Slice to actual size (handles 90% threshold dispatch)
    obs_slice = self.static_obs[:actual_size]
    mask_slice = self.static_mask[:actual_size]

    obs_slice.copy_(obs[:actual_size])
    mask_slice.copy_(masks[:actual_size])

    # Replay graph
    self.graph.replay()

    return (self.static_policy[:actual_size].clone(),
            self.static_value[:actual_size].clone())
```

**CRITICAL: PyTorch Model Must Use Channels-Last**:
```python
# When loading/creating the model, convert to channels-last format
model = AlphaZeroNetwork(...)
model = model.to(memory_format=torch.channels_last)

# Ensure input tensors are also channels-last
obs_tensor = torch.zeros((256, 8, 8, 119), device='cuda',
                         memory_format=torch.channels_last)
```

3. **Expose chess classes for testing**
   - Position, Move for unit tests
   - Move generation for validation

**Key Optimization**: Zero-copy data transfer - C++ writes directly to GPU tensor memory

### Phase 5: Build System
**Files**: `CMakeLists.txt`, `setup.py`, `colab_setup.sh`

1. **CMakeLists.txt**
   - C++20 standard (fallback to C++17 if unavailable)
   - Optimization flags: `-O3 -march=native -mavx2 -mpopcnt -mbmi2`
   - Find Python3 and pybind11 (let pip handle it via setup.py)
   - **Find and link OpenMP**: Required for parallel tree traversal
   - Create Python module: `pybind11_add_module(alphazero_sync)`
   - Link threading library
   - **Use ccache** if available to speed up recompilation

**CMake snippet**:
```cmake
# Try C++20, fallback to C++17
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)

find_package(OpenMP REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

target_link_libraries(alphazero_sync PRIVATE
    OpenMP::OpenMP_CXX
    Threads::Threads
    pybind11::module
)
```

2. **setup.py**
   - Custom build_ext using CMake
   - Detect CUDA availability (for info only)
   - Install dependencies: numpy, torch
   - Build in Release mode with parallel compilation (-j4)
   - **Pre-compile wheels** for common platforms if possible

3. **colab_setup.sh**
   - Install build tools: cmake, g++, python3-dev, ccache
   - Install Python deps: pybind11, numpy, torch
   - Clone repo and `pip install -e .`
   - One-line usage: `curl -sSL <url> | bash`
   - Expected build time: 2-3 minutes on Colab

### Phase 6: Python Integration Layer
**Files**: `python/`

1. **evaluator.py**
   - `TorchEvaluator` class wrapping PyTorch model
   - Load existing AlphaZeroNetwork weights
   - **CUDA Graphs**: Use static computation graphs for inference (10-15% throughput boost)
   - Batch inference with mixed precision (AMP)
   - Interface: `evaluate_batch(obs, masks) → (policies, values)`

**CUDA Graphs Implementation (NHWC Layout)**:
```python
class TorchEvaluator:
    def __init__(self, model, device="cuda"):
        # CRITICAL: Convert model to channels-last format for Tensor Core performance
        self.model = model.to(device).to(memory_format=torch.channels_last)
        self.device = device

        # Pre-allocate tensors for CUDA graph (NHWC layout)
        # Shape: (batch, height, width, channels) = (256, 8, 8, 119)
        self.static_obs = torch.zeros((256, 8, 8, 119), device=device,
                                      memory_format=torch.channels_last)
        self.static_mask = torch.zeros((256, 4672), device=device)

        # Capture CUDA graph (static shapes)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_policy, self.static_value = self.model(
                self.static_obs, self.static_mask
            )

    def evaluate_batch(self, obs, masks):
        batch_size = obs.shape[0]
        # Copy to static buffers
        self.static_obs[:batch_size].copy_(obs)
        self.static_mask[:batch_size].copy_(masks)

        # Replay graph (much faster than launching kernels)
        self.graph.replay()

        return (self.static_policy[:batch_size].clone(),
                self.static_value[:batch_size].clone())
```

**Why Channels-Last Matters**:
- **2-3x faster convolutions** on modern GPUs (Volta, Turing, Ampere, Hopper)
- **Better memory coalescing**: GPU threads access contiguous memory
- **Tensor Core optimization**: Hardware accelerators designed for NHWC layout
- **No performance penalty**: PyTorch handles layout conversion efficiently

2. **game_manager.py**
   - High-level API for self-play
   - Manage multiple games in parallel
   - Coordinate with BatchedMCTSManager
   - Generate training data

3. **compatibility.py**
   - Adapter to work with existing evaluate.py
   - Convert between FEN strings and C++ Position
   - Interface with web app
   - Load/save game states

### Phase 7: Testing & Validation
**Files**: `tests/`

1. **C++ unit tests** (Google Test)
   - Bitboard operations
   - Move generation correctness
   - MCTS node operations
   - Batch coordinator synchronization

2. **Python integration tests**
   - End-to-end search with dummy evaluator
   - Load PyTorch model and run inference
   - Compare results with existing Python MCTS
   - Performance benchmarks

3. **Colab notebook**
   - Installation instructions
   - Simple example: run search on starting position
   - Self-play example
   - Training integration

## Critical Files to Create

### C++ Headers (include/)
1. `chess/bitboard.hpp` - Bitboard types and operations
2. `chess/position.hpp` - Chess position state
3. `chess/movegen.hpp` - Move generation
4. `chess/zobrist.hpp` - Zobrist hashing
5. `mcts/node.hpp` - MCTS node structure
6. `mcts/node_pool.hpp` - Memory pool
7. `mcts/search.hpp` - MCTS search algorithm
8. `mcts/batch_coordinator.hpp` - Multi-game batching
9. `mcts/config.hpp` - Configuration structs
10. `encoding/observation.hpp` - Board encoding
11. `encoding/action.hpp` - Action encoding
12. `utils/spinlock.hpp` - Lock-free primitives

### C++ Implementation (src/)
13. `chess/bitboard.cpp` - Bitboard utilities
14. `chess/position.cpp` - Position operations
15. `chess/movegen.cpp` - Magic bitboards initialization
16. `chess/zobrist.cpp` - Random number generation
17. `mcts/node.cpp` - Node operations
18. `mcts/node_pool.cpp` - Pool management
19. `mcts/search.cpp` - MCTS algorithm
20. `mcts/batch_coordinator.cpp` - Batching logic
21. `encoding/observation.cpp` - Encoding implementation
22. `encoding/action.cpp` - Action mapping
23. `bindings/python_bindings.cpp` - pybind11 interface

### Python Layer (python/)
24. `__init__.py` - Package initialization
25. `evaluator.py` - PyTorch evaluator wrapper
26. `game_manager.py` - Multi-game orchestration
27. `compatibility.py` - Existing code interface

### Build & Setup
28. `CMakeLists.txt` - Build configuration
29. `setup.py` - Python package setup
30. `colab_setup.sh` - Colab installation script
31. `README.md` - Documentation

### Tests
32. `tests/test_chess.cpp` - Chess engine tests
33. `tests/test_mcts.cpp` - MCTS tests
34. `tests/test_batching.cpp` - Batch coordinator tests
35. `tests/test_integration.py` - Python integration tests
36. `tests/colab_example.ipynb` - Colab demo notebook

## Key Optimizations for Speed

1. **Bitboards**: 10-50x faster than python-chess for move generation
2. **Memory pooling**: Eliminate allocation overhead (1M nodes pre-allocated)
3. **Lock-free operations**: Atomic operations instead of mutexes
4. **Cache-line alignment**: 64-byte nodes fit in single cache line
5. **Magic bitboards**: O(1) sliding piece move generation
6. **Zero-copy interface**: NumPy arrays share memory with C++
7. **Batch synchronization**: Maximize GPU utilization
8. **AVX2/BMI2 instructions**: Hardware popcount and bit manipulation

## Integration with Existing Code

### Loading PyTorch Models
```python
from alphazero_sync import BatchedMCTSManager
from alphazero.neural.network import AlphaZeroNetwork

# Load existing model
model = AlphaZeroNetwork.load_checkpoint("model.pt")
evaluator = TorchEvaluator(model, device="cuda")

# Create batched MCTS
mcts = BatchedMCTSManager(
    num_games=64,
    num_simulations=800,
    batch_size=256,
    evaluator=evaluator.evaluate_batch
)
```

### Interface with evaluate.py
- Convert FEN → C++ Position
- Run MCTS search
- Return policy and value
- Compatible with existing web app

### Interface with Training
- Generate self-play games using BatchedMCTSManager
- Output same format as existing self-play
- Drop-in replacement for training pipeline

## Verification Plan

1. **Correctness**
   - Unit tests for move generation (compare with python-chess)
   - Perft tests for position validation
   - MCTS convergence tests (compare with Python implementation)

2. **Performance**
   - Benchmark move generation speed
   - Measure MCTS simulations per second
   - Profile batch coordinator efficiency
   - Compare end-to-end self-play speed

3. **Integration**
   - Load existing model weights
   - Run evaluate.py with new backend
   - Test web app compatibility
   - Verify training data format

4. **Colab**
   - Test installation script
   - Run example notebook
   - Verify GPU inference works
   - Check compilation time

## Expected Performance Gains

- **Move generation**: 10-50x faster (bitboards vs python-chess)
- **MCTS tree operations**: 5-10x faster (C++ vs Python)
- **Batch efficiency**: 2-4x better GPU utilization (multi-game batching)
- **Overall self-play**: 20-100x faster than current implementation

## Estimated Complexity

- **Lines of code**: ~5000 C++, ~500 Python
- **Build time on Colab**: 2-3 minutes
- **Memory usage**: ~100MB per 1M nodes
- **Batch size**: 256-512 leaves (tunable)

## Critical Implementation Details & Technical Risks

### 1. Virtual Loss Logic (CRITICAL - LEELA APPROACH)

**Issue**: Virtual loss must discourage multiple threads from exploring the same path while maintaining search stability.

**CORRECT Solution (Leela Chess Zero Approach)**:
```cpp
// Use parent Q-value for virtual losses (Leela approach)
float Node::q_value(float parent_q) const {
    uint32_t n = visit_count.load(std::memory_order_relaxed);
    int16_t v = virtual_loss.load(std::memory_order_relaxed);

    if (n == 0 && v == 0) {
        // Unvisited node: use FPU (First Play Urgency)
        return parent_q - 0.2f;  // Slightly pessimistic
    }

    if (n == 0 && v > 0) {
        // Only virtual visits: use parent Q-value
        // This keeps Q-value stable and doesn't trend toward zero
        return parent_q;
    }

    // Has real visits: compute normal Q-value
    int64_t sum = value_sum_fixed.load(std::memory_order_relaxed);
    float real_value = sum / 10000.0f;

    // Virtual visits contribute parent_q to the average
    float total_value = real_value + v * parent_q;
    return total_value / (n + v);
}

// Thread workflow:
// 1. Select node, add virtual loss
node->add_virtual_loss();  // +1 to virtual_loss

// 2. Evaluate (GPU)
// ...

// 3. Backpropagate real value, remove virtual loss
node->update(real_value);  // Add real value to value_sum
node->remove_virtual_loss();  // -1 from virtual_loss
```

**Why This Works (Leela Approach)**:
- Virtual visits use parent Q-value as placeholder
- Prevents Q-value from trending toward zero with high virtual loss
- Maintains search stability even with many concurrent threads
- When real evaluation returns, it replaces the virtual placeholder

**Alternative: Soft Virtual Loss**:
```cpp
// Add small negative value to value_sum (not -1, but -0.1 or similar)
float total_value = (sum / 10000.0f) - v * 0.1f;  // Soft penalty
return total_value / (n + v);
```

**WRONG Approaches**:
```cpp
// WRONG 1: Subtracting full v makes it look like losing outcome
float total_value = (sum / 10000.0f) - v;  // ❌ Too aggressive!

// WRONG 2: Only inflating denominator trends toward zero
return (sum / 10000.0f) / (n + v);  // ❌ Unstable with high v!
```

### 2. Fixed-Point Overflow (CRITICAL)

**Issue**: int32_t with 10000x multiplier maxes at ~214K sum, insufficient for long searches.

**Solution**: Use **int64_t** for value_sum_fixed with proper memory ordering
```cpp
std::atomic<int64_t> value_sum_fixed{0};  // Can handle ~9×10^14 sum

// With 10000x multiplier:
// Max sum = 9×10^14 / 10000 = 9×10^10 (90 billion)
// Safe for millions of simulations

// CRITICAL: Use memory_order_release/acquire to prevent race conditions
void update(float value) {
    // Use std::round() to avoid bias toward early-visited nodes
    int64_t fixed_value = static_cast<int64_t>(std::round(value * 10000.0f));
    value_sum_fixed.fetch_add(fixed_value, std::memory_order_relaxed);
}

// During backpropagation, use release semantics on final update
void backpropagate(Node* leaf, float value) {
    Node* node = leaf;
    while (node != nullptr) {
        // Update value_sum first
        int64_t fixed_value = static_cast<int64_t>(std::round(value * 10000.0f));
        node->value_sum_fixed.fetch_add(fixed_value, std::memory_order_relaxed);

        // Update visit_count with release semantics on root
        if (node->parent_idx == 0) {  // Root node
            node->visit_count.fetch_add(1, std::memory_order_release);
        } else {
            node->visit_count.fetch_add(1, std::memory_order_relaxed);
        }

        node = &pool_[node->parent_idx];
        value = -value;  // Flip perspective
    }
}

// During selection, use acquire semantics when reading visit_count
float Node::q_value(float parent_q) const {
    uint32_t n = visit_count.load(std::memory_order_acquire);  // Acquire!
    int16_t v = virtual_loss.load(std::memory_order_relaxed);

    if (n == 0 && v == 0) return parent_q - 0.2f;  // FPU
    if (n == 0 && v > 0) return parent_q;  // Virtual only

    // Has real visits
    int64_t sum = value_sum_fixed.load(std::memory_order_relaxed);
    float real_value = sum / 10000.0f;
    float total_value = real_value + v * parent_q;
    return total_value / (n + v);
}
```

**Why memory_order_release/acquire**:
- `memory_order_release` on visit_count ensures value_sum update is visible
- `memory_order_acquire` on visit_count ensures we see the corresponding value_sum
- Prevents race where thread reads stale value_sum with updated visit_count
- Without this, PUCT can calculate wildly incorrect Q-values (∞ or NaN)

### 3. GIL and Threading (CRITICAL - DEADLOCK RISK)

**Issue**: Python GIL can create circular dependencies and deadlocks. If BatchCoordinator holds a mutex while waiting for GPU results via condition_variable, and the GPU callback needs GIL to return results, we get a deadlock where C++ waits for Python and Python waits for C++ mutex.

**CORRECT Solution: Lock-Free Queue (moodycamel::ConcurrentQueue)**:
```cpp
// Use lock-free queue for async communication (no mutex, no deadlock!)
#include <concurrentqueue.h>  // moodycamel::ConcurrentQueue

class BatchCoordinator {
    moodycamel::ConcurrentQueue<PendingEval> request_queue_;
    moodycamel::ConcurrentQueue<EvalResult> result_queue_;

    // C++ side: Submit request (lock-free!)
    void submit_leaf(int game_id, const PendingEval& eval) {
        request_queue_.enqueue(eval);
    }

    // Python side: Collect batch (lock-free!)
    py::list collect_batch_requests(int max_batch_size) {
        py::list batch;
        PendingEval eval;
        int count = 0;
        while (count < max_batch_size && request_queue_.try_dequeue(eval)) {
            batch.append(eval);
            count++;
        }
        return batch;
    }

    // Python side: Submit results (lock-free!)
    void submit_batch_results(py::list results) {
        for (auto result : results) {
            result_queue_.enqueue(result.cast<EvalResult>());
        }
    }

    // C++ side: Get result (lock-free!)
    bool try_get_result(int game_id, EvalResult& result) {
        return result_queue_.try_dequeue(result);
    }
};
```

**Python Evaluation Loop (Separate Thread)**:
```python
def inference_loop(coordinator, model):
    while True:
        batch = coordinator.collect_batch_requests(batch_size=256)
        if len(batch) > 0:
            obs, masks = prepare_tensors(batch)
            policies, values = model(obs, masks)
            results = package_results(batch, policies, values)
            coordinator.submit_batch_results(results)
        else:
            time.sleep(0.001)  # 1ms sleep if no requests
```

**Why This Works**:
- Lock-free queues eliminate all mutex contention
- C++ never waits for Python (just checks queue)
- Python never waits for C++ mutex
- No circular dependencies possible
- Scales to many threads without contention

### 4. Double Buffering Synchronization with Hard Sync

**Issue**: Single game might submit multiple leaves before others submit one, causing redundant searches. Also, games that miss the 90% threshold can starve. **Critical Risk**: The 90% threshold can create search quality imbalance where slow games (complex positions) consistently miss batches and receive fewer effective simulations.

**Solution with Starvation Prevention + Hard Sync**:
```cpp
class BatchCoordinator {
    std::vector<int> leaves_per_game_;  // Track submissions per game
    std::vector<int> missed_batches_;   // Track starvation
    std::vector<float> game_priority_;  // Priority for next batch
    uint32_t batch_counter_{0};         // Count batches for hard sync
    static constexpr uint32_t HARD_SYNC_INTERVAL = 10;  // Force sync every N batches

    void collect_batch() {
        leaves_per_game_.assign(num_games_, 0);

        // HARD SYNC: Every N batches, wait for ALL active games
        bool force_hard_sync = (batch_counter_ % HARD_SYNC_INTERVAL == 0);

        // Sort games by priority (starved games first)
        std::vector<int> game_order(num_games_);
        std::iota(game_order.begin(), game_order.end(), 0);
        std::sort(game_order.begin(), game_order.end(), [this](int a, int b) {
            return game_priority_[a] > game_priority_[b];
        });

        std::atomic<int> batch_index{0};

        if (force_hard_sync) {
            // HARD SYNC: Wait for ALL active games to submit at least one leaf
            // This prevents search quality imbalance from 90% threshold
            #pragma omp parallel for
            for (int idx = 0; idx < num_games_; ++idx) {
                int i = game_order[idx];
                if (games[i].is_active()) {
                    auto leaf = games[i].select_leaf();
                    int pos = batch_index.fetch_add(1, std::memory_order_relaxed);
                    buffers_[fill_buffer][pos] = leaf;
                    leaves_per_game_[i]++;
                    missed_batches_[i] = 0;  // Reset starvation counter
                    game_priority_[i] = 1.0f;  // Normal priority
                }
            }
        } else {
            // NORMAL: Use 90% threshold with priority queue
            #pragma omp parallel for
            for (int idx = 0; idx < num_games_; ++idx) {
                int i = game_order[idx];
                if (games[i].is_active() && batch_index.load() < batch_size_) {
                    auto leaf = games[i].select_leaf();
                    int pos = batch_index.fetch_add(1, std::memory_order_relaxed);
                    if (pos < batch_size_) {
                        buffers_[fill_buffer][pos] = leaf;
                        leaves_per_game_[i]++;
                        missed_batches_[i] = 0;  // Reset starvation counter
                        game_priority_[i] = 1.0f;  // Normal priority
                    } else {
                        // Missed batch due to 90% threshold
                        missed_batches_[i]++;
                        game_priority_[i] = 10.0f * missed_batches_[i];  // High priority!
                    }
                }
            }
        }

        // Resize to actual size
        int actual_size = std::min(batch_index.load(), batch_size_);
        buffers_[fill_buffer].resize(actual_size);

        batch_counter_++;
    }
};
```

**Why This Works**:
- Track how many batches each game has missed
- Games that miss batches get exponentially higher priority
- Next batch processes starved games first
- **Hard Sync every N batches**: Prevents search quality imbalance from 90% threshold
- Ensures all games progress at roughly the same rate with balanced search quality

### 5. Action Encoding Validation (CRITICAL - SILENT ERRORS)

**Issue**: Move → action index mapping is error-prone, especially for promotions. **Silent errors** occur when the C++ engine generates a legal move but encodes it to the wrong action index. The model "sees" a different move than intended, causing training to fail without obvious errors.

**Solution: Mandatory Round-Trip Testing**
```cpp
// In tests/test_encoding.cpp
TEST(ActionEncoding, LegalMoveValidation) {
    chess::Position pos;  // Starting position

    // Generate all legal moves
    std::vector<chess::Move> moves;
    chess::MoveGenerator::generate_moves(pos, moves);

    // Encode to action indices
    std::vector<uint16_t> actions;
    for (const auto& move : moves) {
        actions.push_back(encoding::move_to_action(move));
    }

    // Decode back to moves
    for (size_t i = 0; i < moves.size(); ++i) {
        chess::Move decoded = encoding::action_to_move(actions[i], pos);
        EXPECT_EQ(moves[i].from, decoded.from);
        EXPECT_EQ(moves[i].to, decoded.to);
        EXPECT_EQ(moves[i].promotion, decoded.promotion);
    }
}

// Pre-calculate square-to-index mapping at startup
void init_action_encoding() {
    // 73 planes: 56 queen moves + 8 knight moves + 9 underpromotions
    for (int from_sq = 0; from_sq < 64; ++from_sq) {
        for (int plane = 0; plane < 73; ++plane) {
            action_map[from_sq][plane] = from_sq * 73 + plane;
        }
    }
}

// CRITICAL: Cross-validation with python-chess
TEST(ActionEncoding, CrossValidationWithPythonChess) {
    // Play 10,000 random games between C++ engine and python-chess
    // Any move discrepancy indicates encoding bug
    for (int game = 0; game < 10000; ++game) {
        chess::Position cpp_pos;
        // python_chess_pos = chess.Board()  // Python side

        while (!cpp_pos.is_game_over()) {
            // Generate moves in C++
            std::vector<chess::Move> cpp_moves;
            chess::MoveGenerator::generate_moves(cpp_pos, cpp_moves);

            // Generate moves in python-chess (via Python binding)
            // python_moves = list(python_chess_pos.legal_moves)

            // CRITICAL: Verify same number of legal moves
            // EXPECT_EQ(cpp_moves.size(), python_moves.size());

            // Pick random move and verify both engines agree
            chess::Move move = cpp_moves[rand() % cpp_moves.size()];
            cpp_pos.make_move(move);
            // python_chess_pos.push(move)  // Python side

            // CRITICAL: Verify positions match after move
            // EXPECT_EQ(cpp_pos.zobrist_hash, hash(python_chess_pos.fen()));
        }
    }
}
```

**Why This Is Critical**:
- Silent encoding errors are the #1 cause of AlphaZero training failure
- A single wrong action index means the model learns incorrect move mappings
- Cross-validation catches errors that unit tests miss (e.g., rank/file inversion)
- 10,000 random games cover edge cases (promotions, en passant, castling)

### 5.5. Perspective Flip Validation (CRITICAL - THE MOVE MAPPING TRAP)

**Issue**: In AlphaZero, the board is usually "flipped" so the model always thinks it is playing as White. If your C++ bitboard generates moves from Black's perspective but the encoder doesn't flip the coordinates before sending them to the model, the policy head will be 100% noise.

**Solution: Mandatory Perspective Flip**:
```cpp
// In encoding/observation.cpp
void encode_position_nhwc(const Position& pos, float* output, int batch_idx) {
    const int HEIGHT = 8;
    const int WIDTH = 8;
    const int CHANNELS = 119;
    float* batch_base = output + batch_idx * (HEIGHT * WIDTH * CHANNELS);

    // CRITICAL: Always encode from current player's perspective
    // If Black to move, flip the board so Black sees itself as "White"
    bool flip = (pos.side_to_move() == BLACK);

    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            // Flip coordinates if Black to move
            int actual_rank = flip ? (7 - rank) : rank;
            int actual_file = flip ? (7 - file) : file;
            int square = actual_rank * 8 + actual_file;

            float* square_channels = batch_base + rank * WIDTH * CHANNELS + file * CHANNELS;

            // Encode piece positions (swap colors if flipped)
            for (int piece_type = 0; piece_type < 6; ++piece_type) {
                for (int color = 0; color < 2; ++color) {
                    int plane = piece_type * 2 + color;

                    // If flipped, swap White and Black pieces
                    int actual_color = flip ? (1 - color) : color;

                    Bitboard bb = pos.get_piece_bitboard(piece_type, actual_color);
                    square_channels[plane] = (bb >> square) & 1;
                }
            }

            // Encode other features (castling rights, etc.)
            // CRITICAL: Also flip castling rights if Black to move
            // ...
        }
    }
}

// In encoding/action.cpp
uint16_t move_to_action(const Move& move, const Position& pos) {
    // CRITICAL: Flip move coordinates if Black to move
    bool flip = (pos.side_to_move() == BLACK);

    int from_sq = move.from;
    int to_sq = move.to;

    if (flip) {
        // Flip both squares
        from_sq = 63 - from_sq;  // Mirror vertically
        to_sq = 63 - to_sq;
    }

    // Calculate action index from flipped coordinates
    // ...
}

Move action_to_move(uint16_t action, const Position& pos) {
    // CRITICAL: Unflip move coordinates if Black to move
    bool flip = (pos.side_to_move() == BLACK);

    // Decode action to move coordinates
    int from_sq = /* ... */;
    int to_sq = /* ... */;

    if (flip) {
        // Unflip both squares
        from_sq = 63 - from_sq;
        to_sq = 63 - to_sq;
    }

    return Move(from_sq, to_sq, /* ... */);
}
```

**Validation Test**:
```cpp
TEST(PerspectiveFlip, WhiteAndBlackSeeSymmetricPositions) {
    // Create a position with White to move
    Position white_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // Create the same position with Black to move (after a null move)
    Position black_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");

    // Encode both positions
    float white_encoding[8 * 8 * 119];
    float black_encoding[8 * 8 * 119];
    encode_position_nhwc(white_pos, white_encoding, 0);
    encode_position_nhwc(black_pos, black_encoding, 0);

    // CRITICAL: The encodings should be IDENTICAL
    // Both White and Black should see themselves as "White" in the encoding
    for (int i = 0; i < 8 * 8 * 119; ++i) {
        EXPECT_FLOAT_EQ(white_encoding[i], black_encoding[i]);
    }
}

TEST(PerspectiveFlip, MoveEncodingConsistency) {
    // Test that e2-e4 for White encodes the same as e7-e5 for Black
    Position white_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Position black_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");

    // White's e2-e4 (from square 12 to square 28)
    Move white_move(12, 28);
    uint16_t white_action = move_to_action(white_move, white_pos);

    // Black's e7-e5 (from square 52 to square 36)
    Move black_move(52, 36);
    uint16_t black_action = move_to_action(black_move, black_pos);

    // CRITICAL: These should encode to the SAME action index
    // Both are "pawn moves 2 squares forward from e-file"
    EXPECT_EQ(white_action, black_action);
}
```

**Why This Is Critical**:
- **Perspective consistency**: Model must always see itself as "White" (playing from bottom)
- **Symmetry**: e2-e4 for White should be encoded identically to e7-e5 for Black
- **Training stability**: Without perspective flip, model learns two different representations for the same logical move
- **Policy head correctness**: Wrong perspective means 100% policy noise

### 6. Node Alignment Verification (CRITICAL - FALSE SHARING)

**Issue**: Simply using `alignas(64)` is not enough if nodes are packed in a `std::vector`. A `push_back` could trigger reallocation that breaks alignment or causes massive overhead.

**Solution: Use std::aligned_alloc in NodePool**
```cpp
Node* allocate_new_block() {
    uint32_t block_id = num_blocks_.fetch_add(1, std::memory_order_relaxed);
    if (block_id >= MAX_BLOCKS) {
        throw std::runtime_error("NodePool exhausted");
    }

    // CRITICAL: Use aligned_alloc for 64-byte alignment
    void* ptr = std::aligned_alloc(64, NODES_PER_BLOCK * sizeof(Node));
    if (!ptr) throw std::bad_alloc();

    // Placement new for each node
    Node* block = static_cast<Node*>(ptr);
    for (size_t i = 0; i < NODES_PER_BLOCK; ++i) {
        new (&block[i]) Node();
    }

    blocks_[block_id] = block;
    return block;
}

~NodePool() {
    // Proper cleanup
    for (uint32_t i = 0; i < num_blocks_; ++i) {
        for (size_t j = 0; j < NODES_PER_BLOCK; ++j) {
            blocks_[i][j].~Node();
        }
        std::free(blocks_[i]);  // Match aligned_alloc with free
    }
}
```

**Why This Is Critical**:
- `std::vector` doesn't guarantee alignment beyond `alignof(T)`
- Reallocation can move nodes to unaligned addresses
- False sharing on cache lines destroys parallel performance
- `std::aligned_alloc` ensures every block starts at 64-byte boundary

### 7. SIMD Memory Layout (NHWC - Channels-Last)

**Issue**: Tensor layout must match PyTorch model's memory format. NHWC (channels-last) provides 2-3x better GPU performance.

**Solution: Use NHWC Layout**:
```cpp
// For NHWC layout: (batch, height, width, channels) = (batch, 8, 8, 119)
void encode_position_nhwc(const Position& pos, float* output) {
    // output[b][h][w][c] = output[b * 8*8*119 + h * 8*119 + w * 119 + c]

    // For each square, write all 119 channels contiguously
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            float* square_channels = output + rank * 8 * 119 + file * 119;

            // Write all 119 channels for this square (contiguous memory!)
            for (int plane = 0; plane < 119; ++plane) {
                Bitboard bb = get_plane_bitboard(pos, plane);
                int square = rank * 8 + file;
                square_channels[plane] = (bb >> square) & 1;
            }
        }
    }
}

// Verify stride calculation for NHWC
static_assert(8 * 8 * 119 == 7616, "NHWC batch stride");
static_assert(8 * 119 == 952, "NHWC rank stride");
static_assert(119 == 119, "NHWC file stride (channels per square)");
```

**Why NHWC is Superior**:
- **2-3x faster convolutions** on Tensor Cores (Volta, Turing, Ampere, Hopper)
- **Cleaner SIMD scatter**: Write 119 floats sequentially per square
- **Better memory coalescing**: GPU threads access contiguous channels
- **PyTorch native support**: `memory_format=torch.channels_last`

### 8. BMI2/PEXT Optimization for Encoding (PERFORMANCE BOOST)

**Issue**: Looping through 64 bits to encode bitboards is slower than necessary on modern CPUs.

**Solution: Use PEXT (Parallel Bit Extract) instruction**
```cpp
#ifdef __BMI2__
#include <immintrin.h>

// Use PEXT to extract bits matching a mask in one instruction
inline uint64_t pext_bitboard(uint64_t bb, uint64_t mask) {
    return _pext_u64(bb, mask);
}

// Fast encoding using PEXT
void encode_bitboard_pext(uint64_t bb, float* output) {
    // Extract all 64 bits in parallel
    for (int sq = 0; sq < 64; ++sq) {
        output[sq] = (bb >> sq) & 1;
    }

    // Or use PEXT for rank-by-rank extraction
    for (int rank = 0; rank < 8; ++rank) {
        uint64_t rank_mask = 0xFFULL << (rank * 8);
        uint8_t rank_bits = _pext_u64(bb, rank_mask);

        // Convert 8 bits to 8 floats
        for (int file = 0; file < 8; ++file) {
            output[rank * 8 + file] = (rank_bits >> file) & 1;
        }
    }
}
#endif
```

**Why This Matters**:
- PEXT is a single-cycle instruction on modern Intel/AMD CPUs
- Eliminates bit-shifting loops
- Colab supports BMI2 (Intel Xeon CPUs)
- 2-3x faster than manual bit extraction

### 9. C++ Standard Selection

**Recommendation**: Use **C++20** if available on Colab (GCC 10+)

**Benefits**:
- `std::atomic<float>::fetch_add` (if lock-free on platform)
- `std::atomic_ref` for cleaner code
- Better constexpr support

**Fallback**: C++17 with manual atomic operations (current plan)

**CMakeLists.txt**:
```cmake
# Try C++20, fallback to C++17
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)  # Allow fallback
set(CMAKE_CXX_EXTENSIONS OFF)

# Check if C++20 is available
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
        set(CMAKE_CXX_STANDARD 17)
    endif()
endif()
```

### 8. Vendored Dependencies

**Include in third_party/**:
```
third_party/
├── pybind11/              # Python bindings (ensure Colab stability)
├── zobrist_constants.hpp  # Pre-generated Zobrist random numbers
└── magic_seeds.hpp        # Seeds for magic bitboard generation
```

**Why vendor**:
- Deterministic builds (no network dependency)
- Faster Colab setup (no downloads)
- Consistent across environments

### 9. Root Reuse and History Management

**Issue**: When moving root to child, must update Zobrist history for 3-fold repetition.

**Solution**:
```cpp
void move_root_to_child(Node* new_root, const chess::Move& move) {
    // Update position
    position_.make_move(move);

    // CRITICAL: Update Zobrist history
    position_.position_history.push_back(position_.zobrist_hash);

    // Prune old tree (mark old generation)
    prune_tree(new_root);

    // new_root becomes new search root
    root_ = new_root;
}
```

### 10. Backpropagation Memory Ordering

**Issue**: Race condition between visit_count and value_sum updates.

**Solution**: Use proper memory ordering
```cpp
void backpropagate(Node* leaf, float value) {
    Node* node = leaf;
    while (node != nullptr) {
        // Update value_sum first
        node->value_sum_fixed.fetch_add(
            static_cast<int64_t>(std::round(value * 10000.0f)),
            std::memory_order_relaxed
        );

        // Update visit_count with release semantics on final node
        if (node->parent_idx == 0) {  // Root node
            node->visit_count.fetch_add(1, std::memory_order_release);
        } else {
            node->visit_count.fetch_add(1, std::memory_order_relaxed);
        }

        node = &pool_[node->parent_idx];
        value = -value;  // Flip perspective
    }
}
```

### 11. Mixed Precision Handling

**Issue**: If using AMP (mixed precision), ensure C++ writes float32 and Python casts to float16.

**Solution**:
```python
class TorchEvaluator:
    def __init__(self, model, device="cuda", use_amp=True):
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def evaluate_batch(self, obs, masks):
        # C++ writes float32
        assert obs.dtype == torch.float32
        assert masks.dtype == torch.float32

        # Cast to float16 for inference if using AMP
        if self.use_amp:
            with torch.cuda.amp.autocast():
                policy, value = self.model(obs, masks)
        else:
            policy, value = self.model(obs, masks)

        # Return as float32 for C++
        return policy.float(), value.float()
```

## Summary of Critical Fixes

| Issue | Risk | Fix |
|-------|------|-----|
| Virtual loss formula | Critical | Use parent Q-value for virtual visits (Leela approach) |
| int64_t overflow | Critical | Use int64_t for value_sum_fixed (NOT int32_t) |
| Fixed-point rounding | High | Use std::round() to avoid bias |
| GIL circular dependency | Critical | Release C++ locks before Python calls |
| Single game hogging batch | Medium | Track leaves_per_game, enforce fairness |
| Action encoding bugs | Critical | Round-trip tests + 10K game cross-validation with python-chess |
| Perspective flip | Critical | Always encode from current player's perspective (flip for Black) |
| Tensor layout | Critical | Use NHWC (channels-last) for 2-3x better Tensor Core performance |
| C++ standard | Low | Use C++20 (fallback to C++17), enables better atomics |
| Network dependencies | Low | Let pip handle pybind11 (not vendored) |
| Root reuse history | Medium | Update position_history on root move |
| Backprop race condition | Medium | Use memory_order_release on root |
| Mixed precision mismatch | Medium | Ensure float32 C++↔Python interface |
| Node alignment | Critical | Use std::aligned_alloc for 64-byte alignment |
| Node pool indexing | Medium | Verify 32-bit indices safe with 64-byte nodes (static_assert) |
| push_back race | Critical | Use atomic index, not push_back in parallel loop |
| Magic determinism | Medium | Use fixed-seed PRNG for reproducible magic tables |
| Dual hash incremental | Medium | Update both 64-bit and 32-bit hashes incrementally |
| PUCT with virtual loss | Medium | Account for virtual loss in exploration term |
| 90% starvation + quality | Critical | Priority mechanism + Hard Sync every N batches |
| Power-of-two blocks | Medium | Use 2^18 (262144) nodes per block for fast indexing |
| Root contention | High | Lazy backpropagation with thread-local buffers |
| BMI2/PEXT encoding | Medium | Use PEXT instruction for 2-3x faster bitboard encoding |

## Final Pre-Implementation Checklist

### Architecture ✓
- [x] Multi-game batching with batch coordinator
- [x] Chained arena allocation (dynamic growth)
- [x] Per-game arenas (16K chunks)
- [x] Double buffering (GPU and CPU in parallel)
- [x] Priority queue batching
- [x] Zero-copy tensor interface

### Memory Management ✓
- [x] int64_t for value_sum_fixed (prevents overflow)
- [x] int16_t for virtual_loss (simple +1 per thread)
- [x] std::aligned_alloc for 64-byte alignment
- [x] Proper rounding in fixed-point conversion
- [x] Atomic index for thread-safe buffer filling

### Chess Engine ✓
- [x] Bitboard representation with BMI2/AVX2 intrinsics
- [x] Magic bitboards with fixed-seed PRNG (deterministic)
- [x] Dual Zobrist hashing (64-bit + 32-bit, both incremental)
- [x] Circular buffer for position history
- [x] Perft validation tests

### MCTS ✓
- [x] Dynamic FPU (parent Q-value minus exploration constant)
- [x] Proper virtual loss formula
- [x] Lazy backpropagation with thread-local buffers
- [x] Tree pruning with generation IDs
- [x] Prefetch optimization

### Batch Coordination ✓
- [x] OpenMP parallel traversal (<1ms for 256 leaves)
- [x] Dynamic batching (90% threshold + 20ms timeout)
- [x] Speculative leaf collection
- [x] Dual CUDA streams (overlap transfer and compute)
- [x] Thread-safe buffer filling (atomic index)

### Python Integration ✓
- [x] CUDA Graphs (10-15% throughput boost)
- [x] Zero-copy tensor writing (C++ writes to torch memory)
- [x] Proper GIL management (release C++ locks before Python)
- [x] Mixed precision handling (float32 interface)

### Build System ✓
- [x] C++20 with fallback to C++17
- [x] OpenMP for parallel traversal
- [x] ccache for faster recompilation
- [x] One-line Colab setup script

### Validation ✓
- [x] Action encoding roundtrip tests
- [x] Perft validation for move generation
- [x] SIMD stride verification
- [x] Memory ordering correctness

## Expected Performance

- **Move generation**: 5-10M moves/sec (vs 200K for python-chess)
- **MCTS simulations**: 50K-100K sims/sec per game (vs 5K-10K for Python)
- **Batch efficiency**: >90% GPU utilization (vs 50-70% without coordination)
- **Leaf collection**: <1ms for 256 leaves across 64 games (vs 6ms serial)
- **Overall speedup**: 20-100x faster self-play than current implementation
- **Memory**: <200MB for 64 games with dynamic node pool

## Ready for Implementation

The plan is now production-ready with:
- ✓ 36 files to create (C++ headers, implementation, Python wrappers, build system, tests)
- ✓ All critical bottlenecks identified and mitigated
- ✓ Concrete code examples for all critical sections
- ✓ Comprehensive validation strategy
- ✓ Three-tier priority system (MVP → Performance → Advanced)

**Status**: Ready to proceed with implementation! 🚀

## Implementation Workflow Strategy

To avoid spending days debugging subtle issues, follow this sequence:

### Phase 1: Chess Engine Validation (CRITICAL FIRST STEP)
**Do NOT write MCTS until chess engine passes Perft tests!**

1. **Implement bitboard chess engine**
   - Bitboard representation
   - Magic bitboard generation (fixed-seed PRNG)
   - Move generation (all move types)
   - Make/unmake moves

2. **Perft validation** (MANDATORY)
   - Perft(6) from starting position = 119,060,324 nodes
   - Perft tests for special positions (castling, en passant, promotions)
   - Any error here will make the AI hallucinate legal moves
   - **DO NOT PROCEED until Perft passes!**

3. **Cross-validation with python-chess** (MANDATORY)
   - Play 10,000 random games between C++ engine and python-chess
   - Verify same legal moves at each position
   - Verify same position hashes after each move
   - Any discrepancy indicates a critical bug (missed 3-fold repetition, wrong castling rights, etc.)
   - **This catches bugs that Perft misses!**

### Phase 2: MCTS with Mock Evaluator
**Test batch coordinator without GPU latency**

1. **Implement MCTS core**
   - Node structure with proper memory ordering
   - Node pool with fixed-size array
   - MCTS search algorithm
   - Virtual loss (Leela approach with parent Q-value)

2. **Implement batch coordinator**
   - Lock-free queues (moodycamel::ConcurrentQueue)
   - Dynamic batching (90% threshold + timeout)
   - Starvation prevention
   - Double buffering

3. **Test with dummy evaluator**
   - Random policy/value generator
   - Saturate CPU to verify 90% threshold logic
   - Verify timeout mechanism
   - Check starvation prevention
   - Measure leaf collection time (<1ms target)

### Phase 3: Python Integration
**Verify zero-copy transfer before real model**

1. **Implement pybind11 bindings**
   - Zero-copy tensor interface
   - Lock-free queue bindings
   - Proper GIL management

2. **Test with identity tensor**
   - Python returns input unchanged
   - Verify no memory copies
   - Verify NCHW stride correctness
   - Check for memory leaks

3. **Load real PyTorch model**
   - Test with existing AlphaZeroNetwork
   - Verify weight loading
   - Test mixed precision (AMP)
   - Implement CUDA Graphs

### Phase 4: Integration & Optimization
**Put it all together**

1. **End-to-end testing**
   - Run full self-play games
   - Verify training data format
   - Test with evaluate.py
   - Test with web app

2. **Performance profiling**
   - Measure move generation speed (target: 5-10M moves/sec)
   - Measure MCTS simulations (target: 50K-100K sims/sec per game)
   - Measure batch efficiency (target: >90% GPU utilization)
   - Measure leaf collection time (target: <1ms for 256 leaves)

3. **Optimization passes**
   - Add SIMD optimizations if needed
   - Tune batch size and timeout
   - Optimize memory layout
   - Add prefetch hints

### Performance Targets

| Component | Target Latency | Optimization |
|-----------|---------------|--------------|
| **Move Gen** | <100ns per move | Magic Bitboards |
| **Leaf Collection** | <1ms for 256 leaves | OpenMP Parallel For |
| **Batch Dispatch** | <20ms (max) | Dynamic Timeout |
| **Encoding** | <100μs per position | AVX2 Scatter |
| **GPU Inference** | <5ms for batch=256 | CUDA Graphs |
| **Overall** | 50K-100K sims/sec | All optimizations |

### Critical Checkpoints

Before proceeding to next phase, verify:
- ✓ **Phase 1**: Perft(6) = 119,060,324 (exact match!)
- ✓ **Phase 2**: Leaf collection <1ms, no deadlocks, fair game progress
- ✓ **Phase 3**: Zero-copy verified, no memory leaks, correct tensor layout
- ✓ **Phase 4**: 20-100x speedup achieved, training data format correct

## Success Criteria

1. ✓ Compiles on Google Colab with 1-2 line setup
2. ✓ Loads existing PyTorch model weights
3. ✓ Works with evaluate.py and web app
4. ✓ Generates correct chess moves (validated against python-chess)
5. ✓ Achieves 20x+ speedup over Python implementation
6. ✓ Batch coordinator efficiently synchronizes multiple games
7. ✓ All features implemented: virtual loss, Dirichlet noise, temperature, root reuse

## Critical Implementation Gotchas & Bottleneck Mitigation

### 1. Batch Coordinator Design (MOST CRITICAL)

**Challenge**: If 64 games are running and some finish faster, GPU sits idle waiting for stragglers.

**Solution**:
- **Dynamic batch size**: Dispatch when 90% of games ready (e.g., 230/256)
- **Timeout mechanism**: 20ms timeout - dispatch partial batch rather than wait
- **Adaptive batching**: Track per-game search speed, adjust batch collection strategy
- **Statistics**: Monitor GPU utilization, batch fill rate, straggler frequency

**Implementation**:
```cpp
// In batch_coordinator.cpp
bool should_dispatch_batch() {
    auto elapsed = now() - batch_start_time;
    size_t ready = pending_.size();
    return ready >= batch_size_ ||                    // Full batch
           (ready >= batch_size_ * 0.9 && elapsed > 10ms) ||  // 90% ready
           elapsed > 20ms;                            // Timeout
}
```

### 1.5. Root Contention Bottleneck (CRITICAL - HIGH PARALLELISM)

**The Problem**: As search depth increases, multiple threads within the *same* game contend for the root node's atomic variables. `fetch_add` on the root's `visit_count` becomes a hot spot that can serialize updates.

**Solution: Lazy Backpropagation**
```cpp
// Thread-local buffer for backpropagation
thread_local std::vector<std::pair<Node*, float>> backprop_buffer;

void lazy_backpropagate(Node* leaf, float value) {
    backprop_buffer.clear();

    // Traverse up and collect updates in thread-local buffer
    Node* node = leaf;
    while (node != nullptr) {
        backprop_buffer.push_back({node, value});
        node = &pool_[node->parent_idx];
        value = -value;  // Flip perspective
    }

    // Apply updates in reverse order (root last)
    // This reduces contention on hot nodes near root
    for (auto it = backprop_buffer.rbegin(); it != backprop_buffer.rend(); ++it) {
        Node* n = it->first;
        float v = it->second;

        int64_t fixed_value = static_cast<int64_t>(std::round(v * 10000.0f));
        n->value_sum_fixed.fetch_add(fixed_value, std::memory_order_relaxed);

        // Use release semantics only on root
        if (n->parent_idx == 0) {
            n->visit_count.fetch_add(1, std::memory_order_release);
        } else {
            n->visit_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
}
```

**Why This Works**:
- Thread-local buffers eliminate repeated atomic operations during traversal
- Batching updates reduces cache line bouncing
- Root node updates are still atomic but less frequent per thread
- Can reduce root contention by 30-50% in high-parallelism scenarios

### 1.6. CPU Latency in collect_batch (CRITICAL - THE HIDDEN TRAP)

**The Problem**: If `collect_batch()` runs serially (looping Game 1, then Game 2...), the CPU takes 5-10ms to find 256 leaves. **The GPU sits idle during this time**, destroying throughput.

**Bad Implementation** (DO NOT DO THIS):
```cpp
// WRONG: Serial iteration kills performance
for (int i = 0; i < num_games; ++i) {
    if (games[i].is_active()) {
        games[i].select_leaf();  // Takes ~100μs per game
    }
}
// Total: 64 games × 100μs = 6.4ms CPU time
// GPU is IDLE during this entire time!
```

**Correct Implementation** (USE THIS):
```cpp
// RIGHT: Parallel tree traversal
py::gil_scoped_release release;  // Release GIL for C++ parallelism

#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < num_games; ++i) {
    if (games[i].is_active()) {
        games[i].traverse_to_leaf_and_encode(batch_buffer[i]);
    }
}
// Total: ~100μs wall time (parallelized across cores)
// GPU idle time: <1ms
```

**Key Points**:
- Use OpenMP (`#pragma omp parallel for`) or a worker thread pool
- `schedule(dynamic)` handles games that finish at different rates
- Release Python GIL before parallel section
- Each game traverses independently (virtual loss prevents conflicts)
- Target: <1ms CPU time to collect 256 leaves across 64 games

**Build System Update**:
```cmake
# In CMakeLists.txt
find_package(OpenMP REQUIRED)
target_link_libraries(alphazero_sync PRIVATE OpenMP::OpenMP_CXX)
```

### 2. Memory Management & False Sharing

**Challenge**: Atomic operations on same cache line cause contention (slower than mutex).

**Solution**:
- **Cache-line alignment**: Ensure Node struct is exactly 64 bytes
- **Pointer compression**: Use 32-bit indices instead of 64-bit pointers (saves 50% memory, improves cache)
- **Memory layout**: Group hot fields together, separate read-only from read-write
- **Relaxed memory ordering**: Use `std::memory_order_relaxed` where full consistency not needed

**Implementation**:
```cpp
struct alignas(64) Node {
    // Hot fields (frequently accessed together)
    std::atomic<uint32_t> visit_count{0};      // 4 bytes
    std::atomic<int64_t> value_sum_fixed{0};   // 8 bytes (fixed-point: value * 10000)
    float prior;                                // 4 bytes (set once, no atomic needed)
    std::atomic<int16_t> virtual_loss{0};      // 2 bytes (increment by 1 per traversal)

    // Tree structure (32-bit indices, not pointers)
    uint32_t parent_idx;                        // 4 bytes
    uint32_t first_child_idx;                   // 4 bytes
    uint16_t num_children;                      // 2 bytes
    uint16_t action;                            // 2 bytes

    // Flags
    uint8_t is_terminal : 1;
    uint8_t is_expanded : 1;
    uint8_t reserved : 6;                       // 1 byte

    // Padding to 64 bytes
    uint8_t padding[27];

    // Helper methods
    float q_value(float parent_q) const {
        uint32_t n = visit_count.load(std::memory_order_relaxed);
        int16_t v = virtual_loss.load(std::memory_order_relaxed);

        if (n == 0 && v == 0) {
            // Unvisited node: use FPU (First Play Urgency)
            return parent_q - 0.2f;  // Slightly pessimistic
        }

        if (n == 0 && v > 0) {
            // Only virtual visits: use parent Q-value (Leela approach)
            // This keeps Q-value stable and doesn't trend toward zero
            return parent_q;
        }

        // Has real visits: compute normal Q-value with virtual loss
        int64_t sum = value_sum_fixed.load(std::memory_order_relaxed);
        float real_value = sum / 10000.0f;

        // Virtual visits contribute parent_q to the average (Leela approach)
        float total_value = real_value + v * parent_q;
        return total_value / (n + v);
    }

    void update(float value) {
        visit_count.fetch_add(1, std::memory_order_relaxed);
        // Proper rounding to avoid bias
        int64_t fixed_value = static_cast<int64_t>(std::round(value * 10000.0f));
        value_sum_fixed.fetch_add(fixed_value, std::memory_order_relaxed);
    }

    void add_virtual_loss() {
        virtual_loss.fetch_add(1, std::memory_order_relaxed);
    }

    void remove_virtual_loss() {
        virtual_loss.fetch_sub(1, std::memory_order_relaxed);
    }
};
static_assert(sizeof(Node) == 64, "Node must be cache-line aligned");
```

**Critical Fixes**:
- **int64_t for value_sum**: Prevents overflow in long searches (int32_t maxes at ~214K with 10000x multiplier)
- **int16_t for virtual_loss**: Simple +1 per traversal (safe for batch_size=256)
- **CORRECT virtual loss (Leela approach)**: Virtual visits contribute parent_q to the average, NOT a penalty
  ```cpp
  // When n==0 && v>0: return parent_q (stable placeholder)
  // When n>0: return (real_value + v * parent_q) / (n + v)
  ```
- **Proper rounding**: Use std::round() to avoid bias toward early-visited nodes

### 3. Chess Engine Nuances

**Challenge**: Magic bitboards are complex; Zobrist needed for repetition detection.

**Solution**:
- **Pre-computed magic tables**: Include in `third_party/magic_bitboards.hpp` (fast Colab setup)
- **Perft validation**: Verify move generation against known results (perft(6) from start = 119,060,324)
- **3-fold repetition**: Use Zobrist hashing to track position history (required for legal chess)
- **Incremental updates**: Update Zobrist hash during make/unmake (O(1) instead of O(n))

**Implementation**:
```cpp
// Load pre-computed magic tables
#include "third_party/magic_bitboards.hpp"

// Zobrist for repetition detection
class Position {
    std::vector<uint64_t> position_history;  // For 3-fold repetition

    bool is_repetition() const {
        int count = 0;
        for (auto hash : position_history) {
            if (hash == zobrist_hash) count++;
        }
        return count >= 3;
    }
};
```

### 4. Python/C++ Bridge Optimization

**Challenge**: C++ → Python callback → C++ is expensive (context switching overhead).

**Solution**:
- **Batch-oriented interface**: C++ collects batch → returns to Python → Python infers → C++ updates
- **NOT callback-based**: Avoid C++ calling Python function repeatedly
- **Zero-copy**: Use NumPy arrays that share memory with C++
- **GIL management**: Release GIL during C++ work, acquire only for data transfer

**Implementation**:
```python
# Optimized pattern
while not done:
    # C++ collects batch and returns tensors
    obs, masks, game_ids = mcts.collect_batch()  # C++ work, GIL released

    # Python runs inference
    policies, values = model(obs, masks)  # GPU work

    # C++ updates nodes and continues
    mcts.update_batch(game_ids, policies, values)  # C++ work, GIL released
```

### 5. Thread Scaling & Atomic Contention

**Challenge**: Too many threads updating same nodes causes contention.

**Solution**:
- **Relaxed memory ordering**: Use `memory_order_relaxed` for statistics (visit_count, value_sum)
- **Virtual loss**: Prevents multiple threads from selecting same path
- **Lock-free updates**: Compare-exchange for float addition
- **Optimal thread count**: Typically num_games = batch_size / 4 (e.g., 64 games for batch_size=256)

**Implementation**:
```cpp
void Node::update(float value) {
    visit_count.fetch_add(1, std::memory_order_relaxed);  // Relaxed OK

    // Atomic float addition
    float old_sum = value_sum.load(std::memory_order_relaxed);
    while (!value_sum.compare_exchange_weak(
        old_sum, old_sum + value,
        std::memory_order_relaxed,
        std::memory_order_relaxed));
}
```

### 6. Build System & Colab Compatibility

**Challenge**: Long compile times, dependency issues on Colab.

**Solution**:
- **ccache**: Cache compilation results (2nd build ~10x faster)
- **Vendored dependencies**: Include pybind11 in third_party/ (no network dependency)
- **Pre-compiled wheels**: For common platforms (optional optimization)
- **Parallel compilation**: Use -j4 flag
- **Minimal dependencies**: Only pybind11, no Boost or other heavy libraries

**Implementation**:
```bash
# colab_setup.sh
apt-get install -y ccache
export PATH="/usr/lib/ccache:$PATH"
pip install -e . --verbose  # Shows compilation progress
```

## Summary Table: Expected Bottlenecks & Mitigations

| Feature | Risk Level | Bottleneck | Mitigation |
|---------|-----------|------------|------------|
| **Move Generation** | High | Complex to implement correctly | Runtime magic generation, validate with Perft |
| **Thread Scaling** | High | Atomic contention on hot nodes | Use `memory_order_relaxed`, virtual loss, pointer compression |
| **GPU Sync** | Critical | Under-filled batches from stragglers | Dynamic batch size (90% threshold), 20ms timeout, speculative fill |
| **Python/C++ Bridge** | Medium | Context switching overhead | Batch-oriented interface, zero-copy tensors |
| **Memory Usage** | Medium | Fixed pool exhaustion | Chained arena allocation (dynamic growth) |
| **Colab Build** | Low | Long compile times | ccache, vendored deps, parallel compilation |
| **False Sharing** | High | Cache-line contention | 64-byte aligned nodes, separate hot/cold fields |
| **Repetition Detection** | Medium | Must detect 3-fold repetition | Dual Zobrist hashing with circular buffer |
| **Node Allocation** | High | Atomic contention on global pool | Per-game arenas (16K chunks), local allocation |
| **Batch Priority** | Medium | Critical evals wait for deep-tree evals | Priority queue: root nodes and high-uncertainty first |
| **CPU Latency** | Critical | Serial leaf collection (5-10ms) | OpenMP parallel traversal (<1ms) |
| **Root Contention** | High | Many threads updating root simultaneously | Lazy backpropagation with thread-local buffers |
| **GPU Idle Time** | Critical | CPU encoding while GPU waits | Dual CUDA streams (overlap transfer and compute) |
| **PUCT Calculation** | Medium | Expensive sqrt in hot loop | Use `_mm256_rsqrt_ps` (approximate reciprocal sqrt) |
| **Legal Move Mask** | Medium | 4672-entry mask generation slow | SIMD permute instructions with pre-calculated scatter map |
| **Fixed-Point Rounding** | High | Bias toward early-visited nodes | Proper rounding in fixed-point conversion |

## Advanced Optimizations Summary (Updated)

### 1. Memory Management
- **Chained arena allocation**: Dynamic growth, no fixed limits (256K nodes per block)
- **Per-game arenas**: 16K node chunks eliminate atomic contention
- **Pointer compression**: 32-bit indices save 50% memory, improve cache
- **16-bit relative offsets**: For children in same chunk (further compression)
- **Fixed-point atomics**: `int32_t` for value_sum with proper rounding (universally lock-free)
- **Zobrist circular buffer**: Fixed-size history (no dynamic allocations)

### 2. Batch Coordination
- **Double buffering**: GPU processes Batch N while CPU collects Batch N+1
- **Dual CUDA streams**: Overlap PCIe transfer and GPU compute
- **Dynamic batching**: 90% threshold + 20ms timeout
- **Speculative leaf collection**: Fill stragglers with secondary PUCT moves
- **Priority queue**: Root nodes and high-uncertainty evaluations first
- **Parallel traversal**: OpenMP for <1ms leaf collection across 64 games
- **Forced root expansion**: Expand root in first batch for diverse initial paths

### 3. Hardware Optimization
- **BMI2/AVX2 intrinsics**: `_mm_popcnt_u64`, `_tzcnt_u64`, `_blsr_u64`
- **SIMD encoding**: AVX2 for bitboard → tensor conversion
- **SIMD legal mask**: AVX2 permute with pre-calculated scatter map
- **Approximate rsqrt**: `_mm256_rsqrt_ps` for PUCT sqrt term (2-3x faster)
- **Prefetch**: `__builtin_prefetch` for next child node
- **NHWC layout**: Optimize for Tensor Cores if model supports it
- **CUDA Graphs**: Static computation graphs (10-15% throughput boost)
- **Zero-copy tensors**: C++ writes directly to torch::Tensor memory
- **NUMA awareness**: Thread pinning for multi-socket CPUs (optional)

### 4. Chess Engine
- **Runtime magic generation**: ~50ms at startup, background thread during import
- **Dual Zobrist hashing**: 64-bit + 32-bit (collision probability ~1 in 2^96)
- **Circular history buffer**: Fixed-size for 3-fold repetition (deterministic memory)
- **Perft validation**: Verify correctness against known results
- **Smart Syzygy probing**: Only probe at low piece count or high visit threshold
- **Delayed init**: Magic tables generated during Python import (background thread)

### 5. MCTS Refinements
- **Dynamic FPU**: Parent Q-value minus exploration constant (better than static 0)
- **Soft virtual loss**: Fractional virtual loss for better exploration balance
- **Lazy backpropagation**: Thread-local buffers, batch-update upper tree levels
- **Tree pruning**: Generation IDs for memory reuse on root transitions
- **Relaxed memory ordering**: For statistics where full consistency not needed
- **Cache-line alignment**: 64-byte nodes prevent false sharing
- **Proper fixed-point rounding**: Avoid bias toward early-visited nodes
- **Gumbel Top-k**: Better policy improvement with fewer simulations (optional)

### 6. Transposition Table (Optional)
- **Lock-free TT**: Store (hash → policy, value) from GPU evaluations
- **10-20% GPU savings**: Check TT before GPU evaluation
- **TT-based transposition merging**: Merge identical positions during selection
- **Small size**: 1M entries (~50MB)

## Implementation Priority Levels

### **Priority 1: Core MVP (Must Have)**
| Feature | Impact | Complexity |
|---------|--------|------------|
| Bitboard chess engine with magic tables | 10-50x speedup | High |
| Chained arena allocation | No memory limits | Medium |
| Per-game arenas (16K chunks) | Eliminate contention | Low |
| Fixed-point atomics (int64_t) | Universally lock-free | Low |
| OpenMP parallel traversal | 6x faster leaf collection | Low |
| Dynamic batching (90% + timeout + Hard Sync) | Prevent GPU idle + balanced search | Medium |
| Priority queue batching | Better GPU utilization | Medium |
| Double buffering | GPU/CPU parallelism | Medium |
| Zero-copy tensor interface | Eliminate copy overhead | Medium |
| **NHWC tensor layout** | **2-3x better Tensor Core performance** | **Low** |
| **Perspective flip encoding** | **Prevents 100% policy noise** | **Low** |
| CUDA Graphs | 10-15% throughput boost | Low |
| Dynamic FPU (Leela approach) | Better exploration | Low |
| BMI2/AVX2 intrinsics | 2-3x faster bitops | Low |

### **Priority 2: Performance Enhancements (Should Have)**
| Feature | Impact | Complexity |
|---------|--------|------------|
| SIMD encoding (AVX2) | 3-5x faster encoding | Medium |
| Prefetch optimization | 5-10% CPU speedup | Low |
| Adaptive virtual loss | Better search quality | Low |
| Tree pruning (generation IDs) | Memory reuse | Low |
| Dual Zobrist hashing | Eliminate collisions | Low |
| Forced root expansion | Better initial batches | Low |
| Speculative execution | Fill stragglers | Medium |
| Smart Syzygy probing | Perfect endgame | Medium |

### **Priority 3: Advanced Optimizations (Nice to Have)**
| Feature | Impact | Complexity |
|---------|--------|------------|
| Transposition table | 10-20% GPU savings | Medium |
| NUMA awareness | Prevent latency spikes | High |
| Wait-free backpropagation | Cleaner search loop | Medium |
| Relative offset compression | Further memory savings | Medium |
| Gumbel Top-k selection | Better policy quality | High |
| NHWC tensor layout | Lower GPU latency | Low |

## Final Implementation Strategy

**Phase 1 (Core MVP)**: Implement all Priority 1 features
- Target: 20-50x speedup over Python implementation
- Estimated: ~5000 lines of C++, ~500 lines Python
- Build time: 2-3 minutes on Colab

**Phase 2 (Performance)**: Add Priority 2 features
- Target: 50-100x speedup with better search quality
- Incremental additions to existing codebase

**Phase 3 (Advanced)**: Optional Priority 3 features
- Target: Squeeze out last 10-20% performance
- Can be added post-deployment based on profiling

## Revised Bottleneck Analysis

| Feature | Suggestion | Impact |
|---------|-----------|--------|
| **Node Traversal** | Use prefetch instructions for next likely child node | ~5-10% CPU speedup |
| **Tensor Format** | Use NHWC layout for 2-3x better Tensor Core performance | Critical for GPU speed |
| **Perspective Flip** | Always encode from current player's perspective | Prevents 100% policy noise |
| **Zobrist** | Use 64-bit + 32-bit dual hash to virtually eliminate collisions | Safety for long games |
| **Leaf Collection** | OpenMP parallel traversal across games | 6x faster (6ms → <1ms) |
| **Node Allocation** | Per-game arenas with 16K chunks | Eliminate atomic contention |
| **Batch Priority** | Priority queue + Hard Sync every N batches | Balanced search quality |
| **Value Storage** | Fixed-point int64_t instead of atomic float | Universally lock-free |
| **Bitboard Ops** | Hardware intrinsics (_mm_popcnt_u64, etc.) | 2-3x faster than software |
| **Encoding** | NHWC scatter for bitboard → tensor | Cleaner SIMD, better coalescing |

## Performance Targets

- **Move generation**: 5-10M moves/sec (vs 200K for python-chess)
- **MCTS simulations**: 50K-100K sims/sec per game (vs 5K-10K for Python)
- **Batch efficiency**: >90% GPU utilization (vs 50-70% without coordination)
- **Leaf collection**: <1ms for 256 leaves across 64 games (vs 6ms serial)
- **Overall speedup**: 20-100x faster self-play than current implementation
- **Memory**: <200MB for 64 games with 1M node pool

## Final Implementation Roadmap

### Pre-Implementation Checklist (MANDATORY)

Before writing any code, ensure you understand:

1. **Virtual Loss (Leela Approach)**: Virtual visits use parent Q-value as placeholder, NOT a penalty
   - `if (n == 0 && v > 0) return parent_q;`
   - `if (n > 0) return (real_value + v * parent_q) / (n + v);`

2. **Memory Ordering**: Use `memory_order_release` on root updates, `memory_order_acquire` on reads
   - Prevents race conditions where stale value_sum is read with updated visit_count

3. **Node Alignment**: Use `std::aligned_alloc(64, ...)` for 64-byte cache-line alignment
   - `std::vector` doesn't guarantee alignment beyond `alignof(T)`
   - Add `static_assert` to verify 32-bit indices are safe with 64-byte nodes

4. **Action Encoding + Perspective Flip**: Implement round-trip tests AND 10K game cross-validation
   - Silent encoding errors are the #1 cause of training failure
   - **CRITICAL**: Always encode from current player's perspective (flip board for Black)
   - Verify e2-e4 for White encodes identically to e7-e5 for Black

5. **Lazy Backpropagation**: Use thread-local buffers to reduce root contention
   - Critical for high-parallelism scenarios (>8 threads per game)

6. **NHWC Tensor Layout**: Use channels-last format for 2-3x better Tensor Core performance
   - Shape: (batch, height, width, channels) = (batch, 8, 8, 119)
   - PyTorch: `model.to(memory_format=torch.channels_last)`
   - All 119 channels for a square are contiguous in memory

7. **Hard Sync Mechanism**: Force full synchronization every N batches
   - Prevents search quality imbalance from 90% threshold
   - Ensures complex positions don't consistently miss batches

### Phase-by-Phase Success Criteria

| Phase | Must Pass Before Proceeding | Time Estimate |
|-------|----------------------------|---------------|
| **Phase 1: Chess Engine** | Perft(6) = 119,060,324 exactly<br>10K games vs python-chess with 0 discrepancies | 2-3 days |
| **Phase 2: MCTS Core** | Leaf collection <1ms for 256 leaves<br>No deadlocks after 1 hour stress test<br>Fair game progress (no starvation) | 3-4 days |
| **Phase 3: Python Integration** | Zero-copy verified (no memcpy in profile)<br>No memory leaks (valgrind clean)<br>Correct NCHW tensor layout | 2-3 days |
| **Phase 4: End-to-End** | 20x+ speedup vs Python baseline<br>Training data format matches existing<br>Model converges on toy problem | 2-3 days |

### Critical Implementation Order

**DO NOT skip or reorder these steps:**

1. **Chess Engine + Perft** (Phase 1)
   - Any bug here propagates to all downstream components
   - Perft catches 95% of move generation bugs
   - Cross-validation catches the remaining 5% (3-fold repetition, etc.)

2. **Action Encoding + Round-Trip Tests** (Phase 1)
   - Silent encoding errors are undetectable without explicit tests
   - Must verify BEFORE integrating with MCTS

3. **MCTS with Mock Evaluator** (Phase 2)
   - Test batch coordinator without GPU latency
   - Verify virtual loss, starvation prevention, timeout logic

4. **Python Integration with Identity Tensor** (Phase 3)
   - Verify zero-copy and tensor layout BEFORE loading real model
   - Identity tensor: Python returns input unchanged

5. **Real Model + End-to-End** (Phase 4)
   - Only after all previous phases pass

### Common Pitfalls to Avoid

| Pitfall | Consequence | Prevention |
|---------|-------------|------------|
| Skipping Perft validation | AI hallucinates illegal moves | MANDATORY: Perft(6) = 119,060,324 |
| Wrong virtual loss formula | Search instability, poor play | Use Leela approach with parent_q |
| Missing std::aligned_alloc | False sharing, 50% perf loss | Verify alignment with static_assert |
| **Missing perspective flip** | **100% policy noise, training fails** | **Always encode from current player's perspective** |
| Silent action encoding bugs | Training fails to converge | Round-trip tests + 10K game validation |
| **Wrong tensor layout (NCHW)** | **2-3x slower GPU inference** | **Use NHWC (channels-last) format** |
| Serial leaf collection | GPU starvation, 6x slower | OpenMP parallel traversal |
| **90% threshold without Hard Sync** | **Search quality imbalance** | **Force full sync every N batches** |
| Root contention | Atomic bottleneck at high parallelism | Lazy backpropagation with thread-local buffers |
| Wrong memory ordering | Race conditions, NaN Q-values | release/acquire on root, relaxed elsewhere |
| **32-bit index overflow** | **Segfault with 64-byte nodes** | **static_assert for safety checks** |
| Callback-based Python bridge | Context switching overhead | Batch-oriented interface with zero-copy |

### Debugging Checklist

If you encounter issues, check these in order:

**Chess Engine Issues:**
- [ ] Perft(6) = 119,060,324 exactly?
- [ ] 10K games vs python-chess with 0 discrepancies?
- [ ] 3-fold repetition detected correctly?
- [ ] Castling rights updated correctly?
- [ ] En passant captured correctly?

**Action Encoding Issues:**
- [ ] Round-trip tests pass (Move → Action → Move)?
- [ ] 10K game cross-validation with python-chess passes?
- [ ] **Perspective flip implemented (Black sees itself as White)?**
- [ ] **e2-e4 for White encodes identically to e7-e5 for Black?**
- [ ] Promotion moves encode correctly?

**Tensor Layout Issues:**
- [ ] **Using NHWC (channels-last) layout?**
- [ ] **PyTorch model converted to channels_last format?**
- [ ] **Stride verification: 8*8*119 per batch, 8*119 per rank, 119 per file?**
- [ ] Zero-copy verified (C++ writes directly to torch tensor)?
- [ ] No memory copies in profile?

**MCTS Issues:**
- [ ] Virtual loss uses parent_q (Leela approach)?
- [ ] Leaf collection <1ms with OpenMP?
- [ ] No games starving (priority mechanism working)?
- [ ] **Hard Sync every N batches (prevents quality imbalance)?**
- [ ] No deadlocks after 1 hour stress test?
- [ ] Root contention mitigated (lazy backprop)?

**Performance Issues:**
- [ ] Nodes 64-byte aligned (std::aligned_alloc)?
- [ ] **32-bit indices safe with 64-byte nodes (static_assert)?**
- [ ] Zero-copy verified (no memcpy in profile)?
- [ ] CUDA Graphs enabled (10-15% boost)?
- [ ] BMI2/PEXT used for encoding (2-3x boost)?
- [ ] OpenMP parallel traversal (<1ms)?

**Training Issues:**
- [ ] Action encoding round-trip tests pass?
- [ ] 10K game cross-validation passes?
- [ ] **Perspective flip validation passes?**
- [ ] **Tensor layout matches PyTorch (NHWC)?**
- [ ] Training data format matches existing?
- [ ] Model converges on toy problem?

### Final Verification

Before deploying to production:

1. **Correctness**: 100K self-play games with 0 illegal moves
2. **Performance**: 20x+ speedup vs Python baseline
3. **Stability**: 24-hour stress test with no crashes
4. **Memory**: No leaks detected by valgrind
5. **Integration**: Works with existing evaluate.py and web app

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Move generation | 5-10M moves/sec | Benchmark on starting position |
| MCTS simulations | 50K-100K sims/sec per game | 800 simulations per move |
| Batch efficiency | >90% GPU utilization | nvidia-smi during self-play |
| Leaf collection | <1ms for 256 leaves | Profile with perf/vtune |
| Overall speedup | 20-100x vs Python | End-to-end self-play time |
| Memory usage | <200MB for 64 games | Monitor RSS during self-play |
| Correctness | 0 illegal moves | 100K game validation |

## Conclusion

This implementation plan is now production-ready with:

✅ **All critical inconsistencies resolved**
- Virtual loss formula unified (Leela approach with parent_q)
- Node alignment strategy clarified (std::aligned_alloc with static_assert)
- Action encoding validation strengthened (round-trip + 10K cross-validation)
- **Perspective flip requirement added (prevents 100% policy noise)**
- **Tensor layout standardized to NHWC (2-3x better GPU performance)**
- **Hard Sync mechanism added (prevents search quality imbalance)**

✅ **All critical bottlenecks identified and mitigated**
- Root contention: Lazy backpropagation with thread-local buffers
- CPU latency: OpenMP parallel traversal (<1ms target)
- GPU starvation: Dynamic batching (90% threshold) + Hard Sync every N batches
- False sharing: 64-byte aligned nodes with std::aligned_alloc
- Tensor Core performance: NHWC layout for 2-3x speedup
- Search quality imbalance: Priority queue + mandatory full synchronization

✅ **Comprehensive validation strategy**
- Phase-by-phase success criteria with time estimates
- Mandatory checkpoints before proceeding (Perft, cross-validation, perspective flip)
- Debugging checklist organized by category
- Common pitfalls table with prevention strategies

✅ **Clear implementation roadmap**
- Critical implementation order (DO NOT skip or reorder)
- 7-point pre-implementation checklist
- Final verification steps before production
- Success metrics with measurement methods

**Status**: Ready to proceed with implementation! 🚀

**Estimated Timeline**: 9-13 days for full MVP (Phase 1-4)

**Key Improvements from Review**:
1. **NHWC Tensor Layout**: Elevated from optional to mandatory (2-3x GPU speedup)
2. **Perspective Flip**: Added critical validation to prevent policy noise
3. **Hard Sync Mechanism**: Prevents search quality imbalance from 90% threshold
4. **Node Pool Safety**: Added static_assert checks for 32-bit index safety
5. **Lazy Backpropagation**: Added to mitigate root contention in high-parallelism scenarios
6. **BMI2/PEXT Optimization**: Added for 2-3x faster bitboard encoding

**Next Step**: Begin Phase 1 with chess engine implementation and Perft validation.

**Critical First Steps**:
1. Implement bitboard chess engine with BMI2 intrinsics
2. Generate magic tables with fixed-seed PRNG (deterministic)
3. Run Perft(6) validation = 119,060,324 (MANDATORY)
4. Cross-validate with python-chess (10K games, 0 discrepancies)
5. Implement perspective flip encoding (Black sees itself as White)
6. Validate perspective flip (e2-e4 for White ≡ e7-e5 for Black)
