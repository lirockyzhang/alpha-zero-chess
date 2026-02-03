Comprehensive Exploration Report: alphazero-cpp/ Directory

1. C++ Components Implemented

The alphazero-cpp/ directory contains a complete, production-ready C++ chess engine with MCTS and neural
network integration:

Chess Engine (include/chess/ and src/chess/):
- bitboard.hpp/cpp - Bitboard representation for fast move generation
- move.hpp/cpp - Move representation and validation
- position.hpp/cpp - Full chess position state management
- movegen.hpp/cpp - Legal move generation
- zobrist.hpp/cpp - Zobrist hashing for position caching

Encoding (include/encoding/ and src/encoding/):
- position_encoder.hpp/cpp - Converts chess positions to 119-plane neural network input (NHWC layout:
8×8×119)
- move_encoder.hpp/cpp - Encodes/decodes moves to/from 4672-dimensional policy space
- Batch encoding with OpenMP parallelization (5.45x speedup, <1ms for 256 positions)
- Zero-copy interface for GPU memory efficiency

MCTS (include/mcts/ and src/mcts/):
- node.hpp - Cache-line aligned (64-byte) MCTS nodes with atomic operations
- node_pool.hpp - Memory pool for efficient node allocation
- search.hpp/cpp - Single-threaded MCTS with PUCT formula
- batch_search.hpp/cpp - Batched MCTS that collects leaves for batch NN evaluation
- batch_coordinator.hpp/cpp - Coordinates parallel games with 90% threshold + Hard Sync mechanism

Performance Metrics (from PHASE3_FINAL_SUMMARY.md):
- Chess engine: 189-422M nodes/sec (20-40x faster than Python)
- MCTS simulations: 362K NPS (9x faster)
- Position encoding: 5.9μs per position (17x faster)
- Batch encoding: 0.272ms for 256 positions (5.45x speedup)

---
2. Python Bindings (pybind11)

File: src/bindings/python_bindings.cpp (200+ lines)

Exported Classes:
1. PyMCTSSearch - Single-threaded MCTS wrapper
    - search(fen, policy, value) → returns visit counts as numpy array
    - select_move(temperature) → selects best move
    - reset() → clears search tree
2. PyBatchCoordinator - Multi-game batch coordinator
    - add_game(game_id, fen) - Add game to batch
    - is_game_complete(game_id) - Check if game finished
    - remove_game(game_id) - Remove completed game
    - get_stats() - Get batch statistics
3. PyBatchedMCTSSearch - Proper AlphaZero with leaf batching (200+ lines, partially shown)
    - init_search(fen, policy, value) - Initialize search
    - collect_leaves(obs_buffer, mask_buffer, max_batch_size) - Collect pending leaves
    - update_leaves(policies, values) - Update with NN results
    - is_search_complete() - Check if search done
    - get_visit_counts() - Get policy target

Exported Functions:
- encode_position(fen) → numpy array (8, 8, 119) NHWC layout
- encode_position_to_buffer(fen, buffer) → zero-copy encoding
- encode_batch(fens, buffer, use_parallel) → batch encoding with OpenMP
- move_to_index(uci_move, fen) → policy index
- index_to_move(index, fen) → UCI move string

Module Name: alphazero_cpp (built by CMake with pybind11)

---
3. alphazero-cpp/train.py vs scripts/train.py

alphazero-cpp/train.py (22,453 bytes):
- Standalone training script that uses C++ MCTS directly
- Imports alphazero_cpp module (C++ bindings)
- Implements minimal AlphaZero network (64 filters, 5 residual blocks)
- Uses BatchedEvaluator class for batched GPU inference
- Implements ReplayBuffer and TrainingDataset locally
- Self-contained: doesn't depend on alphazero/ package
- Designed for fast iteration with C++ backend

scripts/train.py (19,372 bytes):
- Uses the full alphazero/ package infrastructure
- Imports from alphazero.config, alphazero.neural, alphazero.selfplay, alphazero.mcts
- Uses SelfPlayCoordinator and BatchedSelfPlayCoordinator for multi-actor training
- Supports multiple MCTS backends (Python, Cython, C++)
- Implements iterative training with buffer refresh
- More modular and production-ready
- Supports inference server and actor processes

Key Differences:
┌───────────────┬───────────────────────────────────────┬────────────────────────────────────────────┐
│    Aspect     │        alphazero-cpp/train.py         │              scripts/train.py              │
├───────────────┼───────────────────────────────────────┼────────────────────────────────────────────┤
│ Dependencies  │ Minimal (torch, chess, alphazero_cpp) │ Full alphazero/ package                    │
├───────────────┼───────────────────────────────────────┼────────────────────────────────────────────┤
│ MCTS Backend  │ C++ only                              │ Pluggable (Python/Cython/C++)              │
├───────────────┼───────────────────────────────────────┼────────────────────────────────────────────┤
│ Architecture  │ Standalone, self-contained            │ Modular, uses package components           │
├───────────────┼───────────────────────────────────────┼────────────────────────────────────────────┤
│ Training Loop │ Simple iterative                      │ Sophisticated with actors/inference server │
├───────────────┼───────────────────────────────────────┼────────────────────────────────────────────┤
│ Use Case      │ Fast prototyping/testing              │ Production training                        │
└───────────────┴───────────────────────────────────────┴────────────────────────────────────────────┘
---
4. C++ Architecture & Design Patterns

Key Design Patterns:

1. Cache-Line Alignment (Node structure)
    - 64-byte aligned nodes to prevent false sharing
    - Atomic operations for thread-safe MCTS
    - Fixed-point arithmetic (×10000) for precision without floats
2. Memory Pooling (NodePool)
    - Pre-allocated node pool for O(1) allocation
    - Reduces garbage collection overhead
    - Enables efficient tree reuse
3. Batching Architecture (BatchedMCTSSearch)
    - Collects leaves during selection phase
    - Batches them for single GPU call
    - Proper AlphaZero pattern: every leaf gets NN evaluation
4. Batch Coordination (BatchCoordinator)
    - 90% threshold: dispatch when 90% of games ready
    - Hard Sync: every 10th batch waits for ALL games
    - Prevents starvation while maintaining throughput
5. Zero-Copy Interface
    - Direct buffer writing for GPU memory
    - Eliminates data copies between C++ and Python
    - 1.81x speedup verified
6. NHWC Layout (Channels-Last)
    - Optimized for Tensor Cores (2-3x GPU speedup)
    - Contiguous channels per square
    - PyTorch channels_last compatible

---
5. Gaps: What's NOT in alphazero-cpp/

Missing from alphazero-cpp/ (compared to alphazero/):

1. Neural Network Training
    - alphazero-cpp/train.py has minimal network
    - No loss functions, optimizers, or training loops in C++
    - Relies on PyTorch for all NN operations
2. Self-Play Infrastructure
    - No selfplay/ module equivalent
    - No actor/coordinator pattern
    - No inference server
    - alphazero-cpp/train.py uses simple sequential games
3. Evaluation Framework
    - No evaluation/ module equivalent
    - No arena.py, stockfish.py, or endgame_eval.py
    - No tournament/rating system
4. Training Utilities
    - No training/learner.py equivalent
    - No metrics_logger.py, replay_buffer.py, trajectory.py
    - No training_monitor.py
5. Configuration System
    - No config.py equivalent
    - No MCTSConfig, NetworkConfig, TrainingConfig classes
    - No profile system (PROFILES, TrainingProfile)
6. Visualization & Web
    - No visualization/ module
    - No web/ module
    - No web app for monitoring training
7. Advanced MCTS Features
    - No Cython backend equivalent
    - No Python backend for comparison
    - No evaluator.py (NetworkEvaluator, RandomEvaluator, CachedEvaluator)
8. Move Encoding Variants
    - Only one move encoder (4672 moves)
    - No moves_cpp_aligned.py equivalent in C++
    - No alternative encodings

---
6. Current State Summary

What's Complete:
- ✅ Full chess engine (189-422M nps)
- ✅ Position encoding (NHWC, 5.9μs)
- ✅ Move encoding (4672 moves)
- ✅ MCTS with batching (362K NPS)
- ✅ Batch coordinator (90% threshold + Hard Sync)
- ✅ Python bindings (pybind11)
- ✅ Zero-copy GPU interface
- ✅ OpenMP parallelization
- ✅ Comprehensive testing (cross-validation with python-chess)

What's Partial:
- ⚠️ Training script (standalone, not integrated with alphazero/)
- ⚠️ Neural network (minimal, not production-grade)

What's Missing:
- ❌ Integration with alphazero/ package
- ❌ Self-play infrastructure (actors, coordinators)
- ❌ Evaluation framework
- ❌ Training utilities and monitoring
- ❌ Configuration system
- ❌ Visualization and web interface

---
7. File Structure

alphazero-cpp/
├── CMakeLists.txt                    # Build configuration
├── train.py                          # Standalone training script
├── include/
│   ├── chess/                        # Chess engine headers
│   │   ├── bitboard.hpp
│   │   ├── move.hpp
│   │   ├── position.hpp
│   │   ├── movegen.hpp
│   │   └── zobrist.hpp
│   ├── encoding/                     # Encoding headers
│   │   ├── position_encoder.hpp      # 119-plane NHWC encoder
│   │   └── move_encoder.hpp          # 4672-move encoder
│   └── mcts/                         # MCTS headers
│       ├── node.hpp                  # 64-byte aligned nodes
│       ├── node_pool.hpp
│       ├── search.hpp                # Single-threaded MCTS
│       ├── batch_search.hpp          # Batched MCTS
│       └── batch_coordinator.hpp     # Multi-game coordinator
├── src/
│   ├── chess/                        # Chess engine implementations
│   ├── encoding/                     # Encoding implementations
│   ├── mcts/                         # MCTS implementations
│   └── bindings/
│       └── python_bindings.cpp       # pybind11 exports
├── tests/                            # C++ and Python tests
│   ├── *.cpp                         # C++ unit tests
│   └── *.py                          # Python integration tests
├── third_party/
│   └── chess-library/                # External chess library
└── build/                            # CMake build directory

---
Conclusion

alphazero-cpp/ is a complete, high-performance C++ implementation of the core AlphaZero components (chess
engine, MCTS, encoding). It achieves 20-40x speedups over Python and is production-ready for the
computational bottlenecks.

However, it's not a standalone training system - it's designed as a performance backend for the main
alphazero/ package. The alphazero-cpp/train.py is a proof-of-concept, not the primary training entry point.
The full training infrastructure (actors, coordinators, evaluation, monitoring) lives in alphazero/ and uses
the C++ backend via pybind11 bindings.

To make alphazero-cpp/ a complete standalone implementation, you would need to port the selfplay/, training/,
and evaluation/ modules to C++ or create C++ wrappers for them.