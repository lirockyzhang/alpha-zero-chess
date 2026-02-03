# Parallel Self-Play Implementation Plan

## Overview

Transform AlphaZero training from a **latency-bound** system (GPU waits for single MCTS) to a **throughput-bound** system (GPU fed constant stream of cross-game data).

**Target**: 4-6x speedup, GPU utilization 80-95%

---

## Architecture Comparison

### Current (Sequential, Latency-Bound)
```
Game 1: [MCTS]→[GPU 64]→[MCTS]→[GPU 64]→...→[Complete]
Game 2:                                        [MCTS]→[GPU 64]→...
        ↑ GPU idle between batches, one game at a time
```

### Target (Parallel, Throughput-Bound)
```
Workers 1-16: ═══[MCTS]═══[MCTS]═══[MCTS]═══  (continuous)
                   ↓         ↓         ↓
              ┌────┴─────────┴─────────┴────┐
              │   Global Evaluation Queue   │
              │   (Pre-allocated, Pinned)   │
              └─────────────┬───────────────┘
                            ↓
GPU Thread:   ════[Batch 512]════[Batch 512]════  (continuous)
```

---

## Phase 1: Foundation (Existing Infrastructure)

### Already Implemented ✓
- `MCTSSearch` with batched leaf collection
- `Node` with **Virtual Loss** (atomic operations)
- `NodePool` arena allocator (per-worker isolation)
- `ReplayBuffer` lock-free circular buffer
- `SelfPlayCoordinator` multi-threaded orchestration

### Virtual Loss Verification

**Location**: `include/mcts/node.hpp` lines 125-132

```cpp
// Already implemented - ensures parallel MCTS correctness
void apply_virtual_loss() {
    virtual_loss.fetch_add(1, std::memory_order_relaxed);
}

void revert_virtual_loss() {
    virtual_loss.fetch_sub(1, std::memory_order_relaxed);
}
```

**Usage in Selection**: When a worker traverses a path to submit a leaf:
1. Apply virtual loss to all nodes in path
2. Submit leaf to evaluation queue
3. On result return, revert virtual loss and backpropagate actual value

---

## Phase 2: Global Evaluation Queue

### Design Principles

1. **Fixed-Size Buffers**: No heap allocation per request
2. **Pre-allocated Memory Pool**: Avoid fragmentation
3. **Pinned Memory**: Enable async GPU transfers
4. **Condition Variables**: Efficient batching with timeout

### New File: `include/selfplay/evaluation_queue.hpp`

```cpp
#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <array>
#include <chrono>

namespace alphazero {

// Constants
constexpr size_t OBS_SIZE = 8 * 8 * 122;      // 7808 floats
constexpr size_t POLICY_SIZE = 4672;
constexpr size_t MAX_QUEUE_SIZE = 2048;        // Max pending requests
constexpr size_t MAX_BATCH_SIZE = 512;         // Max GPU batch

// Single evaluation request (fixed-size, no heap allocation)
struct alignas(64) EvalRequest {
    int32_t worker_id;
    int32_t request_id;
    int32_t path_length;                       // For virtual loss revert
    std::array<float, OBS_SIZE> observation;   // Fixed-size array
    std::array<float, POLICY_SIZE> legal_mask;
    // Path stored separately in worker's local storage
};

// Evaluation result
struct EvalResult {
    int32_t worker_id;
    int32_t request_id;
    std::array<float, POLICY_SIZE> policy;
    float value;
};

// Pre-allocated ring buffer for requests
class EvalRequestPool {
public:
    EvalRequestPool(size_t capacity = MAX_QUEUE_SIZE);

    // Get next available slot (lock-free)
    EvalRequest* acquire();

    // Release slot back to pool
    void release(EvalRequest* req);

private:
    std::vector<EvalRequest> pool_;
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
};

// Thread-safe evaluation queue with smart batching
class GlobalEvaluationQueue {
public:
    GlobalEvaluationQueue(size_t max_batch_size = MAX_BATCH_SIZE);
    ~GlobalEvaluationQueue();

    // === Worker Thread Interface ===

    // Submit leaves for evaluation (called by workers)
    // Returns request IDs for tracking
    void submit_leaves(int worker_id,
                       const float* observations,    // N x 7808
                       const float* legal_masks,     // N x 4672
                       int num_leaves,
                       std::vector<int32_t>& out_request_ids);

    // Wait for results (blocking with timeout)
    // Returns number of results retrieved
    int get_results(int worker_id,
                    float* out_policies,             // N x 4672
                    float* out_values,               // N
                    int max_results,
                    int timeout_ms = 100);

    // === GPU Thread Interface ===

    // Collect batch for GPU evaluation
    // Blocks until: (1) batch_size reached OR (2) timeout
    // Returns pointer to pinned memory buffer
    int collect_batch(float** out_obs_ptr,           // Pinned memory
                      float** out_mask_ptr,          // Pinned memory
                      int timeout_ms = 5);

    // Submit GPU results back to workers
    void submit_results(const float* policies,       // N x 4672
                        const float* values,         // N
                        int num_results);

    // === Memory Access (Zero-Copy) ===

    // Get raw pointer to pinned observation buffer
    uintptr_t get_obs_buffer_ptr() const { return reinterpret_cast<uintptr_t>(pinned_obs_buffer_); }
    uintptr_t get_mask_buffer_ptr() const { return reinterpret_cast<uintptr_t>(pinned_mask_buffer_); }

    // === Metrics ===

    struct Metrics {
        std::atomic<uint64_t> total_batches{0};
        std::atomic<uint64_t> total_leaves{0};
        std::atomic<uint64_t> total_wait_time_us{0};
        std::atomic<uint64_t> gpu_idle_time_us{0};

        double avg_batch_size() const {
            return total_batches > 0 ?
                   static_cast<double>(total_leaves) / total_batches : 0;
        }

        double batch_fill_ratio() const {
            return avg_batch_size() / MAX_BATCH_SIZE;
        }
    };

    const Metrics& get_metrics() const { return metrics_; }

private:
    // Pinned memory buffers (cudaMallocHost)
    float* pinned_obs_buffer_;    // MAX_BATCH_SIZE x OBS_SIZE
    float* pinned_mask_buffer_;   // MAX_BATCH_SIZE x POLICY_SIZE
    float* pinned_policy_buffer_; // MAX_BATCH_SIZE x POLICY_SIZE
    float* pinned_value_buffer_;  // MAX_BATCH_SIZE

    // Request queue
    EvalRequestPool request_pool_;
    std::vector<EvalRequest*> pending_requests_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Result distribution (per-worker queues)
    static constexpr int MAX_WORKERS = 32;
    std::array<std::vector<EvalResult>, MAX_WORKERS> worker_results_;
    std::array<std::mutex, MAX_WORKERS> worker_mutexes_;
    std::array<std::condition_variable, MAX_WORKERS> worker_cvs_;

    // Batch tracking
    std::vector<std::pair<int, int>> current_batch_mapping_; // (worker_id, request_id)
    size_t max_batch_size_;

    // Metrics
    Metrics metrics_;

    // Lifecycle
    std::atomic<bool> shutdown_{false};
};

} // namespace alphazero
```

### Key Implementation Details

#### 1. Pinned Memory Allocation

```cpp
// In constructor
GlobalEvaluationQueue::GlobalEvaluationQueue(size_t max_batch_size)
    : max_batch_size_(max_batch_size)
{
    // Allocate pinned memory for async GPU transfers
    #ifdef USE_CUDA
    cudaMallocHost(&pinned_obs_buffer_, max_batch_size * OBS_SIZE * sizeof(float));
    cudaMallocHost(&pinned_mask_buffer_, max_batch_size * POLICY_SIZE * sizeof(float));
    cudaMallocHost(&pinned_policy_buffer_, max_batch_size * POLICY_SIZE * sizeof(float));
    cudaMallocHost(&pinned_value_buffer_, max_batch_size * sizeof(float));
    #else
    // Fallback to aligned allocation
    pinned_obs_buffer_ = static_cast<float*>(
        std::aligned_alloc(64, max_batch_size * OBS_SIZE * sizeof(float)));
    // ... etc
    #endif
}
```

#### 2. Smart Batching with Timeout

```cpp
int GlobalEvaluationQueue::collect_batch(float** out_obs_ptr,
                                          float** out_mask_ptr,
                                          int timeout_ms) {
    std::unique_lock<std::mutex> lock(queue_mutex_);

    auto start = std::chrono::steady_clock::now();

    // Wait until: batch full OR timeout OR shutdown
    queue_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
        return pending_requests_.size() >= max_batch_size_ ||
               shutdown_.load() ||
               (!pending_requests_.empty() && /* timeout check */);
    });

    if (shutdown_ || pending_requests_.empty()) {
        return 0;
    }

    // Collect up to max_batch_size requests
    int batch_size = std::min(pending_requests_.size(), max_batch_size_);
    current_batch_mapping_.clear();

    for (int i = 0; i < batch_size; ++i) {
        EvalRequest* req = pending_requests_[i];

        // Copy to pinned buffer (contiguous for GPU)
        std::memcpy(pinned_obs_buffer_ + i * OBS_SIZE,
                    req->observation.data(),
                    OBS_SIZE * sizeof(float));
        std::memcpy(pinned_mask_buffer_ + i * POLICY_SIZE,
                    req->legal_mask.data(),
                    POLICY_SIZE * sizeof(float));

        current_batch_mapping_.emplace_back(req->worker_id, req->request_id);
    }

    // Remove processed requests
    pending_requests_.erase(pending_requests_.begin(),
                           pending_requests_.begin() + batch_size);

    // Update metrics
    auto elapsed = std::chrono::steady_clock::now() - start;
    metrics_.gpu_idle_time_us +=
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    metrics_.total_batches++;
    metrics_.total_leaves += batch_size;

    *out_obs_ptr = pinned_obs_buffer_;
    *out_mask_ptr = pinned_mask_buffer_;
    return batch_size;
}
```

#### 3. Result Distribution

```cpp
void GlobalEvaluationQueue::submit_results(const float* policies,
                                            const float* values,
                                            int num_results) {
    // Distribute results to worker-specific queues
    for (int i = 0; i < num_results; ++i) {
        auto [worker_id, request_id] = current_batch_mapping_[i];

        EvalResult result;
        result.worker_id = worker_id;
        result.request_id = request_id;
        std::memcpy(result.policy.data(),
                    policies + i * POLICY_SIZE,
                    POLICY_SIZE * sizeof(float));
        result.value = values[i];

        {
            std::lock_guard<std::mutex> lock(worker_mutexes_[worker_id]);
            worker_results_[worker_id].push_back(std::move(result));
        }
        worker_cvs_[worker_id].notify_one();
    }
}
```

---

## Phase 2b: Enhanced Parallel Coordinator

### New File: `include/selfplay/parallel_coordinator.hpp`

```cpp
#pragma once

#include "evaluation_queue.hpp"
#include "game.hpp"
#include "../mcts/search.hpp"
#include "../training/replay_buffer.hpp"

#include <thread>
#include <vector>
#include <atomic>

namespace alphazero {

struct ParallelSelfPlayConfig {
    int num_workers = 16;           // 16-32 for GPU saturation
    int games_per_worker = 4;       // Games each worker plays
    int num_simulations = 800;
    int mcts_batch_size = 64;       // Per-game batch size
    float c_puct = 1.5f;
    int temperature_moves = 30;
    int gpu_batch_size = 512;       // Global GPU batch size
    int gpu_timeout_ms = 5;         // Max wait for batch
};

class ParallelSelfPlayCoordinator {
public:
    ParallelSelfPlayCoordinator(const ParallelSelfPlayConfig& config,
                                 ReplayBuffer* replay_buffer);
    ~ParallelSelfPlayCoordinator();

    // Start generation (non-blocking)
    // evaluator_callback is called by GPU thread
    template<typename EvaluatorFn>
    void start(EvaluatorFn&& evaluator);

    // Wait for completion
    void wait();

    // Stop early (graceful shutdown)
    void stop();

    // Get statistics
    struct Stats {
        std::atomic<int> games_completed{0};
        std::atomic<int> total_moves{0};
        std::atomic<int> total_simulations{0};
        std::atomic<int> total_nn_evals{0};
        std::atomic<int> white_wins{0};
        std::atomic<int> black_wins{0};
        std::atomic<int> draws{0};
    };
    const Stats& get_stats() const { return stats_; }
    const GlobalEvaluationQueue::Metrics& get_queue_metrics() const {
        return eval_queue_.get_metrics();
    }

private:
    void worker_thread(int worker_id);
    void gpu_thread_func();

    ParallelSelfPlayConfig config_;
    GlobalEvaluationQueue eval_queue_;
    ReplayBuffer* replay_buffer_;

    std::vector<std::thread> worker_threads_;
    std::thread gpu_thread_;
    std::function<void(const float*, const float*, int, float*, float*)> evaluator_;

    std::atomic<bool> running_{false};
    std::atomic<bool> shutdown_{false};
    Stats stats_;
};

// Worker thread implementation
void ParallelSelfPlayCoordinator::worker_thread(int worker_id) {
    // Thread-local resources
    NodePool node_pool;
    std::vector<chess::Board> position_history;

    // Pre-allocated buffers for this worker
    std::vector<float> obs_buffer(config_.mcts_batch_size * OBS_SIZE);
    std::vector<float> mask_buffer(config_.mcts_batch_size * POLICY_SIZE);
    std::vector<float> policy_buffer(config_.mcts_batch_size * POLICY_SIZE);
    std::vector<float> value_buffer(config_.mcts_batch_size);
    std::vector<int32_t> request_ids(config_.mcts_batch_size);

    for (int game = 0; game < config_.games_per_worker && !shutdown_; ++game) {
        chess::Board board;
        position_history.clear();
        std::vector<std::vector<float>> game_observations;
        std::vector<std::vector<float>> game_policies;
        int move_count = 0;

        while (!board.isGameOver() && move_count < 512 && !shutdown_) {
            node_pool.reset();

            // Create MCTS search for this position
            MCTSSearch mcts(config_.num_simulations,
                           config_.mcts_batch_size,
                           config_.c_puct);

            // Get root evaluation first
            encode_position(board, position_history, obs_buffer.data());
            get_legal_mask(board, mask_buffer.data());

            eval_queue_.submit_leaves(worker_id,
                                      obs_buffer.data(),
                                      mask_buffer.data(),
                                      1, request_ids);

            int num_results = eval_queue_.get_results(worker_id,
                                                       policy_buffer.data(),
                                                       value_buffer.data(),
                                                       1, 100);

            mcts.init_search(board, position_history,
                            policy_buffer.data(), value_buffer[0]);

            // MCTS simulation loop
            while (!mcts.is_search_complete() && !shutdown_) {
                // Collect leaves (applies virtual loss internally)
                int num_leaves = mcts.collect_leaves(obs_buffer.data(),
                                                      mask_buffer.data());
                if (num_leaves == 0) break;

                // Submit to global queue
                eval_queue_.submit_leaves(worker_id,
                                          obs_buffer.data(),
                                          mask_buffer.data(),
                                          num_leaves, request_ids);

                // Wait for results
                num_results = eval_queue_.get_results(worker_id,
                                                       policy_buffer.data(),
                                                       value_buffer.data(),
                                                       num_leaves, 100);

                // Update MCTS (reverts virtual loss, backpropagates)
                mcts.update_leaves(policy_buffer.data(),
                                   value_buffer.data(),
                                   num_results);

                stats_.total_nn_evals += num_results;
            }

            stats_.total_simulations += mcts.get_simulations_completed();

            // Get policy from visit counts
            std::vector<float> policy(POLICY_SIZE);
            mcts.get_policy(policy.data(), move_count < config_.temperature_moves);

            // Store for training
            game_observations.push_back(
                std::vector<float>(obs_buffer.begin(),
                                   obs_buffer.begin() + OBS_SIZE));
            game_policies.push_back(policy);

            // Select and make move
            chess::Move move = select_move(board, policy,
                                           move_count < config_.temperature_moves);
            board.makeMove(move);
            position_history.push_back(board);
            if (position_history.size() > 8) {
                position_history.erase(position_history.begin());
            }

            move_count++;
            stats_.total_moves++;
        }

        // Game complete - determine result
        float result = get_game_result(board);

        // Add to replay buffer (thread-safe)
        for (size_t i = 0; i < game_observations.size(); ++i) {
            float value = (i % 2 == 0) ? result : -result;
            replay_buffer_->add_sample(game_observations[i].data(),
                                       game_policies[i].data(),
                                       value);
        }

        // Update statistics
        stats_.games_completed++;
        if (result > 0.5f) stats_.white_wins++;
        else if (result < -0.5f) stats_.black_wins++;
        else stats_.draws++;
    }
}

// GPU thread implementation
template<typename EvaluatorFn>
void ParallelSelfPlayCoordinator::gpu_thread_func() {
    while (running_ && !shutdown_) {
        float* obs_ptr;
        float* mask_ptr;

        // Collect batch (blocks with timeout)
        int batch_size = eval_queue_.collect_batch(&obs_ptr, &mask_ptr,
                                                    config_.gpu_timeout_ms);

        if (batch_size == 0) continue;

        // Allocate output buffers (could also be pinned)
        std::vector<float> policies(batch_size * POLICY_SIZE);
        std::vector<float> values(batch_size);

        // Call neural network evaluator
        // This is where PyTorch inference happens
        evaluator_(obs_ptr, mask_ptr, batch_size,
                   policies.data(), values.data());

        // Distribute results to workers
        eval_queue_.submit_results(policies.data(), values.data(), batch_size);
    }
}

} // namespace alphazero
```

---

## Phase 3: Double-Buffered Pipeline

### Concept

```
Time →
Buffer A: [Fill]────────[GPU Eval]────────[Fill]────────[GPU Eval]
Buffer B: ────[GPU Eval]────────[Fill]────────[GPU Eval]────────
              ↑               ↑               ↑
         Overlap!        Overlap!        Overlap!
```

### Implementation

```cpp
class DoubleBufferedQueue {
    // Two complete buffer sets
    struct BufferSet {
        float* obs;      // Pinned
        float* mask;     // Pinned
        float* policy;   // Pinned
        float* value;    // Pinned
        int size;
        std::vector<std::pair<int,int>> mapping;
    };

    BufferSet buffers_[2];
    std::atomic<int> fill_buffer_{0};    // Workers fill this
    std::atomic<int> eval_buffer_{1};    // GPU evaluates this

    std::mutex swap_mutex_;
    std::condition_variable swap_cv_;
    std::atomic<bool> buffer_ready_{false};

public:
    // Workers fill current fill_buffer
    void submit_to_fill_buffer(...);

    // GPU thread: swap buffers and evaluate
    int swap_and_get_batch(float** obs, float** mask) {
        std::unique_lock<std::mutex> lock(swap_mutex_);
        swap_cv_.wait(lock, [this] {
            return buffer_ready_ || shutdown_;
        });

        // Swap buffer indices
        int old_fill = fill_buffer_.exchange(eval_buffer_);
        eval_buffer_ = old_fill;
        buffer_ready_ = false;

        // Return eval buffer for GPU
        *obs = buffers_[eval_buffer_].obs;
        *mask = buffers_[eval_buffer_].mask;
        return buffers_[eval_buffer_].size;
    }
};
```

---

## Python Integration

### New Python Bindings

```cpp
// In python_bindings.cpp

py::class_<ParallelSelfPlayCoordinator>(m, "ParallelSelfPlayCoordinator")
    .def(py::init<const ParallelSelfPlayConfig&, ReplayBuffer*>())

    .def("generate_games", [](ParallelSelfPlayCoordinator& coord,
                               py::function evaluator) {
        // Release GIL for C++ threads
        py::gil_scoped_release release;

        coord.start([&evaluator](const float* obs, const float* mask,
                                  int batch_size, float* policies, float* values) {
            // Acquire GIL for Python callback
            py::gil_scoped_acquire acquire;

            // Create tensors from pointers (zero-copy with pin_memory)
            auto obs_tensor = torch::from_blob(
                const_cast<float*>(obs),
                {batch_size, 122, 8, 8},
                torch::kFloat32
            ).to(torch::kCUDA, /*non_blocking=*/true);

            auto mask_tensor = torch::from_blob(
                const_cast<float*>(mask),
                {batch_size, POLICY_SIZE},
                torch::kFloat32
            ).to(torch::kCUDA, /*non_blocking=*/true);

            // Call Python evaluator
            auto result = evaluator(obs_tensor, mask_tensor);
            auto [policy_tensor, value_tensor] = result.cast<std::tuple<
                torch::Tensor, torch::Tensor>>();

            // Copy results back (async if pinned)
            policy_tensor.cpu().contiguous();
            value_tensor.cpu().contiguous();

            std::memcpy(policies, policy_tensor.data_ptr<float>(),
                       batch_size * POLICY_SIZE * sizeof(float));
            std::memcpy(values, value_tensor.data_ptr<float>(),
                       batch_size * sizeof(float));
        });

        coord.wait();
        return coord.get_stats();
    })

    .def("get_queue_metrics", &ParallelSelfPlayCoordinator::get_queue_metrics)
    .def("stop", &ParallelSelfPlayCoordinator::stop);

// Config binding
py::class_<ParallelSelfPlayConfig>(m, "ParallelSelfPlayConfig")
    .def(py::init<>())
    .def_readwrite("num_workers", &ParallelSelfPlayConfig::num_workers)
    .def_readwrite("games_per_worker", &ParallelSelfPlayConfig::games_per_worker)
    .def_readwrite("num_simulations", &ParallelSelfPlayConfig::num_simulations)
    .def_readwrite("mcts_batch_size", &ParallelSelfPlayConfig::mcts_batch_size)
    .def_readwrite("c_puct", &ParallelSelfPlayConfig::c_puct)
    .def_readwrite("temperature_moves", &ParallelSelfPlayConfig::temperature_moves)
    .def_readwrite("gpu_batch_size", &ParallelSelfPlayConfig::gpu_batch_size)
    .def_readwrite("gpu_timeout_ms", &ParallelSelfPlayConfig::gpu_timeout_ms);
```

### Updated train.py Usage

```python
# In train.py

def run_parallel_selfplay(network, replay_buffer, args):
    """Run parallel self-play with cross-game batching."""

    # Configure parallel coordinator
    config = alphazero_cpp.ParallelSelfPlayConfig()
    config.num_workers = 16              # 16-32 for GPU saturation
    config.games_per_worker = args.games_per_iter // config.num_workers
    config.num_simulations = args.simulations
    config.mcts_batch_size = 64          # Per-game MCTS batch
    config.gpu_batch_size = 512          # Global GPU batch
    config.gpu_timeout_ms = 5            # Smart batching timeout
    config.c_puct = args.c_puct
    config.temperature_moves = args.temperature_moves

    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(
        config, replay_buffer
    )

    # Neural network evaluator (called from GPU thread)
    @torch.no_grad()
    def gpu_evaluator(obs_tensor, mask_tensor):
        """Evaluate batch of positions on GPU."""
        with torch.cuda.amp.autocast():
            policy, value = network(obs_tensor, mask_tensor)
        return policy, value

    # Generate games (blocking)
    stats = coordinator.generate_games(gpu_evaluator)

    # Get metrics
    queue_metrics = coordinator.get_queue_metrics()
    print(f"  Batch fill ratio: {queue_metrics.batch_fill_ratio():.1%}")
    print(f"  Avg batch size: {queue_metrics.avg_batch_size():.1f}")

    return stats
```

---

## Metrics to Track

### Key Performance Indicators

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Batch Fill Ratio** | >80% | `avg_batch_size / max_batch_size` |
| **GPU Idle Time** | <10% | Time waiting in `collect_batch()` |
| **Worker Wait Time** | <20% | Time waiting for results |
| **GPU Utilization** | >85% | `nvidia-smi` or PyTorch profiler |

### Diagnostic Output

```python
def print_parallel_metrics(coordinator):
    stats = coordinator.get_stats()
    queue = coordinator.get_queue_metrics()

    print(f"\n  ═══ Parallel Self-Play Metrics ═══")
    print(f"  Games completed:    {stats.games_completed}")
    print(f"  Total moves:        {stats.total_moves:,}")
    print(f"  Total NN evals:     {stats.total_nn_evals:,}")
    print(f"  ")
    print(f"  Batch fill ratio:   {queue.batch_fill_ratio():.1%}")
    print(f"  Avg batch size:     {queue.avg_batch_size():.1f} / {MAX_BATCH_SIZE}")
    print(f"  GPU idle time:      {queue.gpu_idle_time_us / 1e6:.2f}s")
    print(f"  ═══════════════════════════════════")
```

---

## Tuning Guidelines

### Worker Count Selection

| GPU | VRAM | Recommended Workers | Batch Size |
|-----|------|---------------------|------------|
| RTX 4060 | 8GB | 16-24 | 512 |
| RTX 4070 | 12GB | 24-32 | 512-768 |
| RTX 4080 | 16GB | 32-48 | 768-1024 |
| RTX 4090 | 24GB | 48-64 | 1024-2048 |

### Timeout Tuning

```cpp
// If batch_fill_ratio < 70%:
//   - Increase num_workers
//   - Decrease gpu_timeout_ms (forces smaller batches more often)

// If GPU idle time > 20%:
//   - Increase num_workers
//   - Check if CPU encoding is bottleneck

// If worker wait time > 30%:
//   - Decrease gpu_batch_size
//   - Increase gpu_timeout_ms
```

---

## Implementation Checklist

### Phase 2: Global Evaluation Queue
- [ ] Create `include/selfplay/evaluation_queue.hpp`
- [ ] Implement `EvalRequestPool` with pre-allocated slots
- [ ] Implement `GlobalEvaluationQueue` with:
  - [ ] Pinned memory buffers (cudaMallocHost)
  - [ ] Smart batching (size threshold + timeout)
  - [ ] Per-worker result queues
  - [ ] Metrics tracking
- [ ] Create `include/selfplay/parallel_coordinator.hpp`
- [ ] Implement worker threads with virtual loss flow
- [ ] Implement GPU thread with evaluator callback
- [ ] Add Python bindings
- [ ] Update CMakeLists.txt

### Phase 3: Double-Buffered Pipeline
- [ ] Implement `DoubleBufferedQueue`
- [ ] Add async CUDA memory copy support
- [ ] Integrate with `ParallelSelfPlayCoordinator`
- [ ] Profile and verify overlap

### Integration
- [ ] Update `train.py` to use parallel coordinator
- [ ] Add command-line arguments for parallel config
- [ ] Add metrics logging
- [ ] Benchmark and tune

---

## Expected Results

| Metric | Current | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| GPU Utilization | ~40% | ~75% | ~90% |
| Games/hour | ~150 | ~450 | ~600 |
| Moves/sec | ~9 | ~30 | ~45 |
| Batch Fill Ratio | N/A | ~75% | ~85% |

**Total expected speedup: 4-6x**
