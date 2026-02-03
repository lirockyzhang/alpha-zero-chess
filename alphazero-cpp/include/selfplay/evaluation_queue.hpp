#pragma once

#include <atomic>
#include <array>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstring>
#include <cstdlib>

namespace selfplay {

// ============================================================================
// Constants
// ============================================================================

constexpr size_t OBS_SIZE = 8 * 8 * 122;      // 7808 floats per observation
constexpr size_t POLICY_SIZE = 4672;           // Policy vector size
constexpr size_t MAX_WORKERS = 32;             // Maximum concurrent workers
constexpr size_t DEFAULT_MAX_BATCH = 512;      // Default max batch size
constexpr size_t DEFAULT_QUEUE_CAPACITY = 8192; // Request queue capacity (increased from 4096 to prevent exhaustion)

// ============================================================================
// Evaluation Request (Fixed-Size, Cache-Aligned)
// ============================================================================

// Single evaluation request - uses fixed-size arrays to avoid heap allocation
// Aligned to cache line to prevent false sharing
struct alignas(64) EvalRequest {
    int32_t worker_id;
    int32_t request_id;

    // Fixed-size observation buffer (NHWC format: 8x8x122)
    // Using separate allocation to avoid bloating this struct
    float* observation;   // Points to pool memory
    float* legal_mask;    // Points to pool memory

    EvalRequest() : worker_id(-1), request_id(-1), observation(nullptr), legal_mask(nullptr) {}
};

// ============================================================================
// Evaluation Result
// ============================================================================

struct EvalResult {
    int32_t worker_id;
    int32_t request_id;
    std::vector<float> policy;  // 4672 floats
    float value;

    EvalResult() : worker_id(-1), request_id(-1), policy(POLICY_SIZE), value(0.0f) {}
};

// ============================================================================
// Memory Pool for Observations (Pre-allocated, Aligned)
// ============================================================================

class ObservationPool {
public:
    explicit ObservationPool(size_t capacity = DEFAULT_QUEUE_CAPACITY)
        : capacity_(capacity)
        , allocated_(0)
    {
        // Allocate aligned memory for observations and masks
        size_t obs_bytes = capacity * OBS_SIZE * sizeof(float);
        size_t mask_bytes = capacity * POLICY_SIZE * sizeof(float);

        // Use aligned allocation (64-byte for cache line)
        #ifdef _WIN32
        obs_buffer_ = static_cast<float*>(_aligned_malloc(obs_bytes, 64));
        mask_buffer_ = static_cast<float*>(_aligned_malloc(mask_bytes, 64));
        #else
        obs_buffer_ = static_cast<float*>(std::aligned_alloc(64, obs_bytes));
        mask_buffer_ = static_cast<float*>(std::aligned_alloc(64, mask_bytes));
        #endif

        if (!obs_buffer_ || !mask_buffer_) {
            throw std::bad_alloc();
        }
    }

    ~ObservationPool() {
        #ifdef _WIN32
        _aligned_free(obs_buffer_);
        _aligned_free(mask_buffer_);
        #else
        std::free(obs_buffer_);
        std::free(mask_buffer_);
        #endif
    }

    // Get pointers for slot index
    float* get_obs_ptr(size_t index) {
        return obs_buffer_ + index * OBS_SIZE;
    }

    float* get_mask_ptr(size_t index) {
        return mask_buffer_ + index * POLICY_SIZE;
    }

    // Allocate next slot (returns slot index, or -1 if full)
    int allocate() {
        size_t idx = allocated_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= capacity_) {
            allocated_.fetch_sub(1, std::memory_order_relaxed);
            return -1;
        }
        return static_cast<int>(idx);
    }

    // Reset allocation counter (call when queue is cleared)
    void reset() {
        allocated_.store(0, std::memory_order_relaxed);
    }

    size_t capacity() const { return capacity_; }
    size_t allocated() const { return allocated_.load(std::memory_order_relaxed); }

private:
    float* obs_buffer_;
    float* mask_buffer_;
    size_t capacity_;
    std::atomic<size_t> allocated_;
};

// ============================================================================
// Queue Metrics (Atomic, Cache-Line Aligned)
// ============================================================================

struct alignas(64) QueueMetrics {
    std::atomic<uint64_t> total_batches{0};
    std::atomic<uint64_t> total_leaves{0};
    std::atomic<uint64_t> gpu_wait_time_us{0};
    std::atomic<uint64_t> worker_wait_time_us{0};
    std::atomic<uint64_t> total_requests_submitted{0};
    std::atomic<uint64_t> total_results_returned{0};

    // Diagnostic counters for debugging parallel self-play issues
    std::atomic<uint64_t> pool_exhaustion_count{0};   // Times pool ran out of slots
    std::atomic<uint64_t> partial_submissions{0};     // Times queued < requested
    std::atomic<uint64_t> submission_drops{0};        // Total leaves dropped due to pool exhaustion
    std::atomic<uint64_t> pool_resets{0};             // Times pool was reset

    double avg_batch_size() const {
        uint64_t batches = total_batches.load(std::memory_order_relaxed);
        uint64_t leaves = total_leaves.load(std::memory_order_relaxed);
        return batches > 0 ? static_cast<double>(leaves) / batches : 0.0;
    }

    double batch_fill_ratio(size_t max_batch_size) const {
        return avg_batch_size() / static_cast<double>(max_batch_size);
    }

    void reset() {
        total_batches = 0;
        total_leaves = 0;
        gpu_wait_time_us = 0;
        worker_wait_time_us = 0;
        total_requests_submitted = 0;
        total_results_returned = 0;
        pool_exhaustion_count = 0;
        partial_submissions = 0;
        submission_drops = 0;
        pool_resets = 0;
    }
};

// ============================================================================
// Global Evaluation Queue
// ============================================================================
//
// Thread-safe queue for collecting NN evaluation requests from multiple workers
// and batching them for efficient GPU evaluation.
//
// Design principles:
// - Pre-allocated memory pools (no heap allocation in hot path)
// - Smart batching (batch_size threshold OR timeout)
// - Per-worker result queues (efficient result distribution)
// - Pinned memory ready (for async GPU transfers)
//
class GlobalEvaluationQueue {
public:
    // Constructor
    // max_batch_size: Maximum leaves in one GPU batch
    // queue_capacity: Maximum pending requests
    explicit GlobalEvaluationQueue(
        size_t max_batch_size = DEFAULT_MAX_BATCH,
        size_t queue_capacity = DEFAULT_QUEUE_CAPACITY
    );

    ~GlobalEvaluationQueue();

    // =========================================================================
    // Worker Thread Interface
    // =========================================================================

    // Submit observations for evaluation
    // Copies data into internal pool and queues for processing
    // Returns: number of requests actually queued (may be less if queue full)
    int submit_for_evaluation(
        int worker_id,
        const float* observations,      // num_leaves x OBS_SIZE
        const float* legal_masks,       // num_leaves x POLICY_SIZE
        int num_leaves,
        std::vector<int32_t>& out_request_ids  // Filled with assigned request IDs
    );

    // Wait for and retrieve results for this worker
    // Blocks until at least min_results are available or timeout
    // Returns: number of results retrieved
    int get_results(
        int worker_id,
        float* out_policies,            // max_results x POLICY_SIZE
        float* out_values,              // max_results
        int max_results,
        int timeout_ms = 100
    );

    // =========================================================================
    // GPU Thread Interface
    // =========================================================================

    // Collect a batch of requests for GPU evaluation
    // Blocks until: (1) batch is full OR (2) timeout expires and queue non-empty
    // Returns: number of leaves in batch (0 if shutdown or timeout with empty queue)
    int collect_batch(
        float** out_obs_ptr,            // Pointer to observation batch
        float** out_mask_ptr,           // Pointer to mask batch
        int timeout_ms = 5
    );

    // Submit results back after GPU evaluation
    // Must be called after collect_batch with matching batch size
    void submit_results(
        const float* policies,          // batch_size x POLICY_SIZE
        const float* values,            // batch_size
        int batch_size
    );

    // =========================================================================
    // Lifecycle Management
    // =========================================================================

    // Signal shutdown (unblocks all waiting threads)
    void shutdown();

    // Check if shutdown was requested
    bool is_shutdown() const { return shutdown_.load(std::memory_order_acquire); }

    // Reset queue state (for reuse)
    void reset();

    // =========================================================================
    // Metrics
    // =========================================================================

    const QueueMetrics& get_metrics() const { return metrics_; }
    size_t pending_count() const;
    size_t max_batch_size() const { return max_batch_size_; }

    // =========================================================================
    // Direct Memory Access (for zero-copy Python integration)
    // =========================================================================

    // Get raw pointers to batch buffers (for torch.from_blob)
    uintptr_t get_batch_obs_ptr() const {
        return reinterpret_cast<uintptr_t>(batch_obs_buffer_);
    }
    uintptr_t get_batch_mask_ptr() const {
        return reinterpret_cast<uintptr_t>(batch_mask_buffer_);
    }
    uintptr_t get_batch_policy_ptr() const {
        return reinterpret_cast<uintptr_t>(batch_policy_buffer_);
    }
    uintptr_t get_batch_value_ptr() const {
        return reinterpret_cast<uintptr_t>(batch_value_buffer_);
    }

private:
    // Allocate pinned/aligned memory for batch buffers
    void allocate_batch_buffers();
    void free_batch_buffers();

    // Configuration
    size_t max_batch_size_;
    size_t queue_capacity_;

    // Memory pool for incoming requests
    ObservationPool obs_pool_;

    // Request queue
    std::vector<EvalRequest> pending_requests_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Batch buffers (aligned/pinned for GPU)
    float* batch_obs_buffer_;      // max_batch_size x OBS_SIZE
    float* batch_mask_buffer_;     // max_batch_size x POLICY_SIZE
    float* batch_policy_buffer_;   // max_batch_size x POLICY_SIZE
    float* batch_value_buffer_;    // max_batch_size

    // Mapping for current batch (worker_id, request_id) pairs
    std::vector<std::pair<int32_t, int32_t>> batch_mapping_;

    // Per-worker result queues
    std::array<std::vector<EvalResult>, MAX_WORKERS> worker_results_;
    std::array<std::mutex, MAX_WORKERS> worker_mutexes_;
    std::array<std::condition_variable, MAX_WORKERS> worker_cvs_;

    // Request ID generator (per worker to avoid contention)
    std::array<std::atomic<int32_t>, MAX_WORKERS> worker_request_ids_;

    // Lifecycle
    std::atomic<bool> shutdown_{false};

    // Metrics
    QueueMetrics metrics_;
};

} // namespace selfplay
