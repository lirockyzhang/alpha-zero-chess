#pragma once

#include <atomic>
#include <array>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <tuple>
#include <thread>
#include <memory>

namespace selfplay {

// ============================================================================
// Constants
// ============================================================================

constexpr size_t OBS_SIZE = 8 * 8 * 122;      // 7808 floats per observation
constexpr size_t POLICY_SIZE = 4672;           // Policy vector size
constexpr size_t MAX_WORKERS = 128;            // Maximum concurrent workers
constexpr size_t DEFAULT_MAX_BATCH = 512;      // Default max batch size
constexpr size_t DEFAULT_QUEUE_CAPACITY = 8192; // Request queue capacity (increased from 4096 to prevent exhaustion)

// EvalRequest has been replaced by StagedRequest (see below)
// Workers now write directly into pre-allocated staging buffers

// ============================================================================
// Evaluation Result
// ============================================================================

struct EvalResult {
    int32_t worker_id;
    int32_t request_id;
    uint32_t generation;      // Propagated from request (for stale result filtering)
    int32_t batch_index;      // Index into batch_policy_buffers_[buffer_id] (zero-copy reference)
    int8_t buffer_id;         // Which of the 2 double-buffers this result references
    float value;
    float wdl[3];             // WDL probabilities (win, draw, loss) from NN evaluation

    EvalResult() : worker_id(-1), request_id(-1), generation(0),
                   batch_index(-1), buffer_id(0), value(0.0f), wdl{0.0f, 0.0f, 0.0f} {}
};

// ============================================================================
// Staged Request (replaces ObservationPool + EvalRequest indirection)
// ============================================================================

struct StagedRequest {
    int32_t worker_id;
    int32_t request_id;
    uint32_t generation;
    int32_t staging_slot;    // Index into staging buffers

    StagedRequest() : worker_id(-1), request_id(-1), generation(0), staging_slot(-1) {}
    StagedRequest(int32_t wid, int32_t rid, uint32_t gen, int32_t slot)
        : worker_id(wid), request_id(rid), generation(gen), staging_slot(slot) {}
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

    // Flush all pending results for this worker and increment generation.
    // Makes all in-flight results stale so they'll be discarded on arrival.
    // Returns number of stale results flushed (for diagnostics).
    int flush_worker_results(int worker_id);

    // Wait for and retrieve results for this worker
    // Blocks until at least min_results are available or timeout
    // Returns: number of results retrieved
    int get_results(
        int worker_id,
        float* out_policies,            // max_results x POLICY_SIZE
        float* out_values,              // max_results
        int max_results,
        int timeout_ms = 100,
        float* out_wdl = nullptr        // max_results x 3 (WDL probs, optional for Phase 2)
    );

    // Non-blocking result retrieval: returns immediately with whatever results are available
    // Returns: number of results retrieved (0 if none ready)
    int try_get_results(
        int worker_id,
        float* out_policies,            // max_results x POLICY_SIZE
        float* out_values,              // max_results
        int max_results,
        float* out_wdl = nullptr        // max_results x 3 (WDL probs, optional for Phase 2)
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
        const float* values,            // batch_size (scalar values after WDL→value conversion)
        int batch_size,
        const float* wdl_probs = nullptr // batch_size x 3 (WDL probs, optional for Phase 2)
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
    size_t get_queue_capacity() const { return queue_capacity_; }

    // =========================================================================
    // Direct Memory Access (for zero-copy Python integration)
    // =========================================================================

    // Get raw pointers to batch buffers (for torch.from_blob)
    uintptr_t get_batch_obs_ptr() const {
        return reinterpret_cast<uintptr_t>(batch_obs_nchw_buffer_);
    }
    uintptr_t get_batch_mask_ptr() const {
        return reinterpret_cast<uintptr_t>(batch_mask_buffer_);
    }
    uintptr_t get_batch_policy_ptr() const {
        return reinterpret_cast<uintptr_t>(batch_policy_buffers_[current_policy_buffer_]);
    }
    uintptr_t get_batch_value_ptr() const {
        return reinterpret_cast<uintptr_t>(batch_value_buffer_);
    }

private:
    // Allocate pinned/aligned memory for batch buffers
    void allocate_batch_buffers();
    void free_batch_buffers();

    // Compact remaining staged requests to front of staging buffers (called under lock)
    void compact_remaining_locked();

    // Configuration
    size_t max_batch_size_;
    size_t queue_capacity_;

    // Pre-allocated staging buffers: workers write directly into these slots
    // via atomic staging_write_head_ (no lock needed for data write)
    float* staging_obs_buffer_;    // queue_capacity × OBS_SIZE
    float* staging_mask_buffer_;   // queue_capacity × POLICY_SIZE
    std::atomic<int32_t> staging_write_head_{0};

    // Per-slot ready flags: workers set after memcpy completes (lock-free signaling)
    // GPU thread spin-waits on these before reading staging data
    // NOTE: Using unique_ptr<atomic[]> instead of vector<atomic> because
    // std::atomic has deleted copy/move constructors, incompatible with std::vector on MSVC
    std::unique_ptr<std::atomic<uint8_t>[]> slot_ready_;  // queue_capacity entries

    // Request metadata queue (lightweight — only metadata, no observation data)
    std::vector<StagedRequest> staged_requests_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Deferred compaction flag: set when staging head > 75% capacity,
    // executed at start of NEXT collect_batch (after all previous slots consumed)
    bool needs_compaction_{false};

    // Batch buffers (aligned/pinned for GPU)
    // NOTE: batch_obs_buffer_ (NHWC intermediate) removed — fused transpose
    // reads directly from staging_obs_buffer_ into batch_obs_nchw_buffer_
    float* batch_obs_nchw_buffer_; // max_batch_size x OBS_SIZE (NCHW for GPU)
    float* batch_mask_buffer_;     // max_batch_size x POLICY_SIZE
    float* batch_policy_buffers_[2]; // Double-buffered: GPU writes [current], workers read [previous]
    float* batch_value_buffer_;    // max_batch_size
    int current_policy_buffer_{0}; // Which buffer GPU writes to next (only GPU thread writes; ordering via worker_cvs_ notify/wait)

    // Staging slot indices for current batch (used for lock-free Phase 2 reads)
    std::vector<int32_t> batch_staging_slots_;

    // Mapping for current batch (worker_id, request_id, generation) tuples
    std::vector<std::tuple<int32_t, int32_t, uint32_t>> batch_mapping_;

    // Per-worker result queues
    std::array<std::vector<EvalResult>, MAX_WORKERS> worker_results_;
    std::array<std::mutex, MAX_WORKERS> worker_mutexes_;
    std::array<std::condition_variable, MAX_WORKERS> worker_cvs_;

    // Request ID generator (per worker to avoid contention)
    std::array<std::atomic<int32_t>, MAX_WORKERS> worker_request_ids_;

    // Per-worker generation counters (incremented on flush to invalidate stale results)
    std::array<std::atomic<uint32_t>, MAX_WORKERS> worker_generation_{};

    // Lifecycle
    std::atomic<bool> shutdown_{false};

    // Metrics
    QueueMetrics metrics_;
};

} // namespace selfplay
