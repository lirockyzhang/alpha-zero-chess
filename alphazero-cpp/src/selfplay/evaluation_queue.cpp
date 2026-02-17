#include "../../include/selfplay/evaluation_queue.hpp"
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

namespace selfplay {

// ============================================================================
// Constructor / Destructor
// ============================================================================

GlobalEvaluationQueue::GlobalEvaluationQueue(size_t max_batch_size, size_t queue_capacity)
    : max_batch_size_(max_batch_size)
    , queue_capacity_(queue_capacity)
    , staging_obs_buffer_(nullptr)
    , staging_mask_buffer_(nullptr)
    , batch_obs_buffer_(nullptr)
    , batch_mask_buffer_(nullptr)
    , batch_policy_buffers_{nullptr, nullptr}
    , batch_value_buffer_(nullptr)
{
    // Allocate staging buffers (workers write directly into these)
    size_t staging_obs_bytes = queue_capacity * OBS_SIZE * sizeof(float);
    size_t staging_mask_bytes = queue_capacity * POLICY_SIZE * sizeof(float);

    #ifdef _WIN32
    staging_obs_buffer_ = static_cast<float*>(_aligned_malloc(staging_obs_bytes, 64));
    staging_mask_buffer_ = static_cast<float*>(_aligned_malloc(staging_mask_bytes, 64));
    #else
    staging_obs_buffer_ = static_cast<float*>(std::aligned_alloc(64, staging_obs_bytes));
    staging_mask_buffer_ = static_cast<float*>(std::aligned_alloc(64, staging_mask_bytes));
    #endif

    if (!staging_obs_buffer_ || !staging_mask_buffer_) {
        throw std::bad_alloc();
    }

    // Initialize per-slot ready flags (all zero = not ready)
    // PaddedAtomicFlag default-initializes value to 0
    slot_ready_ = std::make_unique<PaddedAtomicFlag[]>(queue_capacity);

    // Pre-allocate staged request metadata queue
    staged_requests_.reserve(queue_capacity);

    // Pre-allocate batch mapping and staging slot tracking
    batch_mapping_.reserve(max_batch_size);
    batch_staging_slots_.reserve(max_batch_size);

    // Allocate batch buffers (aligned for GPU)
    allocate_batch_buffers();

    // Initialize per-worker result queues
    for (size_t i = 0; i < MAX_WORKERS; ++i) {
        worker_results_[i].reserve(max_batch_size);
        worker_request_ids_[i].store(0, std::memory_order_relaxed);
    }

    metrics_.reset();
}

GlobalEvaluationQueue::~GlobalEvaluationQueue() {
    shutdown();
    free_batch_buffers();

    #ifdef _WIN32
    if (staging_obs_buffer_) _aligned_free(staging_obs_buffer_);
    if (staging_mask_buffer_) _aligned_free(staging_mask_buffer_);
    #else
    if (staging_obs_buffer_) std::free(staging_obs_buffer_);
    if (staging_mask_buffer_) std::free(staging_mask_buffer_);
    #endif
}

// ============================================================================
// Memory Management
// ============================================================================

void GlobalEvaluationQueue::allocate_batch_buffers() {
    size_t obs_bytes = max_batch_size_ * OBS_SIZE * sizeof(float);
    size_t mask_bytes = max_batch_size_ * POLICY_SIZE * sizeof(float);
    size_t policy_bytes = max_batch_size_ * POLICY_SIZE * sizeof(float);
    size_t value_bytes = max_batch_size_ * sizeof(float);

    // Allocate aligned memory (64-byte for cache line / GPU compatibility)
    // batch_obs_buffer_ holds NHWC data (same layout as staging)
    #ifdef _WIN32
    batch_obs_buffer_ = static_cast<float*>(_aligned_malloc(obs_bytes, 64));
    batch_mask_buffer_ = static_cast<float*>(_aligned_malloc(mask_bytes, 64));
    batch_policy_buffers_[0] = static_cast<float*>(_aligned_malloc(policy_bytes, 64));
    batch_policy_buffers_[1] = static_cast<float*>(_aligned_malloc(policy_bytes, 64));
    batch_value_buffer_ = static_cast<float*>(_aligned_malloc(value_bytes, 64));
    #else
    batch_obs_buffer_ = static_cast<float*>(std::aligned_alloc(64, obs_bytes));
    batch_mask_buffer_ = static_cast<float*>(std::aligned_alloc(64, mask_bytes));
    batch_policy_buffers_[0] = static_cast<float*>(std::aligned_alloc(64, policy_bytes));
    batch_policy_buffers_[1] = static_cast<float*>(std::aligned_alloc(64, policy_bytes));
    batch_value_buffer_ = static_cast<float*>(std::aligned_alloc(64, value_bytes));
    #endif

    if (!batch_obs_buffer_ || !batch_mask_buffer_ ||
        !batch_policy_buffers_[0] || !batch_policy_buffers_[1] || !batch_value_buffer_) {
        free_batch_buffers();
        throw std::bad_alloc();
    }
}

void GlobalEvaluationQueue::free_batch_buffers() {
    #ifdef _WIN32
    if (batch_obs_buffer_) _aligned_free(batch_obs_buffer_);
    if (batch_mask_buffer_) _aligned_free(batch_mask_buffer_);
    if (batch_policy_buffers_[0]) _aligned_free(batch_policy_buffers_[0]);
    if (batch_policy_buffers_[1]) _aligned_free(batch_policy_buffers_[1]);
    if (batch_value_buffer_) _aligned_free(batch_value_buffer_);
    #else
    if (batch_obs_buffer_) std::free(batch_obs_buffer_);
    if (batch_mask_buffer_) std::free(batch_mask_buffer_);
    if (batch_policy_buffers_[0]) std::free(batch_policy_buffers_[0]);
    if (batch_policy_buffers_[1]) std::free(batch_policy_buffers_[1]);
    if (batch_value_buffer_) std::free(batch_value_buffer_);
    #endif

    batch_obs_buffer_ = nullptr;
    batch_mask_buffer_ = nullptr;
    batch_policy_buffers_[0] = nullptr;
    batch_policy_buffers_[1] = nullptr;
    batch_value_buffer_ = nullptr;
}

// ============================================================================
// Worker Thread Interface
// ============================================================================

int GlobalEvaluationQueue::submit_for_evaluation(
    int worker_id,
    const float* observations,
    const float* legal_masks,
    int num_leaves,
    std::vector<int32_t>& out_request_ids)
{
    if (shutdown_.load(std::memory_order_acquire) || num_leaves <= 0) {
        return 0;
    }

    if (worker_id < 0 || worker_id >= static_cast<int>(MAX_WORKERS)) {
        return 0;
    }

    out_request_ids.clear();
    out_request_ids.reserve(num_leaves);

    int actual_leaves = 0;
    int first_slot = 0;

    // =====================================================================
    // Phase 1 (under lock, ~1µs): Claim staging slots + push metadata
    // =====================================================================
    // Only lightweight operations: integer arithmetic + vector push_back.
    // The expensive memcpy is deferred to Phase 2 (no lock).
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        // Claim staging slots (synchronized with GPU's compaction reset)
        first_slot = staging_write_head_.load(std::memory_order_relaxed);
        actual_leaves = num_leaves;

        if (first_slot + num_leaves > static_cast<int>(queue_capacity_)) {
            actual_leaves = std::max(0, static_cast<int>(queue_capacity_) - first_slot);
            if (actual_leaves < num_leaves) {
                metrics_.pool_exhaustion_count.fetch_add(1, std::memory_order_relaxed);
                metrics_.submission_drops.fetch_add(num_leaves - actual_leaves, std::memory_order_relaxed);
            }
        }

        if (actual_leaves <= 0) {
            return 0;
        }

        staging_write_head_.store(first_slot + actual_leaves, std::memory_order_relaxed);

        // Push metadata (no data copy — just slot indices)
        uint32_t gen = worker_generation_[worker_id].load(std::memory_order_acquire);
        for (int i = 0; i < actual_leaves; ++i) {
            int32_t req_id = worker_request_ids_[worker_id].fetch_add(1, std::memory_order_relaxed);
            staged_requests_.emplace_back(worker_id, req_id, gen, first_slot + i);
            out_request_ids.push_back(req_id);
        }

        metrics_.total_requests_submitted.fetch_add(actual_leaves, std::memory_order_relaxed);

        if (actual_leaves < num_leaves && actual_leaves > 0) {
            metrics_.partial_submissions.fetch_add(1, std::memory_order_relaxed);
        }
    }
    // Lock released — other workers can claim slots concurrently

    // =====================================================================
    // Phase 2 (NO lock): memcpy observation + mask data to claimed slots
    // =====================================================================
    // Safe because each worker owns its claimed slots exclusively:
    // - Slots [first_slot, first_slot + actual_leaves) belong to this worker
    // - No other worker can claim the same slots (staging_write_head_ was
    //   incremented atomically under lock)
    // - GPU thread won't read these slots until slot_ready_ flags are set
    for (int i = 0; i < actual_leaves; ++i) {
        int slot = first_slot + i;
        std::memcpy(staging_obs_buffer_ + slot * OBS_SIZE,
                    observations + i * OBS_SIZE,
                    OBS_SIZE * sizeof(float));
        std::memcpy(staging_mask_buffer_ + slot * POLICY_SIZE,
                    legal_masks + i * POLICY_SIZE,
                    POLICY_SIZE * sizeof(float));
    }

    // =====================================================================
    // Phase 3 (NO lock): Signal slot readiness + notify GPU thread
    // =====================================================================
    // release semantics ensure the preceding memcpy is visible to the GPU
    // thread when it does an acquire load on slot_ready_[slot]
    for (int i = 0; i < actual_leaves; ++i) {
        slot_ready_[first_slot + i].value.store(1, std::memory_order_release);
    }

    // Notify OUTSIDE the lock — the GPU thread can immediately acquire
    // the mutex when woken instead of blocking on the notifier's lock.
    queue_cv_.notify_one();

    return actual_leaves;
}

int GlobalEvaluationQueue::get_results(
    int worker_id,
    float* out_policies,
    float* out_values,
    int max_results,
    int timeout_ms,
    float* out_wdl)
{
    if (worker_id < 0 || worker_id >= static_cast<int>(MAX_WORKERS)) {
        return 0;
    }

    auto start = std::chrono::steady_clock::now();
    int retrieved = 0;

    std::unique_lock<std::mutex> lock(worker_mutexes_[worker_id]);

    // Wait for results
    bool success = worker_cvs_[worker_id].wait_for(
        lock,
        std::chrono::milliseconds(timeout_ms),
        [this, worker_id, max_results]() {
            return !worker_results_[worker_id].empty() ||
                   shutdown_.load(std::memory_order_acquire);
        }
    );

    if (shutdown_.load(std::memory_order_acquire) && worker_results_[worker_id].empty()) {
        return 0;
    }

    // Filter stale results (from timed-out requests that completed after flush)
    auto& results = worker_results_[worker_id];
    uint32_t current_gen = worker_generation_[worker_id].load(std::memory_order_acquire);
    results.erase(
        std::remove_if(results.begin(), results.end(),
            [current_gen](const EvalResult& r) { return r.generation < current_gen; }),
        results.end()
    );

    // Retrieve available results
    int to_retrieve = std::min(static_cast<int>(results.size()), max_results);

    for (int i = 0; i < to_retrieve; ++i) {
        const auto& result = results[i];

        // Copy policy from double-buffered batch buffer using stored buffer_id
        if (result.batch_index >= 0) {
            std::memcpy(out_policies + i * POLICY_SIZE,
                        batch_policy_buffers_[result.buffer_id] + result.batch_index * POLICY_SIZE,
                        POLICY_SIZE * sizeof(float));
        }

        // Copy value
        out_values[i] = result.value;

        // Copy WDL probabilities if caller wants them (Phase 2)
        if (out_wdl != nullptr) {
            out_wdl[i * 3 + 0] = result.wdl[0];
            out_wdl[i * 3 + 1] = result.wdl[1];
            out_wdl[i * 3 + 2] = result.wdl[2];
        }
        retrieved++;
    }

    // Remove retrieved results
    if (to_retrieve > 0) {
        results.erase(results.begin(), results.begin() + to_retrieve);
    }

    // Update metrics
    auto elapsed = std::chrono::steady_clock::now() - start;
    metrics_.worker_wait_time_us.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count(),
        std::memory_order_relaxed
    );

    return retrieved;
}

int GlobalEvaluationQueue::try_get_results(
    int worker_id,
    float* out_policies,
    float* out_values,
    int max_results,
    float* out_wdl)
{
    if (worker_id < 0 || worker_id >= static_cast<int>(MAX_WORKERS)) {
        return 0;
    }

    // Non-blocking: try to acquire lock immediately
    std::unique_lock<std::mutex> lock(worker_mutexes_[worker_id], std::try_to_lock);
    if (!lock.owns_lock()) {
        return 0;
    }

    // Filter stale results
    auto& results = worker_results_[worker_id];
    uint32_t current_gen = worker_generation_[worker_id].load(std::memory_order_acquire);
    results.erase(
        std::remove_if(results.begin(), results.end(),
            [current_gen](const EvalResult& r) { return r.generation < current_gen; }),
        results.end()
    );

    if (results.empty()) {
        return 0;
    }

    int to_retrieve = std::min(static_cast<int>(results.size()), max_results);
    int retrieved = 0;

    for (int i = 0; i < to_retrieve; ++i) {
        const auto& result = results[i];

        if (result.batch_index >= 0) {
            std::memcpy(out_policies + i * POLICY_SIZE,
                        batch_policy_buffers_[result.buffer_id] + result.batch_index * POLICY_SIZE,
                        POLICY_SIZE * sizeof(float));
        }

        out_values[i] = result.value;

        // Copy WDL probabilities if caller wants them (Phase 2)
        if (out_wdl != nullptr) {
            out_wdl[i * 3 + 0] = result.wdl[0];
            out_wdl[i * 3 + 1] = result.wdl[1];
            out_wdl[i * 3 + 2] = result.wdl[2];
        }
        retrieved++;
    }

    if (to_retrieve > 0) {
        results.erase(results.begin(), results.begin() + to_retrieve);
    }

    return retrieved;
}

int GlobalEvaluationQueue::flush_worker_results(int worker_id) {
    if (worker_id < 0 || worker_id >= static_cast<int>(MAX_WORKERS)) {
        return 0;
    }
    // Increment generation FIRST (makes all in-flight results stale)
    worker_generation_[worker_id].fetch_add(1, std::memory_order_release);

    // Then clear any already-arrived stale results
    int count;
    {
        std::lock_guard<std::mutex> lock(worker_mutexes_[worker_id]);
        count = static_cast<int>(worker_results_[worker_id].size());
        worker_results_[worker_id].clear();
    }
    return count;
}

// ============================================================================
// GPU Thread Interface
// ============================================================================

int GlobalEvaluationQueue::collect_batch(
    float** out_obs_ptr,
    float** out_mask_ptr,
    int timeout_ms)
{
    auto start = std::chrono::steady_clock::now();
    size_t batch_size = 0;

    // =====================================================================
    // Phase 1 (under queue_mutex_, ~10µs): Metadata extraction
    // =====================================================================
    // Only lightweight operations: deferred compaction, wait, vector swap.
    // NO memcpy of observation/mask data under lock.
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        // --- Deferred compaction from previous batch ---
        // By deferring to here (start of NEXT collect_batch), we guarantee
        // all previous-batch slot_ready_ flags have been consumed and cleared
        // in the previous Phase 2, so compaction won't overwrite live slots.
        if (needs_compaction_) {
            compact_remaining_locked();
            needs_compaction_ = false;
        }

        // TWO-PHASE WAIT with stall detection (UNCHANGED — it works well).
        //
        // Phase 1: Wait for ANY request (full timeout).
        //   Handles the idle GPU case (between iterations, shutdown).
        //
        // Phase 2: Stall-detecting accumulation.
        //   Keep waiting as long as workers are actively submitting
        //   (queue growing). Once no new submissions arrive within 1ms,
        //   fire immediately with whatever we have.

        // Phase 1: Wait for at least 1 request
        queue_cv_.wait_for(
            lock,
            std::chrono::milliseconds(timeout_ms),
            [this]() {
                return !staged_requests_.empty() ||
                       shutdown_.load(std::memory_order_acquire);
            }
        );

        if (staged_requests_.empty()) {
            return 0;
        }

        // Phase 2: Accumulate while workers are still submitting
        {
            auto outer_deadline = std::chrono::steady_clock::now()
                + std::chrono::milliseconds(timeout_ms);

            while (staged_requests_.size() < max_batch_size_ &&
                   std::chrono::steady_clock::now() < outer_deadline &&
                   !shutdown_.load(std::memory_order_acquire))
            {
                size_t prev_size = staged_requests_.size();

                // Wait for a new submission or 1ms stall timeout
                queue_cv_.wait_for(
                    lock,
                    std::chrono::milliseconds(1),
                    [this, prev_size]() {
                        return staged_requests_.size() > prev_size ||
                               staged_requests_.size() >= max_batch_size_ ||
                               shutdown_.load(std::memory_order_acquire);
                    }
                );

                // Stall detected: no new submissions in 1ms → fire now
                if (staged_requests_.size() == prev_size) {
                    break;
                }
            }
        }

        // Collect up to max_batch_size requests — METADATA ONLY
        batch_size = std::min(staged_requests_.size(), max_batch_size_);

        batch_mapping_.clear();
        batch_mapping_.reserve(batch_size);
        batch_staging_slots_.clear();
        batch_staging_slots_.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
            const auto& req = staged_requests_[i];
            batch_mapping_.emplace_back(req.worker_id, req.request_id, req.generation);
            batch_staging_slots_.push_back(req.staging_slot);
        }

        // Remove collected requests from the metadata queue
        staged_requests_.erase(staged_requests_.begin(),
                                staged_requests_.begin() + batch_size);

        // LAZY COMPACTION: Schedule for next collect_batch if head > 75%.
        // Deferring is safe: collected slots' data is read in Phase 2 below,
        // and remaining requests' slot_ready_ flags are already set by workers.
        int head = staging_write_head_.load(std::memory_order_relaxed);
        if (head > static_cast<int>(queue_capacity_ * 3 / 4)) {
            needs_compaction_ = true;
        }
    }
    // Lock released — workers can submit new leaves during Phase 2

    // =====================================================================
    // Phase 2 (NO lock, OpenMP parallel): Spin-wait + NHWC memcpy + mask copy
    // =====================================================================
    // For each batch entry:
    //   1. Spin-wait on slot_ready_[slot] (typically instant — worker finished
    //      its memcpy long before GPU reached stall detection)
    //   2. Direct NHWC memcpy: staging_obs → batch_obs (no transpose needed,
    //      Python uses torch.permute for zero-copy channels_last layout)
    //   3. memcpy mask: staging_mask → batch_mask
    //   4. Clear slot_ready_[slot] for reuse
    //
    // Safe because:
    //   - Workers are blocked on get_results() during GPU's Phase 2
    //   - Each batch entry reads from a unique staging slot
    //   - Each batch entry writes to a unique batch index

    int batch_size_i = static_cast<int>(batch_size);

    // Skip OpenMP fork overhead for tiny batches (fork cost ~1-5µs > work cost)
    bool use_omp = batch_size_i >= 8;
    int num_threads = std::min(batch_size_i,
        static_cast<int>(std::thread::hardware_concurrency()));

    #pragma omp parallel for schedule(static) num_threads(num_threads) if(use_omp)
    for (int b = 0; b < batch_size_i; ++b) {
        int slot = batch_staging_slots_[b];

        // Two-tier spin-wait: _mm_pause (fast) → yield (fallback)
        // Workers finish memcpy well before the GPU thread completes stall
        // detection, so this almost never exceeds the pause tier.
        // NOTE: No sleep_for — on Windows, sleep_for(1µs) actually sleeps
        // 1-15ms due to 15.6ms timer resolution, which blocks the entire
        // OMP parallel region and causes catastrophic queue overflow.
        bool slot_valid = true;
        {
            int spin_count = 0;
            auto spin_start = std::chrono::steady_clock::now();
            while (slot_ready_[slot].value.load(std::memory_order_acquire) == 0) {
                if (++spin_count <= 64) {
                    _mm_pause();  // x86 PAUSE: ~10ns, saves power, prevents pipeline flush
                } else {
                    std::this_thread::yield();
                }
                if (spin_count % 1024 == 0) {
                    // Check shutdown to prevent hard-lock during teardown
                    if (shutdown_.load(std::memory_order_relaxed)) {
                        slot_valid = false;
                        break;
                    }
                    // Warn if worker is severely descheduled (>100ms)
                    auto elapsed = std::chrono::steady_clock::now() - spin_start;
                    if (elapsed > std::chrono::milliseconds(100)) {
                        fprintf(stderr, "[EvalQueue] WARNING: slot %d not ready after 100ms "
                                "(worker descheduled?), still waiting...\n", slot);
                        spin_start = std::chrono::steady_clock::now();
                    }
                }
            }
        }

        if (!slot_valid) {
            // Shutdown requested while waiting — zero out this batch entry
            // to prevent garbage data from reaching the GPU. The result
            // will be discarded anyway since the evaluator loop checks shutdown.
            std::memset(batch_obs_buffer_ + static_cast<size_t>(b) * OBS_SIZE,
                        0, OBS_SIZE * sizeof(float));
            std::memset(batch_mask_buffer_ + static_cast<size_t>(b) * POLICY_SIZE,
                        0, POLICY_SIZE * sizeof(float));
            continue;
        }

        // Direct NHWC memcpy: staging is already NHWC, batch buffer stays NHWC.
        // Python receives NHWC and uses torch.permute (zero-copy metadata swap)
        // to create channels_last tensors for cuDNN's native NHWC kernels.
        std::memcpy(batch_obs_buffer_ + static_cast<size_t>(b) * OBS_SIZE,
                    staging_obs_buffer_ + static_cast<size_t>(slot) * OBS_SIZE,
                    OBS_SIZE * sizeof(float));

        // Copy mask directly from staging to batch buffer
        std::memcpy(batch_mask_buffer_ + static_cast<size_t>(b) * POLICY_SIZE,
                    staging_mask_buffer_ + static_cast<size_t>(slot) * POLICY_SIZE,
                    POLICY_SIZE * sizeof(float));

        // Clear ready flag for slot reuse (relaxed — no ordering needed,
        // slot won't be reused until after compaction under lock)
        slot_ready_[slot].value.store(0, std::memory_order_relaxed);
    }

    // Update metrics
    auto elapsed = std::chrono::steady_clock::now() - start;
    metrics_.gpu_wait_time_us.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count(),
        std::memory_order_relaxed
    );
    metrics_.total_batches.fetch_add(1, std::memory_order_relaxed);
    metrics_.total_leaves.fetch_add(batch_size, std::memory_order_relaxed);

    *out_obs_ptr = batch_obs_buffer_;
    *out_mask_ptr = batch_mask_buffer_;

    return static_cast<int>(batch_size);
}

void GlobalEvaluationQueue::submit_results(
    const float* policies,
    const float* values,
    int batch_size,
    const float* wdl_probs)
{
    if (batch_size <= 0 || batch_size > static_cast<int>(batch_mapping_.size())) {
        return;
    }

    // Write policies to the CURRENT double-buffer slot
    int buf_id = current_policy_buffer_;
    std::memcpy(batch_policy_buffers_[buf_id], policies, batch_size * POLICY_SIZE * sizeof(float));
    std::memcpy(batch_value_buffer_, values, batch_size * sizeof(float));

    // Track which workers need notification (avoid redundant notifications)
    bool workers_to_notify[MAX_WORKERS] = {false};

    // BATCH PREPARATION: Build lightweight results (no policy copy!)
    // Group results by worker to minimize lock acquisitions
    std::vector<EvalResult> results_by_worker[MAX_WORKERS];

    for (int i = 0; i < batch_size; ++i) {
        auto [worker_id, request_id, generation] = batch_mapping_[i];

        if (worker_id < 0 || worker_id >= static_cast<int>(MAX_WORKERS)) {
            continue;
        }

        EvalResult result;
        result.worker_id = worker_id;
        result.request_id = request_id;
        result.generation = generation;
        result.batch_index = i;           // Index within this batch's policy buffer
        result.buffer_id = static_cast<int8_t>(buf_id);  // Which double-buffer to read from
        result.value = values[i];

        // Copy WDL probabilities if provided (Phase 2: root WDL accumulation)
        if (wdl_probs != nullptr) {
            result.wdl[0] = wdl_probs[i * 3 + 0];
            result.wdl[1] = wdl_probs[i * 3 + 1];
            result.wdl[2] = wdl_probs[i * 3 + 2];
        }

        results_by_worker[worker_id].push_back(std::move(result));
        workers_to_notify[worker_id] = true;
    }

    // BATCHED DELIVERY: One lock acquisition per worker (not per result)
    for (size_t worker_id = 0; worker_id < MAX_WORKERS; ++worker_id) {
        if (!workers_to_notify[worker_id]) continue;

        auto& results = results_by_worker[worker_id];
        {
            std::lock_guard<std::mutex> lock(worker_mutexes_[worker_id]);
            for (auto& result : results) {
                worker_results_[worker_id].push_back(std::move(result));
            }
            // Single notification per worker after all results added
            worker_cvs_[worker_id].notify_one();
        }
    }

    // Flip to the other buffer for the NEXT batch
    current_policy_buffer_ = 1 - current_policy_buffer_;

    metrics_.total_results_returned.fetch_add(batch_size, std::memory_order_relaxed);
}

// ============================================================================
// Compaction (called under lock)
// ============================================================================

void GlobalEvaluationQueue::compact_remaining_locked() {
    // Move remaining staged requests' data to the front of staging buffers.
    // Called at the START of collect_batch (under lock), so all previous-batch
    // slots have been consumed and their slot_ready_ flags cleared in the
    // prior Phase 2.
    //
    // For remaining requests (not yet collected), we also need to wait for
    // their slot_ready_ flags before compacting, because the worker may still
    // be in Phase 2 (memcpy) when compaction fires. In practice, workers
    // finish their memcpy during the GPU's ~10ms compute cycle, so this
    // wait is nearly instant.
    size_t remaining = staged_requests_.size();

    for (size_t i = 0; i < remaining; ++i) {
        int old_slot = staged_requests_[i].staging_slot;

        // Bounded spin-wait for this slot's data to be fully written
        {
            int spin_count = 0;
            auto spin_start = std::chrono::steady_clock::now();
            while (slot_ready_[old_slot].value.load(std::memory_order_acquire) == 0) {
                std::this_thread::yield();
                if (++spin_count % 1024 == 0) {
                    if (shutdown_.load(std::memory_order_relaxed)) {
                        // Shutdown requested — abort compaction, leave state as-is.
                        // The staging buffer may be partially compacted, but that's
                        // OK since no more batches will be collected after shutdown.
                        staging_write_head_.store(
                            static_cast<int>(remaining), std::memory_order_release);
                        return;
                    }
                    auto elapsed = std::chrono::steady_clock::now() - spin_start;
                    if (elapsed > std::chrono::milliseconds(100)) {
                        fprintf(stderr, "[EvalQueue] WARNING: compaction waiting on slot %d "
                                "for >100ms (worker descheduled?)\n", old_slot);
                        spin_start = std::chrono::steady_clock::now();
                    }
                }
            }
        }

        if (old_slot != static_cast<int>(i)) {
            std::memcpy(staging_obs_buffer_ + i * OBS_SIZE,
                        staging_obs_buffer_ + old_slot * OBS_SIZE,
                        OBS_SIZE * sizeof(float));
            std::memcpy(staging_mask_buffer_ + i * POLICY_SIZE,
                        staging_mask_buffer_ + old_slot * POLICY_SIZE,
                        POLICY_SIZE * sizeof(float));

            // Move the ready flag to the new slot position
            slot_ready_[i].value.store(1, std::memory_order_relaxed);
            slot_ready_[old_slot].value.store(0, std::memory_order_relaxed);

            staged_requests_[i].staging_slot = static_cast<int>(i);
        }
        // If old_slot == i, slot is already in the right place — no move needed
    }

    staging_write_head_.store(static_cast<int>(remaining), std::memory_order_release);
    metrics_.pool_resets.fetch_add(1, std::memory_order_relaxed);
}

// ============================================================================
// Lifecycle Management
// ============================================================================

void GlobalEvaluationQueue::shutdown() {
    shutdown_.store(true, std::memory_order_release);

    // Wake up GPU thread
    queue_cv_.notify_all();

    // Wake up all worker threads
    for (size_t i = 0; i < MAX_WORKERS; ++i) {
        worker_cvs_[i].notify_all();
    }
}

void GlobalEvaluationQueue::reset() {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    shutdown_.store(false, std::memory_order_release);
    staged_requests_.clear();
    batch_mapping_.clear();
    batch_staging_slots_.clear();
    staging_write_head_.store(0, std::memory_order_release);
    current_policy_buffer_ = 0;
    needs_compaction_ = false;

    // Clear all slot ready flags
    for (size_t i = 0; i < queue_capacity_; ++i) {
        slot_ready_[i].value.store(0, std::memory_order_relaxed);
    }

    // Clear worker results and reset generation counters
    for (size_t i = 0; i < MAX_WORKERS; ++i) {
        std::lock_guard<std::mutex> wlock(worker_mutexes_[i]);
        worker_results_[i].clear();
        worker_request_ids_[i].store(0, std::memory_order_relaxed);
        worker_generation_[i].store(0, std::memory_order_relaxed);
    }

    metrics_.reset();
}

size_t GlobalEvaluationQueue::pending_count() const {
    // Note: This is a snapshot, may change immediately
    return static_cast<size_t>(staging_write_head_.load(std::memory_order_relaxed));
}

} // namespace selfplay
