#include "../../include/selfplay/evaluation_queue.hpp"
#include <algorithm>
#include <stdexcept>
#include <thread>

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
    , batch_obs_nchw_buffer_(nullptr)
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

    // Pre-allocate staged request metadata queue
    staged_requests_.reserve(queue_capacity);

    // Pre-allocate batch mapping
    batch_mapping_.reserve(max_batch_size);

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
    #ifdef _WIN32
    batch_obs_buffer_ = static_cast<float*>(_aligned_malloc(obs_bytes, 64));
    batch_obs_nchw_buffer_ = static_cast<float*>(_aligned_malloc(obs_bytes, 64));
    batch_mask_buffer_ = static_cast<float*>(_aligned_malloc(mask_bytes, 64));
    batch_policy_buffers_[0] = static_cast<float*>(_aligned_malloc(policy_bytes, 64));
    batch_policy_buffers_[1] = static_cast<float*>(_aligned_malloc(policy_bytes, 64));
    batch_value_buffer_ = static_cast<float*>(_aligned_malloc(value_bytes, 64));
    #else
    batch_obs_buffer_ = static_cast<float*>(std::aligned_alloc(64, obs_bytes));
    batch_obs_nchw_buffer_ = static_cast<float*>(std::aligned_alloc(64, obs_bytes));
    batch_mask_buffer_ = static_cast<float*>(std::aligned_alloc(64, mask_bytes));
    batch_policy_buffers_[0] = static_cast<float*>(std::aligned_alloc(64, policy_bytes));
    batch_policy_buffers_[1] = static_cast<float*>(std::aligned_alloc(64, policy_bytes));
    batch_value_buffer_ = static_cast<float*>(std::aligned_alloc(64, value_bytes));
    #endif

    if (!batch_obs_buffer_ || !batch_obs_nchw_buffer_ || !batch_mask_buffer_ ||
        !batch_policy_buffers_[0] || !batch_policy_buffers_[1] || !batch_value_buffer_) {
        free_batch_buffers();
        throw std::bad_alloc();
    }
}

void GlobalEvaluationQueue::free_batch_buffers() {
    #ifdef _WIN32
    if (batch_obs_buffer_) _aligned_free(batch_obs_buffer_);
    if (batch_obs_nchw_buffer_) _aligned_free(batch_obs_nchw_buffer_);
    if (batch_mask_buffer_) _aligned_free(batch_mask_buffer_);
    if (batch_policy_buffers_[0]) _aligned_free(batch_policy_buffers_[0]);
    if (batch_policy_buffers_[1]) _aligned_free(batch_policy_buffers_[1]);
    if (batch_value_buffer_) _aligned_free(batch_value_buffer_);
    #else
    if (batch_obs_buffer_) std::free(batch_obs_buffer_);
    if (batch_obs_nchw_buffer_) std::free(batch_obs_nchw_buffer_);
    if (batch_mask_buffer_) std::free(batch_mask_buffer_);
    if (batch_policy_buffers_[0]) std::free(batch_policy_buffers_[0]);
    if (batch_policy_buffers_[1]) std::free(batch_policy_buffers_[1]);
    if (batch_value_buffer_) std::free(batch_value_buffer_);
    #endif

    batch_obs_buffer_ = nullptr;
    batch_obs_nchw_buffer_ = nullptr;
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

    // Step 1: Claim staging slots atomically (no lock needed for data writes)
    int first_slot = staging_write_head_.fetch_add(num_leaves, std::memory_order_relaxed);
    int actual_leaves = num_leaves;

    // Check if we exceeded capacity
    if (first_slot + num_leaves > static_cast<int>(queue_capacity_)) {
        actual_leaves = std::max(0, static_cast<int>(queue_capacity_) - first_slot);
        if (actual_leaves < num_leaves) {
            // Roll back the excess
            staging_write_head_.fetch_sub(num_leaves - actual_leaves, std::memory_order_relaxed);
            metrics_.pool_exhaustion_count.fetch_add(1, std::memory_order_relaxed);
            metrics_.submission_drops.fetch_add(num_leaves - actual_leaves, std::memory_order_relaxed);
        }
    }

    if (actual_leaves <= 0) {
        return 0;
    }

    // Step 2: Write observation and mask data directly into staging buffers (no lock!)
    for (int i = 0; i < actual_leaves; ++i) {
        int slot = first_slot + i;
        std::memcpy(staging_obs_buffer_ + slot * OBS_SIZE,
                    observations + i * OBS_SIZE,
                    OBS_SIZE * sizeof(float));
        std::memcpy(staging_mask_buffer_ + slot * POLICY_SIZE,
                    legal_masks + i * POLICY_SIZE,
                    POLICY_SIZE * sizeof(float));
    }

    // Step 3: Push lightweight metadata under lock (no data copies here)
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
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
    int timeout_ms)
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
    int max_results)
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

    // Phase 1 (under queue_mutex_): Collect staged requests, copy staging → batch buffers
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        // PHASE 1a: Wait for at least 1 request (or timeout/shutdown).
        // Using target=1 ensures we wake on the FIRST notification instead of
        // requiring 8+ requests to accumulate. Notifications sent while we were
        // busy (inference/result distribution) are lost, so we must be responsive
        // to the first one that arrives while we're actually waiting.
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

        // PHASE 1b: Brief coalescing window — let more workers submit before
        // we grab the batch. Release the lock momentarily so blocked workers
        // can push their metadata, then re-acquire and collect everything.
        if (staged_requests_.size() < max_batch_size_) {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            lock.lock();
        }

        // Collect up to max_batch_size requests
        batch_size = std::min(staged_requests_.size(), max_batch_size_);

        // Build batch mapping and copy data from staging slots to batch buffers
        batch_mapping_.clear();
        batch_mapping_.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
            const auto& req = staged_requests_[i];

            // Copy from staging slot directly to batch buffer (NHWC)
            std::memcpy(batch_obs_buffer_ + i * OBS_SIZE,
                        staging_obs_buffer_ + req.staging_slot * OBS_SIZE,
                        OBS_SIZE * sizeof(float));

            std::memcpy(batch_mask_buffer_ + i * POLICY_SIZE,
                        staging_mask_buffer_ + req.staging_slot * POLICY_SIZE,
                        POLICY_SIZE * sizeof(float));

            batch_mapping_.emplace_back(req.worker_id, req.request_id, req.generation);
        }

        // Remove collected requests and compact staging
        staged_requests_.erase(staged_requests_.begin(),
                                staged_requests_.begin() + batch_size);

        // STAGING COMPACTION: If remaining requests exist, compact their staging slots
        // to the front so the write head can be reset
        size_t remaining = staged_requests_.size();
        if (remaining > 0) {
            for (size_t i = 0; i < remaining; ++i) {
                int old_slot = staged_requests_[i].staging_slot;
                if (old_slot != static_cast<int>(i)) {
                    std::memcpy(staging_obs_buffer_ + i * OBS_SIZE,
                                staging_obs_buffer_ + old_slot * OBS_SIZE,
                                OBS_SIZE * sizeof(float));
                    std::memcpy(staging_mask_buffer_ + i * POLICY_SIZE,
                                staging_mask_buffer_ + old_slot * POLICY_SIZE,
                                POLICY_SIZE * sizeof(float));
                    staged_requests_[i].staging_slot = static_cast<int>(i);
                }
            }
        }
        staging_write_head_.store(static_cast<int>(remaining), std::memory_order_release);
        metrics_.pool_resets.fetch_add(1, std::memory_order_relaxed);
    }
    // Lock released — workers can submit new leaves during Phase 2

    // Phase 2 (NO lock): NHWC→NCHW transpose on batch_obs_buffer_ → batch_obs_nchw_buffer_
    // Safe: both buffers are only accessed by the GPU thread
    for (size_t b = 0; b < batch_size; ++b) {
        const float* nhwc = batch_obs_buffer_ + b * OBS_SIZE;
        float* nchw = batch_obs_nchw_buffer_ + b * OBS_SIZE;
        for (int h = 0; h < 8; ++h) {
            for (int w = 0; w < 8; ++w) {
                for (int c = 0; c < 122; ++c) {
                    nchw[c * 64 + h * 8 + w] = nhwc[h * (8 * 122) + w * 122 + c];
                }
            }
        }
    }

    // Update metrics
    auto elapsed = std::chrono::steady_clock::now() - start;
    metrics_.gpu_wait_time_us.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count(),
        std::memory_order_relaxed
    );
    metrics_.total_batches.fetch_add(1, std::memory_order_relaxed);
    metrics_.total_leaves.fetch_add(batch_size, std::memory_order_relaxed);

    *out_obs_ptr = batch_obs_nchw_buffer_;
    *out_mask_ptr = batch_mask_buffer_;

    return static_cast<int>(batch_size);
}

void GlobalEvaluationQueue::submit_results(
    const float* policies,
    const float* values,
    int batch_size)
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
    staging_write_head_.store(0, std::memory_order_release);
    current_policy_buffer_ = 0;

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
