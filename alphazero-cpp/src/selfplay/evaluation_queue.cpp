#include "../../include/selfplay/evaluation_queue.hpp"
#include <algorithm>
#include <stdexcept>

namespace selfplay {

// ============================================================================
// Constructor / Destructor
// ============================================================================

GlobalEvaluationQueue::GlobalEvaluationQueue(size_t max_batch_size, size_t queue_capacity)
    : max_batch_size_(max_batch_size)
    , queue_capacity_(queue_capacity)
    , obs_pool_(queue_capacity)
    , batch_obs_buffer_(nullptr)
    , batch_mask_buffer_(nullptr)
    , batch_policy_buffer_(nullptr)
    , batch_value_buffer_(nullptr)
{
    // Pre-allocate request queue
    pending_requests_.reserve(queue_capacity);

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
    batch_mask_buffer_ = static_cast<float*>(_aligned_malloc(mask_bytes, 64));
    batch_policy_buffer_ = static_cast<float*>(_aligned_malloc(policy_bytes, 64));
    batch_value_buffer_ = static_cast<float*>(_aligned_malloc(value_bytes, 64));
    #else
    batch_obs_buffer_ = static_cast<float*>(std::aligned_alloc(64, obs_bytes));
    batch_mask_buffer_ = static_cast<float*>(std::aligned_alloc(64, mask_bytes));
    batch_policy_buffer_ = static_cast<float*>(std::aligned_alloc(64, policy_bytes));
    batch_value_buffer_ = static_cast<float*>(std::aligned_alloc(64, value_bytes));
    #endif

    if (!batch_obs_buffer_ || !batch_mask_buffer_ ||
        !batch_policy_buffer_ || !batch_value_buffer_) {
        free_batch_buffers();
        throw std::bad_alloc();
    }
}

void GlobalEvaluationQueue::free_batch_buffers() {
    #ifdef _WIN32
    if (batch_obs_buffer_) _aligned_free(batch_obs_buffer_);
    if (batch_mask_buffer_) _aligned_free(batch_mask_buffer_);
    if (batch_policy_buffer_) _aligned_free(batch_policy_buffer_);
    if (batch_value_buffer_) _aligned_free(batch_value_buffer_);
    #else
    if (batch_obs_buffer_) std::free(batch_obs_buffer_);
    if (batch_mask_buffer_) std::free(batch_mask_buffer_);
    if (batch_policy_buffer_) std::free(batch_policy_buffer_);
    if (batch_value_buffer_) std::free(batch_value_buffer_);
    #endif

    batch_obs_buffer_ = nullptr;
    batch_mask_buffer_ = nullptr;
    batch_policy_buffer_ = nullptr;
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

    int queued = 0;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        for (int i = 0; i < num_leaves; ++i) {
            // Allocate slot in observation pool
            int slot = obs_pool_.allocate();
            if (slot < 0) {
                // Pool full, can't accept more - track for diagnostics
                metrics_.pool_exhaustion_count.fetch_add(1, std::memory_order_relaxed);
                metrics_.submission_drops.fetch_add(num_leaves - i, std::memory_order_relaxed);
                break;
            }

            // Copy observation and mask to pool
            float* obs_ptr = obs_pool_.get_obs_ptr(slot);
            float* mask_ptr = obs_pool_.get_mask_ptr(slot);

            std::memcpy(obs_ptr, observations + i * OBS_SIZE, OBS_SIZE * sizeof(float));
            std::memcpy(mask_ptr, legal_masks + i * POLICY_SIZE, POLICY_SIZE * sizeof(float));

            // Create request
            EvalRequest req;
            req.worker_id = worker_id;
            req.request_id = worker_request_ids_[worker_id].fetch_add(1, std::memory_order_relaxed);
            req.observation = obs_ptr;
            req.legal_mask = mask_ptr;

            pending_requests_.push_back(req);
            out_request_ids.push_back(req.request_id);
            queued++;
        }

        // Notify GPU thread INSIDE the lock to avoid lost wakeup race condition
        if (queued > 0) {
            metrics_.total_requests_submitted.fetch_add(queued, std::memory_order_relaxed);
            queue_cv_.notify_one();
        }

        // Track partial submissions for diagnostics
        if (queued < num_leaves && queued > 0) {
            metrics_.partial_submissions.fetch_add(1, std::memory_order_relaxed);
        }
    }

    return queued;
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

    // Retrieve available results
    auto& results = worker_results_[worker_id];
    int to_retrieve = std::min(static_cast<int>(results.size()), max_results);

    for (int i = 0; i < to_retrieve; ++i) {
        const auto& result = results[i];

        // Copy policy
        std::memcpy(out_policies + i * POLICY_SIZE,
                    result.policy.data(),
                    POLICY_SIZE * sizeof(float));

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

// ============================================================================
// GPU Thread Interface
// ============================================================================

int GlobalEvaluationQueue::collect_batch(
    float** out_obs_ptr,
    float** out_mask_ptr,
    int timeout_ms)
{
    auto start = std::chrono::steady_clock::now();

    // Local staging area for requests - allows unlocking early
    std::vector<EvalRequest> staging;
    staging.reserve(max_batch_size_);
    bool reset_pool = false;

    {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        // ADAPTIVE BATCHING: Use a low threshold to avoid root evaluation stalls
        // Root evaluations submit 1 leaf per worker, so we need to wake quickly.
        // For simulation phase (64 leaves per worker), we'll naturally get larger batches.
        // Target: ~8 leaves or any work after timeout (whichever comes first)
        size_t target_batch = std::max(size_t(1), std::min(size_t(8), max_batch_size_ / 64));

        queue_cv_.wait_for(
            lock,
            std::chrono::milliseconds(timeout_ms),
            [this, target_batch]() {
                return pending_requests_.size() >= target_batch ||
                       shutdown_.load(std::memory_order_acquire);
            }
        );

        // No work available
        if (pending_requests_.empty()) {
            return 0;
        }

        // Collect up to max_batch_size requests
        size_t batch_size = std::min(pending_requests_.size(), max_batch_size_);

        // QUICK SWAP: Move requests to staging area to minimize lock time
        std::move(pending_requests_.begin(),
                  pending_requests_.begin() + batch_size,
                  std::back_inserter(staging));
        pending_requests_.erase(pending_requests_.begin(),
                                pending_requests_.begin() + batch_size);

        // Check if pool should be reset (before unlocking)
        // Reset more aggressively when below threshold to prevent exhaustion
        // This is safe because we hold the queue_mutex_ which blocks allocate()
        reset_pool = pending_requests_.size() < queue_capacity_ / 4;

        // Lock released here - workers can now submit new requests
    }

    // HEAVY OPERATIONS OUTSIDE LOCK: memcpy to batch buffers
    size_t batch_size = staging.size();
    batch_mapping_.clear();
    batch_mapping_.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        const auto& req = staging[i];

        // Copy to contiguous batch buffer
        std::memcpy(batch_obs_buffer_ + i * OBS_SIZE,
                    req.observation,
                    OBS_SIZE * sizeof(float));

        std::memcpy(batch_mask_buffer_ + i * POLICY_SIZE,
                    req.legal_mask,
                    POLICY_SIZE * sizeof(float));

        // Track mapping for result distribution
        batch_mapping_.emplace_back(req.worker_id, req.request_id);
    }

    // Reset observation pool when queue is low (need to re-lock briefly)
    // This prevents pool exhaustion by freeing slots more frequently
    if (reset_pool) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        // Re-check - another thread may have added requests
        // Reset when below threshold to prevent exhaustion under load
        if (pending_requests_.size() < queue_capacity_ / 4) {
            obs_pool_.reset();
            metrics_.pool_resets.fetch_add(1, std::memory_order_relaxed);
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

    *out_obs_ptr = batch_obs_buffer_;
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

    // Track which workers need notification (avoid redundant notifications)
    bool workers_to_notify[MAX_WORKERS] = {false};

    // BATCH PREPARATION: Build results outside locks first
    // Group results by worker to minimize lock acquisitions
    std::vector<EvalResult> results_by_worker[MAX_WORKERS];

    for (int i = 0; i < batch_size; ++i) {
        auto [worker_id, request_id] = batch_mapping_[i];

        if (worker_id < 0 || worker_id >= static_cast<int>(MAX_WORKERS)) {
            continue;
        }

        EvalResult result;
        result.worker_id = worker_id;
        result.request_id = request_id;
        result.value = values[i];

        // Copy policy
        std::memcpy(result.policy.data(),
                    policies + i * POLICY_SIZE,
                    POLICY_SIZE * sizeof(float));

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
    pending_requests_.clear();
    batch_mapping_.clear();
    obs_pool_.reset();

    // Clear worker results
    for (size_t i = 0; i < MAX_WORKERS; ++i) {
        std::lock_guard<std::mutex> wlock(worker_mutexes_[i]);
        worker_results_[i].clear();
        worker_request_ids_[i].store(0, std::memory_order_relaxed);
    }

    metrics_.reset();
}

size_t GlobalEvaluationQueue::pending_count() const {
    // Note: This is a snapshot, may change immediately
    return obs_pool_.allocated();
}

} // namespace selfplay
