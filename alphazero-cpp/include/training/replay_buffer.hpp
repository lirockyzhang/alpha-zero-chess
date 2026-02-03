#pragma once

#include <vector>
#include <atomic>
#include <mutex>
#include <random>
#include <string>
#include <cstdint>

namespace training {

/**
 * Thread-safe circular replay buffer for training data.
 *
 * Stores (observation, policy, value) tuples from self-play games.
 * Supports:
 * - Thread-safe addition of samples
 * - Random batch sampling
 * - Save/load to disk (.npz format compatible with Python)
 *
 * Implementation uses a circular buffer with mutex protection for simplicity.
 * Lock-free implementation can be added later if profiling shows contention.
 */
class ReplayBuffer {
public:
    /**
     * Create replay buffer with fixed capacity.
     *
     * @param capacity Maximum number of samples to store
     */
    explicit ReplayBuffer(size_t capacity);

    ~ReplayBuffer() = default;

    /**
     * Add a single sample to the buffer.
     * Thread-safe.
     *
     * @param observation Position encoding (8x8x122 = 7808 floats)
     * @param policy Target policy (4672 floats)
     * @param value Target value (1 float)
     */
    void add_sample(
        const std::vector<float>& observation,
        const std::vector<float>& policy,
        float value
    );

    /**
     * Add a batch of samples from a game trajectory.
     * Thread-safe.
     *
     * @param observations Batch of observations (N x 7808)
     * @param policies Batch of policies (N x 4672)
     * @param values Batch of values (N)
     */
    void add_batch(
        const std::vector<std::vector<float>>& observations,
        const std::vector<std::vector<float>>& policies,
        const std::vector<float>& values
    );

    /**
     * Add a batch of samples with zero-copy (from raw pointers).
     * Thread-safe. Used by Python bindings for NumPy arrays.
     *
     * @param batch_size Number of samples
     * @param observations_ptr Pointer to flat observations array (batch_size * 7808)
     * @param policies_ptr Pointer to flat policies array (batch_size * 4672)
     * @param values_ptr Pointer to values array (batch_size)
     */
    void add_batch_raw(
        size_t batch_size,
        const float* observations_ptr,
        const float* policies_ptr,
        const float* values_ptr
    );

    /**
     * Sample a random batch of training data.
     * Thread-safe.
     *
     * @param batch_size Number of samples to return
     * @param observations Output: sampled observations (batch_size x 7808)
     * @param policies Output: sampled policies (batch_size x 4672)
     * @param values Output: sampled values (batch_size)
     * @return true if successful, false if buffer has fewer than batch_size samples
     */
    bool sample(
        size_t batch_size,
        std::vector<float>& observations,
        std::vector<float>& policies,
        std::vector<float>& values
    );

    /**
     * Save replay buffer to disk (.npz format).
     *
     * Format compatible with NumPy:
     * - observations: float32 array (size, 7808)
     * - policies: float32 array (size, 4672)
     * - values: float32 array (size,)
     * - metadata: capacity, total_added, total_games
     *
     * @param path File path to save to
     * @return true if successful, false on error
     */
    bool save(const std::string& path) const;

    /**
     * Load replay buffer from disk (.npz format).
     * Replaces current buffer contents.
     *
     * @param path File path to load from
     * @return true if successful, false on error
     */
    bool load(const std::string& path);

    /**
     * Get current number of samples in buffer.
     * Lock-free: computed as min(total_added, capacity).
     * No CAS contention - just a simple load and comparison.
     */
    size_t size() const {
        uint64_t added = total_added_.value.load(std::memory_order_acquire);
        return static_cast<size_t>(std::min(added, static_cast<uint64_t>(capacity_)));
    }

    /**
     * Get buffer capacity.
     */
    size_t capacity() const {
        return capacity_;
    }

    /**
     * Get total number of samples ever added (including overwrites).
     */
    uint64_t total_added() const {
        return total_added_.value.load(std::memory_order_relaxed);
    }

    /**
     * Get total number of games added.
     */
    uint64_t total_games() const {
        return total_games_.value.load(std::memory_order_relaxed);
    }

    /**
     * Check if buffer has enough samples for training.
     * Lock-free: no CAS contention.
     */
    bool is_ready(size_t min_size) const {
        return size() >= min_size;
    }

    /**
     * Clear all samples from buffer.
     */
    void clear();

    /**
     * Get buffer statistics.
     */
    struct Stats {
        size_t size;
        size_t capacity;
        uint64_t total_added;
        uint64_t total_games;
        float utilization;
    };

    Stats get_stats() const;

private:
    // Buffer storage
    const size_t capacity_;
    std::vector<float> observations_;  // Flat array: capacity * 7808
    std::vector<float> policies_;      // Flat array: capacity * 4672
    std::vector<float> values_;        // Array: capacity

    // Cache-line aligned atomic counters to prevent false sharing
    // Each counter is on its own 64-byte cache line
    struct alignas(64) AlignedAtomicSize {
        std::atomic<size_t> value{0};
    };

    struct alignas(64) AlignedAtomicU64 {
        std::atomic<uint64_t> value{0};
    };

    // Circular buffer state (lock-free atomic, cache-line isolated)
    AlignedAtomicSize write_pos_;      // Next position to write

    // Statistics (cache-line isolated to prevent false sharing)
    AlignedAtomicU64 total_added_;     // Total samples ever added
    AlignedAtomicU64 total_games_;     // Total games ever added

    // NOTE: current_size_ ELIMINATED!
    // Size is computed as: min(total_added_, capacity_)
    // This removes the CAS contention entirely.

    // Thread safety (only for save/load operations)
    mutable std::mutex io_mutex_;

    // Random number generator (thread-local)
    thread_local static std::mt19937 rng_;

    // Constants
    static constexpr size_t OBS_SIZE = 8 * 8 * 122;  // 7808
    static constexpr size_t POLICY_SIZE = 4672;
};

} // namespace training
