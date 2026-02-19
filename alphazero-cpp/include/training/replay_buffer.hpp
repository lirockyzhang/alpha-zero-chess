#pragma once

#include <vector>
#include <atomic>
#include <random>
#include <mutex>
#include <memory>
#include <cstdint>
#include <string>

namespace training {

// Forward declaration
class SumTree;

// ============================================================================
// Per-Sample Metadata
// ============================================================================

enum class Termination : uint8_t {
    CHECKMATE = 0,
    STALEMATE = 1,
    REPETITION = 2,
    FIFTY_MOVE = 3,
    INSUFFICIENT = 4,
    MAX_MOVES = 5,
    UNKNOWN = 255
};

struct SampleMeta {
    uint16_t iteration;      // training iteration that generated this
    uint8_t  game_result;    // 0=white_win, 1=draw, 2=black_win (0xFF=uninitialized)
    uint8_t  termination;    // Termination enum value
    uint16_t move_number;    // position in game (0-indexed ply)
    uint16_t game_length;    // total plies in source game
};
static_assert(sizeof(SampleMeta) == 8, "SampleMeta must be 8 bytes");

/**
 * Thread-safe circular replay buffer for training data.
 *
 * Stores (observation, policy, value, wdl_target) tuples from self-play games.
 * Supports:
 * - Thread-safe addition of samples
 * - Random batch sampling
 * - Optional WDL soft targets (3 floats: P(win), P(draw), P(loss))
 *
 * Implementation uses a circular buffer with atomic write position.
 */
class ReplayBuffer {
public:
    /**
     * Create replay buffer with fixed capacity.
     *
     * @param capacity Maximum number of samples to store
     */
    explicit ReplayBuffer(size_t capacity);

    ~ReplayBuffer();

    /**
     * Add a single sample to the buffer.
     * Thread-safe.
     *
     * @param observation Position encoding (8x8x123 = 7872 floats)
     * @param policy Target policy (4672 floats)
     * @param value Target value (1 float)
     * @param wdl_target Optional WDL soft target (3 floats: P(win), P(draw), P(loss))
     * @param soft_value Optional ERM risk-adjusted root value
     * @param meta Optional per-sample metadata (iteration, game_result, termination, etc.)
     */
    void add_sample(
        const std::vector<float>& observation,
        const std::vector<float>& policy,
        float value,
        const float* wdl_target = nullptr,
        const float* soft_value = nullptr,
        const SampleMeta* meta = nullptr
    );

    /**
     * Add a batch of samples from a game trajectory.
     * Thread-safe.
     *
     * @param observations Batch of observations (N x 7872)
     * @param policies Batch of policies (N x 4672)
     * @param values Batch of values (N)
     * @param wdl_targets Optional WDL soft targets (N x 3), nullptr to skip
     */
    void add_batch(
        const std::vector<std::vector<float>>& observations,
        const std::vector<std::vector<float>>& policies,
        const std::vector<float>& values,
        const float* wdl_targets = nullptr
    );

    /**
     * Add a batch of samples with zero-copy (from raw pointers).
     * Thread-safe. Used by Python bindings for NumPy arrays.
     *
     * @param batch_size Number of samples
     * @param observations_ptr Pointer to flat observations array (batch_size * 7872)
     * @param policies_ptr Pointer to flat policies array (batch_size * 4672)
     * @param values_ptr Pointer to values array (batch_size)
     * @param wdl_ptr Optional WDL soft targets (batch_size * 3), nullptr to skip
     * @param soft_values_ptr Optional ERM risk-adjusted values (batch_size)
     * @param meta_ptr Optional per-sample metadata array (batch_size)
     */
    void add_batch_raw(
        size_t batch_size,
        const float* observations_ptr,
        const float* policies_ptr,
        const float* values_ptr,
        const float* wdl_ptr = nullptr,
        const float* soft_values_ptr = nullptr,
        const SampleMeta* meta_ptr = nullptr
    );

    /**
     * Sample a random batch of training data.
     * Thread-safe.
     *
     * @param batch_size Number of samples to return
     * @param observations Output: sampled observations (batch_size x 7808)
     * @param policies Output: sampled policies (batch_size x 4672)
     * @param values Output: sampled values (batch_size)
     * @param wdl_targets Output: sampled WDL targets (batch_size x 3), nullptr to skip
     * @return true if successful, false if buffer has fewer than batch_size samples
     */
    bool sample(
        size_t batch_size,
        std::vector<float>& observations,
        std::vector<float>& policies,
        std::vector<float>& values,
        std::vector<float>* wdl_targets = nullptr,
        std::vector<float>* soft_values = nullptr
    );

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
     * Set the current training iteration (called before each self-play generation).
     * Metadata for newly added samples will use this iteration number.
     */
    void set_iteration(uint16_t iter) { current_iteration_ = iter; }

    /**
     * Get the current training iteration.
     */
    uint16_t current_iteration() const { return current_iteration_; }

    /**
     * Get W/D/L composition of buffer contents.
     */
    struct Composition {
        uint64_t wins;
        uint64_t draws;
        uint64_t losses;
    };
    Composition get_composition() const;

    /**
     * Save buffer contents to binary file (.rpbf format).
     * @param path Output file path
     * @return true if successful
     */
    bool save(const std::string& path) const;

    /**
     * Load buffer contents from binary file (.rpbf format).
     * @param path Input file path
     * @return true if successful
     */
    bool load(const std::string& path);

    // ========================================================================
    // Prioritized Experience Replay (PER)
    // ========================================================================

    /**
     * Enable PER with the given priority exponent (alpha).
     * Allocates sum-tree and initializes existing samples with uniform priority.
     * @param alpha Priority exponent (0=uniform, 0.6=recommended)
     */
    void enable_per(float alpha);

    /**
     * Check if PER is enabled.
     */
    bool per_enabled() const { return sum_tree_ != nullptr; }

    /**
     * Get the priority exponent (alpha).
     */
    float priority_exponent() const { return priority_exponent_; }

    /**
     * Sample a prioritized batch with IS weights.
     * Thread-safe (holds per_mutex_).
     *
     * @param batch_size Number of samples to return
     * @param beta IS correction exponent (0=no correction, 1=full correction)
     * @param observations Output: sampled observations
     * @param policies Output: sampled policies
     * @param values Output: sampled values
     * @param wdl_targets Output: sampled WDL targets (nullable)
     * @param soft_values Output: sampled soft values (nullable)
     * @param out_indices Output: indices of sampled leaves (for update_priorities)
     * @param out_is_weights Output: IS correction weights (normalized so max=1)
     * @return true if successful
     */
    bool sample_prioritized(
        size_t batch_size,
        float beta,
        std::vector<float>& observations,
        std::vector<float>& policies,
        std::vector<float>& values,
        std::vector<float>* wdl_targets,
        std::vector<float>* soft_values,
        std::vector<uint32_t>& out_indices,
        std::vector<float>& out_is_weights
    );

    /**
     * Update priorities for sampled indices.
     * Thread-safe (holds per_mutex_).
     *
     * @param indices Leaf indices (from sample_prioritized)
     * @param priorities New priority values (typically loss + epsilon)
     */
    void update_priorities(
        const std::vector<uint32_t>& indices,
        const std::vector<float>& priorities
    );

    /**
     * Get buffer statistics.
     */
    struct Stats {
        size_t size;
        size_t capacity;
        uint64_t total_added;
        uint64_t total_games;
        float utilization;
        uint64_t wins;
        uint64_t draws;
        uint64_t losses;
    };

    Stats get_stats() const;

private:
    // Buffer storage
    const size_t capacity_;
    std::vector<float> observations_;  // Flat array: capacity * 7872
    std::vector<float> policies_;      // Flat array: capacity * 4672
    std::vector<float> values_;        // Array: capacity
    std::vector<float> wdl_targets_;   // Flat array: capacity * 3 (WDL soft targets)
    std::vector<float> soft_values_;   // Array: capacity (ERM risk-adjusted root values)

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

    // Per-sample metadata (8 bytes each, parallel to obs/policy/value arrays)
    std::vector<SampleMeta> metadata_;

    // Composition counters (atomic, cache-line aligned)
    AlignedAtomicU64 wins_count_;
    AlignedAtomicU64 draws_count_;
    AlignedAtomicU64 losses_count_;

    // Current training iteration (set by Python before self-play)
    uint16_t current_iteration_{0};

    // Helper: update composition counters when writing to a slot
    void update_composition_counters(size_t pos, const SampleMeta& new_meta);

    // Random number generator (thread-local)
    thread_local static std::mt19937 rng_;

    // Constants
    static constexpr size_t OBS_SIZE = 8 * 8 * 123;  // 7872
    static constexpr size_t POLICY_SIZE = 4672;

    // Prioritized Experience Replay (PER) state
    std::unique_ptr<SumTree> sum_tree_;         // nullptr when PER disabled
    float priority_exponent_{0.0f};             // alpha
    bool tree_needs_rebuild_{false};            // set by add_sample, cleared by sample_prioritized
    mutable std::mutex per_mutex_;              // protects sum_tree_ during sample + update
};

} // namespace training
