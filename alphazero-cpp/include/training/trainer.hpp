#pragma once

#include <string>
#include <vector>
#include <memory>

namespace training {

class ReplayBuffer;

/**
 * Simple training coordinator for C++ backend.
 *
 * This class is minimal - it primarily serves as a holder for training
 * configuration and statistics. The actual training happens in Python
 * using PyTorch, following the implementation plan's approach of keeping
 * PyTorch in Python rather than using LibTorch.
 *
 * The Trainer can:
 * - Track training statistics
 * - Coordinate with ReplayBuffer
 * - Provide simple training loop helpers
 */
class Trainer {
public:
    struct Config {
        size_t batch_size = 256;
        size_t min_buffer_size = 1000;  // Minimum samples before training
        float learning_rate = 0.001f;

        // Training schedule
        size_t num_epochs_per_iteration = 5;
        size_t batches_per_epoch = 100;
    };

    struct Stats {
        uint64_t total_steps = 0;
        uint64_t total_samples_trained = 0;
        float last_loss = 0.0f;
        float last_policy_loss = 0.0f;
        float last_value_loss = 0.0f;
    };

    /**
     * Create trainer with configuration.
     */
    Trainer();
    explicit Trainer(const Config& config);

    ~Trainer() = default;

    /**
     * Check if buffer is ready for training.
     */
    bool is_ready(const ReplayBuffer& buffer) const;

    /**
     * Get training configuration.
     */
    const Config& get_config() const { return config_; }

    /**
     * Get training statistics.
     */
    const Stats& get_stats() const { return stats_; }

    /**
     * Update statistics after a training step.
     */
    void record_step(size_t batch_size, float loss, float policy_loss, float value_loss);

    /**
     * Reset statistics.
     */
    void reset_stats();

private:
    Config config_;
    Stats stats_;
};

} // namespace training
