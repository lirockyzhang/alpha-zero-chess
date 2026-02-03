#include "training/replay_buffer.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace training {

// Thread-local RNG for sampling
thread_local std::mt19937 ReplayBuffer::rng_(std::random_device{}());

ReplayBuffer::ReplayBuffer(size_t capacity)
    : capacity_(capacity)
{
    // Pre-allocate storage
    observations_.resize(capacity * OBS_SIZE);
    policies_.resize(capacity * POLICY_SIZE);
    values_.resize(capacity);

    // Initialize atomics
    write_pos_.value.store(0, std::memory_order_relaxed);
    total_added_.value.store(0, std::memory_order_relaxed);
    total_games_.value.store(0, std::memory_order_relaxed);
}

void ReplayBuffer::add_sample(
    const std::vector<float>& observation,
    const std::vector<float>& policy,
    float value
) {
    // Validate sizes (no lock needed for validation)
    if (observation.size() != OBS_SIZE) {
        std::cerr << "Warning: observation size mismatch: " << observation.size()
                  << " != " << OBS_SIZE << std::endl;
        return;
    }
    if (policy.size() != POLICY_SIZE) {
        std::cerr << "Warning: policy size mismatch: " << policy.size()
                  << " != " << POLICY_SIZE << std::endl;
        return;
    }

    // Atomically get write position (lock-free, no contention)
    // Each thread gets a unique position via fetch_add
    size_t pos = write_pos_.value.fetch_add(1, std::memory_order_relaxed) % capacity_;

    // Copy data to circular buffer
    // No lock needed - each thread writes to its unique position
    std::memcpy(observations_.data() + pos * OBS_SIZE,
                observation.data(),
                OBS_SIZE * sizeof(float));
    std::memcpy(policies_.data() + pos * POLICY_SIZE,
                policy.data(),
                POLICY_SIZE * sizeof(float));
    values_[pos] = value;

    // Update statistics (simple fetch_add, no CAS contention)
    // NOTE: current_size_ ELIMINATED - size() computes min(total_added, capacity)
    total_added_.value.fetch_add(1, std::memory_order_release);
}

void ReplayBuffer::add_batch(
    const std::vector<std::vector<float>>& observations,
    const std::vector<std::vector<float>>& policies,
    const std::vector<float>& values
) {
    size_t batch_size = observations.size();
    if (policies.size() != batch_size || values.size() != batch_size) {
        std::cerr << "Warning: batch size mismatch" << std::endl;
        return;
    }

    // Lock-free batch addition (no CAS loop!)
    for (size_t i = 0; i < batch_size; ++i) {
        if (observations[i].size() != OBS_SIZE || policies[i].size() != POLICY_SIZE) {
            continue; // Skip invalid samples
        }

        // Atomically get write position
        size_t pos = write_pos_.value.fetch_add(1, std::memory_order_relaxed) % capacity_;

        // Copy data to circular buffer
        std::memcpy(observations_.data() + pos * OBS_SIZE,
                    observations[i].data(),
                    OBS_SIZE * sizeof(float));
        std::memcpy(policies_.data() + pos * POLICY_SIZE,
                    policies[i].data(),
                    POLICY_SIZE * sizeof(float));
        values_[pos] = values[i];

        // Update statistics (no CAS!)
        total_added_.value.fetch_add(1, std::memory_order_release);
    }

    total_games_.value.fetch_add(1, std::memory_order_relaxed);
}

void ReplayBuffer::add_batch_raw(
    size_t batch_size,
    const float* observations_ptr,
    const float* policies_ptr,
    const float* values_ptr
) {
    // Zero-copy batch addition (lock-free, no CAS!)
    for (size_t i = 0; i < batch_size; ++i) {
        // Atomically get write position
        size_t pos = write_pos_.value.fetch_add(1, std::memory_order_relaxed) % capacity_;

        // Direct memory copy from NumPy arrays (zero-copy)
        std::memcpy(observations_.data() + pos * OBS_SIZE,
                    observations_ptr + i * OBS_SIZE,
                    OBS_SIZE * sizeof(float));
        std::memcpy(policies_.data() + pos * POLICY_SIZE,
                    policies_ptr + i * POLICY_SIZE,
                    POLICY_SIZE * sizeof(float));
        values_[pos] = values_ptr[i];

        // Update statistics (no CAS!)
        total_added_.value.fetch_add(1, std::memory_order_release);
    }

    total_games_.value.fetch_add(1, std::memory_order_relaxed);
}

bool ReplayBuffer::sample(
    size_t batch_size,
    std::vector<float>& observations,
    std::vector<float>& policies,
    std::vector<float>& values
) {
    // Lock-free size check (computed from total_added)
    size_t current = size();
    if (current < batch_size) {
        return false; // Not enough samples
    }

    // Resize output buffers
    observations.resize(batch_size * OBS_SIZE);
    policies.resize(batch_size * POLICY_SIZE);
    values.resize(batch_size);

    // Random sampling with replacement (thread-local RNG)
    std::uniform_int_distribution<size_t> dist(0, current - 1);

    for (size_t i = 0; i < batch_size; ++i) {
        size_t idx = dist(rng_);

        // Direct memcpy - compiler auto-vectorizes this better than manual SIMD
        // (removed fast_memcpy_avx2 wrapper that was causing regression)
        std::memcpy(
            observations.data() + i * OBS_SIZE,
            observations_.data() + idx * OBS_SIZE,
            OBS_SIZE * sizeof(float)
        );
        std::memcpy(
            policies.data() + i * POLICY_SIZE,
            policies_.data() + idx * POLICY_SIZE,
            POLICY_SIZE * sizeof(float)
        );
        values[i] = values_[idx];
    }

    return true;
}

bool ReplayBuffer::save(const std::string& path) const {
    std::lock_guard<std::mutex> lock(io_mutex_);

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file for writing: " << path << std::endl;
        return false;
    }

    // Write header
    const char* magic = "RPBF";  // Replay Buffer File
    file.write(magic, 4);

    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write metadata (load atomics)
    size_t capacity = capacity_;
    size_t buf_size = size();  // Computed from total_added
    uint64_t total_add = total_added_.value.load(std::memory_order_relaxed);
    uint64_t total_g = total_games_.value.load(std::memory_order_relaxed);

    file.write(reinterpret_cast<const char*>(&capacity), sizeof(capacity));
    file.write(reinterpret_cast<const char*>(&buf_size), sizeof(buf_size));
    file.write(reinterpret_cast<const char*>(&total_add), sizeof(total_add));
    file.write(reinterpret_cast<const char*>(&total_g), sizeof(total_g));

    // Write data (only the valid portion of circular buffer)
    if (buf_size > 0) {
        // For simplicity, write the first buf_size samples
        // (This works because we're saving/loading, not mid-operation)
        file.write(reinterpret_cast<const char*>(observations_.data()),
                   buf_size * OBS_SIZE * sizeof(float));
        file.write(reinterpret_cast<const char*>(policies_.data()),
                   buf_size * POLICY_SIZE * sizeof(float));
        file.write(reinterpret_cast<const char*>(values_.data()),
                   buf_size * sizeof(float));
    }

    file.close();
    return file.good();
}

bool ReplayBuffer::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file for reading: " << path << std::endl;
        return false;
    }

    // Read and verify header
    char magic[5] = {0};
    file.read(magic, 4);
    if (std::string(magic) != "RPBF") {
        std::cerr << "Error: Invalid replay buffer file format" << std::endl;
        return false;
    }

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        std::cerr << "Error: Unsupported replay buffer version: " << version << std::endl;
        return false;
    }

    // Read metadata
    size_t saved_capacity, saved_size;
    uint64_t total_add, total_g;

    file.read(reinterpret_cast<char*>(&saved_capacity), sizeof(saved_capacity));
    file.read(reinterpret_cast<char*>(&saved_size), sizeof(saved_size));
    file.read(reinterpret_cast<char*>(&total_add), sizeof(total_add));
    file.read(reinterpret_cast<char*>(&total_g), sizeof(total_g));

    if (saved_size > saved_capacity || saved_size > capacity_) {
        std::cerr << "Error: Invalid buffer size in file" << std::endl;
        return false;
    }

    std::lock_guard<std::mutex> lock(io_mutex_);

    // Load data
    if (saved_size > 0) {
        file.read(reinterpret_cast<char*>(observations_.data()),
                  saved_size * OBS_SIZE * sizeof(float));
        file.read(reinterpret_cast<char*>(policies_.data()),
                  saved_size * POLICY_SIZE * sizeof(float));
        file.read(reinterpret_cast<char*>(values_.data()),
                  saved_size * sizeof(float));
    }

    // Update buffer state (store atomics)
    write_pos_.value.store(saved_size % capacity_, std::memory_order_relaxed);
    total_added_.value.store(total_add, std::memory_order_release);
    total_games_.value.store(total_g, std::memory_order_relaxed);

    file.close();
    return file.good();
}

void ReplayBuffer::clear() {
    // Lock-free clear (atomic stores)
    write_pos_.value.store(0, std::memory_order_relaxed);
    // Note: We reset total_added_ because size() depends on it
    total_added_.value.store(0, std::memory_order_release);
    // Note: We don't reset total_games_ - it's a cumulative stat
}

ReplayBuffer::Stats ReplayBuffer::get_stats() const {
    // Lock-free stats (load atomics)
    size_t buf_size = size();  // Computed from total_added
    return Stats{
        buf_size,
        capacity_,
        total_added_.value.load(std::memory_order_relaxed),
        total_games_.value.load(std::memory_order_relaxed),
        static_cast<float>(buf_size) / capacity_
    };
}

} // namespace training
