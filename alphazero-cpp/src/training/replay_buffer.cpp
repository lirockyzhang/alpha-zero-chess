#include "training/replay_buffer.hpp"
#include <iostream>
#include <fstream>
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
    wdl_targets_.resize(capacity * 3, 0.0f);  // 12 bytes per sample (~12MB for 1M capacity)
    soft_values_.resize(capacity, 0.0f);       // 4 bytes per sample (~4MB for 1M capacity)

    // Initialize metadata with sentinel game_result=0xFF (uninitialized)
    SampleMeta sentinel{};
    sentinel.game_result = 0xFF;
    metadata_.resize(capacity, sentinel);

    // Initialize atomics
    write_pos_.value.store(0, std::memory_order_relaxed);
    total_added_.value.store(0, std::memory_order_relaxed);
    total_games_.value.store(0, std::memory_order_relaxed);
    wins_count_.value.store(0, std::memory_order_relaxed);
    draws_count_.value.store(0, std::memory_order_relaxed);
    losses_count_.value.store(0, std::memory_order_relaxed);
}

void ReplayBuffer::update_composition_counters(size_t pos, const SampleMeta& new_meta) {
    // Decrement old counter if overwriting a valid (non-sentinel) entry
    uint8_t old_result = metadata_[pos].game_result;
    if (old_result != 0xFF) {
        if (old_result == 0) wins_count_.value.fetch_sub(1, std::memory_order_relaxed);
        else if (old_result == 1) draws_count_.value.fetch_sub(1, std::memory_order_relaxed);
        else if (old_result == 2) losses_count_.value.fetch_sub(1, std::memory_order_relaxed);
    }

    // Write new metadata
    metadata_[pos] = new_meta;

    // Increment new counter
    if (new_meta.game_result == 0) wins_count_.value.fetch_add(1, std::memory_order_relaxed);
    else if (new_meta.game_result == 1) draws_count_.value.fetch_add(1, std::memory_order_relaxed);
    else if (new_meta.game_result == 2) losses_count_.value.fetch_add(1, std::memory_order_relaxed);
}

void ReplayBuffer::add_sample(
    const std::vector<float>& observation,
    const std::vector<float>& policy,
    float value,
    const float* wdl_target,
    const float* soft_value,
    const SampleMeta* meta
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

    // Copy WDL target if provided, otherwise zero-fill
    if (wdl_target != nullptr) {
        std::memcpy(wdl_targets_.data() + pos * 3, wdl_target, 3 * sizeof(float));
    } else {
        wdl_targets_[pos * 3 + 0] = 0.0f;
        wdl_targets_[pos * 3 + 1] = 0.0f;
        wdl_targets_[pos * 3 + 2] = 0.0f;
    }

    // Copy ERM risk-adjusted value if provided
    soft_values_[pos] = (soft_value != nullptr) ? *soft_value : 0.0f;

    // Update metadata and composition counters
    SampleMeta new_meta;
    if (meta != nullptr) {
        new_meta = *meta;
    } else {
        // Default: draw, unknown termination, current iteration
        new_meta = {current_iteration_, 1, static_cast<uint8_t>(Termination::UNKNOWN), 0, 0};
    }
    update_composition_counters(pos, new_meta);

    // Update statistics (simple fetch_add, no CAS contention)
    // NOTE: current_size_ ELIMINATED - size() computes min(total_added, capacity)
    total_added_.value.fetch_add(1, std::memory_order_release);
}

void ReplayBuffer::add_batch(
    const std::vector<std::vector<float>>& observations,
    const std::vector<std::vector<float>>& policies,
    const std::vector<float>& values,
    const float* wdl_targets
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

        if (wdl_targets != nullptr) {
            std::memcpy(wdl_targets_.data() + pos * 3, wdl_targets + i * 3, 3 * sizeof(float));
        } else {
            wdl_targets_[pos * 3 + 0] = 0.0f;
            wdl_targets_[pos * 3 + 1] = 0.0f;
            wdl_targets_[pos * 3 + 2] = 0.0f;
        }

        // Default metadata for batch add (no per-sample meta)
        SampleMeta new_meta{current_iteration_, 1, static_cast<uint8_t>(Termination::UNKNOWN), 0, 0};
        update_composition_counters(pos, new_meta);

        // Update statistics (no CAS!)
        total_added_.value.fetch_add(1, std::memory_order_release);
    }

    total_games_.value.fetch_add(1, std::memory_order_relaxed);
}

void ReplayBuffer::add_batch_raw(
    size_t batch_size,
    const float* observations_ptr,
    const float* policies_ptr,
    const float* values_ptr,
    const float* wdl_ptr,
    const float* soft_values_ptr,
    const SampleMeta* meta_ptr
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

        if (wdl_ptr != nullptr) {
            std::memcpy(wdl_targets_.data() + pos * 3, wdl_ptr + i * 3, 3 * sizeof(float));
        } else {
            wdl_targets_[pos * 3 + 0] = 0.0f;
            wdl_targets_[pos * 3 + 1] = 0.0f;
            wdl_targets_[pos * 3 + 2] = 0.0f;
        }

        soft_values_[pos] = (soft_values_ptr != nullptr) ? soft_values_ptr[i] : 0.0f;

        // Update metadata
        SampleMeta new_meta;
        if (meta_ptr != nullptr) {
            new_meta = meta_ptr[i];
        } else {
            new_meta = {current_iteration_, 1, static_cast<uint8_t>(Termination::UNKNOWN), 0, 0};
        }
        update_composition_counters(pos, new_meta);

        // Update statistics (no CAS!)
        total_added_.value.fetch_add(1, std::memory_order_release);
    }

    total_games_.value.fetch_add(1, std::memory_order_relaxed);
}

bool ReplayBuffer::sample(
    size_t batch_size,
    std::vector<float>& observations,
    std::vector<float>& policies,
    std::vector<float>& values,
    std::vector<float>* wdl_targets,
    std::vector<float>* soft_values
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
    if (wdl_targets != nullptr) {
        wdl_targets->resize(batch_size * 3);
    }
    if (soft_values != nullptr) {
        soft_values->resize(batch_size);
    }

    // Random sampling with replacement (thread-local RNG)
    std::uniform_int_distribution<size_t> dist(0, current - 1);

    for (size_t i = 0; i < batch_size; ++i) {
        size_t idx = dist(rng_);

        // Direct memcpy - compiler auto-vectorizes this better than manual SIMD
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

        if (wdl_targets != nullptr) {
            std::memcpy(
                wdl_targets->data() + i * 3,
                wdl_targets_.data() + idx * 3,
                3 * sizeof(float)
            );
        }
        if (soft_values != nullptr) {
            (*soft_values)[i] = soft_values_[idx];
        }
    }

    return true;
}

void ReplayBuffer::clear() {
    // Lock-free clear (atomic stores)
    write_pos_.value.store(0, std::memory_order_relaxed);
    // Note: We reset total_added_ because size() depends on it
    total_added_.value.store(0, std::memory_order_release);
    // Note: We don't reset total_games_ - it's a cumulative stat

    // Reset composition counters
    wins_count_.value.store(0, std::memory_order_relaxed);
    draws_count_.value.store(0, std::memory_order_relaxed);
    losses_count_.value.store(0, std::memory_order_relaxed);

    // Reset metadata to sentinel
    SampleMeta sentinel{};
    sentinel.game_result = 0xFF;
    std::fill(metadata_.begin(), metadata_.end(), sentinel);
}

ReplayBuffer::Composition ReplayBuffer::get_composition() const {
    return {
        wins_count_.value.load(std::memory_order_relaxed),
        draws_count_.value.load(std::memory_order_relaxed),
        losses_count_.value.load(std::memory_order_relaxed)
    };
}

ReplayBuffer::Stats ReplayBuffer::get_stats() const {
    // Lock-free stats (load atomics)
    size_t buf_size = size();  // Computed from total_added
    return Stats{
        buf_size,
        capacity_,
        total_added_.value.load(std::memory_order_relaxed),
        total_games_.value.load(std::memory_order_relaxed),
        static_cast<float>(buf_size) / capacity_,
        wins_count_.value.load(std::memory_order_relaxed),
        draws_count_.value.load(std::memory_order_relaxed),
        losses_count_.value.load(std::memory_order_relaxed)
    };
}

// ============================================================================
// Binary Save/Load (.rpbf format)
// ============================================================================
//
// Header (72 bytes):
//   magic:        char[4] = "RPBF"
//   version:      uint32_t = 2
//   capacity:     uint64_t   (original buffer capacity)
//   num_samples:  uint64_t   (= size(), i.e. min(total_added, capacity))
//   write_pos:    uint64_t
//   total_added:  uint64_t
//   total_games:  uint64_t
//   obs_size:     uint32_t
//   policy_size:  uint32_t
//   reserved:     uint8_t[8] (future use)
//
// Data (contiguous):
//   observations: num_samples * OBS_SIZE * 4 bytes
//   policies:     num_samples * POLICY_SIZE * 4 bytes
//   values:       num_samples * 4 bytes
//   wdl_targets:  num_samples * 3 * 4 bytes
//   soft_values:  num_samples * 4 bytes
//   metadata:     num_samples * 8 bytes

#pragma pack(push, 1)
struct RpbfHeader {
    char magic[4];          // "RPBF"
    uint32_t version;       // 2
    uint64_t capacity;
    uint64_t num_samples;
    uint64_t write_pos;
    uint64_t total_added;
    uint64_t total_games;
    uint32_t obs_size;
    uint32_t policy_size;
    uint8_t reserved[8];
};
#pragma pack(pop)
static_assert(sizeof(RpbfHeader) == 64, "RpbfHeader must be 64 bytes");

bool ReplayBuffer::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "[ReplayBuffer] Failed to open file for writing: " << path << std::endl;
        return false;
    }

    uint64_t num_samples = size();

    // Write header
    RpbfHeader header{};
    header.magic[0] = 'R'; header.magic[1] = 'P'; header.magic[2] = 'B'; header.magic[3] = 'F';
    header.version = 2;
    header.capacity = static_cast<uint64_t>(capacity_);
    header.num_samples = num_samples;
    header.write_pos = write_pos_.value.load(std::memory_order_relaxed);
    header.total_added = total_added_.value.load(std::memory_order_relaxed);
    header.total_games = total_games_.value.load(std::memory_order_relaxed);
    header.obs_size = static_cast<uint32_t>(OBS_SIZE);
    header.policy_size = static_cast<uint32_t>(POLICY_SIZE);
    std::memset(header.reserved, 0, sizeof(header.reserved));

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    if (num_samples == 0) {
        out.close();
        return out.good();
    }

    // Write data arrays (first num_samples slots)
    out.write(reinterpret_cast<const char*>(observations_.data()),
              num_samples * OBS_SIZE * sizeof(float));
    out.write(reinterpret_cast<const char*>(policies_.data()),
              num_samples * POLICY_SIZE * sizeof(float));
    out.write(reinterpret_cast<const char*>(values_.data()),
              num_samples * sizeof(float));
    out.write(reinterpret_cast<const char*>(wdl_targets_.data()),
              num_samples * 3 * sizeof(float));
    out.write(reinterpret_cast<const char*>(soft_values_.data()),
              num_samples * sizeof(float));
    out.write(reinterpret_cast<const char*>(metadata_.data()),
              num_samples * sizeof(SampleMeta));

    out.close();

    if (!out.good()) {
        std::cerr << "[ReplayBuffer] Write error during save to: " << path << std::endl;
        return false;
    }

    std::cerr << "[ReplayBuffer] Saved " << num_samples << " samples to " << path << std::endl;
    return true;
}

bool ReplayBuffer::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[ReplayBuffer] Failed to open file for reading: " << path << std::endl;
        return false;
    }

    // Read header
    RpbfHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in.good()) {
        std::cerr << "[ReplayBuffer] Failed to read header from: " << path << std::endl;
        return false;
    }

    // Validate magic
    if (header.magic[0] != 'R' || header.magic[1] != 'P' ||
        header.magic[2] != 'B' || header.magic[3] != 'F') {
        std::cerr << "[ReplayBuffer] Invalid magic in: " << path << std::endl;
        return false;
    }

    // Validate version
    if (header.version != 2) {
        std::cerr << "[ReplayBuffer] Unsupported version " << header.version
                  << " in: " << path << std::endl;
        return false;
    }

    // Validate sizes
    if (header.obs_size != OBS_SIZE || header.policy_size != POLICY_SIZE) {
        std::cerr << "[ReplayBuffer] Size mismatch: obs=" << header.obs_size
                  << " (expected " << OBS_SIZE << "), policy=" << header.policy_size
                  << " (expected " << POLICY_SIZE << ")" << std::endl;
        return false;
    }

    // Determine how many samples to load (truncate if file has more than capacity)
    uint64_t num_samples = header.num_samples;
    if (num_samples > capacity_) {
        std::cerr << "[ReplayBuffer] Warning: file has " << num_samples
                  << " samples but buffer capacity is " << capacity_
                  << ". Truncating to " << capacity_ << std::endl;
        num_samples = capacity_;
    }

    if (num_samples == 0) {
        clear();
        in.close();
        return true;
    }

    // When truncating (num_samples < header.num_samples), we need to seek past
    // unread entries in each section since they're stored contiguously in the file.
    uint64_t file_samples = header.num_samples;

    // Read data arrays into slots 0..num_samples-1
    in.read(reinterpret_cast<char*>(observations_.data()),
            num_samples * OBS_SIZE * sizeof(float));
    if (num_samples < file_samples) {
        in.seekg((file_samples - num_samples) * OBS_SIZE * sizeof(float), std::ios::cur);
    }

    in.read(reinterpret_cast<char*>(policies_.data()),
            num_samples * POLICY_SIZE * sizeof(float));
    if (num_samples < file_samples) {
        in.seekg((file_samples - num_samples) * POLICY_SIZE * sizeof(float), std::ios::cur);
    }

    in.read(reinterpret_cast<char*>(values_.data()),
            num_samples * sizeof(float));
    if (num_samples < file_samples) {
        in.seekg((file_samples - num_samples) * sizeof(float), std::ios::cur);
    }

    in.read(reinterpret_cast<char*>(wdl_targets_.data()),
            num_samples * 3 * sizeof(float));
    if (num_samples < file_samples) {
        in.seekg((file_samples - num_samples) * 3 * sizeof(float), std::ios::cur);
    }

    in.read(reinterpret_cast<char*>(soft_values_.data()),
            num_samples * sizeof(float));
    if (num_samples < file_samples) {
        in.seekg((file_samples - num_samples) * sizeof(float), std::ios::cur);
    }

    in.read(reinterpret_cast<char*>(metadata_.data()),
            num_samples * sizeof(SampleMeta));

    if (!in.good()) {
        std::cerr << "[ReplayBuffer] Read error during load from: " << path << std::endl;
        // Reset to avoid inconsistent state
        clear();
        return false;
    }

    in.close();

    // If we truncated, we need to skip past the remaining data in the file
    // (already handled since we only read num_samples worth)

    // Initialize sentinel metadata for unused slots
    SampleMeta sentinel{};
    sentinel.game_result = 0xFF;
    for (size_t i = num_samples; i < capacity_; ++i) {
        metadata_[i] = sentinel;
    }

    // Restore atomics
    write_pos_.value.store(static_cast<size_t>(num_samples % capacity_), std::memory_order_relaxed);
    total_added_.value.store(num_samples, std::memory_order_release);
    total_games_.value.store(header.total_games, std::memory_order_relaxed);

    // Recompute composition counters from loaded metadata
    uint64_t wins = 0, draws = 0, losses = 0;
    for (size_t i = 0; i < num_samples; ++i) {
        uint8_t r = metadata_[i].game_result;
        if (r == 0) ++wins;
        else if (r == 1) ++draws;
        else if (r == 2) ++losses;
    }
    wins_count_.value.store(wins, std::memory_order_relaxed);
    draws_count_.value.store(draws, std::memory_order_relaxed);
    losses_count_.value.store(losses, std::memory_order_relaxed);

    std::cerr << "[ReplayBuffer] Loaded " << num_samples << " samples from " << path
              << " (W=" << wins << " D=" << draws << " L=" << losses << ")" << std::endl;
    return true;
}

} // namespace training
