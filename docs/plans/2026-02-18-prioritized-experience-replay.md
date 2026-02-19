# Prioritized Experience Replay (PER) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add loss-based Prioritized Experience Replay to the AlphaZero training pipeline so high-loss positions are sampled more frequently, with importance sampling correction to debias gradients.

**Architecture:** A C++ sum-tree (segment tree) is added to `ReplayBuffer` for O(log N) proportional sampling. New samples get max-priority (ensuring at least one training pass). After each training batch, per-sample losses are fed back as updated priorities. IS weights with beta annealing correct the gradient bias from non-uniform sampling. When `--per-alpha 0` (default), the system behaves identically to current uniform sampling with zero overhead.

**Tech Stack:** C++20, pybind11, PyTorch (mixed precision via GradScaler), CMake

---

## Background: How PER Works

Classic PER (Schaul et al., 2015) applied to AlphaZero:

1. Each sample `i` has a priority `p_i` (= its most recent training loss + epsilon)
2. Sampling probability: `P(i) = p_i^alpha / sum(p_j^alpha)` where alpha controls how much prioritization matters
3. Importance sampling weight: `w_i = (N * P(i))^(-beta)`, normalized so `max(w) = 1`
4. Loss becomes: `L = mean(w_i * loss_i)` instead of `mean(loss_i)`
5. Beta anneals from ~0.4 to 1.0 over training (fully correcting bias by end)

**Key invariant:** When `alpha=0`, all `P(i) = 1/N` (uniform), all `w_i = 1` -- identical to current behavior.

---

## Task 1: SumTree Data Structure (C++ Header + Implementation)

**Files:**
- Create: `alphazero-cpp/include/training/sum_tree.hpp`
- Create: `alphazero-cpp/src/training/sum_tree.cpp`

The sum-tree is a complete binary tree stored in a flat array. Leaves hold `priority^alpha` values. Internal nodes hold the sum of their children. This gives O(log N) proportional sampling and O(log N) priority updates.

```
Example (capacity=4):
         [10]           <-- tree_[1] = root = total sum
        /    \
      [6]    [4]         <-- tree_[2], tree_[3]
      / \    / \
    [3] [3] [4] [0]     <-- tree_[4..7] = leaves (capacity_ .. 2*capacity_-1)
```

Sampling: draw uniform `u ~ [0, total_sum)`, descend from root: go left if `u < left_child`, else `u -= left_child` and go right.

### Step 1: Write the sum_tree.hpp header

```cpp
// alphazero-cpp/include/training/sum_tree.hpp
#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

namespace training {

/// Segment tree (sum-tree) for O(log N) proportional sampling.
/// Leaves hold priority values; internal nodes hold prefix sums.
/// Capacity is rounded up to the next power of 2 internally.
class SumTree {
public:
    /// Construct with given leaf capacity (rounded up to next power of 2).
    explicit SumTree(size_t capacity);

    /// Update priority at leaf index. O(log N).
    /// @param leaf_idx Buffer position in [0, capacity)
    /// @param priority Non-negative priority value (already raised to alpha)
    void update(size_t leaf_idx, float priority);

    /// Sample one leaf index proportional to priorities. O(log N).
    /// @param uniform Random value in [0, total_sum())
    /// @return Leaf index in [0, capacity)
    size_t sample(float uniform) const;

    /// Get priority at a specific leaf. O(1).
    float get(size_t leaf_idx) const;

    /// Total sum of all leaf priorities. O(1).
    float total() const;

    /// Rebuild all internal node sums from leaves. O(N).
    /// Call after bulk leaf writes (e.g., from concurrent add_sample).
    void rebuild();

    /// Reset all nodes to 0.
    void clear();

    /// Maximum leaf priority currently stored. O(1) via tracking.
    float max_priority() const { return max_priority_; }

    /// Logical capacity (original, before rounding).
    size_t capacity() const { return logical_capacity_; }

private:
    size_t logical_capacity_;   // Original capacity requested
    size_t tree_capacity_;      // Rounded up to next power of 2
    // tree_[0] unused. tree_[1] = root. Leaves at [tree_capacity_, 2*tree_capacity_).
    std::vector<float> tree_;
    float max_priority_;

    /// Next power of 2 >= n.
    static size_t next_power_of_2(size_t n);

    /// Propagate sum change from leaf up to root.
    void propagate(size_t tree_idx);
};

} // namespace training
```

### Step 2: Write the sum_tree.cpp implementation

```cpp
// alphazero-cpp/src/training/sum_tree.cpp
#include "training/sum_tree.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>

namespace training {

size_t SumTree::next_power_of_2(size_t n) {
    if (n == 0) return 1;
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

SumTree::SumTree(size_t capacity)
    : logical_capacity_(capacity)
    , tree_capacity_(next_power_of_2(capacity))
    , tree_(2 * next_power_of_2(capacity), 0.0f)
    , max_priority_(1.0f)   // Default priority for new samples
{
}

void SumTree::update(size_t leaf_idx, float priority) {
    assert(leaf_idx < logical_capacity_);
    size_t tree_idx = tree_capacity_ + leaf_idx;
    tree_[tree_idx] = priority;
    if (priority > max_priority_) {
        max_priority_ = priority;
    }
    propagate(tree_idx);
}

void SumTree::propagate(size_t tree_idx) {
    while (tree_idx > 1) {
        tree_idx >>= 1;  // parent
        tree_[tree_idx] = tree_[tree_idx * 2] + tree_[tree_idx * 2 + 1];
    }
}

size_t SumTree::sample(float uniform) const {
    size_t idx = 1;  // root
    while (idx < tree_capacity_) {
        size_t left = idx * 2;
        if (uniform < tree_[left]) {
            idx = left;
        } else {
            uniform -= tree_[left];
            idx = left + 1;
        }
    }
    size_t leaf = idx - tree_capacity_;
    // Clamp to logical capacity (power-of-2 padding leaves have priority 0)
    return std::min(leaf, logical_capacity_ - 1);
}

float SumTree::get(size_t leaf_idx) const {
    assert(leaf_idx < logical_capacity_);
    return tree_[tree_capacity_ + leaf_idx];
}

float SumTree::total() const {
    return tree_[1];
}

void SumTree::rebuild() {
    // Bottom-up rebuild of all internal nodes from leaves
    for (size_t i = tree_capacity_ - 1; i >= 1; --i) {
        tree_[i] = tree_[i * 2] + tree_[i * 2 + 1];
    }
    // Recompute max_priority from leaves
    max_priority_ = 1.0f;  // minimum default
    for (size_t i = 0; i < logical_capacity_; ++i) {
        max_priority_ = std::max(max_priority_, tree_[tree_capacity_ + i]);
    }
}

void SumTree::clear() {
    std::fill(tree_.begin(), tree_.end(), 0.0f);
    max_priority_ = 1.0f;
}

} // namespace training
```

### Step 3: Run build to verify compilation

Run: `cd "alphazero-cpp" && cmake -B build && cmake --build build --config Release 2>&1 | tail -5`
Expected: Build succeeds (sum_tree.cpp compiles cleanly)

### Step 4: Commit

```bash
git add alphazero-cpp/include/training/sum_tree.hpp alphazero-cpp/src/training/sum_tree.cpp
git commit -m "feat: add SumTree data structure for prioritized replay"
```

---

## Task 2: Add SumTree to CMakeLists.txt

**Files:**
- Modify: `alphazero-cpp/CMakeLists.txt:119-122`

### Step 1: Add sum_tree.cpp to training library

Change the training library definition from:

```cmake
add_library(training STATIC
    src/training/replay_buffer.cpp
    src/training/trainer.cpp
)
```

To:

```cmake
add_library(training STATIC
    src/training/replay_buffer.cpp
    src/training/sum_tree.cpp
    src/training/trainer.cpp
)
```

### Step 2: Run build to verify

Run: `cd "alphazero-cpp" && cmake -B build && cmake --build build --config Release 2>&1 | tail -5`
Expected: Build succeeds with sum_tree.cpp linked into training library

### Step 3: Commit

```bash
git add alphazero-cpp/CMakeLists.txt
git commit -m "build: add sum_tree.cpp to training library"
```

---

## Task 3: Add PER Fields and Methods to ReplayBuffer

**Files:**
- Modify: `alphazero-cpp/include/training/replay_buffer.hpp`
- Modify: `alphazero-cpp/src/training/replay_buffer.cpp`

This is the core C++ work. We add the sum-tree, priority tracking, prioritized sampling, and priority update methods.

### Step 1: Add PER includes and fields to replay_buffer.hpp

Add after the existing `#include` block at the top:

```cpp
#include "training/sum_tree.hpp"
#include <memory>
#include <mutex>
```

Add these new private members at the end of the `private:` section (before the closing `};`):

```cpp
    // ---- Prioritized Experience Replay (PER) ----
    // nullptr when PER disabled (alpha=0). Allocated by enable_per().
    std::unique_ptr<SumTree> sum_tree_;
    float per_alpha_{0.0f};        // Priority exponent (0 = uniform, 0.6 = recommended)
    bool tree_needs_rebuild_{false};  // Set by add_sample, cleared by sample_prioritized
    std::mutex per_mutex_;         // Protects sum_tree_ during sample + update
```

Add these new public methods after `Stats get_stats() const;`:

```cpp
    /**
     * Enable Prioritized Experience Replay.
     * Must be called before adding samples. alpha=0 disables PER.
     *
     * @param alpha Priority exponent. 0=uniform, 0.6=recommended.
     */
    void enable_per(float alpha);

    /**
     * Sample a prioritized batch with importance sampling weights.
     *
     * @param batch_size Number of samples
     * @param beta IS correction exponent (0..1, annealed toward 1.0)
     * @param observations Output observations (batch_size x 7872)
     * @param policies Output policies (batch_size x 4672)
     * @param values Output values (batch_size)
     * @param wdl_targets Output WDL targets (batch_size x 3), nullptr to skip
     * @param soft_values Output soft values (batch_size), nullptr to skip
     * @param indices Output: buffer positions for update_priorities
     * @param is_weights Output: importance sampling weights (normalized, max=1.0)
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
        std::vector<uint32_t>& indices,
        std::vector<float>& is_weights
    );

    /**
     * Update priorities for previously sampled indices.
     * Call after computing per-sample loss in the training loop.
     *
     * @param indices Buffer positions (from sample_prioritized)
     * @param new_priorities Raw loss values (will be raised to alpha internally)
     */
    void update_priorities(
        const std::vector<uint32_t>& indices,
        const std::vector<float>& new_priorities
    );

    /** Check if PER is enabled (alpha > 0). */
    bool per_enabled() const { return sum_tree_ != nullptr; }

    /** Get PER alpha value. */
    float per_alpha() const { return per_alpha_; }
```

### Step 2: Implement PER methods in replay_buffer.cpp

Add at the end of the file (before the closing `} // namespace training`):

```cpp
// ============================================================================
// Prioritized Experience Replay (PER)
// ============================================================================

void ReplayBuffer::enable_per(float alpha) {
    if (alpha <= 0.0f) {
        sum_tree_.reset();
        per_alpha_ = 0.0f;
        return;
    }
    per_alpha_ = alpha;
    sum_tree_ = std::make_unique<SumTree>(capacity_);

    // Initialize existing samples with default priority
    size_t current = size();
    float default_p = std::pow(1.0f, alpha);  // 1.0^alpha = 1.0
    for (size_t i = 0; i < current; ++i) {
        sum_tree_->update(i, default_p);
    }

    std::cerr << "[ReplayBuffer] PER enabled: alpha=" << alpha
              << ", capacity=" << capacity_ << std::endl;
}

bool ReplayBuffer::sample_prioritized(
    size_t batch_size,
    float beta,
    std::vector<float>& observations,
    std::vector<float>& policies,
    std::vector<float>& values,
    std::vector<float>* wdl_targets,
    std::vector<float>* soft_values,
    std::vector<uint32_t>& indices,
    std::vector<float>& is_weights
) {
    if (!sum_tree_) {
        return false;  // PER not enabled
    }

    size_t current = size();
    if (current < batch_size) {
        return false;
    }

    std::lock_guard<std::mutex> lock(per_mutex_);

    // Rebuild internal sums if self-play workers wrote leaves since last sample
    if (tree_needs_rebuild_) {
        sum_tree_->rebuild();
        tree_needs_rebuild_ = false;
    }

    float total = sum_tree_->total();
    if (total <= 0.0f) {
        // All priorities are 0 (shouldn't happen with epsilon). Fall back to uniform.
        return sample(batch_size, observations, policies, values, wdl_targets, soft_values);
    }

    // Resize output buffers
    observations.resize(batch_size * OBS_SIZE);
    policies.resize(batch_size * POLICY_SIZE);
    values.resize(batch_size);
    if (wdl_targets) wdl_targets->resize(batch_size * 3);
    if (soft_values) soft_values->resize(batch_size);
    indices.resize(batch_size);
    is_weights.resize(batch_size);

    // Stratified sampling: divide [0, total) into batch_size equal segments,
    // sample one point uniformly within each segment.
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    float segment = total / static_cast<float>(batch_size);

    // Compute IS weights: w_i = (N * P(i))^(-beta)
    // P(i) = priority_i / total
    // We normalize by max weight so all weights are in [0, 1].
    float max_weight = 0.0f;

    for (size_t i = 0; i < batch_size; ++i) {
        // Stratified sample point within segment i
        float low = segment * static_cast<float>(i);
        float u = low + uniform(rng_) * segment;
        u = std::min(u, total - 1e-6f);  // Clamp to avoid out-of-bounds

        size_t idx = sum_tree_->sample(u);
        // Clamp to valid range
        if (idx >= current) idx = current - 1;

        indices[i] = static_cast<uint32_t>(idx);

        // Compute IS weight
        float priority = sum_tree_->get(idx);
        float prob = priority / total;
        // Avoid division by zero
        prob = std::max(prob, 1e-8f);
        float weight = std::pow(static_cast<float>(current) * prob, -beta);
        is_weights[i] = weight;
        max_weight = std::max(max_weight, weight);

        // Copy sample data
        std::memcpy(observations.data() + i * OBS_SIZE,
                    observations_.data() + idx * OBS_SIZE,
                    OBS_SIZE * sizeof(float));
        std::memcpy(policies.data() + i * POLICY_SIZE,
                    policies_.data() + idx * POLICY_SIZE,
                    POLICY_SIZE * sizeof(float));
        values[i] = values_[idx];

        if (wdl_targets) {
            std::memcpy(wdl_targets->data() + i * 3,
                        wdl_targets_.data() + idx * 3,
                        3 * sizeof(float));
        }
        if (soft_values) {
            (*soft_values)[i] = soft_values_[idx];
        }
    }

    // Normalize IS weights by max weight so all are in [0, 1]
    if (max_weight > 0.0f) {
        float inv_max = 1.0f / max_weight;
        for (size_t i = 0; i < batch_size; ++i) {
            is_weights[i] *= inv_max;
        }
    }

    return true;
}

void ReplayBuffer::update_priorities(
    const std::vector<uint32_t>& indices,
    const std::vector<float>& new_priorities
) {
    if (!sum_tree_ || indices.size() != new_priorities.size()) {
        return;
    }

    std::lock_guard<std::mutex> lock(per_mutex_);

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        if (idx >= capacity_) continue;

        // Store priority^alpha in the tree
        float p = std::pow(std::max(new_priorities[i], 1e-6f), per_alpha_);
        sum_tree_->update(idx, p);
    }
}
```

### Step 3: Modify add_sample and add_batch_raw to set leaf priorities

In `add_sample()`, add this block right before the `total_added_.value.fetch_add(1, ...)` line (line ~112):

```cpp
    // PER: set default priority for new sample (leaf only, no propagation)
    if (sum_tree_) {
        float default_p = std::pow(sum_tree_->max_priority(), per_alpha_);
        // Direct leaf write (no propagation). Rebuild happens at next sample_prioritized().
        // Access tree_ directly to avoid propagation during concurrent writes.
        sum_tree_->update(pos, default_p);
        tree_needs_rebuild_ = true;
    }
```

Wait -- `update()` calls `propagate()`. For concurrent safety during self-play, we want leaf-only writes. We need to add a `set_leaf()` method to SumTree that writes the leaf without propagating. Let me adjust.

**Additional method to add to SumTree:**

In `sum_tree.hpp`, add to public:
```cpp
    /// Write a leaf value WITHOUT propagating to parents. O(1).
    /// Use when multiple concurrent writers need to set leaves;
    /// call rebuild() afterward to fix internal nodes.
    void set_leaf(size_t leaf_idx, float priority);
```

In `sum_tree.cpp`:
```cpp
void SumTree::set_leaf(size_t leaf_idx, float priority) {
    assert(leaf_idx < logical_capacity_);
    tree_[tree_capacity_ + leaf_idx] = priority;
    if (priority > max_priority_) {
        max_priority_ = priority;
    }
}
```

Then in `add_sample()` and `add_batch_raw()`, use `set_leaf()` instead of `update()`:

```cpp
    // PER: set default priority for new sample (leaf only, deferred rebuild)
    if (sum_tree_) {
        float default_p = std::pow(sum_tree_->max_priority(), per_alpha_);
        sum_tree_->set_leaf(pos, default_p);
        tree_needs_rebuild_ = true;
    }
```

Add the same block in `add_batch_raw()` at the equivalent location (before the `total_added_` fetch_add).

### Step 4: Modify clear() to also clear the sum-tree

In the existing `clear()` method, add after the metadata reset:

```cpp
    // Reset PER tree
    if (sum_tree_) {
        sum_tree_->clear();
        tree_needs_rebuild_ = false;
    }
```

### Step 5: Run build to verify compilation

Run: `cd "alphazero-cpp" && cmake -B build && cmake --build build --config Release 2>&1 | tail -10`
Expected: Build succeeds

### Step 6: Commit

```bash
git add alphazero-cpp/include/training/sum_tree.hpp \
        alphazero-cpp/src/training/sum_tree.cpp \
        alphazero-cpp/include/training/replay_buffer.hpp \
        alphazero-cpp/src/training/replay_buffer.cpp
git commit -m "feat: add PER support to ReplayBuffer (sum-tree, prioritized sampling, IS weights)"
```

---

## Task 4: RPBF v3 Persistence (Save/Load Priorities)

**Files:**
- Modify: `alphazero-cpp/src/training/replay_buffer.cpp:313-536`

### Step 1: Update RpbfHeader reserved field and version handling

Change the version comment and use `reserved[0]` as a flags byte:

In `save()`:
```cpp
    header.version = 3;
    // reserved[0] bit 0: has_priorities
    header.reserved[0] = sum_tree_ ? 0x01 : 0x00;
```

After the existing metadata write in `save()`, add:

```cpp
    // Write priorities if PER is enabled
    if (sum_tree_) {
        // Extract leaf priorities from sum-tree
        std::vector<float> priorities(num_samples);
        for (size_t i = 0; i < num_samples; ++i) {
            priorities[i] = sum_tree_->get(i);
        }
        out.write(reinterpret_cast<const char*>(priorities.data()),
                  num_samples * sizeof(float));
    }
```

### Step 2: Update load() to handle v2 and v3

Change the version validation:
```cpp
    if (header.version != 2 && header.version != 3) {
        std::cerr << "[ReplayBuffer] Unsupported version " << header.version
                  << " in: " << path << std::endl;
        return false;
    }
```

After the existing metadata read, add:

```cpp
    // Read priorities (v3 only)
    bool has_priorities = (header.version >= 3) && (header.reserved[0] & 0x01);
    if (has_priorities && sum_tree_) {
        std::vector<float> priorities(num_samples);
        in.read(reinterpret_cast<char*>(priorities.data()),
                num_samples * sizeof(float));
        for (size_t i = 0; i < num_samples; ++i) {
            sum_tree_->update(i, priorities[i]);
        }
        tree_needs_rebuild_ = false;
        std::cerr << "[ReplayBuffer] Loaded priorities for " << num_samples << " samples" << std::endl;
    } else if (has_priorities && !sum_tree_) {
        // File has priorities but PER not enabled — skip past them
        in.seekg(num_samples * sizeof(float), std::ios::cur);
    } else if (!has_priorities && sum_tree_) {
        // v2 file loaded into PER buffer — initialize with uniform priority
        float default_p = std::pow(1.0f, per_alpha_);
        for (size_t i = 0; i < num_samples; ++i) {
            sum_tree_->update(i, default_p);
        }
        std::cerr << "[ReplayBuffer] No priorities in file; initialized uniform" << std::endl;
    }
```

Also handle the truncation case for priorities (if `num_samples < file_samples` and has_priorities):
```cpp
    if (has_priorities && num_samples < file_samples) {
        in.seekg((file_samples - num_samples) * sizeof(float), std::ios::cur);
    }
```

### Step 3: Run build to verify

Run: `cd "alphazero-cpp" && cmake -B build && cmake --build build --config Release 2>&1 | tail -5`
Expected: Build succeeds

### Step 4: Commit

```bash
git add alphazero-cpp/src/training/replay_buffer.cpp
git commit -m "feat: RPBF v3 persistence for PER priorities (backward compat with v2)"
```

---

## Task 5: Python Bindings for PER

**Files:**
- Modify: `alphazero-cpp/src/bindings/python_bindings.cpp`

### Step 1: Add enable_per binding

After the existing `.def("save", ...)` binding on the ReplayBuffer class, add:

```cpp
        .def("enable_per", &training::ReplayBuffer::enable_per,
             py::arg("alpha"),
             "Enable Prioritized Experience Replay.\n"
             "alpha=0 disables (uniform). 0.6 recommended for PER.")
        .def("per_enabled", &training::ReplayBuffer::per_enabled,
             "Check if PER is enabled.")
        .def("per_alpha", &training::ReplayBuffer::per_alpha,
             "Get PER alpha exponent.")
```

### Step 2: Add sample_prioritized binding

```cpp
        .def("sample_prioritized", [](training::ReplayBuffer& self,
                                      size_t batch_size, float beta) {
            std::vector<float> observations, policies, values, wdl_targets, soft_values;
            std::vector<uint32_t> indices;
            std::vector<float> is_weights;

            bool success = self.sample_prioritized(
                batch_size, beta, observations, policies, values,
                &wdl_targets, &soft_values, indices, is_weights
            );
            if (!success) {
                throw std::runtime_error("PER sampling failed (not enabled or not enough samples)");
            }

            constexpr size_t OBS_FLAT = encoding::PositionEncoder::TOTAL_SIZE;  // 7872
            py::array_t<float> obs_array(std::vector<size_t>{batch_size, OBS_FLAT});
            py::array_t<float> pol_array(std::vector<size_t>{batch_size, 4672UL});
            py::array_t<float> val_array(std::vector<size_t>{batch_size});
            py::array_t<float> wdl_array(std::vector<size_t>{batch_size, 3UL});
            py::array_t<float> sv_array(std::vector<size_t>{batch_size});
            py::array_t<uint32_t> idx_array(std::vector<size_t>{batch_size});
            py::array_t<float> wt_array(std::vector<size_t>{batch_size});

            std::memcpy(obs_array.mutable_data(), observations.data(),
                       batch_size * OBS_FLAT * sizeof(float));
            std::memcpy(pol_array.mutable_data(), policies.data(),
                       batch_size * 4672 * sizeof(float));
            std::memcpy(val_array.mutable_data(), values.data(),
                       batch_size * sizeof(float));
            std::memcpy(wdl_array.mutable_data(), wdl_targets.data(),
                       batch_size * 3 * sizeof(float));
            std::memcpy(sv_array.mutable_data(), soft_values.data(),
                       batch_size * sizeof(float));
            std::memcpy(idx_array.mutable_data(), indices.data(),
                       batch_size * sizeof(uint32_t));
            std::memcpy(wt_array.mutable_data(), is_weights.data(),
                       batch_size * sizeof(float));

            return py::make_tuple(obs_array, pol_array, val_array,
                                  wdl_array, sv_array, idx_array, wt_array);
        },
        py::arg("batch_size"), py::arg("beta"),
        "Sample prioritized batch with IS weights.\n"
        "Returns: (obs, policies, values, wdl, soft_values, indices, is_weights)")
```

### Step 3: Add update_priorities binding

```cpp
        .def("update_priorities", [](training::ReplayBuffer& self,
                                     py::array_t<uint32_t> indices,
                                     py::array_t<float> priorities) {
            auto idx_buf = indices.request();
            auto pri_buf = priorities.request();

            std::vector<uint32_t> idx_vec(
                static_cast<uint32_t*>(idx_buf.ptr),
                static_cast<uint32_t*>(idx_buf.ptr) + idx_buf.size);
            std::vector<float> pri_vec(
                static_cast<float*>(pri_buf.ptr),
                static_cast<float*>(pri_buf.ptr) + pri_buf.size);

            self.update_priorities(idx_vec, pri_vec);
        },
        py::arg("indices"), py::arg("priorities"),
        "Update priorities for sampled indices after computing per-sample loss.")
```

### Step 4: Build and verify

Run: `cd "alphazero-cpp" && cmake -B build && cmake --build build --config Release 2>&1 | tail -5`
Expected: Build succeeds

### Step 5: Commit

```bash
git add alphazero-cpp/src/bindings/python_bindings.cpp
git commit -m "feat: Python bindings for PER (sample_prioritized, update_priorities, enable_per)"
```

---

## Task 6: PER Integration Test (Python)

**Files:**
- Create: `alphazero-cpp/tests/test_per.py`

### Step 1: Write the test

```python
#!/usr/bin/env python3
"""Tests for Prioritized Experience Replay."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "build" / "Release"))
import alphazero_cpp

OBS_SIZE = 7872
POLICY_SIZE = 4672


def test_per_basic():
    """PER samples high-priority items more often than low-priority ones."""
    buf = alphazero_cpp.ReplayBuffer(capacity=100)
    buf.enable_per(alpha=0.6)

    assert buf.per_enabled()
    assert buf.per_alpha() == 0.6

    # Add 100 samples with uniform data
    for i in range(100):
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        pol = np.zeros(POLICY_SIZE, dtype=np.float32)
        pol[0] = 1.0
        buf.add_sample(obs, pol, float(i) / 100.0)

    # First sample: all have default (max) priority, so roughly uniform
    obs, pol, val, wdl, sv, indices, weights = buf.sample_prioritized(50, beta=0.4)
    assert len(indices) == 50
    assert len(weights) == 50
    assert all(w > 0 and w <= 1.0 for w in weights)

    # Now give samples 0-9 very high priority, 10-99 very low
    high_indices = np.arange(10, dtype=np.uint32)
    high_priorities = np.full(10, 10.0, dtype=np.float32)
    buf.update_priorities(high_indices, high_priorities)

    low_indices = np.arange(10, 100, dtype=np.uint32)
    low_priorities = np.full(90, 0.01, dtype=np.float32)
    buf.update_priorities(low_indices, low_priorities)

    # Sample many times and check that indices 0-9 dominate
    high_count = 0
    total = 0
    for _ in range(20):
        _, _, _, _, _, idx, _ = buf.sample_prioritized(50, beta=1.0)
        for i in idx:
            if i < 10:
                high_count += 1
            total += 1

    ratio = high_count / total
    print(f"  High-priority ratio: {ratio:.2f} (expected >> 0.10)")
    assert ratio > 0.5, f"Expected high-priority items to dominate, got ratio={ratio:.2f}"


def test_per_disabled():
    """When alpha=0, PER is disabled — sample() works normally."""
    buf = alphazero_cpp.ReplayBuffer(capacity=100)
    assert not buf.per_enabled()

    for i in range(50):
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        pol = np.zeros(POLICY_SIZE, dtype=np.float32)
        pol[0] = 1.0
        buf.add_sample(obs, pol, 0.5)

    obs, pol, val, wdl, sv = buf.sample(25)
    assert len(val) == 25


def test_per_is_weights_beta():
    """IS weights should be more uniform when beta=1.0 vs beta=0."""
    buf = alphazero_cpp.ReplayBuffer(capacity=50)
    buf.enable_per(alpha=0.6)

    for i in range(50):
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        pol = np.zeros(POLICY_SIZE, dtype=np.float32)
        pol[0] = 1.0
        buf.add_sample(obs, pol, float(i) / 50.0)

    # Set varied priorities
    indices = np.arange(50, dtype=np.uint32)
    priorities = np.linspace(0.1, 10.0, 50).astype(np.float32)
    buf.update_priorities(indices, priorities)

    # With beta=0, IS correction is off -> weights should all be 1.0
    _, _, _, _, _, _, w0 = buf.sample_prioritized(25, beta=0.0)
    # With beta=0, (N*P)^0 = 1 for all, so all weights = 1.0
    assert all(abs(w - 1.0) < 1e-5 for w in w0), f"beta=0 weights not all 1.0: {w0}"

    # With beta=1.0, weights should vary
    _, _, _, _, _, _, w1 = buf.sample_prioritized(25, beta=1.0)
    weight_std = np.std(w1)
    print(f"  beta=1.0 weight std: {weight_std:.4f}")
    # With varied priorities and beta=1, weights should have some spread
    # (not guaranteed to be large, but non-zero)


def test_per_save_load():
    """Priorities should survive save/load cycle (RPBF v3)."""
    import tempfile
    import os

    buf = alphazero_cpp.ReplayBuffer(capacity=50)
    buf.enable_per(alpha=0.6)

    for i in range(30):
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        pol = np.zeros(POLICY_SIZE, dtype=np.float32)
        pol[0] = 1.0
        buf.add_sample(obs, pol, float(i) / 30.0)

    # Set custom priorities
    indices = np.arange(30, dtype=np.uint32)
    priorities = np.linspace(1.0, 10.0, 30).astype(np.float32)
    buf.update_priorities(indices, priorities)

    # Sample before save to verify it works
    _, _, _, _, _, idx_before, wt_before = buf.sample_prioritized(10, beta=0.5)

    # Save and load
    with tempfile.NamedTemporaryFile(suffix='.rpbf', delete=False) as f:
        tmp_path = f.name
    try:
        assert buf.save(tmp_path)

        buf2 = alphazero_cpp.ReplayBuffer(capacity=50)
        buf2.enable_per(alpha=0.6)
        assert buf2.load(tmp_path)

        assert buf2.size() == 30

        # Priorities should be restored — sampling should work
        _, _, _, _, _, idx_after, wt_after = buf2.sample_prioritized(10, beta=0.5)
        assert len(idx_after) == 10
        print("  Save/load with priorities: OK")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    print("Test 1: PER basic sampling bias")
    test_per_basic()
    print("  PASSED")

    print("\nTest 2: PER disabled (alpha=0)")
    test_per_disabled()
    print("  PASSED")

    print("\nTest 3: IS weights with beta")
    test_per_is_weights_beta()
    print("  PASSED")

    print("\nTest 4: PER save/load (RPBF v3)")
    test_per_save_load()
    print("  PASSED")

    print("\nAll PER tests passed!")
```

### Step 2: Run the test

Run: `uv run python alphazero-cpp/tests/test_per.py`
Expected: All 4 tests pass

### Step 3: Commit

```bash
git add alphazero-cpp/tests/test_per.py
git commit -m "test: add PER integration tests (sampling bias, IS weights, save/load)"
```

---

## Task 7: Training Loop Changes (Per-Sample Loss + IS Weighting)

**Files:**
- Modify: `alphazero-cpp/scripts/train.py:2031-2145` (train_iteration function)
- Modify: `alphazero-cpp/scripts/train.py:325-361` (IterationMetrics)

### Step 1: Add per_beta field to IterationMetrics

At line ~361, add before `buffer_size`:

```python
    per_beta: float = 0.0          # Current PER beta this iteration
```

### Step 2: Add per_beta parameter to train_iteration

Change the function signature (line ~2031) from:

```python
def train_iteration(
    network: nn.Module,
    optimizer: optim.Optimizer,
    replay_buffer,  # alphazero_cpp.ReplayBuffer
    batch_size: int,
    epochs: int,
    device: str,
    scaler: GradScaler,
) -> dict:
```

To:

```python
def train_iteration(
    network: nn.Module,
    optimizer: optim.Optimizer,
    replay_buffer,  # alphazero_cpp.ReplayBuffer
    batch_size: int,
    epochs: int,
    device: str,
    scaler: GradScaler,
    per_beta: float = 0.0,
) -> dict:
```

### Step 3: Modify the sampling and loss computation

Replace the sampling + loss section (lines ~2060-2097) with:

```python
    for epoch in range(epochs):
        # Sample batch — prioritized or uniform
        indices = None
        is_weights_tensor = None
        if replay_buffer.per_enabled() and per_beta > 0:
            obs, policies, values, wdl_targets, soft_values, indices, is_weights = \
                replay_buffer.sample_prioritized(batch_size, per_beta)
            is_weights_tensor = torch.from_numpy(is_weights).to(device)
        else:
            obs, policies, values, wdl_targets, soft_values = replay_buffer.sample(batch_size)

        # Convert to PyTorch tensors
        # obs is (batch, 7872) flat, need to reshape to (batch, 123, 8, 8)
        obs = obs.reshape(-1, 8, 8, INPUT_CHANNELS)  # (batch, 8, 8, 123) NHWC
        # permute is zero-copy metadata swap; .to(device) preserves channels_last
        obs_tensor = torch.from_numpy(obs).permute(0, 3, 1, 2).to(device)
        policy_target = torch.from_numpy(policies).to(device)
        value_target = torch.from_numpy(values).to(device)  # (batch,)
        mcts_wdl_target = torch.from_numpy(wdl_targets).to(device)  # (batch, 3)

        optimizer.zero_grad()

        with autocast('cuda', enabled=(device == "cuda")):
            # Forward pass (no mask during training - targets already masked)
            policy_pred, value_pred, policy_logits, wdl_logits = network(obs_tensor)

            # Per-sample policy loss: sum over actions, keep batch dim
            per_sample_policy = -torch.sum(
                torch.nan_to_num(policy_target * F.log_softmax(policy_logits, dim=1), nan=0.0),
                dim=1
            )  # (batch,)

            # Value loss: soft WDL cross-entropy with pure game outcome targets
            outcome_wdl = torch.zeros_like(mcts_wdl_target)
            outcome_wdl[:, 0] = (value_target > 0.5).float()
            outcome_wdl[:, 1] = ((value_target >= -0.5) & (value_target <= 0.5)).float()
            outcome_wdl[:, 2] = (value_target < -0.5).float()

            per_sample_value = -torch.sum(
                torch.nan_to_num(outcome_wdl * F.log_softmax(wdl_logits, dim=1), nan=0.0),
                dim=1
            )  # (batch,)

            per_sample_total = per_sample_policy + per_sample_value

            # Apply IS weights if using PER
            if is_weights_tensor is not None:
                loss = torch.mean(is_weights_tensor * per_sample_total)
            else:
                loss = torch.mean(per_sample_total)

            # Unweighted averages for logging
            policy_loss = per_sample_policy.mean()
            value_loss = per_sample_value.mean()
```

After the optimizer step (around line ~2121), add priority update:

```python
        # Update PER priorities with per-sample loss
        if indices is not None:
            with torch.no_grad():
                new_priorities = per_sample_total.detach().float().cpu().numpy()
                new_priorities = new_priorities + 1e-6  # Epsilon to avoid zero priority
                replay_buffer.update_priorities(indices, new_priorities)
```

### Step 4: Verify no test regressions

Run: `uv run python alphazero-cpp/tests/test_per.py`
Expected: All pass (C++ changes are independent)

### Step 5: Commit

```bash
git add alphazero-cpp/scripts/train.py
git commit -m "feat: integrate PER into training loop (per-sample loss, IS weights, priority updates)"
```

---

## Task 8: CLI Arguments and Beta Annealing

**Files:**
- Modify: `alphazero-cpp/scripts/train.py:2238+` (argparse)
- Modify: `alphazero-cpp/scripts/train.py:2458+` (buffer init)
- Modify: `alphazero-cpp/scripts/train.py:3200+` (main loop, beta annealing)

### Step 1: Add CLI arguments

After the existing `--buffer-size` argument (~line 2247), add:

```python
    # Prioritized Experience Replay
    parser.add_argument("--per-alpha", type=float, default=0.0,
                        help="PER priority exponent. 0=uniform (default), 0.6=recommended")
    parser.add_argument("--per-beta", type=float, default=0.4,
                        help="PER IS correction initial beta (default: 0.4)")
    parser.add_argument("--per-beta-final", type=float, default=1.0,
                        help="PER IS correction final beta (default: 1.0)")
    parser.add_argument("--per-beta-warmup", type=int, default=0,
                        help="Iterations to anneal beta. 0=anneal over all iterations")
```

### Step 2: Enable PER on buffer

Right after the `replay_buffer = alphazero_cpp.ReplayBuffer(...)` line (~2458), add:

```python
    if args.per_alpha > 0:
        replay_buffer.enable_per(args.per_alpha)
        print(f"PER enabled: alpha={args.per_alpha}, "
              f"beta={args.per_beta}->{args.per_beta_final} "
              f"(warmup={args.per_beta_warmup or 'all'} iters)")
```

### Step 3: Add beta annealing before train_iteration call

Right before the `train_metrics = train_iteration(...)` call (~line 3202), add:

```python
                # Compute current PER beta (annealed)
                current_per_beta = 0.0
                if args.per_alpha > 0:
                    warmup = args.per_beta_warmup if args.per_beta_warmup > 0 else args.iterations
                    progress = min(1.0, iteration / max(warmup, 1))
                    current_per_beta = args.per_beta + progress * (args.per_beta_final - args.per_beta)
```

Then pass it to the call:

```python
                train_metrics = train_iteration(
                    network, optimizer, replay_buffer,
                    args.train_batch, args.epochs, device, scaler,
                    per_beta=current_per_beta,
                )
```

### Step 4: Log PER beta in metrics

Where `metrics.grad_norm_avg` is set (~line 3212), add:

```python
                if args.per_alpha > 0:
                    metrics.per_beta = current_per_beta
```

### Step 5: Add PER config to print block

Find the configuration print section (search for `"MCTS simulations"` or similar) and add:

```python
    if args.per_alpha > 0:
        print(f"PER:                 alpha={args.per_alpha}, "
              f"beta={args.per_beta}->{args.per_beta_final} "
              f"(warmup={args.per_beta_warmup or 'all'} iters)")
```

### Step 6: Add PER config to checkpoint save

Where the config dict is saved in checkpoints, add:

```python
        'per_alpha': args.per_alpha,
        'per_beta': args.per_beta,
        'per_beta_final': args.per_beta_final,
```

### Step 7: Commit

```bash
git add alphazero-cpp/scripts/train.py
git commit -m "feat: add PER CLI args (--per-alpha, --per-beta, --per-beta-warmup) with beta annealing"
```

---

## Task 9: Update LLM Operation Manual

**Files:**
- Modify: `alphazero-cpp/scripts/llm_operation_manual.md`

### Step 1: Add PER section

Add a new section documenting the PER feature:

```markdown
## Section 10k: Prioritized Experience Replay (PER)

**What it does:** Samples training positions proportional to their loss (how "wrong" the model was), rather than uniformly. High-loss positions get trained on more often.

**When to use:** After early training stabilizes (iteration 5+). Do NOT enable during the first few iterations when the model is learning basic piece movement.

**Parameters:**
- `--per-alpha 0.6` — Priority exponent. Higher = more aggressive prioritization. 0 = uniform (off).
- `--per-beta 0.4` — Initial IS correction. Anneals to `--per-beta-final` (default 1.0).
- `--per-beta-warmup 50` — Iterations to anneal beta. 0 = anneal over all iterations.

**Recommended first use:**
```bash
uv run python alphazero-cpp/scripts/train.py \
    --per-alpha 0.6 --per-beta 0.4 --per-beta-final 1.0 \
    --per-beta-warmup 50 [other args...]
```

**How it works internally:**
- New samples get max-priority (sampled at least once before priority update)
- After each training batch, per-sample loss updates priorities in C++ sum-tree
- IS weights debias the gradient: `loss = mean(w_i * loss_i)` where `w_i = (N * P(i))^(-beta)`
- Beta anneals from 0.4 to 1.0, fully correcting bias by the end of training

**RPBF v3:** Buffer save/load now includes priorities. v2 files load with uniform priority.
```

### Step 2: Commit

```bash
git add alphazero-cpp/scripts/llm_operation_manual.md
git commit -m "docs: add PER section to LLM operation manual"
```

---

## Memory Impact

| Buffer capacity | Sum-tree memory | Priority disk (RPBF) |
|----------------|----------------|---------------------|
| 100K           | ~1 MB          | 400 KB              |
| 500K           | ~4 MB          | 2 MB                |
| 1M             | ~8 MB          | 4 MB                |

Negligible vs observation storage (100K * 7872 * 4 = ~3 GB).

## Zero-Overhead Guarantee (alpha=0)

When `--per-alpha 0` (the default):
- `enable_per()` is never called, `sum_tree_` stays `nullptr`
- `add_sample()` skips the null-check branch (1 instruction)
- `sample()` called instead of `sample_prioritized()` — identical to current
- No IS weights, no `update_priorities()`, no extra memory
- RPBF saves as v3 but without priorities section (same disk size as v2)
