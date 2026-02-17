# The Mathematics and Engineering of AlphaZero Chess

## Preamble

This document is a unified reference covering **both the mathematics and the systems engineering** of the AlphaZero chess implementation in this repository. Every formula traces to a specific source file and line number. Every engineering decision is grounded in the mathematics that motivates it.

The system rests on three pillars — **self-play**, **training**, and **inference** — forming a closed learning loop:

```
                         THE ALPHAZERO LEARNING LOOP

  +-----> Self-Play (MCTS + Network) -----> Game Trajectories -----+
  |         N workers in parallel             (s, pi, wdl, z)      |
  |         cross-game GPU batching                                |
  |                                                                v
  |                                                         Replay Buffer
  |                                                         (circular, C++)
  |                                                                |
  +<----- Train Network on samples <----- Sample mini-batches <---+
            L = L_policy + L_value
            Adam + FP16 + GradScaler
```

Each iteration: workers play games using MCTS guided by the current network, store training samples, train the network to predict MCTS policies and game outcomes, and the improved network produces stronger searches, which produce better training data.

---

## 1. The Core Idea: Policy Iteration Through Self-Play

AlphaZero learns chess by repeatedly playing against itself and improving from the outcomes. This is a form of **generalized policy iteration** — an idea from reinforcement learning where you alternate between:

1. **Policy Evaluation**: Estimate how good each position is (the value function `v(s)`)
2. **Policy Improvement**: Choose better moves based on those estimates (the policy `pi(a|s)`)

Unlike classical RL, AlphaZero fuses both into a single neural network `f_theta(s) = (p, v)` that outputs a **policy** (move probabilities) and a **value** (position evaluation) simultaneously. Monte Carlo Tree Search (MCTS) acts as a **policy improvement operator** — it uses the network's raw predictions to produce a stronger, search-refined policy.

Each iteration:
1. Play games using MCTS guided by the current network
2. Store (position, MCTS policy, game outcome, root WDL) tuples
3. Train the network to predict the MCTS policy and game outcome
4. The improved network produces better MCTS searches, which produce better training data

This is a **fixed-point** process: the network converges when its raw predictions match what MCTS would produce — at that point, search adds no further improvement.

---

## 2. Position Encoding: 122-Channel Board Representation

**Source:** `position_encoder.hpp`, `position_encoder.cpp`

### 2.1 Channel Breakdown

A board state `s` is encoded as a tensor `x in R^{8 x 8 x 122}` in NHWC (channels-last) layout. Everything is from the **current player's perspective** — when Black moves, the board is vertically flipped.

| Channels | Count | Content |
|----------|-------|---------|
| 0-5 | 6 | Current player's pieces (P, N, B, R, Q, K) — binary planes |
| 6-11 | 6 | Opponent's pieces (P, N, B, R, Q, K) — binary planes |
| 12-13 | 2 | Repetition count (reserved, currently zeros) |
| 14 | 1 | Color to move (constant 1.0 plane) |
| 15 | 1 | Move count: `min(1.0, fullmoves / 100)` |
| 16 | 1 | Castling rights: `{both: 1.0, K-side: 0.67, Q-side: 0.33, none: 0.0}` |
| 17 | 1 | No-progress count (50-move rule): `min(1.0, halfmoves / 50)` |
| 18-121 | 104 | 8 historical positions x 13 channels (12 piece planes + 1 repetition marker) |

**Total:** 6 + 6 + 2 + 1 + 1 + 1 + 1 + (8 x 13) = **122 channels**

**Source:** `position_encoder.hpp:29` — `static constexpr int CHANNELS = 122;`

### 2.2 Memory Layout (NHWC)

The encoding uses **channels-last** layout: `(8, 8, 122)` = 7808 floats per position.

```
Memory: [r0c0_ch0, r0c0_ch1, ..., r0c0_ch121, r0c1_ch0, ..., r7c7_ch121]
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              one pixel = 122 contiguous floats
```

This NHWC format is used in the C++ encoding and staging buffers. A fused NHWC-to-NCHW transpose happens during batch collection before GPU inference (see [Section 6.3](#63-batch-collection-collect_batch)).

**Source:** `position_encoder.hpp:26-30` — `TOTAL_SIZE = HEIGHT * WIDTH * CHANNELS = 7808`

### 2.3 Perspective Flipping

When Black is to move, the board is rank-mirrored so Black's pieces appear on the "home" ranks:

```
flip(rank, file) = (7 - rank, file)
flip(square)     = (7 - sq/8) * 8 + sq % 8
```

This ensures the network always sees positions "from below," regardless of which side is moving.

**Source:** `position_encoder.hpp:69-74` — `flip_square()`

### 2.4 Zero-Copy and Batch Encoding

Two encoding paths exist:
- **`encode(board)`** — allocates and returns a new `vector<float>`, convenient but allocates
- **`encode_to_buffer(board, buffer)`** — writes directly into a pre-allocated buffer (hot path)
- **`encode_batch(fens, buffer, use_parallel)`** — OpenMP-parallelized batch encoding from FEN strings

The hot path during self-play uses `encode_to_buffer` to avoid allocation overhead.

**Source:** `position_encoder.hpp:39-54`

---

## 3. Neural Network Architecture

**Source:** `network.py`

### 3.1 Architecture Diagram

```
Input x: (batch, 122, 8, 8)           [after NHWC->NCHW transpose]
    |
    v
[ConvBlock] Conv2d(122 -> F, 3x3, pad=1, no bias) + BN(F) + ReLU
    |
    v
[ResidualTower] B x ResidualBlock(F)
    |
    +---------------------------+
    |                           |
    v                           v
[PolicyHead]               [ValueHead (WDL)]
Conv(F->2, 1x1) + BN + ReLU    Conv(F->1, 1x1) + BN + ReLU
Flatten(128)                    Flatten(64)
Linear(128 -> 4672)             Linear(64 -> 256) + ReLU
    |                           Linear(256 -> 3)
    v                           |
policy_logits (4672)            v
    |                       wdl_logits (3)
    v                           |
masked_fill(-1e4) + softmax     v
    |                       softmax -> [P(win), P(draw), P(loss)]
    v                       value = P(win) - P(loss)
  policy (4672)                 |
    |                           v
    +---------+---------+-------+
              |
              v
    4-tuple: (policy, value, policy_logits, wdl_logits)
```

**Default configuration:** F=192 filters, B=15 residual blocks, ~27M parameters.

**Source:** `network.py:130-134` — `AlphaZeroNet.__init__`

### 3.2 Building Blocks

**ConvBlock** — the initial feature extractor:
```
ConvBlock(x) = ReLU(BN(Conv2d(x)))       where Conv2d is 3x3, pad=1, no bias
```

**Source:** `network.py:33-43`

**ResidualBlock** — the core building block of the residual tower:
```
ResBlock(x) = ReLU( BN(Conv(ReLU(BN(Conv(x))))) + x )
                                                   ^-- skip connection
```

The skip connection lets gradients flow directly through the network, enabling training of very deep architectures (He et al., 2015). Each block learns a *residual correction* to its input rather than a complete transformation.

**Source:** `network.py:46-61`

### 3.3 Policy Head

```
PolicyHead(x) = Linear(128 -> 4672)( Flatten( ReLU(BN(Conv(F -> 2, 1x1)))(x) ) )
```

Raw output: logits `z_p in R^4672`. Illegal move masking uses `-1e4` (not `-inf`) for FP16 safety:

```python
policy_logits = policy_logits.masked_fill(mask == 0, -1e4)
policy = softmax(policy_logits, dim=1)
```

**Source:** `network.py:64-77` (head definition), `network.py:169-172` (masking + softmax)

### 3.4 Value Head (WDL)

```
ValueHead(x) = Linear(256 -> 3)( ReLU( Linear(64 -> 256)( Flatten( ReLU(BN(Conv(F -> 1, 1x1)))(x) ) ) ) )
```

Outputs 3 logits for Win/Draw/Loss. Converted to probabilities and a scalar:

```
[P(win), P(draw), P(loss)] = softmax(z_w)
value = P(win) - P(loss)    in [-1, 1]
```

The WDL head provides richer gradient information than a scalar value head — a position evaluated as "60% win, 30% draw, 10% loss" encodes more information than just "value = 0.5."

**Source:** `network.py:80-107` (head definition), `network.py:177-183` (WDL conversion)

### 3.5 The 4-Tuple Return

The network always returns `(policy, value, policy_logits, wdl_logits)`:
- **policy** — softmax probabilities over actions (used during inference)
- **value** — scalar in [-1, 1] (used during MCTS backpropagation)
- **policy_logits** — raw logits before softmax (used in `log_softmax` during training loss)
- **wdl_logits** — raw WDL logits (used for WDL soft cross-entropy loss)

**Source:** `network.py:185`

---

## 4. Action Space: 4672 Moves

**Source:** `move_encoder.cpp`

### 4.1 Three Encoding Ranges

Every possible chess move maps to one of 4672 indices across three contiguous ranges:

| Range | Count | Type | Encoding Formula |
|-------|-------|------|-----------------|
| 0-3583 | 3584 | Queen-type moves | `from * 56 + dir * 7 + (dist - 1)` |
| 3584-4095 | 512 | Knight moves | `3584 + from * 8 + knight_idx` |
| 4096-4671 | 576 | Underpromotions (N, B, R) | `4096 + from * 9 + dir * 3 + piece_idx` |
| **Total** | **4672** | | |

**Verification:** 64 x 56 = 3584, 64 x 8 = 512, 64 x 9 = 576; 3584 + 512 + 576 = **4672**

### 4.2 Queen-Type Moves (0-3583)

"Queen-type" encompasses all sliding moves — queen, rook, bishop, king, and pawn pushes (including queen promotions). These share direction/distance encoding.

8 directions x 7 distances x 64 from-squares = 3584 indices.

```
Directions:  N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
Distances:   1 through 7
Index:       from * 56 + direction * 7 + (distance - 1)
```

**Source:** `move_encoder.cpp:94-113` — `encode_queen_move()`

### 4.3 Knight Moves (3584-4095)

8 L-shaped offsets x 64 from-squares = 512 indices.

```
Knight offsets (rank_diff, file_diff):
  0: (+2,+1)  1: (+1,+2)  2: (-1,+2)  3: (-2,+1)
  4: (-2,-1)  5: (-1,-2)  6: (+1,-2)  7: (+2,-1)

Index: 3584 + from * 8 + knight_idx
```

**Source:** `move_encoder.cpp:115-148` — `encode_knight_move()`

### 4.4 Underpromotions (4096-4671)

3 piece types x 3 directions x 64 from-squares = 576 indices.

```
Piece types:  Knight=0, Bishop=1, Rook=2   (queen promotion is in queen-type range)
Directions:   left=-1 -> 0, straight=0 -> 1, right=+1 -> 2
Index:        4096 + from * 9 + direction * 3 + piece_idx
```

**Source:** `move_encoder.cpp:151-181` — `encode_underpromotion()`

### 4.5 Black Perspective Rotation

For Black, both `from` and `to` squares undergo 180-degree rotation before encoding:

```
sq_rotated = 63 - sq
```

This ensures the policy vector has identical semantics regardless of which side is moving — the network always sees the board "from below."

**Source:** `move_encoder.cpp:99-103`, `move_encoder.cpp:121-124`

---

## 5. Monte Carlo Tree Search

**Source:** `search.hpp`, `search.cpp`, `node.hpp`

MCTS converts the network's imperfect predictions into a much stronger policy through lookahead search. Each "simulation" walks from the root to a leaf, expands it, evaluates it with the network, and backpropagates the result.

### 5.1 The Four Phases

```
  Root ──select──> ... ──select──> Leaf
                                    |
                                 expand()
                                    |
                               [NN evaluate]
                                    |
                               backpropagate
                               (leaf -> root)
```

**Phase 1 — Selection:** Starting at the root, repeatedly pick the child with the highest PUCT score until reaching an unexpanded node (leaf). Virtual loss is applied to each node on the path.

**Phase 2 — Expansion:** Generate all legal moves at the leaf. Create child nodes, setting each child's prior from the network's policy output.

**Phase 3 — Evaluation:** Feed the leaf position through the neural network to get `(policy, value, wdl)`. In the parallel system, this happens via the shared evaluation queue (see [Section 6](#6-parallel-self-play-system)).

**Phase 4 — Backpropagation:** Walk from the leaf back to the root, updating visit counts and value sums. Negate the value at each level (my win = your loss).

In the batched variant, multiple selections happen first (collecting leaves), then all leaves are evaluated as a single GPU batch, then all results are backpropagated.

**Source:** `search.cpp:120-163` — `run_selection_phase()`

### 5.2 PUCT Selection Formula

At each internal node, select the child `a` maximizing:

```
PUCT(s, a) = Q(s, a)  +  c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s,a) + VL(s,a))
             ^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
             exploitation                    exploration
```

Where:
- `Q(s, a)` — risk-adjusted mean value of child (see [Section 5.3](#53-q-value-with-dynamic-fpu))
- `P(s, a)` — prior probability from the neural network (stored as `uint16` fixed-point)
- `N(s)` — parent visit count
- `N(s, a)` — child visit count
- `VL(s, a)` — virtual loss count (for parallel search)
- `c_puct = 1.5` — exploration constant (default)

**Intuition:** Unvisited children with high priors get explored first — `sqrt(N)/1` is large. As a child accumulates visits, the exploration bonus shrinks — `sqrt(N)/(1+N_child)` approaches 0, and the Q-value dominates. This balances breadth (try what the network recommends) with depth (follow lines that prove good).

**Source:** `search.cpp:316-332` — `puct_score()`

### 5.3 Q-Value with Dynamic FPU

The Q-value computation depends on the node's state. Three cases:

**Case 1 — Unvisited, no virtual loss (N=0, VL=0) — Dynamic FPU:**
```
Q(s, a) = Q_parent - fpu_base * (1 - P(s, a))
```

High-prior moves (P=0.9) get a small penalty (0.1 * fpu_base), while low-prior moves (P=0.03) get a harsh penalty (0.97 * fpu_base). This makes MCTS focus on the network's top recommendations before trying unlikely moves.

**Case 2 — Unvisited with virtual loss (N=0, VL>0):**
```
Q(s, a) = Q_parent
```

Falls back to the parent's Q-value (Leela Chess Zero convention).

**Case 3 — Visited (N>0):**
```
real_value = value_sum_fixed / 10000
Q(s, a) = (real_value + VL * Q_parent) / (N + VL)
```

The true mean value blended with virtual-loss pessimism. Virtual losses temporarily lower Q, discouraging other threads from exploring the same path.

**Source:** `node.hpp:90-109` — `q_value()`

### 5.4 Risk-Adjusted Q-Value (Entropic Risk Measure)

When `risk_beta != 0`, the system uses a variance-adjusted Q-value based on the Entropic Risk Measure.

**The Universal Equation:**
```
V_beta(s) = (1/beta) * ln E[exp(beta * v)]
```

**Taylor expansion** around beta=0:
```
V_beta ~= E[v] + (beta/2) * Var(v) + O(beta^2)
```

**Implementation (visited nodes, N>0):**
```
mean    = value_sum_fixed / (10000 * N)
mean_sq = value_sum_sq_fixed / (10000 * N)
var     = max(0, mean_sq - mean^2)          // guard fixed-point residuals

Q_risk  = mean + (beta / 2) * var
```

With virtual loss blending:
```
Q_risk  = (N * Q_risk + VL * Q_parent) / (N + VL)
Q_risk  = clamp(Q_risk, -1, 1)
```

**The spectrum of agents:**

| beta | Style | Behavior |
|------|-------|----------|
| beta << 0 | "Iron Logic" | Avoids complexity, seeks simplified positions |
| beta = 0 | Standard AlphaZero | Pure expected value maximization |
| beta >> 0 | "The Swindler" | Maximizes chaos, seeks high-variance positions |

**Design decisions:**
- **beta=0 fast path:** When `beta == 0.0f`, `q_value_risk()` delegates directly to `q_value()` — zero overhead, identical behavior to standard AlphaZero.
- **FPU stays risk-neutral:** Unvisited nodes (N=0) have no variance data, so they use the standard FPU formula regardless of beta.
- **Terminal draws are zero:** A terminal draw has value=0 and variance=0, so `Q_beta = 0` regardless of beta.
- **Training labels are pure:** beta is search-time only. The network always trains on `{+1, 0, -1}` outcomes. One network can be deployed at different beta values without retraining.

**Source:** `node.hpp:116-153` — `q_value_risk()`

### 5.5 Node Storage (64-Byte Cache-Line Design)

Each MCTS node is exactly **64 bytes** — one cache line — for optimal CPU performance. The `alignas(64)` annotation prevents false sharing between cores.

```
Byte Layout (53 data bytes + 11 padding = 64):

  Offset   Size   Type                  Field
  ------   ----   ----                  -----
  [0-7]      8    Node*                 parent
  [8-15]     8    Node*                 first_child
  [16-23]    8    Node*                 next_sibling
  [24-31]    8    atomic<int64_t>       value_sum_fixed      (Sum v * 10000)
  [32-39]    8    atomic<int64_t>       value_sum_sq_fixed   (Sum v^2 * 10000)
  [40-43]    4    atomic<uint32_t>      visit_count          N(s,a)
  [44-45]    2    atomic<int16_t>       virtual_loss         VL(s,a)
  [46-49]    4    chess::Move           move                 (uint16 move_ + int16 score_)
  [50-51]    2    uint16_t              prior_fixed          P(s,a) * 10000
  [52]       1    uint8_t               flags                bit 0: terminal, bit 1: expanded
  [53-63]   11    uint8_t[11]           padding
  ------   ----
  Total:    64 bytes
```

**Fixed-point arithmetic:** Values are stored as `int64_t(round(value * 10000))`. This avoids floating-point atomics while maintaining 0.0001 precision. The `int64_t` range prevents overflow up to ~2^31 visits.

**`chess::Move` is 4 bytes**, not 2 — it contains both `uint16_t move_` and `int16_t score_`. This affects padding math.

**Source:** `node.hpp:31-64` (struct definition), `node.hpp:82` + `node.hpp:224-225` (static_asserts)

### 5.6 Dirichlet Noise at Root

To ensure exploration in self-play, the root node's priors are perturbed:

```
P'(s, a) = (1 - epsilon) * P(s, a) + epsilon * eta_a

where eta ~ Dir(alpha, alpha, ..., alpha)
```

- `alpha = 0.3` — chess-specific from the AlphaZero paper; lower = spikier noise
- `epsilon = 0.25` — 25% noise, 75% network prior

The Dirichlet distribution is generated via K independent `Gamma(alpha, 1)` variates, then normalized to sum to 1. This injects just enough randomness that the search occasionally tries moves the network would never recommend, preventing the learning loop from getting stuck in local optima.

**Source:** `search.cpp:344-377` — `add_dirichlet_noise()`

### 5.7 WDL-to-Value Conversion

The network outputs WDL probabilities, which are converted to a scalar for MCTS backpropagation:

```
value = P(win) - P(loss)
```

This is simple and unbiased. Risk sensitivity (preferring decisive or safe positions) is handled at the node level through `q_value_risk()`, not at the leaf conversion level.

**Critical:** Risk adjustment is **search-time only**. Terminal draws in search always produce `value = 0.0f`, `variance = 0` — so `Q_beta = 0` regardless of beta.

**Source:** `search.hpp:43-44` — `wdl_to_value()`

### 5.8 Backpropagation

After leaf evaluation, walk from leaf to root:

```python
node = leaf
while node is not None:
    remove_virtual_loss(node)

    node.value_sum_fixed     += round(value * 10000)   # fixed-point
    node.value_sum_sq_fixed  += round(value^2 * 10000) # for variance
    node.visit_count         += 1

    if node is root:
        root_wdl_sum += [pw, pd, pl]   # accumulate WDL at root only
        root_wdl_count += 1
        # use memory_order_release for visibility guarantee

    node = node.parent
    value = -value           # flip for opponent
    swap(pw, pl)             # W<->L flip, D unchanged
```

The value negation at each level captures the zero-sum nature of chess: if a position is +0.8 for me, it is -0.8 for my opponent.

Root node uses `memory_order_release` on visit_count to ensure all value updates are visible to other threads.

**Source:** `search.cpp:276-314` — both `backpropagate()` overloads

### 5.9 Virtual Loss

When multiple CPU workers search simultaneously, virtual loss prevents them from all exploring the same promising line:

```
Selection:        VL(node) += 1    for each node on the path   (fetch_add)
Backpropagation:  VL(node) -= 1    for each node on the path   (fetch_sub)
```

While a worker is in-flight, other workers see a temporarily deflated Q-value for that path, steering them toward alternative lines. The loss is "virtual" — it disappears when the real evaluation returns.

Implementation: `atomic<int16_t>` — lightweight at 2 bytes, using `memory_order_relaxed`.

**Source:** `node.hpp:192-199` — `add_virtual_loss()`, `remove_virtual_loss()`

### 5.10 LogSumExp Diagnostic

For analysis, the root's risk-adjusted value can be computed via the exact LogSumExp formula (rather than the Taylor approximation used per-node):

```
V_risk = max_q + (1/beta) * log( sum_a exp(beta * (Q_risk_a - max_q)) )
```

The shift by `max_q` is the standard "shift trick" for numerical stability. Only visited children contribute.

**Source:** `search.cpp:553-581` — `get_root_risk_value()`

---

## 6. Parallel Self-Play System

**Source:** `parallel_coordinator.hpp/cpp`, `evaluation_queue.hpp/cpp`

### 6.1 Architecture Overview

```
  Worker 0 ──+                                     +──> Results[0]
  Worker 1 ──+                                     +──> Results[1]
  Worker 2 ──+──> GlobalEvaluationQueue ──> GPU ───+──> Results[2]
     ...     |      (cross-game batching)  Thread   |      ...
  Worker N ──+                                     +──> Results[N]

  Each worker:          Shared queue:          GPU thread:
  - plays full games    - 3-phase pipeline     - collect_batch()
  - runs MCTS           - pre-allocated        - NN forward pass
  - submits leaves        staging buffers      - submit_results()
  - gets results back   - lock-free data       - WDL -> value
                          copy                   conversion
```

N worker threads play independent games. When any worker reaches an MCTS leaf needing neural network evaluation, it submits the position to the shared `GlobalEvaluationQueue`. The GPU thread collects leaves from **all workers across all games** into a single batch — this cross-game batching is what achieves high GPU utilization.

**Source:** `parallel_coordinator.hpp:199-369` — `ParallelSelfPlayCoordinator`

### 6.2 The Lock-Free Three-Phase Evaluation Queue

The queue is designed for minimal contention between workers and the GPU thread:

```
  Worker Thread                              GPU Thread
  ─────────────                              ──────────
  Phase 1 (lock ~1us):
    claim staging slots
    push metadata
  Phase 2 (NO lock):
    memcpy obs + mask
    into staging buffers
  Phase 3 (NO lock):
    set slot_ready_ flags ──────────────>    collect_batch():
    notify GPU thread                          Phase 1 (lock ~10us):
                                                 deferred compaction
                                                 stall-detect wait
                                                 take metadata
                                               Phase 2 (NO lock, OpenMP):
                                                 spin-wait slot_ready_
                                                 fused NHWC->NCHW transpose
                                                 mask copy
```

#### Phase 1 — Claim Slots (lock ~1us)

Under the queue mutex, the worker:
1. Atomically claims N contiguous slots via `staging_write_head_`
2. Pushes lightweight `StagedRequest` metadata (worker_id, request_id, generation, slot index)
3. No observation data is copied under the lock — only integer arithmetic and vector push_back

```cpp
staging_write_head_.store(first_slot + actual_leaves);   // claim slots
staged_requests_.emplace_back(worker_id, req_id, gen, first_slot + i);  // metadata only
```

**Source:** `evaluation_queue.cpp:160-199` — Phase 1 of `submit_for_evaluation()`

#### Phase 2 — Data Copy (no lock)

After releasing the lock, the worker `memcpy`s observation and mask data into its claimed staging slots:

```cpp
memcpy(staging_obs_buffer_  + slot * OBS_SIZE,  observations + i * OBS_SIZE,  ...)
memcpy(staging_mask_buffer_ + slot * POLICY_SIZE, legal_masks + i * POLICY_SIZE, ...)
```

This is safe because each worker owns its claimed slots exclusively — no other worker can claim the same slots (the write head was incremented atomically under lock), and the GPU thread won't read these slots until the ready flags are set.

**Source:** `evaluation_queue.cpp:210-218` — Phase 2

#### Phase 3 — Signal Ready (no lock)

After the memcpy completes, the worker sets per-slot ready flags with `release` semantics:

```cpp
slot_ready_[slot].store(1, std::memory_order_release);   // data visible to GPU
queue_cv_.notify_one();                                   // wake GPU thread
```

The `release` semantics ensure the preceding memcpy is visible to the GPU thread when it does an `acquire` load on `slot_ready_[slot]`.

The `slot_ready_` array uses `unique_ptr<atomic<uint8_t>[]>` rather than `vector<atomic>` because `std::atomic` has deleted copy/move constructors, which is incompatible with `std::vector` on MSVC.

**Source:** `evaluation_queue.cpp:225-231` — Phase 3; `evaluation_queue.hpp:255` — `slot_ready_` declaration

### 6.3 Batch Collection (`collect_batch`)

The GPU thread's batch collection has two phases:

**Phase 1 (under lock, ~10us): Metadata extraction**

1. **Deferred compaction:** If `needs_compaction_` flag was set (staging head > 75% of capacity), compact remaining requests to the front of staging buffers. Deferring to the start of the *next* `collect_batch` guarantees all previous-batch slot_ready flags have been consumed and cleared.

2. **Two-phase wait with stall detection:**
   - Wait for at least 1 request (full timeout)
   - Accumulate while workers are actively submitting (queue growing)
   - Once no new submissions arrive within 1ms, fire immediately

3. Extract up to `max_batch_size` request metadata entries (slot indices, worker IDs, generations)

**Phase 2 (no lock, OpenMP parallel): Data extraction + transpose**

For each batch entry (parallelized with `#pragma omp parallel for`):

1. **Spin-wait** on `slot_ready_[slot]` — typically instant because workers finish memcpy long before the GPU completes stall detection
2. **Fused NHWC-to-NCHW transpose** directly from staging to batch output:

```cpp
// NHWC: staging_obs[slot * OBS_SIZE + h * (8*122) + w * 122 + c]
// NCHW: batch_obs[b * OBS_SIZE + c * 64 + h * 8 + w]
for (h : 0..7)
  for (w : 0..7)
    for (c : 0..121)
      nchw[c * 64 + h * 8 + w] = nhwc[h * (8 * 122) + w * 122 + c];
```

This fused transpose eliminates the intermediate `batch_obs_buffer_` (NHWC), saving 32MB of memory.

3. **Mask copy:** simple memcpy (no transpose needed — masks are 1D policy vectors)
4. **Clear** `slot_ready_[slot]` for reuse

**Source:** `evaluation_queue.cpp:397-598` — `collect_batch()`

### 6.4 Result Distribution

After GPU evaluation, the GPU thread distributes results:

1. Write policies to the current double-buffer slot: `batch_policy_buffers_[current_policy_buffer_]`
2. For each batch entry, create a lightweight `EvalResult` (no policy copy — just a batch_index and buffer_id for zero-copy reference)
3. Group results by worker and deliver with one lock acquisition per worker (not per result)
4. Flip `current_policy_buffer_` (0 <-> 1) — GPU writes to one buffer while workers read from the other

**Generation-based staleness:** Each worker has a generation counter. When a worker calls `flush_worker_results()`, its generation increments, making all in-flight results stale. Stale results (generation mismatch) are silently discarded in `get_results()`. This prevents stale NN results from corrupting new search trees after a tree reset.

**Source:** `evaluation_queue.cpp:601-668` — `submit_results()`; `evaluation_queue.cpp:376-391` — `flush_worker_results()`

### 6.5 Memory Architecture

All buffers are 64-byte aligned for cache line and GPU compatibility:

```
Staging Buffers (per queue):
  staging_obs_buffer_:    queue_capacity * OBS_SIZE * 4 bytes     (~244 MB at 8192 cap)
  staging_mask_buffer_:   queue_capacity * POLICY_SIZE * 4 bytes  (~146 MB at 8192 cap)
  slot_ready_:            queue_capacity * 1 byte                 (atomic<uint8_t>[])

Batch Buffers (per queue):
  batch_obs_nchw_buffer_: max_batch_size * OBS_SIZE * 4 bytes     (NCHW for GPU)
  batch_mask_buffer_:     max_batch_size * POLICY_SIZE * 4 bytes
  batch_policy_buffers_:  2 * max_batch_size * POLICY_SIZE * 4    (double-buffered)
  batch_value_buffer_:    max_batch_size * 4 bytes

Platform-specific allocation:
  Windows: _aligned_malloc() / _aligned_free()
  Linux:   std::aligned_alloc() / std::free()
```

**Constants:**
- `MAX_WORKERS = 128` — workers beyond this limit silently fail all evaluations
- `DEFAULT_QUEUE_CAPACITY = 8192`
- `DEFAULT_MAX_BATCH = 512`
- `OBS_SIZE = 8 * 8 * 122 = 7808` floats
- `POLICY_SIZE = 4672` floats

**Source:** `evaluation_queue.hpp:21-25` — constants; `evaluation_queue.cpp:83-133` — buffer allocation

### 6.6 Game Loop and Move Selection

Each worker runs `play_single_game()`:

1. **NodePool reset between moves** (not just between games) — prevents OOM on long games with 800+ simulations per move
2. **Per-game asymmetric risk:** if configured, one side uses `risk_beta` while the opponent samples from `[opponent_risk_min, opponent_risk_max]`, randomly assigned per game to avoid color bias
3. **Root NN evaluation with retry:** up to `root_eval_retries` attempts with stale result flushing between retries

**Temperature-based move selection:**
```
If move_number < T_moves (default 30):
    pi(a) = N(s, a)^(1/tau) / sum_j N(s, a_j)^(1/tau)    with tau = 1.0
    Sample move ~ pi

Else:
    Select move = argmax_a N(s, a)                          (greedy)
```

**Policy target (training label):**
```
pi_target(a) = N(s, a) / sum_j N(s, j)
```

This is a **soft** target — a position where MCTS spent 60% of visits on e4 and 30% on d4 produces `{e4: 0.6, d4: 0.3, ...}`. The network learns the search distribution, not just the best move.

**Value target (assigned after game ends):**
```
z = +1.0   if current player won
z =  0.0   if draw               (pure label — risk_beta is NOT baked in)
z = -1.0   if current player lost
```

**Per-move stored data:**
- Observation (7808 floats)
- Policy distribution (4672 floats)
- Root WDL: `[avg P(win), avg P(draw), avg P(loss)]` across all simulations
- Soft value: LogSumExp risk-adjusted root value (0.0 if `risk_beta == 0`)
- Game outcome value (assigned retroactively)

**Source:** `parallel_coordinator.cpp:306-584` — `play_single_game()`; `game.hpp:58-76` — `set_outcomes()`

### 6.7 NodePool: Arena Allocator for MCTS Nodes

The NodePool is a chained arena allocator that provides O(1) allocation without fragmentation:

```
Configuration:
  NODES_PER_BLOCK = 16384   (16K nodes per block = 1MB)
  MAX_BLOCKS      = 512     (max 512 blocks = 512MB per worker)
```

Each block is 64-byte aligned to prevent false sharing. The pool resets by simply resetting `current_block_` and `current_offset_` to 0 — no individual node deallocation. Reset happens between moves during self-play.

**Source:** `node_pool.hpp:16-17` — constants; `node_pool.hpp:57-60` — `reset()`

---

## 7. GPU Inference Pipeline

**Source:** `train.py` (evaluator callback, CUDA graph capture)

### 7.1 Evaluator Callback

The neural network evaluator is a Python function passed to the C++ coordinator:

```python
def neural_evaluator(obs_array, mask_array, batch_size, out_policies, out_values):
    """Called from C++ GPU thread — must be thread-safe."""
```

- **Input:** NCHW `float32` numpy arrays (C++ handles the NHWC-to-NCHW transpose)
- **Output:** policies `(batch x 4672)` and WDL probabilities `(batch x 3)` — zero-copy into C++ buffers via `np.copyto()`
- **WDL-to-value conversion** happens in C++ after the callback returns: `value = pw - pl`
- Decorated with `@torch.no_grad()` — inference only

The callback routes each batch to the tightest-fitting CUDA graph tier (see below).

**Source:** `train.py:1492-1584` — `neural_evaluator()`

### 7.2 Multi-Tier CUDA Graph System

**Problem:** GPU kernel launch overhead dominates for small batches — the fixed per-launch cost (~0.1ms) is comparable to or exceeds the actual compute time for batches of 16-64 positions.

**Solution:** Pre-captured CUDA graphs at geometric sizes. A CUDA graph records a sequence of GPU operations once, then replays them with near-zero launch overhead.

```
  Batch arrives (size B)
       |
       v
  B <= mini_size  AND  B >= mini_threshold?  ──yes──>  MINI GRAPH (64)
       |no
       v
  B <= small_size AND  B >= small_threshold? ──yes──>  SMALL GRAPH (128)
       |no
       v
  B <= medium_size AND B >= medium_threshold? ─yes──>  MEDIUM GRAPH (eval_batch/2)
       |no
       v
  B > large_threshold?  ────────────────────── yes──>  LARGE GRAPH (eval_batch)
       |no
       v
  EAGER FALLBACK (variable size, no padding)
```

| Tier | Graph Size | When Used | Tradeoff |
|------|-----------|-----------|----------|
| Mini | 64 | batch >= mini_threshold | Minimal padding for tiny batches |
| Small | 128 | batch >= small_threshold | Small graph for small-medium batches |
| Medium | eval_batch / 2 | batch >= medium_threshold | Half-batch for mid-range |
| Large | eval_batch | batch > ~15/16 of eval_batch | Near-full batches only |
| Eager | variable | below all thresholds | No padding waste, but launch overhead |

**Routing is smallest-first** — picks the tightest-fitting graph to minimize padding waste. A batch of size 50 uses the mini graph (64), not the small graph (128).

**Capture process:**
```python
# Pre-allocate static tensors at graph size
static_obs = torch.zeros(graph_size, 122, 8, 8, device='cuda')
static_mask = torch.zeros(graph_size, POLICY_SIZE, device='cuda')

# Warm-up forward pass (required by CUDA graph API)
with torch.no_grad(), autocast('cuda'):
    network(static_obs, static_mask)

# Capture graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    with torch.no_grad(), autocast('cuda'):
        network(static_obs, static_mask)

# At inference time: copy data into static tensors, then replay
static_obs[:batch_size].copy_(torch.from_numpy(obs_array[:batch_size]))
static_mask[:batch_size].copy_(torch.from_numpy(mask_array[:batch_size]))
graph.replay()   # near-zero launch overhead
```

Per-tier timing statistics are tracked for performance analysis.

**Source:** `train.py:1385-1461` — graph capture; `train.py:1492-1584` — tier routing in evaluator

### 7.3 Calibration

At startup, `calibrate_cuda_graphs()` measures eager vs. graph overhead to determine crossover thresholds:

```
Graph wins when:  graph_replay_time  <  eager_overhead + per_sample * batch_size
Crossover batch:  B = (graph_time - eager_overhead) / per_sample
```

The crossover point depends on GPU model, network size, and batch size distribution. The calibration runs warmup iterations, measures timing at multiple batch sizes, and computes optimal thresholds.

**Source:** `train.py:887-1060` — `calibrate_cuda_graphs()`

### 7.4 Mixed Precision

The system uses multiple precision levels for different operations:

- **TF32 for matmuls:** `torch.set_float32_matmul_precision("high")` — 19-bit mantissa, 2x faster than FP32
- **cuDNN TF32 for convolutions:** `cudnn.conv.fp32_precision = "tf32"`
- **FP16 autocast** for forward and backward passes — `autocast('cuda')`
- **GradScaler** for FP16 stability during training — handles loss scaling and inf detection

Illegal move masking uses `-1e4` (not `-inf`) because `-inf` can cause NaN in FP16 softmax.

---

## 8. Training Loop

**Source:** `train.py`

### 8.1 Replay Buffer

**Source:** `replay_buffer.hpp/cpp`

A circular buffer stores the most recent training samples from self-play:

| Field | Size per sample | Description |
|-------|----------------|-------------|
| observation | 7808 floats (8 x 8 x 122) | Board encoding (NHWC) |
| policy | 4672 floats | MCTS visit distribution |
| value | 1 float | Game outcome z in {+1, 0, -1} |
| wdl_target | 3 floats | Root WDL probabilities [P(win), P(draw), P(loss)] |
| soft_value | 1 float | ERM risk-adjusted root value (0 if disabled) |

**Total per sample:** 7808 + 4672 + 1 + 3 + 1 = **12,485 floats** (49,940 bytes)

Sampling is **uniform random with replacement**. The buffer uses a lock-free atomic write position — no CAS contention. Size is computed as `min(total_added, capacity)`, eliminating the old `current_size_` CAS bottleneck.

Optional **stratified buffers** route samples by game outcome (win/draw/loss) for balanced sampling during training.

**Source:** `replay_buffer.hpp:21-197` — `ReplayBuffer` class

### 8.2 Loss Function

The total loss has two components:

**Policy Loss — NaN-safe cross-entropy:**
```
L_policy = -(1/B) * sum_i sum_a  pi_target(a) * log_softmax(z_p)(a)
```

Implemented with NaN safety for the `0 * -inf = NaN` edge case:
```python
policy_loss = -torch.sum(
    torch.nan_to_num(policy_target * F.log_softmax(policy_logits, dim=1), nan=0.0)
) / batch_size
```

**Value Loss — WDL soft cross-entropy with pure game outcomes:**

Construct the WDL target from the scalar game result:
```python
# Build outcome WDL from scalar game result
outcome_wdl[:, 0] = (value_target > 0.5).float()     # win  -> [1, 0, 0]
outcome_wdl[:, 1] = (|value_target| <= 0.5).float()   # draw -> [0, 1, 0]
outcome_wdl[:, 2] = (value_target < -0.5).float()     # loss -> [0, 0, 1]
```

Then compute the loss:
```
L_value = -(1/B) * sum_i sum_{c in {W,D,L}}  outcome_wdl(c) * log_softmax(z_w)(c)
```

**Total:**
```
L = L_policy + L_value
```

There is no explicit regularization term — weight decay is handled by the optimizer (see [Section 8.4](#84-optimizer)).

**Source:** `train.py:1983-2007` — loss computation within `train_iteration()`

### 8.3 Value Labels: Pure Game Outcomes

The value loss always uses **pure game outcome WDL** — no blending with MCTS root WDL.

**Rationale:** With Entropic Risk Measure (ERM), risk sensitivity is a search-time knob (`risk_beta`). Blending MCTS root WDL into training labels would leak risk-adjusted search artifacts into the network weights, breaking the separation principle that allows a single network to be deployed at any `risk_beta`. Game outcomes `{+1, 0, -1}` are unbiased ground truth.

> **Historical note:** An earlier version supported a `--wdl-blend` schedule that ramped blending of MCTS root WDL into value labels. This was removed when ERM was adopted — see `docs/WDL+contemp.md` for the original design.

### 8.4 Optimizer

```python
optimizer = Adam([
    {'params': conv_and_linear_weights, 'weight_decay': 1e-4},
    {'params': batchnorm_and_bias_params, 'weight_decay': 0.0},
], lr=0.001)
```

- **Adam** with default betas (0.9, 0.999)
- **L2 regularization** (weight_decay=1e-4) applied to convolution and linear weights only
- **No regularization** on BatchNorm parameters and biases — penalizing the 1-filter BN bottleneck in the value head was found to cause WDL collapse
- Parameter groups split by keyword: any parameter name containing `'bn'` or `'bias'` gets zero weight decay
- **GradScaler** for FP16 stability — internally skips optimizer step if `unscale_()` found inf gradients

**Source:** `train.py:2327-2339` — optimizer and parameter group setup

### 8.5 Training Step

Per iteration:
```python
for epoch in range(num_epochs):           # default: 5 epochs
    batch = replay_buffer.sample(B)       # default B = 256
    obs, policies, values, wdl, soft_values = batch

    optimizer.zero_grad()

    with autocast('cuda'):
        policy_pred, value_pred, policy_logits, wdl_logits = network(obs_tensor)
        loss = policy_loss(policy_logits, policies) + value_loss(wdl_logits, wdl_target)

    # NaN detection: skip batch if loss is NaN/inf
    if torch.isnan(loss) or torch.isinf(loss):
        continue

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    # Gradient norm monitoring (no hard clipping by default)
    grad_norm = compute_grad_norm(network.parameters())

    scaler.step(optimizer)    # internally skips if unscale_ found inf
    scaler.update()
```

**NaN safety:** Before training, all parameters are checked for NaN/inf to catch corrupted checkpoints early.

**Source:** `train.py:1877-2050` — `train_iteration()`

---

## 9. The Virtuous Cycle

The key insight is that **MCTS is a policy improvement operator**. Given any policy `p` and value function `v` (even random), MCTS produces a strictly better policy `pi` by looking ahead:

```
pi(a|s) = N(s, a) / sum_j N(s, j)      (MCTS visit distribution)
```

This `pi` is better than `p` because MCTS combines the network's priors with actual lookahead — it discovers tactical sequences the network misses, and avoids moves the network likes but that lead to bad outcomes.

Training then distills this search-improved policy back into the network:

```
theta  <--  theta - lr * grad_theta [ L_policy(pi, p_theta) + L_value(z, v_theta) ]
```

After training:
- The network's **policy** becomes closer to the MCTS distribution (it learns which moves the search prefers)
- The network's **value** becomes more accurate at predicting game outcomes

With a better network, the next round of MCTS produces even stronger games, which produce even better training data:

```
Random network
    |
    v  MCTS makes it play okay
Self-play games (mediocre quality)
    |
    v  Training distills MCTS knowledge
Slightly better network
    |
    v  MCTS makes it play well
Self-play games (decent quality)
    |
    v  Training distills more MCTS knowledge
Much better network
    |
    ...continues until convergence
```

### 9.1 Why This Works (Theoretical Grounding)

The process is analogous to **policy iteration** in dynamic programming:

1. **Policy evaluation**: The value head learns `v_theta(s) ~= E[z | s]` — the expected game outcome from position `s` under the current policy
2. **Policy improvement**: MCTS uses `v_theta` to compute a better policy `pi` via lookahead
3. **Policy distillation**: The policy head learns `p_theta(a|s) ~= pi(a|s)`

The convergence argument: if `p_theta = pi` (network matches MCTS), then search adds no value, and the system is at a fixed point. Any gap between `p_theta` and `pi` represents room for improvement that training will close.

### 9.2 The Role of Each Component

| Component | Mathematical Role |
|-----------|-------------------|
| **Neural Network** | Function approximator: `f_theta(s) -> (p, v)` compresses the game tree into a compact representation |
| **MCTS** | Policy improvement operator: converts `(p, v)` into a stronger policy `pi` via lookahead |
| **Self-Play** | Data generator: produces `(s, pi, z, wdl)` tuples under the current best policy |
| **Replay Buffer** | Experience memory: circular buffer of recent training data for SGD sampling |
| **Training Loss** | Objective: minimize `CE(pi, p_theta) + CE_wdl(wdl_target, v_theta)` to align network with search |
| **Dirichlet Noise** | Exploration: prevents the learning loop from collapsing to deterministic play |
| **Temperature** | Diversity: early-game stochastic move selection (tau=1.0, first 30 moves) ensures broad game coverage |
| **WDL Head** | Richer signal: three-class output provides more gradient than a scalar; enables WDL blend schedule |
| **Risk Beta (ERM)** | Search-time risk sensitivity: beta > 0 prefers complex positions, beta < 0 prefers safe positions |
| **Evaluation Queue** | Throughput: lock-free cross-game batching maximizes GPU utilization across N workers |
| **CUDA Graphs** | Latency: pre-captured graphs eliminate kernel launch overhead for small batches |

---

## 10. Key Parameters Reference

### MCTS Parameters

| Parameter | Default | Source | Purpose |
|-----------|---------|--------|---------|
| `num_simulations` | 800 | `search.hpp:28` | MCTS simulations per move |
| `c_puct` | 1.5 | `search.hpp:30` | Exploration-exploitation balance |
| `dirichlet_alpha` | 0.3 | `search.hpp:31` | Noise concentration (chess-specific) |
| `dirichlet_epsilon` | 0.25 | `search.hpp:32` | Noise mixing weight |
| `fpu_base` | 1.0 | `search.hpp:36` | Dynamic FPU penalty scale |
| `risk_beta` | 0.0 | `search.hpp:35` | ERM risk sensitivity (>0 seeking, <0 averse) |

### Network Parameters

| Parameter | Default | Source | Purpose |
|-----------|---------|--------|---------|
| `input_channels` | 122 | `network.py:25` | Board encoding dimensionality |
| `num_filters` | 192 | `network.py:133` | Residual tower channel width |
| `num_blocks` | 15 | `network.py:134` | Residual tower depth |
| `policy_filters` | 2 | `network.py:136` | Policy head 1x1 conv filters |
| `value_filters` | 1 | `network.py:137` | Value head 1x1 conv filters |
| `value_hidden` | 256 | `network.py:138` | Value head hidden layer size |
| `num_actions` | 4672 | `network.py:26` | Action space dimensionality |

### Training Parameters

| Parameter | Default | Source | Purpose |
|-----------|---------|--------|---------|
| `lr` | 0.001 | `train.py` args | Adam learning rate |
| `weight_decay` | 1e-4 | `train.py:2330` | L2 regularization (conv/linear only) |
| `train_batch` | 256 | `train.py` args | SGD mini-batch size |
| `epochs` | 5 | `train.py` args | Training passes per iteration |
### Self-Play Parameters

| Parameter | Default | Source | Purpose |
|-----------|---------|--------|---------|
| `num_workers` | 16 | `parallel_coordinator.hpp:30` | Parallel game workers |
| `games_per_worker` | 4 | `parallel_coordinator.hpp:31` | Games each worker plays |
| `temperature_moves` | 30 | `parallel_coordinator.hpp:41` | Moves with tau=1.0 sampling |
| `max_moves_per_game` | 512 | `parallel_coordinator.hpp:42` | Max moves before forced draw |
| `mcts_batch_size` | 1 | `parallel_coordinator.hpp:35` | Per-game leaf batch (1 = optimal for cross-game) |

### GPU and Queue Parameters

| Parameter | Default | Source | Purpose |
|-----------|---------|--------|---------|
| `gpu_batch_size` | 512 | `parallel_coordinator.hpp:45` | Maximum GPU batch size |
| `gpu_timeout_ms` | 20 | `parallel_coordinator.hpp:46` | GPU batch timeout |
| `worker_timeout_ms` | 2000 | `parallel_coordinator.hpp:47` | Worker result timeout |
| `queue_capacity` | 8192 | `evaluation_queue.hpp:25` | Evaluation queue capacity |
| `MAX_WORKERS` | 128 | `evaluation_queue.hpp:23` | Maximum concurrent workers |
| `NODES_PER_BLOCK` | 16384 | `node_pool.hpp:16` | Nodes per arena block (1MB) |
| `MAX_BLOCKS` | 512 | `node_pool.hpp:17` | Max blocks per worker (512MB) |

### Risk Schedule Parameters

| Parameter | Default | Source | Purpose |
|-----------|---------|--------|---------|
| `risk_beta` | 0.0 | `parallel_coordinator.hpp:61` | ERM risk beta (start) |
| `risk_beta_final` | (same) | `train.py` args | Risk beta at end of schedule |
| `risk_beta_warmup` | 0 | `train.py` args | Iterations before risk beta ramp |
| `opponent_risk_min` | 0.0 | `parallel_coordinator.hpp:65` | Asymmetric opponent risk range min |
| `opponent_risk_max` | 0.0 | `parallel_coordinator.hpp:66` | Asymmetric opponent risk range max |
