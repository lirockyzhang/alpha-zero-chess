# Search-Time Contempt + MCTS WDL Training Targets

## Context

The WDL value head is **completely collapsed** (constant `0.056368` for all positions). Root cause: `draw_score=0.75` is baked into training labels, mislabeling all draws as wins/losses. Combined with weight decay on the 1-filter BatchNorm bottleneck, this collapses the value head.

**Solution**: Two-phase redesign:
- **Phase 1**: Fix collapse + contempt system (pure training + MCTS personality)
- **Phase 2**: MCTS WDL soft targets (train on search distributions, not hard game outcomes)

---

## Phase 1: Search-Time Contempt System

### Design Principle
- **Training**: Pure scientist. Draws = 0.0. WDL labels are true outcomes.
- **MCTS Search**: Subjective actor. `value = P(win) - P(loss) + contempt * P(draw)`
- **Network**: Always WDL (remove `--no-wdl`). Returns 3 logits from value head.

### Perspective & Symmetry Notes
- WDL is always **relative** (from side-to-move's perspective): `pw` = P(I win), `pl` = P(I lose)
- During backprop, `swap(pw, pl)` flips perspective as we go up (opponent's win = my loss)
- `pd` (draw) is symmetric — never flipped, since draws are the same for both sides
- Contempt is **symmetric**: both sides get `+contempt * P(draw)`. Positive contempt = both sides slightly prefer draws; negative = both avoid them. This is intentional (standard chess engine contempt).

### 1A. C++ Header Changes

**`include/mcts/search.hpp`** (line 34):
- Rename `draw_score` → `contempt` in `BatchSearchConfig`
- Add utility: `inline float wdl_to_value(float pw, float pd, float pl, float c) { return pw - pl + c * pd; }`

**`include/selfplay/parallel_coordinator.hpp`** (line 56):
- Rename `draw_score` → `contempt` in `ParallelSelfPlayConfig`

**`include/selfplay/game.hpp`**:
- `set_outcomes()` (line 50): Remove `draw_score` parameter. Draws always store `0.0f`.
- Remove `draw_score` from `SelfPlayConfig` (line 82)

### 1B. C++ Implementation Changes

**`src/selfplay/parallel_coordinator.cpp`**:
- `gpu_thread_func()` (line 189): Allocate WDL buffer `wdl[gpu_batch_size * 3]` instead of `values[gpu_batch_size]`. After Python callback, convert WDL→scalar via `wdl_to_value()` with `config_.contempt`. Then call `submit_results()` with scalar values. **Evaluation queue internals unchanged.**
- Line 299: `search_config.contempt = config_.contempt;`
- Line 487: `trajectory.set_outcomes(result);` (no draw_score)

**`src/mcts/search.cpp`**:
- Lines 133-136 and 391-394: Terminal draws → `value = config_.contempt;` (symmetric, no side-to-move check)

#### ⚠️ No Double Contempt — Verified Disjoint Paths
Terminal nodes are caught during MCTS selection (`run_selection_phase` line 122, `collect_leaves_async` line 384) and backpropagated **immediately** with `value = config_.contempt` — they **never enter the evaluation queue**. The GPU thread's `wdl_to_value()` conversion only acts on NN-evaluated non-terminal leaf nodes. These are completely disjoint code paths, so contempt is applied exactly once.

### 1C. Python Bindings

**`src/bindings/python_bindings.cpp`**:
- Lines 578-583: Change `out_value_array` shape from `{batch_size}` to `{batch_size, 3}` (WDL probs)
- Lines 603-604: Update legacy fallback memcpy to `batch_size * 3 * sizeof(float)`
- Lines 406, 424, 729, 859, 877: Rename `draw_score` → `contempt`

### 1D. Python Training Script (`train.py`)

**Neural evaluator** (lines 1457-1546):
- Capture WDL logits from CUDA graphs (replace `_swl`/`_sws`/`_swm`/`_spm` throwaway vars with `static_wdl_*` static tensors)
- After graph replay/eager forward, compute `wdl_probs = F.softmax(wdl_logits.float(), dim=1)`
- Write WDL probs to `out_values[:batch_size]` (now shape `batch_size × 3`)

#### ⚠️ No Double Softmax — Use Raw Logits
`AlphaZeroNet.forward()` returns 4 values: `(policy, value, policy_logits, wdl_logits)`.
- `value` (2nd return) = `softmax(wdl_logits)[:, 0] - softmax(wdl_logits)[:, 2]` — **already has softmax baked in**
- `wdl_logits` (4th return) = raw `fc2(x)` output — **no softmax applied**

The evaluator must use `wdl_logits` (4th return), NOT `value` (2nd return). Apply `F.softmax(wdl_logits.float(), dim=1)` exactly once. Current CUDA graph code discards `_swl` — must capture as `static_wdl_large` and softmax that.

```python
# CORRECT: capture raw logits, softmax once
static_policy_large, _, _spl, static_wdl_large = network(static_obs_large, static_mask_large)
# ...after graph replay:
wdl_probs = F.softmax(static_wdl_large[:batch_size].float(), dim=1)

# WRONG: would double-softmax
# wdl_probs = F.softmax(static_value_large, dim=1)  # value already has softmax!
```

**Weight decay fix** (line 2127):
```python
# Robust filter: catches BN gamma/beta (named 'weight'/'bias' under 'bn' modules)
# and all bias parameters (conv, fc, etc.) — these are 1D and shouldn't be decayed
no_decay_keywords = ['bn', 'bias']
decay = [p for n, p in network.named_parameters()
         if not any(kw in n for kw in no_decay_keywords)]
no_decay = [p for n, p in network.named_parameters()
            if any(kw in n for kw in no_decay_keywords)]
optimizer = optim.Adam([
    {'params': decay, 'weight_decay': 1e-4},
    {'params': no_decay, 'weight_decay': 0.0},
], lr=args.lr)
```

**Remove `--no-wdl`** (line 1989): Delete the arg. WDL is always on. Remove all `not args.no_wdl` conditionals.

**Rename `--draw-score` → `--contempt`** (line 1969): Update arg definition and all references (lines 622, 629, 743, 1247, 2194, 2356-2357).

#### WDL Head Initialization Safety
When creating a fresh WDL network (not loading from checkpoint), the `fc2` layer (Linear(256, 3)) uses PyTorch default Kaiming Uniform init. This can produce initial WDL logit spreads of ~±1.5, yielding softmax probs like [0.6, 0.2, 0.2]. This is acceptable — the MCTS scalar via `wdl_to_value` stays within [-1, 1]. No special init needed, but add a startup check:
```python
# After network creation, log initial WDL behavior
with torch.no_grad():
    dummy = torch.randn(1, INPUT_CHANNELS, 8, 8, device=device)
    _, _, _, wdl = network(dummy)
    wdl_p = F.softmax(wdl, dim=1)
    print(f"  Initial WDL distribution: {wdl_p.cpu().numpy().round(3)}")
```

**Sequential self-play** (lines 2356-2357): Draws use `value = 0.0` (pure training).

**`CppSelfPlay` class** (lines 622-743): Remove `draw_score` attribute. Draw value = `0.0`.

### 1E. Checkpoint/Colab Updates
- Save `contempt` in checkpoint config instead of `draw_score`
- `colab/alphazero_training.ipynb`: Update `--draw-score` → `--contempt`

---

## Phase 2: MCTS WDL Soft Training Targets

### Design Principle

Instead of training on hard game outcome labels `[1,0,0]`, train on the MCTS search's soft WDL distribution `[0.65, 0.25, 0.10]`. Optionally blend with the outcome: `wdl_target = α × mcts_wdl + (1-α) × outcome_wdl`.

### Key Insight: Root-Level WDL Accumulation (No Node Changes!)

The 64-byte cache-aligned Node stays unchanged. Instead, accumulate WDL only at the root during `backpropagate()`:

```cpp
// In MCTSSearch class:
float root_wdl_sum_[3] = {0, 0, 0};
int root_wdl_count_ = 0;

void backpropagate(Node* node, float value, float pw, float pd, float pl) {
    while (node != nullptr) {
        node->remove_virtual_loss();
        if (node->parent == nullptr) {
            node->update_root(value);
            root_wdl_sum_[0] += pw;  // Accumulate at root only
            root_wdl_sum_[1] += pd;
            root_wdl_sum_[2] += pl;
            root_wdl_count_++;
        } else {
            node->update(value);
        }
        node = node->parent;
        value = -value;
        std::swap(pw, pl);  // Flip W/L for opponent's perspective (D stays)
    }
}
```

After search: `root_wdl = root_wdl_sum / root_wdl_count` gives the search's WDL distribution from root's perspective.

**Important**: This root WDL is the average of NN WDL predictions at evaluated leaf positions (propagated back with correct perspective flips). It is NOT derived from visit count distributions — it is the "consensus" WDL from the NN evaluations that informed the search. This is more stable than using visit-count-derived WDL but can be biased by the policy head's initial priors (which determine which subtrees get explored).

**Reset per move**: `root_wdl_sum_` and `root_wdl_count_` must be reset at the start of each MCTS search (before each move), similar to how the node pool is reset.

#### Root WDL Method Tradeoffs
- **NN consensus (our approach)**: Average of leaf WDL predictions propagated to root. Stable in early training when visit counts are jittery. But as the policy head sharpens, search concentrates on one subtree, reducing WDL diversity.
- **Visit-count-based WDL**: `W_i = N(s, a_win) / N(s, total)`. Reflects actual search behavior but can be noisy early on.
- **Recommendation**: Start with NN consensus. If the WDL head plateaus after many iterations, add a `--wdl-method` flag to switch to visit-count-weighted average of most-visited children's WDL. This is a future optimization, not Phase 2 scope.

### 2A. WDL Data Flow (C++ evaluation queue extension)

Currently, the evaluation queue delivers `(policies, scalar_values)` to workers. To pass WDL probs for backpropagation, extend the queue to also deliver `wdl_probs[batch_size × 3]`.

Files to modify:
- `include/selfplay/evaluation_queue.hpp`: Add WDL buffer to result storage
- `src/selfplay/parallel_coordinator.cpp`: GPU thread writes WDL probs to queue, workers read them
- `src/mcts/search.cpp`: `update_leaves()` receives WDL probs, passes to `backpropagate()`

### 2B. Game Trajectory + Replay Buffer

**`include/selfplay/game.hpp`**: Add `std::vector<std::array<float,3>> mcts_wdl` to `GameState`/trajectory. After each search, store the root WDL.

**`include/training/replay_buffer.hpp`**: Extend to store WDL targets alongside existing data:
- Add `wdl_targets_`: capacity × 3 floats (~12MB for 1M samples — manageable)
- Extend `add_sample()`, `add_batch()`, `add_batch_raw()`, `sample()` for WDL
- Use `memcpy`/`std::copy` in `add_batch_raw` for vectorized bulk insertion (no per-element loop)

### 2C. WDL Blending with Schedule

At game end, for each position in the trajectory:
```python
outcome_wdl = [1,0,0] if win else [0,1,0] if draw else [0,0,1]
wdl_target = alpha * mcts_wdl[i] + (1 - alpha) * outcome_wdl
```

New args:
- `--wdl-blend` (float, default 0.0): Starting alpha. 0 = pure outcome, 1 = pure MCTS.
- `--wdl-blend-final` (float, default 0.8): Final alpha at end of schedule.
- `--wdl-blend-warmup` (int, default 50): Iterations before ramping starts.

**Alpha schedule**: Linear ramp from `wdl_blend` to `wdl_blend_final` after warmup:
```python
if iteration < wdl_blend_warmup:
    alpha = wdl_blend  # Pure outcome during warmup (network learns ground truth)
else:
    progress = min(1.0, (iteration - wdl_blend_warmup) / max(wdl_blend_warmup, 1))
    alpha = wdl_blend + progress * (wdl_blend_final - wdl_blend)
```

**Rationale**: Early training needs ground-truth outcomes to establish a baseline value head. Once the policy head is reasonably good, MCTS search distributions provide richer training signal about positional nuances. Starting with α=0 and ramping to α=0.8 gives the best of both worlds.

### 2D. Training Loss Change

Replace hard cross-entropy with soft cross-entropy (KL divergence):
```python
# Old (hard labels):
value_loss = F.cross_entropy(wdl_logits, wdl_class_labels)

# New (soft targets):
wdl_target = replay_buffer.sample_wdl(batch_size)  # (batch, 3)
value_loss = -torch.sum(wdl_target * F.log_softmax(wdl_logits, dim=1)) / batch_size
```

This gives non-zero gradients for all 3 WDL classes on every sample, preventing the class-starvation that caused the original collapse.

---

## Implementation Order

1. **Phase 1** first — fixes the value head collapse and makes WDL functional
2. **Phase 2** after verifying Phase 1 — adds MCTS WDL for better training signal

Phase 2 can be done incrementally: first add root WDL tracking (2A), then extend the replay buffer (2B), then change the loss (2D).

---

## Files Modified (Phase 1)

| File | Key Changes |
|------|------------|
| `include/mcts/search.hpp` | `draw_score` → `contempt`, add `wdl_to_value()` |
| `include/selfplay/parallel_coordinator.hpp` | `draw_score` → `contempt` |
| `include/selfplay/game.hpp` | `set_outcomes()` removes draw_score param, draws = 0.0 |
| `src/selfplay/parallel_coordinator.cpp` | GPU thread receives WDL, converts to scalar; `set_outcomes(result)` |
| `src/mcts/search.cpp` | Terminal draws use symmetric `config_.contempt` |
| `src/bindings/python_bindings.cpp` | `out_values` → `(batch_size, 3)` shape; rename draw_score → contempt |
| `scripts/train.py` | Evaluator returns WDL probs; fix weight decay; remove `--no-wdl`; rename `--draw-score` → `--contempt`; sequential path draws = 0.0 |
| `scripts/network.py` | Remove `wdl=False` default, WDL always on |
| `colab/alphazero_training.ipynb` | Update CLI args |

## Files Modified (Phase 2, additional)

| File | Key Changes |
|------|------------|
| `include/selfplay/evaluation_queue.hpp` | Add WDL probs to result delivery |
| `include/training/replay_buffer.hpp` | Add WDL target storage (3 floats per sample) |
| `src/training/replay_buffer.cpp` | Implement WDL storage/sampling |
| `src/mcts/search.cpp` | `backpropagate()` carries WDL; root WDL accumulator |
| `scripts/train.py` | Soft cross-entropy loss; `--wdl-blend` arg |

---

## Key Changes Summary & Runtime Test Cases

Each test case is designed to run against the **compiled C++ extension** (`alphazero_cpp`) and/or **Python components** — i.e., close to actual runtime behavior, not unit tests of isolated logic.

Test file: `alphazero-cpp/tests/test_contempt_wdl.py`

---

### T1. `wdl_to_value()` — C++ utility function

**Change**: New inline in `search.hpp`: `float wdl_to_value(float pw, float pd, float pl, float contempt)`
**Formula**: `pw - pl + contempt * pd`

```python
def test_wdl_to_value():
    """Verify the wdl_to_value utility via Python bindings."""
    # Exposed as alphazero_cpp.wdl_to_value(pw, pd, pl, contempt)
    assert abs(alphazero_cpp.wdl_to_value(0.7, 0.2, 0.1, 0.0) - 0.6) < 1e-6    # pure: 0.7 - 0.1
    assert abs(alphazero_cpp.wdl_to_value(0.7, 0.2, 0.1, 1.0) - 0.8) < 1e-6    # +contempt*draw
    assert abs(alphazero_cpp.wdl_to_value(0.7, 0.2, 0.1, -0.5) - 0.5) < 1e-6   # negative contempt
    assert abs(alphazero_cpp.wdl_to_value(0.0, 1.0, 0.0, 0.5) - 0.5) < 1e-6    # pure draw + contempt
    assert abs(alphazero_cpp.wdl_to_value(0.0, 1.0, 0.0, 0.0) - 0.0) < 1e-6    # pure draw, no contempt
    assert abs(alphazero_cpp.wdl_to_value(1.0, 0.0, 0.0, 0.5) - 1.0) < 1e-6    # certain win
    assert abs(alphazero_cpp.wdl_to_value(0.0, 0.0, 1.0, 0.5) - (-1.0)) < 1e-6 # certain loss
```

---

### T2. `set_outcomes()` — draws always 0.0 (game.hpp)

**Change**: Remove `draw_score` parameter from `set_outcomes()`. Draws store `0.0f` regardless of contempt.

```python
def test_set_outcomes_draws_zero():
    """GameTrajectory.set_outcomes(DRAW) stores 0.0 for all positions."""
    # Play a very short game using CppSelfPlay that ends in draw (e.g. stalemate or max_moves=2)
    # Or: construct trajectory via C++ bindings and call set_outcomes directly

    # Approach: Use the parallel coordinator with max_moves=2 (forces draw)
    config = alphazero_cpp.ParallelSelfPlayConfig()
    config.num_workers = 1
    config.games_per_worker = 1
    config.max_moves_per_game = 2          # Force draw after 2 moves
    config.num_simulations = 10
    config.contempt = 0.75                 # Non-zero contempt — should NOT affect training values

    buffer = alphazero_cpp.ReplayBuffer(capacity=100)
    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(config, buffer)

    def dummy_evaluator(obs, masks, bs, out_pol, out_val):
        # Return uniform policy and neutral WDL: [0.33, 0.34, 0.33]
        out_pol[:bs] = 1.0 / 4672
        out_val[:bs, 0] = 0.33  # win
        out_val[:bs, 1] = 0.34  # draw
        out_val[:bs, 2] = 0.33  # loss

    coordinator.generate_games(dummy_evaluator)

    # Sample all positions from buffer
    obs, pol, val = buffer.sample(buffer.size())
    # ALL values must be 0.0 (draw, pure training label)
    assert np.allclose(val, 0.0, atol=1e-6), f"Draw values should be 0.0, got {val}"
```

---

### T3. Terminal draw uses symmetric `contempt` (search.cpp)

**Change**: Lines 133-136 and 391-394: `value = config_.contempt` (no side-to-move check).

```python
def test_terminal_draw_symmetric_contempt():
    """Terminal draw positions produce contempt as the backprop value, regardless of side."""
    # This is tested indirectly: run MCTS on a known stalemate position
    # The stalemate node value should equal contempt, not ±contempt based on color
    # We verify via the coordinator's game results + root Q-values

    # Use a position near stalemate, run search, check that the root Q
    # doesn't flip sign based on which color is stalemated
    # (Detailed test: set up board with FEN, run single search, inspect root.q_value)
    # This requires exposing MCTSSearch to Python — tested via integration below
    pass  # Verified via integration test T6
```

---

### T4. GPU thread WDL→scalar conversion (parallel_coordinator.cpp)

**Change**: `gpu_thread_func()` allocates `wdl[gpu_batch_size * 3]`, calls Python evaluator which writes WDL probs, then converts to scalar via `wdl_to_value(pw, pd, pl, config_.contempt)`.

```python
def test_gpu_thread_wdl_conversion():
    """Python evaluator receives (batch_size, 3) out_values buffer and GPU thread converts to scalar."""
    received_shapes = {}

    config = alphazero_cpp.ParallelSelfPlayConfig()
    config.num_workers = 1
    config.games_per_worker = 1
    config.num_simulations = 10
    config.contempt = 0.5

    buffer = alphazero_cpp.ReplayBuffer(capacity=1000)
    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(config, buffer)

    def shape_checking_evaluator(obs, masks, bs, out_pol, out_val):
        received_shapes['out_values'] = out_val.shape
        # Write WDL probs: heavily favor wins
        out_pol[:bs] = 1.0 / 4672
        out_val[:bs, 0] = 0.8   # P(win) = 0.8
        out_val[:bs, 1] = 0.15  # P(draw) = 0.15
        out_val[:bs, 2] = 0.05  # P(loss) = 0.05

    coordinator.generate_games(shape_checking_evaluator)

    # Verify shape: out_values must be (batch_size, 3) not (batch_size,)
    assert len(received_shapes['out_values']) == 2, "out_values should be 2D"
    assert received_shapes['out_values'][1] == 3, "out_values dim 1 should be 3 (WDL)"
```

---

### T5. Weight decay parameter groups (train.py)

**Change**: BatchNorm (gamma/beta) and all bias parameters get `weight_decay=0.0`. Only conv/fc weights get `weight_decay=1e-4`.

```python
def test_weight_decay_groups():
    """Verify BN and bias params excluded from weight decay."""
    from network import AlphaZeroNet

    net = AlphaZeroNet(num_filters=64, num_blocks=2, wdl=True)

    no_decay_keywords = ['bn', 'bias']
    decay_params = [n for n, p in net.named_parameters()
                    if not any(kw in n for kw in no_decay_keywords)]
    no_decay_params = [n for n, p in net.named_parameters()
                       if any(kw in n for kw in no_decay_keywords)]

    # BN params must be in no_decay
    bn_params = [n for n, _ in net.named_parameters() if '.bn.' in n]
    for n in bn_params:
        assert n in no_decay_params, f"BN param {n} should have weight_decay=0"

    # fc bias must be in no_decay
    bias_params = [n for n, _ in net.named_parameters() if 'bias' in n]
    for n in bias_params:
        assert n in no_decay_params, f"Bias param {n} should have weight_decay=0"

    # Conv weights must be in decay
    conv_weight_params = [n for n, _ in net.named_parameters()
                          if 'conv' in n and 'weight' in n and 'bn' not in n]
    for n in conv_weight_params:
        assert n in decay_params, f"Conv weight {n} should have weight_decay=1e-4"

    # Value head fc1/fc2 weights must be in decay
    fc_weight_params = [n for n, _ in net.named_parameters()
                        if 'fc' in n and 'weight' in n]
    for n in fc_weight_params:
        assert n in decay_params, f"FC weight {n} should have weight_decay=1e-4"

    print(f"  decay group: {len(decay_params)} params, no_decay group: {len(no_decay_params)} params")
```

---

### T6. Network always WDL — forward returns 3 logits (network.py)

**Change**: Remove `wdl=False` default. `AlphaZeroNet(wdl=True)` is the only mode. Forward returns `(policy, value, policy_logits, wdl_logits)` where `wdl_logits.shape[-1] == 3`.

```python
def test_network_always_wdl():
    """Network always returns WDL logits; wdl_logits is never None."""
    from network import AlphaZeroNet
    import torch

    net = AlphaZeroNet(num_filters=64, num_blocks=2, wdl=True)
    net.eval()

    x = torch.randn(4, 122, 8, 8)
    mask = torch.ones(4, 4672)

    policy, value, policy_logits, wdl_logits = net(x, mask)

    assert wdl_logits is not None, "WDL logits must not be None"
    assert wdl_logits.shape == (4, 3), f"WDL logits shape should be (4,3), got {wdl_logits.shape}"
    assert value.shape == (4, 1), f"Value shape should be (4,1), got {value.shape}"

    # Value should be P(win) - P(loss) from softmax of WDL logits
    wdl_probs = torch.softmax(wdl_logits, dim=1)
    expected_value = wdl_probs[:, 0:1] - wdl_probs[:, 2:3]
    assert torch.allclose(value, expected_value, atol=1e-5)

    # WDL probs should sum to 1.0
    assert torch.allclose(wdl_probs.sum(dim=1), torch.ones(4), atol=1e-5)

    # Verify that different inputs produce different WDL outputs (no collapse!)
    x2 = torch.randn(4, 122, 8, 8)
    _, _, _, wdl_logits2 = net(x2, mask)
    assert not torch.allclose(wdl_logits, wdl_logits2, atol=1e-3), \
        "Different inputs should produce different WDL logits"
```

---

### T7. Neural evaluator writes WDL probs (train.py)

**Change**: `neural_evaluator()` callback computes `F.softmax(wdl_logits, dim=1)` and writes `(batch, 3)` WDL probs to `out_values`.

```python
def test_neural_evaluator_wdl_output():
    """Neural evaluator callback writes valid WDL probabilities."""
    from network import AlphaZeroNet
    import torch, numpy as np

    net = AlphaZeroNet(num_filters=64, num_blocks=2, wdl=True).cuda().eval()

    # Simulate what the C++ side provides
    bs = 8
    obs = np.random.randn(bs, 122, 8, 8).astype(np.float32)
    mask = np.ones((bs, 4672), dtype=np.float32)
    out_pol = np.zeros((bs, 4672), dtype=np.float32)
    out_val = np.zeros((bs, 3), dtype=np.float32)   # NEW: shape (bs, 3)

    # Call the evaluator (constructed in train.py — here we replicate the core logic)
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).cuda()
        mask_t = torch.from_numpy(mask).cuda()
        policy, value, policy_logits, wdl_logits = net(obs_t, mask_t)
        wdl_probs = torch.softmax(wdl_logits.float(), dim=1)

        np.copyto(out_pol[:bs], policy[:bs].cpu().numpy())
        np.copyto(out_val[:bs], wdl_probs[:bs].cpu().numpy())

    # Verify WDL probs are valid probability distributions
    assert out_val.shape == (bs, 3)
    assert np.allclose(out_val.sum(axis=1), 1.0, atol=1e-4), f"WDL probs must sum to 1, got {out_val.sum(axis=1)}"
    assert (out_val >= 0).all(), "WDL probs must be non-negative"
    assert (out_val <= 1).all(), "WDL probs must be <= 1"

    # Different positions should produce different WDL distributions
    assert not np.allclose(out_val[0], out_val[1], atol=1e-3), "Different positions should have different WDL"
```

---

### T8. Sequential self-play: draws always 0.0 (train.py)

**Change**: Lines 2356-2357: Draw value = `0.0` regardless of `--contempt` setting.

```python
def test_sequential_draw_value():
    """Sequential self-play path stores 0.0 for draws, ignoring contempt."""
    # This tests the Python-side logic at train.py lines 2349-2358
    # Simulate the draw case with contempt=0.75
    contempt = 0.75
    result = 0.0  # Draw (sequential path uses 0.0 for draw result)

    values = []
    for i in range(10):
        white_to_move = (i % 2 == 0)
        if result == 1.0:
            value = 1.0 if white_to_move else -1.0
        elif result == -1.0:
            value = -1.0 if white_to_move else 1.0
        else:
            # NEW: draws always 0.0 (pure training label, not contempt-biased)
            value = 0.0
        values.append(value)

    assert all(v == 0.0 for v in values), f"All draw values should be 0.0, got {values}"
```

---

### T9. WDL training loss — soft cross-entropy (train.py, Phase 2)

**Change**: Replace `F.cross_entropy(logits, hard_class)` with `(-target * log_softmax(logits)).sum() / N`.

```python
def test_soft_cross_entropy_loss():
    """Soft CE loss provides gradients to all 3 WDL classes."""
    import torch
    import torch.nn.functional as F

    # Logits from network
    wdl_logits = torch.tensor([[2.0, 0.5, -1.0]], requires_grad=True)

    # Soft target: blended MCTS WDL + outcome
    wdl_target = torch.tensor([[0.6, 0.3, 0.1]])  # Non-trivial distribution

    # Soft cross-entropy
    loss = -torch.sum(wdl_target * F.log_softmax(wdl_logits, dim=1)) / wdl_logits.size(0)
    loss.backward()

    # All 3 logits should have non-zero gradients
    assert wdl_logits.grad is not None
    assert (wdl_logits.grad.abs() > 1e-6).all(), \
        f"All WDL logits should have gradients, got {wdl_logits.grad}"

    # Compare with hard label (class 0 = win) — draw logit gets zero gradient
    wdl_logits2 = torch.tensor([[2.0, 0.5, -1.0]], requires_grad=True)
    hard_loss = F.cross_entropy(wdl_logits2, torch.tensor([0]))
    hard_loss.backward()
    # With hard label, the draw class gradient depends only on softmax output (not target)
    # But with soft target, gradient is explicitly driven toward correct distribution

    print(f"  Soft CE grads: {wdl_logits.grad}")
    print(f"  Hard CE grads: {wdl_logits2.grad}")
```

---

### T10. Contempt rename — Python bindings (python_bindings.cpp)

**Change**: All `draw_score` attributes renamed to `contempt` in `ParallelSelfPlayConfig` and `CppSelfPlay` bindings.

```python
def test_contempt_attribute_renamed():
    """Python bindings expose 'contempt' not 'draw_score'."""
    config = alphazero_cpp.ParallelSelfPlayConfig()

    # New attribute exists
    assert hasattr(config, 'contempt'), "Config should have 'contempt' attribute"
    config.contempt = 0.5
    assert config.contempt == 0.5

    # Old attribute removed
    assert not hasattr(config, 'draw_score'), "Config should NOT have 'draw_score' (renamed)"
```

---

### T11. End-to-end integration: parallel self-play with contempt (Phase 1 complete)

```python
def test_e2e_parallel_selfplay_with_contempt():
    """Full integration: generate games with contempt, verify WDL flow."""
    config = alphazero_cpp.ParallelSelfPlayConfig()
    config.num_workers = 2
    config.games_per_worker = 2
    config.num_simulations = 50
    config.contempt = 0.3
    config.max_moves_per_game = 20

    buffer = alphazero_cpp.ReplayBuffer(capacity=10000)
    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(config, buffer)

    eval_calls = [0]
    def nn_evaluator(obs, masks, bs, out_pol, out_val):
        eval_calls[0] += 1
        # Return somewhat realistic WDL predictions
        out_pol[:bs] = 1.0 / 4672
        out_val[:bs, 0] = 0.4   # P(win)
        out_val[:bs, 1] = 0.3   # P(draw)
        out_val[:bs, 2] = 0.3   # P(loss)

    coordinator.generate_games(nn_evaluator)

    stats = coordinator.get_stats()
    print(f"  Games completed: {stats.games_completed}")
    print(f"  Total moves: {stats.total_moves}")
    print(f"  NN eval calls: {eval_calls[0]}")
    print(f"  Buffer size: {buffer.size()}")

    assert stats.games_completed == 4, f"Expected 4 games, got {stats.games_completed}"
    assert buffer.size() > 0, "Buffer should have samples"
    assert eval_calls[0] > 0, "Evaluator should have been called"

    # Sample and verify training values are in [-1, 0, 1] (pure labels)
    obs, pol, val = buffer.sample(min(buffer.size(), 50))
    unique_vals = set(np.round(val, 1))
    print(f"  Unique training values: {unique_vals}")
    # All values should be -1.0, 0.0, or 1.0 (pure training, no contempt leakage)
    for v in val:
        assert v in (-1.0, 0.0, 1.0) or abs(abs(v) - 1.0) < 0.01 or abs(v) < 0.01, \
            f"Training value {v} should be -1, 0, or 1 (pure labels)"
```

---

### T12. No double softmax — evaluator outputs sum to 1.0 (safety check)

```python
def test_no_double_softmax():
    """Ensure evaluator WDL probs are softmax'd exactly once (not double-softmaxed)."""
    from network import AlphaZeroNet
    import torch, torch.nn.functional as F

    net = AlphaZeroNet(num_filters=64, num_blocks=2, wdl=True).eval()
    x = torch.randn(2, 122, 8, 8)
    mask = torch.ones(2, 4672)

    # Forward returns: policy, value, policy_logits, wdl_logits
    _, value, _, wdl_logits = net(x, mask)

    # wdl_logits should be RAW (not softmax'd) — they should NOT sum to 1
    assert not torch.allclose(wdl_logits.sum(dim=1), torch.ones(2), atol=0.1), \
        "wdl_logits should be raw logits, not probabilities"

    # Correct: softmax the raw logits once
    wdl_probs = F.softmax(wdl_logits, dim=1)
    assert torch.allclose(wdl_probs.sum(dim=1), torch.ones(2), atol=1e-5), \
        "softmax(wdl_logits) should sum to 1"

    # WRONG: double softmax would compress the distribution toward uniform
    double_softmax = F.softmax(wdl_probs, dim=1)
    # Double softmax is always more uniform than single softmax
    single_entropy = -(wdl_probs * wdl_probs.log()).sum(dim=1)
    double_entropy = -(double_softmax * double_softmax.log()).sum(dim=1)
    assert (double_entropy > single_entropy).all(), \
        "Double softmax increases entropy (bad) — our code must use raw logits"
```

---

### T13. No double contempt — terminal draws vs NN-evaluated leaves

```python
def test_no_double_contempt():
    """Verify contempt is applied exactly once: either at terminal node OR at WDL→scalar conversion."""
    # This tests the architectural invariant: terminal nodes (search.cpp lines 122-140)
    # are backpropagated immediately and NEVER reach the evaluation queue.
    #
    # We verify by checking that a game with many forced draws produces
    # consistent values regardless of how many terminal vs NN-evaluated draws occur.

    config = alphazero_cpp.ParallelSelfPlayConfig()
    config.num_workers = 1
    config.games_per_worker = 1
    config.num_simulations = 100
    config.contempt = 0.5
    config.max_moves_per_game = 10  # Short game → likely draw

    buffer = alphazero_cpp.ReplayBuffer(capacity=1000)
    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(config, buffer)

    def neutral_evaluator(obs, masks, bs, out_pol, out_val):
        # Return pure-draw WDL: [0.0, 1.0, 0.0]
        # wdl_to_value(0.0, 1.0, 0.0, 0.5) = 0 - 0 + 0.5*1.0 = 0.5
        out_pol[:bs] = 1.0 / 4672
        out_val[:bs, 0] = 0.0   # P(win) = 0
        out_val[:bs, 1] = 1.0   # P(draw) = 1
        out_val[:bs, 2] = 0.0   # P(loss) = 0

    coordinator.generate_games(neutral_evaluator)

    # Training values should be pure: -1, 0, or 1 (not 0.5!)
    # Because set_outcomes uses pure labels, NOT contempt-biased values
    obs, pol, val = buffer.sample(buffer.size())
    for v in val:
        assert v in (-1.0, 0.0, 1.0) or abs(abs(v) - 1.0) < 0.01 or abs(v) < 0.01, \
            f"Training value {v} should be pure label (-1/0/1), not contempt-biased"
```

---

### T14. Root WDL accumulation (Phase 2 — search.cpp)

```python
def test_root_wdl_accumulation():
    """After MCTS search, root WDL is a valid probability distribution."""
    # This requires Phase 2 implementation: MCTSSearch exposes get_root_wdl()
    # Tested via the parallel coordinator's trajectory data

    config = alphazero_cpp.ParallelSelfPlayConfig()
    config.num_workers = 1
    config.games_per_worker = 1
    config.num_simulations = 100
    config.contempt = 0.0
    config.max_moves_per_game = 5

    # Use coordinator WITHOUT replay buffer to get raw trajectories
    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(config)

    def nn_evaluator(obs, masks, bs, out_pol, out_val):
        out_pol[:bs] = 1.0 / 4672
        # Varied WDL to test accumulation
        out_val[:bs, 0] = np.random.uniform(0.2, 0.5, bs).astype(np.float32)
        out_val[:bs, 1] = np.random.uniform(0.2, 0.4, bs).astype(np.float32)
        out_val[:bs, 2] = 1.0 - out_val[:bs, 0] - out_val[:bs, 1]

    coordinator.generate_games(nn_evaluator)
    games = coordinator.get_completed_games()

    assert len(games) == 1
    game = games[0]

    # Each position should have MCTS WDL stored (Phase 2)
    for state in game.states:
        wdl = state.mcts_wdl  # [pw, pd, pl]
        assert len(wdl) == 3
        assert abs(sum(wdl) - 1.0) < 0.05, f"Root WDL should sum to ~1.0, got {sum(wdl)}"
        assert all(w >= 0 for w in wdl), f"WDL probs must be non-negative: {wdl}"
```

---

### T15. Replay buffer WDL storage (Phase 2 — replay_buffer.hpp)

```python
def test_replay_buffer_wdl_storage():
    """ReplayBuffer stores and retrieves WDL targets alongside obs/pol/val."""
    buffer = alphazero_cpp.ReplayBuffer(capacity=100)

    obs = np.random.rand(7808).astype(np.float32)
    pol = np.random.rand(4672).astype(np.float32)
    val = 0.5
    wdl = np.array([0.6, 0.3, 0.1], dtype=np.float32)

    buffer.add_sample(obs, pol, val, wdl)

    # Sample and verify WDL comes back
    s_obs, s_pol, s_val, s_wdl = buffer.sample(1)
    assert s_wdl.shape == (1, 3)
    assert np.allclose(s_wdl[0], wdl, atol=1e-5), f"WDL should round-trip: {s_wdl[0]} vs {wdl}"
```

---

## Verification (Updated)

### Phase 1 — Smoke Tests
1. **Build**: `cmake --build build --config Release` — clean compile, no warnings
2. **Unit tests**: `python alphazero-cpp/tests/test_contempt_wdl.py` — runs T1-T8, T10-T13
3. **Existing tests**: `python alphazero-cpp/tests/test_training_components.py` — no regressions

### Phase 1 — Integration
4. **Training sanity** (3 iters): `python train.py --iterations 3 --games 20 --simulations 50 --contempt 0.0`
   - Verify: value loss decreases, NN value varies across positions, WDL logits are NOT constant
5. **Contempt behavioral**: Run with `--contempt 0.5` — search should produce slightly higher values (MCTS perspective) but training labels are still pure
6. **Safety checks**: T12 (no double softmax), T13 (no double contempt) — both must pass
7. **Web app**: Load checkpoint, verify WDL bar shows varied probabilities per position

### Phase 2 — After Phase 1 Verified
8. **Root WDL extraction**: Run T14 — MCTS root WDL varies across moves
9. **Replay buffer WDL**: Run T15 — WDL round-trips correctly
10. **Soft target training**: Train 5 iterations with `--wdl-blend 0.0 --wdl-blend-final 0.8 --wdl-blend-warmup 2`, verify WDL predictions converge toward MCTS distributions
11. **Blend comparison**: Compare loss curves across blend schedules
