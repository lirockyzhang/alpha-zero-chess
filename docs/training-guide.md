# AlphaZero Training & Evaluation Guide

Complete guide for training and evaluating AlphaZero chess models using the C++ backend.

## Quick Start

### Training (Recommended)

Each training run creates an **organized output directory** with all artifacts:
```
checkpoints/f192-b15_2024-02-03_14-30-00/
├── model_iter_001.pt            # Checkpoints (every N iterations)
├── model_iter_005.pt
├── model_final.pt               # Final checkpoint
├── training_log.jsonl           # Append-only JSONL: config + per-iteration metrics
├── evaluation_results.json      # vs_random win rate, endgame scores
├── summary.html                 # Visual training summary (default)
└── replay_buffer.rpbf           # Replay buffer (saved after each self-play phase)
```

```bash
# Quick test run (~10 minutes, verifies setup works)
uv run python alphazero-cpp/scripts/train.py \
    --iterations 1 \
    --games-per-iter 5 \
    --simulations 100 \
    --filters 64 \
    --blocks 3

# Development training (~2-4 hours on RTX 4060)
uv run python alphazero-cpp/scripts/train.py \
    --iterations 20 \
    --games-per-iter 25 \
    --simulations 400 \
    --filters 64 \
    --blocks 5

# Fast training with parallel self-play (2-5x faster)
uv run python alphazero-cpp/scripts/train.py \
    --iterations 20 \
    --games-per-iter 64 \
    --workers 16 \
    --filters 64 \
    --blocks 5

# Production training (~12-24 hours, Gumbel SH is default)
uv run python alphazero-cpp/scripts/train.py \
    --iterations 100 \
    --games-per-iter 50 \
    --simulations 800 \
    --filters 192 \
    --blocks 15

# Same but with classic AlphaZero PUCT (for comparison)
uv run python alphazero-cpp/scripts/train.py \
    --iterations 100 \
    --games-per-iter 50 \
    --simulations 800 \
    --filters 192 \
    --blocks 15 \
    --search-algorithm puct
```

### Evaluation

**Note:** Evaluation now runs automatically before each checkpoint save (disable with `--no-eval`).
The results are saved to `evaluation_results.json` in the run directory.

```bash
# Evaluate against random player (quick sanity check)
uv run python alphazero/scripts/evaluate.py \
    --checkpoint checkpoints/f64-b5_2024-02-03_14-30-00/model_iter_020.pt \
    --opponent random \
    --games 20 \
    --simulations 100

# Evaluate endgame positions
uv run python alphazero/scripts/evaluate.py \
    --checkpoint checkpoints/f64-b5_2024-02-03_14-30-00/model_final.pt \
    --opponent endgame \
    --simulations 200
```

---

## Training Script Reference

**Location**: `alphazero-cpp/scripts/train.py`

### All Parameters

Parameters are grouped by which phase of training they affect:

#### Self-Play Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iterations` | 100 | Number of training iterations |
| `--games-per-iter` | 50 | Self-play games per iteration |
| `--workers` | 1 | Self-play workers. 1=sequential, >1=parallel with cross-game batching |
| `--temperature-moves` | 30 | Moves with temperature=1 for exploration |

#### MCTS Search Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--simulations` | 800 | MCTS simulations per move |
| `--c-explore` | 1.5 | MCTS exploration constant |
| `--search-algorithm` | gumbel | Root search: `gumbel` (Sequential Halving) or `puct` (Dirichlet) |
| `--gumbel-top-k` | 16 | Initial candidate actions for Sequential Halving |
| `--gumbel-c-visit` | 50.0 | Gumbel sigma() visit constant |
| `--gumbel-c-scale` | 1.0 | Gumbel sigma() scale factor |

#### Neural Network Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--filters` | 192 | Network filter count |
| `--blocks` | 15 | Residual block count |
| `--eval-batch` | auto | Auto-computed as workers x search_batch rounded to 32; user value is raised to this floor if too low |
| `--batch-timeout-ms` | 5 | Max wait time to fill GPU batch in ms |

#### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train-batch` | 256 | Samples per training gradient step |
| `--lr` | 0.001 | Learning rate |
| `--epochs` | 5 | Training passes per iteration |
| `--buffer-size` | 100000 | Replay buffer capacity |

#### System Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--device` | cuda | Device (cuda or cpu) |
| `--save-dir` | checkpoints | Base checkpoint directory |
| `--resume` | - | Resume from checkpoint path (.pt file) or run directory |
| `--save-interval` | 1 | Save every N iterations |
| `--progress-interval` | 30 | Print performance stats every N seconds |

#### Prioritized Experience Replay (PER)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--priority-exponent` | 0.0 | Priority exponent α (0=uniform/disabled, 0.6=recommended) |
| `--per-beta` | 0.4 | IS correction β (0=none, 0.4=moderate, 1.0=full). See operation manual Section 10q |

#### Visualization & Evaluation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--no-visualization` | False | Disable summary.html generation (enabled by default) |
| `--no-eval` | False | Skip evaluation before checkpoint (enabled by default) |
| `--live` | False | Enable LIVE web dashboard (real-time, requires flask) |
| `--dashboard-port` | 5000 | Port for live dashboard server |

### Architecture (AlphaZero Paper Standard)

The training script uses the optimal architecture:
- **Input channels**: 123 (full AlphaZero encoding with 4 castling planes)
- **Policy filters**: 2 (paper standard)
- **Value hidden**: 256 (paper standard)

---

## Recommended Configurations

### 1. Testing (~10-30 minutes)

For verifying the setup works:

```bash
uv run python alphazero-cpp/scripts/train.py \
    --iterations 2 \
    --games-per-iter 5 \
    --simulations 100 \
    --filters 64 \
    --blocks 3 \
    --train-batch 128
```

**Expected output:**
- Creates run directory: `checkpoints/f64-b3_YYYY-MM-DD_HH-MM-SS/`
- ~50 positions generated
- Training skipped (buffer too small)
- Checkpoints saved at `model_iter_001.pt`, `model_iter_002.pt`, `model_final.pt`
- `summary.html` generated with training metrics

### 2. Development (~2-4 hours, 8GB GPU)

For iterating on the codebase:

```bash
uv run python alphazero-cpp/scripts/train.py \
    --iterations 20 \
    --games-per-iter 25 \
    --simulations 400 \
    --filters 64 \
    --blocks 5 \
    --train-batch 256 \
    --buffer-size 50000
```

**Expected results:**
- Creates run directory: `checkpoints/f64-b5_YYYY-MM-DD_HH-MM-SS/`
- ~500 games generated
- Model should beat random player 80%+ (check `evaluation_results.json`)
- ~10-15 moves/sec during self-play
- `summary.html` updated on each checkpoint with loss curves and evaluation metrics

### 3. Development with Parallel Self-Play (~1-2 hours, 8GB GPU)

Same as above but 2-3x faster using cross-game batching:

```bash
uv run python alphazero-cpp/scripts/train.py \
    --iterations 20 \
    --games-per-iter 64 \
    --simulations 400 \
    --workers 16 \
    --filters 64 \
    --blocks 5 \
    --train-batch 256 \
    --buffer-size 50000
```

**Expected results:**
- ~1,280 games generated
- ~30-50 moves/sec (2-3x faster)
- Higher GPU utilization (~80%)

### 4. Serious Training (~12-24 hours, 12GB+ GPU)

For a reasonably strong model:

```bash
uv run python alphazero-cpp/scripts/train.py \
    --iterations 50 \
    --games-per-iter 50 \
    --simulations 800 \
    --filters 128 \
    --blocks 10 \
    --train-batch 512 \
    --buffer-size 200000
```

**Expected results:**
- Creates run directory: `checkpoints/f128-b10_YYYY-MM-DD_HH-MM-SS/`
- ~2,500 games generated
- Model should beat random 95%+ (tracked in `evaluation_results.json`)
- Basic tactical awareness (endgame puzzles 3-4/5)

### 5. Production (~24-48 hours, 16GB+ GPU)

For maximum strength:

```bash
uv run python alphazero-cpp/scripts/train.py \
    --iterations 100 \
    --games-per-iter 100 \
    --simulations 800 \
    --workers 16 \
    --filters 192 \
    --blocks 15 \
    --train-batch 1024 \
    --buffer-size 500000
```

**Expected results:**
- Creates run directory: `checkpoints/f192-b15_YYYY-MM-DD_HH-MM-SS/`
- ~10,000 games generated
- Strong tactical play
- ~1500-1800 Elo estimate
- `summary.html` tracks all training progress and evaluation metrics

---

## Key Parameters Explained

### Understanding the Training Pipeline

Each training iteration follows this flow:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING ITERATION                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. SELF-PLAY PHASE                                                      │
│     --workers, --games-per-iter, --temperature-moves                     │
│                                                                          │
│     For each move:                                                       │
│     ┌────────────────────────────────────────────────────────────────┐  │
│     │  2. MCTS SEARCH                                                 │  │
│     │     --simulations, --c-explore                                  │  │
│     │                                                                 │  │
│     │     Repeat until simulations complete:                          │  │
│     │     ┌──────────────────────────────────────────────────────┐   │  │
│     │     │  3. NEURAL NETWORK EVALUATION                        │   │  │
│     │     │     --eval-batch (auto-computed GPU batch)            │   │  │
│     │     │     --filters, --blocks (network architecture)       │   │  │
│     │     └──────────────────────────────────────────────────────┘   │  │
│     └────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│     Store results in REPLAY BUFFER (--buffer-size)                       │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  4. TRAINING PHASE                                                       │
│     --train-batch, --epochs, --lr                                        │
│     Sample from buffer, compute gradients, update network                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Network Size (`--filters`, `--blocks`)

| Config | Filters | Blocks | Parameters | VRAM | Use Case |
|--------|---------|--------|------------|------|----------|
| Tiny | 64 | 3 | ~0.3M | 2GB | Testing |
| Small | 64 | 5 | ~0.9M | 3GB | Development |
| Medium | 128 | 10 | ~4M | 6GB | Serious training |
| Large | 192 | 15 | ~11M | 10GB | Production |
| Paper | 256 | 19 | ~20M | 16GB+ | Research |

### MCTS Settings (`--simulations`)

| Simulations | Speed | Strength | Recommended For |
|-------------|-------|----------|-----------------|
| 100 | Fast | Weak | Testing, debugging |
| 400 | Medium | Decent | Development |
| 800 | Slow | Strong | Production (paper default) |
| 1600 | Very slow | Strongest | Evaluation only |

**Note:** The number of leaves evaluated per MCTS iteration (`search_batch`) is now auto-derived from the search algorithm: Gumbel mode uses `gumbel_top_k`, PUCT mode uses 1 (with virtual loss). This is no longer a user-facing parameter.

### Parallel Self-Play (`--workers`, `--eval-batch`)

When `--workers > 1`, parallel self-play batches NN evaluations **across games**:

```
Sequential (--workers 1):
  Game 1: MCTS → GPU(64) → MCTS → GPU(64) → ...
  GPU utilization: ~40% (waiting between calls)

Parallel (--workers 16):
  Game 1: MCTS → submit leaves ─┐
  Game 2: MCTS → submit leaves ─┤
  ...                           ├─→ GPU(512) → distribute results
  Game 16: MCTS → submit leaves─┘
  GPU utilization: ~85% (always full batches)
```

| Parameter | Effect |
|-----------|--------|
| `--workers 16` | 16 games run concurrently |
| `--eval-batch` | Auto-computed as workers x search_batch (rounded to 32); can override with a higher value |
| `--batch-timeout-ms 5` | Don't wait longer than 5ms for batch to fill |
| `--queue-capacity 8192` | Evaluation queue capacity (increase for many workers) |
| `--root-eval-retries 3` | Max retries for root NN eval timeout before giving up |
| `--worker-timeout-ms 2000` | Worker wait time for NN results (per attempt) |

### Parameter Optimization

This section explains how the parallel self-play parameters interact and how to size them for your hardware.

#### Auto-Computed `eval_batch`

`eval_batch` is now **automatically computed** as `workers x search_batch`, rounded up to the nearest multiple of 32. The `search_batch` value is itself auto-derived from the search algorithm (Gumbel uses `gumbel_top_k`, PUCT uses 1). You only need to set `--workers` and optionally `--eval-batch` (which acts as a floor -- the auto-computed value is raised to your override if it would be lower).

```
Auto formula: eval_batch = ceil(workers × search_batch / 32) × 32

Example (Gumbel, 16 workers, top_k=16):
  search_batch = 16 (auto from gumbel_top_k)
  eval_batch   = ceil(16 × 16 / 32) × 32 = 256

Example (Gumbel, 64 workers, top_k=16):
  search_batch = 16
  eval_batch   = ceil(64 × 16 / 32) × 32 = 1024
```

If you pass `--eval-batch 2048` but the auto-computed value is 1024, the system uses 2048 (your value is a floor). If you pass `--eval-batch 512` and the auto-computed value is 1024, the system uses 1024 (auto floor wins).

#### Sizing `eval-batch` by GPU Memory

`eval_batch` is ultimately limited by GPU VRAM. Larger batches amortize kernel launch overhead but require more memory. Use these as starting points and adjust based on actual memory usage:

| GPU VRAM | Small network | f192-b15 (default) |
|----------|--------------|-------------------|
| 8 GB | ≤ 2048 | 512–1024 |
| 12 GB | ≤ 4096 | 1024–2048 |
| 24 GB | ≤ 8192 | 2048–4096 |
| 40 GB+ (A100) | ≤ 12288 | 4096–8192 |

#### Simulations as a Hyperparameter

`--simulations` controls how many MCTS nodes are expanded per move. It's the primary quality-vs-quantity knob:

- **More simulations** → better move quality, slower games, fewer games/hour
- **Fewer simulations** → noisier moves, faster games, more games/hour

**Key insight for early training:** An untrained network benefits more from game quantity than move quality. Random weights produce random evaluations regardless of search depth, so more games (with fewer sims each) cover more of the state space faster. As the network improves, increase simulations to generate higher-quality training data.

**Interaction with the parallel pipeline:** Higher simulations means each move takes longer, so workers submit leaves at a steadier rate (less bursty). Lower simulations means moves finish quickly and workers cycle rapidly. Since `eval_batch` is now auto-computed from workers and search_batch, it adjusts automatically.

Typical progression: start with 400 sims, increase to 800–1600 as the network matures.

#### Note on `search_batch` (Auto-Derived)

`search_batch` (the number of leaves evaluated per MCTS iteration within a single game) is now automatically derived:
- **Gumbel mode**: `search_batch = gumbel_top_k` (default 16). Sequential Halving uses round-robin allocation, so the batch size matches the number of candidate actions.
- **PUCT mode**: `search_batch = 1`. Classic PUCT selects one leaf at a time with virtual loss.

This is no longer a user-facing parameter. GPU utilization in parallel mode comes from cross-game batching via `eval_batch`, not from within-game batching.

#### Sizing `workers`

`--workers` controls how many self-play games run concurrently. Since `eval_batch` is now auto-computed from workers, you only need to choose workers based on:

- **Must not exceed `--games-per-iter`** — excess workers sit idle
- **More workers = better GPU utilization** — more leaves available for batching
- **Memory:** each worker holds an MCTS tree in RAM (~50-200 MB depending on simulations), so 128 workers may need 10-25 GB of system RAM
- **Typical values**: 16 for 8GB GPUs, 64 for 12-24GB GPUs, 128 for 40GB+ GPUs

#### Quick Reference: Parallel Configurations

| GPU | workers | eval_batch (auto) | Notes |
|-----|---------|-------------------|-------|
| 8 GB | 16 | 256 | Gumbel default (16 x 16 = 256) |
| 12 GB | 64 | 1024 | Gumbel default (64 x 16 = 1024) |
| 24 GB | 64 | 1024 | Room to override `--eval-batch 2048` |
| 40 GB (A100) | 128 | 2048 | Gumbel default (128 x 16 = 2048) |

### Training Settings

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `--train-batch` | 256-1024 | Higher = faster but more VRAM |
| `--lr` | 0.001 | Adam optimizer default |
| `--epochs` | 5 | Training passes per iteration |
| `--buffer-size` | 100k-500k | Larger = more diverse samples |

**Understanding epochs and learning rate:**
- `--epochs` is the number of training passes **per iteration**, not over the entire dataset
- Each "epoch" samples a fresh batch from the replay buffer
- With Adam optimizer at lr=0.001, epochs=5 provides good stability
- Alternative configurations:
  - **Faster convergence**: `--lr 0.003 --epochs 3` (slightly less stable)
  - **More stable**: `--lr 0.0003 --epochs 10` (slower convergence)
- The AlphaZero paper uses SGD with lr=0.2→0.02→0.002, but requires millions of samples

### Root Search Algorithm (`--search-algorithm`)

Two root-level search algorithms are available. Non-root tree traversal always uses PUCT regardless.

**Gumbel Top-k Sequential Halving** (default: `--search-algorithm gumbel`):
- Replaces Dirichlet noise with **Gumbel noise** for theoretically principled exploration
- Uses **Sequential Halving** to efficiently allocate the simulation budget across candidate moves
- Produces **improved policy** training targets: `softmax(logit + sigma(Q_completed))` — contains search knowledge for all legal moves, not just visited ones
- Move selection during temperature phase uses the SH winner (a valid sample from the improved policy via the Gumbel-Max trick)

**Standard PUCT + Dirichlet** (`--search-algorithm puct`):
- Classic AlphaZero approach: Dirichlet noise on root priors, PUCT selection, visit-count policy targets
- Use for comparison/ablation or if you want the traditional algorithm

```
Gumbel (default):
  Root:  Gumbel noise + Sequential Halving → improved policy target
  Tree:  Standard PUCT (unchanged)
  Target: softmax(logit + sigma(Q_completed)) for ALL children

PUCT:
  Root:  Dirichlet noise + PUCT selection
  Tree:  Standard PUCT (unchanged)
  Target: visit_count / total_visits (temperature-applied)
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--gumbel-top-k` | 16 | How many candidate moves enter Sequential Halving. Higher = wider exploration, more sims needed |
| `--gumbel-c-visit` | 50.0 | Controls how much search Q-values influence the improved policy. Higher = search results matter more |
| `--gumbel-c-scale` | 1.0 | Scales the sigma() value transform. Usually left at 1.0 |

### Exploration (`--temperature-moves`, `--c-explore`)

- `--temperature-moves 30`: First 30 moves use temperature=1 (exploration), then greedy
- `--c-explore 1.5`: Exploration constant for PUCT tree traversal (higher = more exploration)
- In Gumbel mode, temperature phase uses the SH winner (stochastic); post-temperature uses argmax of improved policy (deterministic)

---

## Parameter Tuning Guide

This section explains how each parameter affects training and how to tune them for your hardware and goals.

### Note: `search_batch` is Auto-Derived

The `search_batch` value (leaves per MCTS iteration) is no longer a user parameter. It is automatically set based on the search algorithm:
- **Gumbel mode** (default): `search_batch = gumbel_top_k` (default 16)
- **PUCT mode**: `search_batch = 1`

GPU utilization in parallel mode is driven by `--workers` and the auto-computed `eval_batch`, not by within-game batching. To improve GPU utilization, increase `--workers`.

### Understanding `--simulations`

Controls MCTS search depth (quality vs speed tradeoff):

```
More simulations = Better moves but slower games
                 = More training data quality
                 = Fewer games per hour

Fewer simulations = Faster games but noisier moves
                  = Lower quality training data
                  = More games per hour
```

**When to adjust:**

| Goal | Simulations | Rationale |
|------|-------------|-----------|
| Fast iteration / debugging | 100-200 | Quick feedback, accept lower quality |
| Development | 400 | Good balance of speed and quality |
| Production training | 800 | AlphaZero paper default |
| Evaluation / playing | 1600 | Maximum strength for testing |

**Important insight:** Early training benefits more from **quantity** (more games with fewer sims) while later training benefits from **quality** (fewer games with more sims).

### Understanding `--games-per-iter`

Controls how much new data is generated before each training update:

```
More games/iter = More diverse training data
                = Longer time between weight updates
                = More stable but slower learning

Fewer games/iter = Faster weight updates
                 = More responsive to recent games
                 = Less stable, may overfit to recent patterns
```

**Tuning by training phase:**

| Phase | games-per-iter | Why |
|-------|----------------|-----|
| Early (iter 1-20) | 10-25 | Learn basic moves quickly |
| Middle (iter 20-50) | 25-50 | Build diverse experience |
| Late (iter 50+) | 50-100 | Refine with quality data |

### Understanding `--train-batch` (Training)

Controls how many positions are used per gradient update:

```
Larger train-batch = More stable gradients
                   = Better GPU utilization during training
                   = Requires more VRAM

Smaller train-batch = Noisier gradients (can help escape local minima)
                    = Less VRAM required
                    = Slower training (more gradient updates)
```

**Relationship with buffer size:**
- `train-batch` should be << `buffer-size` (at least 100x smaller)
- Example: buffer=100000 → train-batch up to 1024 is fine

| GPU VRAM | Recommended train-batch |
|----------|-------------------------|
| 4 GB | 128 |
| 8 GB | 256-512 |
| 12 GB | 512-1024 |
| 24 GB+ | 1024-2048 |

### Understanding `--buffer-size`

Controls replay buffer capacity (how many positions to remember):

```
Larger buffer = More diverse training samples
              = Prevents catastrophic forgetting
              = Uses more RAM

Smaller buffer = Focuses on recent games
               = Adapts faster to new strategies
               = May forget old lessons
```

**Tuning strategy:**
- **Minimum**: 10 × games_per_iter × avg_game_length (≈10 × 50 × 60 = 30,000)
- **Recommended**: 50-100 × games_per_iter × avg_game_length
- **Maximum**: Limited by system RAM (each position ≈ 32KB)

| System RAM | Max buffer-size |
|------------|-----------------|
| 8 GB | 100,000 |
| 16 GB | 250,000 |
| 32 GB | 500,000 |
| 64 GB+ | 1,000,000 |

### Understanding `--c-explore`

Controls exploration vs exploitation in MCTS tree traversal (PUCT formula):

```
Higher c_explore = More exploration (try new moves)
                 = Better for early training
                 = May waste simulations on bad moves

Lower c_explore  = More exploitation (trust the network)
                 = Better for late training / evaluation
                 = May miss good moves the network undervalues
```

| Value | Behavior | When to Use |
|-------|----------|-------------|
| 1.0 | Conservative | Evaluation, late training |
| 1.5 | **Balanced (default)** | Most training |
| 2.0 | Exploratory | Early training, stuck models |
| 2.5+ | Very exploratory | Debugging, testing |

### Understanding `--temperature-moves`

Controls when to switch from exploration to exploitation:

```
Move 1 to temperature_moves: Random sampling (temperature=1)
  → Encourages diverse openings
  → Creates varied training data

Move temperature_moves+1 onwards: Greedy selection (best move)
  → Plays strongest moves
  → Realistic game conclusions
```

| Value | Effect |
|-------|--------|
| 15 | Short exploration phase (more deterministic games) |
| 30 | **Default** - Good balance for chess |
| 50 | Long exploration (very diverse but less realistic endgames) |

### Understanding `--search-algorithm` (Gumbel vs PUCT)

The root search algorithm determines how exploration noise is added, how the simulation budget is allocated across root children, and what policy targets are produced for training.

**Gumbel Top-k Sequential Halving** (default) implements the algorithm from "Policy improvement by planning with Gumbel" (Danihelka et al., 2022). It works in three stages:

```
1. INIT:    Sample Gumbel(0,1) noise g(a) for each legal action a
            Compute logit(a) = log(prior(a))
            Select top-m actions by logit(a) + g(a)

2. HALVING: For ceil(log2(m)) phases:
              Allocate sims evenly across active actions (round-robin)
              Score: logit(a) + g(a) + sigma(Q_completed(a))
              Prune bottom half

3. OUTPUT:  Training target: softmax(logit(a) + sigma(Q_completed(a))) for ALL children
            Move selection: SH winner (temperature phase) or argmax improved policy
```

Key concepts:
- **Completed Q**: `Q_hat(a) = N/(1+N) * Q(a) + 1/(1+N) * V(root)` — smoothly interpolates between the NN value prior (unvisited) and MCTS Q-value (visited)
- **sigma()**: `(c_visit + max_visits) * c_scale * q_normalized` — transforms Q-values to the same scale as logits
- **Round-robin**: Instead of PUCT selection at the root, Gumbel mode cycles through active actions. This eliminates root collisions without virtual loss (which would bias Q-values)

**When to use which:**

| Scenario | Recommendation |
|----------|---------------|
| New training run | Gumbel (default) — better policy targets, principled exploration |
| A/B comparison | Run identical configs with `--search-algorithm gumbel` vs `puct` |
| Resuming old run | Use `--search-algorithm puct` if the model was trained with PUCT |
| Very few simulations (<50) | Gumbel still works but SH phases become very short |
| Evaluation / play | Either works; Gumbel is slightly more efficient with small budgets |

**Tuning `--gumbel-top-k`:**

| top_k | Budget allocation | Best for |
|-------|-------------------|----------|
| 8 | Concentrated (fewer candidates, more sims each) | Low sim budgets (100-200) |
| **16** | **Balanced (default)** | Most training scenarios |
| 32 | Wide (many candidates, fewer sims each) | High sim budgets (800+), opening diversity |

Rule of thumb: `top_k` should be ≤ simulations / 4 so each action gets multiple simulations per phase.

### Parameter Combinations by Goal

**Goal: Maximum Training Speed (development)**
```bash
--simulations 200 \
--games-per-iter 10 \
--train-batch 256 \
--epochs 3
```

**Goal: Maximum Training Speed with Parallel (2-5x faster)**
```bash
--simulations 400 \
--workers 16 \
--games-per-iter 64 \
--train-batch 256
```

**Goal: Maximum Model Quality (production)**
```bash
--simulations 800 \
--games-per-iter 100 \
--train-batch 1024 \
--epochs 5 \
--buffer-size 500000
```

**Goal: Low VRAM (4-6 GB GPU)**
```bash
--simulations 400 \
--filters 64 \
--blocks 5 \
--train-batch 128 \
--buffer-size 50000
```

**Goal: Debugging / Testing**
```bash
--simulations 50 \
--games-per-iter 2 \
--iterations 1
```

### Monitoring and Adjusting

Watch these metrics during training:

| Metric | Healthy Range | If Too Low | If Too High |
|--------|---------------|------------|-------------|
| moves/sec | 10-50 | Increase --workers for better batching | Expected with larger networks |
| GPU util | 60-90% | Use --workers 16 or more | Expected (good!) |
| Loss | Decreasing | Normal | Check if stuck (reduce lr) |
| policy_loss | 2.0-5.0 | Good convergence | May need more exploration |
| value_loss | 0.1-0.5 | Good convergence | May need more games |

---

## Resuming Training

Resume training from either a **checkpoint file** or a **run directory**:

```bash
# Resume from a specific checkpoint file
uv run python alphazero-cpp/scripts/train.py \
    --resume checkpoints/f192-b15_2024-02-03_14-30-00/model_iter_050.pt \
    --iterations 100

# Resume from a run directory (finds latest checkpoint automatically)
uv run python alphazero-cpp/scripts/train.py \
    --resume checkpoints/f192-b15_2024-02-03_14-30-00 \
    --iterations 100
```

This will:
1. Load the network weights and optimizer state
2. Start from the next iteration (e.g., iteration 51)
3. Load `training_log.jsonl` and append new metrics
4. Load `evaluation_results.json` and append new evaluations
5. Load replay buffer from `replay_buffer.rpbf`
6. Continue saving to the same run directory


## Graceful Shutdown (Ctrl+C)

The training script supports **graceful shutdown**. When you press Ctrl+C:

1. **Finishes the current game** (doesn't interrupt mid-game)
2. **Saves an emergency checkpoint** to `{run_dir}/model_iter_XXX_emergency.pt`
3. **Saves the replay buffer** to `replay_buffer.rpbf`
4. **Saves training metrics** to `training_log.jsonl`
5. **Prints resume instructions**

```bash
# Example: Press Ctrl+C during training
# Output:
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ! SHUTDOWN REQUESTED - Finishing current game and saving...
# ! Press Ctrl+C again to force quit (may lose data)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# To resume after graceful shutdown (use run directory):
uv run python alphazero-cpp/scripts/train.py \
    --resume checkpoints/f192-b15_2024-02-03_14-30-00 \
    --iterations 100
```

**Note:** Press Ctrl+C **twice** to force quit immediately (may lose unsaved progress).

---

## Progress Monitoring

The training script prints **live performance statistics** every 30 seconds (configurable with `--progress-interval`):

```
    ⏱ 30s | Games: 5/50 | Moves: 18.3/s | Sims: 14,640/s | NN: 1,830/s | Buffer: 312 | ETA: 0:04:30
    ⏱ 60s | Games: 11/50 | Moves: 19.1/s | Sims: 15,280/s | NN: 1,910/s | Buffer: 687 | ETA: 0:03:25
```

**What each metric means:**

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `⏱ Xs` | Elapsed time since iteration started | - |
| `Games: X/Y` | Games completed / total games this iteration | - |
| `Moves: X/s` | Moves processed per second | 10-50 |
| `Sims: X/s` | MCTS simulations per second | 5,000-20,000 |
| `NN: X/s` | Neural network evaluations per second | 500-5,000 |
| `Buffer: X` | Current replay buffer size | Growing |
| `ETA` | Estimated time remaining for this iteration | - |

**Customizing the interval:**
```bash
# More frequent updates (every 10 seconds)
--progress-interval 10

# Less frequent updates (every 60 seconds)
--progress-interval 60

# Disable (set very high)
--progress-interval 9999
```

---

## Training Outputs & Dashboards

### Default Output: summary.html

By default, a **summary.html** file is generated in the run directory on every checkpoint save:

```bash
# Default behavior - summary.html is generated automatically
uv run python alphazero-cpp/scripts/train.py \
    --iterations 50 \
    --games-per-iter 25

# Output: checkpoints/f192-b15_YYYY-MM-DD_HH-MM-SS/summary.html
```

The summary includes:
- Training loss curves (total, policy, value)
- Evaluation metrics (vs_random win rate, endgame puzzle scores)
- Configuration summary
- Performance statistics

**To disable summary.html generation:**
```bash
--no-visualization
```

### Option: Live Web Dashboard (`--live`)

For real-time monitoring with **instant WebSocket updates**:

```bash
uv run python alphazero-cpp/scripts/train.py \
    --iterations 50 \
    --games-per-iter 25 \
    --live
```

**Features:**
- Opens automatically in your browser
- Updates instantly after each iteration (no refresh needed)
- Interactive Plotly.js charts with zoom/pan
- Shows connection status and auto-reconnects
- Beautiful dark theme with responsive design

**Requirements:**
```bash
pip install flask flask-socketio
```

**Custom port:**
```bash
--live --dashboard-port 8080
```

### Using Both Together

You can enable both outputs simultaneously:

```bash
uv run python alphazero-cpp/scripts/train.py \
    --iterations 50 \
    --live
```

This gives you:
- Real-time updates in the browser (`--live`)
- Saved summary.html in the run directory (default)

### Automatic Evaluation

**Evaluation runs automatically before each checkpoint save** (disable with `--no-eval`):

| Evaluation | Description | Time |
|------------|-------------|------|
| vs Random | 5 games against random opponent | ~30-60 sec |
| Endgame Puzzles | 5 tactical positions | ~10 sec |

Results are saved to `evaluation_results.json` and displayed in `summary.html`.

### Generated Files

After training, you'll find in the run directory:

```
checkpoints/f192-b15_YYYY-MM-DD_HH-MM-SS/
├── model_iter_001.pt         # Checkpoint at iteration 1
├── model_iter_005.pt         # Checkpoint at iteration 5
├── model_final.pt            # Final checkpoint (same as last iteration)
├── training_log.jsonl     # Per-iteration metrics (loss, games, etc.)
├── evaluation_results.json   # Evaluation metrics per checkpoint
├── summary.html              # Visual summary with charts
└── replay_buffer.rpbf        # Replay buffer (saved after each self-play phase)
```

### Viewing the Summary

```bash
# On Windows
start checkpoints/f192-b15_YYYY-MM-DD_HH-MM-SS/summary.html

# On macOS
open checkpoints/f192-b15_YYYY-MM-DD_HH-MM-SS/summary.html

# On Linux
xdg-open checkpoints/f192-b15_YYYY-MM-DD_HH-MM-SS/summary.html
```

---

## Evaluation Script Reference

**Location**: `alphazero/scripts/evaluate.py`

**Note:** Basic evaluation (vs_random + endgame puzzles) now runs automatically before each checkpoint save. Results are stored in `evaluation_results.json` in the run directory.

### Quick Evaluation Examples

```bash
# Against random player
uv run python alphazero/scripts/evaluate.py \
    --checkpoint checkpoints/f192-b15_2024-02-03_14-30-00/model_final.pt \
    --opponent random \
    --games 50 \
    --simulations 200

# Against endgame positions
uv run python alphazero/scripts/evaluate.py \
    --checkpoint checkpoints/f192-b15_2024-02-03_14-30-00/model_final.pt \
    --opponent endgame \
    --simulations 400

# Against Stockfish (if installed)
uv run python alphazero/scripts/evaluate.py \
    --checkpoint checkpoints/f192-b15_2024-02-03_14-30-00/model_final.pt \
    --opponent stockfish \
    --stockfish-path /path/to/stockfish \
    --stockfish-elo 1500 \
    --games 20
```

### Evaluation Metrics

| Opponent | What it Tests | Expected @ 50 iters |
|----------|---------------|---------------------|
| Random | Basic play | 90%+ win rate |
| Endgame | Tactical ability | 60-80% score |
| Stockfish 1000 | Beginner level | 30-50% score |
| Stockfish 1500 | Intermediate | 10-30% score |

---

## Web Interface

Play against your trained model:

```bash
# Using a trained checkpoint
uv run python web/run.py \
    --checkpoint checkpoints/f192-b15_2024-02-03_14-30-00/model_final.pt \
    --simulations 400 \
    --port 5000
```

Open http://localhost:5000 in your browser.

---

## Performance Expectations

### Training Speed (RTX 4060 Laptop GPU)

| Network | Moves/sec | Games/hour | Iteration time |
|---------|-----------|------------|----------------|
| 64×5 | 30-50 | 50-80 | 5-10 min |
| 128×10 | 15-25 | 25-40 | 15-25 min |
| 192×15 | 8-15 | 15-25 | 30-50 min |

### Memory Usage

| Network | GPU VRAM | System RAM |
|---------|----------|------------|
| 64×5 | 2-3 GB | 4 GB |
| 128×10 | 4-6 GB | 8 GB |
| 192×15 | 8-12 GB | 12 GB |

---

## Troubleshooting

### Out of Memory (OOM)

Reduce these parameters:
1. `--train-batch` (e.g., 128 instead of 256)
2. `--workers` (fewer workers = smaller auto-computed eval_batch)
3. `--eval-batch` (override with a lower value if needed)
4. Network size (`--filters`, `--blocks`)

### Training Too Slow

1. Use parallel self-play: `--workers 16`
2. Use smaller network for development
3. Reduce `--simulations` (400 instead of 800)
4. Ensure CUDA is being used (`--device cuda`)

### Low GPU Utilization

1. Enable parallel self-play: `--workers 16` (eval_batch auto-computed)
2. Increase `--workers` for better cross-game batching
3. Increase `--train-batch` during training phase

### High MCTS Timeout Evals (Stale Results & Root Retries)

When workers wait for GPU results and time out (`worker_timeout_ms` exceeded), two problems arise:

1. **Root eval timeout**: The root position gets no NN evaluation → MCTS has zero visits → uniform random move (bad training data).
2. **Stale result corruption**: A timed-out request may still be processing on the GPU. When it eventually completes, the result arrives in the worker's queue. On the *next* evaluation, this stale result gets consumed as if it were for the new position — silently corrupting the search tree with a policy/value computed for the wrong board state.

**Built-in protections (v2.1):**

- **Root eval retry loop** (`--root-eval-retries 3`): If root evaluation times out, the worker flushes stale results and retries up to 3 more times. This reduces root failure probability from P to P⁴ (e.g., 1% → 0.000001%).
- **Generation-based stale filtering**: Each worker has a monotonic generation counter. On flush, the generation increments. All new requests carry the current generation. `get_results()` silently discards any result with a stale (older) generation, closing the race window between flush and result arrival.

**Diagnosing issues via the live dashboard:**

| Metric | Meaning |
|--------|---------|
| `root_retries` | Times root eval was retried after timeout. If high but `mcts_failures` is low, the retry mechanism is working. |
| `stale_results_flushed` | Stale results discarded. Non-zero means timeouts occurred but corruption was prevented. |
| `mcts_failures` (Timeout Evals) | Moves where MCTS returned zero visits (all retries exhausted). Should be near zero. |

**If `mcts_failures` is high:**

1. Increase `--worker-timeout-ms` (e.g., 4000 instead of 2000) — gives more time for GPU
2. Increase `--root-eval-retries` (e.g., 5 instead of 3) — more retry attempts
3. Reduce `--workers` — less GPU contention
4. Increase `--queue-capacity` (e.g., 16384) — prevents observation pool exhaustion
5. Reduce `--eval-batch` — smaller batches complete faster

### Model Not Improving

1. Increase `--games-per-iter` (more training data)
2. Increase `--buffer-size` (more diverse samples)
3. Train for more iterations
4. Check that loss is decreasing

---

## Checkpoint Compatibility

Checkpoints from `alphazero-cpp/scripts/train.py` use:
- **123-channel input** (C++ encoding with 4 binary castling planes)
- **AlphaZero paper architecture** (2 policy filters, 256 value hidden)
- **Compatible with**: Web interface, evaluation script, Python MCTS

The checkpoint format includes:
```python
{
    'iteration': int,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'config': {
        'input_channels': 123,
        'num_filters': int,
        'num_blocks': int,
        'num_actions': 4672,
        'policy_filters': 2,
        'value_filters': 1,
        'value_hidden': 256,
        'wdl': True,
        'se_reduction': int,
    },
    'backend': 'cpp',
    'version': '2.0'
}
```

---

## Quick Reference Card

```bash
# Test (10 min)
train.py --iterations 2 --games-per-iter 5 --simulations 100 --filters 64 --blocks 3

# Dev Sequential (2-4 hours)
train.py --iterations 20 --games-per-iter 25 --simulations 400 --filters 64 --blocks 5

# Dev Parallel (1-2 hours, 2-3x faster)
train.py --iterations 20 --games-per-iter 64 --workers 16 --filters 64 --blocks 5

# Serious (12-24 hours)
train.py --iterations 50 --games-per-iter 50 --simulations 800 --filters 128 --blocks 10

# Production (24-48 hours)
train.py --iterations 100 --games-per-iter 100 --workers 16 --filters 192 --blocks 15

# Resume from run directory
train.py --resume checkpoints/f192-b15_2024-02-03_14-30-00 --iterations 100

# With buffer persistence
train.py --iterations 50 --buffer-persistence

# Use classic PUCT instead of Gumbel (A/B testing)
train.py --iterations 50 --search-algorithm puct

# Disable evaluation and visualization (faster checkpointing)
train.py --iterations 50 --no-eval --no-visualization

# Evaluate
evaluate.py --checkpoint checkpoints/f192-b15_.../model_final.pt --opponent random --games 50
```

### Parameter Quick Reference

| When Used | Parameter | Default | Purpose |
|-----------|-----------|---------|---------|
| Root Search | `--search-algorithm` | gumbel | `gumbel` (SH) or `puct` (Dirichlet) |
| Gumbel | `--gumbel-top-k` | 16 | Candidate actions for Sequential Halving |
| Parallel Eval | `--eval-batch` | auto | Auto-computed GPU batch size (workers x search_batch, rounded to 32) |
| Training | `--train-batch` | 256 | Samples per gradient step |

### New in v3.1

| Feature | Default | Description |
|---------|---------|-------------|
| Prioritized Experience Replay | `--priority-exponent 0` (disabled) | Loss-proportional sampling with IS correction. Enable with `--priority-exponent 0.6` |
| IS correction | `--per-beta 0.4` | Fixed importance sampling correction (0=none, 1=full). Tunable per phase |
| Per-sample loss | Enabled when PER active | Training computes per-sample policy+value loss for priority updates |
| RPBF v3 priorities | Auto | Buffer files include priority section when PER enabled |

### New in v3.0

| Feature | Default | Description |
|---------|---------|-------------|
| Gumbel Top-k SH | `--search-algorithm gumbel` | Gumbel MuZero root search replaces Dirichlet + PUCT at root (default) |
| Sequential Halving | `--gumbel-top-k 16` | Efficient sim budget allocation via halving candidate set |
| Improved policy targets | Enabled | `softmax(logit + sigma(Q_completed))` training targets with theoretical guarantees |
| PUCT fallback | `--search-algorithm puct` | Original AlphaZero algorithm still available for A/B testing |
| Gumbel tuning | 3 params | `--gumbel-c-visit`, `--gumbel-c-scale`, `--gumbel-top-k` |

### New in v2.1

| Feature | Default | Description |
|---------|---------|-------------|
| Root eval retry | 3 retries | Auto-retry root NN eval on timeout (`--root-eval-retries`) |
| Stale result filtering | Enabled | Generation-based filtering prevents corrupted evaluations |
| Queue capacity tuning | 8192 | Configurable from Python (`--queue-capacity`) |
| Dashboard: root retries | Shown | Pipeline health shows root retry and stale flush counts |

### New in v2.0

| Feature | Default | Description |
|---------|---------|-------------|
| Organized output | Enabled | Run directory: `f{filters}-b{blocks}_{timestamp}/` |
| summary.html | Enabled | Visual training summary (disable: `--no-visualization`) |
| Evaluation | Enabled | vs_random + endgame puzzles (disable: `--no-eval`) |
| Buffer persistence | Disabled | Fresh buffer each run (enable: `--buffer-persistence`) |

---

**Last Updated**: 2026-02-18


│ GPU Batch Collection & Evaluation Pipeline Redesign                                                                           │
│                                                                                                                               │
│ Goal                                                                                                                          │
│                                                                                                                               │
│ Reduce lock contention in GlobalEvaluationQueue by ~200x, achieving near-100% GPU utilization. The core technique: separate   │
│ lock-protected metadata from lock-free data writes.                                                                           │
│                                                                                                                               │
│ Key Decision: CUDA Graph Padding vs Variable Batches                                                                          │
│                                                                                                                               │
│ Recommendation: Variable-size batches (no padding). Here's why:                                                               │
│ ┌──────────────────────────┬────────────────────────────────┬─────────────────────────────┐                                   │
│ │          Factor          │      CUDA Graph + Padding      │      Variable Batches       │                                   │
│ ├──────────────────────────┼────────────────────────────────┼─────────────────────────────┤                                   │
│ │ GPU compute (batch 1024) │ ~12ms                          │ ~10ms for 820 leaves        │                                   │
│ ├──────────────────────────┼────────────────────────────────┼─────────────────────────────┤                                   │
│ │ CPU overhead savings     │ ~0.05ms (CUDA Graph replay)    │ N/A                         │                                   │
│ ├──────────────────────────┼────────────────────────────────┼─────────────────────────────┤                                   │
│ │ Padding waste (820/1024) │ +2-3ms wasted GPU on zeros     │ 0                           │                                   │
│ ├──────────────────────────┼────────────────────────────────┼─────────────────────────────┤                                   │
│ │ Net per-batch            │ ~12.05ms for 820 useful leaves │ ~10ms for 820 useful leaves │                                   │
│ └──────────────────────────┴────────────────────────────────┴─────────────────────────────┘                                   │
│ CUDA Graph saves <1% CPU overhead but padding wastes 20%+ GPU compute. The existing dual-path (CUDA Graph when exact match,   │
│ regular forward otherwise) is already optimal. No changes to the Python evaluator's CUDA Graph logic.                         │
│                                                                                                                               │
│ Current Bottlenecks (ranked)                                                                                                  │
│                                                                                                                               │
│ 1. collect_batch staging-to-batch copy under lock: ~1.7ms - 51MB memcpy blocks all workers                                    │
│ 2. submit_for_evaluation memcpy under lock: ~80us/worker - 64 workers = 5.12ms serialized                                     │
│ 3. NHWC-to-NCHW transpose: ~0.5ms - Serial, cache-unfriendly (already outside lock)                                           │
│ 4. Total lock-held time per batch cycle: ~6.8ms (in a ~12ms GPU compute cycle)                                                │
│                                                                                                                               │
│ Design Overview                                                                                                               │
│                                                                                                                               │
│ submit_for_evaluation (workers)                                                                                               │
│                                                                                                                               │
│ 1. Under lock (~1us): claim staging slots + push metadata                                                                     │
│ 2. No lock: memcpy obs/mask data to claimed slots                                                                             │
│ 3. No lock: set per-slot atomic slot_ready_ flags, notify_one()                                                               │
│                                                                                                                               │
│ collect_batch (GPU thread)                                                                                                    │
│                                                                                                                               │
│ 1. Under lock (~10us): deferred compaction (from prev batch), wait for requests (stall detection - unchanged), take metadata  │
│ 2. No lock: spin-wait on slot_ready_ per slot, fused NHWC-to-NCHW transpose directly from staging (skip intermediate buffer), │
│  OpenMP parallel                                                                                                              │
│                                                                                                                               │
│ Files to Modify                                                                                                               │
│                                                                                                                               │
│ 1. alphazero-cpp/include/selfplay/evaluation_queue.hpp                                                                        │
│                                                                                                                               │
│ - Add std::vector<std::atomic<uint8_t>> slot_ready_ member (per-slot ready flags)                                             │
│ - Add bool needs_compaction_{false} member (deferred compaction)                                                              │
│ - Add void compact_remaining_locked() private method                                                                          │
│ - Remove float* batch_obs_buffer_ (NHWC intermediate - no longer needed)                                                      │
│ - Add #include <thread> for std::this_thread::yield() fallback                                                                │
│                                                                                                                               │
│ 2. alphazero-cpp/src/selfplay/evaluation_queue.cpp (primary changes)                                                          │
│                                                                                                                               │
│ Constructor: Init slot_ready_.resize(queue_capacity_) to zero                                                                 │
│                                                                                                                               │
│ submit_for_evaluation() - Split into 3 phases:                                                                                │
│ Phase 1 (lock ~1us): claim slots via staging_write_head_ increment, push StagedRequest metadata                               │
│ Phase 2 (no lock):   memcpy obs + mask to claimed slots (each worker owns its slots exclusively)                              │
│ Phase 3 (no lock):   slot_ready_[slot].store(1, release) for each slot, then notify_one()                                     │
│                                                                                                                               │
│ collect_batch() - Split into 2 phases:                                                                                        │
│ Phase 1 (lock ~10us):                                                                                                         │
│   - Deferred compaction from previous batch (if needs_compaction_)                                                            │
│   - Two-phase stall detection wait (UNCHANGED - it works well)                                                                │
│   - Take batch metadata + save staging slot indices                                                                           │
│   - Erase taken requests, set needs_compaction_ flag if head > 75%                                                            │
│                                                                                                                               │
│ Phase 2 (no lock, OpenMP parallel):                                                                                           │
│   - For each batch entry:                                                                                                     │
│     - Spin-wait on slot_ready_[slot] (typically instant - worker finished long ago)                                           │
│     - Fused NHWC->NCHW transpose directly from staging_obs -> batch_obs_nchw                                                  │
│     - memcpy mask from staging_mask -> batch_mask                                                                             │
│     - Clear slot_ready_[slot] = 0                                                                                             │
│                                                                                                                               │
│ New compact_remaining_locked(): Extracted from current inline compaction logic. Moves remaining staged request data +         │
│ metadata to front of buffer.                                                                                                  │
│                                                                                                                               │
│ allocate_batch_buffers(): Remove batch_obs_buffer_ allocation (saves 32MB)                                                    │
│                                                                                                                               │
│ free_batch_buffers(): Remove batch_obs_buffer_ cleanup                                                                        │
│                                                                                                                               │
│ reset(): Clear all slot_ready_ flags, reset needs_compaction_                                                                 │
│                                                                                                                               │
│ 3. alphazero-cpp/CMakeLists.txt                                                                                               │
│                                                                                                                               │
│ - Add explicit OpenMP::OpenMP_CXX to selfplay target (1 line)                                                                 │
│                                                                                                                               │
│ 4. alphazero-cpp/scripts/train.py (minor)                                                                                     │
│                                                                                                                               │
│ - Add 5-line eval_batch auto-tune suggestion near line 2108:                                                                  │
│ suggested = (workers * search_batch rounded up to 32)                                                                         │
│                                                                                                                               │
│ Safety Arguments                                                                                                              │
│ Concern: Worker writes data after GPU takes metadata                                                                          │
│ Why it's safe: GPU spin-waits on slot_ready_ before reading; release/acquire semantics guarantee visibility                   │
│ ────────────────────────────────────────                                                                                      │
│ Concern: Two workers claim overlapping slots                                                                                  │
│ Why it's safe: Slot claiming (staging_write_head_++) is under queue_mutex_ - serialized                                       │
│ ────────────────────────────────────────                                                                                      │
│ Concern: Compaction overwrites in-progress slots                                                                              │
│ Why it's safe: Compaction is deferred to NEXT collect_batch, by which time all previous slots are consumed                    │
│ ────────────────────────────────────────                                                                                      │
│ Concern: OpenMP conflicts with worker threads                                                                                 │
│ Why it's safe: Workers are blocked on get_results() during GPU's parallel transpose                                           │
│ What NOT to Change                                                                                                            │
│                                                                                                                               │
│ - Batch firing strategy: Keep two-phase stall detection (adaptive, works well)                                                │
│ - CUDA Graph logic in train.py: Keep dual-path (exact match -> graph, else -> eager)                                          │
│ - Position encoder format: Keep NHWC (transpose is now parallelized + amortized)                                              │
│ - Double-buffered policy results: Working correctly as-is                                                                     │
│ - Per-worker result queues: No changes needed                                                                                 │
│ - submit_results(): No changes needed                                                                                         │
│ - Python bindings: References batch_obs_nchw_buffer_ and batch_mask_buffer_ (both unchanged)                                  │
│                                                                                                                               │
│ Expected Impact                                                                                                               │
│ ┌─────────────────────────────┬──────────────────────┬──────────────────────────┐                                             │
│ │           Metric            │        Before        │          After           │                                             │
│ ├─────────────────────────────┼──────────────────────┼──────────────────────────┤                                             │
│ │ submit lock hold            │ ~80us/worker         │ ~1us/worker              │                                             │
│ ├─────────────────────────────┼──────────────────────┼──────────────────────────┤                                             │
│ │ collect lock hold           │ ~1.7ms/batch         │ ~10us/batch              │                                             │
│ ├─────────────────────────────┼──────────────────────┼──────────────────────────┤                                             │
│ │ Total lock time/cycle       │ ~6.8ms               │ ~0.07ms                  │                                             │
│ ├─────────────────────────────┼──────────────────────┼──────────────────────────┤                                             │
│ │ Worker contention           │ 42% lock utilization │ <1%                      │                                             │
│ ├─────────────────────────────┼──────────────────────┼──────────────────────────┤                                             │
│ │ Transpose                   │ ~0.5ms serial        │ ~0.15ms OpenMP           │                                             │
│ ├─────────────────────────────┼──────────────────────┼──────────────────────────┤                                             │
│ │ Memory saved                │ -                    │ 32MB (batch_obs_buffer_) │                                             │
│ ├─────────────────────────────┼──────────────────────┼──────────────────────────┤                                             │
│ │ Expected NN throughput gain │ -                    │ 30-60%                   │                                             │
│ └─────────────────────────────┴──────────────────────┴──────────────────────────┘                                             │
│ Verification                                                                                                                  │
│                                                                                                                               │
│ 1. Build on Windows (MSVC) and Linux (gcc) - both must compile clean                                                          │
│ 2. Run test_eval_queue_staleness.py (existing test)                                                                           │
│ 3. Small config: --workers 4 --games-per-iter 8 --simulations 100 --eval-batch 64                                             │
│ 4. Production config: --workers 64 --simulations 800 --eval-batch 1024                                                        │
│ 5. Check: no pool_exhaustion, no submission_drops, avg_batch near eval_batch                                                  │
│ 6. Compare NN evals/s and batch fill ratio vs before                                                                          │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

● I'll implement this GPU batch collection pipeline redesign. Let me start by reading the current source files to understand the
  exact code I'll be modifying.

● Read 3 files (ctrl+o to expand)

● Now let me also read the train.py section around line 2108 to find where to add the eval_batch suggestion:

● Read 1 file (ctrl+o to expand)

● Now I have a thorough understanding of all the code. Let me create tasks and begin implementing.

● Now let's start with Task 1 — the header file changes:

● ★ Insight ─────────────────────────────────────
  Why std::vector<std::atomic<uint8_t>> for slot readiness?
  - Each worker writes data to its claimed staging slots without holding the lock. The GPU thread needs to know when the data is
  actually written. A per-slot atomic flag with release/acquire semantics provides this guarantee without any lock.
  - std::atomic<uint8_t> is only 1 byte per slot (vs 4 for atomic<int>), keeping the flag array cache-friendly. With 8192 slots,
  that's only 8KB.
  - The release store in the worker ensures the preceding memcpy is visible before the flag becomes 1. The acquire load in the
  GPU thread ensures it sees the complete data.
  ─────────────────────────────────────────────────

  Let me apply the header changes:

● Update(alphazero-cpp\include\selfplay\evaluation_queue.hpp)
  ⎿  Added 1 line
       9  #include <cstring>
      10  #include <cstdlib>
      11  #include <tuple>
      12 +#include <thread>
      13
      14  namespace selfplay {
      15

● Now remove batch_obs_buffer_ and add the new members:

● Update(alphazero-cpp\include\selfplay\evaluation_queue.hpp)
  ⎿  Added 16 lines, removed 1 line
      229      void allocate_batch_buffers();
      230      void free_batch_buffers();
      231
      232 +    // Compact remaining staged requests to front of staging buffers (called under lock)
      233 +    void compact_remaining_locked();
      234 +
      235      // Configuration
      236      size_t max_batch_size_;
      237      size_t queue_capacity_;
     ...
      242      float* staging_mask_buffer_;   // queue_capacity × POLICY_SIZE
      243      std::atomic<int32_t> staging_write_head_{0};
      244
      245 +    // Per-slot ready flags: workers set after memcpy completes (lock-free signaling)
      246 +    // GPU thread spin-waits on these before reading staging data
      247 +    std::vector<std::atomic<uint8_t>> slot_ready_;  // queue_capacity entries
      248 +
      249      // Request metadata queue (lightweight — only metadata, no observation data)
      250      std::vector<StagedRequest> staged_requests_;
      251      std::mutex queue_mutex_;
      252      std::condition_variable queue_cv_;
      253
      254 +    // Deferred compaction flag: set when staging head > 75% capacity,
      255 +    // executed at start of NEXT collect_batch (after all previous slots consumed)
      256 +    bool needs_compaction_{false};
      257 +
      258      // Batch buffers (aligned/pinned for GPU)
      248 -    float* batch_obs_buffer_;      // max_batch_size x OBS_SIZE (NHWC staging)
      259 +    // NOTE: batch_obs_buffer_ (NHWC intermediate) removed — fused transpose
      260 +    // reads directly from staging_obs_buffer_ into batch_obs_nchw_buffer_
      261      float* batch_obs_nchw_buffer_; // max_batch_size x OBS_SIZE (NCHW for GPU)
      262      float* batch_mask_buffer_;     // max_batch_size x POLICY_SIZE
      263      float* batch_policy_buffers_[2]; // Double-buffered: GPU writes [current], workers read [previous]
      264      float* batch_value_buffer_;    // max_batch_size
      265      int current_policy_buffer_{0}; // Which buffer GPU writes to next (only GPU thread writes; ordering via wor
           ker_cvs_ notify/wait)
      266
      267 +    // Staging slot indices for current batch (used for lock-free Phase 2 reads)
      268 +    std::vector<int32_t> batch_staging_slots_;
      269 +
      270      // Mapping for current batch (worker_id, request_id, generation) tuples
      271      std::vector<std::tuple<int32_t, int32_t, uint32_t>> batch_mapping_;
      272

● Now the big one — Task 2, the .cpp rewrite:

● ★ Insight ─────────────────────────────────────
  The key insight of this redesign: separating metadata from data
  - Previously, the entire submit_for_evaluation (slot claim + memcpy ~80µs per worker) was under the lock, serializing all 64
  workers.
  - Now: only the slot claim (~1µs) is locked. The actual data write happens lock-free because each worker has exclusive
  ownership of its claimed slots.
  - The GPU thread uses a similar pattern: metadata extraction under lock (~10µs), then spin-waits on slot_ready_ flags to safely
   read the data without any lock.
  - This is a classic "publish-subscribe with ownership transfer" pattern — the mutex protects the metadata handshake, atomics
  protect the data visibility.
  ─────────────────────────────────────────────────

  Let me rewrite the .cpp file. I'll apply the changes in order:

  1. Constructor — initialize slot_ready_, remove batch_obs_buffer_:

● Update(alphazero-cpp\src\selfplay\evaluation_queue.cpp)
  ⎿  Added 12 lines, removed 2 lines
       2  #include <algorithm>
       3  #include <stdexcept>
       4
       5 +#ifdef _OPENMP
       6 +#include <omp.h>
       7 +#endif
       8 +
       9  namespace selfplay {
      10
      11  // ============================================================================
     ...
      17      , queue_capacity_(queue_capacity)
      18      , staging_obs_buffer_(nullptr)
      19      , staging_mask_buffer_(nullptr)
      16 -    , batch_obs_buffer_(nullptr)
      20      , batch_obs_nchw_buffer_(nullptr)
      21      , batch_mask_buffer_(nullptr)
      22      , batch_policy_buffers_{nullptr, nullptr}
     ...
      38          throw std::bad_alloc();
      39      }
      40
      41 +    // Initialize per-slot ready flags (all zero = not ready)
      42 +    slot_ready_.resize(queue_capacity);
      43 +    for (size_t i = 0; i < queue_capacity; ++i) {
      44 +        slot_ready_[i].store(0, std::memory_order_relaxed);
      45 +    }
      46 +
      47      // Pre-allocate staged request metadata queue
      48      staged_requests_.reserve(queue_capacity);
      49
      41 -    // Pre-allocate batch mapping
      50 +    // Pre-allocate batch mapping and staging slot tracking
      51      batch_mapping_.reserve(max_batch_size);
      52 +    batch_staging_slots_.reserve(max_batch_size);
      53
      54      // Allocate batch buffers (aligned for GPU)
      55      allocate_batch_buffers();

● 2. allocate_batch_buffers — remove batch_obs_buffer_:

● Update(alphazero-cpp\src\selfplay\evaluation_queue.cpp)
  ⎿  Added 3 lines, removed 3 lines
       87      size_t value_bytes = max_batch_size_ * sizeof(float);
       88
       89      // Allocate aligned memory (64-byte for cache line / GPU compatibility)
       90 +    // NOTE: batch_obs_buffer_ (NHWC intermediate) removed — fused transpose
       91 +    // now reads directly from staging into batch_obs_nchw_buffer_
       92      #ifdef _WIN32
       91 -    batch_obs_buffer_ = static_cast<float*>(_aligned_malloc(obs_bytes, 64));
       93      batch_obs_nchw_buffer_ = static_cast<float*>(_aligned_malloc(obs_bytes, 64));
       94      batch_mask_buffer_ = static_cast<float*>(_aligned_malloc(mask_bytes, 64));
       95      batch_policy_buffers_[0] = static_cast<float*>(_aligned_malloc(policy_bytes, 64));
       96      batch_policy_buffers_[1] = static_cast<float*>(_aligned_malloc(policy_bytes, 64));
       97      batch_value_buffer_ = static_cast<float*>(_aligned_malloc(value_bytes, 64));
       98      #else
       98 -    batch_obs_buffer_ = static_cast<float*>(std::aligned_alloc(64, obs_bytes));
       99      batch_obs_nchw_buffer_ = static_cast<float*>(std::aligned_alloc(64, obs_bytes));
      100      batch_mask_buffer_ = static_cast<float*>(std::aligned_alloc(64, mask_bytes));
      101      batch_policy_buffers_[0] = static_cast<float*>(std::aligned_alloc(64, policy_bytes));
     ...
      103      batch_value_buffer_ = static_cast<float*>(std::aligned_alloc(64, value_bytes));
      104      #endif
      105
      106 -    if (!batch_obs_buffer_ || !batch_obs_nchw_buffer_ || !batch_mask_buffer_ ||
      106 +    if (!batch_obs_nchw_buffer_ || !batch_mask_buffer_ ||
      107          !batch_policy_buffers_[0] || !batch_policy_buffers_[1] || !batch_value_buffer_) {
      108          free_batch_buffers();
      109          throw std::bad_alloc();

● 3. free_batch_buffers — remove batch_obs_buffer_:

● Update(alphazero-cpp\src\selfplay\evaluation_queue.cpp)
  ⎿  Removed 3 lines
      112
      113  void GlobalEvaluationQueue::free_batch_buffers() {
      114      #ifdef _WIN32
      115 -    if (batch_obs_buffer_) _aligned_free(batch_obs_buffer_);
      115      if (batch_obs_nchw_buffer_) _aligned_free(batch_obs_nchw_buffer_);
      116      if (batch_mask_buffer_) _aligned_free(batch_mask_buffer_);
      117      if (batch_policy_buffers_[0]) _aligned_free(batch_policy_buffers_[0]);
      118      if (batch_policy_buffers_[1]) _aligned_free(batch_policy_buffers_[1]);
      119      if (batch_value_buffer_) _aligned_free(batch_value_buffer_);
      120      #else
      122 -    if (batch_obs_buffer_) std::free(batch_obs_buffer_);
      121      if (batch_obs_nchw_buffer_) std::free(batch_obs_nchw_buffer_);
      122      if (batch_mask_buffer_) std::free(batch_mask_buffer_);
      123      if (batch_policy_buffers_[0]) std::free(batch_policy_buffers_[0]);
     ...
      125      if (batch_value_buffer_) std::free(batch_value_buffer_);
      126      #endif
      127
      130 -    batch_obs_buffer_ = nullptr;
      128      batch_obs_nchw_buffer_ = nullptr;
      129      batch_mask_buffer_ = nullptr;
      130      batch_policy_buffers_[0] = nullptr;

● 4. submit_for_evaluation — the 3-phase split (most critical change):

● Update(alphazero-cpp\src\selfplay\evaluation_queue.cpp)
  ⎿  Added 38 lines, removed 21 lines
      155      out_request_ids.reserve(num_leaves);
      156
      157      int actual_leaves = 0;
      158 +    int first_slot = 0;
      159
      159 -    // All staging operations happen under queue_mutex_ to synchronize with
      160 -    // collect_batch's staging_write_head_ reset. This eliminates the race
      161 -    // where workers claim slots via atomic fetch_add while the GPU thread
      162 -    // resets the head, causing unbounded head growth and pool exhaustion.
      163 -    //
      164 -    // Lock hold time: ~80µs for 32 leaves (memcpy of ~1.6MB). Acceptable
      165 -    // given batch cycles of ~10-80ms.
      160 +    // =====================================================================
      161 +    // Phase 1 (under lock, ~1µs): Claim staging slots + push metadata
      162 +    // =====================================================================
      163 +    // Only lightweight operations: integer arithmetic + vector push_back.
      164 +    // The expensive memcpy is deferred to Phase 2 (no lock).
      165      {
      166          std::lock_guard<std::mutex> lock(queue_mutex_);
      167
      169 -        // Claim staging slots (synchronized with GPU's reset)
      170 -        int first_slot = staging_write_head_.load(std::memory_order_relaxed);
      168 +        // Claim staging slots (synchronized with GPU's compaction reset)
      169 +        first_slot = staging_write_head_.load(std::memory_order_relaxed);
      170          actual_leaves = num_leaves;
      171
      172          if (first_slot + num_leaves > static_cast<int>(queue_capacity_)) {
     ...
      183
      184          staging_write_head_.store(first_slot + actual_leaves, std::memory_order_relaxed);
      185
      187 -        // Write observation and mask data into staging buffers
      188 -        for (int i = 0; i < actual_leaves; ++i) {
      189 -            int slot = first_slot + i;
      190 -            std::memcpy(staging_obs_buffer_ + slot * OBS_SIZE,
      191 -                        observations + i * OBS_SIZE,
      192 -                        OBS_SIZE * sizeof(float));
      193 -            std::memcpy(staging_mask_buffer_ + slot * POLICY_SIZE,
      194 -                        legal_masks + i * POLICY_SIZE,
      195 -                        POLICY_SIZE * sizeof(float));
      196 -        }
      197 -
      198 -        // Push metadata
      186 +        // Push metadata (no data copy — just slot indices)
      187          uint32_t gen = worker_generation_[worker_id].load(std::memory_order_acquire);
      188          for (int i = 0; i < actual_leaves; ++i) {
      189              int32_t req_id = worker_request_ids_[worker_id].fetch_add(1, std::memory_order_relaxed);
     ...
      197              metrics_.partial_submissions.fetch_add(1, std::memory_order_relaxed);
      198          }
      199      }
      200 +    // Lock released — other workers can claim slots concurrently
      201 +
      202 +    // =====================================================================
      203 +    // Phase 2 (NO lock): memcpy observation + mask data to claimed slots
      204 +    // =====================================================================
      205 +    // Safe because each worker owns its claimed slots exclusively:
      206 +    // - Slots [first_slot, first_slot + actual_leaves) belong to this worker
      207 +    // - No other worker can claim the same slots (staging_write_head_ was
      208 +    //   incremented atomically under lock)
      209 +    // - GPU thread won't read these slots until slot_ready_ flags are set
      210 +    for (int i = 0; i < actual_leaves; ++i) {
      211 +        int slot = first_slot + i;
      212 +        std::memcpy(staging_obs_buffer_ + slot * OBS_SIZE,
      213 +                    observations + i * OBS_SIZE,
      214 +                    OBS_SIZE * sizeof(float));
      215 +        std::memcpy(staging_mask_buffer_ + slot * POLICY_SIZE,
      216 +                    legal_masks + i * POLICY_SIZE,
      217 +                    POLICY_SIZE * sizeof(float));
      218 +    }
      219 +
      220 +    // =====================================================================
      221 +    // Phase 3 (NO lock): Signal slot readiness + notify GPU thread
      222 +    // =====================================================================
      223 +    // release semantics ensure the preceding memcpy is visible to the GPU
      224 +    // thread when it does an acquire load on slot_ready_[slot]
      225 +    for (int i = 0; i < actual_leaves; ++i) {
      226 +        slot_ready_[first_slot + i].store(1, std::memory_order_release);
      227 +    }
      228 +
      229      // Notify OUTSIDE the lock — the GPU thread can immediately acquire
      230      // the mutex when woken instead of blocking on the notifier's lock.
      231      queue_cv_.notify_one();

● 5. collect_batch — the 2-phase redesign with deferred compaction + fused transpose:

● Update(alphazero-cpp\src\selfplay\evaluation_queue.cpp)
  ⎿  Added 66 lines, removed 52 lines
      386      auto start = std::chrono::steady_clock::now();
      387      size_t batch_size = 0;
      388
      389 -    // Phase 1 (under queue_mutex_): Collect staged requests, copy staging → batch buffers
      389 +    // =====================================================================
      390 +    // Phase 1 (under queue_mutex_, ~10µs): Metadata extraction
      391 +    // =====================================================================
      392 +    // Only lightweight operations: deferred compaction, wait, vector swap.
      393 +    // NO memcpy of observation/mask data under lock.
      394      {
      395          std::unique_lock<std::mutex> lock(queue_mutex_);
      396
      393 -        // TWO-PHASE WAIT with stall detection.
      397 +        // --- Deferred compaction from previous batch ---
      398 +        // By deferring to here (start of NEXT collect_batch), we guarantee
      399 +        // all previous-batch slot_ready_ flags have been consumed and cleared
      400 +        // in the previous Phase 2, so compaction won't overwrite live slots.
      401 +        if (needs_compaction_) {
      402 +            compact_remaining_locked();
      403 +            needs_compaction_ = false;
      404 +        }
      405 +
      406 +        // TWO-PHASE WAIT with stall detection (UNCHANGED — it works well).
      407          //
      408          // Phase 1: Wait for ANY request (full timeout).
      409          //   Handles the idle GPU case (between iterations, shutdown).
     ...
      412          //   Keep waiting as long as workers are actively submitting
      413          //   (queue growing). Once no new submissions arrive within 1ms,
      414          //   fire immediately with whatever we have.
      402 -        //
      403 -        //   This is optimal because:
      404 -        //   - If all 64 workers submit in 5ms → fires at 5ms with 1024
      405 -        //   - If 62 workers submit in 4ms, then nothing for 1ms → fires
      406 -        //     at 5ms with 992 (doesn't waste 10ms waiting for stragglers)
      407 -        //   - Adapts to any worker count / search_batch combination
      415
      416          // Phase 1: Wait for at least 1 request
      417          queue_cv_.wait_for(
     ...
      456              }
      457          }
      458
      452 -        // Collect up to max_batch_size requests
      459 +        // Collect up to max_batch_size requests — METADATA ONLY
      460          batch_size = std::min(staged_requests_.size(), max_batch_size_);
      461
      455 -        // Build batch mapping and copy data from staging slots to batch buffers
      462          batch_mapping_.clear();
      463          batch_mapping_.reserve(batch_size);
      464 +        batch_staging_slots_.clear();
      465 +        batch_staging_slots_.reserve(batch_size);
      466
      467          for (size_t i = 0; i < batch_size; ++i) {
      468              const auto& req = staged_requests_[i];
      461 -
      462 -            // Copy from staging slot directly to batch buffer (NHWC)
      463 -            std::memcpy(batch_obs_buffer_ + i * OBS_SIZE,
      464 -                        staging_obs_buffer_ + req.staging_slot * OBS_SIZE,
      465 -                        OBS_SIZE * sizeof(float));
      466 -
      467 -            std::memcpy(batch_mask_buffer_ + i * POLICY_SIZE,
      468 -                        staging_mask_buffer_ + req.staging_slot * POLICY_SIZE,
      469 -                        POLICY_SIZE * sizeof(float));
      470 -
      469              batch_mapping_.emplace_back(req.worker_id, req.request_id, req.generation);
      470 +            batch_staging_slots_.push_back(req.staging_slot);
      471          }
      472
      474 -        // Remove collected requests
      473 +        // Remove collected requests from the metadata queue
      474          staged_requests_.erase(staged_requests_.begin(),
      475                                  staged_requests_.begin() + batch_size);
      476
      478 -        // LAZY COMPACTION: Only compact when staging buffer is >75% full.
      479 -        // Without compaction, collected slots at the front are "wasted" and
      480 -        // staging_write_head_ grows monotonically. This is fine as long as
      481 -        // head < queue_capacity. Compaction reclaims wasted slots by moving
      482 -        // remaining data to the front and resetting the head.
      483 -        //
      484 -        // Previously, compaction ran EVERY batch — O(remaining) memcpy under
      485 -        // the lock. With eval_batch << queue_size (e.g., 10 vs 8192), this
      486 -        // meant copying ~50MB per batch, causing O(n²) degradation.
      487 -        size_t remaining = staged_requests_.size();
      477 +        // LAZY COMPACTION: Schedule for next collect_batch if head > 75%.
      478 +        // Deferring is safe: collected slots' data is read in Phase 2 below,
      479 +        // and remaining requests' slot_ready_ flags are already set by workers.
      480          int head = staging_write_head_.load(std::memory_order_relaxed);
      489 -
      481          if (head > static_cast<int>(queue_capacity_ * 3 / 4)) {
      491 -            // Compact remaining items to the front
      492 -            for (size_t i = 0; i < remaining; ++i) {
      493 -                int old_slot = staged_requests_[i].staging_slot;
      494 -                if (old_slot != static_cast<int>(i)) {
      495 -                    std::memcpy(staging_obs_buffer_ + i * OBS_SIZE,
      496 -                                staging_obs_buffer_ + old_slot * OBS_SIZE,
      497 -                                OBS_SIZE * sizeof(float));
      498 -                    std::memcpy(staging_mask_buffer_ + i * POLICY_SIZE,
      499 -                                staging_mask_buffer_ + old_slot * POLICY_SIZE,
      500 -                                POLICY_SIZE * sizeof(float));
      501 -                    staged_requests_[i].staging_slot = static_cast<int>(i);
      502 -                }
      503 -            }
      504 -            staging_write_head_.store(static_cast<int>(remaining), std::memory_order_release);
      505 -            metrics_.pool_resets.fetch_add(1, std::memory_order_relaxed);
      482 +            needs_compaction_ = true;
      483          }
      484      }
      485      // Lock released — workers can submit new leaves during Phase 2
      486
      510 -    // Phase 2 (NO lock): NHWC→NCHW transpose on batch_obs_buffer_ → batch_obs_nchw_buffer_
      511 -    // Safe: both buffers are only accessed by the GPU thread
      512 -    for (size_t b = 0; b < batch_size; ++b) {
      513 -        const float* nhwc = batch_obs_buffer_ + b * OBS_SIZE;
      514 -        float* nchw = batch_obs_nchw_buffer_ + b * OBS_SIZE;
      487 +    // =====================================================================
      488 +    // Phase 2 (NO lock, OpenMP parallel): Spin-wait + fused transpose + mask copy
      489 +    // =====================================================================
      490 +    // For each batch entry:
      491 +    //   1. Spin-wait on slot_ready_[slot] (typically instant — worker finished
      492 +    //      its memcpy long before GPU reached stall detection)
      493 +    //   2. Fused NHWC→NCHW transpose: staging_obs → batch_obs_nchw (skip
      494 +    //      intermediate batch_obs_buffer_, saving 32MB)
      495 +    //   3. memcpy mask: staging_mask → batch_mask
      496 +    //   4. Clear slot_ready_[slot] for reuse
      497 +    //
      498 +    // Safe because:
      499 +    //   - Workers are blocked on get_results() during GPU's Phase 2
      500 +    //   - Each batch entry reads from a unique staging slot
      501 +    //   - Each batch entry writes to a unique batch index
      502 +
      503 +    int batch_size_i = static_cast<int>(batch_size);
      504 +
      505 +    #pragma omp parallel for schedule(static)
      506 +    for (int b = 0; b < batch_size_i; ++b) {
      507 +        int slot = batch_staging_slots_[b];
      508 +
      509 +        // Spin-wait until worker has finished writing this slot's data.
      510 +        // In practice, workers finish memcpy well before the GPU thread
      511 +        // completes stall detection, so this almost never actually spins.
      512 +        while (slot_ready_[slot].load(std::memory_order_acquire) == 0) {
      513 +            // Yield to avoid burning CPU in the rare case of a slow worker
      514 +            std::this_thread::yield();
      515 +        }
      516 +
      517 +        // Fused NHWC→NCHW transpose directly from staging to batch output
      518 +        const float* nhwc = staging_obs_buffer_ + static_cast<size_t>(slot) * OBS_SIZE;
      519 +        float* nchw = batch_obs_nchw_buffer_ + static_cast<size_t>(b) * OBS_SIZE;
      520          for (int h = 0; h < 8; ++h) {
      521              for (int w = 0; w < 8; ++w) {
      522                  for (int c = 0; c < 122; ++c) {
     ...
      524                  }
      525              }
      526          }
      527 +
      528 +        // Copy mask directly from staging to batch buffer
      529 +        std::memcpy(batch_mask_buffer_ + static_cast<size_t>(b) * POLICY_SIZE,
      530 +                    staging_mask_buffer_ + static_cast<size_t>(slot) * POLICY_SIZE,
      531 +                    POLICY_SIZE * sizeof(float));
      532 +
      533 +        // Clear ready flag for slot reuse (relaxed — no ordering needed,
      534 +        // slot won't be reused until after compaction under lock)
      535 +        slot_ready_[slot].store(0, std::memory_order_relaxed);
      536      }
      537
      538      // Update metrics

● 6. Add compact_remaining_locked() method and update reset():

● Update(alphazero-cpp\src\selfplay\evaluation_queue.cpp)
  ⎿  Added 43 lines
      612  }
      613
      614  // ============================================================================
      615 +// Compaction (called under lock)
      616 +// ============================================================================
      617 +
      618 +void GlobalEvaluationQueue::compact_remaining_locked() {
      619 +    // Move remaining staged requests' data to the front of staging buffers.
      620 +    // Called at the START of collect_batch, so all previous-batch slots have
      621 +    // been consumed and their slot_ready_ flags cleared in the prior Phase 2.
      622 +    //
      623 +    // For remaining requests (not yet collected), we also need to wait for
      624 +    // their slot_ready_ flags before compacting, because the worker may still
      625 +    // be in Phase 2 (memcpy) when compaction fires.
      626 +    size_t remaining = staged_requests_.size();
      627 +
      628 +    for (size_t i = 0; i < remaining; ++i) {
      629 +        int old_slot = staged_requests_[i].staging_slot;
      630 +
      631 +        // Wait for this slot's data to be fully written by its worker
      632 +        while (slot_ready_[old_slot].load(std::memory_order_acquire) == 0) {
      633 +            std::this_thread::yield();
      634 +        }
      635 +
      636 +        if (old_slot != static_cast<int>(i)) {
      637 +            std::memcpy(staging_obs_buffer_ + i * OBS_SIZE,
      638 +                        staging_obs_buffer_ + old_slot * OBS_SIZE,
      639 +                        OBS_SIZE * sizeof(float));
      640 +            std::memcpy(staging_mask_buffer_ + i * POLICY_SIZE,
      641 +                        staging_mask_buffer_ + old_slot * POLICY_SIZE,
      642 +                        POLICY_SIZE * sizeof(float));
      643 +
      644 +            // Move the ready flag to the new slot position
      645 +            slot_ready_[i].store(1, std::memory_order_relaxed);
      646 +            slot_ready_[old_slot].store(0, std::memory_order_relaxed);
      647 +
      648 +            staged_requests_[i].staging_slot = static_cast<int>(i);
      649 +        }
      650 +        // If old_slot == i, slot is already in the right place — no move needed
      651 +    }
      652 +
      653 +    staging_write_head_.store(static_cast<int>(remaining), std::memory_order_release);
      654 +    metrics_.pool_resets.fetch_add(1, std::memory_order_relaxed);
      655 +}
      656 +
      657 +// ============================================================================
      658  // Lifecycle Management
      659  // ============================================================================
      660

● 7. Update reset() to clear slot_ready_ and needs_compaction_:

● Update(alphazero-cpp\src\selfplay\evaluation_queue.cpp)
  ⎿  Added 7 lines
      676      shutdown_.store(false, std::memory_order_release);
      677      staged_requests_.clear();
      678      batch_mapping_.clear();
      679 +    batch_staging_slots_.clear();
      680      staging_write_head_.store(0, std::memory_order_release);
      681      current_policy_buffer_ = 0;
      682 +    needs_compaction_ = false;
      683
      684 +    // Clear all slot ready flags
      685 +    for (size_t i = 0; i < slot_ready_.size(); ++i) {
      686 +        slot_ready_[i].store(0, std::memory_order_relaxed);
      687 +    }
      688 +
      689      // Clear worker results and reset generation counters
      690      for (size_t i = 0; i < MAX_WORKERS; ++i) {
      691          std::lock_guard<std::mutex> wlock(worker_mutexes_[i]);

● Now Task 3 — Add OpenMP to selfplay target:

● Update(alphazero-cpp\CMakeLists.txt)
  ⎿  Added 1 line
      112      mcts_core
      113      encoding
      114      training
      115 +    OpenMP::OpenMP_CXX
      116  )
      117
      118  # Training library

● Now Task 4 — eval_batch auto-tune suggestion in train.py:

● Update(alphazero-cpp\scripts\train.py)
  ⎿  Added 6 lines
      2109      print(f"  eval_batch={args.eval_batch}, gpu_timeout={args.gpu_batch_timeout_ms}ms, "
      2110            f"queue_capacity={queue_capacity}, root_eval_retries={args.root_eval_retries}")
      2111
      2112 +    # Suggest optimal eval_batch based on worker count and search_batch
      2113 +    suggested_eval_batch = ((args.workers * args.search_batch + 31) // 32) * 32
      2114 +    if suggested_eval_batch != args.eval_batch and suggested_eval_batch > 0:
      2115 +        print(f"  💡 Hint: eval_batch={suggested_eval_batch} matches workers×search_batch "
      2116 +              f"({args.workers}×{args.search_batch}={args.workers * args.search_batch}, rounded to 32)")
      2117 +
      2118      # Push initial progress to live dashboard (show we're in self-play phase)
      2119      if live_dashboard is not None:
      2120          live_dashboard.push_progress(

● Now Task 5 — Build and verify. Let me first verify the final state of the files, then build:

● Read 2 files (ctrl+o to expand)

● All files look correct. Now let me build:

● Bash(cd "C:\Users\liroc\OneDrive - Duke University\2025-2026\projects\alpha-zero-chess\alphazero-cpp\build2" && cmake --build .
       --config Release --target selfplay 2>…)
  ⎿  Error: Exit code 1
     CMake is re-running because C:/Users/liroc/OneDrive - Duke
     University/2025-2026/projects/alpha-zero-chess/alphazero-cpp/build2/CMakeFiles/generate.stamp is out-of-date.
       the file 'C:/Users/liroc/OneDrive - Duke University/2025-2026/projects/alpha-zero-chess/alphazero-cpp/CMakeLists.txt'
       is newer than 'C:/Users/liroc/OneDrive - Duke
     University/2025-2026/projects/alpha-zero-chess/alphazero-cpp/build2/CMakeFiles/generate.stamp.depend'
       result='-1'
     -- Found Python: C:/Program
     Files/WindowsApps/PythonSoftwareFoundation.Python.3.13_3.13.2544.0_x64__qbz5n2kfra8p0/python3.13.exe
     -- Found pybind11: C:/Users/liroc/OneDrive - Duke
     University/2025-2026/projects/alpha-zero-chess/.venv/Lib/site-packages/pybind11/include (found version "3.0.1")
     -- Found pybind11: C:/Users/liroc/OneDrive - Duke
     University/2025-2026/projects/alpha-zero-chess/.venv/Lib/site-packages/pybind11/share/cmake/pybind11
     -- Python bindings will be built
     -- Configuring done (1.0s)
     … +77 lines (ctrl+o to see all)

● ★ Insight ─────────────────────────────────────
  MSVC and std::vector<std::atomic<T>>: The std::atomic types have deleted copy constructors, and std::vector::resize() attempts
  to copy-construct elements when growing. This is a well-known MSVC incompatibility — std::vector<std::atomic<T>> isn't usable
  because the vector requires copyable/movable elements. The fix: use a raw std::unique_ptr<std::atomic<uint8_t>[]> array
  instead.
  ─────────────────────────────────────────────────

  The issue is that std::vector<std::atomic<uint8_t>> requires copy/move constructors which atomics don't have. Let me switch to
  a std::unique_ptr<std::atomic<uint8_t>[]>:

● Update(alphazero-cpp\include\selfplay\evaluation_queue.hpp)
  ⎿  Added 1 line
      10  #include <cstdlib>
      11  #include <tuple>
      12  #include <thread>
      13 +#include <memory>
      14
      15  namespace selfplay {
      16

● Update(alphazero-cpp\include\selfplay\evaluation_queue.hpp)
  ⎿  Added 3 lines, removed 1 line
      245
      246      // Per-slot ready flags: workers set after memcpy completes (lock-free signaling)
      247      // GPU thread spin-waits on these before reading staging data
      248 -    std::vector<std::atomic<uint8_t>> slot_ready_;  // queue_capacity entries
      248 +    // NOTE: Using unique_ptr<atomic[]> instead of vector<atomic> because
      249 +    // std::atomic has deleted copy/move constructors, incompatible with std::vector on MSVC
      250 +    std::unique_ptr<std::atomic<uint8_t>[]> slot_ready_;  // queue_capacity entries
      251
      252      // Request metadata queue (lightweight — only metadata, no observation data)
      253      std::vector<StagedRequest> staged_requests_;

● Now update the .cpp to use the new array syntax:

● Update(alphazero-cpp\src\selfplay\evaluation_queue.cpp)
  ⎿  Added 1 line, removed 1 line
      39      }
      40
      41      // Initialize per-slot ready flags (all zero = not ready)
      42 -    slot_ready_.resize(queue_capacity);
      42 +    slot_ready_ = std::make_unique<std::atomic<uint8_t>[]>(queue_capacity);
      43      for (size_t i = 0; i < queue_capacity; ++i) {
      44          slot_ready_[i].store(0, std::memory_order_relaxed);
      45      }

● Now update the reset() function — change slot_ready_.size() to queue_capacity_:

● Update(alphazero-cpp\src\selfplay\evaluation_queue.cpp)
  ⎿  Added 1 line, removed 1 line
      682      needs_compaction_ = false;
      683
      684      // Clear all slot ready flags
      685 -    for (size_t i = 0; i < slot_ready_.size(); ++i) {
      685 +    for (size_t i = 0; i < queue_capacity_; ++i) {
      686          slot_ready_[i].store(0, std::memory_order_relaxed);
      687      }
      688

● Now let's rebuild:

● Bash(cd "C:\Users\liroc\OneDrive - Duke University\2025-2026\projects\alpha-zero-chess\alphazero-cpp\build2" && cmake --build .
       --config Release --target selfplay 2>…)
  ⎿ MSBuild version 18.0.5+e22287bf1 for .NET Framework

      encoding.vcxproj -> C:\Users\liroc\OneDrive - Duke University\2025-2026\projects\alpha-zero-chess\alphazero-cpp\build2
    … +36 lines (ctrl+o to expand)

  ⎿  (timeout 2m)

● selfplay.lib built successfully. Now let's build the full pybind11 module to make sure everything links:

● Bash(cd "C:\Users\liroc\OneDrive - Duke University\2025-2026\projects\alpha-zero-chess\alphazero-cpp\build2" && cmake --build .
       --config Release --target alphazero_c…)
  ⎿ MSBuild version 18.0.5+e22287bf1 for .NET Framework

      encoding.vcxproj -> C:\Users\liroc\OneDrive - Duke University\2025-2026\projects\alpha-zero-chess\alphazero-cpp\build2
    … +27 lines (ctrl+o to expand)

  ⎿  (timeout 2m)

● Build succeeded — alphazero_cpp.cp313-win_amd64.pyd generated cleanly. Now let me copy the new .pyd to the scripts directory
  and run a quick import test:

● Bash(copy "C:\Users\liroc\OneDrive - Duke
      University\2025-2026\projects\alpha-zero-chess\alphazero-cpp\build2\Release\alphazero_cpp.cp313-win_amd64.pyd"
      "C:\Users\li…)
  ⎿  Error: Exit code 127
     /usr/bin/bash: line 1: copy: command not found

● Bash(cmd /c copy "C:\Users\liroc\OneDrive - Duke
      University\2025-2026\projects\alpha-zero-chess\alphazero-cpp\build2\Release\alphazero_cpp.cp313-win_amd64.pyd" "C:\U…)
  ⎿  Microsoft Windows [Version 10.0.26100.7623]
     (c) Microsoft Corporation. All rights reserved.

     C:\Users\liroc\OneDrive - Duke University\2025-2026\projects\alpha-zero-chess>

● Bash(cd "C:\Users\liroc\OneDrive - Duke University\2025-2026\projects\alpha-zero-chess\alphazero-cpp\scripts" && python -c
      "import alphazero_cpp; q = alphazero_cpp.G…)
  ⎿  Error: Exit code 1
     Traceback (most recent call last):
       File "<string>", line 1, in <module>
         import alphazero_cpp; q = alphazero_cpp.GlobalEvaluationQueue(64, 256); print('Queue created OK, max_batch:',
     q.max_batch_size()); q.reset(); print('Reset OK'); print('SUCCESS')
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     AttributeError: module 'alphazero_cpp' has no attribute 'GlobalEvaluationQueue'

● The class is likely exposed under a different name. Let me check the Python bindings:

● Searched for 1 pattern (ctrl+o to expand)

● The GlobalEvaluationQueue is accessed through ParallelSelfPlayCoordinator in the Python bindings, not directly exposed. Let me
  do a quick import test:

● Bash(cd "C:\Users\liroc\OneDrive - Duke University\2025-2026\projects\alpha-zero-chess\alphazero-cpp\scripts" && python -c
      "import alphazero_cpp; print(dir(alphazero…)
  ⎿ ['BatchedMCTSSearch', 'ParallelSelfPlayCoordinator', 'ReplayBuffer', 'SelfPlayCoordinator', 'Trainer', 'TrainerConfig',
    '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '__version__', 'encode_batch', 'encode_posit
    ion', 'encode_position_to_buffer', 'index_to_move', 'move_to_index']
    Import OK


● Module imports fine. The GlobalEvaluationQueue is used internally by ParallelSelfPlayCoordinator. The build is clean and the
  module loads correctly.

● Now let me update the memory file with the new architecture details:

● Recalled 1 memory, wrote 2 memories (ctrl+o to expand)

● All tasks are complete. Here's a summary of everything that was implemented:

  ---
  Implementation Summary

  Files Modified

  1. evaluation_queue.hpp — Header changes
  - Added #include <thread> and #include <memory>
  - Added std::unique_ptr<std::atomic<uint8_t>[]> slot_ready_ — per-slot readiness flags for lock-free data signaling
  - Added bool needs_compaction_{false} — deferred compaction flag
  - Added std::vector<int32_t> batch_staging_slots_ — tracks staging slot indices for lock-free Phase 2 reads
  - Added void compact_remaining_locked() private method
  - Removed float* batch_obs_buffer_ — the NHWC intermediate buffer (saves 32MB)

  2. evaluation_queue.cpp — Core pipeline rewrite
  - submit_for_evaluation() — Split from 1 phase to 3:
    - Phase 1 (lock ~1µs): claim slots via staging_write_head_ + push metadata
    - Phase 2 (no lock): memcpy obs/mask data to claimed slots
    - Phase 3 (no lock): set slot_ready_ flags with release semantics + notify_one()
  - collect_batch() — Split from 1 phase to 2:
    - Phase 1 (lock ~10µs): deferred compaction, two-phase stall detection (unchanged), metadata extraction
    - Phase 2 (no lock, #pragma omp parallel for): spin-wait on slot_ready_, fused NHWC→NCHW transpose directly from staging,
  mask copy, clear flags
  - Added compact_remaining_locked(): Waits for slot_ready_ flags, then moves remaining data to front of staging buffers
  - Removed batch_obs_buffer_ from allocate_batch_buffers() and free_batch_buffers()
  - Updated reset(): Clears slot_ready_ flags, batch_staging_slots_, needs_compaction_

  3. CMakeLists.txt — Added OpenMP::OpenMP_CXX to selfplay target

  4. train.py — Added eval_batch auto-tune suggestion (5 lines near line 2108)

  Key Design Insight

  ★ Insight ─────────────────────────────────────
  - Before: Lock held for ~80µs/worker (memcpy) + ~1.7ms/batch (staging→batch copy) = ~6.8ms total lock time per batch cycle
  - After: Lock held for ~1µs/worker (metadata) + ~10µs/batch (metadata extraction) = ~0.07ms total lock time per batch cycle
  - The trick is ownership transfer via atomic flags: the mutex protects only the metadata handshake (who owns which slots),
  while slot_ready_ atomics with release/acquire semantics guarantee data visibility without holding a lock
  - MSVC gotcha: std::vector<std::atomic<T>> won't compile because resize() tries to copy-construct atomics (deleted copy ctor).
  Solution: std::unique_ptr<std::atomic<T>[]>
  ─────────────────────────────────────────────────

  Verification

  - ✅ Builds clean on Windows (MSVC) — only expected C4819 Unicode warnings
  - ✅ Python module imports successfully
  - ✅ No changes to Python bindings, submit_results, or CUDA Graph logic

  Next Steps for Testing

  1. Run existing test_eval_queue_staleness.py
  2. Small config: --workers 4 --games-per-iter 8 --simulations 100 --eval-batch 64
  3. Production config: --workers 64 --simulations 800 --eval-batch 1024
  4. Monitor: no pool_exhaustion, no submission_drops, avg_batch near eval_batch

✻ Cogitated for 7m 50s · 2 agents running in the background

❯ Based on your summary, the implementation is **architecturally complete** and aligns with the redesign goals. You have
  successfully identified and bypassed the two biggest performance killers: the  data movement under lock and the
  `std::vector<std::atomic>` MSVC compilation trap.

  However, there is one critical detail in the **Phase 1 (Lock) / Phase 2 (No Lock)** split that requires a specific
  implementation pattern to prevent a rare but fatal race condition.

  ### 1. The "Ghost Slot" Race Condition

  In your plan, Phase 1 (under lock) claims the slots. If a worker is interrupted *immediately* after Phase 1 but *before*
  Phase 3 (setting `slot_ready_`), the GPU thread in `collect_batch` will enter its Phase 2 and hit a `spin-wait` on a slot
  that hasn't even begun its `memcpy` yet.

  **The Fix:** Your `collect_batch` Phase 2 (the OpenMP part) must use a **bounded spin-wait with a fallback**.

  * If the GPU spins for more than, say, 100ms without the flag being set, it implies a worker crashed or is severely
  descheduled. You should include a `std::this_thread::yield()` or a logged warning to prevent the GPU thread from
  hard-locking the entire system.

  ### 2. Refined `compact_remaining_locked()`

  Since you are using deferred compaction, this method is the most dangerous part of the new design. It must be
  "Worker-Aware."

  **Implementation Check:**

  * **Must Wait:** `compact_remaining_locked` must also spin-wait on the `slot_ready_` flags for the "remaining" items it is
   about to move. If you move data while a worker is still writing to a high-index slot, you will corrupt the evaluation
  data.
  * **Flag Reset:** After moving the data to the front (indices ), you must `store(1)` in the new slot flags and `store(0)`
  in the old ones to maintain consistency for the next `collect_batch`.

  ### 3. OpenMP & Transpose Logic

  The fused NHWC  NCHW transpose is where you'll see the 32MB memory savings.

  **Verification Tip:** Ensure your OpenMP schedule is set to `static`. Since every "leaf" evaluation is the same size, a
  static schedule avoids the overhead of dynamic work-stealing, which is important when your target latency is in the
  range.

  ```cpp
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < batch_size; ++i) {
      int staging_idx = batch_staging_slots_[i];
      // 1. Bounded spin-wait
      while (slot_ready_[staging_idx].load(std::memory_order_acquire) == 0) {
          std::this_thread::yield();
      }
      // 2. Fused Transpose (NHWC -> NCHW)
      // 3. Mask Memcpy
  }

  ```

  ### 4. Evaluation of the "Plan vs. Implementation"

  | Plan Component | Implementation Match | Status |
  | --- | --- | --- |
  | **Lock Reduction** | ~200x reduction targeted | **Confirmed.** Metadata-only locks are . |
  | **Memory Savings** | 32MB via buffer removal | **Confirmed.** Direct staging-to-batch transpose. |
  | **Portability** | MSVC Compatibility | **Fixed.** `unique_ptr<atomic[]>` solves the `vector` issue. |
  | **Stall Detection** | Keep existing logic | **Confirmed.** Maintaining the two-phase wait is wise. |

  ### Final Recommendation

  The use of `batch_staging_slots_` to track indices is a great addition—it decouples the physical staging location from the
   logical batch position, which makes the "Deferred Compaction" much easier to manage.

  **Would you like me to write the specific C++ loop for the fused NHWC to NCHW transpose to ensure it's optimized for
  OpenMP cache-line alignment?**

● Great analysis. Let me address each point:

  Point 2 (compact_remaining_locked) — already implemented with spin-wait on slot_ready_ and proper flag migration. ✅

  Point 3 (OpenMP static) — already using schedule(static). ✅

  Point 1 (Ghost Slot race) — this is the critical fix. You're right: if a worker is descheduled between Phase 1 and Phase 3, the
   GPU thread will spin indefinitely. My current code has yield() but no shutdown_ escape hatch, so it would hard-lock on
  graceful shutdown attempts too. Let me fix both spin-wait sites:

● Read 1 file (ctrl+o to expand)

● Good. Now let me apply the bounded spin-wait fixes to both locations:

  ★ Insight ─────────────────────────────────────
  Why a bounded spin-wait matters even though "it almost never spins":
  - The typical case is fine — workers finish memcpy during the GPU's ~5ms stall-detection wait, so slot_ready_ is already 1 by
  the time Phase 2 reads it.
  - The pathological case: a worker thread gets descheduled by the OS scheduler right between Phase 1 (metadata pushed) and Phase
   3 (flag set). On a loaded system, descheduling can last 10-100ms. Without a shutdown_ escape, calling shutdown() from Python
  would block forever.
  - Inside an OpenMP parallel region, we can't easily "break" out of the loop. But we can skip work for a given slot and zero the
   batch entry, preserving system liveness. The evaluation result for that slot will be stale anyway.
  ─────────────────────────────────────────────────

  Fix 1: collect_batch Phase 2 — bounded spin with shutdown escape:

● Update(alphazero-cpp\src\selfplay\evaluation_queue.cpp)
  ⎿  Added 36 lines, removed 4 lines
      506      for (int b = 0; b < batch_size_i; ++b) {
      507          int slot = batch_staging_slots_[b];
      508
      509 -        // Spin-wait until worker has finished writing this slot's data.
      509 +        // Bounded spin-wait until worker has finished writing this slot's data.
      510          // In practice, workers finish memcpy well before the GPU thread
      511          // completes stall detection, so this almost never actually spins.
      512 -        while (slot_ready_[slot].load(std::memory_order_acquire) == 0) {
      513 -            // Yield to avoid burning CPU in the rare case of a slow worker
      514 -            std::this_thread::yield();
      512 +        // The spin count check (~every 1024 yields) provides:
      513 +        //   1. shutdown_ escape to prevent hard-lock during graceful shutdown
      514 +        //   2. Detection of severely descheduled workers (>100ms warning)
      515 +        bool slot_valid = true;
      516 +        {
      517 +            int spin_count = 0;
      518 +            auto spin_start = std::chrono::steady_clock::now();
      519 +            while (slot_ready_[slot].load(std::memory_order_acquire) == 0) {
      520 +                std::this_thread::yield();
      521 +                if (++spin_count % 1024 == 0) {
      522 +                    // Check shutdown to prevent hard-lock during teardown
      523 +                    if (shutdown_.load(std::memory_order_relaxed)) {
      524 +                        slot_valid = false;
      525 +                        break;
      526 +                    }
      527 +                    // Warn if worker is severely descheduled (>100ms)
      528 +                    auto elapsed = std::chrono::steady_clock::now() - spin_start;
      529 +                    if (elapsed > std::chrono::milliseconds(100)) {
      530 +                        fprintf(stderr, "[EvalQueue] WARNING: slot %d not ready after 100ms "
      531 +                                "(worker descheduled?), still waiting...\n", slot);
      532 +                        spin_start = std::chrono::steady_clock::now();
      533 +                    }
      534 +                }
      535 +            }
      536          }
      537
      538 +        if (!slot_valid) {
      539 +            // Shutdown requested while waiting — zero out this batch entry
      540 +            // to prevent garbage data from reaching the GPU. The result
      541 +            // will be discarded anyway since the evaluator loop checks shutdown.
      542 +            std::memset(batch_obs_nchw_buffer_ + static_cast<size_t>(b) * OBS_SIZE,
      543 +                        0, OBS_SIZE * sizeof(float));
      544 +            std::memset(batch_mask_buffer_ + static_cast<size_t>(b) * POLICY_SIZE,
      545 +                        0, POLICY_SIZE * sizeof(float));
      546 +            continue;
      547 +        }
      548 +
      549          // Fused NHWC→NCHW transpose directly from staging to batch output
      550          const float* nhwc = staging_obs_buffer_ + static_cast<size_t>(slot) * OBS_SIZE;
      551          float* nchw = batch_obs_nchw_buffer_ + static_cast<size_t>(b) * OBS_SIZE;

● Fix 2: compact_remaining_locked — same bounded spin with shutdown escape:

● Update(alphazero-cpp\src\selfplay\evaluation_queue.cpp)
  ⎿  Added 29 lines, removed 6 lines
      649
      650  void GlobalEvaluationQueue::compact_remaining_locked() {
      651      // Move remaining staged requests' data to the front of staging buffers.
      652 -    // Called at the START of collect_batch, so all previous-batch slots have
      653 -    // been consumed and their slot_ready_ flags cleared in the prior Phase 2.
      652 +    // Called at the START of collect_batch (under lock), so all previous-batch
      653 +    // slots have been consumed and their slot_ready_ flags cleared in the
      654 +    // prior Phase 2.
      655      //
      656      // For remaining requests (not yet collected), we also need to wait for
      657      // their slot_ready_ flags before compacting, because the worker may still
      657 -    // be in Phase 2 (memcpy) when compaction fires.
      658 +    // be in Phase 2 (memcpy) when compaction fires. In practice, workers
      659 +    // finish their memcpy during the GPU's ~10ms compute cycle, so this
      660 +    // wait is nearly instant.
      661      size_t remaining = staged_requests_.size();
      662
      663      for (size_t i = 0; i < remaining; ++i) {
      664          int old_slot = staged_requests_[i].staging_slot;
      665
      663 -        // Wait for this slot's data to be fully written by its worker
      664 -        while (slot_ready_[old_slot].load(std::memory_order_acquire) == 0) {
      665 -            std::this_thread::yield();
      666 +        // Bounded spin-wait for this slot's data to be fully written
      667 +        {
      668 +            int spin_count = 0;
      669 +            auto spin_start = std::chrono::steady_clock::now();
      670 +            while (slot_ready_[old_slot].load(std::memory_order_acquire) == 0) {
      671 +                std::this_thread::yield();
      672 +                if (++spin_count % 1024 == 0) {
      673 +                    if (shutdown_.load(std::memory_order_relaxed)) {
      674 +                        // Shutdown requested — abort compaction, leave state as-is.
      675 +                        // The staging buffer may be partially compacted, but that's
      676 +                        // OK since no more batches will be collected after shutdown.
      677 +                        staging_write_head_.store(
      678 +                            static_cast<int>(remaining), std::memory_order_release);
      679 +                        return;
      680 +                    }
      681 +                    auto elapsed = std::chrono::steady_clock::now() - spin_start;
      682 +                    if (elapsed > std::chrono::milliseconds(100)) {
      683 +                        fprintf(stderr, "[EvalQueue] WARNING: compaction waiting on slot %d "
      684 +                                "for >100ms (worker descheduled?)\n", old_slot);
      685 +                        spin_start = std::chrono::steady_clock::now();
      686 +                    }
      687 +                }
      688 +            }
      689          }
      690
      691          if (old_slot != static_cast<int>(i)) {

● Now let me rebuild and verify:

● Bash(cd "C:\Users\liroc\OneDrive - Duke University\2025-2026\projects\alpha-zero-chess\alphazero-cpp\build2" && cmake --build .
       --config Release --target alphazero_c…)
  ⎿ MSBuild version 18.0.5+e22287bf1 for .NET Framework

      encoding.vcxproj -> C:\Users\liroc\OneDrive - Duke University\2025-2026\projects\alpha-zero-chess\alphazero-cpp\build2
    … +29 lines (ctrl+o to expand)

  ⎿  (timeout 2m)

● Clean build. Let me copy the updated .pyd and run a quick import test:

● Bash(cmd /c copy "C:\Users\liroc\OneDrive - Duke
      University\2025-2026\projects\alpha-zero-chess\alphazero-cpp\build2\Release\alphazero_cpp.cp313-win_amd64.pyd" "C:\U…)
  ⎿  Running in the background (↓ to manage)
  ⎿  (timeout 15s)

● Read 1 file (ctrl+o to expand)

● Bash(cd "C:\Users\liroc\OneDrive - Duke University\2025-2026\projects\alpha-zero-chess\alphazero-cpp\scripts" && python -c
      "import alphazero_cpp; print('Import OK')"…)
  ⎿  Import OK

● All good. Now, regarding your question about the transpose optimization — let me analyze whether the loop reordering would
  help:

  ★ Insight ─────────────────────────────────────
  NHWC→NCHW transpose cache analysis for 8×8×122:
  - Each observation is 8 × 8 × 122 = 7808 floats = 31KB. Both source and destination fit in L1 cache (typically 48-64KB), so for
   a single observation, any loop ordering is fast.
  - Current order (h→w→c): Reads are sequential in NHWC memory layout ✓, but writes stride by 64 floats for each c increment
  (scattered across 122 planes).
  - Write-sequential order (c→h→w): Writes become sequential ✓, but reads stride by 122 for each w increment.
  - Since both source + dest fit in L1, the performance difference between orderings is negligible (~1-2%). The real win is
  OpenMP parallelization — splitting 1024 observations across 8+ cores turns a 0.5ms serial transpose into ~0.06-0.15ms.
  ─────────────────────────────────────────────────

  Given the analysis, the current transpose loop order is fine. The L1-resident 31KB per observation makes cache-line
  optimization irrelevant here. If you still want to try the write-sequential version for completeness, I can swap the loop to
  c→h→w, but the measured difference should be within noise. The OpenMP parallelization is doing the heavy lifting.

  Would you like me to swap the loop order anyway, or shall we focus on the production testing (--workers 64 --simulations 800
  --eval-batch 1024) to measure the actual throughput improvement?

  Why W_ra = f * W is the optimal split (the load-balancing derivation):

  All workers share one GPU via GlobalEvaluationQueue. The GPU processes Theta evals/sec regardless of which worker
  submitted them. Each worker gets Theta/W of the GPU's capacity. For self-play and reanalysis to finish simultaneously:

  Time_selfplay = G * L * N / (W_sp/W * Theta)
  Time_reanalysis = R * N / (W_ra/W * Theta)

  Setting equal and substituting R = f/(1-f) * G * L:
    G*L / W_sp = f/(1-f) * G*L / W_ra
    W_ra / W_sp = f / (1-f)
    => W_ra = f * W,  W_sp = (1-f) * W

  The CPU overhead difference between self-play (~1ms/move for game logic) and reanalysis (~0.3ms/position for buffer
  I/O) is <1% of the GPU-dominated time, so the simple proportional split is accurate.

  Why eval_batch should equal W * S (not larger):

  The CUDA graph large-graph tier activates at ~87.5% fill. If eval_batch = W * S, synchronized workers fill the batch
  100% -- large graph every time. If eval_batch = 2 * W * S, workers only fill 50% -- batches fall to the medium/small
  graph tier, wasting GPU capacity on zero-padding. The 50us stall-detection spin-poll in collect_batch() naturally
  handles desynchronization without needing extra headroom.

  The buffer freshness insight is critical: Reanalysis re-evaluates positions with the current network to get improved
  policy targets. If the buffer only holds 1-2 iterations of data, the generating network and current network are nearly
   identical -- the KL divergence is tiny and reanalysis is wasted GPU work. You need >= 3 iterations of buffer depth
  before the "staleness gap" produces meaningful policy improvements.

  ─────────────────────────────────────────────────

  Here's a summary of everything added:

  New --recommend mode

  # See optimal worker allocation for your config
  uv run python alphazero-cpp/scripts/test_reanalysis_impact.py \
      --recommend --workers 16 --reanalyze-fraction 0.25 \
      --search-batch 16 --simulations 200 --games-per-iter 50 \
      --buffer-size 100000

  The key formulas

  ┌───────────────────────┬────────────────────────────┬─────────────────────────────────────────────────────┐
  │         What          │          Formula           │                        Notes                        │
  ├───────────────────────┼────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Worker split          │ W_sp = (1-f)*W, W_ra = f*W │ Load-balanced completion; GPU-bottlenecked          │
  ├───────────────────────┼────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Eval batch            │ ceil(W*S / 32) * 32        │ Matches train.py auto-sizing; W = total workers     │
  ├───────────────────────┼────────────────────────────┼─────────────────────────────────────────────────────┤
  │ GPU overhead          │ 1 / (1-f)                  │ f=0.25 => 1.33x more NN evals                       │
  ├───────────────────────┼────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Buffer threshold      │ buffer_size >= 3 * G * L   │ Need >= 3 iterations of data for meaningful KL      │
  ├───────────────────────┼────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Batch fill sweet spot │ eval_batch = W_total * S   │ Larger wastes padding; smaller adds queuing latency │
  └───────────────────────┴────────────────────────────┴─────────────────────────────────────────────────────┘


    1. Optimal Self-Play vs Reanalysis Worker Distribution

  The core constraint is that both worker types share the same GlobalEvaluationQueue — they compete for identical GPU
  bandwidth. Each position requires ~`N_sims` NN evaluations whether it's fresh self-play or reanalysis.

  The formula:

  If F is the reanalyze fraction (fraction of training batch from reanalysis) and both paths use the same simulation
  count:

  W_ra / W_sp = F / ((1 - F) * α * (N_sims / N_sims_ra))

  Where:
  - α = efficiency ratio (reanalysis positions per sec / self-play positions per sec, per worker)
  - N_sims = self-play simulations
  - N_sims_ra = reanalysis simulations (can be lower)

  Self-play is slower per-position because it has game overhead (move making, legality checks, position encoding with
  history, game-over detection). Reanalysis just runs MCTS on a single already-encoded position. Empirically α ≈ 1.2–1.5
   depending on game length.

  Simplified (same sims, α ≈ 1):

  W_sp = W * (1 - F)
  W_ra = W * F

  ┌────────────────────┬─────────────┬──────────────┬───────────────┐
  │ Reanalyze Fraction │  8 Workers  │  16 Workers  │  32 Workers   │
  ├────────────────────┼─────────────┼──────────────┼───────────────┤
  │ F = 0.10           │ 7 sp + 1 ra │ 14 sp + 2 ra │ 29 sp + 3 ra  │
  ├────────────────────┼─────────────┼──────────────┼───────────────┤
  │ F = 0.25           │ 6 sp + 2 ra │ 12 sp + 4 ra │ 24 sp + 8 ra  │
  ├────────────────────┼─────────────┼──────────────┼───────────────┤
  │ F = 0.50           │ 4 sp + 4 ra │ 8 sp + 8 ra  │ 16 sp + 16 ra │
  └────────────────────┴─────────────┴──────────────┴───────────────┘

  With different reanalysis sims (e.g., N_sims_ra = N_sims / 2 — reanalysis needs fewer sims since it just updates
  policy, not explores deeply):

  W_ra = W * F / (F + (1-F) * α * N_sims_ra/N_sims)

  This means fewer reanalysis workers are needed if they use fewer sims, because each worker produces positions faster.

  Buffer size consideration: The buffer size constrains how many unique positions are available for reanalysis. If
  buffer_size < W_ra * positions_per_worker_per_iteration, workers will re-reanalyze the same positions — wasted GPU.
  So:

  W_ra ≤ buffer_size / (positions_per_worker_per_iteration)
       = buffer_size / (target_reanalyzed_per_iter / W_ra)

  Practically: ensure F * games_per_iter * avg_game_length ≤ buffer_size.

  2. Batch Size ≥ Total Workers × Search Batch

  The current code already auto-computes this:

  input_batch_size = round_up(workers * search_batch, 32)

  This is optimal for self-play only. When adding reanalysis workers, the formula should become:

  input_batch_size = round_up((W_sp + W_ra) * search_batch, 32)

  Why this matters:

  The GPU thread's collect_batch() has a spin-poll phase (~50μs stall window) where it waits for the batch to fill. If
  eval_batch < total_workers * search_batch:

  1. Some workers' leaves spill to the next batch → pipeline bubbles
  2. Workers wait on worker_wait_ms → throughput drops
  3. Queue backpressure triggers at 75% fill → cooldown stalls everyone

  If eval_batch > total_workers * search_batch:

  1. Batches never fully fill → CUDA graph padding waste
  2. The CUDA graph routing compensates (small/medium/large graphs), but you're always hitting the "padded" path

  The sweet spot is eval_batch = total_workers * search_batch — which gives full batches with zero padding. The
  32-alignment is for GPU memory coalescing.

  3. Why GPU Utilization Is Zero

  The system does not track actual NVIDIA GPU SM utilization (no nvidia-smi integration or CUDA profiling). What it
  tracks are proxy metrics. These are now persisted in training_log.jsonl alongside game/training metrics:

  ┌──────────────────────────┬─────────────────────────────────────┬────────────┐
  │        Metric            │         Where It Lives              │ Persisted? │
  ├──────────────────────────┼─────────────────────────────────────┼────────────┤
  │ batch_fill_ratio         │ training_log.jsonl + live dashboard │ Yes        │
  ├──────────────────────────┼─────────────────────────────────────┼────────────┤
  │ gpu_wait_ms              │ training_log.jsonl + live dashboard │ Yes        │
  ├──────────────────────────┼─────────────────────────────────────┼────────────┤
  │ worker_wait_ms           │ training_log.jsonl + live dashboard │ Yes        │
  ├──────────────────────────┼─────────────────────────────────────┼────────────┤
  │ batch_fill_p50/p90       │ training_log.jsonl + live dashboard │ Yes        │
  ├──────────────────────────┼─────────────────────────────────────┼────────────┤
  │ nn_evals_per_sec         │ training_log.jsonl + live dashboard │ Yes        │
  ├──────────────────────────┼─────────────────────────────────────┼────────────┤
  │ gpu_memory_allocated_mb  │ training_log.jsonl + live dashboard │ Yes        │
  ├──────────────────────────┼─────────────────────────────────────┼────────────┤
  │ cuda_graph_pct           │ training_log.jsonl + live dashboard │ Yes        │
  └──────────────────────────┴─────────────────────────────────────┴────────────┘

  The closest proxy for GPU utilization is batch_fill_ratio:
  - > 0.8 → GPU saturated (close to 100% utilized)
  - 0.3–0.8 → balanced
  - < 0.3 → GPU underutilized (workers can't feed it fast enough)

  To see actual SM utilization during training, you'd need to run nvidia-smi alongside training, or add a metric that
  samples torch.cuda.utilization().
  3. Add torch.cuda.utilization() sampling to get actual GPU SM % during training?


