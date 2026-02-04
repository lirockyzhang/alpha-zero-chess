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
├── training_metrics.json        # Loss, games, moves per iteration
├── evaluation_results.json      # vs_random win rate, endgame scores
├── summary.html                 # Visual training summary (default)
└── replay_buffer.rpbf           # Only if --buffer-persistence
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

# Production training (~12-24 hours)
uv run python alphazero-cpp/scripts/train.py \
    --iterations 100 \
    --games-per-iter 50 \
    --simulations 800 \
    --filters 192 \
    --blocks 15
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
| `--search-batch` | 64 | Leaves to evaluate per MCTS iteration |
| `--c-puct` | 1.5 | MCTS exploration constant |

#### Neural Network Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--filters` | 192 | Network filter count |
| `--blocks` | 15 | Residual block count |
| `--eval-batch` | 512 | Max positions per GPU call in parallel mode |
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
| `--save-interval` | 5 | Save every N iterations |
| `--progress-interval` | 30 | Print performance stats every N seconds |

#### Buffer Persistence
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--buffer-persistence` | False | Save/load replay buffer between runs (default: fresh buffer each run) |

#### Visualization & Evaluation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--no-visualization` | False | Disable summary.html generation (enabled by default) |
| `--no-eval` | False | Skip evaluation before checkpoint (enabled by default) |
| `--live` | False | Enable LIVE web dashboard (real-time, requires flask) |
| `--dashboard-port` | 5000 | Port for live dashboard server |

### Architecture (AlphaZero Paper Standard)

The training script uses the optimal architecture:
- **Input channels**: 122 (full AlphaZero encoding)
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
│     │     --simulations, --c-puct                                     │  │
│     │                                                                 │  │
│     │     Repeat until simulations complete:                          │  │
│     │     ┌──────────────────────────────────────────────────────┐   │  │
│     │     │  3. NEURAL NETWORK EVALUATION                        │   │  │
│     │     │     --search-batch (leaves per iteration)            │   │  │
│     │     │     --eval-batch (GPU batch in parallel mode)        │   │  │
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

### MCTS Settings (`--simulations`, `--search-batch`)

| Simulations | Speed | Strength | Recommended For |
|-------------|-------|----------|-----------------|
| 100 | Fast | Weak | Testing, debugging |
| 400 | Medium | Decent | Development |
| 800 | Slow | Strong | Production (paper default) |
| 1600 | Very slow | Strongest | Evaluation only |

**Search batch (`--search-batch`)**: Controls how many leaf positions are collected before sending to GPU for evaluation.

```
MCTS with --simulations 800 and --search-batch 64:

Iteration 1: Collect 64 leaves → GPU eval → Backprop → 64 sims done
Iteration 2: Collect 64 leaves → GPU eval → Backprop → 128 sims done
...
Iteration 12-13: → 800 simulations complete

GPU calls per move: ~13 (800 ÷ 64)
```

| search-batch | GPU Calls/Move | GPU Utilization | Best For |
|--------------|----------------|-----------------|----------|
| 16 | Many (~50) | Low (~30%) | Very low VRAM |
| 32 | Moderate (~25) | Medium (~50%) | 4GB GPUs |
| **64** | Few (~13) | Good (~70%) | **Default** (8GB) |
| 128 | Very few (~7) | High (~85%) | 12GB+ GPUs |

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
| `--eval-batch 512` | Collect up to 512 leaves from ALL games before GPU call |
| `--batch-timeout-ms 5` | Don't wait longer than 5ms for batch to fill |
| `--queue-capacity 8192` | Evaluation queue capacity (increase for many workers) |
| `--root-eval-retries 3` | Max retries for root NN eval timeout before giving up |
| `--worker-timeout-ms 2000` | Worker wait time for NN results (per attempt) |

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

### Exploration (`--temperature-moves`, `--c-puct`)

- `--temperature-moves 30`: First 30 moves use temperature=1 (exploration), then greedy
- `--c-puct 1.5`: PUCT exploration constant (higher = more exploration)

---

## Parameter Tuning Guide

This section explains how each parameter affects training and how to tune them for your hardware and goals.

### Understanding `--search-batch` (Critical for Performance)

The `--search-batch` parameter controls **how many leaf nodes are evaluated in a single GPU batch** during MCTS search. This is one of the most important parameters for performance.

```
MCTS Search Flow:
┌─────────────────────────────────────────────────────────────────┐
│  1. Select leaves (C++ tree traversal)                          │
│  2. Collect up to search-batch leaves                           │
│  3. Send batch to GPU for neural network evaluation             │
│  4. Backpropagate results (C++ tree update)                     │
│  5. Repeat until simulations complete                           │
└─────────────────────────────────────────────────────────────────┘
```

**How it affects performance:**

| search-batch | GPU Calls per Move | GPU Utilization | Latency | Best For |
|--------------|-------------------|-----------------|---------|----------|
| 8 | Many (100+) | Low (~20%) | High | Not recommended |
| 32 | Moderate (25-50) | Medium (~50%) | Medium | Low VRAM GPUs (4GB) |
| 64 | Few (12-25) | Good (~70%) | Low | **Default** (8GB GPUs) |
| 128 | Very few (6-12) | High (~85%) | Very low | High-end GPUs (12GB+) |
| 256 | Minimal (3-6) | Highest (~90%) | Minimal | Multi-GPU / A100 |

**Tuning strategy:**
```bash
# Start with default
--search-batch 64

# If you see "CUDA out of memory", reduce:
--search-batch 32

# If GPU utilization is low (check with nvidia-smi), increase:
--search-batch 128
```

**Relationship with `--simulations`:**
- `search-batch` should be ≤ `simulations / 4` for efficiency
- Example: 800 simulations → use search-batch 32-128
- Example: 200 simulations → use search-batch 16-64

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

### Understanding `--c-puct`

Controls exploration vs exploitation in MCTS:

```
Higher c-puct = More exploration (try new moves)
              = Better for early training
              = May waste simulations on bad moves

Lower c-puct = More exploitation (trust the network)
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

### Parameter Combinations by Goal

**Goal: Maximum Training Speed (development)**
```bash
--simulations 200 \
--search-batch 64 \
--games-per-iter 10 \
--train-batch 256 \
--epochs 3
```

**Goal: Maximum Training Speed with Parallel (2-5x faster)**
```bash
--simulations 400 \
--workers 16 \
--games-per-iter 64 \
--search-batch 64 \
--eval-batch 512 \
--train-batch 256
```

**Goal: Maximum Model Quality (production)**
```bash
--simulations 800 \
--search-batch 64 \
--games-per-iter 100 \
--train-batch 1024 \
--epochs 5 \
--buffer-size 500000
```

**Goal: Low VRAM (4-6 GB GPU)**
```bash
--simulations 400 \
--search-batch 32 \
--filters 64 \
--blocks 5 \
--train-batch 128 \
--buffer-size 50000
```

**Goal: Debugging / Testing**
```bash
--simulations 50 \
--search-batch 16 \
--games-per-iter 2 \
--iterations 1
```

### Monitoring and Adjusting

Watch these metrics during training:

| Metric | Healthy Range | If Too Low | If Too High |
|--------|---------------|------------|-------------|
| moves/sec | 10-50 | Increase search-batch or use --workers | Expected with larger networks |
| GPU util | 60-90% | Use --workers 16 or increase search-batch | Expected (good!) |
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
3. Load `training_metrics.json` and append new metrics
4. Load `evaluation_results.json` and append new evaluations
5. Load replay buffer (only if `--buffer-persistence` is set)
6. Continue saving to the same run directory


## Graceful Shutdown (Ctrl+C)

The training script supports **graceful shutdown**. When you press Ctrl+C:

1. **Finishes the current game** (doesn't interrupt mid-game)
2. **Saves an emergency checkpoint** to `{run_dir}/model_iter_XXX_emergency.pt`
3. **Saves the replay buffer** (only if `--buffer-persistence` is enabled)
4. **Saves training metrics** to `training_metrics.json`
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
├── training_metrics.json     # Per-iteration metrics (loss, games, etc.)
├── evaluation_results.json   # Evaluation metrics per checkpoint
├── summary.html              # Visual summary with charts
└── replay_buffer.rpbf        # Only if --buffer-persistence
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
2. `--search-batch` (e.g., 32 instead of 64)
3. `--eval-batch` (e.g., 256 instead of 512, for parallel mode)
4. Network size (`--filters`, `--blocks`)

### Training Too Slow

1. Use parallel self-play: `--workers 16`
2. Use smaller network for development
3. Reduce `--simulations` (400 instead of 800)
4. Ensure CUDA is being used (`--device cuda`)

### Low GPU Utilization

1. Enable parallel self-play: `--workers 16 --eval-batch 512`
2. Increase `--search-batch` (e.g., 128 instead of 64)
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
- **122-channel input** (C++ encoding)
- **AlphaZero paper architecture** (2 policy filters, 256 value hidden)
- **Compatible with**: Web interface, evaluation script, Python MCTS

The checkpoint format includes:
```python
{
    'iteration': int,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'config': {
        'input_channels': 122,
        'num_filters': int,
        'num_blocks': int,
        'policy_filters': 2,
        'value_hidden': 256,
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

# Disable evaluation and visualization (faster checkpointing)
train.py --iterations 50 --no-eval --no-visualization

# Evaluate
evaluate.py --checkpoint checkpoints/f192-b15_.../model_final.pt --opponent random --games 50
```

### Parameter Quick Reference

| When Used | Parameter | Default | Purpose |
|-----------|-----------|---------|---------|
| MCTS Search | `--search-batch` | 64 | Leaves per MCTS GPU call |
| Parallel Eval | `--eval-batch` | 512 | Max leaves across all games |
| Training | `--train-batch` | 256 | Samples per gradient step |

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

**Last Updated**: 2026-02-03
