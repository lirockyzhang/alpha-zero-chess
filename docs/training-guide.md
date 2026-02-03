# AlphaZero Training & Evaluation Guide

Complete guide for training and evaluating AlphaZero chess models using the C++ backend.

## Quick Start

### Training (Recommended)

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

```bash
# Evaluate against random player (quick sanity check)
uv run python alphazero/scripts/evaluate.py \
    --checkpoint checkpoints/cpp_iter_20.pt \
    --opponent random \
    --games 20 \
    --simulations 100

# Evaluate endgame positions
uv run python alphazero/scripts/evaluate.py \
    --checkpoint checkpoints/cpp_iter_20.pt \
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
| `--save-dir` | checkpoints | Checkpoint directory |
| `--resume` | - | Resume from checkpoint path |
| `--save-interval` | 5 | Save every N iterations |
| `--buffer-path` | replay_buffer/latest.rpbf | Replay buffer persistence |
| `--progress-interval` | 30 | Print performance stats every N seconds |

#### Dashboard Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--visual` | False | Enable HTML dashboard (auto-refresh, requires plotly) |
| `--live` | False | Enable LIVE web dashboard (real-time, requires flask) |
| `--dashboard-dir` | training_dashboard | Dashboard output directory |
| `--dashboard-interval` | 1 | Update dashboard every N iterations |
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
    --train-batch 128 \
    --save-dir checkpoints/test
```

**Expected output:**
- ~50 positions generated
- Training skipped (buffer too small)
- Checkpoint saved at `checkpoints/test/cpp_iter_2.pt`

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
    --buffer-size 50000 \
    --save-dir checkpoints/dev
```

**Expected results:**
- ~500 games generated
- Model should beat random player 80%+
- ~10-15 moves/sec during self-play

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
    --buffer-size 50000 \
    --save-dir checkpoints/dev
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
    --buffer-size 200000 \
    --save-dir checkpoints/serious
```

**Expected results:**
- ~2,500 games generated
- Model should beat random 95%+
- Basic tactical awareness

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
    --buffer-size 500000 \
    --save-dir checkpoints/production
```

**Expected results:**
- ~10,000 games generated
- Strong tactical play
- ~1500-1800 Elo estimate

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

To resume from a checkpoint:

```bash
uv run python alphazero-cpp/scripts/train.py \
    --resume checkpoints/cpp_iter_50.pt \
    --iterations 100
```

This will:
1. Load the network weights and optimizer state
2. Start from iteration 51
3. Load replay buffer from `replay_buffer/latest.rpbf` (if exists)

---

## Graceful Shutdown (Ctrl+C)

The training script supports **graceful shutdown**. When you press Ctrl+C:

1. **Finishes the current game** (doesn't interrupt mid-game)
2. **Saves an emergency checkpoint** to `checkpoints/cpp_iter_X_emergency.pt`
3. **Saves the replay buffer** to preserve all training data
4. **Prints resume instructions**

```bash
# Example: Press Ctrl+C during training
# Output:
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ! SHUTDOWN REQUESTED - Finishing current game and saving...
# ! Press Ctrl+C again to force quit (may lose data)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# To resume after graceful shutdown:
uv run python alphazero-cpp/scripts/train.py \
    --resume checkpoints/cpp_iter_15_emergency.pt \
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

## Training Dashboards

Two dashboard options are available for monitoring training progress:

### Option 1: Live Web Dashboard (`--live`) ⭐ Recommended

Real-time dashboard with **instant WebSocket updates**:

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

### Option 2: Static HTML Dashboard (`--visual`)

Generates HTML files that auto-refresh every 30 seconds:

```bash
uv run python alphazero-cpp/scripts/train.py \
    --iterations 50 \
    --games-per-iter 25 \
    --visual
```

This creates an HTML dashboard at `training_dashboard/dashboard.html`.

**Requirements:**
```bash
pip install plotly
```

### Using Both Together

You can enable both dashboards simultaneously:

```bash
uv run python alphazero-cpp/scripts/train.py \
    --iterations 50 \
    --live \
    --visual
```

This gives you:
- Real-time updates in the browser (`--live`)
- Saved HTML/JSON files for later analysis (`--visual`)

### Dashboard Features

The dashboard displays **9 interactive charts** tracking all key metrics:

| Chart | Metrics Shown |
|-------|---------------|
| 📉 Training Loss | Total loss over iterations |
| ⚡ Performance | Moves per second |
| 🎮 Games/Hour | Training throughput |
| 🎯 Policy vs Value Loss | Individual loss components |
| 🔬 MCTS Sims/s | Search efficiency |
| 📊 Win/Draw/Loss | Game outcome distribution |
| 💾 Buffer Size | Replay buffer growth |
| ⏱️ Time Breakdown | Self-play vs training time |
| 📈 Cumulative Progress | Total games and moves |

### Dashboard Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--visual` | False | Enable the dashboard |
| `--dashboard-dir` | training_dashboard | Output directory |
| `--dashboard-interval` | 1 | Update every N iterations |

### Example Output

```
Initializing visual dashboard...
  Dashboard enabled: training_dashboard/dashboard.html
  Open in browser: file:///path/to/training_dashboard/dashboard.html

Starting training...
```

### Generated Files

After training, you'll find:

```
training_dashboard/
├── dashboard.html    # Interactive charts (auto-refreshes)
├── summary.html      # Final training summary
└── metrics.json      # Raw data for external analysis
```

### Requirements

The dashboard requires **plotly**:

```bash
pip install plotly
# or with uv:
uv pip install plotly
```

If plotly is not installed, training continues without the dashboard.

### Viewing the Dashboard

1. **During training**: Open `dashboard.html` in your browser - it auto-refreshes every 30 seconds
2. **After training**: Open `summary.html` for a final report with key statistics

```bash
# On Windows
start training_dashboard/dashboard.html

# On macOS
open training_dashboard/dashboard.html

# On Linux
xdg-open training_dashboard/dashboard.html
```

---

## Evaluation Script Reference

**Location**: `alphazero/scripts/evaluate.py`

### Quick Evaluation Examples

```bash
# Against random player
uv run python alphazero/scripts/evaluate.py \
    --checkpoint checkpoints/cpp_iter_50.pt \
    --opponent random \
    --games 50 \
    --simulations 200

# Against endgame positions
uv run python alphazero/scripts/evaluate.py \
    --checkpoint checkpoints/cpp_iter_50.pt \
    --opponent endgame \
    --simulations 400

# Against Stockfish (if installed)
uv run python alphazero/scripts/evaluate.py \
    --checkpoint checkpoints/cpp_iter_50.pt \
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
# Using C++ checkpoint
uv run python web/run.py \
    --checkpoint checkpoints/cpp_iter_50.pt \
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

# Evaluate
evaluate.py --checkpoint model.pt --opponent random --games 50
```

### Parameter Quick Reference

| When Used | Parameter | Default | Purpose |
|-----------|-----------|---------|---------|
| MCTS Search | `--search-batch` | 64 | Leaves per MCTS GPU call |
| Parallel Eval | `--eval-batch` | 512 | Max leaves across all games |
| Training | `--train-batch` | 256 | Samples per gradient step |

---

**Last Updated**: 2026-02-03
