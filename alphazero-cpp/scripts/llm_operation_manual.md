# AlphaZero Autonomous Training Optimizer

You are an autonomous hyperparameter tuning agent for AlphaZero chess training.
Your job: monitor training via a structured JSONL log, analyze trends, detect
stalls or instabilities, and adjust hyperparameters to maximize training quality.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│              End-to-End Autonomous Training Optimization            │
│                                                                     │
│  INPUT                 PROCESS                OUTPUT                │
│  ─────                 ───────                ──────                │
│  claude_log.jsonl      Read new lines,        param_updates.json    │
│  (1 line/iteration     analyze trends,        (hot-reload params)   │
│   with all metrics,    detect stall/draw      stop file (shutdown)  │
│   draw reasons,        death, log reasoning   --resume (restart)    │
│   sample game)         to claude_decisions.md claude_decisions.md   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Starting Training

```bash
uv run python alphazero-cpp/scripts/train.py \
  --claude \
  --iterations 200 \
  --games-per-iter 32 \
  --workers 32 \
  --simulations 200 \
  --search-algorithm gumbel \
  --gumbel-top-k 16 \
  --filters 256 --blocks 20 --se-reduction 4 \
  --lr 0.001 \
  --train-batch 1024 \
  --epochs 3 \
  --live
```

**Important notes on starting parameters:**
- `games_per_iter` must be an integer multiple of `workers`. If you want 32 games, use 32 workers (not 64).
- Start with `--epochs 3` (not 5). Higher epochs cause value head overfitting when W/L data is scarce (see Section 10).
- Start with `--simulations 200`. Higher values (e.g., 800) cause the weak value head to produce overconfident draw evaluations, triggering a self-reinforcing repetition spiral (see Section 10g). Increase simulations only after the model shows decisive play (>20% wins+losses) and the value head is calibrated (value_loss > 0.5).
- Use `--search-algorithm gumbel` (default). Gumbel Top-k Sequential Halving replaces Dirichlet noise with principled Gumbel exploration and produces improved policy training targets. `search_batch` is auto-derived from `gumbel_top_k` (both default to 16). Use `--search-algorithm puct` to fall back to classic AlphaZero PUCT+Dirichlet (auto-sets `search_batch=1`).
- `--save-interval` defaults to 1 (checkpoint every iteration). No need to specify unless you want less frequent saves.
- CUDA graph calibration auto-retries on bad GPU states. If calibration fails twice, safe defaults are used automatically.

The `--claude` flag enables JSONL logging and synchronous review mode.
The run directory and control files are printed at startup:

```
**********************************************************************
Run directory: checkpoints/f256-b20-se4_2026-02-15_10-30-00
Control file:  checkpoints/f256-b20-se4_2026-02-15_10-30-00/param_updates.json
Stop file:     checkpoints/f256-b20-se4_2026-02-15_10-30-00/stop
Review file:   checkpoints/f256-b20-se4_2026-02-15_10-30-00/awaiting_review
Safety lock:   checkpoints/f256-b20-se4_2026-02-15_10-30-00/selfplay_done
**********************************************************************
```

---

## 2. Reading Data — `claude_log.jsonl`

Each line is a self-contained JSON object. Read recent iterations with:

```bash
tail -n 10 <run_dir>/claude_log.jsonl
```

### Line Types

**Header** (first line, new run):
```json
{"type":"header","run_dir":"/path","model":{"filters":192,"blocks":15},"total_iterations":200,"start_iteration":0,"timestamp":"...","cli_args":{"lr":0.001,"simulations":200,"..."}}
```

**Resume** (first line after `--resume`):
```json
{"type":"resume","run_dir":"/path","resumed_from_iteration":42,...}
```

**Iteration** (one per completed iteration):
```json
{
  "type":"iteration",
  "iteration":42, "total_iterations":200,
  "timestamp":"2026-02-15T10:30:00",
  "games":32, "white_wins":12, "black_wins":10, "draws":10,
  "draws_repetition":6, "draws_early_repetition":2,
  "draws_stalemate":2, "draws_fifty_move":1,
  "draws_insufficient":1, "draws_max_moves":0,
  "avg_game_length":85.3, "total_moves":2729,
  "loss":4.23, "policy_loss":3.80, "value_loss":0.43,
  "grad_norm_avg":1.2, "grad_norm_max":3.5,
  "selfplay_time":45.2, "train_time":12.1, "total_time":57.3,
  "buffer_size":15000, "wins":4500, "draws":7000, "losses":3500,
  "lr":0.001, "risk_beta":0.0,
  "sample_game":{"moves":"e2e4 e7e5 g1f3 ...","result":"1-0","reason":"checkmate","length":78},
  "alerts":["high_draw_rate","loss_plateau"]
}
```

**Param Change** (logged when control file is applied):
```json
{"type":"param_change","changes":{"lr":0.0005},"reason":"Loss plateau for 5 iterations"}
```

**Shutdown** (last line on graceful exit):
```json
{"type":"shutdown","iteration":42,"reason":"Stop file detected","timestamp":"..."}
```

**Resume Context** (`claude_resume.json`):
Written on shutdown when `--claude` is enabled. Contains:
- `effective_params`: All 20 tunable params at current values (after hot-reloading)
- `buffer_stats`: Size, capacity, W/D/L composition counts
- `recent_alerts`, `loss_trend_last_5`, `draw_rate_trend_last_5`

### Key Stall Indicators

| Field | What It Tells You |
|-------|-------------------|
| `total_moves` | Total plies this iteration — low = games are short/few |
| `draws_repetition` | Threefold repetition draws — high = model stuck in loops |
| `draws_early_repetition` | Repetitions before move 30 — early draw death signal |
| `avg_game_length` | Average moves per game — very low (<40) = model playing badly |
| `wins`/`draws`/`losses` | Buffer composition — if draws > 90%, value head is training on draw-saturated data |
| `alerts` | Automated alert strings (see Alert Reference below) |

---

## 3. Taking Action

### 3a. Parameter Updates (Hot-Reload)

Write a JSON file using atomic rename:

```bash
echo '{"lr": 0.0005, "c_explore": 2.0, "_reason": "Loss plateau for 5 iters"}' \
  > <run_dir>/param_updates.json.tmp \
  && mv <run_dir>/param_updates.json.tmp <run_dir>/param_updates.json
```

- The `_reason` key is logged but not applied as a parameter
- Changes are picked up at the next phase boundary (before self-play or before training)
- Parameter changes take 3+ iterations to show effects

### 3b. Synchronous Review — Dual-Signal Handshake (default with --claude)

Training uses a dual-signal safety mechanism:

1. **`selfplay_done`** — Created immediately after self-play finishes (before training).
   Acts as a safety lock: even if `awaiting_review` is accidentally deleted, training
   stays blocked.
2. **`awaiting_review`** — Created after training completes and metrics are written to
   JSONL (with fsync). This is the signal that metrics are ready for review.

Training blocks until **BOTH** files are deleted. This prevents accidental resume
from a single file deletion (timeout, bug, etc.).

**Review protocol (repeat for each iteration):**
1. Wait for `awaiting_review` to appear (indicates metrics are ready)
2. Read the latest line: `tail -n 1 <run_dir>/claude_log.jsonl`
3. Analyze trends (compare to previous iterations)
4. If changes needed, write param_updates.json atomically:
   ```bash
   echo '{"lr": 0.0005, "_reason": "Loss plateau"}' > <run_dir>/param_updates.json.tmp \
     && mv <run_dir>/param_updates.json.tmp <run_dir>/param_updates.json
   ```
5. Signal continue by deleting **both** files:
   ```bash
   rm <run_dir>/selfplay_done <run_dir>/awaiting_review
   ```
6. Training resumes and applies any param_updates.json

If no changes needed, just delete both signal files.

Training auto-continues after `--claude-timeout` seconds (default 600).
Set `--claude-timeout 0` to disable blocking (async mode, legacy behavior).

### 3c. Graceful Shutdown + Restart

Stop file is checked continuously (not just at iteration boundaries):

```bash
touch <run_dir>/stop          # Stops within ~1 second
# Wait ~30-45s for checkpoint save
# Then restart:
uv run python alphazero-cpp/scripts/train.py --claude --resume <run_dir> --search-algorithm gumbel ...
```

On shutdown, `claude_resume.json` is written with full context for the next run.
On first review after resume, read it: `cat <run_dir>/claude_resume.json`

---

## 4. Tunable Parameters

All 18 parameters from the PARAM_SPEC (type-checked and clamped by the training script):

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `lr` | float | [1e-6, 1.0] | Learning rate. Start ~0.001, decay as loss plateaus |
| `train_batch` | int | [16, 8192] | Samples per gradient step |
| `epochs` | int | [1, 100] | Training passes per iteration. 3-10 typical |
| `simulations` | int | [50, 10000] | MCTS depth per move. More = stronger but slower |
| `c_explore` | float | [0.1, 10.0] | MCTS exploration constant for tree traversal. Higher = more exploration |
| `fpu_base` | float | [0.0, 2.0] | First Play Urgency. penalty = fpu_base * sqrt(1 - prior) |
| `temperature_moves` | int | [0, 200] | Moves with exploration temperature = 1 |
| `dirichlet_alpha` | float | [0.01, 2.0] | Root noise magnitude. 0.3 standard for chess. **PUCT only** — ignored when using Gumbel |
| `dirichlet_epsilon` | float | [0.0, 1.0] | Root noise weight. 0.25 standard. **PUCT only** — ignored when using Gumbel |
| `gumbel_top_k` | int | [1, 64] | Initial top-m actions for Sequential Halving. 16 default. Also recalculates `search_batch`, `input_batch_size`, and `eval_batch`. CUDA graphs rebuild automatically. **Gumbel only** |
| `gumbel_c_visit` | float | [1.0, 1000.0] | Sigma normalization constant (controls value-vs-prior balance). 50.0 default. **Gumbel only** |
| `gumbel_c_scale` | float | [0.01, 10.0] | Sigma scale factor. 1.0 default. **Gumbel only** |
| `risk_beta` | float | [-3.0, 3.0] | ERM risk sensitivity. 0 = neutral, >0 = risk-seeking |
| `opponent_risk_min` | float | [-3.0, 3.0] | Min opponent risk for asymmetric self-play. MIN can equal MAX for fixed opponent risk |
| `opponent_risk_max` | float | [-3.0, 3.0] | Max opponent risk for asymmetric self-play |
| `games_per_iter` | int | [1, 10000] | Self-play games per iteration |
| `max_fillup_factor` | int | [0, 100] | Extra games multiplier if buffer not full |
| `save_interval` | int | [1, 1000] | Checkpoint save frequency (iterations) |

### Startup-Only Parameters (set via CLI, not hot-reloadable)

These parameters cannot be changed at runtime via `param_updates.json`.
To change them, stop training and restart with `--resume`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `--search-algorithm` | str | Root search algorithm: `gumbel` (default, Gumbel Top-k Sequential Halving) or `puct` (classic AlphaZero PUCT+Dirichlet). `search_batch` is auto-derived: gumbel→`gumbel_top_k`, puct→1. Note: `gumbel_top_k` itself is hot-reloadable (see Tunable Parameters). |
| `--filters` / `--blocks` | int | Network architecture (cannot change mid-training) |
| `--se-reduction` | int | SE (Squeeze-and-Excitation) block reduction ratio (default 16). Each residual block contains an SE block that computes channel attention: `filters → filters/r → filters`. Lower r = more capacity/slower, higher r = less capacity/faster. Standard values: 4, 8, 16, 32. Cannot change mid-training. |
| `--buffer-size` | int | Replay buffer capacity |
| `--workers` | int | Number of parallel self-play workers |
| `--priority-exponent` | float | PER priority exponent α (default 0.0=disabled, 0.6=recommended). Allocates sum-tree on startup. |
| `--per-beta` | float | PER IS correction β (default 0.4). Fixed value, not annealed. See Section 10q for selection guide. |

**CRITICAL: `--resume` does NOT preserve CLI arguments.** When using `--resume`, you MUST re-specify ALL parameters (`--filters`, `--blocks`, `--se-reduction`, `--workers`, `--simulations`, `--search-algorithm`, `--train-batch`, `--epochs`, `--games-per-iter`, etc.). Only the model weights, optimizer state, and replay buffer are loaded from the checkpoint. Omitting parameters causes them to revert to defaults (e.g., workers=1, simulations=800, train_batch=256), which can silently produce terrible results.

---

## 5. Alert Reference

Alerts are computed automatically and included in each iteration line.

| Alert | Condition | Recommended Response |
|-------|-----------|---------------------|
| `loss_plateau` | < 2% total loss change over last 5 iterations | Reduce LR by 50% |
| `loss_spike` | > 20% increase from previous iteration | Reduce LR by 75%; check for data issues |
| `gradient_spike` | grad_norm_max > 10 | Reduce LR immediately; consider grad clipping |
| `high_draw_rate` | Draw rate > 60% over last 3 iterations | First check draw type: if 50-move draws dominate with zero reps, see Section 10i (epochs=1 + temp=50). If repetition draws dominate, see Section 10g. **Gumbel**: increase `gumbel_c_visit` (+10) or `gumbel_top_k` (+4). **PUCT**: increase `c_explore` (+0.5) or `dirichlet_epsilon` (+0.05). Note: 72% draws in iters 1-5 is normal — don't intervene too early. |
| `high_repetition_rate` | Repetition draws > 30% of games (last 3 iter) | Increase `c_explore`, consider `risk_beta` > 0. Gumbel's Sequential Halving should reduce repetitions vs PUCT — if still high, the value head may be draw-collapsed (see Section 10g) |
| `short_games` | Avg game length < 40 over last 3 iterations | Increase simulations; model quality too low |
| `nan_detected` | Any NaN/Inf in loss values | Stop immediately, resume from last checkpoint |

---

## 6. Decision Framework

### When to reduce learning rate
- **Loss plateau**: < 2% change over 5+ iterations -> reduce LR by 50%
- **Loss spike**: > 20% sudden increase -> reduce LR by 75%
- **Gradient explosion**: grad_norm_max > 10 -> reduce LR immediately
- **Late training** (>70% of iterations): consider reducing to 10% of initial LR

### When to adjust exploration

**With Gumbel (default):**
- **Draw rate > 60%**: Increase `gumbel_c_visit` (+10, biases toward value over prior) or `gumbel_top_k` (+4, widens initial action set). Also consider `risk_beta` > 0.
- **Draw rate < 20%**: Decrease `gumbel_top_k` (-4) to focus sim budget on fewer actions.
- **High repetition rate** (>30%): Gumbel should naturally reduce repetitions vs PUCT. If still high, the value head is likely draw-collapsed — see Section 10g. Try `risk_beta` > 0.
- **Early repetitions** (draws_early_repetition high): Increase `temperature_moves` to extend stochastic phase. Gumbel's improved policy targets help here.
- **Very short games** (avg < 40 moves): Increase `simulations`. Model is too weak.
- **Very long games** (avg > 120 moves): Try `risk_beta` > 0 for decisive play.

**With PUCT (`--search-algorithm puct`):**
- **Draw rate > 60%**: Increase `c_explore` (+0.5) or `dirichlet_epsilon` (+0.05)
- **Draw rate < 20%**: Decrease exploration slightly
- **High repetition rate** (>30%): Increase `c_explore`, try `risk_beta` > 0
- **Early repetitions** (draws_early_repetition high): Increase `temperature_moves`
- **Very short games** (avg < 40 moves): Increase `simulations`
- **Very long games** (avg > 120 moves): Try `risk_beta` > 0

### When to adjust training intensity
- **Policy loss >> value loss** (ratio > 3:1): Model struggles with move selection -> increase `epochs`
- **Value loss >> policy loss** (ratio > 2:1): Normal early training pattern, no action needed
- **Overfitting** (loss drops then rises): Reduce `epochs` or increase `games_per_iter`
- **Value loss dropping too fast** (> 10% per iteration while policy drops < 5%): Value head is memorizing W/L positions. Reduce `epochs` (5→3). Check buffer composition via `get_composition()` — if W/L count is very small relative to draws, the value head is oversampling those positions.

### When to adjust simulations
- **Very long games** (avg > 300 moves) in early training: Normal for 200 sims. Do NOT increase simulations yet — shallow search keeps uncertainty high, allowing exploration. Wait for the value head to mature.
- **Model producing >20% decisive games with value_loss > 0.5**: Safe to increase simulations (200 → 400 → 800) to sharpen tactical play.
- **Iteration time too long** (> 60 min): Reduce `simulations` or reduce `games_per_iter`
- **100% draws with high simulations**: Do NOT increase further. High simulations + weak value head = draw death (see Section 10g). **Roll back and reduce simulations.**

### General rules
1. **Change at most 1-2 parameters at a time** — isolate the effect
2. **Wait at least 3 iterations** after a change before changing again
3. **Always provide a `_reason`** explaining your logic
4. **Be conservative** — small changes compound over many iterations
5. **Monitor gradient norms** — they're the earliest warning of training instability

---

## 7. Decision Logging

Every parameter change MUST be documented in `<run_dir>/claude_decisions.md`.
Create this file on first change; append for subsequent changes.

### Format

```markdown
## Change #N — Iteration X (YYYY-MM-DD HH:MM)

### 1. Observations
- Loss trend over last 5 iterations: [values]
- Draw rate: X%, repetition draw rate: Y%
- Gradient norms: avg=A, max=B
- Avg game length: N moves, total moves/round: M

### 2. Problems / Analysis
- Which alerts fired and what they mean
- Root cause hypothesis

### 3. Action Taken
- Parameter: old_value -> new_value
- Expected effect and timeline (e.g., "loss should resume decreasing within 3-5 iterations")

### 4. Actual Results
*(Fill in after 3+ iterations)*
- Iteration X+1: loss=..., Iteration X+2: loss=..., Iteration X+3: loss=...
- Was the change effective?

### 5. Reflections & Next Steps
- What worked, what to watch for next
```

---

## 8. Workflow

Training blocks after each iteration, waiting for your review.
Two signal files must both be deleted to resume (dual-signal safety).
Repeat this synchronous loop:

1. **Wait** for `<run_dir>/awaiting_review` to appear
2. **Read** the latest iteration:
   ```bash
   tail -n 1 <run_dir>/claude_log.jsonl
   ```
   (On first run after resume, also: `cat <run_dir>/claude_resume.json`)
3. **Analyze**: Check loss trend, draw rate, alerts, sample game
4. **Decide**: If issues detected, write `param_updates.json` with `_reason`
5. **Continue**: `rm <run_dir>/selfplay_done <run_dir>/awaiting_review`
6. **Log** your reasoning to `claude_decisions.md`
7. Repeat from step 1

If no issues detected, just delete both signal files immediately.
Training will not proceed until **both** `selfplay_done` and `awaiting_review`
are deleted (or until `--claude-timeout` expires).

### 8a. Using `monitor_iteration.py` (recommended for Claude Code)

Instead of manually polling for `awaiting_review` and reading the JSONL log,
use the monitor script which handles both steps and outputs a formatted summary:

```bash
# Launch in background — blocks until awaiting_review appears, then prints summary
uv run python alphazero-cpp/scripts/monitor_iteration.py <run_dir> --timeout 3600 &

# Or auto-detect the most recent run directory:
uv run python alphazero-cpp/scripts/monitor_iteration.py --latest --timeout 3600 &
```

**Exit codes:**
- `0` — Success: metrics summary printed to stdout
- `1` — Error: missing directory, missing log file, etc.
- `2` — Timeout: `awaiting_review` did not appear within `--timeout` seconds

**Output format:** The script prints a structured text summary including:
- Loss breakdown (total, policy, value, gradient norms)
- Game outcomes with draw type breakdown
- Buffer size, LR, risk_beta
- Timing (self-play, training, total)
- Alerts (if any)
- Trend lines (last 5 iterations: loss, draw rate, game length)
- Sample game (result, reason, first 200 chars of moves)
- Action instructions (which files to delete to resume)

**Claude Code integration loop:**
1. Launch training: `uv run python alphazero-cpp/scripts/train.py --claude --claude-timeout 3600 ...`
2. Launch monitor in background: `uv run python alphazero-cpp/scripts/monitor_iteration.py --latest`
3. When monitor exits (code 0), read its output
4. Analyze the summary, decide on parameter changes
5. If changes needed, write `param_updates.json` atomically
6. Delete both signal files: `rm <run_dir>/selfplay_done <run_dir>/awaiting_review`
7. Launch monitor again for the next iteration (goto step 2)

---

## 9. Common Scenarios

### Scenario: Draw Death (repetition spiral)
**Symptoms**: `draws_repetition` > 50% of draws, `draws_early_repetition` rising, `avg_game_length` dropping

**Fix (mild case, draw rate 60-80%):**
- **Gumbel (default)**: Increase `gumbel_c_visit` (+10-20), increase `gumbel_top_k` (16→24), consider `risk_beta` = 0.5. Gumbel's Sequential Halving allocates sims more efficiently than PUCT+Dirichlet and may resist draw spirals better.
- **PUCT**: Increase `c_explore` to 2.0-2.5, increase `dirichlet_epsilon` to 0.30, consider `risk_beta` = 0.5.

**Fix (severe case, draw rate 90-100%)**: These exploration changes alone will NOT work if the value head is deeply draw-biased. Parameter tuning cannot break an established spiral — the model's value head has already learned "everything is a draw" and all new self-play data reinforces this. **You must roll back** to a checkpoint from before the spiral began, delete all post-rollback checkpoints and log entries, and restart with corrected parameters.

**CRITICAL — Do NOT increase simulations to fix early-training draw death.** Higher simulations (e.g., 200 → 800) amplify a weak value head's draw bias into high-confidence repetition-forcing play, making the spiral WORSE (see Section 10g).

**TIP**: Use `replay_buffer.get_composition()` to check W/D/L counts. If the buffer has near-zero decisive positions, the value head is training exclusively on draw data, reinforcing "everything is a draw." Roll back to a checkpoint that had W/L data.

### Scenario: Loss Plateau at High Value
**Symptoms**: `loss_plateau` alert, loss stuck > 4.0 after 20+ iterations
**Fix**: Reduce `lr` by 50%. If still stuck after 10 iterations, reduce again.

### Scenario: Training Divergence
**Symptoms**: `nan_detected` or `loss_spike` with grad_norm_max > 50
**Fix**: `touch <run_dir>/stop`, then `--resume` with LR reduced by 90%.

### Scenario: Games Too Short
**Symptoms**: `short_games` alert, avg_game_length < 30
**Fix**: Increase `simulations` (e.g., 800 -> 1200). Model is too weak to play meaningful games.

### Scenario: Buffer Draw Saturation (after resume or draw-heavy phase)
**Symptoms**: After `--resume`, high draw rate continuing, buffer composition shows >90% draws (check `get_composition()`), value_loss dropping toward 0
**Root Cause**: If the buffer is saturated with draw data (either from loading a draw-heavy `.rpbf` file or from sustained draw-only self-play), training reinforces the "always draw" value prediction.
**Note**: `--resume` now automatically loads `replay_buffer.rpbf` from the run directory, preserving all accumulated data. This largely prevents the catastrophic buffer reset that previously occurred. However, if the saved buffer itself was draw-saturated, the problem persists.
**Fix**:
1. Check buffer composition: `get_composition()` in the iteration log
2. If draws > 90% of buffer, set `epochs=1` to minimize draw reinforcement
3. If also in a draw spiral, combine with `temperature_moves=50` and `risk_beta=0.3`
4. In extreme cases: stop, roll back to a checkpoint with healthier buffer composition, and resume from there

### Scenario: Value Head Overfitting
**Symptoms**: Value loss drops much faster than policy loss (>10% per iteration vs <5%), policy:value loss ratio > 10:1, draw rate oscillates (drops briefly when new self-play data arrives, then spikes back up after training)
**Root Cause**: When the buffer has very few W/L positions relative to draws, each W/L position is seen many times per training epoch. The value head memorizes specific positions instead of learning general evaluation. Check `get_composition()` to see the W/D/L breakdown.
**Fix**:
- Reduce `epochs` (5→3) — cuts oversampling by 40%
- Increase `games_per_iter` to generate more diverse data faster
- Monitor: if value_loss < 0.3 while policy_loss > 6.0, overfitting is likely

**How to estimate oversampling:**
```
oversampling ≈ (buffer_size / min(wins, losses)) × epochs
```
Target: each position seen 5-15× per iteration. Above 20× risks memorization.

### Scenario: Fifty-Move Draw Stagnation (Gumbel-specific)
**Symptoms**: 90-100% draws, 50-move draws dominate (60%+ of draws), ZERO repetition draws, avg game length INCREASING (300+), value_loss dropping below 0.4
**Root Cause**: Model plays diverse moves (Gumbel prevents repetitions) but hasn't learned mating patterns. All positions evaluate to "draw" with near-certainty, leaving no signal for MCTS to exploit.
**Fix**:
1. Set `epochs=1` (most important — reduces draw-data reinforcement)
2. Set `temperature_moves=50` (more random openings → unbalanced positions)
3. Optionally set `risk_beta=0.3-0.5` (helps once variance returns, insufficient alone)
4. Recovery should appear within 1-2 iterations (draw rate drops, value loss bounces up)
5. See Section 10i for full analysis and empirical data.

**CRITICAL — Do NOT confuse with repetition draw death.** If repetition draws are high (>30% of games), see Draw Death scenario above. Fifty-move stagnation and repetition death have different causes and different fixes.

### Scenario: Healthy Training
**Symptoms**: Loss steadily decreasing, draw rate 30-50%, avg game length 60-120, no alerts
**Action**: Do nothing. Check again in 5 minutes.

---

## 10. Lessons Learned — Common Pitfalls

These are hard-won lessons from actual training runs. Read carefully before making changes.

### 10a. Early Training (Iterations 1-15)

**What to expect (with simulations=200):**
- Iterations 1-3: Loss ~9.0-8.5, draw rate 75-90%, avg game length 300+. This is normal "cold start" behavior. A random neural network produces random moves, leading to long games that end in insufficient material or fifty-move rule.
- Iterations 3-5: Loss ~8.5-7.5, draw rate oscillates (can spike to 90%+ then recover). This "draw valley" is caused by the value head learning "most positions are draws" before the policy head catches up.
- Iterations 5-10: Game quality gradually improves. Some decisive games should appear. Value loss should stay above 0.5 (healthy uncertainty).

**Do NOT:**
- Set `epochs > 3`. More epochs = more oversampling of the tiny W/L data pool.
- Panic about high draw rates. 75-90% draws in iterations 1-10 is normal.
- Increase exploration (c_explore, risk_beta, gumbel_c_visit) before iteration 5. The model hasn't learned enough for exploration to help.
- **Use `simulations > 200` during this phase.** A weak value head + deep search produces overconfident draw evaluations that trigger repetition spirals (see Section 10g). Shallow search (200 sims) keeps MCTS uncertain enough for exploration to work.

**DO:**
- Use `--search-algorithm gumbel` (default). Gumbel's Sequential Halving allocates simulation budget more efficiently than PUCT+Dirichlet and produces improved policy targets with theoretical guarantees.
- Start with `simulations=200` for appropriate search depth during cold start.
- Use `epochs=3` to prevent value overfitting.
- Watch for loss declining steadily — that's the key health indicator.
- Monitor buffer composition via `get_composition()` — watch for W/L data emerging.
- Plan to increase simulations later (after iter 15-20) once the model shows >20% decisive games.

### 10b. Resume Pitfalls

**`--resume` does NOT preserve CLI arguments.** This is the most dangerous pitfall. When resuming, you MUST explicitly pass every parameter. Missing parameters silently revert to defaults:
- `--workers` defaults to 1 (not 32) — makes self-play 32× slower
- `--simulations` defaults to 800 (may differ from your run)
- `--train-batch` defaults to 256 (not 1024)
- `--games-per-iter` defaults to 50 (not 32)
- `--search-algorithm` defaults to `gumbel` — if your run used `puct`, you must re-specify it

**Always copy your original launch command and modify it** rather than relying on `--resume` alone.

**Buffer state after resume:**
- The replay buffer is saved to `replay_buffer.rpbf` (single fixed file, overwritten each iteration) after every self-play phase and during emergency saves. On `--resume`, this file is loaded automatically. All accumulated data (observations, policies, values, WDL targets, per-sample metadata) is preserved.
- After resume, check the logged buffer composition (`wins`/`draws`/`losses` in the iteration log) to verify data was loaded correctly.
- If the loaded buffer is draw-saturated (>90% draws), set `epochs=1` immediately to minimize draw reinforcement (see Section 9: Buffer Draw Saturation).
- `max_fillup_factor` may still trigger expanded game generation if the loaded buffer is below capacity.

### 10c. Search Batch Size (Automatic)

**`search_batch` is now auto-derived** and no longer a user-facing parameter. In Gumbel mode, `search_batch` equals `gumbel_top_k`. In PUCT mode, `search_batch` is always 1. The information below explains WHY these values are optimal.

**With Gumbel (default) — `search_batch` = `gumbel_top_k` (auto-derived):**

Virtual loss is automatically disabled. Gumbel uses round-robin root selection — each simulation in a batch is deterministically assigned to a different active action, eliminating root collisions without needing virtual loss. This removes the "virtual loss + draw bias" problem entirely and makes higher batch sizes safe.

| search_batch | GPU Utilization | Root Fan-out (top_k=16) | Recommendation |
|-------------|-----------------|------------------------|----------------|
| 1 | Low — workers idle waiting for GPU | 1 action per round-trip | **Avoid** — defeats round-robin advantage (see 10j) |
| 8 | Good | 8 of 16 actions per round-trip | Acceptable |
| 16 | Best — matches top_k | All 16 actions per round-trip | **Optimal** — empirically proven to prevent repetition draw death (see 10j) |
| 32+ | Diminishing returns | Wraps around active set | Unnecessary |

**Why Gumbel benefits from higher `search_batch`:** With round-robin, a batch of 16 leaves fans out across all 16 active root actions with zero collisions. Each GPU round-trip evaluates one leaf per action — perfect utilization. With `search_batch=1`, each GPU round-trip only advances one action, wasting 15/16 of the parallelism opportunity.

**With PUCT (`--search-algorithm puct`) — use `search_batch=1` ONLY:**

`search_batch` controls how many MCTS leaves each worker selects per GPU round-trip. Virtual loss is applied during selection to prevent duplicate leaves, but this biases Q-values:

| search_batch | GPU Throughput | Selection Rounds (800 sims) | Search Quality | Risk |
|-------------|----------------|----------------------------|----------------|------|
| 1 | Baseline | 800 rounds | Best | None |
| 2 | ~2.5× | 400 rounds | Good | Mild draw bias with weak value head |
| 4+ | ~3× | 200 rounds | Degraded | **Severe draw bias — avoid entirely** |

**Virtual loss + draw-biased value head = toxic combination:**
When the value head predicts "everything is a draw" (Q ≈ 0), virtual loss on the most promising line makes it look like a loss. With `search_batch=1`, the real evaluation corrects this before the next selection. With `search_batch=2+`, two consecutive selections happen under this pessimistic distortion, biasing MCTS toward safe repetitions.

**Recommendation for PUCT:** Always use `search_batch=1`. The GPU throughput gains from higher batch sizes are not worth the draw-bias risk. If you need better GPU utilization, switch to Gumbel instead.

### 10d. `games_per_iter` and `workers`

- `games_per_iter` is rounded up to a multiple of `workers`. Setting `workers=64` with `games_per_iter=32` wastes half the workers or produces 64 games.
- In early training, iteration time is dominated by game LENGTH (300+ moves), not game COUNT. Reducing `games_per_iter` saves less time than expected.
- `max_fillup_factor` multiplies game count when buffer is not full. With factor=3: up to 4× games in early iterations (e.g., 128 instead of 32). Set to 0 if iteration time is a concern.
- More training iterations per hour > more games per iteration, especially in early training when the model changes rapidly and older self-play data becomes stale.

### 10e. Value Head Monitoring

Track the policy:value loss ratio across iterations:

| Ratio | Phase | Action |
|-------|-------|--------|
| 8-9:1 | Normal early training | Expected — policy space (4672 moves) is much larger than value space (3 classes) |
| 10-13:1 | Warning zone | Value converging too fast. Check buffer composition (`get_composition()`). Consider reducing epochs. |
| > 15:1 | Overfitting likely | Value head is memorizing. Reduce epochs to 2-3. Increase `games_per_iter` for more diverse data. |

**Value loss benchmarks (WDL soft cross-entropy):**
- ~1.1: Random (ln(3), uniform 1/3 prediction)
- 0.7-0.9: Early learning (model distinguishes W/D/L somewhat)
- 0.4-0.6: Decent evaluation (healthy mid-training)
- < 0.3: Suspiciously low — likely overfitting unless the model is very strong

### 10f. Cold Start Progression

Typical progression of chess understanding in early training (with 200 sims):

| Iteration | Expected Behavior |
|-----------|-------------------|
| 1-2 | Random moves, accidental checkmates, 300+ move games, loss ~9.0 |
| 3-5 | Some structure emerges, draws via insufficient material / fifty-move rule, loss ~8.0 |
| 5-10 | Value head developing, game length starts to decrease, some decisive games appear |
| 10-15 | Piece development (Nf3, d4), draw rate may begin declining, value_loss 0.5-0.9 |
| 15-20 | **Simulation increase window**: If >20% decisive games and value_loss > 0.5, consider increasing simulations to 400 |
| 20-30 | Opening repertoire forms, buffer composition should show increasing W/L counts |
| 30+ | Real chess quality, consider simulations=800, LR reduction |

**Key difference from 800 sims:** With 200 sims, early games are longer and more "random," but the value head stays honestly uncertain (value_loss > 0.5). This healthy uncertainty prevents the repetition spiral that 800 sims can trigger.

### 10g. Simulation Depth Draw Death (Critical Lesson)

**Root Cause:** Deep MCTS search (800+ simulations) with a weak, uncalibrated value head produces overconfident draw evaluations. The MCTS tree averages hundreds of node evaluations from the value head, and if the value head is even slightly draw-biased (normal in early training), deep search amplifies this into high-confidence "this position is definitely a draw" assessments. The model then systematically prefers repetition-forcing moves (e.g., bishop oscillation Bb7-Ba8) as the "safest" option.

**The self-reinforcing spiral:**
```
Weak value head (slightly draw-biased)
  → Deep search amplifies bias into confident "draw" evaluations
    → MCTS prefers safe repetition-forcing moves
      → Self-play produces 90-100% draws
        → Training data is all draws
          → Value head becomes MORE draw-biased
            → Loop (irreversible without rollback)
```

**Observed progression (from actual run, iters 5→53 with sims=800):**

| Iteration | Decisive % | Repetition Draws | Value Loss | Avg Length | Status |
|-----------|-----------|-----------------|------------|------------|--------|
| 5 (200 sims) | 22% | 16/32 | 0.89 | 315 | Healthy |
| 6 (800 sims) | 22% | 16/32 | 0.89 | 315 | OK — spiral hasn't started |
| 7 | 9% | 27/32 | 0.51 | 153 | Warning — value loss dropping fast |
| 8 | 3% | 31/32 | 0.45 | 95 | Critical — parameter change attempted (sims=400, c_explore=2.0, eps=0.35) |
| 9-53 | 0% | 32/32 | 0.02 | 42 | Dead — bishop oscillation patterns, parameter changes had no effect |

**Key observations:**
1. **The spiral starts within 2-3 iterations.** Iter 6 looked fine; by iter 8 it was irreversible.
2. **Parameter tuning cannot fix it.** Changing sims=400, c_explore=2.0, dirichlet_epsilon=0.35 after iter 7 made things WORSE (iter 8: 97% draws vs 91%).
3. **Value loss is the canary.** Healthy value loss is 0.5-0.9 in early training. A drop below 0.5 with rising draw rate means the value head is collapsing toward "always predict draw."
4. **Game length is the second signal.** Avg game length dropping from 300+ to <100 with rising repetitions means the model is actively forcing draws earlier and earlier.
5. **The terminal state is ~42-move games** where the model opens with a few moves then oscillates a piece between two squares until threefold repetition.

**Prevention:**
- Start with `simulations=200` (shallow search preserves healthy uncertainty)
- Use `--search-algorithm gumbel` (default). Gumbel's Sequential Halving allocates simulation budget across multiple actions via principled pruning, rather than concentrating on the PUCT "best" child. This may reduce the draw amplification effect by distributing search effort more broadly, though the fundamental vulnerability (weak value head + deep search) still applies.
- Only increase simulations when: (a) >20% decisive games sustained, (b) value_loss > 0.5, (c) model shows actual chess understanding
- Increase gradually: 200 → 400 → 800, waiting 5+ iterations at each level
- Monitor value_loss closely after any simulation increase — if it drops below 0.5 within 3 iterations, roll back immediately

**Recovery (if caught early, within 2-3 iterations):**
1. Stop training immediately
2. Roll back to the last checkpoint BEFORE the simulation increase
3. Delete all post-rollback checkpoints and trim `claude_log.jsonl`
4. Restart with `simulations=200`

**Recovery is NOT possible after 5+ iterations of 100% REPETITION draws.** The replay buffer is saturated with draw data and the value head is irreversibly biased. You must roll back to a clean checkpoint.

**Note:** Fifty-move draw stagnation (a distinct failure mode, see Section 10i) IS recoverable even after 4+ iterations of 100% draws. The key difference is zero repetition draws — the model is still exploring, just failing to checkmate. See Section 10i for the recovery protocol.

### 10h. Simulation Increase Protocol

When increasing simulations mid-training, follow this protocol:

1. **Prerequisites** (ALL must be met):
   - Draw rate < 80% sustained over last 3 iterations
   - Value loss > 0.5
   - At least some decisive games (>10% W+L)
   - Model showing actual chess patterns (purposeful development, captures)

2. **Step size**: Increase by at most 2× (200→400, 400→800)

3. **Monitoring after increase**: Watch every iteration for 5 iterations:
   - If value_loss drops below 0.5 → roll back
   - If draw rate jumps by >20 percentage points → roll back
   - If repetition draws suddenly dominate → roll back

4. **Stabilization**: Wait 10+ iterations at the new level before considering another increase

### 10i. Fifty-Move Draw Stagnation (Distinct from Repetition Death)

**Discovered during Gumbel proof-of-concept run (f256-b20-se4, 200 sims, search_batch auto-derived from gumbel_top_k=16).**

This is a NEW failure mode that is qualitatively different from repetition draw death (Section 10g). The model plays diverse, non-repeating moves but fails to deliver checkmate, causing games to exhaust the fifty-move rule.

**Key differences from repetition draw death:**

| | Repetition Draw Death (10g) | Fifty-Move Draw Stagnation (10i) |
|---|---|---|
| **Draw mechanism** | Threefold repetition (piece oscillation) | Fifty-move rule (no captures/pawn moves) |
| **Repetition draws** | 80-100% of draws | 0% of draws |
| **Game length** | Collapsing (300→42 moves) | Expanding (300→380 moves) |
| **Model behavior** | Deliberately forcing safe repetitions | Playing varied moves, can't find checkmate |
| **Search algorithm** | Primarily PUCT | Gumbel (search_batch auto-derived from gumbel_top_k=16 prevents repetitions) |
| **Recoverable?** | NO (after 5+ iterations) | **YES** (epochs=1 + temperature_moves=50) |
| **Value loss trajectory** | Monotonic collapse (0.89→0.02) | Drop then recovery possible (0.89→0.12→0.39) |

**Observed progression (from proof-of-concept run, Gumbel + search_batch auto-derived from gumbel_top_k=16):**

| Iteration | Decisive % | Repetition Draws | 50-Move Draws | Value Loss | Avg Length | Status |
|-----------|-----------|-----------------|---------------|------------|------------|--------|
| 1-3 | 28% | 3-7 | 5-7 | 0.47-0.89 | 311-358 | Healthy |
| 4 | 50% | 4 | 6 | 0.55 | 289 | Breakthrough |
| 5 | 34% | 5 | 6 | 0.57 | 300 | Stable |
| 6 | 0% | 0 | 25 | 0.50 | 381 | **Stagnation begins** |
| 7 | 0% | 0 | 29 | 0.40 | 368 | Stagnation deepens |
| 8* | 6% | 1 | 29 | 0.15 | 341 | risk_beta=0.5 added |
| 9 | 6% | 0 | 27 | 0.12 | 365 | risk_beta insufficient |
| 10 | 28% | 6 | 11 | 0.30 | 330 | **Recovery (epochs=1, temp=50)** |
| 11 | 38% | 5 | 6 | 0.39 | 298 | Recovery confirmed |

*Iter 8 included buffer reset from resume, which accelerated value loss collapse.

**Root cause:** The model learns to play "real" chess moves (no repetitions) but hasn't developed tactical patterns (forks, pins, mating nets). With 200 simulations and a draw-biased value head, MCTS evaluates all positions as roughly equal ("draw"), so the model shuffles pieces without creating decisive advantages. The fifty-move rule (100 plies without capture or pawn push) eventually triggers.

**Why Gumbel prevents repetition but not fifty-move draws:** Gumbel's round-robin root selection distributes search across all 16 top-k actions. This prevents the "safe repetition" behavior that PUCT exhibits (where virtual loss + draw bias concentrates search on one "safest" move). However, diversity of exploration doesn't help if the model's VALUE predictions are uniformly "draw" — exploring different moves doesn't matter when they all evaluate identically.

**Failed intervention — risk_beta alone:**
Setting `risk_beta=0.5` (ERM risk-seeking) produced only 2 decisive games per iteration (6%) at iters 8-9. The ERM adjustment `Q_β ≈ E[v] + (β/2) · Var(v)` requires variance in value predictions. When `value_loss < 0.15`, the value head predicts "draw" with near-certainty for ALL positions — there is no variance for risk_beta to exploit.

**Successful intervention — epochs=1 + temperature_moves=50:**
| Change | Mechanism | Effect |
|--------|-----------|--------|
| `epochs=1` (from 3) | Reduces draw-data reinforcement per iteration. With 3 epochs, each iteration drove the value head deeper into "everything is draw." With 1 epoch, the natural data distribution drives learning. | Draw rate dropped 94%→72%→62% |
| `temperature_moves=50` (from 30) | More random opening moves create unbalanced positions that are harder to draw. Positions with material imbalances or asymmetric pawn structures naturally produce more decisive games. | Fifty-move draws dropped 27→11→6 |
| `risk_beta=0.5` | Mild assistance once variance returns. Amplifies value differences once the value head starts differentiating positions again. | Contributes after epochs=1 takes effect |

**Recovery protocol for fifty-move draw stagnation:**
1. **Detection**: Draw rate > 90% for 2+ iterations AND zero repetition draws AND fifty-move draws > 60% of draws AND value_loss < 0.4
2. **Immediate action**: Set `epochs=1` and `temperature_moves=50`
3. **Optional**: Set `risk_beta=0.3-0.5` (helps once variance returns, but insufficient alone)
4. **Monitor**: Draw rate should drop within 1-2 iterations. Value loss should increase (healthy recalibration).
5. **Restore**: Once draw rate < 70% sustained for 3+ iterations and value_loss > 0.4, gradually restore `epochs` to 2, then 3. Keep `temperature_moves=50` until the model shows strong opening play.

**Prevention:**
- Use `epochs=3` (not 5) from the start
- Consider `epochs=2` if early training shows 90%+ draws for 3+ consecutive iterations
- Monitor value_loss — if it drops below 0.4 while draw rate > 80%, reduce epochs preemptively
- `temperature_moves=30` is fine for normal training; increase to 50 only as a recovery intervention

### 10j. Gumbel vs PUCT — Empirical Comparison (search_batch Effects)

**Proof-of-concept run parameters:** f256-b20-se4, 200 sims, 32 workers, train_batch=1024, 32 games/iter.

| Setting | Previous PUCT Run | Gumbel POC Run |
|---------|-------------------|----------------|
| `--search-algorithm` | gumbel (default, but...) | gumbel |
| `search_batch` (auto-derived) | **1** (gumbel_top_k=16, but overridden) | **16** (from gumbel_top_k=16) |
| `--gumbel-top-k` | 16 | 16 |
| `--epochs` | 5 (iters 1-4), 3 (iters 5+) | 3 (iters 1-7), 1 (iters 10+) |
| `--risk-beta` | 0 (iters 1-4), 0.5 (iters 5+) | 0 (iters 1-7), 0.3-0.5 (iters 8+) |

**Head-to-head comparison (same iteration, no risk_beta):**

| Metric | PUCT sb=1 (iter 4) | Gumbel sb=16 (iter 4) | Winner |
|--------|--------------------|-----------------------|--------|
| Draw rate | 78% | **50%** | Gumbel |
| Decisive games | 22% | **50%** | Gumbel |
| Repetition draws | 1 | 4 | PUCT |
| Value loss | 0.71 | 0.55 | Tie |
| Total loss | 8.18 | 8.44 | PUCT |

**Critical finding: search_batch=16 eliminates repetition draw death.**
Across 11 iterations of the Gumbel run, there were ZERO early repetition draws. The maximum repetition draws per iteration was 7 (iteration 3), compared to the PUCT run where repetitions exploded to 22-29 per iteration from iter 6 onward. The round-robin root selection in Gumbel mode distributes search across all top-k actions, making it impossible for the search to "lock in" on a single safe repetition-forcing line.

**GPU utilization with search_batch=16:**
- NN throughput: ~2,200 evals/s (vs ~1,200 evals/s with search_batch=1 in equivalent setups)
- Batch sizes: 440-460 (hitting the large CUDA graph for maximum efficiency)
- Each GPU round-trip evaluates leaves from 16 different root actions simultaneously

**search_batch=1 with Gumbel — a missed optimization:**
The previous run used `search_batch=1` with Gumbel (the PUCT recommendation). This defeats Gumbel's round-robin advantage: with batch=1, each GPU round-trip only advances 1 of 16 top-k actions. The model still uses Gumbel noise for action selection, but Sequential Halving must do 16× more round-trips per action to gather the same statistics. Match `search_batch` to `gumbel_top_k` for optimal results.

**Updated implementation:** `search_batch` is now auto-derived from `--gumbel-top-k` (both default to 16). This ensures:
1. All top-k actions get one leaf per GPU round-trip (perfect fan-out)
2. No root collisions (round-robin assignment)
3. Maximum GPU utilization (large batches = better amortization)
4. Faster self-play (16× fewer GPU round-trips per move vs search_batch=1)

### 10k. CUDA Graph Recalibration Bug

**Problem:** The `run_parallel_selfplay()` function recalibrates CUDA graphs every iteration. Occasionally, the GPU is in a degraded state after training (warm caches, VRAM fragmentation), causing the calibration to produce garbage measurements:

```
Good calibration:  Eager: 0.9ms overhead + 0.337ms/sample (R²=0.994)
Bad calibration:   Eager: 184.5ms overhead + 0.142ms/sample (R²=0.025)
```

When R² is low, the crossover analysis selects degenerate graph sizes (all tiers set to the same size). This can cause the CUDA graph to hang, producing "32 MCTS timeouts" with 0 simulations/s.

**Symptoms:** After a working iteration, the next iteration shows:
- "Calibrating CUDA graphs..." with R² < 0.5
- Degenerate size selection (e.g., "mini=512, small=512, large_threshold=512")
- All workers timing out: "Failures: 32 MCTS timeouts"
- 0 games complete, 0 sims/s, 0 NN evals/s indefinitely

**Auto-retry:** Calibration now automatically retries on bad GPU states. If R² < 0.5 or CV > 15%, it waits 15 seconds and retries. If the retry also fails, safe default graph sizes are used automatically. No manual intervention needed.

**Implementation:** R² and CV validation is now built into `calibrate_cuda_graphs()`. Auto-retry with 15-second cooldown handles transient GPU degradation. Persistent failures fall back to safe defaults.

### 10l. Buffer Persistence on Resume

**`--resume` now automatically loads the replay buffer.** The buffer is saved to a single `replay_buffer.rpbf` file (overwritten each iteration) after every self-play phase and during emergency saves (Ctrl+C, crash, stop file). On resume, this file is loaded directly, preserving all accumulated training data including per-sample metadata (iteration, game result, termination reason, move number, game length). For backward compatibility, if `replay_buffer.rpbf` is not found, the loader falls back to the old `buffer_iter_*.rpbf` naming convention.

**What is saved in `.rpbf` files:**
- All observations, policies, values, WDL targets, and soft values
- Per-sample metadata: which iteration generated it, W/D/L result, termination type (checkmate, stalemate, repetition, fifty-move, insufficient, max_moves), move number and game length
- Buffer write position and total sample counts
- Composition is recomputed on load from metadata

**Buffer composition tracking:**
- Each iteration log now includes `wins`, `draws`, `losses` counts from `get_composition()`
- The buffer tracks W/D/L composition atomically as samples are added and overwritten
- Use composition to detect draw saturation: if draws > 90%, set `epochs=1`

**Historical note:** Prior to this feature, `--resume` created a fresh buffer and all accumulated data was lost. In the proof-of-concept run, this caused value_loss to crash from 0.40 → 0.145 in one iteration when the fresh buffer was filled with draw-heavy data. This failure mode is now prevented by automatic buffer persistence.

### 10m. Max-Moves Draw Pollution (New Failure Amplifier)

**Discovered during second Gumbel run (f256-b20-se4, 200 sims, 32 workers, epochs=3→2→1, 32 games/iter).**

The `max_moves_per_game` constant (hardcoded to 512 plies in `parallel_coordinator.hpp:42` and `game.hpp:87`, not exposed to Python) creates a draw mechanism that does not exist in real chess. When a game reaches 512 plies without a natural termination, it is declared a draw and all 512 positions are labeled as draws in the replay buffer. This artificial mechanism amplifies draw saturation far more aggressively than any natural chess draw mechanism.

**Why max-moves draws are uniquely destructive:**

A single max-moves draw generates up to 512 draw-labeled positions. A typical decisive game lasting 80 moves generates 80 W/L-labeled positions. The per-game buffer pollution ratio:

```
Max-moves draw:  ~512 draw positions per game
Decisive game:   ~80  W/L  positions per game
Amplification:   6.4× more draw data per max-moves draw vs decisive game
```

With 14-17 max-moves draws per iteration (observed in this run's crisis phase):
- 17 max-moves draws × 512 = 8,704 draw positions injected per iteration
- 3 decisive games × 80 = 240 W/L positions per iteration
- **Effective draw:decisive ratio = 36:1** in the worst iterations

This creates a vicious cycle: the value head trains on overwhelmingly draw data → MCTS evaluates all positions as "draw" → games play aimlessly to the 512 cap → more max-moves draws → more draw data.

**Observed progression (second Gumbel run):**

| Iteration | W | L | D | Draw% | Rep | MaxMov | AvgLen | ValLoss | Intervention |
|-----------|---|---|---|-------|-----|--------|--------|---------|-------------|
| 1 | 3 | 0 | 29 | 91% | 14 | 7 | 336 | 0.954 | None (cold start) |
| 2 | 1 | 3 | 28 | 88% | 7 | 9 | 361 | 0.481 | None |
| 3 | 4 | 3 | 25 | 78% | 3 | 11 | 394 | 0.383 | None |
| 4 | 3 | 2 | 27 | 84% | 8 | 8 | 362 | 0.355 | epochs=2 (val < 0.4 + draw > 80%) |
| 5 | 1 | 1 | 30 | 94% | 4 | 17 | 450 | 0.335 | epochs=1, temp_moves=50 |
| 6 | 1 | 5 | 26 | 81% | 7 | 8 | 366 | 0.342 | (holding) |
| 7 | 1 | 5 | 26 | 81% | 8 | 6 | 405 | 0.294 | (holding) |
| 8 | 1 | 0 | 31 | 97% | 4 | 14 | 436 | 0.368 | (noise — 32 game variance) |
| 9 | 1 | 2 | 29 | 91% | 4 | 16 | 451 | 0.305 | (holding) |
| 10 | 2 | 1 | 29 | 91% | 17 | 9 | 347 | 0.321 | Stopped for analysis |

Buffer at shutdown: 126K positions — W=4,364 (3.5%), D=116,509 (92.1%), L=5,558 (4.4%)

**This is a hybrid failure mode.** It does not cleanly match either repetition draw death (Section 10g) or fifty-move stagnation (Section 10i):

| Aspect | Repetition Death (10g) | Fifty-Move Stagnation (10i) | Max-Moves Pollution (this) |
|--------|----------------------|---------------------------|--------------------------|
| Dominant draw type | Threefold repetition | Fifty-move rule | Max-moves cap |
| Game length trend | Collapsing (300→42) | Expanding (300→380) | Oscillating (336→451→347) |
| Model behavior | Forces safe repetitions | Diverse but can't checkmate | Aimless shuffling with occasional tactics |
| Value loss trajectory | Monotonic collapse | Drop then recoverable | Steady decline, stuck ~0.3 |
| Policy learning | Stalled | Stalled | **Continuous improvement** (8.4→7.5) |
| epochs=1 recovery? | No (after 5+ iters) | **Yes** (1-2 iterations) | Partial (slows but doesn't reverse) |

**Key observation:** Policy loss improved monotonically every iteration (8.40 → 7.49) — the model IS learning better move selection. The value head is the sole bottleneck, trapped by the 92% draw buffer.

**What the model actually learned (from sample game analysis):**
- Positive: Discovered checkmates, promotions (including underpromotion to knight!), captures, and piece coordination
- Negative: Hundreds of moves of aimless shuffling (Qe1/Qd1/Qe1, Ra1/Ra2/Ra3) punctuated by occasional tactical finds
- The model plays like a beginner who can spot tactics but has no positional understanding

**Why epochs=1 + temp=50 was insufficient (unlike Section 10i):**

In the POC run (Section 10i), recovery happened in 1-2 iterations because the value head had fully collapsed (value_loss=0.12), leaving maximum room for recalibration. In this run, value_loss hovered at 0.29-0.37 — a frustrating middle ground where the model is overconfident but not so collapsed that natural variance provides recovery signal. Additionally, the max-moves mechanism was continuously injecting hundreds of new draw positions per iteration (8,700+ in the worst iterations), outpacing the ability of epochs=1 to limit draw reinforcement.

**Structural root cause: `max_moves_per_game=512` is not a real chess rule.** Unlike threefold repetition, fifty-move, stalemate, and insufficient material — which are legitimate chess termination conditions — the 512-ply cap is an arbitrary training heuristic. It creates a draw mechanism the model cannot learn to avoid (the model has no concept of "move 512"), and each max-moves draw floods the buffer with hundreds of artificial draw labels.

**Encoding limitation (fixed):** The halfmove clock (fifty-move rule input) was encoded as `min(1.0, halfmoves / 50.0)`, saturating at 50 half-moves. This has been fixed to `halfmoves / 100.0` (see Resolution below).

**Resolution (applied):**

Three root causes have been fixed:

1. **`max_moves_per_game` default changed from 512 to 10,000** in `parallel_coordinator.hpp:42` and `game.hpp:87`. The 10,000 cap is a safety net only — all real chess termination rules (fifty-move, threefold repetition, stalemate, insufficient material, checkmate) fire long before this. The sequential fallback in `train.py` (lines ~747, ~834) was updated to match.
2. **Halfmove clock encoding fixed from `/50.0` to `/100.0`** in `position_encoder.cpp` (now channel 18). The fifty-move rule triggers at `halfMoveClock() >= 100`, so normalizing by 100.0 gives the network full resolution across the entire range [0, 100].
3. **Castling encoding restructured to match AlphaZero paper** (123 channels, up from 122). The old encoding had 2 dead channels (12-13, always zero) and packed only the current player's castling rights into a single scalar (channel 16). The new encoding:
   - Removes the 2 dead channels
   - Adds 4 separate binary castling planes (ch 14-17): P1 kingside, P1 queenside, P2 kingside, P2 queenside
   - Shifts the no-progress count to channel 18 and history planes to channels 19-122
   - The network can now independently observe each player's castling rights

**Channel layout (123 channels):**

| Channels | Content |
|----------|---------|
| 0-11 | Piece planes (6 own + 6 opponent) |
| 12 | Color to move (constant 1.0) |
| 13 | Total move count (fullmoves / 100) |
| 14 | Current player castling kingside (binary) |
| 15 | Current player castling queenside (binary) |
| 16 | Opponent castling kingside (binary) |
| 17 | Opponent castling queenside (binary) |
| 18 | No-progress count (halfmoves / 100) |
| 19-122 | History (8 × 13 = 104 planes) |

**Breaking change:** The encoding restructure changes the total channel count (122→123) and the layout of channels 12+, making all existing checkpoints incompatible. This is acceptable — already doing a fresh run due to the halfmove clock fix.

### 10n. Early Epochs Progression

**Learned from both the POC run and the second Gumbel run:** The default starting `epochs=3` may be too aggressive when combined with Gumbel + search_batch=16 (which produces efficient but draw-heavy early games).

**Recommended early-training epochs schedule:**

| Phase | Condition | epochs | Rationale |
|-------|-----------|--------|-----------|
| Start | Iteration 1 | 2 | Conservative: avoids rapid value head overfitting on cold-start draw data |
| Healthy | Draw rate < 80%, value_loss > 0.4 for 3+ iters | 3 | Value head has enough W/L signal to benefit from more training |
| Stagnation | Draw rate > 80% AND value_loss < 0.4 | 1 | Emergency: minimize draw reinforcement (see 10i, 10m) |
| Recovery | Draw rate < 70% sustained for 3+ iters after stagnation | 2 (then 3) | Gradual restoration |

**The second run's mistake:** Starting with `epochs=3` allowed value_loss to collapse from 0.95 to 0.38 in just 3 iterations. By the time we intervened (epochs=2 at iter 4, epochs=1 at iter 5), the buffer was already >90% draws. Starting with `epochs=2` would have slowed the collapse and kept the value head more receptive to future decisive games.

### 10o. Buffet Approach — High-Temperature Bootstrap (Experimental)

**Status: THEORY VALIDATED (partial), KEY CONSTRAINT DISCOVERED**

The "Buffet" approach uses extremely high temperature (`temperature_moves=80`) with shallow search (`simulations=64`) to generate diverse, decisive training data from iteration 1. The theory: 80 random moves place the game in wild midgame/endgame positions with material imbalances, which are inherently decisive. See `Road_to_2500.md` for the full training plan.

**Experiment 1: Buffet on draw-biased model (FAILED)**

Applied Buffet parameters (temp=80, sims=64, epochs=1, train_batch=1024) to a model at iteration 20 that had a collapsed value head (value_loss=0.273, buffer 92% draws).

| Metric | Expected (fresh model) | Observed (iter 20 model) |
|--------|----------------------|-------------------------|
| Decisive games | 40-60%+ | **1/128 (0.8%)** |
| Draw rate | <60% | **99%** |
| Fifty-move draws | ~0 | **117/128** |
| Value loss after training | >0.5 | **0.098 (further collapse)** |
| Avg game length | 80-150 | **355.7** |

**Root cause: The Buffet approach requires FRESH random weights.** A draw-biased model defeats the randomization mechanism:

1. After 80 random moves, the model's POLICY takes over for the remaining 200+ moves
2. The draw-biased VALUE HEAD evaluates all positions as ~0.0 ("draw"), giving MCTS no signal to pursue winning lines
3. The model is "too smart to lose accidentally" (knows how to avoid checkmate) but "not smart enough to win intentionally" (can't convert material advantages)
4. Result: games drift aimlessly for 200+ moves until the fifty-move rule triggers

**Experiment 2: Buffet on fresh random model (CONFIRMED — partial)**

Started a new run from random weights with the same Buffet parameters. The very first game to complete was decisive (black win, 132 moves). The run was stopped early for analysis, but the single data point confirms random models DO produce accidental checkmates that the Buffet approach depends on.

**Critical constraint (new):** The Buffet approach is a COLD-START strategy only. It cannot be retroactively applied to a model that has already developed draw bias. If draw death has set in, the only recovery is to start from scratch — no combination of temperature, simulations, epochs, or risk_beta can overcome a deeply entrenched draw-biased value head in this regime.

**Why parameter tuning fails on a draw-collapsed model:**

| Intervention | Why it fails |
|-------------|-------------|
| `temperature_moves=80` | Random moves create diverse POSITIONS, but the model's policy still plays for draws after move 80 |
| `epochs=1` | Reduces draw reinforcement per iteration, but with 99% draw games there's no W/L signal to amplify |
| `risk_beta=0.5` | ERM needs value variance (`Var(v)`). When value_loss < 0.1, the model predicts "draw" with certainty — zero variance for risk_beta to exploit |
| `simulations=64` (low) | Shallow search helps random models stay uncertain, but a trained model's value head overrides search uncertainty |

**Recommended protocol:**
1. If starting a new training run: use Buffet from iteration 1 (temp=80, sims=64, epochs=1)
2. If current run has value_loss > 0.4 and draw rate < 80%: apply Section 10n epochs schedule
3. If current run has value_loss < 0.3 and draw rate > 90%: **start fresh** — the model cannot be recovered
4. The threshold between "recoverable" and "start fresh" is approximately value_loss ≈ 0.3-0.4 combined with buffer draw saturation > 90%

### 10p. Self-Play Timing with Low Simulations

**Observed (from Buffet runs with sims=64, 32 workers, gumbel_top_k=16):**

| Metric | Expected (plan) | Observed |
|--------|----------------|----------|
| Self-play time per iter | 10-20 sec | **~20 min** |
| Games per second | ~6-12 | **~0.1** |
| Avg game length | 80-150 moves | **320-360 moves** |
| NN throughput | ~2,200 evals/s | ~2,200 evals/s (confirmed) |
| Moves per second | ~100+ | **~37** |

**Why self-play is slower than expected:** The plan assumed short games (80-150 moves) with temp=80, but early-training games with random weights are LONG (300+ moves). Each game requires 300+ moves × 64 sims × 16 search_batch leaf evaluations, not the 80-150 moves assumed. The NN throughput is as expected (~2,200 evals/s with large CUDA graph batches of ~415), but the sheer number of moves per game makes each iteration take ~20 minutes instead of 10-20 seconds.

**Implication for the training plan:** Phase 1 iterations take ~20 minutes each (not seconds). With 50 iterations planned, Phase 1 takes ~17 hours. This is still much faster than sims=200 runs (which took 20-40 min per iter with similarly long games), but the "training becomes the bottleneck" prediction was wrong — self-play still dominates.

### 10q. Prioritized Experience Replay (PER)

**What it is:** PER (Schaul et al., 2015) replaces uniform replay buffer sampling with loss-proportional sampling. Positions the model struggles with (high training loss) are sampled more frequently, focusing gradient updates where they matter most.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--priority-exponent` | 0.0 | Priority exponent α. 0=uniform (disabled), 0.6=recommended |
| `--per-beta` | 0.4 | IS correction exponent β. Fixed value (not annealed). See selection guide below |

**How it works internally:**

1. A **sum-tree** (segment tree) is allocated alongside the replay buffer. Each leaf stores `priority^α` for one buffer sample. Internal nodes store child sums, enabling O(log N) proportional sampling.
2. **New samples** enter with `max_priority` (the highest priority seen so far), guaranteeing they are drawn at least once for training.
3. **Self-play workers** call `set_leaf()` (O(1), no lock) when adding samples. A `tree_needs_rebuild_` flag is set.
4. **Training thread** calls `sample_prioritized()`, which first does a deferred `rebuild()` (O(N) bottom-up fix) if flagged, then performs stratified sampling from the sum-tree.
5. **Per-sample loss** (policy CE + value CE) is computed during training. After the optimizer step, `update_priorities(indices, losses + ε)` writes fresh priorities into the tree.
6. **IS (Importance Sampling) weights** correct the gradient bias: `w_i = (N × P(i))^(-β)`, normalized so `max(w) = 1`. β anneals from `--per-beta` to `--per-beta-final` over `--per-beta-warmup` iterations.

**Priority lifecycle:**
- Sample added → `max_priority` (guaranteed first draw)
- Drawn into batch → forward pass computes current per-sample loss
- After backward pass → `update_priorities()` writes fresh loss as new priority
- Sits in buffer → priority becomes "stale" (reflects old model's loss)
- Drawn again → priority refreshed with current model's loss

Staleness is acceptable: high-loss samples are drawn more often (stay fresher), low-loss samples change less (staleness matters less), and the circular buffer naturally evicts the oldest samples.

**Zero-overhead guarantee (α=0):**
When `--priority-exponent 0` (the default), `enable_per()` is never called. The sum-tree is never allocated. `add_sample()` skips one null-pointer check. `sample()` is called instead of `sample_prioritized()`. No IS weights, no `update_priorities()`, no extra memory. RPBF saves as v3 without a priorities section.

**When to enable PER:**
- After the model has learned basic piece movement (typically iteration 10+)
- When training loss has plateaued and you want more efficient gradient usage
- When buffer is large (100K+) and many samples are "easy" for the current model

**When NOT to enable PER:**
- During initial random-weight phase (all losses are similarly high — no benefit from prioritization)
- If buffer is very small (<10K) — uniform sampling already covers most positions

**Choosing `--per-beta` (IS correction strength):**

Beta controls how much importance sampling corrects the gradient bias from non-uniform sampling. PER samples high-loss positions more often, which biases the gradient toward those positions. IS weights undo this bias so the network doesn't overfit to outliers.

| β value | IS correction | Gradient behavior | When to use |
|---------|--------------|-------------------|-------------|
| 0.0 | None | Fully biased toward high-loss samples. Fastest loss reduction, highest overfit risk. | Never recommended for sustained training. |
| 0.2-0.3 | Light | Mostly biased. High-loss positions still heavily overrepresented. | Aggressive catch-up: when training is far behind and you need the model to rapidly learn hard positions (e.g., first 5-10 iters after enabling PER on a stale buffer). |
| **0.4** | **Moderate (default)** | **Good balance. High-loss positions are prioritized but IS weights prevent runaway overfitting. Gradient direction slightly biased but magnitude is controlled.** | **Most training scenarios. Start here.** |
| 0.6-0.7 | Strong | Near-unbiased gradients. Priority sampling still focuses on hard positions, but weights significantly dampen their influence. | Stable late-training (loss < 3.0) where convergence quality matters more than learning speed. Also when grad_norm_max is elevated (>5) — higher beta smooths gradient spikes from outlier positions. |
| 1.0 | Full | Mathematically unbiased — equivalent to uniform sampling's gradient expectations, but with priority-biased sample selection. | Theoretical correctness. In practice, overly conservative — negates most of PER's benefit. Only use if you observe training instability (oscillating loss, grad spikes) that lower beta values can't fix. |

**Practical decision tree for `--per-beta`:**

```
Is this a new PER-enabled run (first time enabling)?
  → Start with --per-beta 0.4 (default)

Is grad_norm_max consistently > 5?
  → Increase to --per-beta 0.7 (dampens outlier gradients)

Is loss oscillating (up-down-up-down over 5+ iters)?
  → Increase to --per-beta 0.7 or 1.0

Is training progressing but slowly (loss declining < 1% per iter)?
  → Decrease to --per-beta 0.3 (more aggressive prioritization)

Is the buffer heavily draw-saturated (>90% draws)?
  → Use --per-beta 0.3 (amplify the rare W/L positions more)

Is loss < 2.0 and you're fine-tuning for quality?
  → Increase to --per-beta 0.6-0.7 (stability over speed)
```

**Changing beta requires restart:** `--per-beta` is a CLI argument, not hot-reloadable via `param_updates.json`. To change it: `touch <run_dir>/stop`, then restart with `--resume <run_dir> --priority-exponent 0.6 --per-beta <new_value>`.

**PER in the Buffet training strategy:**

| Phase | PER? | Recommended β | Rationale |
|-------|------|---------------|-----------|
| Phase 1 (Buffet, iters 1-20) | No | — | All positions are from random play. Losses are uniformly high — no "easy" positions to deprioritize. PER overhead with no benefit. |
| Phase 2 (Tasting Menu, iters 20-60) | Optional | 0.4 | As model improves, buffer accumulates easy positions. PER starts to help by focusing on hard positions. Enable at Phase 2 restart if loss < 7.0. |
| Phase 3 (Real Chess, iters 60-150) | Yes | 0.4 | Large buffer with wide loss distribution. Default beta provides good balance between prioritization and stability. |
| Phase 4+ (Deep Training, iters 150+) | Yes | 0.6-0.7 | Late training prioritizes convergence quality. Higher beta smooths gradients for fine-grained improvement. |

**PER and value head collapse:** PER can help prevent/recover from value head collapse (Section 10o) by amplifying the gradient signal from decisive positions. When the buffer is 90%+ draws, the few W/L positions have high value loss and get sampled more frequently under PER, keeping the value head responsive. Use `--per-beta 0.3` in this scenario to maximize the amplification. However, PER alone cannot fix a deeply collapsed value head (value_loss < 0.15) — apply epochs=1 first to reset overfitting, then enable PER on restart.

**PER requires restart:** Because the sum-tree is allocated at startup, PER cannot be enabled or disabled via `param_updates.json`. To enable PER, stop training (`touch <run_dir>/stop`) and restart with `--resume <run_dir> --priority-exponent 0.6 --per-beta 0.4`.

**RPBF v3 format:** Buffer files saved with PER enabled include a priorities section after metadata (flag in `reserved[0]` bit 0). Loading a file with priorities into a PER-enabled buffer restores the priority distribution. Loading without PER enabled skips the priorities section. Files saved without PER are also valid v3 (no priorities section).

**Memory overhead:** Negligible. A 100K-capacity buffer uses ~1 MB for the sum-tree vs ~3 GB for observations. On disk, priorities add 400 KB per 100K samples.

### 10r. MCTS Reanalysis (Concurrent Policy Refresh)

**What it does:** Each iteration, a configurable fraction of replay buffer positions are re-searched with the current (improved) neural network using full MCTS. The resulting search policies replace the original (stale) policies stored from self-play. Value and WDL targets are NOT updated — they remain the ground-truth game outcomes.

**Why it helps:** As the model improves, old policy targets become stale — they reflect the search quality of a weaker network. Reanalysis refreshes these targets so the model trains on policies that match its current strength, reducing the "staleness gap" between buffer contents and the model's actual capabilities. This is particularly valuable for large replay buffers where positions persist for many iterations.

**Architecture — zero additional wall-clock time:**

Reanalysis runs concurrently with self-play by sharing the coordinator's GPU thread:

```
Self-play workers → PRIMARY queue ─┐
                                    ├─ GPU thread (single inference call) → results routed back
Reanalysis workers → SECONDARY queue─┘
```

- The GPU thread serves the primary queue (self-play) first with full timeout
- Secondary queue (reanalysis) only fills spare batch capacity — if self-play saturates the GPU, reanalysis gets almost no slots (correct: self-play throughput matters more)
- When self-play workers have natural gaps (search computation, result processing), reanalysis fills what would otherwise be padding zeros — free compute
- After self-play workers finish, the GPU thread transitions to serving reanalysis only (the "reanalysis tail" phase)

**CLI parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--reanalyze-fraction` | 0.0 | Fraction of buffer to reanalyze each iteration (0=disabled) |
| `--reanalyze-simulations-ratio` | 0.25 | Reanalysis sims as fraction of self-play sims |
| `--reanalyze-workers` | 0 (auto) | Number of reanalysis workers (0 = half of self-play workers) |

**How it works internally:**
1. Before self-play, `N = floor(buffer_size * reanalyze_fraction)` positions are randomly sampled
2. Each position's FEN + 8 Zobrist history hashes (stored in the replay buffer) are used to reconstruct a `chess::Board` with proper repetition detection
3. The stored observation (full 123-channel encoding from the original game) is used for root evaluation
4. MCTS runs with NO Dirichlet noise and NO temperature — producing a pure search policy
5. The updated policy overwrites the original in the replay buffer; value/WDL targets stay as game outcomes

**FEN + Zobrist hash storage:**
- Enabled automatically when `--reanalyze-fraction > 0`
- Each replay buffer sample stores: FEN string (128-byte fixed slot) + 8 Zobrist hashes (64 bytes) + hash count (1 byte) = 193 bytes overhead per sample (~0.6% of the 31,488-byte observation)
- Persisted in RPBF v4 format (backward-compatible: v3 files load correctly with empty FENs)
- FEN storage must be enabled BEFORE loading a buffer on `--resume` (the code handles this automatically)

**When to enable:**

| Training Phase | Reanalysis? | Recommended Settings | Rationale |
|---------------|-------------|---------------------|-----------|
| Phase 1 (Buffet, iters 1-20) | No | — | Model changes rapidly, buffer is small. Positions cycle out quickly — staleness is minimal. |
| Phase 2 (Tasting Menu, iters 20-60) | Optional | `--reanalyze-fraction 0.15` | As buffer grows and model improves, older positions have noticeably stale policies. Light reanalysis helps. |
| Phase 3 (Real Chess, iters 60-150) | Yes | `--reanalyze-fraction 0.25` | Large buffer with significant staleness. Reanalysis provides meaningful policy improvement at negligible cost. |
| Phase 4+ (Deep Training, iters 150+) | Yes | `--reanalyze-fraction 0.25 --reanalyze-simulations-ratio 0.5` | Deep search reanalysis is most valuable when the model is strong and positions are complex. Higher sim ratio justified. |

**Interaction with PER:**
- Reanalysis does NOT directly change PER priorities
- Updated policy targets produce different training losses when sampled, and those losses naturally update priorities via the existing `update_priorities()` flow
- This is correct: PER measures "how much can the model learn," which changes organically after policy refresh

**Reanalysis requires restart:** Like PER, FEN storage is allocated at startup. To enable reanalysis on an existing run, stop training and restart with `--resume <run_dir> --reanalyze-fraction 0.25`. Positions generated before FEN storage was enabled will have empty FENs and will be skipped during reanalysis (counted as `positions_skipped` in stats).

**Monitoring:** The training log reports per-iteration: positions updated, positions skipped, reanalysis NN evals, and tail time (seconds spent on reanalysis after self-play finished). A healthy run shows most positions completing (low skip rate) and short tail times (< 30% of self-play time).
