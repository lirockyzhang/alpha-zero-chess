# Road to 2500

## Philosophy: "Breadth-First Learning"

Traditional AlphaZero training starts from the opening position with moderate search depth, hoping decisive games emerge naturally. This leads to 70-80% draws in early iterations because a random network + real openings = aimless play that drifts into draws.

**The Buffet approach inverts this.** Start with extremely high randomization (temp=50-80+) and very shallow search (50-100 sims) to create a *buffet* of thousands of diverse midgame/endgame positions with clear decisive outcomes. The value head gets strong W/L gradient signal from iteration 1. Then gradually reduce temperature and increase simulations as the network learns fundamentals.

**Why this works:**
- 50-80 random opening moves place the game in wild midgame/endgame positions with material imbalances
- These positions are inherently decisive — no draw death risk
- sims=50-100 means each game is 4-10x faster — more games per iteration at the same wall-clock cost
- Massive position diversity = fast learning of piece values, tactics, basic endgames
- Gradual transition to "real" chess prevents the cold-start problem entirely

---

## Architecture

| Component | Value |
|-----------|-------|
| Network | f256-b20-se4 (256 filters, 20 residual blocks, SE reduction 4) |
| Search | Gumbel Top-k Sequential Halving (gumbel_top_k=16, search_batch auto-derived) |
| Encoding | 123-channel input, 4672-move policy |

**Fixed across all phases:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--search-algorithm` | gumbel | Gumbel Top-k SH throughout |
| `--gumbel-top-k` | 16 | search_batch auto-derived to match |
| `--filters` | 256 | Not hot-reloadable |
| `--blocks` | 20 | Not hot-reloadable |
| `--se-reduction` | 4 | Not hot-reloadable |
| `--buffer-size` | 100000 | Not hot-reloadable |
| `--workers` | 32 | Not hot-reloadable |
| `--claude` | — | Enables Claude Code review loop |
| `--claude-timeout` | 1800 | 30 min per review |
| `--live` | — | Real-time dashboard |

---

## Essential Practices

### Monitor Every Iteration During Training

Always run `monitor_iteration.py` alongside training to get structured metrics summaries and Claude Code review prompts:

```bash
uv run python alphazero-cpp/scripts/monitor_iteration.py --latest --timeout 7200
```

This blocks until the current iteration completes self-play, then outputs a formatted summary of losses, game outcomes, draw types, trends, and alerts. Without this, training runs blind — problems like value head collapse or draw death can go unnoticed for hours. See [Operational Reference](#claude-code-integration-with-monitor_iterationpy) for the full review loop.

### Preserve Milestone Checkpoints

**This is critical.** Training is irreversible — once the model overfits, collapses, or enters a draw spiral, you cannot go back unless you saved a checkpoint. The training script auto-saves model weights, but the replay buffer is overwritten in place (`replay_buffer.rpbf`).

**At every important milestone** (phase transitions, before parameter changes, when metrics look good), manually snapshot both the model and the replay buffer:

```bash
# Example: preserving state at iteration 40 (Phase 1→2 transition)
cp <run_dir>/replay_buffer.rpbf <run_dir>/replay_buffer_iter040.rpbf
cp <run_dir>/checkpoint_iter040.pt <run_dir>/milestone_phase1_complete.pt
```

**When to snapshot:**
- Before any phase transition (Phases 1→2, 2→3, etc.)
- Before risky parameter changes (reducing temperature, increasing simulations)
- When draw rate is at its lowest / loss is at a new low
- Before enabling PER or changing `--per-beta`
- Any time you think "this looks good, I don't want to lose this"

**Why the replay buffer matters:** The model checkpoint alone is not enough to resume training effectively. The replay buffer contains hundreds of thousands of training positions with their search targets. Without it, the model resumes training on an empty buffer and can catastrophically forget. By saving `replay_buffer_iter040.rpbf`, you can revert to that exact training state by:

```bash
cp <run_dir>/replay_buffer_iter040.rpbf <run_dir>/replay_buffer.rpbf
uv run python alphazero-cpp/scripts/train.py --resume <run_dir> ...
```

### Updating This Document

This document is split into two parts — keep them separate:

1. **Training Plan** (Phases, Architecture, Operational Reference) — Update when the *plan changes*. If a phase definition needs adjustment based on experimental evidence, update the phase parameters and note what changed and why. Keep this section forward-looking.

2. **Experimental Record** (Run Log, Lessons Learned) — Update after *every training run or significant event*. Use the standardized run format:

   ```
   ### Run N: `<run_dir_name>`
   **Date:** YYYY-MM-DD
   **Phase:** Which phase, what kind of attempt
   **Starting point:** Fresh or resumed from what

   **Parameters (Planned vs Actual):**
   | Parameter | Planned | Actual | Note |
   (fill in deviations)

   **Iteration Data:**
   (table of metrics)

   **What happened:** (narrative)
   **Why:** (analysis)
   **Outcome:** Success/Failed — led to Lesson LN / started Run M
   ```

   For Lessons Learned, tag each with `CONFIRMED` (observed and validated), `THEORY` (predicted but untested), or `RESOLVED` (fixed, kept for reference), and cross-reference the run that produced the insight.

---

## Training Phases

### Phase 1: "Buffet" — Maximum Diversity (Iters 1-20)

**Goal:** Create a buffet of diverse midgame/endgame positions with decisive outcomes. Bootstrap the value head with strong W/L gradient signal from random games.

**Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `simulations` | 64 | Minimum viable search. Games ~4x faster than sims=200. Draw death (Section 10g) does not apply — positions too random for systematic draws. |
| `temperature_moves` | 80 | ~80 random moves before policy sharpens. Games effectively start from random midgame/endgame positions. |
| `games_per_iter` | 128 | 4x more games than old plan. Each game with sims=64 takes ~2-5s. With 32 workers, self-play is fast — training becomes the bottleneck (good!). |
| `lr` | 0.001 | Standard initial learning rate. |
| `train_batch` | 1024 | Large batches for stable gradients. |
| `epochs` | 1 | With ~10,000 unique positions/iter, learn broadly, don't memorize. Keeps value head uncertain (high loss), healthy for exploration. |

**Launch command:**
```bash
uv run python alphazero-cpp/scripts/train.py \
  --claude --claude-timeout 1800 \
  --iterations 50 \
  --games-per-iter 128 --workers 32 \
  --simulations 64 \
  --search-algorithm gumbel --gumbel-top-k 16 \
  --filters 256 --blocks 20 --se-reduction 4 \
  --lr 0.001 --train-batch 1024 --epochs 1 \
  --buffer-size 100000 \
  --temperature-moves 80 \
  --live
```

**Transition criteria (ALL must be met):**
- Loss below 7.0 and declining
- Beats random 100% (`evaluation.py --evaluators vs_random`)
- Draw rate stable (not increasing)

**Expected benchmarks:**
- Iters 1-3: High loss (~9.0), but **40-60%+ decisive games immediately** — this is the key difference from the old plan
- Iters 5-10: Loss drops fast (learning piece values), game length shortens
- Iters 10-20: Clear tactical understanding, beats random easily

**Intervene if:**
- Loss not declining after 5 iterations → check for bugs
- GPU utilization very low → increase `games_per_iter` or `workers`
- grad_norm_max > 10 → reduce LR by 50%

---

### Phase 2: "Tasting Menu" — Less Random, More Depth (Iters 20-60)

**Goal:** Gradually transition from chaotic Buffet positions toward structured chess. Reduce temperature and increase simulations in steps, monitoring for draw death at each transition.

**Gradual transition via `param_updates.json`:**

| Iteration | `temperature_moves` | `simulations` | `epochs` | `lr` | Rationale |
|-----------|---------------------|---------------|----------|------|-----------|
| ~20 | 60 | 100 | 1 | 0.001 | Start seeing more structured openings |
| ~30 | 50 | 150 | 2 | 0.001 | Network strong enough for 2 passes |
| ~40 | 40 | 200 | 2 | 0.001 | Approaching "real" chess territory |
| ~50 | 30 | 200 | 3 | 0.0005 | Standard training regime begins |

**Critical monitoring:**

This is the most dangerous phase — reducing temperature can trigger draw death if done too fast.

- Watch draw rate after each temperature reduction. If draw rate jumps >20 percentage points within 3 iterations, **revert** the temperature change.
- If fifty-move draws suddenly dominate (zero repetition draws), set `{"epochs": 1, "temperature_moves": 50}` (recovery protocol from Section 10i).
- If value loss drops below 0.4 while draw rate > 80%, set `{"epochs": 1}` — the value head has collapsed.

**Transition criteria (to Phase 3):**
- Loss below 5.0
- Draw rate < 60%
- Endgame puzzles 2+/5

**Expected benchmarks:**
- Iter 20: Beat random 100%
- Iter 40: Endgame puzzles 2-3/5
- Iter 60: Score vs Stockfish depth=1 (~ELO 800-1000)

---

### Phase 3: "Chef's Table" — Real Chess (Iters 60-150)

**Goal:** Full-strength chess training with deep search. Enable PER to focus on hard positions. Introduce asymmetric self-play for robustness.

**Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `simulations` | 400 → 800 | Start at 400; increase to 800 after draw rate stable <60% for 5+ iters. |
| `temperature_moves` | 30 | Near-standard openings. |
| `games_per_iter` | 64 | Fewer games, deeper search. |
| `lr` | 0.0005 | Phase start; decay schedule below. |
| `train_batch` | 1024 | Unchanged. |
| `epochs` | 3 | Network benefits from multiple passes. |
| `priority_exponent` | 0.6 | PER enabled — buffer has 100K+ positions with wide loss distribution. |
| `per_beta` | 0.4 | Moderate IS correction (balance prioritization vs gradient stability). See Section 10q. |

**Why enable PER here:** By Phase 3 the buffer contains 100K+ positions with a wide loss distribution — many "easy" positions the model has mastered, and fewer "hard" ones. PER focuses gradient updates on the hard positions, improving training efficiency.

**LR decay schedule:**

| Iteration | LR | Trigger |
|-----------|-----|---------|
| 60 | 0.0005 | Phase start |
| 80-90 | 0.0002 | Loss plateau |
| 110-130 | 0.0001 | Second plateau |

**Simulation increase:** Start at 400, increase to 800 only after draw rate is stable < 60% for 5+ iterations. Apply via `param_updates.json`:
```json
{"simulations": 800, "_reason": "Draw rate stable <60% for 5 iters"}
```

**Optional — asymmetric self-play (iter 80-100+):**
```json
{"opponent_risk_min": -0.5, "opponent_risk_max": 0.5}
```

**Launch command (stop and restart required for PER):**
```bash
touch <run_dir>/stop
# Wait for graceful shutdown, then:
uv run python alphazero-cpp/scripts/train.py \
  --claude --claude-timeout 1800 \
  --resume <run_dir> \
  --iterations 300 \
  --games-per-iter 64 --workers 32 \
  --simulations 800 \
  --search-algorithm gumbel --gumbel-top-k 16 \
  --filters 256 --blocks 20 --se-reduction 4 \
  --lr 0.0005 --train-batch 1024 --epochs 3 \
  --buffer-size 100000 \
  --temperature-moves 30 \
  --priority-exponent 0.6 --per-beta 0.4 \
  --live
```

**Transition criteria (to Phase 4):**
- Loss below 2.0
- Endgame puzzles 5/5
- ELO ~1800-2000

**Expected benchmarks:**
- Iter 80: vs Stockfish ELO 1200-1400
- Iter 100: vs Stockfish ELO 1600-1800, endgame puzzles 5/5
- Iter 150: vs Stockfish ELO ~2000

---

### Phase 4: Deep Training (Iters 150-300+)

**Goal:** Push toward grandmaster-level play with deep search, low learning rate, and stronger PER correction.

**Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `simulations` | 1600 | Deep search for positional understanding. |
| `temperature_moves` | 20 | Near-book openings. |
| `games_per_iter` | 64 | Unchanged. |
| `lr` | 0.0001 | Fine-tuning regime. |
| `train_batch` | 1024 | Unchanged. |
| `epochs` | 5 | Multiple passes for extraction. |
| `priority_exponent` | 0.6 | PER continues. |
| `per_beta` | 0.6 | Smoother convergence at this stage. |
| `opponent_risk` | -0.3:0.3 | Asymmetric self-play for robustness. |

**LR decay:** 0.0001 → 0.00005 → 0.00002 → 0.00001 (at each loss plateau)

**Launch command (stop and restart required):**
```bash
uv run python alphazero-cpp/scripts/train.py \
  --claude --claude-timeout 1800 \
  --resume <run_dir> \
  --iterations 500 \
  --games-per-iter 64 --workers 32 \
  --simulations 1600 \
  --search-algorithm gumbel --gumbel-top-k 16 \
  --filters 256 --blocks 20 --se-reduction 4 \
  --lr 0.0001 \
  --train-batch 1024 --epochs 5 \
  --buffer-size 100000 \
  --temperature-moves 20 \
  --priority-exponent 0.6 --per-beta 0.6 \
  --opponent-risk "-0.3:0.3" \
  --live
```

**Expected benchmarks** (eval every 20-30 iters with `evaluation.py --evaluators vs_stockfish --stockfish-elo XXXX`):
- Iter 180-220: ELO ~2100-2300
- Iter 270-300: ELO ~2400-2500

---

### Phase 5: Polishing (Iters 300-400+, if needed)

**Goal:** Final push to 2500 if not yet reached. Maximum search depth, minimal learning rate, high epoch count.

Only enter this phase if ELO 2300-2400 reached but not 2500 yet.

**Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `simulations` | 3200 | Maximum depth (~3-4 hours/iter). |
| `temperature_moves` | 10 | Near-deterministic openings. |
| `games_per_iter` | 128 | More data volume for fine-tuning. |
| `lr` | 0.000005 | Minimal learning rate. |
| `epochs` | 10 | Maximum extraction. |

---

## Operational Reference

### Claude Code Integration with `monitor_iteration.py`

#### The Review Loop

> **CRITICAL — DO NOT EDIT SCRIPTS**
>
> During the review loop the autonomous trainer must **never** modify
> `train.py`, `monitor_iteration.py`, or any other source code. Its only
> permitted write actions are:
> - `param_updates.json` — hot-reload hyperparameters
> - `claude_decisions.md` — decision log
> - Signal file management (`selfplay_done` / `awaiting_review` deletion, `stop` creation)
>
> If a code change appears necessary, log the recommendation in
> `claude_decisions.md` and flag it for human review.

For each training iteration:

**1. Launch training** with `--claude` flag (see phase launch commands above).

**2. Launch monitor** in background:
```bash
uv run python alphazero-cpp/scripts/monitor_iteration.py --latest --timeout 7200
```

**3. Monitor blocks** until the `awaiting_review` signal file appears in the run directory. It polls via `os.path.exists()` every 2 seconds. Progress messages go to stderr; the metrics summary goes to stdout.

**4. Monitor outputs** a formatted summary containing:
- Iteration number and status header
- Losses (total, policy, value, grad norm avg/max)
- Game outcomes (white wins, black wins, draws, draw rate %, draw type breakdown)
- Quality metrics (avg game length, buffer size, LR, risk beta)
- Timing (self-play time, training time, total time)
- Alerts (from the training script's alert system)
- Trends (last 5 iterations — loss, draw rate, game length)
- Sample game (result, reason, length, moves truncated at 200 chars)
- ACTION REQUIRED block with exact `rm` command to resume

**5. Claude Code analyzes** the output against `llm_operation_manual.md` decision framework (Section 6).

**6. If changes needed**, write `param_updates.json` atomically to the run directory:
```bash
# Write to temp file first, then rename (atomic on most filesystems)
echo '{"temperature_moves": 60, "simulations": 100, "_reason": "Phase 2 transition"}' > <run_dir>/param_updates.json.tmp
mv <run_dir>/param_updates.json.tmp <run_dir>/param_updates.json
```

**7. Resume training** by deleting both signal files:
```bash
rm <run_dir>/selfplay_done <run_dir>/awaiting_review
```

Training unblocks only when **both** files are absent. If neither Claude Code nor the user deletes them within `--claude-timeout` seconds, training auto-continues (deleting them itself).

**8. Re-launch monitor** for the next iteration.

#### `param_updates.json` Keys

| Key | Type | Min | Max |
|-----|------|-----|-----|
| `lr` | float | 1e-6 | 1.0 |
| `train_batch` | int | 16 | 8192 |
| `epochs` | int | 1 | 100 |
| `simulations` | int | 50 | 10000 |
| `c_explore` | float | 0.1 | 10.0 |
| `risk_beta` | float | -3.0 | 3.0 |
| `temperature_moves` | int | 0 | 200 |
| `dirichlet_alpha` | float | 0.01 | 2.0 |
| `dirichlet_epsilon` | float | 0.0 | 1.0 |
| `fpu_base` | float | 0.0 | 2.0 |
| `opponent_risk_min` | float | -3.0 | 3.0 |
| `opponent_risk_max` | float | -3.0 | 3.0 |
| `games_per_iter` | int | 1 | 10000 |
| `max_fillup_factor` | int | 0 | 100 |
| `save_interval` | int | 1 | 1000 |
| `gumbel_top_k` | int | 1 | 64 |
| `gumbel_c_visit` | float | 1.0 | 1000.0 |
| `gumbel_c_scale` | float | 0.01 | 10.0 |

Keys prefixed with `_` (e.g., `_reason`) are treated as metadata — logged but not applied. Unknown keys are logged as warnings and skipped. Values are clamped to their valid range.

**Not hot-reloadable (require restart):** `--priority-exponent`, `--per-beta`, `--filters`, `--blocks`, `--se-reduction`, `--buffer-size`, `--workers`, `--search-algorithm`. To change these, stop training and restart with `--resume`.

#### Monitor Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success — metrics summary printed to stdout |
| 1 | Error — run directory or `claude_log.jsonl` not found |
| 2 | Timeout — `awaiting_review` did not appear within `--timeout` seconds |

#### Review Frequency

| Phase | Iterations | Frequency | Rationale |
|-------|-----------|-----------|-----------|
| 1 (Buffet) | 1-20 | Every iteration | Verify loss is declining, catch unexpected issues |
| 2 (Tasting Menu) | 20-60 | Every iteration | Critical transition — draw death risk during temp reduction |
| 3+ (Chef's Table) | 60+ | Every 2 iterations | Stable training, less intervention needed |

### Key Files

- `alphazero-cpp/scripts/train.py` — Main training script
- `alphazero-cpp/scripts/monitor_iteration.py` — Claude Code async monitor
- `alphazero-cpp/scripts/evaluation.py` — Standalone ELO benchmarking (vs_random, endgame, vs_stockfish)
- `alphazero-cpp/scripts/llm_operation_manual.md` — Full decision framework and alert reference
- `alphazero-cpp/scripts/live_dashboard.py` — Real-time Plotly dashboard

### Verification Checklist

- After launch: confirm `claude_log.jsonl` is created in the run directory
- After first iteration: monitor exits with code 0, metrics summary printed
- First few iterations: **40-60%+ decisive games** (the key signal that Buffet is working)
- Periodically: run `evaluation.py` gauntlet against Stockfish at increasing ELO levels
- Final: consistent >50% win rate vs Stockfish at ELO 2500 with 1600 sims

---

# Experimental Record

## Run Log

### Run 1: `f256-b20-se4_2026-02-18_01-43-26`

**Date:** 2026-02-18
**Phase:** Phase 1 (Buffet) — traditional start with non-standard parameters, then Buffet retrofit attempt
**Starting point:** Fresh training with parameter deviations from plan

**Parameters (Planned vs Actual):**

| Parameter | Planned | Actual | Note |
|-----------|---------|--------|------|
| `simulations` | 64 | 64 | Match |
| `temperature_moves` | 80 | 50 | Deviation — less random than intended |
| `workers` | 32 | 24 | Deviation — fewer workers |
| `train_batch` | 1024 | 256 | Deviation — inconsistent across resumes |
| `epochs` | 1 | 1 | Match |
| `risk_beta` | 0 | 0.3 | Deviation — not in plan |
| `games_per_iter` | 128 | 128 | Match |
| `lr` | 0.001 | 0.001 | Match |

**Iteration Data:**

| Iteration | Loss | Policy | Value | W | B | D | Draw% | Fifty-Move | Rep | AvgLen |
|-----------|------|--------|-------|---|---|---|-------|-----------|-----|--------|
| 1 | 9.65 | 8.49 | 1.16 | - | - | - | ~90% | - | - | 300+ |
| 5 | ~8.5 | ~8.0 | ~0.5 | - | - | - | ~85% | - | - | 300+ |
| 10 | ~8.0 | ~7.7 | ~0.35 | - | - | - | ~85% | - | - | 330 |
| 18 | 7.91 | 7.57 | 0.34 | 5 | 11 | 112 | 87.5% | 41 | 64 | ~330 |
| 19 | 7.85 | 7.51 | 0.35 | 10 | 26 | 92 | 71.9% | 60 | 28 | ~330 |
| 20 | 7.76 | 7.49 | 0.27 | 3 | 5 | 112 | 93.3% | 96 | 10 | 320 |

**Buffer at iter 20:** 188,453 positions — W=7,254 (3.8%), D=173,775 (92.1%), L=7,424 (3.9%)

**What happened:**
1. Policy loss improved monotonically (8.49 → 7.49) — the model IS learning move selection
2. Value loss collapsed (1.16 → 0.27) — the value head became deeply draw-biased
3. Fifty-move draws dominate, NOT repetition draws (Gumbel prevents repetitions)
4. The model CAN find checkmates (sample game at iter 20: 292-move checkmate) but rarely does
5. With temp=50, games still average 320+ moves — not enough randomization
6. `train_batch=256` was a parameter mismatch from the plan (should have been 1024)
7. Multiple resumes with inconsistent CLI args compounded the deviations

**Buffet retrofit attempt (iter 21):**

Resumed from iter 20 with corrected Buffet params (temp=80, sims=64, train_batch=1024, workers=32, epochs=1):

| Iter | Games | W | D | L | Draw% | Fifty-Move | ValLoss | AvgLen |
|------|-------|---|---|---|-------|-----------|---------|--------|
| 21 | 128 | 1 | 127 | 0 | **99.2%** | 117 | **0.098** | 355.7 |

**Why:**
Buffet (temp=80) on a draw-biased model produced 99% draws and collapsed value_loss further (0.27 → 0.098). The model is "too smart to lose accidentally" (avoids checkmate for 200+ moves after 80 random moves) but "too dumb to win intentionally." The Buffet approach generates decisive games through accidental checkmates by random models — a trained model defeats this mechanism. See `llm_operation_manual.md` Section 10o for full analysis.

**Outcome:** Failed — led to Lessons L1, L2, L4. Demonstrated that Buffet requires fresh random weights.

---

### Run 2: Fresh Buffet Start (preliminary test only)

**Date:** 2026-02-18 (preliminary test only)
**Phase:** Phase 1 (Buffet) — fresh start from random weights
**Starting point:** Random initialization

**Parameters:** As specified in Phase 1 plan (all matched).

**Preliminary test (1 game only):**
- Launched fresh run from random weights, first game completed: **decisive** (black win, 132 moves)
- Confirms random models DO produce accidental checkmates that bootstrap the value head
- Run stopped before full iteration per user request

**Timing (measured from Run 1 and aborted Run 2):**
- Self-play: ~20 min per iteration (NOT 10-20 seconds as originally predicted)
- Reason: early games average 320-360 moves, much longer than predicted 80-150
- NN throughput: ~2,200 evals/s (matches expectations)
- Phase 1 total: ~17 hours for 50 iterations (not minutes)

**Outcome:** Preliminary result confirms Buffet works with fresh weights. Full run not yet executed. Led to Lesson L3.

---

### Run 3: `f256-b20-se4_2026-02-18_23-43-17`

**Date:** 2026-02-18 to 2026-02-19
**Phase:** Phase 1 (Buffet) — fresh start from random weights, full training run
**Starting point:** Random initialization

**Parameters (Planned vs Actual):**

| Parameter | Planned | Actual | Note |
|-----------|---------|--------|------|
| `simulations` | 64 | 64 | Match |
| `temperature_moves` | 80 | 80 (→50 at iter 14) | Match initially, lowered during recovery |
| `games_per_iter` | 128 | **64** | Half the planned games — fewer decisive samples per iteration |
| `workers` | 32 | 32 | Match |
| `train_batch` | 1024 | 1024 | Match |
| `epochs` | 1 | 1 | Match |
| `lr` | 0.001 | 0.001 | Match |
| `risk_beta` | 0 | 0 (→0.3 at iter 14) | Applied during recovery attempt |
| PER | disabled | **enabled (α=0.6, β=0.4)** | Deviation — not planned for Phase 1 |
| `claude_timeout` | 1800 | **3600** | Longer review window |

**Iteration Data:**

| Iter | Loss | Policy | Value | W | B | D | Draw% | 50-move | Rep | AvgLen | Notes |
|------|------|--------|-------|---|---|---|-------|---------|-----|--------|-------|
| 1 | 9.65 | 8.47 | 1.18 | 3 | 3 | 58 | 90.6% | 13 | 8 | 313 | Initial random play |
| 2 | 5.94 | 8.37 | 0.83 | 4 | 5 | 55 | 85.9% | 17 | 8 | 348 | Big loss drop |
| 3 | 5.44 | 8.32 | 0.58 | 5 | 3 | 56 | 87.5% | 15 | 10 | 327 | |
| 4 | 5.21 | 8.24 | 0.49 | 3 | 10 | 51 | 79.7% | 11 | 5 | 353 | |
| 5 | 5.04 | 8.19 | 0.43 | 4 | 6 | 54 | 84.4% | 6 | 13 | 329 | Buffer full (100K) |
| 6 | 7.85 | 8.12 | 0.38 | 5 | 6 | 53 | 82.8% | 10 | 14 | 356 | PER loss inflation artifact |
| 7 | 7.89 | 8.07 | 0.39 | 6 | 10 | 48 | **75.0%** | 9 | 11 | 335 | **Best iteration** — snapshot saved |
| 8 | 7.84 | 8.01 | 0.44 | 9 | 10 | 45 | **70.3%** | 7 | 15 | 316 | Best draw rate |
| 9 | 7.53 | 7.95 | 0.45 | 6 | 7 | 51 | 79.7% | 6 | 16 | 354 | |
| 10 | 7.62 | 7.90 | 0.44 | 6 | 2 | 56 | 87.5% | 14 | 15 | 377 | Draw rate climbing |
| 11 | 7.34 | 7.85 | 0.48 | 10 | 7 | 47 | 73.4% | 24 | 9 | 343 | Brief recovery |
| 12 | 7.16 | 7.80 | 0.40 | 4 | 4 | 56 | 87.5% | 15 | 6 | 340 | |
| 13 | 7.13 | 7.74 | 0.32 | 0 | 0 | 64 | **100%** | 56 | 0 | 353 | **Stagnation onset** |
| 14 | 6.95 | 7.65 | 0.27 | 0 | 0 | 64 | 100% | 64 | 0 | 315 | Applied temp=50, risk_beta=0.3 |
| 15 | 6.89 | 7.59 | 0.19 | 0 | 0 | 64 | 100% | 57 | 0 | 368 | Recovery not yet visible |
| 16 | 6.46 | 7.54 | **0.07** | 0 | 0 | 64 | 100% | 59 | 0 | 387 | **Value head near-collapse** |
| 17 | 6.23 | 7.45 | 0.15 | 7 | 8 | 49 | 76.6% | 41 | 2 | 383 | Risk_beta shock — brief recovery |
| 18 | 6.69 | 7.40 | 0.23 | 7 | 3 | 54 | 84.4% | 32 | 10 | 348 | Snapshot saved, training stopped |
| 19 | 6.68 | 7.34 | 0.30 | 4 | 4 | 56 | 87.5% | 43 | 4 | 398 | Resumed from iter 18 |
| 20 | 6.35 | 7.29 | 0.31 | 4 | 2 | 58 | 90.6% | 55 | 1 | 467 | Draw rate climbing again |
| 21 | 6.32 | 7.19 | 0.19 | 0 | 3 | 61 | **95.3%** | 57 | 3 | 504 | **Re-collapse — training stopped** |

**Buffer composition (key moments):**
- Iter 7 (best): W=3,344 (3.3%) D=89,504 (89.5%) L=7,152 (7.2%)
- Iter 16 (worst): W=291 (0.3%) D=99,709 (99.7%) L=0 (0%)
- Iter 21 (final): ~W=3,500 (3.5%) D=94,500 (94.5%) L=2,000 (2.0%)

**Snapshots preserved:**
- `milestone_iter005_healthy.pt` + `replay_buffer_iter005_healthy.rpbf` — value_loss=0.43, 15.6% decisive
- `milestone_iter007_best.pt` + `replay_buffer_iter007_best.rpbf` — value_loss=0.39, 25% decisive, best overall health
- `milestone_iter018_recovery.pt` + `replay_buffer_iter018_recovery.rpbf` — recovery attempt checkpoint

**What happened:**

1. **Iters 1-8 — Healthy training.** Policy loss improved monotonically (8.47 → 8.01). Value loss declined but stayed in healthy range (1.18 → 0.44). Decisive games peaked at 29.7% (iter 8), with draw rate bottoming at 70.3%. The Buffet strategy was working.

2. **Iters 9-12 — Gradual decline.** Draw rate oscillated but trended upward (80% → 88%). Value loss stayed around 0.40-0.48. Fifty-move draws increased (6 → 24 → 15). The model was learning to survive random positions.

3. **Iters 13-16 — Fifty-move stagnation.** Draw rate hit 100% at iter 13, entirely via fifty-move rule (no repetitions — Gumbel prevents those). Value loss collapsed: 0.32 → 0.27 → 0.19 → **0.069**. Applied recovery (temp=50, risk_beta=0.3) at iter 14. Buffer became 99.7% draws at iter 16.

4. **Iters 17-18 — Brief risk_beta recovery.** The risk-seeking shock produced 15 decisive games at iter 17 (best since iter 8). Value loss began recovering (0.07 → 0.15 → 0.23). But this was temporary — the model was adapting to play aggressively without losing.

5. **Iters 19-21 — Recovery failed.** Decisive games declined each iteration (8 → 6 → 3). Value loss plateaued at 0.30 then dropped back to 0.19. Game length ballooned to 504 plies. The model learned to survive risky positions, neutralizing risk_beta's effect. The "too smart to lose, too dumb to win" pattern re-established.

**Why:**

Three compounding factors caused the stagnation:

1. **Low simulations (64) cannot find checkmates.** Delivering checkmate requires 5-10 move tactical sequences. With sims=64, the search sees 2-3 moves ahead — enough to avoid blunders but not to execute mating plans. This is both an exploration problem and a search depth problem.

2. **Defense improves faster than offense.** Avoiding checkmate requires recognizing 1-2 bad moves; delivering checkmate requires coordinating a multi-move plan. The model's defensive skill outpaced its attacking skill, creating the asymmetry.

3. **Fifty-move draws poison the buffer.** Each fifty-move draw trains the value head to predict "draw" for that position. With 90%+ draws in the buffer, the value head converges to predicting draws for everything (value_loss → 0). This creates a death spiral: more draws → more draw predictions → even more draws.

4. **risk_beta provides a temporary shock, not a structural fix.** The ERM mechanism makes the model prefer risky lines, but within 3-4 iterations the policy adapts to play "aggressive defense" — taking risks without actually losing. The decisive rate declines back to near-zero.

5. **games_per_iter=64 was half the planned 128.** With only 64 games and 90% draws, each iteration added ~6 decisive positions. Too few to overcome the 95K draws in the buffer.

**Outcome:** Failed — the Buffet strategy with sims=64 cannot sustain decisive games past ~12 iterations. Led to Lessons L8, L9, L10. Iter 7 snapshot (`milestone_iter007_best.pt`) preserved as recovery point with healthy value head (VL=0.39) and best policy available at that state (PL=8.07).

---

## Lessons Learned

### L1. Buffet Requires Fresh Weights `CONFIRMED` → Run 1

The Buffet approach generates decisive games through accidental checkmates by random models. A trained model with value_loss < 0.3 defeats this mechanism — it avoids checkmate for 200+ moves after randomization. **Never apply Buffet to a model that has developed draw bias.** Start fresh.

### L2. CLI Parameter Consistency is Critical `CONFIRMED` → Run 1

`--resume` does NOT preserve CLI arguments. Run 1 used inconsistent `train_batch` (256 vs plan's 1024) and `workers` (24 vs plan's 32) across resumes. **Always copy the full launch command and modify it when resuming.**

### L3. Self-Play Time Estimates Were Wrong `CONFIRMED` → Run 2

With sims=64, the plan predicted 10-20 second self-play. Actual: ~20 minutes. The error was assuming 80-150 move games, but early random games average 320-360 moves. Each move still requires 64 sims × 16 search_batch evaluations, so long games dominate wall-clock time.

### L4. Value Head Collapse Is the Primary Failure Mode `CONFIRMED` → Run 1

In all runs so far, the value head collapses toward "always predict draw" within 3-10 iterations. The policy head improves steadily. The bottleneck is not learning to play chess — it's learning to EVALUATE positions. This is why the Buffet approach (which maximizes decisive training signal for the value head) is the right strategy.

### L5. PER Is Most Valuable After Phase 1 `CONFIRMED` → Run 3

PER (Prioritized Experience Replay) samples positions proportional to training loss, focusing gradient updates on positions the model struggles with. During Phase 1 (Buffet), all positions have similarly high loss — PER provides no benefit and inflates reported loss metrics (see L6). Run 3 confirmed: PER enabled from iter 1 did not prevent value head collapse and did not amplify decisive-position signal effectively when the buffer was 95%+ draws. **Enable PER from Phase 2 onward** when the buffer has meaningful loss distribution. Use `--priority-exponent 0.6 --per-beta 0.4` on restart. Use `--per-beta 0.6` in Phase 4+ for smoother convergence. See `llm_operation_manual.md` Section 10q and L11.

### L6. PER Inflates All Reported Loss Metrics `CONFIRMED` → Code Analysis

**When PER is active, reported loss/policy_loss/value_loss are all higher than the true uniform-sample values.** This is a reporting artifact, not a real problem — unless game quality metrics also deteriorate.

**Why:** PER oversamples high-loss positions. The `policy_loss` and `value_loss` logged in `train.py` (lines 2124-2125) are unweighted `.mean()` over the PER-sampled batch — the code comment says "comparable regardless of PER" but this is wrong because the batch itself is biased. The IS-weighted `loss` (line 2119) partially corrects with `per_beta`, but with beta < 1.0 the correction is incomplete.

**Diagnostic — is the loss increase PER artifact or real overfitting?**

| Signal | PER Artifact | Real Problem |
|--------|-------------|--------------|
| Loss jumped when PER was enabled | Yes — sampling bias | No |
| Loss gradually increasing over iterations | Less likely | Yes — overfitting |
| Draw rate, game length, puzzle accuracy | Stable or improving | Deteriorating |
| Policy loss vs value loss | Both inflated equally | Value loss rising disproportionately |

**Action:** When PER is active, rely on game quality metrics (draw rate, ELO, endgame puzzles) as ground truth — these are PER-independent. Only reduce LR if game quality deteriorates alongside rising loss.

### L7. Buffer Persistence Fixed `RESOLVED`

Buffer persistence was previously only saved at checkpoint time (inside the `save_checkpoint_now` block), which meant it could be skipped entirely if training crashed or was interrupted before reaching a save interval. Now fixed: `replay_buffer.rpbf` is saved after every self-play phase and during emergency saves (Ctrl+C, crash, stop file). A single fixed filename replaces the old per-iteration `buffer_iter_NNN.rpbf` naming.

### L8. ERM (risk_beta) Recovery Is Temporary `CONFIRMED` → Run 3

Applying `risk_beta=0.3` during fifty-move stagnation produces a 1-2 iteration "shock" of decisive games (Run 3 iter 17: 15 decisive after 4 iterations of 0). But the model's policy adapts within 3-4 iterations to play "aggressive defense" — taking risky lines without actually losing. Decisive rate declines back toward zero (15 → 10 → 8 → 6 → 3). **risk_beta is a band-aid, not a structural fix.** It buys time but does not solve the underlying problem of insufficient search depth or draw-biased training data.

### L9. Fifty-Move Draws: Exploration Problem AND Value Collapse `CONFIRMED` → Run 3

Fifty-move draws at low simulations have two components:

1. **Execution problem (low sims):** The model may "know" a position is winning (value head says +0.6) but sims=64 can't calculate the 5-10 move mating sequence. This is identifiable when value_loss stays HIGH (>0.3) with fifty-move draws — the value head maintains discrimination but search can't convert.

2. **Value collapse:** Each fifty-move draw trains the value head to predict "draw," poisoning the buffer. This is identifiable when value_loss DROPS alongside rising fifty-move draws — the model stops believing positions can be decisive.

In Run 3, BOTH were present. The solution must address both: higher simulations (for execution) AND a healthy value head (for evaluation). Increasing sims alone won't help if the value head is collapsed; fixing the value head alone won't help if search is too shallow to find mates.

### L10. Defense Outpaces Offense in Self-Play `CONFIRMED` → Run 3

The model learns to avoid checkmate (recognize 1-2 bad moves) much faster than it learns to deliver checkmate (execute a 5-10 move plan). This creates the "too smart to lose, too dumb to win" asymmetry. In Run 3, the draw rate bottomed at 70% (iter 8) then climbed back toward 100% over the next 5 iterations as defensive skill caught up. **Implication:** to sustain decisive games, the model needs enough search depth (sims) to find mating sequences, not just enough to avoid blunders.

### L11. PER Cannot Rescue a Draw-Saturated Buffer `CONFIRMED` → Run 3

PER was enabled from iteration 1 in Run 3. It did not prevent value head collapse because PER amplifies signal proportional to what exists in the buffer — when 95%+ of the buffer is draws, even PER-amplified decisive positions are too few to maintain the value head. PER also inflated reported loss metrics (L6), making monitoring harder. **However**, PER should help when the buffer has meaningful decisive content (>5-10%). With a healthier buffer and higher simulations generating more decisive games, PER's prioritization of hard positions actively supports value head health. **Key condition:** PER is beneficial when the buffer's decisive fraction can be sustained by self-play; it cannot create signal that doesn't exist.
