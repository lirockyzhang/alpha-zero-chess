# Plan: Train AlphaZero Chess Model from Scratch to ELO 2500

## Philosophy: "Breadth-First Learning"

Traditional AlphaZero training starts from the opening position with moderate search depth, hoping decisive games emerge naturally. This leads to 70-80% draws in early iterations because a random network + real openings = aimless play that drifts into draws.

**The Buffet approach inverts this.** Start with extremely high randomization (temp=50-80+) and very shallow search (50-100 sims) to create a *buffet* of thousands of diverse midgame/endgame positions with clear decisive outcomes. The value head gets strong W/L gradient signal from iteration 1. Then gradually reduce temperature and increase simulations as the network learns fundamentals.

**Why this works:**
- 50-80 random opening moves place the game in wild midgame/endgame positions with material imbalances
- These positions are inherently decisive — no draw death risk
- sims=50-100 means each game is 4-10x faster — more games per iteration at the same wall-clock cost
- Massive position diversity = fast learning of piece values, tactics, basic endgames
- Gradual transition to "real" chess prevents the cold-start problem entirely

**Network**: f256-b20-se4 (256 filters, 20 residual blocks, SE reduction 4)
**Search**: Gumbel Top-k Sequential Halving (gumbel_top_k=16, search_batch auto-derived)

---

## Phase 1: "Buffet" — Maximum Diversity (Iters 1-20)

```bash
uv run python alphazero-cpp/scripts/train.py \
  --claude --claude-timeout 3600 \
  --iterations 50 \
  --games-per-iter 128 --workers 32 \
  --simulations 64 \
  --search-algorithm gumbel --gumbel-top-k 16 \
  --filters 256 --blocks 20 --se-reduction 4 \
  --lr 0.001 --train-batch 1024 --epochs 1 \
  --buffer-size 500000 \
  --temperature-moves 80 \
  --live
```

**Why these values:**
- `simulations=64`: Minimum viable search. Games are ~4x faster than sims=200. Draw death (Section 10g) does not apply here — the positions are so random that systematic draws are impossible.
- `temperature_moves=80`: ~80 random moves before policy sharpens. Games effectively start from random midgame/endgame positions.
- `games_per_iter=128`: 4x more games than the old plan. Each game with sims=64 takes ~2-5 seconds. With 32 workers, that's 4 games per worker. Self-play finishes in ~10-20 seconds — training becomes the bottleneck (good!).
- `epochs=1`: With 128 games x ~80 moves = ~10,000 unique positions per iter, we want the network to learn broadly, not memorize. Keeps value head uncertain (high loss), which is healthy for exploration.
- `--claude-timeout 3600`: 1-hour timeout gives Claude Code plenty of time to review.

**Expected behavior (normal, do NOT intervene):**
- Iters 1-3: High loss (~9.0), but **40-60%+ decisive games immediately** — this is the key difference from the old plan
- Iters 5-10: Loss drops fast (learning piece values), game length shortens
- Iters 10-20: Clear tactical understanding, beats random easily

**Intervene if:**
- Loss not declining after 5 iterations → check for bugs
- GPU utilization very low → increase `games_per_iter` or `workers`
- grad_norm_max > 10 → reduce LR by 50%

---

## Phase 2: "Tasting Menu" — Less Random, More Depth (Iters 20-60)

**Transition criteria (ALL must be met):**
- Loss below 7.0 and declining
- Beats random 100% (`uv run python alphazero-cpp/scripts/evaluation.py --checkpoint <run_dir>/ --evaluators vs_random`)
- Draw rate stable (not increasing)

**Gradual transition via `param_updates.json`:**

| Iteration | Changes | Rationale |
|-----------|---------|-----------|
| ~20 | `{"temperature_moves": 60, "simulations": 100}` | Start seeing more structured openings |
| ~30 | `{"temperature_moves": 50, "simulations": 150, "epochs": 2}` | Network strong enough for 2 passes |
| ~40 | `{"temperature_moves": 40, "simulations": 200, "epochs": 2}` | Approaching "real" chess territory |
| ~50 | `{"temperature_moves": 30, "simulations": 200, "epochs": 3, "lr": 0.0005}` | Standard training regime begins |

**Critical monitoring during Phase 2:**

This is the most dangerous phase — reducing temperature can trigger draw death if done too fast.

- Watch draw rate after each temperature reduction. If draw rate jumps >20 percentage points within 3 iterations, **revert** the temperature change.
- If fifty-move draws suddenly dominate (zero repetition draws), set `{"epochs": 1, "temperature_moves": 50}` (recovery protocol from Section 10i).
- If value loss drops below 0.4 while draw rate > 80%, set `{"epochs": 1}` — the value head has collapsed.

**ELO benchmarks:**
- Iter 20: Beat random 100%
- Iter 40: Endgame puzzles 2-3/5
- Iter 60: Score vs Stockfish depth=1 (~ELO 800-1000)

---

## Phase 3: "Chef's Table" — Real Chess (Iters 60-150)

**Transition criteria:**
- Loss below 5.0, draw rate < 60%, endgame puzzles 2+/5

**Stop and restart** (to increase simulations):
```bash
touch <run_dir>/stop
# Wait for graceful shutdown, then:
uv run python alphazero-cpp/scripts/train.py \
  --claude --claude-timeout 3600 \
  --resume <run_dir> \
  --iterations 300 \
  --games-per-iter 64 --workers 32 \
  --simulations 800 \
  --search-algorithm gumbel --gumbel-top-k 16 \
  --filters 256 --blocks 20 --se-reduction 4 \
  --lr 0.0005 --train-batch 1024 --epochs 3 \
  --buffer-size 500000 \
  --temperature-moves 30 \
  --live
```

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

**ELO benchmarks:**
- Iter 80: vs Stockfish ELO 1200-1400
- Iter 100: vs Stockfish ELO 1600-1800, endgame puzzles 5/5
- Iter 150: vs Stockfish ELO ~2000

---

## Phase 4: Deep Training (Iters 150-300+)

**Transition criteria:**
- Loss below 2.0, endgame puzzles 5/5, ELO ~1800-2000

**Stop and restart:**
```bash
uv run python alphazero-cpp/scripts/train.py \
  --claude --claude-timeout 3600 \
  --resume <run_dir> \
  --iterations 500 \
  --games-per-iter 64 --workers 32 \
  --simulations 1600 \
  --search-algorithm gumbel --gumbel-top-k 16 \
  --filters 256 --blocks 20 --se-reduction 4 \
  --lr 0.0001 \
  --train-batch 1024 --epochs 5 \
  --buffer-size 500000 \
  --temperature-moves 20 \
  --opponent-risk "-0.3:0.3" \
  --live
```

**LR decay**: 0.0001 → 0.00005 → 0.00002 → 0.00001 (at each loss plateau)

**ELO benchmarks** (eval every 20-30 iters with `evaluation.py --evaluators vs_stockfish --stockfish-elo XXXX`):
- Iter 180-220: ELO ~2100-2300
- Iter 270-300: ELO ~2400-2500

---

## Phase 5: Polishing (Iters 300-400+, if needed)

Only if ELO 2300-2400 reached but not 2500 yet.

- Increase `simulations` to 3200 (~3-4 hours/iter)
- Reduce `temperature_moves` to 10
- LR at 0.000005 with `epochs=10`
- Consider `games_per_iter=128` for more data volume

---

## Claude Code Integration with `monitor_iteration.py`

### The Review Loop

For each training iteration:

**1. Launch training** with `--claude` flag (see phase commands above).

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

Valid keys for `param_updates.json` (with ranges):
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

**7. Resume training** by deleting both signal files:
```bash
rm <run_dir>/selfplay_done <run_dir>/awaiting_review
```

Training unblocks only when **both** files are absent. If neither Claude Code nor the user deletes them within `--claude-timeout` seconds, training auto-continues (deleting them itself).

**8. Re-launch monitor** for the next iteration.

### Monitor Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success — metrics summary printed to stdout |
| 1 | Error — run directory or `claude_log.jsonl` not found |
| 2 | Timeout — `awaiting_review` did not appear within `--timeout` seconds |

### Review Frequency

| Phase | Iterations | Frequency | Rationale |
|-------|-----------|-----------|-----------|
| 1 (Buffet) | 1-20 | Every iteration | Verify loss is declining, catch unexpected issues |
| 2 (Tasting Menu) | 20-60 | Every iteration | Critical transition — draw death risk during temp reduction |
| 3+ (Chef's Table) | 60+ | Every 2 iterations | Stable training, less intervention needed |

---

## Key Files

- `alphazero-cpp/scripts/train.py` — Main training script
- `alphazero-cpp/scripts/monitor_iteration.py` — Claude Code async monitor
- `alphazero-cpp/scripts/evaluation.py` — Standalone ELO benchmarking (vs_random, endgame, vs_stockfish)
- `alphazero-cpp/scripts/llm_operation_manual.md` — Full decision framework and alert reference
- `alphazero-cpp/scripts/live_dashboard.py` — Real-time Plotly dashboard

## Verification

- After launch: confirm `claude_log.jsonl` is created in the run directory
- After first iteration: monitor exits with code 0, metrics summary printed
- First few iterations: **40-60%+ decisive games** (the key signal that Buffet is working)
- Periodically: run `evaluation.py` gauntlet against Stockfish at increasing ELO levels
- Final: consistent >50% win rate vs Stockfish at ELO 2500 with 1600 sims
