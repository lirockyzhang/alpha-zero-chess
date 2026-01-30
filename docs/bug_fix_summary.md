# Bug Fix and Testing Summary

## Date: 2026-01-28

## Critical Bug Fixed

### Issue: TypeError in Batched Inference
**Location:** `alphazero/selfplay/batched_actor.py`

**Error Message:**
```
TypeError: bad operand type for unary -: 'GameResult'
```

**Root Cause:**
The `GameState.get_result()` method returns a `GameResult` dataclass object, not a primitive float. The batched actor code was attempting to negate this object directly using the unary `-` operator, which Python doesn't support for custom objects unless `__neg__()` is implemented.

**Affected Code:**
- `BatchedActorProcess._play_game()` at line 166
- `BatchedActor.play_game()` at line 225

**Fix Applied:**
```python
# Before (broken):
result = state.get_result()
ts.value = result if ts.player == 0 else -result  # ❌ TypeError

# After (fixed):
game_result = state.get_result()
result = game_result.value_for_white  # ✅ Extract numeric value
ts.value = result if ts.player == 0 else -result  # ✅ Works correctly
```

**Impact:**
- **Before:** Batched inference actors would crash when games completed, preventing training with `--batched-inference` flag
- **After:** Batched inference works correctly, enabling efficient multi-actor training with centralized GPU inference

---

## Date: 2026-01-29

### Issue 1: AttributeError in Endgame Evaluation
**Location:** `alphazero/evaluation/endgame_eval.py`

**Error Message:**
```
AttributeError: property 'board' of 'GameState' object has no setter
```

**Root Cause:**
The `GameState` class uses an immutable design pattern where the `board` property is read-only (no setter). The endgame evaluation code attempted to directly assign a chess.Board to `state.board`, which is not allowed. Additionally, the code tried to mutate the board state with `state.board.push(move)` instead of using the immutable `apply_move()` method.

**Affected Code:**
- `EndgameEvaluator.evaluate_position()` at lines 474 and 497

**Fix Applied:**
```python
# Before (broken):
state = GameState()
state.board = chess.Board(position.fen)  # ❌ AttributeError: no setter
# ... later ...
state.board.push(move)  # ❌ Mutates internal state incorrectly

# After (fixed):
state = GameState.from_fen(position.fen)  # ✅ Use factory method
# ... later ...
state = state.apply_move(move)  # ✅ Returns new immutable state
```

**Impact:**
- **Before:** Endgame evaluation script would crash immediately when trying to evaluate any position
- **After:** Endgame evaluation works correctly, enabling testing on 50 curated endgame positions

**Design Pattern Note:**
The `GameState` class follows functional programming principles with immutability:
- Private `_board` attribute stores the actual board
- Public `board` property only has a getter (returns a copy)
- State changes return new `GameState` objects rather than mutating existing ones
- Factory method `from_fen()` properly initializes state from FEN strings

---

### Issue 2: ImportError for action_to_move Function
**Location:** Multiple files

**Error Message:**
```
ImportError: cannot import name 'action_to_move' from 'alphazero.chess_env.moves'
```

**Root Cause:**
The code incorrectly attempted to import `action_to_move` as a standalone function from the `alphazero.chess_env.moves` module. However, `action_to_move` is actually a method of the `GameState` class (defined in `alphazero/chess_env/board.py:188-190`), not a standalone function in the moves module.

**Affected Code:**
- `alphazero/evaluation/endgame_eval.py` at line 492
- `alphazero/web/app.py` at line 218
- `Google Colab/train_alphazero.ipynb` Cell 10 (lines 1003-1004)

**Fix Applied:**

**File 1: `alphazero/evaluation/endgame_eval.py`**
```python
# Before (broken):
from alphazero.chess_env.moves import action_to_move
# ... later ...
move = action_to_move(action, state.board)

# After (fixed):
# No import needed - use GameState method
# ... later ...
move = state.action_to_move(action)
```

**File 2: `alphazero/web/app.py`**
```python
# Before (broken):
from alphazero.chess_env.moves import action_to_move
# ... later in _get_model_move() ...
move = action_to_move(action, game.board)
game.board.push(move)  # ❌ Also mutates state incorrectly

# After (fixed):
# No import needed - use GameState method
# ... later in _get_model_move() ...
move = game.action_to_move(action)
self.games[session_id] = game.apply_move(move)  # ✅ Immutable update
```

**File 3: `Google Colab/train_alphazero.ipynb` Cell 10**
```python
# Before (broken):
from alphazero.chess_env.moves import action_to_move
move = action_to_move(action, state.board)
state.board.push(move)  # ❌ Also mutates state incorrectly

# After (fixed):
# No import needed - use GameState method
move = state.action_to_move(action)
state = state.apply_move(move)  # ✅ Immutable update
```

**Impact:**
- **Before:** Endgame evaluation, web interface, and Colab notebook would crash when trying to convert actions to moves
- **After:** All three components work correctly, properly using the GameState API

**Why This Bug Occurred:**
1. Misunderstanding of the codebase architecture - assuming `action_to_move` was a utility function
2. Not checking the actual location of the function before importing
3. The bug appeared in multiple newly created files (endgame eval, web interface, Colab notebook)
4. Existing code in `scripts/play.py` and demo files correctly used `state.action_to_move(action)`

**Prevention for Future:**
- Always verify function/method locations before importing
- Check existing codebase for usage patterns before implementing new features
- Use IDE features or grep to find correct import paths
- Follow the immutable design pattern consistently (use `apply_move()` instead of `board.push()`)

---

## Test Suite Additions

### 1. Batched Inference Tests (`tests/test_batched_inference.py`)

**Coverage:** 15+ comprehensive test cases

**Test Classes:**
- `TestInferenceRequest`: Dataclass creation and validation
- `TestInferenceResponse`: Response structure validation
- `TestBatchedEvaluator`: Client-side evaluator functionality
  - Request/response flow with mock server
  - Timeout handling
  - Request counter management
- `TestInferenceServer`: Server-side inference processing
  - Single request handling
  - Batched request processing
  - Weight update mechanism
  - Response queue management
- `TestBatchedActor`: Actor game playing with batched inference
  - Complete game trajectory generation
  - Integration with inference server
  - Trajectory validation
- `TestBatchedInferenceIntegration`: End-to-end integration tests
  - Multiple actors with single server
  - Parallel game execution
  - Error handling for missing response queues

**Key Features:**
- Uses actual multiprocessing with `InferenceServer` process
- Tests real network inference with small test networks (32 filters, 2 blocks)
- Validates complete request/response pipeline
- Tests concurrent actor scenarios
- Includes timeout and error handling tests

### 2. MCTS Backend Tests (`tests/test_mcts_backends.py`)

**Coverage:** 20+ comprehensive test cases

**Test Classes:**
- `TestMCTSBackendAvailability`: Backend detection
  - Python backend always available
  - Detection of Cython and C++ backends
- `TestMCTSBackendInterface`: Interface compliance
  - Parametrized tests across all available backends
  - Search return types validation
  - Simulation count verification
  - Legal move enforcement
  - Temperature function correctness
  - Dirichlet noise application
- `TestMCTSBackendConsistency`: Cross-backend consistency
  - Deterministic search comparison
  - Top action overlap verification
- `TestMCTSBackendEdgeCases`: Edge case handling
  - Terminal position searches
  - Positions with few legal moves
  - Zero simulation handling
  - Multiple searches with same instance
- `TestCythonMCTS`: Cython-specific tests
  - Basic search functionality
  - Consistency with Python implementation
- `TestCppMCTS`: C++-specific tests
  - Basic search functionality
  - Consistency with Python implementation
  - Memory safety validation
- `TestMCTSBackendPerformance`: Performance comparison (informational)

**Key Features:**
- Parametrized tests run against all available backends
- Consistency checks ensure implementations produce similar results
- Edge case coverage for robustness
- Performance benchmarking for optimization insights
- Memory safety tests for C++ implementation

---

## Documentation Updates

### Files Modified:
1. **README.md**
   - Added new test modules to testing section
   - Updated test coverage documentation
   - Added references to `test_mcts_backends.py` and `test_batched_inference.py`

2. **CHANGELOG.md** (new file)
   - Comprehensive changelog following Keep a Changelog format
   - Detailed bug description and fix
   - Test suite additions
   - Technical details section explaining the root cause

3. **Bug Fix Summary** (this document)
   - Complete analysis of the bug
   - Test coverage summary
   - Implementation details

---

## Testing Instructions

### Run All Tests:
```bash
uv run pytest tests/ -v
```

### Run Specific Test Suites:
```bash
# Batched inference tests
uv run pytest tests/test_batched_inference.py -v

# MCTS backend tests
uv run pytest tests/test_mcts_backends.py -v

# All MCTS tests
uv run pytest tests/test_mcts.py tests/test_mcts_backends.py -v
```

### Run Tests with Output:
```bash
# Show print statements and detailed output
uv run pytest tests/test_batched_inference.py -v -s

# Run specific test class
uv run pytest tests/test_batched_inference.py::TestInferenceServer -v
```

---

## Verification Checklist

- [x] Bug identified in both `BatchedActorProcess` and `BatchedActor`
- [x] Fix applied to both locations
- [x] Comprehensive test suite for batched inference created
- [x] Comprehensive test suite for MCTS backends created
- [x] Documentation updated (README.md)
- [x] Changelog created (CHANGELOG.md)
- [x] Bug fix summary documented

---

## Technical Notes

### GameResult Dataclass Structure:
```python
@dataclass(frozen=True)
class GameResult:
    winner: Optional[bool]  # True=White, False=Black, None=Draw
    termination: str        # "checkmate", "stalemate", etc.

    @property
    def value_for_white(self) -> float:
        """Get result value from white's perspective."""
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner else -1.0
```

### Why the Bug Occurred:
1. The project plan (line 112) documented `get_result()` as returning a float
2. The actual implementation returns a `GameResult` dataclass
3. The regular actor (`selfplay/game.py`) correctly extracts `.value_for_white`
4. The batched actor was implemented without checking the actual return type
5. Python's type system didn't catch this at development time (no static type checking)

### Prevention for Future:
- Consider adding type hints to all public methods
- Run mypy or pyright for static type checking
- Ensure all code paths follow the same patterns (regular actor vs batched actor)
- Add integration tests early in development

---

## Performance Impact

The bug fix has **no performance impact** - it's purely a correctness fix. The batched inference system now works as intended:

- Actors send inference requests to centralized GPU server
- Server batches requests for efficient GPU utilization
- Actors receive responses and continue MCTS
- Games complete successfully and generate training trajectories

**Expected Performance:**
- Batched inference: ~2-3x faster than individual CPU inference per actor
- Scales well with 4+ actors
- Better GPU utilization with larger batch sizes
