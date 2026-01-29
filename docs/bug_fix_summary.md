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
