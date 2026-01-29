# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Critical bug in batched inference**: Fixed `TypeError: bad operand type for unary -: 'GameResult'` in `batched_actor.py`
  - `state.get_result()` returns a `GameResult` dataclass, not a float
  - The code was attempting to negate the `GameResult` object directly: `ts.value = result if ts.player == 0 else -result`
  - Fixed by extracting the numeric value first: `result = game_result.value_for_white`
  - This bug affected both `BatchedActorProcess._play_game()` (line 166) and `BatchedActor.play_game()` (line 225)
  - Impact: Batched inference actors would crash when games completed, preventing training with `--batched-inference` flag

### Added
- **Comprehensive test suite for batched inference** (`tests/test_batched_inference.py`)
  - Unit tests for `InferenceRequest` and `InferenceResponse` dataclasses
  - Unit tests for `BatchedEvaluator` including timeout handling
  - Integration tests for `InferenceServer` with single and batched requests
  - Tests for weight updates in inference server
  - Tests for `BatchedActor` playing complete games
  - Multi-actor integration tests simulating parallel self-play
  - Edge case tests for missing response queues

- **Comprehensive test suite for MCTS backends** (`tests/test_mcts_backends.py`)
  - Backend availability detection tests
  - Interface compliance tests for all backends (Python, Cython, C++)
  - Consistency tests comparing outputs across backends
  - Edge case tests: terminal positions, few legal moves, zero simulations
  - Performance comparison tests (informational)
  - Cython-specific tests with Python comparison
  - C++-specific tests with Python comparison and memory safety checks

### Changed
- Updated `README.md` to include new test modules in testing section
- Added documentation for `test_mcts_backends.py` and `test_batched_inference.py`

### Technical Details

#### Bug Root Cause
The `GameState.get_result()` method returns a `GameResult` dataclass (defined in `chess_env/board.py:16-27`) with the following structure:
```python
@dataclass(frozen=True)
class GameResult:
    winner: Optional[bool]  # True=White, False=Black, None=Draw
    termination: str

    @property
    def value_for_white(self) -> float:
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner else -1.0
```

The batched actor code was treating this as a numeric value and attempting to negate it directly, which Python doesn't support for custom objects unless `__neg__()` is implemented.

#### Fix Implementation
Changed from:
```python
result = state.get_result()
ts.value = result if ts.player == 0 else -result  # ❌ TypeError
```

To:
```python
game_result = state.get_result()
result = game_result.value_for_white  # ✅ Extract float first
ts.value = result if ts.player == 0 else -result  # ✅ Now works
```

This matches the pattern used in the regular actor (`selfplay/game.py:96-97`).

#### Test Coverage
The new test suites provide comprehensive coverage:

**Batched Inference Tests:**
- 15+ test cases covering the entire batched inference pipeline
- Tests run actual inference server processes with multiprocessing
- Validates request/response flow, batching behavior, and weight updates
- Integration tests with multiple actors running in parallel

**MCTS Backend Tests:**
- 20+ test cases for backend implementations
- Parametrized tests that run against all available backends
- Consistency checks ensure different implementations produce similar results
- Edge case coverage for terminal positions, limited moves, and error conditions

## [0.1.0] - 2026-01-28

### Added
- Initial implementation of AlphaZero chess engine
- Neural network with ResNet architecture (15 blocks, 192 filters)
- MCTS with PUCT selection and Dirichlet noise
- Multi-backend MCTS support (Python, Cython, C++)
- Self-play pipeline with multi-process actors
- Batched GPU inference server for efficient multi-actor training
- Training pipeline with replay buffer and SGD optimizer
- Evaluation against Stockfish
- Comprehensive test suite for core components
