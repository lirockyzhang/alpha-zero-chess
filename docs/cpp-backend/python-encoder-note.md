# Note on Python Encoder (alphazero/chess_env/encoding.py)

## Status: NOT UPDATED (Intentional)

The Python position encoder in `alphazero/chess_env/encoding.py` still uses the old 119-channel format and was **intentionally not updated** for the following reasons:

### Why Not Updated

1. **Not Used in Training Pipeline**:
   - The C++ encoder (`alphazero-cpp/src/encoding/position_encoder.cpp`) is the primary encoding path
   - Used for all self-play, training, and inference
   - Much faster than Python implementation

2. **Legacy/Reference Code**:
   - Python encoder appears to be for reference or testing only
   - Used in some demo scripts but not in production training
   - `GameState` class is imported but encoding functions are not called

3. **Different Architecture**:
   - Python encoder has a different channel layout than C++
   - Would require significant restructuring to match C++ layout
   - Not worth the effort for unused code

### Usage Analysis

Files that import from `chess_env`:
- Import `GameState` only (game logic)
- Do NOT import or use `encode_position()` or encoding functions
- C++ backend handles all position encoding

### If You Need to Update Python Encoder

If you ever need the Python encoder to match the C++ version (unlikely), here's what would be needed:

1. Change `TOTAL_PLANES = 119` to `TOTAL_PLANES = 122`
2. Restructure history planes to use 8 × 13 layout (currently uses 8 × 12)
3. Add repetition marker plane for each historical position
4. Update all channel offsets and plane indices

**Estimated effort**: 2-3 hours
**Value**: Low (since it's not used)

### Recommendation

**Leave the Python encoder as-is** unless:
- You need to use Python-only MCTS backend
- You're benchmarking Python vs C++ encoding
- You need it for testing or validation

For all production use cases, the C++ encoder at 122 channels is sufficient and much faster.

---

**Bottom line**: The 122-channel update is complete for the production C++ backend. The Python encoder doesn't need updating since it's not used in the training pipeline.
