# Codebase Reorganization Summary

**Date:** 2026-01-30

## Overview

This document summarizes the codebase reorganization performed to improve structure, maintainability, and user experience.

## Changes Made

### 1. Web Application Reorganization

**Objective:** Create a dedicated, self-contained web application directory at the root level.

**Changes:**
- **Created:** `web/` directory at project root
- **Moved:** Web application from `alphazero/web/` to `web/`
- **Structure:**
  ```
  web/
  ├── __init__.py          # Python package marker
  ├── app.py               # Flask application (updated with sys.path)
  ├── run.py               # Entry point script (NEW)
  ├── README.md            # Comprehensive web app documentation (NEW)
  ├── templates/           # HTML templates
  │   └── chess.html       # Interactive chessboard interface
  └── static/              # Static assets (CSS, JS, images)
  ```

**Benefits:**
- ✅ **Self-contained:** All web resources in one location
- ✅ **Independent:** Can be deployed separately from main package
- ✅ **Documented:** Dedicated README with API docs and usage examples
- ✅ **Maintainable:** Clear separation of concerns

**Migration:**
```bash
# Old way (deprecated)
python scripts/web_play.py --checkpoint model.pt

# New way
python web/run.py --checkpoint model.pt
```

**Files Updated:**
- `web/app.py`: Added `sys.path.insert(0, ...)` for proper imports
- `web/run.py`: New launcher script
- `web/README.md`: Complete documentation with API reference
- `README.md`: Updated with new web interface instructions

**Files Removed:**
- `scripts/web_play.py` (replaced by `web/run.py`)

---

### 2. Endgame Evaluation Integration

**Objective:** Consolidate evaluation functionality into a single unified script.

**Changes:**
- **Integrated:** Endgame evaluation into `scripts/evaluate.py`
- **Added:** `--opponent endgame` option to main evaluate script
- **Removed:** Standalone `scripts/evaluate_endgames.py`

**New Usage:**
```bash
# Evaluate on endgame positions
python scripts/evaluate.py \
  --checkpoint model.pt \
  --opponent endgame \
  --simulations 400

# Filter by category
python scripts/evaluate.py \
  --checkpoint model.pt \
  --opponent endgame \
  --category basic_mate

# Filter by difficulty
python scripts/evaluate.py \
  --checkpoint model.pt \
  --opponent endgame \
  --difficulty 3
```

**Opponent Types:**
- `random`: Play against random player
- `stockfish`: Play against Stockfish engine
- `self`: Self-play evaluation
- `endgame`: Evaluate on 50 curated endgame positions (NEW)

**Benefits:**
- ✅ **Unified interface:** Single script for all evaluation types
- ✅ **Consistent API:** Same command-line interface across opponents
- ✅ **Reduced duplication:** Shared code for model loading and MCTS
- ✅ **Better discoverability:** Users find all evaluation options in one place

**Files Updated:**
- `scripts/evaluate.py`:
  - Added `endgame` to opponent choices
  - Added `--max-moves`, `--category`, `--difficulty` arguments
  - Imported `EndgameEvaluator` and `ENDGAME_POSITIONS`
  - Added endgame evaluation logic (lines 189-243)
- `README.md`: Updated evaluation examples

**Files Removed:**
- `scripts/evaluate_endgames.py` (functionality moved to `evaluate.py`)

---

## Updated Documentation

### Main README.md

**Sections Updated:**

1. **Play Against the Model** (lines 155-177):
   - Added web interface instructions
   - Included link to `web/README.md`
   - Kept terminal play instructions

2. **Evaluate Model Strength** (lines 178-242):
   - Updated endgame evaluation to use `--opponent endgame`
   - Maintained all filtering options (category, difficulty)
   - Kept quick and full evaluation examples

### New Documentation

1. **`web/README.md`** (NEW):
   - Complete web interface documentation
   - Quick start guide
   - Command-line options reference
   - API endpoint documentation
   - Troubleshooting section
   - Performance tips

---

## Migration Guide

### For Users

**Web Interface:**
```bash
# Old (deprecated)
python scripts/web_play.py --checkpoint model.pt

# New
python web/run.py --checkpoint model.pt
```

**Endgame Evaluation:**
```bash
# Old (deprecated)
python scripts/evaluate_endgames.py --checkpoint model.pt

# New
python scripts/evaluate.py --checkpoint model.pt --opponent endgame
```

### For Developers

**Importing Web App:**
```python
# Old (deprecated)
from alphazero.web.app import ChessWebInterface

# New
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from web.app import ChessWebInterface
```

**Endgame Evaluation:**
```python
# Still works (no change needed)
from alphazero.evaluation.endgame_eval import EndgameEvaluator, ENDGAME_POSITIONS
```

---

## Directory Structure

### Before
```
alpha-zero-chess/
├── alphazero/
│   └── web/              # Web app mixed with package
│       ├── app.py
│       ├── templates/
│       └── static/
├── scripts/
│   ├── web_play.py       # Separate launcher
│   ├── evaluate.py
│   └── evaluate_endgames.py  # Duplicate evaluation logic
```

### After
```
alpha-zero-chess/
├── web/                  # Dedicated web app directory
│   ├── __init__.py
│   ├── app.py           # Updated with sys.path
│   ├── run.py           # New launcher
│   ├── README.md        # Complete documentation
│   ├── templates/
│   └── static/
├── alphazero/
│   └── web/             # Can be deprecated/removed later
├── scripts/
│   └── evaluate.py      # Unified evaluation (includes endgame)
```

---

## Benefits Summary

### Web Application
- **Better organization:** Clear separation from core package
- **Easier deployment:** Self-contained with all resources
- **Better documentation:** Dedicated README with examples
- **Future-ready:** Can add more web features independently

### Evaluation System
- **Unified interface:** Single script for all opponent types
- **Reduced complexity:** Less code duplication
- **Better UX:** Consistent command-line interface
- **Easier maintenance:** Changes apply to all evaluation types

---

## Testing Checklist

- [x] Web interface launches correctly: `python web/run.py --checkpoint model.pt`
- [x] Web interface serves HTML template from `web/templates/`
- [x] Endgame evaluation works: `python scripts/evaluate.py --opponent endgame --checkpoint model.pt`
- [x] Category filtering works: `--category basic_mate`
- [x] Difficulty filtering works: `--difficulty 3`
- [x] Random opponent still works: `--opponent random`
- [x] Stockfish opponent still works: `--opponent stockfish`
- [x] Self-play opponent still works: `--opponent self`
- [x] README.md updated with new instructions
- [x] `web/README.md` created with complete documentation

---

## Future Improvements

### Web Application
- [ ] Add CSS/JS assets to `web/static/`
- [ ] Add game analysis features
- [ ] Add move suggestions/hints
- [ ] Add opening book integration
- [ ] Add game export (PGN format)

### Evaluation System
- [ ] Add parallel endgame evaluation
- [ ] Add custom position evaluation from FEN
- [ ] Add evaluation report generation (PDF/HTML)
- [ ] Add comparison between multiple checkpoints

---

## Backward Compatibility

**Deprecated (but still functional):**
- `scripts/web_play.py` - Use `web/run.py` instead
- `scripts/evaluate_endgames.py` - Use `scripts/evaluate.py --opponent endgame` instead

**Note:** Deprecated scripts have been removed. Users should update their workflows to use the new commands.

---

## Related Issues

- Fixed web interface stale reference bug (see `docs/bug_fix_summary.md`)
- Improved code organization for future development
- Enhanced documentation for better user experience

---

## Contributors

- Reorganization implemented: 2026-01-30
- Documentation updated: 2026-01-30
- Testing completed: 2026-01-30
