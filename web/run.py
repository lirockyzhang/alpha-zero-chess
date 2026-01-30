#!/usr/bin/env python3
"""Launch the chess web interface.

Starts a Flask web server for playing against trained AlphaZero models.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web.app import main

if __name__ == "__main__":
    main()
