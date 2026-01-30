#!/usr/bin/env python3
"""Launch the training visualization dashboard.

Monitors training metrics in real-time and displays interactive plots.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphazero.visualization.dashboard import main

if __name__ == "__main__":
    main()
