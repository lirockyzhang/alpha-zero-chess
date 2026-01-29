"""Utility functions for AlphaZero."""

import re
from pathlib import Path
from typing import Optional, Tuple


def parse_checkpoint_architecture(checkpoint_path: str) -> Optional[Tuple[int, int]]:
    """Parse network architecture from checkpoint filename.

    Expects filename format: checkpoint_<step>_f<filters>_b<blocks>.pt
    Example: checkpoint_1000_f64_b5.pt -> (64, 5)

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Tuple of (num_filters, num_blocks) if found, None otherwise
    """
    filename = Path(checkpoint_path).name

    # Pattern: checkpoint_<step>_f<filters>_b<blocks>.pt
    pattern = r'checkpoint_\d+_f(\d+)_b(\d+)\.pt'
    match = re.match(pattern, filename)

    if match:
        num_filters = int(match.group(1))
        num_blocks = int(match.group(2))
        return (num_filters, num_blocks)

    return None


def load_checkpoint_with_architecture(checkpoint_path: str, device: str = "cuda"):
    """Load checkpoint and return state dict with architecture info.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on

    Returns:
        Tuple of (state_dict, num_filters, num_blocks)
    """
    import torch

    state = torch.load(checkpoint_path, map_location=device)

    # Try to get architecture from checkpoint metadata (new format)
    if 'num_filters' in state and 'num_blocks' in state:
        return state, state['num_filters'], state['num_blocks']

    # Try to parse from filename (fallback)
    arch = parse_checkpoint_architecture(checkpoint_path)
    if arch:
        return state, arch[0], arch[1]

    # No architecture info found
    return state, None, None
