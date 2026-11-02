"""
Reproducibility utilities for MMAction2.
"""

import os
import random
import numpy as np
import torch


def set_random_seed(seed=42, deterministic=False):
    """
    Set random seeds across Python, NumPy, and PyTorch for exact experiment reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_rng_states():
    """Capture full RNG states across Python, NumPy, and PyTorch (CPU + all GPUs)."""
    return {
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
    }


def set_rng_states(states):
    """Restore full RNG states to exactly continue an interrupted experiment."""
    if not states or not isinstance(states, dict):
        return
    if 'torch_rng_state' in states and states['torch_rng_state'] is not None:
        torch.set_rng_state(states['torch_rng_state'])
    if 'cuda_rng_state' in states and states['cuda_rng_state'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states['cuda_rng_state'])
    if 'numpy_rng_state' in states and states['numpy_rng_state'] is not None:
        np.random.set_state(states['numpy_rng_state'])
    if 'python_rng_state' in states and states['python_rng_state'] is not None:
        random.setstate(states['python_rng_state'])


def load_checkpoint(filepath, map_location=None):
    """
    Safely load checkpoints across PyTorch versions, handling PyTorch 2.6+
    default weights_only=True restriction when unpickling optimizer and RNG states.
    """
    try:
        return torch.load(filepath, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(filepath, map_location=map_location)

