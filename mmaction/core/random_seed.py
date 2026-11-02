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
    try:
        if 'torch_rng_state' in states and states['torch_rng_state'] is not None:
            torch_state = states['torch_rng_state']
            if isinstance(torch_state, torch.Tensor):
                torch_state = torch_state.cpu()
            torch.set_rng_state(torch_state)
        if 'cuda_rng_state' in states and states['cuda_rng_state'] is not None and torch.cuda.is_available():
            cuda_states = states['cuda_rng_state']
            if isinstance(cuda_states, list):
                cuda_states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in cuda_states]
            elif isinstance(cuda_states, torch.Tensor):
                cuda_states = cuda_states.cpu()
            torch.cuda.set_rng_state_all(cuda_states)
        if 'numpy_rng_state' in states and states['numpy_rng_state'] is not None:
            np.random.set_state(states['numpy_rng_state'])
        if 'python_rng_state' in states and states['python_rng_state'] is not None:
            random.setstate(states['python_rng_state'])
    except Exception as e:
        print(f"=> [Warning] Could not restore RNG state: {e}. Continuing with active RNG.")


def load_checkpoint(filepath, map_location=None):
    """
    Safely load checkpoints across PyTorch versions, handling PyTorch 2.6+
    default weights_only=True restriction when unpickling optimizer and RNG states.
    """
    try:
        return torch.load(filepath, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(filepath, map_location=map_location)


def safe_load_optimizer_state(optimizer, state_dict, model=None):
    """
    Safely load optimizer state dict, handling parameter count mismatches
    (e.g., when new heads like C_TAP are decoupled during architecture evolution).
    Preserves momentum buffers and hyperparameters for existing weights.
    """
    if state_dict is None or not isinstance(state_dict, dict):
        return
    try:
        optimizer.load_state_dict(state_dict)
    except (ValueError, KeyError) as e:
        print(f"=> [Note] Adapting optimizer state to match current parameters: {e}")
        try:
            saved_param_groups = state_dict.get('param_groups', [])
            if saved_param_groups and optimizer.param_groups:
                saved_group = saved_param_groups[0]
                cur_group = optimizer.param_groups[0]
                for k, v in saved_group.items():
                    if k != 'params':
                        cur_group[k] = v
                num_saved = len(saved_group.get('params', []))
                saved_states = state_dict.get('state', {})
                if model is not None:
                    cur_params = list(model.parameters())
                    for old_idx in range(min(num_saved, len(cur_params))):
                        if old_idx in saved_states:
                            cur_p = cur_params[old_idx]
                            optimizer.state[cur_p] = saved_states[old_idx]
                    print(f"=> Restored momentum buffers for {len(optimizer.state)}/{len(cur_params)} parameters.")
        except Exception as adapt_err:
            print(f"=> [Warning] Could not adapt optimizer state: {adapt_err}. Optimizer will reinitialize momentum.")

