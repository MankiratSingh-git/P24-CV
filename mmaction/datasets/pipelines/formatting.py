"""
Formatting, Normalization, and Collection transforms in MMAction2.
"""

import numpy as np
import torch
from ..builder import PIPELINES

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _normalize_frames(frames):
    """Input: list of PIL Images. Output: torch.Tensor of shape (3, T, H, W)."""
    arr = np.stack([np.array(f, dtype=np.float32) for f in frames], axis=0)  # (T, H, W, C)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)  # (C, T, H, W)
    return tensor


@PIPELINES.register_module()
class Normalize:
    """Normalize video frames with ImageNet mean and standard deviation."""

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        imgs = results['imgs']
        if isinstance(imgs[0], list):
            # List of clips (e.g. 30 views)
            results['imgs'] = torch.stack([_normalize_frames(c) for c in imgs], dim=0)  # (30, C, T, H, W)
        else:
            results['imgs'] = _normalize_frames(imgs)  # (C, T, H, W)

        if 'imgs_tg' in results:
            results['imgs_tg'] = _normalize_frames(results['imgs_tg'])

        return results


@PIPELINES.register_module()
class FormatShape:
    """Format input tensor shape."""

    def __init__(self, input_format='NCTHW'):
        self.input_format = input_format

    def __call__(self, results):
        return results


@PIPELINES.register_module()
class ToTensor:
    """Convert labels and extra items to tensors."""

    def __init__(self, keys=['imgs', 'label']):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            if key in results:
                val = results[key]
                if isinstance(val, (int, float)):
                    results[key] = torch.tensor(val, dtype=torch.long)
                elif isinstance(val, np.ndarray):
                    results[key] = torch.from_numpy(val)
        return results


@PIPELINES.register_module()
class Collect:
    """Collect specified keys from the results dict."""

    def __init__(self, keys, meta_keys=()):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        for key in self.keys:
            if key in results:
                data[key] = results[key]
        return data
