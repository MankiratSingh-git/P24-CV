"""
Video Data Augmentation Module for FixMatch and F2-FixMatch.
Faithfully ported for video action recognition:
1. Synchronized Weak Augmentations: Shared random crop coordinates & flips across modalities (RGB + TG).
2. Video RandAugment: Uniformly samples operations and randomized signs (+/-) across frames.
3. Video Cutout: Spatially masks a 128x128 square patch across all frames.
"""

import math
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# Standard ImageNet normalization parameters used in 3D-ResNet-18
IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


# ==============================================================================
# 1. Video RandAugment Transformations (Temporally Coherent with Random Sign)
# ==============================================================================

def _apply_op_to_clip(clip, op_fn, val):
    """Apply an image operation to all PIL frames in a video clip using identical parameters."""
    return [op_fn(frame, val) for frame in clip]


def _auto_contrast(img, _):
    return ImageOps.autocontrast(img)


def _equalize(img, _):
    return ImageOps.equalize(img)


def _invert(img, _):
    return ImageOps.invert(img)


def _rotate(img, val):
    # val is in degrees [-30, 30]
    return img.rotate(val, resample=Image.BILINEAR)


def _posterize(img, val):
    # val in [4, 8] bits
    bits = max(1, min(8, int(val)))
    return ImageOps.posterize(img, bits)


def _solarize(img, val):
    # val in [0, 256] threshold
    thresh = int(val)
    return ImageOps.solarize(img, thresh)


def _solarize_add(img, val):
    threshold = 128
    lut = []
    for i in range(256):
        if i < threshold:
            lut.append(min(255, i + int(val)))
        else:
            lut.append(i)
    return img.point(lut)


def _color(img, val):
    # val in [0.1, 1.9]
    return ImageEnhance.Color(img).enhance(val)


def _contrast(img, val):
    # val in [0.1, 1.9]
    return ImageEnhance.Contrast(img).enhance(val)


def _brightness(img, val):
    # val in [0.1, 1.9]
    return ImageEnhance.Brightness(img).enhance(val)


def _sharpness(img, val):
    # val in [0.1, 1.9]
    return ImageEnhance.Sharpness(img).enhance(val)


def _shear_x(img, val):
    # val in [-0.3, 0.3]
    return img.transform(img.size, Image.AFFINE, (1, val, 0, 0, 1, 0), resample=Image.BILINEAR)


def _shear_y(img, val):
    # val in [-0.3, 0.3]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, val, 1, 0), resample=Image.BILINEAR)


def _translate_x(img, val):
    # val in [-0.3, 0.3] fraction of width
    offset = int(val * img.size[0])
    return img.transform(img.size, Image.AFFINE, (1, 0, offset, 0, 1, 0), resample=Image.BILINEAR)


def _translate_y(img, val):
    # val in [-0.3, 0.3] fraction of height
    offset = int(val * img.size[1])
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, offset), resample=Image.BILINEAR)


class VideoRandAugment:
    """
    Video RandAugment policy matching FixMatch / MMAction2:
    Randomly chooses N transformations from a pool of 13 operations,
    evaluating them with random sign (+/-) and applying identically across all frames.
    """
    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m

    def _get_random_sign(self):
        return 1.0 if random.random() < 0.5 else -1.0

    def _sample_op(self):
        sign = self._get_random_sign()
        ops = [
            (_auto_contrast, 0),
            (_equalize, 0),
            (_rotate, sign * float(self.m) / 30.0 * 30.0),                     # [-30, 30] deg
            (_solarize, 256.0 - float(self.m) / 30.0 * 256.0),                # [0, 256]
            (_color, 1.0 + sign * float(self.m) / 30.0 * 0.9),                # [0.1, 1.9]
            (_contrast, 1.0 + sign * float(self.m) / 30.0 * 0.9),             # [0.1, 1.9]
            (_brightness, 1.0 + sign * float(self.m) / 30.0 * 0.9),           # [0.1, 1.9]
            (_sharpness, 1.0 + sign * float(self.m) / 30.0 * 0.9),            # [0.1, 1.9]
            (_shear_x, sign * float(self.m) / 30.0 * 0.3),                    # [-0.3, 0.3]
            (_shear_y, sign * float(self.m) / 30.0 * 0.3),                    # [-0.3, 0.3]
            (_translate_x, sign * float(self.m) / 30.0 * 0.3),                # [-0.3, 0.3]
            (_translate_y, sign * float(self.m) / 30.0 * 0.3),                # [-0.3, 0.3]
            (_posterize, 8 - int(float(self.m) / 30.0 * 4.0)),                # [4, 8]
        ]
        return random.choice(ops)

    def __call__(self, clip):
        sampled_ops = [self._sample_op() for _ in range(self.n)]
        processed = clip
        for op, val in sampled_ops:
            processed = _apply_op_to_clip(processed, op, val)
        return processed, sampled_ops

    def apply_preset_ops(self, clip, sampled_ops):
        """Apply already-sampled operations to a second modality (e.g. TG) for exact alignment."""
        processed = clip
        for op, val in sampled_ops:
            processed = _apply_op_to_clip(processed, op, val)
        return processed


# ==============================================================================
# 2. Synchronized Video Cutout (128x128 Patch across all frames)
# ==============================================================================

class VideoCutout:
    """
    Applies spatial cutout of a fixed size (128x128 in paper Section 4.2).
    The square patch is zeroed/masked consistently across all frames of the clip.
    """
    def __init__(self, size=128):
        self.size = size

    def sample_patch(self, w, h):
        half = self.size // 2
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        return x1, y1, x2, y2

    def __call__(self, clip, patch=None):
        if len(clip) == 0:
            return clip, patch
        w, h = clip[0].size
        if patch is None:
            patch = self.sample_patch(w, h)
        x1, y1, x2, y2 = patch

        processed = []
        for frame in clip:
            f = frame.copy()
            # Draw black box over cutout region
            black_box = Image.new('RGB', (x2 - x1, y2 - y1), (0, 0, 0))
            f.paste(black_box, (x1, y1))
            processed.append(f)
        return processed, patch


# ==============================================================================
# 3. Synchronized Spatial Resizing, Cropping, and Flipping
# ==============================================================================

def sample_crop_parameters(orig_w, orig_h, min_size=256, max_size=320, crop_size=224, is_train=True):
    """Samples geometric parameters once so multiple modalities share identical crops and flips."""
    if is_train:
        target_short = random.randint(min_size, max_size)
    else:
        target_short = min_size

    if orig_h < orig_w:
        new_h = target_short
        new_w = int(round(orig_w * (target_short / float(orig_h))))
    else:
        new_w = target_short
        new_h = int(round(orig_h * (target_short / float(orig_w))))

    new_w = max(new_w, crop_size)
    new_h = max(new_h, crop_size)

    if is_train:
        crop_x = random.randint(0, new_w - crop_size)
        crop_y = random.randint(0, new_h - crop_size)
        flip = (random.random() < 0.5)
    else:
        crop_x = (new_w - crop_size) // 2
        crop_y = (new_h - crop_size) // 2
        flip = False

    return {
        'new_w': new_w,
        'new_h': new_h,
        'crop_x': crop_x,
        'crop_y': crop_y,
        'crop_size': crop_size,
        'flip': flip
    }


def apply_crop_parameters(pil_frames, params):
    """Applies preset spatial crop and flip parameters to a video clip."""
    if len(pil_frames) == 0:
        return pil_frames

    new_w = params['new_w']
    new_h = params['new_h']
    crop_x = params['crop_x']
    crop_y = params['crop_y']
    crop_size = params['crop_size']
    flip = params['flip']

    processed = []
    for frame in pil_frames:
        resized = frame.resize((new_w, new_h), Image.BILINEAR)
        cropped = resized.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))
        if flip:
            cropped = ImageOps.mirror(cropped)
        processed.append(cropped)

    return processed


def normalize_clip(frames_array):
    """
    Input: numpy array of shape (T, H, W, 3) in uint8 [0, 255].
    Output: torch.Tensor of shape (3, T, H, W) normalized with ImageNet mean and std.
    """
    import torch
    arr = frames_array.astype(np.float32)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)  # (3, T, H, W)
    return tensor
