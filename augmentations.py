import random
import math
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# Normalization constants (ImageNet standard)
IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
FILL_COLOR = (124, 116, 104)


# -------------------------------------------------------------
# Video-Level RandAugment Primitive Operations
# -------------------------------------------------------------
def _auto_contrast(img, _):
    return ImageOps.autocontrast(img)


def _equalize(img, _):
    return ImageOps.equalize(img)


def _rotate(img, v, fillcolor=FILL_COLOR):
    return img.rotate(v, fillcolor=fillcolor)


def _solarize(img, v):
    return ImageOps.solarize(img, int(v))


def _color(img, v):
    return ImageEnhance.Color(img).enhance(v)


def _contrast(img, v):
    return ImageEnhance.Contrast(img).enhance(v)


def _brightness(img, v):
    return ImageEnhance.Brightness(img).enhance(v)


def _sharpness(img, v):
    return ImageEnhance.Sharpness(img).enhance(v)


def _shear_x(img, v, fillcolor=FILL_COLOR):
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0), fillcolor=fillcolor)


def _shear_y(img, v, fillcolor=FILL_COLOR):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0), fillcolor=fillcolor)


def _translate_x(img, v, fillcolor=FILL_COLOR):
    # v in [-0.3, 0.3] relative to width
    shift = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, shift, 0, 1, 0), fillcolor=fillcolor)


def _translate_y(img, v, fillcolor=FILL_COLOR):
    # v in [-0.3, 0.3] relative to height
    shift = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, shift), fillcolor=fillcolor)


def _posterize(img, v):
    bits = max(1, min(8, int(v)))
    return ImageOps.posterize(img, bits)


# List of candidate RandAugment operations with (function, min_val, max_val)
RANDAUG_CANDIDATES = [
    (_auto_contrast, 0, 1),
    (_equalize, 0, 1),
    (_rotate, -30, 30),
    (_solarize, 0, 256),
    (_color, 0.05, 0.95),
    (_contrast, 0.05, 0.95),
    (_brightness, 0.05, 0.95),
    (_sharpness, 0.05, 0.95),
    (_shear_x, -0.3, 0.3),
    (_shear_y, -0.3, 0.3),
    (_translate_x, -0.3, 0.3),
    (_translate_y, -0.3, 0.3),
    (_posterize, 4, 8),
]


class VideoRandAugment:
    """
    Video-level RandAugment from FixMatch (Sohn et al., 2020).
    Sampled operations and parameters are applied consistently across all frames of a clip
    to maintain temporal coherence.
    """
    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m

    def __call__(self, pil_frames):
        # Choose n random operations
        chosen_ops = random.choices(RANDAUG_CANDIDATES, k=self.n)
        
        # Determine continuous parameter values based on magnitude m in [0, 30]
        params = []
        for op, minval, maxval in chosen_ops:
            val = (float(self.m) / 30.0) * float(maxval - minval) + minval
            params.append((op, val))

        # Apply each operation to all frames in the clip
        augmented_frames = []
        for img in pil_frames:
            res_img = img
            for op, val in params:
                res_img = op(res_img, val)
            augmented_frames.append(res_img)

        return augmented_frames


class VideoCutout:
    """
    Video-level Cutout: removes a square patch (e.g. 128x128 as specified in F2-FixMatch Section 4.2)
    at the exact same spatial coordinates across all frames of the clip.
    """
    def __init__(self, size=128):
        self.size = size

    def __call__(self, pil_frames):
        if len(pil_frames) == 0:
            return pil_frames

        w, h = pil_frames[0].size
        cut_w = min(self.size, w)
        cut_h = min(self.size, h)

        x0 = random.randint(0, max(0, w - cut_w))
        y0 = random.randint(0, max(0, h - cut_h))

        cut_frames = []
        for img in pil_frames:
            arr = np.array(img, copy=True)
            arr[y0:y0+cut_h, x0:x0+cut_w] = FILL_COLOR
            cut_frames.append(Image.fromarray(arr))

        return cut_frames


# -------------------------------------------------------------
# Spatial Geometric Transforms for Video Clips
# -------------------------------------------------------------
def video_resize_and_crop(pil_frames, min_size=256, max_size=320, crop_size=224, is_train=True):
    """
    Weak augmentation spatial pipeline:
    1. Resize shorter side randomly sampled in [min_size, max_size] (or min_size for test).
    2. Random crop (train) or center crop (test) to (crop_size, crop_size).
    3. Random horizontal flip with p=0.5 (train only).
    Applied consistently to all frames in the clip.
    """
    if len(pil_frames) == 0:
        return pil_frames

    orig_w, orig_h = pil_frames[0].size

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

    # Crop coordinates
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
    # (T, H, W, 3) -> (T, 3, H, W) -> float -> normalize -> (3, T, H, W)
    arr = frames_array.astype(np.float32)
    # Subtract mean and divide by std
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)  # (3, T, H, W)
    return tensor
