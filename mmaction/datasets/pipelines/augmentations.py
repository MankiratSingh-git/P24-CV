"""
Spatial, RandAugment, and Cutout Augmentations for MMAction2.
Section 4.2 of Tu et al. (ACM TOMM 2025):
- Weak: Resize [256, 320], Random crop 224x224, Horizontal flip 0.5.
- Strong: Video RandAugment (13 operations, uniform +/- sampling) + Cutout (128x128).
Synchronized across RGB and TG modalities.
"""

import math
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from ..builder import PIPELINES


# ==============================================================================
# Helper transformations for RandAugment
# ==============================================================================

def _apply_op(img, op_name, val):
    if op_name == 'autocontrast':
        return ImageOps.autocontrast(img)
    elif op_name == 'equalize':
        return ImageOps.equalize(img)
    elif op_name == 'rotate':
        return img.rotate(val, resample=Image.BILINEAR)
    elif op_name == 'solarize':
        return ImageOps.solarize(img, int(val))
    elif op_name == 'color':
        return ImageEnhance.Color(img).enhance(val)
    elif op_name == 'contrast':
        return ImageEnhance.Contrast(img).enhance(val)
    elif op_name == 'brightness':
        return ImageEnhance.Brightness(img).enhance(val)
    elif op_name == 'sharpness':
        return ImageEnhance.Sharpness(img).enhance(val)
    elif op_name == 'shear_x':
        return img.transform(img.size, Image.AFFINE, (1, val, 0, 0, 1, 0), resample=Image.BILINEAR)
    elif op_name == 'shear_y':
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, val, 1, 0), resample=Image.BILINEAR)
    elif op_name == 'translate_x':
        offset = int(val * img.size[0])
        return img.transform(img.size, Image.AFFINE, (1, 0, offset, 0, 1, 0), resample=Image.BILINEAR)
    elif op_name == 'translate_y':
        offset = int(val * img.size[1])
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, offset), resample=Image.BILINEAR)
    elif op_name == 'posterize':
        bits = max(1, min(8, int(val)))
        return ImageOps.posterize(img, bits)
    return img


@PIPELINES.register_module()
class VideoRandAugment:
    """
    Video RandAugment applying N operations with magnitude M across all frames uniformly.
    Samples random signs (+/-) for directional operations.
    """

    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m

    def _sample_ops(self):
        ops = []
        for _ in range(self.n):
            sign = 1.0 if random.random() < 0.5 else -1.0
            choices = [
                ('autocontrast', 0),
                ('equalize', 0),
                ('rotate', sign * float(self.m) / 30.0 * 30.0),
                ('solarize', 256.0 - float(self.m) / 30.0 * 256.0),
                ('color', 1.0 + sign * float(self.m) / 30.0 * 0.9),
                ('contrast', 1.0 + sign * float(self.m) / 30.0 * 0.9),
                ('brightness', 1.0 + sign * float(self.m) / 30.0 * 0.9),
                ('sharpness', 1.0 + sign * float(self.m) / 30.0 * 0.9),
                ('shear_x', sign * float(self.m) / 30.0 * 0.3),
                ('shear_y', sign * float(self.m) / 30.0 * 0.3),
                ('translate_x', sign * float(self.m) / 30.0 * 0.3),
                ('translate_y', sign * float(self.m) / 30.0 * 0.3),
                ('posterize', 8 - int(float(self.m) / 30.0 * 4.0)),
            ]
            ops.append(random.choice(choices))
        return ops

    def apply_ops_to_clip(self, clip, ops):
        processed = clip
        for op_name, val in ops:
            processed = [_apply_op(f, op_name, val) for f in processed]
        return processed

    def __call__(self, results):
        ops = self._sample_ops()
        results['imgs'] = self.apply_ops_to_clip(results['imgs'], ops)
        if 'imgs_tg' in results:
            results['imgs_tg'] = self.apply_ops_to_clip(results['imgs_tg'], ops)
        results['randaug_ops'] = ops
        return results


@PIPELINES.register_module()
class VideoCutout:
    """
    Video Cutout: masks a 128x128 spatial patch across all frames.
    Synchronized across RGB and TG modalities.
    """

    def __init__(self, size=128):
        self.size = size

    def __call__(self, results):
        clip = results['imgs']
        if len(clip) == 0:
            return results

        w, h = clip[0].size
        half = self.size // 2
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

        patch = Image.new('RGB', (x2 - x1, y2 - y1), (0, 0, 0))

        processed_rgb = []
        for frame in clip:
            f = frame.copy()
            f.paste(patch, (x1, y1))
            processed_rgb.append(f)
        results['imgs'] = processed_rgb

        if 'imgs_tg' in results:
            processed_tg = []
            for frame in results['imgs_tg']:
                f = frame.copy()
                f.paste(patch, (x1, y1))
                processed_tg.append(f)
            results['imgs_tg'] = processed_tg

        results['cutout_patch'] = (x1, y1, x2, y2)
        return results


@PIPELINES.register_module()
class RandomResizedCrop:
    """
    Weak spatial cropping transform:
    Resizes shorter side to [min_size, max_size], extracts crop_size x crop_size, and flips (p=0.5).
    Synchronized identically between RGB and TG.
    """

    def __init__(self, min_size=256, max_size=320, crop_size=224, flip_ratio=0.5, is_train=True):
        self.min_size = min_size
        self.max_size = max_size
        self.crop_size = crop_size
        self.flip_ratio = flip_ratio
        self.is_train = is_train

    def _sample_params(self, orig_w, orig_h):
        if self.is_train:
            target_short = random.randint(self.min_size, self.max_size)
            flip = (random.random() < self.flip_ratio)
        else:
            target_short = self.min_size
            flip = False

        if orig_h < orig_w:
            new_h = target_short
            new_w = int(round(orig_w * (target_short / float(orig_h))))
        else:
            new_w = target_short
            new_h = int(round(orig_h * (target_short / float(orig_w))))

        new_w = max(new_w, self.crop_size)
        new_h = max(new_h, self.crop_size)

        if self.is_train:
            crop_x = random.randint(0, new_w - self.crop_size)
            crop_y = random.randint(0, new_h - self.crop_size)
        else:
            crop_x = (new_w - self.crop_size) // 2
            crop_y = (new_h - self.crop_size) // 2

        return new_w, new_h, crop_x, crop_y, flip

    def _apply_to_clip(self, clip, params):
        new_w, new_h, crop_x, crop_y, flip = params
        processed = []
        for frame in clip:
            resized = frame.resize((new_w, new_h), Image.BILINEAR)
            cropped = resized.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))
            if flip:
                cropped = ImageOps.mirror(cropped)
            processed.append(cropped)
        return processed

    def __call__(self, results):
        clip = results['imgs']
        orig_w, orig_h = clip[0].size
        params = self._sample_params(orig_w, orig_h)

        results['imgs'] = self._apply_to_clip(clip, params)
        if 'imgs_tg' in results:
            results['imgs_tg'] = self._apply_to_clip(results['imgs_tg'], params)

        results['crop_params'] = params
        return results


@PIPELINES.register_module()
class CenterCrop:
    """Center crop for validation / testing."""

    def __init__(self, crop_size=224, scale=256):
        self.crop_size = crop_size
        self.scale = scale

    def _apply_to_clip(self, clip):
        orig_w, orig_h = clip[0].size
        if orig_h < orig_w:
            new_h = self.scale
            new_w = int(round(orig_w * (self.scale / float(orig_h))))
        else:
            new_w = self.scale
            new_h = int(round(orig_h * (self.scale / float(orig_w))))

        crop_x = (new_w - self.crop_size) // 2
        crop_y = (new_h - self.crop_size) // 2

        processed = []
        for frame in clip:
            resized = frame.resize((new_w, new_h), Image.BILINEAR)
            cropped = resized.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))
            processed.append(cropped)
        return processed

    def __call__(self, results):
        clip = results['imgs']
        if isinstance(clip[0], list):
            # Multiple clips
            results['imgs'] = [self._apply_to_clip(c) for c in clip]
        else:
            results['imgs'] = self._apply_to_clip(clip)
        return results
