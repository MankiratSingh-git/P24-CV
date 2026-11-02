"""
Multi-Crop transforms for MMAction2 evaluation.
ThreeCrop extracts left/top, center, and right/bottom crops.
When combined with 10 temporal clips, it yields the official 30-view evaluation protocol (Section 5).
"""

from PIL import Image
from ..builder import PIPELINES


@PIPELINES.register_module()
class ThreeCrop:
    """Extract three spatial crops along the longer side."""

    def __init__(self, crop_size=224, scale=256):
        self.crop_size = crop_size
        self.scale = scale

    def _crop_clip(self, clip):
        """Input: list of PIL Images. Output: 3 lists of PIL Images (left, center, right)."""
        w, h = clip[0].size
        if h < w:
            new_h = self.scale
            new_w = int(round(w * (self.scale / float(h))))
        else:
            new_w = self.scale
            new_h = int(round(h * (self.scale / float(w))))

        # Resize all frames
        resized_clip = [f.resize((new_w, new_h), Image.BILINEAR) for f in clip]

        # 3 crop locations along the longer side
        crops = []
        if new_w >= new_h:
            # Horizontal crops: left, center, right
            x_offsets = [0, (new_w - self.crop_size) // 2, new_w - self.crop_size]
            y_offset = (new_h - self.crop_size) // 2
            for x_off in x_offsets:
                crop = [f.crop((x_off, y_offset, x_off + self.crop_size, y_offset + self.crop_size)) for f in resized_clip]
                crops.append(crop)
        else:
            # Vertical crops: top, center, bottom
            x_offset = (new_w - self.crop_size) // 2
            y_offsets = [0, (new_h - self.crop_size) // 2, new_h - self.crop_size]
            for y_off in y_offsets:
                crop = [f.crop((x_offset, y_off, x_offset + self.crop_size, y_off + self.crop_size)) for f in resized_clip]
                crops.append(crop)

        return crops

    def __call__(self, results):
        raw_clips = results['imgs']
        if isinstance(raw_clips[0], list):
            # Already a list of 10 clips: (10, 8)
            all_30_crops = []
            for clip in raw_clips:
                three_crops = self._crop_clip(clip)
                all_30_crops.extend(three_crops)  # 10 * 3 = 30 clips
            results['imgs'] = all_30_crops
        else:
            # Single clip
            results['imgs'] = self._crop_clip(raw_clips)
        return results
