"""
Semi-Supervised Dataset for MMAction2 (FixMatch and F2-FixMatch).
Handles bi-modal generation (RGB and Temporal Gradient) and produces
synchronized Weak and Strong augmentation pairs.
"""

import copy
from torch.utils.data import Dataset
from .builder import DATASETS
from .rawframe_dataset import RawframeDataset
from .pipelines.compose import Compose
from .pipelines.sampling import SampleFrames
from .pipelines.loading import RawFrameDecode
from .pipelines.augmentations import RandomResizedCrop, VideoRandAugment, VideoCutout
from .pipelines.formatting import Normalize


@DATASETS.register_module()
class SemiRawframeDataset(RawframeDataset):
    """
    Unlabeled semi-supervised dataset returning synchronized Weak & Strong views.
    Optionally includes Temporal Gradient (TG) modality for F2-FixMatch.
    """

    def __init__(self,
                 ann_file,
                 data_prefix=None,
                 include_tg=False,
                 filename_tmpl='img_{:05d}.jpg',
                 start_index=0,
                 **kwargs):
        # We handle the weak and strong pipelines internally to guarantee strict modality synchronization
        super().__init__(ann_file=ann_file, pipeline=[], data_prefix=data_prefix,
                         test_mode=False, filename_tmpl=filename_tmpl, start_index=start_index)
        self.include_tg = include_tg

        self.sampler = SampleFrames(clip_len=8, frame_interval=8, num_clips=1, test_mode=False)
        self.decoder = RawFrameDecode(filename_tmpl=filename_tmpl, include_tg=include_tg)
        self.weak_cropper = RandomResizedCrop(min_size=256, max_size=320, crop_size=224, flip_ratio=0.5, is_train=True)
        self.rand_aug = VideoRandAugment(n=2, m=10)
        self.cutout = VideoCutout(size=128)
        self.strong_cropper = RandomResizedCrop(min_size=256, max_size=320, crop_size=224, flip_ratio=0.5, is_train=True)
        self.normalizer = Normalize()

    def __getitem__(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['start_index'] = self.start_index
        results['filename_tmpl'] = self.filename_tmpl

        # 1. Temporal sampling (stride 8 over 64 frames)
        results = self.sampler(results)

        # 2. Frame decoding & optional dynamic Temporal Gradient computation
        results = self.decoder(results)
        raw_rgb = results['imgs']
        raw_tg = results.get('imgs_tg', None)

        # 3. Weak Augmentation (shared crop & flip parameters across RGB and TG)
        weak_dict = {'imgs': [f.copy() for f in raw_rgb]}
        if self.include_tg and raw_tg is not None:
            weak_dict['imgs_tg'] = [f.copy() for f in raw_tg]

        weak_dict = self.weak_cropper(weak_dict)
        weak_dict = self.normalizer(weak_dict)
        rgb_w = weak_dict['imgs']
        tg_w = weak_dict.get('imgs_tg', None)

        # 4. Strong Augmentation (RandAugment + Cutout + Crop synchronized across RGB and TG)
        strong_dict = {'imgs': [f.copy() for f in raw_rgb]}
        if self.include_tg and raw_tg is not None:
            strong_dict['imgs_tg'] = [f.copy() for f in raw_tg]

        strong_dict = self.rand_aug(strong_dict)
        strong_dict = self.cutout(strong_dict)
        strong_dict = self.strong_cropper(strong_dict)
        strong_dict = self.normalizer(strong_dict)
        rgb_s = strong_dict['imgs']
        tg_s = strong_dict.get('imgs_tg', None)

        if self.include_tg and tg_w is not None and tg_s is not None:
            return {
                'rgb_w': rgb_w,
                'rgb_s': rgb_s,
                'tg_w': tg_w,
                'tg_s': tg_s
            }

        return {
            'imgs_weak': rgb_w,
            'imgs_strong': rgb_s
        }
