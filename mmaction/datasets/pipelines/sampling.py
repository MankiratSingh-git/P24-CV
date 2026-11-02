"""
Frame Sampling transforms in MMAction2.
Section 4.2: clip_len=8, frame_interval=8 covering 64 consecutive frames.
"""

import random
import numpy as np
from ..builder import PIPELINES


@PIPELINES.register_module()
class SampleFrames:
    """
    Sample frames according to clip_len and frame_interval.
    Supports single clip training sampling and multi-clip test sampling (num_clips=10).
    """

    def __init__(self,
                 clip_len=8,
                 frame_interval=8,
                 num_clips=1,
                 temporal_jitter=False,
                 test_mode=False):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.test_mode = test_mode

    def _sample_train_indices(self, total_frames):
        required_span = self.clip_len * self.frame_interval
        if total_frames > required_span:
            start = random.randint(0, total_frames - required_span)
        else:
            start = 0

        indices = []
        next_indices = []
        for i in range(self.clip_len):
            idx = start + i * self.frame_interval
            if total_frames > 0:
                idx = idx % total_frames
            indices.append(idx)
            next_idx = (idx + 1) % total_frames if total_frames > 0 else idx
            next_indices.append(next_idx)

        return np.array(indices), np.array(next_indices)

    def _sample_test_indices(self, total_frames):
        """Uniformly samples num_clips across the video duration for 30-view evaluation."""
        required_span = self.clip_len * self.frame_interval
        if self.num_clips == 1:
            start = max(0, (total_frames - required_span) // 2)
            starts = [start]
        else:
            if total_frames > required_span:
                starts = np.linspace(0, total_frames - required_span, self.num_clips, dtype=int).tolist()
            else:
                starts = [0] * self.num_clips

        all_clips = []
        for start in starts:
            clip_idx = []
            for i in range(self.clip_len):
                idx = start + i * self.frame_interval
                if total_frames > 0:
                    idx = idx % total_frames
                clip_idx.append(idx)
            all_clips.append(clip_idx)

        return np.array(all_clips)

    def __call__(self, results):
        total_frames = results.get('total_frames', 100)
        start_index = results.get('start_index', 0)

        if not self.test_mode:
            frame_inds, next_inds = self._sample_train_indices(total_frames)
            results['frame_inds'] = frame_inds + start_index
            results['next_frame_inds'] = next_inds + start_index
            results['clip_len'] = self.clip_len
            results['frame_interval'] = self.frame_interval
            results['num_clips'] = 1
        else:
            all_clips = self._sample_test_indices(total_frames)
            results['frame_inds'] = all_clips + start_index
            results['clip_len'] = self.clip_len
            results['frame_interval'] = self.frame_interval
            results['num_clips'] = self.num_clips

        return results
