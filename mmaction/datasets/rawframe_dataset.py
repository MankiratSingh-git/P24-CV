"""
Rawframe Dataset for MMAction2.
Supports UCF-101 frame directories or video files with annotations.
"""

import os
import copy
from torch.utils.data import Dataset
from .builder import DATASETS
from .pipelines.compose import Compose


@DATASETS.register_module()
class RawframeDataset(Dataset):
    """
    Standard Rawframe Dataset in MMAction2 for supervised and validation splits.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05d}.jpg',
                 start_index=0,
                 include_tg=False,
                 **kwargs):
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.filename_tmpl = filename_tmpl
        self.start_index = start_index
        self.include_tg = include_tg

        if isinstance(pipeline, list):
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = pipeline

        self.video_infos = self.load_annotations()

    def load_annotations(self):
        video_infos = []
        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")

        with open(self.ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                rel_path = parts[0]
                total_frames = int(parts[1]) if len(parts) > 1 else 100
                label = int(parts[2]) if len(parts) > 2 else -1

                # Locate directory or video
                full_path = None
                candidates = []
                if self.data_prefix:
                    candidates.extend([
                        os.path.join(self.data_prefix, 'rawframes', rel_path),
                        os.path.join(self.data_prefix, 'videos', 'UCF-101', rel_path + '.avi'),
                        os.path.join(self.data_prefix, 'videos', rel_path + '.avi'),
                        os.path.join(self.data_prefix, rel_path),
                        os.path.join(self.data_prefix, rel_path + '.avi'),
                    ])
                candidates.extend([
                    os.path.join('data/ucf101/rawframes', rel_path),
                    os.path.join('data/ucf101/videos/UCF-101', rel_path + '.avi'),
                    os.path.join('data/ucf101/videos', rel_path + '.avi'),
                    os.path.join('data/ucf101', rel_path),
                ])

                for cand in candidates:
                    if os.path.exists(cand):
                        full_path = cand
                        break
                if full_path is None:
                    full_path = candidates[0]

                video_infos.append({
                    'frame_dir': full_path,
                    'total_frames': total_frames,
                    'label': label
                })

        return video_infos

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['start_index'] = self.start_index
        results['filename_tmpl'] = self.filename_tmpl
        return self.pipeline(results)
