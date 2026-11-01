import os
import glob
import random
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from augmentations import (
    VideoRandAugment,
    VideoCutout,
    video_resize_and_crop,
    normalize_clip
)


def load_video_frames(video_path, frame_indices):
    """
    Load frames at specific indices from a video file (.avi) or a folder of frames.
    Returns: list of PIL Images in RGB format.
    """
    frames = []
    
    # Check if video_path is a directory of extracted frames
    if os.path.isdir(video_path):
        all_imgs = sorted(glob.glob(os.path.join(video_path, 'img_*.jpg')) or 
                          glob.glob(os.path.join(video_path, '*.jpg')) or 
                          glob.glob(os.path.join(video_path, '*.png')))
        total = len(all_imgs)
        if total == 0:
            raise ValueError(f"No frames found in directory: {video_path}")
        for idx in frame_indices:
            clamped_idx = min(max(0, idx), total - 1)
            img = Image.open(all_imgs[clamped_idx]).convert('RGB')
            frames.append(img)
        return frames

    # Otherwise read directly from video file via OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1

    for idx in frame_indices:
        target_idx = min(max(0, idx), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            # Fallback: rewind or duplicate previous
            if len(frames) > 0:
                frame = np.array(frames[-1])
            else:
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()
    return frames


def sample_frame_indices(total_frames, clip_len=8, frame_interval=8, is_train=True):
    """
    Extract 8 frames with sampling stride 8 covering 64 consecutive frames:
    indices: [start + 0*8, start + 1*8, ..., start + 7*8]
    Also returns subsequent frames [idx + 1] for temporal gradient computation.
    """
    required_span = clip_len * frame_interval
    if total_frames > required_span:
        if is_train:
            start = random.randint(0, total_frames - required_span)
        else:
            start = max(0, (total_frames - required_span) // 2)
    else:
        start = 0

    rgb_indices = []
    next_indices = []
    for i in range(clip_len):
        idx = start + i * frame_interval
        if total_frames > 0:
            idx = idx % total_frames
        rgb_indices.append(idx)
        next_idx = (idx + 1) % total_frames if total_frames > 0 else idx
        next_indices.append(next_idx)

    return rgb_indices, next_indices


def compute_temporal_gradient(curr_frames, next_frames):
    """
    Formula from Section 3.1:
    x_t^G = (x_t^V - x_{t+1}^V + 255) / 2
    """
    tg_frames = []
    for curr_img, next_img in zip(curr_frames, next_frames):
        curr_arr = np.array(curr_img, dtype=np.float32)
        next_arr = np.array(next_img, dtype=np.float32)
        tg = (curr_arr - next_arr + 255.0) / 2.0
        tg = np.clip(tg, 0, 255).astype(np.uint8)
        tg_frames.append(Image.fromarray(tg))
    return tg_frames


class UCF101Dataset(Dataset):
    """
    PyTorch Dataset for UCF-101 Semi-Supervised Action Recognition.
    Supports both labeled and unlabeled sets, dynamic temporal gradient generation,
    and weak/strong video augmentations.
    """
    def __init__(self, data_root, split_file, is_unlabeled=False, is_train=True,
                 clip_len=8, frame_interval=8, include_tg=True):
        self.data_root = data_root
        self.split_file = split_file
        self.is_unlabeled = is_unlabeled
        self.is_train = is_train
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.include_tg = include_tg

        self.rand_aug = VideoRandAugment(n=2, m=10)
        self.cutout = VideoCutout(size=128)

        self.samples = []
        self._load_split()

    def _load_split(self):
        with open(self.split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # Line format: <rel_path> <total_frames> [<class_id>]
                rel_path = parts[0]
                total_frames = int(parts[1]) if len(parts) > 1 else 100
                label = int(parts[2]) if len(parts) > 2 else -1

                # Locate video file or extracted frame folder
                # Possibilities:
                # 1. data_root / videos / rel_path + '.avi'
                # 2. data_root / rawframes / rel_path
                # 3. data_root / rel_path + '.avi'
                full_path = None
                candidates = [
                    os.path.join(self.data_root, 'videos', 'UCF-101', rel_path + '.avi'),
                    os.path.join(self.data_root, 'videos', rel_path + '.avi'),
                    os.path.join(self.data_root, 'rawframes', rel_path),
                    os.path.join(self.data_root, rel_path + '.avi'),
                    os.path.join(self.data_root, rel_path),
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        full_path = cand
                        break
                
                # If dataset not yet downloaded, use relative candidate as fallback
                if full_path is None:
                    full_path = candidates[0]

                self.samples.append({
                    'path': full_path,
                    'total_frames': total_frames,
                    'label': label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        video_path = sample['path']
        total_frames = sample['total_frames']
        label = sample['label']

        # Sample 8 frame indices and consecutive indices for TG
        rgb_idx, next_idx = sample_frame_indices(
            total_frames, self.clip_len, self.frame_interval, is_train=self.is_train
        )

        try:
            raw_rgb = load_video_frames(video_path, rgb_idx)
            if self.include_tg and self.is_train:
                raw_next = load_video_frames(video_path, next_idx)
                raw_tg = compute_temporal_gradient(raw_rgb, raw_next)
            else:
                raw_tg = None
        except Exception:
            # Fallback dummy clip if video read error occurs
            raw_rgb = [Image.fromarray(np.zeros((240, 320, 3), dtype=np.uint8)) for _ in range(self.clip_len)]
            raw_tg = [Image.fromarray(np.zeros((240, 320, 3), dtype=np.uint8)) for _ in range(self.clip_len)] if self.include_tg else None

        if not self.is_train:
            # Validation / Test: center crop, resize, normalize RGB
            val_rgb = video_resize_and_crop(raw_rgb, min_size=256, max_size=256, crop_size=224, is_train=False)
            arr = np.stack([np.array(f) for f in val_rgb], axis=0)
            tensor_rgb = normalize_clip(arr)
            return tensor_rgb, label

        # Training Weak Augmentations
        weak_rgb = video_resize_and_crop(raw_rgb, min_size=256, max_size=320, crop_size=224, is_train=True)
        weak_rgb_arr = np.stack([np.array(f) for f in weak_rgb], axis=0)
        tensor_rgb_w = normalize_clip(weak_rgb_arr)

        tensor_tg_w = None
        if self.include_tg and raw_tg is not None:
            weak_tg = video_resize_and_crop(raw_tg, min_size=256, max_size=320, crop_size=224, is_train=True)
            weak_tg_arr = np.stack([np.array(f) for f in weak_tg], axis=0)
            tensor_tg_w = normalize_clip(weak_tg_arr)

        if not self.is_unlabeled:
            # Labeled sample: return weakly augmented inputs + ground-truth label
            if self.include_tg and tensor_tg_w is not None:
                return tensor_rgb_w, tensor_tg_w, label
            return tensor_rgb_w, label

        # Unlabeled sample: also produce Strongly Augmented version
        strong_rgb = self.rand_aug(raw_rgb)
        strong_rgb = self.cutout(strong_rgb)
        strong_rgb = video_resize_and_crop(strong_rgb, min_size=256, max_size=320, crop_size=224, is_train=True)
        strong_rgb_arr = np.stack([np.array(f) for f in strong_rgb], axis=0)
        tensor_rgb_s = normalize_clip(strong_rgb_arr)

        tensor_tg_s = None
        if self.include_tg and raw_tg is not None:
            strong_tg = self.rand_aug(raw_tg)
            strong_tg = self.cutout(strong_tg)
            strong_tg = video_resize_and_crop(strong_tg, min_size=256, max_size=320, crop_size=224, is_train=True)
            strong_tg_arr = np.stack([np.array(f) for f in strong_tg], axis=0)
            tensor_tg_s = normalize_clip(strong_tg_arr)

        if self.include_tg and tensor_tg_w is not None and tensor_tg_s is not None:
            return tensor_rgb_w, tensor_rgb_s, tensor_tg_w, tensor_tg_s
        return tensor_rgb_w, tensor_rgb_s
