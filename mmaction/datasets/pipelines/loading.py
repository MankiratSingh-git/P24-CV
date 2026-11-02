"""
Frame and Video Decoding transforms in MMAction2.
"""

import os
import glob
import cv2
import numpy as np
from PIL import Image
from ..builder import PIPELINES


def _load_frame(video_path, frame_idx, filename_tmpl='img_{:05d}.jpg'):
    """Load a single frame from folder or video file."""
    if os.path.isdir(video_path):
        filename = os.path.join(video_path, filename_tmpl.format(frame_idx))
        if not os.path.exists(filename):
            # Fallback to 1-based or 0-based alternative
            alt1 = os.path.join(video_path, filename_tmpl.format(frame_idx + 1))
            if os.path.exists(alt1):
                filename = alt1
            else:
                all_imgs = sorted(glob.glob(os.path.join(video_path, '*.jpg')) or
                                  glob.glob(os.path.join(video_path, '*.png')))
                if len(all_imgs) > 0:
                    clamped = min(max(0, frame_idx), len(all_imgs) - 1)
                    filename = all_imgs[clamped]
                else:
                    raise IOError(f"No image frames found in directory: {video_path}")

        img = cv2.imread(filename)
        if img is None:
            raise IOError(f"Failed to read image: {filename}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)
    else:
        # Read from video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise IOError(f"Failed to read frame {frame_idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)


@PIPELINES.register_module()
class RawFrameDecode:
    """
    Decodes sampled raw frames into PIL Images.
    Computes temporal gradient modality (G-data) when required.
    """

    def __init__(self, io_backend='disk', filename_tmpl='img_{:05d}.jpg', include_tg=False):
        self.io_backend = io_backend
        self.filename_tmpl = filename_tmpl
        self.include_tg = include_tg

    def __call__(self, results):
        video_path = results['frame_dir']
        frame_inds = results['frame_inds']

        if frame_inds.ndim == 1:
            # Single clip of shape (clip_len,)
            imgs = [_load_frame(video_path, idx, self.filename_tmpl) for idx in frame_inds]
            results['imgs'] = imgs

            if self.include_tg and 'next_frame_inds' in results:
                next_inds = results['next_frame_inds']
                next_imgs = [_load_frame(video_path, idx, self.filename_tmpl) for idx in next_inds]
                # Formula: x_t^G = (x_t^V - x_{t+1}^V + 255) / 2
                tg_imgs = []
                for curr_img, next_img in zip(imgs, next_imgs):
                    c_arr = np.array(curr_img, dtype=np.float32)
                    n_arr = np.array(next_img, dtype=np.float32)
                    tg = np.clip((c_arr - n_arr + 255.0) / 2.0, 0, 255).astype(np.uint8)
                    tg_imgs.append(Image.fromarray(tg))
                results['imgs_tg'] = tg_imgs
        else:
            # Multi-clip (e.g. 10 clips for 30-view evaluation): shape (num_clips, clip_len)
            num_clips = frame_inds.shape[0]
            multi_clip_imgs = []
            for c_i in range(num_clips):
                clip_imgs = [_load_frame(video_path, idx, self.filename_tmpl) for idx in frame_inds[c_i]]
                multi_clip_imgs.append(clip_imgs)
            results['imgs'] = multi_clip_imgs

        return results
