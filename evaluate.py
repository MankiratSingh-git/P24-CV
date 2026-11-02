import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from models.resnet3d import ResNet3D18, F2FixMatchModel
from augmentations import IMAGENET_MEAN, IMAGENET_STD
from dataset import load_video_frames


def three_crop(pil_frames, crop_size=224, scale=256):
    """
    Takes 3 spatial crops (left/top, center, right/bottom) of size crop_size x crop_size
    after scaling shorter side to scale (default 256 pixels), as in MMAction2 and Section 4.2.
    """
    orig_w, orig_h = pil_frames[0].size
    if orig_h < orig_w:
        new_h = scale
        new_w = int(round(orig_w * (scale / float(orig_h))))
    else:
        new_w = scale
        new_h = int(round(orig_h * (scale / float(orig_w))))

    new_w = max(new_w, crop_size)
    new_h = max(new_h, crop_size)

    crops_indices = []
    if new_w >= new_h:
        # 3 horizontal crops along longer side: left, center, right
        y_offset = (new_h - crop_size) // 2
        x_offsets = [0, (new_w - crop_size) // 2, new_w - crop_size]
        crops_indices = [(x, y_offset) for x in x_offsets]
    else:
        # 3 vertical crops along longer side: top, center, bottom
        x_offset = (new_w - crop_size) // 2
        y_offsets = [0, (new_h - crop_size) // 2, new_h - crop_size]
        crops_indices = [(x_offset, y) for y in y_offsets]

    crop_tensors = []
    for (x0, y0) in crops_indices:
        frames_arr = []
        for frame in pil_frames:
            resized = frame.resize((new_w, new_h), Image.BILINEAR)
            cropped = resized.crop((x0, y0, x0 + crop_size, y0 + crop_size))
            frames_arr.append(np.array(cropped))
        arr = np.stack(frames_arr, axis=0).astype(np.float32)
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)  # (3, T, H, W)
        crop_tensors.append(tensor)

    return crop_tensors


def center_crop(pil_frames, crop_size=224, scale=256):
    """
    Takes 1 spatial center crop of size crop_size x crop_size
    after scaling shorter side to scale (default 256 pixels).
    """
    orig_w, orig_h = pil_frames[0].size
    if orig_h < orig_w:
        new_h = scale
        new_w = int(round(orig_w * (scale / float(orig_h))))
    else:
        new_w = scale
        new_h = int(round(orig_h * (scale / float(orig_w))))

    new_w = max(new_w, crop_size)
    new_h = max(new_h, crop_size)

    x0 = (new_w - crop_size) // 2
    y0 = (new_h - crop_size) // 2

    frames_arr = []
    for frame in pil_frames:
        resized = frame.resize((new_w, new_h), Image.BILINEAR)
        cropped = resized.crop((x0, y0, x0 + crop_size, y0 + crop_size))
        frames_arr.append(np.array(cropped))
    arr = np.stack(frames_arr, axis=0).astype(np.float32)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)  # (3, T, H, W)
    return [tensor]


class UCF101FullTestDataset(Dataset):
    """
    Dataset for test evaluation:
    - If eval_mode == 'single': 1 temporal center clip x 1 spatial center crop (1 view per video).
    - If eval_mode == 'full': 10 temporal clips x 3 spatial crops (30 views per video, official paper protocol).
    """
    def __init__(self, data_root, split_file, clip_len=8, frame_interval=8, num_clips=10, crop_size=224, eval_mode='full'):
        self.data_root = data_root
        self.split_file = split_file
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = 1 if eval_mode == 'single' else num_clips
        self.crop_size = crop_size
        self.eval_mode = eval_mode
        self.samples = []
        self._load_split()

    def _load_split(self):
        with open(self.split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                rel_path = parts[0]
                total_frames = int(parts[1]) if len(parts) > 1 else 100
                label = int(parts[2]) if len(parts) > 2 else -1

                candidates = [
                    os.path.join(self.data_root, 'videos', 'UCF-101', rel_path + '.avi'),
                    os.path.join(self.data_root, 'videos', rel_path + '.avi'),
                    os.path.join(self.data_root, 'rawframes', rel_path),
                    os.path.join(self.data_root, rel_path + '.avi'),
                    os.path.join(self.data_root, rel_path),
                ]
                full_path = candidates[0]
                for cand in candidates:
                    if os.path.exists(cand):
                        full_path = cand
                        break

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

        span = self.clip_len * self.frame_interval
        if self.eval_mode == 'single':
            # Single view: 1 temporal center clip x 1 spatial center crop (exact training validation standard)
            start = max(0, (total_frames - span) // 2) if total_frames > span else 0
            frame_indices = [int((start + i * self.frame_interval) % max(1, total_frames)) for i in range(self.clip_len)]
            frames = load_video_frames(video_path, frame_indices)
            crops = center_crop(frames, crop_size=self.crop_size, scale=256)
            tensors = torch.stack(crops, dim=0)  # (1, 3, 8, crop_size, crop_size)
            return tensors, label

        # Full view: 10 temporal clips x 3 spatial crops = 30 views (official paper protocol)
        if total_frames > span:
            intervals = np.linspace(0, total_frames - span, self.num_clips, dtype=int)
        else:
            intervals = np.zeros(self.num_clips, dtype=int)

        all_crops = []
        for start in intervals:
            frame_indices = [int((start + i * self.frame_interval) % max(1, total_frames)) for i in range(self.clip_len)]
            frames = load_video_frames(video_path, frame_indices)
            crops = three_crop(frames, crop_size=self.crop_size, scale=256)
            all_crops.extend(crops)

        # all_crops contains 30 tensors, each (3, 8, crop_size, crop_size)
        tensors = torch.stack(all_crops, dim=0)  # (30, 3, 8, crop_size, crop_size)
        return tensors, label


def main():
    parser = argparse.ArgumentParser(description="Evaluation on UCF-101 Validation Set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data-root", type=str, default="data/ucf101", help="Path to UCF-101 dataset")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Path to splits directory")
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--model-type", type=str, default="auto", choices=["auto", "resnet", "f2"], help="Model architecture")
    parser.add_argument("--eval-mode", type=str, default="full", choices=["single", "full"],
                        help="'single' = 1-clip center-crop, 'full' = 10-clip x 3-crop (30 views, official paper protocol)")
    parser.add_argument("--crop-size", type=int, default=256, help="Spatial crop resolution (256x256 as specified in Section 4.2 of paper)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 60)
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Evaluation Mode: {args.eval_mode} | Crop Size: {args.crop_size}x{args.crop_size} | Device: {device}")
    print("=" * 60)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Detect model type
    is_f2 = any(k.startswith('backbone.') for k in state_dict.keys())
    if is_f2:
        model = F2FixMatchModel(num_classes=101)
        model.load_state_dict(state_dict)
        eval_fn = model.forward_eval
    else:
        model = ResNet3D18(num_classes=101)
        model.load_state_dict(state_dict)
        eval_fn = model

    model = model.to(device).eval()

    val_split = os.path.join(args.splits_dir, f"ucf101_val_split_{args.split}_rawframes.txt")
    val_dataset = UCF101FullTestDataset(
        data_root=args.data_root, 
        split_file=val_split, 
        num_clips=(10 if args.eval_mode == 'full' else 1),
        crop_size=args.crop_size,
        eval_mode=args.eval_mode
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    class_correct = {}
    class_total = {}
    start_time = time.time()

    # Load class names if available
    class_names = {}
    class_file = os.path.join(args.splits_dir, "class_indices.txt")
    if os.path.exists(class_file):
        with open(class_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    class_names[int(parts[0])] = parts[1]

    with torch.no_grad():
        for i, (video_crops, label) in enumerate(val_loader):
            # video_crops: (1, num_views, 3, T, H, W)
            video_crops = video_crops.squeeze(0).to(device, non_blocking=True)
            label = label.to(device)
            lbl_item = label.item()

            logits = eval_fn(video_crops)  # (num_views, 101)
            probs = F.softmax(logits, dim=-1).mean(dim=0, keepdim=True)  # average across all views (1, 101)

            pred_top1 = probs.argmax(dim=-1).item()
            is_correct1 = (pred_top1 == lbl_item)
            if is_correct1:
                correct_top1 += 1

            _, pred_top5 = probs.topk(5, dim=-1)
            if lbl_item in pred_top5[0].tolist():
                correct_top5 += 1

            total += 1
            class_total[lbl_item] = class_total.get(lbl_item, 0) + 1
            if is_correct1:
                class_correct[lbl_item] = class_correct.get(lbl_item, 0) + 1

            if (i + 1) % 100 == 0 or (i + 1) == len(val_loader):
                cur_acc1 = 100.0 * correct_top1 / total
                cur_acc5 = 100.0 * correct_top5 / total
                print(f"\rProgress: [{i+1:04d}/{len(val_loader):04d}] | Top-1: {cur_acc1:.2f}% | Top-5: {cur_acc5:.2f}%", end="", flush=True)

    print("\n" + "=" * 60)
    final_top1 = 100.0 * correct_top1 / total
    final_top5 = 100.0 * correct_top5 / total
    print(f"Final Results on UCF-101 Split {args.split} (Total Videos: {total}):")
    print(f"Top-1 Accuracy: {final_top1:.2f}%")
    print(f"Top-5 Accuracy: {final_top5:.2f}%")
    print(f"Elapsed Time: {time.time() - start_time:.1f}s")
    print("=" * 60)

    # Calculate per-class accuracies
    per_class_stats = []
    for cid in sorted(class_total.keys()):
        tot = class_total[cid]
        corr = class_correct.get(cid, 0)
        acc = round(100.0 * corr / tot, 2)
        cname = class_names.get(cid, f"Class_{cid}")
        per_class_stats.append({"class_id": cid, "class_name": cname, "accuracy": acc, "total": tot})

    sorted_by_acc = sorted(per_class_stats, key=lambda x: x["accuracy"])
    worst_5 = sorted_by_acc[:5]
    best_5 = sorted_by_acc[-5:][::-1]

    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    # Save metrics JSON alongside the checkpoint
    import json
    ckpt_dir = os.path.dirname(args.checkpoint)
    mode_suffix = "full30view" if args.eval_mode == "full" else "singlecrop"
    result_file = os.path.join(ckpt_dir, f"eval_metrics_{mode_suffix}.json")
    results_data = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": args.split,
        "eval_mode": args.eval_mode,
        "num_eval_videos": total,
        "model_params_M": round(total_params, 2),
        "top1_accuracy": round(final_top1, 2),
        "top5_accuracy": round(final_top5, 2),
        "best_5_classes": best_5,
        "worst_5_classes": worst_5,
        "per_class_accuracy": per_class_stats,
        "elapsed_seconds": round(time.time() - start_time, 1)
    }
    with open(result_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved evaluation metrics to: {result_file}")


if __name__ == "__main__":
    main()
