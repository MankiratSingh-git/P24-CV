"""
MMAction2 Test and Evaluation Tool (30-view Multi-Crop Benchmark).
Usage:
    python tools/test.py configs/f2_fixmatch/f2_fixmatch_r3d_r18_ucf101_10percent.py checkpoints/f2_fixmatch/best_model.pth --eval top_k_accuracy
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader

from mmaction.utils.config import Config
from mmaction.models.builder import build_recognizer
from mmaction.datasets.builder import build_dataset, build_dataloader
from mmaction.core.evaluation import top_k_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="MMAction2 Evaluation Script (30-view protocol)")
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--eval", default="top_k_accuracy", help="Evaluation metric")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    device = torch.device(args.device)

    print("=" * 65)
    print("MMAction2 30-View Benchmark Evaluator (10 clips x 3 crops)")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print("=" * 65)

    # 1. Build Model
    model = build_recognizer(cfg.model, test_cfg=cfg.get('test_cfg'))
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # 2. Build Test Dataset & DataLoader (with 10 clips x 3 crops pipeline)
    test_dataset = build_dataset(cfg.data.test)
    test_loader = build_dataloader(
        test_dataset,
        videos_per_gpu=4,
        workers_per_gpu=2,
        shuffle=False
    )
    print(f"Evaluation Dataset: {len(test_dataset)} videos (30 views per video)")

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            imgs = batch['imgs'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            # Model forward test averages across the 30 views
            scores = model(imgs, return_loss=False)
            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())

            if (i + 1) % 50 == 0 or (i + 1) == len(test_loader):
                print(f"Processed [{i+1:04d}/{len(test_loader):04d}] batches...", flush=True)

    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    top1, top5 = top_k_accuracy(all_scores, all_labels, topk=(1, 5))

    print("=" * 65)
    print(f"--> [30-View Test Results] Top-1 Accuracy: {top1:.2f}% | Top-5 Accuracy: {top5:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
