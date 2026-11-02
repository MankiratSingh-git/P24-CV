import os
import sys
import json
import argparse
import torch
from mmaction.core.random_seed import load_checkpoint


def dump_checkpoint_metrics(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"File not found: {ckpt_path}")
        return

    ckpt = load_checkpoint(ckpt_path, map_location='cpu')
    epoch = ckpt.get('epoch', 'N/A')
    val_top1 = ckpt.get('val_top1', 'N/A')
    val_top5 = ckpt.get('val_top5', 'N/A')

    metrics = {
        "checkpoint": os.path.abspath(ckpt_path),
        "best_epoch": epoch,
        "val_top1": round(float(val_top1), 2) if isinstance(val_top1, (int, float)) else val_top1,
        "val_top5": round(float(val_top5), 2) if isinstance(val_top5, (int, float)) else val_top5
    }

    out_file = os.path.join(os.path.dirname(ckpt_path), "metrics.json")
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics extracted from {ckpt_path}:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export metrics from checkpoint to JSON")
    parser.add_argument("checkpoint", nargs="?", default="checkpoints/supervised_baseline/best_model.pth")
    args = parser.parse_args()
    dump_checkpoint_metrics(args.checkpoint)
