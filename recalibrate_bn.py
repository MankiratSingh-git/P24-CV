#!/usr/bin/env python3
"""
recalibrate_bn.py

PreciseBN Recalibration Diagnostic for 3D Video Models:
1. Implements the PreciseBN recalibration protocol (Wu et al. / PySlowFast):
   - Loads checkpoint weights into the model.
   - Freezes all trainable parameters (param.requires_grad = False).
   - Resets running statistics (running_mean and running_var) across all BatchNorm3d layers.
   - Sets momentum = None to compute exact cumulative sample averages across all mini-batches.
   - Runs forward passes over training clips under model.train() with torch.no_grad().
   - Restores original momentum parameters and sets model.eval().
2. Performs strict parameter invariance verification to mathematically guarantee that
   zero weights or biases are updated during recalibration.
3. Computes comprehensive layer-by-layer BN shift diagnostics.
4. Saves the recalibrated checkpoint and a detailed JSON audit report.
5. Includes a dedicated verification mode on synthetic tensors/batches for environments
   lacking full UCF-101 AVI video assets.
"""

import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import sys
import json
import argparse
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.resnet3d import ResNet3D18, F2FixMatchModel
from mmaction.core.random_seed import load_checkpoint
from dataset import UCF101Dataset


def find_bn_modules(model: nn.Module) -> List[Tuple[str, nn.modules.batchnorm._BatchNorm]]:
    """
    Finds all BatchNorm modules within the model architecture.
    """
    bn_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            bn_modules.append((name, module))
    return bn_modules


def load_model_and_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    model_type: str = "auto"
) -> Tuple[nn.Module, Dict[str, Any], bool]:
    """
    Loads checkpoint weights safely (handling PyTorch 2.6+ weights_only) and instantiates the model.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    is_f2 = any(k.startswith("backbone.") for k in state_dict.keys()) or (model_type == "f2")
    if model_type == "resnet":
        is_f2 = False

    if is_f2:
        model = F2FixMatchModel(num_classes=101)
    else:
        model = ResNet3D18(num_classes=101)

    model.load_state_dict(state_dict)
    model = model.to(device)
    return model, checkpoint, is_f2


def snapshot_model_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Snapshots all model parameter tensors for bitwise invariance verification.
    """
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def verify_parameter_invariance(model: nn.Module, snapshot: Dict[str, torch.Tensor]) -> Tuple[bool, int, str]:
    """
    Verifies that zero weights or biases changed during BN recalibration.
    """
    param_count = 0
    for name, param in model.named_parameters():
        param_count += 1
        if name not in snapshot:
            return False, param_count, f"Missing parameter {name} in snapshot"
        if not torch.equal(param, snapshot[name]):
            max_diff = (param - snapshot[name]).abs().max().item()
            return False, param_count, f"Parameter {name} modified! Max diff: {max_diff}"
    return True, param_count, "All parameters bitwise identical"


def recalibrate_bn_statistics(
    model: nn.Module,
    device: torch.device,
    is_synthetic: bool = False,
    data_root: str = "data/ucf101",
    split_file: str = "data/splits/ucf101_train_split_1_rawframes.txt",
    num_batches: int = 200,
    batch_size: int = 16,
    log_interval: int = 25
) -> Dict[str, Any]:
    """
    Executes the PreciseBN recalibration protocol over training clips or synthetic batches:
    1. Freezes all parameters.
    2. Resets running stats and sets momentum = None for exact cumulative average.
    3. Runs forward passes in model.train() under torch.no_grad().
    4. Restores original momentum and sets model.eval().
    5. Returns detailed shift diagnostics.
    """
    # 1. Identify all BN layers
    bn_modules = find_bn_modules(model)
    print(f"Found {len(bn_modules)} BatchNorm modules in architecture.")

    # 2. Snapshot parameters to guarantee immutability
    param_snapshot = snapshot_model_parameters(model)
    for p in model.parameters():
        p.requires_grad = False

    # 3. Record initial BN running stats
    initial_stats = {}
    original_momentums = {}
    for name, bn in bn_modules:
        initial_stats[name] = {
            "running_mean": bn.running_mean.detach().clone() if bn.running_mean is not None else None,
            "running_var": bn.running_var.detach().clone() if bn.running_var is not None else None,
            "num_batches_tracked": bn.num_batches_tracked.detach().clone() if bn.num_batches_tracked is not None else None
        }
        original_momentums[name] = bn.momentum

        # Reset stats and switch to cumulative average
        bn.reset_running_stats()
        bn.momentum = None
        bn.track_running_stats = True

    # 4. Put model in train mode for BN updates
    model.train()

    # Determine forward function: extract_feat passes through all backbone conv+BN stages
    if hasattr(model, "extract_feat"):
        feature_extractor = model.extract_feat
    elif hasattr(model, "backbone") and hasattr(model.backbone, "extract_feat"):
        feature_extractor = model.backbone.extract_feat
    elif hasattr(model, "forward_backbone"):
        feature_extractor = model.forward_backbone
    else:
        feature_extractor = model

    print(f"\n[PreciseBN] Starting recalibration ({'Synthetic Batches' if is_synthetic else 'Real Video Clips'})...")
    print(f"Target Batches: {num_batches} | Batch Size: {batch_size} | Device: {device}")

    start_time = time.time()
    batches_processed = 0

    with torch.no_grad():
        if is_synthetic:
            # Synthetic tensor mode: standardized normal video tensors (B, 3, 8, 224, 224)
            for b in range(1, num_batches + 1):
                batch_tensor = torch.randn(batch_size, 3, 8, 224, 224, device=device)
                _ = feature_extractor(batch_tensor)
                batches_processed = b
                if b % log_interval == 0 or b == num_batches:
                    elapsed = time.time() - start_time
                    rate = batches_processed / max(1e-4, elapsed)
                    print(f"  Recalibrating BN: [{b:04d}/{num_batches:04d}] batches ({rate:.1f} batch/s)", flush=True)
        else:
            # Real dataset mode: support both labeled and unlabeled splits
            is_unlabeled = "unlabeled" in os.path.basename(split_file).lower()
            dataset = UCF101Dataset(
                data_root=data_root,
                split_file=split_file,
                is_train=True,
                is_unlabeled=is_unlabeled,
                include_tg=False
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
            if len(loader) == 0:
                raise ValueError(
                    f"Dataset in '{split_file}' has {len(dataset)} samples, which is fewer than batch_size={batch_size} with drop_last=True."
                )

            # Cycle loader across epochs if num_batches > len(loader) (PreciseBN protocol)
            def _cycle_loader(dl):
                while True:
                    for batch in dl:
                        yield batch

            loader_iter = _cycle_loader(loader)
            for b in range(1, num_batches + 1):
                batch_data = next(loader_iter)
                # batch_data is (tensor_rgb, label) or (tensor_rgb_w, tensor_tg_w, label) or (tensor_rgb_w, tensor_rgb_s)
                tensor_rgb = batch_data[0].to(device)
                _ = feature_extractor(tensor_rgb)
                batches_processed = b
                if b % log_interval == 0 or b == num_batches:
                    elapsed = time.time() - start_time
                    rate = batches_processed / max(1e-4, elapsed)
                    print(f"  Recalibrating BN: [{b:04d}/{num_batches:04d}] batches ({rate:.1f} batch/s)", flush=True)

    # 5. Restore original momentum values and set model.eval()
    for name, bn in bn_modules:
        bn.momentum = original_momentums[name]
    model.eval()

    # 6. Verify parameter invariance
    is_invariant, p_count, inv_msg = verify_parameter_invariance(model, param_snapshot)
    if not is_invariant:
        raise RuntimeError(f"CRITICAL: Parameter invariance check failed: {inv_msg}")
    print(f"\n[Verification] Parameter Invariance Check PASSED: {p_count} parameters verified bitwise identical.")

    # 7. Compute layer-by-layer statistics shift
    if len(bn_modules) == 0:
        return {
            "num_bn_layers": 0,
            "total_batches_processed": batches_processed,
            "mode": "synthetic" if is_synthetic else "real_clips",
            "device": str(device),
            "elapsed_seconds": round(time.time() - start_time, 2),
            "parameter_invariance_verified": is_invariant,
            "parameters_checked": p_count,
            "aggregate_mean_l1_shift": 0.0,
            "aggregate_var_l1_shift": 0.0,
            "max_mean_shift_layer": "None",
            "max_mean_shift_val": 0.0,
            "max_var_shift_layer": "None",
            "max_var_shift_val": 0.0,
            "layers": []
        }

    layer_diagnostics = []
    total_mean_l1 = 0.0
    total_var_l1 = 0.0

    for name, bn in bn_modules:
        orig_mean = initial_stats[name]["running_mean"]
        orig_var = initial_stats[name]["running_var"]
        new_mean = bn.running_mean
        new_var = bn.running_var

        if orig_mean is not None and new_mean is not None:
            mean_l1 = (new_mean - orig_mean).abs().mean().item()
            mean_max = (new_mean - orig_mean).abs().max().item()
        else:
            mean_l1 = 0.0
            mean_max = 0.0

        if orig_var is not None and new_var is not None:
            var_l1 = (new_var - orig_var).abs().mean().item()
            var_max = (new_var - orig_var).abs().max().item()
        else:
            var_l1 = 0.0
            var_max = 0.0

        total_mean_l1 += mean_l1
        total_var_l1 += var_l1

        layer_diagnostics.append({
            "layer_name": name,
            "num_features": bn.num_features,
            "batches_tracked": bn.num_batches_tracked.item() if bn.num_batches_tracked is not None else batches_processed,
            "mean_l1_shift": round(mean_l1, 6),
            "mean_max_shift": round(mean_max, 6),
            "var_l1_shift": round(var_l1, 6),
            "var_max_shift": round(var_max, 6),
            "orig_mean_norm": round(orig_mean.norm().item(), 4) if orig_mean is not None else 0.0,
            "new_mean_norm": round(new_mean.norm().item(), 4) if new_mean is not None else 0.0,
            "orig_var_norm": round(orig_var.norm().item(), 4) if orig_var is not None else 0.0,
            "new_var_norm": round(new_var.norm().item(), 4) if new_var is not None else 0.0
        })

    avg_mean_l1 = total_mean_l1 / len(bn_modules)
    avg_var_l1 = total_var_l1 / len(bn_modules)

    max_mean_layer = max(layer_diagnostics, key=lambda x: x["mean_l1_shift"])
    max_var_layer = max(layer_diagnostics, key=lambda x: x["var_l1_shift"])

    summary = {
        "num_bn_layers": len(bn_modules),
        "total_batches_processed": batches_processed,
        "mode": "synthetic" if is_synthetic else "real_clips",
        "device": str(device),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "parameter_invariance_verified": is_invariant,
        "parameters_checked": p_count,
        "aggregate_mean_l1_shift": round(avg_mean_l1, 6),
        "aggregate_var_l1_shift": round(avg_var_l1, 6),
        "max_mean_shift_layer": max_mean_layer["layer_name"],
        "max_mean_shift_val": max_mean_layer["mean_l1_shift"],
        "max_var_shift_layer": max_var_layer["layer_name"],
        "max_var_shift_val": max_var_layer["var_l1_shift"],
        "layers": layer_diagnostics
    }

    return summary


def print_summary_table(summary: Dict[str, Any]):
    """Prints a structured summary of BN recalibration results."""
    print("\n" + "=" * 80)
    print("PRECISE-BN RECALIBRATION AUDIT RESULTS")
    print("=" * 80)
    print(f"Total BN Layers:         {summary['num_bn_layers']}")
    print(f"Batches Processed:       {summary['total_batches_processed']} ({summary['mode']})")
    print(f"Parameter Invariance:    {'PASSED (Zero weights/biases changed)' if summary['parameter_invariance_verified'] else 'FAILED'}")
    print(f"Aggregate Mean L1 Shift: {summary['aggregate_mean_l1_shift']:.6f}")
    print(f"Aggregate Var L1 Shift:  {summary['aggregate_var_l1_shift']:.6f}")
    print(f"Max Mean Shift Layer:    {summary['max_mean_shift_layer']} (Δ={summary['max_mean_shift_val']:.6f})")
    print(f"Max Var Shift Layer:     {summary['max_var_shift_layer']} (Δ={summary['max_var_shift_val']:.6f})")
    print("-" * 80)
    print(f"{'Layer Name':<32} | {'Channels':<8} | {'Mean L1 Shift':<14} | {'Var L1 Shift':<14}")
    print("-" * 80)
    for layer in summary["layers"]:
        print(f"{layer['layer_name']:<32} | {layer['num_features']:<8} | {layer['mean_l1_shift']:<14.6f} | {layer['var_l1_shift']:<14.6f}")
    print("=" * 80)


def save_recalibrated_checkpoint(
    checkpoint: Dict[str, Any],
    model: nn.Module,
    summary: Dict[str, Any],
    output_path: str
):
    """
    Saves the recalibrated model checkpoint, preserving all auxiliary checkpoint keys
    while updating model_state_dict with the recalibrated BN running statistics.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Clone checkpoint dict to preserve optimizer_state_dict, epoch, etc.
    saved_dict = {}
    for k, v in checkpoint.items():
        if k != "model_state_dict":
            saved_dict[k] = v

    # Insert updated model state dict
    saved_dict["model_state_dict"] = model.state_dict()

    # Append PreciseBN metadata
    saved_dict["precise_bn"] = {
        "recalibrated": True,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "batches": summary["total_batches_processed"],
        "mode": summary["mode"],
        "mean_l1_shift": summary["aggregate_mean_l1_shift"],
        "var_l1_shift": summary["aggregate_var_l1_shift"]
    }

    torch.save(saved_dict, output_path)
    print(f"\n[Saved] Recalibrated checkpoint saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PreciseBN Recalibration Protocol for 3D Video ResNet Checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to input checkpoint (.pth)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save recalibrated checkpoint (.pth)")
    parser.add_argument("--output-report", type=str, default=None,
                        help="Path to save JSON diagnostic report")
    parser.add_argument("--data-root", type=str, default="data/ucf101",
                        help="Path to UCF-101 video directory")
    parser.add_argument("--split-file", type=str, default="data/splits/ucf101_train_split_1_rawframes.txt",
                        help="Training split file to sample clips from")
    parser.add_argument("--num-batches", type=int, default=100,
                        help="Number of batches for running statistic accumulation")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Mini-batch size for forward passes")
    parser.add_argument("--log-interval", type=int, default=20,
                        help="Logging interval in batches")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Execution device ('cuda' or 'cpu')")
    parser.add_argument("--model-type", type=str, default="auto", choices=["auto", "f2", "resnet"])
    parser.add_argument("--synthetic", "--verify", action="store_true",
                        help="Run verification mode on synthetic tensors/batches")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 70)
    print("PHASE 2: PRECISE-BN RECALIBRATION DIAGNOSTIC")
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Device:       {device}")
    print("=" * 70)

    # 1. Load model and checkpoint
    model, checkpoint, is_f2 = load_model_and_checkpoint(
        args.checkpoint, device=device, model_type=args.model_type
    )
    print(f"Loaded architecture: {'F2FixMatchModel' if is_f2 else 'ResNet3D18'}")

    # Determine default output paths
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_checkpoint = args.output or os.path.join(ckpt_dir, f"{stem}_precise_bn.pth")
    out_report = args.output_report or os.path.join(ckpt_dir, f"{stem}_precise_bn_report.json")

    # 2. Check dataset availability
    has_videos = os.path.isdir(args.data_root) and len(os.listdir(args.data_root)) > 0
    is_synthetic = args.synthetic or (not has_videos)

    if is_synthetic and not args.synthetic:
        print(f"[Notice] Video directory '{args.data_root}' not found. Defaulting to synthetic tensor mode.")

    # 3. Execute PreciseBN recalibration
    summary = recalibrate_bn_statistics(
        model=model,
        device=device,
        is_synthetic=is_synthetic,
        data_root=args.data_root,
        split_file=args.split_file,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        log_interval=args.log_interval
    )

    # 4. Display results table
    print_summary_table(summary)

    # 5. Save recalibrated checkpoint
    save_recalibrated_checkpoint(
        checkpoint=checkpoint,
        model=model,
        summary=summary,
        output_path=out_checkpoint
    )

    # 6. Save JSON report
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] Detailed diagnostic JSON: {out_report}")


if __name__ == "__main__":
    main()
