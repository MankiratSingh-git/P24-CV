import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import sys
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.resnet3d import F2FixMatchModel
from dataset import UCF101Dataset
from mmaction.core.random_seed import set_random_seed, get_rng_states, set_rng_states, load_checkpoint, safe_load_optimizer_state


def parse_args():
    parser = argparse.ArgumentParser(description="F2-FixMatch Training on UCF-101 (10% Labeled Data)")
    parser.add_argument("--data-root", type=str, default="data/ucf101", help="Path to UCF-101 dataset")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Path to split annotation files")
    parser.add_argument("--labeled-percent", type=int, default=10, choices=[5, 10, 20], help="Percentage of labeled videos")
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3], help="Official UCF-101 split")

    # Hyperparameters from Section 4.2 & 4.5
    parser.add_argument("--epochs", type=int, default=220, help="Total training epochs (220 for 10% subset)")
    parser.add_argument("--warmup-epochs", type=int, default=40, help="Warmup epochs with only labeled data")
    parser.add_argument("--batch-size", "--batch-size-labeled", type=int, default=4, dest="batch_size", help="Labeled batch size B_l per GPU")
    parser.add_argument("--batch-size-unlabeled", type=int, default=None, help="Explicit unlabeled batch size B_u (overrides --mu if set)")
    parser.add_argument("--mu", type=int, default=4, help="Unlabeled batch size factor: B_u = mu * B_l")
    parser.add_argument("--lr", type=float, default=0.2, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="SGD weight decay")
    parser.add_argument("--threshold", type=float, default=0.3, help="Pseudo-labeling threshold tau (0.3 in paper)")

    # Weight coefficients for unsupervised losses (Section 4.5: alpha=2, beta=2, gamma=1)
    parser.add_argument("--alpha", type=float, default=2.0, help="Weight for frame-level strong-aug loss L_u^s")
    parser.add_argument("--beta", type=float, default=2.0, help="Weight for frame-&feature weak-aug loss L_u^{w_p}")
    parser.add_argument("--gamma", type=float, default=1.0, help="Weight for frame-&feature strong-aug loss L_u^{s_p}")
    parser.add_argument("--rampup-epochs", type=int, default=30, help="Epochs to linearly ramp up unsupervised loss after warmup (default: 30)")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps to imitate 2-GPU mini-batch (Section 4.2)")

    # System settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="checkpoints/f2_fixmatch")
    parser.add_argument("--eval-freq", type=int, default=5, help="Validation evaluation frequency (epochs)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    return parser.parse_args()


def adjust_learning_rate(optimizer, epoch, total_epochs, warmup_epochs, base_lr):
    """Linear warmup + Cosine annealing schedule."""
    if epoch < warmup_epochs:
        lr = base_lr * (float(epoch + 1) / float(max(1, warmup_epochs)))
    else:
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def evaluate(model, val_loader, device):
    """Evaluate Top-1 and Top-5 accuracy on the validation set using RGB modality only."""
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for clips, labels in val_loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model.forward_eval(clips)
            total += labels.size(0)

            # Top 1
            preds = logits.argmax(dim=-1)
            correct_top1 += (preds == labels).sum().item()

            # Top 5
            _, top5_preds = logits.topk(min(5, logits.size(-1)), dim=-1)
            correct_top5 += top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)).sum().item()

    top1_acc = 100.0 * correct_top1 / max(1, total)
    top5_acc = 100.0 * correct_top5 / max(1, total)
    return top1_acc, top5_acc


def main():
    args = parse_args()
    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 65)
    print("F2-FixMatch: Frame- and Feature-level Progressive Augmentation")
    print(f"Backbone: 3D-ResNet-18 | Device: {device}")
    print(f"Dataset: UCF-101 ({args.labeled_percent}% labeled) | Split: {args.split}")
    unlabeled_bs = args.batch_size_unlabeled if args.batch_size_unlabeled is not None else args.batch_size * args.mu
    print(f"Batch Size: B_l={args.batch_size}, B_u={unlabeled_bs}")
    print(f"Loss Weights: alpha={args.alpha}, beta={args.beta}, gamma={args.gamma} | Tau: {args.threshold} | Rampup: {args.rampup_epochs} epochs")
    print("=" * 65)

    # Dataset filepaths
    labeled_split = os.path.join(
        args.splits_dir,
        f"ucf101_train_{args.labeled_percent}_percent_labeled_split_{args.split}_rawframes.txt"
    )
    unlabeled_split = os.path.join(
        args.splits_dir,
        f"ucf101_train_{args.labeled_percent}_percent_unlabeled_split_{args.split}_rawframes.txt"
    )
    val_split = os.path.join(
        args.splits_dir,
        f"ucf101_val_split_{args.split}_rawframes.txt"
    )

    # Build Datasets with bi-modal RGB and TG enabled
    labeled_dataset = UCF101Dataset(
        data_root=args.data_root, split_file=labeled_split,
        is_unlabeled=False, is_train=True, include_tg=True
    )
    unlabeled_dataset = UCF101Dataset(
        data_root=args.data_root, split_file=unlabeled_split,
        is_unlabeled=True, is_train=True, include_tg=True
    )
    val_dataset = UCF101Dataset(
        data_root=args.data_root, split_file=val_split,
        is_unlabeled=False, is_train=False, include_tg=False
    )

    print(f"Samples: Labeled={len(labeled_dataset)}, Unlabeled={len(unlabeled_dataset)}, Val={len(val_dataset)}")

    # Data Loaders
    labeled_loader = DataLoader(
        labeled_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == 'cuda')
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=unlabeled_bs, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )

    # Initialize F2-FixMatch model
    model = F2FixMatchModel(num_classes=101, dropout_ratio=0.5).to(device)

    # Optimizer: SGD with momentum and weight decay
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    ce_loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val_top1 = 0.0
    history = []

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'...")
            checkpoint = load_checkpoint(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                safe_load_optimizer_state(optimizer, checkpoint['optimizer_state_dict'], model)
            if 'rng_state' in checkpoint:
                set_rng_states(checkpoint['rng_state'])
                print("=> Restored full RNG state (Python, NumPy, PyTorch CPU/CUDA)")
            start_epoch = checkpoint.get('epoch', 0)
            best_val_top1 = checkpoint.get('best_val_top1', checkpoint.get('val_top1', 0.0))
            print(f"=> Resumed from Epoch {start_epoch} (Best Val Top-1: {best_val_top1:.2f}%)")

            # Load and prune previous history up to start_epoch
            hist_file = os.path.join(args.output_dir, "training_history.json")
            if os.path.isfile(hist_file):
                try:
                    import json
                    with open(hist_file, "r") as f:
                        old_history = json.load(f)
                        history = [h for h in old_history if h.get("epoch", 0) <= start_epoch]
                    print(f"=> Preserved {len(history)} prior history entries up to Epoch {start_epoch}")
                except Exception as e:
                    print(f"=> Note: Could not load prior history: {e}")
        else:
            print(f"[Warning] Checkpoint '{args.resume}' not found! Starting training from scratch.")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        current_lr = adjust_learning_rate(optimizer, epoch, args.epochs, args.warmup_epochs, args.lr)

        unlabeled_iter = iter(unlabeled_loader)
        running_loss_s = 0.0
        running_loss_u_s = 0.0
        running_loss_u_wp = 0.0
        running_loss_u_sp = 0.0
        total_confident_samples = 0
        total_unlabeled_samples = 0
        num_batches = 0
        optimizer.zero_grad()

        for batch_idx, (rgb_l, tg_l, targets_l) in enumerate(labeled_loader):
            rgb_l = rgb_l.to(device, non_blocking=True)
            tg_l = tg_l.to(device, non_blocking=True)
            targets_l = targets_l.to(device, non_blocking=True)
            bz_l = rgb_l.size(0)

            # -------------------------------------------------------------
            # 1. Supervised Loss L_s (RGB and TG modalities)
            # -------------------------------------------------------------
            feat_rgb_l = model.forward_backbone(rgb_l)
            logits_rgb_l = model.forward_logits(feat_rgb_l)

            feat_tg_l = model.forward_backbone(tg_l)
            logits_tg_l = model.forward_logits(feat_tg_l)

            loss_s_rgb = ce_loss_fn(logits_rgb_l, targets_l)
            loss_s_tg = ce_loss_fn(logits_tg_l, targets_l)
            loss_s = loss_s_rgb + loss_s_tg  # Equation 13: L_s = L_s^V + L_s^G (exact sum, not average)
            total_loss = loss_s

            loss_u_s_val = 0.0
            loss_u_wp_val = 0.0
            loss_u_sp_val = 0.0

            # -------------------------------------------------------------
            # 2. Unsupervised P-Aug + MPL Loss (After Warmup)
            # -------------------------------------------------------------
            rampup_weight = 0.0
            if epoch >= args.warmup_epochs:
                try:
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled_batch = next(unlabeled_iter)

                rgb_w, rgb_s, tg_w, tg_s = unlabeled_batch
                rgb_w = rgb_w.to(device, non_blocking=True)
                rgb_s = rgb_s.to(device, non_blocking=True)
                tg_w = tg_w.to(device, non_blocking=True)
                tg_s = tg_s.to(device, non_blocking=True)
                bz_u = rgb_w.size(0)

                # Linear ramp-up weight after warmup
                if args.rampup_epochs > 0:
                    rampup_weight = min(1.0, float(epoch - args.warmup_epochs + 1) / float(args.rampup_epochs))
                else:
                    rampup_weight = 1.0

                # Extract features for weakly augmented RGB and TG under torch.no_grad()
                # Weak views act strictly as the teacher (pseudo-label generator).
                # Detaching Z_w prevents mode collapse and circular gradient feedback.
                with torch.no_grad():
                    z_w_rgb = model.forward_backbone(rgb_w)
                    logits_w_rgb = model.forward_logits(z_w_rgb)

                    z_w_tg = model.forward_backbone(tg_w)
                    logits_w_tg = model.forward_logits(z_w_tg)

                    # Multihead Pseudo-Labeling (MPL) aggregation: targets are detached
                    p_w_rgb = F.softmax(logits_w_rgb, dim=-1)
                    p_w_tg = F.softmax(logits_w_tg, dim=-1)
                    p_w_agg = 0.5 * (p_w_rgb + p_w_tg)
                    max_probs, pseudo_labels = torch.max(p_w_agg, dim=-1)
                    confident_mask = (max_probs >= args.threshold)
                    num_select = confident_mask.sum().item()

                if num_select > 0:
                    conf_idx = confident_mask.nonzero(as_tuple=False).squeeze(1)
                    conf_labels = pseudo_labels[conf_idx]

                    # --- Frame-level Strong Augmentation features Z_s ---
                    # Strongly augmented views act as the student; gradients flow to backbone
                    z_s_rgb = model.forward_backbone(rgb_s)
                    logits_s_rgb = model.forward_logits(z_s_rgb)

                    z_s_tg = model.forward_backbone(tg_s)
                    logits_s_tg = model.forward_logits(z_s_tg)

                    loss_u_s_rgb = F.cross_entropy(logits_s_rgb[conf_idx], conf_labels, reduction='sum')
                    loss_u_s_tg = F.cross_entropy(logits_s_tg[conf_idx], conf_labels, reduction='sum')
                    loss_u_s = (loss_u_s_rgb + loss_u_s_tg) / (2.0 * float(bz_u))

                    # --- P-Aug: Feature-level perturbation on Z_w and Z_s ---
                    # Z_{w_p}: frame weak, feature perturbed.
                    # z_w is detached to prevent self-training collapse into trivial representations;
                    # gradients flow into the TAP classification head.
                    _, logits_wp_rgb = model.forward_paug(z_w_rgb.detach())
                    _, logits_wp_tg = model.forward_paug(z_w_tg.detach())

                    loss_u_wp_rgb = F.cross_entropy(logits_wp_rgb[conf_idx], conf_labels, reduction='sum')
                    loss_u_wp_tg = F.cross_entropy(logits_wp_tg[conf_idx], conf_labels, reduction='sum')
                    loss_u_wp = (loss_u_wp_rgb + loss_u_wp_tg) / (2.0 * float(bz_u))

                    # Z_{s_p}: frame strong, feature perturbed (gradients flow through z_s to backbone)
                    _, logits_sp_rgb = model.forward_paug(z_s_rgb)
                    _, logits_sp_tg = model.forward_paug(z_s_tg)

                    loss_u_sp_rgb = F.cross_entropy(logits_sp_rgb[conf_idx], conf_labels, reduction='sum')
                    loss_u_sp_tg = F.cross_entropy(logits_sp_tg[conf_idx], conf_labels, reduction='sum')
                    loss_u_sp = (loss_u_sp_rgb + loss_u_sp_tg) / (2.0 * float(bz_u))

                    # Total Loss: L = L_s + rampup * (alpha * L_u^s + beta * L_u^{w_p} + gamma * L_u^{s_p})
                    total_loss = (
                        total_loss + 
                        rampup_weight * (
                            args.alpha * loss_u_s + 
                            args.beta * loss_u_wp + 
                            args.gamma * loss_u_sp
                        )
                    )

                    loss_u_s_val = loss_u_s.item()
                    loss_u_wp_val = loss_u_wp.item()
                    loss_u_sp_val = loss_u_sp.item()
                    total_confident_samples += num_select

                total_unlabeled_samples += bz_u

            # Gradient accumulation: properly scale even if dataset size is not divisible
            steps_in_window = min(
                args.accum_steps,
                len(labeled_loader) - (batch_idx // args.accum_steps) * args.accum_steps
            )
            loss_for_backward = total_loss / float(steps_in_window)
            loss_for_backward.backward()

            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(labeled_loader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss_s += loss_s.item()
            running_loss_u_s += loss_u_s_val
            running_loss_u_wp += loss_u_wp_val
            running_loss_u_sp += loss_u_sp_val
            num_batches += 1

        avg_loss_s = running_loss_s / max(1, num_batches)
        avg_loss_us = running_loss_u_s / max(1, num_batches)
        avg_loss_uwp = running_loss_u_wp / max(1, num_batches)
        avg_loss_usp = running_loss_u_sp / max(1, num_batches)
        conf_ratio = (total_confident_samples / max(1, total_unlabeled_samples)) * 100.0

        ramp_str = f" | Ramp: {rampup_weight:.2f}" if (args.rampup_epochs > 0 and epoch >= args.warmup_epochs and rampup_weight < 1.0) else ""
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.warmup_epochs:
            print(
                f"Epoch [{epoch+1:03d}/{args.epochs:03d}] | LR: {current_lr:.5f} | "
                f"L_S: {avg_loss_s:.4f} | L_u_s: {avg_loss_us:.4f} | "
                f"L_u_wp: {avg_loss_uwp:.4f} | L_u_sp: {avg_loss_usp:.4f}{ramp_str} | "
                f"Pseudo Mask: {conf_ratio:.1f}% ({total_confident_samples}/{total_unlabeled_samples})",
                flush=True
            )

        # Validation
        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs:
            print(f"--> [Epoch {epoch+1:03d}] Evaluating validation set ({len(val_loader.dataset)} videos)...", flush=True)
            val_top1, val_top5 = evaluate(model, val_loader, device)
            is_best = val_top1 > best_val_top1
            if is_best:
                best_val_top1 = val_top1
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_top1': val_top1,
                    'val_top5': val_top5,
                    'rng_state': get_rng_states(),
                }, os.path.join(args.output_dir, "best_model.pth"))

            print(
                f"--> [Val Epoch {epoch+1:03d}] Top-1 Acc: {val_top1:.2f}% (Best: {best_val_top1:.2f}%) | "
                f"Top-5 Acc: {val_top5:.2f}%",
                flush=True
            )

            history.append({
                "epoch": epoch + 1,
                "lr": round(current_lr, 6),
                "loss_s": round(avg_loss_s, 4),
                "loss_u_s": round(avg_loss_us, 4),
                "loss_u_wp": round(avg_loss_uwp, 4),
                "loss_u_sp": round(avg_loss_usp, 4),
                "rampup_weight": round(rampup_weight, 2) if epoch >= args.warmup_epochs else 0.0,
                "pseudo_mask_percent": round(conf_ratio, 2),
                "val_top1": round(val_top1, 2),
                "val_top5": round(val_top5, 2)
            })

            # Save training metrics history JSON
            import json
            with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
                json.dump(history, f, indent=2)

        # Save latest checkpoint after every epoch for seamless resumption
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_top1': best_val_top1,
            'rng_state': get_rng_states(),
        }, os.path.join(args.output_dir, "latest_checkpoint.pth"))

        if (epoch + 1) == args.epochs:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_top1': best_val_top1,
                'rng_state': get_rng_states(),
            }, os.path.join(args.output_dir, "final_model.pth"))

    elapsed = time.time() - start_time
    import json
    summary = {
        "model": "F2-FixMatch (Bi-modal RGB + TG, P-Aug, MPL)",
        "dataset": "UCF-101 Split 1 (10% Labeled)",
        "best_epoch": max(history, key=lambda x: x["val_top1"])["epoch"] if history else args.epochs,
        "best_val_top1": round(best_val_top1, 2),
        "total_training_hours": round(elapsed / 3600.0, 2),
        "checkpoint": os.path.join(args.output_dir, "best_model.pth")
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 65)
    print(f"Training completed in {elapsed / 3600:.2f} hours. Best Val Top-1: {best_val_top1:.2f}%")
    print(f"Best checkpoint saved to: {os.path.join(args.output_dir, 'best_model.pth')}")
    print(f"Metrics saved to: {os.path.join(args.output_dir, 'metrics.json')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
