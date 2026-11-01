import os
import sys
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.resnet3d import ResNet3D18
from dataset import UCF101Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="FixMatch Baseline on UCF-101 (10% Labeled Data)")
    parser.add_argument("--data-root", type=str, default="data/ucf101", help="Path to UCF-101 dataset")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Path to split annotation files")
    parser.add_argument("--labeled-percent", type=int, default=10, choices=[5, 10, 20], help="Percentage of labeled videos")
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3], help="Official UCF-101 split")
    
    # Hyperparameters from paper (Section 4.2 & Table 5)
    parser.add_argument("--epochs", type=int, default=220, help="Total training epochs (220 for 10% split)")
    parser.add_argument("--warmup-epochs", type=int, default=40, help="Warmup epochs with only labeled data")
    parser.add_argument("--batch-size", type=int, default=4, help="Labeled batch size B_l")
    parser.add_argument("--mu", type=int, default=4, help="Unlabeled batch size factor: B_u = mu * B_l")
    parser.add_argument("--lr", type=float, default=0.2, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="SGD weight decay")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold tau (0.3 in paper)")
    parser.add_argument("--lambda-u", type=float, default=1.0, help="Unsupervised loss weight")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps to imitate 2-GPU mini-batch (Section 4.2)")

    # System / Dataloader settings
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="checkpoints/fixmatch_baseline")
    parser.add_argument("--eval-freq", type=int, default=5, help="Validation evaluation frequency (epochs)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    return parser.parse_args()


def adjust_learning_rate(optimizer, epoch, total_epochs, warmup_epochs, base_lr):
    """Linear warmup + Cosine annealing schedule as described in Section 4.2."""
    if epoch < warmup_epochs:
        # Linear warm-up
        lr = base_lr * (float(epoch + 1) / float(max(1, warmup_epochs)))
    else:
        # Cosine decaying schedule
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def evaluate(model, val_loader, device):
    """Evaluate Top-1 and Top-5 accuracy on the validation set."""
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for clips, labels in val_loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(clips)
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
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("FixMatch Baseline Training (RGB only)")
    print(f"Backbone: 3D-ResNet-18 | Device: {device}")
    print(f"Labeled Percentage: {args.labeled_percent}% | Split: {args.split}")
    print(f"Batch Size: Labeled={args.batch_size}, Unlabeled={args.batch_size * args.mu}")
    print(f"Total Epochs: {args.epochs} (Warmup: {args.warmup_epochs}) | Tau: {args.threshold} | LR: {args.lr}")
    print("=" * 60)

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

    # Build Datasets
    labeled_dataset = UCF101Dataset(
        data_root=args.data_root, split_file=labeled_split,
        is_unlabeled=False, is_train=True, include_tg=False
    )
    unlabeled_dataset = UCF101Dataset(
        data_root=args.data_root, split_file=unlabeled_split,
        is_unlabeled=True, is_train=True, include_tg=False
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
        unlabeled_dataset, batch_size=args.batch_size * args.mu, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )

    # Initialize 3D-ResNet-18
    model = ResNet3D18(num_classes=101, dropout_ratio=0.5).to(device)

    # Optimizer: SGD with momentum and weight decay
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    ce_loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val_top1 = 0.0

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'...")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_top1 = checkpoint.get('best_val_top1', checkpoint.get('val_top1', 0.0))
            print(f"=> Resumed from Epoch {start_epoch} (Best Val Top-1: {best_val_top1:.2f}%)")
        else:
            print(f"[Warning] Checkpoint '{args.resume}' not found! Starting training from scratch.")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        current_lr = adjust_learning_rate(optimizer, epoch, args.epochs, args.warmup_epochs, args.lr)

        unlabeled_iter = iter(unlabeled_loader)
        running_loss_s = 0.0
        running_loss_u = 0.0
        total_confident_samples = 0
        total_unlabeled_samples = 0
        num_batches = 0
        optimizer.zero_grad()

        for batch_idx, (labeled_clips, labeled_targets) in enumerate(labeled_loader):
            labeled_clips = labeled_clips.to(device, non_blocking=True)
            labeled_targets = labeled_targets.to(device, non_blocking=True)
            bz_l = labeled_clips.size(0)

            loss_u_val = 0.0

            if epoch < args.warmup_epochs:
                # Warmup: purely supervised on labeled clips
                logits_l = model(labeled_clips)
                loss_s = ce_loss_fn(logits_l, labeled_targets)
                loss = loss_s
            else:
                # FixMatch: joint forward pass (labeled + weak + strong) for unified BatchNorm stats
                try:
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled_batch = next(unlabeled_iter)

                unlabeled_weak, unlabeled_strong = unlabeled_batch
                unlabeled_weak = unlabeled_weak.to(device, non_blocking=True)
                unlabeled_strong = unlabeled_strong.to(device, non_blocking=True)
                bz_u = unlabeled_weak.size(0)

                # Concatenate labeled, weak, and strong batches into a single joint tensor
                joint_inputs = torch.cat([labeled_clips, unlabeled_weak, unlabeled_strong], dim=0)
                joint_logits = model(joint_inputs)

                logits_l = joint_logits[:bz_l]
                logits_u_w, logits_u_s = joint_logits[bz_l:].chunk(2)

                loss_s = ce_loss_fn(logits_l, labeled_targets)

                # Pseudo-label assignment from weak views (detached target)
                with torch.no_grad():
                    probs_weak = F.softmax(logits_u_w, dim=-1)
                    max_probs, pseudo_labels = torch.max(probs_weak, dim=-1)
                    mask = (max_probs >= args.threshold)
                    num_confident = mask.sum().item()

                if num_confident > 0:
                    loss_u = F.cross_entropy(logits_u_s[mask], pseudo_labels[mask], reduction='sum') / float(bz_u)
                    loss = loss_s + args.lambda_u * loss_u
                    loss_u_val = loss_u.item()
                    total_confident_samples += num_confident
                else:
                    loss = loss_s

                total_unlabeled_samples += bz_u

            # Gradient accumulation: properly scale current window even if dataset size isn't divisible
            steps_in_window = min(
                args.accum_steps,
                len(labeled_loader) - (batch_idx // args.accum_steps) * args.accum_steps
            )
            loss_for_backward = loss / float(steps_in_window)
            loss_for_backward.backward()

            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(labeled_loader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss_s += loss_s.item()
            running_loss_u += loss_u_val
            num_batches += 1

        avg_loss_s = running_loss_s / max(1, num_batches)
        avg_loss_u = running_loss_u / max(1, num_batches)
        conf_ratio = (total_confident_samples / max(1, total_unlabeled_samples)) * 100.0

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.warmup_epochs:
            print(
                f"Epoch [{epoch+1:03d}/{args.epochs:03d}] | LR: {current_lr:.5f} | "
                f"Loss_S: {avg_loss_s:.4f} | Loss_U: {avg_loss_u:.4f} | "
                f"Pseudo Mask: {conf_ratio:.1f}% ({total_confident_samples}/{total_unlabeled_samples})",
                flush=True
            )

        # Validation
        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs:
            val_top1, val_top5 = evaluate(model, val_loader, device)
            is_best = val_top1 > best_val_top1
            if is_best:
                best_val_top1 = val_top1
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_top1': best_val_top1,
                    'val_top1': val_top1,
                    'val_top5': val_top5,
                }, os.path.join(args.output_dir, "best_model.pth"))

            print(
                f"--> [Val Epoch {epoch+1:03d}] Top-1 Acc: {val_top1:.2f}% (Best: {best_val_top1:.2f}%) | "
                f"Top-5 Acc: {val_top5:.2f}%",
                flush=True
            )

        # Save latest checkpoint after every epoch for seamless resumption
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_top1': best_val_top1,
        }, os.path.join(args.output_dir, "latest_checkpoint.pth"))

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"Training completed in {elapsed / 3600:.2f} hours. Best Val Top-1: {best_val_top1:.2f}%")
    print(f"Best model checkpoint saved to: {os.path.join(args.output_dir, 'best_model.pth')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
