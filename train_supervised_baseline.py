import os
import sys
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.resnet3d import ResNet3D18
from dataset import UCF101Dataset
from mmaction.core.random_seed import set_random_seed, get_rng_states, set_rng_states, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised Baseline Training on UCF-101 (10% Labeled Data Only)")
    parser.add_argument("--data-root", type=str, default="data/ucf101", help="Path to UCF-101 dataset")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Path to split annotation files")
    parser.add_argument("--labeled-percent", type=int, default=10, choices=[5, 10, 20, 50, 100], help="Percentage of labeled videos")
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3], help="Official UCF-101 split")

    parser.add_argument("--epochs", type=int, default=120, help="Total training epochs for supervised only")
    parser.add_argument("--batch-size", "--batch-size-labeled", type=int, default=8, dest="batch_size", help="Batch size B_l")
    parser.add_argument("--lr", type=float, default=0.2, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="SGD weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Linear warmup epochs")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps to imitate 2-GPU mini-batch")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="checkpoints/supervised_baseline")
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
    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("Supervised Baseline Training (10% Labeled Data Only)")
    print(f"Backbone: 3D-ResNet-18 | Device: {device}")
    print(f"Dataset: UCF-101 Split {args.split} ({args.labeled_percent}% labeled)")
    print(f"Batch Size: {args.batch_size} | Epochs: {args.epochs} | Base LR: {args.lr}")
    print("=" * 60)

    if args.labeled_percent == 100:
        labeled_split = os.path.join(
            args.splits_dir,
            f"ucf101_train_split_{args.split}_rawframes.txt"
        )
    else:
        labeled_split = os.path.join(
            args.splits_dir,
            f"ucf101_train_{args.labeled_percent}_percent_labeled_split_{args.split}_rawframes.txt"
        )
    val_split = os.path.join(
        args.splits_dir,
        f"ucf101_val_split_{args.split}_rawframes.txt"
    )

    labeled_dataset = UCF101Dataset(
        data_root=args.data_root, split_file=labeled_split,
        is_unlabeled=False, is_train=True, include_tg=False
    )
    val_dataset = UCF101Dataset(
        data_root=args.data_root, split_file=val_split,
        is_unlabeled=False, is_train=False, include_tg=False
    )

    print(f"Samples: Labeled Train={len(labeled_dataset)}, Val={len(val_dataset)}")

    labeled_loader = DataLoader(
        labeled_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )

    model = ResNet3D18(num_classes=101, dropout_ratio=0.5).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    ce_loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val_top1 = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'...")
            checkpoint = load_checkpoint(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'rng_state' in checkpoint:
                set_rng_states(checkpoint['rng_state'])
                print("=> Restored full RNG state (Python, NumPy, PyTorch CPU/CUDA)")
            start_epoch = checkpoint.get('epoch', 0)
            best_val_top1 = checkpoint.get('best_val_top1', checkpoint.get('val_top1', 0.0))
            print(f"=> Resumed from Epoch {start_epoch} (Best Val Top-1: {best_val_top1:.2f}%)")
        else:
            print(f"[Warning] Checkpoint '{args.resume}' not found! Starting training from scratch.")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        current_lr = adjust_learning_rate(optimizer, epoch, args.epochs, args.warmup_epochs, args.lr)

        running_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for batch_idx, (clips, targets) in enumerate(labeled_loader):
            clips = clips.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(clips)
            loss = ce_loss_fn(logits, targets)

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

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{args.epochs:03d}] | LR: {current_lr:.5f} | Train Loss: {avg_loss:.4f}", flush=True)

        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs:
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

        # Save latest checkpoint after every epoch
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
    print("=" * 60)
    print(f"Supervised baseline training completed in {elapsed / 3600:.2f} hours. Best Val Top-1: {best_val_top1:.2f}%")
    print(f"Checkpoint saved to: {os.path.join(args.output_dir, 'best_model.pth')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
