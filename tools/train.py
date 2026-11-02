"""
MMAction2 Training Tool.
Usage:
    python tools/train.py configs/f2_fixmatch/f2_fixmatch_r3d_r18_ucf101_10percent.py
    python tools/train.py configs/fixmatch/fixmatch_r3d_r18_ucf101_10percent.py --resume checkpoints/fixmatch_baseline/best_model.pth
"""

import os
import sys
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmaction.utils.config import Config
from mmaction.models.builder import build_recognizer
from mmaction.datasets.builder import build_dataset, build_dataloader
from mmaction.core.evaluation import top_k_accuracy
from mmaction.core.random_seed import set_random_seed, get_rng_states, set_rng_states


def parse_args():
    parser = argparse.ArgumentParser(description="MMAction2 Train Script for FixMatch and F2-FixMatch")
    parser.add_argument("config", help="Path to MMAction2 config file")
    parser.add_argument("--work-dir", help="Directory to save logs and checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--accum-steps", type=int, default=None, help="Gradient accumulation steps override")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic CUDNN")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def adjust_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        lr = base_lr * (float(epoch + 1) / float(max(1, warmup_epochs)))
    else:
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def evaluate(model, val_loader, device):
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['imgs'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            scores = model(imgs, return_loss=False)
            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())

    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    top1, top5 = top_k_accuracy(all_scores, all_labels, topk=(1, 5))
    return top1, top5


def main():
    args = parse_args()
    set_random_seed(args.seed, args.deterministic)
    cfg = Config.fromfile(args.config)

    work_dir = args.work_dir if args.work_dir else cfg.get('work_dir', 'checkpoints/exp')
    os.makedirs(work_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 65)
    print(f"MMAction2 Action Recognition Toolbox (v0.24.0)")
    print(f"Config: {args.config}")
    print(f"Model: {cfg.model.type} | Backbone: {cfg.model.backbone.type}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 65)

    # 1. Build Model
    model = build_recognizer(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model = model.to(device)

    # 2. Build Datasets & Dataloaders
    train_cfg_data = cfg.data
    val_dataset = build_dataset(train_cfg_data.val)
    val_loader = build_dataloader(
        val_dataset,
        videos_per_gpu=train_cfg_data.videos_per_gpu * 2,
        workers_per_gpu=train_cfg_data.workers_per_gpu,
        shuffle=False
    )

    is_semi = ('train_unlabeled' in train_cfg_data)
    if is_semi:
        labeled_dataset = build_dataset(train_cfg_data.train_labeled)
        unlabeled_dataset = build_dataset(train_cfg_data.train_unlabeled)
        unlabeled_ratio = train_cfg_data.get('unlabeled_ratio', 4)

        labeled_loader = build_dataloader(
            labeled_dataset,
            videos_per_gpu=train_cfg_data.videos_per_gpu,
            workers_per_gpu=train_cfg_data.workers_per_gpu,
            shuffle=True, drop_last=True
        )
        unlabeled_loader = build_dataloader(
            unlabeled_dataset,
            videos_per_gpu=train_cfg_data.videos_per_gpu * unlabeled_ratio,
            workers_per_gpu=train_cfg_data.workers_per_gpu,
            shuffle=True, drop_last=True
        )
        print(f"Dataset: Labeled={len(labeled_dataset)}, Unlabeled={len(unlabeled_dataset)}, Val={len(val_dataset)}")
    else:
        train_dataset = build_dataset(train_cfg_data.train)
        train_loader = build_dataloader(
            train_dataset,
            videos_per_gpu=train_cfg_data.videos_per_gpu,
            workers_per_gpu=train_cfg_data.workers_per_gpu,
            shuffle=True, drop_last=True
        )
        print(f"Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # 3. Optimizer & Schedulers
    opt_cfg = cfg.get('optimizer', dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=1e-4))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt_cfg.get('lr', 0.2),
        momentum=opt_cfg.get('momentum', 0.9),
        weight_decay=opt_cfg.get('weight_decay', 1e-4)
    )

    start_epoch = 0
    best_val_top1 = 0.0

    # Resume from checkpoint
    resume_path = args.resume if args.resume else cfg.get('resume_from', None)
    if resume_path and os.path.isfile(resume_path):
        print(f"=> Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'rng_state' in ckpt:
            set_rng_states(ckpt['rng_state'])
            print("=> Restored full RNG states (Python, NumPy, PyTorch CPU/CUDA)")
        start_epoch = ckpt.get('epoch', 0)
        best_val_top1 = ckpt.get('best_val_top1', ckpt.get('val_top1', 0.0))
        print(f"=> Resumed at Epoch {start_epoch} with Best Top-1: {best_val_top1:.2f}%")

    total_epochs = cfg.get('total_epochs', 220)
    warmup_epochs = cfg.get('warmup_epochs', 40)
    accum_steps = args.accum_steps if args.accum_steps is not None else cfg.get('accum_steps', 2)
    eval_freq = cfg.get('eval_freq', 5)

    print(f"Starting Training: Epochs={total_epochs} (Warmup={warmup_epochs}) | Accum Steps={accum_steps}")
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs):
        model.train()
        current_lr = adjust_lr(optimizer, epoch, total_epochs, warmup_epochs, opt_cfg.get('lr', 0.2))

        running_loss_s = 0.0
        running_loss_u = 0.0
        total_conf = 0
        total_u = 0
        num_batches = 0
        optimizer.zero_grad()

        if is_semi:
            unlabeled_iter = iter(unlabeled_loader)
            loader_len = len(labeled_loader)

            for batch_idx, batch_l in enumerate(labeled_loader):
                try:
                    batch_u = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    batch_u = next(unlabeled_iter)

                # Move to device
                for k in batch_l:
                    if isinstance(batch_l[k], torch.Tensor):
                        batch_l[k] = batch_l[k].to(device, non_blocking=True)
                for k in batch_u:
                    if isinstance(batch_u[k], torch.Tensor):
                        batch_u[k] = batch_u[k].to(device, non_blocking=True)

                if cfg.model.type == 'F2FixMatchRecognizer3D':
                    if 'imgs_tg' not in batch_l:
                        raise KeyError(
                            "F2FixMatchRecognizer3D requires 'imgs_tg' in the labeled batch for bi-modal supervised loss, "
                            "but 'imgs_tg' was not found in batch_l. Ensure the labeled data pipeline includes "
                            "RawFrameDecode(include_tg=True) and Collect(keys=['imgs', 'imgs_tg', 'label'])."
                        )
                    loss_dict = model.forward_train(
                        imgs_rgb_l=batch_l['imgs'],
                        imgs_tg_l=batch_l['imgs_tg'],
                        labels=batch_l['label'],
                        rgb_w=batch_u['rgb_w'],
                        rgb_s=batch_u['rgb_s'],
                        tg_w=batch_u['tg_w'],
                        tg_s=batch_u['tg_s'],
                        cur_epoch=epoch
                    )
                    total_loss = loss_dict['loss_s'] + loss_dict['loss_u_s'] + loss_dict['loss_u_wp'] + loss_dict['loss_u_sp']
                    u_val = (loss_dict['loss_u_s'] + loss_dict['loss_u_wp'] + loss_dict['loss_u_sp']).item()
                else:
                    loss_dict = model.forward_train(
                        imgs=batch_l['imgs'],
                        labels=batch_l['label'],
                        imgs_weak=batch_u['imgs_weak'],
                        imgs_strong=batch_u['imgs_strong'],
                        cur_epoch=epoch
                    )
                    total_loss = loss_dict['loss_cls'] + loss_dict['loss_u']
                    u_val = loss_dict['loss_u'].item()

                steps_in_window = min(accum_steps, loader_len - (batch_idx // accum_steps) * accum_steps)
                loss_for_backward = total_loss / float(steps_in_window)
                loss_for_backward.backward()

                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == loader_len:
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss_s += (loss_dict.get('loss_s', loss_dict.get('loss_cls'))).item()
                running_loss_u += u_val
                total_conf += loss_dict.get('conf_samples', 0)
                total_u += loss_dict.get('total_u_samples', 0)
                num_batches += 1
        else:
            loader_len = len(train_loader)
            for batch_idx, batch in enumerate(train_loader):
                imgs = batch['imgs'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                loss_dict = model.forward_train(imgs, labels)
                loss = loss_dict['loss_cls']

                steps_in_window = min(accum_steps, loader_len - (batch_idx // accum_steps) * accum_steps)
                loss_for_backward = loss / float(steps_in_window)
                loss_for_backward.backward()

                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == loader_len:
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss_s += loss.item()
                num_batches += 1

        avg_s = running_loss_s / max(1, num_batches)
        avg_u = running_loss_u / max(1, num_batches)
        mask_ratio = (total_conf / max(1, total_u)) * 100.0 if total_u > 0 else 0.0

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == warmup_epochs:
            print(
                f"Epoch [{epoch+1:03d}/{total_epochs:03d}] | LR: {current_lr:.5f} | "
                f"Loss_S: {avg_s:.4f} | Loss_U: {avg_u:.4f} | "
                f"Pseudo Mask: {mask_ratio:.1f}% ({total_conf}/{total_u})",
                flush=True
            )

        # Validation
        if (epoch + 1) % eval_freq == 0 or (epoch + 1) == total_epochs:
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
                    'rng_state': get_rng_states(),
                }, os.path.join(work_dir, "best_model.pth"))

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
        }, os.path.join(work_dir, "latest_checkpoint.pth"))

    elapsed = time.time() - start_time
    print("=" * 65)
    print(f"Training completed in {elapsed / 3600:.2f} hours. Best Val Top-1: {best_val_top1:.2f}%")
    print(f"Best checkpoint: {os.path.join(work_dir, 'best_model.pth')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
