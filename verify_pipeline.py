import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from models.resnet3d import ResNet3D18, FeaturePerturbator, F2FixMatchModel
from augmentations import VideoRandAugment, VideoCutout, video_resize_and_crop, normalize_clip
from dataset import sample_frame_indices, compute_temporal_gradient


def test_models():
    print("[-] Testing ResNet3D-18 Architecture and Shapes...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Running on device: {device}")

    # Input: (B=2, C=3, T=8, H=224, W=224)
    x = torch.randn(2, 3, 8, 224, 224, device=device)
    model = ResNet3D18(num_classes=101).to(device)
    model.train()

    # Forward pass
    feat = model.extract_feat(x)
    assert feat.shape == (2, 512, 8, 7, 7), f"Expected feat shape (2, 512, 8, 7, 7), got {feat.shape}"
    print(f"    [OK] ResNet3D18 Backbone feature shape: {feat.shape}")

    logits = model.forward_3d_feat(feat)
    assert logits.shape == (2, 101), f"Expected logits shape (2, 101), got {logits.shape}"
    print(f"    [OK] ResNet3D18 Logits shape: {logits.shape}")

    # Test P-Aug Feature Perturbator
    print("[-] Testing P-Aug Feature Perturbator...")
    perturbator = FeaturePerturbator().to(device)
    feat_p = perturbator(feat)
    assert feat_p.shape == (2, 512, 7, 7), f"Expected TAP shape (2, 512, 7, 7), got {feat_p.shape}"
    logits_p = model.forward_tap_feat(feat_p)
    assert logits_p.shape == (2, 101), f"Expected TAP logits shape (2, 101), got {logits_p.shape}"
    print(f"    [OK] P-Aug Perturbed TAP feature shape: {feat_p.shape}, logits shape: {logits_p.shape}")

    # Test F2FixMatchModel
    print("[-] Testing F2FixMatchModel Integrated Module...")
    f2_model = F2FixMatchModel(num_classes=101).to(device)
    f2_model.train()
    feat_3d = f2_model.forward_backbone(x)
    feat_tap, tap_logits = f2_model.forward_paug(feat_3d)
    assert tap_logits.shape == (2, 101)
    print(f"    [OK] F2FixMatchModel forward_paug: {tap_logits.shape}")


def test_augmentations():
    print("[-] Testing Video Augmentations (RandAugment, Cutout, Spatial)...")
    # 8 dummy frames of 240x320
    dummy_frames = [Image.fromarray(np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)) for _ in range(8)]

    # Weak spatial
    weak_frames = video_resize_and_crop(dummy_frames, min_size=256, max_size=320, crop_size=224, is_train=True)
    assert len(weak_frames) == 8
    assert weak_frames[0].size == (224, 224)
    print("    [OK] Weak spatial resize & crop to (224, 224)")

    # RandAugment
    rand_aug = VideoRandAugment(n=2, m=10)
    ra_frames = rand_aug(dummy_frames)
    assert len(ra_frames) == 8
    print("    [OK] VideoRandAugment across 8 frames")

    # Cutout
    cutout = VideoCutout(size=128)
    co_frames = cutout(dummy_frames)
    assert len(co_frames) == 8
    print("    [OK] VideoCutout (128x128 patch) across 8 frames")

    # Temporal Gradient
    next_frames = [Image.fromarray(np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)) for _ in range(8)]
    tg_frames = compute_temporal_gradient(dummy_frames, next_frames)
    assert len(tg_frames) == 8
    print("    [OK] Temporal gradient computed on-the-fly")


def test_loss_backward():
    print("[-] Testing FixMatch & F2-FixMatch Loss Computation and Backpropagation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f2_model = F2FixMatchModel(num_classes=101).to(device)
    optimizer = torch.optim.SGD(f2_model.parameters(), lr=0.01, momentum=0.9)

    # Labeled batch: B_l = 2
    rgb_l = torch.randn(2, 3, 8, 224, 224, device=device)
    tg_l = torch.randn(2, 3, 8, 224, 224, device=device)
    targets_l = torch.tensor([5, 42], device=device)

    # Supervised Loss
    logits_rgb_l = f2_model.forward_logits(f2_model.forward_backbone(rgb_l))
    logits_tg_l = f2_model.forward_logits(f2_model.forward_backbone(tg_l))
    loss_s = 0.5 * (F.cross_entropy(logits_rgb_l, targets_l) + F.cross_entropy(logits_tg_l, targets_l))

    # Unlabeled batch: B_u = 4
    rgb_w = torch.randn(4, 3, 8, 224, 224, device=device)
    rgb_s = torch.randn(4, 3, 8, 224, 224, device=device)
    tg_w = torch.randn(4, 3, 8, 224, 224, device=device)
    tg_s = torch.randn(4, 3, 8, 224, 224, device=device)

    with torch.no_grad():
        p_w_rgb = F.softmax(f2_model.forward_logits(f2_model.forward_backbone(rgb_w)), dim=-1)
        p_w_tg = F.softmax(f2_model.forward_logits(f2_model.forward_backbone(tg_w)), dim=-1)
        p_w_agg = 0.5 * (p_w_rgb + p_w_tg)
        max_p, pseudo_y = torch.max(p_w_agg, dim=-1)
        mask = (max_p >= 0.0)  # select all for test

    z_s_rgb = f2_model.forward_backbone(rgb_s)
    z_s_tg = f2_model.forward_backbone(tg_s)
    logits_s_rgb = f2_model.forward_logits(z_s_rgb)
    logits_s_tg = f2_model.forward_logits(z_s_tg)
    loss_u_s = (F.cross_entropy(logits_s_rgb[mask], pseudo_y[mask]) + F.cross_entropy(logits_s_tg[mask], pseudo_y[mask])) / 8.0

    total_loss = loss_s + 2.0 * loss_u_s
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print(f"    [OK] Backpropagation successful! Loss: {total_loss.item():.4f}")


if __name__ == "__main__":
    print("==================================================")
    print("Running FixMatch & F2-FixMatch Verification Suite")
    print("==================================================")
    test_models()
    test_augmentations()
    test_loss_backward()
    print("==================================================")
    print("All architecture, data, and loss tests PASSED!")
    print("==================================================")
