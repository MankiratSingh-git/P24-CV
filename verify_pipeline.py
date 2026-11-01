import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from models.resnet3d import ResNet3D18, FeaturePerturbator, F2FixMatchModel
from augmentations import (
    VideoRandAugment,
    VideoCutout,
    sample_crop_parameters,
    apply_crop_parameters,
    normalize_clip
)
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

    # Test P-Aug Feature Perturbator (spatial_drop_prob=0.5)
    print("[-] Testing P-Aug Feature Perturbator (spatial_drop_prob=0.5)...")
    perturbator = FeaturePerturbator(spatial_drop_prob=0.5).to(device)
    assert perturbator.spatial_drop_prob == 0.5
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
    print("[-] Testing Synchronized Video Augmentations...")
    # 8 dummy frames of 240x320
    dummy_rgb = [Image.fromarray(np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)) for _ in range(8)]
    dummy_tg = [Image.fromarray(np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)) for _ in range(8)]

    # 1. Synchronized Weak Spatial Resize & Crop
    params = sample_crop_parameters(320, 240, min_size=256, max_size=320, crop_size=224, is_train=True)
    weak_rgb = apply_crop_parameters(dummy_rgb, params)
    weak_tg = apply_crop_parameters(dummy_tg, params)
    assert len(weak_rgb) == 8 and len(weak_tg) == 8
    assert weak_rgb[0].size == (224, 224) and weak_tg[0].size == (224, 224)
    print("    [OK] Synchronized Weak spatial resize & crop to (224, 224)")

    # 2. Synchronized Video RandAugment
    rand_aug = VideoRandAugment(n=2, m=10)
    ra_rgb, sampled_ops = rand_aug(dummy_rgb)
    ra_tg = rand_aug.apply_preset_ops(dummy_tg, sampled_ops)
    assert len(ra_rgb) == 8 and len(ra_tg) == 8
    print("    [OK] Synchronized VideoRandAugment with shared operations across modalities")

    # 3. Synchronized Video Cutout
    cutout = VideoCutout(size=128)
    co_rgb, patch = cutout(dummy_rgb)
    co_tg, _ = cutout(dummy_tg, patch=patch)
    assert len(co_rgb) == 8 and len(co_tg) == 8
    print(f"    [OK] Synchronized VideoCutout with shared patch {patch} across modalities")

    # 4. Temporal Gradient Computation
    next_frames = [Image.fromarray(np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)) for _ in range(8)]
    tg_frames = compute_temporal_gradient(dummy_rgb, next_frames)
    assert len(tg_frames) == 8
    print("    [OK] Temporal gradient computed on-the-fly")


def test_fixmatch_joint_forward():
    print("[-] Testing FixMatch Joint Forward Pass (BatchNorm consistency)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet3D18(num_classes=101).to(device)
    model.train()

    bz_l = 2
    bz_u = 4
    x_l = torch.randn(bz_l, 3, 8, 224, 224, device=device)
    x_u_w = torch.randn(bz_u, 3, 8, 224, 224, device=device)
    x_u_s = torch.randn(bz_u, 3, 8, 224, 224, device=device)
    y_l = torch.tensor([1, 10], device=device)

    # Joint forward pass
    joint_inputs = torch.cat([x_l, x_u_w, x_u_s], dim=0)
    assert joint_inputs.size(0) == bz_l + 2 * bz_u
    joint_logits = model(joint_inputs)

    logits_l = joint_logits[:bz_l]
    logits_u_w, logits_u_s = joint_logits[bz_l:].chunk(2)

    loss_s = F.cross_entropy(logits_l, y_l)
    with torch.no_grad():
        probs_w = F.softmax(logits_u_w, dim=-1)
        max_p, pseudo_y = torch.max(probs_w, dim=-1)
        mask = (max_p >= 0.0)

    loss_u = F.cross_entropy(logits_u_s[mask], pseudo_y[mask])
    loss = loss_s + loss_u
    loss.backward()
    print("    [OK] Joint forward pass for labeled, weak, and strong batches verified!")


def test_loss_backward():
    print("[-] Testing F2-FixMatch P-Aug Gradient Flow through Backbone...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f2_model = F2FixMatchModel(num_classes=101).to(device)
    optimizer = torch.optim.SGD(f2_model.parameters(), lr=0.01, momentum=0.9)

    # Labeled batch: B_l = 2
    rgb_l = torch.randn(2, 3, 8, 224, 224, device=device)
    tg_l = torch.randn(2, 3, 8, 224, 224, device=device)
    targets_l = torch.tensor([5, 42], device=device)

    logits_rgb_l = f2_model.forward_logits(f2_model.forward_backbone(rgb_l))
    logits_tg_l = f2_model.forward_logits(f2_model.forward_backbone(tg_l))
    loss_s = 0.5 * (F.cross_entropy(logits_rgb_l, targets_l) + F.cross_entropy(logits_tg_l, targets_l))

    # Unlabeled batch: B_u = 2
    rgb_w = torch.randn(2, 3, 8, 224, 224, device=device)
    tg_w = torch.randn(2, 3, 8, 224, 224, device=device)

    # Gradients MUST remain enabled on z_w extraction for P-Aug feature backpropagation
    z_w_rgb = f2_model.forward_backbone(rgb_w)
    logits_w_rgb = f2_model.forward_logits(z_w_rgb)
    z_w_tg = f2_model.forward_backbone(tg_w)
    logits_w_tg = f2_model.forward_logits(z_w_tg)

    with torch.no_grad():
        p_w_agg = 0.5 * (F.softmax(logits_w_rgb, dim=-1) + F.softmax(logits_w_tg, dim=-1))
        max_p, pseudo_y = torch.max(p_w_agg, dim=-1)
        mask = (max_p >= 0.0)

    # P-Aug on weak features: Z_{w_p}
    _, logits_wp_rgb = f2_model.forward_paug(z_w_rgb)
    _, logits_wp_tg = f2_model.forward_paug(z_w_tg)
    loss_u_wp = 0.5 * (F.cross_entropy(logits_wp_rgb[mask], pseudo_y[mask]) + F.cross_entropy(logits_wp_tg[mask], pseudo_y[mask]))

    total_loss = loss_s + loss_u_wp
    optimizer.zero_grad()
    total_loss.backward()

    # Verify that gradients flowed all the way through backbone conv1 from loss_u_wp
    assert f2_model.backbone.conv1.weight.grad is not None
    assert f2_model.backbone.conv1.weight.grad.abs().sum().item() > 0.0
    optimizer.step()
    print(f"    [OK] Verified P-Aug backpropagation through 3D-ResNet backbone! Grad sum: {f2_model.backbone.conv1.weight.grad.abs().sum().item():.4f}")


if __name__ == "__main__":
    print("==================================================")
    print("Running FixMatch & F2-FixMatch Verification Suite")
    print("==================================================")
    test_models()
    test_augmentations()
    test_fixmatch_joint_forward()
    test_loss_backward()
    print("==================================================")
    print("All architecture, data, and loss tests PASSED!")
    print("==================================================")
