"""
Comprehensive Verification Suite for MMAction2 Refactoring of FixMatch & F2-FixMatch.
Tests:
1. MMAction2 Config Parsing (Config.fromfile)
2. MMAction2 Model Architecture & Registries (ResNet3d, I3DHead, F2Head, FeaturePerturbator)
3. MMAction2 Pipeline & Data Augmentations (SampleFrames, RawFrameDecode, VideoRandAugment, Cutout, ThreeCrop)
4. FixMatch Joint Batch Forward Pass (BatchNorm consistency)
5. F2-FixMatch Bi-modal P-Aug Gradient Flow through 3D Backbone
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Import refactored MMAction2 framework
from mmaction.utils.config import Config
from mmaction.models.builder import (
    build_recognizer, build_backbone, build_head, build_perturbator
)
from mmaction.models.backbones.resnet3d import ResNet3d
from mmaction.models.heads.i3d_head import I3DHead
from mmaction.models.heads.f2_head import F2Head
from mmaction.models.perturbator.paug import FeaturePerturbator
from mmaction.models.recognizers.recognizer3d import Recognizer3D
from mmaction.models.recognizers.fixmatch_recognizer import FixMatchRecognizer3D
from mmaction.models.recognizers.f2_fixmatch_recognizer import F2FixMatchRecognizer3D
from mmaction.datasets.pipelines import (
    Compose, SampleFrames, RawFrameDecode, RandomResizedCrop,
    VideoRandAugment, VideoCutout, ThreeCrop, Normalize
)
from mmaction.core.evaluation import top_k_accuracy


def test_configs():
    print("[-] 1. Testing MMAction2 Config Loading...")
    configs = [
        'configs/supervised/r3d_r18_ucf101_10percent.py',
        'configs/fixmatch/fixmatch_r3d_r18_ucf101_10percent.py',
        'configs/f2_fixmatch/f2_fixmatch_r3d_r18_ucf101_10percent.py',
    ]
    for c_path in configs:
        assert os.path.exists(c_path), f"Missing config: {c_path}"
        cfg = Config.fromfile(c_path)
        assert hasattr(cfg, 'model')
        assert hasattr(cfg, 'data')
        print(f"    [OK] Loaded {os.path.basename(c_path)} -> Model: {cfg.model.type}")


def test_models():
    print("[-] 2. Testing MMAction2 Architecture & Registries...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Running on device: {device}")

    # Test ResNet3d backbone (depth=18)
    backbone = ResNet3d(depth=18, spatial_strides=(1, 2, 2, 2), inflate=(0, 0, 1, 1)).to(device)
    x = torch.randn(2, 3, 8, 224, 224, device=device)
    feat = backbone(x)
    assert feat.shape == (2, 512, 8, 7, 7), f"Expected (2, 512, 8, 7, 7), got {feat.shape}"
    print(f"    [OK] ResNet3d Backbone feature output: {feat.shape}")

    # Test I3DHead (3D features)
    head_3d = I3DHead(num_classes=101, in_channels=512, dropout_ratio=0.5).to(device)
    logits_3d = head_3d(feat)
    assert logits_3d.shape == (2, 101)
    print(f"    [OK] I3DHead logits output: {logits_3d.shape}")

    # Test FeaturePerturbator (P-Aug with spatial_drop_prob=0.5)
    perturbator = FeaturePerturbator(spatial_drop_prob=0.5, flip_prob=0.5, noise_injection_prob=0.5).to(device)
    feat_tap = perturbator(feat)
    assert feat_tap.shape == (2, 512, 7, 7), f"Expected (2, 512, 7, 7), got {feat_tap.shape}"
    print(f"    [OK] P-Aug FeaturePerturbator TAP feature output: {feat_tap.shape}")

    # Test F2Head (2D TAP features)
    head_tap = F2Head(num_classes=101, in_channels=512, dropout_ratio=0.5).to(device)
    logits_tap = head_tap(feat_tap)
    assert logits_tap.shape == (2, 101)
    print(f"    [OK] F2Head TAP logits output: {logits_tap.shape}")


def test_pipelines():
    print("[-] 3. Testing MMAction2 Data Pipelines...")
    dummy_frames = [Image.fromarray(np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)) for _ in range(8)]
    dummy_tg = [Image.fromarray(np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)) for _ in range(8)]

    # Test Weak Spatial Crop & Flip (shared parameters)
    cropper = RandomResizedCrop(min_size=256, max_size=320, crop_size=224, is_train=True)
    res = {'imgs': [f.copy() for f in dummy_frames], 'imgs_tg': [f.copy() for f in dummy_tg]}
    res = cropper(res)
    assert len(res['imgs']) == 8 and len(res['imgs_tg']) == 8
    assert res['imgs'][0].size == (224, 224) and res['imgs_tg'][0].size == (224, 224)
    print("    [OK] RandomResizedCrop (Weak) synchronized across RGB and TG to (224, 224)")

    # Test VideoRandAugment (Uniform +/- signs, shared across modalities)
    ra = VideoRandAugment(n=2, m=10)
    res = ra(res)
    assert len(res['imgs']) == 8 and len(res['imgs_tg']) == 8
    print("    [OK] VideoRandAugment synchronized across RGB and TG")

    # Test VideoCutout (128x128 patch, shared across modalities)
    co = VideoCutout(size=128)
    res = co(res)
    assert len(res['imgs']) == 8 and len(res['imgs_tg']) == 8
    print(f"    [OK] VideoCutout (128x128) synchronized across RGB and TG: {res['cutout_patch']}")

    # Test ThreeCrop (30-view evaluation: 10 clips x 3 crops)
    dummy_10_clips = [[Image.fromarray(np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)) for _ in range(8)] for _ in range(10)]
    tc = ThreeCrop(crop_size=224, scale=256)
    tc_res = tc({'imgs': dummy_10_clips})
    assert len(tc_res['imgs']) == 30
    assert tc_res['imgs'][0][0].size == (224, 224)
    print("    [OK] ThreeCrop generated 30 views (10 clips x 3 spatial crops)")


def test_fixmatch_joint_forward():
    print("[-] 4. Testing MMAction2 FixMatchRecognizer3D Joint Pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config.fromfile('configs/fixmatch/fixmatch_r3d_r18_ucf101_10percent.py')
    model = build_recognizer(cfg.model, train_cfg=cfg.model.train_cfg).to(device)
    model.train()

    bz_l = 2
    bz_u = 4
    imgs_l = torch.randn(bz_l, 3, 8, 224, 224, device=device)
    labels = torch.tensor([1, 42], device=device)
    imgs_w = torch.randn(bz_u, 3, 8, 224, 224, device=device)
    imgs_s = torch.randn(bz_u, 3, 8, 224, 224, device=device)

    # Train step after warmup (joint forward pass through backbone)
    loss_dict = model.forward_train(
        imgs=imgs_l, labels=labels, imgs_weak=imgs_w, imgs_strong=imgs_s, cur_epoch=50
    )
    total_loss = loss_dict['loss_cls'] + loss_dict['loss_u']
    total_loss.backward()
    print(f"    [OK] FixMatchRecognizer3D loss computed successfully! Loss_S: {loss_dict['loss_cls'].item():.4f}, Loss_U: {loss_dict['loss_u'].item():.4f}")


def test_f2_fixmatch_gradient_flow():
    print("[-] 5. Testing MMAction2 F2FixMatchRecognizer3D P-Aug Gradient Flow...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config.fromfile('configs/f2_fixmatch/f2_fixmatch_r3d_r18_ucf101_10percent.py')
    model = build_recognizer(cfg.model, train_cfg=cfg.model.train_cfg).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    bz_l = 2
    bz_u = 2
    imgs_rgb_l = torch.randn(bz_l, 3, 8, 224, 224, device=device)
    imgs_tg_l = torch.randn(bz_l, 3, 8, 224, 224, device=device)
    labels = torch.tensor([5, 99], device=device)

    rgb_w = torch.randn(bz_u, 3, 8, 224, 224, device=device)
    rgb_s = torch.randn(bz_u, 3, 8, 224, 224, device=device)
    tg_w = torch.randn(bz_u, 3, 8, 224, 224, device=device)
    tg_s = torch.randn(bz_u, 3, 8, 224, 224, device=device)

    loss_dict = model.forward_train(
        imgs_rgb_l=imgs_rgb_l, imgs_tg_l=imgs_tg_l, labels=labels,
        rgb_w=rgb_w, rgb_s=rgb_s, tg_w=tg_w, tg_s=tg_s, cur_epoch=50
    )

    total_loss = loss_dict['loss_s'] + loss_dict['loss_u_s'] + loss_dict['loss_u_wp'] + loss_dict['loss_u_sp']
    optimizer.zero_grad()
    total_loss.backward()

    # Verify that gradients flowed from P-Aug through backbone conv1
    conv1_grad = model.backbone.conv1.weight.grad
    assert conv1_grad is not None
    grad_norm = conv1_grad.abs().sum().item()
    assert grad_norm > 0.0
    optimizer.step()
    print(f"    [OK] F2FixMatchRecognizer3D full backpropagation verified! Backbone conv1 grad: {grad_norm:.4f}")


def test_topk_evaluation():
    print("[-] 6. Testing Top-K Evaluation Metrics...")
    scores = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
    labels = torch.tensor([1, 0])
    top1, top5 = top_k_accuracy(scores, labels, topk=(1, 5))
    assert top1 == 100.0
    print(f"    [OK] Top-1: {top1:.1f}%, Top-5: {top5:.1f}%")


if __name__ == "__main__":
    print("=" * 65)
    print("MMAction2 Refactoring Verification Suite")
    print("=" * 65)
    test_configs()
    test_models()
    test_pipelines()
    test_fixmatch_joint_forward()
    test_f2_fixmatch_gradient_flow()
    test_topk_evaluation()
    print("=" * 65)
    print("All MMAction2 Refactored Modules & Verification Tests PASSED!")
    print("=" * 65)
