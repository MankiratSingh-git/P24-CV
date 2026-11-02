"""
3D-ResNet-18 Backbone, P-Aug Module, and F2-FixMatch Model.
Directly subclasses and wraps the MMAction2 implementation (mmaction/) to ensure
100% architectural and operational unification across standalone and config-based runs.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.models.backbones.resnet3d import ResNet3d, BasicBlock3d
from mmaction.models.perturbator.paug import P_Aug

# Alias for backwards compatibility
BasicBlock3D = BasicBlock3d


class ResNet3D18(ResNet3d):
    """
    3D-ResNet-18 Backbone and Classifier Head as specified in Table 2 of F2-FixMatch.
    Subclasses MMAction2 ResNet3d to eliminate code duplication and maintain
    strict state_dict compatibility with existing checkpoints.

    Stage   Layer                   Output Size (for 3x8x224x224)
    conv1   1x7x7, 64, stride 1,2,2 64 x 8 x 112 x 112 (stride 1 in time, 2 in space)
    pool1   1x3x3 max, stride 1,2,2 64 x 8 x 56 x 56   (stride 1 in time, 2 in space)
    res2    [1x3x3, 64; 1x3x3, 64]x2 64 x 8 x 56 x 56  (temporal stride 1)
    res3    [1x3x3, 128; 1x3x3, 128]x2 128 x 8 x 28 x 28  (stride 1 in time, 2 in space)
    res4    [3x3x3, 256; 3x3x3, 256]x2 256 x 8 x 14 x 14  (stride 1 in time, 2 in space, 3D inflated)
    res5    [3x3x3, 512; 3x3x3, 512]x2 512 x 8 x 7 x 7    (stride 1 in time, 2 in space, 3D inflated)
    head    AdaptiveAvgPool3d(1), Dropout(0.5), FC(512, num_classes)
    """
    def __init__(self, num_classes=101, dropout_ratio=0.5, **kwargs):
        super().__init__(depth=18, **kwargs)

        # Classifier head for 3D spatio-temporal features
        self.avgpool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Classifier head for 2D TAP (Temporal-Avg-Pooled) features
        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc = nn.Linear(512, num_classes)

        self._init_head_weights()

    def _init_head_weights(self):
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

    def extract_feat(self, x):
        """Extract high-level feature representation Z in R^{B x 512 x 8 x 7 x 7}."""
        return super().forward(x)

    def forward_3d_feat(self, feat):
        """Pass 3D feature through pooling, dropout, and FC head."""
        pooled = self.avgpool3d(feat).flatten(1)
        dropped = self.dropout(pooled)
        logits = self.fc(dropped)
        return logits

    def forward_tap_feat(self, feat_2d):
        """Pass 2D TAP feature through pooling, dropout, and FC head."""
        pooled = self.avgpool2d(feat_2d).flatten(1)
        dropped = self.dropout(pooled)
        logits = self.fc(dropped)
        return logits

    def forward(self, x):
        """Standard end-to-end forward pass."""
        feat = self.extract_feat(x)
        return self.forward_3d_feat(feat)


class FeaturePerturbator(P_Aug):
    """
    Feature-level Perturbator A_feat for Progressive Augmentation (P-Aug).
    Subclasses MMAction2 P_Aug to ensure unified behavior across all pipelines.
    1. Temporal-Avg-Pooling (TAP): averages over temporal dimension T (Z -> R^{B x C x H x W}).
    2. Spatial Dropout: randomly zeroes entire channels/patches with probability 0.5.
    3. Noise Injection: CutMix / patch swapping at feature level between batch samples.
    4. Feature Flipping: randomly flips features horizontally.
    """
    def __init__(self, spatial_drop_prob=0.5, flip_prob=0.5, noise_injection_prob=0.5):
        super().__init__(
            spatial_drop_prob=spatial_drop_prob,
            flip_prob=flip_prob,
            noise_injection_prob=noise_injection_prob
        )


class F2FixMatchModel(nn.Module):
    """
    Complete F2-FixMatch Network combining:
    - Shared MMAction2 3D-ResNet-18 Backbone for RGB and Temporal Gradient (TG) modalities
    - MMAction2 Feature Perturbator (P-Aug)
    - Classification Heads for 3D and TAP representations
    """
    def __init__(self, num_classes=101, dropout_ratio=0.5):
        super(F2FixMatchModel, self).__init__()
        self.backbone = ResNet3D18(num_classes=num_classes, dropout_ratio=dropout_ratio)
        self.perturbator = FeaturePerturbator()

    def forward_backbone(self, x):
        return self.backbone.extract_feat(x)

    def forward_logits(self, feat_3d):
        return self.backbone.forward_3d_feat(feat_3d)

    def forward_paug(self, feat_3d):
        """Apply P-Aug and compute predictions on perturbed feature."""
        feat_p = self.perturbator(feat_3d)
        logits_p = self.backbone.forward_tap_feat(feat_p)
        return feat_p, logits_p

    def forward_eval(self, x):
        """Evaluation forward pass (RGB only)."""
        return self.backbone(x)
