"""
Feature Perturbator for Progressive Augmentation (P-Aug) in F2-FixMatch.
Section 3.2 of Tu et al. (ACM TOMM 2025):
1. Temporal-Avg-Pooling (TAP): averages over temporal dimension T (Z -> R^{B x C x H x W}).
2. Spatial Dropout: randomly zeroes entire channels/patches with probability 0.5.
3. Feature CutMix / Noise Injection: swaps spatial patches between samples in the mini-batch.
4. Feature Flipping: randomly flips features horizontally with probability 0.5.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import PERTURBATORS


@PERTURBATORS.register_module()
class FeaturePerturbator(nn.Module):
    def __init__(self, spatial_drop_prob=0.5, flip_prob=0.5, noise_injection_prob=0.5):
        super().__init__()
        self.spatial_drop_prob = spatial_drop_prob
        self.flip_prob = flip_prob
        self.noise_injection_prob = noise_injection_prob

    def temporal_avg_pool(self, feat):
        """TAP: average along temporal dimension (dim=2). B x C x T x H x W -> B x C x H x W."""
        return feat.mean(dim=2)

    def apply_spatial_dropout(self, feat_2d):
        if self.spatial_drop_prob > 0 and self.training:
            return F.dropout2d(feat_2d, p=self.spatial_drop_prob, training=self.training)
        return feat_2d

    def apply_feature_flipping(self, feat_2d):
        if random.random() < self.flip_prob and self.training:
            return torch.flip(feat_2d, dims=[-1])
        return feat_2d

    def apply_noise_injection(self, feat_2d):
        if (self.noise_injection_prob > 0 and feat_2d.size(0) > 1 and
                self.training and random.random() < self.noise_injection_prob):
            b, c, h, w = feat_2d.size()
            rand_indices = torch.randperm(b, device=feat_2d.device)
            patch_h = max(1, random.randint(1, max(1, h // 2)))
            patch_w = max(1, random.randint(1, max(1, w // 2)))
            top = random.randint(0, h - patch_h)
            left = random.randint(0, w - patch_w)

            perturbed = feat_2d.clone()
            perturbed[:, :, top:top+patch_h, left:left+patch_w] = feat_2d[rand_indices, :, top:top+patch_h, left:left+patch_w]
            return perturbed
        return feat_2d

    def forward(self, feat_3d):
        """Input: 3D feature Z (B x C x T x H x W). Output: Perturbed TAP feature (B x C x H x W)."""
        feat_2d = self.temporal_avg_pool(feat_3d)
        feat_2d = self.apply_spatial_dropout(feat_2d)
        feat_2d = self.apply_feature_flipping(feat_2d)
        feat_2d = self.apply_noise_injection(feat_2d)
        return feat_2d
