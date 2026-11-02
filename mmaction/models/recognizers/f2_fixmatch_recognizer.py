"""
F2-FixMatch Recognizer for MMAction2.
Faithfully reproduces Tu et al. (ACM TOMM 2025):
1. Bi-Modal Architecture: Shared 3D-ResNet-18 Backbone for RGB and Temporal Gradient (TG).
2. Progressive Augmentation (P-Aug): 4 feature types (Z_w, Z_s, Z_{w_p}, Z_{s_p}) via FeaturePerturbator.
3. Multihead Pseudo-Labeling (MPL): Weak RGB + TG prediction aggregation, tau=0.3 thresholding,
   and 3-way consistency alignment weighted by alpha=2, beta=2, gamma=1.
4. Preserves full gradient flow from L_u^{w_p} back through 3D backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import RECOGNIZERS, build_head, build_perturbator
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class F2FixMatchRecognizer3D(BaseRecognizer):
    """
    F2-FixMatch Semi-Supervised 3D Action Recognizer.
    """

    def __init__(self,
                 backbone,
                 cls_head=None,
                 tap_head=None,
                 perturbator=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)

        # 3D Spatio-Temporal classification head
        self.cls_head_3d = self.cls_head

        # 2D TAP (Temporal-Average-Pooling) classification head
        if tap_head is not None:
            self.cls_head_tap = build_head(tap_head)
        else:
            from ..heads.f2_head import F2Head
            num_classes = cls_head.get('num_classes', 101) if cls_head else 101
            in_channels = cls_head.get('in_channels', 512) if cls_head else 512
            self.cls_head_tap = F2Head(num_classes=num_classes, in_channels=in_channels, dropout_ratio=0.5)

        # Feature Perturbator (P-Aug)
        if perturbator is not None:
            self.perturbator = build_perturbator(perturbator)
        else:
            from ..perturbator.paug import FeaturePerturbator
            self.perturbator = FeaturePerturbator(spatial_drop_prob=0.5, flip_prob=0.5, noise_injection_prob=0.5)

        # Hyperparameters
        self.threshold = self.train_cfg.get('threshold', 0.3)
        self.warmup_epochs = self.train_cfg.get('warmup_epochs', 40)
        self.alpha = self.train_cfg.get('alpha', 2.0)
        self.beta = self.train_cfg.get('beta', 2.0)
        self.gamma = self.train_cfg.get('gamma', 1.0)

    def forward_train(self,
                      imgs_rgb_l,
                      imgs_tg_l,
                      labels,
                      rgb_w=None,
                      rgb_s=None,
                      tg_w=None,
                      tg_s=None,
                      cur_epoch=0,
                      **kwargs):
        """
        Executes F2-FixMatch training step.
        """
        bz_l = imgs_rgb_l.size(0)
        gt_labels = labels.squeeze()

        # -------------------------------------------------------------
        # 1. Supervised Loss L_s (RGB and TG modalities)
        # -------------------------------------------------------------
        feat_rgb_l = self.extract_feat(imgs_rgb_l)
        logits_rgb_l = self.cls_head_3d(feat_rgb_l)

        feat_tg_l = self.extract_feat(imgs_tg_l)
        logits_tg_l = self.cls_head_3d(feat_tg_l)

        loss_s_rgb = F.cross_entropy(logits_rgb_l, gt_labels)
        loss_s_tg = F.cross_entropy(logits_tg_l, gt_labels)
        loss_s = 0.5 * (loss_s_rgb + loss_s_tg)

        loss_dict = dict(
            loss_s=loss_s,
            loss_u_s=torch.tensor(0.0, device=imgs_rgb_l.device),
            loss_u_wp=torch.tensor(0.0, device=imgs_rgb_l.device),
            loss_u_sp=torch.tensor(0.0, device=imgs_rgb_l.device),
            conf_samples=0,
            total_u_samples=0
        )

        # Check if unsupervised branch is active
        if (rgb_w is None or rgb_s is None or tg_w is None or tg_s is None or
                cur_epoch < self.warmup_epochs):
            return loss_dict

        bz_u = rgb_w.size(0)

        # -------------------------------------------------------------
        # 2. Extract Weak Features with Gradients Intact for P-Aug
        # -------------------------------------------------------------
        z_w_rgb = self.extract_feat(rgb_w)
        logits_w_rgb = self.cls_head_3d(z_w_rgb)

        z_w_tg = self.extract_feat(tg_w)
        logits_w_tg = self.cls_head_3d(z_w_tg)

        # Multihead Pseudo-Labeling (MPL) aggregation: targets detached
        with torch.no_grad():
            p_w_rgb = F.softmax(logits_w_rgb, dim=-1)
            p_w_tg = F.softmax(logits_w_tg, dim=-1)
            p_w_agg = 0.5 * (p_w_rgb + p_w_tg)
            max_probs, pseudo_labels = torch.max(p_w_agg, dim=-1)
            mask = (max_probs >= self.threshold)
            num_confident = mask.sum().item()

        if num_confident > 0:
            conf_idx = mask.nonzero(as_tuple=False).squeeze(1)
            conf_labels = pseudo_labels[conf_idx]

            # --- Frame-level Strong Augmentation features Z_s ---
            z_s_rgb = self.extract_feat(rgb_s)
            logits_s_rgb = self.cls_head_3d(z_s_rgb)

            z_s_tg = self.extract_feat(tg_s)
            logits_s_tg = self.cls_head_3d(z_s_tg)

            loss_u_s_rgb = F.cross_entropy(logits_s_rgb[conf_idx], conf_labels, reduction='sum')
            loss_u_s_tg = F.cross_entropy(logits_s_tg[conf_idx], conf_labels, reduction='sum')
            loss_u_s = (loss_u_s_rgb + loss_u_s_tg) / (2.0 * float(bz_u))

            # --- P-Aug: Z_{w_p} (frame weak, feature perturbed) ---
            # Gradients flow smoothly back through z_w to backbone
            feat_wp_rgb = self.perturbator(z_w_rgb)
            logits_wp_rgb = self.cls_head_tap(feat_wp_rgb)

            feat_wp_tg = self.perturbator(z_w_tg)
            logits_wp_tg = self.cls_head_tap(feat_wp_tg)

            loss_u_wp_rgb = F.cross_entropy(logits_wp_rgb[conf_idx], conf_labels, reduction='sum')
            loss_u_wp_tg = F.cross_entropy(logits_wp_tg[conf_idx], conf_labels, reduction='sum')
            loss_u_wp = (loss_u_wp_rgb + loss_u_wp_tg) / (2.0 * float(bz_u))

            # --- P-Aug: Z_{s_p} (frame strong, feature perturbed) ---
            feat_sp_rgb = self.perturbator(z_s_rgb)
            logits_sp_rgb = self.cls_head_tap(feat_sp_rgb)

            feat_sp_tg = self.perturbator(z_s_tg)
            logits_sp_tg = self.cls_head_tap(feat_sp_tg)

            loss_u_sp_rgb = F.cross_entropy(logits_sp_rgb[conf_idx], conf_labels, reduction='sum')
            loss_u_sp_tg = F.cross_entropy(logits_sp_tg[conf_idx], conf_labels, reduction='sum')
            loss_u_sp = (loss_u_sp_rgb + loss_u_sp_tg) / (2.0 * float(bz_u))

            loss_dict['loss_u_s'] = self.alpha * loss_u_s
            loss_dict['loss_u_wp'] = self.beta * loss_u_wp
            loss_dict['loss_u_sp'] = self.gamma * loss_u_sp

        loss_dict['conf_samples'] = num_confident
        loss_dict['total_u_samples'] = bz_u
        return loss_dict

    def forward_test(self, imgs, **kwargs):
        if imgs.dim() == 6:
            b, num_views, c, t, h, w = imgs.shape
            imgs = imgs.view(-1, c, t, h, w)
            x = self.extract_feat(imgs)
            cls_score = self.cls_head_3d(x)
            cls_score = cls_score.view(b, num_views, -1)
            return cls_score.mean(dim=1)
        else:
            x = self.extract_feat(imgs)
            return self.cls_head_3d(x)
