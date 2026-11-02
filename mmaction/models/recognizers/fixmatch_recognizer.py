"""
FixMatch Baseline 3D Recognizer for MMAction2.
Implements joint batch forward pass for consistent BatchNorm tracking,
weak-to-strong pseudo-labeling, and confidence masking (tau=0.3).
"""

import torch
import torch.nn.functional as F
from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class FixMatchRecognizer3D(BaseRecognizer):
    """FixMatch Semi-Supervised 3D Recognizer."""

    def __init__(self,
                 backbone,
                 cls_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(backbone, cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        self.threshold = self.train_cfg.get('threshold', 0.3)
        self.lambda_u = self.train_cfg.get('lambda_u', 1.0)
        self.warmup_epochs = self.train_cfg.get('warmup_epochs', 40)

    def forward_train(self,
                      imgs,
                      labels,
                      imgs_weak=None,
                      imgs_strong=None,
                      cur_epoch=0,
                      **kwargs):
        """
        Defines the computation for FixMatch.
        Warmup: only labeled data is used.
        After warmup: joint forward pass [imgs, imgs_weak, imgs_strong].
        """
        bz_l = imgs.size(0)
        gt_labels = labels.squeeze()

        if imgs_weak is None or imgs_strong is None or cur_epoch < self.warmup_epochs:
            # Supervised warmup phase
            feat_l = self.extract_feat(imgs)
            logits_l = self.cls_head(feat_l)
            loss_dict = self.cls_head.loss(logits_l, gt_labels)
            loss_dict['loss_u'] = torch.tensor(0.0, device=imgs.device)
            loss_dict['conf_samples'] = 0
            loss_dict['total_u_samples'] = 0
            return loss_dict

        bz_u = imgs_weak.size(0)

        # Joint forward pass through backbone & head for unified BatchNorm running stats
        joint_imgs = torch.cat([imgs, imgs_weak, imgs_strong], dim=0)
        joint_feat = self.extract_feat(joint_imgs)
        joint_logits = self.cls_head(joint_feat)

        logits_l = joint_logits[:bz_l]
        logits_u_w, logits_u_s = joint_logits[bz_l:].chunk(2)

        # Supervised loss L_s
        loss_dict = self.cls_head.loss(logits_l, gt_labels)

        # Unsupervised FixMatch pseudo-labeling
        with torch.no_grad():
            probs_w = F.softmax(logits_u_w, dim=-1)
            max_probs, pseudo_labels = torch.max(probs_w, dim=-1)
            mask = (max_probs >= self.threshold)
            num_confident = mask.sum().item()

        if num_confident > 0:
            loss_u = F.cross_entropy(logits_u_s[mask], pseudo_labels[mask], reduction='sum') / float(bz_u)
            loss_dict['loss_u'] = self.lambda_u * loss_u
        else:
            loss_dict['loss_u'] = torch.tensor(0.0, device=imgs.device)

        loss_dict['conf_samples'] = num_confident
        loss_dict['total_u_samples'] = bz_u
        return loss_dict

    def forward_test(self, imgs, **kwargs):
        if imgs.dim() == 6:
            b, num_views, c, t, h, w = imgs.shape
            imgs = imgs.view(-1, c, t, h, w)
            x = self.extract_feat(imgs)
            cls_score = self.cls_head(x)
            cls_score = cls_score.view(b, num_views, -1)
            return cls_score.mean(dim=1)
        else:
            x = self.extract_feat(imgs)
            return self.cls_head(x)
