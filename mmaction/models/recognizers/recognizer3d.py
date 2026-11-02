"""
Supervised 3D Recognizer for MMAction2.
"""

import torch
import torch.nn.functional as F
from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer3D(BaseRecognizer):
    """Standard supervised 3D recognizer."""

    def forward_train(self, imgs, labels, **kwargs):
        """Input: imgs (B, C, T, H, W), labels (B,). Returns: dict(loss_cls=...)."""
        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x)
        losses = self.cls_head.loss(cls_score, labels.squeeze())
        return losses

    def forward_test(self, imgs, **kwargs):
        """
        Supports both single-clip test (B, C, T, H, W)
        and multi-view 30-view test (B, num_views, C, T, H, W).
        """
        if imgs.dim() == 6:
            # (B, num_views, C, T, H, W)
            b, num_views, c, t, h, w = imgs.shape
            imgs = imgs.view(-1, c, t, h, w)
            x = self.extract_feat(imgs)
            cls_score = self.cls_head(x)
            cls_score = cls_score.view(b, num_views, -1)
            # Average scores across all 30 views
            cls_score = cls_score.mean(dim=1)
            return cls_score
        else:
            x = self.extract_feat(imgs)
            return self.cls_head(x)
