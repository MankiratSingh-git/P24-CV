"""
I3D Classification Head for 3D Spatio-Temporal features in MMAction2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import HEADS


@HEADS.register_module()
class I3DHead(nn.Module):
    """
    Standard classification head for 3D representations.
    Performs AdaptiveAvgPool3d((1, 1, 1)), optional dropout, and linear projection.
    """

    def __init__(self,
                 num_classes=101,
                 in_channels=512,
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        if spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            raise NotImplementedError(f"Unsupported spatial_type: {spatial_type}")

        if dropout_ratio > 0:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = nn.Identity()

        self.fc_cls = nn.Linear(in_channels, num_classes)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, self.init_std)
        if self.fc_cls.bias is not None:
            nn.init.constant_(self.fc_cls.bias, 0)

    def forward(self, x):
        """Input: (B, C, T, H, W). Output: (B, num_classes)."""
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        cls_score = self.fc_cls(x)
        return cls_score

    def loss(self, cls_score, labels, **kwargs):
        """Cross entropy loss calculation."""
        loss_cls = F.cross_entropy(cls_score, labels, **kwargs)
        return dict(loss_cls=loss_cls)
