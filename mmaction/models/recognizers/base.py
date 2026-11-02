"""
Base Recognizer for MMAction2.
"""

from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from ..builder import build_backbone, build_head


class BaseRecognizer(nn.Module, metaclass=ABCMeta):
    """Base class for recognizers."""

    def __init__(self,
                 backbone,
                 cls_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        if cls_head is not None:
            self.cls_head = build_head(cls_head)
        else:
            self.cls_head = None
        self.train_cfg = train_cfg if train_cfg is not None else dict()
        self.test_cfg = test_cfg if test_cfg is not None else dict()

    def extract_feat(self, imgs):
        return self.backbone(imgs)

    @abstractmethod
    def forward_train(self, imgs, labels, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, imgs, **kwargs):
        pass

    def forward(self, imgs, label=None, return_loss=True, **kwargs):
        if return_loss:
            if label is None:
                raise ValueError("Label must be provided when return_loss=True")
            return self.forward_train(imgs, label, **kwargs)
        return self.forward_test(imgs, **kwargs)
