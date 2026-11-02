"""
MMAction2 Model Builders and Registries.
"""

from ..utils.registry import Registry, build_from_cfg

BACKBONES = Registry('backbones')
HEADS = Registry('heads')
RECOGNIZERS = Registry('recognizers')
LOSSES = Registry('losses')
PERTURBATORS = Registry('perturbators')


def build_backbone(cfg):
    return build_from_cfg(cfg, BACKBONES)


def build_head(cfg):
    return build_from_cfg(cfg, HEADS)


def build_recognizer(cfg, train_cfg=None, test_cfg=None):
    default_args = dict(train_cfg=train_cfg, test_cfg=test_cfg)
    return build_from_cfg(cfg, RECOGNIZERS, default_args=default_args)


def build_loss(cfg):
    return build_from_cfg(cfg, LOSSES)


def build_perturbator(cfg):
    return build_from_cfg(cfg, PERTURBATORS)
