"""
MMAction2 Action Recognition Toolbox for FixMatch and F2-FixMatch.
"""

from .models.builder import (
    BACKBONES, HEADS, RECOGNIZERS, LOSSES, PERTURBATORS,
    build_backbone, build_head, build_recognizer, build_loss, build_perturbator
)
from .datasets.builder import (
    DATASETS, PIPELINES, build_dataset, build_pipeline, build_dataloader
)
from .utils.config import Config, ConfigDict

__version__ = '0.24.0'

__all__ = [
    'BACKBONES', 'HEADS', 'RECOGNIZERS', 'LOSSES', 'PERTURBATORS',
    'build_backbone', 'build_head', 'build_recognizer', 'build_loss', 'build_perturbator',
    'DATASETS', 'PIPELINES', 'build_dataset', 'build_pipeline', 'build_dataloader',
    'Config', 'ConfigDict', '__version__'
]
