"""
MMAction2 Dataset Builders and Registries.
"""

from torch.utils.data import DataLoader
from ..utils.registry import Registry, build_from_cfg

DATASETS = Registry('datasets')
PIPELINES = Registry('pipelines')


def build_dataset(cfg, default_args=None):
    return build_from_cfg(cfg, DATASETS, default_args=default_args)


def build_pipeline(cfg, default_args=None):
    return build_from_cfg(cfg, PIPELINES, default_args=default_args)


def build_dataloader(dataset,
                     videos_per_gpu,
                     workers_per_gpu,
                     shuffle=True,
                     drop_last=False,
                     pin_memory=True,
                     **kwargs):
    """Build PyTorch DataLoader matching MMAction2 signature."""
    dataloader = DataLoader(
        dataset,
        batch_size=videos_per_gpu,
        num_workers=workers_per_gpu,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        **kwargs
    )
    return dataloader
