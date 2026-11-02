"""
Compose transforms pipeline in MMAction2.
"""

from ..builder import PIPELINES, build_pipeline


@PIPELINES.register_module()
class Compose:
    """Compose multiple transforms sequentially."""

    def __init__(self, transforms):
        assert isinstance(transforms, list)
        self.transforms = []
        for transform in transforms:
            if callable(transform):
                self.transforms.append(transform)
            elif isinstance(transform, dict):
                transform = build_pipeline(transform)
                self.transforms.append(transform)
            else:
                raise TypeError(f"transform must be callable or a dict, but got {type(transform)}")

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_str = self.__class__.__name__ + '(\n'
        for t in self.transforms:
            format_str += f'    {t}\n'
        format_str += ')'
        return format_str
