"""
Config parser for Python config files matching MMAction2 / mmcv Config interface.
"""

import os
import sys
import types
from importlib.machinery import SourceFileLoader


class ConfigDict(dict):
    """Dictionary that allows attribute-style access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")


def _dict_to_configdict(d):
    if isinstance(d, dict):
        return ConfigDict({k: _dict_to_configdict(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_configdict(v) for v in d]
    return d


class Config(ConfigDict):
    """Configuration class that loads settings from Python files or dictionaries."""

    @staticmethod
    def fromfile(filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Config file not found: {filename}")

        filename = os.path.abspath(filename)
        module_name = os.path.splitext(os.path.basename(filename))[0]

        loader = SourceFileLoader(module_name, filename)
        mod = types.ModuleType(loader.name)
        loader.exec_module(mod)

        cfg_dict = {
            k: v for k, v in mod.__dict__.items()
            if not k.startswith('__') and not isinstance(v, types.ModuleType)
        }

        cfg = Config(_dict_to_configdict(cfg_dict))
        cfg.filename = filename
        return cfg

    def dump(self):
        import pprint
        return pprint.pformat(self)
