"""
Pure PyTorch Registry implementation matching OpenMMLab / MMAction2 interface.
Eliminates C++/CUDA compilation dependencies while preserving the exact MMAction2
modular class registration and builder architecture.
"""

import inspect
from functools import partial


class Registry:
    """A registry to map strings to classes or functions."""

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return key in self._module_dict

    def __repr__(self):
        format_str = self.__class__.__name__ + f"(name={self._name}, items={list(self._module_dict.keys())})"
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """Get the registry record."""
        scope, real_key = self._split_scope_key(key)
        if real_key not in self._module_dict:
            raise KeyError(f"'{real_key}' is not in the '{self.name}' registry. Available: {list(self._module_dict.keys())}")
        return self._module_dict[real_key]

    def _split_scope_key(self, key):
        """Split scope and real key."""
        split_index = key.find('.')
        if split_index != -1:
            return key[:split_index], key[split_index + 1:]
        return None, key

    def register_module(self, name=None, force=False, module=None):
        """Register a module."""
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, got {type(force)}")

        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class) and not inspect.isfunction(module_class):
            raise TypeError(f"module must be a class or a function, but got {type(module_class)}")

        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_names = [module_name]
        else:
            module_names = module_name

        for name in module_names:
            if not force and name in self._module_dict:
                raise KeyError(f"'{name}' is already registered in '{self.name}'")
            self._module_dict[name] = module_class


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict."""
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
    if 'type' not in cfg:
        raise KeyError(f"the cfg dict must contain the key 'type', but got {cfg}")
    if not isinstance(registry, Registry):
        raise TypeError(f"registry must be an Registry object, but got {type(registry)}")
    if default_args is not None and not isinstance(default_args, dict):
        raise TypeError(f"default_args must be a dict, but got {type(default_args)}")

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
    elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")

    try:
        return obj_cls(**args)
    except Exception as e:
        raise type(e)(f"{obj_cls.__name__}: {e}")
