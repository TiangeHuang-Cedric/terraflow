from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from terraflow.prepare.readers.abstract import (
        TopographyReaderConfig,
        BackgroundReaderConfig,
    )
    from . import bkg as bkg
    from . import topo as topo
    from . import grist as grist

_LAZY_CLASS = {
    "TopographyReaderConfig": "terraflow.prepare.readers.abstract.TopographyReaderConfig",
    "BackgroundReaderConfig": "terraflow.prepare.readers.abstract.BackgroundReaderConfig",
}


_LAZY_SUBMODULES = {
    "bkg": "terraflow.core.readers.bkg",
    "topo": "terraflow.core.readers.topo",
    "grist": "terraflow.core.readers.grist",
}

__all__ = [
    "TopographyReaderConfig",
    "BackgroundReaderConfig",
    "bkg",
    "topo",
    "grist"
]


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(_LAZY_SUBMODULES[name])
        globals()[name] = module
        return module
    if name in _LAZY_CLASS:
        module_path, class_name = _LAZY_CLASS[name].rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
