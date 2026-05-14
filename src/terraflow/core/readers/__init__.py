from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import bkg as bkg
    from . import topo as topo


_LAZY_SUBMODULES = {
    "bkg": "terraflow.core.readers.bkg",
    "topo": "terraflow.core.readers.topo",
}

__all__ = [
    "bkg",
    "topo",
]


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(_LAZY_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
