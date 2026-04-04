from __future__ import annotations
from typing import TYPE_CHECKING
import importlib


if TYPE_CHECKING:
    from terraflow.core.util import history as history
    from terraflow.util import datetime as datetime
    from terraflow.util import device as device
    from terraflow.util import easygems as easygems
    from terraflow.util import log as log
    from terraflow.util import post_process as post_process
    from terraflow.util import uxarray as uxarray


_LAZY_SUBMODULES = {
    "datetime": "terraflow.util.datetime",
    "device": "terraflow.util.device",
    "easygems": "terraflow.util.easygems",
    "post_process": "terraflow.util.post_process",
    "uxarray": "terraflow.util.uxarray",
    "log": "terraflow.util.log",
    "history": "terraflow.core.util.history",
}


__all__ = [
    "datetime",
    "device",
    "easygems",
    "post_process",
    "uxarray",
    "history",
    "log",
]


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(_LAZY_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
