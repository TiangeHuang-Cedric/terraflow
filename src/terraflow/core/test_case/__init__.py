from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import swm
    from . import adv
    from . import baroclinic


_LAZY_SUBMODULES = {
    "swm": "terraflow.core.test_case.swm",
    "adv": "terraflow.core.test_case.adv",
    "baroclinic": "terraflow.core.test_case.baroclinic",
}

__all__ = [
    "swm",
    "adv",
    "baroclinic",
]


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(_LAZY_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
