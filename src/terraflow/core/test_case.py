from __future__ import annotations
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from terraflow.test_case import baroclinic as baroclinic
    from terraflow.test_case import swe as swe
    from terraflow.test_case import adv as adv


_LAZY_SUBMODULES = {
    "swe": "terraflow.test_case.swe",
    "baroclinic": "terraflow.test_case.baroclinic",
    "adv": "terraflow.test_case.adv"
}

__all__ = [
    "swe",
    "baroclinic",
    "adv"
]


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(_LAZY_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
