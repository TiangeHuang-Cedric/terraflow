from __future__ import annotations
from typing import TYPE_CHECKING
import importlib

if TYPE_CHECKING:
    from terraflow.dynamics.shallow_water.types import ShallowWaterEquationBuilder as ShallowWaterEquation
    from terraflow.dynamics.baroclinic.hydrostatic import DryHydrostaticEquationBuilder as DryHydrostaticEquation
    from terraflow.dynamics.adv.types import AdvectionEquation2DBuilder as AdvectionEquation2D


_LAZY_CLASS = {
    "ShallowWaterEquation": "terraflow.dynamics.shallow_water.types.ShallowWaterEquationBuilder",
    "DryHydrostaticEquation": "terraflow.dynamics.baroclinic.hydrostatic.DryHydrostaticEquationBuilder",
    "AdvectionEquation2D": "terraflow.dynamics.adv.types.AdvectionEquation2DBuilder"
}


__all__ = [
    "ShallowWaterEquation",
    "DryHydrostaticEquation",
    "AdvectionEquation2D"
]


def __getattr__(name: str):
    if name in _LAZY_CLASS:
        module_path, class_name = _LAZY_CLASS[name].rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
