from . import shallow_water

from .shallow_water import ShallowWaterEquationBuilder as ShallowWaterEquation
from .baroclinic import DryHydrostaticEquationBuilder as DryHydrostaticEquation

__all__ = [
    "ShallowWaterEquation",
    "DryHydrostaticEquation",
]
