from . import shallow_water

from .shallow_water import ShallowWaterEquationBuilder as ShallowWaterEquation
from .shallow_water import ShallowWaterVariablesBuilder as ShallowWaterVariables

__all__ = [
    "ShallowWaterEquation",
    "ShallowWaterVariables",
]
