from typing import TypeVar, Generic
from abc import ABC, abstractmethod

from terraflow.planet import PlanetParameterSet
from terraflow.mesh.abstract import AbstractMesh
from terraflow.vert_coord.abstract import AbstractVertCoordinate
from terraflow.layout.abstract import AbstractLayout, CoordinateSystem


HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
LO = TypeVar("LO", bound=AbstractLayout)


class AbstractDiscretizationMethod(ABC):
    """Abstract base class for operators."""

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        """Return the alias name of the operator."""
        raise NotImplementedError


class FiniteVolumeMethod(AbstractDiscretizationMethod):
    """Finite Volume Method operator implementation."""

    @staticmethod
    def alias_name() -> str:
        return "fvm"


class AbstractOperator(Generic[HM, VM, LO], ABC):
    """Abstract base class for operators."""

    def __init__(
        self,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet
    ):
        self._coord = coord
        self._para = para

    @property
    def coord(self) -> CoordinateSystem[HM, VM, LO]:
        return self._coord

    @property
    def para(self) -> PlanetParameterSet:
        return self._para

    @property
    def mesh(self) -> HM:
        return self.coord.mesh

    @property
    def vert(self) -> VM:
        return self.coord.vert

    @property
    def layout(self) -> LO:
        return self.coord.layout

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


__all__ = [
    "AbstractOperator",
    "AbstractDiscretizationMethod",
    "FiniteVolumeMethod"
]
