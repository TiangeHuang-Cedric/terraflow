from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import Callable, Generic, TypeVar
from typing import Any, Sequence
import jax

from terraflow.util import typing
from terraflow.mesh import AbstractMesh
from terraflow.vert_coord import AbstractVertCoordinate
from terraflow.layout import CoordinateSystem, AbstractLayout
from terraflow.planet import PlanetParameterSet

AxisName = Hashable
Numeric = typing.Numeric
Array = typing.Array


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


class FiniteVolumeMethodOperator(AbstractOperator[HM, VM, LO]):
    """Finite Volume Method operator base class."""

    pass
