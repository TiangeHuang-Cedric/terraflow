from abc import ABC, abstractmethod
from typing import TypeVar, Dict, Generic
import equinox as eqx

from terraflow.util import typing
from terraflow.mesh import AbstractMesh
from terraflow.vert_coord import AbstractVertCoordinate


Array = typing.Array
ArrayWithUnits = typing.ArrayWithUnits

HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
LO = TypeVar("LO", bound="AbstractLayout")


class MeshEntityElement:
    Vtx = "vtx"
    Edge = "edge"
    Cell = "cell"


MeshEntity = MeshEntityElement()


class VertCoordLocation:
    Cell = "cell"
    Lev = "lev"


VertLoc = VertCoordLocation()


class BasisEnum:
    Scalar = "scalar"
    NormalX = "normal_x"
    NormalY = "normal_y"
    Normal = "normal"
    Tangent = "tangent"
    Vector2D = "vector_2d"
    Vector3D = "vector_3d"


Basis = BasisEnum()


class CoordinateInfo:
    horizontal_coord: Dict[str, Array]
    vertical_coord: Array

    def __init__(self, horizontal_coord: Dict[str, Array], vertical_coord: Array):
        self.horizontal_coord = horizontal_coord
        self.vertical_coord = vertical_coord


class CoordinateSystem(eqx.Module, Generic[HM, VM, LO]):
    _hori: HM
    _vert: VM
    _layout: LO

    def __init__(self, hori: HM, vert: VM, layout: LO):
        self._hori = hori
        self._vert = vert
        self._layout = layout

    @property
    def mesh(self) -> HM:
        return self._hori

    @property
    def vert(self) -> VM:
        return self._vert

    @property
    def layout(self) -> LO:
        return self._layout

    def get_layout(
        self,
        hori_loc: str,
        vert_loc: str,
        basis: str,
    ) -> CoordinateInfo:
        return self.layout.get_layout(
            self,
            hori_loc,
            vert_loc,
            basis,
        )


class AbstractLayout(eqx.Module, Generic[HM, VM, LO], ABC):

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        """Return the alias name of the layout."""
        raise NotImplementedError

    @abstractmethod
    def get_layout(
        self,
        coord: CoordinateSystem[HM, VM, LO],
        hori_loc: str,
        vert_loc: str,
        basis: str,
    ) -> CoordinateInfo:
        pass


class AbstractUnstaggeringLayout(AbstractLayout):
    pass


class AbstractStaggeringLayout(AbstractLayout):
    pass


class AbstractCGridStaggeringLayout(AbstractStaggeringLayout):
    pass
