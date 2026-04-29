from abc import ABC, abstractmethod
from typing import TypeVar, Dict, Generic, Optional
import equinox as eqx
import jax.numpy as jnp

from terraflow.util import typing, device, log
from terraflow.mesh.abstract import AbstractMesh
from terraflow.vert_coord.abstract import AbstractVertCoordinate


IntType = typing.IntType
Array = typing.Array
ArrayWithUnits = typing.ArrayWithUnits

HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
HLO = TypeVar("HLO", bound="AbstractLayout")
VLO = TypeVar("VLO", bound="AbstractLayout")


class MeshEntityElement:
    Vtx = "vtx"
    Edge = "edge"
    Cell = "cell"


MeshEntity = MeshEntityElement()


class VertCoordLocation:
    Cell = "cell"
    Lev = "lev"
    Sfc = "sfc"


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

    def __init__(
        self,
        horizontal_coord: Optional[Dict[str, Array]] = None,
        vertical_coord: Optional[Array] = None,
    ):
        self.horizontal_coord = horizontal_coord
        self.vertical_coord = vertical_coord


class CoordinateSystem(eqx.Module, Generic[HM, VM, HLO, VLO]):
    _hori: HM
    _vert: VM
    _hori_layout: HLO
    _vert_layout: VLO

    def __init__(
        self,
        hori: HM,
        vert: VM,
        hori_layout: Optional[HLO] = None,
        vert_layout: Optional[VLO] = None,
    ):
        self._hori = hori
        self._vert = vert
        if hori_layout is None:
            _hori_layout = hori.get_layout()
        else:
            _hori_layout = hori_layout
        if vert_layout is None:
            _vert_layout = vert.get_layout()
        else:
            _vert_layout = vert_layout
        self._hori_layout = _hori_layout
        self._vert_layout = _vert_layout
        log.logger.success("Initialized coordinate system.")

    @property
    def mesh(self) -> HM:
        return self._hori

    @property
    def vert(self) -> VM:
        return self._vert

    @property
    def hori_layout(self) -> HLO:
        return self._hori_layout

    @property
    def vert_layout(self) -> VLO:
        return self._vert_layout

    @property
    def dev(self) -> device.AbstractDevice:
        return self.mesh.dev

    def fill_halo(
        self,
        field: Array,
        hori_loc: str,
        vert_loc: str,
        is_index: bool = False,
    ) -> Array:
        field = self.mesh.fill_halo(field, loc=hori_loc, is_index=is_index)
        field = self.vert.fill_halo(field, loc=vert_loc, is_index=is_index)
        return field

    def zeros_like(
        self,
        hori_loc: str,
        vert_loc: str,
        basis: str,
        is_tracer: bool = False,
        ntracers: int = 0,
        is_expand_dims: bool = False,
        expand_axis: int = 0,
        expand_axis_length: int = 1,
    ) -> Array:
        hori_shape = self.hori_layout.shape(
            self,
            loc=hori_loc,
            basis=basis,
        )
        vert_shape = self.vert_layout.shape(
            self,
            loc=vert_loc,
        )
        dtype = self.mesh.dev.float_type
        arch = self.mesh.dev()

        if vert_shape[0] == 0:
            if vert_loc != VertLoc.Sfc:
                vert_shape = (1,)
                shape = vert_shape + hori_shape
            else:
                shape = hori_shape
        else:
            shape = vert_shape + hori_shape

        if is_tracer:
            shape = (ntracers,) + shape

        if is_expand_dims:
            shape = (
                shape[:expand_axis]
                + (expand_axis_length,)
                + shape[expand_axis:]
            )

        return jnp.zeros(shape, dtype=dtype, device=arch)

    def get_hori_layout(
        self,
        loc: str,
        basis: str,
    ) -> CoordinateInfo:
        return self.hori_layout.get_layout(
            self,
            loc,
            basis,
        )

    def get_vert_layout(
        self,
        loc: str,
    ) -> CoordinateInfo:
        return self.vert_layout.get_layout(
            self,
            loc,
        )


class AbstractLayout(eqx.Module, Generic[HM, VM, HLO, VLO], ABC):

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        """Return the alias name of the layout."""
        raise NotImplementedError

    @abstractmethod
    def shape(
        self,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        loc: str,
        *args,
        **kwargs,
    ) -> tuple[int, ...]:
        """Return the shape of the specified layout position."""
        raise NotImplementedError

    @abstractmethod
    def get_layout(
        self,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        loc: str,
        *args,
        **kwargs,
    ) -> CoordinateInfo:
        raise NotImplementedError

    @abstractmethod
    def get_idx(
        self,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        loc: str,
        domain: str,
        *args,
        **kwargs,
    ) -> tuple[IntType, IntType]:
        raise NotImplementedError


class AbstractUnstaggeringLayout(AbstractLayout):
    pass


class AbstractStaggeringLayout(AbstractLayout):
    pass


class AbstractCGridStaggeringLayout(AbstractStaggeringLayout):
    pass
