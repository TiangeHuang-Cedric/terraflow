from __future__ import annotations
from typing import Self, TYPE_CHECKING, TypeVar

from terraflow.util import typing
from terraflow.discretization.fvm.scvt.trisk.common.adv import batch
from terraflow.discretization.fvm.scvt.trisk.common.adv import types, upwind

Array = typing.Array

if TYPE_CHECKING:
    from terraflow.discretization.fvm.scvt.trisk.dynamics_types import Block
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout

    CoordSysOneLayer = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        OneLayerVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        OneLayerVertCoordLayout
    ]

    CoordSysHybrid = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        HybridVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        HybridVertCoordLayout
    ]

    CoordSys = TypeVar("CoordSys", CoordSysOneLayer, CoordSysHybrid)


class HorizontalAdvectionSchemeAtCell:

    def __init__(self):
        self._built = False

    def calc_mass_hflx(
        self,
        m: Array,
        batch: batch.AdvBatchType,
        coord: CoordSys,
        aux: types.AdvectionAuxiliaryDataset,
    ):
        return self.scheme.calc_mass_hflx(
            m=m,
            batch=batch,
            coord=coord,
            aux=aux
        )

    def calc_tracer_hflx(
        self,
        q: Array,
        batch: batch.AdvBatchType,
        coord: CoordSys,
        aux: types.AdvectionAuxiliaryDataset,
    ) -> batch.AdvBatchType:
        return self.scheme.calc_tracer_hflx(
            q=q,
            batch=batch,
            coord=coord,
            aux=aux
        )

    def upwind(
        self,
        order: int = 3,
        optimized: bool = True
    ) -> upwind.UpwindHorizontalAdvectionAtCell:
        self.scheme = upwind.UpwindHorizontalAdvectionAtCell(
            order=order, optimized=optimized
        )
        self._built = True
        return self.scheme

    def prepare_block(self, block: Block) -> Block:
        return self.scheme.prepare_block(block)

    def build_default(self) -> Self:
        if not self._built:
            self.upwind()
            self._built = True
        return self


class HorizontalAdvectionSchemeAtVtx:

    def __init__(self):
        self._built = False

    def __call__(
        self,
        q: Array,
        batch: batch.AdvBatchType,
        coord: CoordSys,
        aux: types.AdvectionAuxiliaryDataset,
    ) -> batch.AdvBatchType:
        return self.scheme.calc_tracer_hflx(
            q=q,
            batch=batch,
            coord=coord,
            aux=aux
        )

    def upwind(
        self,
        order: int = 3,
        use_mbs: bool = True,
        optimized: bool = True
    ) -> upwind.UpwindHorizontalAdvectionAtVtx:
        self.scheme = upwind.UpwindHorizontalAdvectionAtVtx(
            order=order,
            use_mbs=use_mbs,
            optimized=optimized
        )
        self._built = True
        return self.scheme

    def prepare_block(self, block: Block) -> Block:
        return self.scheme.prepare_block(block)

    def build_default(self) -> Self:
        if not self._built:
            self.upwind()
            self._built = True
        return self


class VerticalAdvectionScheme:

    def __init__(self):
        self._built = False

    def calc_tracer_vflx(
        self,
        q: Array,
        batch: batch.AdvBatchType,
        coord: CoordSysHybrid,
    ) -> batch.AdvBatchType:
        return self.scheme.calc_tracer_vflx(
            q=q,
            coord=coord,
            batch=batch,
        )

    def upwind(self, order: int = 3) -> upwind.UpwindVerticalAdvection:
        self.scheme = upwind.UpwindVerticalAdvection(order=order)
        self._built = True
        return self.scheme

    def prepare_block(self, block: Block) -> Block:
        return self.scheme.prepare_block(block)

    def build_default(self) -> Self:
        if not self._built:
            self.upwind()
            self._built = True
        return self
