from __future__ import annotations
from typing import Self, TYPE_CHECKING

from terraflow.util import typing
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.layout.abstract import CoordinateSystem
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
from terraflow.discretization.fvm.odm.trisk.common.adv import batch
from terraflow.discretization.fvm.odm.trisk.common.adv import types, upwind

Array = typing.Array

if TYPE_CHECKING:
    from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block


CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    HybridVertCoordLayout
]


class HorizontalAdvectionSchemeAtCell:

    def __init__(self):
        self._built = False

    def __call__(
        self,
        q: Array,
        batch: batch.AdvBatchType,
        coord: CoordSys,
        aux: types.AdvectionAuxiliaryDataset,
    ) -> batch.AdvBatchType:
        return self.scheme(
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
        return self.scheme(
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

    def __call__(
        self,
        q: Array,
        batch: batch.AdvBatchType,
    ) -> batch.AdvBatchType:
        return self.scheme(
            q=q,
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
