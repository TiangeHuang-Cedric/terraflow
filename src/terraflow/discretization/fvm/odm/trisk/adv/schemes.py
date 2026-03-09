from __future__ import annotations
from typing import Self, TYPE_CHECKING

from terraflow.util import typing
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.layout.abstract import CoordinateSystem
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
from terraflow.operator import FiniteVolumeMethod
from terraflow.discretization.fvm.odm.trisk.adv import upwind, batch, types
from terraflow.discretization.abstract import AbstractEquationSetAuxiliaryTerms

Array = typing.Array

if TYPE_CHECKING:
    from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block


CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    HybridVertCoordLayout
]


Term = AbstractEquationSetAuxiliaryTerms[FiniteVolumeMethod]


class HorizontalAdvectionScheme:

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

    def upwind(self, order: int = 3) -> upwind.UpwindHorizontalAdvection:
        self.scheme = upwind.UpwindHorizontalAdvection(order=order)
        self._built = True
        return self.scheme

    def prepare(self, block: Block) -> Block:
        return self.scheme.prepare(block)

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

    def prepare(self, block: Block) -> Block:
        return self.scheme.prepare(block)

    def build_default(self) -> Self:
        if not self._built:
            self.upwind()
            self._built = True
        return self
