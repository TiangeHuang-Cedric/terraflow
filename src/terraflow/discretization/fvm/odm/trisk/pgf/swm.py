from __future__ import annotations
from typing import TYPE_CHECKING
import equinox as eqx

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh import operator as mesh_op
from terraflow.vert_coord import HybridVertCoordinate
from terraflow.discretization.abstract import (
    DiscretizationInfo,
    AbstractEquationSetAuxiliaryTerms,
)
from terraflow.layout.abstract import CoordinateSystem
from terraflow.layout.field import ArrayField, StaggeredFieldMetaData
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout

if TYPE_CHECKING:
    from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block

Float = typing.Float

Array = typing.Array

FDT = ArrayField[StaggeredFieldMetaData]

CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    HybridVertCoordLayout
]

Term = AbstractEquationSetAuxiliaryTerms[FiniteVolumeMethod]


class PressureGradientForce(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=False,
            hydrostatic=False,
            nonhydrostatic=False,
            ideal_dry_core=False,
            advection=False
        )

    def __call__(self, block: Block) -> Block:
        coord = block.coord
        para = block.para
        gz = block.state.gz.unwrap()
        dundt_pgf = - mesh_op.grad_at_dual_edge(
            x_at_primal_cell=gz,
            coord=coord,
            para=para
        )
        return eqx.tree_at(
            lambda b: b.tend.dundt_pgf.data,
            block,
            replace=dundt_pgf
        )

    def build_default(self) -> "PressureGradientForce":
        self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return block
