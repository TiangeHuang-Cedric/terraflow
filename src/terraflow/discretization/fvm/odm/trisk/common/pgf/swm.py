from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar
import equinox as eqx

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh import operators as mesh_op
from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
from terraflow.discretization.abstract import (
    AbstractEquationMode,
    DiscretizationInfo,
    AbstractEquationOperator,
)
from terraflow.layout.abstract import CoordinateSystem
from terraflow.layout.field import ArrayField, StaggeredFieldMetaData
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
from terraflow.dynamics.shallow_water.types import ShallowWaterEquationMode

if TYPE_CHECKING:
    from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block, State

Float = typing.Float

Array = typing.Array

FDT = ArrayField[StaggeredFieldMetaData]

CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    OneLayerVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    OneLayerVertCoordLayout
]

EM = TypeVar("EM", bound=AbstractEquationMode)

Operator = AbstractEquationOperator[FiniteVolumeMethod, EM]


class PressureGradientForce(Operator):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=ShallowWaterEquationMode()
        )

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        coord = block.coord
        para = block.para
        gz = star_state.gz.unwrap()
        dundt_pgf = - mesh_op.grad_at_dual_edge(
            x_at_primal_cell=gz,
            coord=coord,
            para=para
        )
        block = eqx.tree_at(
            lambda b: b.tend.dundt_pgf.data,
            block,
            replace=dundt_pgf
        )
        return block, star_state

    def build_default(self) -> "PressureGradientForce":
        self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block
