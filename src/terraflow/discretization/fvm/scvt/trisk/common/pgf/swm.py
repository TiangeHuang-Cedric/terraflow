from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.scvt import operators as mesh_op
from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
from terraflow.discretization.abstract import (
    AbstractEquationMode,
    DiscretizationInfo,
    AbstractDynamicalEquationOperator,
)
from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
from terraflow.dynamics.shallow_water.types import ShallowWaterEquationMode
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at

if TYPE_CHECKING:
    from terraflow.discretization.fvm.scvt.trisk.dynamics_types import Block

    Int = typing.Int
    Float = typing.Float
    Array = typing.Array

    BLOCK = Block[OneLayerVertCoordinate, OneLayerVertCoordLayout]


EM = TypeVar("EM", bound=AbstractEquationMode)

DynOperator = AbstractDynamicalEquationOperator[FiniteVolumeMethod, EM]


class PressureGradientForce(DynOperator):

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

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        coord = block.coord
        para = block.para
        ids, ide = coord.mesh.edge_ids, coord.mesh.edge_ide
        kds, kde = coord.vert.full_kds, coord.vert.full_kde
        gz = block.state.gz.unwrap()[star_idx, :, :]
        dundt_pgf = block.tend.dundt_pgf.unwrap()
        # ==========================================================
        dundt_pgf = dundt_pgf.at[kds:kde+1, ids:ide+1].set(
            - mesh_op.grad_at_dual_edge(
                x_at_primal_cell=gz,
                coord=coord,
                para=para
            )
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: b.tend.dundt_pgf,
            pytree=block,
            replace=dundt_pgf
        )

    def build_default(self) -> "PressureGradientForce":
        self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block
