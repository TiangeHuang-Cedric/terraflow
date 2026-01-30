from __future__ import annotations
from typing import TYPE_CHECKING
import equinox as eqx

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh import operator as op
from terraflow.vert_coord import OneLayerVertCoordinate, HybridVertCoordinate
from terraflow.layout import CoordinateSystem, Field
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.discretization.abstract import DiscretizationInfo


from terraflow.discretization.fvm.odm.trisk.single_lev.aux import AuxiliaryTerms
from terraflow.dynamics.shallow_water.types import (
    DivergenceOfMassFluxProtocol,
    NonlinearCoriolisTermProtocol,
    PressureGradientForceTermProtocol,
    GradientOfKineticEnergyProtocol,
)

if TYPE_CHECKING:
    from terraflow.discretization.fvm.odm.trisk.single_lev.dynamics_types import Block

Float = typing.Float
Array = typing.Array
CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    OneLayerVertCoordinate | HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
]


class DivergenceOfMassFluxAtPrimalCell(
    DivergenceOfMassFluxProtocol[FiniteVolumeMethod, AuxiliaryTerms]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        aux: AuxiliaryTerms
    ):
        super().__init__(disc=disc, method=method, aux=aux)

    def __call__(self, block: Block) -> Block:
        un = block.state.un.unwrap()
        z_e = block.aux.z_e.unwrap()
        res = - op.div_at_primal_cell(
            x_at_edge=z_e * un,
            coord=block.coord,
            para=block.para
        )
        return eqx.tree_at(
            lambda b: b.tend.dzdt_mfn_e.data,
            block,
            replace=res
        )

    def unwrap(self, block: Block) -> Field:
        return block.tend.dzdt_mfn_e


class NonlinearCoriolisTermAtDualEdge(
    NonlinearCoriolisTermProtocol[FiniteVolumeMethod, AuxiliaryTerms]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        aux: AuxiliaryTerms
    ):
        super().__init__(disc=disc, method=method, aux=aux)

    def __call__(self, block: Block) -> Block:
        return self.aux.coriolis(block)

    def unwrap(self, block: Block) -> Field:
        return block.tend.dundt_coriolis


class PressureGradientForceAtDualEdge(
    PressureGradientForceTermProtocol[FiniteVolumeMethod, AuxiliaryTerms]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        aux: AuxiliaryTerms
    ):
        super().__init__(disc=disc, method=method, aux=aux)

    def __call__(self, block: Block) -> Block:
        coord = block.coord
        para = block.para
        z = block.state.z.unwrap()
        zs = block.static.zs.unwrap()
        gravity = para.gravity.unwrap()
        gz = gravity * (z + zs)
        dundt_pgf = - op.grad_at_dual_edge(
            x_at_primal_cell=gz,
            coord=coord,
            para=para
        )
        return eqx.tree_at(
            lambda b: b.tend.dundt_pgf.data,
            block,
            replace=dundt_pgf
        )

    def unwrap(self, block: Block) -> Field:
        return block.tend.dundt_pgf


class GradientOfKineticEnergyAtDualEdge(
    GradientOfKineticEnergyProtocol[FiniteVolumeMethod, AuxiliaryTerms]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        aux: AuxiliaryTerms
    ):
        super().__init__(disc=disc, method=method, aux=aux)

    def __call__(self, block: Block) -> Block:
        ke = self.aux.ke(block).aux.ke.unwrap()
        res = - op.grad_at_dual_edge(
            x_at_primal_cell=ke,
            coord=block.coord,
            para=block.para
        )
        return eqx.tree_at(
            lambda b: b.tend.dundt_grad_ke.data,
            block,
            replace=res
        )

    def unwrap(self, block: Block) -> Field:
        return block.tend.dundt_grad_ke
