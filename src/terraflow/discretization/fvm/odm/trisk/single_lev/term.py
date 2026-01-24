import equinox as eqx

from terraflow.util import typing
from terraflow.planet import PlanetParameterSet
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh import operator as op
from terraflow.vert_coord import OneLayerVertCoordinate, HybridVertCoordinate
from terraflow.layout import CoordinateSystem
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.discretization.abstract import DiscretizationInfo
from terraflow.discretization.abstract import AbstractEquationTerm
from terraflow.discretization.fvm.odm.trisk import coriolis, base
from terraflow.discretization.fvm.odm.trisk.single_lev.dynamics_types import BlockArrays
from terraflow.discretization.fvm.odm.trisk.single_lev.aux import AuxiliaryTerms

from terraflow.dynamics.shallow_water.types import (
    DivergenceOfMassFluxProtocol,
    NonlinearCoriolisTermProtocol,
    PressureGradientForceTermProtocol,
    GradientOfKineticEnergyProtocol,
)

Float = typing.Float
Array = typing.Array
CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    OneLayerVertCoordinate | HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
]

Term = AbstractEquationTerm[
    OrthogonalDualMesh,
    OneLayerVertCoordinate | HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    FiniteVolumeMethod,
    AuxiliaryTerms,
    BlockArrays
]


class DivergenceOfMassFluxAtPrimalCell(DivergenceOfMassFluxProtocol, Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod,
        aux: AuxiliaryTerms
    ):
        super().__init__(disc, coord, para, method, aux)
        self.div_at_primal_cell = op.DivAtPrimalCell(
            coord, para
        )

    def __call__(self, block: BlockArrays) -> BlockArrays:
        un_e = block.state.un_e
        h_e = block.aux.h_e
        res = - self.div_at_primal_cell(h_e * un_e)
        return eqx.tree_at(
            lambda b: b.tend.dhdt_mfx,
            block,
            replace=res
        )

    def unwrap(self, block: BlockArrays) -> Array:
        return block.tend.dhdt_mfx

    def build_default(self) -> "DivergenceOfMassFluxAtPrimalCell":
        if not self.has_built():
            self._built = True
        return self


class NonlinearCoriolisTermAtDualEdge(NonlinearCoriolisTermProtocol, Term):
    scheme: coriolis.CoriolisForceAtDualEdge

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod,
        aux: AuxiliaryTerms
    ):
        super().__init__(disc, coord, para, method, aux)
        self.scheme = coriolis.CoriolisForceAtDualEdge(
            coord, para
        )

    def __call__(self, block: BlockArrays) -> BlockArrays:
        return self.scheme(block)

    def unwrap(self, block: BlockArrays) -> Array:
        return block.tend.dundt_coriolis

    def build_default(self) -> "NonlinearCoriolisTermAtDualEdge":
        if not self.has_built():
            self._built = True
        return self


class PressureGradientForceAtDualEdge(PressureGradientForceTermProtocol, Term):
    grad_at_dual_edge: op.GradAtDualEdge

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod,
        aux: AuxiliaryTerms
    ):
        super().__init__(disc, coord, para, method, aux)
        self.grad_at_dual_edge = op.GradAtDualEdge(
            coord, para
        )

    def __call__(self, block: BlockArrays) -> BlockArrays:
        h_pc = block.state.h_pc
        topo = block.static.topo
        gravity = self.para.gravity.unwrap()
        gz = gravity * (h_pc + topo)
        res = - self.grad_at_dual_edge(gz)
        return eqx.tree_at(
            lambda b: b.tend.dundt_pgf,
            block,
            replace=res
        )

    def unwrap(self, block: BlockArrays) -> Array:
        return block.tend.dundt_pgf

    def build_default(self) -> "PressureGradientForceAtDualEdge":
        if not self.has_built():
            self._built = True
        return self


class GradientOfKineticEnergyAtDualEdge(GradientOfKineticEnergyProtocol, Term):
    ke_at_primal_cell: base.KineticEnergyAtPrimalCell

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod,
        aux: AuxiliaryTerms
    ):
        super().__init__(disc, coord, para, method, aux)
        self.ke_at_primal_cell = base.KineticEnergyAtPrimalCell(
            disc, coord, para, method
        )
        self.grad_at_dual_edge = op.GradAtDualEdge(
            coord, para
        )

    def __call__(self, block: BlockArrays) -> BlockArrays:
        ke_pc = self.ke_at_primal_cell(block).aux.ke_pc
        res = - self.grad_at_dual_edge(ke_pc)
        return eqx.tree_at(
            lambda b: b.tend.dundt_grad_ke,
            block,
            replace=res
        )

    def unwrap(self, block: BlockArrays) -> Array:
        return block.tend.dundt_grad_ke

    def build_default(self) -> "GradientOfKineticEnergyAtDualEdge":
        if not self.has_built():
            self.ke_at_primal_cell.build_default()
            self._built = True
        return self

        block = self.ut_e.recon_tangent_velocity(block)
        block = self.h_e(block)
        block = self.mfx_e(block)
        block = self.h_dc(block)
        block = self.rel_vor_dc(block)
        block = self.abs_vor_dc(block)
        block = self.pv_dc(block)
        block = self.pv_e(block)
        block = self.ke_pc(block)
        return block
