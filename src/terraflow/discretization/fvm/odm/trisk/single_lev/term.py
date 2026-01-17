from terraflow.util import typing
from terraflow.planet import PlanetParameterSet
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh import operator as op
from terraflow.vert_coord import OneLayerVertCoordinate
from terraflow.layout import CoordinateSystem
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.discretization.abstract import DiscretizationInfo
from terraflow.discretization.fvm.odm.trisk import coriolis, base as trisk
from terraflow.discretization.fvm.odm.trisk.single_lev.vars import StateArrays
from terraflow.discretization.fvm.odm.trisk.single_lev.vars import StaticArrays

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
    OneLayerVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
]


class DivergenceOfMassFluxAtPrimalCell(trisk.Term, DivergenceOfMassFluxProtocol):
    flux_recon_at_primal_edge: trisk.FluxReconstructAtPrimalEdge

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)
        self.flux_recon_at_primal_edge = trisk.FluxReconstructAtPrimalEdge(
            disc, coord, para, method
        )
        self.div_at_primal_cell = op.DivAtPrimalCell(
            coord, para
        )

    def __call__(self, state: StateArrays) -> Float[Array, "nPrimal_cell"]:
        thickness_at_primal_cell = state.h_pc
        normal_u_at_edge = state.un_e
        thickness_at_edge = self.flux_recon_at_primal_edge(
            thickness_at_primal_cell, normal_u_at_edge
        )
        return self.div_at_primal_cell(
            thickness_at_edge * normal_u_at_edge,
        )

    def build_default(self) -> "DivergenceOfMassFluxAtPrimalCell":
        if not self.has_built():
            self.flux_recon_at_primal_edge.build_default()
            self._built = True
        return self


class NonlinearCoriolisTermAtDualEdge(trisk.Term, NonlinearCoriolisTermProtocol):
    scheme: coriolis.CoriolisForceAtDualEdge
    flux_recon_at_primal_edge: trisk.FluxReconstructAtPrimalEdge
    pv_at_dual_edge: trisk.PotentialVorticityAtDualEdge

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)
        self.scheme = coriolis.CoriolisForceAtDualEdge(
            coord, para
        )
        self.flux_recon_at_primal_edge = trisk.FluxReconstructAtPrimalEdge(
            disc, coord, para, method
        )
        self.pv_at_dual_edge = trisk.PotentialVorticityAtDualEdge(
            disc, coord, para, method
        )

    def __call__(
        self,
        state: StateArrays,
        recon_tangent_velocity: bool = False
    ) -> Float[Array, "nEdge"]:
        normal_u_at_edge = state.un_e
        thickness_at_primal_cell = state.h_pc
        if recon_tangent_velocity:
            return self.scheme.recon_tangent_velocity(normal_u_at_edge)
        thickness_at_primal_edge = self.flux_recon_at_primal_edge(
            x_at_primal_cell=thickness_at_primal_cell,
            normal_u_at_edge=normal_u_at_edge
        )
        tangent_u_at_edge = self.scheme.recon_tangent_velocity(
            normal_u_at_edge=normal_u_at_edge
        )
        pv_at_dual_edge = self.pv_at_dual_edge(
            normal_u_at_edge=normal_u_at_edge,
            tangent_u_at_edge=tangent_u_at_edge,
            thickness_at_primal_cell=thickness_at_primal_cell
        )
        return self.scheme(
            normal_u_at_edge=normal_u_at_edge,
            thickness_at_primal_edge=thickness_at_primal_edge,
            pv_at_dual_edge=pv_at_dual_edge
        )

    def build_default(self) -> "NonlinearCoriolisTermAtDualEdge":
        if not self.has_built():
            self.flux_recon_at_primal_edge.build_default()
            self.pv_at_dual_edge.build_default()
            self._built = True
        return self


class PressureGradientForceAtDualEdge(trisk.Term, PressureGradientForceTermProtocol):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)
        self.grad_at_dual_edge = op.GradAtDualEdge(
            coord, para
        )

    def __call__(self, state: StateArrays, static: StaticArrays) -> Float[Array, "nDual_edge"]:
        thickness_at_primal_cell = state.h_pc
        topo = static.topo
        gravity = self.para.gravity.unwrap()
        gz = gravity * (thickness_at_primal_cell + topo)
        return self.grad_at_dual_edge(gz)

    def build_default(self) -> "PressureGradientForceAtDualEdge":
        if not self.has_built():
            self._built = True
        return self


class GradientOfKineticEnergyAtDualEdge(trisk.Term, GradientOfKineticEnergyProtocol):
    ke_at_primal_cell: trisk.KineticEnergyAtPrimalCell

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)
        self.ke_at_primal_cell = trisk.KineticEnergyAtPrimalCell(
            disc, coord, para, method
        )
        self.grad_at_dual_edge = op.GradAtDualEdge(
            coord, para
        )

    def __call__(self, state: StateArrays) -> Float[Array, "nDual_edge"]:
        normal_velocity_at_edge = state.un_e
        ke_at_primal_cell = self.ke_at_primal_cell(
            normal_velocity_at_edge
        )
        return self.grad_at_dual_edge(ke_at_primal_cell)

    def build_default(self) -> "GradientOfKineticEnergyAtDualEdge":
        if not self.has_built():
            self.ke_at_primal_cell.build_default()
            self._built = True
        return self
