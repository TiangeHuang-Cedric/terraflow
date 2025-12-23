from terraflow.util import typing
from terraflow.planet import PlanetParameterSet
from terraflow.mesh import OrthogonalDualMesh
from terraflow.vert_coordinates import OneLayerVertCoordinate
from terraflow.layout import CoordinateSystem
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.operator import FiniteVolumeMethodOperator
from terraflow.operator.fvm.odm import base
from terraflow.discretization.abstract import DiscretizationInfo
import terraflow.discretization.term.fvm.odm.base as tm
from terraflow.discretization.equation.shallow_water.vars.unstructure import StateArrays
from terraflow.discretization.equation.shallow_water.vars.unstructure import StaticArrays

from terraflow.discretization.equation.shallow_water.base import (
    DivergenceOfThicknessTermProtocol,
    NonlinearCoriolisTermProtocol,
    GradientOfGeopotentialTermProtocol,
    GradientOfKineticEnergyTermProtocol,
)


Array = typing.Array
CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    OneLayerVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
]


class DivergenceOfThicknessFluxAtPrimalCell(tm.Term, DivergenceOfThicknessTermProtocol):
    flux_recon_at_primal_edge: tm.FluxReconstructAtPrimalEdge

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        operator: FiniteVolumeMethodOperator
    ):
        super().__init__(disc, coord, para, operator)
        self.flux_recon_at_primal_edge = tm.FluxReconstructAtPrimalEdge(
            disc, coord, para, operator
        )

    def __call__(self, dstate: StateArrays) -> Array:
        thickness_at_primal_cell = dstate.h_pc
        normal_velocity_at_edge = dstate.un_e
        thickness_at_edge = self.flux_recon_at_primal_edge(
            thickness_at_primal_cell
        )
        return base.div_at_primal_cell(
            thickness_at_edge * normal_velocity_at_edge,
            self.para.radius.unwrap(),
            self.mesh._plg.ed,
            self.mesh._plg.nr,
            self.mesh._plg.area,
            self.mesh._edp.length
        )

    def build_default(self) -> "DivergenceOfThicknessFluxAtPrimalCell":
        if not self.has_built():
            self.flux_recon_at_primal_edge = self.flux_recon_at_primal_edge.build_default()
            self._built = True
        return self


class NonlinearCoriolisTermAtDualEdge(tm.Term, NonlinearCoriolisTermProtocol):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        operator: FiniteVolumeMethodOperator
    ):
        super().__init__(disc, coord, para, operator)

    def __call__(
        self,
        dstate: StateArrays,
        recon_tangent_velocity: bool = False
    ) -> Array:
        normal_velocity_at_edge = dstate.un_e
        thickness_at_primal_cell = dstate.h_pc
        return self._scheme(
            normal_velocity_at_edge,
            thickness_at_primal_cell,
            recon_tangent_velocity=recon_tangent_velocity
        )

    def build_with_trisk(self) -> tm.TRiSKSchemeBasedNonlinearCoriolisTermAtDualEdge:
        self._scheme_name = "trisk"
        self._scheme: tm.TRiSKSchemeBasedNonlinearCoriolisTermAtDualEdge = (
            tm.TRiSKSchemeBasedNonlinearCoriolisTermAtDualEdge(
                self._disc, self.coord, self.para, self.operator
            )
        )
        return self._scheme

    def build_default(self) -> "NonlinearCoriolisTermAtDualEdge":
        if not self.has_built():
            self._scheme = self.build_with_trisk().build_default()
            self._built = True
        return self


class GradientOfGeopotentialAtDualEdge(tm.Term, GradientOfGeopotentialTermProtocol):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        operator: FiniteVolumeMethodOperator
    ):
        super().__init__(disc, coord, para, operator)

    def __call__(self, dstate: StateArrays, static: StaticArrays) -> Array:
        thickness_at_primal_cell = dstate.h_pc
        topo = static.topo
        gravity = self.para.gravity.unwrap()
        gz = gravity * (thickness_at_primal_cell + topo)
        return base.grad_at_edge(
            gz,
            self.para.radius.unwrap(),
            self.mesh._edd.v,
            self.mesh._edd.length,
            self.mesh._aux.edpnr_eddtg
        )

    def build_default(self) -> "GradientOfGeopotentialAtDualEdge":
        if not self.has_built():
            self._built = True
        return self


class GradientOfKineticEnergyAtDualEdge(tm.Term, GradientOfKineticEnergyTermProtocol):
    kinetic_energy_at_primal_cell: tm.KineticEnergyAtPrimalCell

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        operator: FiniteVolumeMethodOperator
    ):
        super().__init__(disc, coord, para, operator)
        self.kinetic_energy_at_primal_cell = tm.KineticEnergyAtPrimalCell(
            disc, coord, para, operator
        )

    def __call__(self, dstate: StateArrays) -> Array:
        normal_velocity_at_edge = dstate.un_e
        kinetic_energy_at_primal_cell = self.kinetic_energy_at_primal_cell(
            normal_velocity_at_edge
        )
        return base.grad_at_edge(
            kinetic_energy_at_primal_cell,
            self.para.radius.unwrap(),
            self.mesh._edd.v,
            self.mesh._edd.length,
            self.mesh._aux.edpnr_eddtg
        )

    def build_default(self) -> "GradientOfKineticEnergyAtDualEdge":
        if not self.has_built():
            self.kinetic_energy_at_primal_cell = self.kinetic_energy_at_primal_cell.build_default()
            self._built = True
        return self
