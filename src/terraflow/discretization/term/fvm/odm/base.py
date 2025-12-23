from typing import TypeVar, Self, Protocol
from abc import ABC
import jax.numpy as jnp

from terraflow.util import typing
from terraflow.mesh import OrthogonalDualMesh
from terraflow.vert_coordinates import OneLayerVertCoordinate
from terraflow.layout import CoordinateSystem
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.operator import FiniteVolumeMethodOperator
from terraflow.operator.fvm.odm import base, flux_reconstruct as fr, interp
from terraflow.discretization.abstract import DiscretizationInfo
from terraflow.discretization.abstract import AbstractEquationTerm
from terraflow.planet import PlanetParameterSet

T = TypeVar("T")

IntType = typing.IntType
FloatType = typing.FloatType
Int = typing.Int
Float = typing.Float
Array = typing.Array


class ShallowWaterEquationTerm(
    AbstractEquationTerm[
        OrthogonalDualMesh,
        OneLayerVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        FiniteVolumeMethodOperator
    ],
    ABC
):
    pass


Term = ShallowWaterEquationTerm

CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    OneLayerVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
]


class CentralDifferenceReconstructAtDualEdge(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        operator: FiniteVolumeMethodOperator
    ):
        self._order: int = 2
        super().__init__(disc, coord, para, operator)

    def __call__(
        self,
        x_at_dual_cell: Float[Array, "nDual_cell"]
    ) -> Float[Array, "nEdge"]:
        return fr.flux_reconstruct_central_difference_2nd_order(
            x_at_dual_cell,
            self.mesh._edp.v
        )

    def build_default(self) -> Self:
        self._inited = True
        return self


class FluxReconstructAtDualEdge(Term):

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
        x_at_dual_cell: Float[Array, "nDual_cell"]
    ) -> Float[Array, "nEdge"]:
        return self._scheme(x_at_dual_cell)

    def with_central_difference(self) -> CentralDifferenceReconstructAtDualEdge:
        self._scheme_name = "central_difference"
        self._scheme = CentralDifferenceReconstructAtDualEdge(
            self._disc, self.coord, self.para, self.operator
        )
        self.order = self._scheme._order
        self._inited = True
        return self._scheme

    def build_default(self) -> "FluxReconstructAtDualEdge":
        if not self.has_built():
            self.with_central_difference()
        return self


class CentralDifferenceReconstructAtPrimalEdge(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        operator: FiniteVolumeMethodOperator
    ):
        self._order: int = 2
        super().__init__(disc, coord, para, operator)

    def __call__(
        self,
        x_at_primal_cell: Float[Array, "nPrimal_cell"]
    ) -> Float[Array, "nEdge"]:
        return fr.flux_reconstruct_central_difference_2nd_order(
            x_at_primal_cell,
            self.mesh._edd.v
        )

    def build_default(self) -> Self:
        self._inited = True
        return self


class FluxReconstructAtPrimalEdge(Term):

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
        x_at_primal_cell: Float[Array, "nPrimal_cell"]
    ) -> Float[Array, "nEdge"]:
        return self._scheme(x_at_primal_cell)

    def build_with_central_difference(self) -> CentralDifferenceReconstructAtPrimalEdge:
        self._scheme_name = "central_difference"
        self._scheme = CentralDifferenceReconstructAtPrimalEdge(
            self._disc, self.coord, self.para, self.operator
        )
        self.order = self._scheme._order
        self._inited = True
        return self._scheme

    def build_default(self) -> "FluxReconstructAtPrimalEdge":
        if not self.has_built():
            self.build_with_central_difference()
        return self


class RelativeVorticityAtDualCell(Term):

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
        normal_velocity_at_edge: Float[Array, "nEdge"]
    ) -> Float[Array, "nDual_cell"]:
        return base.curl_at_dual_cell(
            normal_velocity_at_edge,
            self.para.radius.unwrap(),
            self.mesh._dua.ed,
            self.mesh._dua.nr,
            self.mesh._dua.area,
            self.mesh._edd.length,
            self.mesh._aux.edptg_eddnr
        )

    def build_default(self) -> Self:
        self._inited = True
        return self


class AbsoluteVorticityAtDualCell(Term):
    relative_vorticity: RelativeVorticityAtDualCell

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        operator: FiniteVolumeMethodOperator
    ):
        super().__init__(disc, coord, para, operator)
        self.relative_vorticity = RelativeVorticityAtDualCell(
            disc, coord, para, operator
        )

    def __call__(
        self,
        normal_velocity_at_edge: Float[Array, "nEdge"]
    ) -> Float[Array, "nDual_cell"]:
        rel_vor = self.relative_vorticity(normal_velocity_at_edge)
        omega = self.para.omega.unwrap()
        coriolis = 2 * omega * jnp.sin(self.mesh._dua.centroid.lat)
        return coriolis + rel_vor

    def build_default(self) -> Self:
        self._inited = True
        return self


class ThicknessAtDualCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        operator: FiniteVolumeMethodOperator
    ):
        super().__init__(disc, coord, para, operator)

    def __call__(
        self, thickness_at_primal_cell: Float[Array, "nPrimal_cell"]
    ) -> Float[Array, "nDual_cell"]:
        return interp.interp_weighted_add_plg2dua(
            thickness_at_primal_cell,
            self.mesh.dual.cell.nodes,
            self.mesh.dual.cell.kite_area,
            self.mesh.dual.cell.area
        )

    def build_default(self) -> Self:
        self._inited = True
        return self


class PotentialVorticityAtDualEdge(Term):

    class _PVAtDualCell(Protocol):
        def __call__(
            self,
            abs_vor_at_dual_cell: Array,
            thickness_at_dual_cell: Array
        ) -> Array:
            ...

    abs_vor_at_dual_cell: AbsoluteVorticityAtDualCell
    thickness_at_dual_cell: ThicknessAtDualCell
    pv_at_dual_cell: _PVAtDualCell
    flux_recon_at_dual_edge: FluxReconstructAtDualEdge

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        operator: FiniteVolumeMethodOperator
    ):
        self.abs_vor_at_dual_cell = AbsoluteVorticityAtDualCell(
            disc, coord, para, operator
        )
        self.flux_recon_at_dual_edge = FluxReconstructAtDualEdge(
            disc, coord, para, operator
        )
        self.thickness_at_dual_cell = ThicknessAtDualCell(
            disc, coord, para, operator
        )

        def _pv_at_dual_cell(
            abs_vor_at_dual_cell: Float[Array, "nDual_cell"],
            thickness_at_dual_cell: Float[Array, "nDual_cell"]
        ) -> Float[Array, "nDual_cell"]:
            return abs_vor_at_dual_cell / thickness_at_dual_cell

        self.pv_at_dual_cell = _pv_at_dual_cell
        super().__init__(disc, coord, para, operator)

    def __call__(
        self,
        normal_velocity_at_edge: Float[Array, "nEdge"],
        thickness_at_primal_cell: Float[Array, "nPrimal_cell"]
    ) -> Float[Array, "nEdge"]:
        abs_vor_at_dual_cell = self.abs_vor_at_dual_cell(
            normal_velocity_at_edge
        )
        thickness_at_dual_cell = self.thickness_at_dual_cell(
            thickness_at_primal_cell,
        )
        pv_at_dual_cell = self.pv_at_dual_cell(
            abs_vor_at_dual_cell, thickness_at_dual_cell
        )
        return self.flux_recon_at_dual_edge(pv_at_dual_cell)

    def build_default(self) -> "PotentialVorticityAtDualEdge":
        if not self.has_built():
            builted = self.flux_recon_at_dual_edge.build_default()
            self.flux_recon_at_dual_edge = builted
        return self


class TRiSKSchemeBasedNonlinearCoriolisTermAtDualEdge(Term):
    edp_numb_on_edge: Array
    edp_edge_on_edge: Array
    edp_trsk_on_edge: Array
    edp_edpl_on_edge: Array
    max_number_on_edge: IntType
    flux_recon_at_primal_edge: FluxReconstructAtPrimalEdge
    pv_at_dual_edge: PotentialVorticityAtDualEdge

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        operator: FiniteVolumeMethodOperator
    ):
        super().__init__(disc, coord, para, operator)
        collect = interp.trisk_weighted_interpolation_operator(coord.mesh)
        numb, edge, trsk, edpl = collect
        self.edp_numb_on_edge = numb
        self.edp_edge_on_edge = edge
        self.edp_trsk_on_edge = trsk
        self.edp_edpl_on_edge = edpl
        self.max_number_on_edge = int(jnp.max(numb))
        self.flux_recon_at_primal_edge = FluxReconstructAtPrimalEdge(
            disc, coord, para, operator
        )
        self.pv_at_dual_edge = PotentialVorticityAtDualEdge(
            disc, coord, para, operator
        )

    def __call__(
        self,
        normal_velocity_at_edge: Float[Array, "nEdge"],
        thickness_at_primal_cell: Float[Array, "nPrimal_cell"],
        recon_tangent_velocity: bool = False
    ) -> Float[Array, "nEdge"]:
        if recon_tangent_velocity:
            ones_tile = jnp.ones_like(self.edp_edge_on_edge)
            return interp.interp_weighted_add_pair_meeting_edges2edge(
                normal_velocity_at_edge,
                self.edp_edge_on_edge,
                self.edp_trsk_on_edge,
                self.edp_edpl_on_edge,
                self.mesh._edd.length,
                aux_tile=ones_tile
            )
        thickness_at_primal_edge = self.flux_recon_at_primal_edge(
            thickness_at_primal_cell
        )
        pv_eep = self.energy_conserving_coriolis_term(
            normal_velocity_at_edge, thickness_at_primal_cell
        )
        thickness_flux_at_primal_edge = normal_velocity_at_edge * thickness_at_primal_edge
        return interp.interp_weighted_add_pair_meeting_edges2edge(
            thickness_flux_at_primal_edge,  # thickness flux, not just normal wind
            self.edp_edge_on_edge,
            self.edp_trsk_on_edge,
            self.edp_edpl_on_edge,
            self.mesh._edd.length,
            aux_tile=pv_eep
        )

    def build_default(self) -> "TRiSKSchemeBasedNonlinearCoriolisTermAtDualEdge":
        if not self.has_built():
            self.flux_recon_at_primal_edge = self.flux_recon_at_primal_edge.build_default()
            self.pv_at_dual_edge = self.pv_at_dual_edge.build_default()
        return self

    def energy_conserving_coriolis_term(
        self,
        normal_velocity_at_edge: Float[Array, "nEdge"],
        thickness_at_primal_cell: Float[Array, "nPrimal_cell"]
    ) -> Float[Array, "nEdge maxnumb*2"]:
        pv_at_dual_edge = self.pv_at_dual_edge(
            normal_velocity_at_edge, thickness_at_primal_cell
        )  # (nedge, )
        pv_iee = pv_at_dual_edge
        pv_iep = jnp.take(
            pv_at_dual_edge,
            self.edp_edge_on_edge,
            axis=0
        )  # (nedge, max_numb*2)
        return (pv_iee[:, None] + pv_iep) / 2  # (nedge, max_numb*2)


class Ringler2010KineticEnergyAtPrimalCell(Term):

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
        normal_velocity_at_edge: Float[Array, "nEdge"]
    ) -> Float[Array, "nPrimal_cell"]:
        return interp.interp_weighted_add_edp2plg(
            normal_velocity_at_edge ** 2,
            self.mesh._plg.ed,
            self.mesh._plg.area,
            self.mesh._edp.length,
            self.mesh._edd.length
        )

    def build_default(self) -> Self:
        self._inited = True
        return self


class Gassmann2013KineticEnergyAtPrimalCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        operator: FiniteVolumeMethodOperator
    ):
        super().__init__(disc, coord, para, operator)
        self._ringler_2010 = Ringler2010KineticEnergyAtPrimalCell(
            disc, coord, para, operator
        )
        self._gamma: FloatType = 0.9

    def __call__(
        self,
        normal_velocity_at_edge: Float[Array, "nEdge"]
    ) -> Float[Array, "nPrimal_cell"]:
        ke_edge = normal_velocity_at_edge ** 2
        ke_dc = interp.interp_weighted_add_edd2dua(
            ke_edge,
            self.mesh._dua.ed,
            self.mesh._dua.area,
            self.mesh._edd.length,
            self.mesh._edp.length
        )
        ke_pc = self._ringler_2010(normal_velocity_at_edge)
        ke_dc_pc = interp.interp_weighted_add_dua2plg(
            ke_dc,
            self.mesh._plg.v,
            self.mesh._plg.kite_area,
            self.mesh._plg.area
        )
        return 1 * self.gamma * ke_pc + (1 - self.gamma) * ke_dc_pc

    @property
    def gamma(self) -> FloatType:
        return self._gamma

    def build_default(self) -> Self:
        self._inited = True
        return self


class KineticEnergyAtPrimalCell(Term):

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
        normal_velocity_at_edge: Float[Array, "nEdge"]
    ) -> Float[Array, "nPrimal_cell"]:
        return self._scheme(normal_velocity_at_edge)

    def build_with_gassmann_2013(
        self,
        gamma: FloatType = 0.9
    ) -> Gassmann2013KineticEnergyAtPrimalCell:
        self._scheme_name = "gassmann_2013"
        self._scheme = Gassmann2013KineticEnergyAtPrimalCell(
            self._disc, self.coord, self.para, self.operator
        )
        self._scheme._gamma = gamma
        self._inited = True
        return self._scheme

    def build_with_ringler_2010(self) -> Ringler2010KineticEnergyAtPrimalCell:
        self._scheme_name = "ringler_2010"
        self._scheme = Ringler2010KineticEnergyAtPrimalCell(
            self._disc, self.coord, self.para, self.operator
        )
        self._inited = True
        return self._scheme

    def build_default(self) -> "KineticEnergyAtPrimalCell":
        if not self.has_built():
            self.build_with_gassmann_2013()
        return self
