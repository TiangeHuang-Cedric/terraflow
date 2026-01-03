from ast import In
from typing import TypeVar, Self, Protocol
from abc import ABC
import jax.numpy as jnp

from terraflow.util import typing
from terraflow.mesh import OrthogonalDualMesh
from terraflow.vert_coord import OneLayerVertCoordinate
from terraflow.layout import CoordinateSystem
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.discretization.operator import FiniteVolumeMethod
from terraflow.discretization.operator.fvm.odm.trisk.single_lev import (
    flux_reconstruct as fr,
    interp
)
from terraflow.discretization.operator.fvm.odm.trisk.single_lev import base
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
        FiniteVolumeMethod
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


class FluxReconstructAtPrimalEdge(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)

    def __call__(
        self,
        x_at_primal_cell: Float[Array, "nPrimal_cell"],
        normal_u_at_edge: Float[Array, "nEdge"]
    ) -> Float[Array, "nEdge"]:
        return self.scheme(x_at_primal_cell, normal_u_at_edge)

    def central_difference(self) -> fr.CentralDifference2AtPrimalEdge:
        self.scheme_name = "central_difference"
        self.scheme = fr.CentralDifference2AtPrimalEdge(
            self.coord, self.para
        )
        self._built = True
        return self.scheme

    def wrf3(self) -> fr.WRF3AtPrimalEdge:
        self.scheme_name = "wrf3"
        self.scheme = fr.WRF3AtPrimalEdge(
            self.coord, self.para
        )
        self._built = True
        return self.scheme

    def wrf4(self) -> fr.WRF4AtPrimalEdge:
        self.scheme_name = "wrf4"
        self.scheme = fr.WRF4AtPrimalEdge(
            self.coord, self.para
        )
        self._built = True
        return self.scheme

    def hybrid_wrf34(self, beta: FloatType = 0.25) -> fr.HybridWRF34AtPrimalEdge:
        self.scheme_name = "hybrid_wrf34"
        self.scheme = fr.HybridWRF34AtPrimalEdge(
            self.coord, self.para, beta
        )
        self._built = True
        return self.scheme

    def wrf5(self) -> fr.WRF5AtPrimalEdge:
        self.scheme_name = "wrf5"
        self.scheme = fr.WRF5AtPrimalEdge(
            self.coord, self.para
        )
        self._built = True
        return self.scheme

    def wrf6(self) -> fr.WRF6AtPrimalEdge:
        self.scheme_name = "wrf6"
        self.scheme = fr.WRF6AtPrimalEdge(
            self.coord, self.para
        )
        self._built = True
        return self.scheme

    def hybrid_wrf56(self, beta: FloatType = 0.25) -> fr.HybridWRF56AtPrimalEdge:
        self.scheme_name = "hybrid_wrf56"
        self.scheme = fr.HybridWRF56AtPrimalEdge(
            self.coord, self.para, beta
        )
        self._built = True
        return self.scheme

    def build_default(self) -> "FluxReconstructAtPrimalEdge":
        if not self.has_built():
            self.central_difference()
        return self


class FluxReconstructAtDualEdge(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)

    def __call__(
        self,
        x_at_dual_cell: Float[Array, "nDual_cell"],
        tangent_u_at_edge: Float[Array, "nEdge"]
    ) -> Float[Array, "nEdge"]:
        return self.scheme(x_at_dual_cell, tangent_u_at_edge)

    def central_difference(self) -> fr.CentralDifference2AtDualEdge:
        self.scheme_name = "central_difference"
        self.scheme = fr.CentralDifference2AtDualEdge(
            self.coord, self.para
        )
        self._built = True
        return self.scheme

    def wrf3(self) -> fr.WRF3AtDualEdge:
        self.scheme_name = "wrf3"
        self.scheme = fr.WRF3AtDualEdge(
            self.coord, self.para
        )
        self._built = True
        return self.scheme

    def wrf4(self) -> fr.WRF4AtDualEdge:
        self.scheme_name = "wrf4"
        self.scheme = fr.WRF4AtDualEdge(
            self.coord, self.para
        )
        self._built = True
        return self.scheme

    def hybrid_wrf34(self, beta: FloatType = 0.25) -> fr.HybridWRF34AtDualEdge:
        self.scheme_name = "hybrid_wrf34"
        self.scheme = fr.HybridWRF34AtDualEdge(
            self.coord, self.para, beta
        )
        self._built = True
        return self.scheme

    def wrf5(self, use_mbs: bool = False) -> fr.WRF5AtDualEdge:
        self.scheme_name = "wrf5"
        self.scheme = fr.WRF5AtDualEdge(
            self.coord, self.para, use_mbs=use_mbs
        )
        self._built = True
        return self.scheme

    def wrf6(self, use_mbs: bool = False) -> fr.WRF6AtDualEdge:
        self.scheme_name = "wrf6"
        self.scheme = fr.WRF6AtDualEdge(
            self.coord, self.para, use_mbs=use_mbs
        )
        self._built = True
        return self.scheme

    def hybrid_wrf56(
        self,
        beta: FloatType = 0.25,
        use_mbs: bool = False
    ) -> fr.HybridWRF56AtDualEdge:
        self.scheme_name = "hybrid_wrf56"
        self.scheme = fr.HybridWRF56AtDualEdge(
            self.coord, self.para, beta, use_mbs=use_mbs
        )
        self._built = True
        return self.scheme

    def build_default(self) -> "FluxReconstructAtDualEdge":
        if not self.has_built():
            self.central_difference()
        return self


class RelativeVorticityAtDualCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)
        self.curl_at_dual_cell = base.CurlAtDualCell(
            coord, para
        )

    def __call__(
        self,
        normal_velocity_at_edge: Float[Array, "nEdge"]
    ) -> Float[Array, "nDual_cell"]:
        return self.curl_at_dual_cell(normal_velocity_at_edge)

    def build_default(self) -> Self:
        self._built = True
        return self


class AbsoluteVorticityAtDualCell(Term):
    relative_vorticity: RelativeVorticityAtDualCell

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)
        self.relative_vorticity = RelativeVorticityAtDualCell(
            disc, coord, para, method
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
        self._built = True
        return self


class ThicknessAtDualCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)
        self.interp_weighted_add_plg2dua = interp.InterpWeightedAddPlg2Dua(
            coord, para
        )

    def __call__(
        self, thickness_at_primal_cell: Float[Array, "nPrimal_cell"]
    ) -> Float[Array, "nDual_cell"]:
        return self.interp_weighted_add_plg2dua(
            thickness_at_primal_cell
        )

    def build_default(self) -> Self:
        self._built = True
        return self


class InterpPotentialVorticityMidpoint(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)
        self.recon_scheme = fr.CentralDifference2AtDualEdge(
            coord, para
        )
        self.order = 2

    def __call__(
        self,
        pv_at_dual_cell: Float[Array, "nDual_cell"],
        normal_u_at_edge: Float[Array, "nEdge"],
        tangent_u_at_edge: Float[Array, "nEdge"],
    ) -> Float[Array, "nEdge"]:
        return self.recon_scheme(
            x_at_dual_cell=pv_at_dual_cell,
            normal_u_at_edge=normal_u_at_edge,
            tangent_u_at_edge=tangent_u_at_edge
        )

    def build_default(self) -> Self:
        self._built = True
        return self


class InterpPotentialVorticityUpwind(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod,
        order: int = 3
    ):
        super().__init__(disc, coord, para, method)
        self.central_difference_scheme = fr.CentralDifference2AtDualEdge(
            coord, para
        )
        if order == 3:
            self.recon_scheme = fr.WRF3AtDualEdge(coord, para)
        elif order == 5:
            self.recon_scheme = fr.WRF5AtDualEdge(coord, para)
        else:
            raise ValueError(
                f"Unsupported order {order} for upwind interpolation."
            )
        self.order = order

    def __call__(
        self,
        pv_at_dual_cell: Float[Array, "nDual_cell"],
        normal_u_at_edge: Float[Array, "nEdge"],
        tangent_u_at_edge: Float[Array, "nEdge"],
    ) -> Float[Array, "nEdge"]:
        b = jnp.abs(normal_u_at_edge) / jnp.sqrt(
            normal_u_at_edge ** 2 + tangent_u_at_edge ** 2 + 1.0
        )
        flux1 = self.central_difference_scheme(
            x_at_dual_cell=pv_at_dual_cell,
            tangent_u_at_edge=tangent_u_at_edge
        )
        flux2 = self.recon_scheme(
            x_at_dual_cell=pv_at_dual_cell,
            tangent_u_at_edge=tangent_u_at_edge
        )
        return (1 - b) * flux1 + b * flux2

    def build_default(self) -> Self:
        self._built = True
        return self


class InterpPotentialVorticityAtDualEdge(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)

    def __call__(
        self,
        pv_at_dual_cell: Float[Array, "nDual_cell"],
        normal_u_at_edge: Float[Array, "nEdge"],
        tangent_u_at_edge: Float[Array, "nEdge"],
    ) -> Float[Array, "nEdge"]:
        return self.adv_scheme(
            pv_at_dual_cell=pv_at_dual_cell,
            normal_u_at_edge=normal_u_at_edge,
            tangent_u_at_edge=tangent_u_at_edge
        )

    def midpoint(self) -> Self:
        self.adv_scheme = InterpPotentialVorticityMidpoint(
            self._disc, self.coord, self.para, self.method
        )
        self._built = True
        return self

    def upwind(self, order: int) -> Self:
        self.adv_scheme = InterpPotentialVorticityUpwind(
            self._disc, self.coord, self.para, self.method, order=order
        )
        self._built = True
        return self

    def build_default(self) -> Self:
        if not self.has_built():
            self.upwind(order=3)
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
    interp: InterpPotentialVorticityAtDualEdge

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        self.abs_vor_at_dual_cell = AbsoluteVorticityAtDualCell(
            disc, coord, para, method
        )
        self._flux_recon_central_difference = fr.CentralDifference2AtDualEdge(
            coord, para
        )
        self.interp = InterpPotentialVorticityAtDualEdge(
            disc, coord, para, method
        )
        self.thickness_at_dual_cell = ThicknessAtDualCell(
            disc, coord, para, method
        )

        def _pv_at_dual_cell(
            abs_vor_at_dual_cell: Float[Array, "nDual_cell"],
            thickness_at_dual_cell: Float[Array, "nDual_cell"]
        ) -> Float[Array, "nDual_cell"]:
            return abs_vor_at_dual_cell / thickness_at_dual_cell

        self.pv_at_dual_cell = _pv_at_dual_cell
        super().__init__(disc, coord, para, method)

    def __call__(
        self,
        normal_u_at_edge: Float[Array, "nEdge"],
        tangent_u_at_edge: Float[Array, "nEdge"],
        thickness_at_primal_cell: Float[Array, "nPrimal_cell"]
    ) -> Float[Array, "nEdge"]:
        abs_vor_at_dual_cell = self.abs_vor_at_dual_cell(
            normal_u_at_edge
        )
        thickness_at_dual_cell = self.thickness_at_dual_cell(
            thickness_at_primal_cell,
        )
        pv_at_dual_cell = self.pv_at_dual_cell(
            abs_vor_at_dual_cell, thickness_at_dual_cell
        )
        return self.interp(
            pv_at_dual_cell=pv_at_dual_cell,
            normal_u_at_edge=normal_u_at_edge,
            tangent_u_at_edge=tangent_u_at_edge
        )

    def build_default(self) -> "PotentialVorticityAtDualEdge":
        if not self.has_built():
            self.interp.build_default()
            self._built = True
        return self


class Ringler2010KineticEnergyAtPrimalCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)
        self.interp_weighted_add_edp2plg = interp.InterpWeightedAddEdp2Plg(
            coord, para
        )

    def __call__(
        self,
        normal_u_at_edge: Float[Array, "nEdge"]
    ) -> Float[Array, "nPrimal_cell"]:
        return self.interp_weighted_add_edp2plg(
            normal_u_at_edge ** 2,
        )

    def build_default(self) -> Self:
        self._built = True
        return self


class Gassmann2013KineticEnergyAtPrimalCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)
        self.interp_weighted_add_edd2dua = interp.InterpWeightedAddEdd2Dua(
            coord, para
        )
        self.interp_weighted_add_dua2plg = interp.InterpWeightedAddDua2Plg(
            coord, para
        )
        self.ringler_2010 = Ringler2010KineticEnergyAtPrimalCell(
            disc, coord, para, method
        )
        self.gamma: FloatType = 0.9

    def __call__(
        self,
        normal_velocity_at_edge: Float[Array, "nEdge"]
    ) -> Float[Array, "nPrimal_cell"]:
        ke_edge = normal_velocity_at_edge ** 2
        ke_dc = self.interp_weighted_add_edd2dua(
            ke_edge,
        )
        ke_pc = self.ringler_2010(normal_velocity_at_edge)
        ke_dc_pc = self.interp_weighted_add_dua2plg(
            ke_dc,
        )
        return 1 * self.gamma * ke_pc + (1 - self.gamma) * ke_dc_pc

    def build_default(self) -> Self:
        self._built = True
        return self


class KineticEnergyAtPrimalCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)

    def __call__(
        self,
        normal_velocity_at_edge: Float[Array, "nEdge"]
    ) -> Float[Array, "nPrimal_cell"]:
        return self.scheme(normal_velocity_at_edge)

    def gassmann_2013(
        self,
        gamma: FloatType = 0.9
    ) -> Gassmann2013KineticEnergyAtPrimalCell:
        self.scheme_name = "gassmann_2013"
        self.scheme = Gassmann2013KineticEnergyAtPrimalCell(
            self._disc, self.coord, self.para, self.method
        )
        self.scheme.gamma = gamma
        self._built = True
        return self.scheme

    def ringler_2010(self) -> Ringler2010KineticEnergyAtPrimalCell:
        self.scheme_name = "ringler_2010"
        self.scheme = Ringler2010KineticEnergyAtPrimalCell(
            self._disc, self.coord, self.para, self.method
        )
        self._built = True
        return self.scheme

    def build_default(self) -> "KineticEnergyAtPrimalCell":
        if not self.has_built():
            self.gassmann_2013()
        return self
