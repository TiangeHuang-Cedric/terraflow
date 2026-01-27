from typing import TypeVar, Self
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh import operator as op
from terraflow.discretization.fvm.odm.trisk.adv import flux_reconstruct as fr
from terraflow.vert_coord import OneLayerVertCoordinate, HybridVertCoordinate
from terraflow.layout import CoordinateSystem, Field
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.discretization.abstract import (
    AbstractEquationSetAuxiliary,
    DiscretizationInfo
)
from terraflow.discretization.fvm.odm.trisk.single_lev.dynamics_types import Block as swm_Block
from terraflow.planet import PlanetParameterSet


IntType = typing.IntType
FloatType = typing.FloatType
Int = typing.Int
Float = typing.Float
Array = typing.Array


VM = TypeVar("VM", OneLayerVertCoordinate, HybridVertCoordinate)
BLOCK = TypeVar("BLOCK", bound=swm_Block)
Block = swm_Block

Term = AbstractEquationSetAuxiliary[
    OrthogonalDualMesh,
    OneLayerVertCoordinate | HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    FiniteVolumeMethod,
    BLOCK
]

CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    VM,
    TriSKBasedCGridStaggeringLayout,
]


class RelativeVorticityAtDualCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)
        self.curl_at_dual_cell = op.CurlAtDualCell(
            coord, para
        )

    def __call__(self, block: BLOCK) -> BLOCK:
        un_e = block.state.un_e.unwrap()
        res = self.curl_at_dual_cell(un_e)
        return eqx.tree_at(
            lambda b: b.aux.rel_vor_dc.data,
            block,
            replace=res
        )

    def unwrap(self, block: Block) -> Field:
        return block.aux.rel_vor_dc

    def build_default(self) -> Self:
        self._built = True
        return self


class AbsoluteVorticityAtDualCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod,
    ):
        super().__init__(disc, coord, para, method)

    def __call__(self, block: BLOCK) -> BLOCK:
        rel_vor = block.aux.rel_vor_dc.unwrap()
        omega = self.para.omega.unwrap()
        coriolis = 2 * omega * jnp.sin(self.mesh._dua.centroid.lat)
        res = coriolis[None, :] + rel_vor
        return eqx.tree_at(
            lambda b: b.aux.abs_vor_dc.data,
            block,
            replace=res
        )

    def unwrap(self, block: Block) -> Field:
        return block.aux.abs_vor_dc

    def build_default(self) -> Self:
        self._built = True
        return self


class GeopotentialHeightAtPrimalEdge(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)
        self.recon_scheme = fr.FluxReconstructAtPrimalEdge(coord, para)

    def __call__(self, block: BLOCK) -> BLOCK:
        un_e = block.state.un_e.unwrap()
        z_pc = block.state.z_pc.unwrap()
        res = self.recon_scheme(
            x_at_primal_cell=z_pc,
            normal_u_at_edge=un_e,
        )
        return eqx.tree_at(
            lambda b: b.aux.z_e.data,
            block,
            replace=res
        )

    def unwrap(self, block: Block) -> Field:
        return block.aux.z_e

    def build_default(self) -> Self:
        self._built = True
        return self


class MassFluxAtPrimalEdge(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)

    def __call__(self, block: BLOCK) -> BLOCK:
        un_e = block.state.un_e.unwrap()
        z_e = block.aux.z_e.unwrap()
        res = un_e * z_e
        return eqx.tree_at(
            lambda b: b.aux.mfx_e.data,
            block,
            replace=res
        )

    def unwrap(self, block: Block) -> Field:
        return block.aux.mfx_e

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self


class GeopotentialHeightAtDualCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)
        self.interp_weighted_add_plg2dua = op.InterpWeightedAddPlg2Dua(
            coord, para
        )

    def __call__(self, block: BLOCK) -> BLOCK:
        z_pc = block.state.z_pc.unwrap()
        res = self.interp_weighted_add_plg2dua(z_pc)
        return eqx.tree_at(
            lambda b: b.aux.z_dc.data,
            block,
            replace=res
        )

    def unwrap(self, block: Block) -> Field:
        return block.aux.z_dc

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

    def __call__(self, block: BLOCK) -> BLOCK:
        un_e = block.state.un_e.unwrap()
        pv_dc = block.aux.pv_dc.unwrap()
        ut_e = block.aux.ut_e.unwrap()
        res = self.recon_scheme(
            x_at_dual_cell=pv_dc,
            tangent_u_at_edge=ut_e,
            normal_u_at_edge=un_e,
        )
        return eqx.tree_at(
            lambda b: b.aux.pv_e.data,
            block,
            replace=res
        )

    def unwrap(self, block: Block) -> Field:
        return block.aux.pv_e

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
        order: int = 3,
        use_mbs: bool = True
    ):
        super().__init__(disc, coord, para, method)
        self.central_difference_scheme = fr.CentralDifference2AtDualEdge(
            coord, para
        )
        if order == 3:
            self.recon_scheme = fr.WRF3AtDualEdge(coord, para, use_mbs=use_mbs)
        elif order == 5:
            self.recon_scheme = fr.WRF5AtDualEdge(coord, para, use_mbs=use_mbs)
        else:
            raise ValueError(
                f"Unsupported order {order} for upwind interpolation."
            )
        self.order = order

    def __call__(self, block: BLOCK) -> BLOCK:
        un_e = block.state.un_e.unwrap()
        pv_dc = block.aux.pv_dc.unwrap()
        ut_e = block.aux.ut_e.unwrap()
        b = jnp.abs(un_e) / jnp.sqrt(
            un_e ** 2 + ut_e ** 2 + 1.0
        )
        flux1 = self.central_difference_scheme(
            x_at_dual_cell=pv_dc,
            tangent_u_at_edge=ut_e
        )
        flux2 = self.recon_scheme(
            x_at_dual_cell=pv_dc,
            tangent_u_at_edge=ut_e
        )
        res = (1 - b) * flux1 + b * flux2
        return eqx.tree_at(
            lambda b: b.aux.pv_e.data,
            block,
            replace=res
        )

    def unwrap(self, block: Block) -> Field:
        return block.aux.pv_e

    def build_default(self) -> Self:
        self._built = True
        return self


class InterpPotentialVorticityPureUpwind(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod,
        order: int = 3,
        use_mbs: bool = True
    ):
        super().__init__(disc, coord, para, method)
        self.recon_scheme = fr.FluxReconstructAtDualEdge(coord, para)
        if order == 3:
            self.recon_scheme.wrf3(use_mbs=use_mbs)
        elif order == 5:
            self.recon_scheme.wrf5(use_mbs=use_mbs)
        else:
            raise ValueError(
                f"Unsupported order {order} for pure upwind interpolation."
            )
        self.order = order

    def __call__(self, block: BLOCK) -> BLOCK:
        un_e = block.state.un_e.unwrap()
        pv_dc = block.aux.pv_dc.unwrap()
        ut_e = block.aux.ut_e.unwrap()
        res = self.recon_scheme(
            x_at_dual_cell=pv_dc,
            normal_u_at_edge=un_e,
            tangent_u_at_edge=ut_e,
        )
        return eqx.tree_at(
            lambda b: b.aux.pv_e.data,
            block,
            replace=res
        )

    def unwrap(self, block: Block) -> Field:
        return block.aux.pv_e

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

    def __call__(self, block: BLOCK) -> BLOCK:
        return self.adv_scheme(block)

    def midpoint(self) -> Self:
        self.adv_scheme = InterpPotentialVorticityMidpoint(
            disc=self._disc,
            coord=self.coord,
            para=self.para,
            method=self.method
        )
        self._built = True
        return self

    def upwind(self, order: int, use_mbs: bool = True) -> Self:
        self.adv_scheme = InterpPotentialVorticityUpwind(
            disc=self._disc,
            coord=self.coord,
            para=self.para,
            method=self.method,
            order=order,
            use_mbs=use_mbs
        )
        self._built = True
        return self

    def pure_upwind(self, order: int, use_mbs: bool = True) -> Self:
        self.adv_scheme = InterpPotentialVorticityPureUpwind(
            disc=self._disc,
            coord=self.coord,
            para=self.para,
            method=self.method,
            order=order,
            use_mbs=use_mbs
        )
        self._built = True
        return self

    def unwrap(self, block: Block) -> Field:
        return block.aux.pv_e

    def build_default(self) -> Self:
        if not self.has_built():
            self.upwind(order=3)
        return self


class PotentialVorticityAtDualCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)

    def __call__(self, block: BLOCK) -> BLOCK:
        abs_vor_dc = block.aux.abs_vor_dc.data
        z_dc = block.aux.z_dc.data
        res = abs_vor_dc / z_dc
        return eqx.tree_at(
            lambda b: b.aux.pv_dc.data,
            block,
            replace=res
        )

    def unwrap(self, block: Block) -> Field:
        return block.aux.pv_dc

    def build_default(self) -> Self:
        self._built = True
        return self


class PotentialVorticityAtDualEdge(Term):
    interp: InterpPotentialVorticityAtDualEdge

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod,
    ):
        self.interp = InterpPotentialVorticityAtDualEdge(
            disc=disc,
            coord=coord,
            para=para,
            method=method
        )
        super().__init__(disc, coord, para, method)

    def __call__(self, block: BLOCK) -> BLOCK:
        return self.interp(block)

    def unwrap(self, block: Block) -> Field:
        return block.aux.pv_e

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
        self.interp_weighted_add_edp2plg = op.InterpWeightedAddEdp2Plg(
            coord, para
        )

    def __call__(self, block: BLOCK) -> BLOCK:
        un_e = block.state.un_e.unwrap()
        res = self.interp_weighted_add_edp2plg(un_e ** 2)
        return eqx.tree_at(
            lambda b: b.aux.ke_pc.data,
            block,
            replace=res
        )

    def unwrap(self, block: Block) -> Field:
        return block.aux.ke_pc

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
        self.interp_weighted_add_edd2dua = op.InterpWeightedAddEdd2Dua(
            coord, para
        )
        self.interp_weighted_add_dua2plg = op.InterpWeightedAddDua2Plg(
            coord, para
        )
        self.ringler_2010 = Ringler2010KineticEnergyAtPrimalCell(
            disc, coord, para, method
        )
        self.gamma: FloatType = 0.9

    def __call__(self, block: BLOCK) -> BLOCK:
        un_e = block.state.un_e.unwrap()
        ke_edge = un_e ** 2
        ke_dc = self.interp_weighted_add_edd2dua(ke_edge)
        ke_pc = self.ringler_2010(block).aux.ke_pc.unwrap()
        ke_dc_pc = self.interp_weighted_add_dua2plg(ke_dc)
        res = 1 * self.gamma * ke_pc + (1 - self.gamma) * ke_dc_pc
        return eqx.tree_at(
            lambda b: b.aux.ke_pc.data,
            block,
            replace=res
        )

    def unwrap(self, block: Block) -> Field:
        return block.aux.ke_pc

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

    def __call__(self, block: BLOCK) -> BLOCK:
        return self.scheme(block)

    def unwrap(self, block: Block) -> Field:
        return block.aux.ke_pc

    def gassmann_2013(
        self,
        gamma: FloatType = 0.9
    ) -> Gassmann2013KineticEnergyAtPrimalCell:
        self.scheme_name = "gassmann_2013"
        self.scheme = Gassmann2013KineticEnergyAtPrimalCell(
            disc=self._disc,
            coord=self.coord,
            para=self.para,
            method=self.method
        )
        self.scheme.gamma = gamma
        self._built = True
        return self.scheme

    def ringler_2010(self) -> Ringler2010KineticEnergyAtPrimalCell:
        self.scheme_name = "ringler_2010"
        self.scheme = Ringler2010KineticEnergyAtPrimalCell(
            disc=self._disc,
            coord=self.coord,
            para=self.para,
            method=self.method
        )
        self._built = True
        return self.scheme

    def build_default(self) -> "KineticEnergyAtPrimalCell":
        if not self.has_built():
            self.gassmann_2013()
        return self
