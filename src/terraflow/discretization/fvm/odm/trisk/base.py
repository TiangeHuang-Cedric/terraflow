from __future__ import annotations
from typing import TypeVar, Self, TYPE_CHECKING
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh import operator as op
from terraflow.discretization.fvm.odm.trisk.adv import flux_reconstruct as fr
from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.layout.abstract import CoordinateSystem
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
from terraflow.discretization.abstract import (
    AbstractEquationSetAuxiliaryTerms,
    DiscretizationInfo
)

if TYPE_CHECKING:
    from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block

IntType = typing.IntType
FloatType = typing.FloatType
Int = typing.Int
Float = typing.Float
Array = typing.Array


VM = TypeVar("VM", OneLayerVertCoordinate, HybridVertCoordinate)
VLO = TypeVar("VLO", OneLayerVertCoordLayout, HybridVertCoordLayout)


CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    VM,
    TriSKBasedCGridStaggeringLayout,
    VLO
]


Term = AbstractEquationSetAuxiliaryTerms[FiniteVolumeMethod]


class DivergenceOfMassFluxAtCell(AbstractEquationSetAuxiliaryTerms[FiniteVolumeMethod]):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        baroclinic: bool,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=baroclinic,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        mfn = block.aux.mfn.unwrap()
        dmf = op.div_at_primal_cell(
            x_at_edge=mfn,
            coord=block.coord,
            para=block.para
        )
        return eqx.tree_at(
            lambda b: b.aux.dmf.data,
            block,
            replace=dmf
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return block


class RelativeVorticityAtDualCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        baroclinic: bool,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=baroclinic,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        coord = block.coord
        para = block.para
        un = block.state.un.unwrap()
        vor = op.curl_at_dual_cell(
            x_at_dual_edge=un,
            coord=coord,
            para=para
        )
        return eqx.tree_at(
            lambda b: b.aux.vor.data,
            block,
            replace=vor
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return block


class MassFlux(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        baroclinic: bool,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=baroclinic,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        un = block.state.un.unwrap()
        dmg_e = block.aux.dmg_e.unwrap()
        mfn = un * dmg_e
        return eqx.tree_at(
            lambda b: b.aux.mfn.data,
            block,
            replace=mfn
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return block


class InterpPotentialVorticityMidpoint(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        baroclinic: bool,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=baroclinic,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.recon_scheme = fr.CentralDifference2AtEdgeOnDualMesh()
        self.order = 2

    def __call__(self, block: Block) -> Block:
        pv_vtx = block.aux.pv_vtx.unwrap()
        pv_e = self.recon_scheme(
            x_at_dual_cell=pv_vtx,
            coord=block.coord,
        )
        return eqx.tree_at(
            lambda b: b.aux.pv_e.data,
            block,
            replace=pv_e
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return self.recon_scheme.prepare(block)


class InterpPotentialVorticityUpwind(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        baroclinic: bool,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool,
        order: int = 3,
        use_mbs: bool = True
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=baroclinic,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.central_difference_scheme = fr.CentralDifference2AtEdgeOnDualMesh()
        if order == 3:
            self.recon_scheme = fr.WRF3AtEdgeOnDualMesh(use_mbs=use_mbs)
        elif order == 5:
            self.recon_scheme = fr.WRF5AtEdgeOnDualMesh(use_mbs=use_mbs)
        else:
            raise ValueError(
                f"Unsupported order {order} for upwind interpolation."
            )
        self.order = order

    def __call__(self, block: Block) -> Block:
        coord = block.coord
        un = block.state.un.unwrap()
        pv_vtx = block.aux.pv_vtx.unwrap()
        ut = block.aux.ut.unwrap()
        b = jnp.abs(un) / jnp.sqrt(
            un ** 2 + ut ** 2 + 1.0
        )
        flux1 = self.central_difference_scheme(
            x_at_dual_cell=pv_vtx,
            coord=coord,
        )
        flux2 = self.recon_scheme(
            x_at_dual_cell=pv_vtx,
            tangent_u_at_edge=ut,
            coord=coord,
            aux=block.equaux.adv
        )
        res = (1 - b) * flux1 + b * flux2
        return eqx.tree_at(
            lambda b: b.aux.pv_e.data,
            block,
            replace=res
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        block = self.central_difference_scheme.prepare(block)
        return self.recon_scheme.prepare(block)


class InterpPotentialVorticityPureUpwind(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        baroclinic: bool,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool,
        order: int = 3,
        use_mbs: bool = True
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=baroclinic,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.recon_scheme = fr.FluxReconstructAtEdgeOnDualMesh()
        if order == 3:
            self.recon_scheme.wrf3(use_mbs=use_mbs)
        elif order == 5:
            self.recon_scheme.wrf5(use_mbs=use_mbs)
        else:
            raise ValueError(
                f"Unsupported order {order} for pure upwind interpolation."
            )
        self.order = order

    def __call__(self, block: Block) -> Block:
        un = block.state.un.unwrap()
        pv_vtx = block.aux.pv_vtx.unwrap()
        ut = block.aux.ut.unwrap()
        res = self.recon_scheme(
            x_at_dual_cell=pv_vtx,
            normal_u_at_edge=un,
            tangent_u_at_edge=ut,
            coord=block.coord,
            aux=block.equaux.adv
        )
        return eqx.tree_at(
            lambda b: b.aux.pv_e.data,
            block,
            replace=res
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return self.recon_scheme.prepare(block)


class InterpPotentialVorticityAtDualEdge(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        baroclinic: bool,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=baroclinic,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        return self.adv_scheme(block)

    def midpoint(self) -> Self:
        self.adv_scheme = InterpPotentialVorticityMidpoint(
            disc=self._disc,
            method=self.method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self._built = True
        return self

    def upwind(self, order: int, use_mbs: bool = True) -> Self:
        self.adv_scheme = InterpPotentialVorticityUpwind(
            disc=self._disc,
            method=self.method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection,
            order=order,
            use_mbs=use_mbs
        )
        self._built = True
        return self

    def pure_upwind(self, order: int, use_mbs: bool = True) -> Self:
        self.adv_scheme = InterpPotentialVorticityPureUpwind(
            disc=self._disc,
            method=self.method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection,
            order=order,
            use_mbs=use_mbs
        )
        self._built = True
        return self

    def build_default(self) -> Self:
        if not self.has_built():
            self.upwind(order=3)
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return self.adv_scheme.prepare(block)


class PotentialVorticity(Term):
    interp: InterpPotentialVorticityAtDualEdge

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        baroclinic: bool,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        self.interp = InterpPotentialVorticityAtDualEdge(
            disc=disc,
            method=method,
            baroclinic=baroclinic,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=baroclinic,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        mesh = block.coord.mesh
        omega = block.para.omega.unwrap()
        vor = block.aux.vor.unwrap()
        coriolis = 2 * omega * jnp.sin(mesh._dua.centroid.lat)
        abs_vor = coriolis[None, :] + vor
        dmg_vtx = block.aux.dmg_vtx.unwrap()
        pv_vtx = abs_vor / dmg_vtx
        block = eqx.tree_at(
            lambda b: b.aux.pv_vtx.data,
            block,
            replace=pv_vtx
        )
        return self.interp(block)

    def build_default(self) -> "PotentialVorticity":
        if not self.has_built():
            self.interp.build_default()
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return self.interp.prepare(block)


class Ringler2010KineticEnergyAtPrimalCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        baroclinic: bool,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=baroclinic,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        coord = block.coord
        un = block.state.un.unwrap()
        res = op.interp_weighted_add_edp2plg(
            x_at_primal_edge=un ** 2,
            coord=coord,
        )
        return eqx.tree_at(
            lambda b: b.aux.ke.data,
            block,
            replace=res
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return block


class Gassmann2013KineticEnergyAtPrimalCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        baroclinic: bool,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool,
        gamma: FloatType = 0.9
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=baroclinic,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.ringler_2010 = Ringler2010KineticEnergyAtPrimalCell(
            disc=disc,
            method=method,
            baroclinic=baroclinic,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.gamma: FloatType = gamma

    def __call__(self, block: Block) -> Block:
        coord = block.coord
        un = block.state.un.unwrap()
        ke_edge = un ** 2
        ke_vtx = op.interp_weighted_add_edd2dua(
            x_at_dual_edge=ke_edge,
            coord=coord,
        )
        ke_cell = self.ringler_2010(block).aux.ke.unwrap()
        ke_vtx_cell = op.interp_weighted_add_dua2plg(
            x_at_dual_cell=ke_vtx,
            coord=coord,
        )
        res = 1 * self.gamma * ke_cell + (1 - self.gamma) * ke_vtx_cell
        return eqx.tree_at(
            lambda b: b.aux.ke.data,
            block,
            replace=res
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self.ringler_2010.build_default()
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return block


class KineticEnergyAtPrimalCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        baroclinic: bool,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=baroclinic,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        return self.scheme(block)

    def gassmann_2013(
        self,
        gamma: FloatType = 0.9
    ) -> Gassmann2013KineticEnergyAtPrimalCell:
        self.scheme_name = "gassmann_2013"
        self.scheme = Gassmann2013KineticEnergyAtPrimalCell(
            disc=self._disc,
            method=self.method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection,
        )
        self.scheme.gamma = gamma
        self._built = True
        return self.scheme

    def ringler_2010(self) -> Ringler2010KineticEnergyAtPrimalCell:
        self.scheme_name = "ringler_2010"
        self.scheme = Ringler2010KineticEnergyAtPrimalCell(
            disc=self._disc,
            method=self.method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self._built = True
        return self.scheme

    def build_default(self) -> "KineticEnergyAtPrimalCell":
        if not self.has_built():
            self.gassmann_2013()
        return self

    def prepare(self, block: Block) -> Block:
        return self.scheme.prepare(block)
