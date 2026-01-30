from __future__ import annotations
from typing import TypeVar, Self, TYPE_CHECKING
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
    AbstractEquationSetAuxiliaryTerms,
    DiscretizationInfo
)

if TYPE_CHECKING:
    from terraflow.discretization.fvm.odm.trisk.single_lev.dynamics_types import Block as swm_Block
    BLOCK = TypeVar("BLOCK", bound=swm_Block)
    Block = swm_Block

IntType = typing.IntType
FloatType = typing.FloatType
Int = typing.Int
Float = typing.Float
Array = typing.Array


VM = TypeVar("VM", OneLayerVertCoordinate, HybridVertCoordinate)


CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    VM,
    TriSKBasedCGridStaggeringLayout,
]


Term = AbstractEquationSetAuxiliaryTerms[FiniteVolumeMethod]


class RelativeVorticityAtDualCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc=disc, method=method)

    def __call__(self, block: BLOCK) -> BLOCK:
        coord = block.coord
        para = block.para
        un = block.state.un.unwrap()
        rel_vor = op.curl_at_dual_cell(
            x_at_dual_edge=un,
            coord=coord,
            para=para
        )
        return eqx.tree_at(
            lambda b: b.aux.rel_vor.data,
            block,
            replace=rel_vor
        )

    def prepare(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block) -> Field:
        return block.aux.rel_vor

    def build_default(self) -> Self:
        self._built = True
        return self


class AbsoluteVorticityAtDualCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
    ):
        super().__init__(disc=disc, method=method)

    def __call__(self, block: BLOCK) -> BLOCK:
        mesh = block.coord.mesh
        para = block.para
        rel_vor = block.aux.rel_vor.unwrap()
        omega = para.omega.unwrap()
        coriolis = 2 * omega * jnp.sin(mesh._dua.centroid.lat)
        res = coriolis[None, :] + rel_vor
        return eqx.tree_at(
            lambda b: b.aux.abs_vor.data,
            block,
            replace=res
        )

    def prepare(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block) -> Field:
        return block.aux.abs_vor

    def build_default(self) -> Self:
        self._built = True
        return self


class GeopotentialHeightAtPrimalEdge(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc=disc, method=method)
        self.recon_scheme = fr.FluxReconstructAtEdgeOnPrimalMesh()

    def __call__(self, block: BLOCK) -> BLOCK:
        un = block.state.un.unwrap()
        z = block.state.z.unwrap()
        res = self.recon_scheme(
            x_at_primal_cell=z,
            normal_u_at_edge=un,
            coord=block.coord,
            aux=block.equaux.adv
        )
        return eqx.tree_at(
            lambda b: b.aux.z_e.data,
            block,
            replace=res
        )

    def prepare(self, block: Block) -> Block:
        return self.recon_scheme.prepare(block)

    def unwrap(self, block: Block) -> Field:
        return block.aux.z_e

    def build_default(self) -> Self:
        if not self.has_built():
            self.recon_scheme.build_default()
            self._built = True
        return self


class MassFluxAtPrimalEdge(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc=disc, method=method)

    def __call__(self, block: BLOCK) -> BLOCK:
        un = block.state.un.unwrap()
        z_e = block.aux.z_e.unwrap()
        res = un * z_e
        return eqx.tree_at(
            lambda b: b.aux.mfn_e.data,
            block,
            replace=res
        )

    def prepare(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block) -> Field:
        return block.aux.mfn_e

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self


class GeopotentialHeightAtDualCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc=disc, method=method)

    def __call__(self, block: BLOCK) -> BLOCK:
        coord = block.coord
        z = block.state.z.unwrap()
        z_vtx = op.interp_weighted_add_plg2dua(
            x_at_primal_cell=z,
            coord=coord,
        )
        return eqx.tree_at(
            lambda b: b.aux.z_vtx.data,
            block,
            replace=z_vtx
        )

    def prepare(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block) -> Field:
        return block.aux.z_vtx

    def build_default(self) -> Self:
        self._built = True
        return self


class InterpPotentialVorticityMidpoint(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc=disc, method=method)
        self.recon_scheme = fr.CentralDifference2AtEdgeOnDualMesh()
        self.order = 2

    def __call__(self, block: BLOCK) -> BLOCK:
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

    def prepare(self, block: Block) -> Block:
        return self.recon_scheme.prepare(block)

    def unwrap(self, block: Block) -> Field:
        return block.aux.pv_e

    def build_default(self) -> Self:
        self._built = True
        return self


class InterpPotentialVorticityUpwind(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        order: int = 3,
        use_mbs: bool = True
    ):
        super().__init__(disc=disc, method=method)
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

    def __call__(self, block: BLOCK) -> BLOCK:
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

    def prepare(self, block: Block) -> Block:
        block = self.central_difference_scheme.prepare(block)
        return self.recon_scheme.prepare(block)

    def unwrap(self, block: Block) -> Field:
        return block.aux.pv_e

    def build_default(self) -> Self:
        self._built = True
        return self


class InterpPotentialVorticityPureUpwind(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        order: int = 3,
        use_mbs: bool = True
    ):
        super().__init__(disc=disc, method=method)
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

    def __call__(self, block: BLOCK) -> BLOCK:
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

    def prepare(self, block: Block) -> Block:
        return self.recon_scheme.prepare(block)

    def unwrap(self, block: Block) -> Field:
        return block.aux.pv_e

    def build_default(self) -> Self:
        self._built = True
        return self


class InterpPotentialVorticityAtDualEdge(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc=disc, method=method)

    def __call__(self, block: BLOCK) -> BLOCK:
        return self.adv_scheme(block)

    def midpoint(self) -> Self:
        self.adv_scheme = InterpPotentialVorticityMidpoint(
            disc=self._disc,
            method=self.method
        )
        self._built = True
        return self

    def upwind(self, order: int, use_mbs: bool = True) -> Self:
        self.adv_scheme = InterpPotentialVorticityUpwind(
            disc=self._disc,
            method=self.method,
            order=order,
            use_mbs=use_mbs
        )
        self._built = True
        return self

    def pure_upwind(self, order: int, use_mbs: bool = True) -> Self:
        self.adv_scheme = InterpPotentialVorticityPureUpwind(
            disc=self._disc,
            method=self.method,
            order=order,
            use_mbs=use_mbs
        )
        self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return self.adv_scheme.prepare(block)

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
        method: FiniteVolumeMethod
    ):
        super().__init__(disc=disc, method=method)

    def __call__(self, block: BLOCK) -> BLOCK:
        abs_vor = block.aux.abs_vor.data
        z_vtx = block.aux.z_vtx.data
        res = abs_vor / z_vtx
        return eqx.tree_at(
            lambda b: b.aux.pv_vtx.data,
            block,
            replace=res
        )

    def prepare(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block) -> Field:
        return block.aux.pv_vtx

    def build_default(self) -> Self:
        self._built = True
        return self


class PotentialVorticityAtDualEdge(Term):
    interp: InterpPotentialVorticityAtDualEdge

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
    ):
        self.interp = InterpPotentialVorticityAtDualEdge(
            disc=disc,
            method=method
        )
        super().__init__(disc=disc, method=method)

    def __call__(self, block: BLOCK) -> BLOCK:
        return self.interp(block)

    def prepare(self, block: Block) -> Block:
        return self.interp.prepare(block)

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
        method: FiniteVolumeMethod
    ):
        super().__init__(disc=disc, method=method)

    def __call__(self, block: BLOCK) -> BLOCK:
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

    def prepare(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block) -> Field:
        return block.aux.ke

    def build_default(self) -> Self:
        self._built = True
        return self


class Gassmann2013KineticEnergyAtPrimalCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc=disc, method=method)
        self.ringler_2010 = Ringler2010KineticEnergyAtPrimalCell(
            disc=disc, method=method
        )
        self.gamma: FloatType = 0.9

    def __call__(self, block: BLOCK) -> BLOCK:
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

    def prepare(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block) -> Field:
        return block.aux.ke

    def build_default(self) -> Self:
        self._built = True
        return self


class KineticEnergyAtPrimalCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc=disc, method=method)

    def __call__(self, block: BLOCK) -> BLOCK:
        return self.scheme(block)

    def prepare(self, block: Block) -> Block:
        return self.scheme.prepare(block)

    def unwrap(self, block: Block) -> Field:
        return block.aux.ke

    def gassmann_2013(
        self,
        gamma: FloatType = 0.9
    ) -> Gassmann2013KineticEnergyAtPrimalCell:
        self.scheme_name = "gassmann_2013"
        self.scheme = Gassmann2013KineticEnergyAtPrimalCell(
            disc=self._disc,
            method=self.method
        )
        self.scheme.gamma = gamma
        self._built = True
        return self.scheme

    def ringler_2010(self) -> Ringler2010KineticEnergyAtPrimalCell:
        self.scheme_name = "ringler_2010"
        self.scheme = Ringler2010KineticEnergyAtPrimalCell(
            disc=self._disc,
            method=self.method
        )
        self._built = True
        return self.scheme

    def build_default(self) -> "KineticEnergyAtPrimalCell":
        if not self.has_built():
            self.gassmann_2013()
        return self
