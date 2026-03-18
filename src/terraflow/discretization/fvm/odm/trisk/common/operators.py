from __future__ import annotations
from typing import TypeVar, Self, TYPE_CHECKING
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh import operators as op
from terraflow.discretization.fvm.odm.trisk.common.adv import \
    flux_reconstruct as fr
from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.layout.abstract import CoordinateSystem
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
from terraflow.discretization.abstract import (
    AbstractEquationMode,
    AbstractEquationOperator,
    DiscretizationInfo
)

if TYPE_CHECKING:
    from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block, State

IntType = typing.IntType
FloatType = typing.FloatType
Int = typing.Int
Float = typing.Float
Array = typing.Array


VM = TypeVar("VM", OneLayerVertCoordinate, HybridVertCoordinate)
VLO = TypeVar("VLO", OneLayerVertCoordLayout, HybridVertCoordLayout)
EM = TypeVar("EM", bound=AbstractEquationMode)


CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    VM,
    TriSKBasedCGridStaggeringLayout,
    VLO
]


Term = AbstractEquationOperator[FiniteVolumeMethod, EM]


class RelativeVorticityAtDualCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: AbstractEquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        coord = block.coord
        para = block.para
        un = star_state.un.unwrap()
        vor = op.curl_at_dual_cell(
            x_at_dual_edge=un,
            coord=coord,
            para=para
        )
        block = eqx.tree_at(
            lambda b: b.aux.vor.data,
            block,
            replace=vor
        )
        return block, star_state

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block


class NormalMassFluxAtEdge(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: AbstractEquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        un = star_state.un.unwrap()
        dmg_e = star_state.dmg_e.unwrap()
        mfn = un * dmg_e
        block = eqx.tree_at(
            lambda b: b.aux.mfn.data,
            block,
            replace=mfn
        )
        return block, star_state

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block


class DivergenceOfMassFluxAtCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: AbstractEquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        mfn = block.aux.mfn.unwrap()
        dmf = op.div_at_primal_cell(
            x_at_edge=mfn,
            coord=block.coord,
            para=block.para
        )
        block = eqx.tree_at(
            lambda b: b.aux.dmf.data,
            block,
            replace=dmf
        )
        return block, star_state

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block


class InterpPotentialVorticityMidpoint(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: AbstractEquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.recon_scheme = fr.CentralDifference2AtEdgeOnDualMesh()
        self.order = 2

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        pv_vtx = block.aux.pv_vtx.unwrap()
        pv_e = self.recon_scheme(
            x_at_dual_cell=pv_vtx,
            coord=block.coord,
        )
        block = eqx.tree_at(
            lambda b: b.aux.pv_e.data,
            block,
            replace=pv_e
        )
        return block, star_state

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return self.recon_scheme.prepare_block(block)


class InterpPotentialVorticityUpwind(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: AbstractEquationMode,
        order: int = 3,
        use_mbs: bool = True
    ):
        super().__init__(disc=disc, method=method, mode=mode)
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

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        coord = block.coord
        un = star_state.un.unwrap()
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
        pv_e = (1 - b) * flux1 + b * flux2
        block = eqx.tree_at(
            lambda b: b.aux.pv_e.data,
            block,
            replace=pv_e
        )
        return block, star_state

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        block = self.central_difference_scheme.prepare_block(block)
        return self.recon_scheme.prepare_block(block)


class InterpPotentialVorticityPureUpwind(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: AbstractEquationMode,
        order: int = 3,
        use_mbs: bool = True
    ):
        super().__init__(disc=disc, method=method, mode=mode)
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

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        un = star_state.un.unwrap()
        pv_vtx = block.aux.pv_vtx.unwrap()
        ut = block.aux.ut.unwrap()
        pv_e = self.recon_scheme(
            x_at_dual_cell=pv_vtx,
            normal_u_at_edge=un,
            tangent_u_at_edge=ut,
            coord=block.coord,
            aux=block.equaux.adv
        )
        block = eqx.tree_at(
            lambda b: b.aux.pv_e.data,
            block,
            replace=pv_e
        )
        return block, star_state

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return self.recon_scheme.prepare_block(block)


class InterpPotentialVorticityAtDualEdge(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: AbstractEquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        return self.adv_scheme(block, star_state)

    def midpoint(self) -> Self:
        self.adv_scheme = InterpPotentialVorticityMidpoint(
            disc=self._disc, method=self.method, mode=self.mode
        )
        self._built = True
        return self

    def upwind(self, order: int, use_mbs: bool = True) -> Self:
        self.adv_scheme = InterpPotentialVorticityUpwind(
            disc=self._disc,
            method=self.method,
            mode=self.mode,
            order=order,
            use_mbs=use_mbs
        )
        self._built = True
        return self

    def pure_upwind(self, order: int, use_mbs: bool = True) -> Self:
        self.adv_scheme = InterpPotentialVorticityPureUpwind(
            disc=self._disc,
            method=self.method,
            mode=self.mode,
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

    def prepare_block(self, block: Block) -> Block:
        return self.adv_scheme.prepare_block(block)


class PotentialVorticity(Term):
    interp: InterpPotentialVorticityAtDualEdge

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: AbstractEquationMode,
    ):
        self.interp = InterpPotentialVorticityAtDualEdge(
            disc=disc, method=method, mode=mode
        )
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        mesh = block.coord.mesh
        omega = block.para.omega.unwrap()
        vor = block.aux.vor.unwrap()
        coriolis = 2 * omega * jnp.sin(mesh._dua.centroid.lat)
        abs_vor = coriolis[None, :] + vor
        dmg_vtx = star_state.dmg_vtx.unwrap()
        pv_vtx = abs_vor / dmg_vtx
        block = eqx.tree_at(
            lambda b: b.aux.pv_vtx.data,
            block,
            replace=pv_vtx
        )
        return self.interp(block, star_state)

    def build_default(self) -> "PotentialVorticity":
        if not self.has_built():
            self.interp.build_default()
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return self.interp.prepare_block(block)


class Ringler2010KineticEnergyAtPrimalCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: AbstractEquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        coord = block.coord
        un = star_state.un.unwrap()
        ke = op.interp_weighted_add_edp2plg(
            x_at_primal_edge=un ** 2,
            coord=coord,
        )
        block = eqx.tree_at(
            lambda b: b.aux.ke.data,
            block,
            replace=ke
        )
        return block, star_state

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block


class Gassmann2013KineticEnergyAtPrimalCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: AbstractEquationMode,
        gamma: FloatType = 0.9
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.ringler_2010 = Ringler2010KineticEnergyAtPrimalCell(
            disc=disc, method=method, mode=mode
        )
        self.gamma: FloatType = gamma

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        coord = block.coord
        un = star_state.un.unwrap()
        ke_edge = un ** 2
        ke_vtx = op.interp_weighted_add_edd2dua(
            x_at_dual_edge=ke_edge,
            coord=coord,
        )
        ke_cell = self.ringler_2010(block, star_state)[0].aux.ke.unwrap()
        ke_vtx_cell = op.interp_weighted_add_dua2plg(
            x_at_dual_cell=ke_vtx,
            coord=coord,
        )
        ke = 1 * self.gamma * ke_cell + (1 - self.gamma) * ke_vtx_cell
        block = eqx.tree_at(
            lambda b: b.aux.ke.data,
            block,
            replace=ke
        )
        return block, star_state

    def build_default(self) -> Self:
        if not self.has_built():
            self.ringler_2010.build_default()
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block


class KineticEnergyAtPrimalCell(Term):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: AbstractEquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        return self.scheme(block, star_state)

    def gassmann_2013(
        self,
        gamma: FloatType = 0.9
    ) -> Gassmann2013KineticEnergyAtPrimalCell:
        self.scheme_name = "gassmann_2013"
        self.scheme = Gassmann2013KineticEnergyAtPrimalCell(
            disc=self._disc, method=self.method, mode=self.mode,
        )
        self.scheme.gamma = gamma
        self._built = True
        return self.scheme

    def ringler_2010(self) -> Ringler2010KineticEnergyAtPrimalCell:
        self.scheme_name = "ringler_2010"
        self.scheme = Ringler2010KineticEnergyAtPrimalCell(
            disc=self._disc, method=self.method, mode=self.mode
        )
        self._built = True
        return self.scheme

    def build_default(self) -> "KineticEnergyAtPrimalCell":
        if not self.has_built():
            self.gassmann_2013()
        return self

    def prepare_block(self, block: Block) -> Block:
        return self.scheme.prepare_block(block)
