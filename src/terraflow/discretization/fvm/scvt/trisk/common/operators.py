from __future__ import annotations
from typing import Dict, TypeVar, Self, Any, TYPE_CHECKING
import jax
import jax.numpy as jnp
import equinox as eqx

from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.scvt import operators as op
from terraflow.discretization.abstract import (
    AbstractDynamicalEquationOperator,
    EquationMode
)
from terraflow.util.model_config import ModelConfig
from ..parallel import tree_at

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.util.model_config import ModelConfig
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
    from terraflow.discretization.abstract import DiscretizationInfo
    from terraflow.discretization.fvm.scvt.trisk.single_lev.dynamics_types import Block as SingleLevBlock
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as HybridLevBlock

    IntType = typing.IntType
    FloatType = typing.FloatType
    Int = typing.Int
    Float = typing.Float
    Array = typing.Array

    VM = TypeVar("VM", OneLayerVertCoordinate, HybridVertCoordinate)
    VLO = TypeVar("VLO", OneLayerVertCoordLayout, HybridVertCoordLayout)

    CoordSys = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        VM,
        TriSKBasedCGridStaggeringLayout,
        VLO
    ]

    BLOCK = TypeVar("BLOCK", SingleLevBlock, HybridLevBlock)

DynOperator = AbstractDynamicalEquationOperator[FiniteVolumeMethod]


class RelativeVorticityAtDualCell(DynOperator):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        coord = block.coord
        para = block.para
        un = block.state.un.unwrap()[star_idx, :, :]
        vor = block.aux.vor.unwrap()
        # ==========================================================
        vor = op.curl_at_dual_cell(
            x_at_dual_edge=un,   # in
            x_at_dual_cell=vor,  # inout
            coord=coord,
            para=para
        )
        # ==========================================================
        return tree_at(
            coord=coord,
            where=lambda b: b.aux.vor,
            pytree=block,
            replace=vor,
            fill_halo=True
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class DivergenceOfVelocityAtCell(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        coord = block.coord
        para = block.para
        un = block.state.un.unwrap()[star_idx, :, :]
        div = block.aux.div.unwrap()
        # ==========================================================
        div = op.div_at_primal_cell(
            x_at_primal_edge=un,   # in
            x_at_primal_cell=div,  # inout
            coord=coord,
            para=para
        )
        # ==========================================================
        return tree_at(
            coord=coord,
            where=lambda b: b.aux.div,
            pytree=block,
            replace=div,
            fill_halo=True
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class TriSKTangentVelocityAtEdge(DynOperator):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        """Reconstruct tangent velocity at dual edges from normal velocity at edges."""
        coord = block.coord
        ids, ide = coord.mesh.edge_ids, coord.mesh.edge_ide
        kds, kde = coord.vert.full_kds, coord.vert.full_kde
        coriolis = block.equaux.coriolis.trisk
        edd_length = coord.mesh.dual.edge.length  # (nedge,)
        trisk_idx = coriolis.edp_edge_on_edge     # (nedge, maxnnb*2)
        weight = coriolis.edp_trsk_on_edge        # (nedge, maxnnb*2)
        lenp = coriolis.edp_edpl_on_edge          # (nedge, maxnnb*2)
        un = block.state.un.unwrap()[star_idx, :]
        ut = block.aux.ut.unwrap()
        # ==========================================================
        ep_idx = trisk_idx[ids:ide+1, :]

        def _wrap(un: Float[Array, "nEdge"]) -> Float[Array, "nEdge"]:
            wind_tile = un[ep_idx]
            wind_tile = jnp.where(ep_idx < 0, 0.0, wind_tile)
            return jnp.sum(
                weight[ids:ide+1, :] * lenp[ids:ide+1, :] * wind_tile,
                axis=-1
            ) / edd_length[ids:ide+1]

        ut = ut.at[kds:kde+1, ids:ide+1].set(
            jax.vmap(
                _wrap,
                in_axes=0,
                out_axes=0
            )(un[kds:kde+1])
        )
        # ==========================================================
        return tree_at(
            coord=coord,
            where=lambda b: b.aux.ut,
            pytree=block,
            replace=ut,
            fill_halo=True
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        if not block.equaux.coriolis.trisk.has_initialized():
            from .coriolis.trisk import TriSKCoriolisForceAtEdgeData
            coord = block.coord
            block = eqx.tree_at(
                lambda b: b.equaux.coriolis.trisk,
                block,
                replace=TriSKCoriolisForceAtEdgeData.from_coord(coord)
            )
        return block


class TangentVelocityAtEdge(DynOperator):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        return self.scheme(block, star_idx)

    def trisk(self) -> TriSKTangentVelocityAtEdge:
        self.scheme = TriSKTangentVelocityAtEdge(
            disc=self.disc, method=self.method, mode=self.mode
        )
        self._cfg["ut_scheme"] = "trisk"
        self._built = True
        return self.scheme

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            if cfg.coriolis_scheme == "trisk":
                self.trisk()
            else:
                raise ValueError(
                    f"Unsupported coriolis scheme {cfg.coriolis_scheme} for tangent velocity reconstruction."
                )
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.scheme.prepare_block(block)


class NormalMassFluxAtEdge(DynOperator):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ihs, ihe = mesh.edge_ihs[0], mesh.edge_ihe[0]
        kds, kde = vert.full_kds, vert.full_kde
        un = block.state.un.unwrap()[star_idx, :, :]
        dmg_e = block.state.dmg_e.unwrap()[star_idx, :, :]
        mfn = block.aux.mfn.unwrap()
        # ==========================================================
        mfn = mfn.at[kds:kde+1, ihs:ihe+1].set(
            un[kds:kde+1, ihs:ihe+1] * dmg_e[kds:kde+1, ihs:ihe+1]
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: b.aux.mfn,
            pytree=block,
            replace=mfn
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class DivergenceOfMassFluxAtCell(DynOperator):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        mfn = block.aux.mfn.unwrap()
        dmf = block.aux.dmf.unwrap()
        # ==========================================================
        dmf = op.div_at_primal_cell(
            x_at_primal_edge=mfn,
            x_at_primal_cell=dmf,
            coord=block.coord,
            para=block.para
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: b.aux.dmf,
            pytree=block,
            replace=dmf,
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class InterpPotentialVorticityMidpoint:

    def __init__(self):
        from .adv.flux_reconstruct import CentralDifference2AtEdgeOnDualMesh
        self.recon_scheme = CentralDifference2AtEdgeOnDualMesh()
        self.order = 2
        self._cfg: dict[str, Any] = {}
        self._built = False

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> Array:
        pv_vtx = block.aux.pv_vtx.unwrap()
        pv_e = block.aux.pv_e.unwrap()
        # ==========================================================
        return self.recon_scheme(
            x_at_dual_cell=pv_vtx,  # in
            x_at_dual_edge=pv_e,    # inout
            coord=block.coord,
        )

    def has_built(self) -> bool:
        return self._built

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.recon_scheme.prepare_block(block)


class InterpPotentialVorticityUpwind:

    def __init__(
        self,
        order: int = 3,
        use_mbs: bool = True,
        optimized: bool = True
    ):
        from .adv.flux_reconstruct import CentralDifference2AtEdgeOnDualMesh
        self.central_difference_scheme = CentralDifference2AtEdgeOnDualMesh()
        if order == 3:
            if optimized:
                from .adv.flux_reconstruct import WRFPure3AtEdgeOnDualMesh
                self.recon_scheme = WRFPure3AtEdgeOnDualMesh(
                    use_mbs=use_mbs
                )
            else:
                from .adv.flux_reconstruct import WRF3AtEdgeOnDualMesh
                self.recon_scheme = WRF3AtEdgeOnDualMesh(use_mbs=use_mbs)
        elif order == 5:
            from .adv.flux_reconstruct import WRF5AtEdgeOnDualMesh
            self.recon_scheme = WRF5AtEdgeOnDualMesh(use_mbs=use_mbs)
        else:
            raise ValueError(
                f"Unsupported order {order} for upwind interpolation."
            )
        self.order = order
        self._cfg: dict[str, Any] = {}
        self._built = False

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> Array:
        coord = block.coord
        ids, ide = coord.mesh.edge_ids, coord.mesh.edge_ide
        kds, kde = coord.vert.full_kds, coord.vert.full_kde
        un = block.state.un.unwrap()[star_idx, :, :]
        ut = block.aux.ut.unwrap()
        pv_vtx = block.aux.pv_vtx.unwrap()
        pv_e = block.aux.pv_e.unwrap()
        # ==========================================================
        un_slice = un[kds:kde+1, ids:ide+1]
        ut_slice = ut[kds:kde+1, ids:ide+1]
        b = jnp.abs(ut_slice) / jnp.sqrt(
            un_slice ** 2 + ut_slice ** 2 + 1.0
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
        return pv_e.at[kds:kde+1, ids:ide+1].set((1 - b) * flux1 + b * flux2)

    def has_built(self) -> bool:
        return self._built

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.central_difference_scheme.prepare_block(block)
        return self.recon_scheme.prepare_block(block)


class InterpPotentialVorticityPureUpwind:

    def __init__(
        self,
        order: int = 3,
        use_mbs: bool = True,
        optimized: bool = True
    ):
        from .adv.flux_reconstruct import FluxReconstructAtEdgeOnDualMesh
        self.recon_scheme = FluxReconstructAtEdgeOnDualMesh()
        if order == 3:
            self.recon_scheme.wrf3(use_mbs=use_mbs, optimized=optimized)
        elif order == 5:
            self.recon_scheme.wrf5(use_mbs=use_mbs)
        else:
            raise ValueError(
                f"Unsupported order {order} for pure upwind interpolation."
            )
        self.order = order
        self._cfg: dict[str, Any] = {}
        self._built = False

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> Array:
        ut = block.aux.ut.unwrap()
        pv_vtx = block.aux.pv_vtx.unwrap()
        pv_e = block.aux.pv_e.unwrap()
        # ==========================================================
        return self.recon_scheme(
            x_at_dual_cell=pv_vtx,  # in
            x_at_dual_edge=pv_e,    # inout
            tangent_u_at_edge=ut,
            coord=block.coord,
            aux=block.equaux.adv
        )

    def has_built(self) -> bool:
        return self._built

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.recon_scheme.prepare_block(block)


class InterpPotentialVorticityAtDualEdge:

    def __init__(self):
        self._cfg: dict[str, Any] = {}
        self._built = False

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> Array:
        return self.adv_scheme(block, star_idx)

    def midpoint(self) -> Self:
        self.adv_scheme = InterpPotentialVorticityMidpoint()
        self._cfg["pv_adv_scheme"] = "midpoint"
        self._built = True
        return self

    def upwind(
        self,
        order: int,
        use_mbs: bool = True,
        optimized: bool = True
    ) -> Self:
        self.adv_scheme = InterpPotentialVorticityUpwind(
            order=order,
            use_mbs=use_mbs,
            optimized=optimized
        )
        self._cfg["pv_adv_scheme"] = "upwind"
        self._cfg["upwind_order_pv"] = order
        self._cfg["upwind_use_mbs_pv"] = int(use_mbs)
        self._cfg["upwind_optimized_pv"] = int(optimized)
        self._built = True
        return self

    def pure_upwind(
        self,
        order: int,
        use_mbs: bool = True,
        optimized: bool = True
    ) -> Self:
        self.adv_scheme = InterpPotentialVorticityPureUpwind(
            order=order,
            use_mbs=use_mbs,
            optimized=optimized
        )
        self._cfg["pv_adv_scheme"] = "pure_upwind"
        self._cfg["upwind_order_pv"] = order
        self._cfg["upwind_use_mbs_pv"] = int(use_mbs)
        self._cfg["upwind_optimized_pv"] = int(optimized)
        self._built = True
        return self

    def has_built(self) -> bool:
        return self._built

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            if cfg.pv_adv_scheme == "upwind":
                self.upwind(
                    order=cfg.upwind_order_pv,
                    use_mbs=cfg.upwind_use_mbs_pv,
                    optimized=cfg.upwind_optimized_pv
                )
            elif cfg.pv_adv_scheme == "pure_upwind":
                self.pure_upwind(
                    order=cfg.upwind_order_pv,
                    use_mbs=cfg.upwind_use_mbs_pv,
                    optimized=cfg.upwind_optimized_pv
                )
            elif cfg.pv_adv_scheme == "midpoint":
                self.midpoint()
            else:
                raise ValueError(
                    f"Unsupported PV advection scheme {cfg.pv_adv_scheme}."
                )
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.adv_scheme.prepare_block(block)


class PotentialVorticity(DynOperator):
    interp: InterpPotentialVorticityAtDualEdge

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode,
    ):
        self.interp = InterpPotentialVorticityAtDualEdge()
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ide = mesh.dual_ids, mesh.dual_ide
        kds, kde = vert.full_kds, vert.full_kde
        omega = block.para.omega.unwrap()
        vor = block.aux.vor.unwrap()
        dmg_vtx = block.state.dmg_vtx.unwrap()[star_idx, :, :]
        lat = mesh._dua.centroid.lat
        pv_vtx = block.aux.pv_vtx.unwrap()
        # ==========================================================
        # Calc pv at vtx
        coriolis = 2 * omega * jnp.sin(lat[ids:ide+1])
        abs_vor = coriolis[None, :] + vor[kds:kde+1, ids:ide+1]
        _pv_vtx = abs_vor / dmg_vtx[kds:kde+1, ids:ide+1]
        pv_vtx = pv_vtx.at[kds:kde+1, ids:ide+1].set(_pv_vtx)
        block = tree_at(
            coord=block.coord,
            where=lambda b: b.aux.pv_vtx,
            pytree=block,
            replace=pv_vtx,
            fill_halo=True
        )
        # ==========================================================
        # Calc pv at edge
        pv_e = self.interp(block, star_idx)
        return tree_at(
            coord=block.coord,
            where=lambda b: b.aux.pv_e,
            pytree=block,
            replace=pv_e,
            fill_halo=True
        )

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.interp.config2dict()
        self._cfg.update(cfg)
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.interp.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.interp.prepare_block(block)


class Ringler2010KineticEnergyAtPrimalCell(DynOperator):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        coord = block.coord
        ids, ide = coord.mesh.edge_ids, coord.mesh.edge_ide
        kds, kde = coord.vert.full_kds, coord.vert.full_kde
        un = block.state.un.unwrap()[star_idx, :, :]
        ke = block.aux.ke.unwrap()
        # ==========================================================
        ke = ke.at[kds:kde+1, ids:ide+1].set(
            self._evaluate_at_domain(un=un, coord=coord)
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: b.aux.ke,
            pytree=block,
            replace=ke,
            fill_halo=True,
        )

    def _evaluate_at_domain(
        self,
        un: Float[Array, "nlev nEdge"],
        coord: CoordSys
    ) -> Float[Array, "nlev nPrimal_cell"]:
        return op.interp_weighted_add_edp2plg(
            x_at_primal_edge=un,
            coord=coord,
            field_fn=lambda x: x**2
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class Gassmann2013KineticEnergyAtPrimalCell(DynOperator):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode,
        gamma: FloatType = 0.9
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.ringler_2010 = Ringler2010KineticEnergyAtPrimalCell(
            disc=disc, method=method, mode=mode
        )
        self.gamma: FloatType = gamma

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        coord = block.coord
        ids, ide = coord.mesh.edge_ids, coord.mesh.edge_ide
        kds, kde = coord.vert.full_kds, coord.vert.full_kde
        un = block.state.un.unwrap()[star_idx, :, :]
        ke = block.aux.ke.unwrap()
        # ==========================================================
        ke_vtx = op.interp_weighted_add_edd2dua(
            x_at_dual_edge=un,
            coord=coord,
            field_fn=lambda x: x**2
        )
        ke_cell = self.ringler_2010._evaluate_at_domain(
            un=un,
            coord=coord
        )
        ke_vtx_cell = op.interp_weighted_add_dua2plg(
            x_at_dual_cell=ke_vtx,
            coord=coord,
        )
        ke = ke.at[kds:kde+1, ids:ide+1].set(
            1 * self.gamma * ke_cell + (1 - self.gamma) * ke_vtx_cell
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: b.aux.ke,
            pytree=block,
            replace=ke,
            fill_halo=True,
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.ringler_2010.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class KineticEnergyAtPrimalCell(DynOperator):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        return self.scheme(block, star_idx)

    def gassmann_2013(
        self,
        gamma: FloatType = 0.9
    ) -> Gassmann2013KineticEnergyAtPrimalCell:
        self.scheme_name = "gassmann_2013"
        self.scheme = Gassmann2013KineticEnergyAtPrimalCell(
            disc=self._disc, method=self.method, mode=self.mode,
        )
        self.scheme.gamma = gamma
        self._cfg["ke_scheme"] = "gassmann_2013"
        self._cfg["ke_gamma"] = gamma
        self._built = True
        return self.scheme

    def ringler_2010(self) -> Ringler2010KineticEnergyAtPrimalCell:
        self.scheme_name = "ringler_2010"
        self.scheme = Ringler2010KineticEnergyAtPrimalCell(
            disc=self._disc, method=self.method, mode=self.mode
        )
        self._cfg["ke_scheme"] = "ringler_2010"
        self._built = True
        return self.scheme

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.scheme.config2dict()
        self._cfg.update(cfg)
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            if cfg.ke_scheme == "ringler_2010":
                self.ringler_2010()
            elif cfg.ke_scheme == "gassmann_2013":
                self.gassmann_2013(gamma=cfg.ke_gamma)
            else:
                raise ValueError(
                    f"Unsupported kinetic energy scheme {cfg.ke_scheme}."
                )
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.scheme.prepare_block(block)
