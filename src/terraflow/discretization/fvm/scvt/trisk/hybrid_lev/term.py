from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Self
import equinox as eqx
import jax.numpy as jnp

from terraflow.util import typing, formula
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.scvt.operators import (
    div_at_primal_cell,
    grad_at_dual_edge,
    interp_weighted_add_plg2dua
)
from terraflow.discretization.abstract import (
    AbstractDynamicalEquationOperator,
    AbstractPrepareSpaceOperators,
)
from terraflow.discretization.fvm.scvt.trisk.common.adv.term import (
    SetWindForBackground,
    AccumWind,
    TendencyOfTracerDueToDivergenceOfTracerFlux,
    TendencyOfTracerDueToVerticalAdvection,
)
from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.adv import TracerRun3D
from terraflow.discretization.fvm.scvt.trisk.hybrid_lev import operators as hybrid_op
from terraflow.dynamics.baroclinic import types
from ..parallel import tree_at

if TYPE_CHECKING:
    from terraflow.util.model_config import ModelConfig
    from terraflow.discretization.fvm.scvt.trisk.common.adv.upwind import (
        UpwindHorizontalAdvectionAtCell,
        UpwindVerticalAdvection
    )
    from terraflow.discretization.abstract import DiscretizationInfo, EquationMode
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as BLOCK

    Float = typing.Float
    Int = typing.Int
    Array = typing.Array

DynOperator = AbstractDynamicalEquationOperator[FiniteVolumeMethod]


class Temperature(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ihe = mesh.polg_ids, mesh.polg_ihe[0]
        full_kds, full_kde = vert.full_kds, vert.full_kde
        idx_qv = block.tracers.idx_qv
        pt = block.state.pt.unwrap()[star_idx, :, :]
        p = block.state.p.unwrap()[star_idx, :, :]
        t = block.aux.t.unwrap()
        tv = block.aux.tv.unwrap()
        # ==========================================================
        if idx_qv >= 0:
            old_idx = block.adv_batches["moist"].get_time_idx(itime_str="old")
            q = block.tracers.q.unwrap()[old_idx, idx_qv, :, :]
            rd_o_cpd = block.para.rd_o_cpd.unwrap()
            t = t.at[full_kds:full_kde+1, ids:ihe+1].set(
                formula.temperature(
                    pt=pt[full_kds:full_kde+1, ids:ihe+1],
                    p=p[full_kds:full_kde+1, ids:ihe+1],
                    qv=q[full_kds:full_kde+1, ids:ihe+1],
                    para=block.para
                )
            )
            tv = tv.at[full_kds:full_kde+1, ids:ihe+1].set(
                formula.virtual_temperature_from_modified_potential_temperature(
                    pt=pt[full_kds:full_kde+1, ids:ihe+1],
                    pk=p[full_kds:full_kde+1, ids:ihe+1]**rd_o_cpd,
                    qm=q[full_kds:full_kde+1, ids:ihe+1],
                    para=block.para
                )
            )
        else:
            t = t.at[full_kds:full_kde+1, ids:ihe+1].set(
                formula.temperature(
                    pt=pt[full_kds:full_kde+1, ids:ihe+1],
                    p=p[full_kds:full_kde+1, ids:ihe+1],
                    qv=jnp.asarray(0.0, dtype=pt.dtype),
                    para=block.para
                )
            )
            tv = tv.at[full_kds:full_kde+1, ids:ihe+1].set(
                t[full_kds:full_kde+1, ids:ihe+1]
            )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: (b.aux.t, b.aux.tv),
            pytree=block,
            replace=(t, tv)
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class TangentMassFluxAtEdge(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ihs, ihe = mesh.edge_ihs[0], mesh.edge_ihe[0]
        kds, kde = vert.half_kds, vert.half_kde
        ut = block.aux.ut.unwrap()
        dmg_e = block.state.dmg_e.unwrap()[star_idx, :, :]
        mft = block.aux.mft.unwrap()
        # ==========================================================
        mft = mft.at[kds:kde+1, ihs:ihe+1].set(
            dmg_e[kds:kde+1, ihs:ihe+1] * ut[kds:kde+1, ihs:ihe+1]
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: b.aux.mft,
            pytree=block,
            replace=mft
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class VerticalMassFluxAtHalfLev(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        half_kds = block.coord.vert.half_kds
        half_kde = block.coord.vert.half_kde
        polg_ids = block.coord.mesh.polg_ids
        polg_ide = block.coord.mesh.polg_ide
        dmgsdt = block.tend.dmgsdt.unwrap()
        dmf = block.aux.dmf.unwrap()
        mfz_lev = block.aux.mfz_lev.unwrap()
        # ==========================================================
        dmgdt_lev = block.coord.vert.calc_dmgdt_lev(
            dmgsdt[polg_ids:polg_ide+1],
            ks=half_kds + 1,
            ke=half_kde - 1
        )
        mfz_lev = mfz_lev.at[half_kds+1:half_kde].set(
            - dmgdt_lev - jnp.cumsum(
                dmf[:half_kde-1, polg_ids:polg_ide+1],
                axis=0
            )
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: b.aux.mfz_lev,
            pytree=block,
            replace=mfz_lev,
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class DryAirWeight(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        mesh = block.coord.mesh
        vert = block.coord.vert
        polg_ids = mesh.polg_ids
        polg_ihe = mesh.polg_ihe[0]
        full_kds = vert.full_kds
        full_kde = vert.full_kde
        half_kds = vert.half_kds
        half_kde = vert.half_kde
        mgs = block.state.mgs.unwrap()[star_idx, :]
        mg_lev = block.state.mg_lev.unwrap()[star_idx, :, :]
        mg = block.state.mg.unwrap()[star_idx, :, :]
        # ==========================================================
        mg_lev = mg_lev.at[half_kds:half_kde+1, polg_ids:polg_ihe+1].set(
            block.coord.vert.calc_mg_lev(
                mgs[polg_ids:polg_ihe+1],
                ks=half_kds,
                ke=half_kde
            )
        )
        mg = mg.at[full_kds:full_kde+1, polg_ids:polg_ihe+1].set(
            0.5 * (
                mg_lev[full_kds:full_kde+1, polg_ids:polg_ihe+1]
                + mg_lev[full_kds+1:full_kde+2, polg_ids:polg_ihe+1]
            )
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: (b.state.mg, b.state.mg_lev),
            pytree=block,
            replace=(mg, mg_lev),
            at_dims=(0, 0),
            at_idxes=(star_idx, star_idx)
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class DeltaDryAirWeight(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.interp = hybrid_op.InterpMethod()

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        coord = block.coord
        kds, kde = coord.vert.full_kds, coord.vert.full_kde
        mg = block.state.mg.unwrap()[star_idx, :, :]
        mg_lev = block.state.mg_lev.unwrap()[star_idx, :, :]
        dmg = block.state.dmg.unwrap()[star_idx, :, :]
        dmg_lev = block.state.dmg_lev.unwrap()[star_idx, :, :]
        dmg_e = block.state.dmg_e.unwrap()[star_idx, :, :]
        dmg_vtx = block.state.dmg_vtx.unwrap()[star_idx, :, :]
        # ==========================================================
        dmg = hybrid_op.delta_at_full_lev(
            coord=coord,
            x_lev=mg_lev,
            dx=dmg,
        )
        dmg = eqx.error_if(
            x=dmg,
            pred=jnp.any(dmg < 0),
            msg="The dry-air weight levels are not monotonic!",
            on_error="raise"
        )
        dmg_lev = hybrid_op.delta_at_half_lev(
            coord=coord,
            x=mg,
            x_lev=mg_lev,
            dx_lev=dmg_lev
        )
        dmg_e = self.interp.cell2edge(
            x=dmg,
            y=dmg_e,
            coord=coord
        )
        dmg_vtx = interp_weighted_add_plg2dua(
            x_at_primal_cell=dmg,
            x_at_dual_cell=dmg_vtx,
            coord=coord,
        )
        # ==========================================================
        return tree_at(
            coord=coord,
            where=lambda b: (
                b.state.dmg, b.state.dmg_lev,
                b.state.dmg_e, b.state.dmg_vtx
            ),
            pytree=block,
            replace=(dmg, dmg_lev, dmg_e, dmg_vtx),
            at_dims=(0, 0, 0, 0),
            at_idxes=(star_idx, star_idx, star_idx, star_idx)
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class DensityOfDryAir(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ihe = mesh.polg_ids, mesh.polg_ihe[0]
        kds, kde = vert.full_kds, vert.full_kde
        gz_lev = block.state.gz_lev.unwrap()[star_idx, :, :]
        dmg = block.state.dmg.unwrap()[star_idx, :, :]
        rhod = block.aux.rhod.unwrap()
        # ==========================================================
        dgz = hybrid_op.delta_at_full_lev(
            coord=block.coord,
            x_lev=gz_lev,
            hori_entity="polg",
            hori_end_at="halo0",
        )
        rhod = rhod.at[kds:kde+1, ids:ihe+1].set(
            - dmg[kds:kde+1, ids:ihe+1] / dgz
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: b.aux.rhod,
            pytree=block,
            replace=rhod
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class Geopotential(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ihe = mesh.polg_ids, mesh.polg_ihe[0]
        full_kds, full_kde = vert.full_kds, vert.full_kde
        half_kds, half_kde = vert.half_kds, vert.half_kde
        para = block.para
        gzs = block.static.gzs.unwrap()
        ph_lev = block.state.ph_lev.unwrap()[star_idx, :, :]
        tv = block.aux.tv.unwrap()
        rd = para.rd.unwrap()
        gz_lev = block.state.gz_lev.unwrap()[star_idx, :, :]
        gz = block.state.gz.unwrap()[star_idx, :, :]
        # ====================================================
        # Calculat gz_lev
        kds, kde = half_kds, half_kde
        # =======================================
        # From half_kde to half_kds
        tmp = rd * tv * jnp.log(
            ph_lev[kds+1:kde+1, ids:ihe+1] / ph_lev[kds:kde, ids:ihe+1]
        )
        _gz_lev = gzs + jnp.cumulative_sum(
            jnp.flip(tmp, axis=0),
            axis=0,
            include_initial=True
        )
        # =======================================
        # Flip back, from half_kds to half_kde
        _gz_lev = jnp.flip(_gz_lev, axis=0)
        # =======================================
        # Wrap Back
        gz_lev = gz_lev.at[kds:kde+1, ids:ihe+1].set(_gz_lev)
        # ===================================================
        kds, kde = full_kds, full_kde
        # Calculate gz
        gz = gz.at[kds:kde+1, ids:ihe+1].set(
            0.5 * (
                gz_lev[kds:kde+1, ids:ihe+1]
                + gz_lev[kds+1:kde+2, ids:ihe+1]
            )
        )
        # ===================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: (b.state.gz, b.state.gz_lev),
            pytree=block,
            replace=(gz, gz_lev),
            at_dims=(0, 0),
            at_idxes=(star_idx, star_idx)
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class HydrostaticPressure(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ihe = mesh.polg_ids, mesh.polg_ihe[0]
        full_kds, full_kde = vert.full_kds, vert.full_kde
        half_kds, half_kde = vert.half_kds, vert.half_kde
        mg_lev = block.state.mg_lev.unwrap()[star_idx, :, :]
        dmg = block.state.dmg.unwrap()[star_idx, :, :]
        ph_lev = block.state.ph_lev.unwrap()[star_idx, :, :]
        ph = block.state.ph.unwrap()[star_idx, :, :]
        phs = block.state.phs.unwrap()[star_idx, :]
        # =======================================================
        # Calculate ph at half level
        kds, kde = half_kds, half_kde
        if self.mode.is_moist():
            qm = block.tracers.qm.unwrap()[:, :]
            dmg_ = dmg[kds:kde+1, ids:ihe+1]
            qm_ = qm[kds:kde+1, ids:ihe+1]
            moist_dmg = dmg_ * (1.0 + qm_)
            ph_lev = ph_lev.at[kds:kde+1, ids:ihe+1].set(
                mg_lev[kds, :] + jnp.cumulative_sum(
                    moist_dmg,
                    axis=0,
                    include_initial=True
                )
            )
        else:
            ph_lev = ph_lev.at[kds:kde+1, ids:ihe+1].set(
                mg_lev[kds, :] + jnp.cumulative_sum(
                    dmg[kds:kde+1, ids:ihe+1],
                    axis=0,
                    include_initial=True
                )
            )
        # =======================================================
        # Calculate ph at full level
        kds, kde = full_kds, full_kde
        ph = ph.at[kds:kde+1, ids:ihe+1].set(
            0.5 * (
                ph_lev[kds:kde+1, ids:ihe+1]
                + ph_lev[kds+1:kde+2, ids:ihe+1]
            )
        )
        # =======================================================
        # Calculate phs
        kde = half_kde
        phs = phs.at[ids:ihe+1].set(ph_lev[kde, ids:ihe+1])
        # =======================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: (b.state.phs, b.state.ph, b.state.ph_lev),
            pytree=block,
            replace=(phs, ph, ph_lev),
            at_dims=(0, 0, 0),
            at_idxes=(star_idx, star_idx, star_idx)
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class Pressure(DynOperator):
    # nh
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ihe = mesh.polg_ids, mesh.polg_ihe[0]
        full_kds, full_kde = vert.full_kds, vert.full_kde
        half_kds, half_kde = vert.half_kds, vert.half_kde
        ph = block.state.ph.unwrap()[star_idx, :, :]
        ph_lev = block.state.ph_lev.unwrap()[star_idx, :, :]
        p = block.state.p.unwrap()[star_idx, :, :]
        p_lev = block.state.p_lev.unwrap()[star_idx, :, :]
        # =======================================================
        if self.mode.hydrostatic:
            p_lev = p_lev.at[half_kds:half_kde+1, ids:ihe+1].set(
                ph_lev[half_kds:half_kde+1, ids:ihe+1]
            )
            p = p.at[full_kds:full_kde+1, ids:ihe+1].set(
                ph[full_kds:full_kde+1, ids:ihe+1]
            )
        else:
            raise NotImplementedError
        # =======================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: (b.state.p, b.state.p_lev),
            pytree=block,
            replace=(p, p_lev),
            at_dims=(0, 0),
            at_idxes=(star_idx, star_idx)
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class PotentialTemperatureHorizontalAdvection(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        from ..common.adv.schemes import HorizontalAdvectionSchemeAtCell
        self.adv_scheme = HorizontalAdvectionSchemeAtCell()

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        pt = block.state.pt.unwrap()[star_idx, :, :]
        batch = block.adv_batch_pt
        coord = block.coord
        aux = block.equaux.adv
        # ==========================================================
        batch = self.adv_scheme.calc_tracer_hflx(
            q=pt,
            batch=batch,
            coord=coord,
            aux=aux
        )
        # ==========================================================
        return eqx.tree_at(
            lambda b: b.adv_batch_pt,
            block,
            batch
        )

    def upwind(self, order: int = 3, optimized: bool = True) -> UpwindHorizontalAdvectionAtCell:
        self.adv_scheme.upwind(order=order, optimized=optimized)
        self._cfg["adv_scheme"] = "upwind"
        self._cfg["upwind_order"] = order
        self._cfg["upwind_optimized"] = int(optimized)
        return self.adv_scheme.scheme

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.adv_scheme.build_with_config(target="pt", cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.adv_scheme.prepare_block(block)


class PotentialTemperatureVerticalAdvection(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        from ..common.adv.schemes import VerticalAdvectionScheme
        self.adv_scheme = VerticalAdvectionScheme()

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        pt = block.state.pt.unwrap()[star_idx, :, :]
        batch = block.adv_batch_pt
        # ==========================================================
        batch = self.adv_scheme.calc_tracer_vflx(
            q=pt,
            batch=batch,
            coord=block.coord
        )
        # ==========================================================
        return eqx.tree_at(
            lambda b: b.adv_batch_pt,
            block,
            batch
        )

    def upwind(self, order: int = 3) -> UpwindVerticalAdvection:
        self.adv_scheme.upwind(order=order)
        self._cfg["upwind_order"] = order
        return self.adv_scheme.scheme

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.adv_scheme.prepare_block(block)

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.adv_scheme.build_with_config(target="pt", cfg=cfg)
            self._built = True
        return self


class TendencyOfDryAirMassAtSurface(
    types.DivergenceOfMassFluxAtSurfaceProtocol[FiniteVolumeMethod]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> BLOCK:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ide = mesh.polg_ids, mesh.polg_ide
        kds, kde = vert.full_kds, vert.full_kde
        dmf = block.aux.dmf.unwrap()
        dmgsdt_dmf = block.tend.dmgsdt_dmf.unwrap()
        dmgsdt = block.tend.dmgsdt.unwrap()
        # ==========================================================
        dmgsdt_dmf = dmgsdt_dmf.at[ids:ide+1].set(
            - jnp.sum(dmf[kds:kde+1, ids:ide+1], axis=0)
        )
        dmgsdt = dmgsdt.at[ids:ide+1].set(
            dmgsdt[ids:ide+1] + dmgsdt_dmf[ids:ide+1]
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: (b.tend.dmgsdt, b.tend.dmgsdt_dmf),
            pytree=block,
            replace=(dmgsdt, dmgsdt_dmf)
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class TendencyOfPotentialTemperatureDueToDivergenceOfFlux(
    types.DivergenceOfThermodynamicMassFluxProtocol[FiniteVolumeMethod]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.adv_scheme = PotentialTemperatureHorizontalAdvection(
            disc=disc, method=method, mode=mode
        )

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> BLOCK:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ide = mesh.polg_ids, mesh.polg_ide
        kds, kde = vert.full_kds, vert.full_kde
        un = block.state.un.unwrap()[star_idx, :, :]
        dmg = block.state.dmg.unwrap()[star_idx, :, :]
        mfn = block.aux.mfn.unwrap()
        mfz_lev = block.aux.mfz_lev.unwrap()
        batch = block.adv_batch_pt
        dptdt_dptfn = block.tend.dptdt_dptfn.unwrap()
        dptdt = block.tend.dptdt.unwrap()
        # ==========================================================
        batch = batch.associate(
            un=un,
            mfn=mfn,
            mfz=mfz_lev,
            m=dmg
        )
        block = eqx.tree_at(
            lambda b: b.adv_batch_pt,
            block,
            batch
        )
        block = self.adv_scheme(block, star_idx)
        ptfn = block.adv_batch_pt.qmfn.unwrap()
        dptdt_dptfn = dptdt_dptfn.at[kds:kde+1, ids:ide+1].set(
            - div_at_primal_cell(
                x_at_primal_edge=ptfn,
                coord=block.coord,
                para=block.para
            )
        )
        dptdt = dptdt.at[kds:kde+1, ids:ide+1].set(
            dptdt[kds:kde+1, ids:ide+1] + dptdt_dptfn[kds:kde+1, ids:ide+1]
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: (b.tend.dptdt, b.tend.dptdt_dptfn),
            pytree=block,
            replace=(dptdt, dptdt_dptfn)
        )

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.adv_scheme.config2dict()
        self._update_config(cfg=cfg)
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.adv_scheme.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.adv_scheme.prepare_block(block)
        return block


class TendencyOfPotentialTemperatureDueToVerticalAdvection(
    types.VerticalAdvectionOfThermodynamicTermProtocol[FiniteVolumeMethod]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.adv_scheme = PotentialTemperatureVerticalAdvection(
            disc=disc, method=method, mode=mode
        )

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> BLOCK:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ide = mesh.polg_ids, mesh.polg_ide
        kds, kde = vert.full_kds, vert.full_kde
        dptdt_ptvf = block.tend.dptdt_ptvf.unwrap()
        dptdt = block.tend.dptdt.unwrap()
        # ==========================================================
        block = self.adv_scheme(block, star_idx)
        dptdt_ptvf = dptdt_ptvf.at[kds:kde+1, ids:ide+1].set(
            - hybrid_op.delta_at_full_lev(
                coord=block.coord,
                x_lev=block.adv_batch_pt.qmfz.unwrap()
            )
        )
        dptdt = dptdt.at[kds:kde+1, ids:ide+1].set(
            dptdt[kds:kde+1, ids:ide+1] + dptdt_ptvf[kds:kde+1, ids:ide+1]
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: (b.tend.dptdt, b.tend.dptdt_ptvf),
            pytree=block,
            replace=(dptdt, dptdt_ptvf)
        )

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.adv_scheme.config2dict()
        self._update_config(cfg=cfg)
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.adv_scheme.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.adv_scheme.prepare_block(block)
        return block


class TendencyOfNormalVelocityDueToVerticalAdvection(
    types.VerticalAdvectionOfMomentumTermProtocol[FiniteVolumeMethod]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.interp = hybrid_op.InterpMethod()

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> BLOCK:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ide = mesh.edge_ids, mesh.edge_ide
        kds, kde = vert.full_kds, vert.full_kde
        un = block.state.un.unwrap()[star_idx, :, :]
        un_lev = block.aux.un_lev.unwrap()
        dmg_e = block.state.dmg_e.unwrap()[star_idx, :, :]
        mfz_lev = block.aux.mfz_lev.unwrap()
        mfz_e_lev = block.aux.mfz_e_lev.unwrap()
        dundt_wedundeta = block.tend.dundt_wedundeta.unwrap()
        dundt = block.tend.dundt.unwrap()
        # ==========================================================
        un_lev = self.interp.edge2lev(
            coord=block.coord,
            x=un,
            y=un_lev,
            extrap_opt=False
        )
        mfz_e_lev = self.interp.lev2edge(
            x=mfz_lev,
            y=mfz_e_lev,
            coord=block.coord,
        )
        dundt_wedundeta = dundt_wedundeta.at[kds:kde+1, ids:ide+1].set(
            - hybrid_op.adv_vert_at_full_lev(
                coord=block.coord,
                x=un,
                x_lev=un_lev,
                mfz_lev=mfz_e_lev,
                dmg=dmg_e,
                hori_entity="edge",
            )
        )
        dundt = dundt.at[kds:kde+1, ids:ide+1].set(
            dundt[kds:kde+1, ids:ide+1] + dundt_wedundeta[kds:kde+1, ids:ide+1]
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: (b.tend.dundt, b.tend.dundt_wedundeta),
            pytree=block,
            replace=(dundt, dundt_wedundeta)
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class TendencyOfNormalVelocityDueToPressureGradientForce(
    types.PressureGradientForceTermProtocol[FiniteVolumeMethod]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        from ..common.pgf.schemes import PressureGradientForce
        self.pgf_scheme = PressureGradientForce(
            disc=disc, method=method, mode=mode
        )

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> BLOCK:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ide = mesh.edge_ids, mesh.edge_ide
        kds, kde = vert.full_kds, vert.full_kde
        dundt = block.tend.dundt.unwrap()
        # ==========================================================
        block = self.pgf_scheme(block, star_idx=star_idx)
        dundt_pgf = block.tend.dundt_pgf.unwrap()
        dundt = dundt.at[kds:kde+1, ids:ide+1].set(
            dundt[kds:kde+1, ids:ide+1] + dundt_pgf[kds:kde+1, ids:ide+1]
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: b.tend.dundt,
            pytree=block,
            replace=dundt
        )

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.pgf_scheme.config2dict()
        self._update_config(cfg=cfg)
        return self._cfg

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.pgf_scheme.prepare_block(block)

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.pgf_scheme.build_with_config(cfg=cfg)
            self._built = True
        return self


class TendencyOfNormalVelocityDueToNonlinearCoriolisTermAtDualEdge(
    types.NonlinearCoriolisTermProtocol[FiniteVolumeMethod]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        from ..common.coriolis.schemes import CoriolisForceAtEdge
        self.scheme = CoriolisForceAtEdge(
            disc=disc, method=method, mode=mode
        )

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> BLOCK:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ide = mesh.edge_ids, mesh.edge_ide
        kds, kde = vert.full_kds, vert.full_kde
        dundt = block.tend.dundt.unwrap()
        # ==========================================================
        block = self.scheme(block, star_idx)
        dundt_coriolis = block.tend.dundt_coriolis.unwrap()
        dundt = dundt.at[kds:kde+1, ids:ide+1].set(
            dundt[kds:kde+1, ids:ide+1] + dundt_coriolis[kds:kde+1, ids:ide+1]
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: (b.tend.dundt, b.tend.dundt_coriolis),
            pytree=block,
            replace=(dundt, dundt_coriolis)
        )

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.scheme.config2dict()
        self._update_config(cfg=cfg)
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.scheme.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.scheme.prepare_block(block)


class TendencyOfNormalVelocityDueToGradientOfKineticEnergyAtDualEdge(
    types.GradientOfKineticEnergyProtocol[FiniteVolumeMethod]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> BLOCK:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ide = mesh.edge_ids, mesh.edge_ide
        kds, kde = vert.full_kds, vert.full_kde
        ke = block.aux.ke.unwrap()
        dundt_grad_ke = block.tend.dundt_grad_ke.unwrap()
        dundt = block.tend.dundt.unwrap()
        # ==========================================================
        dundt_grad_ke = dundt_grad_ke.at[kds:kde+1, ids:ide+1].set(
            - grad_at_dual_edge(
                x_at_primal_cell=ke,
                coord=block.coord,
                para=block.para
            )
        )
        dundt = dundt.at[kds:kde+1, ids:ide+1].set(
            dundt[kds:kde+1, ids:ide+1] + dundt_grad_ke[kds:kde+1, ids:ide+1]
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: (b.tend.dundt, b.tend.dundt_grad_ke),
            pytree=block,
            replace=(dundt, dundt_grad_ke)
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class ExPrepareSpaceOperators(
    AbstractPrepareSpaceOperators[FiniteVolumeMethod]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        from ..common import operators as common_op
        self.mg = DryAirWeight(
            disc=disc, method=method, mode=mode
        )
        self.dmg = DeltaDryAirWeight(
            disc=disc, method=method, mode=mode
        )
        self.mfn = common_op.NormalMassFluxAtEdge(
            disc=disc, method=method, mode=mode
        )
        self.mft = TangentMassFluxAtEdge(
            disc=disc, method=method, mode=mode
        )
        self.dmf = common_op.DivergenceOfMassFluxAtCell(
            disc=disc, method=method, mode=mode
        )
        self.ph = HydrostaticPressure(
            disc=disc, method=method, mode=mode
        )
        self.p = Pressure(
            disc=disc, method=method, mode=mode
        )
        self.t = Temperature(
            disc=disc, method=method, mode=mode
        )
        self.gz = Geopotential(
            disc=disc, method=method, mode=mode
        )
        self.ut = common_op.TangentVelocityAtEdge(
            disc=disc, method=method, mode=mode
        )
        self.div = common_op.DivergenceOfVelocityAtCell(
            disc=disc, method=method, mode=mode
        )
        self.vor = common_op.RelativeVorticityAtDualCell(
            disc=disc, method=method, mode=mode
        )
        self.pv = common_op.PotentialVorticity(
            disc=disc, method=method, mode=mode
        )
        self.ke = common_op.KineticEnergyAtPrimalCell(
            disc=disc, method=method, mode=mode
        )
        self.rhod = DensityOfDryAir(
            disc=disc, method=method, mode=mode
        )

    def first_call(self, block: BLOCK) -> BLOCK:
        old = block.old
        block = self.mg(block, old)
        block = self.dmg(block, old)
        block = self.mfn(block, old)
        block = self.dmf(block, old)
        block = self.ph(block, old)
        block = self.p(block, old)
        block = self.t(block, old)
        block = self.gz(block, old)
        block = self.ut(block, old)
        block = self.mft(block, old)
        block = self.vor(block, old)
        block = self.pv(block, old)
        block = self.ke(block, old)
        block = self.rhod(block, old)
        return block

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        pass_tag: str,
        explicit: bool
    ) -> BLOCK:
        # TODO: optimize by using forward-backward scheme
        if pass_tag == "all_pass":
            block = self.ut(block, star_idx)
            block = self.mfn(block, star_idx)
            block = self.mft(block, star_idx)
            block = self.dmf(block, star_idx)
            block = self.ph(block, star_idx)
            block = self.p(block, star_idx)
            block = self.t(block, star_idx)
            block = self.gz(block, star_idx)
            block = self.vor(block, star_idx)
            block = self.pv(block, star_idx)
            block = self.ke(block, star_idx)
            block = self.rhod(block, star_idx)
        elif pass_tag == "forward_pass":
            block = self.ut(block, star_idx)
            block = self.mfn(block, star_idx)
            block = self.mft(block, star_idx)
            block = self.dmf(block, star_idx)
            block = self.vor(block, star_idx)
            block = self.pv(block, star_idx)
            block = self.ke(block, star_idx)
        elif pass_tag == "backward_pass":
            block = self.ph(block, star_idx)
            block = self.p(block, star_idx)
            block = self.t(block, star_idx)
            block = self.gz(block, star_idx)
            block = self.rhod(block, star_idx)
        else:
            raise ValueError(f"Unsupported pass tag: {pass_tag}!")
        return block

    def config2dict(self) -> Dict[str, Any]:
        self._cfg.update(self.ut.config2dict())
        self._cfg.update(self.mfn.config2dict())
        self._cfg.update(self.mft.config2dict())
        self._cfg.update(self.dmf.config2dict())
        self._cfg.update(self.ph.config2dict())
        self._cfg.update(self.p.config2dict())
        self._cfg.update(self.t.config2dict())
        self._cfg.update(self.gz.config2dict())
        self._cfg.update(self.vor.config2dict())
        self._cfg.update(self.pv.config2dict())
        self._cfg.update(self.ke.config2dict())
        self._cfg.update(self.rhod.config2dict())
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.mg.build_with_config(cfg=cfg)
            self.dmg.build_with_config(cfg=cfg)
            self.mfn.build_with_config(cfg=cfg)
            self.mft.build_with_config(cfg=cfg)
            self.dmf.build_with_config(cfg=cfg)
            self.ph.build_with_config(cfg=cfg)
            self.p.build_with_config(cfg=cfg)
            self.t.build_with_config(cfg=cfg)
            self.gz.build_with_config(cfg=cfg)
            self.div.build_with_config(cfg=cfg)
            self.vor.build_with_config(cfg=cfg)
            self.ut.build_with_config(cfg=cfg)
            self.pv.build_with_config(cfg=cfg)
            self.ke.build_with_config(cfg=cfg)
            self.rhod.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.mg.prepare_block(block)
        block = self.dmg.prepare_block(block)
        block = self.mfn.prepare_block(block)
        block = self.mft.prepare_block(block)
        block = self.dmf.prepare_block(block)
        block = self.ph.prepare_block(block)
        block = self.p.prepare_block(block)
        block = self.t.prepare_block(block)
        block = self.gz.prepare_block(block)
        block = self.div.prepare_block(block)
        block = self.vor.prepare_block(block)
        block = self.ut.prepare_block(block)
        block = self.pv.prepare_block(block)
        block = self.ke.prepare_block(block)
        block = self.rhod.prepare_block(block)
        return block


class ExSpaceOperators(
    types.ExVifBaroclinicEquationSet4UGRIDSpaceOperators[
        TendencyOfDryAirMassAtSurface,
        TendencyOfPotentialTemperatureDueToDivergenceOfFlux,
        TendencyOfPotentialTemperatureDueToVerticalAdvection,
        TendencyOfNormalVelocityDueToNonlinearCoriolisTermAtDualEdge,
        TendencyOfNormalVelocityDueToGradientOfKineticEnergyAtDualEdge,
        TendencyOfNormalVelocityDueToVerticalAdvection,
        TendencyOfNormalVelocityDueToPressureGradientForce,
        TendencyOfTracerDueToDivergenceOfTracerFlux,
        TendencyOfTracerDueToVerticalAdvection,
        FiniteVolumeMethod,
        ExPrepareSpaceOperators,
    ]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.mfz_lev = VerticalMassFluxAtHalfLev(
            disc=disc, method=method, mode=mode
        )
        if hasattr(self, "dqdt"):
            self.set_bg_wind = SetWindForBackground(
                disc=disc, method=method, mode=mode
            )
            self.accum_wind = AccumWind(
                disc=disc, method=method, mode=mode
            )

    def __call__(
        self,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        star_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
        pass_tag: str,
        explicit: bool
    ) -> BLOCK:
        block = block.reset(pass_tag=pass_tag)
        if pass_tag == "all_pass":
            # ==================================================
            # Prepare state and block for tendency calculation
            block = self.prepare(
                block=block,
                star_idx=star_idx,
                pass_tag=pass_tag,
                explicit=explicit
            )
            # ==================================================
            # Calculate tendencies
            block = self.dmgsdt(block, star_idx, dt, substep)
            block = self.mfz_lev(block, star_idx)
            block = self.dptdt(block, star_idx, dt, substep)
            block = self.dundt(block, star_idx, dt, substep)
            # ==================================================
        elif pass_tag == "forward_pass":
            # ==================================================
            # Prepare state and block for tendency calculation
            block = self.prepare(
                block=block,
                star_idx=star_idx,
                pass_tag=pass_tag,
                explicit=explicit
            )
            # ==================================================
            # Calculate tendencies
            block = self.dmgsdt(block, star_idx, dt, substep)
            block = self.mfz_lev(block, star_idx)
            block = self.dptdt(block, star_idx, dt, substep)
            # ==================================================
        elif pass_tag == "backward_pass":
            # ==================================================
            # Prepare state and block for tendency calculation
            block = self.prepare(
                block=block,
                star_idx=new_idx,
                pass_tag=pass_tag,
                explicit=explicit
            )
            # ==================================================
            # Calculate tendencies
            block = self.dundt.grad_ke(block, star_idx, dt, substep)
            block = self.dundt.pgf(block, new_idx, dt, substep)
            block = self.dundt.coriolis(block, star_idx, dt, substep)
            block = self.dundt.vadv(block, star_idx, dt, substep)
            # ==================================================
        else:
            raise ValueError(
                f"Unsupported pass tag {pass_tag}!"
            )
        return block

    def advection_stepper(self, block: BLOCK, explicit: bool) -> BLOCK:
        if not self.mode.ideal_dry_core:
            block = self.tracer_run(block, explicit=explicit)
        return block

    def diagnose(
        self,
        block: BLOCK,
        itime_str: str,
    ) -> BLOCK:
        itime = block.get_time_idx(itime_str=itime_str)
        block = self.prepare.div(block, itime)
        block = block.diagnose(itime_str=itime_str)
        return block

    def hvsplit_method(self, method: str = "V-H") -> Self:
        if method not in ["H-V", "V-H"]:
            raise ValueError(f"Unsupported hvsplit method: {method}")
        self._hvsplit_method = method
        return self

    def _update_mgs(
        self,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: float,
        pass_tag: str
    ) -> BLOCK:
        if pass_tag == "all_pass" or pass_tag == "forward_pass":
            mesh = block.coord.mesh
            ids, ide = mesh.polg_ids, mesh.polg_ide
            mgs = block.state.mgs.unwrap()
            old_mgs = mgs[old_idx, :]
            new_mgs = mgs[new_idx, :]
            dmgsdt = block.tend.dmgsdt.unwrap()
            # ==================================================
            # Update mgs
            new_mgs = new_mgs.at[ids:ide+1].set(
                old_mgs[ids:ide+1]
                + dmgsdt[ids:ide+1] * dt
            )
            # ==================================================
            # Wrap to new state and update dmg
            block = tree_at(
                coord=block.coord,
                where=lambda b: b.state.mgs,
                pytree=block,
                replace=new_mgs,
                at_dims=0,
                at_idxes=new_idx,
                fill_halo=True
            )
            # ==================================================
            # Update dmg
            block = self.prepare.mg(block, new_idx)
            block = self.prepare.dmg(block, new_idx)
            return block
        elif pass_tag == "backward_pass":
            return block
        else:
            raise ValueError(
                f"Unsupported pass tag {pass_tag}!"
            )

    def _update_pt(
        self,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: float,
        pass_tag: str
    ) -> BLOCK:
        if pass_tag == "all_pass" or pass_tag == "forward_pass":
            mesh = block.coord.mesh
            vert = block.coord.vert
            ids, ide = mesh.polg_ids, mesh.polg_ide
            kds, kde = vert.full_kds, vert.full_kde
            old_pt = block.state.pt.unwrap()[old_idx, :, :]
            new_pt = block.state.pt.unwrap()[new_idx, :, :]
            old_dmg = block.state.dmg.unwrap()[old_idx, :, :]
            new_dmg = block.state.dmg.unwrap()[new_idx, :, :]
            dptdt = block.tend.dptdt.unwrap()
            # ==================================================
            new_pt = new_pt.at[kds:kde+1, ids:ide+1].set(
                (
                    old_pt[kds:kde+1, ids:ide+1]
                    * old_dmg[kds:kde+1, ids:ide+1]
                    + dptdt[kds:kde+1, ids:ide+1] * dt
                )/new_dmg[kds:kde+1, ids:ide+1]
            )
            # ==================================================
            block = tree_at(
                coord=block.coord,
                where=lambda b: b.state.pt,
                pytree=block,
                replace=new_pt,
                at_dims=0,
                at_idxes=new_idx,
                fill_halo=True
            )
            return block
        elif pass_tag == "backward_pass":
            return block
        else:
            raise ValueError(
                f"Unsupported pass tag {pass_tag}!"
            )

    def _update_un(
        self,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: float,
        pass_tag: str
    ) -> BLOCK:
        if pass_tag == "all_pass" or pass_tag == "backward_pass":
            mesh = block.coord.mesh
            vert = block.coord.vert
            ids, ide = mesh.edge_ids, mesh.edge_ide
            kds, kde = vert.full_kds, vert.full_kde
            old_un = block.state.un.unwrap()[old_idx, :, :]
            new_un = block.state.un.unwrap()[new_idx, :, :]
            dundt = block.tend.dundt.unwrap()
            # ==================================================
            new_un = new_un.at[kds:kde+1, ids:ide+1].set(
                old_un[kds:kde+1, ids:ide+1] + dundt[kds:kde+1, ids:ide+1] * dt
            )
            # ==================================================
            block = tree_at(
                coord=block.coord,
                where=lambda b: b.state.un,
                pytree=block,
                replace=new_un,
                at_dims=0,
                at_idxes=new_idx,
                fill_halo=True
            )
            return block
        elif pass_tag == "forward_pass":
            return block
        else:
            raise ValueError(
                f"Unsupported pass tag {pass_tag}!"
            )

    def update_state(
        self,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: float,
        substep: int,
        pass_tag: str,
    ) -> BLOCK:
        # ================================================
        # Update mgs
        block = self._update_mgs(
            block=block,
            old_idx=old_idx,
            new_idx=new_idx,
            dt=dt,
            pass_tag=pass_tag
        )
        # ================================================
        # Update pt
        block = self._update_pt(
            block=block,
            old_idx=old_idx,
            new_idx=new_idx,
            dt=dt,
            pass_tag=pass_tag
        )
        # ================================================
        # Update un
        block = self._update_un(
            block=block,
            old_idx=old_idx,
            new_idx=new_idx,
            dt=dt,
            pass_tag=pass_tag
        )
        # ================================================
        return block

    def config2dict(self) -> Dict[str, Any]:
        self._cfg.update(self.mfz_lev.config2dict())
        self._cfg.update(self.dmgsdt.config2dict())
        self._cfg.update(self.dptdt.config2dict())
        self._cfg.update(self.dundt.config2dict())
        if hasattr(self, "dqdt"):
            self._cfg.update(self.dqdt.config2dict())
            self._cfg.update(self.set_bg_wind.config2dict())
            self._cfg.update(self.accum_wind.config2dict())
        self._cfg.update(self.prepare.config2dict())
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.hvsplit_method()
            self.mfz_lev.build_with_config(cfg=cfg)
            self.dmgsdt.build_with_config(cfg=cfg)
            self.dptdt.build_with_config(cfg=cfg)
            self.dundt.build_with_config(cfg=cfg)
            if hasattr(self, "dqdt"):
                self.dqdt.build_with_config(cfg=cfg)
                self.set_bg_wind.build_with_config(cfg=cfg)
                self.accum_wind.build_with_config(cfg=cfg)
            self.prepare.build_with_config(cfg=cfg)
            self.tracer_run = TracerRun3D(space_operators=self)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.dmgsdt.prepare_block(block)
        block = self.dptdt.prepare_block(block)
        block = self.dundt.prepare_block(block)
        if hasattr(self, "dqdt"):
            block = self.dqdt.prepare_block(block)
            block = self.set_bg_wind.prepare_block(block)
            block = self.accum_wind.prepare_block(block)
        block = self.mfz_lev.prepare_block(block)
        block = self.prepare.prepare_block(block)
        return block
