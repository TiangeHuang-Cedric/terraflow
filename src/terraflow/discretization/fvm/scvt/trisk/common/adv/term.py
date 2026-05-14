from __future__ import annotations
from typing import Any, Dict, Self, TypeVar, TYPE_CHECKING
import jax
import equinox as eqx

from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.scvt import operators as mesh_op
from terraflow.discretization.abstract import (
    AbstractDynamicalEquationOperator,
    AbstractAdvectionEquationOperator
)
from terraflow.discretization.fvm.scvt.trisk.common.adv.schemes import (
    HorizontalAdvectionSchemeAtCell,
    VerticalAdvectionScheme
)
from terraflow.discretization.fvm.scvt.trisk.common.adv.flux_limiter import (
    HorizontalLimiterAtCell,
    VerticalLimiterAtCell
)
from terraflow.discretization.fvm.scvt.trisk.hybrid_lev import operators as hybrid_op
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at
from terraflow.dynamics.adv import types
from terraflow.solver.base import ex_rk3_advection_stepper


if TYPE_CHECKING:
    from terraflow.util import typing, model_config
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    from terraflow.discretization.abstract import DiscretizationInfo, EquationMode
    from terraflow.discretization.fvm.scvt.trisk.common.adv.batch import AdvBatchType
    from terraflow.discretization.fvm.scvt.trisk.single_lev.dynamics_types import Block as SingleLevBlock
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as HybridLevBlock

    Float = typing.Float
    Int = typing.Int
    Array = typing.Array
    CoordSys = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        OneLayerVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        OneLayerVertCoordLayout
    ]

    BLOCK = TypeVar("BLOCK", SingleLevBlock, HybridLevBlock)


DynOperator = AbstractDynamicalEquationOperator[FiniteVolumeMethod]

AdvOperator = AbstractAdvectionEquationOperator[FiniteVolumeMethod]


class MassFluxAtEdge(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.adv_scheme = HorizontalAdvectionSchemeAtCell()

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        coord = block.coord
        batch = block.adv_batch_bg
        m = block.state.dmg.unwrap()[star_idx, ...]
        # ===================================
        # Calculate mass flux at edge
        batch = self.adv_scheme.calc_mass_hflx(
            m=m,
            batch=batch,
            coord=coord,
            aux=block.equaux.adv,
        )
        # ===================================
        return eqx.tree_at(
            where=lambda b: b.adv_batch_bg,
            pytree=block,
            replace=batch,
        )

    def upwind(
        self,
        order: int = 3,
        optimized: bool = True
    ) -> Self:
        self.adv_scheme.upwind(order=order, optimized=optimized)
        self._cfg = {
            "bg_adv_scheme": "upwind",
            "upwind_order_bg": order,
            "upwind_optimized_bg": int(optimized)
        }
        return self

    def config2dict(self) -> Dict[str, Any]:
        self._cfg.update(
            self.adv_scheme.config2dict()
        )
        return self._cfg

    def build_with_config(self, cfg: model_config.ModelConfig) -> Self:
        if not self.has_built():
            self.adv_scheme.build_with_config(
                target="bg",
                cfg=cfg,
            )
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.adv_scheme.prepare_block(block)


class SetWindForBackground(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, itime: Int[Array, ""] | int) -> BLOCK:
        if self.mode.is_2d_model():
            un = block.state.un.unwrap()[itime, ...]
            m = block.state.dmg.unwrap()[itime, ...]
            return eqx.tree_at(
                lambda b: (
                    b.adv_batch_bg.un.data,
                    b.adv_batch_bg.m.data,
                ),
                block,
                (un, m)
            )
        elif self.mode.is_3d_model():
            un = block.state.un.unwrap()[itime, ...]
            m = block.state.dmg.unwrap()[itime, ...]
            mfz_lev = block.aux.mfz_lev.unwrap()
            return eqx.tree_at(
                lambda b: (
                    b.adv_batch_bg.un.data,
                    b.adv_batch_bg.m.data,
                    b.adv_batch_bg.mfz.data
                ),
                block,
                (un, m, mfz_lev)
            )
        else:
            raise NotImplementedError(
                f"Unsupported model mode: {self.mode}"
            )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: model_config.ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class AccumWind(DynOperator):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK) -> BLOCK:
        if self.mode.is_2d_model():
            m = block.adv_batch_bg.m.unwrap()
            mfn = block.adv_batch_bg.mfn.unwrap()
            for batch_name, batch in block.adv_batches.items():
                batch = batch.accum_wind(
                    coord=block.coord,
                    m=m,
                    mfn=mfn,
                )
                block = eqx.tree_at(
                    lambda b: b.adv_batches[batch_name],
                    block,
                    batch
                )
        elif self.mode.is_3d_model():
            m = block.adv_batch_bg.m.unwrap()
            mfn = block.adv_batch_bg.mfn.unwrap()
            mfz = block.adv_batch_bg.mfz.unwrap()
            for batch_name, batch in block.adv_batches.items():
                batch = batch.accum_wind(
                    coord=block.coord,
                    m=m,
                    mfn=mfn,
                    mfz=mfz,
                )
                block = eqx.tree_at(
                    lambda b: b.adv_batches[batch_name],
                    block,
                    batch
                )
        else:
            raise NotImplementedError(
                f"Unsupported model mode: {self.mode}"
            )
        return block

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: model_config.ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class TendencyOfMassDueToDivergenceOfMassFlux(
    types.DivergenceOfMassFluxProtocol[FiniteVolumeMethod]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.mfn = MassFluxAtEdge(disc=disc, method=method, mode=mode)

    def __call__(
        self,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        star_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
        pass_tag: str
    ) -> BLOCK:
        coord = block.coord
        para = block.para
        m_old = block.state.dmg.unwrap()[old_idx, :, :]
        # ===================================
        block = self.mfn(block=block, star_idx=star_idx)
        mfn = block.adv_batch_bg.mfn.unwrap()
        dmf = mesh_op.div_at_primal_cell(
            x_at_primal_edge=mfn,
            coord=coord,
            para=para,
        )
        m_new = m_old - dt * dmf
        # ===================================
        return tree_at(
            coord=coord,
            where=lambda b: b.state.dmg,
            pytree=block,
            replace=m_new,
            at_dims=0,
            at_idxes=new_idx
        )

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.mfn.config2dict()
        return self._update_config(cfg=cfg)

    def build_with_config(self, cfg: model_config.ModelConfig) -> Self:
        if not self.has_built():
            self.mfn.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.mfn.prepare_block(block)


class TendencyOfMassFluxDueToVerticalAdvection(
    types.VerticalAdvectionOfMassFluxProtocol[FiniteVolumeMethod]
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
        block: HybridLevBlock,
        old_idx: Int[Array, ""] | int,
        star_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
        pass_tag: str
    ) -> HybridLevBlock:
        coord = block.coord
        mesh = coord.mesh
        vert = coord.vert
        ids, ide = mesh.polg_ids, mesh.polg_ide
        kds, kde = vert.half_kds, vert.half_kde
        mfz_lev = block.aux.mfz_lev.unwrap()
        m_old = block.state.dmg.unwrap()[old_idx, kds:kde+1, ids:ide+1]
        # ===================================
        dmdt = hybrid_op.delta_at_full_lev(
            coord=coord,
            x_lev=mfz_lev,
        )
        m_new = m_old - dt * dmdt
        # ===================================
        return tree_at(
            coord=coord,
            where=lambda b: b.state.dmg,
            pytree=block,
            replace=m_new,
            at_dims=0,
            at_idxes=new_idx
        )

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: model_config.ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: HybridLevBlock) -> HybridLevBlock:
        return block


class TendencyOfTracerDueToDivergenceOfTracerFlux(
    types.DivergenceOfTracerFluxProtocol[FiniteVolumeMethod]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc, method, mode)
        self.adv_scheme = HorizontalAdvectionSchemeAtCell()
        self.limiter = HorizontalLimiterAtCell()

    def __call__(self, block: BLOCK,) -> BLOCK:
        return self.call_fun(block=block)

    def calc_at_batch(
        self,
        batch: AdvBatchType,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        star_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int
    ) -> tuple[AdvBatchType, BLOCK]:
        coord = block.coord
        idx = batch.idx
        q_old = block.tracers.q.unwrap()[old_idx, idx, ...]
        q_star = block.tracers.q.unwrap()[star_idx, idx, ...]
        q_new = block.tracers.q.unwrap()[new_idx, ...]
        mq_old = block.tracers.mq.unwrap()[old_idx, idx, ...]
        # ==========================================
        # Run tracers
        q_new_, batch = self._calc_at_tracers(
            q=q_star,
            q_old=q_old,
            mq_old=mq_old,
            batch=batch,
            block=block,
            dt=dt,
            substep=substep
        )
        q_new = q_new.at[idx, :, :].set(q_new_)
        block = tree_at(
            coord=coord,
            where=lambda b: b.tracers.q,
            pytree=block,
            replace=q_new,
            at_dims=0,
            at_idxes=new_idx
        )
        # ==========================================
        return batch, block

    def _calc_at_tracers(
        self,
        q: Float[Array, "ntracers nlev ncell"],
        q_old: Float[Array, "ntracers nlev ncell"],
        mq_old: Float[Array, "ntracers nlev ncell"],
        batch: AdvBatchType,
        block: BLOCK,
        dt: Float[Array, ""] | float,
        substep: int
    ) -> tuple[Float[Array, "ntracers nlev nedges"], AdvBatchType]:

        def body_fn(
            carry_batch: AdvBatchType,
            xs: tuple[
                Float[Array, "nlev ncell"],
                Float[Array, "nlev ncell"],
                Float[Array, "nlev ncell"]
            ]
        ) -> tuple[AdvBatchType, Float[Array, "nlev nedges"]]:
            q_in, q_old_in, mq_old_in = xs
            q_new, carry_batch = self._calc_at_single_tracer(
                q=q_in,
                q_old=q_old_in,
                mq_old=mq_old_in,
                batch=carry_batch,
                block=block,
                dt=dt,
                substep=substep
            )
            return carry_batch, q_new

        batch, q_new = jax.lax.scan(
            f=body_fn,
            init=batch,
            xs=(q, q_old, mq_old)
        )
        return q_new, batch

    def _calc_at_single_tracer(
        self,
        q: Float[Array, "nlev npoints"],
        q_old: Float[Array, "nlev npoints"],
        mq_old: Float[Array, "nlev npoints"],
        batch: AdvBatchType,
        block: BLOCK,
        dt: Float[Array, ""] | float,
        substep: int
    ) -> tuple[Float[Array, "nlev nedges"], AdvBatchType]:
        coord = block.coord
        mesh = coord.mesh
        vert = coord.vert
        ids, ide = mesh.polg_ids, mesh.polg_ide
        kds, kde = vert.full_kds, vert.full_kde
        para = block.para
        aux = block.equaux.adv
        mq_old_ = mq_old[kds:kde+1, ids:ide+1]
        m_new = block.adv_batch_bg.m.unwrap()[kds:kde+1, ids:ide+1]
        # ==========================================
        batch = self.adv_scheme.calc_tracer_hflx(
            q=q,
            batch=batch,
            coord=coord,
            aux=aux
        )
        batch = self.limiter(
            q=q_old,
            mq_old=mq_old,
            batch=batch,
            block=block,
            dt=dt,
            substep=substep
        )
        dqdt = mesh_op.div_at_primal_cell(
            x_at_primal_edge=batch.qmfn.unwrap(),
            coord=coord,
            para=para,
        )
        # ==========================================
        q_new = (mq_old_ - dt * dqdt) / m_new
        return q_new, batch

    def select_limiter(
        self,
        limiter_type: str = "fct",
        fct_regression: bool = True
    ) -> Self:
        if limiter_type == "fct":
            self.limiter.fct(regression=fct_regression)
            self._cfg = {
                "limiter_type": "fct",
                "fct_regression": int(fct_regression)
            }
        elif limiter_type == "none":
            self.limiter.no_limiter()
            self._cfg = {
                "limiter_type": "none"
            }
        else:
            raise NotImplementedError(
                f"Unsupported limiter type: {limiter_type}"
            )
        return self

    def rk3_upwind(
        self,
        order: int = 3,
        optimized: bool = True,
        limiter_type: str = "fct",
        fct_regression: bool = True,
    ) -> Self:
        self.adv_scheme.upwind(order=order, optimized=optimized)
        self.select_limiter(
            limiter_type=limiter_type,
            fct_regression=fct_regression
        )
        self._cfg = {
            "adv_scheme": "upwind",
            "upwind_order": order,
            "upwind_optimized": int(optimized),
        }
        self.call_fun = ex_rk3_advection_stepper(
            space_operators=self.calc_at_batch
        )
        return self

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.adv_scheme.config2dict()
        self._update_config(cfg=cfg)
        cfg = self.limiter.config2dict()
        self._update_config(cfg=cfg)
        return self._cfg

    def build_with_config(self, cfg: model_config.ModelConfig) -> Self:
        if not self.has_built():
            if not self.adv_scheme.has_built() and not self.limiter.has_built():
                if cfg.adv_scheme == "upwind":
                    self.rk3_upwind(
                        order=cfg.upwind_order_h,
                        optimized=cfg.upwind_optimized,
                        limiter_type=cfg.limiter_type,
                        fct_regression=cfg.fct_regression_h
                    )
                    self.limiter.build_with_config(cfg=cfg)
            else:
                if self.adv_scheme.has_built() and self.adv_scheme.scheme_name == "upwind":
                    if cfg.limiter_type == "fct":
                        self.select_limiter(
                            limiter_type=cfg.limiter_type,
                            fct_regression=cfg.fct_regression_h
                        )
                        self.call_fun = ex_rk3_advection_stepper(
                            space_operators=self.calc_at_batch
                        )
                        self.limiter.build_with_config(cfg=cfg)
                    else:
                        raise NotImplementedError(
                            f"Unsupported limiter type: {cfg.limiter_type}"
                        )
                if self.limiter.has_built() and self.limiter.scheme_name == "fct":
                    if cfg.adv_scheme == "upwind":
                        self.adv_scheme.upwind(
                            order=cfg.upwind_order_h,
                            optimized=cfg.upwind_optimized
                        )
                        self.call_fun = ex_rk3_advection_stepper(
                            space_operators=self.calc_at_batch
                        )
                        self.limiter.build_with_config(cfg=cfg)
                    else:
                        raise NotImplementedError(
                            f"Unsupported advection scheme type: {cfg.adv_scheme}"
                        )
                if self.adv_scheme.has_built() and self.limiter.has_built():
                    if (
                        self.adv_scheme.scheme_name == "upwind"
                        and (
                            self.limiter.scheme_name == "fct" or self.limiter.scheme_name == "none"
                        )
                    ):
                        self.limiter.build_with_config(cfg=cfg)
                    else:
                        raise NotImplementedError(
                            f"Unsupported combination of advection scheme and limiter:"
                            f" {self.adv_scheme.scheme_name} + {self.limiter.scheme_name}"
                        )
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.adv_scheme.prepare_block(block)
        block = self.limiter.prepare_block(block)
        return block


class TendencyOfTracerDueToVerticalAdvection(
    types.VerticalAdvectionOfTracerProtocol[FiniteVolumeMethod]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.adv_scheme = VerticalAdvectionScheme()
        self.limiter = VerticalLimiterAtCell()

    def __call__(self, block: HybridLevBlock) -> HybridLevBlock:
        return self.call_fun(block=block)

    def calc_at_batch(
        self,
        batch: AdvBatchType,
        block: HybridLevBlock,
        old_idx: Int[Array, ""] | int,
        star_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> tuple[AdvBatchType, HybridLevBlock]:
        coord = block.coord
        idx = batch.idx
        q_old = block.tracers.q.unwrap()[old_idx, idx, ...]
        q_star = block.tracers.q.unwrap()[star_idx, idx, ...]
        q_new = block.tracers.q.unwrap()[new_idx, ...]
        mq_old = block.tracers.mq.unwrap()[old_idx, idx, ...]
        # ===========================================
        # Run tracers
        q_new_, batch = self._calc_at_tracers(
            q=q_star,
            q_old=q_old,
            mq_old=mq_old,
            batch=batch,
            block=block,
            dt=dt,
            substep=substep
        )
        q_new = q_new.at[idx, :, :].set(q_new_)
        block = tree_at(
            coord=coord,
            where=lambda b: b.tracers.q,
            pytree=block,
            replace=q_new,
            at_dims=0,
            at_idxes=new_idx
        )
        # ==========================================
        return batch, block

    def _calc_at_tracers(
        self,
        q: Float[Array, "ntracers nlev ncell"],
        q_old: Float[Array, "ntracers nlev ncell"],
        mq_old: Float[Array, "ntracers nlev ncell"],
        batch: AdvBatchType,
        block: HybridLevBlock,
        dt: Float[Array, ""] | float,
        substep: int
    ) -> tuple[Float[Array, "ntracers nlev nedges"], AdvBatchType]:
        def body_fn(
            carry_batch: AdvBatchType,
            xs: tuple[
                Float[Array, "nlev ncell"],
                Float[Array, "nlev ncell"],
                Float[Array, "nlev ncell"]
            ]
        ) -> tuple[AdvBatchType, Float[Array, "nlev nedges"]]:
            q_in, q_old_in, mq_old_in = xs
            q_new, carry_batch = self._calc_at_single_tracer(
                q=q_in,
                q_old=q_old_in,
                mq_old=mq_old_in,
                batch=carry_batch,
                block=block,
                dt=dt,
                substep=substep
            )
            return carry_batch, q_new

        batch, q_new = jax.lax.scan(
            f=body_fn,
            init=batch,
            xs=(q, q_old, mq_old)
        )
        return q_new, batch

    def _calc_at_single_tracer(
        self,
        q: Float[Array, "nlev npoints"],
        q_old: Float[Array, "nlev npoints"],
        mq_old: Float[Array, "nlev npoints"],
        batch: AdvBatchType,
        block: HybridLevBlock,
        dt: Float[Array, ""] | float,
        substep: int
    ) -> tuple[Float[Array, "nlev nedges"], AdvBatchType]:
        coord = block.coord
        mesh = coord.mesh
        vert = coord.vert
        ids, ide = mesh.polg_ids, mesh.polg_ide
        kds, kde = vert.half_kds, vert.half_kde
        mq_old_ = mq_old[kds:kde+1, ids:ide+1]
        m_new = block.adv_batch_bg.m.unwrap()[kds:kde+1, ids:ide+1]
        # ==========================================
        batch = self.adv_scheme.calc_tracer_vflx(
            q=q,
            batch=batch,
            coord=coord
        )
        batch = self.limiter(
            q=q_old,
            mq_old=mq_old_,
            batch=batch,
            block=block,
            dt=dt,
            substep=substep
        )
        dqdt = hybrid_op.delta_at_full_lev(
            coord=coord,
            x_lev=batch.qmfz.unwrap()
        )
        # ==========================================
        q_new = (mq_old_ - dt * dqdt) / m_new
        return q_new, batch

    def select_limiter(
        self,
        limiter_type: str = "fct",
        fct_regression: bool = True
    ) -> Self:
        if limiter_type == "fct":
            self.limiter.fct(regression=fct_regression)
            self._cfg = {
                "limiter_type": "fct",
                "fct_regression_v": int(fct_regression)
            }
        elif limiter_type == "none":
            self.limiter.no_limiter()
            self._cfg = {
                "limiter_type": "none"
            }
        else:
            raise NotImplementedError(
                f"Unsupported limiter type: {limiter_type}"
            )
        return self

    def rk3_upwind(
        self,
        order: int = 3,
        wgt: float = 1.0,
        limiter_type: str = "fct",
        fct_regression: bool = True
    ) -> Self:
        self.adv_scheme.upwind(order=order, wgt=wgt)
        self.select_limiter(
            limiter_type=limiter_type,
            fct_regression=fct_regression
        )
        self._cfg = {
            "adv_scheme": "upwind",
            "upwind_order": order,
            "upwind_wgt": wgt,
        }
        self.call_fun = ex_rk3_advection_stepper(
            space_operators=self.calc_at_batch
        )
        return self

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.adv_scheme.config2dict()
        self._update_config(cfg=cfg)
        cfg = self.limiter.config2dict()
        self._update_config(cfg=cfg)
        return self._cfg

    def build_with_config(self, cfg: model_config.ModelConfig) -> Self:
        if not self.has_built():
            if not self.adv_scheme.has_built() and not self.limiter.has_built():
                if cfg.adv_scheme == "upwind":
                    self.rk3_upwind(
                        order=cfg.upwind_order_v,
                        wgt=cfg.upwind_wgt,
                        limiter_type=cfg.limiter_type,
                        fct_regression=cfg.fct_regression_v
                    )
                    self.limiter.build_with_config(cfg=cfg)
            else:
                if self.adv_scheme.has_built() and self.adv_scheme.scheme_name == "upwind":
                    if cfg.limiter_type == "fct":
                        self.select_limiter(
                            limiter_type=cfg.limiter_type,
                            fct_regression=cfg.fct_regression_v
                        )
                        self.limiter.build_with_config(cfg=cfg)
                    else:
                        raise NotImplementedError(
                            f"Unsupported limiter type: {cfg.limiter_type}"
                        )
                if self.limiter.has_built() and self.limiter.scheme_name == "fct":
                    if cfg.adv_scheme == "upwind":
                        self.adv_scheme.upwind(
                            order=cfg.upwind_order_v, wgt=cfg.upwind_wgt
                        )
                        self.call_fun = ex_rk3_advection_stepper(
                            space_operators=self.calc_at_batch
                        )
                        self.limiter.build_with_config(cfg=cfg)
                    else:
                        raise NotImplementedError(
                            f"Unsupported advection scheme: {cfg.adv_scheme}"
                        )
                if self.adv_scheme.has_built() and self.limiter.has_built():
                    if (
                        self.adv_scheme.scheme_name == "upwind"
                        and (
                            self.limiter.scheme_name == "fct" or self.limiter.scheme_name == "none"
                        )
                    ):
                        self.limiter.build_with_config(cfg=cfg)
                    else:
                        raise NotImplementedError(
                            f"Unsupported combination of advection scheme and limiter: "
                            f"{self.adv_scheme.scheme_name} + {self.limiter.scheme_name}"
                        )
            self._built = True
        return self

    def prepare_block(self, block: HybridLevBlock) -> HybridLevBlock:
        block = self.adv_scheme.prepare_block(block)
        block = self.limiter.prepare_block(block)
        return block
