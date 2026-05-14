from __future__ import annotations
from typing import Any, Dict, Self, TYPE_CHECKING
import equinox as eqx
import jax.numpy as jnp

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.discretization.abstract import (
    AbstractDynamicalEquationOperator,
    AbstractPrepareSpaceOperators
)
from terraflow.discretization.fvm.scvt.trisk.common.adv.term import (
    SetWindForBackground,
    AccumWind,
    TendencyOfMassDueToDivergenceOfMassFlux,
    TendencyOfMassFluxDueToVerticalAdvection,
    TendencyOfTracerDueToDivergenceOfTracerFlux,
    TendencyOfTracerDueToVerticalAdvection
)
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at
from terraflow.dynamics.adv import types
from terraflow.test_case import adv as test_case_adv


DynOperator = AbstractDynamicalEquationOperator[FiniteVolumeMethod]

if TYPE_CHECKING:
    from terraflow.util.model_config import ModelConfig
    from terraflow.discretization.abstract import DiscretizationInfo, EquationMode
    from .term import ExSpaceOperators as BaroclinicExSpaceOperators
    from .dynamics_types import Block as BLOCK

    Float = typing.Float
    Int = typing.Int
    Array = typing.Array


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

    def first_call(self, block: BLOCK) -> BLOCK:
        # ===============================================
        # Prepare background wind and mass for the first time step
        m_old = block.state.dmg.unwrap()[block.old, ...]
        un_old = block.state.un.unwrap()[block.old, ...]
        block = eqx.tree_at(
            lambda b: (
                b.adv_batch_bg.m.data,
                b.adv_batch_bg.un.data
            ),
            block,
            (m_old, un_old),
        )
        # ===============================================
        # Prepare tracer mass for the first time step
        q_old = block.tracers.q.unwrap()[block.old, ...]
        mq_old = m_old[None, ...] * q_old
        block = tree_at(
            coord=block.coord,
            where=lambda b: b.tracers.mq,
            pytree=block,
            replace=mq_old,
            at_dims=0,
            at_idxes=block.old
        )
        # ===============================================
        # Prepare wind and mass for all batches for the first time step
        for batch_name in block.adv_batches.keys():
            block = eqx.tree_at(
                lambda b: (
                    b.adv_batches[batch_name].m.data,
                    b.adv_batches[batch_name].un.data,
                ),
                block,
                (m_old, un_old),
            )
        # ===============================================
        return block

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        explicit: bool
    ) -> BLOCK:
        return block

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class UpdatePressureLevels(DynOperator):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.term import (
            DryAirWeight,
            DeltaDryAirWeight,
            Geopotential
        )
        super().__init__(disc=disc, method=method, mode=mode)
        self.mg = DryAirWeight(disc=disc, method=method, mode=mode)
        self.dmg = DeltaDryAirWeight(disc=disc, method=method, mode=mode)
        self.gz = Geopotential(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, itime: Int[Array, ""] | int) -> BLOCK:
        if block.test_case is None:
            raise ValueError(
                "Test case is required for updating pressure levels."
            )
        if isinstance(block.test_case, test_case_adv.DCMIP12TestCase):
            block = self.dcmip12(
                test_case=block.test_case,
                block=block,
                itime=itime
            )
        return block

    def dcmip12(
        self,
        test_case: test_case_adv.DCMIP12TestCase,
        block: BLOCK,
        itime: Int[Array, ""] | int
    ):
        p0 = test_case.p0
        T0 = test_case.T0
        mgs = block.state.mgs.unwrap()[itime, ...]
        block = tree_at(
            coord=block.coord,
            where=lambda b: b.state.mgs,
            pytree=block,
            replace=jnp.ones_like(mgs) * p0,
            at_dims=0,
            at_idxes=itime
        )
        block = self.mg(block, itime)
        block = self.dmg(block, itime)
        ph_lev = block.state.mg_lev.unwrap()[itime, ...]
        ph = block.state.mg.unwrap()[itime, ...]
        tv = jnp.ones_like(ph) * T0
        block = tree_at(
            coord=block.coord,
            where=lambda b: (b.state.ph_lev, b.state.ph),
            pytree=block,
            replace=(ph_lev, ph),
            at_dims=(0, 0),
            at_idxes=(itime, itime)
        )
        block = tree_at(
            coord=block.coord,
            where=lambda b: b.aux.tv,
            pytree=block,
            replace=tv,
        )
        block = self.gz(block, itime)
        return block

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.mg.config2dict()
        self._cfg.update(cfg=cfg)
        cfg = self.dmg.config2dict()
        self._cfg.update(cfg=cfg)
        cfg = self.gz.config2dict()
        self._cfg.update(cfg=cfg)
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.mg.build_with_config(cfg=cfg)
            self.dmg.build_with_config(cfg=cfg)
            self.gz.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.mg.prepare_block(block)
        block = self.dmg.prepare_block(block)
        block = self.gz.prepare_block(block)
        return block


class TracerRun3D:
    def __init__(self, space_operators: ExSpaceOperators | BaroclinicExSpaceOperators):
        self.space_operators = space_operators

    def __call__(self, block: BLOCK, explicit: bool) -> BLOCK:
        # ==========================================
        block = self.d2a_coupling(block=block)
        # ==========================================
        if self.space_operators._hvsplit_method == "H-V":
            # ======================================
            block = self.space_operators.dqdt.dtf(block=block)
            block = block.set_new_tracer_mass4batches()
            block = block.swap_batches_time_indices()
            # ======================================
            block = self.space_operators.dqdt.vadv(block=block)
            block = block.set_new_tracer_mass4batches()
            block = block.swap_batches_time_indices()
        # ==========================================
        elif self.space_operators._hvsplit_method == "V-H":
            # ======================================
            block = self.space_operators.dqdt.vadv(block=block)
            block = block.set_new_tracer_mass4batches()
            block = block.swap_batches_time_indices()
            # ======================================
            block = self.space_operators.dqdt.dtf(block=block)
            block = block.set_new_tracer_mass4batches()
            block = block.swap_batches_time_indices()
        # ==========================================
        else:
            raise ValueError(
                f"Unsupported hvsplit method: {self.space_operators._hvsplit_method}"
            )
        # ==========================================
        # Set new mass for all batches
        block = block.set_new_mass4batches()
        # Update istep for all batches
        block = block.update_batches_istep()
        # ==========================================
        return block

    def d2a_coupling(self, block: BLOCK) -> BLOCK:
        block = self.space_operators.set_bg_wind(block, block.old)
        block = self.space_operators.accum_wind(block)
        return block


class ExSpaceOperators(
    types.ExVif3DAdvectionEquationSet4UGRIDSpaceOperators[
        TendencyOfMassDueToDivergenceOfMassFlux,
        TendencyOfMassFluxDueToVerticalAdvection,
        TendencyOfTracerDueToDivergenceOfTracerFlux,
        TendencyOfTracerDueToVerticalAdvection,
        FiniteVolumeMethod,
        ExPrepareSpaceOperators
    ],
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.set_bg_wind = SetWindForBackground(
            disc=disc, method=method, mode=mode
        )
        self.accum_wind = AccumWind(
            disc=disc, method=method, mode=mode
        )
        self.update_pressure_levels = UpdatePressureLevels(
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
        explicit: bool,
    ) -> BLOCK:
        block = self.prepare(
            block=block,
            star_idx=star_idx,
            explicit=explicit
        )
        block = self.dmdt.dmf(
            block=block,
            old_idx=old_idx,
            star_idx=star_idx,
            new_idx=new_idx,
            dt=dt,
            substep=substep,
            pass_tag=pass_tag
        )
        block = self.dmdt.vadv(
            block=block,
            old_idx=new_idx,
            star_idx=star_idx,
            new_idx=new_idx,
            dt=dt,
            substep=substep,
            pass_tag=pass_tag
        )
        return block

    def hvsplit_method(self, method: str = "V-H") -> Self:
        if method not in ["H-V", "V-H"]:
            raise ValueError(f"Unsupported hvsplit method: {method}")
        self._hvsplit_method = method
        return self

    def advection_stepper(self, block: BLOCK, explicit: bool) -> BLOCK:
        block = self.update_pressure_levels(block, block.old)
        block = block.testcase_set_uv(block.old)
        return self.tracer_run(block=block, explicit=explicit)

    def diagnose(
        self,
        block: BLOCK,
        itime_str: str,
    ) -> BLOCK:
        block = block.diagnose(itime_str=itime_str)
        return block

    def update_state(
        self,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: float,
        substep: int,
        pass_tag: str,
    ) -> BLOCK:
        return block

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.dmdt.config2dict()
        self._cfg.update(cfg=cfg)
        cfg = self.dqdt.config2dict()
        self._cfg.update(cfg=cfg)
        cfg = self.set_bg_wind.config2dict()
        self._cfg.update(cfg=cfg)
        cfg = self.accum_wind.config2dict()
        self._cfg.update(cfg=cfg)
        cfg = self.prepare.config2dict()
        self._cfg.update(cfg=cfg)
        cfg = self.update_pressure_levels.config2dict()
        self._cfg.update(cfg=cfg)
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.hvsplit_method()
            self.dmdt.build_with_config(cfg=cfg)
            self.dqdt.build_with_config(cfg=cfg)
            self.set_bg_wind.build_with_config(cfg=cfg)
            self.accum_wind.build_with_config(cfg=cfg)
            self.prepare.build_with_config(cfg=cfg)
            self.update_pressure_levels.build_with_config(cfg=cfg)
            self.tracer_run = TracerRun3D(space_operators=self)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.dmdt.prepare_block(block)
        block = self.dqdt.prepare_block(block)
        block = self.set_bg_wind.prepare_block(block)
        block = self.accum_wind.prepare_block(block)
        block = self.prepare.prepare_block(block)
        block = self.update_pressure_levels.prepare_block(block)
        return block
