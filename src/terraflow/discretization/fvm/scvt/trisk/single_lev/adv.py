from __future__ import annotations
from typing import Any, Dict, Self, TYPE_CHECKING
import equinox as eqx

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.discretization.abstract import AbstractPrepareSpaceOperators
from terraflow.discretization.fvm.scvt.trisk.common.adv.term import (
    SetWindForBackground,
    AccumWind,
    TendencyOfMassDueToDivergenceOfMassFlux,
    TendencyOfTracerDueToDivergenceOfTracerFlux
)
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at
from terraflow.dynamics.adv import types


if TYPE_CHECKING:
    from terraflow.util.model_config import ModelConfig
    from terraflow.discretization.abstract import DiscretizationInfo, EquationMode
    from .dynamics_types import Block

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

    def first_call(self, block: Block) -> Block:
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
        block: Block,
        star_idx: Int[Array, ""] | int,
        explicit: bool
    ) -> Block:
        return block

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block


class TracerRun2D:
    def __init__(self, space_operators: ExSpaceOperators):
        self.space_operators = space_operators

    def __call__(self, block: Block, explicit: bool) -> Block:
        # ==========================================
        block = self.d2a_coupling(block=block)
        # ==========================================
        block = self.space_operators.dqdt.dtf(block=block)
        block = block.set_new_tracer_mass4batches()
        block = block.swap_batches_time_indices()
        # ==========================================
        # Set new mass for all batches
        block = block.set_new_mass4batches()
        # Update istep for all batches
        block = block.update_batches_istep()
        # ==========================================
        return block

    def d2a_coupling(self, block: Block) -> Block:
        block = self.space_operators.set_bg_wind(block, block.old)
        block = self.space_operators.accum_wind(block)
        return block


class ExSpaceOperators(
    types.ExVif2DAdvectionEquationSet4UGRIDSpaceOperators[
        TendencyOfMassDueToDivergenceOfMassFlux,
        TendencyOfTracerDueToDivergenceOfTracerFlux,
        FiniteVolumeMethod,
        ExPrepareSpaceOperators
    ]
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

    def __call__(
        self,
        block: Block,
        old_idx: Int[Array, ""] | int,
        star_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
        pass_tag: str,
        explicit: bool,
    ) -> Block:
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
        return block

    def advection_stepper(self, block: Block, explicit: bool) -> Block:
        block = block.testcase_set_uv(block.old)
        return self.tracer_run(block=block, explicit=explicit)

    def diagnose(
        self,
        block: Block,
        itime_str: str,
    ) -> Block:
        block = block.diagnose(itime_str=itime_str)
        return block

    def update_state(
        self,
        block: Block,
        old_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: float,
        substep: int,
        pass_tag: str,
    ) -> Block:
        return block

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.dmdt.config2dict()
        cfg.update(self.dqdt.config2dict())
        cfg.update(self.set_bg_wind.config2dict())
        cfg.update(self.accum_wind.config2dict())
        cfg.update(self.prepare.config2dict())
        self._cfg.update(cfg)
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.dmdt.build_with_config(cfg=cfg)
            self.dqdt.build_with_config(cfg=cfg)
            self.set_bg_wind.build_with_config(cfg=cfg)
            self.accum_wind.build_with_config(cfg=cfg)
            self.prepare.build_with_config(cfg=cfg)
            self.tracer_run = TracerRun2D(space_operators=self)
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        block = self.dmdt.prepare_block(block)
        block = self.dqdt.prepare_block(block)
        block = self.set_bg_wind.prepare_block(block)
        block = self.accum_wind.prepare_block(block)
        block = self.prepare.prepare_block(block)
        return block
