from __future__ import annotations
from typing import Self, TYPE_CHECKING
import jax


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.discretization.fvm.scvt.trisk.dynamics_types import Block
    from terraflow.discretization.fvm.scvt.trisk.common.adv import batch
    from terraflow.discretization.fvm.scvt.trisk.common.adv.fct import (
        FluxCorrectedTransportHorizontalLimiterAtCell,
        FluxCorrectedTransportVerticalLimiterAtCell
    )

    Int = typing.Int
    Float = typing.Float
    Array = typing.Array


class HorizontalLimiterAtCell:

    def __init__(self):
        self._built = False

    def __call__(
        self,
        q: Float[Array, "nlev ncell"],
        q_old: Float[Array, "nlev ncell"],
        batch: batch.AdvBatchType,
        block: Block,
        dt: Float[Array, ""] | float,
        substep: int
    ) -> batch.AdvBatchType:
        def do_limiter(batch: batch.AdvBatchType) -> batch.AdvBatchType:
            return self.scheme(
                q=q,
                q_old=q_old,
                batch=batch,
                block=block,
                dt=dt
            )

        def not_do_limiter(batch: batch.AdvBatchType) -> batch.AdvBatchType:
            return batch

        return jax.lax.cond(
            pred=substep == batch.nsubsteps - 1,
            true_fun=do_limiter,
            false_fun=not_do_limiter,
            operand=batch
        )

    def fct(self, regression: bool = True) -> FluxCorrectedTransportHorizontalLimiterAtCell:
        from terraflow.discretization.fvm.scvt.trisk.common.adv import fct
        self.scheme = fct.FluxCorrectedTransportHorizontalLimiterAtCell(
            regression=regression
        )
        self.use_limiter = True
        self.scheme_name = "fct"
        self._built = True
        return self.scheme

    def no_limiter(self) -> NoLimiter:
        self.scheme = NoLimiter()
        self.use_limiter = False
        self.scheme_name = "none"
        self._built = True
        return self.scheme

    def has_built(self) -> bool:
        return self._built

    def build_default(self) -> Self:
        if not self._built:
            self.fct().build_default()
        if not self.scheme.has_built():
            self.scheme.build_default()
        return self

    def prepare_block(self, block: Block) -> Block:
        return self.scheme.prepare_block(block)


class VerticalLimiterAtCell:

    def __init__(self):
        self._built = False

    def __call__(
        self,
        q: Float[Array, "nlev ncell"],
        q_old: Float[Array, "nlev ncell"],
        batch: batch.AdvBatchType,
        block: Block,
        dt: Float[Array, ""] | float,
        substep: int
    ) -> batch.AdvBatchType:

        def do_limiter(batch: batch.AdvBatchType) -> batch.AdvBatchType:
            return self.scheme(
                q=q,
                q_old=q_old,
                batch=batch,
                block=block,
                dt=dt
            )

        def not_do_limiter(batch: batch.AdvBatchType) -> batch.AdvBatchType:
            return batch

        return jax.lax.cond(
            pred=substep == batch.nsubsteps - 1,
            true_fun=do_limiter,
            false_fun=not_do_limiter,
            operand=batch
        )

    def fct(self, regression: bool = True) -> FluxCorrectedTransportVerticalLimiterAtCell:
        from terraflow.discretization.fvm.scvt.trisk.common.adv import fct
        self.scheme = fct.FluxCorrectedTransportVerticalLimiterAtCell(
            regression=regression
        )
        self._built = True
        return self.scheme

    def no_limiter(self) -> NoLimiter:
        self.scheme = NoLimiter()
        self._built = True
        return self.scheme

    def has_built(self) -> bool:
        return self._built

    def build_default(self) -> Self:
        if not self._built:
            self.fct()
        if not self.scheme.has_built():
            self.scheme.build_default()
        return self

    def prepare_block(self, block: Block) -> Block:
        return self.scheme.prepare_block(block)


class NoLimiter:

    def __init__(self):
        self._built = False

    def __call__(
        self,
        q_old: Float[Array, "nlev ncell"],
        batch: batch.AdvBatchType,
        block: Block,
        dt: Float[Array, ""] | float,
        *args,
        **kwargs
    ) -> batch.AdvBatchType:
        return batch

    def has_built(self) -> bool:
        return self._built

    def build_default(self) -> Self:
        self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block
