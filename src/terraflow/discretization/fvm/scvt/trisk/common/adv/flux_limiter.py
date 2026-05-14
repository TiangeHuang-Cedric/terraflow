from __future__ import annotations
from typing import Self, Any, TypeVar, TYPE_CHECKING
import jax


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.util.model_config import ModelConfig
    from terraflow.discretization.fvm.scvt.trisk.common.adv import batch
    from terraflow.discretization.fvm.scvt.trisk.common.adv.fct import (
        FluxCorrectedTransportHorizontalLimiterAtCell,
        FluxCorrectedTransportVerticalLimiterAtCell
    )
    from terraflow.discretization.fvm.scvt.trisk.single_lev.dynamics_types import Block as SingleLevBlock
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as HybridLevBlock

    Int = typing.Int
    Float = typing.Float
    Array = typing.Array

    BLOCK = TypeVar("BLOCK", SingleLevBlock, HybridLevBlock)


class HorizontalLimiterAtCell:

    def __init__(self):
        self._cfg: dict[str, Any] = {}
        self._built = False

    def __call__(
        self,
        q: Float[Array, "nlev ncell"],
        mq_old: Float[Array, "nlev ncell"],
        batch: batch.AdvBatchType,
        block: BLOCK,
        dt: Float[Array, ""] | float,
        substep: int
    ) -> batch.AdvBatchType:
        def do_limiter(batch: batch.AdvBatchType) -> batch.AdvBatchType:
            return self.scheme(
                q=q,
                mq_old=mq_old,
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
        self.scheme_name = "fct"
        self._cfg["limiter_type"] = "fct"
        self._cfg["fct_regression_h"] = int(regression)
        self._built = True
        return self.scheme

    def no_limiter(self) -> NoLimiter:
        self.scheme = NoLimiter()
        self.scheme_name = "none"
        self._cfg["limiter_type"] = "none"
        self._built = True
        return self.scheme

    def has_built(self) -> bool:
        return self._built

    def config2dict(self) -> dict[str, Any]:
        self._cfg.update(
            self.scheme.config2dict()
        )
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self._built:
            if cfg.limiter_type == "fct":
                self.fct(regression=cfg.fct_regression_h)
            elif cfg.limiter_type == "none":
                self.no_limiter()
            else:
                raise ValueError(
                    f"Unsupported limiter type: {cfg.limiter_type}"
                )
        if not self.scheme.has_built():
            self.scheme.build_with_config(cfg=cfg)
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.scheme.prepare_block(block)


class VerticalLimiterAtCell:

    def __init__(self):
        self._cfg: dict[str, Any] = {}
        self._built = False

    def __call__(
        self,
        q: Float[Array, "nlev ncell"],
        mq_old: Float[Array, "nlev ncell"],
        batch: batch.AdvBatchType,
        block: BLOCK,
        dt: Float[Array, ""] | float,
        substep: int
    ) -> batch.AdvBatchType:

        def do_limiter(batch: batch.AdvBatchType) -> batch.AdvBatchType:
            return self.scheme(
                q=q,
                mq_old=mq_old,
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
        self.scheme_name = "fct"
        self._cfg["limiter_type"] = "fct"
        self._cfg["fct_regression_v"] = int(regression)
        self._built = True
        return self.scheme

    def no_limiter(self) -> NoLimiter:
        self.scheme = NoLimiter()
        self.scheme_name = "none"
        self._built = True
        return self.scheme

    def has_built(self) -> bool:
        return self._built

    def config2dict(self) -> dict[str, Any]:
        self._cfg.update(
            self.scheme.config2dict()
        )
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self._built:
            if cfg.limiter_type == "fct":
                self.fct(regression=cfg.fct_regression_v)
            elif cfg.limiter_type == "none":
                self.no_limiter()
            else:
                raise ValueError(
                    f"Unsupported limiter type: {cfg.limiter_type}"
                )
        if not self.scheme.has_built():
            self.scheme.build_with_config(cfg=cfg)
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.scheme.prepare_block(block)


class NoLimiter:

    def __init__(self):
        self._built = False

    def __call__(
        self,
        q: Float[Array, "nlev ncell"],
        mq_old: Float[Array, "nlev ncell"],
        batch: batch.AdvBatchType,
        block: BLOCK,
        dt: Float[Array, ""] | float,
        *args,
        **kwargs
    ) -> batch.AdvBatchType:
        return batch

    def has_built(self) -> bool:
        return self._built

    def config2dict(self) -> dict[str, Any]:
        return {}

    def build_with_config(self, cfg: ModelConfig) -> Self:
        self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block
