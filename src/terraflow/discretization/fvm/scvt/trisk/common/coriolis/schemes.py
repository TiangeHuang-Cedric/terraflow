from __future__ import annotations
from typing import Any, Dict, Self, TypeVar, TYPE_CHECKING

from terraflow.operator import FiniteVolumeMethod
from terraflow.discretization.abstract import (
    EquationMode,
    AbstractDynamicalEquationOperator,
    DiscretizationInfo,
)
from terraflow.discretization.fvm.scvt.trisk.common.coriolis import trisk

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.util.model_config import ModelConfig
    from terraflow.discretization.fvm.scvt.trisk.single_lev.dynamics_types import Block as SingleLevBlock
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as HybridLevBlock

    Int = typing.Int
    Array = typing.Array

    BLOCK = TypeVar("BLOCK", SingleLevBlock, HybridLevBlock)

DynOperator = AbstractDynamicalEquationOperator[FiniteVolumeMethod]


class CoriolisForceAtEdge(DynOperator):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        return self.scheme(block, star_idx)

    def trisk(self) -> trisk.TriSKCoriolisForceAtEdge:
        self.scheme = trisk.TriSKCoriolisForceAtEdge(
            disc=self.disc, method=self.method, mode=self.mode
        )
        self._cfg["coriolis_scheme"] = "trisk"
        self._built = True
        return self.scheme

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self._built:
            if cfg.coriolis_scheme == "trisk":
                self.trisk()
            else:
                raise ValueError(
                    f"Unsupported Coriolis force scheme {cfg.coriolis_scheme}."
                )
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.scheme.prepare_block(block)
