from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Self

from terraflow.operator import FiniteVolumeMethod
from terraflow.discretization.abstract import (
    EquationMode,
    DiscretizationInfo,
    AbstractDynamicalEquationOperator
)
from terraflow.discretization.fvm.scvt.trisk.common.pgf import ptb

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.util.model_config import ModelConfig
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dry.hdc.dynamics_types import Block

    Int = typing.Int
    Array = typing.Array


DynOperator = AbstractDynamicalEquationOperator[FiniteVolumeMethod]


class PressureGradientForce(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: Block, star_idx: Int[Array, ""] | int) -> Block:
        return self.scheme(block, star_idx)

    def ptb(self) -> ptb.PerturbationPressureGradientForce:
        self.scheme = ptb.PerturbationPressureGradientForce(
            disc=self.disc, method=self.method, mode=self.mode
        )
        self._cfg["pgf_scheme"] = "ptb"
        self._built = True
        return self.scheme

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self._built:
            if cfg.pgf_scheme == "ptb":
                self.ptb()
            else:
                raise ValueError(
                    f"Unsupported PGF scheme {cfg.pgf_scheme}."
                )
        return self

    def prepare_block(self, block: Block) -> Block:
        return self.scheme.prepare_block(block)
