from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar

from terraflow.operator import FiniteVolumeMethod
from terraflow.discretization.abstract import (
    AbstractEquationMode,
    DiscretizationInfo,
    AbstractDynamicalEquationOperator
)
from terraflow.discretization.fvm.scvt.trisk.common.pgf import ptb

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dry.hdc.dynamics_types import Block

    Int = typing.Int
    Array = typing.Array


EM = TypeVar("EM", bound=AbstractEquationMode)


DynOperator = AbstractDynamicalEquationOperator[FiniteVolumeMethod, EM]


class PressureGradientForce(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: AbstractEquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: Block, star_idx: Int[Array, ""] | int) -> Block:
        return self.scheme(block, star_idx)

    def ptb(self) -> ptb.PerturbationPressureGradientForce:
        self.scheme = ptb.PerturbationPressureGradientForce(
            disc=self.disc, method=self.method, mode=self.mode
        )
        self._built = True
        return self.scheme

    def prepare_block(self, block: Block) -> Block:
        return self.scheme.prepare_block(block)

    def build_default(self) -> "PressureGradientForce":
        if not self._built:
            self.ptb()
            self._built = True
        return self
