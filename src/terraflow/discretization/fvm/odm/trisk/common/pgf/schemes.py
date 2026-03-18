from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar

from terraflow.operator import FiniteVolumeMethod
from terraflow.discretization.abstract import (
    AbstractEquationMode,
    DiscretizationInfo,
    AbstractEquationOperator
)
from terraflow.discretization.fvm.odm.trisk.common.pgf import ptb

if TYPE_CHECKING:
    from terraflow.discretization.fvm.odm.trisk.hybrid_lev.dry.hdc.dynamics_types import Block, State


EM = TypeVar("EM", bound=AbstractEquationMode)


Operator = AbstractEquationOperator[FiniteVolumeMethod, EM]


class PressureGradientForce(Operator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: AbstractEquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        return self.scheme(block, star_state)

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
