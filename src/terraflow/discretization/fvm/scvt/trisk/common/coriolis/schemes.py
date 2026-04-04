from __future__ import annotations
from typing import Self, TypeVar, TYPE_CHECKING

from terraflow.operator import FiniteVolumeMethod
from terraflow.discretization.abstract import (
    AbstractEquationMode,
    AbstractEquationOperator,
    DiscretizationInfo,
)
from terraflow.discretization.fvm.scvt.trisk.common.coriolis import trisk

if TYPE_CHECKING:
    from terraflow.discretization.fvm.scvt.trisk.dynamics_types import Block, State


EM = TypeVar("EM", bound=AbstractEquationMode)

Operator = AbstractEquationOperator[FiniteVolumeMethod, EM]


class CoriolisForceAtEdge(Operator):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: AbstractEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        return self.scheme(block, star_state)

    def trisk(self) -> trisk.TriSKCoriolisForceAtEdge:
        self.scheme = trisk.TriSKCoriolisForceAtEdge(
            disc=self.disc, method=self.method, mode=self.mode
        )
        self._built = True
        return self.scheme

    def prepare_block(self, block: Block) -> Block:
        return self.scheme.prepare_block(block)

    def build_default(self) -> Self:
        if not self._built:
            self.trisk()
            self._built = True
        return self
