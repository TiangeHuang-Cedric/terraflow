from __future__ import annotations
from typing import Any, Dict, Self, TypeVar, TYPE_CHECKING

from terraflow.operator import FiniteVolumeMethod
from terraflow.discretization.abstract import (
    AbstractEquationMode,
    AbstractDynamicalEquationOperator,
    DiscretizationInfo,
)
from terraflow.discretization.fvm.scvt.trisk.common.coriolis import trisk

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.util.model_config import ModelConfig
    from terraflow.discretization.fvm.scvt.trisk.dynamics_types import Block

    Int = typing.Int
    Array = typing.Array


EM = TypeVar("EM", bound=AbstractEquationMode)

DynOperator = AbstractDynamicalEquationOperator[FiniteVolumeMethod, EM]


class CoriolisForceAtEdge(DynOperator):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: AbstractEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: Block, star_idx: Int[Array, ""] | int) -> Block:
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

    def prepare_block(self, block: Block) -> Block:
        return self.scheme.prepare_block(block)
