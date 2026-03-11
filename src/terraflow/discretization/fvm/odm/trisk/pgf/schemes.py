from __future__ import annotations
from typing import TYPE_CHECKING

from terraflow.operator import FiniteVolumeMethod
from terraflow.discretization.abstract import (
    DiscretizationInfo,
    AbstractEquationOperator
)
from terraflow.discretization.fvm.odm.trisk.pgf.ptb import PerturbationPressureGradientForce

if TYPE_CHECKING:
    from terraflow.discretization.fvm.odm.trisk.hybrid_lev.dry.hdc.dynamics_types import Block


Term = AbstractEquationOperator[FiniteVolumeMethod]


class PressureGradientForce(Term):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        baroclinic: bool,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=baroclinic,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        return self.scheme(block)

    def ptb(self) -> PerturbationPressureGradientForce:
        self.scheme = PerturbationPressureGradientForce(
            disc=self.disc,
            method=self.method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
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
