from __future__ import annotations
from typing import Self, TypeVar, TYPE_CHECKING

from terraflow.operator import FiniteVolumeMethod
from terraflow.discretization.fvm.odm.trisk import coriolis
from terraflow.discretization.abstract import (
    AbstractEquationSetAuxiliaryTerms,
    DiscretizationInfo,
)

if TYPE_CHECKING:
    from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block


Term = AbstractEquationSetAuxiliaryTerms[FiniteVolumeMethod]


class CoriolisForceAtEdge(Term):

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

    def recon_tangent_velocity(self, block: Block) -> Block:
        return self.scheme.recon_tangent_velocity(block)

    def trisk(self) -> coriolis.trisk.TriSKCoriolisForceAtEdge:
        self.scheme = coriolis.trisk.TriSKCoriolisForceAtEdge(
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

    def prepare(self, block: Block) -> Block:
        return self.scheme.prepare(block)

    def build_default(self) -> Self:
        if not self._built:
            self.trisk()
            self._built = True
        return self
