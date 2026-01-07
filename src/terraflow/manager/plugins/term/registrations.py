from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from terraflow.manager.registry import Catalog
else:
    ModelTableFactory: TypeAlias = "terraflow.manager.registry.Catalog"


def register(model: Catalog) -> None:
    from terraflow.manager.registry import Catalog

    assert isinstance(model, Catalog)

    from terraflow.discretization.fvm.odm.trisk.single_lev.term import builder

    belong_key = "earth::atm::swe::vif::normal_momentum_equ::explicit::fvm::triskC::odm::1level"

    model.equterm.register(
        belong_key,
        group="nonlinear_coriolis",
        alias="builder",
        factory=builder.NonlinearCoriolisTermAtDualEdge
    )

    model.equterm.register(
        belong_key,
        group="grad_of_geopotential",
        alias="builder",
        factory=builder.PressureGradientForceAtDualEdge
    )

    model.equterm.register(
        belong_key,
        group="grad_of_kinetic_energy",
        alias="builder",
        factory=builder.GradientOfKineticEnergyAtDualEdge
    )

    belong_key = "earth::atm::swe::vif::continuity_equ::explicit::fvm::triskC::odm::1level"

    model.equterm.register(
        belong_key,
        group="div_of_thickness",
        alias="builder",
        factory=builder.DivergenceOfMassFluxAtPrimalCell
    )
