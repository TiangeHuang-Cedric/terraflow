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

    from terraflow.discretization.fvm.odm.trisk.single_lev import term

    belong_key = "earth::atm::swe::vif::normal_momentum_equ::explicit::fvm::triskC::odm::1level"

    model.equterm.register(
        belong_key,
        group="nonlinear_coriolis",
        alias="builder",
        factory=term.TendencyOfNormalVelocityDueToNonlinearCoriolisTermAtDualEdge
    )

    model.equterm.register(
        belong_key,
        group="grad_of_geopotential",
        alias="builder",
        factory=term.TendencyOfNormalVelocityDueToPressureGradientForceAtDualEdge
    )

    model.equterm.register(
        belong_key,
        group="grad_of_kinetic_energy",
        alias="builder",
        factory=term.TendencyOfNormalVelocityDueToGradientOfKineticEnergyAtDualEdge
    )

    belong_key = "earth::atm::swe::vif::continuity_equ::explicit::fvm::triskC::odm::1level"

    model.equterm.register(
        belong_key,
        group="div_of_mfx",
        alias="builder",
        factory=term.TendencyOfDivergenceOfMassFlux
    )

    from terraflow.discretization.fvm.odm.trisk.hybrid_lev import term

    belong_key = "earth::atm::dry_hdc::vif::continuity_equ::explicit::fvm::triskC::odm::hybrid"

    model.equterm.register(
        belong_key,
        group="div_of_mfs",
        alias="builder",
        factory=term.TendencyOfDryAirMassAtSurface
    )

    belong_key = "earth::atm::dry_hdc::vif::thermodynamic_equ::explicit::fvm::triskC::odm::hybrid"

    model.equterm.register(
        belong_key,
        group="div_of_mft",
        alias="builder",
        factory=term.TendencyOfPotentialTemperatureDueToDivergenceOfFlux
    )

    model.equterm.register(
        belong_key,
        group="vert_adv_th",
        alias="builder",
        factory=term.TendencyOfPotentialTemperatureDueToVerticalAdvection
    )

    belong_key = "earth::atm::dry_hdc::vif::normal_momentum_equ::explicit::fvm::triskC::odm::hybrid"

    model.equterm.register(
        belong_key,
        group="nonlinear_coriolis",
        alias="builder",
        factory=term.TendencyOfNormalVelocityDueToNonlinearCoriolisTermAtDualEdge
    )

    model.equterm.register(
        belong_key,
        group="grad_of_kinetic_energy",
        alias="builder",
        factory=term.TendencyOfNormalVelocityDueToGradientOfKineticEnergyAtDualEdge
    )

    model.equterm.register(
        belong_key,
        group="pgf",
        alias="builder",
        factory=term.TendencyOfNormalVelocityDueToPressureGradientForce
    )

    model.equterm.register(
        belong_key,
        group="vert_adv",
        alias="builder",
        factory=term.TendencyOfNormalVelocityDueToVerticalAdvection
    )
