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

    from terraflow.discretization.fvm.scvt.trisk.single_lev import term

    belong_key = "earth::atm::swe::vif::explicit::fvm::triskC::scvt::1level"

    model.equaux.register(
        belong_key,
        alias="prepare_space_operators",
        factory=term.ExPrepareSpaceOperators
    )

    model.equaux.register(
        belong_key,
        alias="space_operators",
        factory=term.ExSpaceOperators,
    )

    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev import term
    belong_key = "earth::atm::dry_hdc::vif::explicit::fvm::triskC::scvt::hybrid"

    model.equaux.register(
        belong_key,
        alias="prepare_space_operators",
        factory=term.ExPrepareSpaceOperators,
    )

    model.equaux.register(
        belong_key,
        alias="space_operators",
        factory=term.ExSpaceOperators,
    )

    from terraflow.discretization.fvm.scvt.trisk.single_lev import adv as term
    belong_key = "earth::atm::adv::vif::explicit::fvm::triskC::scvt::1level"

    model.equaux.register(
        belong_key,
        alias="prepare_space_operators",
        factory=term.ExPrepareSpaceOperators,
    )

    model.equaux.register(
        belong_key,
        alias="space_operators",
        factory=term.ExSpaceOperators,
    )

    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev import adv as term
    belong_key = "earth::atm::adv::vif::explicit::fvm::triskC::scvt::hybrid"

    model.equaux.register(
        belong_key,
        alias="prepare_space_operators",
        factory=term.ExPrepareSpaceOperators,
    )

    model.equaux.register(
        belong_key,
        alias="space_operators",
        factory=term.ExSpaceOperators,
    )
