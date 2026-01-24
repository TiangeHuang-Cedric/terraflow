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

    belong_key = "earth::atm::swe::vif::explicit::fvm::triskC::odm::1level"

    model.equaux.register(
        belong_key,
        factory=term.AuxiliaryTerms
    )
