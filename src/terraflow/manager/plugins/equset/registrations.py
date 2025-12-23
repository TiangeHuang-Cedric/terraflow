from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from terraflow.manager.registry import Catalog
else:
    ModelTableFactory: TypeAlias = "terraflow.manager.registry.Catalog"


def register(model: Catalog) -> None:
    from terraflow.manager.registry import Catalog
    import terraflow.discretization.equation.shallow_water as swe

    assert isinstance(model, Catalog)

    # Register equset
    model.equset.register(
        form="vif",
        mesh_kind="unstructured",
        ode_kind="explicit",
        factory=swe.base.ExVifShallowWaterEquationSet4UGRID
    )
