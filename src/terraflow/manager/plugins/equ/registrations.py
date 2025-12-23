from __future__ import annotations

from re import A
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

    # Register equation
    model.equ.register(
        equset="swe",
        form="vif",
        mesh_kind="unstructured",
        ode_kind="explicit",
        factory=swe.base.ExContinuityEqu
    )

    model.equ.register(
        equset="swe",
        form="vif",
        mesh_kind="unstructured",
        ode_kind="explicit",
        factory=swe.base.ExNormalMomentumEqu4UGRID
    )
