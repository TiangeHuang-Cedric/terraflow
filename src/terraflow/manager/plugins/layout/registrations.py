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

    import terraflow.layout.orthogonal_dual_mesh as odm

    # Register method
    model.layout.register(
        name="C",
        horizontal_mesh="odm",
        vertical_coordinate="1level",
        description="A staggered grid arrangement commonly used in geophysical fluid dynamics",
        factory=odm.TriSKBasedCGridStaggeringLayout
    )
