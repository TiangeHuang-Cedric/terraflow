from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from terraflow.manager.registry import Catalog
else:
    ModelTableFactory: TypeAlias = "terraflow.manager.registry.Catalog"


def register(model: Catalog) -> None:
    from terraflow.manager.registry import Catalog
    import terraflow.vert_coordinates as vert

    assert isinstance(model, Catalog)

    # Register vertical coordinate
    model.vert.register(
        kind="one_layer", factory=vert.OneLayerVertCoordinate
    )
