from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from terraflow.manager.registry import Catalog
else:
    ModelTableFactory: TypeAlias = "terraflow.manager.registry.Catalog"


def register(model: Catalog) -> None:
    from terraflow.manager.registry import Catalog
    import terraflow.mesh as mesh

    assert isinstance(model, Catalog)

    # Register mesh
    model.mesh.register(
        kind="unstructured", alias="odm", factory=mesh.OrthogonalDualMesh
    )
