from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from terraflow.manager.registry import Catalog
else:
    ModelTableFactory: TypeAlias = "terraflow.manager.registry.Catalog"


def register(model: Catalog) -> None:
    from terraflow.manager.registry import Catalog
    import terraflow.operator as op

    assert isinstance(model, Catalog)

    # Register method
    model.method.register(
        name="finite_volume_method",
        description="A method for solving partial differential equations using control volumes",
        factory=op.FiniteVolumeMethodOperator
    )
