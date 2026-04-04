from __future__ import annotations
from typing import TYPE_CHECKING
import equinox as eqx

from terraflow.util import log, typing
from terraflow.util.history.setup.vert import OneLayerVerticalCoordinateSetup
from terraflow.vert_coord.abstract import AbstractVertCoordinate

if TYPE_CHECKING:
    from terraflow.util.typing import Array
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    import numpy as np

IntType = typing.IntType


class OneLayerVertCoordinate(AbstractVertCoordinate[OneLayerVerticalCoordinateSetup]):
    """
    OneLayerVertCoordinate represents a vertical coordinate system with a single layer.
    """
    nlev: IntType = eqx.field(static=True, default=1)
    full_kds: IntType = eqx.field(static=True, default=0)
    full_kde: IntType = eqx.field(static=True, default=0)
    half_kds: IntType = eqx.field(static=True, default=0)
    half_kde: IntType = eqx.field(static=True, default=0)

    def __init__(self):
        self.nlev = 1
        self.full_kds = 0
        self.full_kde = 0
        self.half_kds = 0
        self.half_kde = 0
        log.logger.success("Initialized OneLayerVertCoordinate.")

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the one-layer vertical coordinate system."""
        return "1level"

    def __len__(self):
        return 1

    @property
    def shape(self):
        return (1,)

    def get_layout(self) -> OneLayerVertCoordLayout:
        from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
        return OneLayerVertCoordLayout()

    def history_builder(
        self,
        file_path: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64]
    ) -> OneLayerVerticalCoordinateSetup:
        return OneLayerVerticalCoordinateSetup(
            file_path=file_path,
            vert=self,
            float_type=float_type,
            int_type=int_type
        )

    def get_idx(self, entity: str, domain: str) -> tuple[IntType, IntType]:
        """Parse the position string and return the corresponding index."""
        if entity == "full" or entity == "half":
            if domain == "domain" or domain == "memory":
                return 0, 0
            else:
                raise ValueError(
                    f"Invalid domain '{domain}' for entity '{entity}' in OneLayerVertCoordinate."
                )
        else:
            raise ValueError(
                f"Invalid entity '{entity}' for OneLayerVertCoordinate. Expected 'full' or 'half'."
            )

    def fill_halo(self, field: Array, is_index: bool = False, *args, **kwargs) -> Array:
        """Fill the halo regions of the mesh."""
        return field
