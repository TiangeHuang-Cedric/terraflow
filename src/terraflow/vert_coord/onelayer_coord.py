from __future__ import annotations
from typing import TYPE_CHECKING
from loguru import logger

from terraflow.util.history.setup.vert import OneLayerVerticalCoordinateSetup
from terraflow.vert_coord.abstract import AbstractVertCoordinate

if TYPE_CHECKING:
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    import numpy as np


class OneLayerVertCoordinate(AbstractVertCoordinate[OneLayerVerticalCoordinateSetup]):
    """
    OneLayerVertCoordinate represents a vertical coordinate system with a single layer.
    """

    def __init__(self):
        logger.success("Initialized OneLayerVertCoordinate.")

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
