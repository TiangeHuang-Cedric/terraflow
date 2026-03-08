from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, TYPE_CHECKING
import equinox as eqx
from loguru import logger

from terraflow.util.history.setup.vert import AbstractVertHistorySetup
from terraflow.util.history.setup.vert import OneLayerVerticalCoordinateSetup


if TYPE_CHECKING:
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    import numpy as np


VMST = TypeVar("VMST", bound=AbstractVertHistorySetup)


class AbstractVertCoordinate(eqx.Module, Generic[VMST], ABC):
    """
    AbstractVertCoordinate is the base class for vertical coordinate systems.
    """

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        """Return the alias name of the vertical coordinate system."""
        raise NotImplementedError

    @abstractmethod
    def get_layout(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def history_builder(
        self,
        file_path: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64],
        *args,
        **kwargs
    ) -> VMST:
        raise NotImplementedError


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


class AbstractMultiLayerVertCoordinate(AbstractVertCoordinate):
    """
    AbstractMultiLayerVertCoordinate represents a vertical coordinate system with multiple layers.
    """
    pass


class AbstractLorenzCoordinate(AbstractMultiLayerVertCoordinate):
    """
    LorenzCoordinate represents a vertical coordinate system following the Lorenz staggering.
    """
    @abstractmethod
    def get_full_lev(self):
        raise NotImplementedError

    @abstractmethod
    def get_half_lev(self):
        raise NotImplementedError

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the Lorenz vertical coordinate system."""
        return "lorenz"
