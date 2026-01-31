from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import equinox as eqx
from loguru import logger

from terraflow.util.history.setup.vert import AbstractVertHistorySetup
from terraflow.util.history.setup.vert import OneLayerVerticalCoordinateSetup


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
    def history_builder(
        self,
        file_path: str,
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

    def history_builder(self, file_path: str) -> OneLayerVerticalCoordinateSetup:
        return OneLayerVerticalCoordinateSetup(file_path, self)


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
