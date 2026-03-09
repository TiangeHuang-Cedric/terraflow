from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, TYPE_CHECKING
import equinox as eqx

from terraflow.util.history.setup.vert import AbstractVertHistorySetup


if TYPE_CHECKING:
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
