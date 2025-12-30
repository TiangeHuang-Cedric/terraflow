from abc import ABC, abstractmethod


class AbstractVertCoordinate(ABC):
    """
    AbstractVertCoordinate is the base class for vertical coordinate systems.
    """

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        """Return the alias name of the vertical coordinate system."""
        raise NotImplementedError


class OneLayerVertCoordinate(AbstractVertCoordinate):
    """
    OneLayerVertCoordinate represents a vertical coordinate system with a single layer.
    """

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the one-layer vertical coordinate system."""
        return "1level"


class AbstractMultiLayerVertCoordinate(AbstractVertCoordinate):
    """
    AbstractMultiLayerVertCoordinate represents a vertical coordinate system with multiple layers.
    """
    pass


class LorenzCoordinate(AbstractMultiLayerVertCoordinate):
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
