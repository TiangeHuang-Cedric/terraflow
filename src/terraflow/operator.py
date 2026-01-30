from abc import ABC, abstractmethod


class AbstractDiscretizationMethod(ABC):
    """Abstract base class for operators."""

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        """Return the alias name of the operator."""
        raise NotImplementedError


class FiniteVolumeMethod(AbstractDiscretizationMethod):
    """Finite Volume Method operator implementation."""

    @staticmethod
    def alias_name() -> str:
        return "fvm"
