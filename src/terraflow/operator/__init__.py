from abc import ABC, abstractmethod
from turtle import st


class AbstractOperator(ABC):
    """Abstract base class for operators."""

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        """Return the alias name of the operator."""
        raise NotImplementedError


class FiniteVolumeMethodOperator(AbstractOperator):
    """Finite Volume Method operator implementation."""

    @staticmethod
    def alias_name() -> str:
        return "fvm"


__all__ = [
    "AbstractOperator",
    "FiniteVolumeMethodOperator"
]
