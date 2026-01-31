from abc import ABC, abstractmethod
from loguru import logger


class AbstractDiscretizationMethod(ABC):
    """Abstract base class for operators."""

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        """Return the alias name of the operator."""
        raise NotImplementedError


class FiniteVolumeMethod(AbstractDiscretizationMethod):
    """Finite Volume Method operator implementation."""

    def __init__(self) -> None:
        logger.success("Initialized finite volume method.")

    @staticmethod
    def alias_name() -> str:
        return "fvm"
