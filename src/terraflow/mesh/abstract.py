from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from terraflow.util import typing

if TYPE_CHECKING:
    from terraflow.util import uxarray as tux
Array = typing.Array


class AbstractMesh(ABC):
    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        """Return the alias name of the mesh type."""
        raise NotImplementedError

    @abstractmethod
    def to_ugrid(self, **kwargs) -> tux.UGRIDGrid:
        """Convert the mesh to UGRID format Dataset."""
        raise NotImplementedError


class AbstractStructuredMesh(AbstractMesh):
    pass


class AbstractUnstructuredMesh(AbstractMesh):
    pass
