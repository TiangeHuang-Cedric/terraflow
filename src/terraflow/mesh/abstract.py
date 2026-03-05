from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
import equinox as eqx

from terraflow.util import typing, device

if TYPE_CHECKING:
    from terraflow.util import uxarray as tux
    from terraflow.util.history.setup.mesh import AbstractMeshHistorySetup


Array = typing.Array


class AbstractMesh(eqx.Module, ABC):
    _dev: device.AbstractDevice = eqx.field(static=True)

    @property
    def dev(self) -> device.AbstractDevice:
        """Return the device associated with the mesh."""
        return self._dev

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        """Return the alias name of the mesh type."""
        raise NotImplementedError

    @abstractmethod
    def get_layout(self) -> Any:
        """Return the layout associated with the mesh."""
        raise NotImplementedError

    @abstractmethod
    def to_ugrid(self, **kwargs) -> tux.UGRIDGrid:
        """Convert the mesh to UGRID format Dataset."""
        raise NotImplementedError

    @abstractmethod
    def history_builder(
        self,
        file_path: str,
        output_type: str,
        *args,
        **kwargs
    ) -> AbstractMeshHistorySetup:
        """Get history builder for the mesh."""
        raise NotImplementedError


class AbstractStructuredMesh(AbstractMesh):
    pass


class AbstractUnstructuredMesh(AbstractMesh):
    pass
