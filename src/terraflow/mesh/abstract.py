from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
import equinox as eqx

from terraflow.util import typing

if TYPE_CHECKING:
    from terraflow.util import uxarray as tux
    from terraflow.util.history.setup.mesh import AbstractMeshHistorySetup
    import numpy as np


Array = typing.Array


class AbstractMesh(eqx.Module, ABC):

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        """Return the alias name of the mesh type."""
        raise NotImplementedError

    @abstractmethod
    def get_coords(self, *args, **kwargs) -> Any:
        """Return the layout associated with the mesh."""
        raise NotImplementedError

    @abstractmethod
    def history_builder(
        self,
        file_path: str,
        output_type: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64],
        * args,
        **kwargs
    ) -> AbstractMeshHistorySetup:
        """Get history builder for the mesh."""
        raise NotImplementedError


class AbstractStructuredMesh(AbstractMesh):
    pass


class AbstractUnstructuredMesh(AbstractMesh):
    pass


class AbstractRefMesh(eqx.Module, ABC):
    pass

class AbstractHalo(eqx.Module, ABC):
    pass
