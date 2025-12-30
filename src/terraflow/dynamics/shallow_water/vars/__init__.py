from __future__ import annotations
from typing import Type, TypeVar, Tuple, overload, cast, TYPE_CHECKING

from terraflow.mesh.abstract import AbstractMesh
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.vert_coord.abstract import AbstractVertCoordinate
from terraflow.layout.abstract import AbstractLayout

from . import unstructure

if TYPE_CHECKING:
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.discretization.abstract import AbstractFieldBasedVariableSet

HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
LO = TypeVar("LO", bound=AbstractLayout)


class ShallowWaterVariablesBuilder:

    @overload
    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[OrthogonalDualMesh, VM, LO],
    ) -> Tuple[
        Type[unstructure.StateFields],
        Type[unstructure.StaticFields]
    ]:
        ...

    @overload
    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[HM, VM, LO],
    ) -> Tuple[
        Type[AbstractFieldBasedVariableSet],
        Type[AbstractFieldBasedVariableSet]
    ]:
        ...

    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[HM, VM, LO],
    ) -> Tuple[
        Type[AbstractFieldBasedVariableSet],
        Type[AbstractFieldBasedVariableSet]
    ]:
        if isinstance(coord.mesh, OrthogonalDualMesh):
            return (cast(Type[unstructure.StateFields], unstructure.StateFields),
                    cast(Type[unstructure.StaticFields], unstructure.StaticFields))
        else:
            raise NotImplementedError(
                "ShallowWaterVariablesBuilder only supports unstructured mesh currently."
            )
