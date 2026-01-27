from __future__ import annotations
from typing import Type, TypeVar, overload, TYPE_CHECKING

from terraflow.mesh.abstract import AbstractMesh
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.vert_coord.abstract import AbstractVertCoordinate
from terraflow.layout.abstract import AbstractLayout

import terraflow.discretization.fvm.odm.trisk.single_lev.dynamics_types as vars

if TYPE_CHECKING:
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.discretization.abstract import AbstractBlock

HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
LO = TypeVar("LO", bound=AbstractLayout)


class ShallowWaterVariablesBuilder:

    @overload
    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[OrthogonalDualMesh, VM, LO],
    ) -> Type[vars.Block]:
        ...

    @overload
    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[HM, VM, LO],
    ) -> Type[AbstractBlock]:
        ...

    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[HM, VM, LO],
    ) -> Type[AbstractBlock]:
        if isinstance(coord.mesh, OrthogonalDualMesh):
            return vars.Block
        else:
            raise NotImplementedError(
                "ShallowWaterVariablesBuilder only supports unstructured mesh currently."
            )
