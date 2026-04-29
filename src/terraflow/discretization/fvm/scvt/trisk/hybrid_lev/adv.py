from __future__ import annotations
from typing import Self, TYPE_CHECKING

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.scvt import operators as op

from terraflow.discretization.abstract import (
    AbstractPrepareSpaceOperators,
    AbstractAdvectionEquationOperator
)
from terraflow.dynamics.adv import types
from ..parallel import tree_at


if TYPE_CHECKING:
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
    from terraflow.discretization.abstract import DiscretizationInfo
    from ..dynamics_types import Block

    Float = typing.Float
    Int = typing.Int
    Array = typing.Array
    CoordSys = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        HybridVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        HybridVertCoordLayout
    ]

    BLOCK = Block[HybridVertCoordinate, HybridVertCoordLayout]


class ExPrepareSpaceOperators(
    AbstractPrepareSpaceOperators[
        FiniteVolumeMethod,
        types.AdvectionEquation3DMode
    ]
):
    pass


class ExSpaceOperators(
    types.ExVif3DAdvectionEquationSet4UGRIDSpaceOperators
):
    pass
