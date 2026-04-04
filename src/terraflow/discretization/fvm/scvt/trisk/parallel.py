from typing import (
    cast,
    Callable,
    Sequence,
    Any,
    TypeVar,
)
from equinox._custom_types import sentinel
import equinox as eqx
import jax

from terraflow.util import typing
from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.layout.abstract import CoordinateSystem
from terraflow.layout.field import ArrayField, StaggeredFieldMetaData
from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout


_Node = Any

PyTree = typing.PyTree

FD = ArrayField[StaggeredFieldMetaData]

VM = TypeVar("VM", OneLayerVertCoordinate, HybridVertCoordinate)
VLO = TypeVar("VLO", OneLayerVertCoordLayout, HybridVertCoordLayout)

CoordSys = CoordinateSystem[
    SphericalCentroidVoronoiTessellations,
    VM,
    TriSKBasedCGridStaggeringLayout,
    VLO
]

CoordSysOneLayer = CoordinateSystem[
    SphericalCentroidVoronoiTessellations,
    OneLayerVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    OneLayerVertCoordLayout
]

CoordSysHybrid = CoordinateSystem[
    SphericalCentroidVoronoiTessellations,
    HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    HybridVertCoordLayout
]


def tree_at(
    coord: CoordSys,
    where: Callable[[PyTree], FD | Sequence[FD]],
    pytree: PyTree,
    replace: Any | Sequence[Any] = sentinel,
    replace_fn: Callable[[_Node], Any] = sentinel,
    is_leaf: Callable[[Any], bool] | None = None,
    fill_halo: bool = False
) -> PyTree:
    fields = where(pytree)
    if isinstance(fields, (tuple, list)):
        where_list = [
            (lambda tree, i=i: where(tree)[i])
            for i in range(len(fields))
        ]
    else:
        where_list = cast(list[Callable[[PyTree], FD]], [where])

    where_wrap_back = (
        lambda tree: tuple(where_i(tree).data for where_i in where_list)
    )

    if replace_fn is not sentinel and replace is sentinel:
        if isinstance(replace_fn, (tuple, list)):
            if len(replace_fn) != len(where_list):
                raise ValueError(
                    "Length of replace_fn must match the number of selected fields."
                )
            replace_fn_list = list(replace_fn)
        else:
            replace_fn_list = [replace_fn] * len(where_list)
        replace_list = [sentinel] * len(where_list)
    elif replace_fn is sentinel and replace is not sentinel:
        if isinstance(replace, (tuple, list)):
            if len(replace) != len(where_list):
                raise ValueError(
                    "Length of replace must match the number of selected fields."
                )
            replace_list = list(replace)
        elif len(where_list) == 1:
            replace_list = [replace]
        else:
            raise ValueError(
                "Multiple selected fields require replace to be a sequence."
            )
        replace_fn_list = [sentinel] * len(where_list)
    else:
        raise ValueError(
            "Either replace or replace_fn must be provided, but not both."
        )

    replace_wrap_back = jax.tree_util.tree_map(
        lambda where_i, replace_i, replace_fn_i: _tree_at(
            coord=coord,
            where=where_i,
            pytree=pytree,
            replace=replace_i,
            replace_fn=replace_fn_i,
            fill_halo=fill_halo
        ),
        where_list,
        replace_list,
        replace_fn_list
    )

    return eqx.tree_at(
        where=where_wrap_back,
        pytree=pytree,
        replace=replace_wrap_back,
        is_leaf=is_leaf,
    )


def _tree_at(
    coord: CoordSys,
    where: Callable[[PyTree], ArrayField[StaggeredFieldMetaData]],
    pytree: PyTree,
    replace: Any = sentinel,
    replace_fn: Callable[[_Node], Any] = sentinel,
    fill_halo: bool = False,
    fill_hori_halo: bool = False,
    fill_vert_halo: bool = False
) -> Any:
    if replace_fn is not sentinel and replace is sentinel:
        replace = replace_fn(pytree)
    elif replace is sentinel and replace_fn is sentinel:
        raise ValueError("Either replace or replace_fn must be provided.")
    if fill_halo and (fill_hori_halo or fill_vert_halo):
        raise ValueError(
            "fill_halo cannot be used with fill_hori_halo or fill_vert_halo."
        )
    if fill_halo:
        field = where(pytree)
        return coord.fill_halo(
            field=replace,
            hori_loc=field.meta.hori_location,
            vert_loc=field.meta.vert_location,
        )
    elif fill_hori_halo:
        field = where(pytree)
        return coord.mesh.fill_halo(
            field=replace,
            loc=field.meta.hori_location,
        )
    elif fill_vert_halo:
        field = where(pytree)
        return coord.vert.fill_halo(
            field=replace,
            loc=field.meta.vert_location,
        )
    else:
        return replace
