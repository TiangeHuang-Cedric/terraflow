from __future__ import annotations
from typing import (
    cast,
    Union,
    Callable,
    Sequence,
    Any,
    TypeVar,
    TYPE_CHECKING
)
from equinox._custom_types import sentinel
import equinox as eqx

from terraflow.util import typing

Array = typing.Array

if TYPE_CHECKING:
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

    CoordSys = TypeVar("CoordSys", CoordSysOneLayer, CoordSysHybrid)


def tree_at(
    coord: CoordSys,
    where: Callable[[PyTree], FD | Sequence[FD]],
    pytree: PyTree,
    replace: Any | Sequence[Any] = sentinel,
    replace_fn: Callable[[_Node], Any] = sentinel,
    is_leaf: Callable[[Any], bool] | None = None,
    at_dims: int | Sequence[int] | None = None,
    at_idxes: Union[
        int,
        Sequence[int],
        Array,
        Sequence[Union[int, Array]],
        None
    ] = None,
    fill_halo: bool = False,
    fill_hori_halo: bool = False,
    fill_vert_halo: bool = False,
) -> PyTree:
    def _node_data(node: Any) -> Array:
        # Support selecting either ArrayField-like objects or raw array leaves.
        return cast(Array, node.data if hasattr(node, "data") else node)

    # Extract the fields from the pytree using the where function
    fields = where(pytree)

    # Determine selected fields and number of targets.
    if isinstance(fields, (tuple, list)):
        selected_fields = tuple(fields)
    else:
        selected_fields = (fields,)
    n_where = len(selected_fields)

    # Determine the replace and replace_fn for each field
    if replace_fn is not sentinel and replace is sentinel:
        if isinstance(replace_fn, (tuple, list)):
            if len(replace_fn) != n_where:
                raise ValueError(
                    "Length of replace_fn must match the number of selected fields."
                )
            replace_fn_list = list(replace_fn)
        else:
            replace_fn_list = [replace_fn] * n_where
        replace_list = [sentinel] * n_where
    elif replace_fn is sentinel and replace is not sentinel:
        if isinstance(replace, (tuple, list)):
            if len(replace) != n_where:
                raise ValueError(
                    "Length of replace must match the number of selected fields."
                )
            replace_list = list(replace)
        elif n_where == 1:
            replace_list = [replace]
        else:
            raise ValueError(
                "Multiple selected fields require replace to be a sequence."
            )
        replace_fn_list = [sentinel] * n_where
    else:
        raise ValueError(
            "Either replace or replace_fn must be provided, but not both."
        )

    # Determine the at_dims and at_idxes for each field
    if at_dims is not None and at_idxes is not None:
        if isinstance(at_dims, int):
            at_dims_list = [at_dims] * n_where
        else:
            if len(at_dims) != n_where:
                raise ValueError(
                    "Length of at_dims must match the number of selected fields."
                )
            at_dims_list = list(at_dims)
        if isinstance(at_idxes, int):
            at_idx_list = [at_idxes] * n_where
        elif isinstance(at_idxes, Array):
            if at_idxes.ndim != 0 or at_idxes.shape != ():
                raise ValueError(
                    "at_idxes array must be a scalar with shape ()."
                )
            at_idx_list = [at_idxes]
        elif isinstance(at_idxes, (tuple, list)):
            if len(at_idxes) != n_where:
                raise ValueError(
                    "Length of at_idxes must match the number of selected fields."
                )
            at_idx_list = list(at_idxes)
        else:
            raise ValueError(
                "at_idxes must be an int, a 1D array, or a sequence when at_dims is provided."
            )
    elif at_dims is None and at_idxes is not None:
        raise ValueError(
            "at_dims must be provided if at_idxes is provided."
        )
    elif at_dims is not None and at_idxes is None:
        raise ValueError(
            "at_idxes must be provided if at_dims is provided."
        )
    else:
        at_dims_list = [None] * n_where
        at_idx_list = [None] * n_where

    selected_nodes = tuple(_node_data(field_i) for field_i in selected_fields)

    where_wrap_back_fn: Callable[[Any], Any] = lambda _tree: (_ for _ in ()).throw(
        RuntimeError("where_wrap_back_fn is not initialized.")
    )
    replace_wrap_back_fn: Callable[[Array], Array] = lambda _node: (_ for _ in ()).throw(
        RuntimeError("replace_wrap_back_fn is not initialized.")
    )

    if n_where == 1:
        selected_node_0 = selected_nodes[0]
        selected_field_0 = selected_fields[0]

        def _where_wrap_back_single(tree):
            return _node_data(where(tree))

        def _replace_wrap_back_single(node: Array):
            if id(node) != id(selected_node_0):
                raise ValueError(
                    "Failed to resolve replacement target in tree_at()."
                )

            replace_value = _fill_halo_at(
                coord=coord,
                where=lambda _tree: selected_field_0,
                pytree=pytree,
                replace=replace_list[0],
                replace_fn=replace_fn_list[0],
                fill_halo=fill_halo,
                fill_hori_halo=fill_hori_halo,
                fill_vert_halo=fill_vert_halo,
            )

            at_dims_i = at_dims_list[0]
            at_idx_i = at_idx_list[0]
            if at_dims_i is not None and at_idx_i is not None:
                indexer: list[Any] = [slice(None)] * node.ndim
                indexer[at_dims_i] = at_idx_i
                return node.at[tuple(indexer)].set(replace_value)
            return replace_value

        where_wrap_back_fn = _where_wrap_back_single
        replace_wrap_back_fn = _replace_wrap_back_single
    else:
        node_id_to_i = {
            id(node_i): i for i, node_i in enumerate(selected_nodes)
        }

        def _where_wrap_back_multi(tree):
            selected = where(tree)
            if not isinstance(selected, (tuple, list)):
                raise ValueError(
                    "`where` must return a sequence when multiple fields are selected."
                )
            if len(selected) != n_where:
                raise ValueError(
                    "`where` returned a sequence with unexpected length."
                )
            return tuple(_node_data(selected[i]) for i in range(n_where))

        def _replace_wrap_back_multi(node: Array):
            i = node_id_to_i.get(id(node), None)
            if i is None:
                raise ValueError(
                    "Failed to resolve replacement target in tree_at()."
                )

            replace_value = _fill_halo_at(
                coord=coord,
                where=lambda _tree, i=i: selected_fields[i],
                pytree=pytree,
                replace=replace_list[i],
                replace_fn=replace_fn_list[i],
                fill_halo=fill_halo,
                fill_hori_halo=fill_hori_halo,
                fill_vert_halo=fill_vert_halo,
            )

            at_dims_i = at_dims_list[i]
            at_idx_i = at_idx_list[i]
            if at_dims_i is not None and at_idx_i is not None:
                indexer: list[Any] = [slice(None)] * node.ndim
                indexer[at_dims_i] = at_idx_i
                return node.at[tuple(indexer)].set(replace_value)
            return replace_value

        where_wrap_back_fn = _where_wrap_back_multi
        replace_wrap_back_fn = _replace_wrap_back_multi

    return eqx.tree_at(
        where=where_wrap_back_fn,
        pytree=pytree,
        replace_fn=replace_wrap_back_fn,
        is_leaf=is_leaf,
    )


def _fill_halo_at(
    coord: CoordSys,
    where: Callable[[PyTree], Any],
    pytree: PyTree,
    replace: Any = sentinel,
    replace_fn: Callable[[_Node], Any] = sentinel,
    fill_halo: bool = False,
    fill_hori_halo: bool = False,
    fill_vert_halo: bool = False
) -> Array:
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
        if not hasattr(field, "meta"):
            raise ValueError(
                "fill_halo requires `where` to select a field with metadata."
            )
        return coord.fill_halo(
            field=replace,
            hori_loc=field.meta.hori_location,
            vert_loc=field.meta.vert_location,
        )
    elif fill_hori_halo:
        field = where(pytree)
        if not hasattr(field, "meta"):
            raise ValueError(
                "fill_hori_halo requires `where` to select a field with metadata."
            )
        return coord.mesh.fill_halo(
            field=replace,
            loc=field.meta.hori_location,
        )
    elif fill_vert_halo:
        field = where(pytree)
        if not hasattr(field, "meta"):
            raise ValueError(
                "fill_vert_halo requires `where` to select a field with metadata."
            )
        return coord.vert.fill_halo(
            field=replace,
            loc=field.meta.vert_location,
        )
    else:
        return cast(Array, replace)
