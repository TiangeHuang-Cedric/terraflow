from typing import (
    Union,
    Callable,
    Sequence,
    Any
)
from equinox._custom_types import sentinel
import equinox as eqx


from terraflow.util import typing


Array = typing.Array
PyTree = typing.PyTree


def tree_at(
    where: Callable[[PyTree], Any | Sequence[Any]],
    pytree: PyTree,
    replace: Any | Sequence[Any] = sentinel,
    is_leaf: Callable[[Any], bool] | None = None,
    at_dims: int | Sequence[Union[int, None]] | None = None,
    at_idxes: Union[
        int,
        Sequence[Union[int, None]],
        Array,
        Sequence[Union[int, Array, None]],
        None
    ] = None
) -> PyTree:
    if at_dims == None and at_idxes == None:
        return eqx.tree_at(
            where=where,
            pytree=pytree,
            replace=replace,
            is_leaf=is_leaf
        )
    
    fields = where(pytree)
    
    if isinstance(fields, Array):
        if not isinstance(at_dims, int) or at_idxes is None:
            raise ValueError("Size dismatch between where function and at_dims or at_idxes!")
        indexer: list[Any] = [slice(None)] * fields.ndim
        indexer[at_dims] = at_idxes
        fields_rep = fields.at[tuple(indexer)].set(replace)
        return eqx.tree_at(
            where=where,
            pytree=pytree,
            replace=fields_rep,
            is_leaf=is_leaf
        )
    else:
        if isinstance(at_dims, int) or at_dims is None:
            raise ValueError("Size dismatch between where function and at_dims!")
        if isinstance(at_idxes, int) or at_idxes is None:
            raise ValueError("Size dismatch between where function and at_idxes!")
        
        def replace_fn(i: int):
            field = fields[i]
            if at_dims[i] is None and at_idxes[i] is None:
                return replace[i]
            elif at_dims[i] is not None and at_idxes[i] is not None:
                indexer = [slice(None)] * field.ndim  
                indexer[at_dims[i]] = at_idxes[i]
                return field.at[tuple(indexer)].set(replace[i])
        
        fields_rep = tuple(replace_fn(i) for i, _ in enumerate(fields))
        
        return eqx.tree_at(
            where=where,
            pytree=pytree,
            replace=fields_rep,
            is_leaf=is_leaf
        ) 


def error_if(
    x: PyTree,
    pred: Array,
    msg: str,
    is_metal: bool = False
) -> PyTree:
    if is_metal:
        return eqx.error_if(
            x=x,
            pred=pred,
            msg=msg,
            on_error="nan"
        )
    else:
        return eqx.error_if(
            x=x,
            pred=pred,
            msg=msg,
            on_error="raise"
        )