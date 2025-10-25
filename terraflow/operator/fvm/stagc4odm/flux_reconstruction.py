from jax import numpy as jnp

from terraflow import bind
from terraflow.util import typing
from terraflow.mesh.orthogonal_dual_mesh import PrimeCell, DualCell
from terraflow.mesh.orthogonal_dual_mesh import PrimeEdge, DualEdge


Int = typing.Int
Float = typing.Float
Array = typing.Array


@bind("operator.fvm.stagc4odm", tag="flux_reconstruction")
def flux_reconstruction_central_difference_2nd_order(
    x_at_cell: Array,
    edge: PrimeEdge | DualEdge
) -> Array:
    vertex = edge.vertex()
    v1 = vertex[:, 0]
    v2 = vertex[:, 1]
    x1 = x_at_cell[v1]
    x2 = x_at_cell[v2]
    return 0.5 * (x1 + x2)


@bind("operator.fvm.stagc4odm", tag="flux_reconstruction")
def flux_reconstruction_least_square_3rd4th_order(
    stencil_2nd_num: Array,
    stencil_2nd_idx: Array,
    edd_2nd_v_fcoe: Array,
):
    pass


@bind("operator.fvm.stagc4odm", tag="flux_reconstruction")
def flux_reconstruction_least_square_5th6th_order(
    stencil_2nd_num: Array,
    stencil_2nd_idx: Array,
    edd_2nd_v_fcoe: Array,
    stencil_4th_num: Array,
    stencil_4th_idx: Array,
    edd_4th_v_fcoe: Array
):
    pass
