from terraflow.util import typing


Int = typing.Int
Float = typing.Float
Array = typing.Array


def flux_reconstruct_central_difference_2nd_order(
    x_at_cell: Float[Array, "nCell"],
    edge_v: Int[Array, "nEdge 2"]
) -> Float[Array, "nEdge"]:
    vertex = edge_v
    v1 = vertex[:, 0]
    v2 = vertex[:, 1]
    x1 = x_at_cell[v1]
    x2 = x_at_cell[v2]
    return 0.5 * (x1 + x2)


def flux_reconstruct_least_square_3rd4th_order(
    stencil_2nd_num: Array,
    stencil_2nd_idx: Array,
    edd_2nd_v_fcoe: Array,
):
    pass


def flux_reconstruct_least_square_5th6th_order(
    stencil_2nd_num: Array,
    stencil_2nd_idx: Array,
    edd_2nd_v_fcoe: Array,
    stencil_4th_num: Array,
    stencil_4th_idx: Array,
    edd_4th_v_fcoe: Array
):
    pass
