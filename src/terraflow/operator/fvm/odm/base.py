from jax import numpy as jnp

from terraflow.util import typing

FloatType = typing.FloatType
Int = typing.Int
Float = typing.Float
Array = typing.Array


def div_at_primal_cell(
    x_at_edge: Float[Array, "nEdge"],
    radius: FloatType,
    plg_ed: Int[Array, "nPrimal_cell maxnnb"],
    plg_nr: Float[Array, "nPrimal_cell maxnnb"],
    plg_area: Float[Array, "nPrimal_cell"],
    edp_length: Float[Array, "nEdge"]
) -> Float[Array, "nPrimal_cell"]:
    """Compute divergence at primal cell center.

    Calculates the divergence of a vector field at primal cell centers using
    the finite volume method on an orthogonal dual mesh.
    """
    ed_idx = plg_ed  # (nplg, maxnnb)
    leng_tile = edp_length[ed_idx]  # (nplg, maxnnb)
    x_tile = x_at_edge[ed_idx]  # (nplg, maxnnb)
    x_tile = jnp.where(ed_idx < 0, 0.0, x_tile)
    return jnp.sum(plg_nr * leng_tile * x_tile, axis=-1) / (plg_area * radius)


def grad_at_edge(
    x_at_primal_cell: Float[Array, "nPrimal_cell"],
    radius: FloatType,
    edd_v: Int[Array, "nEdge 2"],
    edd_length: Float[Array, "nEdge"],
    edpNr_eddTg: Int[Array, "nEdge"]
) -> Float[Array, "nEdge"]:
    """Compute gradient at edge.

    Calculates the gradient of a scalar field at edges using finite differences
    between adjacent primal cells.
    """
    vertex = edd_v  # (nedg, 2)
    x_at_vertex = x_at_primal_cell[vertex]  # (nedg, 2)
    q0 = x_at_vertex[:, 0]
    q1 = x_at_vertex[:, 1]
    return (q1 - q0) * edpNr_eddTg / (radius * edd_length)


def curl_at_dual_cell(
    x_at_edge: Float[Array, "nEdge"],
    radius: FloatType,
    dua_ed: Int[Array, "nDual_cell maxnnb"],
    dua_nr: Float[Array, "nDual_cell maxnnb"],
    dua_area: Float[Array, "nDual_cell"],
    edd_length: Float[Array, "nEdge"],
    edpTg_eddNr: Int[Array, "nEdge"]
) -> Float[Array, "nDual_cell"]:
    """Compute curl at dual cell center.

    Calculates the curl of a vector field at dual cell centers using
    the finite volume method on an orthogonal dual mesh.
    """
    ed_idx = dua_ed  # (nDual_cell, maxnnb)
    x_tile = x_at_edge[ed_idx]  # (nDual_cell, maxnnb)
    x_tile = jnp.where(ed_idx < 0, 0.0, x_tile)
    leng_tile = edd_length[ed_idx]  # (nDual_cell, maxnnb)
    edpTg_eddNr_tile = edpTg_eddNr[ed_idx]  # (nDual_cell, maxnnb)
    return jnp.sum(
        x_tile * edpTg_eddNr_tile * dua_nr * leng_tile, axis=-1
    ) / (dua_area * radius)
