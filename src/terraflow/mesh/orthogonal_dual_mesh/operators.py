from typing import TypeVar
from jax import numpy as jnp
import jax

from terraflow.util import typing
from terraflow.planet import PlanetParameterSet
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.layout.abstract import AbstractLayout
from terraflow.layout.abstract import CoordinateSystem

FloatType = typing.FloatType
Int = typing.Int
Float = typing.Float
Array = typing.Array

VM = TypeVar("VM", OneLayerVertCoordinate, HybridVertCoordinate)
HLO = TypeVar("HLO", bound=AbstractLayout)
VLO = TypeVar("VLO", bound=AbstractLayout)

CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    VM,
    HLO,
    VLO,
]


def div_at_primal_cell(
    *,
    x_at_edge: Float[Array, "nlev nEdge"],
    coord: CoordSys,
    para: PlanetParameterSet,
) -> Float[Array, "nlev nPrimal_cell"]:
    """Divergence operator at primal cell center."""
    float_type = coord.dev.float_type
    plg_ed = coord.mesh.primal.cell.edges  # (nplg, maxnnb)
    plg_nr = coord.mesh.primal.cell.orient  # (nplg, maxnnb)
    plg_area = coord.mesh.primal.cell.area  # (nplg,)
    edp_length = coord.mesh.primal.edge.length  # (nedg,)
    radius = para.radius.unwrap()

    def _wrap(
        x_at_edge: Float[Array, "nEdge"],
    ) -> Float[Array, "nPrimal_cell"]:
        ed_idx = plg_ed  # (nplg, maxnnb)
        leng_tile = edp_length[ed_idx]  # (nplg, maxnnb)
        x_tile = x_at_edge[ed_idx]  # (nplg, maxnnb)
        x_tile = jnp.where(ed_idx < 0, 0.0, x_tile)
        return jnp.sum(
            plg_nr * leng_tile * x_tile,
            dtype=jnp.float64,
            axis=-1
        ).astype(float_type) / (plg_area * radius)

    return jax.vmap(_wrap, in_axes=0, out_axes=0)(x_at_edge)


def grad_at_dual_edge(
    *,
    x_at_primal_cell: Float[Array, "nlev nPrimal_cell"],
    coord: CoordSys,
    para: PlanetParameterSet,
) -> Float[Array, "nlev nEdge"]:
    """Gradient operator at edge."""
    edd_v = coord.mesh.dual.edge.endpoints  # (nedg, 2)
    edd_length = coord.mesh.dual.edge.length  # (nedg,)
    edpnr_eddtg = coord.mesh.aux.edpnr_eddtg  # (nedg,)
    radius = para.radius.unwrap()

    def _wrap(
        x_at_primal_cell: Float[Array, "nPrimal_cell"]
    ) -> Float[Array, "nEdge"]:
        x_at_vertex = x_at_primal_cell[edd_v]  # (nedg, 2)
        q0 = x_at_vertex[:, 0]
        q1 = x_at_vertex[:, 1]
        return (q1 - q0) * edpnr_eddtg / (radius * edd_length)

    return jax.vmap(_wrap, in_axes=0, out_axes=0)(x_at_primal_cell)


def curl_at_dual_cell(
    *,
    x_at_dual_edge: Float[Array, "nlev nEdge"],
    coord: CoordSys,
    para: PlanetParameterSet,
) -> Float[Array, "nlev nDual_cell"]:
    """Curl operator at dual cell center."""
    float_type = coord.dev.float_type
    dua_ed = coord.mesh.dual.cell.edges  # (ndua, maxnnb)
    dua_nr = coord.mesh.dual.cell.orient  # (ndua, maxnnb)
    dua_area = coord.mesh.dual.cell.area  # (ndua,)
    edd_length = coord.mesh.dual.edge.length  # (nedg,)
    edptg_eddnr = coord.mesh.aux.edptg_eddnr  # (nedg,)
    radius = para.radius.unwrap()

    def _wrap(
        x_at_dual_edge: Float[Array, "nEdge"],
    ) -> Float[Array, "nDual_cell"]:
        ed_idx = dua_ed  # (nDual_cell, maxnnb)
        x_tile = x_at_dual_edge[ed_idx]  # (nDual_cell, maxnnb)
        x_tile = jnp.where(ed_idx < 0, 0.0, x_tile)
        leng_tile = edd_length[ed_idx]  # (nDual_cell, maxnnb)
        edpTg_eddNr_tile = edptg_eddnr[ed_idx]  # (nDual_cell, maxnnb)
        return jnp.sum(
            x_tile * edpTg_eddNr_tile * dua_nr * leng_tile,
            dtype=jnp.float64,
            axis=-1
        ).astype(float_type) / (dua_area * radius)

    return jax.vmap(_wrap, in_axes=0, out_axes=0)(x_at_dual_edge)


def interp_weighted_add_edp2plg(
    *,
    x_at_primal_edge: Float[Array, "nlev nEdge"],
    coord: CoordSys,
) -> Float[Array, "nlev nPrimal_cell"]:
    """Interpolation from primal edge to primal cell."""
    float_type = coord.dev.float_type

    def _wrap(
        x_at_primal_edge: Float[Array, "nEdge"],
    ) -> Float[Array, "nPrimal_cell"]:
        plg_ed = coord.mesh.primal.cell.edges  # (nplg, maxnnb)
        plg_area = coord.mesh.primal.cell.area  # (nplg,)
        edp_length = coord.mesh.primal.edge.length  # (nedg,)
        edd_length = coord.mesh.dual.edge.length  # (nedg,)
        ed_idx = plg_ed  # (nplg, maxnnb)
        Ae = edp_length[ed_idx] * edd_length[ed_idx] / 4.0
        x_tile = x_at_primal_edge[ed_idx]
        x_tile = jnp.where(ed_idx < 0, 0.0, x_tile)
        return jnp.sum(
            Ae * x_tile,
            dtype=jnp.float64,
            axis=-1
        ).astype(float_type) / plg_area

    return jax.vmap(_wrap, in_axes=0, out_axes=0)(x_at_primal_edge)


def interp_weighted_add_edd2dua(
    *,
    x_at_dual_edge: Float[Array, "nlev nEdge"],
    coord: CoordSys,
) -> Float[Array, "nlev nDual_cell"]:
    """Interpolation from dual edge to dual cell."""
    float_type = coord.dev.float_type

    def _wrap(
        x_at_dual_edge: Float[Array, "nEdge"],
    ) -> Float[Array, "nDual_cell"]:
        dua_ed = coord.mesh.dual.cell.edges  # (ndual, maxnnb)
        dua_area = coord.mesh.dual.cell.area  # (ndual,)
        edp_length = coord.mesh.primal.edge.length  # (nedg,)
        edd_length = coord.mesh.dual.edge.length  # (nedg,)
        ed_idx = dua_ed  # (ndual, maxnnb)
        Ae = edd_length[ed_idx] * edp_length[ed_idx] / 4.0
        x_tile = x_at_dual_edge[ed_idx]
        x_tile = jnp.where(ed_idx < 0, 0.0, x_tile)
        return jnp.sum(
            Ae * x_tile,
            dtype=jnp.float64,
            axis=-1
        ).astype(float_type) / dua_area

    return jax.vmap(_wrap, in_axes=0, out_axes=0)(x_at_dual_edge)


def interp_weighted_add_plg2dua(
    *,
    x_at_primal_cell: Float[Array, "nlev nPrimal_cell"],
    coord: CoordSys,
) -> Float[Array, "nlev nDual_cell"]:
    """Interpolation from primal cell to dual cell."""
    float_type = coord.dev.float_type

    def _wrap(
        x_at_primal_cell: Float[Array, "nPrimal_cell"],
    ) -> Float[Array, "nDual_cell"]:
        dua_v = coord.mesh.dual.cell.nodes  # (ndual, maxnnb)
        dua_kite_area = coord.mesh.dual.cell.kite_area  # (ndual, maxnnb)
        dua_area = coord.mesh.dual.cell.area  # (ndual,)
        plg_idx = dua_v  # (ndual, maxnnb)
        x_tile = x_at_primal_cell[plg_idx]  # (ndual, maxnnb)
        x_tile = jnp.where(plg_idx < 0, 0.0, x_tile)  # (ndual, maxnnb)
        Adv = dua_kite_area
        return jnp.sum(
            Adv * x_tile,
            dtype=jnp.float64,
            axis=-1
        ).astype(float_type) / dua_area

    return jax.vmap(_wrap, in_axes=0, out_axes=0)(x_at_primal_cell)


def interp_weighted_add_dua2plg(
    *,
    x_at_dual_cell: Float[Array, "nlev nDual_cell"],
    coord: CoordSys,
) -> Float[Array, "nlev nPrimal_cell"]:
    """Interpolation from dual cell to primal cell."""
    float_type = coord.dev.float_type

    def _wrap(
        x_at_dual_cell: Float[Array, "nDual_cell"],
    ) -> Float[Array, "nPrimal_cell"]:
        plg_v = coord.mesh.primal.cell.nodes  # (nplg, maxnnb)
        plg_kite_area = coord.mesh.primal.cell.kite_area  # (nplg, maxnnb)
        plg_area = coord.mesh.primal.cell.area  # (nplg,)
        dua_idx = plg_v  # (nplg, maxnnb)
        x_tile = x_at_dual_cell[dua_idx]  # (nplg, maxnnb)
        x_tile = jnp.where(dua_idx < 0, 0.0, x_tile)  # (nplg, maxnnb)
        Adv = plg_kite_area
        return jnp.sum(
            Adv * x_tile,
            dtype=jnp.float64,
            axis=-1
        ).astype(float_type) / plg_area

    return jax.vmap(_wrap, in_axes=0, out_axes=0)(x_at_dual_cell)
