from __future__ import annotations
from typing import TypeVar, Callable, TYPE_CHECKING
from jax import numpy as jnp
import jax

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.planet import PlanetParameterSet
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.abstract import AbstractLayout
    from terraflow.layout.abstract import CoordinateSystem

    IntType = typing.IntType
    FloatType = typing.FloatType
    Int = typing.Int
    Float = typing.Float
    Array = typing.Array

    HLO = TypeVar("HLO", bound=AbstractLayout)
    VLO = TypeVar("VLO", bound=AbstractLayout)

    CoordSys_OneLayer = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        OneLayerVertCoordinate,
        HLO,
        VLO,
    ]

    CoordSys_Hybrid = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        HybridVertCoordinate,
        HLO,
        VLO,
    ]

    CoordSys = CoordSys_OneLayer | CoordSys_Hybrid


def div_at_primal_cell(
    *,
    x_at_primal_edge: Float[Array, "nlev nEdge"],
    coord: CoordSys,
    para: PlanetParameterSet,
    x_at_primal_cell: Float[Array, "nlev nPrimal_cell"] | None = None,
    hori_end_at: str = "domain",
    vert_entity: str = "full",
) -> Float[Array, "nlev nPolg"]:
    """Divergence operator at primal cell center."""
    mesh = coord.mesh
    vert = coord.vert
    i_s, i_e = mesh.get_idx(
        entity="polg",
        start_at="domain",
        end_at=hori_end_at
    )
    k_s, k_e = vert.get_idx(entity=vert_entity, domain="domain")
    plg_ed = mesh.primal.cell.edges[i_s:i_e+1, :]  # (nplg, maxnnb)
    plg_nr = mesh.primal.cell.orient[i_s:i_e+1, :]  # (nplg, maxnnb)
    plg_area = mesh.primal.cell.area[i_s:i_e+1]  # (nplg,)
    edp_length = mesh.primal.edge.length  # (nedg,)
    radius = para.radius.unwrap()
    # ==========================================================

    def _wrap(
        x_at_edge: Float[Array, "nEdge"],
    ) -> Float[Array, "nPolg_cmpt"]:
        ed_idx = plg_ed  # (nplg, maxnnb)
        leng_tile = edp_length[ed_idx]  # (nplg, maxnnb)
        x_tile = x_at_edge[ed_idx]  # (nplg, maxnnb)
        x_tile = jnp.where(ed_idx < 0, 0.0, x_tile)
        return jnp.sum(
            plg_nr * leng_tile * x_tile,
            axis=-1
        ) / (plg_area * radius)

    res = jax.vmap(
        _wrap,
        in_axes=0,
        out_axes=0
    )(x_at_primal_edge[k_s:k_e+1, :])

    # ==========================================================
    if x_at_primal_cell is None:
        return res
    else:
        return x_at_primal_cell.at[k_s:k_e+1, i_s:i_e+1].set(res)


def grad_at_dual_edge(
    *,
    x_at_primal_cell: Float[Array, "nlev nPrimal_cell"],
    coord: CoordSys,
    para: PlanetParameterSet,
    x_at_dual_edge: Float[Array, "nlev nEdge"] | None = None,
    hori_end_at: str = "domain",
    vert_entity: str = "full",
) -> Float[Array, "nlev nEdge"]:
    """Gradient operator at edge."""
    mesh = coord.mesh
    vert = coord.vert
    i_s, i_e = mesh.get_idx(
        entity="edge",
        start_at="domain",
        end_at=hori_end_at
    )
    k_s, k_e = vert.get_idx(entity=vert_entity, domain="domain")
    edd_v = mesh.dual.edge.endpoints[i_s:i_e+1, :]  # (nedg, 2)
    edd_length = mesh.dual.edge.length[i_s:i_e+1]  # (nedg,)
    edpnr_eddtg = mesh.aux.edpnr_eddtg[i_s:i_e+1]  # (nedg,)
    radius = para.radius.unwrap()
    # ==========================================================

    def _wrap(
        x_at_primal_cell: Float[Array, "nPrimal_cell"]
    ) -> Float[Array, "nEdge"]:
        x_at_vertex = x_at_primal_cell[edd_v]  # (nedg, 2)
        q0 = x_at_vertex[:, 0]
        q1 = x_at_vertex[:, 1]
        return (q1 - q0) * edpnr_eddtg / (radius * edd_length)

    res = jax.vmap(
        _wrap,
        in_axes=0,
        out_axes=0
    )(x_at_primal_cell[k_s:k_e+1, :])

    # ==========================================================

    if x_at_dual_edge is None:
        return res
    else:
        return x_at_dual_edge.at[k_s:k_e+1, i_s:i_e+1].set(res)


def curl_at_dual_cell(
    *,
    x_at_dual_edge: Float[Array, "nlev nEdge"],
    coord: CoordSys,
    para: PlanetParameterSet,
    x_at_dual_cell: Float[Array, "nlev nDual_cell"] | None = None,
    hori_end_at: str = "domain",
    vert_entity: str = "full",
) -> Float[Array, "nlev nDual_cell"]:
    """Curl operator at dual cell center."""
    mesh = coord.mesh
    vert = coord.vert
    i_s, i_e = mesh.get_idx(
        entity="dual",
        start_at="domain",
        end_at=hori_end_at
    )
    k_s, k_e = vert.get_idx(entity=vert_entity, domain="domain")
    dua_ed = mesh.dual.cell.edges[i_s:i_e+1, :]  # (ndua, maxnnb)
    dua_nr = mesh.dual.cell.orient[i_s:i_e+1, :]  # (ndua, maxnnb)
    dua_area = mesh.dual.cell.area[i_s:i_e+1]  # (ndua,)
    edd_length = mesh.dual.edge.length  # (nedg,)
    edptg_eddnr = mesh.aux.edptg_eddnr  # (nedg,)
    radius = para.radius.unwrap()
    # ==========================================================

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
            axis=-1
        ) / (dua_area * radius)

    res = jax.vmap(
        _wrap,
        in_axes=0,
        out_axes=0
    )(x_at_dual_edge[k_s:k_e+1, :])

    # ==========================================================

    if x_at_dual_cell is None:
        return res
    else:
        return x_at_dual_cell.at[k_s:k_e+1, i_s:i_e+1].set(res)


def interp_weighted_add_edp2plg(
    *,
    x_at_primal_edge: Float[Array, "nlev nEdge"],
    coord: CoordSys,
    field_fn: Callable = lambda x: x,
    x_at_primal_cell: Float[Array, "nlev nPrimal_cell"] | None = None,
) -> Float[Array, "nlev nPrimal_cell"]:
    """Interpolation from primal edge to primal cell."""
    mesh = coord.mesh
    vert = coord.vert
    ids, ide = mesh.polg_ids, mesh.polg_ide
    kds, kde = vert.full_kds, vert.full_kde
    plg_ed = mesh.primal.cell.edges[ids:ide+1, :]  # (nplg, maxnnb)
    plg_area = mesh.primal.cell.area[ids:ide+1]  # (nplg,)
    edp_length = mesh.primal.edge.length  # (nedg,)
    edd_length = mesh.dual.edge.length  # (nedg,)
    ed_idx = plg_ed  # (nplg, maxnnb)
    # ==========================================================

    def _wrap(
        x_at_primal_edge: Float[Array, "nEdge"],
    ) -> Float[Array, "nPrimal_cell"]:
        Ae = edp_length[ed_idx] * edd_length[ed_idx] / 4.0
        x_tile = jnp.array(
            field_fn(x_at_primal_edge[ed_idx]),
            dtype=x_at_primal_edge.dtype
        )  # (nplg, maxnnb)
        x_tile = jnp.where(ed_idx < 0, 0.0, x_tile)
        return jnp.sum(
            Ae * x_tile,
            axis=-1
        ) / plg_area

    res = jax.vmap(
        _wrap,
        in_axes=0,
        out_axes=0
    )(x_at_primal_edge[kds:kde+1, :])

    # ==========================================================
    if x_at_primal_cell is None:
        return res
    else:
        return x_at_primal_cell.at[kds:kde+1, ids:ide+1].set(res)


def interp_weighted_add_edd2dua(
    *,
    x_at_dual_edge: Float[Array, "nlev nEdge"],
    coord: CoordSys,
    field_fn: Callable = lambda x: x,
    x_at_dual_cell: Float[Array, "nlev nDual_cell"] | None = None,
) -> Float[Array, "nlev nDual_cell"]:
    """Interpolation from dual edge to dual cell."""
    mesh = coord.mesh
    vert = coord.vert
    ids, ide = mesh.dual_ids, mesh.dual_ide
    kds, kde = vert.full_kds, vert.full_kde
    dua_ed = mesh.dual.cell.edges[ids:ide+1, :]  # (ndual, maxnnb)
    dua_area = mesh.dual.cell.area[ids:ide+1]  # (ndual,)
    edp_length = mesh.primal.edge.length  # (nedg,)
    edd_length = mesh.dual.edge.length  # (nedg,)
    ed_idx = dua_ed  # (ndual, maxnnb)
    # ==========================================================

    def _wrap(
        x_at_dual_edge: Float[Array, "nEdge"],
    ) -> Float[Array, "nDual_cell"]:
        Ae = edd_length[ed_idx] * edp_length[ed_idx] / 4.0
        x_tile = jnp.array(
            field_fn(x_at_dual_edge[ed_idx]),
            dtype=x_at_dual_edge.dtype
        )  # (ndual, maxnnb)
        x_tile = jnp.where(ed_idx < 0, 0.0, x_tile)
        return jnp.sum(
            Ae * x_tile,
            axis=-1
        ) / dua_area

    res = jax.vmap(
        _wrap,
        in_axes=0,
        out_axes=0
    )(x_at_dual_edge[kds:kde+1, :])

    # ==========================================================
    if x_at_dual_cell is None:
        return res
    else:
        return x_at_dual_cell.at[kds:kde+1, ids:ide+1].set(res)


def interp_weighted_add_plg2dua(
    *,
    x_at_primal_cell: Float[Array, "nlev nPrimal_cell"],
    coord: CoordSys,
    field_fn: Callable = lambda x: x,
    x_at_dual_cell: Float[Array, "nlev nDual_cell"] | None = None,
) -> Float[Array, "nlev nDual_cell"]:
    """Interpolation from primal cell to dual cell."""
    mesh = coord.mesh
    vert = coord.vert
    ids, ide = mesh.dual_ids, mesh.dual_ide
    kds, kde = vert.full_kds, vert.full_kde
    dua_v = mesh.dual.cell.nodes[ids:ide+1, :]  # (ndual, maxnnb)
    dua_kite_area = mesh.dual.cell.kite_area[ids:ide+1, :]  # (ndual, maxnnb)
    dua_area = mesh.dual.cell.area[ids:ide+1]  # (ndual,)
    plg_idx = dua_v  # (ndual, maxnnb)
    # ==========================================================

    def _wrap(
        x_at_primal_cell: Float[Array, "nPrimal_cell"],
    ) -> Float[Array, "nDual_cell"]:
        x_tile = jnp.array(
            field_fn(x_at_primal_cell[plg_idx]),
            dtype=x_at_primal_cell.dtype
        )  # (ndual, maxnnb)
        x_tile = jnp.where(plg_idx < 0, 0.0, x_tile)  # (ndual, maxnnb)
        Adv = dua_kite_area
        return jnp.sum(
            Adv * x_tile,
            axis=-1
        ) / dua_area

    res = jax.vmap(
        _wrap,
        in_axes=0,
        out_axes=0
    )(x_at_primal_cell[kds:kde+1, :])

    # ==========================================================
    if x_at_dual_cell is None:
        return res
    else:
        return x_at_dual_cell.at[kds:kde+1, ids:ide+1].set(res)


def interp_weighted_add_dua2plg(
    *,
    x_at_dual_cell: Float[Array, "nlev nDual_cell"],
    coord: CoordSys,
    field_fn: Callable = lambda x: x,
    x_at_primal_cell: Float[Array, "nlev nPrimal_cell"] | None = None,
) -> Float[Array, "nlev nPrimal_cell"]:
    """Interpolation from dual cell to primal cell."""
    mesh = coord.mesh
    vert = coord.vert
    ids, ide = mesh.polg_ids, mesh.polg_ide
    kds, kde = vert.full_kds, vert.full_kde
    plg_v = mesh.primal.cell.nodes[ids:ide+1, :]  # (nplg, maxnnb)
    plg_kite_area = mesh.primal.cell.kite_area[ids:ide+1, :]  # (nplg, maxnnb)
    plg_area = mesh.primal.cell.area[ids:ide+1]  # (nplg,)
    dua_idx = plg_v  # (nplg, maxnnb)
    # ==========================================================

    def _wrap(
        x_at_dual_cell: Float[Array, "nDual_cell"],
    ) -> Float[Array, "nPrimal_cell"]:
        x_tile = jnp.array(
            field_fn(x_at_dual_cell[dua_idx]),
            dtype=x_at_dual_cell.dtype
        )  # (nplg, maxnnb)
        x_tile = jnp.where(dua_idx < 0, 0.0, x_tile)  # (nplg, maxnnb)
        Adv = plg_kite_area
        return jnp.sum(
            Adv * x_tile,
            axis=-1
        ) / plg_area

    res = jax.vmap(
        _wrap,
        in_axes=0,
        out_axes=0
    )(x_at_dual_cell[kds:kde+1, :])

    # ==========================================================
    if x_at_primal_cell is None:
        return res
    else:
        return x_at_primal_cell.at[kds:kde+1, ids:ide+1].set(res)
