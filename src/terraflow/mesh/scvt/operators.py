from __future__ import annotations
from typing import TYPE_CHECKING

from terraflow.util.sphere_geometry import (
    project_sphere2plane,
    vector_convert_lonlat2xyz,
    vector_convert_xyz2lonlat
)

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.mesh.scvt.mesh_type import SCVTMesh

    IntType = typing.IntType
    FloatType = typing.FloatType
    Int = typing.Int
    Float = typing.Float
    Array = typing.Array


def div_operator_1d(
    mesh: SCVTMesh,
    f: Float[Array, "nedge"],
    div: Float[Array, "npoint"],
    div_loc: str,
    with_halo: bool = False
) -> Array:
    """Divergence operator"""
    bkd = mesh.dev.backend
    if div_loc == "cell":
        i_s = mesh.cell_ids
        i_e = mesh.cell_ihe[0] if with_halo else mesh.cell_ide
        cell_ed = mesh.cell_ed[i_s:i_e+1, :]  # (ncell, maxnnb)
        cell_nr = mesh.cell_nr[i_s:i_e+1, :]  # (ncell, maxnnb)
        area_cell = mesh.area_cell[i_s:i_e+1]  # (ncell,)
        le = mesh.le  # (nedge,)
        # ==========================================================
        ed_idx  = cell_ed  # (ncell, maxnnb)
        valid_ed = ed_idx >= 0
        safe_ed_idx = bkd.where(valid_ed, ed_idx, 0)
        le_tile = bkd.take(le, safe_ed_idx, axis=0)  # (ncell, maxnnb)
        x_tile  = bkd.take(f, safe_ed_idx, axis=0)  # (ncell, maxnnb)
        div_tile = cell_nr * le_tile * x_tile
        div_tile = bkd.where(valid_ed, div_tile, 0.0)
        res = bkd.sum(
            div_tile,
            axis=-1
        ) / area_cell
        # ==========================================================
    else:
        raise ValueError(f"div_operator_1d: unsupported location {div_loc}!")
    return bkd.slice_update(div, slice(i_s, i_e+1), res)


def div_operator_2d(
    mesh: SCVTMesh,
    f: Float[Array, "nlev nedge"],
    div: Float[Array, "nlev npoint"],
    div_loc: str,
    with_halo: bool = False
) -> Float[Array, "nlev nPolg"]:
    """Divergence operator"""
    bkd = mesh.dev.backend
    k_s = mesh.full_kds if div_loc != "lev" else mesh.half_kds
    k_e = mesh.full_kde if div_loc != "lev" else mesh.half_kde
    if div_loc == "cell" or div_loc == "lev":
        i_s = mesh.cell_ids
        i_e = mesh.cell_ihe[0] if with_halo else mesh.cell_ide
        cell_ed = mesh.cell_ed[i_s:i_e+1, :]  # (ncell, maxnnb)
        cell_nr = mesh.cell_nr[i_s:i_e+1, :]  # (ncell, maxnnb)
        area_cell = mesh.area_cell[i_s:i_e+1]  # (ncell,)
        le = mesh.le  # (nedge,)
        # ======================================================
        def _wrap(f_1d: Float[Array, "nedge"]) -> Float[Array, "ncell"]:
            ed_idx = cell_ed  # (ncell, maxnnb)
            valid_ed = ed_idx >= 0
            safe_ed_idx = bkd.where(valid_ed, ed_idx, 0)
            leng_tile = bkd.take(le, safe_ed_idx, axis=0)  # (ncell, maxnnb)
            x_tile = bkd.take(f_1d, safe_ed_idx, axis=0)  # (ncell, maxnnb)
            div_tile = cell_nr * leng_tile * x_tile
            div_tile = bkd.where(valid_ed, div_tile, 0.0)
            return bkd.sum(
                div_tile,
                axis=-1
            ) / area_cell

        res = bkd.vmap(
            _wrap,
            in_axes=0,
            out_axes=0
        )(f[k_s:k_e+1, :])
    else:
        raise ValueError(f"div_operator_2d: unsupported location {div_loc}!")
    # ==========================================================
    return bkd.slice_update(div, (slice(k_s, k_e+1), slice(i_s, i_e+1)), res)


def curl_operator_1d(
    mesh: SCVTMesh,
    f: Float[Array, "nedge"],
    curl: Float[Array, "npoint"],
    curl_loc: str,
    with_halo: bool = False
) -> Float[Array, "nlev npoint"]:
    """Curl operator"""
    bkd = mesh.dev.backend
    if curl_loc == "vtx":
        i_s = mesh.vtx_ids
        i_e = mesh.vtx_ihe[0] if with_halo else mesh.vtx_ide
        vtx_ed = mesh.vtx_ed[i_s:i_e+1, :]  # (nvtx, maxnnb)
        vtx_nr = mesh.vtx_nr[i_s:i_e+1, :]  # (nvtx, maxnnb)
        area_vtx = mesh.area_vtx[i_s:i_e+1]  # (nvtx,)
        de = mesh.de  # (nedge,)
        edptg_eddnr = mesh.edptg_eddnr  # (nedge,)
        # ==========================================================
        ed_idx = vtx_ed  # (nvtx, maxnnb)
        valid_ed = ed_idx >= 0
        safe_ed_idx = bkd.where(valid_ed, ed_idx, 0)
        f_tile = bkd.take(f, safe_ed_idx, axis=0)  # (nvtx, maxnnb)
        de_tile = bkd.take(de, safe_ed_idx, axis=0)  # (nvtx, maxnnb)
        edptg_eddnr_tile = bkd.take(edptg_eddnr, safe_ed_idx, axis=0)  # (nvtx, maxnnb)
        curl_tile = f_tile * edptg_eddnr_tile * vtx_nr * de_tile
        curl_tile = bkd.where(valid_ed, curl_tile, 0.0)
        res = bkd.sum(
            curl_tile,
            axis=-1
        ) / area_vtx
    else:
        raise ValueError(f"curl_operator_1d: unsupported location {curl_loc}!")
    # ==========================================================
    return bkd.slice_update(curl, slice(i_s, i_e+1), res)


def curl_operator_2d(
    mesh: SCVTMesh,
    f: Float[Array, "nlev nedge"],
    curl: Float[Array, "nlev npoint"],
    curl_loc: str,
    with_halo: bool = False
) -> Float[Array, "nlev npoint"]:
    """Curl operator"""
    bkd = mesh.dev.backend
    k_s = mesh.full_kds if curl_loc != "lev" else mesh.half_kds
    k_e = mesh.full_kde if curl_loc != "lev" else mesh.half_kde
    if curl_loc == "vtx" or curl_loc == "lev_vtx":
        i_s = mesh.vtx_ids
        i_e = mesh.vtx_ihe[0] if with_halo else mesh.vtx_ide
        vtx_ed = mesh.vtx_ed[i_s:i_e+1, :]  # (nvtx, maxnnb)
        vtx_nr = mesh.vtx_nr[i_s:i_e+1, :]  # (nvtx, maxnnb)
        area_vtx = mesh.area_vtx[i_s:i_e+1]  # (nvtx,)
        de = mesh.de  # (nedge,)
        edptg_eddnr = mesh.edptg_eddnr  # (nedge,)
        # ==========================================================

        def _wrap(f_1d: Float[Array, "nedge"]) -> Float[Array, "nvtx"]:
            ed_idx = vtx_ed  # (nvtx, maxnnb)
            valid_ed = ed_idx >= 0
            safe_ed_idx = bkd.where(valid_ed, ed_idx, 0)
            f_tile = bkd.take(f_1d, safe_ed_idx, axis=0)  # (nvtx, maxnnb)
            de_tile = bkd.take(de, safe_ed_idx, axis=0)  # (nvtx, maxnnb)
            edptg_eddnr_tile = bkd.take(edptg_eddnr, safe_ed_idx, axis=0)  # (nvtx, maxnnb)
            curl_tile = f_tile * edptg_eddnr_tile * vtx_nr * de_tile
            curl_tile = bkd.where(valid_ed, curl_tile, 0.0)
            return bkd.sum(
                curl_tile,
                axis=-1
            ) / area_vtx

        res = bkd.vmap(
            _wrap,
            in_axes=0,
            out_axes=0
        )(f[k_s:k_e+1, :])
    else:
        raise ValueError(f"curl_operator_2d: unsupported location {curl_loc}!")
    # ==========================================================
    return bkd.slice_update(curl, (slice(k_s, k_e+1), slice(i_s, i_e+1)), res)


def grad_operator_1d(
    mesh: SCVTMesh,
    f: Float[Array, "npoint"],
    grad: Float[Array, "nedge"],
    f_loc: str,
    with_halo: bool = False
) -> Float[Array, "nlev nEdge"]:
    """Gradient operator"""
    bkd = mesh.dev.backend
    if f_loc == "cell":
        i_s = mesh.edge_ids
        i_e = mesh.edge_ihe[0] if with_halo else mesh.edge_ide
        cell_of_edge = mesh.cell_of_edge[i_s:i_e+1, :]  # (nedge, 2)
        de = mesh.de[i_s:i_e+1]  # (nedge,)
        edpnr_eddtg = mesh.edpnr_eddtg[i_s:i_e+1]  # (nedge,)
        # ======================================================
        q0 = f[cell_of_edge[:, 0]]
        q1 = f[cell_of_edge[:, 1]]
        res = (q1 - q0) * edpnr_eddtg / de
    else:
        raise ValueError(f"grad_operator_1d: unsupported location {f_loc}!")
    # ==========================================================
    return bkd.slice_update(grad, slice(i_s, i_e+1), res)
    

def grad_operator_2d(
    mesh: SCVTMesh,
    f: Float[Array, "nlev npoint"],
    grad: Float[Array, "nlev nedge"],
    f_loc: str,
    with_halo: bool = False
) -> Float[Array, "nlev nEdge"]:
    """Gradient operator"""
    bkd = mesh.dev.backend
    k_s = mesh.full_kds if f_loc != "lev" else mesh.half_kds
    k_e = mesh.full_kde if f_loc != "lev" else mesh.half_kde
    if f_loc == "cell":
        i_s = mesh.edge_ids
        i_e = mesh.edge_ihe[0] if with_halo else mesh.edge_ide
        cell_of_edge = mesh.cell_of_edge[i_s:i_e+1, :]  # (nedge, 2)
        de = mesh.de[i_s:i_e+1]  # (nedge,)
        edpnr_eddtg = mesh.edpnr_eddtg[i_s:i_e+1]  # (nedge,)
        # ======================================================
        def _wrap(f_1d: Float[Array, "npoint"]) -> Float[Array, "nEdge"]:
            q0 = f_1d[cell_of_edge[:, 0]]
            q1 = f_1d[cell_of_edge[:, 1]]
            return (q1 - q0) * edpnr_eddtg / de

        res = bkd.vmap(
            _wrap,
            in_axes=0,
            out_axes=0
        )(f[k_s:k_e+1, :])
    else:
        raise ValueError(f"grad_operator_2d: unsupported location {f_loc} !")
    # ==========================================================
    return bkd.slice_update(grad, (slice(k_s, k_e+1), slice(i_s, i_e+1)), res)


def wind_c2a_operator(
    R: FloatType,
    mesh: SCVTMesh,
    un: Float[Array, "nlev nedge"],
    u: Float[Array, "nlev ncell"],
    v: Float[Array, "nlev, ncell"]
) -> tuple[Float[Array, "nlev, ncell"], Float[Array, "nlev, ncell"]]:
    kds, kde      = mesh.full_kds, mesh.full_kde
    ids, ide      = mesh.cell_ids, mesh.cell_ide
    cell_xyz      = mesh.cell_xyz  [ids:ide+1, :]   # (ncell, 3)
    cell_ed       = mesh.cell_ed    [ids:ide+1, :]  # (ncell, maxnnb)
    cell_nnb      = mesh.cell_nnb  [ids:ide+1]      # (ncell,)
    area_cell     = mesh.area_cell[ids:ide+1]       # (ncell,)
    dual_edge_xyz = mesh.dual_edge_xyz              # (nedge, 3)
    edge_nr       = mesh.edge_nr                    # (nedge, 3)
    le            = mesh.le                         # (nedge,)
    bkd           = mesh.dev.backend

    eps = bkd.eps(cell_xyz.dtype)
    valid_cell_ed = cell_ed >= 0
    safe_cell_ed = bkd.where(valid_cell_ed, cell_ed, 0)

    def _wrap_at_single_lev(un: Float[Array, "nedge"]) -> dict[str, Float[Array, "ncell"]]:

        # Vectorized computation over all cells
        # Gather edge data for all cells at once: (ncell, maxnnb, ...)
        OPi  = bkd.take(dual_edge_xyz, safe_cell_ed, axis=0)  # (ncell, maxnnb, 3)
        nei  = bkd.take(edge_nr, safe_cell_ed, axis=0) * R  # (ncell, maxnnb, 3)
        wind = bkd.take(un, safe_cell_ed, axis=0)  # (ncell, maxnnb)
        leng = bkd.take(le, safe_cell_ed, axis=0)  # (ncell, maxnnb)

        # Create mask for valid neighbors: (ncell, maxnnb)
        # Use shape inference to get maxnnb instead of accessing it as a field
        maxnnb = cell_ed.shape[1]
        pos = bkd.arange(maxnnb)[None, :]  # (1, maxnnb)
        mask = (pos < cell_nnb[:, None]) & valid_cell_ed  # (ncell, maxnnb)

        # Project edge midpoints onto tangent plane at cell centroids
        # OPc: (ncell, 3) -> (ncell, 1, 3) for broadcasting
        OPc = cell_xyz[:, None, :]  # (ncell, 1, 3)
        rei = bkd.vmap(
            lambda R, opc, opi: project_sphere2plane(bkd, R, opc[0], opi),
            in_axes=(None, 0, 0)
        )(R, OPc, OPi)  # (ncell, maxnnb, 3)

        # Compute norms and normalize
        rei_norm = bkd.norm(rei, axis=-1)  # (ncell, maxnnb)
        rei_norm = bkd.maximum(rei_norm, eps * 100)
        rei_unit = rei / rei_norm[:, :, None]  # (ncell, maxnnb, 3)

        # Dot product: nei . rei_unit
        dot_val = bkd.sum(nei * rei_unit, axis=-1)  # (ncell, maxnnb)

        # Contribution per neighbor: rei * wind * dot_val * leng
        # (ncell, maxnnb, 3)
        contrib = rei * (wind * dot_val * leng)[:, :, None]

        # Apply mask to zero out invalid neighbors
        contrib = contrib * mask[:, :, None]  # (ncell, maxnnb, 3)

        # Sum contributions across neighbors
        vector_xyz = bkd.sum(contrib, axis=1)  # (ncell, 3)

        # Divide by cell area
        area = bkd.maximum(area_cell, eps * 100)  # (ncell,)
        vector_xyz = vector_xyz / area[:, None]  # (ncell, 3)

        # Convert to lat-lon coordinates
        result = vector_convert_xyz2lonlat(bkd, R, vector_xyz, cell_xyz)  # returns dict with 'u', 'v'
        return result

    res =  bkd.vmap(
        _wrap_at_single_lev,
        in_axes=0,
        out_axes=0
    )(un[kds:kde+1, :])
    
    u = bkd.slice_update(u, (slice(kds, kde+1), slice(ids, ide+1)), res["u"])
    v = bkd.slice_update(v, (slice(kds, kde+1), slice(ids, ide+1)), res["v"])
    
    return u, v


def wind_a2c_operator(
    R: FloatType,
    mesh: SCVTMesh,
    u: Float[Array, "nlev ncell"],
    v: Float[Array, "nlev ncell"],
    un: Float[Array, "nlev nedge"]
) -> Float[Array, "nlev nedge"]:
    bkd = mesh.dev.backend
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.edge_ids, mesh.edge_ide
    v1 = mesh.cell_of_edge[ids:ide+1, 0]
    v2 = mesh.cell_of_edge[ids:ide+1, 1]
    u_tmp = 0.5 * (u[kds:kde+1, v1] + u[kds:kde+1, v2])
    v_tmp = 0.5 * (v[kds:kde+1, v1] + v[kds:kde+1, v2])
    vec = vector_convert_lonlat2xyz(bkd, R, u_tmp, v_tmp, mesh.dual_edge_xyz[ids:ide+1, :])
    tmp = bkd.sum(vec * mesh.edge_nr[ids:ide+1, :], axis=-1)
    return bkd.slice_update(un, (slice(kds, kde+1), slice(ids, ide+1)), tmp)
