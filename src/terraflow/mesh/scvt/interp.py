from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numba import njit, prange

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.mesh.scvt.mesh_type import SCVTMesh

    FLT = typing.Float
    Array = typing.Array


def scvt_interp_2d(
    mesh: SCVTMesh,
    x: FLT[Array, "nlev npoint1"],
    y: FLT[Array, "nlev npoint2"],
    x_loc: str,
    y_loc: str,
    extrap: bool = True,
    extrap_type: int = 1,
    lagrange: bool = False,
    lagrange_order: int = 1
) -> FLT[Array, "nlev npoint2"]:
    bkd = mesh.dev.backend
    loc_flag = x_loc + ">" + y_loc
    if loc_flag == "cell>edge":
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        v1 = mesh.cell_of_edge[ids:ide+1, 0]
        v2 = mesh.cell_of_edge[ids:ide+1, 1]
        y = bkd.slice_update(y, (slice(kds, kde+1), slice(ids, ide+1)),
            0.5 * (x[kds:kde+1, v1] + x[kds:kde+1, v2])
        )
    elif loc_flag == "cell>lev":
        if mesh.full_nlev == 1:
            return y
        kds, kde = mesh.half_kds, mesh.half_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        if lagrange:
            nfull = mesh.full_kde - mesh.full_kds + 1

            def interp_lagrange_at_k(y, k, start, npt):
                x0 = mesh.half_lev[k]
                yk = bkd.zeros_like(x[start, ids:ide+1])
                for j in range(start, start + npt):
                    wgt = 1.0
                    for m in range(start, start + npt):
                        if m != j:
                            wgt = wgt * (
                                (x0 - mesh.full_lev[m])
                                / (mesh.full_lev[j] - mesh.full_lev[m])
                            )
                    yk = yk + wgt * x[j, ids:ide+1]
                return bkd.slice_update(y, (k, slice(ids, ide+1)), yk)

            if lagrange_order == 1:
                if nfull < 2:
                    raise ValueError("Lagrange interpolation order 1 requires at least 2 full levels.")
                for k in range(kds + 1, kde):
                    y = interp_lagrange_at_k(y, k, k - 1, 2)
                y = interp_lagrange_at_k(y, kds, mesh.full_kds, 2)
                y = interp_lagrange_at_k(y, kde, mesh.full_kde - 1, 2)
            elif lagrange_order == 3:
                if nfull < 4:
                    raise ValueError("Lagrange interpolation order 3 requires at least 4 full levels.")
                for k in range(kds + 2, kde - 1):
                    y = interp_lagrange_at_k(y, k, k - 2, 4)
                y = interp_lagrange_at_k(y, kds, mesh.full_kds, 4)
                y = interp_lagrange_at_k(y, kds + 1, mesh.full_kds, 4)
                y = interp_lagrange_at_k(y, kde - 1, mesh.full_kde - 3, 4)
                y = interp_lagrange_at_k(y, kde, mesh.full_kde - 3, 4)
            elif lagrange_order == 5:
                if nfull < 6:
                    raise ValueError("Lagrange interpolation order 5 requires at least 6 full levels.")
                for k in range(kds + 3, kde - 2):
                    y = interp_lagrange_at_k(y, k, k - 3, 6)
                y = interp_lagrange_at_k(y, kds, mesh.full_kds, 6)
                y = interp_lagrange_at_k(y, kds + 1, mesh.full_kds, 6)
                y = interp_lagrange_at_k(y, kds + 2, mesh.full_kds, 6)
                y = interp_lagrange_at_k(y, kde - 2, mesh.full_kde - 5, 6)
                y = interp_lagrange_at_k(y, kde - 1, mesh.full_kde - 5, 6)
                y = interp_lagrange_at_k(y, kde, mesh.full_kde - 5, 6)
            else:
                raise ValueError(f"Unsupported Lagrange interpolation order: {lagrange_order}!")
        else:
            # -------
            #
            # ===o=== k-1
            #
            # ---?--- k
            #
            # ===o=== k
            #
            # -------
            a = mesh.full_dlev[kds:kde-1] / (2 * mesh.half_dlev[kds+1:kde])
            b = mesh.full_dlev[kds+1:kde] / (2 * mesh.half_dlev[kds+1:kde])
            y = bkd.slice_update(y, (slice(kds+1, kde), slice(ids, ide+1)),
                a[:, None] * x[kds:kde-1, ids:ide+1] + b[:, None] * x[kds+1:kde, ids:ide+1]
            )
        if extrap:
            k = mesh.half_kds
            x1 = mesh.full_lev[k  ] - mesh.half_lev[k]
            x2 = mesh.full_lev[k+1] - mesh.half_lev[k]
            a =  x2 / (x2 - x1)
            b = -x1 / (x2 - x1)
            y = bkd.slice_update(y, (k, slice(ids, ide+1)), a * x[k, ids:ide+1] + b * x[k+1, ids:ide+1])
            k = mesh.half_kde
            x1 = mesh.half_lev[k] - mesh.full_lev[k-1]
            x2 = mesh.half_lev[k] - mesh.full_lev[k-2]
            a =  x2 / (x2 - x1)
            b = -x1 / (x2 - x1)
            y = bkd.slice_update(y, (k, slice(ids, ide+1)), a * x[k-1, ids:ide+1] + b * x[k-2, ids:ide+1])
        else:
            k = mesh.half_kds
            y = bkd.slice_update(y, (k, slice(None)), x[k, :])
            k = mesh.half_kde
            y = bkd.slice_update(y, (k, slice(None)), x[k-1, :])
    elif loc_flag == "cell>vtx":
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.vtx_ids , mesh.vtx_ide
        v1            = mesh.cell_of_vtx  [ids:ide+1, :]
        kite_area_vtx = mesh.kite_area_vtx[ids:ide+1, :]
        area_vtx      = mesh.area_vtx     [ids:ide+1   ]
        valid_v1 = v1 >= 0
        safe_v1 = bkd.where(valid_v1, v1, 0)

        def cell2vtx(x: FLT[Array, "npoint1"]) -> FLT[Array, "npoint2"]:
            x_tile = bkd.take(x, safe_v1, axis=0)
            contrib = x_tile * kite_area_vtx
            contrib = bkd.where(valid_v1, contrib, 0.0)
            local_sum = bkd.sum(contrib, axis=-1)
            return local_sum / area_vtx

        y_tmp = bkd.vmap(cell2vtx, in_axes=0, out_axes=0)(x[kds:kde+1, :])

        y = bkd.slice_update(y, (slice(kds, kde+1), slice(ids, ide+1)), y_tmp)
    elif loc_flag == "vtx>edge":
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        v1 = mesh.vtx_of_edge[ids:ide+1, 0]
        v2 = mesh.vtx_of_edge[ids:ide+1, 1]
        y = bkd.slice_update(y, (slice(kds, kde+1), slice(ids, ide+1)),
            0.5 * (x[kds:kde+1, v1] + x[kds:kde+1, v2])
        )
    elif loc_flag == "vtx>cell":
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        v1             = mesh.vtx_of_cell   [ids:ide+1, :]
        kite_area_cell = mesh.kite_area_cell[ids:ide+1, :]
        area_cell      = mesh.area_cell     [ids:ide+1   ]
        valid_v1 = v1 >= 0
        safe_v1 = bkd.where(valid_v1, v1, 0)

        def vtx2cell(x: FLT[Array, "npoint1"]) -> FLT[Array, "npoint2"]:
            x_tile = bkd.take(x, safe_v1, axis=0)
            contrib = x_tile * kite_area_cell
            contrib = bkd.where(valid_v1, contrib, 0.0)
            local_sum = bkd.sum(contrib, axis=-1)
            return local_sum / area_cell

        y_tmp = bkd.vmap(vtx2cell, in_axes=0, out_axes=0)(x[kds:kde+1, :])

        y = bkd.slice_update(y, (slice(kds, kde+1), slice(ids, ide+1)), y_tmp)
    elif loc_flag == "edge>cell":
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        e_idx     = mesh.cell_ed  [ids:ide+1, :]
        valid_e_idx = e_idx >= 0
        safe_e_idx = bkd.where(valid_e_idx, e_idx, 0)
        area_cell = mesh.area_cell[ids:ide+1   ]
        le        = bkd.take(mesh.le, safe_e_idx, axis=0)
        de        = bkd.take(mesh.de, safe_e_idx, axis=0)

        def edge2cell(x: FLT[Array, "npoint1"]) -> FLT[Array, "npoint2"]:
            x_tile = bkd.take(x, safe_e_idx, axis=0)
            contrib = x_tile * le * de / 4.0
            contrib = bkd.where(valid_e_idx, contrib, 0.0)
            local_sum = bkd.sum(contrib, axis=-1)
            return local_sum / area_cell

        y_tmp = bkd.vmap(edge2cell, in_axes=0, out_axes=0)(x[kds:kde+1, :])

        y = bkd.slice_update(y, (slice(kds, kde+1), slice(ids, ide+1)), y_tmp)
    elif loc_flag == "edge>vtx":
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.vtx_ids , mesh.vtx_ide
        e_idx    = mesh.vtx_ed  [ids:ide+1, :]
        valid_e_idx = e_idx >= 0
        safe_e_idx = bkd.where(valid_e_idx, e_idx, 0)
        area_vtx = mesh.area_vtx[ids:ide+1   ]
        le       = bkd.take(mesh.le, safe_e_idx, axis=0)
        de       = bkd.take(mesh.de, safe_e_idx, axis=0)

        def edge2vtx(x: FLT[Array, "npoint1"]) -> FLT[Array, "npoint2"]:
            x_tile = bkd.take(x, safe_e_idx, axis=0)
            contrib = x_tile * le * de / 4.0
            contrib = bkd.where(valid_e_idx, contrib, 0.0)
            local_sum = bkd.sum(contrib, axis=-1)
            return local_sum / area_vtx

        y_tmp = bkd.vmap(edge2vtx, in_axes=0, out_axes=0)(x[kds:kde+1, :])

        y = bkd.slice_update(y, (slice(kds, kde+1), slice(ids, ide+1)), y_tmp)
    elif loc_flag == "lev>cell":
        # =======
        #
        # ---o--- k
        #
        # ===?=== k
        #
        # ---o--- k+1
        #
        # =======
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        y = bkd.slice_update(y, (slice(kds, kde+1), slice(ids, ide+1)),
            0.5 * (x[kds:kde+1, ids:ide+1] + x[kds+1:kde+2, ids:ide+1])
        )
    elif loc_flag == "edge>lev_edge":
        # -------
        #
        # ===o=== k-1
        #
        # ---?--- k
        #
        # ===o=== k
        #
        # ---o---
        kds, kde = mesh.half_kds, mesh.half_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        a = mesh.full_dlev[kds:kde-1] / (2 * mesh.half_dlev[kds+1:kde])
        b = mesh.full_dlev[kds+1:kde] / (2 * mesh.half_dlev[kds+1:kde])
        y = bkd.slice_update(y, (slice(kds+1, kde), slice(ids, ide+1)),
            a[:, None] * x[kds+1:kde, ids:ide+1] + b[:, None] * x[kds:kde-1, ids:ide+1]
        )
        if extrap_type == 1:
            k = mesh.half_kds
            # ---?--- 1
            #
            # ===o=== 1   x1
            #
            # -------
            #
            # ===o=== 2   x2
            #
            # -------
            #
            # ===o=== 3   x3
            # x1 = mesh.full_lev[k  ] - mesh.half_lev[k]
            # x2 = mesh.full_lev[k+1] - mesh.half_lev[k]
            # x3 = mesh.full_lev[k+2] - mesh.half_lev[k]
            # a =  x2 * x3 / (x1**2 - x1 * x2 - x1 * x3 + x2 * x3)
            # b =  x1 * x3 / (x2**2 - x2 * x1 - x2 * x3 + x1 * x3)
            # c =  x1 * x2 / (x3**2 - x3 * x1 - x3 * x2 + x1 * x2)
            # y = bkd.slice_update(
            #     y,
            #     (k, slice(ids, ide+1)),
            #     a * x[k, ids:ide+1] + b * x[k+1, ids:ide+1] + c * x[k+2, ids:ide+1]
            # )
            y = bkd.slice_update(y, (k, slice(ids, ide+1)), 0.5 * x[k, ids:ide+1])
            k = mesh.half_kde
            # ===o=== NLEV - 2  x3
            #
            # -------
            #
            # ===o=== NLEV - 1  x2
            #
            # -------
            #
            # ===o=== NLEV      x1
            #
            # ---?--- NLEV + 1
            x1 = mesh.half_lev[k] - mesh.full_lev[k-1]
            x2 = mesh.half_lev[k] - mesh.full_lev[k-2]
            x3 = mesh.half_lev[k] - mesh.full_lev[k-3]
            a =  x2 * x3 / (x1**2 - x1 * x2 - x1 * x3 + x2 * x3)
            b =  x1 * x3 / (x2**2 - x2 * x1 - x2 * x3 + x1 * x3)
            c =  x1 * x2 / (x3**2 - x3 * x1 - x3 * x2 + x1 * x2)
            y = bkd.slice_update(y, (k, slice(ids, ide+1)),
                a * x[k-1, ids:ide+1] + b * x[k-2, ids:ide+1] + c * x[k-3, ids:ide+1]
            )
        elif extrap_type == 2:
            k = mesh.half_kds
            y = bkd.slice_update(y, (k, slice(ids, ide+1)), 0.5 * x[k, ids:ide+1])
            k = mesh.half_kde
            y = bkd.slice_update(y, (k, slice(ids, ide+1)), 0.5 * x[k-1, ids:ide+1])
        else:
            raise ValueError(f"scvt_interp_2d: unsupported extrap_type {extrap_type}!")
    elif loc_flag == "lev>lev_edge":
        kds, kde = mesh.half_kds, mesh.half_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        v1 = mesh.cell_of_edge[ids:ide+1, 0]
        v2 = mesh.cell_of_edge[ids:ide+1, 1]
        y = bkd.slice_update(y, (slice(kds, kde+1), slice(ids, ide+1)),
            0.5 * (x[kds:kde+1, v1] + x[kds:kde+1, v2])
        )
    else:
        raise ValueError(f"scvt_interp_2d: unsupported location flag: {loc_flag}!")
    return y

def scvt_average_2d(
    mesh: SCVTMesh,
    x: FLT[Array, "nlev npoint1"],
    y: FLT[Array, "nlev npoint2"],
    x_loc: str,
    y_loc: str
) -> FLT[Array, "nlev npoint2"]:
    bkd = mesh.dev.backend
    loc_flag = x_loc + ">" + y_loc
    if loc_flag == "cell>edge":
       kds, kde = mesh.full_kds, mesh.full_kde
       ids, ide = mesh.edge_ids, mesh.edge_ide
       v1 = mesh.cell_of_edge[ids:ide+1, 0]
       v2 = mesh.cell_of_edge[ids:ide+1, 1]
       y = bkd.slice_update(y, (slice(kds, kde+1), slice(ids, ide+1)),
           0.5 * (x[kds:kde+1, v1] + x[kds:kde+1, v2])
       )
    elif loc_flag == "lev>cell":
       kds, kde = mesh.full_kds, mesh.full_kde
       ids, ide = mesh.cell_ids, mesh.cell_ide
       y = bkd.slice_update(y, (slice(kds, kde+1), slice(ids, ide+1)),
           0.5 * (x[kds:kde+1, ids:ide+1] + x[kds+1:kde+2, ids:ide+1])
       )
    else:
        raise ValueError(f"scvt_average_2d: unsupported location flag {loc_flag}")
    return y


@njit(cache=True, parallel=True)
def _interp_plev_columns(
    pi_cols: np.ndarray,
    xi_cols: np.ndarray,
    po: np.ndarray,
    logp: bool,
) -> np.ndarray:
    """Column interpolation for monotonic pressure columns."""
    nlev = pi_cols.shape[0]
    ncol = pi_cols.shape[1]
    npo = po.shape[0]
    xo_cols = np.empty((npo, ncol), dtype=xi_cols.dtype)

    for col in prange(ncol):
        p_beg = pi_cols[0, col]
        p_end = pi_cols[nlev - 1, col]
        increasing = p_end >= p_beg
        for ko in range(npo):
            p_out = po[ko]
            k = 1
            if increasing:
                if p_out < p_beg or p_out > p_end:
                    xo_cols[ko, col] = np.nan
                    continue
                while k < nlev - 1 and pi_cols[k, col] < p_out:
                    k += 1
            else:
                if p_out > p_beg or p_out < p_end:
                    xo_cols[ko, col] = np.nan
                    continue
                while k < nlev - 1 and pi_cols[k, col] > p_out:
                    k += 1

            p_lo = pi_cols[k - 1, col]
            p_hi = pi_cols[k, col]
            x_lo = xi_cols[k - 1, col]
            x_hi = xi_cols[k, col]
            if logp:
                dp1 = np.log(p_out) - np.log(p_lo)
                dp2 = np.log(p_hi) - np.log(p_out)
            else:
                dp1 = p_out - p_lo
                dp2 = p_hi - p_out
            denom = dp1 + dp2
            if denom == 0.0:
                xo_cols[ko, col] = 0.5 * (x_lo + x_hi)
            else:
                xo_cols[ko, col] = (dp2 * x_lo + dp1 * x_hi) / denom

    return xo_cols


def interp_plev(
    mesh: SCVTMesh,
    pi: np.ndarray, # FLT[Array, "nlev_i nlat nlon"]
    xi: np.ndarray, # FLT[Array, "nlev_i nlat nlon"]
    po: np.ndarray, # FLT[Array, "nlev_o"]
    xo_loc: str,
    logp: bool = False
) -> np.ndarray: # FLT[Array, "nlev_o nlat nlon"]
    if xo_loc == "cell" or xo_loc == "vtx" or xo_loc == "edge":
        k_s, k_e = mesh.full_kds, mesh.full_kde
    elif xo_loc == "lev" or xo_loc == "lev_vtx" or xo_loc == "lev_edge":
        k_s, k_e = mesh.half_kds, mesh.half_kde
    else:
        raise ValueError(f"interp_plev: unsupported location {xo_loc} for xo_loc!")
    pi_src = np.asarray(pi[k_s:k_e+1, ...])
    xi_src = np.asarray(xi[k_s:k_e+1, ...])
    nlev_src = pi_src.shape[0]
    if nlev_src < 2:
        raise ValueError("interp_plev requires at least two source levels in the active vertical domain.")
    if xi_src.shape != pi_src.shape:
        raise ValueError(
            f"interp_plev: pi and xi must have the same active-domain shape, "
            f"got {pi_src.shape} and {xi_src.shape}."
        )

    hori_shape = pi_src.shape[1:]
    ncol = 1
    for dim in hori_shape:
        ncol *= dim

    pi_cols = pi_src.reshape(nlev_src, ncol)
    xi_cols = xi_src.reshape(nlev_src, ncol)
    po_array = np.asarray(po)
    pi_diff = np.diff(pi_cols, axis=0)
    if pi_diff.size > 0:
        scale = max(float(np.nanmax(np.abs(pi_cols))), 1.0)
        if np.issubdtype(pi_cols.dtype, np.floating):
            tol = np.finfo(pi_cols.dtype).eps * scale * 64
        else:
            tol = 0.0
        monotonic = (
            np.all(pi_diff >= -tol, axis=0)
            | np.all(pi_diff <= tol, axis=0)
        )
        if not np.all(monotonic):
            raise ValueError(
                "interp_plev: source pressure must be monotonic in every column."
            )

    xo_cols = _interp_plev_columns(
        pi_cols=pi_cols,
        xi_cols=xi_cols,
        po=po_array,
        logp=logp,
    )
    return xo_cols.reshape((po_array.shape[0],) + hori_shape)
