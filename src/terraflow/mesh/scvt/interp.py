from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
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
    loc_flag = x_loc + ">" + y_loc
    if loc_flag == "cell>edge":
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        v1 = mesh.cell_of_edge[ids:ide+1, 0]
        v2 = mesh.cell_of_edge[ids:ide+1, 1]
        y = y.at[kds:kde+1, ids:ide+1].set(
            0.5 * (x[kds:kde+1, v1] + x[kds:kde+1, v2])
        )
    elif loc_flag == "cell>lev":
        if mesh.full_nlev == 1:
            return y
        kds, kde = mesh.half_kds, mesh.half_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        if lagrange:
            pass
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
            y = y.at[kds+1:kde, ids:ide+1].set(
                a[:, None] * x[kds:kde-1, ids:ide+1] + b[:, None] * x[kds+1:kde, ids:ide+1]
            )
            if extrap:
                k = mesh.half_kds
                x1 = mesh.full_lev[k  ] - mesh.half_lev[k]
                x2 = mesh.full_lev[k+1] - mesh.half_lev[k]
                a =  x2 / (x2 - x1)
                b = -x1 / (x2 - x1)
                y = y.at[k, ids:ide+1].set(a * x[k, ids:ide+1] + b * x[k+1, ids:ide+1])
                k = mesh.half_kde
                x1 = mesh.half_lev[k] - mesh.full_lev[k-1]
                x2 = mesh.half_lev[k] - mesh.full_lev[k-2]
                a =  x2 / (x2 - x1)
                b = -x1 / (x2 - x1)
                y = y.at[k, ids:ide+1].set(a * x[k-1, ids:ide+1] + b * x[k-2, ids:ide+1])
            else:
                k = mesh.half_kds
                y = y.at[k, :].set(x[k, :])
                k = mesh.half_kde
                y = y.at[k, :].set(x[k-1, :])
    elif loc_flag == "cell>vtx":
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.vtx_ids , mesh.vtx_ide
        v1            = mesh.cell_of_vtx  [ids:ide+1, :]
        kite_area_vtx = mesh.kite_area_vtx[ids:ide+1, :]
        area_vtx      = mesh.area_vtx     [ids:ide+1   ] 
        
        def cell2vtx(x: FLT[Array, "npoint1"]) -> FLT[Array, "npoint2"]:
            x_tile = jnp.where(v1 < 0, 0.0, x[v1])
            local_sum = jnp.sum(x_tile * kite_area_vtx, axis=-1)
            return local_sum / area_vtx
        
        y_tmp = jax.vmap(cell2vtx, in_axes=0, out_axes=0)(x[kds:kde+1, :])
        
        y = y.at[kds:kde+1, ids:ide+1].set(y_tmp)
    elif loc_flag == "vtx>edge":
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        v1 = mesh.vtx_of_edge[ids:ide+1, 0]
        v2 = mesh.vtx_of_edge[ids:ide+1, 1]
        y = y.at[kds:kde+1, ids:ide+1].set(
            0.5 * (x[kds:kde+1, v1] + x[kds:kde+1, v2])
        )
    elif loc_flag == "vtx>cell":
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        v1             = mesh.vtx_of_cell   [ids:ide+1, :]
        kite_area_cell = mesh.kite_area_cell[ids:ide+1, :]
        area_cell      = mesh.area_cell     [ids:ide+1   ]
        
        def vtx2cell(x: FLT[Array, "npoint1"]) -> FLT[Array, "npoint2"]:
            x_tile = jnp.where(v1 < 0, 0.0, x[v1])
            local_sum = jnp.sum(x_tile * kite_area_cell, axis=-1)
            return local_sum / area_cell
        
        y_tmp = jax.vmap(vtx2cell, in_axes=0, out_axes=0)(x[kds:kde+1, :])
        
        y = y.at[kds:kde+1, ids:ide+1].set(y_tmp)
    elif loc_flag == "edge>cell": 
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        e_idx     = mesh.cell_ed  [ids:ide+1, :]
        area_cell = mesh.area_cell[ids:ide+1   ]
        le        = mesh.le       [e_idx]
        de        = mesh.de       [e_idx]
        
        def edge2cell(x: FLT[Array, "npoint1"]) -> FLT[Array, "npoint2"]:
            x_tile = jnp.where(e_idx < 0, 0.0, x[e_idx])
            local_sum = jnp.sum(x_tile * le * de / 4.0, axis=-1)
            return local_sum / area_cell
        
        y_tmp = jax.vmap(edge2cell, in_axes=0, out_axes=0)(x[kds:kde+1, :])
        
        y = y.at[kds:kde+1, ids:ide+1].set(y_tmp)
    elif loc_flag == "edge>vtx":
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.vtx_ids , mesh.vtx_ide
        e_idx    = mesh.vtx_ed  [ids:ide+1, :]
        area_vtx = mesh.area_vtx[ids:ide+1   ]
        le       = mesh.le       [e_idx]
        de       = mesh.de       [e_idx]
        
        def edge2vtx(x: FLT[Array, "npoint1"]) -> FLT[Array, "npoint2"]:
            x_tile = jnp.where(e_idx < 0, 0.0, x[e_idx])
            local_sum = jnp.sum(x_tile * le * de / 4.0, axis=-1)
            return local_sum / area_vtx
        
        y_tmp = jax.vmap(edge2vtx, in_axes=0, out_axes=0)(x[kds:kde+1, :])
        
        y = y.at[kds:kde+1, ids:ide+1].set(y_tmp)
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
        y = y.at[kds:kde+1, ids:ide+1].set(
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
        y = y.at[kds+1:kde, ids:ide+1].set(
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
            # y = y.at[k, ids:ide+1].set(
            #     a * x[k, ids:ide+1] + b * x[k+1, ids:ide+1] + c * x[k+2, ids:ide+1]
            # )
            y = y.at[k, ids:ide+1].set(0.5 * x[k, ids:ide+1])
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
            y = y.at[k, ids:ide+1].set(
                a * x[k-1, ids:ide+1] + b * x[k-2, ids:ide+1] + c * x[k-3, ids:ide+1]
            ) 
        elif extrap_type == 2:
            k = mesh.half_kds
            y = y.at[k, ids:ide+1].set(0.5 * x[k, ids:ide+1])
            k = mesh.half_kde
            y = y.at[k, ids:ide+1].set(0.5 * x[k-1, ids:ide+1])
        else:
            raise ValueError(f"scvt_interp_2d: unsupported extrap_type {extrap_type}!")
    elif loc_flag == "lev>lev_edge":
        kds, kde = mesh.half_kds, mesh.half_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        v1 = mesh.cell_of_edge[ids:ide+1, 0]
        v2 = mesh.cell_of_edge[ids:ide+1, 1]
        y = y.at[kds:kde+1, ids:ide+1].set(
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
    loc_flag = x_loc + ">" + y_loc
    if loc_flag == "cell>edge":
       kds, kde = mesh.full_kds, mesh.full_kde
       ids, ide = mesh.edge_ids, mesh.edge_ide
       v1 = mesh.cell_of_edge[ids:ide+1, 0]
       v2 = mesh.cell_of_edge[ids:ide+1, 1]
       y = y.at[kds:kde+1, ids:ide+1].set(
           0.5 * (x[kds:kde+1, v1] + x[kds:kde+1, v2])
       )
    elif loc_flag == "lev>cell":
       kds, kde = mesh.full_kds, mesh.full_kde
       ids, ide = mesh.cell_ids, mesh.cell_ide
       y = y.at[kds:kde+1, ids:ide+1].set(
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
    """Column interpolation for monotonically increasing pressure."""
    nlev = pi_cols.shape[0]
    ncol = pi_cols.shape[1]
    npo = po.shape[0]
    xo_cols = np.empty((npo, ncol), dtype=xi_cols.dtype)

    for col in prange(ncol):
        k = 1
        p_min = pi_cols[0, col]
        p_max = pi_cols[nlev - 1, col]
        for ko in range(npo):
            p_out = po[ko]
            if p_out < p_min or p_out > p_max:
                xo_cols[ko, col] = np.nan
                continue
            while k < nlev - 1 and pi_cols[k, col] < p_out:
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
            xo_cols[ko, col] = (dp2 * x_lo + dp1 * x_hi) / (dp1 + dp2)

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

    hori_shape = pi.shape[1:]
    ncol = 1
    for dim in hori_shape:
        ncol *= dim

    pi_cols = pi_src.reshape(nlev_src, ncol)
    xi_cols = xi_src.reshape(nlev_src, ncol)
    po_array = np.asarray(po)
    if po_array.size > 1 and not np.all(po_array[1:] >= po_array[:-1]):
        po_order = np.argsort(po_array)
        po_inverse = np.empty_like(po_order)
        po_inverse[po_order] = np.arange(po_order.size)
        po_for_interp = po_array[po_order]
    else:
        po_inverse = None
        po_for_interp = po_array

    xo_cols = _interp_plev_columns(
        pi_cols=pi_cols,
        xi_cols=xi_cols,
        po=po_for_interp,
        logp=logp,
    )
    if po_inverse is not None:
        xo_cols = xo_cols[po_inverse, :]
    return xo_cols.reshape((po_array.shape[0],) + hori_shape)
