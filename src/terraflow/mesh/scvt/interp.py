from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
   
    Float = typing.Float
    Array = typing.Array
    
    
def scvt_interp_2d(
    mesh: SCVTMesh,
    x: Float[Array, "nlev npoint1"], 
    y: Float[Array, "nlev npoint2"],
    x_loc: str,
    y_loc: str,
    extrap: bool = True,
    extrap_type: int = 1,
    lagrange: bool = False,
    lagrange_order: int = 1
) -> Float[Array, "nlev npoint2"]: 
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
        
        def cell2vtx(x: Float[Array, "npoint1"]) -> Float[Array, "npoint2"]:
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
        
        def vtx2cell(x: Float[Array, "npoint1"]) -> Float[Array, "npoint2"]:
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
        
        def edge2cell(x: Float[Array, "npoint1"]) -> Float[Array, "npoint2"]:
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
        
        def edge2vtx(x: Float[Array, "npoint1"]) -> Float[Array, "npoint2"]:
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
    x: Float[Array, "nlev npoint1"], 
    y: Float[Array, "nlev npoint2"],
    x_loc: str,
    y_loc: str
) -> Float[Array, "nlev npoint2"]:
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


@partial(jax.jit, static_argnames=("logp",))
def _interp_plev_columns(
    pi_cols: jnp.ndarray,
    xi_cols: jnp.ndarray,
    po: jnp.ndarray,
    logp: bool,
) -> jnp.ndarray:
    """Vectorized column interpolation following the F90 adjacent-pair scan exactly."""
    pi_lo_all = pi_cols[:-1, :]
    pi_hi_all = pi_cols[1:, :]
    xi_lo_all = xi_cols[:-1, :]
    xi_hi_all = xi_cols[1:, :]
    po_cols = po[:, None]

    # Match the F90 condition:
    # if (pi(k-1) <= po(ko) .and. po(ko) <= pi(k)) then
    match = (
        (pi_lo_all[None, :, :] <= po_cols[:, None, :])
        & (po_cols[:, None, :] <= pi_hi_all[None, :, :])
    )
    has_match = jnp.any(match, axis=1)
    pair_idx = jnp.argmax(match, axis=1)

    pi_lo = jnp.take_along_axis(pi_lo_all, pair_idx, axis=0)
    pi_hi = jnp.take_along_axis(pi_hi_all, pair_idx, axis=0)
    xi_lo = jnp.take_along_axis(xi_lo_all, pair_idx, axis=0)
    xi_hi = jnp.take_along_axis(xi_hi_all, pair_idx, axis=0)

    if logp:
        dp1 = jnp.log(po_cols) - jnp.log(pi_lo)
        dp2 = jnp.log(pi_hi) - jnp.log(po_cols)
    else:
        dp1 = po_cols - pi_lo
        dp2 = pi_hi - po_cols

    xo_cols = (dp2 * xi_lo + dp1 * xi_hi) / (dp1 + dp2)
    return jnp.where(has_match, xo_cols, jnp.nan)


def interp_plev(
    mesh: SCVTMesh,
    pi: Float[Array, "nlev_i nlat nlon"],
    xi: Float[Array, "nlev_i nlat nlon"],
    po: Float[Array, "nlev_o"],
    xo_loc: str,
    logp: bool = False
) -> Float[Array, "nlev_o nlat nlon"]:
    if xo_loc == "cell" or xo_loc == "vtx" or xo_loc == "edge":
        k_s, k_e = mesh.full_kds, mesh.full_kde
    elif xo_loc == "lev" or xo_loc == "lev_vtx" or xo_loc == "lev_edge":
        k_s, k_e = mesh.half_kds, mesh.half_kde
    else:
        raise ValueError(f"interp_plev: unsupported location {xo_loc} for xo_loc!")
    pi_src = pi[k_s:k_e+1, ...]
    xi_src = xi[k_s:k_e+1, ...]
    nlev_src = pi_src.shape[0]
    if nlev_src < 2:
        raise ValueError("interp_plev requires at least two source levels in the active vertical domain.")

    hori_shape = pi.shape[1:]
    ncol = 1
    for dim in hori_shape:
        ncol *= dim

    pi_cols = pi_src.reshape(nlev_src, ncol)
    xi_cols = xi_src.reshape(nlev_src, ncol)
    xo_cols = _interp_plev_columns(
        pi_cols=pi_cols,
        xi_cols=xi_cols,
        po=po,
        logp=logp,
    )
    return xo_cols.reshape((po.shape[0],) + hori_shape)
