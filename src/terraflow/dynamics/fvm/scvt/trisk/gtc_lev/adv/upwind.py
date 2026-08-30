from __future__ import annotations
from typing import (
    TYPE_CHECKING, 
    Self,
    Callable, 
    Tuple
)
from abc import ABC, abstractmethod
from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import typing, math, jax_tools, device
from terraflow.util.sphere_geometry import _project_sphere2plane, project_sphere2plane
from terraflow.mesh.scvt.parallel import fill_field2d_array_halo, fill_raw_array_halo
from terraflow.dynamics.abstract import (
    AbstractEquationAuxiliaryData, 
    AbstractEquationAuxiliaryDataset
)
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv import batch

INT       = typing.Int
FLT       = typing.Float
IntType   = typing.IntType
FloatType = typing.FloatType
Array     = typing.Array


if TYPE_CHECKING:
    from terraflow.model.config import UserConfig
    from terraflow.planet import PlanetParameterSet
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block


def upwind_calc_mass_hflx(
    batch: batch.AdvBatchType, 
    block: Block,
    m: FLT[Array, "nlev npoint"], 
    mfn: FLT[Array, "nlev nedge"], 
    dt: float
) -> FLT[Array, "nlev nedge"]:
    if batch.loc == "cell":
        stencil_2nd = block.equaux.adv.upwind.stencil_cell_2nd
        stencil_4th = block.equaux.adv.upwind.stencil_cell_4th
        mesh = block.mesh
        para = block.para
        radius = para.radius.unwrap()
        upwind_order     = block.cfg.upwind_order
        upwind_wgt       = block.cfg.upwind_wgt
        upwind_optimized = block.cfg.upwind_optimized
        un = batch.un.unwrap()
        kds = mesh.full_kds if batch.loc == "cell" else mesh.half_kds
        kde = mesh.full_kde if batch.loc == "cell" else mesh.half_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        if upwind_order == 0:
            v1 = mesh.cell_of_edge[ids:ide+1, 0]
            v2 = mesh.cell_of_edge[ids:ide+1, 1]
            wind = un[kds:kde+1, ids:ide+1]
            tmp  = 0.5 * (m[kds:kde+1, v1] + m[kds:kde+1, v2]) * wind
            mfn = mfn.at[kds:kde+1, ids:ide+1].set(tmp)
        elif upwind_order == 1:
            wind = un[kds:kde+1, ids:ide+1] * mesh.edpnr_eddtg[ids:ide+1]
            tmp  = upwind1_hori(
                q=m, 
                wind=wind, 
                edd_v=mesh.cell_of_edge[ids:ide+1, :],
                wgt=upwind_wgt,
            ) * un[kds:kde+1, ids:ide+1]
            mfn = mfn.at[kds:kde+1, ids:ide+1].set(tmp)
        elif upwind_order == 3:
            wind = un[kds:kde+1, ids:ide+1] * mesh.edpnr_eddtg[ids:ide+1]
            de = mesh.de[ids:ide+1] / radius
            tmp  = upwind3_hori(
                q=m, 
                wind=wind, 
                edd_v=mesh.cell_of_edge[ids:ide+1, :], 
                edd_leng=de,
                stencil_2nd_idx=stencil_2nd.idxes,
                stencil_2nd_wgt=stencil_2nd.wgt,
                wgt=upwind_wgt,
                upwind_optimized=upwind_optimized
            ) * un[kds:kde+1, ids:ide+1]
            mfn = mfn.at[kds:kde+1, ids:ide+1].set(tmp)
        elif upwind_order == 5:
            wind = un[kds:kde+1, ids:ide+1] * mesh.edpnr_eddtg[ids:ide+1]
            de = mesh.de[ids:ide+1] / radius
            tmp  = upwind5_hori(
                q=m, 
                wind=wind, 
                edd_v=mesh.cell_of_edge[ids:ide+1, :], 
                edd_leng=de,
                stencil_2nd_idx=stencil_2nd.idxes,
                stencil_2nd_wgt=stencil_2nd.wgt,
                stencil_4th_idx=stencil_4th.idxes,
                stencil_4th_wgt=stencil_4th.wgt,
                wgt=upwind_wgt,
            ) * un[kds:kde+1, ids:ide+1]
            mfn = mfn.at[kds:kde+1, ids:ide+1].set(tmp)
        else:
            raise ValueError(f"upwind_calc_tracer_hflx: unsupported upwind order {upwind_order}!")
        return fill_field2d_array_halo(
            halo=mesh.halo,
            f=mfn,
            meta=batch.mfn.meta
        )
    else:
        raise ValueError(f"upwind_calc_tracer_hflx: unsupported batch location {batch.loc}!")


def upwind_calc_tracer_hflx(
    batch: batch.AdvBatchType, 
    block: Block,
    q: FLT[Array, "nlev npoint"], 
    qmfn: FLT[Array, "nlev nedge"], 
    dt: float
) -> FLT[Array, "nlev nedge"]:
    if batch.loc == "cell" or batch.loc == "lev":
        stencil_2nd = block.equaux.adv.upwind.stencil_cell_2nd
        stencil_4th = block.equaux.adv.upwind.stencil_cell_4th
        mesh = block.mesh
        para = block.para
        radius = para.radius.unwrap()
        upwind_order     = block.cfg.upwind_order
        upwind_wgt       = block.cfg.upwind_wgt
        upwind_optimized = block.cfg.upwind_optimized
        mfn = batch.mfn.unwrap()
        kds = mesh.full_kds if batch.loc == "cell" else mesh.half_kds
        kde = mesh.full_kde if batch.loc == "cell" else mesh.half_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        if upwind_order == 0:
            v1 = mesh.cell_of_edge[ids:ide+1, 0]
            v2 = mesh.cell_of_edge[ids:ide+1, 1]
            wind = mfn[kds:kde+1, ids:ide+1]
            tmp  = 0.5 * (q[kds:kde+1, v1] + q[kds:kde+1, v2]) * wind
            qmfn = qmfn.at[kds:kde+1, ids:ide+1].set(tmp)
        elif upwind_order == 1:
            wind = mfn[kds:kde+1, ids:ide+1] * mesh.edpnr_eddtg[ids:ide+1]
            tmp  = upwind1_hori(
                q=q, 
                wind=wind, 
                edd_v=mesh.cell_of_edge[ids:ide+1, :],
                wgt=upwind_wgt, 
            ) * mfn[kds:kde+1, ids:ide+1]
            qmfn = qmfn.at[kds:kde+1, ids:ide+1].set(tmp)
        elif upwind_order == 3:
            wind = mfn[kds:kde+1, ids:ide+1] * mesh.edpnr_eddtg[ids:ide+1]
            de = mesh.de[ids:ide+1] / radius
            tmp  = upwind3_hori(
                q=q, 
                wind=wind, 
                edd_v=mesh.cell_of_edge[ids:ide+1, :], 
                edd_leng=de,
                stencil_2nd_idx=stencil_2nd.idxes,
                stencil_2nd_wgt=stencil_2nd.wgt,
                wgt=upwind_wgt,
                upwind_optimized=upwind_optimized
            ) * mfn[kds:kde+1, ids:ide+1]
            qmfn = qmfn.at[kds:kde+1, ids:ide+1].set(tmp)
        elif upwind_order == 5:
            wind = mfn[kds:kde+1, ids:ide+1] * mesh.edpnr_eddtg[ids:ide+1]
            de = mesh.de[ids:ide+1] / radius
            tmp  = upwind5_hori(
                q=q, 
                wind=wind, 
                edd_v=mesh.cell_of_edge[ids:ide+1, :], 
                edd_leng=de,
                stencil_2nd_idx=stencil_2nd.idxes,
                stencil_2nd_wgt=stencil_2nd.wgt,
                stencil_4th_idx=stencil_4th.idxes,
                stencil_4th_wgt=stencil_4th.wgt,
                wgt=upwind_wgt,
            ) * mfn[kds:kde+1, ids:ide+1]
            qmfn = qmfn.at[kds:kde+1, ids:ide+1].set(tmp)
        else:
            raise ValueError(f"upwind_calc_tracer_hflx: unsupported upwind order {upwind_order}!")
        return fill_field2d_array_halo(
            halo=mesh.halo,
            f=qmfn,
            meta=batch.qmfn.meta
        )
    else:
        raise ValueError(f"upwind_calc_tracer_hflx: unsupported batch location {batch.loc}!")


def upwind_calc_tracer_vflx(
    batch: batch.AdvBatchType,
    block: Block, 
    q: FLT[Array, "nlev npoint"], 
    qmfz: FLT[Array, "nlev nedge"], 
    dt: float
) -> FLT[Array, "nlev nedge"]:
    if batch.loc == "cell":
        upwind_order = block.cfg.upwind_order
        upwind_wgt   = block.cfg.upwind_wgt        
        mesh     = block.mesh
        mfz      = batch.mfz.unwrap()
        kds, kde = mesh.half_kds, mesh.half_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        if upwind_order == 0:
            # half_kds + 1 <= k <= half_kde - 1
            res = 0.5 * (
                q[kds:kde-1, ids:ide+1]
                + q[kds+1:kde, ids:ide+1]
            ) * mfz[kds+1:kde, ids:ide+1]
            qmfz = qmfz.at[kds+1:kde, ids:ide+1].set(res) 
        elif upwind_order == 1:
            # half_kds + 1 <= k <= half_kde - 1
            mid_mfz = mfz[kds+1:kde, ids:ide+1]
            direction = jnp.sign(mid_mfz)
            q_0 = q[kds:kde-1, ids:ide+1]
            q_plus_1 = q[kds+1:kde, ids:ide+1]
            res = upwind1_vert(
                direction=direction,
                wgt=upwind_wgt,
                q_0=q_0,
                q_plus_1=q_plus_1
            ) * mid_mfz
            qmfz = qmfz.at[kds+1:kde, ids:ide+1].set(res)
        elif upwind_order == 3:
            # =================================================
            # k = half_kds
            top0 = jnp.zeros_like(q[kds, ids:ide+1])
            # =================================================
            # k = half_kds + 1
            top1 = 0.5 * (
                q[kds, ids:ide+1] + q[kds+1, ids:ide+1]
            ) * mfz[kds+1, ids:ide+1]
            # =================================================
            # half_kds + 2 <= k <= half_kde - 2
            q_minus_1 = q[kds:kde-3, ids:ide+1]
            q_0 = q[kds+1:kde-2, ids:ide+1]
            q_plus_1 = q[kds+2:kde-1, ids:ide+1]
            q_plus_2 = q[kds+3:kde, ids:ide+1]
            mid_mfz   = mfz[kds+2:kde-1, ids:ide+1]
            direction = jnp.sign(mid_mfz)
            res = upwind3_vert(
                direction=direction,
                wgt=upwind_wgt,
                q_minus_1=q_minus_1,
                q_0=q_0,
                q_plus_1=q_plus_1,
                q_plus_2=q_plus_2
            ) * mid_mfz
            # =================================================
            # k = half_kde - 1
            bottom1 = 0.5 * (
                q[kde-2, ids:ide+1] + q[kde-1, ids:ide+1]
            ) * mfz[kde-1, ids:ide+1]
            # =================================================
            # k = half_kde
            bottom0 = jnp.zeros_like(q[kde-1, ids:ide+1])
            # =================================================
            qmfz = qmfz.at[kds        , ids:ide+1].set(top0   )
            qmfz = qmfz.at[kds+1      , ids:ide+1].set(top1   )
            qmfz = qmfz.at[kds+2:kde-1, ids:ide+1].set(res    )
            qmfz = qmfz.at[kde-1      , ids:ide+1].set(bottom1)
            qmfz = qmfz.at[kde        , ids:ide+1].set(bottom0)
        elif upwind_order == 5:
            # =================================================
            # k = half_kds
            top0 = jnp.zeros_like(q[kds, ids:ide+1])
            # =================================================
            # half_kds + 1 <= k <= half_kds + 2
            top1 = 0.5 * (
                q[kds:kds+2, ids:ide+1] + q[kds+1:kds+3, ids:ide+1]
            ) * mfz[kds+1:kds+3, ids:ide+1]
            # =================================================
            # half_kds + 3 <= k <= half_kde - 3
            q_minus_2 = q[kds:kde-5, ids:ide+1]
            q_minus_1 = q[kds+1:kde-4, ids:ide+1]
            q_0 = q[kds+2:kde-3, ids:ide+1]
            q_plus_1 = q[kds+3:kde-2, ids:ide+1]
            q_plus_2 = q[kds+4:kde-1, ids:ide+1]
            q_plus_3 = q[kds+5:kde, ids:ide+1]
            mid_mfz = mfz[kds+3:kde-2, ids:ide+1]
            direction = jnp.sign(mid_mfz)
            res = upwind5_vert(
                direction=direction,
                wgt=upwind_wgt,
                q_minus_2=q_minus_2,
                q_minus_1=q_minus_1,
                q_0=q_0,
                q_plus_1=q_plus_1,
                q_plus_2=q_plus_2,
                q_plus_3=q_plus_3
            ) * mid_mfz
            # =================================================
            # half_kde - 2 <= k <= half_kde - 1
            bottom1 = 0.5 * (
                q[kde-3:kde-1, ids:ide+1] + q[kde-2:kde, ids:ide+1]
            ) * mfz[kde-2:kde, ids:ide+1]
            # =================================================
            # k = half_kde
            bottom0 = jnp.zeros_like(q[kde-1, ids:ide+1])
            # =================================================
            qmfz = qmfz.at[kds        , ids:ide+1].set(top0   )
            qmfz = qmfz.at[kds+1:kds+3, ids:ide+1].set(top1   )
            qmfz = qmfz.at[kds+3:kde-2, ids:ide+1].set(res    )
            qmfz = qmfz.at[kde-2:kde  , ids:ide+1].set(bottom1)
            qmfz = qmfz.at[kde        , ids:ide+1].set(bottom0)
        else:
            raise ValueError(f"upwind_calc_tracer_vflx: unsupported upwind order {upwind_order}!")
        return fill_field2d_array_halo(
            halo=mesh.halo,
            f=qmfz,
            meta=batch.qmfz.meta
        )
    elif batch.loc == "lev":
        upwind_order = block.cfg.upwind_order
        upwind_wgt   = block.cfg.upwind_wgt        
        mesh         = block.mesh
        mfz          = batch.mfz.unwrap()
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        if upwind_order == 0:
            # full_kds <= k <= full_kde
            res = 0.5 * (
                q[kds:kde+1, ids:ide+1] + q[kds+1:kde+2, ids:ide+1]
            ) * mfz[kds:kde+1, ids:ide+1]
            qmfz = qmfz.at[kds:kde+1, ids:ide+1].set(res)
        elif upwind_order == 1:
            # full_kds <= k <= full_kde
            q_0 = q[kds:kde+1, ids:ide+1]
            q_plus_1 = q[kds+1:kde+2, ids:ide+1]
            mfz_tmp = mfz[kds:kde+1, ids:ide+1]
            direction = jnp.sign(mfz_tmp)
            res = upwind1_vert(
                direction=direction,
                wgt=upwind_wgt,
                q_0=q_0,
                q_plus_1=q_plus_1
            ) * mfz_tmp
            qmfz = qmfz.at[kds:kde+1, ids:ide+1].set(res)
        elif upwind_order == 3:
            # =================================================
            # k = full_kds
            top = 0.5 * (
                q[kds, ids:ide+1] + q[kds+1, ids:ide+1]
            ) * mfz[kds, ids:ide+1]
            # =================================================
            # full_kds + 1 <= k <= full_kde - 1
            q_minus_1 = q[kds:kde-1, ids:ide+1]
            q_0 = q[kds+1:kde, ids:ide+1]
            q_plus_1 = q[kds+2:kde+1, ids:ide+1]
            q_plus_2 = q[kds+3:kde+2, ids:ide+1]
            mid_mfz = mfz[kds+1:kde, ids:ide+1]
            direction = jnp.sign(mid_mfz)
            mid = upwind3_vert(
                direction=direction,
                wgt=upwind_wgt,
                q_minus_1=q_minus_1,
                q_0=q_0,
                q_plus_1=q_plus_1,
                q_plus_2=q_plus_2
            ) * mid_mfz
            # =================================================
            # k = full_kde
            bottom = 0.5 * (
                q[kde, ids:ide+1] + q[kde+1, ids:ide+1]
            ) * mfz[kde, ids:ide+1]
            # =================================================
            qmfz = qmfz.at[kds      , ids:ide+1].set(top)
            qmfz = qmfz.at[kds+1:kde, ids:ide+1].set(mid)
            qmfz = qmfz.at[kde      , ids:ide+1].set(bottom)
        elif upwind_order == 5:
            # =================================================
            # full_kds <= k <= full_kds + 1
            top = 0.5 * (
                q[kds:kds+2, ids:ide+1] + q[kds+1:kds+3, ids:ide+1]
            ) * mfz[kds:kds+2, ids:ide+1]
            # =================================================
            # full_kds + 2 <= k <= full_kde - 2
            q_minus_2 = q[kds:kde-3, ids:ide+1]
            q_minus_1 = q[kds+1:kde-2, ids:ide+1]
            q_0 = q[kds+2:kde-1, ids:ide+1]
            q_plus_1 = q[kds+3:kde, ids:ide+1]
            q_plus_2 = q[kds+4:kde+1, ids:ide+1]
            q_plus_3 = q[kds+5:kde+2, ids:ide+1]
            mid_mfz = mfz[kds+2:kde-1, ids:ide+1]
            direction = jnp.sign(mid_mfz)
            mid = upwind5_vert(
                direction=direction,
                wgt=upwind_wgt,
                q_minus_2=q_minus_2,
                q_minus_1=q_minus_1,
                q_0=q_0,
                q_plus_1=q_plus_1,
                q_plus_2=q_plus_2,
                q_plus_3=q_plus_3
            ) * mid_mfz
            # =================================================
            # full_kde - 1 <= k <= full_kde
            bottom = 0.5 * (
                q[kde-1:kde+1, ids:ide+1] + q[kde:kde+2, ids:ide+1]
            ) * mfz[kde-1:kde+1, ids:ide+1]
            # =================================================
            qmfz = qmfz.at[kds:kds+2  , ids:ide+1].set(top   )
            qmfz = qmfz.at[kds+2:kde-1, ids:ide+1].set(mid   )
            qmfz = qmfz.at[kde-1:kde+1, ids:ide+1].set(bottom)
        else:
            raise ValueError(f"upwind_calc_tracer_vflx: unsupported upwind order {upwind_order}!")
        return fill_field2d_array_halo(
            halo=mesh.halo,
            f=qmfz,
            meta=batch.qmfz.meta
        )
    else:
        raise ValueError(f"upwind_calc_tracer_vflx: unsupported batch location {batch.loc}!")

   
def upwind1_hori(
    q: FLT[Array, "nlev nCell"],
    wind: FLT[Array, "nEdge"],
    edd_v: INT[Array, "nEdge 2"],
    wgt: FloatType = 0.75,
) -> FLT[Array, "nlev nEdge"]:
    v1 = edd_v[:, 0]
    v2 = edd_v[:, 1]
    
    def _wrap(q: FLT[Array, "nCell"], wind: FLT[Array, "nEdge"]) -> FLT[Array, "nEdge"]:
        return (
            0.5 * (q[v1] + q[v2])
            - 0.5 * jnp.sign(wind) * wgt * (q[v2] - q[v1])
        )
        
    return jax.vmap(
        _wrap,
        in_axes=(0, 0),
        out_axes=0
    )(q, wind)


def upwind3_hori(
    q: FLT[Array, "nlev nCell"],
    wind: FLT[Array, "nEdge"],
    edd_v: INT[Array, "nEdge 2"],
    edd_leng: FLT[Array, "nEdge"],
    stencil_2nd_idx: INT[Array, "nCell max_m"],
    stencil_2nd_wgt: FLT[Array, "nEdge 2 max_m"],
    wgt: FloatType = 0.75,
    upwind_optimized: bool = True
) -> FLT[Array, "nlev nEdge"]:
    
    def _wrap(q: FLT[Array, "nCell"], wind: FLT[Array, "nEdge"]) -> FLT[Array, "nEdge"]:
        if upwind_optimized:
            nedge = edd_v.shape[0]
            v1 = edd_v[:, 0]
            v2 = edd_v[:, 1]
            local_iv = (-0.5 * jnp.sign(wind) + 0.5).astype(jnp.int32)
            vtx_upwind = edd_v[jnp.arange(nedge), local_iv]  # (nEdge, )
            dx_2nd = calc_2nd_derivative_upwind(
                vtx_upwind, local_iv, q, edd_leng, stencil_2nd_idx, stencil_2nd_wgt
            )
            return operator_pure_3rd_order_formula(
                q[v1], q[v2], dx_2nd
            )
        else:
            v1_idx = edd_v[:, 0]  # (nEdge, )
            v2_idx = edd_v[:, 1]  # (nEdge, )
            x1 = q[v1_idx]  # (nEdge, )
            x2 = q[v2_idx]  # (nEdge, )
            dx1_2nd = calc_2nd_derivative(
                v1_idx, 0, q, edd_leng, stencil_2nd_idx, stencil_2nd_wgt
            )
            dx2_2nd = calc_2nd_derivative(
                v2_idx, 1, q, edd_leng, stencil_2nd_idx, stencil_2nd_wgt
            )
            return operator_3rd4th_order_formula(
                x1, x2, dx1_2nd, dx2_2nd, wind, wgt
            )
    
    return jax.vmap(
        _wrap,
        in_axes=(0, 0),
        out_axes=0
    )(q, wind)


def upwind5_hori(
    q: FLT[Array, "nlev nCell"],
    wind: FLT[Array, "nEdge"],
    edd_v: INT[Array, "nEdge 2"],
    edd_leng: FLT[Array, "nEdge"],
    stencil_2nd_idx: INT[Array, "nCell max_m"],
    stencil_2nd_wgt: FLT[Array, "nEdge 2 max_m"],
    stencil_4th_idx: INT[Array, "nCell max_m"],
    stencil_4th_wgt: FLT[Array, "nEdge 2 max_m"],
    wgt: FloatType = 0.75,
) -> FLT[Array, "nlev nEdge"]:
    
    def _wrap(q: FLT[Array, "nCell"], wind: FLT[Array, "nEdge"]) -> FLT[Array, "nEdge"]:
        v1_idx = edd_v[:, 0]  # (nEdge, )
        v2_idx = edd_v[:, 1]  # (nEdge, )
        x1 = q[v1_idx]  # (nEdge, )
        x2 = q[v2_idx]  # (nEdge, )
        dx1_2nd = calc_2nd_derivative(
            v1_idx, 0, q, edd_leng, stencil_2nd_idx, stencil_2nd_wgt
        )
        dx2_2nd = calc_2nd_derivative(
            v2_idx, 1, q, edd_leng, stencil_2nd_idx, stencil_2nd_wgt
        )
        dx1_4th = calc_4th_derivative(
            v1_idx, 0, q, edd_leng, stencil_4th_idx, stencil_4th_wgt
        )
        dx2_4th = calc_4th_derivative(
            v2_idx, 1, q, edd_leng, stencil_4th_idx, stencil_4th_wgt
        )
        return operator_5th6th_order_formula(
            x1, x2, dx1_2nd, dx2_2nd, dx1_4th, dx2_4th, wind, wgt
        )
        
    return jax.vmap(
        _wrap,
        in_axes=(0, 0),
        out_axes=0
    )(q, wind)


def upwind1_vert(
    direction: FLT[Array, "z, n_points"],
    wgt: float,
    q_0: FLT[Array, "z, n_points"],
    q_plus_1: FLT[Array, "z, n_points"],
) -> FLT[Array, "z, n_points"]:
    c11 = 0.5
    c12 = -0.5
    return c11 * (q_plus_1 + q_0) + c12 * (q_plus_1 - q_0) * wgt * direction


def upwind3_vert(
    direction: FLT[Array, "z, n_points"],
    wgt: float,
    q_minus_1: FLT[Array, "z, n_points"],
    q_0: FLT[Array, "z, n_points"],
    q_plus_1: FLT[Array, "z, n_points"],
    q_plus_2: FLT[Array, "z, n_points"],
) -> FLT[Array, "z, n_points"]:
    c31 = 7.0 / 12.0
    c32 = -1.0 / 12.0
    c33 = 1.0 / 12.0
    return (
        c31 * (q_plus_1 + q_0)
        + c32 * (q_plus_2 + q_minus_1)
        + c33 * (
            q_plus_2 - q_minus_1 - 3 * (q_plus_1 - q_0)
        ) * wgt * direction
    )


def upwind5_vert(
    direction: FLT[Array, "z, n_points"],
    wgt: float,
    q_minus_2: FLT[Array, "z, n_points"],
    q_minus_1: FLT[Array, "z, n_points"],
    q_0: FLT[Array, "z, n_points"],
    q_plus_1: FLT[Array, "z, n_points"],
    q_plus_2: FLT[Array, "z, n_points"],
    q_plus_3: FLT[Array, "z, n_points"],
) -> FLT[Array, "z, n_points"]:
    c41 = 37.0 / 60.0
    c42 = -2.0 / 15.0
    c43 = 1.0 / 60.0
    c44 = -1.0 / 60.0
    return (
        c41 * (q_plus_1 + q_0)
        + c42 * (q_plus_2 + q_minus_1)
        + c43 * (q_plus_3 + q_minus_2)
        + c44 * (
            q_plus_3 - q_minus_2
            - 5 * (q_plus_2 - q_minus_1)
            + 10 * (q_plus_1 - q_0)
        ) * wgt * direction
    )



class UpwindStencilDataset(AbstractEquationAuxiliaryDataset):
    stencil_cell_2nd: UpwindStencilData
    stencil_cell_4th: UpwindStencilData
    stencil_vtx_2nd : UpwindStencilData
    stencil_vtx_4th : UpwindStencilData

    @classmethod
    def empty(cls) -> Self:
        return cls(
            initialized=False,
            stencil_cell_2nd=UpwindStencilData.empty(),
            stencil_cell_4th=UpwindStencilData.empty(),
            stencil_vtx_2nd=UpwindStencilData.empty(),
            stencil_vtx_4th=UpwindStencilData.empty(),
        )
        
    @classmethod
    def initialize(
        cls, 
        cfg: UserConfig,
        para: PlanetParameterSet,
        mesh: SCVTMesh,
        dev: device.AbstractDevice = device.CPU()
    ) -> Self:
        if cfg.adv_scheme == "upwind" or cfg.bg_adv_scheme == "upwind" or cfg.pt_adv_scheme == "upwind":
            if cfg.upwind_order == 3 and cfg.upwind_order_bg == 3:
                stencil_cell_2nd = UpwindStencilData.from_coord(
                    cfg=cfg,
                    para=para,
                    mesh=mesh,
                    loc="cell",
                    polynomial_order=2,
                    dev=dev
                )
                stencil_cell_4th = UpwindStencilData.empty()
            elif cfg.upwind_order == 5 or cfg.upwind_order_bg == 5:
                stencil_cell_2nd = UpwindStencilData.from_coord(
                    cfg=cfg,
                    para=para,
                    mesh=mesh,
                    loc="cell",
                    polynomial_order=2,
                    dev=dev
                )
                stencil_cell_4th = UpwindStencilData.from_coord(
                    cfg=cfg,
                    para=para,
                    mesh=mesh,
                    loc="cell",
                    polynomial_order=4,
                    dev=dev
                )
            else:
                raise ValueError(f"Unsupported upwind_order {cfg.upwind_order} or upwind_order_bg {cfg.upwind_order_bg}!")
        else:
            stencil_cell_2nd = UpwindStencilData.empty()
            stencil_cell_4th = UpwindStencilData.empty()
        if cfg.pv_adv_scheme == "upwind":
            if cfg.upwind_order_pv == 3:
                stencil_vtx_2nd = UpwindStencilData.from_coord(
                    cfg=cfg,
                    para=para,
                    mesh=mesh,
                    loc="vtx",
                    polynomial_order=2,
                    dev=dev
                )
                stencil_vtx_4th = UpwindStencilData.empty()
            elif cfg.upwind_order_pv == 5:
                stencil_vtx_2nd = UpwindStencilData.from_coord(
                    cfg=cfg,
                    para=para,
                    mesh=mesh,
                    loc="vtx",
                    polynomial_order=2,
                    dev=dev
                )
                stencil_vtx_4th = UpwindStencilData.from_coord(
                    cfg=cfg,
                    para=para,
                    mesh=mesh,
                    loc="vtx",
                    polynomial_order=4,
                    dev=dev
                    
                )
            else:
                stencil_vtx_2nd = UpwindStencilData.empty()
                stencil_vtx_4th = UpwindStencilData.empty()
        else:
            stencil_vtx_2nd = UpwindStencilData.empty()
            stencil_vtx_4th = UpwindStencilData.empty()
        return cls(
            initialized=True,
            stencil_cell_2nd=stencil_cell_2nd,
            stencil_cell_4th=stencil_cell_4th,
            stencil_vtx_2nd=stencil_vtx_2nd,
            stencil_vtx_4th=stencil_vtx_4th
        )


class UpwindStencilData(AbstractEquationAuxiliaryData):
    """Variables needed in upwind advection scheme"""
    idxes: FLT[Array, "nEdge nStencil"]
    wgt  : FLT[Array, "nEdge nStencil"]

    @classmethod
    def empty(cls) -> "UpwindStencilData":
        stencil_idxes = jnp.empty(0)
        wgt = jnp.empty(0)
        return cls(
            initialized=False,
            idxes=stencil_idxes,
            wgt=wgt
        )

    @classmethod
    def from_coord(
        cls,
        cfg: UserConfig,
        para: PlanetParameterSet,
        mesh: SCVTMesh,
        loc: str,
        polynomial_order: int,
        dev: device.AbstractDevice = device.CPU()
    ):
        if loc == "cell":
            fn = init_upwind_cell_stencil
        elif loc == "vtx":
            fn = init_upwind_vtx_stencil
        else:
            raise ValueError(
                f"Unsupported mesh type {loc} for flux reconstruction auxiliary data."
            )
        stencil_idxes, wgt = fn(cfg, mesh, para, polynomial_order)
        stencil_idxes = jax.device_put(stencil_idxes, device=dev())
        wgt = jax.device_put(wgt, device=dev())
        return cls(
            initialized=True,
            idxes=stencil_idxes,
            wgt=wgt,
        )
 
class AbstractUpwindStencilInfo(eqx.Module, ABC):
    n_coeff            : int      = eqx.field(static=True)
    polynomial_order   : int      = eqx.field(static=True)
    recon_polynomial_fn: Callable = eqx.field(static=True)
    spec_loc           : int      = eqx.field(static=True)
    polynomial_coef    : int      = eqx.field(static=True)
    mesh_type          : str      = eqx.field(static=True)
    max_member         : int      = eqx.field(static=True)
    stencil_fn         : Callable = eqx.field(static=True)

    def __init__(self, polynomial_order: int):
        if polynomial_order == 2:
            self.n_coeff = 5
            self.polynomial_order = polynomial_order
            self.recon_polynomial_fn = recon_2nd_polynomial
            self.spec_loc = 3 - 1
            self.polynomial_coef = 2
        elif polynomial_order == 4:
            self.n_coeff = 14
            self.polynomial_order = polynomial_order
            self.recon_polynomial_fn = recon_4th_polynomial
            self.spec_loc = 10 - 1
            self.polynomial_coef = 24
        else:
            raise ValueError("Only 2nd and 4th order least square supported.")

    @abstractmethod
    def calc_stencil(
        self,
        mesh: SCVTMesh,
    ) -> Tuple[INT[Array, "ncell"], INT[Array, "ncell max_m"]]:
        raise NotImplementedError

    def recon(
        self,
        local_x: INT[Array, "max_m"],
        local_y: INT[Array, "max_m"],
    ) -> FLT[Array, "n_coeff max_m"]:
        b_matrix = self.recon_polynomial_fn(
            local_x, local_y
        )  # (max_m, n_coeff)
        recon_b_matrix = b_matrix.T @ b_matrix  # (n_coeff, n_coeff)
        recon_b_matrix = math.pinv_svd(recon_b_matrix)  # (n_coeff, n_coeff)
        return recon_b_matrix @ b_matrix.T  # (n_coeff, max_m)
    
    
class UpwindStencilInfoAtCell(AbstractUpwindStencilInfo):
    def __init__(self, polynomial_order: int):
        super().__init__(polynomial_order)
        self.mesh_type = "primal"
        if self.polynomial_order == 2:
            self.max_member = 6
            self.stencil_fn = calc_hx_2nd_stencil
        elif self.polynomial_order == 4:
            self.max_member = 18
            self.stencil_fn = calc_hx_4th_stencil

    def calc_stencil(
        self, 
        mesh: SCVTMesh
    ) -> Tuple[INT[Array, "ncell"], INT[Array, "ncell max_m"]]:
        return self.stencil_fn(mesh, self.max_member)
    
    
class UpwindStencilInfoAtVtx(AbstractUpwindStencilInfo):
    def __init__(
        self, 
        polynomial_order: int, 
        tri_stencil_type: int = 1
    ):
        super().__init__(polynomial_order)
        self.mesh_type = "dual"
        if self.polynomial_order == 2:
            if tri_stencil_type == 1:
                self.max_member = 9
                self.stencil_fn = calc_tri_2nd_stencil_1
            if tri_stencil_type == 2:
                self.max_member = 12
                self.stencil_fn = calc_tri_2nd_stencil_2
            else:
                raise ValueError(f"Unsupported tri_stencil_type: {tri_stencil_type}")
        elif self.polynomial_order == 4:
            if tri_stencil_type == 1:
                self.max_member = 18
                self.stencil_fn = calc_tri_4th_stencil_1
            if tri_stencil_type == 2:
                self.max_member = 21
                self.stencil_fn = calc_tri_4th_stencil_2
            else:
                raise ValueError(f"Unsupported tri_stencil_type: {tri_stencil_type}") 

    def calc_stencil(
        self,
        mesh: SCVTMesh
    ) -> Tuple[INT[Array, "nvtx"], INT[Array, "nvtx max_m"]]:
        return self.stencil_fn(mesh, self.max_member)

    
def calc_2nd_derivative_upwind(
    global_iv: INT[Array, "nEdge"],
    local_iv: INT[Array, "nEdge"],
    x_at_cell: FLT[Array, "nCell"],
    leng: FLT[Array, "nEdge"],
    stencil_idx: INT[Array, "nCell max_m"],
    wgt: FLT[Array, "nEdge 2 max_m"],
) -> FLT[Array, "nEdge"]:
    nedge = wgt.shape[0]
    q0 = x_at_cell[global_iv]  # (nEdge)
    fcoe = wgt[jnp.arange(nedge), local_iv, :]  # (nEdge, max_m)
    idx = stencil_idx[global_iv]  # (nEdge, max_m)
    q = x_at_cell[idx]  # (nEdge, max_m)
    s = q - q0[:, None]  # (nEdge, max_m)
    dqdx_2nd = 2 * jnp.sum(fcoe * s, axis=-1)  # (nEdge)
    dq_2nd = dqdx_2nd * leng ** 2  # (nEdge)
    return dq_2nd


def calc_2nd_derivative(
    global_iv: INT[Array, "nEdge"],
    local_iv: IntType,
    x_at_cell: FLT[Array, "nCell"],
    leng: FLT[Array, "nEdge"],
    stencil_idx: INT[Array, "nCell max_m"],
    wgt: FLT[Array, "nEdge 2 max_m"],
) -> FLT[Array, "nEdge"]:
    q0 = x_at_cell[global_iv]  # (nEdge)
    fcoe = wgt[:, local_iv, :]  # (nEdge, max_m)
    idx = stencil_idx[global_iv]  # (nEdge, max_m)
    q = x_at_cell[idx]  # (nEdge, max_m)
    s = q - q0[:, None]  # (nEdge, max_m)
    dqdx_2nd = 2 * jnp.sum(fcoe * s, axis=-1)  # (nEdge)
    dq_2nd = dqdx_2nd * leng ** 2  # (nEdge)
    return dq_2nd


def operator_pure_3rd_order_formula(
    q1: FLT[Array, "nEdge"],
    q2: FLT[Array, "nEdge"],
    dq_2nd: FLT[Array, "nEdge"],
) -> FLT[Array, "nEdge"]:
    qe = 0.5 * (q1 + q2)
    qe = qe - dq_2nd / 6.0
    return qe


def operator_3rd4th_order_formula(
    q1: FLT[Array, "nEdge"],
    q2: FLT[Array, "nEdge"],
    dq1_2nd: FLT[Array, "nEdge 2"],
    dq2_2nd: FLT[Array, "nEdge 2"],
    normal_u: FLT[Array, "nEdge"],
    wgt: FloatType,
) -> FLT[Array, "nEdge"]:
    qe = 0.5 * (q1 + q2)
    qe = qe - (dq1_2nd + dq2_2nd) / 12.0
    qe = qe + jnp.sign(normal_u) * wgt * (dq2_2nd - dq1_2nd) / 12.0
    return qe


def operator_5th6th_order_formula(
    q1: FLT[Array, "nEdge"],
    q2: FLT[Array, "nEdge"],
    dq1_2nd: FLT[Array, "nEdge 2"],
    dq2_2nd: FLT[Array, "nEdge 2"],
    dq1_4th: FLT[Array, "nEdge 2"],
    dq2_4th: FLT[Array, "nEdge 2"],
    normal_u: FLT[Array, "nEdge"],
    wgt: FloatType,
) -> FLT[Array, "nEdge"]:
    qe = 0.5 * (q1 + q2)
    qe = qe - (dq1_2nd + dq2_2nd) / 12.0 + (dq1_4th + dq2_4th) / 60.0
    qe = qe - jnp.sign(normal_u) * wgt * (dq2_4th - dq1_4th) / 60.0
    return qe


def calc_4th_derivative_upwind(
    global_iv: INT[Array, "nEdge"],
    local_iv: INT[Array, "nEdge"],
    x_at_cell: FLT[Array, "nCell"],
    leng: FLT[Array, "nEdge"],
    stencil_idx: INT[Array, "nCell max_m"],
    wgt: FLT[Array, "nEdge 2 max_m"],
) -> FLT[Array, "nEdge"]:
    nedge = wgt.shape[0]
    q0 = x_at_cell[global_iv]  # (nEdge)
    fcoe = wgt[jnp.arange(nedge), local_iv, :]  # (nEdge, max_m)
    idx = stencil_idx[global_iv]  # (nEdge, max_m)
    q = x_at_cell[idx]  # (nEdge, max_m)
    s = q - q0[:, None]  # (nEdge, max_m)
    dqdx_4th = 24 * jnp.sum(fcoe * s, axis=-1)  # (nEdge)
    dq_4th = dqdx_4th * leng ** 4  # (nEdge)
    return dq_4th


def calc_4th_derivative(
    global_iv: INT[Array, "nEdge"],
    local_iv: IntType,
    x_at_cell: FLT[Array, "nCell"],
    leng: FLT[Array, "nEdge"],
    stencil_idx: INT[Array, "nCell max_m"],
    wgt: FLT[Array, "nEdge 2 max_m"],
) -> FLT[Array, "nEdge"]:
    q0 = x_at_cell[global_iv]  # (nEdge)
    fcoe = wgt[:, local_iv, :]  # (nEdge, max_m)
    idx = stencil_idx[global_iv]  # (nEdge, max_m)
    q = x_at_cell[idx]  # (nEdge, max_m)
    s = q - q0[:, None]  # (nEdge, max_m)
    dqdx_4th = 24 * jnp.sum(fcoe * s, axis=-1)  # (nEdge)
    dq_4th = dqdx_4th * leng ** 4  # (nEdge)
    return dq_4th


@partial(jax.jit, static_argnames=("max_m"), backend="cpu")
def calc_hx_2nd_stencil(mesh: SCVTMesh, max_m: int) -> Tuple[INT[Array, "ncell"], INT[Array, "ncell max_m"]]:
    """5/6 point stencil for 2nd order reconstruction on hexagonal mesh."""
    ims, ime = mesh.cell_ims, mesh.cell_ime
    ids, ihe = mesh.cell_ids, mesh.cell_ihe[0]
    stencil_num = jnp.zeros(ime-ims+1         , dtype=mesh.cell_nb.dtype)
    stencil_idx = jnp.zeros((ime-ims+1, max_m), dtype=mesh.cell_nb.dtype)
    # ==========================================================================
    stencil_num = stencil_num.at[ids:ihe+1   ].set(mesh.cell_nnb[ids:ihe+1   ])
    stencil_idx = stencil_idx.at[ids:ihe+1, :].set(mesh.cell_nb [ids:ihe+1, :])
    # ==========================================================================
    return stencil_num, stencil_idx


@partial(jax.jit, static_argnames=("max_m"), backend="cpu")
def calc_hx_4th_stencil(mesh: SCVTMesh, max_m: int) -> Tuple[INT[Array, "ncell"], INT[Array, "ncell max_m"]]:
    """15/17/18 point stencil for 4th order reconstruction on hexagonal mesh."""
    ims, ime = mesh.cell_ims, mesh.cell_ime
    ids, ihe = mesh.cell_ids, mesh.cell_ihe[0]
    stencil_num = jnp.zeros(ime-ims+1         , dtype=mesh.cell_nb.dtype)
    stencil_idx = jnp.zeros((ime-ims+1, max_m), dtype=mesh.cell_nb.dtype)
    # ==========================================================================
    cell_nb = mesh.cell_nb  # (ncell, maxnnb)
    maxnnb = mesh.maxnnb_cell

    def cell_body(_, cell_idx):
        nb = cell_nb[cell_idx, :]  # (maxnnb,)
        valid_nb = nb >= 0  # (maxnnb,)
        # padding invalid index value using -1
        nb_safe = jnp.where(valid_nb, nb, -1)  # (maxnnb,)
        nb_of_nb = cell_nb[nb_safe]  # (maxnnb, maxnnb)
        # Mask out the home cell contributions
        nb_of_nb = jnp.where(
            nb_of_nb == cell_idx, -1, nb_of_nb
        )  # (maxnnb, maxnnb)
        # Mask out contributions from invalid neighbors
        # filter out horizontal invalid neighbors
        nb_of_nb = jnp.where(
            valid_nb[:, None], nb_of_nb, -1
        )  # (maxnnb, maxnnb)
        # filter out vertical invalid neighbors
        nb_of_nb = jnp.where(
            valid_nb[None, :], nb_of_nb, -1
        )  # (maxnnb, maxnnb)
        stencil_idx = nb_of_nb.reshape((maxnnb * maxnnb,))  # (maxnnb*maxnnb)
        stencil_idx = jax_tools.filter_ordered_unique(
            stencil_idx, size=max_m, fill_value=-1
        )  # (max_m,)
        return None, stencil_idx

    _, stencil_idx_tmp = jax.lax.scan(cell_body, init=None, xs=jnp.arange(ids, ihe+1, 1))
    stencil_num_tmp = jnp.sum(stencil_idx_tmp >= 0, axis=-1)
    # ==========================================================================
    stencil_num = stencil_num.at[ids:ihe+1   ].set(stencil_num_tmp)
    stencil_idx = stencil_idx.at[ids:ihe+1, :].set(stencil_idx_tmp)
    return stencil_num, stencil_idx


@partial(jax.jit, static_argnames=("max_m"), backend="cpu")
def calc_tri_2nd_stencil_1(mesh: SCVTMesh, max_m: int) -> Tuple[INT[Array, "nvtx"], INT[Array, "nvtx max_m"]]:
    """9/10 point stencil for 2nd order reconstruction on triangular mesh."""
    ims, ime = mesh.vtx_ims, mesh.vtx_ime
    ids, ihe = mesh.vtx_ids, mesh.vtx_ihe[0]
    stencil_num = jnp.zeros(ime-ims+1         , dtype=mesh.cell_nb.dtype)
    stencil_idx = jnp.zeros((ime-ims+1, max_m), dtype=mesh.cell_nb.dtype)
    # ==========================================================================
    vtx_nb = mesh.vtx_nb  # (nDual_cell, 3)
    maxnnb = mesh.maxnnb_vtx

    def vtx_body(_, vtx_idx):
        nb = vtx_nb[vtx_idx, :]  # (3,)
        nb_of_nb = vtx_nb[nb, :]  # (3, 3)
        # padding invalid index value using -1
        nb_of_nb = jnp.where(
            nb_of_nb >= 0, nb_of_nb, -1
        )  # (3, maxnnb)
        # Mask out the home cell contributions
        nb_of_nb = jnp.where(
            nb_of_nb == vtx_idx, -1, nb_of_nb
        )  # (3, maxnnb)
        # Gather all neighboring cells
        nb_of_nb = nb_of_nb.reshape((3 * maxnnb, ))  # (3, maxnnb)
        stencil_idx = jnp.concatenate(
            [nb, nb_of_nb],
            axis=0
        )  # (3 + 3*maxnnb,)
        stencil_idx = jax_tools.filter_ordered_unique(
            stencil_idx, size=9, fill_value=-1
        )  # (9,)
        return None, stencil_idx

    _, stencil_idx_tmp = jax.lax.scan(vtx_body, None, xs=jnp.arange(ids, ihe+1, 1))
    stencil_num_tmp = jnp.sum(stencil_idx_tmp >= 0, axis=-1)
    # ==========================================================================
    stencil_num = stencil_num.at[ids:ihe+1   ].set(stencil_num_tmp)
    stencil_idx = stencil_idx.at[ids:ihe+1, :].set(stencil_idx_tmp)   
    return stencil_num, stencil_idx


@partial(jax.jit, static_argnames=("max_m"), backend="cpu")
def calc_tri_2nd_stencil_2(mesh: SCVTMesh, max_m: int) -> Tuple[INT[Array, "nvtx"], INT[Array, "nvtx max_m"]]:
    """11/12 point stencil for 2nd order reconstruction on triangular mesh."""
    ims, ime = mesh.vtx_ims, mesh.vtx_ime
    ids, ihe = mesh.vtx_ids, mesh.vtx_ihe[0]
    stencil_num = jnp.zeros(ime-ims+1         , dtype=mesh.cell_nb.dtype)
    stencil_idx = jnp.zeros((ime-ims+1, max_m), dtype=mesh.cell_nb.dtype)
    # ==========================================================================
    vtx_of_cell = mesh.vtx_of_cell  # (ncell, maxnnb_cell)
    cell_of_vtx = mesh.cell_of_vtx  # (nvtx , maxnnb_vtx )
    maxnnb_cell = mesh.maxnnb_cell

    def vtx_body(_, vtx_idx):
        icell = cell_of_vtx[vtx_idx, :]  # (3,)
        ivtx = vtx_of_cell[icell, :]     # (3, maxnnb)
        # padding invalid index value using -1
        ivtx = jnp.where(ivtx >= 0, ivtx, -1)  # (3, maxnnb)
        # Mask out the home cell contributions
        ivtx = jnp.where(ivtx == vtx_idx, -1, ivtx)     # (3, maxnnb)
        stencil_idx = ivtx.reshape((3 * maxnnb_cell,))  # (3*maxnnb)
        stencil_idx = jax_tools.filter_ordered_unique(
            stencil_idx, size=12, fill_value=-1
        )  # (12,)
        return None, stencil_idx

    _, stencil_idx_tmp = jax.lax.scan(vtx_body, None, xs=jnp.arange(ids, ihe+1, 1))
    stencil_num_tmp = jnp.sum(stencil_idx_tmp >= 0, axis=-1)
    # ==========================================================================
    stencil_num = stencil_num.at[ids:ihe+1   ].set(stencil_num_tmp)
    stencil_idx = stencil_idx.at[ids:ihe+1, :].set(stencil_idx_tmp)
    return stencil_num, stencil_idx


@partial(jax.jit, static_argnames=("max_m"), backend="cpu")
def calc_tri_4th_stencil_1(mesh: SCVTMesh, max_m: int) -> Tuple[INT[Array, "nvtx"], INT[Array, "nvtx max_m"]]:
    """17/18 point stencil for 4th order reconstruction on triangular mesh."""
    ims, ime = mesh.vtx_ims, mesh.vtx_ime
    ids, ihe = mesh.vtx_ids, mesh.vtx_ihe[0]
    stencil_num = jnp.zeros(ime-ims+1         , dtype=mesh.cell_nb.dtype  )
    stencil_idx = jnp.zeros((ime-ims+1, max_m), dtype=mesh.cell_nb.dtype)
    # ==========================================================================
    vtx_nb      = mesh.vtx_nb # (nvtx, 3)
    cell_of_vtx = mesh.cell_of_vtx  # (nvtx, 3)
    vtx_of_cell = mesh.vtx_of_cell  # (ncell, maxnnb)
    maxnnb      = mesh.maxnnb_cell

    def vtx_body(_, vtx_idx):
        icells = cell_of_vtx[vtx_idx, :]  # (3,)
        num_bp = 0
        base_point = jnp.zeros((3,), dtype=icells.dtype)

        def tri_nb_body(carry, inb):
            base_point, num_bp = carry
            ivtx = vtx_nb[vtx_idx, inb]

            def vtx_body(carry, iv):
                base_point, num_bp = carry
                icell = cell_of_vtx[ivtx, iv]
                mask = icell == icells  # (3,)

                def false_fn(carry):
                    return carry

                def true_fn(carry):
                    base_point, num_bp = carry
                    base_point = base_point.at[num_bp].set(icell)
                    return base_point, num_bp + 1

                base_point, num_bp = jax.lax.cond(
                    jnp.sum(mask) >= 1,
                    false_fn,
                    true_fn,
                    operand=(base_point, num_bp)
                )
                return (base_point, num_bp), None

            return jax.lax.scan(
                vtx_body,
                (base_point, num_bp),
                xs=jnp.arange(3)
            )

        base_point, num_bp = jax.lax.scan(
            tri_nb_body,
            (base_point, num_bp),
            xs=jnp.arange(3)
        )[0]

        # Now get neighboring triangles of these vertexes
        stencil_idx = vtx_of_cell[base_point, :]  # (3, maxnnb)
        stencil_idx = stencil_idx.reshape((3 * maxnnb,))  # (3*maxnnb)
        return None, stencil_idx

    _, stencil_idx_tmp = jax.lax.scan(vtx_body, None, xs=jnp.arange(ids, ihe+1, 1))
    stencil_num_tmp = jnp.sum(stencil_idx_tmp >= 0, axis=-1)
    # ==========================================================================
    stencil_num = stencil_num.at[ids:ihe+1   ].set(stencil_num_tmp)
    stencil_idx = stencil_idx.at[ids:ihe+1, :].set(stencil_idx_tmp)
    return stencil_num, stencil_idx


@partial(jax.jit, static_argnames=("max_m"), backend="cpu")
def calc_tri_4th_stencil_2(mesh: SCVTMesh, max_m: int) -> Tuple[INT[Array, "nvtx"], INT[Array, "nvtx max_m"]]:
    """20/21 point stencil for 4th order reconstruction on triangular mesh."""
    ims, ime = mesh.vtx_ims, mesh.vtx_ime
    ids, ihe = mesh.vtx_ids, mesh.vtx_ihe[0]
    stencil_num = jnp.zeros(ime-ims+1         , dtype=mesh.cell_nb.dtype  )
    stencil_idx = jnp.zeros((ime-ims+1, max_m), dtype=mesh.cell_nb.dtype)
    # ==========================================================================
    vtx_nb      = mesh.vtx_nb      # (nvtx, 3)
    cell_of_vtx = mesh.cell_of_vtx # (nvtx, 3)
    vtx_of_cell = mesh.vtx_of_cell # (ncell, maxnnb)
    maxnnb_cell = mesh.maxnnb_cell

    def vtx_body(_, vtx_idx):
        ivtx  = vtx_nb[vtx_idx, :]  # (3,)
        icell = cell_of_vtx[ivtx , :]  # (3, 3)
        ivtx  = vtx_of_cell[icell, :]  # (3, 3, maxnnb)
        # padding invalid index value using -1
        ivtx = jnp.where(ivtx >= 0, ivtx, -1)  # (3, 3, maxnnb)
        # Mask out the home cell contributions
        ivtx = jnp.where(
            ivtx == vtx_idx, -1, ivtx
        )  # (3, 3, maxnnb)
        stencil_idx = ivtx.reshape((3 * 3 * maxnnb_cell,))  # (3*3*maxnnb)
        stencil_idx = jax_tools.filter_ordered_unique(
            stencil_idx, size=max_m, fill_value=-1
        )
        return None, stencil_idx

    _, stencil_idx_tmp = jax.lax.scan(vtx_body, None, xs=jnp.arange(ids, ihe+1, 1))
    stencil_num_tmp = jnp.sum(stencil_idx_tmp >= 0, axis=-1)
    # ==========================================================================
    stencil_num = stencil_num.at[ids:ihe+1   ].set(stencil_num_tmp)
    stencil_idx = stencil_idx.at[ids:ihe+1, :].set(stencil_idx_tmp)
    return stencil_num, stencil_idx


def recon_2nd_polynomial(
    x: INT[Array, "max_m"],
    y: INT[Array, "max_m"]
) -> FLT[Array, "max_m n_coeff"]:
    return jnp.stack(
        [x, y, x ** 2, x * y, y ** 2], axis=-1
    )  # (max_m, 5)


def recon_4th_polynomial(
    x: INT[Array, "max_m"],
    y: INT[Array, "max_m"]
) -> FLT[Array, "max_m n_coeff"]:
    return jnp.stack(
        [
            x, y,
            x ** 2, x * y, y ** 2,
            x ** 3, (x ** 2) * y, x * (y ** 2), y ** 3,
            x ** 4, (x ** 3) * y, (x ** 2) * (y ** 2), x * (y ** 3), y ** 4
        ],
        axis=-1
    )  # (max_m, 14)


def set_lob_xy(
    R: FloatType,
    op0: FLT[Array, "3"],
    op1: FLT[Array, "3"],
) -> Tuple[FLT[Array, "3"], FLT[Array, "3"]]:
    eps = jnp.finfo(op0.dtype).eps
    p0p1 = _project_sphere2plane(R, op0, op1)  # (3, )
    # Avoid division by zero
    norm1 = jnp.linalg.norm(p0p1, axis=-1, keepdims=True)
    norm1 = jnp.maximum(norm1, eps*100)
    tx = p0p1 / norm1  # (3,)

    tmp = jnp.cross(-tx, op0)  # (3,)
    # Avoid division by zero
    norm2 = jnp.linalg.norm(tmp, axis=-1, keepdims=True)
    norm2 = jnp.maximum(norm2, eps*100)
    ty = tmp / norm2  # (3,)
    return tx, ty


def get_lob_xy(
    R: FloatType,
    op0: FLT[Array, "3"],
    opi: FLT[Array, "max_m 3"],
    tx: FLT[Array, "3"],
    ty: FLT[Array, "3"],
    is_valid_stencil: FLT[Array, "max_m"]
) -> Tuple[FLT[Array, "max_m"], FLT[Array, "max_m"]]:
    p0pi = project_sphere2plane(R, op0, opi)  # (max_m, 3)
    local_x = jnp.sum(p0pi * tx[None, :], axis=-1)  # (max_m,)
    local_y = jnp.sum(p0pi * ty[None, :], axis=-1)  # (max_m,)
    # Mask out invalid stencil members
    local_x = jnp.where(is_valid_stencil, local_x, 0.0)  # (max_m,)
    local_y = jnp.where(is_valid_stencil, local_y, 0.0)  # (max_m,)
    return local_x, local_y


def _calc_b_matrix_in_a_direction_of_cell(
    R: FloatType,
    op0: FLT[Array, "3"],
    op1: FLT[Array, "3"],
    opi: FLT[Array, "max_m 3"],
    is_valid_stencil: INT[Array, "max_m"],
    recon_builder: AbstractUpwindStencilInfo
) -> FLT[Array, "n_coeff max_m"]:
    tx, ty = set_lob_xy(R, op0, op1)  # (3,), (3,)
    local_x, local_y = get_lob_xy(
        R, op0, opi, tx, ty, is_valid_stencil
    )  # (max_m,), (max_m,)
    return recon_builder.recon(
        local_x, local_y
    )  # (n_coeff, max_m)


def _calc_b_matrix_at_single_cell(
    R: FloatType,
    op0: FLT[Array, "3"],
    op1: INT[Array, "maxnnb 3"],
    opi: INT[Array, "max_m 3"],
    is_valid_stencil: INT[Array, "max_m"],
    is_valid_nb: INT[Array, "maxnnb"],
    recon_builder: AbstractUpwindStencilInfo
) -> FLT[Array, "maxnnb n_coeff max_m"]:
    b_matrix = jax.vmap(
        _calc_b_matrix_in_a_direction_of_cell,
        in_axes=(None, None, 0, None, None, None),
        out_axes=0
    )(
        R,
        op0,
        op1,
        opi,
        is_valid_stencil,
        recon_builder
    )  # (maxnnb, n_coeff, max_m,)

    # Mask out invalid neighbors
    return jnp.where(
        is_valid_nb[:, None, None],
        b_matrix,
        0.0
    )


@jax.jit
def _calc_b_matrix_at_points(
    R: FloatType,
    op0: FLT[Array, "npoint 3"],
    op1: INT[Array, "npoint maxnnb 3"],
    opi: INT[Array, "npoint max_m 3"],
    is_valid_stencil: INT[Array, "npoint max_m"],
    is_valid_nb: INT[Array, "npoint maxnnb"],
    recon_builder: AbstractUpwindStencilInfo
) -> FLT[Array, "npoint maxnnb n_coeff max_M"]:
    npoint = op0.shape[0]

    def point_body(_, cell_idx):

        b_matrix = _calc_b_matrix_at_single_cell(
            R,
            op0[cell_idx, :],
            op1[cell_idx, :, :],
            opi[cell_idx, :, :],
            is_valid_stencil[cell_idx, :],
            is_valid_nb[cell_idx, :],
            recon_builder
        )

        return None, b_matrix

    _, b_matrix = jax.lax.scan(
        point_body, init=None, xs=jnp.arange(npoint)
    )

    return b_matrix


def _calc_b_matrix_at_vtx(
    cfg: UserConfig,
    mesh: SCVTMesh,
    polynomial_order: int
) -> FLT[Array, "nvtx maxnnb n_coeff max_m"]:
    op0 = mesh.vtx_xyz  # (nvtx, 3)
    vtx_nb = mesh.vtx_nb # (nvtx, maxnnb)
    vtx_xyz = mesh.vtx_xyz # (nvtx, 3)
    recon_builder = UpwindStencilInfoAtVtx(
        polynomial_order=polynomial_order,
        tri_stencil_type=cfg.tri_stencil_type
    )
    stencil_idxes = recon_builder.calc_stencil(mesh)[1]  # (nvtx, max_m)

    op1 = jnp.where(
        vtx_nb[..., None] >= 0,
        vtx_xyz[vtx_nb],
        0.0
    )  # (nvtx, maxnnb, 3)
    opi = jnp.where(
        stencil_idxes[..., None] >= 0,
        vtx_xyz[stencil_idxes],
        0.0
    )  # (nvtx, max_m, 3)
    return _calc_b_matrix_at_points(
        op0,
        op1,
        opi,
        stencil_idxes >= 0,
        vtx_nb >= 0,
        recon_builder
    )


def _calc_b_matrix_at_cell(
    cfg: UserConfig,
    mesh: SCVTMesh,
    polynomial_order: int,
) -> FLT[Array, "ncell maxnnb n_coeff max_m"]:
    op0 = mesh.cell_xyz # (ncell, 3)
    cell_nb = mesh.cell_nb # (ncell, maxnnb)
    cell_xyz = mesh.cell_xyz # (ncell, 3)
    recon_builder = UpwindStencilInfoAtCell(polynomial_order)
    stencil_idxes = recon_builder.calc_stencil(mesh)[1]

    op1 = jnp.where(
        cell_nb[..., None] >= 0,
        cell_xyz[cell_nb],
        0.0
    )  # (ncell, maxnnb, 3)
    opi = jnp.where(
        stencil_idxes[..., None] >= 0,
        cell_xyz[stencil_idxes],
        0.0
    )  # (ncell, max_m, 3)
    return _calc_b_matrix_at_points(
        op0,
        op1,
        opi,
        stencil_idxes >= 0,
        cell_nb >= 0,
        recon_builder
    )


@partial(jax.jit, static_argnames=("para"), backend="cpu")
def _calc_wgt_at_cell(
    mesh: SCVTMesh,
    para: PlanetParameterSet,
    recon_builder: UpwindStencilInfoAtCell
) -> Tuple[INT[Array, "ncell max_m"], FLT[Array, "nedge 2 max_m"]]:
    radius        = para.radius.unwrap()
    cell_xyz      = mesh.cell_xyz # (ncell, 3)
    cell_nb       = mesh.cell_nb  # (ncell, maxnnb)
    maxnnb        = mesh.maxnnb_cell
    nedge         = mesh.edge_ide - mesh.edge_ids + 1
    max_m         = recon_builder.max_member
    spec_loc      = recon_builder.spec_loc
    stencil_idxes = recon_builder.calc_stencil(mesh)[1]

    def edge_body(_, ie):

        def vtx_body(_, iv):
            v_idx = mesh.cell_of_edge[ie, iv]

            def nb_body(_, inb):
                globel_idx_ed = mesh.cell_ed[v_idx, inb]

                def do_nothing():
                    return jnp.zeros((max_m,), dtype=cell_xyz.dtype)

                def get_value():
                    op0 = cell_xyz[v_idx, :] / radius
                    op1 = cell_xyz[cell_nb[v_idx, inb], :] / radius
                    opi = cell_xyz[stencil_idxes[v_idx, :], :] / radius
                    is_valid_stencil = stencil_idxes[v_idx, :] >= 0
                    return _calc_b_matrix_in_a_direction_of_cell(
                        R=1.0,
                        op0=op0,
                        op1=op1,
                        opi=opi,
                        is_valid_stencil=is_valid_stencil,
                        recon_builder=recon_builder
                    )[spec_loc, :]

                return None, jax.lax.cond(
                    globel_idx_ed == ie,
                    get_value,
                    do_nothing,
                )

            _, b_wgt = jax.lax.scan(
                nb_body, init=None, xs=jnp.arange(maxnnb)
            )
            b_wgt = jnp.sum(b_wgt, axis=0)  # (max_m,)
            return None, b_wgt

        _, b_wgt = jax.lax.scan(
            vtx_body, init=None, xs=jnp.arange(2)
        )  # (2, max_m,)
        return None, b_wgt

    _, b_wgt = jax.lax.scan(
        edge_body, init=None, xs=jnp.arange(nedge)
    )  # (nedge, 2, max_m)

    b_wgt = fill_raw_array_halo(mesh.halo, b_wgt, "edge")

    return stencil_idxes, b_wgt


@partial(jax.jit, static_argnames=("para"), backend="cpu")
def _calc_wgt_at_vtx(
    mesh: SCVTMesh,
    para: PlanetParameterSet,
    recon_builder: UpwindStencilInfoAtVtx    
) -> Tuple[INT[Array, "nvtx max_m"], FLT[Array, "nedge 2 max_m"]]:
    radius        = para.radius.unwrap()
    vtx_xyz       = mesh.vtx_xyz  # (nvtx, 3)
    vtx_nb        = mesh.vtx_nb  # (nvtx, maxnnb)
    maxnnb        = mesh.maxnnb_vtx
    nedge         = mesh.edge_ide - mesh.edge_ids + 1
    max_m         = recon_builder.max_member
    spec_loc      = recon_builder.spec_loc
    stencil_idxes = recon_builder.calc_stencil(mesh)[1]

    def edge_body(_, ie):

        def vtx_body(_, iv):
            v_idx = mesh.vtx_of_edge[ie, iv]

            def nb_body(_, inb):
                globel_idx_ed = mesh.vtx_ed[v_idx, inb]

                def do_nothing():
                    return jnp.zeros((max_m,), dtype=vtx_xyz.dtype)

                def get_value():
                    op0 = vtx_xyz[v_idx, :] / radius
                    op1 = vtx_xyz[vtx_nb[v_idx, inb], :] / radius
                    opi = vtx_xyz[stencil_idxes[v_idx, :], :] / radius
                    is_valid_stencil = stencil_idxes[v_idx, :] >= 0
                    return _calc_b_matrix_in_a_direction_of_cell(
                        R=1.0,
                        op0=op0,
                        op1=op1,
                        opi=opi,
                        is_valid_stencil=is_valid_stencil,
                        recon_builder=recon_builder
                    )[spec_loc, :]

                return None, jax.lax.cond(
                    globel_idx_ed == ie,
                    get_value,
                    do_nothing,
                )

            _, b_wgt = jax.lax.scan(
                nb_body, init=None, xs=jnp.arange(maxnnb)
            )
            b_wgt = jnp.sum(b_wgt, axis=0)  # (max_m,)
            return None, b_wgt

        _, b_wgt = jax.lax.scan(
            vtx_body, init=None, xs=jnp.arange(2)
        )  # (2, max_m,)
        return None, b_wgt

    _, b_wgt = jax.lax.scan(
        edge_body, init=None, xs=jnp.arange(nedge)
    )  # (nedge, 2, max_m)

    b_wgt = fill_raw_array_halo(mesh.halo, b_wgt, "edge")

    return stencil_idxes, b_wgt


def init_upwind_cell_stencil(
    cfg: UserConfig,
    mesh: SCVTMesh,
    para: PlanetParameterSet,
    polynomial_order: int,
) -> Tuple[INT[Array, "ncell max_m"], FLT[Array, "nedge 2 max_m"]]:
    recon_builder = UpwindStencilInfoAtCell(polynomial_order=polynomial_order)
    return _calc_wgt_at_cell(
        mesh=mesh,
        para=para,
        recon_builder=recon_builder
    )


def init_upwind_vtx_stencil(
    cfg: UserConfig,
    mesh: SCVTMesh,
    para: PlanetParameterSet,
    polynomial_order: int,
) -> Tuple[INT[Array, "nvtx max_m"], FLT[Array, "nedge 2 max_m"]]:
    recon_builder = UpwindStencilInfoAtVtx(
        polynomial_order=polynomial_order,
        tri_stencil_type=cfg.tri_stencil_type
    )
    return _calc_wgt_at_vtx(
        mesh=mesh,
        para=para,
        recon_builder=recon_builder
    )
