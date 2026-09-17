from __future__ import annotations
from typing import (
    TYPE_CHECKING, 
    Self,
    Callable, 
    Tuple
)
from abc import ABC, abstractmethod
import numpy as np
from numba import njit


from terraflow.util import typing, device
from terraflow.util.profile import profile_scope
from terraflow.util import equinox as eqx
from terraflow.mesh.scvt.parallel import fill_field2d_array_halo, fill_raw_array_halo
from terraflow.dynamics.abstract import (
    AbstractEquationAuxiliaryData, 
    AbstractEquationAuxiliaryDataset
)


INT       = typing.Int
FLT       = typing.Float
IntType   = typing.IntType
FloatType = typing.FloatType
Array     = typing.Array


if TYPE_CHECKING:
    from terraflow.util.planet import PlanetParameterSet
    from terraflow.util.backend import AbstractBackend
    from terraflow.model.config import UserConfig
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv import batch


@profile_scope("upwind_calc_mass_hflx")
def upwind_calc_mass_hflx(
    batch: batch.AdvBatchType, 
    block: Block,
    m: FLT[Array, "nlev npoint"], 
    mfn: FLT[Array, "nlev nedge"], 
    dt: float
) -> FLT[Array, "nlev nedge"]:
    bkd = block.backend
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
                bkd,
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
                bkd,
                q=m, 
                wind=wind, 
                edd_v=mesh.cell_of_edge[ids:ide+1, :], 
                edd_leng=de,
                stencil_2nd_idx=stencil_2nd.idxes,
                stencil_2nd_wgt=stencil_2nd.wgt[ids:ide+1, :],
                wgt=upwind_wgt,
                upwind_optimized=upwind_optimized
            ) * un[kds:kde+1, ids:ide+1]
            mfn = mfn.at[kds:kde+1, ids:ide+1].set(tmp)
        elif upwind_order == 5:
            wind = un[kds:kde+1, ids:ide+1] * mesh.edpnr_eddtg[ids:ide+1]
            de = mesh.de[ids:ide+1] / radius
            tmp  = upwind5_hori(
                bkd,
                q=m, 
                wind=wind, 
                edd_v=mesh.cell_of_edge[ids:ide+1, :], 
                edd_leng=de,
                stencil_2nd_idx=stencil_2nd.idxes,
                stencil_2nd_wgt=stencil_2nd.wgt[ids:ide+1, :],
                stencil_4th_idx=stencil_4th.idxes,
                stencil_4th_wgt=stencil_4th.wgt[ids:ide+1, :],
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


@profile_scope("upwind_calc_tracer_hflx")
def upwind_calc_tracer_hflx(
    batch: batch.AdvBatchType, 
    block: Block,
    q: FLT[Array, "nlev npoint"], 
    qmfn: FLT[Array, "nlev nedge"], 
    dt: float
) -> FLT[Array, "nlev nedge"]:
    bkd = block.backend
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
                bkd,
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
                bkd,
                q=q, 
                wind=wind, 
                edd_v=mesh.cell_of_edge[ids:ide+1, :], 
                edd_leng=de,
                stencil_2nd_idx=stencil_2nd.idxes,
                stencil_2nd_wgt=stencil_2nd.wgt[ids:ide+1, :],
                wgt=upwind_wgt,
                upwind_optimized=upwind_optimized
            ) * mfn[kds:kde+1, ids:ide+1]
            qmfn = qmfn.at[kds:kde+1, ids:ide+1].set(tmp)
        elif upwind_order == 5:
            wind = mfn[kds:kde+1, ids:ide+1] * mesh.edpnr_eddtg[ids:ide+1]
            de = mesh.de[ids:ide+1] / radius
            tmp  = upwind5_hori(
                bkd,
                q=q, 
                wind=wind, 
                edd_v=mesh.cell_of_edge[ids:ide+1, :], 
                edd_leng=de,
                stencil_2nd_idx=stencil_2nd.idxes,
                stencil_2nd_wgt=stencil_2nd.wgt[ids:ide+1, :],
                stencil_4th_idx=stencil_4th.idxes,
                stencil_4th_wgt=stencil_4th.wgt[ids:ide+1, :],
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


@profile_scope("upwind_calc_tracer_vflx")
def upwind_calc_tracer_vflx(
    batch: batch.AdvBatchType,
    block: Block, 
    q: FLT[Array, "nlev npoint"], 
    qmfz: FLT[Array, "nlev nedge"], 
    dt: float
) -> FLT[Array, "nlev nedge"]:
    bkd = block.backend
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
            direction = bkd.sign(mid_mfz)
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
            top0 = bkd.zeros_like(q[kds, ids:ide+1])
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
            direction = bkd.sign(mid_mfz)
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
            bottom0 = bkd.zeros_like(q[kde-1, ids:ide+1])
            # =================================================
            qmfz = qmfz.at[kds        , ids:ide+1].set(top0   )
            qmfz = qmfz.at[kds+1      , ids:ide+1].set(top1   )
            qmfz = qmfz.at[kds+2:kde-1, ids:ide+1].set(res    )
            qmfz = qmfz.at[kde-1      , ids:ide+1].set(bottom1)
            qmfz = qmfz.at[kde        , ids:ide+1].set(bottom0)
        elif upwind_order == 5:
            # =================================================
            # k = half_kds
            top0 = bkd.zeros_like(q[kds, ids:ide+1])
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
            direction = bkd.sign(mid_mfz)
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
            bottom0 = bkd.zeros_like(q[kde-1, ids:ide+1])
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
            direction = bkd.sign(mfz_tmp)
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
            direction = bkd.sign(mid_mfz)
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
            direction = bkd.sign(mid_mfz)
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
    bkd: AbstractBackend,
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
            - 0.5 * bkd.sign(wind) * wgt * (q[v2] - q[v1])
        )
        
    return bkd.vmap(
        _wrap,
        in_axes=(0, 0),
        out_axes=0
    )(q, wind)


def upwind3_hori(
    bkd: AbstractBackend,
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
            local_iv = bkd.astype(-0.5 * bkd.sign(wind) + 0.5, bkd.int32)
            vtx_upwind = edd_v[bkd.arange(nedge), local_iv]  # (nEdge, )
            dx_2nd = calc_2nd_derivative_upwind(
                bkd, vtx_upwind, local_iv, q, edd_leng, stencil_2nd_idx, stencil_2nd_wgt
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
                bkd, v1_idx, 0, q, edd_leng, stencil_2nd_idx, stencil_2nd_wgt
            )
            dx2_2nd = calc_2nd_derivative(
                bkd, v2_idx, 1, q, edd_leng, stencil_2nd_idx, stencil_2nd_wgt
            )
            return operator_3rd4th_order_formula(
                bkd, x1, x2, dx1_2nd, dx2_2nd, wind, wgt
            )
    
    return bkd.vmap(
        _wrap,
        in_axes=(0, 0),
        out_axes=0
    )(q, wind)


def upwind5_hori(
    bkd: AbstractBackend,
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
            bkd, v1_idx, 0, q, edd_leng, stencil_2nd_idx, stencil_2nd_wgt
        )
        dx2_2nd = calc_2nd_derivative(
            bkd, v2_idx, 1, q, edd_leng, stencil_2nd_idx, stencil_2nd_wgt
        )
        dx1_4th = calc_4th_derivative(
            bkd, v1_idx, 0, q, edd_leng, stencil_4th_idx, stencil_4th_wgt
        )
        dx2_4th = calc_4th_derivative(
            bkd, v2_idx, 1, q, edd_leng, stencil_4th_idx, stencil_4th_wgt
        )
        return operator_5th6th_order_formula(
            bkd, x1, x2, dx1_2nd, dx2_2nd, dx1_4th, dx2_4th, wind, wgt
        )
        
    return bkd.vmap(
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
        dev: device.AbstractDevice
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
        stencil_idxes = np.empty(0)
        wgt = np.empty(0)
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
        dev: device.AbstractDevice
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
        stencil_idxes = dev.backend.asarray(stencil_idxes, device=dev())
        wgt = dev.backend.asarray(wgt, device=dev())
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
        bkd,
        local_x: INT[Array, "max_m"],
        local_y: INT[Array, "max_m"],
    ) -> FLT[Array, "n_coeff max_m"]:
        b_matrix = self.recon_polynomial_fn(
            bkd, local_x, local_y
        )  # (max_m, n_coeff)
        return bkd.pinv(b_matrix)  # (..., n_coeff, max_m)
    
    
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
            elif tri_stencil_type == 2:
                self.max_member = 12
                self.stencil_fn = calc_tri_2nd_stencil_2
            else:
                raise ValueError(f"Unsupported tri_stencil_type: {tri_stencil_type}")
        elif self.polynomial_order == 4:
            if tri_stencil_type == 1:
                self.max_member = 18
                self.stencil_fn = calc_tri_4th_stencil_1
            elif tri_stencil_type == 2:
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
    bkd: AbstractBackend,
    global_iv: INT[Array, "nEdge"],
    local_iv: INT[Array, "nEdge"],
    x_at_cell: FLT[Array, "nCell"],
    leng: FLT[Array, "nEdge"],
    stencil_idx: INT[Array, "nCell max_m"],
    wgt: FLT[Array, "nEdge 2 max_m"],
) -> FLT[Array, "nEdge"]:
    nedge = wgt.shape[0]
    q0 = x_at_cell[global_iv]  # (nEdge)
    fcoe = wgt[bkd.arange(nedge), local_iv, :]  # (nEdge, max_m)
    idx = stencil_idx[global_iv]  # (nEdge, max_m)
    q = x_at_cell[idx]  # (nEdge, max_m)
    s = q - q0[:, None]  # (nEdge, max_m)
    dqdx_2nd = 2 * bkd.sum(fcoe * s, axis=-1)  # (nEdge)
    dq_2nd = dqdx_2nd * leng ** 2  # (nEdge)
    return dq_2nd


def calc_2nd_derivative(
    bkd: AbstractBackend,
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
    dqdx_2nd = 2 * bkd.sum(fcoe * s, axis=-1)  # (nEdge)
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
    bkd: AbstractBackend,
    q1: FLT[Array, "nEdge"],
    q2: FLT[Array, "nEdge"],
    dq1_2nd: FLT[Array, "nEdge 2"],
    dq2_2nd: FLT[Array, "nEdge 2"],
    normal_u: FLT[Array, "nEdge"],
    wgt: FloatType,
) -> FLT[Array, "nEdge"]:
    qe = 0.5 * (q1 + q2)
    qe = qe - (dq1_2nd + dq2_2nd) / 12.0
    qe = qe + bkd.sign(normal_u) * wgt * (dq2_2nd - dq1_2nd) / 12.0
    return qe


def operator_5th6th_order_formula(
    bkd: AbstractBackend,
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
    qe = qe - bkd.sign(normal_u) * wgt * (dq2_4th - dq1_4th) / 60.0
    return qe


def calc_4th_derivative_upwind(
    bkd: AbstractBackend,
    global_iv: INT[Array, "nEdge"],
    local_iv: INT[Array, "nEdge"],
    x_at_cell: FLT[Array, "nCell"],
    leng: FLT[Array, "nEdge"],
    stencil_idx: INT[Array, "nCell max_m"],
    wgt: FLT[Array, "nEdge 2 max_m"],
) -> FLT[Array, "nEdge"]:
    nedge = wgt.shape[0]
    q0 = x_at_cell[global_iv]  # (nEdge)
    fcoe = wgt[bkd.arange(nedge), local_iv, :]  # (nEdge, max_m)
    idx = stencil_idx[global_iv]  # (nEdge, max_m)
    q = x_at_cell[idx]  # (nEdge, max_m)
    s = q - q0[:, None]  # (nEdge, max_m)
    dqdx_4th = 24 * bkd.sum(fcoe * s, axis=-1)  # (nEdge)
    dq_4th = dqdx_4th * leng ** 4  # (nEdge)
    return dq_4th


def calc_4th_derivative(
    bkd: AbstractBackend,
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
    dqdx_4th = 24 * bkd.sum(fcoe * s, axis=-1)  # (nEdge)
    dq_4th = dqdx_4th * leng ** 4  # (nEdge)
    return dq_4th


@njit(cache=True)
def _filter_ordered_unique_1d(values: np.ndarray, size: int, fill_value: int) -> np.ndarray:
    out = np.empty((size,), dtype=values.dtype)
    seen = np.empty((size,), dtype=values.dtype)
    for i in range(size):
        out[i] = fill_value
        seen[i] = fill_value

    count = 0
    for i in range(values.shape[0]):
        val = values[i]
        if val == fill_value:
            continue

        exists = False
        for j in range(count):
            if seen[j] == val:
                exists = True
                break
        if not exists and count < size:
            out[count] = val
            seen[count] = val
            count += 1
    return out


@njit(cache=True)
def _calc_hx_2nd_stencil_numba(
    ims: int,
    ime: int,
    ids: int,
    ihe: int,
    max_m: int,
    cell_nnb: np.ndarray,
    cell_nb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stencil_num = np.zeros((ime - ims + 1,), dtype=cell_nb.dtype)
    stencil_idx = np.zeros((ime - ims + 1, max_m), dtype=cell_nb.dtype)
    for ic in range(ids, ihe + 1):
        stencil_num[ic] = cell_nnb[ic]
        for im in range(min(max_m, cell_nb.shape[1])):
            stencil_idx[ic, im] = cell_nb[ic, im]
    return stencil_num, stencil_idx


def calc_hx_2nd_stencil(mesh: SCVTMesh, max_m: int) -> Tuple[INT[Array, "ncell"], INT[Array, "ncell max_m"]]:
    """5/6 point stencil for 2nd order reconstruction on hexagonal mesh."""
    dev = mesh.dev
    stencil_num, stencil_idx = _calc_hx_2nd_stencil_numba(
        int(mesh.cell_ims),
        int(mesh.cell_ime),
        int(mesh.cell_ids),
        int(mesh.cell_ihe[0]),
        max_m,
        dev.to_host(mesh.cell_nnb),
        dev.to_host(mesh.cell_nb),
    )
    return dev.to_backend(stencil_num), dev.to_backend(stencil_idx)


@njit(cache=True)
def _calc_hx_4th_stencil_numba(
    ims: int,
    ime: int,
    ids: int,
    ihe: int,
    max_m: int,
    maxnnb: int,
    cell_nnb: np.ndarray,
    cell_nb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stencil_num = np.zeros((ime - ims + 1,), dtype=cell_nb.dtype)
    stencil_idx = np.zeros((ime - ims + 1, max_m), dtype=cell_nb.dtype)
    scratch = np.empty((maxnnb * maxnnb,), dtype=cell_nb.dtype)

    for ic in range(ids, ihe + 1):
        n = 0
        for iee in range(maxnnb):
            valid_iee = cell_nb[ic, iee] >= 0
            nb = cell_nb[ic, iee] if valid_iee else -1
            for iep in range(maxnnb):
                valid_iep = cell_nb[ic, iep] >= 0
                value = -1
                if valid_iee and valid_iep:
                    value = cell_nb[nb, iep]
                    if value == ic:
                        value = -1
                    if value < 0:
                        value = -1
                scratch[n] = value
                n += 1

        row = _filter_ordered_unique_1d(scratch, max_m, -1)
        count = 0
        for im in range(max_m):
            stencil_idx[ic, im] = row[im]
            if row[im] >= 0:
                count += 1
        stencil_num[ic] = count

    return stencil_num, stencil_idx


def calc_hx_4th_stencil(mesh: SCVTMesh, max_m: int) -> Tuple[INT[Array, "ncell"], INT[Array, "ncell max_m"]]:
    """15/17/18 point stencil for 4th order reconstruction on hexagonal mesh."""
    dev = mesh.dev
    stencil_num, stencil_idx = _calc_hx_4th_stencil_numba(
        int(mesh.cell_ims),
        int(mesh.cell_ime),
        int(mesh.cell_ids),
        int(mesh.cell_ihe[0]),
        max_m,
        int(mesh.maxnnb_cell),
        dev.to_host(mesh.cell_nnb),
        dev.to_host(mesh.cell_nb),
    )
    return dev.to_backend(stencil_num), dev.to_backend(stencil_idx)


@njit(cache=True)
def _calc_tri_2nd_stencil_1_numba(
    ims: int,
    ime: int,
    ids: int,
    ihe: int,
    max_m: int,
    maxnnb: int,
    cell_nb_dtype_ref: np.ndarray,
    vtx_nb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stencil_num = np.zeros((ime - ims + 1,), dtype=cell_nb_dtype_ref.dtype)
    stencil_idx = np.zeros((ime - ims + 1, max_m), dtype=cell_nb_dtype_ref.dtype)
    scratch = np.empty((vtx_nb.shape[1] + vtx_nb.shape[1] * maxnnb,), dtype=cell_nb_dtype_ref.dtype)

    for ivtx in range(ids, ihe + 1):
        n = 0
        for inb in range(vtx_nb.shape[1]):
            scratch[n] = vtx_nb[ivtx, inb]
            n += 1
        for inb in range(vtx_nb.shape[1]):
            nb = vtx_nb[ivtx, inb]
            for jnb in range(maxnnb):
                value = vtx_nb[nb, jnb]
                if value < 0 or value == ivtx:
                    value = -1
                scratch[n] = value
                n += 1

        row = _filter_ordered_unique_1d(scratch, max_m, -1)
        count = 0
        for im in range(max_m):
            stencil_idx[ivtx, im] = row[im]
            if row[im] >= 0:
                count += 1
        stencil_num[ivtx] = count

    return stencil_num, stencil_idx


def calc_tri_2nd_stencil_1(mesh: SCVTMesh, max_m: int) -> Tuple[INT[Array, "nvtx"], INT[Array, "nvtx max_m"]]:
    """9/10 point stencil for 2nd order reconstruction on triangular mesh."""
    dev = mesh.dev
    cell_nb = dev.to_host(mesh.cell_nb)
    stencil_num, stencil_idx = _calc_tri_2nd_stencil_1_numba(
        int(mesh.vtx_ims),
        int(mesh.vtx_ime),
        int(mesh.vtx_ids),
        int(mesh.vtx_ihe[0]),
        max_m,
        int(mesh.maxnnb_vtx),
        np.empty(0, dtype=cell_nb.dtype),
        dev.to_host(mesh.vtx_nb),
    )
    return dev.to_backend(stencil_num), dev.to_backend(stencil_idx)


@njit(cache=True)
def _calc_tri_2nd_stencil_2_numba(
    ims: int,
    ime: int,
    ids: int,
    ihe: int,
    max_m: int,
    maxnnb_cell: int,
    cell_nb_dtype_ref: np.ndarray,
    vtx_of_cell: np.ndarray,
    cell_of_vtx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stencil_num = np.zeros((ime - ims + 1,), dtype=cell_nb_dtype_ref.dtype)
    stencil_idx = np.zeros((ime - ims + 1, max_m), dtype=cell_nb_dtype_ref.dtype)
    scratch = np.empty((cell_of_vtx.shape[1] * maxnnb_cell,), dtype=cell_nb_dtype_ref.dtype)

    for ivtx in range(ids, ihe + 1):
        n = 0
        for icell_idx in range(cell_of_vtx.shape[1]):
            icell = cell_of_vtx[ivtx, icell_idx]
            for j in range(maxnnb_cell):
                value = vtx_of_cell[icell, j]
                if value < 0 or value == ivtx:
                    value = -1
                scratch[n] = value
                n += 1

        row = _filter_ordered_unique_1d(scratch, max_m, -1)
        count = 0
        for im in range(max_m):
            stencil_idx[ivtx, im] = row[im]
            if row[im] >= 0:
                count += 1
        stencil_num[ivtx] = count

    return stencil_num, stencil_idx


def calc_tri_2nd_stencil_2(mesh: SCVTMesh, max_m: int) -> Tuple[INT[Array, "nvtx"], INT[Array, "nvtx max_m"]]:
    """11/12 point stencil for 2nd order reconstruction on triangular mesh."""
    dev = mesh.dev
    cell_nb = dev.to_host(mesh.cell_nb)
    stencil_num, stencil_idx = _calc_tri_2nd_stencil_2_numba(
        int(mesh.vtx_ims),
        int(mesh.vtx_ime),
        int(mesh.vtx_ids),
        int(mesh.vtx_ihe[0]),
        max_m,
        int(mesh.maxnnb_cell),
        np.empty(0, dtype=cell_nb.dtype),
        dev.to_host(mesh.vtx_of_cell),
        dev.to_host(mesh.cell_of_vtx),
    )
    return dev.to_backend(stencil_num), dev.to_backend(stencil_idx)


@njit(cache=True)
def _calc_tri_4th_stencil_1_numba(
    ims: int,
    ime: int,
    ids: int,
    ihe: int,
    max_m: int,
    maxnnb_cell: int,
    cell_nb_dtype_ref: np.ndarray,
    vtx_nb: np.ndarray,
    cell_of_vtx: np.ndarray,
    vtx_of_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stencil_num = np.zeros((ime - ims + 1,), dtype=cell_nb_dtype_ref.dtype)
    stencil_idx = np.zeros((ime - ims + 1, max_m), dtype=cell_nb_dtype_ref.dtype)
    base_point = np.empty((cell_of_vtx.shape[1],), dtype=cell_nb_dtype_ref.dtype)

    for ivtx in range(ids, ihe + 1):
        for i in range(base_point.shape[0]):
            base_point[i] = 0
        num_bp = 0

        for inb in range(vtx_nb.shape[1]):
            nb_vtx = vtx_nb[ivtx, inb]
            for iv in range(cell_of_vtx.shape[1]):
                icell = cell_of_vtx[nb_vtx, iv]
                exists = False
                for k in range(cell_of_vtx.shape[1]):
                    if icell == cell_of_vtx[ivtx, k]:
                        exists = True
                        break
                if not exists and num_bp < base_point.shape[0]:
                    base_point[num_bp] = icell
                    num_bp += 1

        n = 0
        for ibp in range(base_point.shape[0]):
            for j in range(maxnnb_cell):
                if n < max_m:
                    stencil_idx[ivtx, n] = vtx_of_cell[base_point[ibp], j]
                n += 1

        count = 0
        for im in range(max_m):
            if stencil_idx[ivtx, im] >= 0:
                count += 1
        stencil_num[ivtx] = count

    return stencil_num, stencil_idx


def calc_tri_4th_stencil_1(mesh: SCVTMesh, max_m: int) -> Tuple[INT[Array, "nvtx"], INT[Array, "nvtx max_m"]]:
    """17/18 point stencil for 4th order reconstruction on triangular mesh."""
    dev = mesh.dev
    cell_nb = dev.to_host(mesh.cell_nb)
    stencil_num, stencil_idx = _calc_tri_4th_stencil_1_numba(
        int(mesh.vtx_ims),
        int(mesh.vtx_ime),
        int(mesh.vtx_ids),
        int(mesh.vtx_ihe[0]),
        max_m,
        int(mesh.maxnnb_cell),
        np.empty(0, dtype=cell_nb.dtype),
        dev.to_host(mesh.vtx_nb),
        dev.to_host(mesh.cell_of_vtx),
        dev.to_host(mesh.vtx_of_cell),
    )
    return dev.to_backend(stencil_num), dev.to_backend(stencil_idx)


@njit(cache=True)
def _calc_tri_4th_stencil_2_numba(
    ims: int,
    ime: int,
    ids: int,
    ihe: int,
    max_m: int,
    maxnnb_cell: int,
    cell_nb_dtype_ref: np.ndarray,
    vtx_nb: np.ndarray,
    cell_of_vtx: np.ndarray,
    vtx_of_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stencil_num = np.zeros((ime - ims + 1,), dtype=cell_nb_dtype_ref.dtype)
    stencil_idx = np.zeros((ime - ims + 1, max_m), dtype=cell_nb_dtype_ref.dtype)
    scratch = np.empty((vtx_nb.shape[1] * cell_of_vtx.shape[1] * maxnnb_cell,), dtype=cell_nb_dtype_ref.dtype)

    for ivtx in range(ids, ihe + 1):
        n = 0
        for inb in range(vtx_nb.shape[1]):
            nb_vtx = vtx_nb[ivtx, inb]
            for icell_idx in range(cell_of_vtx.shape[1]):
                icell = cell_of_vtx[nb_vtx, icell_idx]
                for j in range(maxnnb_cell):
                    value = vtx_of_cell[icell, j]
                    if value < 0 or value == ivtx:
                        value = -1
                    scratch[n] = value
                    n += 1

        row = _filter_ordered_unique_1d(scratch, max_m, -1)
        count = 0
        for im in range(max_m):
            stencil_idx[ivtx, im] = row[im]
            if row[im] >= 0:
                count += 1
        stencil_num[ivtx] = count

    return stencil_num, stencil_idx


def calc_tri_4th_stencil_2(mesh: SCVTMesh, max_m: int) -> Tuple[INT[Array, "nvtx"], INT[Array, "nvtx max_m"]]:
    """20/21 point stencil for 4th order reconstruction on triangular mesh."""
    dev = mesh.dev
    cell_nb = dev.to_host(mesh.cell_nb)
    stencil_num, stencil_idx = _calc_tri_4th_stencil_2_numba(
        int(mesh.vtx_ims),
        int(mesh.vtx_ime),
        int(mesh.vtx_ids),
        int(mesh.vtx_ihe[0]),
        max_m,
        int(mesh.maxnnb_cell),
        np.empty(0, dtype=cell_nb.dtype),
        dev.to_host(mesh.vtx_nb),
        dev.to_host(mesh.cell_of_vtx),
        dev.to_host(mesh.vtx_of_cell),
    )
    return dev.to_backend(stencil_num), dev.to_backend(stencil_idx)


def recon_2nd_polynomial(
    bkd,
    x: INT[Array, "max_m"],
    y: INT[Array, "max_m"]
) -> FLT[Array, "max_m n_coeff"]:
    return bkd.stack(
        [x, y, x ** 2, x * y, y ** 2], axis=-1
    )  # (max_m, 5)


def recon_4th_polynomial(
    bkd,
    x: INT[Array, "max_m"],
    y: INT[Array, "max_m"]
) -> FLT[Array, "max_m n_coeff"]:
    return bkd.stack(
        [
            x, y,
            x ** 2, x * y, y ** 2,
            x ** 3, (x ** 2) * y, x * (y ** 2), y ** 3,
            x ** 4, (x ** 3) * y, (x ** 2) * (y ** 2), x * (y ** 3), y ** 4
        ],
        axis=-1
    )  # (max_m, 14)


def _cross_last_dim(bkd, a: Array, b: Array) -> Array:
    return bkd.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def _project_sphere2plane_batched(
    bkd,
    R: FloatType,
    op0: Array,
    op1: Array,
) -> Array:
    n0 = bkd.norm(op0, axis=-1, keepdims=False)
    n1 = bkd.norm(op1, axis=-1, keepdims=False)
    eps = bkd.eps(op0.dtype)
    margin = max(float(eps) * 100, 1e-15)
    n0 = bkd.maximum(n0, eps * 100)
    n1 = bkd.maximum(n1, eps * 100)
    cos_lambda = bkd.sum(op0 * op1, axis=-1) / (n0 * n1)
    cos_lambda = bkd.clip(cos_lambda, -1.0 + margin, 1.0 - margin)
    lam = bkd.arccos(cos_lambda)
    op2 = op1 / cos_lambda[..., None]
    p0p2 = op2 - op0
    tan_lam = bkd.tan(lam)
    tan_lam = bkd.where(
        bkd.abs(tan_lam) < eps,
        bkd.where(tan_lam < 0, -eps, eps),
        tan_lam,
    )
    return p0p2 * (lam / tan_lam)[..., None]


def set_lob_xy(
    bkd: AbstractBackend,
    R: FloatType,
    op0: FLT[Array, "3"],
    op1: FLT[Array, "3"],
) -> Tuple[FLT[Array, "3"], FLT[Array, "3"]]:
    eps = bkd.eps(op0.dtype)
    p0p1 = _project_sphere2plane_batched(bkd, R, op0, op1)  # (3, )
    # Avoid division by zero
    norm1 = bkd.norm(p0p1, axis=-1, keepdims=True)
    norm1 = bkd.maximum(norm1, eps*100)
    tx = p0p1 / norm1  # (3,)

    tmp = _cross_last_dim(bkd, -tx, op0)  # (3,)
    # Avoid division by zero
    norm2 = bkd.norm(tmp, axis=-1, keepdims=True)
    norm2 = bkd.maximum(norm2, eps*100)
    ty = tmp / norm2  # (3,)
    return tx, ty


def get_lob_xy(
    bkd: AbstractBackend,
    R: FloatType,
    op0: FLT[Array, "3"],
    opi: FLT[Array, "max_m 3"],
    tx: FLT[Array, "3"],
    ty: FLT[Array, "3"],
    is_valid_stencil: FLT[Array, "max_m"]
) -> Tuple[FLT[Array, "max_m"], FLT[Array, "max_m"]]:
    p0pi = _project_sphere2plane_batched(bkd, R, op0, opi)  # (max_m, 3)
    local_x = bkd.sum(p0pi * tx[None, :], axis=-1)  # (max_m,)
    local_y = bkd.sum(p0pi * ty[None, :], axis=-1)  # (max_m,)
    # Mask out invalid stencil members
    local_x = bkd.where(is_valid_stencil, local_x, 0.0)  # (max_m,)
    local_y = bkd.where(is_valid_stencil, local_y, 0.0)  # (max_m,)
    return local_x, local_y


def _calc_b_matrix_at_points(
    bkd: AbstractBackend,
    R: FloatType,
    op0: FLT[Array, "npoint 3"],
    op1: INT[Array, "npoint maxnnb 3"],
    opi: INT[Array, "npoint max_m 3"],
    is_valid_stencil: INT[Array, "npoint max_m"],
    is_valid_nb: INT[Array, "npoint maxnnb"],
    recon_builder: AbstractUpwindStencilInfo
) -> FLT[Array, "npoint maxnnb n_coeff max_M"]:
    op0_for_dir = op0[:, None, :]
    p0p1 = _project_sphere2plane_batched(bkd, R, op0_for_dir, op1)
    eps   = bkd.eps(op0.dtype)
    norm1 = bkd.norm(p0p1, axis=-1, keepdims=True)
    norm1 = bkd.maximum(norm1, eps * 100)
    tx = p0p1 / norm1

    tmp = _cross_last_dim(bkd, -tx, op0_for_dir)
    norm2 = bkd.norm(tmp, axis=-1, keepdims=True)
    norm2 = bkd.maximum(norm2, eps * 100)
    ty = tmp / norm2

    p0pi = _project_sphere2plane_batched(
        bkd,
        R,
        op0[:, None, None, :],
        opi[:, None, :, :],
    )
    local_x = bkd.sum(p0pi * tx[:, :, None, :], axis=-1)
    local_y = bkd.sum(p0pi * ty[:, :, None, :], axis=-1)
    local_x = bkd.where(is_valid_stencil[:, None, :], local_x, 0.0)
    local_y = bkd.where(is_valid_stencil[:, None, :], local_y, 0.0)
    b_matrix = recon_builder.recon(bkd, local_x, local_y)
    return bkd.where(is_valid_nb[:, :, None, None], b_matrix, 0.0)


def _calc_recon_weights_for_points(
    bkd: AbstractBackend,
    point_xyz: Array,
    point_nb: Array,
    stencil_idxes: Array,
    recon_builder: AbstractUpwindStencilInfo,
) -> Array:
    safe_point_nb = bkd.where(point_nb >= 0, point_nb, 0)
    safe_stencil_idxes = bkd.where(stencil_idxes >= 0, stencil_idxes, 0)
    fallback_point = point_xyz[:, None, :]
    op1 = bkd.where(
        point_nb[..., None] >= 0,
        point_xyz[safe_point_nb],
        fallback_point,
    )
    opi = bkd.where(
        stencil_idxes[..., None] >= 0,
        point_xyz[safe_stencil_idxes],
        fallback_point,
    )
    return _calc_b_matrix_at_points(
        bkd,
        1.0,
        point_xyz,
        op1,
        opi,
        stencil_idxes >= 0,
        point_nb >= 0,
        recon_builder,
    )


def _build_b_wgt_at_edges(
    bkd: AbstractBackend,
    b_wgt_at_point: Array,
    point_of_edge: Array,
    edge_of_point: Array,
    spec_loc: int,
    nedge: int,
) -> Array:
    edge_idx = bkd.arange(nedge, dtype=point_of_edge.dtype)
    point_idx = point_of_edge[:nedge, :]
    edge_idx_of_point = edge_of_point[point_idx]
    matches = edge_idx_of_point == edge_idx[:, None, None]
    valid = bkd.any(matches, axis=-1)
    local_idx = bkd.argmax(bkd.astype(matches, point_of_edge.dtype), axis=-1)
    b_wgt = b_wgt_at_point[point_idx, local_idx, spec_loc, :]
    return bkd.where(valid[..., None], b_wgt, 0.0)


def _calc_wgt_at_cell(
    mesh: SCVTMesh,
    para: PlanetParameterSet,
    recon_builder: UpwindStencilInfoAtCell
) -> Tuple[INT[Array, "ncell max_m"], FLT[Array, "nedge 2 max_m"]]:
    bkd           = mesh.dev.backend
    radius        = para.radius.unwrap()
    nedge         = mesh.edge_ime - mesh.edge_ims + 1
    spec_loc      = recon_builder.spec_loc
    stencil_idxes = recon_builder.calc_stencil(mesh)[1]
    b_wgt_ = _calc_recon_weights_for_points(
        bkd,
        mesh.cell_xyz / radius,
        mesh.cell_nb,
        stencil_idxes,
        recon_builder,
    )
    b_wgt = _build_b_wgt_at_edges(
        bkd,
        b_wgt_,
        mesh.cell_of_edge,
        mesh.cell_ed,
        spec_loc,
        nedge,
    )
    b_wgt = fill_raw_array_halo(mesh.halo, b_wgt, "edge")
    return stencil_idxes, b_wgt


def _calc_wgt_at_vtx(
    mesh: SCVTMesh,
    para: PlanetParameterSet,
    recon_builder: UpwindStencilInfoAtVtx    
) -> Tuple[INT[Array, "nvtx max_m"], FLT[Array, "nedge 2 max_m"]]:
    bkd           = mesh.dev.backend
    radius        = para.radius.unwrap()
    nedge         = mesh.edge_ime - mesh.edge_ims + 1
    spec_loc      = recon_builder.spec_loc
    stencil_idxes = recon_builder.calc_stencil(mesh)[1]
    b_wgt_ = _calc_recon_weights_for_points(
        bkd,
        mesh.vtx_xyz / radius,
        mesh.vtx_nb,
        stencil_idxes,
        recon_builder,
    )
    b_wgt = _build_b_wgt_at_edges(
        bkd,
        b_wgt_,
        mesh.vtx_of_edge,
        mesh.vtx_ed,
        spec_loc,
        nedge,
    )
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
