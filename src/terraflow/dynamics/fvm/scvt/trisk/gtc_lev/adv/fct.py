from __future__ import annotations
from typing import TYPE_CHECKING


from terraflow.util.profile import profile_scope
from terraflow.mesh.scvt.parallel import fill_field2d_array_halo
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv.upwind import upwind1_hori, upwind1_vert


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv import batch
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block

    INT = typing.Int
    FLT = typing.Float
    Array = typing.Array


@profile_scope("fct_calc_hflx")
def fct_calc_hflx(
    batch: batch.AdvBatchType,
    block: Block,
    q: FLT[Array, "nlev npoint"],
    q_old: FLT[Array, "nlev npoint"],
    qmfn: FLT[Array, "nlev nedge"],
    dt: float
) -> FLT[Array, "nlev nedge"]:
    bkd          = block.backend
    mesh         = block.mesh
    kds  , kde   = mesh.full_kds, mesh.full_kde
    dmg_old      = batch.m      .unwrap()
    dmg_new      = batch.m_new  .unwrap()
    mfn          = batch.mfn    .unwrap()
    qmfn_lo      = batch.qmfn_lo.unwrap()
    afn          = batch.afn    .unwrap()
    q_td         = batch.td     .unwrap()
    r_in         = batch.r_in   .unwrap()
    r_out        = batch.r_out  .unwrap()
    # ==========================================================
    # Calculate low-order and anti- diffusive fluxes
    ids, ihe = mesh.edge_ids, mesh.edge_ihe[0]
    # ------ Calculate low-order fluxes ------
    wind = mfn[kds:kde+1, ids:ihe+1] * mesh.edpnr_eddtg[ids:ihe+1]
    qmfn_lo = bkd.slice_update(qmfn_lo, (slice(kds, kde+1), slice(ids, ihe+1)),
        upwind1_hori(
            bkd,
            q=q_old,
            wind=wind,
            edd_v=mesh.cell_of_edge[ids:ihe+1, :],
            wgt=block.cfg.upwind_wgt
        ) * mfn[kds:kde+1, ids:ihe+1]
    )
    qmfn_lo = fill_field2d_array_halo(
        halo=mesh.halo,
        f=qmfn_lo,
        meta=batch.qmfn_lo.meta
    )
    # ------ Calculate anti-diffusive fluxes ------
    afn = bkd.slice_update(afn, (slice(kds, kde+1), slice(ids, ihe+1)),
        (qmfn[kds:kde+1, ids:ihe+1] - qmfn_lo[kds:kde+1, ids:ihe+1]) * mesh.le[ids:ihe+1]
    )
    # ========================================================
    # Calculate the updated low order solution
    ids, ihe  = mesh.cell_ids, mesh.cell_ihe[1]
    q_        = q      [kds:kde+1, ids:ihe+1]
    q_old_    = q_old  [kds:kde+1, ids:ihe+1]
    dmg_old_  = dmg_old[kds:kde+1, ids:ihe+1]
    dmg_new_  = dmg_new[kds:kde+1, ids:ihe+1]
    cell_ed   = mesh.cell_ed  [ids:ihe+1, :]  # (ncell, maxnnb)
    cell_nr   = mesh.cell_nr  [ids:ihe+1, :]  # (ncell, maxnnb)
    area_cell = mesh.area_cell[ids:ihe+1   ]  # (ncell,)
    # --------------------------------------------------------
    def _wrap(f_1d: FLT[Array, "nedge"]) -> FLT[Array, "ncell"]:
        ed_idx = cell_ed  # (ncell, maxnnb)
        leng_tile = mesh.le[ed_idx]  # (ncell, maxnnb)
        x_tile = f_1d[ed_idx]  # (ncell, maxnnb)
        x_tile = bkd.where(ed_idx < 0, 0.0, x_tile)
        return bkd.sum(
            cell_nr * leng_tile * x_tile,
            axis=-1
        ) / area_cell

    div = bkd.vmap(
        _wrap,
        in_axes=0,
        out_axes=0
    )(qmfn_lo[kds:kde+1, :])
    q_td = bkd.slice_update(q_td, (slice(kds, kde+1), slice(ids, ihe+1)), (q_old_ * dmg_old_ - div * dt) / dmg_new_)
    # ========================================================
    # Calculate limiter R values (nlev, ncell)
    ids, ihe  = mesh.cell_ids, mesh.cell_ihe[0]
    q_        = q      [kds:kde+1, ids:ihe+1]
    q_td_     = q_td   [kds:kde+1, ids:ihe+1]
    dmg_new_  = dmg_new[kds:kde+1, ids:ihe+1]
    cell_ed   = mesh.cell_ed  [ids:ihe+1, :]
    cell_nb   = mesh.cell_nb  [ids:ihe+1, :]
    cell_nr   = mesh.cell_nr  [ids:ihe+1, :]
    area_cell = mesh.area_cell[ids:ihe+1   ]
    # ------ Calculate qmax ------
    qmax = - bkd.array(1.e16, dtype=q.dtype)
    qmax = bkd.maximum(qmax, bkd.maximum(q_, q_td_))

    # Filter out the invalid neighbors (with negative index) by setting their q values to very large numbers
    # to ensure they do not affect the calculation of qmax.
    q_nnb = q[kds:kde+1, cell_nb]  # (nlev, ncell, nnb)
    q_td_nnb = q_td[kds:kde+1, cell_nb]  # (nlev, ncell, nnb)
    q_nnb = bkd.where(
        cell_nb[None, :, :] < 0,
        - bkd.array(1.e16, dtype=q.dtype),
        q_nnb
    )
    q_td_nnb = bkd.where(
        cell_nb[None, :, :] < 0,
        - bkd.array(1.e16, dtype=q.dtype),
        q_td_nnb
    )
    qmax = bkd.maximum(
        qmax,
        bkd.maximum(
            bkd.max(q_nnb, axis=-1),
            bkd.max(q_td_nnb, axis=-1)
        )
    )
    # ------ Calculate qmin ------
    qmin = bkd.array(1.e16, dtype=q.dtype)
    qmin = bkd.minimum(qmin, bkd.minimum(q_, q_td_))
    # Filter out the invalid neighbors (with negative index) by setting their q values to very small numbers
    # to ensure they do not affect the calculation of qmin.
    q_nnb = q[kds:kde+1, cell_nb]  # (nlev, ncell, nnb)
    q_td_nnb = q_td[kds:kde+1, cell_nb]  # (nlev, ncell, nnb)
    q_nnb = bkd.where(
        cell_nb[None, :, :] < 0,
        bkd.array(1.e16, dtype=q.dtype),
        q_nnb
    )
    q_td_nnb = bkd.where(
        cell_nb[None, :, :] < 0,
        bkd.array(1.e16, dtype=q.dtype),
        q_td_nnb
    )

    qmin = bkd.minimum(
        qmin,
        bkd.minimum(
            bkd.min(q_nnb, axis=-1),
            bkd.min(q_td_nnb, axis=-1)
        )
    )
    # ------ Calculate p_in and p_out ------
    tmp = cell_nr[None, :, :] * afn[kds:kde+1, cell_ed]  # (nlev, ncell, nnb)
    tmp = bkd.where(
        cell_ed[None, :, :] < 0,
        0.0,
        tmp
    )
    p_in = bkd.minimum(
        bkd.array(0.0, dtype=q.dtype),
        tmp  # (nlev, ncell, nnb)
    ).sum(axis=-1)  # (nlev, ncell)
    p_out = bkd.maximum(
        bkd.array(0.0, dtype=q.dtype),
        tmp  # (nlev, ncell, nnb)
    ).sum(axis=-1)  # (nlev, ncell)
    # ------ Calculate m_in and m_out (nlev, ncell) ------
    maf_in  = dmg_new_ * (qmax - q_td_) * area_cell / dt
    maf_out = dmg_new_ * (q_td_ - qmin) * area_cell / dt
    # Calculate limiter R values (nlev, ncell)
    if block.cfg.fct_regression_h:
        r_in = bkd.slice_update(r_in, (slice(kds, kde+1), slice(ids, ihe+1)),
            bkd.where(
                p_in < 0,
                bkd.minimum(
                    bkd.array(1.0, dtype=q.dtype),
                    maf_in / bkd.abs(p_in)
                ),
                0.0
            )
        )
        r_out = bkd.slice_update(r_out, (slice(kds, kde+1), slice(ids, ihe+1)),
            bkd.where(
                p_out > 0,
                bkd.minimum(
                    bkd.array(1.0, dtype=q.dtype),
                    maf_out / bkd.abs(p_out)
                ),
                0.0
            )
        )
    else:
        if q.dtype == bkd.float32:
            kesi = 1.e-40
        elif q.dtype == bkd.float64:
            kesi = 1.e-80
        else:
            raise ValueError("Unsupported data type for q.")
        r_in = bkd.slice_update(r_in, (slice(kds, kde+1), slice(ids, ihe+1)),
            bkd.minimum(
                bkd.array(1.0, dtype=q.dtype),
                maf_in / (bkd.abs(p_in) + kesi)
            )
        )
        r_out = bkd.slice_update(r_out, (slice(kds, kde+1), slice(ids, ihe+1)),
            bkd.minimum(
                bkd.array(1.0, dtype=q.dtype),
                maf_out / (bkd.abs(p_out) + kesi)
            )
        )
    # ========================================================
    # Calculate limited mass flux at edge
    ids, ide = mesh.edge_ids, mesh.edge_ide
    v0   = mesh.cell_of_edge[ids:ide+1, 0]
    v1   = mesh.cell_of_edge[ids:ide+1, 1]
    v0v1 = mesh.cell_xyz[v1, :] - mesh.cell_xyz[v0, :]
    flag = bkd.sign(
        bkd.sum(
            v0v1 * mesh.edge_nr[ids:ide+1, :],
            axis=-1
        )
    )  # 0 or 1
    mark = bkd.astype(
        bkd.rint(-0.5 * bkd.sign(flag * afn[kds:kde+1, ids:ide+1]) + 0.5),
        bkd.int32
    )  # (nlev, nedge)
    v_upwind = mesh.cell_of_edge[bkd.arange(ids, ide+1, 1),   mark]  # (nlev, nedge,)
    v_dowind = mesh.cell_of_edge[bkd.arange(ids, ide+1, 1), 1-mark]  # (nlev, nedge,)
    # Calculate coef: (nlev, ncell) -> (nlev, nedge)
    coef = bkd.minimum(
        r_in [bkd.arange(kds, kde+1, 1)[:, None], v_dowind],
        r_out[bkd.arange(kds, kde+1, 1)[:, None], v_upwind]
    )
    tmp = (
        qmfn_lo[kds:kde+1, ids:ide+1] +
        coef * (
            qmfn[kds:kde+1, ids:ide+1] - qmfn_lo[kds:kde+1, ids:ide+1]
        )
    ) # (nlev, nedge)
    # ========================================================
    qmfn = bkd.slice_update(qmfn, (slice(kds, kde+1), slice(ids, ide+1)), tmp)
    return fill_field2d_array_halo(
        halo=mesh.halo,
        f=qmfn,
        meta=batch.qmfn.meta
    )


@profile_scope("fct_calc_vflx")
def fct_calc_vflx(
    batch: batch.AdvBatchType,
    block: Block,
    q: FLT[Array, "nlev npoint"],
    q_old: FLT[Array, "nlev npoint"],
    qmfz: FLT[Array, "nlev npoint"],
    dt: float
) -> FLT[Array, "nlev+1 npoint"]:
    bkd          = block.backend
    mesh         = block.mesh
    ids, ide     = mesh.cell_ids, mesh.cell_ide
    dmg_old      = batch.m      .unwrap()
    dmg_new      = batch.m_new  .unwrap()
    mfz          = batch.mfz    .unwrap()
    qmfz_lo      = batch.qmfz_lo.unwrap()
    afz          = batch.afz    .unwrap()
    q_td         = batch.td     .unwrap()
    r_in         = batch.r_in   .unwrap()
    r_out        = batch.r_out  .unwrap()
    # ==========================================================
    # Calculate low-order and anti- diffusive fluxes
    kds, kde  = mesh.half_kds, mesh.half_kde
    # ------ Calculate low-order fluxes ------
    # from half_kds + 1 to half_kde -1
    mid_mfz   = mfz[kds+1:kde, ids:ide+1]
    direction = bkd.sign(mid_mfz)
    q_0       = q[kds:kde-1, ids:ide+1]
    q_plus_1  = q[kds+1:kde, ids:ide+1]
    qmfz_lo = bkd.slice_update(qmfz_lo, (slice(kds+1, kde), slice(ids, ide+1)),
        upwind1_vert(
            direction=direction,
            wgt=block.cfg.upwind_wgt,
            q_0=q_0,
            q_plus_1=q_plus_1
        ) * mid_mfz
    )
    # ------ Calculate anti-diffusive fluxes ------
    # ad_flux is at half level (half_kds to half_kde, ids to ide)
    afz = bkd.slice_update(afz, (slice(kds+1, kde), slice(ids, ide+1)),
        qmfz[kds+1:kde, ids:ide+1] - qmfz_lo[kds+1:kde, ids:ide+1]
    )
    # ========================================================
    # Calculate the updated low order solution
    kds, kde = mesh.full_kds, mesh.full_kde
    q_       = q      [kds:kde+1, ids:ide+1]
    q_old_   = q_old  [kds:kde+1, ids:ide+1]
    dmg_old_ = dmg_old[kds:kde+1, ids:ide+1]
    dmg_new_ = dmg_new[kds:kde+1, ids:ide+1]
    tmp = qmfz_lo[kds+1:kde+2, ids:ide+1] - qmfz_lo[kds:kde+1, ids:ide+1]
    q_td = bkd.slice_update(q_td, (slice(kds, kde+1), slice(ids, ide+1)),
        (q_old_ * dmg_old_ - dt * tmp) / dmg_new_  # (nlev, ncell)
    )
    # ========================================================
    # Calculate limiter R values
    q_td_ = q_td[kds:kde+1, ids:ide+1]
    # ------ Calculate qmax and qmin at half levels ------
    # At top (full_kds)
    qmax_top = bkd.maximum(
        bkd.maximum(q_[0, :], q_[1, :]),
        bkd.maximum(q_td_[0, :], q_td_[1, :])
    )
    qmin_top = bkd.minimum(
        bkd.minimum(q_[0, :], q_[1, :]),
        bkd.minimum(q_td_[0, :], q_td_[1, :])
    )
    # From full_kds + 1 to full_kde - 1
    qmax_mid = bkd.maximum(
        bkd.maximum(
            q_[1:-1, :],
            bkd.maximum(q_[:-2, :], q_[2:, :]),
        ),
        bkd.maximum(
            q_td_[1:-1, :],
            bkd.maximum(q_td_[:-2, :], q_td_[2:, :]),
        )
    )
    qmin_mid = bkd.minimum(
        bkd.minimum(
            q_[1:-1, :],
            bkd.minimum(q_[:-2, :], q_[2:, :]),
        ),
        bkd.minimum(
            q_td_[1:-1, :],
            bkd.minimum(q_td_[:-2, :], q_td_[2:, :]),
        )
    )
    # At bottom (full_kde)
    qmax_bot = bkd.maximum(
        bkd.maximum(q_[-1, :], q_[-2, :]),
        bkd.maximum(q_td_[-1, :], q_td_[-2, :])
    )
    qmin_bot = bkd.minimum(
        bkd.minimum(q_[-1, :], q_[-2, :]),
        bkd.minimum(q_td_[-1, :], q_td_[-2, :])
    )
    # Combine them together
    qmax = bkd.concatenate(
        [qmax_top[None, :],
         qmax_mid,
         qmax_bot[None, :]],
        axis=0
    )  # (nlev, ncell)
    qmin = bkd.concatenate(
        [qmin_top[None, :],
         qmin_mid,
         qmin_bot[None, :]],
        axis=0
    )  # (nlev, ncell)
    # ------ Calculate p_in and p_out ------
    zeros = bkd.array(0.0, dtype=q.dtype)
    p_in = (
        bkd.maximum(zeros, afz[kds:kde+1, ids:ide+1])
        - bkd.minimum(zeros, afz[kds+1:kde+2, ids:ide+1])
    )  # (nlev, ncell)
    p_out = (
        bkd.maximum(zeros, afz[kds+1:kde+2, ids:ide+1])
        - bkd.minimum(zeros, afz[kds:kde+1, ids:ide+1])
    )  # (nlev, ncell)
    # ------ Calculate m_in and m_out ------
    m_in = (qmax - q_td_) * dmg_new_ / dt   # (nlev, ncell)
    m_out = (q_td_ - qmin) * dmg_new_ / dt  # (nlev, ncell)
    # =============================================
    ones = bkd.array(1.0, dtype=q.dtype)
    if block.cfg.fct_regression_v:
        r_in = bkd.slice_update(r_in, (slice(kds, kde+1), slice(ids, ide+1)),
            bkd.where(
                p_in > 0,
                bkd.minimum(ones, m_in / p_in),
                zeros
            )  # (nlev, ncell)
        )
        r_out = bkd.slice_update(r_out, (slice(kds, kde+1), slice(ids, ide+1)),
            bkd.where(
                p_out > 0,
                bkd.minimum(ones, m_out / p_out),
                zeros
            )  # (nlev, ncell)
        )
    else:
        if q.dtype == bkd.float32:
            kesi = 1.e-40
        elif q.dtype == bkd.float64:
            kesi = 1.e-80
        else:
            raise ValueError("Unsupported data type for q.")
        r_in = bkd.slice_update(r_in, (slice(kds, kde+1), slice(ids, ide+1)),
            bkd.minimum(ones, m_in / (p_in + kesi))
        ) # (nlev, ncell)
        r_out = bkd.slice_update(r_out, (slice(kds, kde+1), slice(ids, ide+1)),
            bkd.minimum(ones, m_out / (p_out + kesi))
        ) # (nlev, ncell)
    # ========================================================
    # Calculate limited mass flux at half level
    kds, kde = mesh.half_kds, mesh.half_kde
    cell_range = bkd.arange(ids, ide+1)
    mark = bkd.astype(
        bkd.rint(-0.5 * bkd.sign(afz[kds+1:kde, ids:ide+1]) + 0.5),
        bkd.int32
    ).squeeze()  # 0 or 1 (nlev, ncell)
    ilev = bkd.arange(kds+1, kde)[:, None]
    v_upwind = ilev - (1 - mark)  # (nlev, ncell)
    v_dowind = ilev - mark        # (nlev, ncell)
    coef = bkd.minimum(
        r_in [v_dowind, cell_range],
        r_out[v_upwind, cell_range]
    )  # (nlev, ncell)
    qmfz_lo_ = qmfz_lo[kds+1:kde, ids:ide+1]
    qmfz_    = qmfz   [kds+1:kde, ids:ide+1]
    tmp = qmfz_lo_ + coef * (qmfz_ - qmfz_lo_)
    qmfz = bkd.slice_update(qmfz, (slice(kds+1, kde), slice(ids, ide+1)), tmp)
    return fill_field2d_array_halo(
        halo=mesh.halo,
        f=qmfz,
        meta=batch.qmfz.meta
    )
