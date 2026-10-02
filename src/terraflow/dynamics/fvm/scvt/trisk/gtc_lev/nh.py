from __future__ import annotations
from typing import TYPE_CHECKING


from terraflow.util.profile import profile_scope
from terraflow.util.math import tridiag_thomas
from terraflow.mesh.scvt import interp
from terraflow.mesh.scvt import operators as op
from terraflow.mesh.scvt.parallel import fill_field2d_array_halo
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv import interface
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv.batch import scvt_adv_fill_vhalo


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block
    from terraflow.mesh.scvt.field import SCVTFieldMeta

    FLT = typing.Float
    INT = typing.Int
    Array = typing.Array


@profile_scope("nh_solve")
def nh_solve(
    block: Block,
    old_idx: INT[Array, ""] | int,
    star_idx: INT[Array, ""] | int,
    new_idx: INT[Array, ""] | int,
    dt: float
) -> Block:
    block = interp_wind(block, star_idx, dt)

    w_lev     = block.state.w_lev    .unwrap()[star_idx, :, :]
    adv_w_lev = block.aux  .adv_w_lev.unwrap()
    block, w_lev, adv_w_lev = calc_adv_lev(
        block=block,
        q_lev=w_lev,
        q_meta=block.state.w_lev.meta,
        dqdt_lev=adv_w_lev,
        no_negvals=False,
        dt=dt
    )

    gz_lev     = block.state.gz_lev    .unwrap()[star_idx, :, :]
    adv_gz_lev = block.aux  .adv_gz_lev.unwrap()
    block, gz_lev, adv_gz_lev = calc_adv_lev(
        block=block,
        q_lev=gz_lev,
        q_meta=block.state.gz_lev.meta,
        dqdt_lev=adv_gz_lev,
        no_negvals=True,
        dt=dt
    )

    block = block.backend.tree_at(
        where=lambda b: (
            b.state.w_lev.d, b.state.gz_lev.d,
            b.aux.adv_w_lev.d, b.aux.adv_gz_lev.d
        ),
        pytree=block,
        replace=(w_lev, gz_lev, adv_w_lev, adv_gz_lev),
        at_dims=(0, 0, None, None),
        at_idxes=(star_idx, star_idx, None, None)
    )
    
    block = implicit_w_solver(block, old_idx, star_idx, new_idx, dt)

    block = calc_p(block, old_idx, new_idx, dt)

    bkd    = block.backend
    mesh   = block.mesh
    gz_lev = block.state.gz_lev.unwrap()[new_idx, :, :]
    gz     = block.state.gz    .unwrap()[new_idx, :, :]
    gz     = interp.scvt_average_2d(
        mesh=mesh,
        x=gz_lev,
        y=gz,
        x_loc=block.state.gz_lev.meta.loc,
        y_loc=block.state.gz.meta.loc
    )
    gz = fill_field2d_array_halo(
        halo=mesh.halo,
        f=gz,
        meta=block.state.gz.meta
    )
    return bkd.tree_at(
        where=lambda b: b.state.gz.d,
        pytree=block,
        replace=gz,
        at_dims=0,
        at_idxes=new_idx
    )


@profile_scope("interp_wind")
def interp_wind(
    block: Block,
    star_idx: INT[Array, ""] | int,
    dt: float
) -> Block:
    bkd     = block.backend
    mesh    = block.mesh
    mfn     = block.aux  .mfn    .unwrap()
    mfz_lev = block.aux  .mfz_lev.unwrap()
    dmg_lev = block.state.dmg_lev.unwrap()[star_idx, :, :]
    mfn_lev = block.aux  .mfn_lev.unwrap()
    un_lev  = block.aux  .un_lev .unwrap()
    mfz     = block.aux  .mfz    .unwrap()
    dmf_lev = block.aux  .dmf_lev.unwrap()

    mfn_lev = interp.scvt_interp_2d(
        mesh=mesh,
        x=mfn,
        y=mfn_lev,
        x_loc=block.aux.mfn.meta.loc,
        y_loc=block.aux.mfn_lev.meta.loc,
        extrap=True,
        extrap_type=2
    )
    mfn_lev = fill_field2d_array_halo(
        halo=mesh.halo,
        f=mfn_lev,
        meta=block.aux.mfn_lev.meta
    )
    un_lev = interp.scvt_interp_2d(
        mesh=mesh,
        x=dmg_lev,
        y=un_lev,
        x_loc=block.state.dmg_lev.meta.loc,
        y_loc=block.aux.un_lev.meta.loc
    )
    un_lev = bkd.slice_update(
        un_lev,
        (slice(mesh.half_kds, mesh.half_kde + 1), slice(mesh.edge_ids, mesh.edge_ide + 1)),
        mfn_lev[
            mesh.half_kds:mesh.half_kde + 1,
            mesh.edge_ids:mesh.edge_ide + 1
        ] / un_lev[
            mesh.half_kds:mesh.half_kde + 1,
            mesh.edge_ids:mesh.edge_ide + 1
        ]
    )
    un_lev = fill_field2d_array_halo(
        halo=mesh.halo,
        f=un_lev,
        meta=block.aux.un_lev.meta
    )
    mfz = interp.scvt_interp_2d(
        mesh=mesh,
        x=mfz_lev,
        y=mfz,
        x_loc=block.aux.mfz_lev.meta.loc,
        y_loc=block.aux.mfz.meta.loc
    )
    batch = block.adv_batch_nh.set_wind(
        un=un_lev,
        mfn=mfn_lev,
        mfz=mfz,
        m=dmg_lev,
        mesh=mesh
    )
    dmf_lev = op.div_operator_2d(
        mesh=mesh,
        f=mfn_lev,
        div=dmf_lev,
        div_loc=block.aux.dmf_lev.meta.loc
    )
    return bkd.tree_at(
        where=lambda b: (
            b.aux.mfn_lev.d, b.aux.un_lev.d, b.aux.mfz.d,
            b.aux.dmf_lev.d, b.adv_batch_nh
        ),
        pytree=block,
        replace=(mfn_lev, un_lev, mfz, dmf_lev, batch)
    )


@profile_scope("calc_adv_lev")
def calc_adv_lev(
    block: Block,
    q_lev: FLT[Array, "nlevp ncell"],
    q_meta: SCVTFieldMeta,
    dqdt_lev: FLT[Array, "nlevp ncell"],
    no_negvals: bool,
    dt: float
) -> tuple[Block, FLT[Array, "nlevp ncell"], FLT[Array, "nlevp ncell"]]:
    bkd   = block.backend
    mesh  = block.mesh
    batch = block.adv_batch_nh
    q_lev = fill_field2d_array_halo(
        halo=mesh.halo,
        f=q_lev,
        meta=q_meta
    )
    qmfn = interface.calc_tracer_hflx(
        batch=batch,
        block=block,
        q=q_lev,
        qmfn=batch.qmfn.unwrap(),
        dt=dt
    )
    dqdt_lev = op.div_operator_2d(
        mesh=mesh,
        f=qmfn,
        div=dqdt_lev,
        div_loc=q_meta.loc
    )

    ids, ide = mesh.cell_ids, mesh.cell_ide
    kds, kde = mesh.half_kds, mesh.half_kde
    dmf_lev  = block.aux.dmf_lev.unwrap()
    m        = batch.m  .unwrap()
    mfz      = batch.mfz.unwrap()
    dqdt_lev = bkd.slice_update(
        dqdt_lev,
        (slice(kds, kde + 1), slice(ids, ide + 1)),
        -dqdt_lev[kds:kde + 1, ids:ide + 1]
        + q_lev[kds:kde + 1, ids:ide + 1] * dmf_lev[kds:kde + 1, ids:ide + 1]
    )

    q_lev = scvt_adv_fill_vhalo(
        f=q_lev,
        meta=q_meta,
        mesh=mesh,
        no_negvals=no_negvals
    )
    qmfz = interface.calc_tracer_vflx(
        batch=batch,
        block=block,
        q=q_lev,
        qmfz=batch.qmfz.unwrap(),
        dt=dt
    )
    if kde - kds > 1:
        dqdt_lev = bkd.slice_update(
            dqdt_lev,
            (slice(kds + 1, kde), slice(ids, ide + 1)),
            dqdt_lev[kds + 1:kde, ids:ide + 1]
            - (
                qmfz[kds + 1:kde, ids:ide + 1]
                - qmfz[kds:kde - 1, ids:ide + 1]
                - q_lev[kds + 1:kde, ids:ide + 1] * (
                    mfz[kds + 1:kde, ids:ide + 1]
                    - mfz[kds:kde - 1, ids:ide + 1]
                )
            )
        )
    dqdt_lev = bkd.slice_update(
        dqdt_lev,
        (slice(kds, kde + 1), slice(ids, ide + 1)),
        dqdt_lev[kds:kde + 1, ids:ide + 1] / m[kds:kde + 1, ids:ide + 1]
    )
    batch = bkd.tree_at(
        where=lambda b: (b.qmfn.d, b.qmfz.d),
        pytree=batch,
        replace=(qmfn, qmfz)
    )
    block = bkd.tree_at(
        where=lambda b: b.adv_batch_nh,
        pytree=block,
        replace=batch
    )
    return block, q_lev, dqdt_lev


def apply_bc_w_lev(
    block: Block,
    new_idx: INT[Array, ""] | int
) -> Block:
    bkd      = block.backend
    mesh     = block.mesh
    ids, ide = mesh.cell_ids, mesh.cell_ide
    k = mesh.half_kde
    g = block.para.gravity.unwrap()
    adv_gz_lev = block.aux  .adv_gz_lev.unwrap()
    w_lev      = block.state.w_lev     .unwrap()[new_idx, :, :]
    # ==================================================
    w_lev = bkd.slice_update(
        w_lev,
        (k, slice(ids, ide + 1)),
        -adv_gz_lev[k, ids:ide + 1] / g
    )
     # ==================================================
    return bkd.tree_at(
        where=lambda b: b.state.w_lev.d,
        pytree=block,
        replace=w_lev,
        at_dims=0,
        at_idxes=new_idx
    )


@profile_scope("implicit_w_solver")
def implicit_w_solver(
    block: Block,
    old_idx: INT[Array, ""] | int,
    star_idx: INT[Array, ""] | int,
    new_idx: INT[Array, ""] | int,
    dt: float
) -> Block:
    mesh = block.mesh
    if mesh.half_kde - mesh.half_kds <= 1:
        return block

    block    = apply_bc_w_lev(block, new_idx)
    bkd      = block.backend
    ids, ide = mesh.cell_ids, mesh.cell_ide
    kds, kde = mesh.half_kds, mesh.half_kde
    iks, ike = kds + 1, kde - 1

    beta         = block.cfg .implicit_w_wgt
    g            = block.para.gravity  .unwrap()
    gamma        = block.para.cpd_o_cvd.unwrap()
    gdtbeta      = g * dt * beta
    gdt1mbeta    = g * dt * (1.0 - beta)
    gdtbeta2gam  = (g * dt * beta) ** 2 * gamma

    adv_gz_lev   = block.aux    .adv_gz_lev.unwrap()
    adv_w_lev    = block.aux    .adv_w_lev .unwrap()
    old_p        = block.state  .p         .unwrap()[old_idx, :, :]
    star_p       = block.state  .p         .unwrap()[star_idx, :, :]
    old_w_lev    = block.state  .w_lev     .unwrap()[old_idx, :, :]
    star_w_lev   = block.state  .w_lev     .unwrap()[star_idx, :, :]
    new_w_lev    = block.state  .w_lev     .unwrap()[new_idx, :, :]
    star_dmg_lev = block.state  .dmg_lev   .unwrap()[star_idx, :, :]
    new_dmg_lev  = block.state  .dmg_lev   .unwrap()[new_idx, :, :]
    old_gz_lev   = block.state  .gz_lev    .unwrap()[old_idx, :, :]
    new_gz_lev   = block.state  .gz_lev    .unwrap()[new_idx, :, :]
    old_dmg      = block.state  .dmg       .unwrap()[old_idx, :, :]
    new_dmg      = block.state  .dmg       .unwrap()[new_idx, :, :]
    old_pt       = block.state  .pt        .unwrap()[old_idx, :, :]
    new_pt       = block.state  .pt        .unwrap()[new_idx, :, :]
    qm_lev       = block.tracers.qm_lev    .unwrap()

    gz1 = old_gz_lev + bkd.zeros_like(old_gz_lev)
    gz1 = bkd.slice_update(
        gz1,
        (slice(kds, kde), slice(ids, ide + 1)),
        old_gz_lev[kds:kde, ids:ide + 1]
        + dt * adv_gz_lev[kds:kde, ids:ide + 1]
        + gdt1mbeta * star_w_lev[kds:kde, ids:ide + 1]
    )
    gz1 = bkd.slice_update(
        gz1,
        (kde, slice(ids, ide + 1)),
        old_gz_lev[kde, ids:ide + 1]
    )

    w1 = (
        old_w_lev[iks:ike + 1, ids:ide + 1]
        + dt * adv_w_lev[iks:ike + 1, ids:ide + 1]
        - g * dt
        + gdt1mbeta * (
            star_p[iks:ike + 1, ids:ide + 1]
            - star_p[iks - 1:ike, ids:ide + 1]
        ) / star_dmg_lev[iks:ike + 1, ids:ide + 1]
        / (1.0 + qm_lev[iks:ike + 1, ids:ide + 1])
    )

    dp1 = (
        old_p[iks:ike + 1, ids:ide + 1]
        - old_p[iks - 1:ike, ids:ide + 1]
    ) + gamma * (
        (
            old_p[iks:ike + 1, ids:ide + 1]
            * new_dmg[iks:ike + 1, ids:ide + 1]
            * new_pt[iks:ike + 1, ids:ide + 1]
            / old_dmg[iks:ike + 1, ids:ide + 1]
            / old_pt[iks:ike + 1, ids:ide + 1]
            - old_p[iks - 1:ike, ids:ide + 1]
            * new_dmg[iks - 1:ike, ids:ide + 1]
            * new_pt[iks - 1:ike, ids:ide + 1]
            / old_dmg[iks - 1:ike, ids:ide + 1]
            / old_pt[iks - 1:ike, ids:ide + 1]
        ) - (
            old_p[iks:ike + 1, ids:ide + 1]
            * (
                gz1[iks + 1:ike + 2, ids:ide + 1]
                - gz1[iks:ike + 1, ids:ide + 1]
            )
            / (
                old_gz_lev[iks + 1:ike + 2, ids:ide + 1]
                - old_gz_lev[iks:ike + 1, ids:ide + 1]
            )
            - old_p[iks - 1:ike, ids:ide + 1]
            * (
                gz1[iks:ike + 1, ids:ide + 1]
                - gz1[iks - 1:ike, ids:ide + 1]
            )
            / (
                old_gz_lev[iks:ike + 1, ids:ide + 1]
                - old_gz_lev[iks - 1:ike, ids:ide + 1]
            )
        )
    )
    w1 = (
        w1
        + gdtbeta * dp1 / new_dmg_lev[iks:ike + 1, ids:ide + 1]
        / (1.0 + qm_lev[iks:ike + 1, ids:ide + 1])
    )

    a = (
        gdtbeta2gam
        * old_p[iks - 1:ike, ids:ide + 1]
        / (
            old_gz_lev[iks:ike + 1, ids:ide + 1]
            - old_gz_lev[iks - 1:ike, ids:ide + 1]
        )
        / (1.0 + qm_lev[iks:ike + 1, ids:ide + 1])
    )
    c = (
        gdtbeta2gam
        * old_p[iks:ike + 1, ids:ide + 1]
        / (
            old_gz_lev[iks + 1:ike + 2, ids:ide + 1]
            - old_gz_lev[iks:ike + 1, ids:ide + 1]
        )
        / (1.0 + qm_lev[iks:ike + 1, ids:ide + 1])
    )
    b = new_dmg_lev[iks:ike + 1, ids:ide + 1] - a - c
    d = new_dmg_lev[iks:ike + 1, ids:ide + 1] * w1
    a = bkd.slice_update(a, (0, slice(None)), bkd.zeros_like(a[0, :]))
    c = bkd.slice_update(c, (-1, slice(None)), bkd.zeros_like(c[-1, :]))
    new_w_inner = tridiag_thomas(bkd, a, b, c, d)
    new_w_lev = bkd.slice_update(
        new_w_lev,
        (slice(iks, ike + 1), slice(ids, ide + 1)),
        new_w_inner
    )

    if block.cfg.use_rayleigh_damp_w:
        new_w_lev = rayleigh_damp_w(
            block=block,
            dt=dt,
            gz=block.state.gz_lev.unwrap()[star_idx, :, :],
            w=new_w_lev
        )
    if block.cfg.use_cfl_damp:
        new_w_lev = cfl_damp_w(
            block=block,
            dt=dt,
            mfz=block.aux.mfz_lev.unwrap(),
            dmg=star_dmg_lev,
            w=new_w_lev
        )

    new_gz_lev = bkd.slice_update(
        new_gz_lev,
        (slice(kds, kde), slice(ids, ide + 1)),
        gz1[kds:kde, ids:ide + 1]
        + gdtbeta * new_w_lev[kds:kde, ids:ide + 1]
    )
    # new_gz_lev = bkd.slice_update(
    #     new_gz_lev,
    #     (kde, slice(ids, ide + 1)),
    #     old_gz_lev[kde, ids:ide + 1]
    # )
    new_w_lev = fill_field2d_array_halo(
        halo=mesh.halo,
        f=new_w_lev,
        meta=block.state.w_lev.meta
    )
    new_gz_lev = fill_field2d_array_halo(
        halo=mesh.halo,
        f=new_gz_lev,
        meta=block.state.gz_lev.meta
    )
    return bkd.tree_at(
        where=lambda b: (b.state.w_lev.d, b.state.gz_lev.d),
        pytree=block,
        replace=(new_w_lev, new_gz_lev),
        at_dims=(0, 0),
        at_idxes=(new_idx, new_idx)
    )


def rayleigh_damp_w(
    block: Block,
    dt: float,
    gz: FLT[Array, "nlevp ncell"],
    w: FLT[Array, "nlevp ncell"]
) -> FLT[Array, "nlevp ncell"]:
    bkd = block.backend
    mesh = block.mesh
    ids, ide = mesh.cell_ids, mesh.cell_ide
    kds, kde = mesh.half_kds, mesh.half_kde
    if kde - kds <= 1:
        return w

    gzd = block.cfg.rayleigh_damp_top * block.para.gravity.unwrap()
    gz_top = gz[kds, ids:ide + 1]
    gz_mid = gz[kds + 1:kde, ids:ide + 1]
    mask = gz_mid > gz_top[None, :] - gzd
    coef = block.cfg.rayleigh_damp_w_coef * bkd.sin(
        0.5 * bkd.pi * (1.0 - (gz_top[None, :] - gz_mid) / gzd)
    ) ** 2
    w_mid = bkd.where(
        mask,
        w[kds + 1:kde, ids:ide + 1] / (1.0 + coef * dt),
        w[kds + 1:kde, ids:ide + 1]
    )
    return bkd.slice_update(
        w,
        (slice(kds + 1, kde), slice(ids, ide + 1)),
        w_mid
    )


def cfl_damp_w(
    block: Block,
    dt: float,
    mfz: FLT[Array, "nlevp ncell"],
    dmg: FLT[Array, "nlevp ncell"],
    w: FLT[Array, "nlevp ncell"]
) -> FLT[Array, "nlevp ncell"]:
    bkd = block.backend
    mesh = block.mesh
    ids, ide = mesh.cell_ids, mesh.cell_ide
    kds, kde = mesh.half_kds, mesh.half_kde
    if kde - kds <= 1:
        return w

    cr_num = mfz[kds + 1:kde, ids:ide + 1] * dt / dmg[kds + 1:kde, ids:ide + 1]
    cr_act = bkd.maximum(bkd.zeros_like(cr_num), bkd.abs(cr_num) - 1.0)
    cr_act = cr_act * bkd.where(w[kds + 1:kde, ids:ide + 1] >= 0, 1.0, -1.0)
    w_mid = w[kds + 1:kde, ids:ide + 1] - 0.3 * cr_act * dt
    return bkd.slice_update(
        w,
        (slice(kds + 1, kde), slice(ids, ide + 1)),
        w_mid
    )


@profile_scope("calc_p")
def calc_p(
    block: Block,
    old_idx: INT[Array, ""] | int,
    new_idx: INT[Array, ""] | int,
    dt: float
) -> Block:
    bkd      = block.backend
    mesh     = block.mesh
    ids, ihe = mesh.cell_ids, mesh.cell_ihe[0]
    kds, kde = mesh.full_kds, mesh.full_kde
    hks, hke = mesh.half_kds, mesh.half_kde
    gamma      = block.para .cpd_o_cvd.unwrap()
    old_p      = block.state.p        .unwrap()[old_idx, :, :]
    new_p      = block.state.p        .unwrap()[new_idx, :, :]
    old_pt     = block.state.pt       .unwrap()[old_idx, :, :]
    new_pt     = block.state.pt       .unwrap()[new_idx, :, :]
    old_dmg    = block.state.dmg      .unwrap()[old_idx, :, :]
    new_dmg    = block.state.dmg      .unwrap()[new_idx, :, :]
    old_gz_lev = block.state.gz_lev   .unwrap()[old_idx, :, :]
    new_gz_lev = block.state.gz_lev   .unwrap()[new_idx, :, :]
    new_p_lev  = block.state.p_lev    .unwrap()[new_idx, :, :]

    new_p_calc = old_p[kds:kde + 1, ids:ihe + 1] + gamma * old_p[
        kds:kde + 1, ids:ihe + 1
    ] * (
        (
            new_dmg[kds:kde + 1, ids:ihe + 1]
            * new_pt[kds:kde + 1, ids:ihe + 1]
        ) / (
            old_dmg[kds:kde + 1, ids:ihe + 1]
            * old_pt[kds:kde + 1, ids:ihe + 1]
        ) - (
            new_gz_lev[kds:kde + 1, ids:ihe + 1]
            - new_gz_lev[kds + 1:kde + 2, ids:ihe + 1]
        ) / (
            old_gz_lev[kds:kde + 1, ids:ihe + 1]
            - old_gz_lev[kds + 1:kde + 2, ids:ihe + 1]
        )
    )
    if block.cfg.use_p_damp:
        new_p_calc = (
            new_p_calc
            - block.cfg.p_damp_coef * (
                new_p_calc - old_p[kds:kde + 1, ids:ihe + 1]
            )
        )
    new_p = bkd.slice_update(
        new_p,
        (slice(kds, kde + 1), slice(ids, ihe + 1)),
        new_p_calc
    )
    new_p_lev = interp.scvt_interp_2d(
        mesh=mesh,
        x=new_p,
        y=new_p_lev,
        x_loc=block.state.p.meta.loc,
        y_loc=block.state.p_lev.meta.loc,
        lagrange=True,
        lagrange_order=5
    )
    new_p_lev = bkd.slice_update(
        new_p_lev,
        (hks, slice(ids, mesh.cell_ide + 1)),
        new_p[kds, ids:mesh.cell_ide + 1]
        - 0.5 * new_dmg[kds, ids:mesh.cell_ide + 1]
    )
    new_p_lev = bkd.slice_update(
        new_p_lev,
        (hke, slice(ids, mesh.cell_ide + 1)),
        new_p[kde, ids:mesh.cell_ide + 1]
        + 0.5 * new_dmg[kde, ids:mesh.cell_ide + 1]
    )
    new_p_lev = fill_field2d_array_halo(
        halo=mesh.halo,
        f=new_p_lev,
        meta=block.state.p_lev.meta
    )
    return bkd.tree_at(
        where=lambda b: (b.state.p.d, b.state.p_lev.d),
        pytree=block,
        replace=(new_p, new_p_lev),
        at_dims=(0, 0),
        at_idxes=(new_idx, new_idx)
    )
