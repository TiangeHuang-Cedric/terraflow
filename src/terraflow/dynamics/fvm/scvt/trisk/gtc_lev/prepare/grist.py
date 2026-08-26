from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp

from terraflow.util import typing, log, sphere_geometry, formula
from terraflow.util.pytree import tree_at
from terraflow.mesh.scvt.parallel import (
    fill_field1d_array_halo,
    fill_field2d_array_halo,
    fill_field3d_array_halo
)
from terraflow.prepare.vert_interp import (
    VerticalLinearInterpolator,
    grist_vert_interp_linear
)
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.tracer import tracer_calc_qm
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.space_operators import (
    calc_mg,
    calc_dmg
)
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.prepare import ref
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.prepare.topo_smooth import topo_smooth, calc_zs_slope

if TYPE_CHECKING:
    from terraflow.prepare.readers.grist import GRISTStaticDataReader, GRISTBackgroundReader
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block as BLOCK


Int = typing.Int
Float = typing.Float
Array = typing.Array


def prepare_static(
    block: BLOCK,
    reader: GRISTStaticDataReader,
) -> BLOCK:
    # =====================================
    gzs = fill_field1d_array_halo(
        halo=block.mesh.halo,
        f=reader.phis,
        meta=block.static.gzs.meta
    )
    landmask = fill_field1d_array_halo(
        halo=block.mesh.halo,
        f=reader.mask,
        meta=block.static.landmask.meta
    )
    block = tree_at(
        where=lambda b: (
            b.static.gzs.d,
            b.static.landmask.d,
        ),
        pytree=block,
        replace=(gzs, landmask)
    )
    # =====================================
    gzs = block.static.gzs.unwrap()
    dzsdn = block.static.dzsdn.unwrap()
    dzsdn = calc_zs_slope(
        block,
        gzs,
        dzsdn
    )
    block = tree_at(
        where=lambda b: b.static.dzsdn.d,
        pytree=block,
        replace=dzsdn
    )
    # =====================================
    if reader.cfg.use_zs_grad_filter:
        block = topo_smooth(block=block, reader=reader)
    block = ref.ref_calc_ps(block=block)
    return block


def prepare_bkg(
    block: BLOCK,
    reader: GRISTBackgroundReader,
) -> BLOCK:
    if reader.cfg.vert_interp_type == "grist":
        allow_extrap_opt = False
    elif reader.cfg.vert_interp_type == "native":
        allow_extrap_opt = True
    else:
        raise ValueError(
            f"Unsupported vertical interpolation type: {reader.cfg.vert_interp_type}"
        )
    vert_linear_interp = VerticalLinearInterpolator(
        backend=reader.cfg.vert_interp_backend,
        allow_extrap=allow_extrap_opt
    )
    if reader.cfg.vert_interp_type == "grist":
        block = prepare_phs(block=block, reader=reader)
        block = bkg_calc_ph(block=block)
        block = regrid_wet_qv(
            block,
            reader=reader,
            vert_interp_linear=vert_linear_interp,
            mute=False
        )
        block = tracer_calc_qm(block, "old")
        block = bkg_calc_dry_qv(block=block, mute=False)
        block = tracer_calc_qm(block, "old")
        block = regrid_un_from_uv(
            block=block,
            reader=reader,
            vert_interp_linear=vert_linear_interp
        )
        block = regrid_t(
            block=block,
            reader=reader,
            vert_interp_linear=vert_linear_interp
        )
        block = bkg_calc_mgs(block=block, reader=reader, mute=False)
        block = bkg_calc_mg(block=block, mute=False)
        block = calc_ph(block=block)
        block = bkg_calc_tv(block=block)
        block = bkg_calc_pt(block=block)
    elif reader.cfg.vert_interp_type == "native":
        # Stage 1:
        block = prepare_phs(block=block, reader=reader)
        block = bkg_calc_ph(block=block)
        for cyc in range(3):
            block = regrid_wet_qv(
                block,
                reader=reader,
                vert_interp_linear=vert_linear_interp,
                mute=cyc != 0,
            )
            block = tracer_calc_qm(block, "old")
            # Stage 2:
            block = bkg_calc_mgs(block=block, reader=reader, mute=cyc != 0)
            block = bkg_calc_mg(block=block, mute=cyc != 0)
            block = bkg_calc_dry_qv(block=block, mute=cyc != 0)
            block = tracer_calc_qm(block, "old")
            block = calc_ph(block=block)
        # Stage 3:
        block = regrid_un_from_uv(
            block=block,
            reader=reader,
            vert_interp_linear=vert_linear_interp
        )
        block = regrid_t(
            block=block,
            reader=reader,
            vert_interp_linear=vert_linear_interp
        )
        block = bkg_calc_tv(block=block)
        block = bkg_calc_pt(block=block)
    else:
        raise ValueError(
            f"Unsupported vertical interpolation type: {reader.cfg.vert_interp_type}"
        )
    return block


def prepare_phs(
    block: BLOCK,
    reader: GRISTBackgroundReader,
) -> BLOCK:
    if reader.cfg.vert_interp_type == "grist":
        plev_lev = reader.plev_lev
        ncol_range = jnp.arange(plev_lev.shape[1])
        phs = plev_lev[reader.nlev_eff, ncol_range]
    elif reader.cfg.vert_interp_type == "native":
        phs = reader.ps
    else:
        raise ValueError(
            f"Unsupported vertical interpolation type: {reader.cfg.vert_interp_type}"
        )
    phs = fill_field1d_array_halo(
        halo=block.mesh.halo,
        f=phs,
        meta=block.state.phs.meta
    )
    return tree_at(
        where=lambda b: b.state.phs.d,
        pytree=block,
        replace=phs,
        at_dims=0,
        at_idxes=0
    )


def bkg_calc_ph(block: BLOCK) -> BLOCK:
    log.logger.info("Calculate hydrostatic pressure on each level.")
    phs = block.state.phs.unwrap()[0, ...]
    mesh = block.mesh
    full_kds = mesh.full_kds
    full_kde = mesh.full_kde
    half_kds = mesh.half_kds
    half_kde = mesh.half_kde
    # =================================================================
    ph = block.vert_coord.calc_mg(mgs=phs, ks=full_kds, ke=full_kde)
    ph_lev = block.vert_coord.calc_mg_lev(mgs=phs, ks=half_kds, ke=half_kde)
    ph = fill_field2d_array_halo(
        halo=mesh.halo,
        f=ph,
        meta=block.state.ph.meta
    )
    ph_lev = fill_field2d_array_halo(
        halo=mesh.halo,
        f=ph_lev,
        meta=block.state.ph_lev.meta
    )
    # =================================================================
    if block.mode.is_hydrostatic():
        return tree_at(
            where=lambda b: (
                b.state.ph.d, b.state.ph_lev.d,
                b.state.p.d, b.state.p_lev.d,
                b.state.ps.d
            ),
            pytree=block,
            replace=(ph, ph_lev, ph, ph_lev, phs),
            at_dims=(0, 0, 0, 0, 0),
            at_idxes=(0, 0, 0, 0, 0)
        )
    else:
        return tree_at(
            where=lambda b: (b.state.ph.d, b.state.ph_lev.d),
            pytree=block,
            replace=(ph, ph_lev),
            at_dims=(0, 0),
            at_idxes=(0, 0)
        )


def calc_ph(block: BLOCK) -> BLOCK:
    mesh = block.mesh
    ids, ide = mesh.cell_ids, mesh.cell_ide
    full_kds, full_kde = mesh.full_kds, mesh.full_kde
    half_kds, half_kde = mesh.half_kds, mesh.half_kde
    mg_lev = block.state.mg_lev.unwrap()[0, :, :]
    dmg = block.state.dmg.unwrap()[0, :, :]
    ph_lev = block.state.ph_lev.unwrap()[0, :, :]
    ph = block.state.ph.unwrap()[0, :, :]
    phs = block.state.phs.unwrap()[0, :]
    # =======================================================
    # Calculate ph at half level
    kds, kde = half_kds, half_kde
    if block.mode.is_moist():
        qm = block.tracers.qm.unwrap()
        dmg_ = dmg[kds:kde, ids:ide+1]
        qm_ = qm[kds:kde, ids:ide+1]
        moist_dmg = dmg_ * (1.0 + qm_)
        ph_lev = ph_lev.at[kds:kde+1, ids:ide+1].set(
            mg_lev[kds, ids:ide+1] + jnp.cumulative_sum(
                moist_dmg,
                axis=0,
                include_initial=True
            )
        )
    else:
        ph_lev = ph_lev.at[kds:kde+1, ids:ide+1].set(
            mg_lev[kds, ids:ide+1] + jnp.cumulative_sum(
                dmg[kds:kde, ids:ide+1],
                axis=0,
                include_initial=True
            )
        )
    # =======================================================
    # Calculate ph at full level
    kds, kde = full_kds, full_kde
    ph = ph.at[kds:kde+1, ids:ide+1].set(
        0.5 * (
            ph_lev[kds:kde+1, ids:ide+1]
            + ph_lev[kds+1:kde+2, ids:ide+1]
        )
    )
    # =======================================================
    # Calculate phs
    kde = half_kde
    phs = phs.at[ids:ide+1].set(ph_lev[kde, ids:ide+1])
    phs = fill_field1d_array_halo(
        halo=mesh.halo,
        f=phs,
        meta=block.state.phs.meta
    )
    ph = fill_field2d_array_halo(
        halo=mesh.halo,
        f=ph,
        meta=block.state.ph.meta
    )
    ph_lev = fill_field2d_array_halo(
        halo=mesh.halo,
        f=ph_lev,
        meta=block.state.ph_lev.meta
    )
    # =======================================================
    if block.mode.is_hydrostatic():
        return tree_at(
            where=lambda b: (
                b.state.phs.d, b.state.ph.d, b.state.ph_lev.d,
                b.state.ps.d, b.state.p.d, b.state.p_lev.d
            ),
            pytree=block,
            replace=(phs, ph, ph_lev, phs, ph, ph_lev),
            at_dims=(0, 0, 0, 0, 0, 0),
            at_idxes=(0, 0, 0, 0, 0, 0)
        )
    else:
        return tree_at(
            where=lambda b: (b.state.phs.d, b.state.ph.d, b.state.ph_lev.d),
            pytree=block,
            replace=(phs, ph, ph_lev),
            at_dims=(0, 0, 0),
            at_idxes=(0, 0, 0)
        )


def regrid_un_from_uv(
    block: BLOCK,
    reader: GRISTBackgroundReader,
    vert_interp_linear: VerticalLinearInterpolator
) -> BLOCK:
    radius = block.para.radius.unwrap()
    mesh = block.mesh
    ids, ide = mesh.edge_ids, mesh.edge_ide
    un = block.state.un.unwrap()[0, :, :]
    # =================================================================
    u = _regrid_u(
        block=block,
        reader=reader,
        vert_interp_linear=vert_interp_linear
    )[:, ids:ide+1]
    v = _regrid_v(
        block=block,
        reader=reader,
        vert_interp_linear=vert_interp_linear
    )[:, ids:ide+1]
    vec = sphere_geometry.vector_convert_lonlat2xyz(
        R=radius,
        u=u, 
        v=v,
        xyz=mesh.dual_edge_xyz[ids:ide+1, :]
    )
    un_ = jnp.sum(vec * mesh.edge_nr[ids:ide+1, :], axis=-1)
    un = un.at[:, ids:ide+1].set(un_)
    un = fill_field2d_array_halo(
        halo=mesh.halo,
        f=un,
        meta=block.state.un.meta
    )
    # =================================================================
    return tree_at(
        where=lambda b: b.state.un.d,
        pytree=block,
        replace=un,
        at_dims=0,
        at_idxes=0
    )


def _regrid_u(
    block: BLOCK,
    reader: GRISTBackgroundReader,
    vert_interp_linear: VerticalLinearInterpolator
) -> Float[Array, "nlev nedge"]:
    log.logger.info("Regrid u wind component.")
    mesh = block.mesh
    ids, ide = mesh.edge_ids, mesh.edge_ide
    cell_0 = mesh.cell_of_edge[ids:ide+1, 0]
    cell_1 = mesh.cell_of_edge[ids:ide+1, 1]
    ph = block.state.ph.unwrap()[0, ...]
    ph = fill_field2d_array_halo(
        halo=mesh.halo,
        f=ph,
        meta=block.state.ph.meta
    )
    # =================================================================
    if reader.cfg.vert_interp_type == "grist":
        u_cell = grist_vert_interp_linear(
            vert_interp_linear=vert_interp_linear,
            nlev_eff=reader.nlev_eff,
            x1=reader.plev,
            data1=reader.u,
            x2=ph,
        )
    elif reader.cfg.vert_interp_type == "native":
        u_cell = vert_interp_linear(
            x1=reader.plev,
            data1=reader.u,
            x2=ph,
        )
    else:
        raise ValueError(
            f"Unsupported vertical interpolation type: {reader.cfg.vert_interp_type}"
        )
    u_cell = fill_field2d_array_halo(
        halo=mesh.halo,
        f=u_cell,
        meta=block.state.ph.meta
    )
    u = block.state.un.unwrap()[0, :, :]
    u = u.at[:, ids:ide+1].set(
        0.5 * (u_cell[:, cell_0] + u_cell[:, cell_1])
    )
    u = fill_field2d_array_halo(
        halo=mesh.halo,
        f=u,
        meta=block.state.un.meta
    )
    # =================================================================
    return u


def _regrid_v(
    block: BLOCK,
    reader: GRISTBackgroundReader,
    vert_interp_linear: VerticalLinearInterpolator
) -> Float[Array, "nlev nedge"]:
    log.logger.info("Regrid v wind component.")
    mesh = block.mesh
    ids, ide = mesh.edge_ids, mesh.edge_ide
    cell_0 = mesh.cell_of_edge[ids:ide+1, 0]
    cell_1 = mesh.cell_of_edge[ids:ide+1, 1]
    ph = block.state.ph.unwrap()[0, ...]
    ph = fill_field2d_array_halo(
        halo=mesh.halo,
        f=ph,
        meta=block.state.ph.meta
    )
    # =================================================================
    if reader.cfg.vert_interp_type == "grist":
        v_cell = grist_vert_interp_linear(
            vert_interp_linear=vert_interp_linear,
            nlev_eff=reader.nlev_eff,
            x1=reader.plev,
            data1=reader.v,
            x2=ph,
        )
    elif reader.cfg.vert_interp_type == "native":
        v_cell = vert_interp_linear(
            x1=reader.plev,
            data1=reader.v,
            x2=ph,
        )
    else:
        raise ValueError(
            f"Unsupported vertical interpolation type: {reader.cfg.vert_interp_type}"
        )
    v_cell = fill_field2d_array_halo(
        halo=mesh.halo,
        f=v_cell,
        meta=block.state.ph.meta
    )
    v = block.state.un.unwrap()[0, :, :]
    v = v.at[:, ids:ide+1].set(
        0.5 * (v_cell[:, cell_0] + v_cell[:, cell_1])
    )
    v = fill_field2d_array_halo(
        halo=mesh.halo,
        f=v,
        meta=block.state.un.meta
    )
    # =================================================================
    return v


def regrid_t(
    block: BLOCK,
    reader: GRISTBackgroundReader,
    vert_interp_linear: VerticalLinearInterpolator
) -> BLOCK:
    log.logger.info("Regrid temperature.")
    ph = block.state.ph.unwrap()[0, ...]
    # =================================================================
    if reader.cfg.vert_interp_type == "grist":
        t = grist_vert_interp_linear(
            vert_interp_linear=vert_interp_linear,
            nlev_eff=reader.nlev_eff,
            x1=reader.plev,
            data1=reader.t,
            x2=ph,
        )
    elif reader.cfg.vert_interp_type == "native":
        t = vert_interp_linear(
            x1=reader.plev,
            data1=reader.t,
            x2=ph,
        )
    else:
        raise ValueError(
            f"Unsupported vertical interpolation type: {reader.cfg.vert_interp_type}"
        )
    t = fill_field2d_array_halo(
        halo=block.mesh.halo,
        f=t,
        meta=block.aux.t.meta
    )
    # =================================================================
    return tree_at(
        where=lambda b: b.aux.t.d,
        pytree=block,
        replace=t,
    )


def regrid_wet_qv(
    block: BLOCK,
    reader: GRISTBackgroundReader,
    vert_interp_linear: VerticalLinearInterpolator,
    mute: bool = False
) -> BLOCK:
    idx_qv = block.tracers.idx_qv
    if idx_qv == -1:
        return block

    if not mute:
        log.logger.info("Regrid water vapor wet mixing ratio.")
    # =================================================================
    ph = block.state.ph.unwrap()[0, ...]
    q = block.tracers.q.unwrap()[0, ...]
    # =================================================================
    if reader.cfg.vert_interp_type == "grist":
        qv = grist_vert_interp_linear(
            vert_interp_linear=vert_interp_linear,
            nlev_eff=reader.nlev_eff,
            x1=reader.plev,
            data1=reader.q,
            x2=ph,
        )
    elif reader.cfg.vert_interp_type == "native":
        qv = vert_interp_linear(
            x1=reader.plev,
            data1=reader.q,
            x2=ph,
        )
    else:
        raise ValueError(
            f"Unsupported vertical interpolation type: {reader.cfg.vert_interp_type}"
        )
    qv = jnp.where(qv < 0.0, 0.0, qv)
    q = q.at[idx_qv].set(qv)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_qv,
        meta=block.tracers.q.meta
    )
    # =================================================================
    return tree_at(
        where=lambda b: b.tracers.q.d,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def bkg_calc_mgs(
    block: BLOCK,
    reader: GRISTBackgroundReader,
    mute: bool = False
) -> BLOCK:
    if not mute:
        log.logger.info("Calculate surface dry-air pressure.")
    mesh = block.mesh
    ids, ide = mesh.cell_ids, mesh.cell_ide
    phs = block.state.phs.unwrap()[0, ...]
    mgs = block.state.mgs.unwrap()[0, :]
    if reader.cfg.vert_interp_type == "grist":
        mgs_ = phs[ids:ide+1]
    elif reader.cfg.vert_interp_type == "native":
        kds, kde = mesh.full_kds, mesh.full_kde
        ph_lev = block.state.ph_lev.unwrap()[0, ...]
        qm = block.tracers.qm.unwrap()[:, :]
        # =================================================================
        delta_ph = ph_lev[kds+1:kde+2, ids:ide+1] - ph_lev[kds:kde+1, ids:ide+1]
        mgs_ = phs[ids:ide+1] - jnp.sum(
            delta_ph * qm[kds:kde+1, ids:ide+1],
            axis=0
        )
    else:
        raise ValueError(
            f"Unsupported vertical interpolation type: {reader.cfg.vert_interp_type}"
        )
    mgs = mgs.at[ids:ide+1].set(mgs_)
    mgs = fill_field1d_array_halo(
        halo=mesh.halo,
        f=mgs,
        meta=block.state.mgs.meta
    )
    # =================================================================
    return tree_at(
        where=lambda b: b.state.mgs.d,
        pytree=block,
        replace=mgs,
        at_dims=0,
        at_idxes=0
    )


def bkg_calc_mg(block: BLOCK, mute: bool = False) -> BLOCK:
    if not mute:
        log.logger.info("Calculate dry-air pressure.")
    # =================================================================
    block = calc_mg (block, 0)
    block = calc_dmg(block, 0)
    # ================================================================= 
    return block


def bkg_calc_tv(block: BLOCK) -> BLOCK:
    log.logger.info("Calculate virtual temperature.")
    idx_qv = block.tracers.idx_qv
    t = block.aux.t.unwrap()
    # =================================================================
    if idx_qv == -1:
        tv = t
    else:
        qv = block.tracers.q.unwrap()[0, idx_qv, :, :]
        qm = block.tracers.qm.unwrap()[:, :]
        tv = formula.virtual_temperature_from_dry_mixing_ratio(
            t=t,
            qv=qv,
            qm=qm,
            para=block.para
        )
    tv = fill_field2d_array_halo(
        halo=block.mesh.halo,
        f=tv,
        meta=block.aux.tv.meta
    )
    return tree_at(
        where=lambda b: b.aux.tv.d,
        pytree=block,
        replace=tv,
    )


def bkg_calc_pt(block: BLOCK) -> BLOCK:
    log.logger.info("Calculate modified potential temperature.")
    ph = block.state.ph.unwrap()[0, ...]
    t = block.aux.t.unwrap()
    idx_qv = block.tracers.idx_qv
    # =================================================================
    if idx_qv == -1:
        pt = formula.modified_potential_temperature(
            t=t,
            p=ph,
            qv=jnp.array(
                0.0,
                device=block.dev(),
                dtype=block.dev.float_type
            ),
            para=block.para
        )
    else:
        q = block.tracers.q.unwrap()[0, :, :, :]
        pt = formula.modified_potential_temperature(
            t=t,
            p=ph,
            qv=q[idx_qv, :, :],
            para=block.para
        )
    pt = fill_field2d_array_halo(
        halo=block.mesh.halo,
        f=pt,
        meta=block.state.pt.meta
    )
    # =================================================================
    return tree_at(
        where=lambda b: b.state.pt.d,
        pytree=block,
        replace=pt,
        at_dims=0,
        at_idxes=0
    )


def bkg_calc_dry_qv(block: BLOCK, mute: bool = False) -> BLOCK:
    idx_qv = block.tracers.idx_qv

    if idx_qv == -1:
        return block

    if not mute:
        log.logger.info("Calculate water vapor dry mixing ratio.")
    # =================================================================
    q = block.tracers.q.unwrap()[0, :, :, :]
    qm = block.tracers.qm.unwrap()[:, :]
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    qv = q[idx_qv, :, :] / (1 - qm)
    q = q.at[idx_qv, :, :].set(qv)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_qv,
        meta=block.tracers.q.meta
    )
    # =================================================================
    return tree_at(
        where=lambda b: b.tracers.q.d,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )
