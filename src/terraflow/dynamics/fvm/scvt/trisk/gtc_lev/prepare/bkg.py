from __future__ import annotations
from typing import TYPE_CHECKING


from terraflow.util import typing, log, sphere_geometry, formula
from terraflow.prepare.readers.era5 import ERA5Reader
from terraflow.mesh.scvt.parallel import (
    fill_field1d_array_halo,
    fill_field2d_array_halo,
    fill_field3d_array_halo
)
from terraflow.mesh.scvt.regridder import Regridder
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.space_operators import (
    calc_mg,
    calc_dmg
)
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.tracer import tracer_calc_qm


if TYPE_CHECKING:
    from terraflow.prepare.readers.abstract import AbstractBackgroundReader
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block


Float = typing.Float
Array = typing.Array


def regrid_bkg_field(
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    data: Array,
) -> Array:
    bkd = reader.dev.backend
    if reader.cfg.hori_interp_backend == "esmf" and data.ndim == 3:
        return bkd.stack(
            [regridder(data[k, :, :]) for k in range(int(data.shape[0]))],
            axis=0,
        )
    return regridder(data)


def prepare_bkg(
    block: Block,
    reader: AbstractBackgroundReader
) -> Block:
    # Regridder for background data
    src = Regridder.create_latlon_info(
        lon=reader.lon,
        lat=reader.lat
    )
    dst_at_cell = Regridder.create_scvt_info(
        para=block.para,
        mesh=block.mesh,
        loc="cell",
    )
    if block.dev.is_root(): log.logger.info("Build regridder for background data.")
    regridder_at_cell = Regridder(
        src=src,
        dst=dst_at_cell,
        layout="triskc",
        backend=reader.cfg.hori_interp_backend,
        method=reader.cfg.hori_interp_method,
        extrap=True,
        esmf_extrap_method=reader.cfg.esmf_extrap_method,
        esmf_conserve_opt=reader.cfg.esmf_conserve_opt,
        precompute_weights=True,
    )
    if reader.cfg.interp_at_midpoint:
        regridder_uv = regridder_at_cell
    else:
        dst_at_edge = Regridder.create_scvt_info(
            para=block.para,
            mesh=block.mesh,
            loc="edge",
        )
        regridder_uv = Regridder(
            src=src,
            dst=dst_at_edge,
            layout="triskc",
            backend=reader.cfg.hori_interp_backend,
            method=reader.cfg.hori_interp_method,
            extrap=True,
            esmf_extrap_method=reader.cfg.esmf_extrap_method,
            esmf_conserve_opt=reader.cfg.esmf_conserve_opt,
            precompute_weights=True,
        )

    # Regrid background data and calculate derived variables
    # Stage 1:
    block = regrid_phs(
        block=block,
        reader=reader,
        regridder=regridder_at_cell,
    )
    block = bkg_calc_ph(block=block)
    for cyc in range(3):
        block = regrid_wet_qv(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            mute=cyc != 0,
        )
        block = regrid_wet_qc(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            mute=cyc != 0,
        )
        block = regrid_wet_qi(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            mute=cyc != 0,
        )
        block = regrid_wet_qr(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            mute=cyc != 0,
        )
        block = regrid_wet_qs(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            mute=cyc != 0,
        )
        block = tracer_calc_qm(block, "old")
        # Stage 2:
        block = bkg_calc_mgs(block=block, reader=reader, mute=cyc != 0)
        block = bkg_calc_mg(block=block, mute=cyc != 0)
        block = bkg_calc_dry_qv(block=block, mute=cyc != 0)
        block = bkg_calc_dry_qc(block=block, mute=cyc != 0)
        block = bkg_calc_dry_qi(block=block, mute=cyc != 0)
        block = bkg_calc_dry_qr(block=block, mute=cyc != 0)
        block = bkg_calc_dry_qs(block=block, mute=cyc != 0)
        block = tracer_calc_qm(block, "old")
        block = calc_ph(block=block)
    # Stage 3:
    block = regrid_un_from_uv(
        block=block,
        reader=reader,
        regridder=regridder_uv,
    )
    if reader.cfg.prepare_regrid_gz:
        block = regrid_gz(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
        )
        block = bkg_calc_t(block=block)
    else:
        block = regrid_t(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
        )
    block = bkg_calc_tv(block=block)
    block = bkg_calc_pt(block=block)
    return block


def regrid_phs(
    block: Block,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
) -> Block:
    bkd = block.backend
    if block.dev.is_root(): 
        log.logger.info("Regrid surface hydrostatic pressure.")
    g = block.para.gravity.unwrap()
    mesh = block.mesh
    ids, ide = mesh.cell_ids, mesh.cell_ide
    gzs = block.static.gzs.unwrap()
    phs = block.state .phs.unwrap()[0, :]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        z0 = regrid_bkg_field(reader, regridder, reader.z)
        logp0 = bkd.expand_dims(bkd.log(reader.plev), axis=1)
        logp0 = bkd.tile(logp0, (1, z0.shape[1]))
        tmp = bkd.vert_linear_interp(
            x1=bkd.flip(z0[:, ids:ide+1], axis=0),
            data1=bkd.flip(logp0[:, ids:ide+1], axis=0),
            x2=gzs[ids:ide + 1] / g,
            allow_extrap=True
        )
        tmp = bkd.exp(tmp).squeeze()
        phs = phs.at[ids:ide+1].set(tmp)
    else:
        raise ValueError(
            f"Unsupported reader type for phs interpolation: {type(reader)}"
        )
    phs = fill_field1d_array_halo(
        halo=mesh.halo,
        f=phs,
        meta=block.state.phs.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.state.phs.d,
        pytree=block,
        replace=phs,
        at_dims=0,
        at_idxes=0
    )


def bkg_calc_ph(block: Block) -> Block:
    bkd = block.backend
    if block.dev.is_root(): 
        log.logger.info("Calculate hydrostatic pressure on each level.")
    phs = block.state.phs.unwrap()[0, ...]
    mesh = block.mesh
    ids, ide = mesh.cell_ids, mesh.cell_ide
    full_kds, full_kde = mesh.full_kds, mesh.full_kde
    half_kds, half_kde = mesh.half_kds, mesh.half_kde
    ph = block.state.ph.unwrap()[0, ...]
    ph_lev = block.state.ph_lev.unwrap()[0, ...]
    # =================================================================
    ph = ph.at[full_kds:full_kde+1, ids:ide+1].set(
        block.vert_coord.calc_mg(
            mgs=phs[ids:ide+1],
            ks=full_kds,
            ke=full_kde,
        )
    )
    ph_lev = ph_lev.at[half_kds:half_kde+1, ids:ide+1].set(
        block.vert_coord.calc_mg_lev(
            mgs=phs[ids:ide+1],
            ks=half_kds,
            ke=half_kde,
        )
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
    # =================================================================
    if block.mode.is_hydrostatic():
        return bkd.tree_at(
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
        return bkd.tree_at(
            where=lambda b: (b.state.ph.d, b.state.ph_lev.d),
            pytree=block,
            replace=(ph, ph_lev),
            at_dims=(0, 0),
            at_idxes=(0, 0)
        )


def calc_ph(block: Block) -> Block:
    bkd = block.backend
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
            mg_lev[kds, ids:ide+1] + bkd.cumulative_sum(
                moist_dmg,
                axis=0,
                include_initial=True
            )
        )
    else:
        ph_lev = ph_lev.at[kds:kde+1, ids:ide+1].set(
            mg_lev[kds, ids:ide+1] + bkd.cumulative_sum(
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
        return bkd.tree_at(
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
        return bkd.tree_at(
            where=lambda b: (b.state.phs.d, b.state.ph.d, b.state.ph_lev.d),
            pytree=block,
            replace=(phs, ph, ph_lev),
            at_dims=(0, 0, 0),
            at_idxes=(0, 0, 0)
        )


def regrid_gz(
    block: Block,
    reader: AbstractBackgroundReader,
    regridder: Regridder
) -> Block:
    bkd = block.backend
    if block.dev.is_root(): log.logger.info("Regrid geopotential height.")
    mesh = block.mesh
    ids, ide = mesh.cell_ids, mesh.cell_ide
    half_kde = mesh.half_kde
    full_kds = mesh.full_kds
    full_kde = mesh.full_kde
    g      = block.para  .gravity.unwrap()
    gzs    = block.static.gzs    .unwrap()
    gz     = block.state .gz     .unwrap()[0, ...]
    gz_lev = block.state .gz_lev .unwrap()[0, ...]
    ph_lev = block.state .ph_lev .unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        z0 = regrid_bkg_field(reader, regridder, reader.z)
        p0 = reader.plev
        tmp = bkd.vert_log_linear_interp(
            x1=p0,
            data1=z0[:, ids:ide+1],
            x2=ph_lev[full_kds:full_kde+1, ids:ide + 1],
            allow_extrap=True
        )
        gz_lev = gz_lev.at[full_kds:full_kde+1, ids:ide+1].set(tmp * g)
        gz_lev = gz_lev.at[half_kde, ids:ide+1].set(gzs[ids:ide+1])
        gz = gz.at[full_kds:full_kde+1, ids:ide+1].set(
            0.5 * (
                gz_lev[full_kds:full_kde+1, ids:ide+1]
                + gz_lev[full_kds+1:full_kde+2, ids:ide+1]
            )
        )
    else:
        raise ValueError(
            f"Unsupported reader type for gz interpolation: {type(reader)}"
        )
    gz_lev = fill_field2d_array_halo(
        halo=mesh.halo,
        f=gz_lev,
        meta=block.state.gz_lev.meta
    )
    gz = fill_field2d_array_halo(
        halo=mesh.halo,
        f=gz,
        meta=block.state.gz.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: (b.state.gz.d, b.state.gz_lev.d),
        pytree=block,
        replace=(gz, gz_lev),
        at_dims=(0, 0),
        at_idxes=(0, 0)
    )


def bkg_calc_t(block: Block) -> Block:
    bkd = block.backend
    if block.dev.is_root(): log.logger.info("Calculate temperature.")
    mesh = block.mesh
    ids, ide = mesh.cell_ids, mesh.cell_ide
    kds, kde = mesh.full_kds, mesh.full_kde
    rd = block.para.rd.unwrap()
    idx_qv = block.tracers.idx_qv
    ph_lev = block.state.ph_lev.unwrap()[0, ...]
    gz_lev = block.state.gz_lev.unwrap()[0, ...]
    qm = block.tracers.qm.unwrap()
    t = block.aux.t.unwrap()
    # =================================================================
    delta_gz = (
        gz_lev[kds:kde+1, ids:ide+1] - gz_lev[kds+1:kde+2, ids:ide+1]
    )
    log_ratio = bkd.log(
        ph_lev[kds+1:kde+2, ids:ide+1] / ph_lev[kds:kde+1, ids:ide+1]
    )
    tv = delta_gz / (rd * log_ratio)
    if idx_qv == -1:
        t = t.at[kds:kde+1, ids:ide+1].set(tv)
    else:
        q = block.tracers.q.unwrap()[0, idx_qv, :, :]
        t = t.at[kds:kde+1, ids:ide+1].set(
            formula.temperature_from_virtual_temperature_and_dry_mixing_ratio(
                tv,
                q[kds:kde+1, ids:ide+1],
                qm[kds:kde+1, ids:ide+1],
                block.para
            )
        )
    t = fill_field2d_array_halo(
        halo=mesh.halo,
        f=t,
        meta=block.aux.t.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.aux.t.d,
        pytree=block,
        replace=t,
    )


def regrid_t(
    block: Block,
    reader: AbstractBackgroundReader,
    regridder: Regridder
) -> Block:
    bkd = block.backend
    if block.dev.is_root(): log.logger.info("Regrid temperature.")
    mesh = block.mesh
    ids, ide = mesh.cell_ids, mesh.cell_ide
    kds, kde = mesh.full_kds, mesh.full_kde
    ph = block.state.ph.unwrap()[0, ...]
    t  = block.aux  .t .unwrap()
    # =================================================================
    if isinstance(reader, ERA5Reader):
        t0 = regrid_bkg_field(reader, regridder, reader.t)
        tmp = bkd.vert_log_linear_interp(
            x1=reader.plev,
            data1=t0[:, ids:ide+1],
            x2=ph[kds:kde+1, ids:ide + 1],
            allow_extrap=True
        )
    else:
        raise ValueError(
            f"Unsupported reader type for t interpolation: {type(reader)}"
        )
    t = t.at[kds:kde+1, ids:ide+1].set(tmp)
    t = fill_field2d_array_halo(
        halo=mesh.halo,
        f=t,
        meta=block.aux.t.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.aux.t.d,
        pytree=block,
        replace=t,
    )


def regrid_un_from_uv(
    block: Block,
    reader: AbstractBackgroundReader,
    regridder: Regridder
) -> Block:
    bkd = block.backend
    radius = block.para.radius.unwrap()
    mesh = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.edge_ids, mesh.edge_ide
    un = bkd.zeros_like(block.state.un.unwrap()[0, :, :])
    # =================================================================
    if reader.cfg.interp_at_midpoint:
        cell_0 = mesh.cell_of_edge[ids:ide+1, 0]
        cell_1 = mesh.cell_of_edge[ids:ide+1, 1]
        u_cell = _regrid_u(
            block=block,
            reader=reader,
            regridder=regridder,
        )[kds:kde+1, :]
        u = 0.5 * (u_cell[:, cell_0] + u_cell[:, cell_1])
        v_cell = _regrid_v(
            block=block,
            reader=reader,
            regridder=regridder,
        )[kds:kde+1, :]
        v = 0.5 * (v_cell[:, cell_0] + v_cell[:, cell_1])
    else:
        u = _regrid_u(
            block=block,
            reader=reader,
            regridder=regridder,
        )[kds:kde+1, ids:ide+1]
        v = _regrid_v(
            block=block,
            reader=reader,
            regridder=regridder,
        )[kds:kde+1, ids:ide+1]
    vec_xyz = sphere_geometry.vector_convert_lonlat2xyz(
        bkd=block.backend,
        R=radius,
        u=u, 
        v=v, 
        xyz=mesh.dual_edge_xyz[ids:ide+1, :]
    )
    un_ = bkd.sum(vec_xyz * mesh.edge_nr[ids:ide+1, :], axis=-1)
    un = un.at[kds:kde+1, ids:ide+1].set(un_)
    un = fill_field2d_array_halo(
        halo=mesh.halo,
        f=un,
        meta=block.state.un.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.state.un.d,
        pytree=block,
        replace=un,
        at_dims=0,
        at_idxes=0
    )


def _regrid_u(
    block: Block,
    reader: AbstractBackgroundReader,
    regridder: Regridder
) -> Float[Array, "nlev nedge"]:
    bkd = block.backend
    if block.dev.is_root(): log.logger.info("Regrid u wind component.")
    mesh = block.mesh
    ph = block.state.ph.unwrap()[0, ...]
    ph = fill_field2d_array_halo(
        halo=mesh.halo,
        f=ph,
        meta=block.state.ph.meta
    )
    if reader.cfg.interp_at_midpoint:
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        x2 = ph
        meta = block.state.ph.meta
    else:
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        cell_0 = mesh.cell_of_edge[ids:ide+1, 0]
        cell_1 = mesh.cell_of_edge[ids:ide+1, 1]
        x2 = block.state.un.unwrap()[0, :, :]
        x2 = x2.at[:, ids:ide+1].set(
            0.5 * (ph[:, cell_0] + ph[:, cell_1])
        )
        x2 = fill_field2d_array_halo(
            halo=mesh.halo,
            f=x2,
            meta=block.state.un.meta
        )
        meta = block.state.un.meta
    u = bkd.zeros_like(x2)
    # =================================================================
    if isinstance(reader, ERA5Reader):
        u0 = regrid_bkg_field(reader, regridder, reader.u)
        tmp = bkd.vert_linear_interp(
            x1=reader.plev,
            data1=u0[:, ids:ide+1],
            x2=x2[kds:kde+1, ids:ide+1],
            allow_extrap=True
        )
    else:
        raise ValueError(
            f"Unsupported reader type for u interpolation: {type(reader)}"
        )
    u = u.at[kds:kde+1, ids:ide+1].set(tmp)
    u = fill_field2d_array_halo(
        halo=mesh.halo,
        f=u,
        meta=meta
    )
    # =================================================================
    return u


def _regrid_v(
    block: Block,
    reader: AbstractBackgroundReader,
    regridder: Regridder
) -> Float[Array, "nlev nedge"]:
    bkd = block.backend
    if block.dev.is_root(): log.logger.info("Regrid v wind component.")
    mesh = block.mesh
    ph = block.state.ph.unwrap()[0, ...]
    ph = fill_field2d_array_halo(
        halo=mesh.halo,
        f=ph,
        meta=block.state.ph.meta
    )
    if reader.cfg.interp_at_midpoint:
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        x2 = ph
        meta = block.state.ph.meta
    else:
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        cell_0 = mesh.cell_of_edge[ids:ide+1, 0]
        cell_1 = mesh.cell_of_edge[ids:ide+1, 1]
        x2 = block.state.un.unwrap()[0, :, :]
        x2 = x2.at[:, ids:ide+1].set(
            0.5 * (ph[:, cell_0] + ph[:, cell_1])
        )
        x2 = fill_field2d_array_halo(
            halo=mesh.halo,
            f=x2,
            meta=block.state.un.meta
        )
        meta = block.state.un.meta
    v = bkd.zeros_like(x2)
    # =================================================================
    if isinstance(reader, ERA5Reader):
        v0 = regrid_bkg_field(reader, regridder, reader.v)
        tmp = bkd.vert_linear_interp(
            x1=reader.plev,
            data1=v0[:, ids:ide+1],
            x2=x2[kds:kde+1, ids:ide+1],
            allow_extrap=True
        )
    else:
        raise ValueError(
            f"Unsupported reader type for v interpolation: {type(reader)}"
        )
    v = v.at[kds:kde+1, ids:ide+1].set(tmp)
    v = fill_field2d_array_halo(
        halo=mesh.halo,
        f=v,
        meta=meta
    )
    # =================================================================
    return v


def regrid_wet_qv(
    block: Block,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    mute: bool = False
) -> Block:
    bkd = block.backend
    idx_qv = block.tracers.idx_qv
    if idx_qv == -1:
        return block

    if not mute:
        if block.dev.is_root(): 
            log.logger.info("Regrid water vapor wet mixing ratio.")
    # =================================================================
    mesh = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide
    ph = block.state.ph.unwrap()[0, ...]
    q = block.tracers.q.unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q0 = regrid_bkg_field(reader, regridder, reader.qv)
        tmp = bkd.vert_linear_interp(
            x1=reader.plev,
            data1=q0[:, ids:ide+1],
            x2=ph[kds:kde+1, ids:ide + 1],
            allow_extrap=True
        )
    else:
        raise ValueError(
            f"Unsupported reader type for qv interpolation: {type(reader)}"
        )
    tmp = bkd.where(tmp < 0.0, 0.0, tmp)
    q = q.at[idx_qv, kds:kde+1, ids:ide+1].set(tmp)
    q = fill_field3d_array_halo(
        halo=mesh.halo,
        f=q,
        i0=idx_qv,
        meta=block.tracers.q.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.tracers.q.d,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def regrid_wet_qc(
    block: Block,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    mute: bool = False
) -> Block:
    bkd = block.backend
    idx_qc = block.tracers.idx_qc
    if idx_qc == -1:
        return block

    if not mute:
        if block.dev.is_root(): 
            log.logger.info("Regrid cloud liquid wet mixing ratio.")
    # =================================================================
    mesh = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide
    ph = block.state.ph.unwrap()[0, ...]
    q = block.tracers.q.unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q0 = regrid_bkg_field(reader, regridder, reader.qc)
        tmp = bkd.vert_linear_interp(
            x1=reader.plev,
            data1=q0[:, ids:ide+1],
            x2=ph[kds:kde+1, ids:ide + 1],
            allow_extrap=True
        )
    else:
        raise ValueError(
            f"Unsupported reader type for qc interpolation: {type(reader)}"
        )
    tmp = bkd.where(tmp < 0.0, 0.0, tmp)
    q = q.at[idx_qc, kds:kde+1, ids:ide+1].set(tmp)
    q = fill_field3d_array_halo(
        halo=mesh.halo,
        f=q,
        i0=idx_qc,
        meta=block.tracers.q.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.tracers.q.d,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def regrid_wet_qi(
    block: Block,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    mute: bool = False
) -> Block:
    bkd = block.backend
    idx_qi = block.tracers.idx_qi
    if idx_qi == -1:
        return block

    if not mute:
        if block.dev.is_root(): 
            log.logger.info("Regrid cloud ice wet mixing ratio.")
    # =================================================================
    mesh = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide
    ph = block.state.ph.unwrap()[0, ...]
    q = block.tracers.q.unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q0 = regrid_bkg_field(reader, regridder, reader.qi)
        tmp = bkd.vert_linear_interp(
            x1=reader.plev,
            data1=q0[:, ids:ide+1],
            x2=ph[kds:kde+1, ids:ide + 1],
            allow_extrap=True
        )
    else:
        raise ValueError(
            f"Unsupported reader type for qi interpolation: {type(reader)}"
        )
    tmp = bkd.where(tmp < 0.0, 0.0, tmp)
    q = q.at[idx_qi, kds:kde+1, ids:ide+1].set(tmp)
    q = fill_field3d_array_halo(
        halo=mesh.halo,
        f=q,
        i0=idx_qi,
        meta=block.tracers.q.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.tracers.q.d,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def regrid_wet_nc(
    block: Block,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    mute: bool = False
) -> Block:
    bkd = block.backend
    idx_nc = block.tracers.idx_nc

    if idx_nc == -1:
        return block

    if not mute:
        if block.dev.is_root(): 
            log.logger.info("Regrid cloud liquid number wet mixing ratio.")
    # =================================================================
    mesh = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide
    q = block.tracers.q.unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q_nc = q[idx_nc, kds:kde+1, ids:ide+1]
    else:
        q_nc = q[idx_nc, kds:kde+1, ids:ide+1]
    q_nc = bkd.where(q_nc < 0.0, 0.0, q_nc)
    q = q.at[idx_nc, kds:kde+1, ids:ide+1].set(q_nc)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_nc,
        meta=block.tracers.q.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.tracers.q.d,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def regrid_wet_ni(
    block: Block,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    mute: bool = False
) -> Block:
    bkd = block.backend
    idx_ni = block.tracers.idx_ni

    if idx_ni == -1:
        return block

    if not mute:
        if block.dev.is_root(): 
            log.logger.info("Regrid cloud ice number wet mixing ratio.")
    # ================================================================= 
    mesh = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide
    q = block.tracers.q.unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q_ni = q[idx_ni, kds:kde+1, ids:ide+1]
    else:
        q_ni = q[idx_ni, kds:kde+1, ids:ide+1]
    q_ni = bkd.where(q_ni < 0.0, 0.0, q_ni)
    q = q.at[idx_ni, kds:kde+1, ids:ide+1].set(q_ni)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_ni,
        meta=block.tracers.q.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.tracers.q.d,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def regrid_wet_qr(
    block: Block,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    mute: bool = False
) -> Block:
    bkd = block.backend
    idx_qr = block.tracers.idx_qr

    if idx_qr == -1:
        return block

    if not mute:
        if block.dev.is_root(): 
            log.logger.info("Regrid rain water wet mixing ratio.")
    # =================================================================
    mesh = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide
    ph = block.state  .ph.unwrap()[0, ...]
    q  = block.tracers.q .unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q0 = regrid_bkg_field(reader, regridder, reader.qr)
        tmp = bkd.vert_linear_interp(
            x1=reader.plev,
            data1=q0[:, ids:ide+1],
            x2=ph[kds:kde+1, ids:ide + 1],
            allow_extrap=True
        )
    else:
        raise ValueError(
            f"Unsupported reader type for qr interpolation: {type(reader)}"
        )
    tmp = bkd.where(tmp < 0.0, 0.0, tmp)
    q = q.at[idx_qr, kds:kde+1, ids:ide+1].set(tmp)
    q = fill_field3d_array_halo(
        halo=mesh.halo,
        f=q,
        i0=idx_qr,
        meta=block.tracers.q.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.tracers.q.d,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def regrid_wet_qs(
    block: Block,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    mute: bool = False
) -> Block:
    bkd = block.backend
    idx_qs = block.tracers.idx_qs

    if idx_qs == -1:
        return block

    if not mute:
        if block.dev.is_root(): 
            log.logger.info("Regrid snow water wet mixing ratio.")
    # =================================================================
    mesh = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide
    ph = block.state.ph.unwrap()[0, ...]
    q = block.tracers.q.unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q0 = regrid_bkg_field(reader, regridder, reader.qs)
        tmp = bkd.vert_linear_interp(
            x1=reader.plev,
            data1=q0[:, ids:ide+1],
            x2=ph[kds:kde+1, ids:ide + 1],
            allow_extrap=True
        )
    else:
        raise ValueError(
            f"Unsupported reader type for qs interpolation: {type(reader)}"
        )
    tmp = bkd.where(tmp < 0.0, 0.0, tmp)
    q = q.at[idx_qs, kds:kde+1, ids:ide+1].set(tmp)
    q = fill_field3d_array_halo(
        halo=mesh.halo,
        f=q,
        i0=idx_qs,
        meta=block.tracers.q.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.tracers.q.d,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def bkg_calc_mgs(
    block: Block,
    reader: AbstractBackgroundReader,
    mute: bool = False
) -> Block:
    bkd = block.backend
    if not mute:
        if block.dev.is_root(): 
            log.logger.info("Calculate surface dry-air pressure.")
    mesh = block.mesh
    ids, ide = mesh.cell_ids, mesh.cell_ide
    kds, kde = mesh.full_kds, mesh.full_kde
    phs    = block.state  .phs   .unwrap()[0, :]
    ph_lev = block.state  .ph_lev.unwrap()[0, :, :]
    mgs    = block.state  .mgs   .unwrap()[0, :]
    qm     = block.tracers.qm    .unwrap()[:, :]
    # =================================================================
    delta_ph = ph_lev[kds+1:kde+2, ids:ide+1] - ph_lev[kds:kde+1, ids:ide+1]
    tmp = phs[ids:ide+1] - bkd.sum(
        delta_ph * qm[kds:kde+1, ids:ide+1],
        axis=0
    )
    mgs = mgs.at[ids:ide+1].set(tmp)
    mgs = fill_field1d_array_halo(
        halo=mesh.halo,
        f=mgs,
        meta=block.state.mgs.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.state.mgs.d,
        pytree=block,
        replace=mgs,
        at_dims=0,
        at_idxes=0
    )


def bkg_calc_mg(block: Block, mute: bool = False) -> Block:
    if not mute:
        if block.dev.is_root(): 
            log.logger.info("Calculate dry-air pressure.")
    # =================================================================
    block = calc_mg (block, 0)
    block = calc_dmg(block, 0)
    # ================================================================= 
    return block


def bkg_calc_tv(block: Block) -> Block:
    bkd = block.backend
    if block.dev.is_root(): 
        log.logger.info("Calculate virtual temperature.")
    # =================================================================
    mesh = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide
    idx_qv = block.tracers.idx_qv
    t  = block.aux.t .unwrap()
    tv = block.aux.tv.unwrap()
    # =================================================================
    if idx_qv == -1:
        tv = tv.at[kds:kde+1, ids:ide+1].set(t[kds:kde+1, ids:ide+1])
    else:
        qv = block.tracers.q .unwrap()[0, idx_qv, :, :]
        qm = block.tracers.qm.unwrap()[:, :]
        tv = tv.at[kds:kde+1, ids:ide+1].set(
            formula.virtual_temperature_from_dry_mixing_ratio(
                t=t[kds:kde+1, ids:ide+1],
                qv=qv[kds:kde+1, ids:ide+1],
                qm=qm[kds:kde+1, ids:ide+1],
                para=block.para
            )
        )
    tv = fill_field2d_array_halo(
        halo=block.mesh.halo,
        f=tv,
        meta=block.aux.tv.meta
    )
    return bkd.tree_at(
        where=lambda b: b.aux.tv.d,
        pytree=block,
        replace=tv,
    )


def bkg_calc_pt(block: Block) -> Block:
    bkd = block.backend
    if block.dev.is_root(): 
        log.logger.info("Calculate modified potential temperature.")
    # =================================================================
    mesh   = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide
    idx_qv = block.tracers.idx_qv
    ph     = block.state.ph.unwrap()[0, ...]
    pt     = block.state.pt.unwrap()[0, ...]
    t      = block.aux  .t .unwrap()
    # =================================================================
    if idx_qv == -1:
        tmp = formula.modified_potential_temperature(
            t=t[kds:kde+1, ids:ide+1],
            p=ph[kds:kde+1, ids:ide+1],
            qv=bkd.array(
                0.0,
                device=block.dev(),
                dtype=block.dev.float_type
            ),
            para=block.para
        )
    else:
        q = block.tracers.q.unwrap()[0, :, :, :]
        tmp = formula.modified_potential_temperature(
            t=t[kds:kde+1, ids:ide+1],
            p=ph[kds:kde+1, ids:ide+1],
            qv=q[idx_qv, kds:kde+1, ids:ide+1],
            para=block.para
        )
    pt = pt.at[kds:kde+1, ids:ide+1].set(tmp)
    pt = fill_field2d_array_halo(
        halo=block.mesh.halo,
        f=pt,
        meta=block.state.pt.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.state.pt.d,
        pytree=block,
        replace=pt,
        at_dims=0,
        at_idxes=0
    )


def bkg_calc_dry_qv(block: Block, mute: bool = False) -> Block:
    bkd = block.backend
    idx_qv = block.tracers.idx_qv

    if idx_qv == -1:
        return block

    if not mute:
        if block.dev.is_root(): 
            log.logger.info("Calculate water vapor dry mixing ratio.")
    # =================================================================
    mesh = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide
    q  = block.tracers.q .unwrap()[0, :, :, :]
    qm = block.tracers.qm.unwrap()[:, :]
    # =================================================================
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    qv = q[idx_qv, kds:kde+1, ids:ide+1] / (1 - qm[kds:kde+1, ids:ide+1])
    q = q.at[idx_qv, kds:kde+1, ids:ide+1].set(qv)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_qv,
        meta=block.tracers.q.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.tracers.q.d,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def bkg_calc_dry_qc(block: Block, mute: bool = False) -> Block:
    bkd = block.backend
    idx_qc = block.tracers.idx_qc

    if idx_qc == -1:
        return block

    if not mute:
        if block.dev.is_root(): 
            log.logger.info("Calculate cloud liquid dry mixing ratio.")
    # =================================================================
    mesh = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide    
    q  = block.tracers.q .unwrap()[0, :, :, :]
    qm = block.tracers.qm.unwrap()[:, :]
    # =================================================================
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    qc = q[idx_qc, kds:kde+1, ids:ide+1] / (1 - qm[kds:kde+1, ids:ide+1])
    q = q.at[idx_qc, kds:kde+1, ids:ide+1].set(qc)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_qc,
        meta=block.tracers.q.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.tracers.q.d,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def bkg_calc_dry_qi(block: Block, mute: bool = False) -> Block:
    bkd = block.backend
    idx_qi = block.tracers.idx_qi

    if idx_qi == -1:
        return block

    if not mute:
        if block.dev.is_root(): 
            log.logger.info("Calculate cloud ice dry mixing ratio.")
    # =================================================================
    mesh = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide
    q  = block.tracers.q .unwrap()[0, :, :, :]
    qm = block.tracers.qm.unwrap()[:, :]
    # =================================================================
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    qi = q[idx_qi, kds:kde+1, ids:ide+1] / (1 - qm[kds:kde+1, ids:ide+1])
    q = q.at[idx_qi, kds:kde+1, ids:ide+1].set(qi)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_qi,
        meta=block.tracers.q.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.tracers.q.d,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def bkg_calc_dry_nc(block: Block, mute: bool = False) -> Block:
    bkd = block.backend
    idx_nc = block.tracers.idx_nc

    if idx_nc == -1:
        return block

    if not mute:
        if block.dev.is_root(): 
            log.logger.info("Calculate cloud liquid number dry mixing ratio.")
    # =================================================================
    mesh = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide
    q  = block.tracers.q .unwrap()[0, :, :, :]
    qm = block.tracers.qm.unwrap()[:, :]
    # =================================================================
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    nc = q[idx_nc, kds:kde+1, ids:ide+1] / (1 - qm[kds:kde+1, ids:ide+1])
    q = q.at[idx_nc, kds:kde+1, ids:ide+1].set(nc)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_nc,
        meta=block.tracers.q.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.tracers.q.d,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def bkg_calc_dry_ni(block: Block, mute: bool = False) -> Block:
    bkd = block.backend
    idx_ni = block.tracers.idx_ni

    if idx_ni == -1:
        return block

    if not mute:
        if block.dev.is_root(): 
            log.logger.info("Calculate cloud ice number dry mixing ratio.")
    # =================================================================
    mesh = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide
    q  = block.tracers.q .unwrap()[0, :, :, :]
    qm = block.tracers.qm.unwrap()[:, :]
    # =================================================================
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    ni = q[idx_ni, kds:kde+1, ids:ide+1] / (1 - qm[kds:kde+1, ids:ide+1])
    q = q.at[idx_ni, kds:kde+1, ids:ide+1].set(ni)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_ni,
        meta=block.tracers.q.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.tracers.q.d,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def bkg_calc_dry_qr(block: Block, mute: bool = False) -> Block:
    bkd = block.backend
    idx_qr = block.tracers.idx_qr

    if idx_qr == -1:
        return block

    if not mute:
        if block.dev.is_root(): 
            log.logger.info("Calculate rain water dry mixing ratio.")
    # =================================================================
    mesh = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide
    q  = block.tracers.q .unwrap()[0, :, :, :]
    qm = block.tracers.qm.unwrap()[:, :]
    # =================================================================
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    qr = q[idx_qr, kds:kde+1, ids:ide+1] / (1 - qm[kds:kde+1, ids:ide+1])
    q = q.at[idx_qr, kds:kde+1, ids:ide+1].set(qr)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_qr,
        meta=block.tracers.q.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.tracers.q.d,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def bkg_calc_dry_qs(block: Block, mute: bool = False) -> Block:
    bkd = block.backend
    idx_qs = block.tracers.idx_qs

    if idx_qs == -1:
        return block

    if not mute:
        if block.dev.is_root(): 
            log.logger.info("Calculate snow water dry mixing ratio.")
    # =================================================================
    mesh = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide
    q  = block.tracers.q .unwrap()[0, :, :, :]
    qm = block.tracers.qm.unwrap()[:, :]
    # =================================================================
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    qs = q[idx_qs, kds:kde+1, ids:ide+1] / (1 - qm[kds:kde+1, ids:ide+1])
    q = q.at[idx_qs, kds:kde+1, ids:ide+1].set(qs)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_qs,
        meta=block.tracers.q.meta
    )
    # =================================================================
    return bkd.tree_at(
        where=lambda b: b.tracers.q.d,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )
