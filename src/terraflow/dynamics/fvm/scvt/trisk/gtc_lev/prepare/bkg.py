from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp

from terraflow.util import typing, log, sphere_geometry, formula
from terraflow.util.pytree import tree_at
from terraflow.prepare.readers.era5 import ERA5Reader
from terraflow.mesh.scvt.parallel import (
    fill_field1d_array_halo,
    fill_field2d_array_halo,
    fill_field3d_array_halo
)
from terraflow.prepare.vert_interp import (
    VerticalLinearInterpolator,
    VerticalLogLinearInterpolator,
    grist_vert_interp_linear,
    grist_vert_log_interp_linear
)
from terraflow.mesh.scvt.regridder import Regridder
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.space_operators import (
    calc_mg,
    calc_dmg
)
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.tracer import tracer_calc_qm

if TYPE_CHECKING:
    from terraflow.prepare.readers.abstract import AbstractBackgroundReader
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block as BLOCK


Float = typing.Float
Array = typing.Array


def prepare_bkg(
    block: BLOCK,
    reader: AbstractBackgroundReader
) -> BLOCK:
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
    log.logger.info("Build regridder for background data at cell center.")
    regridder_at_cell = Regridder(
        src=src,
        dst=dst_at_cell,
        layout="triskc",
        backend=reader.cfg.hori_interp_backend,
        ndbounds=(reader.lev.shape[0],),
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
            ndbounds=(reader.lev.shape[0],),
            method=reader.cfg.hori_interp_method,
            extrap=True,
            esmf_extrap_method=reader.cfg.esmf_extrap_method,
            esmf_conserve_opt=reader.cfg.esmf_conserve_opt,
            precompute_weights=True,
        )

    if reader.cfg.vert_interp_type == "native":
        allow_extrap = True
    elif reader.cfg.vert_interp_type == "grist":
        allow_extrap = False
    else:
        raise ValueError(
            f"Unsupported vertical interpolation type: {reader.cfg.vert_interp_type}"
        )
    vert_linear_interp = VerticalLinearInterpolator(
        backend=reader.cfg.vert_interp_backend,
        allow_extrap=allow_extrap
    )
    vert_log_linear_interp = VerticalLogLinearInterpolator(
        backend=reader.cfg.vert_interp_backend,
        allow_extrap=allow_extrap
    )
    # Regrid background data and calculate derived variables
    if reader.cfg.vert_interp_type == "native":
        # Stage 1:
        block = regrid_phs(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            vert_interp_linear=vert_linear_interp,
        )
        block = bkg_calc_ph(block=block)
        for cyc in range(3):
            block = regrid_wet_qv(
                block=block,
                reader=reader,
                regridder=regridder_at_cell,
                mute=cyc != 0,
                vert_interp_linear=vert_linear_interp,
            )
            block = regrid_wet_qc(
                block=block,
                reader=reader,
                regridder=regridder_at_cell,
                mute=cyc != 0,
                vert_interp_linear=vert_linear_interp,
            )
            block = regrid_wet_qi(
                block=block,
                reader=reader,
                regridder=regridder_at_cell,
                mute=cyc != 0,
                vert_interp_linear=vert_linear_interp,
            )
            block = regrid_wet_qr(
                block=block,
                reader=reader,
                regridder=regridder_at_cell,
                mute=cyc != 0,
                vert_interp_linear=vert_linear_interp,
            )
            block = regrid_wet_qs(
                block=block,
                reader=reader,
                regridder=regridder_at_cell,
                mute=cyc != 0,
                vert_interp_linear=vert_linear_interp,
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
            vert_interp_linear=vert_linear_interp,
        )
        if reader.cfg.prepare_regrid_gz:
            block = regrid_gz(
                block=block,
                reader=reader,
                regridder=regridder_at_cell,
                vert_interp_log_linear=vert_log_linear_interp,
            )
            block = bkg_calc_t(block=block)
        else:
            block = regrid_t(
                block=block,
                reader=reader,
                regridder=regridder_at_cell,
                vert_interp_linear=vert_linear_interp,
                vert_interp_log_linear=vert_log_linear_interp,
            )
        block = bkg_calc_tv(block=block)
        block = bkg_calc_pt(block=block)
    elif reader.cfg.vert_interp_type == "grist":
        reader = grist_post_process(
            reader=reader,
            regridder=regridder_at_cell,
        )
        block = regrid_phs(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            vert_interp_linear=vert_linear_interp
        )
        block = bkg_calc_ph(block=block)
        block = regrid_wet_qv(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            vert_interp_linear=vert_linear_interp,
            mute=False
        )
        block = regrid_wet_qc(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            vert_interp_linear=vert_linear_interp,
            mute=False
        )
        block = regrid_wet_qi(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            vert_interp_linear=vert_linear_interp,
            mute=False
        )
        block = regrid_wet_qr(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            vert_interp_linear=vert_linear_interp,
            mute=False
        )
        block = regrid_wet_qs(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            vert_interp_linear=vert_linear_interp,
            mute=False
        )
        block = tracer_calc_qm(block, "old")
        block = bkg_calc_dry_qv(block=block, mute=False)
        block = bkg_calc_dry_qc(block=block, mute=False)
        block = bkg_calc_dry_qi(block=block, mute=False)
        block = bkg_calc_dry_qr(block=block, mute=False)
        block = bkg_calc_dry_qs(block=block, mute=False)
        block = tracer_calc_qm(block, "old")
        block = regrid_un_from_uv(
            block=block,
            reader=reader,
            regridder=regridder_uv,
            vert_interp_linear=vert_linear_interp
        )
        block = regrid_t(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            vert_interp_linear=vert_linear_interp,
            vert_interp_log_linear=vert_log_linear_interp
        )
        block = bkg_calc_mgs(block=block, reader=reader, mute=False)
        block = bkg_calc_mg(block=block, mute=False)
        block = calc_ph(block=block)
        block = bkg_calc_tv(block=block)
        block = bkg_calc_pt(block=block)
    else:
        raise ValueError(
            f"Unsupported vertical interpolation type: {reader.cfg.vert_interp_type}"
        )
    return block


def grist_post_process(
    reader: AbstractBackgroundReader,
    regridder: Regridder,
) -> AbstractBackgroundReader:
    log.logger.info("Post-process background data for GRIST vertical interpolation.")
    if isinstance(reader, ERA5Reader):
        ps = jnp.tile(reader.ps[None, :, :], (reader.lev.shape[0], 1, 1))
        ps = regridder(ps)[0, :]
        qv = regridder(reader.qv)
        qc = regridder(reader.qc)
        qi = regridder(reader.qi)
        qr = regridder(reader.qr)
        qs = regridder(reader.qs)
        reader.grist_post_process(
            ps=ps,
            qv=qv,
            qc=qc,
            qi=qi,
            qr=qr,
            qs=qs,
        )
        return reader
    else:
        raise ValueError(
            f"Unsupported reader type for GRIST post-processing: {type(reader)}"
        )


def regrid_phs(
    block: BLOCK,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    vert_interp_linear: VerticalLinearInterpolator,
) -> BLOCK:
    log.logger.info("Regrid surface hydrostatic pressure.")
    g = block.para.gravity.unwrap()
    gzs = block.static.gzs.unwrap()
    # =================================================================
    if isinstance(reader, ERA5Reader):
        if reader.cfg.vert_interp_type == "native":
            z0 = regridder(reader.z)
            logp0 = jnp.expand_dims(jnp.log(reader.plev), axis=1)
            logp0 = jnp.tile(logp0, (1, z0.shape[1]))
            phs = vert_interp_linear(
                x1=jnp.flip(z0, axis=0),
                data1=jnp.flip(logp0, axis=0),
                x2=gzs / g,
            )
            phs = jnp.exp(phs)
        elif reader.cfg.vert_interp_type == "grist":
            plev_lev = reader.plev_lev
            ncol_range = jnp.arange(plev_lev.shape[1])
            phs = plev_lev[reader.nlev_eff, ncol_range]
        else:
            raise ValueError(
                f"Unsupported vertical interpolation type for phs: {reader.cfg.vert_interp_type}"
            )
    else:
        raise ValueError(
            f"Unsupported reader type for phs interpolation: {type(reader)}"
        )
    phs = fill_field1d_array_halo(
        halo=block.mesh.halo,
        f=phs,
        meta=block.state.phs.meta
    )
    # =================================================================
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
    full_kds, full_kde = mesh.full_kds, mesh.full_kde
    half_kds, half_kde = mesh.half_kds, mesh.half_kde
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


def regrid_gz(
    block: BLOCK,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    vert_interp_log_linear: VerticalLogLinearInterpolator
) -> BLOCK:
    log.logger.info("Regrid geopotential height.")
    mesh = block.mesh
    ids = mesh.cell_ids
    ide = mesh.cell_ide
    half_kde = mesh.half_kde
    full_kds = mesh.full_kds
    full_kde = mesh.full_kde
    g = block.para.gravity.unwrap()
    gzs = block.static.gzs.unwrap()
    gz = block.state.gz.unwrap()[0, ...]
    ph_lev = block.state.ph_lev.unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        z0 = regridder(reader.z)
        p0 = reader.plev
        if reader.cfg.vert_interp_type == "native":
            gz_lev = vert_interp_log_linear(
                x1=p0,
                data1=z0,
                x2=ph_lev,
            )
        elif reader.cfg.vert_interp_type == "grist":
            gz_lev = grist_vert_log_interp_linear(
                vert_log_interp_linear=vert_interp_log_linear,
                nlev_eff=reader.nlev_eff,
                x1=p0,
                data1=z0,
                x2=ph_lev,
            )
        else:
            raise ValueError(
                f"Unsupported vertical interpolation type for gz: {reader.cfg.vert_interp_type}"
            )
        gz_lev = gz_lev * g
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
    return tree_at(
        where=lambda b: (b.state.gz.d, b.state.gz_lev.d),
        pytree=block,
        replace=(gz, gz_lev),
        at_dims=(0, 0),
        at_idxes=(0, 0)
    )


def bkg_calc_t(block: BLOCK) -> BLOCK:
    log.logger.info("Calculate temperature.")
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
    log_ratio = jnp.log(
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
    return tree_at(
        where=lambda b: b.aux.t.d,
        pytree=block,
        replace=t,
    )


def regrid_t(
    block: BLOCK,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    vert_interp_linear: VerticalLinearInterpolator,
    vert_interp_log_linear: VerticalLogLinearInterpolator
) -> BLOCK:
    log.logger.info("Regrid temperature.")
    ph = block.state.ph.unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        t0 = regridder(reader.t)
        if reader.cfg.vert_interp_type == "native":
            t = vert_interp_log_linear(
                x1=reader.plev,
                data1=t0,
                x2=ph,
            )
        elif reader.cfg.vert_interp_type == "grist":
            t = grist_vert_interp_linear(
                vert_interp_linear=vert_interp_linear,
                nlev_eff=reader.nlev_eff,
                x1=reader.plev,
                data1=t0,
                x2=ph,
            )
        else:
            raise ValueError(
                f"Unsupported vertical interpolation type for t: {reader.cfg.vert_interp_type}"
            )
    else:
        raise ValueError(
            f"Unsupported reader type for t interpolation: {type(reader)}"
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


def regrid_un_from_uv(
    block: BLOCK,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    vert_interp_linear: VerticalLinearInterpolator
) -> BLOCK:
    radius = block.para.radius.unwrap()
    mesh = block.mesh
    ids, ide = mesh.edge_ids, mesh.edge_ide
    un = block.state.un.unwrap()[0, :, :]
    # =================================================================
    if reader.cfg.interp_at_midpoint:
        cell_0 = mesh.cell_of_edge[ids:ide+1, 0]
        cell_1 = mesh.cell_of_edge[ids:ide+1, 1]
        u_cell = _regrid_u(
            block=block,
            reader=reader,
            regridder=regridder,
            vert_interp_linear=vert_interp_linear
        )
        u = 0.5 * (u_cell[:, cell_0] + u_cell[:, cell_1])
        v_cell = _regrid_v(
            block=block,
            reader=reader,
            regridder=regridder,
            vert_interp_linear=vert_interp_linear
        )
        v = 0.5 * (v_cell[:, cell_0] + v_cell[:, cell_1])
    else:
        u = _regrid_u(
            block=block,
            reader=reader,
            regridder=regridder,
            vert_interp_linear=vert_interp_linear
        )[:, ids:ide+1]
        v = _regrid_v(
            block=block,
            reader=reader,
            regridder=regridder,
            vert_interp_linear=vert_interp_linear
        )[:, ids:ide+1]
    vec_xyz = sphere_geometry.vector_convert_lonlat2xyz(
        R=radius,
        u=u, 
        v=v, 
        xyz=mesh.dual_edge_xyz[ids:ide+1, :]
    )
    un_ = jnp.sum(vec_xyz * mesh.edge_nr[ids:ide+1, :], axis=-1)
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
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    vert_interp_linear: VerticalLinearInterpolator
) -> Float[Array, "nlev nedge"]:
    log.logger.info("Regrid u wind component.")
    mesh = block.mesh
    ph = block.state.ph.unwrap()[0, ...]
    ph = fill_field2d_array_halo(
        halo=mesh.halo,
        f=ph,
        meta=block.state.ph.meta
    )
    if reader.cfg.interp_at_midpoint:
        x2 = ph
        meta = block.state.ph.meta
    else:
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
    # =================================================================
    if isinstance(reader, ERA5Reader):
        u = regridder(reader.u)
        if reader.cfg.vert_interp_type == "native":
            u = vert_interp_linear(
                x1=reader.plev,
                data1=u,
                x2=x2,
            )
        elif reader.cfg.vert_interp_type == "grist":
            u = grist_vert_interp_linear(
                vert_interp_linear=vert_interp_linear,
                nlev_eff=reader.nlev_eff,
                x1=reader.plev,
                data1=u,
                x2=x2,
            )
        else:
            raise ValueError(
                f"Unsupported vertical interpolation type for u: {reader.cfg.vert_interp_type}"
            )
    else:
        raise ValueError(
            f"Unsupported reader type for u interpolation: {type(reader)}"
        )
    u = fill_field2d_array_halo(
        halo=mesh.halo,
        f=u,
        meta=meta
    )
    # =================================================================
    return u


def _regrid_v(
    block: BLOCK,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    vert_interp_linear: VerticalLinearInterpolator
) -> Float[Array, "nlev nedge"]:
    log.logger.info("Regrid v wind component.")
    mesh = block.mesh
    ph = block.state.ph.unwrap()[0, ...]
    ph = fill_field2d_array_halo(
        halo=mesh.halo,
        f=ph,
        meta=block.state.ph.meta
    )
    if reader.cfg.interp_at_midpoint:
        x2 = ph
        meta = block.state.ph.meta
    else:
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
    # =================================================================
    if isinstance(reader, ERA5Reader):
        v = regridder(reader.v)
        if reader.cfg.vert_interp_type == "native":
            v = vert_interp_linear(
                x1=reader.plev,
                data1=v,
                x2=x2,
            )
        elif reader.cfg.vert_interp_type == "grist":
            v = grist_vert_interp_linear(
                vert_interp_linear=vert_interp_linear,
                nlev_eff=reader.nlev_eff,
                x1=reader.plev,
                data1=v,
                x2=x2,
            )
        else:
            raise ValueError(
                f"Unsupported vertical interpolation type for v: {reader.cfg.vert_interp_type}"
            )
    else:
        raise ValueError(
            f"Unsupported reader type for v interpolation: {type(reader)}"
        )
    v = fill_field2d_array_halo(
        halo=mesh.halo,
        f=v,
        meta=meta
    )
    # =================================================================
    return v


def regrid_wet_qv(
    block: BLOCK,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
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
    if isinstance(reader, ERA5Reader):
        q0 = regridder(reader.qv)
        if reader.cfg.vert_interp_type == "native":
            qv = vert_interp_linear(
                x1=reader.plev,
                data1=q0,
                x2=ph,
            )
        elif reader.cfg.vert_interp_type == "grist":
            qv = grist_vert_interp_linear(
                vert_interp_linear=vert_interp_linear,
                nlev_eff=reader.nlev_eff,
                x1=reader.plev,
                data1=q0,
                x2=ph,
            )
        else:
            raise ValueError(
                f"Unsupported vertical interpolation type for qv: {reader.cfg.vert_interp_type}"
            )
    else:
        raise ValueError(
            f"Unsupported reader type for qv interpolation: {type(reader)}"
        )
    qv = jnp.where(qv < 0.0, 0.0, qv)
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


def regrid_wet_qc(
    block: BLOCK,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    vert_interp_linear: VerticalLinearInterpolator,
    mute: bool = False
) -> BLOCK:
    idx_qc = block.tracers.idx_qc
    if idx_qc == -1:
        return block

    if not mute:
        log.logger.info("Regrid cloud liquid wet mixing ratio.")
    # =================================================================
    ph = block.state.ph.unwrap()[0, ...]
    q = block.tracers.q.unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q0 = regridder(reader.qc)
        if reader.cfg.vert_interp_type == "native":
            qc = vert_interp_linear(
                x1=reader.plev,
                data1=q0,
                x2=ph,
            )
        elif reader.cfg.vert_interp_type == "grist":
            qc = grist_vert_interp_linear(
                vert_interp_linear=vert_interp_linear,
                nlev_eff=reader.nlev_eff,
                x1=reader.plev,
                data1=q0,
                x2=ph,
            )
        else:
            raise ValueError(
                f"Unsupported vertical interpolation type for qc: {reader.cfg.vert_interp_type}"
            )
    else:
        raise ValueError(
            f"Unsupported reader type for qc interpolation: {type(reader)}"
        )
    qc = jnp.where(qc < 0.0, 0.0, qc)
    q = q.at[idx_qc, :, :].set(qc)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_qc,
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


def regrid_wet_qi(
    block: BLOCK,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    vert_interp_linear: VerticalLinearInterpolator,
    mute: bool = False
) -> BLOCK:
    idx_qi = block.tracers.idx_qi
    if idx_qi == -1:
        return block

    if not mute:
        log.logger.info("Regrid cloud ice wet mixing ratio.")
    # =================================================================
    ph = block.state.ph.unwrap()[0, ...]
    q = block.tracers.q.unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q0 = regridder(reader.qi)
        if reader.cfg.vert_interp_type == "native":
            qi = vert_interp_linear(
                x1=reader.plev,
                data1=q0,
                x2=ph,
            )
        elif reader.cfg.vert_interp_type == "grist":
            qi = grist_vert_interp_linear(
                vert_interp_linear=vert_interp_linear,
                nlev_eff=reader.nlev_eff,
                x1=reader.plev,
                data1=q0,
                x2=ph,
            )
        else:
            raise ValueError(
                f"Unsupported vertical interpolation type for qi: {reader.cfg.vert_interp_type}"
            )
    else:
        raise ValueError(
            f"Unsupported reader type for qi interpolation: {type(reader)}"
        )
    qi = jnp.where(qi < 0.0, 0.0, qi)
    q = q.at[idx_qi, :, :].set(qi)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_qi,
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


def regrid_wet_nc(
    block: BLOCK,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    mute: bool = False
) -> BLOCK:
    idx_nc = block.tracers.idx_nc

    if idx_nc == -1:
        return block

    if not mute:
        log.logger.info("Regrid cloud liquid number wet mixing ratio.")
    # =================================================================
    q = block.tracers.q.unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q_nc = q[idx_nc, :, :]
    else:
        q_nc = q[idx_nc, :, :]
    q_nc = jnp.where(q_nc < 0.0, 0.0, q_nc)
    q = q.at[idx_nc, :, :].set(q_nc)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_nc,
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


def regrid_wet_ni(
    block: BLOCK,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    mute: bool = False
) -> BLOCK:
    idx_ni = block.tracers.idx_ni

    if idx_ni == -1:
        return block

    if not mute:
        log.logger.info("Regrid cloud ice number wet mixing ratio.")
    # =================================================================
    q = block.tracers.q.unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q_ni = q[idx_ni, :, :]
    else:
        q_ni = q[idx_ni, :, :]
    q_ni = jnp.where(q_ni < 0.0, 0.0, q_ni)
    q = q.at[idx_ni, :, :].set(q_ni)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_ni,
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


def regrid_wet_qr(
    block: BLOCK,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    vert_interp_linear: VerticalLinearInterpolator,
    mute: bool = False
) -> BLOCK:
    idx_qr = block.tracers.idx_qr

    if idx_qr == -1:
        return block

    if not mute:
        log.logger.info("Regrid rain water wet mixing ratio.")
    # =================================================================
    ph = block.state.ph.unwrap()[0, ...]
    q = block.tracers.q.unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q0 = regridder(reader.qr)
        if reader.cfg.vert_interp_type == "native":
            qr = vert_interp_linear(
                x1=reader.plev,
                data1=q0,
                x2=ph,
            )
        elif reader.cfg.vert_interp_type == "grist":
            qr = grist_vert_interp_linear(
                vert_interp_linear=vert_interp_linear,
                nlev_eff=reader.nlev_eff,
                x1=reader.plev,
                data1=q0,
                x2=ph,
            )
        else:
            raise ValueError(
                f"Unsupported vertical interpolation type for qr: {reader.cfg.vert_interp_type}"
            )
    else:
        raise ValueError(
            f"Unsupported reader type for qr interpolation: {type(reader)}"
        )
    qr = jnp.where(qr < 0.0, 0.0, qr)
    q = q.at[idx_qr, :, :].set(qr)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_qr,
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


def regrid_wet_qs(
    block: BLOCK,
    reader: AbstractBackgroundReader,
    regridder: Regridder,
    vert_interp_linear: VerticalLinearInterpolator,
    mute: bool = False
) -> BLOCK:
    idx_qs = block.tracers.idx_qs

    if idx_qs == -1:
        return block

    if not mute:
        log.logger.info("Regrid snow water wet mixing ratio.")
    # =================================================================
    ph = block.state.ph.unwrap()[0, ...]
    q = block.tracers.q.unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q0 = regridder(reader.qs)
        if reader.cfg.vert_interp_type == "native":
            qs = vert_interp_linear(
                x1=reader.plev,
                data1=q0,
                x2=ph,
            )
        elif reader.cfg.vert_interp_type == "grist":
            qs = grist_vert_interp_linear(
                vert_interp_linear=vert_interp_linear,
                nlev_eff=reader.nlev_eff,
                x1=reader.plev,
                data1=q0,
                x2=ph,
            )
        else:
            raise ValueError(
                f"Unsupported vertical interpolation type for qs: {reader.cfg.vert_interp_type}"
            )
    else:
        raise ValueError(
            f"Unsupported reader type for qs interpolation: {type(reader)}"
        )
    qs = jnp.where(qs < 0.0, 0.0, qs)
    q = q.at[idx_qs, :, :].set(qs)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_qs,
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
    reader: AbstractBackgroundReader,
    mute: bool = False
) -> BLOCK:
    if not mute:
        log.logger.info("Calculate surface dry-air pressure.")
    mesh = block.mesh
    ids, ide = mesh.cell_ids, mesh.cell_ide
    kds, kde = mesh.full_kds, mesh.full_kde
    phs = block.state.phs.unwrap()[0, ...]
    ph_lev = block.state.ph_lev.unwrap()[0, ...]
    mgs = block.state.mgs.unwrap()[0, :]
    qm = block.tracers.qm.unwrap()[:, :]
    # =================================================================
    if reader.cfg.vert_interp_type == "native":
        delta_ph = ph_lev[kds+1:kde+2, ids:ide+1] - ph_lev[kds:kde+1, ids:ide+1]
        mgs_ = phs[ids:ide+1] - jnp.sum(
            delta_ph * qm[kds:kde+1, ids:ide+1],
            axis=0
        )
    elif reader.cfg.vert_interp_type == "grist":
        mgs_ = phs[ids:ide+1]
    else:
        raise ValueError(
            f"Unsupported vertical interpolation type for mgs calculation: {reader.cfg.vert_interp_type}"
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


def bkg_calc_dry_qc(block: BLOCK, mute: bool = False) -> BLOCK:
    idx_qc = block.tracers.idx_qc

    if idx_qc == -1:
        return block

    if not mute:
        log.logger.info("Calculate cloud liquid dry mixing ratio.")
    # =================================================================
    q = block.tracers.q.unwrap()[0, :, :, :]
    qm = block.tracers.qm.unwrap()[:, :]
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    qc = q[idx_qc, :, :] / (1 - qm)
    q = q.at[idx_qc, :, :].set(qc)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_qc,
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


def bkg_calc_dry_qi(block: BLOCK, mute: bool = False) -> BLOCK:
    idx_qi = block.tracers.idx_qi

    if idx_qi == -1:
        return block

    if not mute:
        log.logger.info("Calculate cloud ice dry mixing ratio.")
    # =================================================================
    q = block.tracers.q.unwrap()[0, :, :, :]
    qm = block.tracers.qm.unwrap()[:, :]
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    qi = q[idx_qi, :, :] / (1 - qm)
    q = q.at[idx_qi, :, :].set(qi)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_qi,
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


def bkg_calc_dry_nc(block: BLOCK, mute: bool = False) -> BLOCK:
    idx_nc = block.tracers.idx_nc

    if idx_nc == -1:
        return block

    if not mute:
        log.logger.info("Calculate cloud liquid number dry mixing ratio.")
    # =================================================================
    q = block.tracers.q.unwrap()[0, :, :, :]
    qm = block.tracers.qm.unwrap()[:, :]
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    nc = q[idx_nc, :, :] / (1 - qm)
    q = q.at[idx_nc, :, :].set(nc)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_nc,
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


def bkg_calc_dry_ni(block: BLOCK, mute: bool = False) -> BLOCK:
    idx_ni = block.tracers.idx_ni

    if idx_ni == -1:
        return block

    if not mute:
        log.logger.info("Calculate cloud ice number dry mixing ratio.")
    # =================================================================
    q = block.tracers.q.unwrap()[0, :, :, :]
    qm = block.tracers.qm.unwrap()[:, :]
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    ni = q[idx_ni, :, :] / (1 - qm)
    q = q.at[idx_ni, :, :].set(ni)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_ni,
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


def bkg_calc_dry_qr(block: BLOCK, mute: bool = False) -> BLOCK:
    idx_qr = block.tracers.idx_qr

    if idx_qr == -1:
        return block

    if not mute:
        log.logger.info("Calculate rain water dry mixing ratio.")
    # =================================================================
    q = block.tracers.q.unwrap()[0, :, :, :]
    qm = block.tracers.qm.unwrap()[:, :]
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    qr = q[idx_qr, :, :] / (1 - qm)
    q = q.at[idx_qr, :, :].set(qr)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_qr,
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


def bkg_calc_dry_qs(block: BLOCK, mute: bool = False) -> BLOCK:
    idx_qs = block.tracers.idx_qs

    if idx_qs == -1:
        return block

    if not mute:
        log.logger.info("Calculate snow water dry mixing ratio.")
    # =================================================================
    q = block.tracers.q.unwrap()[0, :, :, :]
    qm = block.tracers.qm.unwrap()[:, :]
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    qs = q[idx_qs, :, :] / (1 - qm)
    q = q.at[idx_qs, :, :].set(qs)
    q = fill_field3d_array_halo(
        halo=block.mesh.halo,
        f=q,
        i0=idx_qs,
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
