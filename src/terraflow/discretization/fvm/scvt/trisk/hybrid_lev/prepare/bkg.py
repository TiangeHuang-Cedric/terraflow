from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import jax.numpy as jnp

from terraflow.util import typing, log, math, formula
from terraflow.prepare.readers.era5 import ERA5Reader
from terraflow.prepare.vert_interp import (
    vert_interp_linear,
    vert_interp_log_linear
)
from terraflow.mesh.scvt.interp import latlon_interp_bilinear_column
from terraflow.mesh.scvt.operators import interp_weighted_add_plg2dua
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at
from terraflow.discretization.fvm.scvt.trisk.hybrid_lev import operators as hybrid_op

if TYPE_CHECKING:
    from terraflow.prepare.readers.abstract import AbstractBackgroundReader
    from terraflow.layout.field import StaggeredFieldMetaData
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as BLOCK


Float = typing.Float
Array = typing.Array


def _get_hori_coord(meta: StaggeredFieldMetaData, coord) -> dict[str, Float[Array, "npoint"]]:
    coord_info = meta.get_hori_position(coord).horizontal_coord
    if coord_info is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for {meta.name} meta."
        )
    coord_info["lon"] = jnp.rad2deg(coord_info["lon"])
    coord_info["lat"] = jnp.rad2deg(coord_info["lat"])
    return coord_info


def _get_hori_interp_fun(interp_method: str) -> Callable:
    if interp_method == "bilinear":
        return latlon_interp_bilinear_column
    else:
        raise ValueError(f"Unsupported interpolation method: {interp_method}")


def regrid_phs(block: BLOCK, reader: AbstractBackgroundReader) -> BLOCK:
    log.logger.info("Regrid surface hydrostatic pressure.")
    g = block.para.gravity.unwrap()
    coord = block.coord
    gzs = block.static.gzs.unwrap()
    phs_meta = block.state.phs.meta
    interp_fun = _get_hori_interp_fun(interp_method=reader.hori_interp_method)
    # =================================================================
    if isinstance(reader, ERA5Reader):
        coord_info = _get_hori_coord(phs_meta, coord)
        dst_lon = coord_info["lon"]
        dst_lat = coord_info["lat"]
        z0 = interp_fun(
            src_lat=reader.lat,
            src_lon=reader.lon,
            src_data=reader.z,
            dst_lon=dst_lon,
            dst_lat=dst_lat,
            extrap=reader.hori_extrap
        )
        logp0 = jnp.expand_dims(jnp.log(reader.lev), axis=1)
        logp0 = jnp.tile(logp0, (1, z0.shape[1]))
        phs = vert_interp_linear(
            x1=jnp.flip(z0, axis=0),
            data1=jnp.flip(logp0, axis=0),
            x2=gzs / g,
            left="extrapolate",
            right="extrapolate"
        )
        phs = jnp.exp(phs)
    else:
        raise ValueError(
            f"Unsupported reader type for phs interpolation: {type(reader)}"
        )
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: b.state.phs,
        pytree=block,
        replace=phs,
        at_dims=0,
        at_idxes=0
    )


def bkg_calc_ph(block: BLOCK) -> BLOCK:
    log.logger.info("Calculate hydrostatic pressure on each level.")
    phs = block.state.phs.unwrap()[0, ...]
    full_kds = block.coord.vert.full_kds
    full_kde = block.coord.vert.full_kde
    half_kds = block.coord.vert.half_kds
    half_kde = block.coord.vert.half_kde
    # =================================================================
    ph = block.coord.vert.calc_mg(mgs=phs, ks=full_kds, ke=full_kde)
    ph_lev = block.coord.vert.calc_mg_lev(mgs=phs, ks=half_kds, ke=half_kde)
    # =================================================================
    return tree_at(
        coord=block.coord,
        where=lambda b: (b.state.ph, b.state.ph_lev),
        pytree=block,
        replace=(ph, ph_lev),
        at_dims=(0, 0),
        at_idxes=(0, 0)
    )


def regrid_gz(block: BLOCK, reader: AbstractBackgroundReader) -> BLOCK:
    log.logger.info("Regrid geopotential height.")
    coord = block.coord
    mesh = coord.mesh
    polg_ids = mesh.polg_ids
    polg_ihe = mesh.polg_ihe[0]
    half_kde = coord.vert.half_kde
    full_kds = coord.vert.full_kds
    full_kde = coord.vert.full_kde
    g = block.para.gravity.unwrap()
    gzs = block.static.gzs.unwrap()
    gz = block.state.gz.unwrap()[0, ...]
    ph_lev = block.state.ph_lev.unwrap()[0, ...]
    gz_lev_meta = block.state.gz_lev.meta
    coord_info = _get_hori_coord(gz_lev_meta, coord)
    interp_fun = _get_hori_interp_fun(interp_method=reader.hori_interp_method)
    # =================================================================
    if isinstance(reader, ERA5Reader):
        dst_lon = coord_info["lon"]
        dst_lat = coord_info["lat"]
        z0 = interp_fun(
            src_lat=reader.lat,
            src_lon=reader.lon,
            src_data=reader.z,
            dst_lon=dst_lon,
            dst_lat=dst_lat,
            extrap=reader.hori_extrap
        )
        p0 = reader.lev
        gz_lev = vert_interp_log_linear(
            x1=p0,
            data1=z0,
            x2=ph_lev,
            left="extrapolate",
            right="extrapolate"
        )
        gz_lev = gz_lev * g
        gz_lev = gz_lev.at[half_kde, polg_ids:polg_ihe+1].set(
            gzs[polg_ids:polg_ihe+1]
        )
        gz = gz.at[full_kds:full_kde+1, polg_ids:polg_ihe+1].set(
            0.5 * (
                gz_lev[full_kds:full_kde+1, polg_ids:polg_ihe+1]
                + gz_lev[full_kds+1:full_kde+2, polg_ids:polg_ihe+1]
            )
        )
    else:
        raise ValueError(
            f"Unsupported reader type for gz interpolation: {type(reader)}"
        )
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: (b.state.gz, b.state.gz_lev),
        pytree=block,
        replace=(gz, gz_lev),
        at_dims=(0, 0),
        at_idxes=(0, 0)
    )


def bkg_calc_t(block: BLOCK) -> BLOCK:
    log.logger.info("Calculate temperature.")
    coord = block.coord
    mesh = coord.mesh
    vert = coord.vert
    ids, ihe = mesh.polg_ids, mesh.polg_ihe[0]
    kds, kde = vert.full_kds, vert.full_kde
    rd = block.para.rd.unwrap()
    idx_qv = block.tracers.idx_qv
    ph_lev = block.state.ph_lev.unwrap()[0, ...]
    gz_lev = block.state.gz_lev.unwrap()[0, ...]
    qm = block.tracers.qm.unwrap()[0, :, :]
    t = block.aux.t.unwrap()
    # =================================================================
    delta_gz = (
        gz_lev[kds:kde+1, ids:ihe+1]
        - gz_lev[kds+1:kde+2, ids:ihe+1]
    )
    log_ratio = jnp.log(
        ph_lev[kds+1:kde+2, ids:ihe+1]
        / ph_lev[kds:kde+1, ids:ihe+1]
    )
    tv = delta_gz / (rd * log_ratio)
    if idx_qv == -1:
        t = t.at[kds:kde+1, ids:ihe+1].set(tv)
    else:
        q = block.tracers.q.unwrap()[0, idx_qv, :, :]
        t = t.at[kds:kde+1, ids:ihe+1].set(
            formula.temperature_from_virtual_temperature_and_dry_mixing_ratio(
                tv,
                q[kds:kde+1, ids:ihe+1],
                qm[kds:kde+1, ids:ihe+1],
                block.para
            )
        )
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: b.aux.t,
        pytree=block,
        replace=t,
    )


def regrid_t(block: BLOCK, reader: AbstractBackgroundReader) -> BLOCK:
    log.logger.info("Regrid temperature.")
    coord = block.coord
    t_meta = block.aux.t.meta
    ph = block.state.ph.unwrap()[0, ...]
    coord_info = _get_hori_coord(t_meta, coord)
    dst_lon = coord_info["lon"]
    dst_lat = coord_info["lat"]
    interp_fun = _get_hori_interp_fun(interp_method=reader.hori_interp_method)
    # =================================================================
    if isinstance(reader, ERA5Reader):
        t0 = interp_fun(
            src_lat=reader.lat,
            src_lon=reader.lon,
            src_data=reader.t,
            dst_lon=dst_lon,
            dst_lat=dst_lat,
            extrap=reader.hori_extrap
        )
        t = vert_interp_log_linear(
            x1=reader.lev,
            data1=t0,
            x2=ph,
            left="extrapolate",
            right="extrapolate"
        )
    else:
        raise ValueError(
            f"Unsupported reader type for t interpolation: {type(reader)}"
        )
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: b.aux.t,
        pytree=block,
        replace=t,
    )


def regrid_un_from_uv(block: BLOCK, reader: AbstractBackgroundReader) -> BLOCK:
    coord = block.coord
    un_meta = block.state.un.meta
    coord_info = _get_hori_coord(un_meta, coord)
    cart = coord_info["cart"]
    edge_normal = coord_info["nr_orient"]
    # =================================================================
    u = _regrid_u(block=block, reader=reader)
    v = _regrid_v(block=block, reader=reader)
    vec_cart = math.convert_vector_ltln2cart(u, v, cart)
    un = jnp.sum(vec_cart * edge_normal, axis=-1)
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: b.state.un,
        pytree=block,
        replace=un,
        at_dims=0,
        at_idxes=0
    )


def _regrid_u(block: BLOCK, reader: AbstractBackgroundReader) -> Float[Array, "nlev nedge"]:
    log.logger.info("Regrid u wind component.")
    un_meta = block.state.un.meta
    edd_v = block.coord.mesh._edd.v
    cell_0 = edd_v[:, 0]
    cell_1 = edd_v[:, 1]
    ph = block.state.ph.unwrap()[0, ...]
    coord_info = _get_hori_coord(un_meta, block.coord)
    dst_lon = coord_info["lon"]
    dst_lat = coord_info["lat"]
    interp_fun = _get_hori_interp_fun(interp_method=reader.hori_interp_method)
    # =================================================================
    ph_e = 0.5 * (ph[:, cell_0] + ph[:, cell_1])
    if isinstance(reader, ERA5Reader):
        u = interp_fun(
            src_lat=reader.lat,
            src_lon=reader.lon,
            src_data=reader.u,
            dst_lon=dst_lon,
            dst_lat=dst_lat,
            extrap=reader.hori_extrap
        )
        u = vert_interp_linear(
            x1=reader.lev,
            data1=u,
            x2=ph_e,
            left="extrapolate",
            right="extrapolate"
        )
    else:
        raise ValueError(
            f"Unsupported reader type for u interpolation: {type(reader)}"
        )
    # =================================================================
    return u


def _regrid_v(block: BLOCK, reader: AbstractBackgroundReader) -> Float[Array, "nlev nedge"]:
    log.logger.info("Regrid v wind component.")
    un_meta = block.state.un.meta
    edd_v = block.coord.mesh._edd.v
    cell_0 = edd_v[:, 0]
    cell_1 = edd_v[:, 1]
    ph = block.state.ph.unwrap()[0, ...]
    coord_info = _get_hori_coord(un_meta, block.coord)
    dst_lon = coord_info["lon"]
    dst_lat = coord_info["lat"]
    interp_fun = _get_hori_interp_fun(interp_method=reader.hori_interp_method)
    # =================================================================
    ph_e = 0.5 * (ph[:, cell_0] + ph[:, cell_1])
    if isinstance(reader, ERA5Reader):
        v = interp_fun(
            src_lat=reader.lat,
            src_lon=reader.lon,
            src_data=reader.v,
            dst_lon=dst_lon,
            dst_lat=dst_lat,
            extrap=reader.hori_extrap
        )
        v = vert_interp_linear(
            x1=reader.lev,
            data1=v,
            x2=ph_e,
            left="extrapolate",
            right="extrapolate"
        )
    else:
        raise ValueError(
            f"Unsupported reader type for v interpolation: {type(reader)}"
        )
    # =================================================================
    return v


def regrid_wet_qv(block: BLOCK, reader: AbstractBackgroundReader, mute: bool = False) -> BLOCK:
    idx_qv = block.tracers.idx_qv
    if idx_qv == -1:
        return block

    if not mute:
        log.logger.info("Regrid water vapor wet mixing ratio.")
    # =================================================================
    coord = block.coord
    q_meta = block.tracers.q.meta
    ph = block.state.ph.unwrap()[0, ...]
    q = block.tracers.q.unwrap()[0, ...]
    coord_info = _get_hori_coord(q_meta, coord)
    dst_lon = coord_info["lon"]
    dst_lat = coord_info["lat"]
    interp_fun = _get_hori_interp_fun(interp_method=reader.hori_interp_method)
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q0 = interp_fun(
            src_lat=reader.lat,
            src_lon=reader.lon,
            src_data=reader.qv,
            dst_lon=dst_lon,
            dst_lat=dst_lat,
            extrap=reader.hori_extrap
        )
        qv = vert_interp_linear(
            x1=reader.lev,
            data1=q0,
            x2=ph,
            left="extrapolate",
            right="extrapolate"
        )
    else:
        raise ValueError(
            f"Unsupported reader type for qv interpolation: {type(reader)}"
        )
    qv = jnp.where(qv < 0.0, 0.0, qv)
    q = q.at[idx_qv, :, :].set(qv)
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: b.tracers.q,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def regrid_wet_qc(block: BLOCK, reader: AbstractBackgroundReader, mute: bool = False) -> BLOCK:
    idx_qc = block.tracers.idx_qc
    if idx_qc == -1:
        return block

    if not mute:
        log.logger.info("Regrid cloud liquid wet mixing ratio.")
    # =================================================================
    coord = block.coord
    q_meta = block.tracers.q.meta
    ph = block.state.ph.unwrap()[0, ...]
    q = block.tracers.q.unwrap()[0, ...]
    coord_info = _get_hori_coord(q_meta, coord)
    dst_lon = coord_info["lon"]
    dst_lat = coord_info["lat"]
    interp_fun = _get_hori_interp_fun(interp_method=reader.hori_interp_method)
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q0 = interp_fun(
            src_lat=reader.lat,
            src_lon=reader.lon,
            src_data=reader.qc,
            dst_lon=dst_lon,
            dst_lat=dst_lat,
            extrap=reader.hori_extrap
        )
        qc = vert_interp_linear(
            x1=reader.lev,
            data1=q0,
            x2=ph,
            left="extrapolate",
            right="extrapolate"
        )
    else:
        raise ValueError(
            f"Unsupported reader type for qc interpolation: {type(reader)}"
        )
    qc = jnp.where(qc < 0.0, 0.0, qc)
    q = q.at[idx_qc, :, :].set(qc)
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: b.tracers.q,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def regrid_wet_qi(block: BLOCK, reader: AbstractBackgroundReader, mute: bool = False) -> BLOCK:
    idx_qi = block.tracers.idx_qi
    if idx_qi == -1:
        return block

    if not mute:
        log.logger.info("Regrid cloud ice wet mixing ratio.")
    # =================================================================
    coord = block.coord
    q_meta = block.tracers.q.meta
    ph = block.state.ph.unwrap()[0, ...]
    q = block.tracers.q.unwrap()[0, ...]
    coord_info = _get_hori_coord(q_meta, coord)
    dst_lon = coord_info["lon"]
    dst_lat = coord_info["lat"]
    interp_fun = _get_hori_interp_fun(interp_method=reader.hori_interp_method)
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q0 = interp_fun(
            src_lat=reader.lat,
            src_lon=reader.lon,
            src_data=reader.qi,
            dst_lon=dst_lon,
            dst_lat=dst_lat,
            extrap=reader.hori_extrap
        )
        qi = vert_interp_linear(
            x1=reader.lev,
            data1=q0,
            x2=ph,
            left="extrapolate",
            right="extrapolate"
        )
    else:
        raise ValueError(
            f"Unsupported reader type for qi interpolation: {type(reader)}"
        )
    qi = jnp.where(qi < 0.0, 0.0, qi)
    q = q.at[idx_qi, :, :].set(qi)
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: b.tracers.q,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def regrid_wet_nc(block: BLOCK, reader: AbstractBackgroundReader, mute: bool = False) -> BLOCK:
    idx_nc = block.tracers.idx_nc

    if idx_nc == -1:
        return block

    if not mute:
        log.logger.info("Regrid cloud liquid number wet mixing ratio.")
    # =================================================================
    coord = block.coord
    q = block.tracers.q.unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q_nc = q[idx_nc, :, :]
    else:
        q_nc = q[idx_nc, :, :]
    q_nc = jnp.where(q_nc < 0.0, 0.0, q_nc)
    q = q.at[idx_nc, :, :].set(q_nc)
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: b.tracers.q,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def regrid_wet_ni(block: BLOCK, reader: AbstractBackgroundReader, mute: bool = False) -> BLOCK:
    idx_ni = block.tracers.idx_ni

    if idx_ni == -1:
        return block

    if not mute:
        log.logger.info("Regrid cloud ice number wet mixing ratio.")
    # =================================================================
    coord = block.coord
    q = block.tracers.q.unwrap()[0, ...]
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q_ni = q[idx_ni, :, :]
    else:
        q_ni = q[idx_ni, :, :]
    q_ni = jnp.where(q_ni < 0.0, 0.0, q_ni)
    q = q.at[idx_ni, :, :].set(q_ni)
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: b.tracers.q,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def regrid_wet_qr(block: BLOCK, reader: AbstractBackgroundReader, mute: bool = False) -> BLOCK:
    idx_qr = block.tracers.idx_qr

    if idx_qr == -1:
        return block

    if not mute:
        log.logger.info("Regrid rain water wet mixing ratio.")
    # =================================================================
    coord = block.coord
    q_meta = block.tracers.q.meta
    ph = block.state.ph.unwrap()[0, ...]
    q = block.tracers.q.unwrap()[0, ...]
    coord_info = _get_hori_coord(q_meta, coord)
    dst_lon = coord_info["lon"]
    dst_lat = coord_info["lat"]
    interp_fun = _get_hori_interp_fun(interp_method=reader.hori_interp_method)
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q0 = interp_fun(
            src_lat=reader.lat,
            src_lon=reader.lon,
            src_data=reader.qr,
            dst_lon=dst_lon,
            dst_lat=dst_lat,
            extrap=reader.hori_extrap
        )
        qr = vert_interp_linear(
            x1=reader.lev,
            data1=q0,
            x2=ph,
            left="extrapolate",
            right="extrapolate"
        )
    else:
        raise ValueError(
            f"Unsupported reader type for qr interpolation: {type(reader)}"
        )
    qr = jnp.where(qr < 0.0, 0.0, qr)
    q = q.at[idx_qr, :, :].set(qr)
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: b.tracers.q,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def regrid_wet_qs(block: BLOCK, reader: AbstractBackgroundReader, mute: bool = False) -> BLOCK:
    idx_qs = block.tracers.idx_qs

    if idx_qs == -1:
        return block

    if not mute:
        log.logger.info("Regrid snow water wet mixing ratio.")
    # =================================================================
    coord = block.coord
    q_meta = block.tracers.q.meta
    ph = block.state.ph.unwrap()[0, ...]
    q = block.tracers.q.unwrap()[0, ...]
    coord_info = _get_hori_coord(q_meta, coord)
    dst_lon = coord_info["lon"]
    dst_lat = coord_info["lat"]
    interp_fun = _get_hori_interp_fun(interp_method=reader.hori_interp_method)
    # =================================================================
    if isinstance(reader, ERA5Reader):
        q0 = interp_fun(
            src_lat=reader.lat,
            src_lon=reader.lon,
            src_data=reader.qs,
            dst_lon=dst_lon,
            dst_lat=dst_lat,
            extrap=reader.hori_extrap
        )
        qs = vert_interp_linear(
            x1=reader.lev,
            data1=q0,
            x2=ph,
            left="extrapolate",
            right="extrapolate"
        )
    else:
        raise ValueError(
            f"Unsupported reader type for qs interpolation: {type(reader)}"
        )
    qs = jnp.where(qs < 0.0, 0.0, qs)
    q = q.at[idx_qs, :, :].set(qs)
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: b.tracers.q,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )


def bkg_calc_mgs(block: BLOCK, mute: bool = False) -> BLOCK:
    if not mute:
        log.logger.info("Calculate surface dry-air pressure.")
    coord = block.coord
    mesh = coord.mesh
    vert = coord.vert
    ids, ide = mesh.polg_ids, mesh.polg_ide
    kds, kde = vert.full_kds, vert.full_kde
    phs = block.state.phs.unwrap()[0, ...]
    ph_lev = block.state.ph_lev.unwrap()[0, ...]
    qm = block.tracers.qm.unwrap()[0, ...]
    # =================================================================
    delta_ph = ph_lev[kds+1:kde+2, ids:ide+1] - ph_lev[kds:kde+1, ids:ide+1]
    mgs = phs - jnp.sum(delta_ph * qm, axis=0)
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: b.state.mgs,
        pytree=block,
        replace=mgs,
        at_dims=0,
        at_idxes=0
    )


def bkg_calc_mg(block: BLOCK, mute: bool = False) -> BLOCK:
    if not mute:
        log.logger.info("Calculate dry-air pressure.")

    coord = block.coord
    mesh = coord.mesh
    ids = mesh.polg_ids
    ihe = mesh.polg_ihe[0]
    full_kds = coord.vert.full_kds
    full_kde = coord.vert.full_kde
    half_kds = coord.vert.half_kds
    half_kde = coord.vert.half_kde
    mgs = block.state.mgs.unwrap()[0, ...]
    mg_lev = block.state.mg_lev.unwrap()[0, ...]
    mg = block.state.mg.unwrap()[0, ...]
    dmg = block.state.dmg.unwrap()[0, ...]
    dmg_lev = block.state.dmg_lev.unwrap()[0, ...]
    dmg_e = block.state.dmg_e.unwrap()[0, ...]
    dmg_vtx = block.state.dmg_vtx.unwrap()[0, ...]
    # =================================================================
    mg_lev = mg_lev.at[half_kds:half_kde+1, ids:ihe+1].set(
        coord.vert.calc_mg_lev(
            mgs[ids:ihe+1],
            ks=half_kds,
            ke=half_kde
        )
    )
    mg = mg.at[full_kds:full_kde+1, ids:ihe+1].set(
        0.5 * (
            mg_lev[full_kds:full_kde+1, ids:ihe+1]
            + mg_lev[full_kds+1:full_kde+2, ids:ihe+1]
        )
    )
    dmg = hybrid_op.delta_at_full_lev(
        coord=coord,
        x_lev=mg_lev,
        dx=dmg,
    )
    dmg_lev = hybrid_op.delta_at_half_lev(
        coord=coord,
        x=mg,
        x_lev=mg_lev,
        dx_lev=dmg_lev
    )
    dmg_e = hybrid_op.InterpMethod().cell2edge(
        coord=coord,
        x=dmg,
        y=dmg_e
    )
    dmg_vtx = interp_weighted_add_plg2dua(
        x_at_primal_cell=dmg,
        coord=coord,
        x_at_dual_cell=dmg_vtx,
    )
    return tree_at(
        coord=coord,
        where=lambda b: (
            b.state.mg, b.state.mg_lev,
            b.state.dmg, b.state.dmg_lev,
            b.state.dmg_e, b.state.dmg_vtx
        ),
        pytree=block,
        replace=(mg, mg_lev, dmg, dmg_lev, dmg_e, dmg_vtx),
        at_dims=(0, 0, 0, 0, 0, 0),
        at_idxes=(0, 0, 0, 0, 0, 0)
    )


def bkg_calc_tv(block: BLOCK) -> BLOCK:
    log.logger.info("Calculate virtual temperature.")
    idx_qv = block.tracers.idx_qv
    t = block.aux.t.unwrap()
    # =================================================================
    if idx_qv == -1:
        tv = t
    else:
        qv = block.tracers.q.unwrap()[0, idx_qv, :, :]
        qm = block.tracers.qm.unwrap()[0, :, :]
        tv = formula.virtual_temperature_from_dry_mixing_ratio(
            t=t,
            qv=qv,
            qm=qm,
            para=block.para
        )
    return tree_at(
        coord=block.coord,
        where=lambda b: b.aux.tv,
        pytree=block,
        replace=tv,
    )


def bkg_calc_pt(block: BLOCK) -> BLOCK:
    log.logger.info("Calculate modified potential temperature.")
    coord = block.coord
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
                device=block.para.dev(),
                dtype=block.para.dev.float_type
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
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: b.state.pt,
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
    qm = block.tracers.qm.unwrap()[0, :, :]
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    qv = q[idx_qv, :, :] / (1 - qm)
    q = q.at[idx_qv, :, :].set(qv)
    # =================================================================
    return tree_at(
        coord=block.coord,
        where=lambda b: b.tracers.q,
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
    qm = block.tracers.qm.unwrap()[0, :, :]
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    qc = q[idx_qc, :, :] / (1 - qm)
    q = q.at[idx_qc, :, :].set(qc)
    # =================================================================
    return tree_at(
        coord=block.coord,
        where=lambda b: b.tracers.q,
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
    qm = block.tracers.qm.unwrap()[0, :, :]
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    qi = q[idx_qi, :, :] / (1 - qm)
    q = q.at[idx_qi, :, :].set(qi)
    # =================================================================
    return tree_at(
        coord=block.coord,
        where=lambda b: b.tracers.q,
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
    qm = block.tracers.qm.unwrap()[0, :, :]
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    nc = q[idx_nc, :, :] / (1 - qm)
    q = q.at[idx_nc, :, :].set(nc)
    # =================================================================
    return tree_at(
        coord=block.coord,
        where=lambda b: b.tracers.q,
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
    qm = block.tracers.qm.unwrap()[0, :, :]
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    ni = q[idx_ni, :, :] / (1 - qm)
    q = q.at[idx_ni, :, :].set(ni)
    # =================================================================
    return tree_at(
        coord=block.coord,
        where=lambda b: b.tracers.q,
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
    qm = block.tracers.qm.unwrap()[0, :, :]
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    qr = q[idx_qr, :, :] / (1 - qm)
    q = q.at[idx_qr, :, :].set(qr)
    # =================================================================
    return tree_at(
        coord=block.coord,
        where=lambda b: b.tracers.q,
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
    qm = block.tracers.qm.unwrap()[0, :, :]
    # NOTE: qm is currently the total wet mixing ratio of water substances.
    qs = q[idx_qs, :, :] / (1 - qm)
    q = q.at[idx_qs, :, :].set(qs)
    # =================================================================
    return tree_at(
        coord=block.coord,
        where=lambda b: b.tracers.q,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0
    )
