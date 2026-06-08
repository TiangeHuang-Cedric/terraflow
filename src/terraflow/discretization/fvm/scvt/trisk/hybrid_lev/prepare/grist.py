from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp

from terraflow.util import typing, log, math, formula
from terraflow.prepare.vert_interp import (
    VerticalLinearInterpolator,
    grist_vert_interp_linear
)
from terraflow.mesh.scvt.operators import interp_weighted_add_plg2dua
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at
from terraflow.discretization.fvm.scvt.trisk.hybrid_lev import operators as hybrid_op
from . import ref
from .topo_smooth import topo_smooth, calc_zs_slope

if TYPE_CHECKING:
    from terraflow.prepare.readers.grist import GRISTStaticDataReader, GRISTBackgroundReader
    from terraflow.layout.field import StaggeredFieldMetaData
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as BLOCK


Int = typing.Int
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


def prepare_static(
    block: BLOCK,
    reader: GRISTStaticDataReader,
) -> BLOCK:
    # =====================================
    block = tree_at(
        coord=block.coord,
        where=lambda b: (
            b.static.gzs.data,
            b.static.landmask.data,
        ),
        pytree=block,
        replace=(reader.phis, reader.mask)
    )
    # =====================================
    gzs = block.static.gzs.unwrap()
    grad_zs = block.static.grad_zs.unwrap()
    grad_zs = calc_zs_slope(
        block,
        gzs.reshape(1, -1),
        grad_zs.reshape(1, -1)
    ).reshape(-1)
    block = tree_at(
        coord=block.coord,
        where=lambda b: b.static.grad_zs.data,
        pytree=block,
        replace=grad_zs
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
        block = block.calc_qm(itime_str="old")
        block = bkg_calc_dry_qv(block=block, mute=False)
        block = block.calc_qm(itime_str="old")
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
            block = block.calc_qm(itime_str="old")
            # Stage 2:
            block = bkg_calc_mgs(block=block, reader=reader, mute=cyc != 0)
            block = bkg_calc_mg(block=block, mute=cyc != 0)
            block = bkg_calc_dry_qv(block=block, mute=cyc != 0)
            block = block.calc_qm(itime_str="old")
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
    return tree_at(
        coord=block.coord,
        where=lambda b: b.state.phs.data,
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


def calc_ph(block: BLOCK) -> BLOCK:
    mesh = block.coord.mesh
    vert = block.coord.vert
    ids, ihe = mesh.polg_ids, mesh.polg_ihe[0]
    full_kds, full_kde = vert.full_kds, vert.full_kde
    half_kds, half_kde = vert.half_kds, vert.half_kde
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
        dmg_ = dmg[kds:kde, ids:ihe+1]
        qm_ = qm[kds:kde, ids:ihe+1]
        moist_dmg = dmg_ * (1.0 + qm_)
        ph_lev = ph_lev.at[kds:kde+1, ids:ihe+1].set(
            mg_lev[kds, :] + jnp.cumulative_sum(
                moist_dmg,
                axis=0,
                include_initial=True
            )
        )
    else:
        ph_lev = ph_lev.at[kds:kde+1, ids:ihe+1].set(
            mg_lev[kds, :] + jnp.cumulative_sum(
                dmg[kds:kde, ids:ihe+1],
                axis=0,
                include_initial=True
            )
        )
    # =======================================================
    # Calculate ph at full level
    kds, kde = full_kds, full_kde
    ph = ph.at[kds:kde+1, ids:ihe+1].set(
        0.5 * (
            ph_lev[kds:kde+1, ids:ihe+1]
            + ph_lev[kds+1:kde+2, ids:ihe+1]
        )
    )
    # =======================================================
    # Calculate phs
    kde = half_kde
    phs = phs.at[ids:ihe+1].set(ph_lev[kde, ids:ihe+1])
    # =======================================================
    return tree_at(
        coord=block.coord,
        where=lambda b: (b.state.phs, b.state.ph, b.state.ph_lev),
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
    coord = block.coord
    un_meta = block.state.un.meta
    coord_info = _get_hori_coord(un_meta, coord)
    cart = coord_info["cart"]
    edge_normal = coord_info["nr_orient"]
    # =================================================================
    u = _regrid_u(
        block=block,
        reader=reader,
        vert_interp_linear=vert_interp_linear
    )
    v = _regrid_v(
        block=block,
        reader=reader,
        vert_interp_linear=vert_interp_linear
    )
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


def _regrid_u(
    block: BLOCK,
    reader: GRISTBackgroundReader,
    vert_interp_linear: VerticalLinearInterpolator
) -> Float[Array, "nlev nedge"]:
    log.logger.info("Regrid u wind component.")
    edd_v = block.coord.mesh._edd.v
    cell_0 = edd_v[:, 0]
    cell_1 = edd_v[:, 1]
    ph = block.state.ph.unwrap()[0, ...]
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
    u = 0.5 * (u_cell[:, cell_0] + u_cell[:, cell_1])
    # =================================================================
    return u


def _regrid_v(
    block: BLOCK,
    reader: GRISTBackgroundReader,
    vert_interp_linear: VerticalLinearInterpolator
) -> Float[Array, "nlev nedge"]:
    log.logger.info("Regrid v wind component.")
    edd_v = block.coord.mesh._edd.v
    cell_0 = edd_v[:, 0]
    cell_1 = edd_v[:, 1]
    ph = block.state.ph.unwrap()[0, ...]
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
    v = 0.5 * (v_cell[:, cell_0] + v_cell[:, cell_1])
    # =================================================================
    return v


def regrid_t(
    block: BLOCK,
    reader: GRISTBackgroundReader,
    vert_interp_linear: VerticalLinearInterpolator
) -> BLOCK:
    log.logger.info("Regrid temperature.")
    coord = block.coord
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
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: b.aux.t,
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
    coord = block.coord
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
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: b.tracers.q,
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
    coord = block.coord
    phs = block.state.phs.unwrap()[0, ...]
    if reader.cfg.vert_interp_type == "grist":
        mgs = phs
    elif reader.cfg.vert_interp_type == "native":
        mesh = coord.mesh
        vert = coord.vert
        ids, ide = mesh.polg_ids, mesh.polg_ide
        kds, kde = vert.full_kds, vert.full_kde
        ph_lev = block.state.ph_lev.unwrap()[0, ...]
        qm = block.tracers.qm.unwrap()[:, :]
        # =================================================================
        delta_ph = ph_lev[kds+1:kde+2, ids:ide+1] - ph_lev[kds:kde+1, ids:ide+1]
        mgs = phs - jnp.sum(delta_ph * qm[kds:kde+1, ids:ide+1], axis=0)
    else:
        raise ValueError(
            f"Unsupported vertical interpolation type: {reader.cfg.vert_interp_type}"
        )
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
        qm = block.tracers.qm.unwrap()[:, :]
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
    qm = block.tracers.qm.unwrap()[:, :]
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
