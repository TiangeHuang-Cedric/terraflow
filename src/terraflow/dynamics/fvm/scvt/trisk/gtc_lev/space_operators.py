from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import equinox as eqx

from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv import upwind
from terraflow.mesh.scvt.parallel import fill_field2d_array_halo
from terraflow.util import formula
from terraflow.util.pytree import tree_at
from terraflow.mesh.scvt import operators as op
from terraflow.mesh.scvt import interp
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.tracer import tracer_calc_qm
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv import interface

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block

    IntType = typing.IntType
    FloatType = typing.FloatType
    INT = typing.Int
    FLT = typing.Float
    Array = typing.Array
    
    
def operators_prepare_1(block: Block, itime: INT[Array, ""] | int, dt: float) -> Block:
    mode  = block.mode
    const = block.const
    cfg   = block.cfg
    if mode.is_baroclinic():
        block = calc_mg(block, itime)
    block = calc_dmg(block, itime)
    block = tracer_calc_qm(block, "old")
    if mode.is_baroclinic():
        block = calc_ph    (block, itime)
    if mode.nonhydrostatic and not mode.restart:
        if (jnp.sum(block.state.p.d[itime, :, :]) == 0):
            # Set pressure to hydrostatic pressure in the initial condition.
            ph     = block.state.ph    .d[itime, :, :]
            ph_lev = block.state.ph_lev.d[itime, :, :] 
            block = tree_at(
                where=lambda b: (b.state.p, b.state.ph),
                pytree=block,
                replace=(ph, ph_lev),
                at_dims=(0, 0),
                at_idxes=(itime, itime)
            )
    if mode.is_baroclinic():
        block = calc_t(block, itime                   )
    block = calc_mf   (block, itime, dt                )
    block = calc_ke   (block, itime,     const.total_substeps)
    block = calc_ut   (block, itime                    )
    block = calc_pv   (block, itime,     const.total_substeps)
    block = interp_pv (block, itime, dt, const.total_substeps)
    if mode.is_baroclinic() and (mode.is_hydrostatic() or cfg.init_hydrostatic_gz):
        block = calc_gz(block, itime)
    if mode.is_baroclinic():
        block = calc_rhod(block, itime)
    return block


def operators_prepare_2(
    block: Block, 
    itime: INT[Array, ""] | int, 
    dt: float,
    pass_tag: str, 
    substep: int
) -> Block:
    mode = block.mode
    if pass_tag == "all_pass":
        if mode.is_baroclinic(): 
            block = calc_t   (block, itime)
        block = calc_mf      (block, itime, dt)
        block = calc_ke      (block, itime,     substep)
        block = calc_ut      (block, itime)
        block = calc_pv      (block, itime,     substep)
        block = interp_pv    (block, itime, dt, substep)
        if mode.is_hydrostatic():
            block = calc_gz  (block, itime)
        if mode.is_baroclinic():
            block = calc_rhod(block, itime)
    elif pass_tag == "forward_pass":
        if mode.is_baroclinic(): 
            block = calc_t   (block, itime)
        block = calc_mf      (block, itime, dt)
        block = calc_ke      (block, itime,     substep)
        block = calc_ut      (block, itime)
        block = calc_pv      (block, itime,     substep)
        block = interp_pv    (block, itime, dt, substep)
    elif pass_tag == "backward_pass":
        if mode.is_hydrostatic(): 
            block = calc_gz  (block, itime)
        if mode.is_baroclinic():
            block = calc_rhod(block, itime)
    else:
        raise ValueError(f"Unsupported pass_tag {pass_tag}!")
    return block


def calc_mg(block: Block, star_idx: INT[Array, ""] | int) -> Block:
    mesh = block.mesh
    ids, ihe = mesh.cell_ids, mesh.cell_ihe[0]
    full_kds, full_kde = mesh.full_kds, mesh.full_kde
    half_kds, half_kde = mesh.half_kds, mesh.half_kde
    mgs    = block.state.mgs   .unwrap()[star_idx, :   ]
    mg_lev = block.state.mg_lev.unwrap()[star_idx, :, :]
    mg     = block.state.mg    .unwrap()[star_idx, :, :]
    # ==========================================================
    kds, kde = half_kds, half_kde
    mg_lev = mg_lev.at[kds:kde+1, ids:ihe+1].set(
        block.vert_coord.calc_mg_lev(
            mgs[ids:ihe+1],
            ks=kds,
            ke=kde
        )
    )
    kds, kde = full_kds, full_kde
    mg = mg.at[kds:kde+1, ids:ihe+1].set(
        0.5 * (
            mg_lev[kds:kde+1, ids:ihe+1]
            + mg_lev[kds+1:kde+2, ids:ihe+1]
        )
    )
    # ==========================================================
    return tree_at(
        where=lambda b: (b.state.mg.d, b.state.mg_lev.d),
        pytree=block,
        replace=(mg, mg_lev),
        at_dims=(0, 0),
        at_idxes=(star_idx, star_idx)
    )
    
def calc_ph(block: Block, star_idx: INT[Array, ""] | int) -> Block:
    mesh = block.mesh
    ids, ihe = mesh.cell_ids, mesh.cell_ihe[0]
    full_kds, full_kde = mesh.full_kds, mesh.full_kde
    half_kds, half_kde = mesh.half_kds, mesh.half_kde
    mg_lev = block.state  .mg_lev.unwrap()[star_idx, :, :]
    dmg    = block.state  .dmg   .unwrap()[star_idx, :, :]
    ph_lev = block.state  .ph_lev.unwrap()[star_idx, :, :]
    ph     = block.state  .ph    .unwrap()[star_idx, :, :]
    phs    = block.state  .phs   .unwrap()[star_idx, :   ]
    qm     = block.tracers.qm    .unwrap()
    # =======================================================
    # Calculate ph at half level
    kds, kde = half_kds, half_kde
    dmg_ = dmg[kds:kde, ids:ihe+1]
    qm_ = qm[kds:kde, ids:ihe+1]
    moist_dmg = dmg_ * (1.0 + qm_)
    ph_lev = ph_lev.at[kds:kde+1, ids:ihe+1].set(
        mg_lev[kds, ids:ihe+1] + jnp.cumulative_sum(
            moist_dmg,
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
    if block.mode.is_hydrostatic():
        return tree_at(
            where=lambda b: (
                b.state.phs.d, b.state.ph.d, b.state.ph_lev.d, 
                b.state.ps.d, b.state.p.d, b.state.p_lev.d
            ),
            pytree=block,
            replace=(phs, ph, ph_lev, phs, ph, ph_lev),
            at_dims=(0, 0, 0, 0, 0, 0),
            at_idxes=(star_idx, star_idx, star_idx, star_idx, star_idx, star_idx)
        )
    else:
        return tree_at(
            where=lambda b: (b.state.phs.d, b.state.ph.d, b.state.ph_lev.d),
            pytree=block,
            replace=(phs, ph, ph_lev),
            at_dims=(0, 0, 0),
            at_idxes=(star_idx, star_idx, star_idx)
        )
    
    
def calc_t(block: Block, star_idx: INT[Array, ""] | int) -> Block:
    mesh = block.mesh
    ids, ihe = mesh.cell_ids, mesh.cell_ihe[0]
    full_kds, full_kde = mesh.full_kds, mesh.full_kde
    idx_qv = block.tracers.idx_qv
    pt = block.state.pt.unwrap()[star_idx, :, :]
    p  = block.state.p .unwrap()[star_idx, :, :]
    t  = block.aux  .t .unwrap()
    tv = block.aux  .tv.unwrap()
    # ==========================================================
    if idx_qv >= 0:
        old_idx = block.adv_batches["moist"].get_time_idx(itime_str="old")
        q = block.tracers.q.unwrap()[old_idx, idx_qv, :, :]
        rd_o_cpd = block.para.rd_o_cpd.unwrap()
        t = t.at[full_kds:full_kde+1, ids:ihe+1].set(
            formula.temperature(
                pt=pt[full_kds:full_kde+1, ids:ihe+1],
                p=p[full_kds:full_kde+1, ids:ihe+1],
                qv=q[full_kds:full_kde+1, ids:ihe+1],
                para=block.para
            )
        )
        tv = tv.at[full_kds:full_kde+1, ids:ihe+1].set(
            formula.virtual_temperature_from_modified_potential_temperature(
                pt=pt[full_kds:full_kde+1, ids:ihe+1],
                pk=p[full_kds:full_kde+1, ids:ihe+1]**rd_o_cpd,
                qm=q[full_kds:full_kde+1, ids:ihe+1],
                para=block.para
            )
        )
    else:
        t = t.at[full_kds:full_kde+1, ids:ihe+1].set(
            formula.temperature(
                pt=pt[full_kds:full_kde+1, ids:ihe+1],
                p=p[full_kds:full_kde+1, ids:ihe+1],
                qv=jnp.asarray(0.0, dtype=pt.dtype),
                para=block.para
            )
        )
        tv = tv.at[full_kds:full_kde+1, ids:ihe+1].set(
            t[full_kds:full_kde+1, ids:ihe+1]
        )
    # ==========================================================
    return tree_at(
        where=lambda b: (b.aux.t.d, b.aux.tv.d),
        pytree=block,
        replace=(t, tv)
    )


def calc_rhod(block: Block, star_idx: INT[Array, ""] | int) -> Block:
    mesh = block.mesh
    ids, ihe = mesh.cell_ids, mesh.cell_ihe[0]
    kds, kde = mesh.full_kds, mesh.full_kde
    gz_lev = block.state.gz_lev.unwrap()[star_idx, :, :]
    dmg    = block.state.dmg   .unwrap()[star_idx, :, :]
    rhod   = block.aux  .rhod  .unwrap()
    # ==========================================================
    dgz = gz_lev[kds:kde+1, ids:ihe+1] - gz_lev[kds+1:kde+2, ids:ihe+1]
    rhod = rhod.at[kds:kde+1, ids:ihe+1].set(
        dmg[kds:kde+1, ids:ihe+1] / dgz
    )
    # ==========================================================
    return tree_at(
        where=lambda b: b.aux.rhod.d,
        pytree=block,
        replace=rhod
    )
    
    
def calc_mfz(block: Block, star_idx: INT[Array, ""] | int) -> Block:
    half_kds = block.mesh.half_kds
    half_kde = block.mesh.half_kde
    polg_ids = block.mesh.cell_ids
    polg_ide = block.mesh.cell_ide
    dmgsdt  = block.tend.dmgsdt .unwrap()
    dmf     = block.aux .dmf    .unwrap()
    mfz_lev = block.aux .mfz_lev.unwrap()
    # ==========================================================
    dmgdt_lev = block.vert_coord.calc_dmgdt_lev(
        dmgsdt[polg_ids:polg_ide+1],
        ks=half_kds + 1,
        ke=half_kde - 1
    )
    mfz_lev = mfz_lev.at[half_kds+1:half_kde].set(
        - dmgdt_lev - jnp.cumsum(
            dmf[:half_kde-1, polg_ids:polg_ide+1],
            axis=0
        )
    )
    # ==========================================================
    return tree_at(
        where=lambda b: b.aux.mfz_lev.d,
        pytree=block,
        replace=mfz_lev,
    )
    
    
def calc_ke(block: Block, star_idx: INT[Array, ""] | int, substep: int) -> Block:
    mesh = block.mesh
    cell_ids, cell_ihe = mesh.cell_ids, mesh.cell_ihe[0]
    vtx_ids , vtx_ihe  = mesh.vtx_ids , mesh.vtx_ihe [0]
    kds, kde = mesh.full_kds, mesh.full_kde
    un     = block.state.un    .unwrap()[star_idx, :, :]
    ke     = block.aux  .ke    .unwrap()
    ke_vtx = block.aux  .ke_vtx.unwrap()
    # ==========================================================
    cell_ed        = mesh.cell_ed       [cell_ids:cell_ihe+1, :] # (ncell, maxnnb_cell)
    vtx_ed         = mesh.vtx_ed        [vtx_ids :vtx_ihe +1, :] # (nvtx , maxnnb_vtx )
    area_cell      = mesh.area_cell     [cell_ids:cell_ihe+1   ] # (ncell,)
    area_vtx       = mesh.area_vtx      [vtx_ids :vtx_ihe +1   ] # (nvtx,)
    vtx_of_cell    = mesh.vtx_of_cell   [cell_ids:cell_ihe+1, :] # (ncell, maxnnb_cell)
    kite_area_cell = mesh.kite_area_cell[cell_ids:cell_ihe+1, :] # (ncell, maxnnb_cell) 
    le = mesh.le
    de = mesh.de
    
    def edge2cell(un: FLT[Array, "nedge"]) -> FLT[Array, "ncell"]:
        ed_idx = cell_ed
        Ae = le[ed_idx] * de[ed_idx] / 4.0 # (ncell,)
        local_sum = jnp.sum(
            jnp.where(ed_idx < 0, 0.0, Ae * un[ed_idx] ** 2), 
            axis=-1
        )
        return local_sum / area_cell  # (ncell,)
    
    ke = ke.at[kds:kde+1, cell_ids:cell_ihe+1].set(
        jax.vmap(
            edge2cell,
            in_axes=0,
            out_axes=0
        )(un[kds:kde+1, :])
    )
    
    if block.cfg.ke_scheme == 2:
        ke_cell = ke[kds:kde+1, cell_ids:cell_ihe+1]
        ke_cell_wgt = block.cfg.ke_cell_wgt
        def edge2vtx(un: FLT[Array, "nedge"]) -> FLT[Array, "nvtx"]:
            ed_idx = vtx_ed
            Ae = le[ed_idx] * de[ed_idx] / 4.0 # (nvtx,)
            local_sum = jnp.sum(
                jnp.where(ed_idx < 0, 0.0, Ae * un[ed_idx] ** 2),
                axis=-1
            )
            return local_sum / area_vtx # (nvtx,)
        
        def vtx2cell(ke_vtx: FLT[Array, "nvtx"]) -> FLT[Array, "ncell"]:
            v_idx = vtx_of_cell
            Ae = kite_area_cell
            local_sum = jnp.sum(
                jnp.where(v_idx < 0, 0.0, Ae * ke_vtx[v_idx]),
                axis=-1
            )
            return local_sum / area_cell # (ncell,)
        
        ke_vtx = ke_vtx.at[kds:kde+1, vtx_ids:vtx_ihe+1].set(
            jax.vmap(
                edge2vtx, 
                in_axes=0, 
                out_axes=0
            )(un[kds:kde+1, :])
        )
        
        ke_vtx_cell = jax.vmap(vtx2cell, in_axes=0, out_axes=0)(ke_vtx[kds:kde+1,:])
        
        ke = ke.at[kds:kde+1, cell_ids:cell_ihe+1].set(
            ke_cell_wgt * ke_cell + (1.0 - ke_cell_wgt) * ke_vtx_cell
        )
    # ==========================================================
    return tree_at(
        where=lambda b: (b.aux.ke.d, b.aux.ke_vtx.d),
        pytree=block,
        replace=(ke, ke_vtx),
    )
    
    
def calc_div(block: Block, star_idx: INT[Array, ""] | int) -> Block:
    mesh = block.mesh
    un   = block.state.un  .unwrap()[star_idx, :, :]
    div  = block.aux  .div .unwrap()
    div2 = block.aux  .div2.unwrap()
    # ==========================================================
    div = op.div_operator_2d(
        mesh=mesh,
        f=un,
        div=div,
        div_loc=block.aux.div.meta.loc,
        with_halo=True
    )
    if block.cfg.div_damp_order == 4:
        divn = block.aux.fn_2d.unwrap()
        div2 = block.aux.div2 .unwrap()
        
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        v1 = mesh.cell_of_edge[ids:ide+1, 0]
        v2 = mesh.cell_of_edge[ids:ide+1, 1]
        divn = (
            (div[kds:kde+1, v2] - div[kds:kde+1, v1]) 
            * mesh.edpnr_eddtg[ids:ide+1] / mesh.de[ids:ide+1]
        )
        div2 = op.div_operator_2d(
            mesh=mesh,
            f=divn,
            div=div2,
            div_loc=block.aux.div2.meta.loc
        )
        div2 = fill_field2d_array_halo(
            halo=mesh.halo,
            f=div2,
            meta=block.aux.div2.meta
        )
    # ==========================================================
    return tree_at(
        where=lambda b: (b.aux.div.d, b.aux.div2.d),
        pytree=block,
        replace=(div, div2),
    )
    
    
def calc_gz(block: Block, star_idx: INT[Array, ""] | int) -> Block:
    mesh = block.mesh
    ids, ihe = mesh.cell_ids, mesh.cell_ihe[0]
    full_kds, full_kde = mesh.full_kds, mesh.full_kde
    half_kds, half_kde = mesh.half_kds, mesh.half_kde
    para   = block.para
    rd     = para.rd.unwrap()
    gzs    = block.static.gzs.unwrap()
    ph_lev = block.state .ph_lev.unwrap()[star_idx, :, :]
    tv     = block.aux   .tv    .unwrap()
    gz_lev = block.state .gz_lev.unwrap()[star_idx, :, :]
    gz     = block.state .gz    .unwrap()[star_idx, :, :]
    # ====================================================
    # Calculat gz_lev
    kds, kde = half_kds, half_kde
    # =======================================
    # From half_kde to half_kds
    tmp = rd * tv * jnp.log(
        ph_lev[kds+1:kde+1, ids:ihe+1] / ph_lev[kds:kde, ids:ihe+1]
    )
    _gz_lev = gzs + jnp.cumulative_sum(
        jnp.flip(tmp, axis=0),
        axis=0,
        include_initial=True
    )
    # =======================================
    # Flip back, from half_kds to half_kde
    _gz_lev = jnp.flip(_gz_lev, axis=0)
    # =======================================
    # Wrap Back
    gz_lev = gz_lev.at[kds:kde+1, ids:ihe+1].set(_gz_lev)
    # ===================================================
    kds, kde = full_kds, full_kde
    # Calculate gz
    gz = gz.at[kds:kde+1, ids:ihe+1].set(
        0.5 * (
            gz_lev[kds:kde+1, ids:ihe+1]
            + gz_lev[kds+1:kde+2, ids:ihe+1]
        )
    )
    # ===================================================
    return tree_at(
        where=lambda b: (b.state.gz.d, b.state.gz_lev.d),
        pytree=block,
        replace=(gz, gz_lev),
        at_dims=(0, 0),
        at_idxes=(star_idx, star_idx)
    )
    
    
def calc_dmg(block: Block, star_idx: INT[Array, ""] | int) -> Block:
    mesh = block.mesh
    ids     , ihe      = mesh.cell_ids, mesh.cell_ihe[0]
    full_kds, full_kde = mesh.full_kds, mesh.full_kde
    half_kds, half_kde = mesh.half_kds, mesh.half_kde
    g       = block.para .gravity .unwrap()
    dmg     = block.state.dmg     .unwrap()[star_idx, :, :]
    dmg_edge= block.aux  .dmg_edge.unwrap()[          :, :]
    dmg_vtx = block.aux  .dmg_vtx .unwrap()[          :, :]
    # ==========================================================
    if block.mode.is_baroclinic() or block.mode.is_advection():
        mg      = block.state .mg    .unwrap()[star_idx, :, :]
        mg_lev  = block.state .mg_lev.unwrap()[star_idx, :, :]
        dmg_lev = block.state.dmg_lev.unwrap()[star_idx, :, :]
        # ======================================================
        kds, kde = full_kds, full_kde
        dmg = dmg.at[kds:kde+1, ids:ihe+1].set(
            mg_lev[kds+1:kde+2, ids:ihe+1] - mg_lev[kds:kde+1  , ids:ihe+1]
        )
        dmg_valid = dmg[kds:kde+1, ids:ihe+1]
        dmg = eqx.error_if(
            x=dmg,
            pred=jnp.any(dmg_valid <= 0),
            msg="The dry-air weight levels are not monotonic!",
            on_error="raise"
        )
        dmg = eqx.error_if(
            x=dmg,
            pred=jnp.any(jnp.isnan(dmg_valid)),
            msg="The dry-air weight levels contain NaN values!",
            on_error="raise"
        )
        # ======================================================
        # half_kds + 1 <= k <= half_kde - 1
        kds, kde = half_kds, half_kde
        dmg_lev = dmg_lev.at[kds+1:kde, ids:ihe+1].set(
            mg[kds+1:kde, ids:ihe+1] - mg[kds:kde-1, ids:ihe+1]
        )
        # Top boundary
        k = half_kds
        dmg_lev = dmg_lev.at[k, ids:ihe+1].set(
            mg[k, ids:ihe+1] - mg_lev[k, ids:ihe+1]
        )
        # Bottom boundary
        k = half_kde
        dmg_lev = dmg_lev.at[k, ids:ihe+1].set(
            mg_lev[k, ids:ihe+1] - mg[k-1, ids:ihe+1]
        )
        # ======================================================
        dmg_edge = interp.scvt_average_2d(
            mesh=mesh,
            x=dmg,
            y=dmg_edge,
            x_loc=block.state.dmg     .meta.loc,
            y_loc=block.aux  .dmg_edge.meta.loc
        )
        dmg_vtx = interp.scvt_interp_2d(
            mesh=mesh,
            x=dmg,
            y=dmg_vtx,
            x_loc=block.state.dmg    .meta.loc,
            y_loc=block.aux  .dmg_vtx.meta.loc
        )
        return tree_at(
            where=lambda b: (
                b.state.dmg.d, b.state.dmg_lev.d,
                b.aux.dmg_edge.d, b.aux.dmg_vtx.d
            ),
            pytree=block,
            replace=(dmg, dmg_lev, dmg_edge, dmg_vtx),
            at_dims=(0, 0, None, None),
            at_idxes=(star_idx, star_idx, None, None)
        )
    else:
        gzs = block.static.gzs.unwrap()
        gz  = block.state .gz .unwrap()[star_idx, :, :]
        # ======================================================
        dmg = dmg.at[0, ids:ihe+1].set(
            (gz[0, ids:ihe+1] - gzs[ids:ihe+1]) / g
        )
        dmg_edge = interp.scvt_average_2d(
            mesh=mesh,
            x=dmg,
            y=dmg_edge,
            x_loc=block.state.dmg     .meta.loc,
            y_loc=block.aux  .dmg_edge.meta.loc
        )
        dmg_vtx = interp.scvt_interp_2d(
            mesh=mesh,
            x=dmg,
            y=dmg_vtx,
            x_loc=block.state.dmg    .meta.loc,
            y_loc=block.aux  .dmg_vtx.meta.loc
        )
        # ====================================================== 
        return tree_at(
            where=lambda b: (
                b.state.dmg.d,
                b.aux.dmg_edge.d, b.aux.dmg_vtx.d
            ),
            pytree=block,
            replace=(dmg, dmg_edge, dmg_vtx),
            at_dims=(0, None, None),
            at_idxes=(star_idx, None, None)
        )
    
    
def calc_mf(block: Block, star_idx: INT[Array, ""] | int, dt: float) -> Block:
    mesh = block.mesh
    ihs, ihe = mesh.edge_ihs[0], mesh.edge_ihe[0]
    kds, kde = mesh.full_kds, mesh.full_kde
    un       = block.state.un      .unwrap()[star_idx, :, :]
    dmg_edge = block.aux  .dmg_edge.unwrap()[          :, :]
    mfn      = block.aux  .mfn     .unwrap()
    # ==========================================================
    mfn = mfn.at[kds:kde+1, ihs:ihe+1].set(
        un[kds:kde+1, ihs:ihe+1] * dmg_edge[kds:kde+1, ihs:ihe+1]
    )
    # ==========================================================
    return tree_at(
        where=lambda b: b.aux.mfn.d,
        pytree=block,
        replace=mfn
    )
    
    
def calc_ut(block: Block, star_idx: INT[Array, ""] | int) -> Block:
    mesh = block.mesh
    ids, ide = mesh.edge_ids, mesh.edge_ide
    kds, kde = mesh.full_kds, mesh.full_kde
    coriolis = block.equaux.coriolis
    de       = mesh.de  # (nedge,)
    ep_idx   = coriolis.edge_of_edge[ids:ide+1]  # (nedge, maxnnb*2)
    wgt      = coriolis.wgt_edge_of_edge         # (nedge, maxnnb*2)
    le       = coriolis.le_edge_of_edge          # (nedge, maxnnb*2)
    un       = block.state.un.unwrap()[star_idx, :]
    ut       = block.aux  .ut.unwrap()
    # ==========================================================

    def _wrap(un: FLT[Array, "nedge"]) -> FLT[Array, "nedge"]:
        wind_tile = un[ep_idx]
        wind_tile = jnp.where(ep_idx < 0, 0.0, wind_tile)
        return jnp.sum(
            wgt[ids:ide+1, :] * le[ids:ide+1, :] * wind_tile,
            axis=-1
        ) / de[ids:ide+1]

    ut = ut.at[kds:kde+1, ids:ide+1].set(
        jax.vmap(
            _wrap,
            in_axes=0,
            out_axes=0
        )(un[kds:kde+1])
    )
    # ==========================================================
    return tree_at(
        where=lambda b: b.aux.ut.d,
        pytree=block,
        replace=ut,
    )


def calc_vor(block: Block, star_idx: INT[Array, ""] | int, with_halo: bool = False) -> Block:
    un = block.state.un.unwrap()[star_idx, :, :]
    vor = block.aux.vor.unwrap()
    # ==========================================================
    vor = op.curl_operator_2d(
        mesh=block.mesh,
        f=un,      # in
        curl=vor,  # inout
        curl_loc=block.aux.vor.meta.loc,
        with_halo=with_halo
    )
    # ==========================================================
    return tree_at(
        where=lambda b: b.aux.vor.d,
        pytree=block,
        replace=vor,
    )
    
    
def calc_pv(block: Block, star_idx: INT[Array, ""] | int, substep: int) -> Block:
    mesh = block.mesh
    ids, ide = mesh.vtx_ids, mesh.vtx_ide
    kds, kde = mesh.full_kds, mesh.full_kde
    f_vtx    = mesh.f_vtx
    dmg_vtx  = block.aux.dmg_vtx.unwrap()[:, :]
    pv       = block.aux.pv     .unwrap()
    # ==========================================================
    block = calc_vor(block, star_idx)
    vor = block.aux.vor.unwrap()[kds:kde+1, ids:ide+1]
    pv = pv.at[kds:kde+1, ids:ide+1].set(
        (vor + f_vtx[ids:ide+1]) / dmg_vtx[kds:kde+1, ids:ide+1]
    )
    # ==========================================================
    return tree_at(
        where=lambda b: b.aux.pv.d,
        pytree=block,
        replace=pv,
    )
    
    
def interp_pv_midpoint(
    block: Block, 
    star_idx: INT[Array, ""] | int, 
    dt: float, 
    substep: int
) -> Block:
    mesh = block.mesh
    ids, ide = mesh.edge_ids, mesh.edge_ide
    kds, kde = mesh.full_kds, mesh.full_kde
    pv      = block.aux.pv     .unwrap()
    pv_edge = block.aux.pv_edge.unwrap()
    # ==========================================================
    v1 = mesh.vtx_of_edge[ids:ide+1, 0]
    v2 = mesh.vtx_of_edge[ids:ide+1, 1]
    pv_edge = pv_edge.at[kds:kde+1, ids:ide+1].set(
        0.5 * (pv[kds:kde+1, v1] + pv[kds:kde+1, v2])
    )
    return tree_at(
        where=lambda b: b.aux.pv_edge.d,
        pytree=block,
        replace=pv_edge
    )
    
    
def interp_pv_upwind(
    block: Block, 
    star_idx: INT[Array, ""] | int, 
    dt: float, 
    substep: int    
) -> Block:
    mesh        = block.mesh
    para        = block.para
    stencil_2nd = block.equaux.adv.upwind.stencil_vtx_2nd
    stencil_4th = block.equaux.adv.upwind.stencil_vtx_4th
    radius          = para.radius.unwrap()
    upwind_order_pv = block.cfg.upwind_order_pv
    ids, ide = mesh.edge_ids, mesh.edge_ide
    kds, kde = mesh.full_kds, mesh.full_kde
    un      = block.state.un     .unwrap()[star_idx, :, :]
    ut      = block.aux  .ut     .unwrap()
    pv      = block.aux  .pv     .unwrap()
    pv_edge = block.aux  .pv_edge.unwrap()
    # ==========================================================
    eps = jnp.finfo(un.dtype).eps
    un_ = un[kds:kde+1, ids:ide+1]
    ut_ = ut[kds:kde+1, ids:ide+1]
    v1 = mesh.vtx_of_edge[ids:ide+1, 0]
    v2 = mesh.vtx_of_edge[ids:ide+1, 1]
    q1 = pv[kds:kde+1, v1]
    q2 = pv[kds:kde+1, v2]
    flux1 = 0.5 * (q1 + q2)
    b = jnp.abs(ut_) / (jnp.sqrt(un_ ** 2 + ut_ ** 2) + eps)
    if upwind_order_pv == 1:
        wind = ut_ * mesh.edp12_edptg[ids:ide+1]
        le   = mesh.le[ids:ide+1] / radius
        flux2 = upwind.upwind1_hori(
            q=pv,
            wind=wind,
            edd_v=mesh.vtx_of_edge[ids:ide+1, :],
            wgt=block.cfg.upwind_wgt_pv
        )
        pv_edge = pv_edge.at[kds:kde+1, ids:ide+1].set(
            b * flux2 + (1 - b) * flux1
        )
    elif upwind_order_pv == 3:
        wind = ut_ * mesh.edp12_edptg[ids:ide+1]
        le   = mesh.le[ids:ide+1] / radius
        flux2 = upwind.upwind3_hori(
            q=pv,
            wind=wind,
            edd_v=mesh.vtx_of_edge[ids:ide+1, :],
            edd_leng=le,
            stencil_2nd_idx=stencil_2nd.idxes,
            stencil_2nd_wgt=stencil_2nd.wgt,
            wgt=block.cfg.upwind_wgt_pv
        )
        pv_edge = pv_edge.at[kds:kde+1, ids:ide+1].set(
            b * flux2 + (1 - b) * flux1
        )
    elif upwind_order_pv == 5:
        wind = ut_ * mesh.edp12_edptg[ids:ide+1]
        le   = mesh.le[ids:ide+1] / radius
        flux2 = upwind.upwind5_hori(
            q=pv,
            wind=wind,
            edd_v=mesh.vtx_of_edge[ids:ide+1, :],
            edd_leng=le,
            stencil_2nd_idx=stencil_2nd.idxes,
            stencil_2nd_wgt=stencil_2nd.wgt,
            stencil_4th_idx=stencil_4th.idxes,
            stencil_4th_wgt=stencil_4th.wgt,
            wgt=block.cfg.upwind_wgt_pv
        )
        pv_edge = pv_edge.at[kds:kde+1, ids:ide+1].set(
            b * flux2 + (1 - b) * flux1
        )
    else:
        raise ValueError(f"interp_pv_upwind: unsupported upwind order {upwind_order_pv}!")
    return tree_at(
        where=lambda b: b.aux.pv_edge.d,
        pytree=block,
        replace=pv_edge
    )
    
    
def interp_pv(
    block: Block, 
    star_idx: INT[Array, ""] | int, 
    dt: float, 
    substep: int   
) -> Block:
    if block.cfg.pv_adv_scheme == "midpoint":
        return interp_pv_midpoint(block, star_idx, dt, substep)
    elif block.cfg.pv_adv_scheme == "upwind":
        return interp_pv_upwind  (block, star_idx, dt, substep)
    else:
        raise ValueError(f"interp_pv: unsupported pv_adv_scheme {block.cfg.pv_adv_scheme}!")
    
    
def calc_coriolis(block: Block, star_idx: INT[Array, ""] | int, dt: float, substep: int) -> Block:
    mesh = block.mesh
    ids, ide = mesh.edge_ids, mesh.edge_ide
    kds, kde = mesh.full_kds, mesh.full_kde
    coriolis = block.equaux.coriolis
    de     = mesh.de[ids:ide+1]  # (nedg,)
    ep_idx = coriolis.edge_of_edge[ids:ide+1, :]  # (nedg, maxnnb*2)
    wgt    = coriolis.wgt_edge_of_edge[ids:ide+1, :]
    le     = coriolis.le_edge_of_edge[ids:ide+1, :]
    mfn            = block.aux .mfn           .unwrap()
    pv_edge        = block.aux .pv_edge       .unwrap()
    dundt          = block.tend.dundt         .unwrap()
    dundt_coriolis = block.tend.dundt_coriolis.unwrap()
    # ==========================================================
    def _wrap(
        mfn: FLT[Array, "nEdge"],
        pv_edge: FLT[Array, "nEdge"],
    ) -> FLT[Array, "nEdge"]:
        pv_iee = pv_edge[ids:ide+1]
        pv_iep = jnp.take(
            pv_edge,
            ep_idx,
            axis=0
        )  # (nedge, max_numb*2)
        pv_eep = (pv_iee[:, None] + pv_iep) / 2  # (nedge, max_numb*2)

        mfx_tile = mfn[ep_idx]  # (nedg, maxnnb*2)
        mfx_tile = jnp.where(ep_idx < 0, 0.0, mfx_tile)
        mfx_tile = mfx_tile * pv_eep
        return jnp.sum(
            wgt * le * mfx_tile,
            axis=-1
        ) / de

    tmp = - jax.vmap(
        _wrap, 
        in_axes=(0, 0), 
        out_axes=0
    )(mfn[kds:kde+1, :], pv_edge[kds:kde+1, :])
    
    dundt = dundt.at[kds:kde+1, ids:ide+1].set(
        dundt[kds:kde+1, ids:ide+1] + tmp
    )
    dundt_coriolis = dundt_coriolis.at[kds:kde+1, ids:ide+1].set(tmp)
    # ==========================================================
    return tree_at(
        where=lambda b: (b.tend.dundt.d, b.tend.dundt_coriolis.d),
        pytree=block,
        replace=(dundt, dundt_coriolis)
    )
    
    
def calc_grad_ke(block: Block, star_idx: INT[Array, ""] | int, dt: float) -> Block:
    mesh = block.mesh
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.edge_ids, mesh.edge_ide
    edpnr_eddtg = mesh.edpnr_eddtg[ids:ide+1]
    de = mesh.de[ids:ide+1]
    ke            = block.aux .ke           .unwrap()
    dundt         = block.tend.dundt        .unwrap()
    dundt_grad_ke = block.tend.dundt_grad_ke.unwrap()
    # ==========================================================
    v1 = mesh.cell_of_edge[ids:ide+1, 0]
    v2 = mesh.cell_of_edge[ids:ide+1, 1]
    tmp = -(ke[kds:kde+1, v2] - ke[kds:kde+1, v1]) * edpnr_eddtg / de
    dundt = dundt.at[kds:kde+1, ids:ide+1].set(
        dundt[kds:kde+1, ids:ide+1] + tmp
    )
    dundt_grad_ke = dundt_grad_ke.at[kds:kde+1, ids:ide+1].set(tmp)
    # ==========================================================
    return tree_at(
        where=lambda b: (b.tend.dundt.d, b.tend.dundt_grad_ke.d),
        pytree=block,
        replace=(dundt, dundt_grad_ke)
    )
    
    
def calc_grad_mf(block: Block, star_idx: INT[Array, ""] | int) -> Block:
    mfn = block.aux.mfn.unwrap()
    dmf = block.aux.dmf.unwrap()
    # ==========================================================
    dmf = op.div_operator_2d(
        mesh=block.mesh,
        f=mfn,
        div=dmf,
        div_loc=block.aux.dmf.meta.loc
    )
    # ==========================================================
    return tree_at(
        where=lambda b: b.aux.dmf.d,
        pytree=block,
        replace=dmf,
    )
    
    
def calc_grad_ptf(block: Block, star_idx: INT[Array, ""] | int, dt: float) -> Block:
    mesh = block.mesh
    ids, ide = mesh.cell_ids, mesh.cell_ide
    kds, kde = mesh.full_kds, mesh.full_kde
    pt      = block.state.pt     .unwrap()[star_idx, :, :] 
    un      = block.state.un     .unwrap()[star_idx, :, :]
    dmg     = block.state.dmg    .unwrap()[star_idx, :, :]
    mfn     = block.aux  .mfn    .unwrap()
    mfz_lev = block.aux  .mfz_lev.unwrap()
    dptdt   = block.tend .dptdt  .unwrap()
    batch   = block.adv_batch_pt
    # ==========================================================
    batch = batch.set_wind(
        un=un,
        mfn=mfn,
        mfz=mfz_lev,
        m=dmg
    )
    ptfn = interface.calc_tracer_hflx(
        block=block,
        batch=batch,
        q=pt,
        qmfn=batch.qmfn.unwrap(),
        dt=dt
    )
    dptdt = op.div_operator_2d(
        mesh=mesh, 
        f=ptfn, 
        div=dptdt, 
        div_loc=block.tend.dptdt.meta.loc
    )
    # TODO: add scvt_adv_fill_vhalo function at here!
    ptfz = interface.calc_tracer_vflx(
        block=block,
        batch=batch,
        q=pt,
        qmfz=batch.qmfz.unwrap(),
        dt=dt
    )
    dptdt = dptdt.at[kds:kde+1, ids:ide+1].set(
        -dptdt[kds:kde+1, ids:ide+1] - 
        (ptfz[kds+1:kde+2, ids:ide+1] - ptfz[kds:kde+1, ids:ide+1])
    )
    
    return tree_at(
        where=lambda b: (b.adv_batch_pt.qmfn.d, b.adv_batch_pt.qmfz.d, b.tend.dptdt.d),
        pytree=block,
        replace=(ptfn, ptfz, dptdt)
    )
    

def calc_dmgsdt(block: Block, star_idx: INT[Array, ""] | int) -> Block:
    mesh = block.mesh
    ids, ide = mesh.cell_ids, mesh.cell_ide
    kds, kde = mesh.full_kds, mesh.full_kde
    dmf    = block.aux .dmf   .unwrap()
    dmgsdt = block.tend.dmgsdt.unwrap()
    # ==========================================================
    dmgsdt = dmgsdt.at[ids:ide+1].set(
        - jnp.sum(dmf[kds:kde+1, ids:ide+1], axis=0)
    )
    # ==========================================================
    return tree_at(
        where=lambda b: b.tend.dmgsdt.d,
        pytree=block,
        replace=dmgsdt
    )
    
    
def calc_wedundlev(block: Block, star_idx: INT[Array, ""] | int, dt: float) -> Block:
    mesh     = block.mesh
    ids, ide = mesh.edge_ids, mesh.edge_ide
    kds, kde = mesh.full_kds, mesh.full_kde
    un              = block.state.un             .unwrap()[star_idx, :, :]
    dmg_edge        = block.aux  .dmg_edge       .unwrap()[          :, :]
    un_lev          = block.aux  .un_lev         .unwrap()
    mfz_lev         = block.aux  .mfz_lev        .unwrap()
    mfz_lev_edge    = block.aux  .mfz_lev_edge   .unwrap()
    dundt_wedundeta = block.tend .dundt_wedundeta.unwrap()
    dundt           = block.tend .dundt          .unwrap()
    # ==========================================================
    if block.cfg.uv_vert_adv_type == 1:
        un_lev = interp.scvt_interp_2d(
            mesh=mesh,
            x=un,
            y=un_lev,
            x_loc=block.state.un.meta.loc,
            y_loc=block.aux.un_lev.meta.loc,
            extrap_type=2
        )
        mfz_lev_edge = interp.scvt_interp_2d(
            mesh=mesh,
            x=mfz_lev,
            y=mfz_lev_edge,
            x_loc=block.aux.mfz_lev.meta.loc,
            y_loc=block.aux.mfz_lev_edge.meta.loc
        )
        tmp = -(
            mfz_lev_edge[kds+1:kde+2, ids:ide+1] * (
                un_lev[kds+1:kde+2, ids:ide+1] - un    [kds:kde+1, ids:ide+1]
            ) + 
            mfz_lev_edge[kds:kde+1  , ids:ide+1] * (
                un    [kds:kde+1  , ids:ide+1] - un_lev[kds:kde+1, ids:ide+1]
            )
        ) / dmg_edge[kds:kde+1, ids:ide+1]
        dundt_wedundeta = dundt_wedundeta.at[kds:kde+1, ids:ide+1].set(tmp)
        dundt = dundt.at[kds:kde+1, ids:ide+1].set(dundt[kds:kde+1, ids:ide+1] + tmp)
    elif block.cfg.uv_vert_adv_type == 2:
        mfz_lev_edge = interp.scvt_interp_2d(
            mesh=mesh, 
            x=mfz_lev, 
            y=mfz_lev_edge, 
            x_loc=block.aux.mfz_lev.meta.loc,
            y_loc=block.aux.mfz_lev_edge.meta.loc
        )
        tmp = - (
            mfz_lev_edge[kds+1:kde+2, ids:ide+1] * (un[kds+1:kde+2, ids:ide+1] - un[kds:kde+1, ids:ide+1]) - 
            mfz_lev_edge[kds:kde+1  , ids:ide+1] * (un[kds-1:kde  , ids:ide+1] - un[kds:kde+1, ids:ide+1])
        ) / dmg_edge[kds:kde+1, ids:ide+1] / 2
        dundt_wedundeta = dundt_wedundeta.at[kds:kde+1, ids:ide+1].set(tmp)
        dundt = dundt.at[kds:kde+1, ids:ide+1].set(dundt[kds:kde+1, ids:ide+1] + tmp)
    else:
        raise ValueError(f"calc_wedundlev: unsupported uv_vert_adv_type {block.cfg.uv_vert_adv_type}!")
    # ==========================================================
    return tree_at(
        where=lambda b: (b.tend.dundt.d, b.tend.dundt_wedundeta.d),
        pytree=block,
        replace=(dundt, dundt_wedundeta)
    )
