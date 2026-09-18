from __future__ import annotations
from typing import TYPE_CHECKING


from terraflow.mesh.scvt.operators import wind_c2a_operator
from terraflow.mesh.scvt.parallel import global_sum
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.pgf.interface import pgf_run
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.space_operators import (
    operators_prepare_2,
    calc_grad_mf,
    calc_dmgsdt,
    calc_mfz,
    calc_grad_ptf,
    calc_grad_ke,
    calc_coriolis,
    calc_wedundlev
)

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block

    INT = typing.Int
    FLT = typing.Float
    Array = typing.Array


def print_logo():
    banner = [
        "████████╗███████╗██████╗ ██████╗  █████╗ ███████╗██╗      ██████╗ ██╗    ██╗",
        "╚══██╔══╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██║     ██╔═══██╗██║    ██║",
        "   ██║   █████╗  ██████╔╝██████╔╝███████║█████╗  ██║     ██║   ██║██║ █╗ ██║",
        "   ██║   ██╔══╝  ██╔══██╗██╔══██╗██╔══██║██╔══╝  ██║     ██║   ██║██║███╗██║",
        "   ██║   ███████╗██║  ██║██║  ██║██║  ██║██║     ███████╗╚██████╔╝╚███╔███╔╝",
        "   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝ ",
    ]
    print("")
    for line in banner:
        print(line)
    print("")


def space_operators(
    block: Block,
    old_idx: INT[Array, ""] | int,
    star_idx: INT[Array, ""] | int,
    new_idx: INT[Array, ""] | int,
    dt: float,
    pass_tag: str,
    substep: int
) -> Block:
    mode  = block.mode
    bkd   = block.backend
    block = block.reset(pass_tag=pass_tag)

    if pass_tag == "all_pass":
        block = operators_prepare_2(block, star_idx, dt, pass_tag, substep)
        if mode.is_baroclinic():
            block = calc_grad_mf   (block, star_idx)
            block = calc_dmgsdt    (block, star_idx)
            block = calc_mfz       (block, star_idx)
            block = calc_grad_ptf  (block, star_idx, dt)
            block = calc_grad_ke   (block, star_idx, dt)
            block = pgf_run        (block, star_idx)
            block = calc_coriolis  (block, star_idx, dt, substep)
            block = calc_wedundlev (block, star_idx, dt)
        else:
            block = calc_grad_mf   (block, star_idx)
            block = calc_grad_ke   (block, star_idx, dt)
            block = pgf_run        (block, star_idx,)
            block = calc_coriolis  (block, star_idx, dt, substep)
            # =============================================
            mesh  = block.mesh
            kds, kde = mesh.full_kds, mesh.full_kde
            ids, ide = mesh.cell_ids, mesh.cell_ide
            dmf   = block.aux.dmf.unwrap()
            g     = block.para.gravity.unwrap()
            dgzdt = block.tend.dgzdt  .unwrap()
            dgzdt = bkd.slice_update(dgzdt, (slice(kds, kde+1), slice(ids, ide+1)),
                -dmf[kds:kde+1, ids:ide+1] * g
            )
            block = bkd.tree_at(
                where=lambda b: (b.tend.dgzdt.d),
                pytree=block,
                replace=dgzdt
            )
            # =============================================
    elif pass_tag == "forward_pass":
        block = operators_prepare_2(block, star_idx, dt, pass_tag, substep)
        if mode.is_baroclinic():
            block = calc_grad_mf   (block, star_idx)
            block = calc_dmgsdt    (block, star_idx, )
            block = calc_mfz       (block, star_idx, )
            block = calc_grad_ptf  (block, star_idx, dt)
            block = calc_grad_ke   (block, star_idx, dt)
            block = calc_coriolis  (block, star_idx, dt, substep)
            block = calc_wedundlev (block, star_idx, dt)
        else:
            block = calc_grad_mf   (block, star_idx)
            block = calc_grad_ke   (block, star_idx, dt)
            block = calc_coriolis  (block, star_idx, dt, substep)
            # =============================================
            mesh  = block.mesh
            kds, kde = mesh.full_kds, mesh.full_kde
            ids, ide = mesh.cell_ids, mesh.cell_ide
            dmf   = block.aux.dmf.unwrap()
            g     = block.para.gravity.unwrap()
            dgzdt = block.tend.dgzdt  .unwrap()
            dgzdt = bkd.slice_update(dgzdt, (slice(kds, kde+1), slice(ids, ide+1)),
                -dmf[kds:kde+1, ids:ide+1] * g
            )
            block = bkd.tree_at(
                where=lambda b: (b.tend.dgzdt.d),
                pytree=block,
                replace=dgzdt
            )
            # =============================================
    elif pass_tag == "backward_pass":
        if mode.is_nonhydrostatic():
            # TODO: Need to add scvt_nh_solve here!
            pass
        block = operators_prepare_2(block, new_idx, dt, pass_tag, substep)
        block = pgf_run            (block, new_idx)

    return block


def diagnose(block: Block, itime_str: str) -> Block:

    def diagnose_uv(block: Block, itime_str: str) -> Block:
        R = block.para.radius.unwrap()
        mesh = block.mesh
        itime = block.get_time_idx(itime_str=itime_str)
        un = block.state.un.unwrap()[itime, :, :]
        u  = block.aux  .u .unwrap()
        v  = block.aux  .v .unwrap()
        # =======================================================
        # Diagnose u and v from un
        u, v = wind_c2a_operator(R, mesh, un, u, v)
        # =======================================================
        return block.backend.tree_at(
            where=lambda b: (b.aux.u.d, b.aux.v.d),
            pytree=block,
            replace=(u, v),
        )

    def diagnose_tm(block: Block, itime_str: str) -> FLT[Array, ""]:
        # =============================================
        bkd = block.backend
        itime = block.get_time_idx(itime_str=itime_str)
        mesh = block.mesh
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        dmg = block.state.dmg.unwrap()[itime, kds:kde+1, ids:ide+1]
        area_cell = mesh.area_cell[ids:ide+1]
        # =============================================
        tm = bkd.array(0.0, dtype=block.dev.float_type)
        return tm + bkd.sum(dmg * area_cell[None, :])

    def diagnose_tqv(block: Block, itime_str: str) -> FLT[Array, ""]:
        idx_qv = block.tracers.idx_qv
        if idx_qv >= 0:
            bkd = block.backend
            itime_qv = block.adv_batches["moist"].get_time_idx(itime_str=itime_str)
            mesh = block.mesh
            kds, kde = mesh.full_kds, mesh.full_kde
            ids, ide = mesh.cell_ids, mesh.cell_ide
            area_cell = mesh.area_cell[ids:ide+1]
            m  = block.adv_batches["moist"].m.unwrap()[kds:kde+1, ids:ide+1]
            qv = block.tracers.q.unwrap()[itime_qv, idx_qv, kds:kde+1, ids:ide+1]
            return bkd.sum(m * qv * area_cell[None, :])
        else:
            return block.backend.array(0.0, dtype=block.dev.float_type)

    def diagnose_te(
        block: Block,
        itime_str: str
    ) -> tuple[
        FLT[Array, ""],
        FLT[Array, ""],
        FLT[Array, ""],
        FLT[Array, ""]
    ]:
        # =============================================
        float_type = block.dev.float_type
        bkd = block.backend
        itime = block.get_time_idx(itime_str=itime_str)
        mesh = block.mesh
        kds, kde = mesh.full_kds, mesh.full_kde
        cids, cide = mesh.cell_ids, mesh.cell_ide
        eids, eide = mesh.edge_ids, mesh.edge_ide
        gzs      = block.static.gzs     .unwrap()[cids:cide+1]
        dmg      = block.state .dmg     .unwrap()[itime, kds:kde+1, cids:cide+1]
        un       = block.state .un      .unwrap()[itime, kds:kde+1, eids:eide+1]
        dmg_edge = block.aux   .dmg_edge.unwrap()[kds:kde+1, eids:eide+1]
        g        = block.para  .gravity .unwrap()
        cpd      = block.para  .cpd     .unwrap()
        # Weighted area
        area_cell = mesh.area_cell[cids:cide+1]
        area_edge = mesh.area_edge[eids:eide+1]
        # =============================================
        te_ke = bkd.array(0.0, dtype=float_type)
        te_ie = bkd.array(0.0, dtype=float_type)
        te_pe = bkd.array(0.0, dtype=float_type)
        # =============================================
        # Total energy due to kinetic energy
        # TODO: nh dynamics kinetic energy
        # Kinetic energy at edge
        te_ke = te_ke + bkd.sum(0.5 * un**2 * dmg_edge * area_edge[None, :] * 2)
        # =============================================
        # Total energy due to internal energy
        if block.mode.is_baroclinic():
            tv = block.aux.tv.unwrap()[kds:kde+1, cids:cide+1]
            te_ie = te_ie + bkd.sum(dmg * cpd * tv * area_cell[None, :])
        # =============================================
        # Total energy due to potential energy
        if block.mode.is_baroclinic():
            mgs = block.state.mgs.unwrap()[itime, cids:cide+1]
            te_pe = te_pe + bkd.sum(gzs * mgs * area_cell)
        else:
            te_pe = te_pe + bkd.sum(
                (dmg.squeeze()**2 * g * 0.5 + dmg.squeeze()*gzs) * area_cell,
            )
        # =============================================
        return te_ke + te_ie + te_pe, te_ke, te_ie, te_pe

    def diagnose_tpe(block: Block, itime_str: str) -> FLT[Array, ""]:
        # =======================================================
        mesh = block.mesh
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        pv_edge  = block.aux.pv_edge .unwrap()[kds:kde+1, ids:ide+1]
        dmg_edge = block.aux.dmg_edge.unwrap()[kds:kde+1, ids:ide+1]
        # Weighted area
        area_edge = mesh.area_edge[ids:ide+1]
        # =======================================================
        # Total potential enstrophy
        return block.backend.sum(dmg_edge * pv_edge**2 * 0.5 * area_edge[None, :] * 2)

    def diagnose_tpt(block: Block, itime_str: str) -> FLT[Array, ""]:
        # =======================================================
        float_type = block.dev.float_type
        bkd = block.backend
        itime = block.get_time_idx(itime_str=itime_str)
        mesh = block.mesh
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        dmg = block.state.dmg.unwrap()[itime, kds:kde+1, ids:ide+1]
        pt  = block.state.pt .unwrap()[itime, kds:kde+1, ids:ide+1]
        # Weighted area
        area_cell = mesh.area_cell[ids:ide+1]
        # =======================================================
        # Total potential temperature
        if block.mode.is_baroclinic():
            return bkd.sum(dmg * pt * area_cell[None, :])
        else:
            return bkd.array(0.0, dtype=float_type)

    def diagnose_tq(block: Block, itime_str: str) -> FLT[Array, ""]:
        # =======================================================
        float_type = block.dev.float_type
        bkd = block.backend
        mesh = block.mesh
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        # Weighted area
        area_cell = mesh.area_cell[ids:ide+1]
        # =======================================================
        tq = bkd.array(0.0, dtype=float_type)
        for batch in block.adv_batches.values():
            dmg = batch.m.unwrap()[kds:kde+1, ids:ide+1]
            itime = batch.get_time_idx(itime_str)
            idx = batch.idx
            q = block.tracers.q.unwrap()[itime, idx, kds:kde+1, ids:ide+1]
            tq = tq + bkd.sum(dmg * q * area_cell[None, :])
        # =======================================================
        return tq

    def diagnose_global_sum(x: FLT[Array, ""]) -> FLT[Array, ""]:
        return global_sum(block.backend, block.dev.proc.comm_model, x)

    def diagnose4barotropic(block: Block, itime_str: str) -> Block:
        # =======================================================
        # Diagnose u and v at primal cell
        block = diagnose_uv(block, itime_str=itime_str)
        # =======================================================
        # Diagnose total mass, total energy, total potential enstrophy
        # =========================================
        # Total mass
        tm = diagnose_tm(block, itime_str=itime_str)
        # =========================================
        # Total energy
        te, te_ke, _, te_pe = diagnose_te(block, itime_str=itime_str)
        # =========================================
        # Total potential enstrophy
        tpe = diagnose_tpe(block, itime_str=itime_str)
        # =========================================
        # Global sums
        tm    = diagnose_global_sum(tm)
        te_ke = diagnose_global_sum(te_ke)
        te_pe = diagnose_global_sum(te_pe)
        tpe   = diagnose_global_sum(tpe)
        te    = te_ke + te_pe
        # =======================================================
        # Wrap diagnosed variables into block
        itime = block.get_time_idx(itime_str=itime_str)
        if block.mode.is_barotropic():
            return block.backend.tree_at(
                where=lambda b: (
                    b.state.tm.d, b.state.te.d, b.state.tpe.d,
                    b.state.te_ke.d, b.state.te_pe.d
                ),
                pytree=block,
                replace=(tm, te, tpe, te_ke, te_pe),
                at_dims=(0,) * 5,
                at_idxes=(itime,) * 5
            )
        else:
            raise ValueError(
                f"Invalid equation mode for barotropic diagnosis. Invalid mode={block.mode}"
            )

    def diagnose4baroclinic(block: Block, itime_str: str) -> Block:
        # =======================================================
        # Diagnose u and v at primal cell
        block = diagnose_uv(block, itime_str)
        # =======================================================
        # Diagnose total mass, total energy, total potential enstrophy
        # =========================================
        # Total mass
        tm = diagnose_tm(block, itime_str=itime_str)
        # =========================================
        # Total water vapor
        tqv = diagnose_tqv(block, itime_str)
        # =========================================
        # Total energy
        out = diagnose_te(block, itime_str=itime_str)
        te, te_ke, te_ie, te_pe = out
        # =========================================
        # Total potential enstrophy
        tpe = diagnose_tpe(block, itime_str=itime_str)
        # =========================================
        # Total potential temperature
        tpt = diagnose_tpt(block, itime_str=itime_str)
        # =========================================
        # Global sums
        tm    = diagnose_global_sum(tm)
        tqv   = diagnose_global_sum(tqv)
        te_ke = diagnose_global_sum(te_ke)
        te_ie = diagnose_global_sum(te_ie)
        te_pe = diagnose_global_sum(te_pe)
        tpe   = diagnose_global_sum(tpe)
        tpt   = diagnose_global_sum(tpt)
        te    = te_ke + te_ie + te_pe
        # =========================================
        # Wrap diagnosed variables into block
        itime = block.get_time_idx(itime_str=itime_str)
        if block.mode.is_baroclinic() and block.mode.is_dry():
            return block.backend.tree_at(
                where=lambda b: (
                    b.state.tm.d, b.state.te.d, b.state.tpe.d, b.state.tpt.d,
                    b.state.te_ke.d, b.state.te_ie.d, b.state.te_pe.d
                ),
                pytree=block,
                replace=(tm, te, tpe, tpt, te_ke, te_ie, te_pe),
                at_dims=(0,) * 7,
                at_idxes=(itime,) * 7
            )
        if block.mode.is_baroclinic() and block.mode.is_moist():
            return block.backend.tree_at(
                where=lambda b: (
                    b.state.tm.d, b.state.tqv.d, b.state.te.d, b.state.tpe.d, b.state.tpt.d,
                    b.state.te_ke.d, b.state.te_ie.d, b.state.te_pe.d
                ),
                pytree=block,
                replace=(tm, tqv, te, tpe, tpt, te_ke, te_ie, te_pe),
                at_dims=(0,) * 8,
                at_idxes=(itime,) * 8
            )
        else:
            raise ValueError(
                f"Invalid equation mode for baroclinic diagnosis. Invalid mode={block.mode}"
            )

    def diagnose4advection(block: Block, itime_str: str) -> Block:
        # =======================================================
        # Diagnose total mass, total tracers
        # =========================================
        # Total mass
        tm = diagnose_tm(block, itime_str)
        # =========================================
        # Total tracers
        tq = diagnose_tq(block, itime_str)
        # =========================================
        # Global sums
        tm = diagnose_global_sum(tm)
        tq = diagnose_global_sum(tq)
        # =======================================================
        # Wrap diagnosed variables into block
        itime = block.get_time_idx(itime_str=itime_str)
        if block.mode.is_advection():
            return block.backend.tree_at(
                where=lambda b: (b.state.tm.d, b.state.tq.d),
                pytree=block,
                replace=(tm, tq),
                at_dims=(0, 0),
                at_idxes=(itime, itime),
            )
        else:
            raise ValueError(
                f"Invalid equation mode for advection diagnosis. Invalid mode={block.mode}"
            )

    if block.mode.is_barotropic():
        return diagnose4barotropic(block, itime_str)
    elif block.mode.is_baroclinic():
        return diagnose4baroclinic(block, itime_str)
    elif block.mode.is_advection():
        return diagnose4advection (block, itime_str)
    else:
        raise ValueError(
            f"Invalid equation mode for diagnosis. Invalid mode={block.mode}"
        )
