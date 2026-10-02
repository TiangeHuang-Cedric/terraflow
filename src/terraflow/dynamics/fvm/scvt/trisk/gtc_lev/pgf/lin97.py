from __future__ import annotations
from typing import TYPE_CHECKING


from terraflow.util import typing


if TYPE_CHECKING:
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block

    Int = typing.Int
    Array = typing.Array


def pgf_lin97_run(block: Block, star_idx: Int[Array, ""] | int) -> Block:
    bkd  = block.backend
    mesh = block.mesh
    full_kds, full_kde = mesh.full_kds, mesh.full_kde
    half_kds, half_kde = mesh.half_kds, mesh.half_kde
    e_ids, e_ide       = mesh.edge_ids, mesh.edge_ide
    c_ids, c_ihe       = mesh.cell_ids, mesh.cell_ihe[0]
    rd_o_cpd  = block.para   .rd_o_cpd .unwrap()
    qm        = block.tracers.qm       .unwrap()
    gz_lev    = block.state  .gz_lev   .unwrap()[star_idx, :, :]
    pk_lev    = block.aux    .pk_lev   .unwrap()
    dundt     = block.tend   .dundt    .unwrap()
    dundt_pgf = block.tend   .dundt_pgf.unwrap()
    # ========================================================
    kds, kde = half_kds, half_kde 
    if block.mode.is_hydrostatic():
        ph_lev = block.state.ph_lev.unwrap()[star_idx, :, :]
        pk_src = ph_lev[kds:kde+1, c_ids:c_ihe+1]
    else:
        p_lev = block.state.p_lev.unwrap()[star_idx, :, :]
        pk_src = p_lev[kds:kde+1, c_ids:c_ihe+1]

    pk_lev = bkd.slice_update(
        pk_lev,
        (slice(kds, kde+1), slice(c_ids, c_ihe+1)),
        pk_src ** rd_o_cpd
    )

    kds, kde = full_kds, full_kde 
    v1 = mesh.cell_of_edge[e_ids:e_ide+1, 0]
    v2 = mesh.cell_of_edge[e_ids:e_ide+1, 1]
    L = 1.0 + 0.5 * (qm[kds:kde+1, v1] + qm[kds:kde+1, v2])

    dgz13 = gz_lev[kds+1:kde+2, v1] - gz_lev[kds  :kde+1, v2]
    dgz42 = gz_lev[kds  :kde+1, v1] - gz_lev[kds+1:kde+2, v2]
    dpk24 = pk_lev[kds+1:kde+2, v2] - pk_lev[kds  :kde+1, v1]
    dpk13 = pk_lev[kds+1:kde+2, v1] - pk_lev[kds  :kde+1, v2]

    tmp = (
        (dgz13 * dpk24 + dgz42 * dpk13)
        / (dpk24 + dpk13)
        * mesh.edpnr_eddtg[e_ids:e_ide+1]
        / mesh.de[e_ids:e_ide+1]
        / L
    )
    dundt = bkd.slice_update(
        dundt,
        (slice(kds, kde+1), slice(e_ids, e_ide+1)),
        dundt[kds:kde+1, e_ids:e_ide+1] + tmp
    )
    dundt_pgf = bkd.slice_update(
        dundt_pgf,
        (slice(kds, kde+1), slice(e_ids, e_ide+1)),
        tmp
    )
    # ========================================================
    return bkd.tree_at(
        where=lambda b: (
            b.aux.pk_lev.d,
            b.tend.dundt.d,
            b.tend.dundt_pgf.d
        ),
        pytree=block,
        replace=(pk_lev, dundt, dundt_pgf)
    )
