from __future__ import annotations
from typing import TYPE_CHECKING


from terraflow.util import log
from terraflow.mesh.scvt.parallel import fill_field1d_array_halo
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.damp.laplace import laplace_damp_1d


if TYPE_CHECKING:
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block as Block


def ref_calc_ps(block: Block) -> Block:
    bkd = block.backend
    if block.dev.is_root(): 
        log.logger.info(
            "Calculating reference surface pressure and its perturbation."
        )
    # ==================================================
    if block.para.planet == "earth":
        p0 = 1.0e5
        t0 = 300.0
        a  = 50.0
        mesh = block.mesh
        rd   = block.para  .rd .unwrap()
        gzs         = block.static.gzs        .unwrap()
        ref_ps      = block.static.ref_ps     .unwrap()
        ref_ps_smth = block.static.ref_ps_smth.unwrap()
        ref_ps_perb = block.static.ref_ps_perb.unwrap()
        # ==============================================
        ids, ide = mesh.cell_ids, mesh.cell_ide
        tmp = p0 * bkd.exp(
            -t0 / a
            + bkd.sqrt((t0 / a) ** 2 - 2 * gzs[ids:ide+1] / a / rd)
        )
        ref_ps = bkd.slice_update(ref_ps, slice(ids, ide+1), tmp)
        ref_ps = fill_field1d_array_halo(
            halo=mesh.halo,
            f=ref_ps,
            meta=block.static.ref_ps.meta
        )
        ref_ps_smth = ref_ps
        for _ in range(50):
            block, ref_ps_smth = laplace_damp_1d(
                block=block,
                f=ref_ps_smth,
                order=2,
                coef=1.0e6,
            )
        ref_ps_smth = fill_field1d_array_halo(
            halo=mesh.halo,
            f=ref_ps_smth,
            meta=block.static.ref_ps_smth.meta
        )
        ref_ps_perb = ref_ps - ref_ps_smth
        ref_ps_perb = fill_field1d_array_halo(
            halo=mesh.halo,
            f=ref_ps_perb,
            meta=block.static.ref_ps_perb.meta
        )
        # ==============================================
        block = bkd.tree_at(
            where=lambda b: (
                b.static.ref_ps.d,
                b.static.ref_ps_smth.d,
                b.static.ref_ps_perb.d
            ),
            pytree=block,
            replace=(ref_ps, ref_ps_smth, ref_ps_perb)
        )
    else:
        raise NotImplementedError(
            f"Reference surface pressure calculation is only implemented for Earth, but got {block.para.planet}"
        )
    # ==================================================
    return block
