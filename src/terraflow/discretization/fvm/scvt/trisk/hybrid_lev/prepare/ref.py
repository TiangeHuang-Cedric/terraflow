from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp

from terraflow.util import log
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at
from terraflow.discretization.fvm.scvt.trisk.common.damp.laplace_damp import laplace_damp_1d

if TYPE_CHECKING:
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as BLOCK


def ref_calc_ps(block: BLOCK) -> BLOCK:
    log.logger.info(
        "Calculating reference surface pressure and its perturbation.")
    # ==================================================
    if block.para.planet == "earth":
        p0 = 1.0e5
        t0 = 300.0
        a = 50.0
        rd = block.para.rd.unwrap()
        gzs = block.static.gzs.unwrap()
        # ==============================================
        ref_ps = p0 * jnp.exp(
            -t0 / a
            + jnp.sqrt((t0 / a) ** 2 - 2 * gzs / a / rd)
        )
        ref_ps_smth = ref_ps
        for _ in range(50):
            block, ref_ps_smth = laplace_damp_1d(
                block=block,
                f_1d=ref_ps_smth.reshape(1, -1),
                order=2,
                coef=1.0e6,
            )
        ref_ps_smth = ref_ps_smth.squeeze()
        ref_ps_perb = ref_ps - ref_ps_smth
        # ==============================================
        block = tree_at(
            coord=block.coord,
            where=lambda b: (
                b.static.ref_ps,
                b.static.ref_ps_smth,
                b.static.ref_ps_perb
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
