from __future__ import annotations
from typing import TYPE_CHECKING
import jax


from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv.fct import (
    fct_calc_hflx, 
    fct_calc_vflx
)

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv import batch
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block

    Int = typing.Int
    Float = typing.Float
    Array = typing.Array


def limiter_hflx(
    batch: batch.AdvBatchType,
    block: Block,
    q: Float[Array, "nlev npoint"],
    q_old: Float[Array, "nlev npoint"],
    qmfn: Float[Array, "nlev nedge"],
    dt: float,
    substep: int
) -> Float[Array, "nlev npoint"]:
    if block.cfg.limiter_type == "fct":
        return jax.lax.cond(
            substep == block.const.adv_total_substeps - 1,
            lambda qmfn: fct_calc_hflx(batch, block, q_old, q_old, qmfn, dt),
            lambda qmfn: qmfn,
            qmfn,
        )
    elif block.cfg.limiter_type == "none":
        return qmfn
    else:
        raise ValueError(f"limiter_hflx: unsupported limiter_type {block.cfg.limiter_type}!")
    
    
def limiter_vflx(
    batch: batch.AdvBatchType,
    block: Block,
    q: Float[Array, "nlev npoint"],
    q_old: Float[Array, "nlev npoint"],
    qmfz: Float[Array, "nlev npoint"],
    dt: float,
    substep: int
) -> Float[Array, "nlev+1 npoint"]:
    if block.cfg.limiter_type == "fct":
        return jax.lax.cond(
            substep == block.const.adv_total_substeps - 1,
            lambda qmfz: fct_calc_vflx(batch, block, q_old, q_old, qmfz, dt),
            lambda qmfz: qmfz,
            qmfz,
        )
    elif block.cfg.limiter_type == "none":
        return qmfz
    else:
        raise ValueError(f"limiter_vflx: unsupported limiter_type {block.cfg.limiter_type}!") 

