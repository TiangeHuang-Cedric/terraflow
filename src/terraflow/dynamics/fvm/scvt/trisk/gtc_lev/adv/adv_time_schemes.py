from __future__ import annotations
from typing import Callable, TYPE_CHECKING
import jax

from terraflow.util.pytree import tree_at


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.model.config import UserConfig
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv.batch import AdvBatchType

    FLT   = typing.Float
    INT   = typing.Int
    Array = typing.Array
    
    
def adv_scheme_init(cfg: UserConfig) -> Callable[
    [
        Callable[
            [AdvBatchType, Block, int, int, int, float, int]
        ]
    ],
    Callable[[Block], Block]
]:
    if cfg.adv_time_scheme == "wrfrk3":
        adv_time_integrator = wrfrk3
    elif cfg.adv_time_scheme == "rk2":
        adv_time_integrator = rk2
    else:
        raise ValueError(f"Unsupported adv_time_scheme: {cfg.adv_time_scheme}!")
    return adv_time_integrator 
    
    
def step_adv(
    adv_operators: Callable[[AdvBatchType, Block, int, int, int, float, int]],
    batch: AdvBatchType,
    block: Block,
    old_idx: int,
    star_idx: int,
    new_idx: int,
    dt: float,
    substep: int,
) -> tuple[AdvBatchType, Block]:
    return adv_operators(
        batch,
        block,
        old_idx,
        star_idx,
        new_idx,
        dt,
        substep,
    )


def rk2(
    adv_operators: Callable[
        [AdvBatchType, Block, int, int, int, float, int]
    ]
) -> Callable[[Block], Block]:

    substep = step_adv
    
    def rk2_fn(block: Block) -> Block:

        for batch_name, batch in block.adv_batches.items():

            def do_nothing(
                carry_in: tuple[AdvBatchType, Block]
            ) -> tuple[AdvBatchType, Block]:
                return carry_in

            def run_tracers(
                carry_in: tuple[AdvBatchType, Block]
            ) -> tuple[AdvBatchType, Block]:
                batch, block = carry_in
                batch, block = substep(
                    adv_operators=adv_operators,
                    batch=batch,
                    block=block,
                    old_idx=batch.old,
                    star_idx=batch.old,
                    new_idx=batch.new,
                    dt=batch.dt / 2.0,
                    substep=0,
                )
                batch, block = substep(
                    adv_operators=adv_operators,
                    batch=batch,
                    block=block,
                    old_idx=batch.old,
                    star_idx=batch.new,
                    new_idx=batch.new,
                    dt=batch.dt,
                    substep=1,
                )
                return batch, block

            batch, block = jax.lax.cond(
                batch.is_run_tracer_istep(),
                run_tracers,
                do_nothing,
                (batch, block)
            )

            block = tree_at(
                lambda b: b.adv_batches[batch_name],
                block,
                batch
            )

        return block

    return rk2_fn
    

def wrfrk3(
    adv_operators: Callable[[AdvBatchType, Block, int, int, int, float, int]]
) -> Callable[[Block], Block]:

    substep = step_adv

    def wrfrk3_fn(block: Block) -> Block:

        for batch_name, batch in block.adv_batches.items():

            def do_nothing(
                carry_in: tuple[AdvBatchType, Block]
            ) -> tuple[AdvBatchType, Block]:
                return carry_in

            def run_tracers(
                carry_in: tuple[AdvBatchType, Block]
            ) -> tuple[AdvBatchType, Block]:
                batch, block = carry_in
                batch, block = substep(
                    adv_operators=adv_operators,
                    batch=batch,
                    block=block,
                    old_idx=batch.old,
                    star_idx=batch.old,
                    new_idx=batch.new,
                    dt=batch.dt / 3.0,
                    substep=0,
                )
                batch, block = substep(
                    adv_operators=adv_operators,
                    batch=batch,
                    block=block,
                    old_idx=batch.old,
                    star_idx=batch.new,
                    new_idx=2,
                    dt=batch.dt / 2.0,
                    substep=1,
                )
                batch, block = substep(
                    adv_operators=adv_operators,
                    batch=batch,
                    block=block,
                    old_idx=batch.old,
                    star_idx=2,
                    new_idx=batch.new,
                    dt=batch.dt,
                    substep=2,
                )
                return batch, block

            batch, block = jax.lax.cond(
                batch.is_run_tracer_istep(),
                run_tracers,
                do_nothing,
                (batch, block)
            )

            block = tree_at(
                lambda b: b.adv_batches[batch_name],
                block,
                batch
            )

        return block

    return wrfrk3_fn
