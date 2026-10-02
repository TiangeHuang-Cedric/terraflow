from __future__ import annotations
from typing import Callable, TYPE_CHECKING


from terraflow.mesh.scvt.parallel import fill_field1d_array_halo, fill_field2d_array_halo
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.space_operators import (
    calc_mg,
    calc_dmg,
    calc_ph,
    calc_t
)


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.model.config import UserConfig
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block

    FLT   = typing.Float
    INT   = typing.Int
    Array = typing.Array

all_pass = "all_pass"
forward_pass = "forward_pass"
backward_pass = "backward_pass"


def time_scheme_init(cfg: UserConfig) -> Callable[
    [
        Callable[[Block, int, int, int, float, str, int], Block],
        bool
    ],
    Callable[[Block, int, int, float], Block]
]:
    if cfg.time_scheme == "wrfrk3":
        time_integrator = wrfrk3
    elif cfg.time_scheme == "euler":
        time_integrator = euler
    elif cfg.time_scheme == "rk2":
        time_integrator = rk2
    elif cfg.time_scheme == "pc2":
        time_integrator = pc2
    else:
        raise ValueError(f"Unsupported time_scheme: {cfg.time_scheme}!")
    return time_integrator


def step_all(
    space_operators: Callable[[Block, int, int, int, float, str, int], Block],
    block: Block,
    old_idx: int,
    star_idx: int,
    new_idx: int,
    dt: float,
    substep: int
) -> Block:
    block = space_operators(
        block,
        old_idx,
        star_idx,
        new_idx,
        dt,
        all_pass,
        substep,
    )
    block = update_state(
        block=block,
        old_idx=old_idx,
        new_idx=new_idx,
        dt=dt,
        substep=substep,
        pass_tag=all_pass
    )
    return block


def step_forward_backward(
    space_operators: Callable[[Block, int, int, int, float, str, int], Block],
    block: Block,
    old_idx: int,
    star_idx: int,
    new_idx: int,
    dt: float,
    substep: int,
) -> Block:
    block = space_operators(
        block,
        old_idx,
        star_idx,
        new_idx,
        dt,
        forward_pass,
        substep
    )
    block = update_state(
        block=block,
        old_idx=old_idx,
        new_idx=new_idx,
        dt=dt,
        substep=substep,
        pass_tag=forward_pass
    )
    block = space_operators(
        block,
        old_idx,
        star_idx,
        new_idx,
        dt,
        backward_pass,
        substep
    )
    block = update_state(
        block=block,
        old_idx=old_idx,
        new_idx=new_idx,
        dt=dt,
        substep=substep,
        pass_tag=backward_pass
    )
    return block


def rk2(space_operators: Callable, forward_backward: bool = True) -> Callable[[Block, int, int, float], Block]:
    substep = step_forward_backward if forward_backward else step_all

    def rk2_fn(block: Block, old: int, new: int, dt: float) -> Block:
        block = substep(
            space_operators=space_operators,
            block=block,
            old_idx=old,
            star_idx=old,
            new_idx=new,
            dt=dt / 2.0,
            substep=0,
        )
        block = substep(
            space_operators=space_operators,
            block=block,
            old_idx=old,
            star_idx=new,
            new_idx=new,
            dt=dt,
            substep=1,
        )
        return block

    return rk2_fn


def pc2(space_operators: Callable, forward_backward: bool = True) -> Callable[[Block, int, int, float], Block]:
    substep = step_forward_backward if forward_backward else step_all

    def pc2_fn(block: Block, old: int, new: int, dt: float) -> Block:
        block = substep(
            space_operators=space_operators,
            block=block,
            old_idx=old,
            star_idx=old,
            new_idx=new,
            dt=dt / 2.0,
            substep=0,
        )
        block = substep(
            space_operators=space_operators,
            block=block,
            old_idx=old,
            star_idx=new,
            new_idx=2,
            dt=dt / 2.0,
            substep=1,
        )
        block = substep(
            space_operators=space_operators,
            block=block,
            old_idx=old,
            star_idx=2,
            new_idx=new,
            dt=dt,
            substep=2,
        )
        return block

    return pc2_fn


def wrfrk3(space_operators: Callable, forward_backward: bool = True) -> Callable[[Block, int, int, float], Block]:
    substep = step_forward_backward if forward_backward else step_all

    def wrfrk3_fn(block: Block, old: int, new: int, dt: float) -> Block:
        block = substep(
            space_operators=space_operators,
            block=block,
            old_idx=old,
            star_idx=old,
            new_idx=new,
            dt=dt / 3.0,
            substep=0,
        )
        block = substep(
            space_operators=space_operators,
            block=block,
            old_idx=old,
            star_idx=new,
            new_idx=2,
            dt=dt / 2.0,
            substep=1,
        )
        block = substep(
            space_operators=space_operators,
            block=block,
            old_idx=old,
            star_idx=2,
            new_idx=new,
            dt=dt,
            substep=2,
        )
        return block

    return wrfrk3_fn


def euler(space_operators: Callable, dt: float, forward_backward: bool = True) -> Callable[[Block, int, int, float], Block]:
    substep = step_forward_backward if forward_backward else step_all

    def euler_fn(block: Block, old: int, new: int, dt: float) -> Block:
        block = substep(
            space_operators=space_operators,
            block=block,
            old_idx=old,
            star_idx=old,
            new_idx=new,
            dt=dt,
            substep=0,
        )
        return block

    return euler_fn


def update_mgs(
    block: Block,
    old_idx: int,
    new_idx: int,
    dt: float,
    pass_tag: str,
    substep: int
) -> Block:
    mode = block.mode
    mesh = block.mesh
    bkd = block.backend
    # =================================================
    if pass_tag == all_pass or pass_tag == forward_pass:
        if mode.is_baroclinic():
            # ------ Update mgs ------
            old_mgs = block.state.mgs   .unwrap()[old_idx, :]
            new_mgs = block.state.mgs   .unwrap()[new_idx, :]
            dmgsdt  = block.tend .dmgsdt.unwrap()

            ids, ide = mesh.cell_ids, mesh.cell_ide
            new_mgs = bkd.slice_update(new_mgs, slice(ids, ide+1),
                old_mgs[ids:ide+1] + dt * dmgsdt[ids:ide+1]
            )
            new_mgs = fill_field1d_array_halo(
                halo=mesh.halo,
                f=new_mgs,
                meta=block.state.mgs.meta
            )
            block = bkd.tree_at(
                where=lambda b: b.state.mgs.d,
                pytree=block,
                replace=new_mgs,
                at_dims=0,
                at_idxes=new_idx
            )
            block = calc_mg (block, new_idx)
            block = calc_dmg(block, new_idx)
            block = calc_ph (block, new_idx)
    # =================================================
    return block


def update_pt(
    block: Block,
    old_idx: int,
    new_idx: int,
    dt: float,
    pass_tag: str,
    substep: int
) -> Block:
    mode = block.mode
    mesh = block.mesh
    bkd = block.backend
    # =================================================
    if mode.is_baroclinic():
        if pass_tag == all_pass or pass_tag == forward_pass:
            # ------ Update pt ------
            old_dmg    = block.state.dmg       .unwrap()[old_idx, :, :]
            new_dmg    = block.state.dmg       .unwrap()[new_idx, :, :]
            dptdt      = block.tend .dptdt     .unwrap()
            dptdt_damp = block.aux  .dptdt_damp.unwrap()

            old_pt  = block.state.pt    .unwrap()[old_idx, :, :]
            new_pt  = block.state.pt    .unwrap()[new_idx, :, :]

            kds, kde = mesh.full_kds, mesh.full_kde
            ids, ide = mesh.cell_ids, mesh.cell_ide
            new_pt = bkd.slice_update(new_pt, (slice(kds, kde+1), slice(ids, ide+1)),
                old_pt[kds:kde+1, ids:ide+1] * old_dmg[kds:kde+1, ids:ide+1]
                + dt * dptdt[kds:kde+1, ids:ide+1]
            )
            if block.cfg.use_smag_damp:
                new_pt = bkd.slice_update(new_pt, (slice(kds, kde+1), slice(ids, ide+1)),
                    new_pt[kds:kde+1, ids:ide+1] + dt * dptdt_damp[kds:kde+1, ids:ide+1]
                )
            new_pt = bkd.slice_update(new_pt, (slice(kds, kde+1), slice(ids, ide+1)),
                new_pt[kds:kde+1, ids:ide+1] / new_dmg[kds:kde+1, ids:ide+1]
            )
            new_pt = fill_field2d_array_halo(
                halo=mesh.halo,
                f=new_pt,
                meta=block.state.pt.meta
            )
            block = bkd.tree_at(
                where=lambda b: b.state.pt.d,
                pytree=block,
                replace=new_pt,
                at_dims=0,
                at_idxes=new_idx
            )
            block = calc_t  (block, new_idx)
    # =================================================
    return block


def update_gz(
    block: Block,
    old_idx: int,
    new_idx: int,
    dt: float,
    pass_tag: str,
    substep: int
) -> Block:
    mode = block.mode
    mesh = block.mesh
    bkd = block.backend
    # =================================================
    if not mode.is_baroclinic():
        if pass_tag == all_pass or pass_tag == forward_pass:
            old_gz  = block.state.gz   .unwrap()[old_idx, :, :]
            new_gz  = block.state.gz   .unwrap()[new_idx, :, :]
            dgzdt   = block.tend .dgzdt.unwrap()

            kds, kde = mesh.full_kds, mesh.full_kde
            ids, ide = mesh.cell_ids, mesh.cell_ide
            new_gz = bkd.slice_update(new_gz, (slice(kds, kde+1), slice(ids, ide+1)),
                old_gz[kds:kde+1, ids:ide+1] + dt * dgzdt[kds:kde+1, ids:ide+1]
            )
            new_gz = fill_field2d_array_halo(
                halo=mesh.halo,
                f=new_gz,
                meta=block.state.gz.meta
            )
            block = bkd.tree_at(
                where=lambda b: b.state.gz.d,
                pytree=block,
                replace=new_gz,
                at_dims=0,
                at_idxes=new_idx
            )
            block = calc_dmg(block, new_idx)
    # =================================================
    return block


def update_un(
    block: Block,
    old_idx: int,
    new_idx: int,
    dt: float,
    pass_tag: str,
    substep: int
) -> Block:
    mesh = block.mesh
    bkd = block.backend
    # =================================================
    if pass_tag == all_pass or pass_tag == backward_pass:
        old_un     = block.state.un       .unwrap()[old_idx, :, :]
        new_un     = block.state.un       .unwrap()[new_idx, :, :]
        dundt      = block.tend.dundt     .unwrap()
        dundt_damp = block.aux .dundt_damp.unwrap()

        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        new_un = bkd.slice_update(new_un, (slice(kds, kde+1), slice(ids, ide+1)),
            old_un[kds:kde+1, ids:ide+1] + dt * dundt[kds:kde+1, ids:ide+1]
        )
        if block.cfg.use_smag_damp:
            new_un = bkd.slice_update(new_un, (slice(kds, kde+1), slice(ids, ide+1)),
                new_un[kds:kde+1, ids:ide+1] + dt * dundt_damp[kds:kde+1, ids:ide+1]
            )
        new_un = fill_field2d_array_halo(
            halo=mesh.halo,
            f=new_un,
            meta=block.state.un.meta
        )
        # =================================================
        block = bkd.tree_at(
            where=lambda b: b.state.un.d,
            pytree=block,
            replace=new_un,
            at_dims=0,
            at_idxes=new_idx
        )
    return block


def update_state(
    block: Block,
    old_idx: int,
    new_idx: int,
    dt: float,
    pass_tag: str,
    substep: int
) -> Block:
    block = update_mgs(block, old_idx, new_idx, dt, pass_tag, substep)
    block = update_pt (block, old_idx, new_idx, dt, pass_tag, substep)
    block = update_gz (block, old_idx, new_idx, dt, pass_tag, substep)
    block = update_un (block, old_idx, new_idx, dt, pass_tag, substep)
    return block
