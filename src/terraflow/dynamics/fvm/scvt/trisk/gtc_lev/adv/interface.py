from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional, Self
import jax

from terraflow.util.device import AbstractDevice, CPU
from terraflow.util.pytree import tree_at
from terraflow.mesh.scvt.operators import div_operator_2d
from terraflow.mesh.scvt.parallel import (
    fill_field2d_array_halo,
    fill_field3d_array_halo
)
from terraflow.dynamics.abstract import AbstractEquationAuxiliaryDataset
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.tracer import tracer_calc_qm
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv import batch
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv.batch import scvt_adv_fill_vhalo
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv import upwind
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv.limiter import (
    limiter_hflx, 
    limiter_vflx
)


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.util.planet import PlanetParameterSet
    from terraflow.model.config import UserConfig
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block

    INT   = typing.Int
    FLT   = typing.Float
    Array = typing.Array


class AdvectionAuxiliaryDataset(AbstractEquationAuxiliaryDataset):
    upwind: upwind.UpwindStencilDataset

    @classmethod
    def empty(cls) -> Self:
        return cls(
            initialized=False,
            upwind=upwind.UpwindStencilDataset.empty()
        )
        
    @classmethod
    def initialize(
        cls, 
        cfg: UserConfig,
        para: PlanetParameterSet,
        mesh: SCVTMesh,
        dev: AbstractDevice = CPU()
    ) -> Self:
        upwind_ds = upwind.UpwindStencilDataset.initialize(
            cfg=cfg,
            para=para,
            mesh=mesh,
            dev=dev
        )
        return cls(
            initialized=True,
            upwind=upwind_ds
        )


def calc_mass_hflx(
    batch: batch.AdvBatchType, 
    block: Block,
    m: FLT[Array, "nlev npoint"], 
    mfn: FLT[Array, "nlev nedge"], 
    dt: float
) -> FLT[Array, "nlev nedge"]:
    if batch.scheme_h == "upwind":
        return upwind.upwind_calc_mass_hflx(batch, block, m, mfn, dt)
    else:
        raise ValueError(f"calc_mass_hflx: unsupported scheme_h {batch.scheme_h} for batch {batch.name}!")


def calc_mass_vflx(
    batch: batch.AdvBatchType, 
    block: Block,
    m: FLT[Array, "nlev npoint"], 
    mfz: FLT[Array, "nlev nedge"], 
    dt: float
) -> FLT[Array, "nlev nedge"]:
    if batch.scheme_v == "upwind":
        return mfz
    else:
        return mfz


def calc_tracer_hflx(
    batch: batch.AdvBatchType, 
    block: Block,
    q: FLT[Array, "nlev npoint"], 
    qmfn: FLT[Array, "nlev nedge"], 
    dt: float
) -> FLT[Array, "nlev nedge"]:
    if batch.scheme_h == "upwind":
        return upwind.upwind_calc_tracer_hflx(batch, block, q, qmfn, dt)
    else:
        raise ValueError(f"calc_tracer_hflx: unsupported scheme_h {batch.scheme_h} for batch {batch.name}!")
    
    
def calc_tracer_vflx(
    batch: batch.AdvBatchType, 
    block: Block,
    q: FLT[Array, "nlev npoint"], 
    qmfz: FLT[Array, "nlev nedge"], 
    dt: float    
) -> FLT[Array, "nlev nedge"]:
    if batch.scheme_v == "upwind":
        return upwind.upwind_calc_tracer_vflx(batch, block, q, qmfz, dt)
    else:
        raise ValueError(f"calc_tracer_vflx: unsupported scheme_v {batch.scheme_v} for batch {batch.name}!")


def adv_prepare(block: Block, itime: int) -> Block:
    if not block.mode.restart:
        m = block.state.dmg.unwrap()[itime, :, :]
        for batch_name, batch in block.adv_batches.items():
            batch = batch.copy_m_old(m, block.mesh)
            block = tree_at(
                    where=lambda b: b.adv_batches[batch_name],
                    pytree=block,
                    replace=batch
                )
    return block


def adv_run_mass(block: Block, old: int, new: int, dt: Optional[float] = None) -> Block:
    
    dt_opt = block.adv_batch_bg.dt if dt is None else dt
    
    block = set_wind(block, old)
    
    mesh  = block.mesh
    m_old = block.state       .dmg.unwrap()[old, :, :]
    m_new = block.state       .dmg.unwrap()[new, :, :]
    mfn   = block.adv_batch_bg.mfn.unwrap()
    mfz   = block.adv_batch_bg.mfz.unwrap()
    dmdt  = block.adv_batch_bg.m  .unwrap()
    batch = block.adv_batch_bg
    # =================================================
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide
    mfn = calc_mass_hflx(batch, block, m_old, mfn,dt_opt)
    dmdt = div_operator_2d(
        mesh=mesh,
        f=mfn,
        div=dmdt,
        div_loc=block.adv_batch_bg.m.meta.loc
    )
    m_new = m_new.at[kds:kde+1, ids:ide+1].set(
        m_old[kds:kde+1, ids:ide+1] - dt_opt * dmdt[kds:kde+1, ids:ide+1]
    )
    mfz = calc_mass_vflx(batch, block, m_new, mfz, dt_opt)
    m_new = m_new.at[kds:kde+1, ids:ide+1].set(
        m_new[kds:kde+1, ids:ide+1] - dt_opt * (
            mfz[kds+1:kde+2, ids:ide+1] - mfz[kds:kde+1, ids:ide+1]
        )
    )
    m_new = fill_field2d_array_halo(
        halo=mesh.halo,
        f=m_new,
        meta=block.state.dmg.meta
    )
    # =================================================
    block = tree_at(
        where=lambda b: b.state.dmg.d,
        pytree=block,
        replace=m_new,
        at_dims=0,
        at_idxes=new
    )
    return tree_at(
        where=lambda b: (
            b.adv_batch_bg.mfn.d, b.adv_batch_bg.mfz.d,
            b.aux.mfn.d, b.aux.mfz_lev.d
        ),
        pytree=block,
        replace=(mfn, mfz, mfn, mfz)
    )


def adv_run_tracers_fn(
    adv_time_integrator: Callable[
    [
        Callable[
            [batch.AdvBatchType, Block, int, int, int, float, int]
        ]
    ],
    Callable[[Block], Block]
]
) -> Callable[[Block, int, int], Block]:
    adv_hori_run_tracers_fn = adv_time_integrator(adv_hori_run_tracers)
    adv_vert_run_tracers_fn = adv_time_integrator(adv_vert_run_tracers)
    
    def return_fn(block: Block, old_idx: int, new_idx: int) -> Block:
        if len(block.adv_batches.keys()) == 0:
            return block
        
        block = accum_wind(block, old_idx)
        
        block = adv_set_new_mass(block, new_idx)
        
        if block.cfg.adv_hvsplit_type == 1:
            block = adv_hori_run_tracers_fn(block)
            if block.mesh.full_nlev > 1:
                block = adv_vert_run_tracers_fn(block)
        elif block.cfg.adv_hvsplit_type == 2:
            if block.mesh.full_nlev > 1:
                block = adv_vert_run_tracers_fn(block)
            block = adv_hori_run_tracers_fn(block)
        else:
            raise ValueError(f"adv_run_tracers: unsupported adv_hvsplit_type {block.cfg.adv_hvsplit_type}!")
        block = tracer_calc_qm(block, "old")
        block = block.update_batches_istep()
        return block        

    return return_fn


def adv_set_new_mass(block: Block, new_idx: int) -> Block:
    for batch_name, batch in block.adv_batches.items():
        
        def do_nothing(
            batch: batch.AdvBatchType, 
            block: Block, 
            new_idx: int
        ) -> Block:
            return block
        
        def set_new_mass(
            batch: batch.AdvBatchType, 
            block: Block, 
            new_idx: int
        ) -> Block:
            if batch.loc == "cell":
                mesh = block.mesh
                batch_m = block.state.dmg.unwrap()[new_idx, :, :]
                batch_m = scvt_adv_fill_vhalo(
                    f=batch_m,
                    meta=batch.m_new.meta,
                    mesh=mesh,
                    no_negvals=True
                )
                batch_m = fill_field2d_array_halo(
                    halo=mesh.halo,
                    f=batch_m,
                    meta=batch.m_new.meta
                )
                block = tree_at(
                    where=lambda b: b.adv_batches[batch_name].m_new.d,
                    pytree=block,
                    replace=batch_m
                )
            return block
        
        block = jax.lax.cond(
            batch.is_run_tracer_istep(),
            set_new_mass,
            do_nothing,
            batch, 
            block, 
            new_idx
        )
    return block


def adv_hori_run_tracers(
    batch: batch.AdvBatchType,
    block: Block,
    old_idx: INT[Array, ""] | int,
    star_idx: INT[Array, ""] | int,
    new_idx: INT[Array, ""] | int,
    dt: float,
    substep: int 
) -> tuple[batch.AdvBatchType, Block]:
    
    def do_nothing(
        batch: batch.AdvBatchType,
        block: Block,
        old_idx: INT[Array, ""] | int,
        star_idx: INT[Array, ""] | int,
        new_idx: INT[Array, ""] | int,
        dt: float,
        substep: int 
    ) -> tuple[batch.AdvBatchType, Block]:
        return batch, block
    
    def hori_run_tracers(
        batch: batch.AdvBatchType,
        block: Block,
        old_idx: INT[Array, ""] | int,
        star_idx: INT[Array, ""] | int,
        new_idx: INT[Array, ""] | int,
        dt: float,
        substep: int
    ) -> tuple[batch.AdvBatchType, Block]:
        # ==========================================
        mesh     = block.mesh
        ids, ide = mesh.cell_ids, mesh.cell_ide
        kds, kde = mesh.full_kds, mesh.full_kde
        idx      = batch.idx
        q_old    = block.tracers.q.unwrap()[old_idx , idx, ...]
        q_star   = block.tracers.q.unwrap()[star_idx, idx, ...]
        q_new    = block.tracers.q.unwrap()[new_idx , idx, ...]
        q_new_   = block.tracers.q.unwrap()[new_idx ,      ...]
        for iq in range(len(idx)):
            q_old = fill_field3d_array_halo(
                halo=mesh.halo,
                f=q_old,
                i0=iq,
                meta=block.tracers.q.meta
            )
            q_star = fill_field3d_array_halo(
                halo=mesh.halo,
                f=q_star,
                i0=iq,
                meta=block.tracers.q.meta
            )
        # ==========================================
        # Run tracers
        def run_single_tracer(
            q_star: FLT[Array, "nlev npoints"],
            q_old: FLT[Array, "nlev npoints"],
            q_new: FLT[Array, "nlev npoints"], 
            batch: batch.AdvBatchType,
            block: Block,
            dt: float,
            substep: int
        ) -> tuple[FLT[Array, "nlev nedges"], batch.AdvBatchType]:
            m_old = batch.m    .unwrap()
            m_new = batch.m_new.unwrap()
            qmfn  = batch.qmfn.unwrap()
            qmfn  = calc_tracer_hflx(batch, block, q_star       , qmfn, dt)
            qmfn  = limiter_hflx    (batch, block, q_star, q_old, qmfn, dt, substep)
            dqdt  = div_operator_2d(mesh, qmfn, q_new, block.tracers.q.meta.loc)
            tmp   = m_old[kds:kde+1, ids:ide+1] * q_old[kds:kde+1, ids:ide+1] - dt * dqdt[kds:kde+1, ids:ide+1]
            q_new = q_new.at[kds:kde+1, ids:ide+1].set(tmp / m_new[kds:kde+1, ids:ide+1])
            return q_new, batch
        
        def run_tracers(
            q: FLT[Array, "ntracers nlev ncell"],
            q_old: FLT[Array, "ntracers nlev ncell"],
            q_new: FLT[Array, "ntracers nlev ncell"],
            batch: batch.AdvBatchType,
            block: Block,
            dt: float,
            substep: int        
        ) -> tuple[FLT[Array, "ntracers nlev nedges"], batch.AdvBatchType]:
            def body_fn(
                carry_batch: batch.AdvBatchType,
                xs: tuple[
                    FLT[Array, "nlev ncell"],
                    FLT[Array, "nlev ncell"],
                    FLT[Array, "nlev ncell"]
                ]
            ) -> tuple[batch.AdvBatchType, FLT[Array, "nlev nedges"]]:
                q_in, q_old_in, q_new_in = xs
                q_new, carry_batch = run_single_tracer(
                    q_star=q_in,
                    q_old=q_old_in,
                    q_new=q_new_in,
                    batch=carry_batch,
                    block=block,
                    dt=dt,
                    substep=substep
                )
                return carry_batch, q_new

            batch, q_new = jax.lax.scan(
                f=body_fn,
                init=batch,
                xs=(q, q_old, q_new)
            )
            return q_new, batch
        
        q_new, batch = run_tracers(q_star, q_old, q_new, batch, block, dt, substep)
        q_new_ = q_new_.at[idx, :, :].set(q_new)
        for iq in idx:
            q_new_ = fill_field3d_array_halo(
                halo=mesh.halo,
                f=q_new_,
                i0=iq,
                meta=block.tracers.q.meta
            )
        # ==========================================
        block = tree_at(
            where=lambda b: b.tracers.q.d,
            pytree=block,
            replace=q_new_,
            at_dims=0,
            at_idxes=new_idx
        )
        
        def substep_at_end(batch: batch.AdvBatchType) -> batch.AdvBatchType:
            batch = batch.copy_m_old(batch.m_new.unwrap(), block.mesh)
            batch = batch.swap_time_indices()
            return batch
        
        def substep_not_at_end(batch: batch.AdvBatchType) -> batch.AdvBatchType:
            return batch
        
        batch = jax.lax.cond(
           substep == block.const.adv_total_substeps - 1,
           substep_at_end,
           substep_not_at_end,
           batch
        )

        return batch, block
    
    return jax.lax.cond(
        batch.is_run_tracer_istep(),
        hori_run_tracers,
        do_nothing,
        batch,
        block,
        old_idx,
        star_idx,
        new_idx,
        dt,
        substep    
    )
    

def adv_vert_run_tracers(
    batch: batch.AdvBatchType,
    block: Block,
    old_idx: INT[Array, ""] | int,
    star_idx: INT[Array, ""] | int,
    new_idx: INT[Array, ""] | int,
    dt: float,
    substep: int 
) -> tuple[batch.AdvBatchType, Block]:

    def do_nothing(
        batch: batch.AdvBatchType,
        block: Block,
        old_idx: INT[Array, ""] | int,
        star_idx: INT[Array, ""] | int,
        new_idx: INT[Array, ""] | int,
        dt: float,
        substep: int 
    ) -> tuple[batch.AdvBatchType, Block]:
        return batch, block
    
    def vert_run_tracers(
        batch: batch.AdvBatchType,
        block: Block,
        old_idx: INT[Array, ""] | int,
        star_idx: INT[Array, ""] | int,
        new_idx: INT[Array, ""] | int,
        dt: float,
        substep: int
    ) -> tuple[batch.AdvBatchType, Block]:    
        # ==========================================
        mesh     = block.mesh
        ids, ide = mesh.cell_ids, mesh.cell_ide
        kds, kde = mesh.full_kds, mesh.full_kde
        idx      = batch.idx
        q_old    = block.tracers.q.unwrap()[old_idx , idx, ...]
        q_star   = block.tracers.q.unwrap()[star_idx, idx, ...]
        q_new    = block.tracers.q.unwrap()[new_idx , idx, ...]
        q_new_   = block.tracers.q.unwrap()[new_idx ,      ...]
        for iq in range(len(idx)):
            q_old = fill_field3d_array_halo(
                halo=mesh.halo,
                f=q_old,
                i0=iq,
                meta=block.tracers.q.meta
            )
            q_star = fill_field3d_array_halo(
                halo=mesh.halo,
                f=q_star,
                i0=iq,
                meta=block.tracers.q.meta
            )
        # ==========================================
        # Run tracers
        def run_single_tracer(
            q_star: FLT[Array, "nlev npoints"],
            q_old: FLT[Array, "nlev npoints"],
            q_new: FLT[Array, "nlev npoints"], 
            batch: batch.AdvBatchType,
            block: Block,
            dt: float,
            substep: int
        ) -> tuple[FLT[Array, "nlev nedges"], batch.AdvBatchType]:
            m_old = batch.m    .unwrap()
            m_new = batch.m_new.unwrap()
            qmfz  = batch.qmfz.unwrap()
            qmfz  = calc_tracer_vflx(batch, block, q_star       , qmfz, dt)
            qmfz  = limiter_vflx    (batch, block, q_star, q_old, qmfz, dt, substep)
            dqdt  = qmfz[kds+1:kde+2, ids:ide+1] - qmfz[kds:kde+1, ids:ide+1]
            tmp   = m_old[kds:kde+1, ids:ide+1] * q_old[kds:kde+1, ids:ide+1] - dt * dqdt
            q_new = q_new.at[kds:kde+1, ids:ide+1].set(tmp / m_new[kds:kde+1, ids:ide+1])
            return q_new, batch
        
        def run_tracers(
            q: FLT[Array, "ntracers nlev ncell"],
            q_old: FLT[Array, "ntracers nlev ncell"],
            q_new: FLT[Array, "ntracers nlev ncell"],
            batch: batch.AdvBatchType,
            block: Block,
            dt: float,
            substep: int        
        ) -> tuple[FLT[Array, "ntracers nlev nedges"], batch.AdvBatchType]:
            def body_fn(
                carry_batch: batch.AdvBatchType,
                xs: tuple[
                    FLT[Array, "nlev ncell"],
                    FLT[Array, "nlev ncell"],
                    FLT[Array, "nlev ncell"]
                ]
            ) -> tuple[batch.AdvBatchType, FLT[Array, "nlev nedges"]]:
                q_in, q_old_in, q_new_in = xs
                q_new, carry_batch = run_single_tracer(
                    q_star=q_in,
                    q_old=q_old_in,
                    q_new=q_new_in,
                    batch=carry_batch,
                    block=block,
                    dt=dt,
                    substep=substep
                )
                return carry_batch, q_new

            batch, q_new = jax.lax.scan(
                f=body_fn,
                init=batch,
                xs=(q, q_old, q_new)
            )
            return q_new, batch
        
        q_new, batch = run_tracers(q_star, q_old, q_new, batch, block, dt, substep)
        q_new_ = q_new_.at[idx, :, :].set(q_new)
        for iq in idx:
            q_new_ = fill_field3d_array_halo(
                halo=mesh.halo,
                f=q_new_,
                i0=iq,
                meta=block.tracers.q.meta
            )
        # ==========================================
        block = tree_at(
            where=lambda b: b.tracers.q.d,
            pytree=block,
            replace=q_new_,
            at_dims=0,
            at_idxes=new_idx
        )
        def substep_at_end(batch: batch.AdvBatchType) -> batch.AdvBatchType:
            batch = batch.copy_m_old(batch.m_new.unwrap(), block.mesh)
            batch = batch.swap_time_indices()
            return batch
        
        def substep_not_at_end(batch: batch.AdvBatchType) -> batch.AdvBatchType:
            return batch
        
        batch = jax.lax.cond(
           substep == block.const.adv_total_substeps - 1,
           substep_at_end,
           substep_not_at_end,
           batch
        )

        return batch, block
    
    return jax.lax.cond(
        batch.is_run_tracer_istep(),
        vert_run_tracers,
        do_nothing,
        batch,
        block,
        old_idx,
        star_idx,
        new_idx,
        dt,
        substep    
    )


def set_wind(block: Block, itime: int) -> Block:
    batch = block.adv_batch_bg.set_wind(
        un=block.state.un.unwrap()[itime, ...],
        w=block.aux.we_lev.unwrap(),
        mfn=block.aux.mfn.unwrap(),
        mfz=block.aux.mfz_lev.unwrap(),
        mesh=block.mesh
    )
    return tree_at(
        where=lambda b: b.adv_batch_bg,
        pytree=block,
        replace=batch
    )

        
def accum_wind(block: Block, old: int) -> Block:
    if block.mode.is_3d_model():
        mfn = block.aux.mfn.unwrap()
        mfz = block.aux.mfz_lev.unwrap()
        for batch_name, batch in block.adv_batches.items():
            batch = batch.accum_wind(mfn=mfn, mfz=mfz)
            block = tree_at(
                where=lambda b: b.adv_batches[batch_name],
                pytree=block,
                replace=batch
            )
    else:
        mfn = block.aux.mfn.unwrap()
        for batch_name, batch in block.adv_batches.items():
            batch = batch.accum_wind(mfn=mfn)
            block = tree_at(
                where=lambda b: b.adv_batches[batch_name],
                pytree=block,
                replace=batch
            )
    return block
