from __future__ import annotations
from typing import Optional, Self, TYPE_CHECKING
import equinox as eqx
import jax
import jax.numpy as jnp

from terraflow.util import typing
from terraflow.mesh.scvt.field import (
    SCVTFieldLike,
    SCVTFieldMeta,
    SCVTField2D,  
    SCVTFields
)
from terraflow.mesh.scvt.parallel import fill_field2d_array_halo
from terraflow.model.abstract import ModelMode

if TYPE_CHECKING:
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
    

INT   = typing.Int
FLT   = typing.Float
Array = typing.Array

NLEVP = "ilev"
NLEV  = "lev"
NCELL = "cell"
NEDGE = "edge"
NVTX  = "vtx"


class AdvBatchType(SCVTFields):
    name       : str         = eqx.field(default=""   , static=True)
    loc        : str         = eqx.field(default=""   , static=True)
    scheme_h   : str         = eqx.field(default=""   , static=True)
    scheme_v   : str         = eqx.field(default=""   , static=True)
    dynamic    : bool        = eqx.field(default=False, static=True)
    passive    : bool        = eqx.field(default=True , static=True)
    semilag    : bool        = eqx.field(default=False, static=True)
    old        : int         = eqx.field(default=0                 )
    new        : int         = eqx.field(default=1                 )
    dt         : float       = eqx.field(default=0    , static=True)
    nsteps     : int         = eqx.field(default=0    , static=True)
    istep      : int         = eqx.field(default=0                 )
    idx        : list[int]   = eqx.field(default_factory=list, static=True     )
    m          : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D() )  # Dry-air weight at n time level
    m_new      : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D() )  # Dry-air weight at n+1 time level
    mfn        : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D() )  # Mass flux in normal direction
    mfz        : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D() )  # Mass flux in z direction
    un         : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D() )
    w          : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D() )
    qmfn       : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D() )
    qmfz       : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D() )
    # FCT variables
    qmfn_lo    : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D() )
    qmfz_lo    : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D() )
    afn        : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D() )
    afz        : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D() )
    td         : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D() )
    r_in       : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D() )
    r_out      : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D() ) 
    
    @classmethod
    def from_mesh(
        cls,
        mode: ModelMode,
        mesh: SCVTMesh,
        scheme: str,
        batch_name: str,
        batch_loc: str,
        dt: float,
        dt_dyn: float,
        dynamic: bool,
        passive: bool,
        semilag: bool,
        idx: Optional[list[int]] = None,
    ):
        if batch_loc not in {"cell", "lev", "vtx"}:
            raise ValueError(f"Unsupported batch_loc {batch_loc} for batch {batch_name}!")
        if idx is None:
            idx = []
        if scheme.count(":") == 1:
            scheme_h = scheme.split(":")[0]
            scheme_v = scheme.split(":")[1]
        else:
            scheme_h = scheme
            scheme_v = scheme
        # =========================================================
        if batch_loc == "cell":
            loc0 = "cell"
            locn = "edge"
            locz = "lev"
        elif batch_loc == "lev":
            loc0 = "lev"
            locn = "lev_edge"
            locz = "cell"
        elif batch_loc == "vtx":
            loc0 = "vtx"
            locn = "edge"
            locz = "none"
        else:
            raise NotImplementedError(
                f"Invalid batch location {batch_loc}."
            )
        fields: dict[str, SCVTFieldLike] = {}
        if mode.is_baroclinic() or mode.is_advection():
            fields["m"] = SCVTField2D.from_mesh(
                name=batch_name + "_m",
                long_name="Dry air weight at n time level",
                units="Pa",
                loc=loc0,
                mesh=mesh,
                output="h0" if mode.is_advection() else "",
            )
            fields["m_new"] = SCVTField2D.from_mesh(
                name=batch_name + "_m_new",
                long_name="Dry air weight at n+1 time level",
                units="Pa",
                loc=loc0,
                mesh=mesh,
                output="",
            )
        if mode.is_baroclinic() or mode.is_advection():
            fields["mfn"] = SCVTField2D.from_mesh(
                name=batch_name + "_mfn",
                long_name="Dry air weight at edge",
                units="Pa",
                loc=locn,
                mesh=mesh,
                output="h1" if mode.is_advection() else "",
            )
        if mode.is_baroclinic() or mode.is_advection():
            fields["mfz"] = SCVTField2D.from_mesh(
                name=batch_name + "_mfz",
                long_name="Dry air weight at level",
                units="Pa",
                loc=locz,
                mesh=mesh,
                output="h1" if mode.is_advection() else "",
            )
        if mode.is_baroclinic() or mode.is_advection():
            fields["un"] = SCVTField2D.from_mesh(
                name=batch_name + "_un",
                long_name="Normal wind at edge",
                units="m s-1",
                loc=locn,
                mesh=mesh,
                output="h0" if mode.is_advection() else "",
            )
        if mode.is_baroclinic() or mode.is_advection():
            fields["w"] = SCVTField2D.from_mesh(
                name=batch_name + "_w",
                long_name="Vertical coordinate velocity",
                units="s-1",
                loc=locz,
                mesh=mesh,
                output="h0" if mode.is_advection() else "",
            )
        if mode.is_baroclinic() or mode.is_advection():
            fields["qmfn"] = SCVTField2D.from_mesh(
                name=batch_name + "_qmfn",
                long_name="Tracer mass flux in normal direction",
                units="Pa kg kg-1 m s-1",
                loc=locn,
                mesh=mesh,
                output="h1",
            )
        if mode.is_baroclinic() or mode.is_advection():
            fields["qmfz"] = SCVTField2D.from_mesh(
                name=batch_name + "_qmfz",
                long_name="Tracer mass flux in z direction",
                units="Pa kg kg-1 m s-1",
                loc=locz,
                mesh=mesh,
                output="h1",
            )
        if mode.is_baroclinic() or mode.is_advection():
            fields["qmfn_lo"] = SCVTField2D.from_mesh(
                name=batch_name + "_qmfn_lo",
                long_name="Tracer mass flux in normal direction",
                units="Pa kg kg-1 m s-1",
                loc=locn,
                mesh=mesh,
                output=""
            )
            fields["qmfz_lo"] = SCVTField2D.from_mesh(
                name=batch_name + "_qmfz_lo",
                long_name="Tracer mass flux in z direction",
                units="Pa kg kg-1 m s-1",
                loc=locz,
                mesh=mesh,
                output=""
            )
            fields["afn"] = SCVTField2D.from_mesh(
                name=batch_name + "_afn",
                long_name="Anti-diffusion flux in normal direction",
                units="Pa kg kg-1 m s-1",
                loc=locn,
                mesh=mesh,
                output=""
            )
            fields["afz"] = SCVTField2D.from_mesh(
                name=batch_name + "_afz",
                long_name="Anti-diffusion flux in z direction",
                units="Pa kg kg-1 m s-1",
                loc=locz,
                mesh=mesh,
                output=""
            )
            fields["td"] = SCVTField2D.from_mesh(
                name=batch_name + "_td",
                long_name="TD solution flux at cell",
                units="Pa kg kg-1",
                loc=loc0,
                mesh=mesh,
                output=""
            )
            fields["r_in"] = SCVTField2D.from_mesh(
                name=batch_name + "_r_in",
                long_name="Temporal array for fct",
                units="1",
                loc=loc0,
                mesh=mesh,
                output=""
            )
            fields["r_out"] = SCVTField2D.from_mesh(
                name=batch_name + "_r_out",
                long_name="Temporal array for fct",
                units="1",
                loc=loc0,
                mesh=mesh,
                output=""
            )

        return cls.from_fields(
            fields=fields,
            name=batch_name,
            loc=batch_loc,
            dt=dt,
            nsteps=int(dt / dt_dyn),
            istep=jnp.array(0),
            idx=idx,
            old=jnp.array(0),
            new=jnp.array(1),
            scheme_h=scheme_h,
            scheme_v=scheme_v,
            dynamic=dynamic,
            passive=passive,
            semilag=semilag
        )
        
    def copy_m_old(self, m: FLT[Array, "nlev ncell"], mesh: SCVTMesh) -> Self:
        m = scvt_adv_fill_vhalo(
            f=m,
            meta=self.m.meta,
            mesh=mesh,
            no_negvals=True
        )
        if not self.passive or self.semilag:
            m = fill_field2d_array_halo(
                halo=mesh.halo,
                f=m,
                meta=self.m.meta
            )
        return eqx.tree_at(
            where=lambda b: b.m.d,
            pytree=self,
            replace=m
        )

    def set_wind(
        self,
        un: FLT[Array, "nlev nedge"],
        w: Optional[FLT[Array, "nlevp ncell"]] = None,
        mfn: Optional[FLT[Array, "nlev nedge"]] = None,
        mfz: Optional[FLT[Array, "nlevp ncell"]] = None,
        m: Optional[FLT[Array, "nlev ncell"]] = None,
        m_new: Optional[FLT[Array, "nlev ncell"]] = None,
        mesh: Optional[SCVTMesh] = None
    ) -> Self:
        self = eqx.tree_at(
            where=lambda b: b.un.d,
            pytree=self,
            replace=un
        )
        if w is not None:
            self = eqx.tree_at(
                where=lambda b: b.w.d,
                pytree=self,
                replace=w
            )
        if mfn is not None:
            self = eqx.tree_at(
                where=lambda b: b.mfn.d,
                pytree=self,
                replace=mfn
            )
        if mfz is not None:
            self = eqx.tree_at(
                where=lambda b: b.mfz.d,
                pytree=self,
                replace=mfz
            )
        if m is not None:
            if mesh is None:
                raise ValueError("AdvBatchType.set_wind: mesh must be provided when m is present.")
            self = self.copy_m_old(m, mesh)
        if m_new is not None:
            self = eqx.tree_at(
                where=lambda b: b.m_new.d,
                pytree=self,
                replace=m_new
            ) 
        return self

    def accum_wind(
        self,
        mfn: FLT[Array, "nlev nedge"],
        mfz: Optional[FLT[Array, "nlevp ncell"]] = None,
    ) -> Self:

        def do_nothing(batch: Self) -> Self:
            return batch

        def istep_at_reset(batch: Self) -> Self:
            # istep == 0
            mfn_zeros = jnp.zeros_like(mfn)
            if mfz is not None:
                mfz_zeros = jnp.zeros_like(mfz)
                return eqx.tree_at(
                    lambda b: (b.mfn.d, b.mfz.d),
                    batch,
                    (mfn_zeros, mfz_zeros)
                )
            else:
                return eqx.tree_at(
                    lambda b: b.mfn.d,
                    batch,
                    mfn_zeros
                )

        def istep_at_end(batch: Self) -> Self:
            # istep == nsteps -1
            nsteps = batch.nsteps
            mfn_at_end = batch.mfn.d / nsteps

            if mfz is not None:
                mfz_at_end = batch.mfz.d / nsteps
                return eqx.tree_at(
                    lambda b: (b.mfn.d, b.mfz.d),
                    batch,
                    (mfn_at_end, mfz_at_end)
                )
            else:
                return eqx.tree_at(
                    lambda b: b.mfn.d,
                    batch,
                    mfn_at_end
                )

        def istep_at_accumulating(batch: Self) -> Self:
            # 0 <= istep <= nsteps -1
            mfn_accum = batch.mfn.d + mfn
            if mfz is not None:
                mfz_accum = batch.mfz.d + mfz
                return eqx.tree_at(
                    lambda b: (b.mfn.d, b.mfz.d),
                    batch,
                    (mfn_accum, mfz_accum)
                )
            else:
                return eqx.tree_at(
                    lambda b: b.mfn.d,
                    batch,
                    mfn_accum
                )

        self = jax.lax.cond(
            pred=self.istep == 0,
            true_fun=istep_at_reset,
            false_fun=do_nothing,
            operand=self
        )

        self = jax.lax.cond(
            pred=jnp.logical_and(
                self.istep >= 0, self.istep <= self.nsteps - 1
            ),
            true_fun=istep_at_accumulating,
            false_fun=do_nothing,
            operand=self
        )

        self = jax.lax.cond(
            pred=self.istep == self.nsteps - 1,
            true_fun=istep_at_end,
            false_fun=do_nothing,
            operand=self
        )

        return self
    
    def swap_time_indices(self) -> Self:
        return eqx.tree_at(
            lambda b: (b.old, b.new),
            self,
            (self.new, self.old)
        )

    def get_time_idx(self, itime_str: str) -> int:
        if itime_str == "old":
            return self.old
        elif itime_str == "new":
            return self.new
        else:
            raise ValueError(
                f"Invalid time index '{itime_str}'. Expected 'old' or 'new'."
            )

    def is_run_tracer_istep(self) -> bool:
        """Determine whether to run tracer."""
        return self.istep == (self.nsteps - 1)

    def update_istep(self) -> Self:
        istep = self.istep
        nsteps = self.nsteps

        def not_reset_istep(batch: Self):
            return eqx.tree_at(
                where=lambda b: b.istep,
                pytree=batch,
                replace=istep + 1
            )

        def reset_istep(batch: Self):
            return eqx.tree_at(
                where=lambda b: b.istep,
                pytree=batch,
                replace=0
            )

        return jax.lax.cond(
            pred=istep == nsteps - 1,
            true_fun=reset_istep,
            false_fun=not_reset_istep,
            operand=self
        )


def scvt_adv_fill_vhalo(
    f: FLT[Array, "nlev ncell"],
    meta: SCVTFieldMeta,
    mesh: SCVTMesh,
    no_negvals: bool
) -> FLT[Array, "nlev ncell"]:
    if meta.loc == "cell":
        kds, kde = mesh.full_kds, mesh.full_kde
        kms, kme = mesh.full_kms, mesh.full_kme
    elif meta.loc == "lev":
        kds, kde = mesh.half_kds, mesh.half_kde
        kms, kme = mesh.half_kms, mesh.half_kme
    else:
        raise ValueError(f"scvt_adv_fill_vhalo: unsupported location {meta.loc}!")

    ids, ide = mesh.cell_ids, mesh.cell_ide
    for k in range(kds - 1, kms - 1, -1):
        f = f.at[k, ids:ide+1].set(
            3 * f[k+1, ids:ide+1] - 3 * f[k+2, ids:ide+1] + f[k+3, ids:ide+1]
        )
    for k in range(kde + 1, kme + 1):
        f = f.at[k, ids:ide+1].set(
            3 * f[k-1, ids:ide+1] - 3 * f[k-2, ids:ide+1] + f[k-3, ids:ide+1]
        )

    if no_negvals:
        if kms < kds:
            top = f[kms:kds, ids:ide+1]
            top_fill = jnp.broadcast_to(f[kds, ids:ide+1], top.shape)
            top = jnp.where(top < 0, top_fill, top)
            f = f.at[kms:kds, ids:ide+1].set(top)
        if kde < kme:
            bottom = f[kde+1:kme+1, ids:ide+1]
            bottom_fill = jnp.broadcast_to(f[kde, ids:ide+1], bottom.shape)
            bottom = jnp.where(bottom < 0, bottom_fill, bottom)
            f = f.at[kde+1:kme+1, ids:ide+1].set(bottom)

    return f
