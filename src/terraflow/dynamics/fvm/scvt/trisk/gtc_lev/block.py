from __future__ import annotations
from typing import (
    Self,
    cast,
    Optional,
    TYPE_CHECKING
)
import jax.numpy as jnp
import numpy as np
import xarray as xr


from terraflow.util.device import AbstractDevice
from terraflow.util.pytree import tree_at
from terraflow.util.uxarray import UGRIDDataset
from terraflow.model.config import Const
from terraflow.mesh.scvt.mesh_type import SCVTMesh
from terraflow.mesh.scvt.field import SCVTFieldMeta, meta2ugrid
from terraflow.mesh.scvt.operators import wind_c2a_operator
from terraflow.mesh.scvt.parallel import fill_field2d_array_halo
from terraflow.vert_coord.abstract import AbstractGeneralizedTerrainFollowingCoordinate
from terraflow.dynamics.abstract import AbstractBlock
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.dynamics_types import (
    State, 
    Tendency, 
    Static, 
    Auxiliary, 
    EquationAuxDataset
)
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.tracer import Tracer
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv.batch import AdvBatchType
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv.interface import adv_prepare
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.space_operators import (
    calc_div,
    operators_prepare_1
)
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.model.common import diagnose


if TYPE_CHECKING:
    import uxarray as uxr
    from terraflow.util import typing
    from terraflow.util.planet import PlanetParameterSet
    from terraflow.model.config import UserConfig
    from terraflow.model.abstract import ModelMode
    from terraflow.mesh.scvt.field import SCVTField3D

    INT    = typing.Int
    FLT    = typing.Float
    Array  = typing.Array
    PyTree = typing.PyTree


class Block(
    AbstractBlock[
        SCVTMesh, 
        AbstractGeneralizedTerrainFollowingCoordinate,
        State, Tendency, Static, Auxiliary, AdvBatchType, Tracer, EquationAuxDataset
    ]
):
    @classmethod
    def from_coord(
        cls,
        mode: ModelMode,
        const: Const,
        cfg: UserConfig,
        para: PlanetParameterSet,
        mesh: SCVTMesh,
        vert_coord: AbstractGeneralizedTerrainFollowingCoordinate,
        dev: AbstractDevice,
        tracers: Optional[Tracer] = None
    ) -> Self:
        # ================================================
        state  = State    .from_coord(mode=mode, cfg=cfg, mesh=mesh, dev=dev)
        static = Static   .from_coord(mode=mode, cfg=cfg, mesh=mesh, dev=dev)
        tend   = Tendency .from_coord(mode=mode, cfg=cfg, mesh=mesh, dev=dev)
        aux    = Auxiliary.from_coord(mode=mode, cfg=cfg, mesh=mesh, dev=dev)
        # ================================================
        # Adv init
        adv_batch_bg = AdvBatchType.from_mesh(
            mode=mode,
            mesh=mesh,
            scheme=cfg.bg_adv_scheme,
            batch_name="bg",
            batch_loc="cell",
            dt=cfg.dt_dyn_seconds,
            dt_dyn=cfg.dt_dyn_seconds,
            dynamic=True,
            passive=False,
            semilag=cfg.bg_adv_scheme=="ffsl",
            dev=dev
        )
        if mode.is_baroclinic():
            adv_batch_pt = AdvBatchType.from_mesh(
                mode=mode,
                mesh=mesh,
                scheme=cfg.pt_adv_scheme, 
                batch_name="pt",
                batch_loc="cell",
                dt=cfg.dt_dyn_seconds,
                dt_dyn=cfg.dt_dyn_seconds,
                dynamic=True,
                passive=True,
                semilag=False,
                dev=dev
            )
        else:
            adv_batch_pt = AdvBatchType()
        if tracers is not None:
            # Parse batch information from tracers_meta
            batch_names: list[str] = []
            batch_idx: dict[str, list[int]] = {}
            for itra, tracer_meta in enumerate(tracers.tracers_meta.values()):
                bname = tracer_meta.batch_name
                if bname in batch_names:
                    batch_idx[bname].append(itra)
                    continue
                batch_names.append(bname)
                batch_idx[bname] = [itra]

            # Initialize adv batches for tracers
            adv_batches: dict[str, AdvBatchType] = {}
            for bname in batch_names:
                adv_batches[bname] = AdvBatchType.from_mesh(
                    mode=mode,
                    mesh=mesh,
                    scheme=cfg.adv_scheme,
                    batch_name=bname,
                    batch_loc="cell",
                    dt=tracers.batch_dts[bname],
                    dt_dyn=cfg.dt_dyn_seconds,
                    dynamic=False,
                    passive=True,
                    semilag=True,
                    idx=batch_idx[bname],
                    dev=dev
                )
        else:
            # if tracers is not provided, initialize an empty adv_batches
            adv_batches = {}
            tracers = Tracer()
        # ================================================
        return cls(
            elapsed_seconds=0.0,
            mode=mode,
            const=const,
            cfg=cfg,
            dt_dyn=cfg.dt_dyn_seconds,
            dt_adv=cfg.dt_adv_seconds,
            old=0,
            new=1,
            dev=dev,
            para=para,
            mesh=mesh,
            vert_coord=vert_coord,
            state=state,
            tend=tend,
            static=static,
            aux=aux,
            adv_batch_bg=adv_batch_bg,
            adv_batch_pt=adv_batch_pt,
            adv_batches=adv_batches,
            tracers=tracers,
            equaux=EquationAuxDataset.empty(),
        )

    def reset(self, pass_tag: str) -> Self:
        if pass_tag == "forward_pass":
            return tree_at(
                lambda b: b.tend.dundt.d,
                self,
                jnp.zeros_like(self.tend.dundt.d)
            )
        else:
           return self
    
    def update_batches_istep(self) -> Self:
        for batch_name, batch in self.adv_batches.items():
            batch = batch.update_istep()
            self = tree_at(
                lambda b: b.adv_batches[batch_name],
                self,
                batch
            )
        return self

    def get_plev(self) -> tuple[SCVTField3D, SCVTField3D]:
        """Get the pressure levels from the static data."""
        return self.state.ph, self.state.ph_lev
    
    def from_parent_after_ic(self) -> "Block":
        # =========================================================
        # Ensure bottom gz_lev is the same as gzs
        if self.mode.is_baroclinic():
            # -----------------------------------------------------
            gz_lev = self.state .gz_lev.unwrap()[0, :, :]
            gzs    = self.static.gzs   .unwrap()
            k        = self.mesh.half_kde
            ids, ide = self.mesh.cell_ids, self.mesh.cell_ide
            # -----------------------------------------------------
            gz_lev   = gz_lev.at[k, ids:ide+1].set(gzs[ids:ide+1])
            gz_lev   = fill_field2d_array_halo(
                halo=self.mesh.halo, 
                f=gz_lev,
                meta=self.state.gz_lev.meta
            )
            self = tree_at(
                where=lambda b: b.state.gz_lev.d,
                pytree=self,
                replace=gz_lev,
                at_dims=0,
                at_idxes=0
            )
        # =========================================================
        # Calculate u and v component at cell
        if not self.mode.is_advection():
            un = self.state.un.unwrap()[0, :, :]
            u  = self.aux  .u .unwrap()
            v  = self.aux  .v .unwrap()
            u, v = wind_c2a_operator(
                R=self.para.radius.unwrap(),
                mesh=self.mesh,
                un=un,
                u=u,
                v=v
            )
            self = tree_at(
                where=lambda b: (b.aux.u.d, b.aux.v.d),
                pytree=self,
                replace=(u, v)
            )
        # =========================================================
        if not self.mode.is_advection(): self = calc_div(self, 0)
        # =========================================================
        if self.mode.is_baroclinic() and not self.mode.restart:
            ref_ps   = self.static.ref_ps.unwrap()
            mgs      = self.state .mgs   .unwrap()[0]
            ims, ime = self.mesh.cell_ims, self.mesh.cell_ime
            ref_ps   = ref_ps.at[ims:ime+1].set(mgs[ims:ime+1])
            self = tree_at(
                where=lambda b: b.static.ref_ps.d,
                pytree=self,
                replace=ref_ps,
            )
        # =========================================================
        equaux = EquationAuxDataset.initialize(
            mode=self.mode,
            cfg=self.cfg,
            para=self.para,
            mesh=self.mesh,
            vert_coord=self.vert_coord,
            ref_ps=self.static.ref_ps.unwrap(),
            dev=self.dev
        )
        self = tree_at(
            where=lambda b: b.equaux,
            pytree=self,
            replace=equaux
        )
        # =========================================================
        self = cast(Block, self)
        if not self.mode.is_advection():
            self = operators_prepare_1(self, self.old, self.dt_dyn)
            self = adv_prepare(self, self.old)
            self = diagnose(self, "old")
        else:
            self = adv_prepare(self, self.old)
        # ========================================================= 
        return self
    
    def trajectory_diag_tools(
        self, 
        trajectory: dict[str, Array],
        use_dual: bool = False
    ) -> UGRIDDataset:
        spatial_meta: dict[str, SCVTFieldMeta] = {}
        non_spatial_names: set[str] = set()
        for fields in (self.state, self.tend, self.static, self.aux):
            for name, field in fields.items():
                if isinstance(field.meta, SCVTFieldMeta):
                    spatial_meta[name] = field.meta
                else:
                    non_spatial_names.add(name)

        q_meta = self.tracers.q.meta
        q_long_name = q_meta.long_name
        for tracer_name, tracer_meta in self.tracers.tracers_meta.items():
            meta_dict = q_meta.to_dict()
            meta_dict.update(
                name=tracer_name,
                long_name=(
                    q_long_name
                    + " of "
                    + tracer_meta.long_name.lower()
                ),
                units=tracer_meta.unit or q_meta.units,
                is_expand_dim0=False,
                expand_dim0_len=0,
                is_tracer=False,
                ntracers=0,
            )
            spatial_meta[tracer_name] = SCVTFieldMeta.from_dict(meta_dict)

        data_vars: dict[str, xr.DataArray] = {}
        for name, value in trajectory.items():
            data = np.asarray(value)
            if name not in spatial_meta:
                if name in non_spatial_names:
                    continue
                raise ValueError(
                    f"Cannot convert trajectory variable '{name}' to UGRID "
                    "because no SCVT field metadata was found."
                )

            meta = spatial_meta[name]
            location = meta.loc
            if location not in [
                "cell", "edge", "vtx", "lev", "lev_vtx", "lev_edge"
            ]:
                raise ValueError(
                    f"hori_location must be 'cell', 'edge' or 'vtx', got {location}."
                )

            if not use_dual:
                mesh_name = "primal"
                if location == "cell" or location == "lev":
                    location = "face"
                elif location == "vtx" or location == "lev_vtx":
                    location = "node"
            else:
                mesh_name = "dual"
                if location == "cell" or location == "lev":
                    location = "node"
                elif location == "vtx" or location == "lev_vtx":
                    location = "face"

            spatial_dims = meta.dims[:-1] + (f"n_{location}",)
            if data.ndim == len(spatial_dims) + 1:
                dims = ("time",) + spatial_dims
            elif data.ndim == len(spatial_dims):
                dims = spatial_dims
            else:
                raise ValueError(
                    f"Trajectory variable '{name}' has shape {data.shape}, "
                    f"which does not match expected dims {spatial_dims} "
                    "with or without a leading time dimension."
                )

            data_vars[name] = xr.DataArray(
                data=data,
                dims=dims,
                attrs={
                    "name": meta.name,
                    "mesh": mesh_name,
                    "location": location,
                    "coordinates": f"{location}_lon {location}_lat",
                    "units": meta.units,
                    "long_name": meta.long_name,
                    "is_static": meta.is_static,
                    "is_tracer": meta.is_tracer,
                    "output": meta.output,
                }
            )

        ds = xr.Dataset(data_vars=data_vars)
        uxgrid = self.mesh.to_ugrid(use_dual=use_dual).to_uxgrid()
        return UGRIDDataset.from_xarray(ds=ds, uxgrid=uxgrid)
    
    def trajectory_post_process(self) -> dict[str, Array]:
        res: dict[str, Array] = {}
        itime = self.get_time_idx("old")
        res.update(self.state.to_dict(dim0_idx=itime))
        res.update(self.aux.u.to_dict(dim0_idx=itime))
        res.update(self.aux.v.to_dict(dim0_idx=itime))
        for batch in self.adv_batches.values():
            tracer_indices = batch.idx
            itime = batch.get_time_idx("old")
            for q_idx, tracer_name in enumerate(self.tracers.tracers_meta.keys()):
                if q_idx in tracer_indices:
                    res[tracer_name] = self.tracers.q.d[itime, q_idx, ...]
        return res
    
    def to_dict(self, itime_str: str) -> dict[str, Array]:
        res: dict[str, Array] = {}
        state_itime = self.get_time_idx(itime_str=itime_str)
        res.update(self.state .to_dict(dim0_idx=state_itime))
        res.update(self.tend  .to_dict())
        res.update(self.static.to_dict())
        res.update(self.aux   .to_dict())
        q = self.tracers.q.d
        tracers_meta = self.tracers.tracers_meta
        for batch in self.adv_batches.values():
            tracer_itime = batch.get_time_idx(itime_str=itime_str)
            tracer_indices = batch.idx
            val = q[tracer_itime, ...]
            for q_idx, tracer_name in enumerate(tracers_meta.keys()):
                if q_idx in tracer_indices:
                    res[tracer_name] = val[q_idx, ...]
        return res
    
    def to_uxarray(
        self,
        use_dual: bool = False,
        itime_str: str = "old",
    ) -> uxr.UxDataset:
        return self.to_ugrid(
            use_dual=use_dual,
            itime_str=itime_str
        ).to_uxarray()
     
    def to_ugrid(
        self,
        use_dual: bool = False,
        itime_str: str = "old",
    ) -> UGRIDDataset:
        state_idx = self.get_time_idx(itime_str=itime_str)
        state_ds = self.state.to_ugrid(
            mesh=self.mesh,
            use_dual=use_dual,
            dim0_idx=state_idx
        )
        tend_ds = self.tend.to_ugrid(
            mesh=self.mesh,
            use_dual=use_dual
        )
        static_ds = self.static.to_ugrid(
            mesh=self.mesh,
            use_dual=use_dual
        )
        aux_ds = self.aux.to_ugrid(
            mesh=self.mesh,
            use_dual=use_dual
        )
        q_vars: dict[str, xr.DataArray] = {}
        q = self.tracers.q.d
        q_meta = self.tracers.q.meta
        tracers_meta = self.tracers.tracers_meta
        for batch in self.adv_batches.values():
            tracer_itime = batch.get_time_idx(itime_str=itime_str)
            tracer_indices = batch.idx
            q_da = meta2ugrid(
                value=q[tracer_itime, ...],
                meta=q_meta,
                use_dual=use_dual
            )
            for q_idx, (tracer_name, tracer_meta) in enumerate(
                tracers_meta.items()
            ):
                if q_idx not in tracer_indices:
                    continue
                tracer_da = q_da.isel(ntracers=q_idx, drop=True)
                tracer_da.attrs = {
                    **q_da.attrs,
                    "name": tracer_name,
                    "units": tracer_meta.unit or q_da.attrs.get("units", ""),
                    "long_name": (
                        q_da.attrs.get("long_name", "")
                        + " of "
                        + tracer_meta.long_name.lower()
                    ),
                }
                q_vars[tracer_name] = tracer_da

        variables = xr.merge(
            [
                state_ds.variables,
                tend_ds.variables,
                static_ds.variables,
                aux_ds.variables,
                xr.Dataset(data_vars=q_vars),
            ],
            compat="no_conflicts"
        )
        return UGRIDDataset.from_xarray(variables, state_ds.uxgrid)
