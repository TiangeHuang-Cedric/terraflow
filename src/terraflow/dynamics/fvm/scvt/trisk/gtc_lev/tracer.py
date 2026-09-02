from __future__ import annotations
from typing import TYPE_CHECKING, Self, Optional, Dict
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import typing
from terraflow.util.device import AbstractDevice
from terraflow.util.pytree import tree_at
from terraflow.model.abstract import ModelMode
from terraflow.mesh.scvt.field import (
    SCVTFieldLike, 
    SCVTField4D, 
    SCVTField2D, 
    SCVTFields
)
from terraflow.mesh.scvt.parallel import fill_field2d_array_halo


if TYPE_CHECKING:
    from terraflow.util.planet import PlanetParameterSet
    from terraflow.model.config import UserConfig
    from terraflow.prepare.readers.abstract import AbstractBackgroundReader
    from terraflow.prepare.readers.grist import GRISTBackgroundReader
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block

    BKGREADER = AbstractBackgroundReader | GRISTBackgroundReader


Int = typing.Int
Array = typing.Array


NLEVP = "ilev"
NLEV = "lev"
NCELL = "cell"
NEDGE = "edge"
NVTX = "vtx"


class TracerMeta:
    batch_name: str   = eqx.field(static=True, default="")
    dt        : float = eqx.field(static=True, default=0 )
    name      : str   = eqx.field(static=True, default="")
    long_name : str   = eqx.field(static=True, default="")
    unit      : str   = eqx.field(static=True, default="")

    def __init__(
        self,
        batch_name: str,
        dt: float,
        name: str,
        long_name: str,
        unit: Optional[str] = None,
    ):
        object.__setattr__(self, "batch_name", batch_name)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "long_name", long_name)
        if unit is None:
            unit = ""
        object.__setattr__(self, "unit", unit)

    @classmethod
    def from_reader(
        cls,
        mode: ModelMode,
        para: PlanetParameterSet,
        reader: BKGREADER,
        dt_adv: float
    ) -> Dict[str, Self]:
        if mode.ideal_dry_core:
            return {}

        if para.planet == "earth":
            if reader.reader_type == "era5":
                meta: Dict[str, Self] = {}
                meta["qv"] = cls(
                    batch_name="moist",
                    dt=dt_adv,
                    name="qv",
                    long_name="Water vapor",
                    unit="kg kg-1"
                )
                meta["qc"] = cls(
                    batch_name="moist",
                    dt=dt_adv,
                    name="qc",
                    long_name="Cloud water",
                    unit="kg kg-1"
                )
                meta["qi"] = cls(
                    batch_name="moist",
                    dt=dt_adv,
                    name="qi",
                    long_name="Cloud ice",
                    unit="kg kg-1"
                )
                meta["qr"] = cls(
                    batch_name="moist",
                    dt=dt_adv,
                    name="qr",
                    long_name="Rain water",
                    unit="kg kg-1"
                )
                meta["qs"] = cls(
                    batch_name="moist",
                    dt=dt_adv,
                    name="qs",
                    long_name="Snow",
                    unit="kg kg-1"
                )
                return meta
            elif reader.reader_type == "grist":
                meta: Dict[str, Self] = {}
                meta["qv"] = cls(
                    batch_name="moist",
                    dt=dt_adv,
                    name="qv",
                    long_name="Water vapor",
                    unit="kg kg-1"
                )
                return meta
            else:
                raise NotImplementedError(
                    f"Reader type '{reader.reader_type}' not supported for TracerMetaData extraction."
                )
        else:
            raise NotImplementedError(
                f"Planet '{para.planet}' not supported for TracerMetaData extraction."
            )


class Tracer(SCVTFields):
    mode              : ModelMode             = eqx.field(static=True, default_factory=lambda: ModelMode())
    nbatches          : int                   = eqx.field(static=True, default=0                )
    batch_dts         : dict[str, float]      = eqx.field(static=True, default_factory=dict     )
    ntracers          : int                   = eqx.field(static=True, default=0                )
    tracers_meta      : dict[str, TracerMeta] = eqx.field(static=True, default_factory=dict     )
    idx_qv            : int                   = eqx.field(static=True, default=-1,  init=False)
    idx_qc            : int                   = eqx.field(static=True, default=-1,  init=False)
    idx_nc            : int                   = eqx.field(static=True, default=-1,  init=False)
    idx_qi            : int                   = eqx.field(static=True, default=-1,  init=False)
    idx_ni            : int                   = eqx.field(static=True, default=-1,  init=False)
    idx_qr            : int                   = eqx.field(static=True, default=-1,  init=False)
    idx_nr            : int                   = eqx.field(static=True, default=-1,  init=False)
    idx_qs            : int                   = eqx.field(static=True, default=-1,  init=False)
    idx_ns            : int                   = eqx.field(static=True, default=-1,  init=False)
    idx_qg            : int                   = eqx.field(static=True, default=-1,  init=False)
    idx_qh            : int                   = eqx.field(static=True, default=-1,  init=False)
    idx_qo3           : int                   = eqx.field(static=True, default=-1,  init=False)
    idx_qso2          : int                   = eqx.field(static=True, default=-1,  init=False)
    is_water_tracer   : list[bool]            = eqx.field(static=True, default_factory=list, init=False)
    idxs_water_tracers: list[int]             = eqx.field(static=True, default_factory=list)
    ntracers_water    : int                   = eqx.field(static=True, default=0 ,  init=False)
    q                 : SCVTField4D           = eqx.field(default_factory=lambda: SCVTField4D())
    qm                : SCVTField2D           = eqx.field(default_factory=lambda: SCVTField2D())
    qm_lev            : SCVTField2D           = eqx.field(default_factory=lambda: SCVTField2D())

    @classmethod
    def initialize(
        cls,
        mode: ModelMode,
        cfg: UserConfig,
        mesh: SCVTMesh,
        tracers_meta: dict[str, TracerMeta],
        dev: AbstractDevice,
    ) -> Self:
        if cfg.adv_time_scheme == "wrfrk3":
            ntracer_tile = 3
        elif cfg.adv_time_scheme == "rk2":
            ntracer_tile = 2
        else:
            raise ValueError(f"Unsupported adv_time_scheme {cfg.adv_time_scheme}!")
        ntracers = len(tracers_meta)
        fields: dict[str, SCVTFieldLike] = {}
        fields["q"] = SCVTField4D.from_mesh(
            name="q",
            long_name="Tracer dry mixing ratio",
            units="kg/kg",
            loc="cell",
            mesh=mesh,
            dev=dev,
            ntracers=ntracers,
            expand_dim0_len=ntracer_tile,
            output="h0",
        )
        fields["qm"] = SCVTField2D.from_mesh(
            name="qm",
            long_name="Total moist tracer dry mixing ratio",
            units="kg/kg",
            loc="cell",
            mesh=mesh,
            dev=dev,
            output="",
        )    
        if mode.is_nonhydrostatic():
            fields["qm_lev"] = SCVTField2D.from_mesh(
                name="qm_lev",
                long_name="Total moist tracer dry mixing ratio on half level",
                units="kg/kg",
                loc="lev",
                mesh=mesh,
                dev=dev,
                output="",
            )
        self = cls.from_fields(
            fields=fields,
            mode=mode,
            nbatches=len(set(meta.batch_name for meta in tracers_meta.values())),
            batch_dts={meta.batch_name: meta.dt for meta in tracers_meta.values()},
            ntracers=len(tracers_meta),
            tracers_meta=tracers_meta,
        )
        tracer_catalog(self)
        return self

    @classmethod
    def add_tracers(
        cls,
        mode: ModelMode,
        cfg : UserConfig,
        mesh: SCVTMesh,
        tracers_meta: dict[str, TracerMeta],
        dev: AbstractDevice,
    ) -> Self:
        return cls.initialize(mode=mode, cfg=cfg, mesh=mesh, tracers_meta=tracers_meta, dev=dev)


def tracer_catalog(tracer: Tracer) -> Tracer:
    tracer_names = list(tracer.tracers_meta.keys())
    is_water_tracer = [False] * len(tracer_names)
    idxs_water_tracers = []
    ntracers_water = 0
    for idx, tracer_name in enumerate(tracer_names):
        if tracer_name == "qv" or tracer_name == "Q":
            object.__setattr__(tracer, "idx_qv", idx)
            is_water_tracer[idx] = True
            idxs_water_tracers.append(idx)
            ntracers_water += 1
        elif tracer_name == "qc" or tracer_name == "CLDLIQ":
            object.__setattr__(tracer, "idx_qc", idx)
            is_water_tracer[idx] = True
            idxs_water_tracers.append(idx)
            ntracers_water += 1
        elif tracer_name == "nc" or tracer_name == "NUMLIQ":
            object.__setattr__(tracer, "idx_nc", idx)
        elif tracer_name == "qi" or tracer_name == "CLDICE":
            object.__setattr__(tracer, "idx_qi", idx)
            is_water_tracer[idx] = True
            idxs_water_tracers.append(idx)
            ntracers_water += 1
        elif tracer_name == "ni" or tracer_name == "NUMICE":
            object.__setattr__(tracer, "idx_ni", idx)
        elif tracer_name == "qr" or tracer_name == "RAINQM":
            object.__setattr__(tracer, "idx_qr", idx)
            is_water_tracer[idx] = True
            idxs_water_tracers.append(idx)
            ntracers_water += 1
        elif tracer_name == "qs" or tracer_name == "SNOWQM":
            object.__setattr__(tracer, "idx_qs", idx)
            is_water_tracer[idx] = True
            idxs_water_tracers.append(idx)
            ntracers_water += 1
        elif tracer_name == "qg":
            object.__setattr__(tracer, "idx_qg", idx)
            ntracers_water += 1
        elif tracer_name == "qh":
            object.__setattr__(tracer, "idx_qh", idx)
            ntracers_water += 1
        elif tracer_name == "qo3":
            object.__setattr__(tracer, "idx_qo3", idx)
        elif tracer_name == "qso2" or tracer_name == "SO2":
            object.__setattr__(tracer, "idx_qso2", idx)
    object.__setattr__(tracer, "is_water_tracer", is_water_tracer)
    object.__setattr__(tracer, "idxs_water_tracers", idxs_water_tracers)
    object.__setattr__(tracer, "ntracers_water", ntracers_water)
    return tracer


def tracer_calc_qm(block: Block, itime_str) -> Block:
    if block.tracers.ntracers_water > 0:
        mesh = block.mesh
        ids, ide = mesh.cell_ids, mesh.cell_ide
        itime = block.adv_batches["moist"].get_time_idx(itime_str=itime_str)
        q = block.tracers.q.unwrap()[itime, :, :, :]
        idxs_water_tracers = block.tracers.idxs_water_tracers
        qm = jnp.zeros_like(block.tracers.qm.unwrap())
        qm = qm.at[:, ids:ide+1].set(
            jnp.sum(q[idxs_water_tracers, :, ids:ide+1], axis=0)
        )
        qm = fill_field2d_array_halo(
            halo=mesh.halo,
            f=qm,
            meta=block.tracers.qm.meta,
        )
        return tree_at(
            where=lambda b: b.tracers.qm.d,
            pytree=block,
            replace=qm,
        )
    else:
        return block
