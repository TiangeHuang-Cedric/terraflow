from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Self, Optional
import equinox as eqx
import datetime

from terraflow.discretization.abstract import AbstractFields
from terraflow.layout.abstract import MeshEntity, VertLoc, Basis
from terraflow.layout.field import ArrayField, StaggeredFieldMetaData, NumericMetaData
from terraflow.discretization.abstract import TracerMetaData

if TYPE_CHECKING:
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
    from terraflow.discretization.abstract import EquationMode

    VM = TypeVar("VM", OneLayerVertCoordinate, HybridVertCoordinate)
    VLO = TypeVar("VLO", OneLayerVertCoordLayout, HybridVertCoordLayout)

    CoordSys = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        VM,
        TriSKBasedCGridStaggeringLayout,
        VLO
    ]

FDT = ArrayField[StaggeredFieldMetaData]
NDT = ArrayField[NumericMetaData]

NLEVP = "ilev"
NLEV = "lev"
NCELL = "cell"
NEDGE = "edge"
NVTX = "vtx"


class Tracer(AbstractFields[StaggeredFieldMetaData]):
    nbatches: int = eqx.field(static=True)
    batch_dts: dict[str, datetime.timedelta] = eqx.field(static=True)
    batch_dts_seconds: dict[str, float] = eqx.field(static=True)
    ntracers: int = eqx.field(static=True)
    tracers_meta: dict[str, TracerMetaData] = eqx.field(static=True)
    idx_qv: int = eqx.field(static=True, init=False, default=-1)
    idx_qc: int = eqx.field(static=True, init=False, default=-1)
    idx_nc: int = eqx.field(static=True, init=False, default=-1)
    idx_qi: int = eqx.field(static=True, init=False, default=-1)
    idx_ni: int = eqx.field(static=True, init=False, default=-1)
    idx_qr: int = eqx.field(static=True, init=False, default=-1)
    idx_nr: int = eqx.field(static=True, init=False, default=-1)
    idx_qs: int = eqx.field(static=True, init=False, default=-1)
    idx_ns: int = eqx.field(static=True, init=False, default=-1)
    idx_qg: int = eqx.field(static=True, init=False, default=-1)
    idx_qh: int = eqx.field(static=True, init=False, default=-1)
    idx_qo3: int = eqx.field(static=True, init=False, default=-1)
    idx_qso2: int = eqx.field(static=True, init=False, default=-1)
    is_water_tracer: list[bool] = eqx.field(
        static=True, init=False, default_factory=list
    )
    ntracers_water: int = eqx.field(static=True, init=False, default=0)
    q: FDT
    mq: FDT
    qm: FDT
    qm_lev: FDT

    def __init__(
        self,
        mode: EquationMode,
        tracers_meta: dict[str, TracerMetaData],
        **fields: ArrayField[StaggeredFieldMetaData] | ArrayField[NumericMetaData]
    ):
        object.__setattr__(
            self,
            "nbatches",
            len(set(meta.batch_name for meta in tracers_meta.values()))
        )
        object.__setattr__(
            self,
            "batch_dts",
            {meta.batch_name: meta.dt for meta in tracers_meta.values()}
        )
        object.__setattr__(
            self,
            "batch_dts_seconds",
            {meta.batch_name: meta.dt_seconds for meta in tracers_meta.values()}
        )
        object.__setattr__(self, "ntracers", len(tracers_meta))
        object.__setattr__(self, "tracers_meta", tracers_meta)
        super().__init__(mode, **fields)
        tracer_catalog(self)

    @classmethod
    def empty(
        cls,
        mode: EquationMode,
    ) -> Self:
        exist_fields: dict[str, FDT] = {}
        for field_name in cls.get_all_fields():
            exist_fields[field_name] = ArrayField.empty()
        return cls(
            mode=mode,
            tracers_meta={},
            **exist_fields
        )

    @classmethod
    def add_tracers(
        cls,
        mode: EquationMode,
        coord: CoordSys,
        tracers_meta: dict[str, TracerMetaData]
    ) -> Self:
        ntracers = len(tracers_meta)
        meta = default_tracer_meta(mode, ntracers)
        if mode.is_baroclinic() or mode.is_advection3d():
            return cls(
                mode=mode,
                tracers_meta=tracers_meta,
                q=meta["q"].zeros_like(coord),
                mq=meta["mq"].zeros_like(coord),
                qm=meta["qm"].zeros_like(coord),
                qm_lev=meta["qm_lev"].zeros_like(coord),
            )
        elif mode.is_advection2d():
            return cls(
                mode=mode,
                tracers_meta=tracers_meta,
                q=meta["q"].zeros_like(coord),
                mq=meta["mq"].zeros_like(coord),
                qm=meta["qm"].zeros_like(coord),
                qm_lev=ArrayField.empty(),
            )
        else:
            raise NotImplementedError(
                f"Unsupported mode {mode} for tracer variables."
            )

    def default_meta(
        self,
        mode: Optional[EquationMode] = None,
        ntracers: Optional[int] = None
    ) -> dict[str, StaggeredFieldMetaData | NumericMetaData]:
        if ntracers is None:
            ntracers = self.ntracers
        if mode is None:
            mode = self.mode
        if ntracers is None and mode is None:
            raise ValueError(
                "At least one of mode and ntracers must be provided."
            )
        return default_tracer_meta(mode, ntracers)


def default_tracer_meta(
    mode: EquationMode,
    ntracers: int
) -> dict[str, StaggeredFieldMetaData | NumericMetaData]:
    meta: dict[str, StaggeredFieldMetaData | NumericMetaData] = {}
    if mode.is_baroclinic() or mode.is_advection3d():
        meta["q"] = StaggeredFieldMetaData(
            name="q",
            long_name="Tracer dry mixing ratio",
            description="Tracer dry mixing ratio",
            dims=(NLEV, NCELL),
            units="kg/kg",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_tracer=True,
            ntracers=ntracers,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0",
        )
        meta["mq"] = StaggeredFieldMetaData(
            name="mq",
            long_name="Tracer mass dry mixing ratio",
            description="Tracer mass dry mixing ratio",
            dims=(NLEV, NCELL),
            units="Pa kg/kg",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_tracer=True,
            ntracers=ntracers,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="",
        )
        meta["qm"] = StaggeredFieldMetaData(
            name="qm",
            long_name="Total moist tracer dry mixing ratio",
            description="Total moist tracer dry mixing ratio",
            dims=(NLEV, NCELL),
            units="kg/kg",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            output="",
        )
        meta["qm_lev"] = StaggeredFieldMetaData(
            name="qm_lev",
            long_name="Total moist tracer dry mixing ratio on half level",
            description="Total moist tracer dry mixing ratio on half level",
            dims=(NLEVP, NCELL),
            units="kg/kg",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Lev,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            output="",
        )
    if mode.is_advection2d():
        meta["q"] = StaggeredFieldMetaData(
            name="q",
            long_name="Tracer dry mixing ratio",
            description="Tracer dry mixing ratio",
            dims=(NCELL,),
            units="kg/kg",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_tracer=True,
            ntracers=ntracers,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0",
        )
        meta["mq"] = StaggeredFieldMetaData(
            name="mq",
            long_name="Tracer mass dry mixing ratio",
            description="Tracer mass dry mixing ratio",
            dims=(NCELL,),
            units="Pa kg/kg",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_tracer=True,
            ntracers=ntracers,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="",
        )
        meta["qm"] = StaggeredFieldMetaData(
            name="qm",
            long_name="Total moist tracer dry mixing ratio",
            description="Total moist tracer dry mixing ratio",
            dims=(NCELL,),
            units="kg/kg",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            output="",
        )
    return meta


def tracer_catalog(tracer: Tracer) -> Tracer:
    tracer_names = list(tracer.tracers_meta.keys())
    is_water_tracer = [False] * len(tracer_names)
    ntracers_water = 0
    for idx, tracer_name in enumerate(tracer_names):
        if tracer_name == "qv" or tracer_name == "Q":
            object.__setattr__(tracer, "idx_qv", idx)
            is_water_tracer[idx] = True
            ntracers_water += 1
        elif tracer_name == "qc" or tracer_name == "CLDLIQ":
            object.__setattr__(tracer, "idx_qc", idx)
            is_water_tracer[idx] = True
            ntracers_water += 1
        elif tracer_name == "nc" or tracer_name == "NUMLIQ":
            object.__setattr__(tracer, "idx_nc", idx)
        elif tracer_name == "qi" or tracer_name == "CLDICE":
            object.__setattr__(tracer, "idx_qi", idx)
            is_water_tracer[idx] = True
            ntracers_water += 1
        elif tracer_name == "ni" or tracer_name == "NUMICE":
            object.__setattr__(tracer, "idx_ni", idx)
        elif tracer_name == "qr" or tracer_name == "RAINQM":
            object.__setattr__(tracer, "idx_qr", idx)
            is_water_tracer[idx] = True
            ntracers_water += 1
        elif tracer_name == "qs" or tracer_name == "SNOWQM":
            object.__setattr__(tracer, "idx_qs", idx)
            is_water_tracer[idx] = True
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
    object.__setattr__(tracer, "ntracers_water", ntracers_water)
    return tracer
