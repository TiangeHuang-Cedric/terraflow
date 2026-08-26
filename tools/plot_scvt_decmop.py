#!/usr/bin/env python3

from __future__ import annotations
import argparse
import colorsys
from dataclasses import dataclass
import glob
import math
import re
from pathlib import Path
from typing import Literal, Protocol, Sequence, cast
import cartopy.crs as ccrs 
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
DomainName = Literal["inner", "bdry", "cmpt", "halo", "full", "memory"]
MeshEntity = Literal["cell", "edge", "vtx"]
OverlayName = Literal["none", "inner", "bdry", "cmpt", "halo", "full", "memory"]
SiteKind = Literal["send", "recv"]
SiteMode = Literal["none", "send", "recv", "sendrecv"]
Point = tuple[float, float]
Segment = tuple[Point, Point]

DOMAIN_CHOICES: tuple[str, ...] = (
    "all",
    "inner",
    "bdry",
    "boundary",
    "cmpt",
    "compute",
    "halo",
    "full",
    "memory",
)
DOMAIN_ALIASES: dict[str, DomainName] = {
    "inner": "inner",
    "bdry": "bdry",
    "boundary": "bdry",
    "cmpt": "cmpt",
    "compute": "cmpt",
    "halo": "halo",
    "full": "full",
    "memory": "memory",
}
DOMAIN_ORDER: tuple[DomainName, ...] = ("inner", "bdry", "halo")
DOMAIN_COLORS: dict[DomainName, str] = {
    "inner": "#000000",
    "bdry": "#E7298A",
    "cmpt": "#0072B2",
    "halo": "#FF7F00",
    "full": "#CC79A7",
    "memory": "#E69F00",
}
DOMAIN_VTX_MARKERS: dict[DomainName, str] = {
    "inner": "^",
    "bdry": "D",
    "cmpt": "s",
    "halo": "P",
    "full": "X",
    "memory": "v",
}
SITE_MARKERS: dict[tuple[SiteKind, MeshEntity], str] = {
    ("send", "cell"): "o",
    ("send", "vtx"): "^",
    ("send", "edge"): "X",
    ("recv", "cell"): "s",
    ("recv", "vtx"): "D",
    ("recv", "edge"): "P",
}
SITE_MARKER_SIZES: dict[MeshEntity, float] = {
    "cell": 16.0,
    "vtx": 22.0,
    "edge": 18.0,
}
CELL_ORDER_COLOR = "#D62728"


@dataclass(frozen=True)
class DomainSelection:
    name: DomainName
    level: int | None = None

    @property
    def label(self) -> str:
        if self.level is None:
            return self.name
        return f"{self.name}[{self.level}]"


class PlotArgs(Protocol):
    input: list[str]
    projection: str
    output: str | None
    domain: list[str]
    overlay: OverlayName
    domain_view: int | None
    overlay_view: int | None
    with_send: bool
    with_recv: bool
    with_sendrecv: bool
    central_lon: float | None
    central_lat: float | None
    auto_center: bool
    show_halo: bool
    halo_alpha: float
    outline_partitions: bool
    overlay_dual: bool
    edges: bool
    edge_color: str
    outline_color: str
    linewidth: float
    outline_width: float
    alpha: float
    overlay_alpha: float
    cmap: str
    coastlines: bool
    gridlines: bool
    legend: bool
    centers: bool
    draw_cell_order: bool
    center_size: float
    title: str | None
    width: float
    height: float
    dpi: int


class GeoAxesLike(Protocol):
    def set_global(self) -> None: ...

    def coastlines(self, *, linewidth: float, color: str) -> object: ...

    def gridlines(
        self, *, crs: ccrs.Projection, color: str, alpha: float, linestyle: str
    ) -> object: ...

    def add_collection(self, collection: object) -> object: ...

    def scatter(
        self,
        x: FloatArray,
        y: FloatArray,
        *,
        s: float,
        color: str,
        marker: str,
        linewidths: float,
        edgecolors: str,
        transform: ccrs.Projection,
        zorder: int,
    ) -> object: ...

    def legend(
        self,
        *,
        handles: Sequence[Patch | Line2D],
        loc: str,
        fontsize: int,
        frameon: bool,
        framealpha: float,
    ) -> object: ...

    def set_title(self, label: str) -> object: ...


def as_float_array(values: object) -> FloatArray:
    if np.ma.isMaskedArray(values):
        values = np.ma.filled(values, 0)
    return np.asarray(values, dtype=np.float64)


def as_int_array(values: object) -> IntArray:
    if np.ma.isMaskedArray(values):
        values = np.ma.filled(values, 0)
    return np.asarray(values, dtype=np.int64)


def optional_int_array(ds: xr.Dataset, name: str) -> IntArray | None:
    if name not in ds.variables:
        return None
    return as_int_array(ds[name].values)


def scalar_int(values: IntArray) -> int:
    return int(np.ravel(values)[0])


def lon_to_degrees(lon_rad: FloatArray) -> FloatArray:
    lon = np.degrees(lon_rad)
    return ((lon + 180.0) % 360.0) - 180.0


def unwrap_polygon_lons(lons: FloatArray | Sequence[float]) -> FloatArray:
    lon_array = np.asarray(lons, dtype=np.float64).copy()
    if lon_array.size > 1 and np.nanmax(lon_array) - np.nanmin(lon_array) > 180.0:
        lon_array[lon_array < 0.0] += 360.0
    return lon_array


def projection_from_name(
    name: str, central_lon: float, central_lat: float
) -> ccrs.Projection:
    key = name.lower().replace("_", "-")
    if key in ("platecarree", "plate-carree", "latlon", "lonlat"):
        return ccrs.PlateCarree(central_longitude=central_lon)
    if key == "robinson":
        return ccrs.Robinson(central_longitude=int(central_lon))
    if key == "orthographic":
        return ccrs.Orthographic(
            central_longitude=central_lon, central_latitude=central_lat
        )
    if key == "mollweide":
        return ccrs.Mollweide(central_longitude=int(central_lon))
    if key in ("equalearth", "equal-earth"):
        return ccrs.EqualEarth(central_longitude=int(central_lon))
    if key in ("northpolar", "north-polar"):
        return ccrs.NorthPolarStereo(central_longitude=central_lon)
    if key in ("southpolar", "south-polar"):
        return ccrs.SouthPolarStereo(central_longitude=central_lon)
    if key == "stereographic":
        return ccrs.Stereographic(
            central_longitude=central_lon, central_latitude=central_lat
        )
    raise ValueError(f"Unsupported projection: {name}")


def parse_domain_args(domain_args: Sequence[str]) -> list[DomainSelection]:
    selected: list[DomainSelection] = []
    seen: set[tuple[DomainName, int | None]] = set()
    for item in domain_args:
        for raw_name in item.split(","):
            name = raw_name.strip().lower()
            if not name:
                continue
            if name == "all":
                for domain in DOMAIN_ORDER:
                    key = (domain, None)
                    if key not in seen:
                        selected.append(DomainSelection(domain))
                        seen.add(key)
                continue

            match = re.fullmatch(r"([a-z]+)(?:\[(\d+)\]|:(\d+))?", name)
            if match is None:
                raise ValueError(f"Unsupported domain: {raw_name}")
            domain_key, bracket_level, colon_level = match.groups()
            if domain_key not in DOMAIN_CHOICES:
                raise ValueError(f"Unsupported domain: {raw_name}")
            if domain_key == "all":
                raise ValueError(f"Layer-qualified 'all' is not supported: {raw_name}")

            domain_name = DOMAIN_ALIASES[domain_key]
            raw_level = bracket_level if bracket_level is not None else colon_level
            level = int(raw_level) if raw_level is not None else None
            if level is not None and level < 0:
                raise ValueError(
                    f"Layer-qualified domains use 0-based levels: {raw_name}"
                )
            if level is not None and domain_name not in ("bdry", "halo"):
                raise ValueError(
                    f"Layer-qualified domains are only supported for bdry/halo: {raw_name}"
                )
            key = (domain_name, level)
            if key not in seen:
                selected.append(DomainSelection(domain_name, level))
                seen.add(key)

    if not selected:
        raise ValueError("At least one domain must be selected")
    return selected


def parse_overlay_arg(overlay_arg: str) -> OverlayName:
    name = overlay_arg.strip().lower()
    if name == "none":
        return "none"
    if name not in DOMAIN_ALIASES:
        raise argparse.ArgumentTypeError(f"Unsupported overlay domain: {overlay_arg}")
    return DOMAIN_ALIASES[name]


def site_mode_from_args(args: PlotArgs) -> SiteMode:
    if args.with_send:
        return "send"
    if args.with_recv:
        return "recv"
    if args.with_sendrecv:
        return "sendrecv"
    return "none"


def site_kinds_for_mode(site_mode: SiteMode) -> tuple[SiteKind, ...]:
    if site_mode == "send":
        return ("send",)
    if site_mode == "recv":
        return ("recv",)
    if site_mode == "sendrecv":
        return ("send", "recv")
    return ()


def site_color(site_kind: SiteKind, neighbor_idx: int) -> str:
    cmap = plt.get_cmap("tab20")
    kind_offset = 0 if site_kind == "send" else 1
    palette_idx = neighbor_idx * 2 + kind_offset
    if palette_idx < 20:
        return to_hex(cmap(palette_idx))

    hue = (neighbor_idx * 0.618033988749895) % 1.0
    saturation = 0.85 if site_kind == "send" else 0.58
    value = 0.88 if site_kind == "send" else 0.68
    return to_hex(colorsys.hsv_to_rgb(hue, saturation, value))


@dataclass(frozen=True)
class SCVTRawData:
    cell_ins: IntArray
    cell_ine: IntArray
    cell_ibs: IntArray
    cell_ibe: IntArray
    cell_ids: IntArray
    cell_ide: IntArray
    cell_ihs: IntArray
    cell_ihe: IntArray
    cell_ims: IntArray
    cell_ime: IntArray
    vtx_ins: IntArray
    vtx_ine: IntArray
    vtx_ibs: IntArray
    vtx_ibe: IntArray
    vtx_ids: IntArray
    vtx_ide: IntArray
    vtx_ihs: IntArray
    vtx_ihe: IntArray
    vtx_ims: IntArray
    vtx_ime: IntArray
    edge_ins: IntArray
    edge_ine: IntArray
    edge_ibs: IntArray
    edge_ibe: IntArray
    edge_ids: IntArray
    edge_ide: IntArray
    edge_ihs: IntArray
    edge_ihe: IntArray
    edge_ims: IntArray
    edge_ime: IntArray
    cell_lon: FloatArray
    cell_lat: FloatArray
    cell_xyz: FloatArray
    cell_nnb: IntArray
    vtx_lon: FloatArray
    vtx_lat: FloatArray
    edge_lon: FloatArray
    edge_lat: FloatArray
    vtx_of_cell: IntArray
    edge_vtx: IntArray
    dual_edge_vtx: IntArray
    area_cell: FloatArray
    cell_of_vtx: IntArray | None = None
    send_ncell: IntArray | None = None
    send_nvtx: IntArray | None = None
    send_nedge: IntArray | None = None
    recv_ncell: IntArray | None = None
    recv_nvtx: IntArray | None = None
    recv_nedge: IntArray | None = None
    send_cell: IntArray | None = None
    send_vtx: IntArray | None = None
    send_edge: IntArray | None = None
    recv_cell: IntArray | None = None
    recv_vtx: IntArray | None = None
    recv_edge: IntArray | None = None

    @classmethod
    def from_file(cls, file_path: Path) -> SCVTRawData:
        with xr.open_dataset(file_path) as ds:
            return cls(
                cell_ins=as_int_array(ds["cell_ins"].values),
                cell_ine=as_int_array(ds["cell_ine"].values),
                cell_ibs=as_int_array(ds["cell_ibs"].values),
                cell_ibe=as_int_array(ds["cell_ibe"].values),
                cell_ids=as_int_array(ds["cell_ids"].values),
                cell_ide=as_int_array(ds["cell_ide"].values),
                cell_ihs=as_int_array(ds["cell_ihs"].values),
                cell_ihe=as_int_array(ds["cell_ihe"].values),
                cell_ims=as_int_array(ds["cell_ims"].values),
                cell_ime=as_int_array(ds["cell_ime"].values),
                vtx_ins=as_int_array(ds["vtx_ins"].values),
                vtx_ine=as_int_array(ds["vtx_ine"].values),
                vtx_ibs=as_int_array(ds["vtx_ibs"].values),
                vtx_ibe=as_int_array(ds["vtx_ibe"].values),
                vtx_ids=as_int_array(ds["vtx_ids"].values),
                vtx_ide=as_int_array(ds["vtx_ide"].values),
                vtx_ihs=as_int_array(ds["vtx_ihs"].values),
                vtx_ihe=as_int_array(ds["vtx_ihe"].values),
                vtx_ims=as_int_array(ds["vtx_ims"].values),
                vtx_ime=as_int_array(ds["vtx_ime"].values),
                edge_ins=as_int_array(ds["edge_ins"].values),
                edge_ine=as_int_array(ds["edge_ine"].values),
                edge_ibs=as_int_array(ds["edge_ibs"].values),
                edge_ibe=as_int_array(ds["edge_ibe"].values),
                edge_ids=as_int_array(ds["edge_ids"].values),
                edge_ide=as_int_array(ds["edge_ide"].values),
                edge_ihs=as_int_array(ds["edge_ihs"].values),
                edge_ihe=as_int_array(ds["edge_ihe"].values),
                edge_ims=as_int_array(ds["edge_ims"].values),
                edge_ime=as_int_array(ds["edge_ime"].values),
                cell_lon=as_float_array(ds["cell_lon"].values),
                cell_lat=as_float_array(ds["cell_lat"].values),
                cell_xyz=as_float_array(ds["cell_xyz"].values),
                cell_nnb=as_int_array(ds["cell_nnb"].values),
                vtx_lon=as_float_array(ds["vtx_lon"].values),
                vtx_lat=as_float_array(ds["vtx_lat"].values),
                edge_lon=as_float_array(ds["edge_lon"].values),
                edge_lat=as_float_array(ds["edge_lat"].values),
                vtx_of_cell=as_int_array(ds["vtx_of_cell"].values),
                edge_vtx=as_int_array(ds["edge_vtx"].values),
                dual_edge_vtx=as_int_array(ds["dual_edge_vtx"].values),
                area_cell=as_float_array(ds["area_cell"].values),
                cell_of_vtx=optional_int_array(ds, "cell_of_vtx"),
                send_ncell=optional_int_array(ds, "send_ncell"),
                send_nvtx=optional_int_array(ds, "send_nvtx"),
                send_nedge=optional_int_array(ds, "send_nedge"),
                recv_ncell=optional_int_array(ds, "recv_ncell"),
                recv_nvtx=optional_int_array(ds, "recv_nvtx"),
                recv_nedge=optional_int_array(ds, "recv_nedge"),
                send_cell=optional_int_array(ds, "send_cell"),
                send_vtx=optional_int_array(ds, "send_vtx"),
                send_edge=optional_int_array(ds, "send_edge"),
                recv_cell=optional_int_array(ds, "recv_cell"),
                recv_vtx=optional_int_array(ds, "recv_vtx"),
                recv_edge=optional_int_array(ds, "recv_edge"),
            )


class SCVTMesh:
    def __init__(self, file_path: Path, rank: int) -> None:
        raw = SCVTRawData.from_file(file_path)

        self.file_path = file_path
        self.rank = rank
        self.cell_ins = raw.cell_ins
        self.cell_ine = raw.cell_ine
        self.cell_ibs = raw.cell_ibs
        self.cell_ibe = raw.cell_ibe
        self.cell_ids = raw.cell_ids
        self.cell_ide = raw.cell_ide
        self.cell_ihs = raw.cell_ihs
        self.cell_ihe = raw.cell_ihe
        self.cell_ims = raw.cell_ims
        self.cell_ime = raw.cell_ime
        self.vtx_ins = raw.vtx_ins
        self.vtx_ine = raw.vtx_ine
        self.vtx_ibs = raw.vtx_ibs
        self.vtx_ibe = raw.vtx_ibe
        self.vtx_ids = raw.vtx_ids
        self.vtx_ide = raw.vtx_ide
        self.vtx_ihs = raw.vtx_ihs
        self.vtx_ihe = raw.vtx_ihe
        self.vtx_ims = raw.vtx_ims
        self.vtx_ime = raw.vtx_ime
        self.edge_ins = raw.edge_ins
        self.edge_ine = raw.edge_ine
        self.edge_ibs = raw.edge_ibs
        self.edge_ibe = raw.edge_ibe
        self.edge_ids = raw.edge_ids
        self.edge_ide = raw.edge_ide
        self.edge_ihs = raw.edge_ihs
        self.edge_ihe = raw.edge_ihe
        self.edge_ims = raw.edge_ims
        self.edge_ime = raw.edge_ime
        self.cell_lon = raw.cell_lon
        self.cell_lat = raw.cell_lat
        self.cell_nnb = raw.cell_nnb
        self.vtx_lon = raw.vtx_lon
        self.vtx_lat = raw.vtx_lat
        self.edge_lon = raw.edge_lon
        self.edge_lat = raw.edge_lat
        self.area_cell = raw.area_cell
        self.send_ncell = raw.send_ncell
        self.send_nvtx = raw.send_nvtx
        self.send_nedge = raw.send_nedge
        self.recv_ncell = raw.recv_ncell
        self.recv_nvtx = raw.recv_nvtx
        self.recv_nedge = raw.recv_nedge
        self.send_cell = raw.send_cell
        self.send_vtx = raw.send_vtx
        self.send_edge = raw.send_edge
        self.recv_cell = raw.recv_cell
        self.recv_vtx = raw.recv_vtx
        self.recv_edge = raw.recv_edge

        self.n_cell = self.cell_lon.size
        self.n_vtx = self.vtx_lon.size
        self.n_edge = self.edge_lon.size
        self.cell_xyz = self._normalize_xyz(raw.cell_xyz, self.n_cell)
        self.vtx_of_cell = self._normalize_table(raw.vtx_of_cell, self.n_cell)
        self.cell_of_vtx = (
            self._normalize_table(raw.cell_of_vtx, self.n_vtx)
            if raw.cell_of_vtx is not None
            else None
        )
        self.edge_vtx = self._normalize_edge_table(raw.edge_vtx)
        self.dual_edge_vtx = self._normalize_edge_table(raw.dual_edge_vtx)

        self.cell_lon_deg = lon_to_degrees(self.cell_lon)
        self.cell_lat_deg = np.degrees(self.cell_lat)
        self.vtx_lon_deg = lon_to_degrees(self.vtx_lon)
        self.vtx_lat_deg = np.degrees(self.vtx_lat)
        self.edge_lon_deg = lon_to_degrees(self.edge_lon)
        self.edge_lat_deg = np.degrees(self.edge_lat)

    @staticmethod
    def rank_from_path(file_path: str | Path, fallback: int) -> int:
        match = re.search(r"\.(\d+)\.nc$", Path(file_path).name)
        return int(match.group(1)) if match else fallback

    @staticmethod
    def _normalize_table(values: IntArray, first_dim: int) -> IntArray:
        if values.ndim != 2:
            raise ValueError(f"Expected a 2-D topology table, got shape {values.shape}")
        if values.shape[0] == first_dim:
            return values
        if values.shape[1] == first_dim:
            return values.T
        raise ValueError(f"Cannot align topology table shape {values.shape} to {first_dim}")

    @staticmethod
    def _normalize_edge_table(values: IntArray) -> IntArray:
        if values.ndim != 2:
            raise ValueError(f"Expected a 2-D edge table, got shape {values.shape}")
        if values.shape[1] == 2:
            return values
        if values.shape[0] == 2:
            return values.T
        raise ValueError(f"Cannot align edge table shape {values.shape} to (*, 2)")

    @staticmethod
    def _normalize_site_table(values: IntArray, n_neighbor: int) -> IntArray:
        if values.ndim != 2:
            raise ValueError(f"Expected a 2-D send/recv site table, got shape {values.shape}")
        if values.shape[0] == n_neighbor:
            return values
        if values.shape[1] == n_neighbor:
            return values.T
        raise ValueError(
            f"Cannot align send/recv site table shape {values.shape} to {n_neighbor} neighbors"
        )

    @staticmethod
    def _normalize_xyz(values: FloatArray, first_dim: int) -> FloatArray:
        if values.ndim != 2:
            raise ValueError(f"Expected a 2-D coordinate table, got shape {values.shape}")
        if values.shape == (first_dim, 3):
            return values
        if values.shape == (3, first_dim):
            return values.T
        raise ValueError(f"Cannot align xyz table shape {values.shape} to {first_dim}")

    def index_range(
        self, domain: DomainName, entity: MeshEntity = "cell", level: int | None = None
    ) -> range:
        if entity == "cell":
            start, end, size = self._index_bounds(
                domain,
                level,
                self.cell_ins,
                self.cell_ine,
                self.cell_ibs,
                self.cell_ibe,
                self.cell_ids,
                self.cell_ide,
                self.cell_ihs,
                self.cell_ihe,
                self.cell_ims,
                self.cell_ime,
                self.n_cell,
            )
        elif entity == "vtx":
            start, end, size = self._index_bounds(
                domain,
                level,
                self.vtx_ins,
                self.vtx_ine,
                self.vtx_ibs,
                self.vtx_ibe,
                self.vtx_ids,
                self.vtx_ide,
                self.vtx_ihs,
                self.vtx_ihe,
                self.vtx_ims,
                self.vtx_ime,
                self.n_vtx,
            )
        elif entity == "edge":
            start, end, size = self._index_bounds(
                domain,
                level,
                self.edge_ins,
                self.edge_ine,
                self.edge_ibs,
                self.edge_ibe,
                self.edge_ids,
                self.edge_ide,
                self.edge_ihs,
                self.edge_ihe,
                self.edge_ims,
                self.edge_ime,
                self.n_edge,
            )
        else:
            raise ValueError(f"Unknown mesh entity: {entity}")
        start = max(start, 0)
        end = min(end, size - 1)
        if start > end:
            return range(0)
        return range(start, end + 1)

    @staticmethod
    def _index_bounds(
        domain: DomainName,
        level: int | None,
        inner_start: IntArray,
        inner_end: IntArray,
        boundary_start: IntArray,
        boundary_end: IntArray,
        compute_start: IntArray,
        compute_end: IntArray,
        halo_start: IntArray,
        halo_end: IntArray,
        memory_start: IntArray,
        memory_end: IntArray,
        size: int,
    ) -> tuple[int, int, int]:
        if domain == "inner":
            return scalar_int(inner_start), scalar_int(inner_end), size
        if domain == "cmpt":
            return scalar_int(compute_start), scalar_int(compute_end), size
        if domain in ("full", "memory"):
            return scalar_int(memory_start), scalar_int(memory_end), size
        if domain == "bdry":
            return SCVTMesh._layered_bounds(boundary_start, boundary_end, level, "bdry", size)
        if domain == "halo":
            return SCVTMesh._layered_bounds(halo_start, halo_end, level, "halo", size)
        raise ValueError(f"Unknown domain: {domain}")

    @staticmethod
    def _layered_bounds(
        starts: IntArray,
        ends: IntArray,
        level: int | None,
        domain: DomainName,
        size: int,
    ) -> tuple[int, int, int]:
        start_values = np.ravel(np.asarray(starts, dtype=np.int64))
        end_values = np.ravel(np.asarray(ends, dtype=np.int64))
        if start_values.size != end_values.size:
            raise ValueError(
                f"{domain} start/end arrays have mismatched sizes: "
                f"{start_values.size} != {end_values.size}"
            )
        if level is None:
            return int(np.min(start_values)), int(np.max(end_values)), size
        level_idx = level
        if level_idx < 0 or level_idx >= start_values.size:
            raise ValueError(
                f"{domain}[{level}] is out of range; available levels are "
                f"0..{start_values.size - 1}"
            )
        return int(start_values[level_idx]), int(end_values[level_idx]), size

    def polygons(self, domain: DomainName, level: int | None = None) -> list[FloatArray]:
        polygons: list[FloatArray] = []
        for cell_idx in self.index_range(domain, level=level):
            nnb = int(self.cell_nnb[cell_idx])
            vtx_ids = self.vtx_of_cell[cell_idx, :nnb]
            vtx_ids = vtx_ids[(vtx_ids >= 0) & (vtx_ids < self.n_vtx)]
            if vtx_ids.size < 3:
                continue

            idx = vtx_ids
            lons = unwrap_polygon_lons(self.vtx_lon_deg[idx])
            lats = self.vtx_lat_deg[idx]
            polygons.append(np.asarray(np.column_stack((lons, lats)), dtype=np.float64))
        return polygons

    def dual_polygons(self, domain: DomainName, level: int | None = None) -> list[FloatArray]:
        if self.cell_of_vtx is None:
            raise ValueError(
                f"{self.file_path} does not contain cell_of_vtx for --overlay-dual"
            )

        polygons: list[FloatArray] = []
        for vtx_idx in self.index_range(domain, "vtx", level):
            cell_ids = self.cell_of_vtx[vtx_idx, :]
            cell_ids = cell_ids[(cell_ids >= 0) & (cell_ids < self.n_cell)]
            if cell_ids.size < 3:
                continue

            idx = cell_ids
            lons = unwrap_polygon_lons(self.cell_lon_deg[idx])
            lats = self.cell_lat_deg[idx]
            polygons.append(np.asarray(np.column_stack((lons, lats)), dtype=np.float64))
        return polygons

    def cell_points(
        self, domain: DomainName, level: int | None = None
    ) -> tuple[FloatArray, FloatArray]:
        indices = list(self.index_range(domain, "cell", level))
        return self.cell_lon_deg[indices], self.cell_lat_deg[indices]

    def cell_order_segments(self, domain: DomainName = "cmpt") -> list[Segment]:
        indices = list(self.index_range(domain, "cell"))
        segments: list[Segment] = []
        for prev_idx, next_idx in zip(indices[:-1], indices[1:]):
            lons = unwrap_polygon_lons(
                [self.cell_lon_deg[prev_idx], self.cell_lon_deg[next_idx]]
            )
            segments.append(
                (
                    (float(lons[0]), float(self.cell_lat_deg[prev_idx])),
                    (float(lons[1]), float(self.cell_lat_deg[next_idx])),
                )
            )
        return segments

    def vtx_points(
        self, domain: DomainName, level: int | None = None
    ) -> tuple[FloatArray, FloatArray]:
        indices = list(self.index_range(domain, "vtx", level))
        return self.vtx_lon_deg[indices], self.vtx_lat_deg[indices]

    def edge_segments(self, domain: DomainName, level: int | None = None) -> list[Segment]:
        segments: list[Segment] = []
        for edge_idx in self.index_range(domain, "edge", level):
            v1, v2 = int(self.edge_vtx[edge_idx, 0]), int(self.edge_vtx[edge_idx, 1])
            if not (0 <= v1 < self.n_vtx and 0 <= v2 < self.n_vtx):
                continue
            lons = unwrap_polygon_lons([self.vtx_lon_deg[v1], self.vtx_lon_deg[v2]])
            lats = [self.vtx_lat_deg[v1], self.vtx_lat_deg[v2]]
            segments.append(
                (
                    (float(lons[0]), float(lats[0])),
                    (float(lons[1]), float(lats[1])),
                )
            )
        return segments

    def site_table(self, site_kind: SiteKind, entity: MeshEntity) -> tuple[IntArray, IntArray]:
        count_name = {
            "cell": f"{site_kind}_ncell",
            "vtx": f"{site_kind}_nvtx",
            "edge": f"{site_kind}_nedge",
        }[entity]
        table_name = f"{site_kind}_{entity}"
        counts = getattr(self, count_name)
        table = getattr(self, table_name)
        if counts is None or table is None:
            raise ValueError(
                f"{self.file_path} does not contain {site_kind} {entity} site variables"
            )

        counts = np.ravel(np.asarray(counts, dtype=np.int64))
        table = self._normalize_site_table(np.asarray(table, dtype=np.int64), counts.size)
        return counts, table

    def n_site_neighbors(self, site_kind: SiteKind) -> int:
        counts, _ = self.site_table(site_kind, "cell")
        return counts.size

    def site_indices_for_neighbor(
        self, site_kind: SiteKind, entity: MeshEntity, neighbor_idx: int
    ) -> IntArray:
        counts, table = self.site_table(site_kind, entity)
        if not (0 <= neighbor_idx < counts.size):
            raise ValueError(
                f"Neighbor index {neighbor_idx + 1} is out of range for {self.file_path}"
            )

        max_index = {"cell": self.n_cell, "vtx": self.n_vtx, "edge": self.n_edge}[entity]
        selected: set[int] = set()
        count = max(int(counts[neighbor_idx]), 0)
        if count == 0 or table.shape[1] == 0:
            return np.asarray([], dtype=np.int64)
        for value in table[neighbor_idx, : min(count, table.shape[1])]:
            idx = int(value)
            if 0 <= idx < max_index:
                selected.add(idx)

        return np.asarray(sorted(selected), dtype=np.int64)

    def site_indices(self, site_kind: SiteKind, entity: MeshEntity) -> IntArray:
        neighbor_indices = [
            self.site_indices_for_neighbor(site_kind, entity, neighbor_idx)
            for neighbor_idx in range(self.n_site_neighbors(site_kind))
        ]
        if not neighbor_indices:
            return np.asarray([], dtype=np.int64)
        return np.unique(np.concatenate(neighbor_indices))

    def site_points(
        self, site_kind: SiteKind, entity: MeshEntity
    ) -> tuple[FloatArray, FloatArray]:
        indices = self.site_indices(site_kind, entity)
        return self.site_points_from_indices(entity, indices)

    def site_points_for_neighbor(
        self, site_kind: SiteKind, entity: MeshEntity, neighbor_idx: int
    ) -> tuple[FloatArray, FloatArray]:
        indices = self.site_indices_for_neighbor(site_kind, entity, neighbor_idx)
        return self.site_points_from_indices(entity, indices)

    def site_points_from_indices(
        self, entity: MeshEntity, indices: IntArray
    ) -> tuple[FloatArray, FloatArray]:
        if entity == "cell":
            return self.cell_lon_deg[indices], self.cell_lat_deg[indices]
        if entity == "vtx":
            return self.vtx_lon_deg[indices], self.vtx_lat_deg[indices]
        return self.edge_lon_deg[indices], self.edge_lat_deg[indices]

    def site_edge_segments(self, site_kind: SiteKind) -> list[Segment]:
        neighbor_segments = [
            self.site_edge_segments_for_neighbor(site_kind, neighbor_idx)
            for neighbor_idx in range(self.n_site_neighbors(site_kind))
        ]
        return [segment for segments in neighbor_segments for segment in segments]

    def site_edge_segments_for_neighbor(
        self, site_kind: SiteKind, neighbor_idx: int
    ) -> list[Segment]:
        segments: list[Segment] = []
        for edge_idx in self.site_indices_for_neighbor(site_kind, "edge", neighbor_idx):
            v1, v2 = int(self.edge_vtx[edge_idx, 0]), int(self.edge_vtx[edge_idx, 1])
            if not (0 <= v1 < self.n_vtx and 0 <= v2 < self.n_vtx):
                continue
            lons = unwrap_polygon_lons([self.vtx_lon_deg[v1], self.vtx_lon_deg[v2]])
            lats = [self.vtx_lat_deg[v1], self.vtx_lat_deg[v2]]
            segments.append(
                (
                    (float(lons[0]), float(lats[0])),
                    (float(lons[1]), float(lats[1])),
                )
            )
        return segments

    def partition_boundary_segments(self) -> list[Segment]:
        compute = set(range(scalar_int(self.cell_ids), scalar_int(self.cell_ide) + 1))
        halo = set(range(scalar_int(self.cell_ide) + 1, scalar_int(self.cell_ime) + 1))
        segments: list[Segment] = []

        for edge_cells, edge_vertices in zip(self.dual_edge_vtx, self.edge_vtx):
            c1, c2 = int(edge_cells[0]), int(edge_cells[1])
            crosses_partition = (c1 in compute and c2 in halo) or (
                c2 in compute and c1 in halo
            )
            if not crosses_partition:
                continue

            v1, v2 = int(edge_vertices[0]), int(edge_vertices[1])
            if not (0 <= v1 < self.n_vtx and 0 <= v2 < self.n_vtx):
                continue

            lons = unwrap_polygon_lons([self.vtx_lon_deg[v1], self.vtx_lon_deg[v2]])
            lats = [self.vtx_lat_deg[v1], self.vtx_lat_deg[v2]]
            segments.append(
                (
                    (float(lons[0]), float(lats[0])),
                    (float(lons[1]), float(lats[1])),
                )
            )
        return segments

    def center(self, domain: DomainName, level: int | None = None) -> Point | None:
        indices = np.fromiter(self.index_range(domain, level=level), dtype=np.int64)
        if indices.size == 0:
            return None

        weights = np.asarray(self.area_cell[indices], dtype=np.float64)
        if not np.all(np.isfinite(weights)) or np.sum(weights) <= 0.0:
            weights = np.ones(indices.size, dtype=np.float64)

        xyz = np.asarray(
            np.average(self.cell_xyz[indices, :], axis=0, weights=weights),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(xyz))
        if norm <= 0.0:
            return None

        xyz_unit = np.asarray(xyz / norm, dtype=np.float64)
        lon = math.degrees(math.atan2(xyz_unit[1], xyz_unit[0]))
        lat = math.degrees(math.asin(np.clip(xyz_unit[2], -1.0, 1.0)))
        return lon, lat


class SCVTMeshDecomposition:
    def __init__(self, input_files: Sequence[Path]) -> None:
        self.input_files = list(input_files)
        self.meshes = [
            SCVTMesh(path, SCVTMesh.rank_from_path(path, idx))
            for idx, path in enumerate(self.input_files)
        ]

    @classmethod
    def from_inputs(cls, inputs: Sequence[str]) -> SCVTMeshDecomposition:
        return cls(cls._expand_inputs(inputs))

    @staticmethod
    def _expand_inputs(inputs: Sequence[str]) -> list[Path]:
        files: list[Path] = []
        for item in inputs:
            path = Path(item).expanduser()
            if path.is_dir():
                files.extend(sorted(path.glob("*.nc")))
                continue

            matches = sorted(glob.glob(str(path)))
            if matches:
                files.extend(Path(match) for match in matches)
            else:
                files.append(path)

        files = [path.resolve() for path in files]
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Input file not found: {missing[0]}")
        if not files:
            raise FileNotFoundError("No NetCDF input files found")

        return sorted(set(files), key=lambda p: (SCVTMesh.rank_from_path(p, 10**9), str(p)))

    def meshes_for_view(self, rank: int | None) -> list[SCVTMesh]:
        if rank is None:
            return self.meshes

        selected = [mesh for mesh in self.meshes if mesh.rank == rank]
        if not selected:
            available = ", ".join(str(mesh.rank) for mesh in self.meshes)
            raise ValueError(f"Rank {rank} is not in the input files. Available ranks: {available}")
        return selected

    def projection_center(
        self,
        args: PlotArgs,
        center_domain: DomainName,
        meshes: Sequence[SCVTMesh],
        center_level: int | None = None,
    ) -> Point:
        central_lon = args.central_lon
        central_lat = args.central_lat

        if args.auto_center:
            centers: list[Point] = []
            for mesh in meshes:
                center = mesh.center(center_domain, center_level)
                if center is not None:
                    centers.append(center)

            if centers:
                xyz: list[list[float]] = []
                for lon, lat in centers:
                    lon_rad = math.radians(lon)
                    lat_rad = math.radians(lat)
                    xyz.append(
                        [
                            math.cos(lat_rad) * math.cos(lon_rad),
                            math.cos(lat_rad) * math.sin(lon_rad),
                            math.sin(lat_rad),
                        ]
                    )
                mean_xyz = np.asarray(np.mean(np.asarray(xyz), axis=0), dtype=np.float64)
                norm = float(np.linalg.norm(mean_xyz))
                if norm > 1.0e-12:
                    mean_xyz_unit = np.asarray(mean_xyz / norm, dtype=np.float64)
                    central_lon = math.degrees(math.atan2(mean_xyz_unit[1], mean_xyz_unit[0]))
                    central_lat = math.degrees(
                        math.asin(np.clip(mean_xyz_unit[2], -1.0, 1.0))
                    )

        if central_lon is None:
            central_lon = 0.0
        if central_lat is None:
            central_lat = 90.0 if args.projection == "orthographic" else 0.0
        return central_lon, central_lat

    def draw_cell_order(
        self,
        ax: GeoAxesLike,
        data_crs: ccrs.Projection,
        meshes: Sequence[SCVTMesh],
        args: PlotArgs,
        *,
        zorder: int,
    ) -> None:
        for mesh in meshes:
            order_segments = mesh.cell_order_segments("cmpt")
            if not order_segments:
                continue
            ax.add_collection(
                LineCollection(
                    order_segments,
                    colors=CELL_ORDER_COLOR,
                    linewidths=max(args.outline_width, 1.25),
                    alpha=0.95,
                    transform=data_crs,
                    zorder=zorder,
                )
            )

    @staticmethod
    def overlay_polygons(
        mesh: SCVTMesh, domain: DomainName, overlay_dual: bool
    ) -> list[FloatArray]:
        if overlay_dual:
            return mesh.dual_polygons(domain)
        return mesh.polygons(domain)

    def plot_sites(self, args: PlotArgs, site_mode: SiteMode) -> Path | None:
        site_kinds = site_kinds_for_mode(site_mode)
        domain_meshes = self.meshes_for_view(args.domain_view)
        overlay_meshes = self.meshes_for_view(args.overlay_view)
        central_lon, central_lat = self.projection_center(args, "full", domain_meshes)
        projection = projection_from_name(
            args.projection, central_lon=central_lon, central_lat=central_lat
        )
        data_crs = ccrs.Geodetic()

        fig = cast(Figure, plt.figure(figsize=(args.width, args.height), facecolor="white"))
        ax = cast(GeoAxesLike, fig.add_subplot(1, 1, 1, projection=projection))
        ax.set_global()

        if args.coastlines:
            ax.coastlines(linewidth=0.35, color="0.35")
        if args.gridlines:
            ax.gridlines(crs=ccrs.PlateCarree(), color="0.55", alpha=0.45, linestyle="--")

        cmap = plt.get_cmap(args.cmap)
        rank_count = max(len(self.meshes) - 1, 1)
        site_legend_handles: list[Line2D] = []
        site_legend_labels: set[str] = set()

        if args.overlay != "none":
            overlay_domain = cast(DomainName, args.overlay)
            for mesh in overlay_meshes:
                color = cmap(mesh.rank / rank_count)
                overlay_polygons = self.overlay_polygons(
                    mesh, overlay_domain, args.overlay_dual
                )
                if not overlay_polygons:
                    continue
                ax.add_collection(
                    PolyCollection(
                        overlay_polygons,
                        facecolors=[color],
                        edgecolors="#A8A8A8" if args.edges else "none",
                        linewidths=args.linewidth,
                        alpha=args.overlay_alpha,
                        transform=data_crs,
                        zorder=1,
                    )
                )

        if args.draw_cell_order:
            self.draw_cell_order(ax, data_crs, domain_meshes, args, zorder=3)

        neighbor_color_offset = 0
        for mesh in domain_meshes:
            if args.outline_partitions:
                segments = mesh.partition_boundary_segments()
                if segments:
                    ax.add_collection(
                        LineCollection(
                            segments,
                            colors=args.outline_color,
                            linewidths=args.outline_width,
                            transform=data_crs,
                            zorder=2,
                        )
                    )

            mesh_neighbor_count = max(
                (mesh.n_site_neighbors(site_kind) for site_kind in site_kinds),
                default=0,
            )
            for site_kind in site_kinds:
                for neighbor_idx in range(mesh.n_site_neighbors(site_kind)):
                    color_idx = neighbor_color_offset + neighbor_idx
                    color = site_color(site_kind, color_idx)
                    edge_segments = mesh.site_edge_segments_for_neighbor(
                        site_kind, neighbor_idx
                    )
                    if edge_segments:
                        ax.add_collection(
                            LineCollection(
                                edge_segments,
                                colors=color,
                                linewidths=max(args.linewidth * 3.0, 0.7),
                                alpha=0.95,
                                transform=data_crs,
                                zorder=4,
                            )
                        )

                    for entity in ("edge", "cell", "vtx"):
                        site_lon, site_lat = mesh.site_points_for_neighbor(
                            site_kind, entity, neighbor_idx
                        )
                        if site_lon.size == 0:
                            continue
                        ax.scatter(
                            site_lon,
                            site_lat,
                            s=SITE_MARKER_SIZES[entity],
                            color=color,
                            marker=SITE_MARKERS[(site_kind, entity)],
                            linewidths=0.25,
                            edgecolors="black",
                            transform=data_crs,
                            zorder=6 if entity != "edge" else 5,
                        )
                        label = (
                            f"rank {mesh.rank} nb {neighbor_idx + 1} "
                            f"{site_kind} {entity}"
                        )
                        if label not in site_legend_labels:
                            site_legend_labels.add(label)
                            site_legend_handles.append(
                                Line2D(
                                    [0],
                                    [0],
                                    color=color,
                                    marker=SITE_MARKERS[(site_kind, entity)],
                                    linestyle="-" if entity == "edge" else "none",
                                    markeredgecolor="black",
                                    label=label,
                                )
                            )
            neighbor_color_offset += mesh_neighbor_count

        if args.legend:
            if args.draw_cell_order:
                site_legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=CELL_ORDER_COLOR,
                        linestyle="-",
                        label="cmpt cell order",
                    )
                )
            ax.legend(
                handles=site_legend_handles,
                loc="lower left",
                fontsize=8,
                frameon=True,
                framealpha=0.9,
            )

        title = args.title
        if title is None:
            view_label = "all ranks" if args.domain_view is None else f"rank {args.domain_view}"
            site_label = "+".join(site_kinds)
            title = f"SCVT mesh {site_label} sites ({view_label}, {args.projection})"
        if title:
            ax.set_title(title)

        if args.output is not None:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            return output

        plt.show()
        return None

    def plot(self, args: PlotArgs) -> Path | None:
        site_mode = site_mode_from_args(args)
        if site_mode != "none":
            return self.plot_sites(args, site_mode)

        domains = parse_domain_args(args.domain)
        domain_meshes = self.meshes_for_view(args.domain_view)
        overlay_meshes = self.meshes_for_view(args.overlay_view)
        center_meshes = domain_meshes if domain_meshes else overlay_meshes
        central_lon, central_lat = self.projection_center(
            args, domains[0].name, center_meshes, domains[0].level
        )
        projection = projection_from_name(
            args.projection, central_lon=central_lon, central_lat=central_lat
        )
        data_crs = ccrs.Geodetic()

        fig = cast(Figure, plt.figure(figsize=(args.width, args.height), facecolor="white"))
        ax = cast(GeoAxesLike, fig.add_subplot(1, 1, 1, projection=projection))
        ax.set_global()

        if args.coastlines:
            ax.coastlines(linewidth=0.35, color="0.35")
        if args.gridlines:
            ax.gridlines(crs=ccrs.PlateCarree(), color="0.55", alpha=0.45, linestyle="--")

        cmap = plt.get_cmap(args.cmap)
        rank_count = max(len(self.meshes) - 1, 1)
        legend_handles: list[Patch | Line2D] = []

        for mesh in overlay_meshes:
            color = cmap(mesh.rank / rank_count)
            if args.show_halo and not any(domain.name == "halo" for domain in domains):
                halo_polygons = mesh.polygons("halo")
                if halo_polygons:
                    ax.add_collection(
                        PolyCollection(
                            halo_polygons,
                            facecolors=[color],
                            edgecolors="0.65",
                            linewidths=args.linewidth * 0.8,
                            alpha=args.halo_alpha,
                            transform=data_crs,
                            zorder=1,
                        )
                    )

            if args.overlay != "none":
                overlay_domain = cast(DomainName, args.overlay)
                overlay_polygons = self.overlay_polygons(
                    mesh, overlay_domain, args.overlay_dual
                )
                if overlay_polygons:
                    ax.add_collection(
                        PolyCollection(
                            overlay_polygons,
                            facecolors=[color],
                            edgecolors="0.65" if args.edges else "none",
                            linewidths=args.linewidth * 0.8,
                            alpha=args.overlay_alpha,
                            transform=data_crs,
                            zorder=2,
                        )
                    )

            overlay_label = "dual overlay" if args.overlay_dual else "overlay"
            legend_handles.append(
                Patch(
                    facecolor=color,
                    edgecolor="black",
                    label=f"rank {mesh.rank} {overlay_label}",
                )
            )

        for mesh in domain_meshes:
            for domain in domains:
                domain_color = DOMAIN_COLORS[domain.name]
                edge_segments = mesh.edge_segments(domain.name, domain.level)
                if edge_segments:
                    ax.add_collection(
                        LineCollection(
                            edge_segments,
                            colors=domain_color,
                            linewidths=max(args.outline_width, 1.35),
                            alpha=0.95,
                            transform=data_crs,
                            zorder=3,
                        )
                    )

                cell_lon, cell_lat = mesh.cell_points(domain.name, domain.level)
                ax.scatter(
                    cell_lon,
                    cell_lat,
                    s=12,
                    color=domain_color,
                    marker="o",
                    linewidths=0.2,
                    edgecolors="white",
                    transform=data_crs,
                    zorder=4,
                )

                vtx_lon, vtx_lat = mesh.vtx_points(domain.name, domain.level)
                ax.scatter(
                    vtx_lon,
                    vtx_lat,
                    s=22,
                    color=domain_color,
                    marker=DOMAIN_VTX_MARKERS[domain.name],
                    linewidths=0.35,
                    edgecolors="black",
                    transform=data_crs,
                    zorder=5,
                )

            if args.outline_partitions:
                segments = mesh.partition_boundary_segments()
                if segments:
                    ax.add_collection(
                        LineCollection(
                            segments,
                            colors=args.outline_color,
                            linewidths=args.outline_width,
                            transform=data_crs,
                            zorder=4,
                        )
                    )

            if args.centers:
                indices = list(mesh.index_range(domains[0].name, "cell", domains[0].level))
                ax.scatter(
                    mesh.cell_lon_deg[indices],
                    mesh.cell_lat_deg[indices],
                    s=args.center_size,
                    color="black",
                    marker=".",
                    linewidths=0,
                    edgecolors="none",
                    transform=data_crs,
                    zorder=6,
                )

        if args.draw_cell_order:
            self.draw_cell_order(ax, data_crs, domain_meshes, args, zorder=6)

        if args.legend:
            for domain in domains:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=DOMAIN_COLORS[domain.name],
                        marker=".",
                        linestyle="-",
                        label=f"{domain.label}: edge/cell/vtx",
                    )
                )
            if args.draw_cell_order:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=CELL_ORDER_COLOR,
                        linestyle="-",
                        label="cmpt cell order",
                    )
                )
            ax.legend(
                handles=legend_handles,
                loc="lower left",
                fontsize=8,
                frameon=True,
                framealpha=0.9,
            )

        title = args.title
        if title is None:
            domain_label = ",".join(domain.label for domain in domains)
            overlay_label = f"{args.overlay} dual" if args.overlay_dual else args.overlay
            title = (
                f"SCVT mesh decomposition "
                f"(overlay {overlay_label}, domain {domain_label}, {args.projection})"
            )
        if title:
            ax.set_title(title)

        if args.output is not None:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            return output

        plt.show()
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot local SCVT mesh decomposition files written by "
            "terraflow.mesh.scvt.mesh_type.SCVTMesh.output()."
        )
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Input .nc file(s), directory, or glob pattern such as 'scvt_mesh.*.nc'.",
    )
    parser.add_argument(
        "-p",
        "--projection",
        default="robinson",
        type=lambda value: value.lower().replace("_", "-"),
        choices=(
            "plate-carree",
            "platecarree",
            "latlon",
            "lonlat",
            "robinson",
            "orthographic",
            "mollweide",
            "equal-earth",
            "equalearth",
            "north-polar",
            "northpolar",
            "south-polar",
            "southpolar",
            "stereographic",
        ),
        help="Cartopy projection used for the output image.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output image path. If omitted, open the interactive plotting window.",
    )
    parser.add_argument(
        "--domain",
        nargs="+",
        default=["all"],
        metavar="DOMAIN",
        help=(
            "Area(s) to mark on top of --overlay. Use one or more values from "
            "all, inner, cmpt, bdry, halo, full, memory; bdry[k]/halo[k] select "
            "a single 0-based boundary/halo layer, e.g. bdry[0],halo[1]. "
            "bdry:k/halo:k are equivalent shell-safe forms for zsh. "
            "Comma-separated values are also accepted. Default: all. 'compute' "
            "and 'boundary' are accepted as compatibility aliases."
        ),
    )
    parser.add_argument(
        "--overlay",
        default="cmpt",
        type=parse_overlay_arg,
        choices=("none", "inner", "bdry", "cmpt", "halo", "full", "memory"),
        help=(
            "Cell index range drawn underneath --domain markers or send/recv site "
            "markers. Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--domain-view",
        type=int,
        default=None,
        help="Only draw --domain markers for this rank. Defaults to all input ranks.",
    )
    site_group = parser.add_mutually_exclusive_group()
    site_group.add_argument(
        "--with-send",
        action="store_true",
        help=(
            "Ignore --domain and draw send cell/vtx/edge sites on the mesh selected "
            "by --domain-view; background mesh is selected by --overlay/--overlay-view."
        ),
    )
    site_group.add_argument(
        "--with-recv",
        action="store_true",
        help=(
            "Ignore --domain and draw recv cell/vtx/edge sites on the mesh selected "
            "by --domain-view; background mesh is selected by --overlay/--overlay-view."
        ),
    )
    site_group.add_argument(
        "--with-sendrecv",
        action="store_true",
        help=(
            "Ignore --domain and draw both send and recv cell/vtx/edge sites on the "
            "mesh selected by --domain-view; background mesh is selected by "
            "--overlay/--overlay-view."
        ),
    )
    parser.add_argument(
        "--overlay-view",
        type=int,
        default=None,
        help="Only draw --overlay polygons for this rank. Defaults to all input ranks.",
    )
    parser.add_argument(
        "--overlay-dual",
        action="store_true",
        help="Draw --overlay with the dual triangular mesh instead of polygon cells.",
    )
    parser.add_argument("--central-lon", type=float, default=None)
    parser.add_argument("--central-lat", type=float, default=None)
    parser.add_argument(
        "--auto-center",
        action="store_true",
        help="Center orthographic/stereographic views on the plotted cell centroids.",
    )
    parser.add_argument("--show-halo", action="store_true", help="Draw halo cells underneath.")
    parser.add_argument("--halo-alpha", type=float, default=0.18)
    parser.add_argument(
        "--outline-partitions",
        dest="outline_partitions",
        action="store_true",
        help="Draw thick local partition boundaries.",
    )
    parser.add_argument("--no-outline-partitions", dest="outline_partitions", action="store_false")
    parser.add_argument("--edges", dest="edges", action="store_true", help="Draw cell edges.")
    parser.add_argument("--no-edges", dest="edges", action="store_false")
    parser.add_argument("--edge-color", default="black")
    parser.add_argument("--outline-color", default="black")
    parser.add_argument("--linewidth", type=float, default=0.18)
    parser.add_argument("--outline-width", type=float, default=1.1)
    parser.add_argument("--alpha", type=float, default=0.72)
    parser.add_argument("--overlay-alpha", type=float, default=0.22)
    parser.add_argument("--cmap", default="tab20")
    parser.add_argument("--coastlines", action="store_true")
    parser.add_argument("--gridlines", action="store_true")
    parser.add_argument("--legend", action="store_true")
    parser.add_argument("--centers", action="store_true", help="Overlay plotted cell centers.")
    parser.add_argument(
        "--draw-cell-order",
        action="store_true",
        help="Connect cmpt-domain cell centers in local index order.",
    )
    parser.add_argument("--center-size", type=float, default=1.0)
    parser.add_argument("--title", default=None, help="Figure title. Use '' to suppress.")
    parser.add_argument("--width", type=float, default=14.0)
    parser.add_argument("--height", type=float, default=9.0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.set_defaults(outline_partitions=True, edges=True)
    return parser


if __name__ == "__main__":
    cli_parser = build_parser()
    cli_args = cast(PlotArgs, cli_parser.parse_args())
    try:
        decomp = SCVTMeshDecomposition.from_inputs(cli_args.input)
        output_path = decomp.plot(cli_args)
    except KeyboardInterrupt:
        raise SystemExit(0)
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        cli_parser.exit(2, f"{cli_parser.prog}: error: {exc}\n")
    if output_path is not None:
        print(f"Saved {output_path}; loaded {len(decomp.meshes)} file(s)")
