from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Generic, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
import numpy as np
import xarray as xr


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.util.device import AbstractDevice
    from terraflow.planet import PlanetParameterSet


    Float = typing.Float
    Array = typing.Array


READERCFG = TypeVar("READERCFG", bound="AbstractReaderConfig")


class AbstractReaderConfig(ABC):
    pass


@dataclass
class TopographyReaderConfig(AbstractReaderConfig):
    # "interp" or "fill"
    regrid_type           : str   = "fill"

    # "esmf" or "native"
    interp_backend        : str   = "esmf"

    # "nearest", "bilinear", "conserve_1st", "conserve_2nd"
    interp_method         : str   = "conserve_1st"

    # norm_opt: "destarea" or "fracarea"
    interp_conserve_opt   : dict[str, Any] = field(default_factory=lambda: {"norm_opt": "destarea"})

    # "scipy" or "jaxkd"
    fill_backend          : str   = "scipy"

    # True or False
    fill_split_with_chunks: bool  = True

    # Int
    fill_lon_chunk_size   : int   = 360
    fill_lat_chunk_size   : int   = 180

    # zs grad filter parameters
    use_zs_grad_filter    : bool  = False
    zs_grad_filter_type   : str   = "laplacian1"  # "laplacian1" or "laplacian2"
    zs_grad_filter_order  : int   = 2
    zs_grad_filter_coef   : float = 1.0e6
    zs_grad_filter_cycles : int   = 100
    topo_max_slope        : float = 0.12

    def __print__(self):
        print("================ Topography Reader Config =================="     )
        if self.regrid_type == "interp":
            print(f"  regrid_type                = {self.regrid_type           }")
            print(f"  interp_backend             = {self.interp_backend        }")
            print(f"  interp_method              = {self.interp_method         }")
            print(f"  interp_conserve_opt        = {self.interp_conserve_opt   }")
        elif self.regrid_type == "fill":
            print(f"  regrid_type                = {self.regrid_type           }")
            print(f"  fill_backend               = {self.fill_backend          }")
            print(f"  fill_split_with_chunks     = {self.fill_split_with_chunks}")
            print(f"  fill_lon_chunk_size        = {self.fill_lon_chunk_size   }")
            print(f"  fill_lat_chunk_size        = {self.fill_lat_chunk_size   }")
        else:
            raise ValueError(
                f"Unsupported regrid_type: {self.regrid_type}. Supported types: 'interp', 'fill'."
            )
        print    (f"  use_zs_grad_filter         = {self.use_zs_grad_filter    }")
        if self.zs_grad_filter_type == "laplacian1":
            print(f"  zs_grad_filter_type        = {self.zs_grad_filter_type   }")
            print(f"  zs_grad_filter_order       = {self.zs_grad_filter_order  }")
            print(f"  zs_grad_filter_coef        = {self.zs_grad_filter_coef   }")
            print(f"  zs_grad_filter_cycles      = {self.zs_grad_filter_cycles }")
            print(f"  topo_max_slope             = {self.topo_max_slope        }")
        elif self.zs_grad_filter_type == "laplacian2":
            print(f"  zs_grad_filter_type        = {self.zs_grad_filter_type   }")
            print(f"  zs_grad_filter_cycles      = {self.zs_grad_filter_cycles }")
        print("============================================================"     )


@dataclass
class BackgroundReaderConfig(AbstractReaderConfig):
    # "esmf" or "native"
    hori_interp_backend: str  = "esmf"

    # "nearest", "bilinear", "conserve_1st", "conserve_2nd", "patch"
    hori_interp_method : str  = "conserve_1st"

    # True or False
    interp_at_midpoint : bool = False

    # "none", "nearest_stod", "nearest_idavg", "creep_fill"
    esmf_extrap_method : str  = "none"

    # norm_opt: "destarea" or "fracarea"
    esmf_conserve_opt  : dict[str, Any] = field(default_factory=lambda: {"norm_opt": "fracarea"})

    # True or False
    prepare_regrid_gz  : bool  = True

    # "jax" or "native"
    vert_interp_backend: str   = "jax"

    def __print__(self):
        print("================ Background Reader Config ==================")
        print(f"  hori_interp_backend        = {self.hori_interp_backend  }")
        print(f"  hori_interp_method         = {self.hori_interp_method   }")
        print(f"  interp_at_midpoint         = {self.interp_at_midpoint   }")
        print(f"  esmf_extrap_method         = {self.esmf_extrap_method   }")
        print(f"  esmf_conserve_opt          = {self.esmf_conserve_opt    }")
        print(f"  prepare_regrid_gz          = {self.prepare_regrid_gz    }")
        print(f"  vert_interp_backend        = {self.vert_interp_backend  }")
        print("============================================================")


class AbstractReader(ABC, Generic[READERCFG]):
    loaded: bool

    def __init__(self, para: PlanetParameterSet, cfg: READERCFG, dev: AbstractDevice):
        self.para = para
        self.cfg = cfg
        self.dev = dev
        self.loaded = False

    @property
    @abstractmethod
    def reader_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def load_data(self, *args, **kwarg):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def lon_range(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def unload(self):
        raise NotImplementedError

    def has_loaded(self) -> bool:
        return self.loaded

    def get_config(self) -> READERCFG:
        return self.cfg

    @staticmethod
    def read_coord(ds: xr.Dataset, dim_name: str) -> tuple[xr.Dataset, np.ndarray]:
        coord = np.asarray(ds[dim_name].values)
        if coord.ndim != 1:
            raise ValueError(f"Coordinate '{dim_name}' must be 1D, got shape {coord.shape}.")
        if coord.size < 2:
            raise ValueError(f"Coordinate '{dim_name}' must have at least 2 points.")
        if np.all(np.diff(coord) > 0.0):
            return ds, coord.astype(np.float64, copy=False)
        if np.all(np.diff(coord) < 0.0):
            ds = ds.isel({dim_name: slice(None, None, -1)})
        else:
            order = np.argsort(coord)
            ds = ds.isel({dim_name: order})
        return ds, np.asarray(ds[dim_name].values, dtype=np.float64)

    @staticmethod
    def _drop_cyclic_endpoint(ds: xr.Dataset, dim_name: str, length: float) -> xr.Dataset:
        coord = np.asarray(ds[dim_name].values, dtype=np.float64)
        if coord.size > 2 and np.isclose(coord[-1] - coord[0], length, rtol=0.0, atol=1.0e-10):
            return ds.isel({dim_name: slice(0, -1)})
        return ds

    @classmethod
    def find_range_indices(
        cls,
        coord: np.ndarray,
        coord_range: tuple[float, float],
        *,
        cyclic: bool = False,
        span: tuple[float, float] | None = None,
    ) -> list[tuple[int, int, float]]:
        min_value, max_value = coord_range
        if not cyclic:
            if min_value > max_value:
                raise ValueError(
                    f"Non-cyclic coordinate range must be increasing, got "
                    f"{min_value} > {max_value}."
                )
            i0 = int(np.searchsorted(coord, min_value, side="left")) - 1
            i1 = int(np.searchsorted(coord, max_value, side="left"))
            i0, i1 = cls._clip_range_indices(i0, i1, coord.size)
            return [(i0, i1 + 1, 0.0)]

        if span is None:
            raise ValueError("Cyclic range selection requires span.")
        length = float(span[1] - span[0])
        if length <= 0.0:
            raise ValueError(f"Cyclic span must be increasing, got {span}.")

        raw_width = max_value - min_value
        if raw_width >= length - 1.0e-10:
            return [(0, coord.size, 0.0)]

        start = (min_value - span[0]) % length + span[0]
        end = (max_value - span[0]) % length + span[0]
        while end <= start:
            end += length

        segments: list[tuple[int, int, float]] = []
        for cycle in (0, 1):
            offset = cycle * length
            coord_shifted = coord + offset
            if coord_shifted[-1] < start or coord_shifted[0] > end:
                continue
            i0 = int(np.searchsorted(coord_shifted, start, side="left")) - 1
            i1 = int(np.searchsorted(coord_shifted, end, side="left"))
            i0, i1 = cls._clip_range_indices(i0, i1, coord.size)
            segments.append((i0, i1 + 1, offset))

        if not segments:
            raise ValueError(
                f"Could not select cyclic range [{min_value}, {max_value}] "
                f"from coordinate span {span}."
            )
        return segments

    @staticmethod
    def _clip_range_indices(i0: int, i1: int, size: int) -> tuple[int, int]:
        i0 = max(i0, 0)
        i1 = min(i1, size - 1)
        if i1 <= i0:
            i0 = max(min(i0, size - 2), 0)
            i1 = min(i0 + 1, size - 1)
        return i0, i1

    @classmethod
    def input_range(
        cls,
        ds: xr.Dataset,
        *,
        lon_name: str,
        lat_name: str,
        lon_span: tuple[float, float],
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
    ) -> xr.Dataset:
        ds, _ = cls.read_coord(ds, lon_name)
        ds, _ = cls.read_coord(ds, lat_name)
        ds = cls._drop_cyclic_endpoint(ds, lon_name, lon_span[1] - lon_span[0])

        lon_coord = np.asarray(ds[lon_name].values, dtype=np.float64)
        lat_coord = np.asarray(ds[lat_name].values, dtype=np.float64)
        lon_segments = cls.find_range_indices(
            lon_coord,
            coord_range=(min_lon, max_lon),
            cyclic=True,
            span=lon_span,
        )
        lat_segments = cls.find_range_indices(
            lat_coord,
            coord_range=(
                max(min_lat, float(lat_coord[0])),
                min(max_lat, float(lat_coord[-1])),
            ),
        )
        lat_start, lat_stop, _ = lat_segments[0]

        parts: list[xr.Dataset] = []
        for lon_start, lon_stop, offset in lon_segments:
            part = ds.isel({
                lon_name: slice(lon_start, lon_stop),
                lat_name: slice(lat_start, lat_stop),
            })
            if offset != 0.0:
                part = part.assign_coords({lon_name: part[lon_name] + offset})
            parts.append(part)

        if len(parts) == 1:
            return parts[0].load()
        selected = xr.concat(parts, dim=lon_name)
        if not isinstance(selected, xr.Dataset):
            raise TypeError("input_range expects an xarray Dataset.")
        return selected.load()

    @staticmethod
    def check_input_range(reader_type: str, lon: np.ndarray, lat: np.ndarray) -> None:
        if lon.size < 2 or lat.size < 2:
            raise ValueError(
                f"{reader_type} bbox selection is too small for bilinear interpolation: "
                f"longitude={lon.size}, latitude={lat.size}"
            )
        if not np.all(np.diff(lon) > 0.0):
            raise ValueError(f"{reader_type} longitude must be strictly increasing.")
        if not np.all(np.diff(lat) > 0.0):
            raise ValueError(f"{reader_type} latitude must be strictly increasing.")


class AbstractTopographyReader(AbstractReader[TopographyReaderConfig]):
    topo_file: str
    lon: Float[Array, "nlon"]
    lat: Float[Array, "nlat"]
    gzs: Float[Array, "nlev"]
    mask: Float[Array, "nlat", "nlon"]

    def __init__(
        self,
        topo_file: str,
        para: PlanetParameterSet,
        dev: AbstractDevice,
        cfg: TopographyReaderConfig = TopographyReaderConfig()
    ):
        super().__init__(para=para, cfg=cfg, dev=dev)
        if not os.path.exists(topo_file):
            raise FileNotFoundError(
                f"Topography file '{topo_file}' not found!"
            )
        if dev.is_root(): self.cfg.__print__()
        self.topo_file = topo_file

    def open_dataset(self) -> xr.Dataset:
        return xr.open_dataset(self.topo_file)


class AbstractBackgroundReader(AbstractReader[BackgroundReaderConfig]):
    bkg_file: str
    time_str: str
    lon: Float[Array, "nlon"]
    lat: Float[Array, "nlat"]
    lev: Float[Array, "nlev"]

    def __init__(
        self,
        bkg_file: str,
        para: PlanetParameterSet,
        dev: AbstractDevice,
        cfg: BackgroundReaderConfig = BackgroundReaderConfig()
    ):
        super().__init__(para=para, cfg=cfg, dev=dev)
        if not os.path.exists(bkg_file):
            raise FileNotFoundError(
                f"Background file '{bkg_file}' not found!"
            )
        if dev.is_root(): self.cfg.__print__()
        self.bkg_file = bkg_file

    def open_dataset(self) -> xr.Dataset:
        return xr.open_dataset(self.bkg_file)
