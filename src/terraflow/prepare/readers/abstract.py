from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Generic, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
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
