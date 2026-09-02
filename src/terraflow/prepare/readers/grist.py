from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
import gc
import jax
import jax.numpy as jnp
import xarray as xr

from terraflow.util import typing, log
from terraflow.prepare.readers.abstract import (
    AbstractReader,
    AbstractReaderConfig
)
if TYPE_CHECKING:
    from terraflow.util.device import AbstractDevice
    from terraflow.util.planet import PlanetParameterSet

Float = typing.Float
Array = typing.Array


@dataclass
class GRISTStaticDataReaderConfig(AbstractReaderConfig):
    # zs grad filter parameters
    use_zs_grad_filter   : bool = False
    zs_grad_filter_type  : str = "laplacian2"  # "laplacian1" or "laplacian2"
    zs_grad_filter_order : int = 2
    zs_grad_filter_coef  : float = 1.0e6
    zs_grad_filter_cycles: int = 100
    topo_max_slope       : float = 0.12

    def __print__(self):
        print("================ GRIST Static Data Reader Config ==================")
        print    (f"  use_zs_grad_filter         = {self.use_zs_grad_filter      }")
        if self.zs_grad_filter_type == "laplacian1":
            print(f"  zs_grad_filter_type        = {self.zs_grad_filter_type     }")
            print(f"  zs_grad_filter_order       = {self.zs_grad_filter_order    }")
            print(f"  zs_grad_filter_coef        = {self.zs_grad_filter_coef     }")
            print(f"  zs_grad_filter_cycles      = {self.zs_grad_filter_cycles   }")
            print(f"  topo_max_slope             = {self.topo_max_slope          }")
        elif self.zs_grad_filter_type == "laplacian2":
            print(f"  zs_grad_filter_type        = {self.zs_grad_filter_type     }")
            print(f"  zs_grad_filter_cycles      = {self.zs_grad_filter_cycles   }")
        print("===================================================================")


@dataclass
class GRISTBackgroundReaderConfig(AbstractReaderConfig):
    # "jax" or "native"
    vert_interp_backend: str = "jax"

    def __print__(self):
        print("================ GRIST Background Reader Config ==================")
        print(f"  vert_interp_backend        = {self.vert_interp_backend}")
        print("==================================================================")


class GRISTStaticDataReader(AbstractReader[GRISTStaticDataReaderConfig]):
    lon: Float[Array, "ncol"]
    lat: Float[Array, "ncol"]
    phis: Float[Array, "ncol"]
    lu_index: Float[Array, "ncol"]
    lu_index_frac: Float[Array, "lu_cat"]
    soiltop: Float[Array, "ncol"]
    soiltop_frac: Float[Array, "ncol"]
    mask: Float[Array, "ncol"]
    soiltemp: Float[Array, "ncol"]
    snoalb: Float[Array, "ncol"]
    greenfrac: Float[Array, "ncol month"]
    albedo: Float[Array, "ncol month"]
    varsso: Float[Array, "ncol"]
    con: Float[Array, "ncol"]
    oa1: Float[Array, "ncol"]
    oa2: Float[Array, "ncol"]
    oa3: Float[Array, "ncol"]
    oa4: Float[Array, "ncol"]
    ol1: Float[Array, "ncol"]
    ol2: Float[Array, "ncol"]
    ol3: Float[Array, "ncol"]
    ol4: Float[Array, "ncol"]
    var2d: Float[Array, "ncol"]

    def __init__(
        self,
        static_file: str,
        para: PlanetParameterSet,
        dev: AbstractDevice,
        cfg: GRISTStaticDataReaderConfig = GRISTStaticDataReaderConfig()
    ):
        super().__init__(
            para=para,
            cfg=cfg,
            dev=dev
        )
        self.static_file = static_file
        if dev.is_root(): log.logger.info(f"Use GRIST static file {static_file} as static data source.")
    
    @property
    def reader_type(self) -> str:
        return "grist"
    
    @property
    def lon_range(self) -> str:
        return "-180>180"
    
    def load_data(self):
        float_type = self.dev.float_type
        ds = xr.open_dataset(self.static_file, engine="netcdf4")
        self.lon = jnp.array(
            ds["lon"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.lat = jnp.array(
            ds["lat"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.phis = jnp.array(
            ds["PHIS"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.lu_index = jnp.array(
            ds["LU_INDEX"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.lu_index_frac = jnp.array(
            ds["LU_INDEX_FRAC"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.soiltop = jnp.array(
            ds["SOILTOP"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.soiltop_frac = jnp.array(
            ds["SOILTOP_FRAC"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.mask = jnp.array(
            ds["MASK"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.soiltemp = jnp.array(
            ds["SOILTEMP"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.snoalb = jnp.array(
            ds["SNOALB"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.greenfrac = jnp.array(
            ds["GREENFRAC"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.albedo = jnp.array(
            ds["ALBEDO"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.varsso = jnp.array(
            ds["VARSSO"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.con = jnp.array(
            ds["CON"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.oa1 = jnp.array(
            ds["OA1"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.oa2 = jnp.array(
            ds["OA2"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.oa3 = jnp.array(
            ds["OA3"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.oa4 = jnp.array(
            ds["OA4"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.ol1 = jnp.array(
            ds["OL1"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.ol2 = jnp.array(
            ds["OL2"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.ol3 = jnp.array(
            ds["OL3"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.ol4 = jnp.array(
            ds["OL4"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.var2d = jnp.array(
            ds["VAR2D"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )

    def unload(self):
        delattr(self, "lon"          )
        delattr(self, "lat"          )
        delattr(self, "phis"         )
        delattr(self, "lu_index"     )
        delattr(self, "lu_index_frac")
        delattr(self, "soiltop"      )
        delattr(self, "soiltop_frac" )
        delattr(self, "mask"         )
        delattr(self, "soiltemp"     )
        delattr(self, "snoalb"       )
        delattr(self, "greenfrac"    )
        delattr(self, "albedo"       )
        delattr(self, "varsso"       )
        delattr(self, "con"          )
        delattr(self, "oa1"          )
        delattr(self, "oa2"          )
        delattr(self, "oa3"          )
        delattr(self, "oa4"          )
        delattr(self, "ol1"          )
        delattr(self, "ol2"          )
        delattr(self, "ol3"          )
        delattr(self, "ol4"          )
        delattr(self, "var2d"        )
        self.loaded = False
        gc.collect
        jax.clear_caches()


class GRISTBackgroundReader(AbstractReader[GRISTBackgroundReaderConfig]):
    time_str: str
    lon: Float[Array, "ncells"]
    lat: Float[Array, "ncells"]
    plev: Float[Array, "plev"]
    ps: Float[Array, "ncells"]
    q: Float[Array, "ncells plev"]
    skintemp: Float[Array, "ncells"]
    snow: Float[Array, "ncells"]
    snowh: Float[Array, "ncells"]
    soilh: Float[Array, "ncells"]
    t: Float[Array, "ncells plev"]
    u: Float[Array, "ncells plev"]
    v: Float[Array, "ncells plev"]
    xice: Float[Array, "ncells"]
    soiltemp: Float[Array, "ncells nsoil_levels"]
    nsoil_levels: Float[Array, "nsoil_levels"]
    soilmoist: Float[Array, "ncells nsoil_levels"]

    def __init__(
        self,
        bkg_file: str,
        para: PlanetParameterSet,
        dev: AbstractDevice,
        cfg: GRISTBackgroundReaderConfig = GRISTBackgroundReaderConfig(),
    ):
        super().__init__(
            para=para,
            cfg=cfg,
            dev=dev
        )
        self.bkg_file = bkg_file
        if dev.is_root(): log.logger.info(f"Use GRIST background file {bkg_file} as background data source.")
    
    @property
    def reader_type(self) -> str:
        return "grist"
    
    @property
    def lon_range(self) -> str:
        return "-180>180"
     
    def load_data(self):
        float_type = self.dev.float_type
        ds = xr.open_dataset(self.bkg_file, engine="netcdf4")
        # ------------- Load data -------------
        self.lon = jnp.array(
            ds["lon"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.lat = jnp.array(
            ds["lat"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.plev = jnp.array(
            ds["plev"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.time_str = str(ds["time"].values)
        self.ps = jnp.array(
            ds["PS"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.q = jnp.array(
            ds["Q"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        ).T
        self.skintemp = jnp.array(
            ds["SKINTEMP"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.snow = jnp.array(
            ds["SNOW"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.snowh = jnp.array(
            ds["SNOWH"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.soilh = jnp.array(
            ds["SOILH"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.t = jnp.array(
            ds["T"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        ).T
        self.u = jnp.array(
            ds["U"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        ).T
        self.v = jnp.array(
            ds["V"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        ).T
        self.xice = jnp.array(
            ds["XICE"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        )
        self.soiltemp = jnp.array(
            ds["SoilTemp"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        ).T
        self.nsoil_levels = jnp.array(
            ds["nSoilLevels"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        ).T
        self.soilmoist = jnp.array(
            ds["SoilMoist"].values,
            device=jax.local_devices(backend="cpu")[0],
            dtype=float_type
        ).T

    def unload(self):
        delattr(self, "time_str"    )
        delattr(self, "lon"         )
        delattr(self, "lat"         )
        delattr(self, "plev"        )
        delattr(self, "ps"          )
        delattr(self, "q"           )
        delattr(self, "skintemp"    )
        delattr(self, "snow"        )
        delattr(self, "snowh"       )
        delattr(self, "soilh"       )
        delattr(self, "t"           )
        delattr(self, "u"           )
        delattr(self, "v"           )
        delattr(self, "xice"        )
        delattr(self, "soiltemp"    )
        delattr(self, "nsoil_levels")
        delattr(self, "soilmoist"   )
        self.loaded = False
        gc.collect
        jax.clear_caches()
