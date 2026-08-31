from __future__ import annotations
from typing import TYPE_CHECKING
import gc
import jax
import jax.numpy as jnp


from terraflow.util import typing, log, formula
from terraflow.prepare.readers.abstract import AbstractBackgroundReader, BackgroundReaderConfig


if TYPE_CHECKING:
    from terraflow.util.device import AbstractDevice
    from terraflow.util.planet import PlanetParameterSet


Float = typing.Float
Array = typing.Array


class ERA5Reader(AbstractBackgroundReader):
    time_name: str
    time_str: str
    level_name: str
    lon: Float[Array, "nlon"]
    lat: Float[Array, "nlat"]
    lev: Float[Array, "nlev"]
    plev: Float[Array, "nlev"]
    u: Float[Array, "nlon nlat nlev"]
    v: Float[Array, "nlon nlat nlev"]
    t: Float[Array, "nlon nlat nlev"]
    tv: Float[Array, "nlon nlat nlev"]
    z: Float[Array, "nlon nlat nlev"]
    qv: Float[Array, "nlon nlat nlev"]
    qc: Float[Array, "nlon nlat nlev"]
    qi: Float[Array, "nlon nlat nlev"]
    qr: Float[Array, "nlon nlat nlev"]
    qs: Float[Array, "nlon nlat nlev"]
    ps: Float[Array, "nlon nlat"]
    zs: Float[Array, "nlon nlat"]

    def __init__(
        self,
        bkg_file: str,
        para: PlanetParameterSet,
        dev: AbstractDevice,
        cfg: BackgroundReaderConfig = BackgroundReaderConfig()
    ):
        if dev.is_root(): log.logger.info(f"Use ERA5 {bkg_file} as background.")
        super().__init__(
            bkg_file=bkg_file,
            para=para,
            dev=dev,
            cfg=cfg
        )

    @property
    def reader_type(self) -> str:
        return "era5"

    @property
    def lon_range(self) -> str:
        return "0>360"

    def load_data(
        self,
        min_lon: float = 0,
        min_lat: float = -90.0,
        max_lon: float = 360.0,
        max_lat: float = 90
    ):
        # ===================================================
        # Load data
        float_type = self.dev.float_type
        ds0 = self.open_dataset()
        ds = self.input_range(
            ds0,
            lon_name="longitude",
            lat_name="latitude",
            lon_span=(0.0, 360.0),
            min_lon=min_lon,
            max_lon=max_lon,
            min_lat=max(float(min_lat), -89.75),
            max_lat=min(float(max_lat), 89.75),
        )
        ds0.close()
        self.check_input_range(
            "ERA5",
            ds["longitude"].values,
            ds["latitude"].values,
        )
        if "level" in ds:
            self.level_name = "level"
        elif "pressure_level" in ds:
            self.level_name = "pressure_level"
        else:
            raise ValueError(
                f"No valid level information in ERA5 file '{self.bkg_file}'!"
            )
        if "time" in ds:
            self.time_name = "time"
        elif "valid_time" in ds:
            self.time_name = "valid_time"
        else:
            raise ValueError(
                f"No valid time information in ERA5 file '{self.bkg_file}'!"
            )
        self.time_str = str(ds[self.time_name].values[0])
        self.lon = jnp.array(
            ds["longitude"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.lat = jnp.array(
            ds["latitude"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.plev = jnp.array(
            ds[self.level_name].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.lev = self.plev
        self.u = jnp.array(
            ds["u"].values[0, :, :, :],
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.v = jnp.array(
            ds["v"].values[0, :, :, :],
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.t = jnp.array(
            ds["t"].values[0, :, :, :],
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.z = jnp.array(
            ds["z"].values[0, :, :, :],
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.qv = jnp.array(
            ds["q"].values[0, :, :, :],
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.ps = jnp.array(
            ds["sp"].values[0, :, :],
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.zs = jnp.array(
            ds["zs"].values[0, :, :],
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        if "clwc" in ds:
            self.qc = jnp.array(
                ds["clwc"].values[0, :, :, :],
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        if "ciwc" in ds:
            self.qi = jnp.array(
                ds["ciwc"].values[0, :, :, :],
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        if "crwc" in ds:
            self.qr = jnp.array(
                ds["crwc"].values[0, :, :, :],
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        if "cswc" in ds:
            self.qs = jnp.array(
                ds["cswc"].values[0, :, :, :],
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        ds.close()
        # ===================================================
        # Reverse vertical order
        if self.plev[0] > self.plev[1]:
            self.plev = self.plev[::-1]
            self.lev = self.lev[::-1]
            self.u = self.u[::-1, :, :]
            self.v = self.v[::-1, :, :]
            self.t = self.t[::-1, :, :]
            self.z = self.z[::-1, :, :]
            self.qv = self.qv[::-1, :, :]
            if hasattr(self, "qc"):
                self.qc = self.qc[::-1, :, :]
            if hasattr(self, "qi"):
                self.qi = self.qi[::-1, :, :]
            if hasattr(self, "qr"):
                self.qr = self.qr[::-1, :, :]
            if hasattr(self, "qs"):
                self.qs = self.qs[::-1, :, :]
        # ===================================================
        # Set u and v to zero at the poles
        self.u = jnp.where(
            jnp.abs(jnp.abs(self.lat[None, :, None]) - 90) < 1e-10,
            0.0,
            self.u
        )
        self.v = jnp.where(
            jnp.abs(jnp.abs(self.lat[None, :, None]) - 90) < 1e-10,
            0.0,
            self.v
        )
        # ===================================================
        # Change units
        g = self.para.gravity.unwrap()
        self.plev = self.plev * 100.0
        self.lev = self.lev * 100.0
        self.z = self.z / g
        self.zs = self.zs / g
        # ===================================================
        # Initialize missing variables
        if not hasattr(self, "qc"):
            self.qc = jnp.zeros(
                (self.plev.shape[0], self.lat.shape[0], self.lon.shape[0]),
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        if not hasattr(self, "qi"):
            self.qi = jnp.zeros(
                (self.plev.shape[0], self.lat.shape[0], self.lon.shape[0]),
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        if not hasattr(self, "qr"):
            self.qr = jnp.zeros(
                (self.plev.shape[0], self.lat.shape[0], self.lon.shape[0]),
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        if not hasattr(self, "qs"):
            self.qs = jnp.zeros(
                (self.plev.shape[0], self.lat.shape[0], self.lon.shape[0]),
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        # ===================================================
        # Calculate virtual temperature
        self.tv = formula.virtual_temperature_from_wet_mixing_ratio(
            t=self.t,
            qv=self.qv,
            para=self.para
        )
        self.loaded = True

    def unload(self):
        delattr(self, "time_name" )
        delattr(self, "time_str"  )
        delattr(self, "level_name")
        delattr(self, "lon"       )
        delattr(self, "lat"       )
        delattr(self, "lev"       )
        delattr(self, "plev"      )
        delattr(self, "u"         )
        delattr(self, "v"         )
        delattr(self, "t"         )
        delattr(self, "tv"        )
        delattr(self, "z"         )
        delattr(self, "qv"        )
        delattr(self, "qc"        )
        delattr(self, "qi"        )
        delattr(self, "qr"        )
        delattr(self, "qs"        )
        delattr(self, "ps"        )
        delattr(self, "zs"        )
        self.loaded = False
        gc.collect()
        jax.clear_caches()
