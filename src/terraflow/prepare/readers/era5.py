from __future__ import annotations
from typing import TYPE_CHECKING
import gc
import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr


from terraflow.util import typing, log, formula
from terraflow.prepare.readers.abstract import AbstractBackgroundReader, BackgroundReaderConfig

if TYPE_CHECKING:
    from terraflow.util.device import AbstractDevice
    from terraflow.planet import PlanetParameterSet

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
        lon_values = np.asarray(ds0["longitude"].values, dtype=np.float64)
        lat_values = np.asarray(ds0["latitude"].values, dtype=np.float64)
        lon_min = float(np.min(lon_values))
        lon_max = float(np.max(lon_values))
        lat_min = max(float(min_lat), -89.75, float(np.min(lat_values)))
        lat_max = min(float(max_lat),  89.75, float(np.max(lat_values)))
        lon0 = float(min_lon)
        lon1 = float(max_lon)

        if self.cfg.hori_interp_backend == "native":
            dlon = float(np.min(np.abs(np.diff(np.sort(lon_values)))))
            dlat = float(np.min(np.abs(np.diff(np.sort(lat_values)))))
            lon0 -= dlon
            lon1 += dlon
            lat_min = max(lat_min - dlat, -89.75, float(np.min(lat_values)))
            lat_max = min(lat_max + dlat,  89.75, float(np.max(lat_values)))

        lon_width = lon1 - lon0 if lon1 >= lon0 else lon1 + 360.0 - lon0
        if lon_width >= 360.0 - 1e-10:
            ds = ds0.sel(
                longitude=slice(lon_min, lon_max),
                latitude=slice(lat_max, lat_min)
            )
        else:
            lon0_norm = (lon0 - lon_min) % 360.0 + lon_min
            lon1_norm = (lon1 - lon_min) % 360.0 + lon_min
            if lon0_norm <= lon1_norm:
                ds = ds0.sel(
                    longitude=slice(lon0_norm, lon1_norm),
                    latitude=slice(lat_max, lat_min)
                )
            else:
                ds_west = ds0.sel(
                    longitude=slice(lon0_norm, lon_max),
                    latitude=slice(lat_max, lat_min)
                )
                ds_east = ds0.sel(
                    longitude=slice(lon_min, lon1_norm),
                    latitude=slice(lat_max, lat_min)
                )
                ds_east = ds_east.assign_coords(
                    longitude=ds_east["longitude"] + 360.0
                )
                ds = xr.concat([ds_west, ds_east], dim="longitude")
        ds = ds.load()
        ds0.close()

        if ds.sizes["longitude"] < 2 or ds.sizes["latitude"] < 2:
            raise ValueError(
                "ERA5 bbox selection is too small for bilinear interpolation: "
                f"longitude={ds.sizes['longitude']}, latitude={ds.sizes['latitude']}"
            )
        if "level" in ds.keys():
            self.level_name = "level"
        elif "pressure_level" in ds.keys():
            self.level_name = "pressure_level"
        else:
            raise ValueError(
                f"No valid level information in ERA5 file '{self.bkg_file}'!"
            )
        if "time" in ds.keys():
            self.time_name = "time"
        elif "valid_time" in ds.keys():
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
            np.flip(ds["latitude"].values),
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
            self.flip_latitude(ds["u"].values[0, :, :, :]),
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.v = jnp.array(
            self.flip_latitude(ds["v"].values[0, :, :, :]),
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.t = jnp.array(
            self.flip_latitude(ds["t"].values[0, :, :, :]),
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.z = jnp.array(
            self.flip_latitude(ds["z"].values[0, :, :, :]),
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.qv = jnp.array(
            self.flip_latitude(ds["q"].values[0, :, :, :]),
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.ps = jnp.array(
            self.flip_latitude(ds["sp"].values[0, :, :]),
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.zs = jnp.array(
            self.flip_latitude(ds["zs"].values[0, :, :]),
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        if "clwc" in ds.keys():
            self.qc = jnp.array(
                self.flip_latitude(ds["clwc"].values[0, :, :, :]),
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        if "ciwc" in ds.keys():
            self.qi = jnp.array(
                self.flip_latitude(ds["ciwc"].values[0, :, :, :]),
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        if "crwc" in ds.keys():
            self.qr = jnp.array(
                self.flip_latitude(ds["crwc"].values[0, :, :, :]),
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        if "cswc" in ds.keys():
            self.qs = jnp.array(
                self.flip_latitude(ds["cswc"].values[0, :, :, :]),
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

    def flip_latitude(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 3 and data.shape == (self.lev.shape[0], self.lat.shape[0], self.lon.shape[0]):
            # (nlev, nlat, nlon)
            return np.flip(data, axis=1)
        elif data.ndim == 2 and data.shape == (self.lat.shape[0], self.lon.shape[0]):
            # (nlat, nlon)
            return np.flip(data, axis=0)
        else:
            raise ValueError(
                f"Unsupported data shape for latitude flipping: {data.shape}"
            )

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
