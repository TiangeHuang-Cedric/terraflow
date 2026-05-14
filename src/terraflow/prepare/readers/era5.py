from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr

from terraflow.util import typing, log, formula
from terraflow.prepare.readers.abstract import AbstractBackgroundReader

if TYPE_CHECKING:
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
        hori_interp_method: str = "bilinear",
        hori_extrap: bool = True
    ):
        log.logger.info(f"Use ERA5 {bkg_file} as background")
        super().__init__(
            bkg_file=bkg_file,
            para=para,
            hori_interp_method=hori_interp_method,
            hori_extrap=hori_extrap,
        )

    def extract_data(self):
        # ===================================================
        # Load data
        float_type = self._para.dev.float_type
        ds = xr.open_dataset(self.bkg_file, engine="netcdf4")
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
        self.lev = jnp.array(
            ds[self.level_name].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
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
            ds["sp"].values[0, :, :],
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.zs = jnp.array(
            ds["zs"].values[0, :, :],
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
        if self.lev[0] > self.lev[1]:
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
        g = self._para.gravity.unwrap()
        self.lev = self.lev * 100.0
        self.z = self.z / g
        self.zs = self.zs / g
        # ===================================================
        # Initialize missing variables
        if not hasattr(self, "qc"):
            self.qc = jnp.zeros(
                (self.lon.shape[0], self.lat.shape[0], self.lev.shape[0]),
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        if not hasattr(self, "qi"):
            self.qi = jnp.zeros(
                (self.lon.shape[0], self.lat.shape[0], self.lev.shape[0]),
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        if not hasattr(self, "qr"):
            self.qr = jnp.zeros(
                (self.lon.shape[0], self.lat.shape[0], self.lev.shape[0]),
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        if not hasattr(self, "qs"):
            self.qs = jnp.zeros(
                (self.lon.shape[0], self.lat.shape[0], self.lev.shape[0]),
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        # ===================================================
        # Calculate virtual temperature
        self.tv = formula.virtual_temperature_from_wet_mixing_ratio(
            t=self.t,
            qv=self.qv,
            para=self._para
        )

    @property
    def reader_type(self) -> str:
        return "era5"

    def flip_latitude(self, data: np.ndarray) -> np.ndarray:
        return np.flip(data, axis=1)
