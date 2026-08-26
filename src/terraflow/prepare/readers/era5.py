from __future__ import annotations
from typing import TYPE_CHECKING
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
    nlev_eff: Float[Array, "ncol"]
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
        log.logger.info(f"Use ERA5 {bkg_file} as background.")
        super().__init__(
            bkg_file=bkg_file,
            para=para,
            dev=dev,
            cfg=cfg
        )

    def extract_data(self):
        # ===================================================
        # Load data
        float_type = self.dev.float_type
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
                (self.lon.shape[0], self.lat.shape[0], self.plev.shape[0]),
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        if not hasattr(self, "qi"):
            self.qi = jnp.zeros(
                (self.lon.shape[0], self.lat.shape[0], self.plev.shape[0]),
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        if not hasattr(self, "qr"):
            self.qr = jnp.zeros(
                (self.lon.shape[0], self.lat.shape[0], self.plev.shape[0]),
                device=jax.devices("cpu")[0],
                dtype=float_type
            )
        if not hasattr(self, "qs"):
            self.qs = jnp.zeros(
                (self.lon.shape[0], self.lat.shape[0], self.plev.shape[0]),
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

    def grist_post_process(
        self,
        ps: Float[Array, "ncol"],
        qv: Float[Array, "nlev ncol"],
        qc: Float[Array, "nlev ncol"],
        qi: Float[Array, "nlev ncol"],
        qr: Float[Array, "nlev ncol"],
        qs: Float[Array, "nlev ncol"],
    ):
        # ------------- GRIST Post process -------------
        plev_in = self.plev
        nlev = plev_in.shape[0]
        ncol = ps.shape[0]

        # Check effective full level number
        mask = plev_in[:, None] >= ps[None, :]
        first_ge_idx = jnp.argmax(mask, axis=0)
        self.nlev_eff = jnp.where(
            jnp.any(mask, axis=0),
            first_ge_idx + 1,
            nlev
        )  # (ncol,)

        # Evaluate dry-mass coordinate at ERA pressure level and surface
        face_idx = jnp.arange(nlev + 1, dtype=jnp.int32)[:, None]  # (nlev+1, 1)
        nlev_eff = self.nlev_eff[None, :]  # (nlev+1, ncol)

        plev_lev = jnp.zeros((nlev + 1, ncol), dtype=plev_in.dtype)
        plev_mid = jnp.pad(
            0.5 * (plev_in[:-1, None] + plev_in[1:, None]),
            ((1, 1), (0, 0)),
            mode='constant'
        )  # (nlev+1, 1)
        plev_lev = jnp.where(
            (face_idx >= 1) & (face_idx < nlev_eff),
            plev_mid,
            plev_lev
        )
        # if ps<pface(nlev_eff), let it be, good assumption! no problem till now...
        plev_lev = jnp.where(
            face_idx == nlev_eff,
            ps[None, :],
            plev_lev
        )

        # hpface(1)=pface(1)=zeros
        # hpface(k)=hpface(k-1)+dp(k-1)*(1-q(k-1)), k=2..nlev_eff+1.
        dp = plev_lev[1:, :] - plev_lev[:-1, :]
        lev_idx = jnp.arange(nlev, dtype=jnp.int32)[:, None]  # (nlev, ncol)
        valid_lev = lev_idx <= nlev_eff - 1  # (nlev, ncol)
        dhp = jnp.where(valid_lev, dp, 0.0)
        qm = qv + qc + qi + qr + qs
        self.plev_lev = jnp.cumulative_sum(
            dhp * (1.0 - qm),
            axis=0,
            include_initial=True
        )  # (nlev+1, ncol)
        self.plev = 0.5 * (self.plev_lev[:-1, :] + self.plev_lev[1:, :])

    @property
    def reader_type(self) -> str:
        return "era5"

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
