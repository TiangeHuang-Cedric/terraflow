from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import xarray as xr

from terraflow.util import typing, log
from terraflow.prepare.readers.abstract import AbstractTopographyReader, TopographyReaderConfig

if TYPE_CHECKING:
    from terraflow.util.device import AbstractDevice
    from terraflow.planet import PlanetParameterSet

Float = typing.Float
Array = typing.Array


class ETOPO1Reader(AbstractTopographyReader):

    def __init__(
        self,
        topo_file: str,
        para: PlanetParameterSet,
        dev: AbstractDevice,
        cfg: TopographyReaderConfig = TopographyReaderConfig()
    ):
        if para.planet != "earth":
            raise ValueError(
                f"Topography file '{topo_file}' is used for the Earth!"
            )
        log.logger.info(f"Use ETOPO1 {topo_file} as topography.")
        super().__init__(
            topo_file=topo_file,
            para=para,
            cfg=cfg,
            dev=dev
        )

    def extract_data(self):
        float_type = self.dev.float_type
        gravity = self.para.gravity.unwrap()
        ds = xr.open_dataset(self.topo_file, engine="netcdf4")
        # Load data
        self.lon = jnp.array(
            ds["x"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.lat = jnp.array(
            ds["y"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.gzs = jnp.array(
            ds["z"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        ) * gravity
        self.mask = jnp.array(
            ds["mask"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        # Convert longitude from [-180, 180] to [0, 360]
        self.lon = jnp.where(
            self.lon < 0.0,
            self.lon + 360.0,
            self.lon
        )
        # Sort the longitude and corresponding data to ensure monotonicity
        sort_idx = jnp.argsort(self.lon)
        self.lon = self.lon[sort_idx]
        self.gzs = self.gzs[:, sort_idx]
        self.mask = self.mask[:, sort_idx]

    @property
    def reader_type(self) -> str:
        return "etopo1"


class GMETED2010Reader(AbstractTopographyReader):

    def __init__(
        self,
        topo_file: str,
        para: PlanetParameterSet,
        dev: AbstractDevice,
        cfg: TopographyReaderConfig = TopographyReaderConfig()
    ):
        if para.planet != "earth":
            raise ValueError(
                f"Topography file '{topo_file}' is used for the Earth!"
            )
        log.logger.info(f"Use GMETED2010 {topo_file} as topography.")
        super().__init__(
            topo_file=topo_file,
            para=para,
            dev=dev,
            cfg=cfg
        )

    def extract_data(self):
        float_type = self.dev.float_type
        gravity = self.para.gravity.unwrap()
        ds = xr.open_dataset(self.topo_file, engine="netcdf4")
        self.lon = jnp.array(
            ds["lon"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.lat = jnp.array(
            ds["lat"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.gzs = jnp.array(
            ds["htopo"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        ) * gravity

    @property
    def reader_type(self) -> str:
        return "gmeted2010"
