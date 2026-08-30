from __future__ import annotations
from typing import TYPE_CHECKING
import gc
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
        if dev.is_root(): log.logger.info(f"Use ETOPO1 {topo_file} as topography.")
        super().__init__(
            topo_file=topo_file,
            para=para,
            cfg=cfg,
            dev=dev
        )
        
    @property
    def reader_type(self) -> str:
        return "etopo1"
    
    @property
    def lon_range(self) -> str:
        return "-180>180"
    
    def load_data(
        self, 
        min_lon: float = -180.0,
        min_lat: float = -90.0, 
        max_lon: float = 180.0,
        max_lat: float = 90
    ):
        float_type = self.dev.float_type
        gravity = self.para.gravity.unwrap()
        if min_lon <= max_lon:
            ds = self.open_dataset().sel(
                x=slice(min_lon, max_lon), 
                y=slice(min_lat, max_lat)
            )
        else:
            ds0 = self.open_dataset()
            ds1 = ds0.sel(
                x=slice(min_lon, 180), 
                y=slice(min_lat, max_lat)
            )
            ds2 = ds0.sel(
                x=slice(-180, max_lon), 
                y=slice(min_lat, max_lat)
            )
            ds = xr.concat([ds1, ds2], dim="x")
            ds = ds.load()
            ds0.close()
            ds1.close()
            ds2.close()
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
        ds.close()
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
        if self.cfg.fill_split_with_chunks:
            if self.lon.shape[0] < self.cfg.fill_lon_chunk_size:
                raise ValueError(
                    f"Chunk size for filling topo is larger than the topo grid size! "
                    f"Please reduce fill_lon_chunk_size in TopographyReaderConfig."
                )
            if self.lat.shape[0] < self.cfg.fill_lat_chunk_size:
                raise ValueError(
                    f"Chunk size for filling topo is larger than the topo grid size! "
                    f"Please reduce fill_lat_chunk_size in TopographyReaderConfig."
                )
        self.loaded = True
        
    def unload(self):
        delattr(self, "lon" )
        delattr(self, "lat" )
        delattr(self, "gzs" )
        delattr(self, "mask")
        self.loaded = False
        gc.collect
        jax.clear_caches()


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
        if dev.is_root(): log.logger.info(f"Use GMETED2010 {topo_file} as topography.")
        super().__init__(
            topo_file=topo_file,
            para=para,
            dev=dev,
            cfg=cfg
        )
    
    @property
    def reader_type(self) -> str:
        return "gmeted2010"
        
    @property
    def lon_range(self) -> str:
        return "0>360"
    
    def load_data(
        self,
        min_lon: float = 0.0,
        min_lat: float = -90.0, 
        max_lon: float = 360.0,
        max_lat: float = 90
    ):
        float_type = self.dev.float_type
        gravity = self.para.gravity.unwrap()
        if min_lon <= max_lon:
            ds = self.open_dataset().sel(
                lon=slice(min_lon, max_lon), 
                lat=slice(min_lat, max_lat)
            )
        else:
            ds0 = self.open_dataset()
            ds1 = ds0.sel(
                lon=slice(min_lon, 360), 
                lat=slice(min_lat, max_lat)
            )
            ds2 = ds0.sel(
                lon=slice(0, max_lon), 
                lat=slice(min_lat, max_lat)
            )
            ds = xr.concat([ds1, ds2], dim="lon")
            ds = ds.load()
            ds0.close()
            ds1.close()
            ds2.close()
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
        ds.close()
        if self.cfg.fill_split_with_chunks:
            if self.lon.shape[0] < self.cfg.fill_lon_chunk_size:
                raise ValueError(
                    f"Chunk size for filling topo is larger than the topo grid size! "
                    f"Please reduce fill_lon_chunk_size in TopographyReaderConfig."
                )
            if self.lat.shape[0] < self.cfg.fill_lat_chunk_size:
                raise ValueError(
                    f"Chunk size for filling topo is larger than the topo grid size! "
                    f"Please reduce fill_lat_chunk_size in TopographyReaderConfig."
                )
        self.loaded = True

    def unload(self):
        delattr(self, "lon" )
        delattr(self, "lat" )
        delattr(self, "gzs" )
        self.loaded = False
        gc.collect
        jax.clear_caches()
