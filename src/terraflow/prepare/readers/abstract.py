from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import os
import xarray as xr

if TYPE_CHECKING:
    from terraflow.planet import PlanetParameterSet
    from terraflow.util import typing

    Float = typing.Float
    Array = typing.Array


class AbstractReader(ABC):
    lon: Float[Array, "nlon"]
    lat: Float[Array, "nlat"]

    def __init__(
        self,
        para: PlanetParameterSet,
        hori_interp_method: str = "bilinear",
        hori_extrap: bool = True
    ):
        self._para = para
        self.hori_interp_method = hori_interp_method
        self.hori_extrap = hori_extrap

    @property
    @abstractmethod
    def reader_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def extract_data(self):
        raise NotImplementedError


class AbstractTopographyReader(AbstractReader):
    topo_file: str
    lon: Float[Array, "nlon nlat"]
    lat: Float[Array, "nlon nlat"]
    gzs: Float[Array, "nlon nlat"]
    mask: Float[Array, "nlon nlat"]

    def __init__(
        self,
        topo_file: str,
        para: PlanetParameterSet,
        hori_interp_method: str = "bilinear",
        hori_extrap: bool = True
    ):
        super().__init__(
            para=para,
            hori_interp_method=hori_interp_method,
            hori_extrap=hori_extrap
        )
        if not os.path.exists(topo_file):
            raise FileNotFoundError(
                f"Topography file '{topo_file}' not found!"
            )
        self.topo_file = topo_file
        self.extract_data()

    def open_dataset(self) -> xr.Dataset:
        return xr.open_dataset(self.topo_file)


class AbstractBackgroundReader(AbstractReader):
    bkg_file: str
    time_str: str

    def __init__(
        self,
        bkg_file: str,
        para: PlanetParameterSet,
        hori_interp_method: str = "bilinear",
        hori_extrap: bool = True
    ):
        super().__init__(
            para=para,
            hori_interp_method=hori_interp_method,
            hori_extrap=hori_extrap
        )
        if not os.path.exists(bkg_file):
            raise FileNotFoundError(
                f"Background file '{bkg_file}' not found!"
            )
        self.bkg_file = bkg_file
        self.extract_data()

    def open_dataset(self) -> xr.Dataset:
        return xr.open_dataset(self.bkg_file)
