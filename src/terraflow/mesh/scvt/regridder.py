from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, Any, Union
import numpy as np

from terraflow.mesh.scvt._native_regridder import NativeRegridder
from terraflow.mesh.scvt._esmf_regridder import ESMFRegridder
from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations as SCVT_MESH

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.layout.abstract import CoordinateSystem, AbstractLayout
    from terraflow.vert_coord.abstract import AbstractVertCoordinate
    from .construct import SphericalCentroidVoronoiTessellations as SCVT_MESH

    Float = typing.Float
    Array = typing.Array

    VM = TypeVar("VM", bound=AbstractVertCoordinate)
    HLO = TypeVar("HLO", bound=AbstractLayout)
    VLO = TypeVar("VLO", bound=AbstractLayout)

    CoordSys = CoordinateSystem[SCVT_MESH, VM, HLO, VLO]

    GRID = Union["LatlonRegridInfo", "SCVTRegridInfo"]


@dataclass(frozen=True)
class LatlonRegridInfo:
    lon: Float[Array, "nlon"]
    lat: Float[Array, "nlat"]


@dataclass(frozen=True)
class SCVTRegridInfo:
    coord: CoordSys
    loc: str

    def get_coord(self) -> dict[str, Float[Array, "npoints"]]:
        return self.coord.mesh.get_coord_info(
            loc=self.loc,
            layout=self.coord.hori_layout.alias_names()
        )


class Regridder:
    """Unified façade of native and ESMF regridders.

    - native backend: bilinear only
    - esmpy backend: bilinear, patch, nearest_stod, nearest_dtos,
      conserve_1st, conserve_2nd
    """

    def __init__(
        self,
        src: GRID,
        dst: GRID,
        *,
        backend: str = "native",
        ndbounds: tuple[int, ...] | None = None,
        method: str = "bilinear",
        extrap: bool = True,
        esmf_extrap_method: str = "none",
        esmf_conserve_opt: dict[str, Any] = {"norm_opt": "fracarea"},
        precompute_weights: bool = True,
    ):
        if isinstance(src, LatlonRegridInfo) and isinstance(dst, LatlonRegridInfo):
            raise ValueError("Latlon -> Latlon remapping is not supported")
        elif isinstance(src, SCVTRegridInfo) and not isinstance(dst, LatlonRegridInfo):
            raise ValueError("SCVT -> SCVT remapping is not supported")
        elif isinstance(src, LatlonRegridInfo) and isinstance(dst, SCVTRegridInfo):
            src_type = "latlon"
            dst_type = "scvt"
        elif isinstance(src, SCVTRegridInfo) and isinstance(dst, LatlonRegridInfo):
            src_type = "scvt"
            dst_type = "latlon"
        else:
            raise ValueError(
                f"Unsupported grid types: {type(src)} -> {type(dst)}"
            )
        self.backend = backend
        self.ndbounds = ndbounds
        self.method = method
        self.extrap = extrap
        self.esmf_extrap_method = esmf_extrap_method
        self.esmf_conserve_opt = esmf_conserve_opt
        self.precompute_weights = precompute_weights
        if self.backend == "native":
            if not isinstance(src, LatlonRegridInfo) or not isinstance(dst, SCVTRegridInfo):
                raise ValueError(
                    f"Native backend only supports latlon -> scvt regridding currently, got "
                    f"{src_type} -> {dst_type}"
                )
            src_info = self.create_latlon_info(lon=src.lon, lat=src.lat)
            dst_info = self.create_scvt_info(
                coord=dst.coord,
                loc=dst.loc,
            )
            dst_coord = dst_info.get_coord()
            self._impl = NativeRegridder(
                src_lon=src_info.lon,
                src_lat=src_info.lat,
                dst_lon=dst_coord["lon"],
                dst_lat=dst_coord["lat"],
                method=method,
                extrap=extrap,
                precompute_weights=precompute_weights,
            )
        elif self.backend == "esmf":
            # ================================================================
            # src (latlon)
            if isinstance(src, LatlonRegridInfo):
                esmf_src = ESMFRegridder.create_latlon_grid(
                    lon=np.array(src.lon),
                    lat=np.array(src.lat)
                )
            # src (scvt)
            elif isinstance(src, SCVTRegridInfo):
                esmf_src = ESMFRegridder.create_scvt_mesh(
                    coord=src.coord,
                    loc=src.loc,
                )
            else:
                raise ValueError(
                    f"Unsupported src grid type for ESMF backend: {type(src)}"
                )
            # ================================================================
            # dst (latlon)
            if isinstance(dst, LatlonRegridInfo):
                esmf_dst = ESMFRegridder.create_latlon_grid(
                    lon=np.array(dst.lon),
                    lat=np.array(dst.lat)
                )
            # dst (scvt)
            elif isinstance(dst, SCVTRegridInfo):
                esmf_dst = ESMFRegridder.create_scvt_mesh(
                    coord=dst.coord,
                    loc=dst.loc,
                )
            else:
                raise ValueError(
                    f"Unsupported dst grid type for ESMF backend: {type(dst)}"
                )
            # ================================================================
            self._impl = ESMFRegridder(
                src=esmf_src,
                dst=esmf_dst,
                method=self.method,
                ndbounds=self.ndbounds,
                extrap=self.extrap,
                extrap_method=self.esmf_extrap_method,
                conserve_opt=self.esmf_conserve_opt,
                precompute_weights=self.precompute_weights,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def __call__(self, src_data: Float[Array, "nlev nlat nlon"]):
        return self.regrid(src_data)

    @staticmethod
    def esmf_backend() -> type[ESMFRegridder]:
        return ESMFRegridder

    @staticmethod
    def native_backend() -> type[NativeRegridder]:
        return NativeRegridder

    def regrid(self, src_data: Float[Array, "nlev nlat nlon"]):
        return self._impl.regrid(src_data)

    @staticmethod
    def create_latlon_info(lon: Float[Array, "nlon"], lat: Float[Array, "nlat"]) -> LatlonRegridInfo:
        return LatlonRegridInfo(lon=lon, lat=lat)

    @staticmethod
    def create_scvt_info(coord: CoordSys, loc: str) -> SCVTRegridInfo:
        return SCVTRegridInfo(coord=coord, loc=loc)
