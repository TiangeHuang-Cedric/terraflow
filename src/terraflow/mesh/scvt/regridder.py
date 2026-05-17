from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union
import numpy as np

from terraflow.mesh.scvt._native_regridder import NativeRegridder
from terraflow.mesh.scvt._esmf_regridder import ESMFRegridder
from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations as SCVT_MESH

if TYPE_CHECKING:
    from terraflow.util import typing

    Float = typing.Float
    Array = typing.Array

    GRID = Union["LatlonRegridInfo", "SCVTRegridInfo"]


@dataclass(frozen=True)
class LatlonRegridInfo:
    lon: Float[Array, "nlon"]
    lat: Float[Array, "nlat"]


@dataclass(frozen=True)
class SCVTRegridInfo:
    mesh: SCVT_MESH
    loc: str
    entity: str

    def get_coord(self) -> dict:
        if self.entity == "primal":
            if self.loc == "cell":
                return {
                    "lon": self.mesh.primal.cell.centroid.lon,
                    "lat": self.mesh.primal.cell.centroid.lat
                }
            elif self.loc == "node":
                return {
                    "lon": self.mesh.primal.node.coordinate.lon,
                    "lat": self.mesh.primal.node.coordinate.lat
                }
            elif self.loc == "edge":
                return {
                    "lon": self.mesh.primal.edge.midpoint.lon,
                    "lat": self.mesh.primal.edge.midpoint.lat
                }
            else:
                raise ValueError(f"Invalid loc for primal: {self.loc}")
        elif self.entity == "dual":
            if self.loc == "cell":
                return {
                    "lon": self.mesh.dual.cell.centroid.lon,
                    "lat": self.mesh.dual.cell.centroid.lat
                }
            elif self.loc == "node":
                return {
                    "lon": self.mesh.dual.node.coordinate.lon,
                    "lat": self.mesh.dual.node.coordinate.lat
                }
            elif self.loc == "edge":
                return {
                    "lon": self.mesh.dual.edge.midpoint.lon,
                    "lat": self.mesh.dual.edge.midpoint.lat
                }
            else:
                raise ValueError(f"Invalid loc for dual: {self.loc}")
        else:
            raise ValueError(f"Invalid entity: {self.entity}")


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
        extrap: bool = False,
        conserve_opt: dict[str, Any] = {"norm_opt": "destarea"},
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
        self.conserve_opt = conserve_opt
        self.precompute_weights = precompute_weights
        if self.backend == "native":
            if not isinstance(src, LatlonRegridInfo) or not isinstance(dst, SCVTRegridInfo):
                raise ValueError(
                    f"Native backend only supports latlon -> scvt regridding currently, got "
                    f"{src_type} -> {dst_type}"
                )
            src_info = self.create_latlon_info(lon=src.lon, lat=src.lat)
            dst_info = self.create_scvt_info(
                mesh=dst.mesh,
                loc=dst.loc,
                entity=dst.entity
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
            conservative_methods = {"conserve_1st", "conserve_2nd"}
            # ================================================================
            # src (latlon) -> dst (scvt)
            if isinstance(src, LatlonRegridInfo):
                esmf_src = ESMFRegridder.create_latlon_grid(
                    lon=np.array(src.lon),
                    lat=np.array(src.lat)
                )
            elif isinstance(src, SCVTRegridInfo):
                if src.loc == "cell" or src.loc == "node":
                    esmf_src = ESMFRegridder.create_scvt_mesh(
                        mesh=src.mesh,
                        loc=src.loc,
                        entity=src.entity
                    )
                elif src.loc == "edge":
                    esmf_src = ESMFRegridder.create_locstream(
                        mesh=src.mesh,
                        loc=src.loc,
                        entity=src.entity
                    )
                else:
                    raise ValueError(
                        f"Invalid loc for ESMF backend: {src.loc}"
                    )
            else:
                raise ValueError(
                    f"Unsupported src grid type for ESMF backend: {type(src)}"
                )
            # ================================================================
            # src (scvt) -> dst (latlon)
            if isinstance(dst, LatlonRegridInfo):
                esmf_dst = ESMFRegridder.create_latlon_grid(
                    lon=np.array(dst.lon),
                    lat=np.array(dst.lat)
                )
            elif isinstance(dst, SCVTRegridInfo):
                if dst.loc == "cell" or dst.loc == "node":
                    esmf_dst = ESMFRegridder.create_scvt_mesh(
                        mesh=dst.mesh,
                        loc=dst.loc,
                        entity=dst.entity
                    )
                elif dst.loc == "edge":
                    if self.method in conservative_methods:
                        raise ValueError(
                            f"Conservative methods are not supported for edge -> latlon regridding"
                        )
                    esmf_dst = ESMFRegridder.create_locstream(
                        mesh=dst.mesh,
                        loc=dst.loc,
                        entity=dst.entity
                    )
                else:
                    raise ValueError(
                        f"Invalid loc for ESMF backend: {dst.loc}"
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
                conserve_opt=self.conserve_opt,
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
    def create_scvt_info(mesh: SCVT_MESH, loc: str, entity: str) -> SCVTRegridInfo:
        return SCVTRegridInfo(mesh=mesh, loc=loc, entity=entity)
