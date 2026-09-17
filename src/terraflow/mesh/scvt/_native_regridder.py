from __future__ import annotations

from numba import UnsupportedError

from terraflow.util import typing
from terraflow.util.backend.abstract import AbstractBackend
from terraflow.mesh.scvt._native_interp import (
    latlon_interp_bilinear_apply,
    latlon_interp_bilinear_column,
    latlon_interp_bilinear_init,
)

Float = typing.Float
Array = typing.Array


class NativeRegridder:
    """Native regridder.

    Only bilinear is supported.
    """

    def __init__(
        self,
        src_lon: Float[Array, "nlon"],
        src_lat: Float[Array, "nlat"],
        dst_lon: Float[Array, "npoint_dst"],
        dst_lat: Float[Array, "npoint_dst"],
        method: str = "bilinear",
        *,
        extrap: bool = True,
        precompute_weights: bool = True,
        backend: AbstractBackend | None = None,
    ):
        self.src_lon = src_lon
        self.src_lat = src_lat
        self.dst_lon = dst_lon
        self.dst_lat = dst_lat
        self.extrap = extrap
        self.backend = backend

        method_norm = method.strip().lower()
        if method_norm != "bilinear":
            raise ValueError(
                "Native backend only supports bilinear. "
                f"Got method={method!r}."
            )

        self._cache: dict[str, object] | None = None
        if precompute_weights:
            self._cache = latlon_interp_bilinear_init(
                src_lon=self.src_lon,
                src_lat=self.src_lat,
                dst_lon=self.dst_lon,
                dst_lat=self.dst_lat,
                extrap=self.extrap,
                bkd=self.backend,
            )

    def regrid(self, src_data: Float[Array, "nlev nlat nlon"]):
        if self._cache is None:
            return latlon_interp_bilinear_column(
                src_lon=self.src_lon,
                src_lat=self.src_lat,
                src_data=src_data,
                dst_lon=self.dst_lon,
                dst_lat=self.dst_lat,
                extrap=self.extrap,
                bkd=self.backend,
            )
        return latlon_interp_bilinear_apply(src_data, self._cache, bkd=self.backend)

    def __call__(self, src_data: Float[Array, "nlev nlat nlon"]):
        return self.regrid(src_data)
