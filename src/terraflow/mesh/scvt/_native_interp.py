import numpy as np

from terraflow.util import typing
from terraflow.util.backend import infer_backend_from_array
from terraflow.util.backend.abstract import AbstractBackend

Float = typing.Float
Array = typing.Array


# ===================================================
# Shared cache and validation helpers
# ===================================================
def check_strictly_increasing_if_concrete(name: str, x) -> None:
    try:
        x_np = np.asarray(x)
    except Exception:
        return

    if x_np.ndim != 1:
        raise ValueError(f"{name} must be 1D, got ndim={x_np.ndim}")
    if x_np.shape[0] < 2:
        raise ValueError(f"{name} must have at least 2 points")
    if not np.all(np.diff(x_np) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")


def cache_get_int(cache: dict[str, object], key: str) -> int:
    value = cache.get(key)
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise ValueError(f"remap_cache[{key!r}] must be int, got {type(value)}")


def require_backend(*values, bkd: AbstractBackend | None = None) -> AbstractBackend:
    if bkd is not None:
        return bkd
    for value in values:
        inferred = infer_backend_from_array(value)
        if inferred is not None:
            return inferred
    raise ValueError("A backend is required and could not be inferred from the input arrays.")


def cache_get_array(cache: dict[str, object], key: str, bkd: AbstractBackend):
    value = cache.get(key)
    if value is None:
        raise ValueError(f"remap_cache missing key: {key!r}")
    return bkd.asarray(value)


def normalize_lon_to_source(bkd: AbstractBackend, lon, start_lon):
    return bkd.mod(lon - start_lon, 360.0) + start_lon


# ===================================================
# Bilinear interpolation
# - Online: latlon_interp_bilinear_column()
# - Precompute + apply: latlon_interp_bilinear_init()/latlon_interp_bilinear_apply()
# ===================================================
def latlon_interp_bilinear_column(
    src_lon: Float[Array, "nlon"],
    src_lat: Float[Array, "nlat"],
    src_data: Float[Array, "nlev nlat nlon"],
    dst_lon: Float[Array, "npoint_dst"],
    dst_lat: Float[Array, "npoint_dst"],
    extrap: bool = True,
    bkd: AbstractBackend | None = None,
) -> Float[Array, "nlev npoint_dst"]:
    bkd = require_backend(src_data, src_lon, src_lat, dst_lon, dst_lat, bkd=bkd)
    src_lon = bkd.asarray(src_lon)
    src_lat = bkd.asarray(src_lat)
    src_data = bkd.asarray(src_data)
    dst_lon = bkd.asarray(dst_lon)
    dst_lat = bkd.asarray(dst_lat)

    if src_data.ndim == 3:
        # When src_data has 3 dimensions, we apply the interpolation to each vertical level independently using vmap
        return bkd.vmap(
            lambda src_data_2d: _latlon_interp_bilinear_column(
                bkd,
                src_lon,
                src_lat,
                src_data_2d,
                dst_lon,
                dst_lat,
                extrap,
            ),
            in_axes=0,
            out_axes=0,
        )(src_data)
    elif src_data.ndim == 2:
        # When src_data has 2 dimensions, we directly apply the interpolation without vmap
        return _latlon_interp_bilinear_column(
            bkd,
            src_lon,
            src_lat,
            src_data,
            dst_lon,
            dst_lat,
            extrap,
        )
    else:
        raise ValueError(f"Invalid src_data shape: {src_data.shape}")


def _latlon_interp_bilinear_column(
    bkd: AbstractBackend,
    src_lon: Float[Array, "nlon"],
    src_lat: Float[Array, "nlat"],
    src_data: Float[Array, "nlat nlon"],
    dst_lon: Float[Array, "npoint_dst"],
    dst_lat: Float[Array, "npoint_dst"],
    extrap: bool = True
) -> Float[Array, "npoint_dst"]:
    nx1 = src_lon.shape[0]
    ny1 = src_lat.shape[0]
    # ===================================================
    # Longitude: searchsorted + wraparound
    dst_lon_norm = normalize_lon_to_source(bkd, dst_lon, src_lon[0])
    i1 = bkd.searchsorted(src_lon, dst_lon_norm, side="right") - 1
    i1 = bkd.clip(i1, 0, nx1 - 1)
    i2 = i1 + 1

    wrap = dst_lon_norm >= src_lon[-1]
    i1 = bkd.where(wrap, nx1 - 1, i1)
    i2 = bkd.where(wrap, 0, i2)

    tmp1x = src_lon[i1]
    tmp2x = bkd.where(wrap, src_lon[i2] + 360.0, src_lon[i2])

    xwgt1 = (tmp2x - dst_lon_norm) / (tmp2x - tmp1x)
    xwgt2 = (dst_lon_norm - tmp1x) / (tmp2x - tmp1x)
    # ===================================================
    # Latitude: searchsorted
    j1 = bkd.searchsorted(src_lat, dst_lat, side="right") - 1
    j1 = bkd.clip(j1, 0, ny1 - 2)
    j2 = j1 + 1

    inside = (dst_lat >= src_lat[0]) & (dst_lat < src_lat[-1])

    if extrap:
        south = dst_lat < src_lat[0]
        north = dst_lat >= src_lat[-1]

        # south: j1=0, j2=1
        j1 = bkd.where(south, 0, j1)
        j2 = bkd.where(south, 1, j2)

        # north: j1=ny-1, j2=ny-2
        j1 = bkd.where(north, ny1 - 1, j1)
        j2 = bkd.where(north, ny1 - 2, j2)

        tmp1y = src_lat[j1]
        tmp2y = src_lat[j2]
        ywgt1 = (tmp2y - dst_lat) / (tmp2y - tmp1y)
        ywgt2 = (dst_lat - tmp1y) / (tmp2y - tmp1y)
    else:
        # When extrapolation is not allowed, weights outside the valid range are set to NaN
        tmp1y = src_lat[j1]
        tmp2y = src_lat[j2]
        ywgt1 = (tmp2y - dst_lat) / (tmp2y - tmp1y)
        ywgt2 = (dst_lat - tmp1y) / (tmp2y - tmp1y)
        ywgt1 = bkd.where(inside, ywgt1, float("nan"))
        ywgt2 = bkd.where(inside, ywgt2, float("nan"))
    # ===================================================
    # Bilinear interpolation
    v11 = src_data[j1, i1]
    v21 = src_data[j1, i2]
    v12 = src_data[j2, i1]
    v22 = src_data[j2, i2]

    tmp1 = v11 * xwgt1 + v21 * xwgt2
    tmp2 = v12 * xwgt1 + v22 * xwgt2
    dst_data = tmp1 * ywgt1 + tmp2 * ywgt2
    # ===================================================
    return dst_data


def latlon_interp_bilinear_init(
    src_lon: Float[Array, "nlon"],
    src_lat: Float[Array, "nlat"],
    dst_lon: Float[Array, "npoint_dst"],
    dst_lat: Float[Array, "npoint_dst"],
    extrap: bool = True,
    bkd: AbstractBackend | None = None,
) -> dict[str, object]:
    bkd = require_backend(src_lon, src_lat, dst_lon, dst_lat, bkd=bkd)
    check_strictly_increasing_if_concrete("src_lon", src_lon)
    check_strictly_increasing_if_concrete("src_lat", src_lat)

    if dst_lon.shape != dst_lat.shape:
        raise ValueError(
            "dst_lon and dst_lat must have the same shape, got "
            f"{dst_lon.shape} and {dst_lat.shape}"
        )

    src_lon = bkd.asarray(src_lon)
    src_lat = bkd.asarray(src_lat)
    dst_lon = bkd.asarray(dst_lon)
    dst_lat = bkd.asarray(dst_lat)

    nx1 = src_lon.shape[0]
    ny1 = src_lat.shape[0]

    # ===================================================
    # Longitude: searchsorted + wraparound
    dst_lon_norm = normalize_lon_to_source(bkd, dst_lon, src_lon[0])
    i1 = bkd.searchsorted(src_lon, dst_lon_norm, side="right") - 1
    i1 = bkd.clip(i1, 0, nx1 - 1)
    i2 = i1 + 1

    wrap = dst_lon_norm >= src_lon[-1]
    i1 = bkd.where(wrap, nx1 - 1, i1)
    i2 = bkd.where(wrap, 0, i2)

    tmp1x = src_lon[i1]
    tmp2x = bkd.where(wrap, src_lon[i2] + 360.0, src_lon[i2])

    xwgt1 = (tmp2x - dst_lon_norm) / (tmp2x - tmp1x)
    xwgt2 = (dst_lon_norm - tmp1x) / (tmp2x - tmp1x)

    # ===================================================
    # Latitude: searchsorted
    j1 = bkd.searchsorted(src_lat, dst_lat, side="right") - 1
    j1 = bkd.clip(j1, 0, ny1 - 2)
    j2 = j1 + 1

    inside = (dst_lat >= src_lat[0]) & (dst_lat < src_lat[-1])

    if extrap:
        south = dst_lat < src_lat[0]
        north = dst_lat >= src_lat[-1]

        j1 = bkd.where(south, 0, j1)
        j2 = bkd.where(south, 1, j2)

        j1 = bkd.where(north, ny1 - 1, j1)
        j2 = bkd.where(north, ny1 - 2, j2)

        tmp1y = src_lat[j1]
        tmp2y = src_lat[j2]
        ywgt1 = (tmp2y - dst_lat) / (tmp2y - tmp1y)
        ywgt2 = (dst_lat - tmp1y) / (tmp2y - tmp1y)
    else:
        tmp1y = src_lat[j1]
        tmp2y = src_lat[j2]
        ywgt1 = (tmp2y - dst_lat) / (tmp2y - tmp1y)
        ywgt2 = (dst_lat - tmp1y) / (tmp2y - tmp1y)
        ywgt1 = bkd.where(inside, ywgt1, float("nan"))
        ywgt2 = bkd.where(inside, ywgt2, float("nan"))

    return {
        "src_nlat": src_lat.shape[0],
        "src_nlon": src_lon.shape[0],
        "i1": i1,
        "i2": i2,
        "j1": j1,
        "j2": j2,
        "xwgt1": xwgt1,
        "xwgt2": xwgt2,
        "ywgt1": ywgt1,
        "ywgt2": ywgt2,
    }


def latlon_interp_bilinear_apply(
    src_data: Float[Array, "nlev nlat nlon"],
    remap_cache: dict[str, object],
    bkd: AbstractBackend | None = None,
) -> Float[Array, "nlev npoint_dst"]:
    bkd = require_backend(src_data, bkd=bkd)
    src_data = bkd.asarray(src_data)
    if src_data.ndim == 2:
        src_data_3d = src_data[None, ...]
        squeeze_out = True
    elif src_data.ndim == 3:
        src_data_3d = src_data
        squeeze_out = False
    else:
        raise ValueError(f"Invalid src_data shape: {src_data.shape}")

    src_nlat = cache_get_int(remap_cache, "src_nlat")
    src_nlon = cache_get_int(remap_cache, "src_nlon")
    if src_data_3d.shape[1:] != (src_nlat, src_nlon):
        raise ValueError(
            "src_data shape mismatch with remap_cache: expected "
            f"(nlev, {src_nlat}, {src_nlon}), got {src_data_3d.shape}"
        )

    i1 = cache_get_array(remap_cache, "i1", bkd)
    i2 = cache_get_array(remap_cache, "i2", bkd)
    j1 = cache_get_array(remap_cache, "j1", bkd)
    j2 = cache_get_array(remap_cache, "j2", bkd)
    xwgt1 = cache_get_array(remap_cache, "xwgt1", bkd)
    xwgt2 = cache_get_array(remap_cache, "xwgt2", bkd)
    ywgt1 = cache_get_array(remap_cache, "ywgt1", bkd)
    ywgt2 = cache_get_array(remap_cache, "ywgt2", bkd)

    v11 = src_data_3d[:, j1, i1]
    v21 = src_data_3d[:, j1, i2]
    v12 = src_data_3d[:, j2, i1]
    v22 = src_data_3d[:, j2, i2]

    tmp1 = v11 * xwgt1[None, :] + v21 * xwgt2[None, :]
    tmp2 = v12 * xwgt1[None, :] + v22 * xwgt2[None, :]
    dst_data = tmp1 * ywgt1[None, :] + tmp2 * ywgt2[None, :]

    return dst_data[0] if squeeze_out else dst_data
