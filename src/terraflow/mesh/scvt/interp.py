import jax
import jax.numpy as jnp

from terraflow.util import typing

Float = typing.Float
Array = typing.Array


def latlon_interp_bilinear_column(
    src_lon: Float[Array, "nlon"],
    src_lat: Float[Array, "nlat"],
    src_data: Float[Array, "nlev nlat nlon"],
    dst_lon: Float[Array, "npoint_dst"],
    dst_lat: Float[Array, "npoint_dst"],
    extrap: bool = False
) -> Float[Array, "nlev npoint_dst"]:
    if src_data.ndim == 3:
        # When src_data has 3 dimensions, we apply the interpolation to each vertical level independently using vmap
        return jax.vmap(
            _latlon_interp_bilinear_column,
            in_axes=(None, None, 0, None, None, None),
            out_axes=0,
        )(
            src_lon,
            src_lat,
            src_data,
            dst_lon,
            dst_lat,
            extrap,
        )
    elif src_data.ndim == 2:
        # When src_data has 2 dimensions, we directly apply the interpolation without vmap
        return _latlon_interp_bilinear_column(
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
    src_lon: Float[Array, "nlon"],
    src_lat: Float[Array, "nlat"],
    src_data: Float[Array, "nlat nlon"],
    dst_lon: Float[Array, "npoint_dst"],
    dst_lat: Float[Array, "npoint_dst"],
    extrap: bool = False
) -> Float[Array, "npoint_dst"]:
    nx1 = src_lon.shape[0]
    ny1 = src_lat.shape[0]
    # ===================================================
    # Longitude: searchsorted + wraparound
    i1 = jnp.searchsorted(src_lon, dst_lon, side="right") - 1
    i1 = jnp.clip(i1, 0, nx1 - 1)
    i2 = i1 + 1

    wrap = dst_lon >= src_lon[-1]
    i1 = jnp.where(wrap, nx1 - 1, i1)
    i2 = jnp.where(wrap, 0, i2)

    tmp1x = src_lon[i1]
    tmp2x = jnp.where(wrap, src_lon[i2] + 360.0, src_lon[i2])

    xwgt1 = (tmp2x - dst_lon) / (tmp2x - tmp1x)
    xwgt2 = (dst_lon - tmp1x) / (tmp2x - tmp1x)
    # ===================================================
    # Latitude: searchsorted
    j1 = jnp.searchsorted(src_lat, dst_lat, side="right") - 1
    j1 = jnp.clip(j1, 0, ny1 - 2)
    j2 = j1 + 1

    inside = (dst_lat >= src_lat[0]) & (dst_lat < src_lat[-1])

    if extrap:
        south = dst_lat < src_lat[0]
        north = dst_lat >= src_lat[-1]

        # south: j1=0, j2=1
        j1 = jnp.where(south, 0, j1)
        j2 = jnp.where(south, 1, j2)

        # north: j1=ny-1, j2=ny-2
        j1 = jnp.where(north, ny1 - 1, j1)
        j2 = jnp.where(north, ny1 - 2, j2)

        tmp1y = jnp.where(
            south,
            src_lat[j1] - dst_lat,
            jnp.where(
                north,
                dst_lat - src_lat[j1],
                src_lat[j1]
            )
        )
        tmp2y = jnp.where(
            south,
            src_lat[j2] - dst_lat,
            jnp.where(
                north,
                dst_lat - src_lat[j2],
                src_lat[j2]
            )
        )

        ywgt1 = jnp.where(
            inside,
            (tmp2y - dst_lat) / (tmp2y - tmp1y),
            tmp2y / (tmp2y - tmp1y)
        )
        ywgt2 = jnp.where(
            inside,
            (dst_lat - tmp1y) / (tmp2y - tmp1y),
            -tmp1y / (tmp2y - tmp1y)
        )
    else:
        # When extrapolation is not allowed, weights outside the valid range are set to NaN
        tmp1y = src_lat[j1]
        tmp2y = src_lat[j2]
        ywgt1 = (tmp2y - dst_lat) / (tmp2y - tmp1y)
        ywgt2 = (dst_lat - tmp1y) / (tmp2y - tmp1y)
        ywgt1 = jnp.where(inside, ywgt1, jnp.nan)
        ywgt2 = jnp.where(inside, ywgt2, jnp.nan)
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
