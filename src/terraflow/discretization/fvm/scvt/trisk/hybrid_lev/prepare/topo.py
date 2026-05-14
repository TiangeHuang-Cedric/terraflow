from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp

from terraflow.util import typing
from terraflow.mesh.scvt.interp import latlon_interp_bilinear_column
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at

if TYPE_CHECKING:
    from terraflow.prepare.readers.topo import AbstractTopographyReader
    from terraflow.layout.field import StaggeredFieldMetaData
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as BLOCK


Float = typing.Float
Array = typing.Array


def _get_hori_coord(meta: StaggeredFieldMetaData, coord) -> dict:
    coord_info = meta.get_hori_position(coord).horizontal_coord
    if coord_info is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for {meta.name} meta."
        )
    coord_info["lon"] = jnp.rad2deg(coord_info["lon"])
    coord_info["lat"] = jnp.rad2deg(coord_info["lat"])
    return coord_info


def fill_grid(
    reader: AbstractTopographyReader,
    dst_lon: Float[Array, "npoints"],
    dst_lat: Float[Array, "npoints"],
) -> Float[Array, "npoints"]:
    para = reader._para
    topo_lon = reader.lon
    topo_lat = reader.lat
    topo_data = reader.gzs
    topo_mask = getattr(reader, "mask", None)
    i_s = jnp.searchsorted(topo_lon, dst_lon, side="right") - 1
    j_s = jnp.searchsorted(topo_lat, dst_lat, side="right") - 1
    i_s = jnp.clip(i_s, 0, topo_lon.shape[0] - 2)
    j_s = jnp.clip(j_s, 0, topo_lat.shape[0] - 2)
    i_e = i_s + 1
    j_e = j_s + 1
    # =================================================================
    if para.planet == "earth":
        if topo_mask is not None:
            mask1 = topo_mask[j_s, i_s] # (npoints,)
            mask2 = topo_mask[j_s, i_e] # (npoints,)
            mask3 = topo_mask[j_e, i_s] # (npoints,)
            mask4 = topo_mask[j_e, i_e] # (npoints,)
            gzs = jnp.where(
                mask1 == 1,
                topo_data[j_s, i_s],
                0.0
            ) + jnp.where(
                mask2 == 1,
                topo_data[j_s, i_e],
                0.0
            ) + jnp.where(
                mask3 == 1,
                topo_data[j_e, i_s],
                0.0
            ) + jnp.where(
                mask4 == 1,
                topo_data[j_e, i_e],
                0.0
            ) # (npoints,)
        else:
            mask1 = topo_data[j_s, i_s] # (npoints,)
            mask2 = topo_data[j_s, i_e] # (npoints,)
            mask3 = topo_data[j_e, i_s] # (npoints,)
            mask4 = topo_data[j_e, i_e] # (npoints,)
            gzs = jnp.where(
                mask1 > 0.0,
                topo_data[j_s, i_s],
                0.0
            ) + jnp.where(
                mask2 > 0.0,
                topo_data[j_s, i_e],
                0.0
            ) + jnp.where(
                mask3 > 0.0,
                topo_data[j_e, i_s],
                0.0
            ) + jnp.where(
                mask4 > 0.0,
                topo_data[j_e, i_e],
                0.0
            ) # (npoints,)
    else:
        raise NotImplementedError(
            f"Filling grid is only implemented for Earth topography currently!"
        )
    cnt = (i_e - i_s + 1) * (j_e - j_s + 1)
    # =================================================================
    return gzs / cnt


def regrid_topo(block: BLOCK, reader: AbstractTopographyReader) -> BLOCK:
    coord = block.coord
    topo_meta = block.static.gzs.meta
    coord_info = _get_hori_coord(topo_meta, coord)
    dst_lon = coord_info["lon"]
    dst_lat = coord_info["lat"]    
    # =================================================================
    if reader.hori_interp_method == "fill":
        gzs = fill_grid(
            reader=reader,
            dst_lon=dst_lon,
            dst_lat=dst_lat,
        )
    elif reader.hori_interp_method == "bilinear":
        gzs = latlon_interp_bilinear_column(
            src_lon=reader.lon,
            src_lat=reader.lat,
            src_data=reader.gzs,
            dst_lon=dst_lon,
            dst_lat=dst_lat,
            extrap=reader.hori_extrap
        )
    else:
        raise ValueError(
            f"Unsupported horizontal interpolation method: {reader.hori_interp_method}"
        )
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: b.static.gzs,
        pytree=block,
        replace=gzs,
    )
