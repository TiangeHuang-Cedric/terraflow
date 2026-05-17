from __future__ import annotations
from typing import TYPE_CHECKING
import scipy.spatial as sp
import jax
import jax.numpy as jnp
import jaxkd as jk
import numpy as np

from terraflow.util import typing, log, math
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at

if TYPE_CHECKING:
    from terraflow.prepare.readers.topo import AbstractTopographyReader
    from terraflow.mesh.scvt.regridder import Regridder
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as BLOCK


Float = typing.Float
Array = typing.Array


def regrid_topo(block: BLOCK, reader: AbstractTopographyReader) -> BLOCK:
    coord = block.coord
    log.logger.info(f"Regrid topography.")
    # =================================================================
    regrid_type = reader.cfg.regrid_type
    if regrid_type == "fill":
        gzs = fill_grid(block=block, reader=reader)
    elif regrid_type == "interp":
        log.logger.info("Build regridder for topography data.")
        src = Regridder.create_latlon_info(
            lon=reader.lon,
            lat=reader.lat
        )
        dst = Regridder.create_scvt_info(
            mesh=block.coord.mesh,
            loc="cell",
            entity="primal"
        )
        regridder = Regridder(
            src=src,
            dst=dst,
            backend=reader.cfg.interp_backend,
            method=reader.cfg.interp_method,
            extrap=reader.cfg.interp_extrap,
            conserve_opt=reader.cfg.interp_conserve_opt,
            precompute_weights=False,
        )
        gzs = regridder(reader.gzs)
    else:
        raise ValueError(
            f"Unsupported horizontal interpolation method: {regrid_type}"
        )
    # =================================================================
    return tree_at(
        coord=coord,
        where=lambda b: b.static.gzs,
        pytree=block,
        replace=gzs,
    )


def fill_grid(block: BLOCK, reader: AbstractTopographyReader) -> Float[Array, "npoints"]:
    """
    Map regular lat-lon topography points to SCVT primal cells with KDTree.

    Stage-1: nearest-cell mapping by KDTree (chunked source traversal).
    Stage-2: strict spherical in-cell check (optional, enabled by default).
    """
    backend = reader.cfg.fill_backend
    para = reader._para
    if para.planet == "earth":
        if backend == "jaxkd":
            return FillGridWithJaxKDBackend(block=block, reader=reader)()
        elif backend == "scipy":
            return FillGridWithScipyBackend(block=block, reader=reader)()
        else:
            raise ValueError(
                f"Unsupported topo_backend: {backend}. Supported backends: 'jaxkd', 'scipy'."
            )
    else:
        raise NotImplementedError(
            "KDTree fill for topography is currently implemented for Earth only."
        )


class FillGridWithJaxKDBackend:
    def __init__(self, block: BLOCK, reader: AbstractTopographyReader):
        self.block = block
        self.reader = reader

    def __call__(self) -> Float[Array, "npoints"]:
        cfg = self.reader.cfg
        if cfg.fill_split_with_chunks:
            return _jaxkd_fill_grid_on_earth_with_split_chunks(
                block=self.block,
                reader=self.reader,
            )
        else:
            return _jaxkd_fill_grid_on_earth(
                block=self.block,
                reader=self.reader,
            )


class FillGridWithScipyBackend:
    def __init__(self, block: BLOCK, reader: AbstractTopographyReader):
        self.block = block
        self.reader = reader

    def __call__(self) -> Float[Array, "npoints"]:
        cfg = self.reader.cfg
        if cfg.fill_split_with_chunks:
            return _scipy_fill_grid_on_earth_with_split_chunks(
                block=self.block,
                reader=self.reader,
            )
        else:
            return _scipy_fill_grid_on_earth(
                block=self.block,
                reader=self.reader,
            )


def _jaxkd_fill_grid_on_earth(
    block: BLOCK,
    reader: AbstractTopographyReader,
) -> Float[Array, "npoints"]:
    radius = block.para.radius.unwrap()
    mesh = block.coord.mesh
    topo_lon = jnp.deg2rad(reader.lon)
    topo_lat = jnp.deg2rad(reader.lat)
    topo_gzs = jnp.asarray(reader.gzs)
    topo_mask = getattr(reader, "mask", None)
    topo_mask = None if topo_mask is None else jnp.asarray(topo_mask)
    # =======================================================
    # KDTree over destination cell centers
    dst_xyz = math.lonlat2xyz(
        lon=mesh.primal.cell.centroid.lon,
        lat=mesh.primal.cell.centroid.lat,
        radius=radius,
    )
    ncell = int(dst_xyz.shape[0])
    kd_tree = jk.build_tree(dst_xyz)

    max_kdtree_distance2 = float(calc_max_kdtree_distance2(block))
    max_kdtree_distance = float(jnp.sqrt(max_kdtree_distance2))
    # =======================================================
    # Convert topo grid to 1D for KDTree querying
    # topo_lon, topo_lat, topo_gzs, topo_mask: (nlat, nlon) -> (topo_npoint,)
    # topo_npoint = nlon * nlat
    topo_lon_2d, topo_lat_2d = jnp.meshgrid(
        topo_lon,
        topo_lat,
        indexing="xy"  # for consistent flattening
    )
    topo_lon = topo_lon_2d.reshape(-1)
    topo_lat = topo_lat_2d.reshape(-1)
    topo_gzs = topo_gzs.reshape(-1)
    topo_mask = topo_mask.reshape(-1) if topo_mask is not None else None

    src_xyz = math.lonlat2xyz(
        lon=topo_lon,
        lat=topo_lat,
        radius=radius,
    )  # (topo_npoint, 3)

    # (topo_npoints, 1), missing points have index==dst_npoint
    nn_ids, nn_dist = jk.query_neighbors(
        tree=kd_tree,
        query=src_xyz,
        k=1,
    )
    nn_ids = nn_ids[:, 0]
    nn_dist = nn_dist[:, 0]

    # TODO: when in parallel mode/LAM mode, should additionally check if the topo data point is in cells on boundarys.
    src_in_cells = nn_ids
    # =======================================================
    # (topo_npoints,)
    valid_dist = nn_dist <= max_kdtree_distance
    valid_src = (src_in_cells >= 0) & (src_in_cells < ncell) & valid_dist
    # (topo_npoints,) boolean
    is_land = topo_mask == 1 if topo_mask is not None else topo_gzs > 0.0
    topo_gzs = jnp.where(valid_src & is_land, topo_gzs, 0.0)

    # (ncell,) accumulate topo_gzs from src points to dst cells
    dst_gzs = jnp.zeros((ncell,), dtype=topo_gzs.dtype)
    cnt = jnp.zeros((ncell,), dtype=jnp.int32)

    dst_gzs = dst_gzs.at[src_in_cells].add(topo_gzs)
    cnt = cnt.at[src_in_cells].add(valid_src.astype(jnp.int32))
    if jnp.any(cnt == 0):
        raise ValueError(
            "No valid topography points found within KDTree search radius."
        )

    dst_gzs = dst_gzs / cnt
    # =======================================================
    return dst_gzs


def _jaxkd_fill_grid_on_earth_with_split_chunks(
    block: BLOCK,
    reader: AbstractTopographyReader,
) -> Float[Array, "npoints"]:
    lon_chunk_size = reader.cfg.fill_lon_chunk_size
    lat_chunk_size = reader.cfg.fill_lat_chunk_size
    radius = block.para.radius.unwrap()
    mesh = block.coord.mesh
    topo_lon = jnp.deg2rad(reader.lon)
    topo_lat = jnp.deg2rad(reader.lat)
    topo_gzs = jnp.asarray(reader.gzs)
    topo_mask = getattr(reader, "mask", None)
    topo_mask = None if topo_mask is None else jnp.asarray(topo_mask)
    # =======================================================
    # KDTree over destination cell centers
    dst_xyz = math.lonlat2xyz(
        lon=mesh.primal.cell.centroid.lon,
        lat=mesh.primal.cell.centroid.lat,
        radius=radius,
    )
    ncell = int(dst_xyz.shape[0])
    kd_tree = jk.build_tree(dst_xyz)

    max_kdtree_distance2 = float(calc_max_kdtree_distance2(block))
    max_kdtree_distance = float(jnp.sqrt(max_kdtree_distance2))
    # =======================================================
    # Split topo points into chunks for memory efficiency
    nlon = int(topo_lon.shape[0])
    nlat = int(topo_lat.shape[0])
    nlon_chunks = (nlon + lon_chunk_size - 1) // lon_chunk_size
    nlat_chunks = (nlat + lat_chunk_size - 1) // lat_chunk_size

    nlon_pad = nlon_chunks * lon_chunk_size
    nlat_pad = nlat_chunks * lat_chunk_size

    topo_lon_pad = jnp.pad(topo_lon, (0, nlon_pad - nlon), mode="edge")
    topo_lat_pad = jnp.pad(topo_lat, (0, nlat_pad - nlat), mode="edge")
    topo_gzs_pad = jnp.pad(
        topo_gzs,
        ((0, nlat_pad - nlat), (0, nlon_pad - nlon)),
        mode="constant",
    )
    if topo_mask is not None:
        topo_mask_pad = jnp.pad(
            topo_mask,
            ((0, nlat_pad - nlat), (0, nlon_pad - nlon)),
            mode="constant",
        )
    else:
        topo_mask_pad = None

    # (ncell,) accumulate topo_gzs from src points to dst cells
    dst_gzs = jnp.zeros((ncell,), dtype=topo_gzs.dtype)
    cnt = jnp.zeros((ncell,), dtype=jnp.int32)

    i_idx = jnp.arange(nlon_chunks, dtype=jnp.int32)
    j_idx = jnp.arange(nlat_chunks, dtype=jnp.int32)
    lon_offset = jnp.arange(lon_chunk_size, dtype=jnp.int32)
    lat_offset = jnp.arange(lat_chunk_size, dtype=jnp.int32)
    nlon_i32 = jnp.int32(nlon)
    nlat_i32 = jnp.int32(nlat)

    def scan_lat(carry, j):
        dst_gzs_lcl, cnt_lcl, i_s, lon_chunk = carry

        j_s = j * lat_chunk_size
        lat_chunk = jax.lax.dynamic_slice(
            topo_lat_pad,
            (j_s,),
            (lat_chunk_size,),
        )
        gzs_chunk = jax.lax.dynamic_slice(
            topo_gzs_pad,
            (j_s, i_s),
            (lat_chunk_size, lon_chunk_size),
        )
        if topo_mask_pad is not None:
            mask_chunk = jax.lax.dynamic_slice(
                topo_mask_pad,
                (j_s, i_s),
                (lat_chunk_size, lon_chunk_size),
            )
        else:
            mask_chunk = None

        # Convert topo grid to 1D for KDTree querying
        # topo_lon, topo_lat, topo_gzs, topo_mask: (nlat, nlon) -> (topo_npoint,)
        # topo_npoint = nlon * nlat
        topo_lon_2d, topo_lat_2d = jnp.meshgrid(
            lon_chunk,
            lat_chunk,
            indexing="xy",  # for consistent flattening
        )
        lon_chunk_flat = topo_lon_2d.reshape(-1)
        lat_chunk_flat = topo_lat_2d.reshape(-1)
        gzs_chunk_flat = gzs_chunk.reshape(-1)
        if mask_chunk is not None:
            mask_chunk_flat = mask_chunk.reshape(-1)
        else:
            mask_chunk_flat = None

        src_xyz = math.lonlat2xyz(
            lon=lon_chunk_flat,
            lat=lat_chunk_flat,
            radius=radius,
        )  # (topo_npoint, 3)

        # (topo_npoints, 1), missing points have index==dst_npoint
        nn_ids, nn_dist = jk.query_neighbors(
            tree=kd_tree,
            query=src_xyz,
            k=1,
        )
        nn_ids = nn_ids[:, 0]
        nn_dist = nn_dist[:, 0]

        # TODO: when in parallel mode/LAM mode, should additionally check if the topo data point is in cells on boundarys.
        src_in_cells = nn_ids
        # =======================================================
        valid_lon = (i_s + lon_offset) < nlon_i32
        valid_lat = (j_s + lat_offset) < nlat_i32
        valid_chunk = (valid_lat[:, None] & valid_lon[None, :]).reshape(-1)

        valid_dist = nn_dist <= max_kdtree_distance
        valid_src = (
            (src_in_cells >= 0)
            & (src_in_cells < ncell)
            & valid_dist
            & valid_chunk
        )
        is_land = (
            mask_chunk_flat == 1
            if mask_chunk_flat is not None
            else gzs_chunk_flat > 0.0
        )
        gzs_chunk_flat = jnp.where(valid_src & is_land, gzs_chunk_flat, 0.0)

        src_in_cells_safe = jnp.where(valid_src, src_in_cells, 0)
        dst_gzs_lcl = dst_gzs_lcl.at[src_in_cells_safe].add(
            gzs_chunk_flat
        )
        cnt_lcl = cnt_lcl.at[src_in_cells_safe].add(
            valid_src.astype(jnp.int32)
        )

        return (dst_gzs_lcl, cnt_lcl, i_s, lon_chunk), None

    def scan_lon(carry, i):
        dst_gzs_lcl, cnt_lcl = carry

        i_s = i * lon_chunk_size
        lon_chunk = jax.lax.dynamic_slice(
            topo_lon_pad,
            (i_s,),
            (lon_chunk_size,),
        )

        (dst_gzs_lcl, cnt_lcl, _, _), _ = jax.lax.scan(
            scan_lat,
            (dst_gzs_lcl, cnt_lcl, i_s, lon_chunk),
            j_idx,
        )

        return (dst_gzs_lcl, cnt_lcl), None

    (dst_gzs, cnt), _ = jax.lax.scan(scan_lon, (dst_gzs, cnt), i_idx)

    if jnp.any(cnt == 0):
        raise ValueError(
            "No valid topography points found within KDTree search radius."
        )

    dst_gzs = dst_gzs / cnt
    # =======================================================
    return dst_gzs


def _scipy_fill_grid_on_earth(
    block: BLOCK,
    reader: AbstractTopographyReader,
) -> Float[Array, "npoints"]:
    radius = block.para.radius.unwrap()
    mesh = block.coord.mesh
    topo_lon = np.deg2rad(reader.lon)
    topo_lat = np.deg2rad(reader.lat)
    topo_gzs = np.asarray(reader.gzs).copy()
    topo_mask = getattr(reader, "mask", None)
    topo_mask = None if topo_mask is None else np.asarray(topo_mask).copy()
    # =======================================================
    # KDTree over destination cell centers
    dst_xyz = np.asarray(
        math.lonlat2xyz(
            lon=mesh.primal.cell.centroid.lon,
            lat=mesh.primal.cell.centroid.lat,
            radius=radius,
        )
    )
    ncell = int(dst_xyz.shape[0])
    kd_tree = sp.KDTree(dst_xyz)

    max_kdtree_distance2 = float(calc_max_kdtree_distance2(block))
    max_kdtree_distance = float(np.sqrt(max_kdtree_distance2))
    # =======================================================
    # Convert topo grid to 1D for KDTree querying
    # topo_lon, topo_lat, topo_gzs, topo_mask: (nlat, nlon) -> (topo_npoint,)
    # topo_npoint = nlon * nlat
    topo_lon_2d, topo_lat_2d = np.meshgrid(
        topo_lon,
        topo_lat,
        indexing="xy"  # for consistent flattening
    )
    topo_lon = topo_lon_2d.reshape(-1)
    topo_lat = topo_lat_2d.reshape(-1)
    topo_gzs = topo_gzs.reshape(-1)
    topo_mask = topo_mask.reshape(-1) if topo_mask is not None else None

    src_xyz = math.lonlat2xyz(
        lon=jnp.asarray(topo_lon),
        lat=jnp.asarray(topo_lat),
        radius=radius,
    )  # (topo_npoint, 3)
    src_xyz = np.asarray(src_xyz)

    # (topo_npoints,), missing points have index==dst_npoint
    _, nn_ids = kd_tree.query(
        x=src_xyz,
        k=1,
        distance_upper_bound=max_kdtree_distance,
        workers=-1,
    )

    # TODO: when in parallel mode/LAM mode, should additionally check if the topo data point is in cells on boundarys.
    src_in_cells = np.asarray(nn_ids)
    # =======================================================
    # (topo_npoints,)
    valid_src = (src_in_cells >= 0) & (src_in_cells < ncell)
    # (topo_npoints,) boolean
    is_land = topo_mask == 1 if topo_mask is not None else topo_gzs > 0.0
    topo_gzs = np.where(valid_src & is_land, topo_gzs, 0.0)

    # (ncell,) accumulate topo_gzs from src points to dst cells
    dst_gzs = np.zeros((ncell,), dtype=topo_gzs.dtype)
    cnt = np.zeros((ncell,), dtype=np.int32)

    if np.any(valid_src):
        np.add.at(dst_gzs, src_in_cells[valid_src], topo_gzs[valid_src])
        np.add.at(cnt, src_in_cells[valid_src], 1)
    if np.any(cnt == 0):
        raise ValueError(
            "No valid topography points found within KDTree search radius."
        )

    dst_gzs /= cnt
    # =======================================================
    return jnp.asarray(dst_gzs, dtype=reader.gzs.dtype)


def _scipy_fill_grid_on_earth_with_split_chunks(
    block: BLOCK,
    reader: AbstractTopographyReader,
) -> Float[Array, "npoints"]:
    lon_chunk_size = reader.cfg.fill_lon_chunk_size
    lat_chunk_size = reader.cfg.fill_lat_chunk_size
    radius = block.para.radius.unwrap()
    mesh = block.coord.mesh
    topo_lon = np.deg2rad(reader.lon)
    topo_lat = np.deg2rad(reader.lat)
    topo_gzs = np.asarray(reader.gzs).copy()
    topo_mask = getattr(reader, "mask", None)
    topo_mask = None if topo_mask is None else np.asarray(topo_mask).copy()
    # =======================================================
    # KDTree over destination cell centers
    dst_xyz = np.asarray(
        math.lonlat2xyz(
            lon=mesh.primal.cell.centroid.lon,
            lat=mesh.primal.cell.centroid.lat,
            radius=radius,
        )
    )
    ncell = int(dst_xyz.shape[0])
    kd_tree = sp.KDTree(dst_xyz)

    max_kdtree_distance2 = float(calc_max_kdtree_distance2(block))
    max_kdtree_distance = float(np.sqrt(max_kdtree_distance2))
    # =======================================================
    # Split topo points into chunks for memory efficiency
    nlon = topo_lon.shape[0]
    nlat = topo_lat.shape[0]
    nlon_chunks = (nlon + lon_chunk_size - 1) // lon_chunk_size
    nlat_chunks = (nlat + lat_chunk_size - 1) // lat_chunk_size

    # (ncell,) accumulate topo_gzs from src points to dst cells
    dst_gzs = np.zeros((ncell,), dtype=topo_gzs.dtype)
    cnt = np.zeros((ncell,), dtype=np.int32)

    for i in range(nlon_chunks):
        i_s = i * lon_chunk_size
        i_e = min((i + 1) * lon_chunk_size, nlon)
        lon_chunk_1d = topo_lon[i_s:i_e]
        for j in range(nlat_chunks):
            j_s = j * lat_chunk_size
            j_e = min((j + 1) * lat_chunk_size, nlat)
            lat_chunk_1d = topo_lat[j_s:j_e]
            gzs_chunk = topo_gzs[j_s:j_e, i_s:i_e]
            if topo_mask is not None:
                mask_chunk = topo_mask[j_s:j_e, i_s:i_e]
            else:
                mask_chunk = None
            # =======================================================
            # Convert topo grid to 1D for KDTree querying
            # topo_lon, topo_lat, topo_gzs, topo_mask: (nlat, nlon) -> (topo_npoint,)
            # topo_npoint = nlon * nlat
            topo_lon_2d, topo_lat_2d = np.meshgrid(
                lon_chunk_1d,
                lat_chunk_1d,
                indexing="xy"  # for consistent flattening
            )
            lon_chunk_flat = topo_lon_2d.reshape(-1)
            lat_chunk_flat = topo_lat_2d.reshape(-1)
            gzs_chunk_flat = gzs_chunk.reshape(-1)
            if mask_chunk is not None:
                mask_chunk_flat = mask_chunk.reshape(-1)
            else:
                mask_chunk_flat = None

            src_xyz = math.lonlat2xyz(
                lon=jnp.asarray(lon_chunk_flat),
                lat=jnp.asarray(lat_chunk_flat),
                radius=radius,
            )  # (topo_npoint, 3)
            src_xyz = np.asarray(src_xyz)

            # (topo_npoints,), missing points have index==dst_npoint
            _, nn_ids = kd_tree.query(
                x=src_xyz,
                k=1,
                distance_upper_bound=max_kdtree_distance,
                workers=-1,
            )

            # TODO: when in parallel mode/LAM mode, should additionally check if the topo data point is in cells on boundarys.
            src_in_cells = np.asarray(nn_ids)
            # =======================================================
            # (topo_npoints,)
            valid_src = (src_in_cells >= 0) & (src_in_cells < ncell)
            # (topo_npoints,) boolean
            is_land = (
                mask_chunk_flat == 1
                if mask_chunk_flat is not None
                else gzs_chunk_flat > 0.0
            )
            active = valid_src & is_land

            gzs_chunk_flat = np.where(active, gzs_chunk_flat, 0.0)
            if np.any(valid_src):
                np.add.at(
                    dst_gzs,
                    src_in_cells[valid_src],
                    gzs_chunk_flat[valid_src]
                )
                np.add.at(cnt, src_in_cells[valid_src], 1)

    if np.any(cnt == 0):
        raise ValueError(
            "No valid topography points found within KDTree search radius."
        )

    dst_gzs = dst_gzs / cnt
    # =======================================================
    return jnp.asarray(dst_gzs, dtype=reader.gzs.dtype)


def sphere_distance(
    lat1: Float[Array, "..."],
    lon1: Float[Array, "..."],
    lat2: Float[Array, "..."],
    lon2: Float[Array, "..."],
    radius: Float[Array, ""]
) -> Float[Array, "..."]:
    """MPAS-consistent great-circle distance."""
    arg1 = jnp.sqrt(
        jnp.sin(0.5 * (lat2 - lat1)) ** 2
        + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(0.5 * (lon2 - lon1)) ** 2
    )
    return 2.0 * radius * jnp.arcsin(arg1)


def calc_max_kdtree_distance2(block: BLOCK) -> Float[Array, ""]:
    """
    max_diameter = 2 * max(cell-center to each cell-vertex great-circle distance)
    max_kdtree_distance2 = 2 * max_diameter**2
    """
    radius = block.para.radius.unwrap()
    mesh = block.coord.mesh
    cell = mesh.primal.cell
    dual = mesh.dual.cell

    vertices = cell.nodes   # (ncell, maxnnb), invalid = -1
    nnb = cell.n_adjacency  # (ncell,)
    maxnnb = vertices.shape[1]

    cell_lat = cell.centroid.lat     # (ncell,)
    cell_lon = cell.centroid.lon     # (ncell,)
    vtx_lat_all = dual.centroid.lat  # (ndual,)
    vtx_lon_all = dual.centroid.lon  # (ndual,)
    # ==========================================
    valid = (jnp.arange(maxnnb)[None, :] < nnb[:, None]) & (vertices >= 0)
    vid = jnp.clip(vertices, 0, vtx_lat_all.shape[0] - 1)

    vtx_lat = vtx_lat_all[vid]  # (ncell, maxnnb)
    vtx_lon = vtx_lon_all[vid]  # (ncell, maxnnb)
    c_lat = cell_lat[:, None]
    c_lon = cell_lon[:, None]

    d = sphere_distance(c_lat, c_lon, vtx_lat, vtx_lon, radius=radius)
    d = jnp.where(valid, d, 0.0)

    max_diameter = 2.0 * jnp.max(jnp.asarray(d))
    return 2.0 * (max_diameter ** 2)
