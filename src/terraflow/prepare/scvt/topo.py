from __future__ import annotations
from typing import TYPE_CHECKING
import scipy.spatial as sp
import numpy as np


from terraflow.util import sphere_geometry


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.util.planet import PlanetParameterSet
    from terraflow.util.backend import AbstractBackend
    from terraflow.prepare.readers.topo import AbstractTopographyReader
    from terraflow.mesh.scvt.mesh_type import SCVTMesh

    Float = typing.Float
    Array = typing.Array


def fill_grid(
    reader: AbstractTopographyReader,
    mesh: SCVTMesh,
    loc: str,
    layout: str,
) -> dict[str, Float[Array, "npoints"]]:
    """
    Map regular lat-lon topography points to SCVT primal cells with KDTree.

    Stage-1: nearest-cell mapping by KDTree (chunked source traversal).
    Stage-2: strict spherical in-cell check (optional, enabled by default).
    """
    para = reader.para
    if para.planet == "earth":
        return FillGridWithScipyBackend()(
                reader=reader,
                mesh=mesh,
                loc=loc,
                layout=layout
            )
    else:
        raise NotImplementedError(
            "KDTree fill for topography is currently implemented for Earth only."
        )


class FillGridWithScipyBackend:

    def __call__(
        self,
        reader: AbstractTopographyReader,
        mesh: SCVTMesh,
        loc: str,
        layout: str,
    ) -> dict[str, Float[Array, "npoints"]]:
        cfg = reader.cfg
        if cfg.fill_split_with_chunks:
            return _scipy_fill_grid_on_earth_with_split_chunks(
                reader=reader,
                mesh=mesh,
                loc=loc,
                layout=layout,
            )
        else:
            return _scipy_fill_grid_on_earth(
                reader=reader,
                mesh=mesh,
                loc=loc,
                layout=layout
            )


def _scipy_fill_grid_on_earth(
    reader: AbstractTopographyReader,
    mesh: SCVTMesh,
    loc: str,
    layout: str,
) -> dict[str, Float[Array, "npoints"]]:
    radius = reader.para.radius.unwrap()
    bkd = reader.dev.backend
    topo_lon = np.deg2rad(reader.lon)
    topo_lat = np.deg2rad(reader.lat)
    topo_gzs = bkd.to_numpy(reader.gzs).copy()
    topo_mask = getattr(reader, "mask", None)
    topo_mask = None if topo_mask is None else bkd.to_numpy(topo_mask).copy()
    float_type = reader.dev.float_type
    # =======================================================
    coord_info = mesh.get_coords(loc=loc, layout=layout, with_halo=True)
    dst_lon = coord_info["lon"]
    dst_lat = coord_info["lat"]
    # KDTree over destination cell centers
    dst_xyz = bkd.to_numpy(
        sphere_geometry.lonlat2xyz(
            bkd=bkd,
            lon=dst_lon,
            lat=dst_lat,
            radius=radius,
        )
    )
    ncell = int(dst_xyz.shape[0])
    kd_tree = sp.KDTree(dst_xyz)

    max_kdtree_distance2 = float(calc_max_kdtree_distance2(mesh, reader.para))
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

    src_xyz = sphere_geometry.lonlat2xyz(
        bkd=bkd,
        lon=bkd.asarray(topo_lon),
        lat=bkd.asarray(topo_lat),
        radius=radius,
    )  # (topo_npoint, 3)
    src_xyz = bkd.to_numpy(src_xyz)

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

    # (ncell,) accumulate topo_gzs from src points to dst cells
    dst_gzs = np.zeros((ncell,), dtype=topo_gzs.dtype)
    dst_lnd = np.zeros((ncell,), dtype=topo_gzs.dtype)
    cnt = np.zeros((ncell,), dtype=np.int32)

    if np.any(valid_src):
        np.add.at(
            dst_gzs,
            src_in_cells,
            topo_gzs * (valid_src & is_land)
        )
        np.add.at(
            dst_lnd,
            src_in_cells,
            (topo_gzs * valid_src > 0.0).astype(topo_gzs.dtype)
        )
        np.add.at(
            cnt,
            src_in_cells,
            1 * valid_src
        )
    if np.any(cnt == 0):
        raise ValueError(
            "No valid topography points found within KDTree search radius."
        )

    dst_gzs = dst_gzs / cnt
    dst_lnd = dst_lnd / cnt
    # =======================================================
    return {
        "gzs": bkd.asarray(dst_gzs, dtype=float_type),
        "lnd": bkd.asarray(dst_lnd, dtype=float_type)
    }


def _scipy_fill_grid_on_earth_with_split_chunks(
    reader: AbstractTopographyReader,
    mesh: SCVTMesh,
    loc: str,
    layout: str,
) -> dict[str, Float[Array, "npoints"]]:
    lon_chunk_size = reader.cfg.fill_lon_chunk_size
    lat_chunk_size = reader.cfg.fill_lat_chunk_size
    radius = reader.para.radius.unwrap()
    bkd = reader.dev.backend
    topo_lon = np.deg2rad(reader.lon)
    topo_lat = np.deg2rad(reader.lat)
    topo_gzs = bkd.to_numpy(reader.gzs).copy()
    topo_mask = getattr(reader, "mask", None)
    topo_mask = None if topo_mask is None else bkd.to_numpy(topo_mask).copy()
    float_type = reader.dev.float_type
    # =======================================================
    coord_info = mesh.get_coords(loc=loc, layout=layout, with_halo=True)
    dst_lon = coord_info["lon"]
    dst_lat = coord_info["lat"]
    # KDTree over destination cell centers
    dst_xyz = bkd.to_numpy(
        sphere_geometry.lonlat2xyz(
            bkd=bkd,
            lon=dst_lon,
            lat=dst_lat,
            radius=radius,
        )
    )
    ncell = int(dst_xyz.shape[0])
    kd_tree = sp.KDTree(dst_xyz)

    max_kdtree_distance2 = float(calc_max_kdtree_distance2(mesh, reader.para))
    max_kdtree_distance = float(np.sqrt(max_kdtree_distance2))
    # =======================================================
    # Split topo points into chunks for memory efficiency
    nlon = topo_lon.shape[0]
    nlat = topo_lat.shape[0]
    nlon_chunks = (nlon + lon_chunk_size - 1) // lon_chunk_size
    nlat_chunks = (nlat + lat_chunk_size - 1) // lat_chunk_size

    # (ncell,) accumulate topo_gzs from src points to dst cells
    dst_gzs = np.zeros((ncell,), dtype=topo_gzs.dtype)
    dst_lnd = np.zeros((ncell,), dtype=topo_gzs.dtype)
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

            src_xyz = sphere_geometry.lonlat2xyz(
                bkd=bkd,
                lon=bkd.asarray(lon_chunk_flat),
                lat=bkd.asarray(lat_chunk_flat),
                radius=radius,
            )  # (topo_npoint, 3)
            src_xyz = bkd.to_numpy(src_xyz)

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

            if np.any(valid_src):
                id = src_in_cells[valid_src]
                gzs_valid = gzs_chunk_flat[valid_src]
                
                if mask_chunk_flat is not None:
                    is_land_valid = mask_chunk_flat[valid_src] == 1
                else:
                    is_land_valid = gzs_valid > 0.0
                
                np.add.at(
                    dst_gzs,
                    id,
                    gzs_valid * is_land_valid
                )
                np.add.at(
                    dst_lnd,
                    id,
                    (gzs_valid > 0.0).astype(topo_gzs.dtype),
                )
                np.add.at(
                    cnt,
                    id,
                    1
                )

    owned = np.zeros(ncell, dtype=bool)
    owned[int(mesh.cell_ids):int(mesh.cell_ide) + 1] = True
    if np.any((cnt == 0) & owned):
        raise ValueError(
            "No valid topography points found within KDTree search radius."
        )

    dst_gzs = dst_gzs / cnt
    dst_lnd = dst_lnd / cnt
    # =======================================================
    return {
        "gzs": bkd.asarray(dst_gzs, dtype=float_type),
        "lnd": bkd.asarray(dst_lnd, dtype=float_type)
    }


def sphere_distance(
    bkd: AbstractBackend,
    lat1: Float[Array, "..."],
    lon1: Float[Array, "..."],
    lat2: Float[Array, "..."],
    lon2: Float[Array, "..."] | float,
    radius: float
) -> Float[Array, "..."]:
    """MPAS-consistent great-circle distance."""
    arg1 = bkd.sqrt(
        bkd.sin(0.5 * (lat2 - lat1)) ** 2
        + bkd.cos(lat1) * bkd.cos(lat2) * bkd.sin(0.5 * (lon2 - lon1)) ** 2
    )
    return 2.0 * radius * bkd.arcsin(arg1)


def calc_max_kdtree_distance2(
    mesh: SCVTMesh,
    para: PlanetParameterSet
) -> Float[Array, ""]:
    """
    max_diameter = 2 * max(cell-center to each cell-vertex great-circle distance)
    max_kdtree_distance2 = 2 * max_diameter**2
    """
    bkd = mesh.dev.backend
    radius = para.radius.unwrap()

    vtx_of_cell = mesh.vtx_of_cell   # (ncell, maxnnb), invalid = -1
    cell_nnb = mesh.cell_nnb  # (ncell,)
    maxnnb_cell = mesh.maxnnb_cell

    cell_lat = mesh.cell_lat     # (ncell,)
    cell_lon = mesh.cell_lon     # (ncell,)
    vtx_lat_all = mesh.vtx_lat  # (ndual,)
    vtx_lon_all = mesh.vtx_lon  # (ndual,)
    # ==========================================
    valid = (bkd.arange(maxnnb_cell)[None, :] < cell_nnb[:, None]) & (vtx_of_cell >= 0)
    vid = bkd.clip(vtx_of_cell, 0, vtx_lat_all.shape[0] - 1)

    vtx_lat = vtx_lat_all[vid]  # (ncell, maxnnb)
    vtx_lon = vtx_lon_all[vid]  # (ncell, maxnnb)
    c_lat = cell_lat[:, None]
    c_lon = cell_lon[:, None]

    d = sphere_distance(bkd, c_lat, c_lon, vtx_lat, vtx_lon, radius=radius)
    d = bkd.where(valid, d, 0.0)

    max_diameter = 2.0 * bkd.max(bkd.asarray(d))
    return 2.0 * (max_diameter ** 2)
