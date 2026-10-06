# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     notebook_metadata_filter: jupytext,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv.upwind import (
    UpwindStencilInfoAtCell,
    UpwindStencilInfoAtVtx,
    calc_hx_2nd_stencil,
    calc_hx_4th_stencil,
    calc_tri_2nd_stencil_1,
    calc_tri_2nd_stencil_2,
    calc_tri_4th_stencil_1,
    calc_tri_4th_stencil_2,
)
import cartopy.crs as ccrs
import geoviews as gv
import numpy as np
import terraflow.core as tf

grist_path = "../../data/GRIST_INPUT/uniform-pl-20211229/"
grist_0d_file = grist_path + "grist.grid_file.g4.ccvt.0d.nc"
grist_2d_file = grist_path + "grist.grid_file.g4.ccvt.2d.nc"

# %%
dev = tf.util.device.CPU(backend="jax", float_type="r8", int_type="i4")

# %%
para = tf.planet.PlanetParameterSet.from_si(
    planet="earth", component="atmosphere", dev=dev
)

mesh = tf.mesh.SCVTMesh.from_grist(
    mesh_0d_file=grist_0d_file,
    mesh_2d_file=grist_2d_file,
    para=para,
    hori_hw=3,
    dev=dev
)

# %%
gv.extension("matplotlib")


def _to_lon180(lon):
    return ((lon + 180.0) % 360.0) - 180.0


def _unwrap_lon(lon, center_lon):
    return ((lon - center_lon + 180.0) % 360.0) + center_lon - 180.0


def _stencil_plot_arrays(mesh, use_dual):
    if use_dual:
        return {
            "center_lon": np.asarray(mesh.vtx_lon_deg),
            "center_lat": np.asarray(mesh.vtx_lat_deg),
            "face_nodes": np.asarray(mesh.cell_of_vtx),
            "face_nnodes": np.asarray(mesh.vtx_nnb),
            "node_lon": np.asarray(mesh.cell_lon_deg),
            "node_lat": np.asarray(mesh.cell_lat_deg),
            "face_neighbors": np.asarray(mesh.vtx_nb),
        }
    return {
        "center_lon": np.asarray(mesh.cell_lon_deg),
        "center_lat": np.asarray(mesh.cell_lat_deg),
        "face_nodes": np.asarray(mesh.vtx_of_cell),
        "face_nnodes": np.asarray(mesh.cell_nnb),
        "node_lon": np.asarray(mesh.vtx_lon_deg),
        "node_lat": np.asarray(mesh.vtx_lat_deg),
        "face_neighbors": np.asarray(mesh.cell_nb),
    }


def _context_faces(cell, member_faces, face_neighbors, layers=1):
    face_ids = {int(cell), *[int(face) for face in member_faces]}
    for _ in range(layers):
        for face in tuple(face_ids):
            for neighbor in face_neighbors[face]:
                if neighbor >= 0:
                    face_ids.add(int(neighbor))
    return np.asarray(sorted(face_ids), dtype=np.int64)


def _face_paths(arrays, face_ids, center_lon):
    paths = []
    for face in face_ids:
        nodes = arrays["face_nodes"][face, : int(arrays["face_nnodes"][face])]
        nodes = nodes[nodes >= 0]
        lon = _unwrap_lon(_to_lon180(arrays["node_lon"][nodes]), center_lon)
        lat = arrays["node_lat"][nodes]
        coords = [(float(x), float(y)) for x, y in zip(lon, lat)]
        paths.append(coords + [coords[0]])
    return paths


def plot_stencil_members(
    mesh,
    cell,
    stencil_num,
    stencil_idxes,
    use_dual,
    global_view=True,
    context_layers=1,
    fig_size=260,
):
    arrays = _stencil_plot_arrays(mesh, use_dual)
    n_member = int(np.asarray(stencil_num)[cell])
    member_faces = np.asarray(stencil_idxes[cell, :n_member]).astype(np.int64)
    member_faces = member_faces[member_faces >= 0]

    face_lon_center = float(_to_lon180(arrays["center_lon"][cell]))
    face_lat_center = float(arrays["center_lat"][cell])
    map_crs = ccrs.Orthographic(
        central_longitude=face_lon_center,
        central_latitude=face_lat_center,
    )
    data_crs = ccrs.PlateCarree()

    if global_view:
        face_ids = np.arange(arrays["center_lon"].shape[0], dtype=np.int64)
    else:
        face_ids = _context_faces(
            cell,
            member_faces,
            arrays["face_neighbors"],
            layers=context_layers,
        )
    paths = _face_paths(arrays, face_ids, face_lon_center)

    face_lon = _unwrap_lon(_to_lon180(arrays["center_lon"][member_faces]), face_lon_center)
    face_lat = arrays["center_lat"][member_faces]

    base = gv.Path(paths, crs=data_crs).opts(
        projection=map_crs,
        color="#1f77b4",
        linewidth=1.2,
    )
    points = gv.Points(
        {
            "Longitude": face_lon,
            "Latitude": face_lat,
            "member": member_faces,
        },
        kdims=["Longitude", "Latitude"],
        vdims=["member"],
        crs=data_crs,
    ).opts(
        color="member",
        cmap="viridis",
        projection=map_crs,
        s=30,
        marker="o",
        edgecolor="black",
    )
    target = gv.Points(
        {
            "Longitude": [face_lon_center],
            "Latitude": [face_lat_center],
        },
        kdims=["Longitude", "Latitude"],
        crs=data_crs,
    ).opts(
        color="red",
        projection=map_crs,
        s=90,
        marker="x",
    )

    path_lons = np.asarray([point[0] for path in paths for point in path])
    path_lats = np.asarray([point[1] for path in paths for point in path])
    projected = map_crs.transform_points(data_crs, path_lons, path_lats)
    extent = np.nanmax(np.abs(projected[:, :2]))
    extent_scale = 1.0 if global_view else 1.08
    extent = max(float(extent) * extent_scale, 1.0)
    view_opts = {
        "fig_size": fig_size,
        "xlim": (-extent, extent),
        "ylim": (-extent, extent),
    }

    return base.opts(**view_opts) * points.opts(**view_opts) * target.opts(**view_opts)


# %%
cell = 800
stencil = UpwindStencilInfoAtCell(polynomial_order=2)
stencil_num, stencil_idxes = calc_hx_2nd_stencil(mesh, stencil.max_member)

plot_stencil_members(mesh, cell, stencil_num, stencil_idxes, use_dual=False)

# %%
cell = 800
stencil = UpwindStencilInfoAtCell(polynomial_order=4)
stencil_num, stencil_idxes = calc_hx_4th_stencil(mesh, stencil.max_member)

plot_stencil_members(mesh, cell, stencil_num, stencil_idxes, use_dual=False)

# %%
cell = 800
stencil = UpwindStencilInfoAtVtx(polynomial_order=2, tri_stencil_type=1)
stencil_num, stencil_idxes = calc_tri_2nd_stencil_1(mesh, stencil.max_member)

plot_stencil_members(mesh, cell, stencil_num, stencil_idxes, use_dual=True)

# %%
cell = 800
stencil = UpwindStencilInfoAtVtx(polynomial_order=2, tri_stencil_type=2)
stencil_num, stencil_idxes = calc_tri_2nd_stencil_2(mesh, stencil.max_member)

plot_stencil_members(mesh, cell, stencil_num, stencil_idxes, use_dual=True)

# %%
cell = 800
stencil = UpwindStencilInfoAtVtx(polynomial_order=4, tri_stencil_type=1)
stencil_num, stencil_idxes = calc_tri_4th_stencil_1(mesh, stencil.max_member)

plot_stencil_members(mesh, cell, stencil_num, stencil_idxes, use_dual=True)

# %%
cell = 800
stencil = UpwindStencilInfoAtVtx(polynomial_order=4, tri_stencil_type=2)
stencil_num, stencil_idxes = calc_tri_4th_stencil_2(mesh, stencil.max_member)

plot_stencil_members(mesh, cell, stencil_num, stencil_idxes, use_dual=True)
