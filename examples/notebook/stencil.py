# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     notebook_metadata_filter: language_info,jupytext,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (terraflow-dev)
#     language: python
#     name: terraflow-dev
# ---

# %%
from terraflow.discretization.fvm.scvt.trisk.common.adv.flux_reconstruct import (
    tr_stencil_member_list_4th_order_mbs
)
import cartopy.crs as ccrs
import holoviews as hv
import terraflow.core as tf
import jax.numpy as jnp

grist_path = "../../data/GRIST_INPUT/uniform-pl-20211229/"
grist_0d_file = grist_path + "grist.grid_file.g4.ccvt.0d.nc"
grist_2d_file = grist_path + "grist.grid_file.g4.ccvt.2d.nc"

# %%
dev = tf.util.device.CPU(float_type=jnp.float64, int_type=jnp.int64)

# %%
para = tf.planet.PlanetParameterSet.from_si(
    planet="earth", component="atmosphere", dev=dev
)

mesh = tf.mesh.SphericalCentroidVoronoiTessellations.from_grist(
    grist_0d_file, grist_2d_file, n_dual=3, dev=dev
)

vert = tf.vert_coord.OneLayerVertCoordinate()

coord = tf.layout.CoordinateSystem(
    hori=mesh,
    vert=vert,
    hori_layout=mesh.get_layout(),
    vert_layout=vert.get_layout()
)

method = tf.operator.FiniteVolumeMethod()
# %%
# matplotlib backend so we can use cartopy projections
hv.extension("matplotlib")
cell = 800  # face index to inspect
uxgrid = mesh.to_ugrid(use_dual=True).to_uxgrid()
_, stencil_idxes = tr_stencil_member_list_4th_order_mbs(mesh.dual)

member_faces = stencil_idxes[cell, :]
member_faces = member_faces[member_faces >= 0]  # ignore padded entries
face_lon_center = uxgrid.face_lon.isel(n_face=cell).values
face_lat_center = uxgrid.face_lat.isel(n_face=cell).values

# projections
src_crs = ccrs.PlateCarree()           # data lon/lat
map_crs = ccrs.Orthographic(
    central_longitude=float(face_lon_center),
    central_latitude=float(face_lat_center)
)  # target map projection

face_lon = uxgrid.face_lon.isel(n_face=member_faces).values
face_lat = uxgrid.face_lat.isel(n_face=member_faces).values

# project lon/lat into the target map projection for matplotlib backend
projected = map_crs.transform_points(src_crs, face_lon, face_lat)
x_proj = projected[:, 0]
y_proj = projected[:, 1]

points = hv.Points(
    {
        "x": x_proj,
        "y": y_proj,
        "member": member_faces,
    },
    kdims=["x", "y"],
    vdims=["member"],
).opts(
    color="member",
    cmap="viridis",
    projection=map_crs,  # set axes projection
    s=30,                # marker size for matplotlib backend
    marker="o",
    edgecolor="black",
)

base = uxgrid.plot(
    projection=map_crs,
    height=500,
    width=1000,
)

base * points
