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
#     display_name: Python (terraflow-dev)
#     language: python
#     name: terraflow-dev
# ---

# %%
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv.upwind import (
    UpwindStencilInfoAtVtx,
    calc_tri_4th_stencil_2
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

mesh = tf.mesh.SCVTMesh.from_grist(
    mesh_0d_file=grist_0d_file,
    mesh_2d_file=grist_2d_file,
    para=para,
    hori_hw=3,
    lev_hw=3,
    dev=dev
)
# %%
# %matplotlib inline

hv.extension("matplotlib")
cell = 800  # face index to inspect
uxgrid = mesh.to_ugrid(use_dual=True).to_uxgrid()

stencil = UpwindStencilInfoAtVtx(polynomial_order=4, tri_stencil_type=2)
_, stencil_idxes = calc_tri_4th_stencil_2(mesh, stencil.max_member)

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
