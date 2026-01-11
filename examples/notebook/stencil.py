# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python (terraflow-dev)
#     language: python
#     name: terraflow-dev
# ---

# %%
import jax.numpy as jnp

import terraflow.util.device as device

grist_path = "../../data/GRIST_INPUT/uniform-pl-20211229/"
grist_0d_file = grist_path + "grist.grid_file.g4.ccvt.0d.nc"
grist_2d_file = grist_path + "grist.grid_file.g4.ccvt.2d.nc"

# %%
dev = device.CPU(float_type=jnp.float64, int_type=jnp.int64)

# %%
import terraflow

mesh = terraflow.mesh.OrthogonalDualMesh.from_grist(
    grist_0d_file, grist_2d_file, n_dual=3, dev=dev
)

vert = terraflow.vert_coord.OneLayerVertCoordinate()

layout = terraflow.layout.orthogonal_dual_mesh.TriSKBasedCGridStaggeringLayout()

coord = terraflow.layout.CoordinateSystem(hori=mesh, vert=vert, layout=layout)

para = terraflow.planet.PlanetParameterSet.from_si(
    planet="earth", component="atmosphere", dev=dev
)

method = terraflow.discretization.FiniteVolumeMethod()

# %%
import holoviews as hv
import cartopy.crs as ccrs

from terraflow.discretization.fvm.odm.trisk.single_lev.operator.flux_reconstruct import (
    tr_stencil_member_list_4th_order_mbs
)

hv.extension("matplotlib")  # matplotlib backend so we can use cartopy projections
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
) # target map projection

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
