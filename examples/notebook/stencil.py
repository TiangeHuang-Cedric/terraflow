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

method = terraflow.discretization.operator.FiniteVolumeMethod()

# %%
import holoviews as hv

from terraflow.discretization.operator.fvm.odm.trisk.single_lev.flux_reconstruct import (
    tr_stencil_member_list_4th_order_mbs
)

hv.extension("bokeh")  # make sure Holoviews renders in the notebook

uxgrid = mesh.to_ugrid(use_dual=True).to_uxgrid()
_, stencil_idxes = tr_stencil_member_list_4th_order_mbs(mesh.dual)
cell = 800  # face index to inspect

member_faces = stencil_idxes[cell, :]
member_faces = member_faces[member_faces >= 0]  # ignore padded entries
face_lon_center = uxgrid.face_lon.isel(n_face=cell).values
face_lat_center = uxgrid.face_lat.isel(n_face=cell).values
face_lon = uxgrid.face_lon.isel(n_face=member_faces).values
face_lat = uxgrid.face_lat.isel(n_face=member_faces).values

points = hv.Points(
    {
        "lon": face_lon,
        "lat": face_lat,
        "member": member_faces,
    },
    kdims=["lon", "lat"],
    vdims=["member"],
).opts(
    color="member",
    cmap="viridis",
    size=5,
    marker="circle",
    line_color="black",
    tools=["hover"],  # inspect member ordering
)

uxgrid.plot(
    height=500,
    width=1000,
) * points
