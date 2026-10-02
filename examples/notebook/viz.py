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

# %% [markdown]
# ## Prepare mesh and data

# %%
# %matplotlib inline
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import holoviews as hv
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import terraflow.core as tf

from terraflow.util.uxarray import UGRIDGrid, UGRIDDataset

grist_path = "../../data/GRIST_INPUT/uniform-pl-20211229/"
grist_0d_file = grist_path + "grist.grid_file.g4.ccvt.0d.nc"
grist_2d_file = grist_path + "grist.grid_file.g4.ccvt.2d.nc"

dev = tf.util.device.CPU(backend="jax", float_type="r8", int_type="i4")

para = tf.planet.PlanetParameterSet.from_si(
    planet="earth", component="atmosphere", dev=dev
)

test_case = tf.test_case.swm.MountainZonalFlowTestCase(para)

mesh = tf.mesh.SCVTMesh.from_grist(
    mesh_0d_file=grist_0d_file,
    mesh_2d_file=grist_2d_file,
    para=para,
    hori_hw=3,
    dev=dev
)

# %%
config = tf.model.UserConfig(
    pv_adv_scheme="upwind",
    upwind_order_pv=3,
    run_time="30 days",
    dt_dyn="300 seconds",
    save_interval="6 hours"
)

# %%
model = tf.model.build_shallow_water_model(
    config=config,
    para=para,
    mesh=mesh,
    method="finite_volume_method",
    form="vector_invariant",
    staggered_kind="triskc",
    test_case=test_case,
    dev=dev,
    restart=False
)

# %% [markdown]
# ## Plot using easygems

# %%
gz = xr.DataArray(np.array(model.block.state.gz.unwrap()[0, :, :]).squeeze())
from_lon = np.rad2deg(mesh.cell_lon)
from_lat = np.rad2deg(mesh.cell_lat)

remap_gz = tf.util.easygems.terraflow2healpix(
    gz, from_lon, from_lat, 10
)

img = tf.util.easygems.plot_healpix_grid(remap_gz)

display(img)

# %% [markdown]
# ## Plot using uxarray

# %% [markdown]
# ### Convert data and mesh into UGRID Format and pack it into uxarray

# %%
ugrid_ds = model.block.state.to_ugrid(mesh, use_dual=False, dim0_idx=0)

ugrid_ds.uxds

# %%
ugrid_ds.uxgrid.plot(
    height=500,
    width=1000,
    backend="matplotlib",
)

# %%
ugrid_ds.uxgrid.plot(
    projection=ccrs.Orthographic(central_latitude=0, central_longitude=0),
    linewidth=0.5,
    title="Spherical voronoi Triangulation",
    height=500,
    width=1000,
    backend="matplotlib",
)

# %% [markdown]
# ### Plot directly

# %% [markdown]
# #### Plot in 2d figure

# %%
ugrid_ds.uxds["gz"].isel(lev=0).plot(height=500, width=1000)

# %% [markdown]
# #### Plot in 2d geomap

# %%
ugrid_ds.uxds["gz"].isel(lev=0).plot(
    height=500,
    width=1000,
    projection=ccrs.Robinson(),
    backend="matplotlib",
    pixel_ratio=0.5,
    cmap="plasma",
    rasterize=True,
    features=["borders", "coastline"]
)

# %% [markdown]
# ### Plot rasterized data

# %%
fig, ax = plt.subplots(
    subplot_kw={"projection": ccrs.Robinson()}, figsize=(9, 6), constrained_layout=True
)

ax.set_global()

raster = ugrid_ds.uxds["gz"].isel(lev=0).to_raster(ax=ax, pixel_ratio=0.5,)
img = ax.imshow(
    raster, cmap="plasma", origin="lower", extent=ax.get_xlim() + ax.get_ylim()
)
ax.set_title("Global Raster")
ax.coastlines()

# Adding a colorbar (the examples below will not include one to keep things concise)
cbar = fig.colorbar(img, ax=ax, fraction=0.03)

display(fig)

# img.figure.savefig("test.png")

# %% [markdown]
# ### Remap

# %%
dst = ugrid_ds.to_structured_grid(
    east_west_resolution=10.0,
    south_north_resolution=10.0,
    remap_method="nearest_neighbor",
    with_reshaped_ds=False
)
dst.uxgrid

# %%
hv.extension("matplotlib")

dst.uxgrid.plot(
    projection=ccrs.Orthographic(central_latitude=0, central_longitude=0),
    linewidth=0.5,
    title="Lat-Lon Grid",
    height=500,
    width=1000,
)

# %% [markdown]
# #### Plot with structured grid and data

# %% [markdown]
# ##### Plot Scalar Field

# %%
_, dst = ugrid_ds.to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="nearest_neighbor",
    with_reshaped_ds=True
)

lon_np = dst["face_lon"]
lat_np = dst["face_lat"]
data = dst["gz"].isel(lev=0)

fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
contour = ax.contourf(
    lon_np,
    lat_np,
    data,
    cmap="plasma",
    transform=ccrs.PlateCarree(),
)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.set_title("Remapped gz on Lat-Lon Grid")
cbar = plt.colorbar(contour, ax=ax, orientation="vertical", pad=0.04)
cbar.set_label("Layer Geopotential (m2/s2)")
plt.tight_layout()
display(fig)

# %%
_, dst = ugrid_ds.to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="bilinear",
    with_reshaped_ds=True
)

lon_np = dst["face_lon"]
lat_np = dst["face_lat"]
data = dst["gz"].isel(lev=0)

fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
contour = ax.contourf(
    lon_np,
    lat_np,
    data,
    cmap="plasma",
    transform=ccrs.PlateCarree(),
)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.set_title("Remapped gz on Lat-Lon Grid")
cbar = plt.colorbar(contour, ax=ax, orientation="vertical", pad=0.04)
cbar.set_label("Layer Geopotential (m2/s2)")
plt.tight_layout()
display(fig)

# %%
_, dst = ugrid_ds.to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="inverse_distance_weighted",
    with_reshaped_ds=True
)

lon_np = dst["face_lon"]
lat_np = dst["face_lat"]
data = dst["gz"].isel(lev=0)

fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
contour = ax.contourf(
    lon_np,
    lat_np,
    data,
    cmap="plasma",
    transform=ccrs.PlateCarree(),
)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.set_title("Remapped gz on Lat-Lon Grid")
cbar = plt.colorbar(contour, ax=ax, orientation="vertical", pad=0.04)
cbar.set_label("Layer Geopotential (m2/s2)")
plt.tight_layout()
display(fig)

# %%
lon_np = np.asarray(lon_np)
lat_np = np.asarray(lat_np)

extent = [lon_np.min(), lon_np.max(), lat_np.min(), lat_np.max()]

fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
img = ax.imshow(
    data,
    origin="lower",
    extent=extent,
    transform=ccrs.PlateCarree(),
    cmap="plasma",
    interpolation="nearest",
)
contour = ax.contour(
    lon_np,
    lat_np,
    data,
    levels=10,
    alpha=0.8,
    colors="black",
    linestyles="--",
    linewidths=1.5,
    transform=ccrs.PlateCarree()
)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.set_title("Remapped gz on Lat-Lon Grid")
cbar = plt.colorbar(img, ax=ax, orientation="vertical", pad=0.04)
cbar.set_label("Layer Geopotential (m2/s2)")
plt.tight_layout()

display(fig)

# %% [markdown]
# ##### Plot UV Wind Vectors on Lat-Lon Grid

# %%
u = model.block.aux.u.unwrap()
v = model.block.aux.v.unwrap()

ugrid_builder = UGRIDGrid.from_scvt_mesh(
    mesh,
    use_dual=False,
    title="Unstructured Primal-Dual Mesh",
)

uxgrid = ugrid_builder.to_uxgrid()

u = xr.DataArray(
    data=u.squeeze(),
    dims=("n_face",),
    attrs={
        "long_name": "u component wind",
        "units": "m/s",
        "mesh": "primal",
        "location": "face",
        "coordinates": "face_lon face_lat",
    },
)

v = xr.DataArray(
    data=v.squeeze(),
    dims=("n_face",),
    attrs={
        "long_name": "v component wind",
        "units": "m/s",
        "mesh": "primal",
        "location": "face",
        "coordinates": "face_lon face_lat",
    },
)

variables = xr.Dataset(
    data_vars={
        "u": u,
        "v": v
    }
)

attrs = {
    "Conventions": "CF-1.11, UGRID-1.0",
    "title": "Shallow Water Model on Unstructured Primal-Dual Mesh",
    "institution": "TerraFlow Project",
    "source": "TerraFlow",
    "history": "Created for demonstration purposes",
}

uv_ugrid_ds = UGRIDDataset.from_xarray(
    ds=variables,
    uxgrid=uxgrid,
    attrs=attrs
)


dst, dst_reshaped = uv_ugrid_ds.to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="bilinear",
    with_reshaped_ds=True
)


lon = dst_reshaped["face_lon"]
lat = dst_reshaped["face_lat"]
u = dst_reshaped["u"]
v = dst_reshaped["v"]
speed = np.sqrt(u**2 + v**2)


fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")
contour = ax.contourf(
    lon, lat, speed,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    levels=20,
    alpha=0.7
)
cbar = plt.colorbar(contour, ax=ax, orientation="vertical", pad=0.04)
q = ax.quiver(
    lon, lat, u, v,
    transform=ccrs.PlateCarree(),
    regrid_shape=20,  # sample every 20 points, to avoid overcrowding
    scale=500,        # ajust arrow length
    width=0.003,     # ajust arrow width
    color="black"
)
cbar.set_label("Wind Speed (m/s)")
ax.set_title("Wind Vectors (u,v)")
plt.tight_layout()

display(fig)


# %%
# Streamline plot over speed contours
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")
filled = ax.contourf(
    lon, lat, speed,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    levels=20,
    alpha=0.7
)
plt.colorbar(
    filled,
    ax=ax,
    orientation="vertical",
    pad=0.04, label="Wind Speed (m/s)"
)
ax.streamplot(
    lon, lat, u, v,
    transform=ccrs.PlateCarree(),
    density=2.0,
    color="black",
    linewidth=0.6,
    arrowsize=0.7,
)
ax.set_title("Wind Streamlines over Speed Contours")
plt.tight_layout()

display(fig)

# %% [markdown]
# #### Plot using remapped dataset directly

# %%
dst = ugrid_ds.to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="nearest_neighbor",
    with_reshaped_ds=False
)


dst["gz"].isel(lev=0).plot(
    height=500,
    width=1000,
    projection=ccrs.Robinson(),
    backend="matplotlib",
    pixel_ratio=0.5,
    cmap="plasma",
    rasterize=True,
    features=["borders", "coastline"],
    periodic_elements="ignore",
)

# %%
dst = ugrid_ds.to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="bilinear",
    with_reshaped_ds=False
)

dst["gz"].isel(lev=0).plot(
    height=500,
    width=1000,
    projection=ccrs.Robinson(),
    backend="matplotlib",
    pixel_ratio=0.5,
    cmap="plasma",
    rasterize=True,
    features=["borders", "coastline"],
    periodic_elements="ignore",
)

# %%
dst = ugrid_ds.to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="inverse_distance_weighted",
    with_reshaped_ds=False
)

dst["gz"].isel(lev=0).plot(
    height=500,
    width=1000,
    projection=ccrs.Robinson(),
    backend="matplotlib",
    pixel_ratio=0.5,
    cmap="plasma",
    rasterize=True,
    features=["borders", "coastline"],
    periodic_elements="ignore",
)
