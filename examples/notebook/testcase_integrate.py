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
from matplotlib.colors import BoundaryNorm
from matplotlib import pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np
import jax.numpy as jnp

import terraflow.core as tf

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
test_case = tf.test_case.swm.MountainZonalFlowTestCase(para)


# %%
config = tf.model.UserConfig(
    pv_adv_scheme="upwind",
    upwind_order_pv=3,
    run_time="20 days",
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

# %%
integrate_fn = tf.model.compiled_with_model(
    model=model,
    fun=model.integrate()
)

# %%
model = integrate_fn(model)

model = tf.model.block_until_ready(model)

final = model.block

# %%
# %matplotlib inline

_, da = final.state.gz.to_ugrid(
    mesh=mesh,
    dim0_idx=final.old
).to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="inverse_distance_weighted",
    with_reshaped_da=True
)

lon_np = da["face_lon"]
lat_np = da["face_lat"]
data = da.squeeze(drop=True).data

lon_np = np.asarray(lon_np)
lat_np = np.asarray(lat_np)

extent = [lon_np.min(), lon_np.max(), lat_np.min(), lat_np.max()]

fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
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
ax.set_title(f"Remapped {da.attrs['name']} on Lat-Lon Grid")
cbar = plt.colorbar(img, ax=ax, orientation="vertical", pad=0.04)
cbar.set_label(f"{da.attrs['name']} at primal cell ({da.attrs['units']})")
plt.tight_layout()

# %%
# %matplotlib inline

_, da = final.aux.u.to_ugrid(mesh=mesh).to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="inverse_distance_weighted",
    with_reshaped_da=True
)

lon_np = da["face_lon"]
lat_np = da["face_lat"]
data = da.squeeze(drop=True).data

lon_np = np.asarray(lon_np)
lat_np = np.asarray(lat_np)

extent = [lon_np.min(), lon_np.max(), lat_np.min(), lat_np.max()]

vmin, vmax, interval = -16, 38, 5
tick_vals = np.arange(vmin, vmax + interval, interval)
bounds = tick_vals
cmap = plt.get_cmap("coolwarm")
norm = BoundaryNorm(bounds, ncolors=cmap.N, extend="both")

fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
img = ax.imshow(
    data,
    origin="lower",
    extent=extent,
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    norm=norm,
    interpolation="bilinear",
)
contour = ax.contour(
    lon_np,
    lat_np,
    data,
    levels=bounds,
    alpha=0.8,
    colors="black",
    # linestyles="--",
    linewidths=1.5,
    transform=ccrs.PlateCarree()
)
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.set_title(f"Remapped {da.attrs['name']} on Lat-Lon Grid")
cbar = plt.colorbar(
    img,
    ax=ax,
    orientation="horizontal",
    pad=0.02,
    shrink=0.8,
    aspect=30,
    ticks=bounds
)
cbar.ax.tick_params(labelsize=10)
cbar.set_label(
    f"{da.attrs['name']} at primal cell ({da.attrs['units']})"
)
plt.tight_layout()

# %%
# # %matplotlib inline
_, u_da = final.aux.u.to_ugrid(mesh=mesh).to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="nearest_neighbor",
    with_reshaped_da=True,
)
_, v_da = final.aux.v.to_ugrid(mesh=mesh).to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="nearest_neighbor",
    with_reshaped_da=True,
)

lon = u_da["face_lon"]
lat = u_da["face_lat"]
u = u_da.squeeze(drop=True).data
v = v_da.squeeze(drop=True).data
speed = np.sqrt(u**2 + v**2)

fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
contour = ax.contourf(
    lon,
    lat,
    speed,
    cmap="plasma",
    transform=ccrs.PlateCarree(),
)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.set_title("Remapped wind vectors on Lat-Lon Grid")
cbar = plt.colorbar(contour, ax=ax, orientation="vertical", pad=0.04)
cbar.set_label("Wind Speed (m/s)")
q = ax.quiver(
    lon, lat, u, v,
    transform=ccrs.PlateCarree(),
    regrid_shape=20,  # sample every 20 points, to avoid overcrowding
    scale=500,  # ajust arrow length
    width=0.003,  # ajust arrow width
    color="black",
)
plt.tight_layout()

plt.show()

# %%
# Streamline plot over speed contours
# # %matplotlib inline
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")
filled = ax.contourf(
    lon, lat, speed,
    transform=ccrs.PlateCarree(),
    cmap="plasma",
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
    arrowsize=0.9,
)
ax.set_title("Wind Streamlines over Speed Contours")
plt.tight_layout()

plt.show()

# %%
final.tend.dundt.to_ugrid(mesh=mesh).isel(lev=0).to_uxarray().plot(
    height=500,
    width=1000,
    projection=ccrs.Robinson(),
    backend="matplotlib",
    pixel_ratio=0.5,
    cmap="plasma",
    # rasterize=True,
    features=["borders", "coastline"],
)
