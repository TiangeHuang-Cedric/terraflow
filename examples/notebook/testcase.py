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
from matplotlib.colors import BoundaryNorm
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import cartopy.feature as cfeature
import cartopy.crs as ccrs
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
integrate_fn = model.trajectory(start_with_input=True, with_jit=True)

# %%
final, trajectory = integrate_fn(model)

final      = dev.block_until_ready(final)
trajectory = dev.block_until_ready(trajectory)

# %%
_, ds = final.to_ugrid(use_dual=False).to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="inverse_distance_weighted",
    with_reshaped_ds=True  
)

# %%
# %matplotlib inline

lon_np = ds["face_lon"]
lat_np = ds["face_lat"]
data = ds["gz"].squeeze(drop=True)

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
ax.set_title(f"Remapped {data.attrs['name']} on Lat-Lon Grid")
cbar = plt.colorbar(img, ax=ax, orientation="vertical", pad=0.04)
cbar.set_label(f"{data.attrs['name']} at primal cell ({data.attrs['units']})")
plt.tight_layout()

# %%
# %matplotlib inline

lon_np = ds["face_lon"]
lat_np = ds["face_lat"]
data = ds["u"].squeeze(drop=True)

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
ax.set_title(f"Remapped {data.attrs['name']} on Lat-Lon Grid")
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
    f"{data.attrs['name']} at primal cell ({data.attrs['units']})"
)
plt.tight_layout()

# %%
_, ds = final.trajectory_diag_tools(
    trajectory=trajectory,
    use_dual=False
).to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="inverse_distance_weighted",
    with_reshaped_ds=True  
)

# %%
# Increase animation embed limit (e.g., to 50 MB)
plt.rcParams["animation.embed_limit"] = 200

# Get number of time steps
data = ds["gz"].squeeze(drop=True)
n_frames = data.shape[0]

fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")

lon_np = np.asarray(ds["face_lon"])
lat_np = np.asarray(ds["face_lat"])
extent = [lon_np.min(), lon_np.max(), lat_np.min(), lat_np.max()]

img = ax.imshow(
    data.isel(time=0).data,
    origin="lower",
    extent=extent,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    interpolation="nearest",
    animated=True,
)

cbar = plt.colorbar(
    img,
    ax=ax,
    orientation="horizontal",
    pad=0.04,
    shrink=0.8,
    aspect=30
)
cbar.set_label(f"{data.attrs['name']} at primal cell ({data.attrs['units']})")
title = ax.set_title("Trajectory Step: 0")

# Initial wind (quiver, updatable)

u_da = ds["u"].squeeze(drop=True)
v_da = ds["v"].squeeze(drop=True)
u_0 = u_da.isel(time=0).data
v_0 = v_da.isel(time=0).data

# Downsample the vector field to avoid shape mismatch on updates
ny, nx = u_0.shape
stride = max(1, int(np.ceil(np.sqrt(u_0.size / 500))))
lon_s = lon_np[::stride, ::stride]
lat_s = lat_np[::stride, ::stride]
u_s0 = u_0[::stride, ::stride]
v_s0 = v_0[::stride, ::stride]

qv = ax.quiver(
    lon_s,
    lat_s,
    u_s0,
    v_s0,
    transform=ccrs.PlateCarree(),
    scale=1500,
    width=0.003,
    color="black",
)


def update(frame):
    img.set_data(data.isel(time=frame).data)
    u = u_da.isel(time=frame).data[::stride, ::stride]
    v = v_da.isel(time=frame).data[::stride, ::stride]
    qv.set_UVC(u, v)
    title.set_text(f"Trajectory Step: {frame}")
    return img, qv, title


# Cartopy axes generally do not support blitting; keep blit=False
anim = animation.FuncAnimation(
    fig, update, frames=n_frames, interval=200, blit=False)
# anim.save(
#     "gz_uv.gif",
#     writer="pillow",
#     dpi=120,
#     fps=5,
#     savefig_kwargs={"pad_inches": 0},
# )
plt.close(fig)
HTML(anim.to_jshtml())

# %%
# # %matplotlib inline
lon = u_da["face_lon"]
lat = u_da["face_lat"]
u = u_da.data[-1, ...]
v = v_da.data[-1, ...]
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
final.block.static.gzs.to_uxarray(mesh=final.block.mesh).plot(
    height=500,
    width=1000,
    projection=ccrs.Robinson(),
    backend="matplotlib",
    pixel_ratio=0.5,
    cmap="plasma",
    # rasterize=True,
    features=["borders", "coastline"],
)
