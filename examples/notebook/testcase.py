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
from matplotlib.colors import BoundaryNorm
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np
import jax
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

mesh = tf.mesh.SphericalCentroidVoronoiTessellations.from_grist(
    grist_0d_file=grist_0d_file,
    grist_2d_file=grist_2d_file,
    n_dual=3,
    dev=dev
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
testcase = tf.test_case.swe.MountainZonalFlowTestCase(para)


# %%
equation = tf.dynamics.ShallowWaterEquation.from_model(
    coord=coord,
    para=para,
    method=method,
    form="vector_invariant",
    ode_kind="explicit",
)

# %%
equation.space_operators.prepare.pv.interp.pure_upwind(3)

# %%
equation.build_default()

# %%
time_manager = tf.util.datetime.DateTimeManager(
    start_time="2000-01-01T00:00:00",
    end_time="2000-01-21T00:00:00",
)

# %%
solver = tf.solver.ExRungeKutta3Solver(
    equation=equation,
    time_manager=time_manager,
    save_interval="6 hours",
    dt_dyn="300 seconds",
    dev=dev,
)

# %%
block_cls = equation.build_block_cls(coord=coord)

block = block_cls.from_testcase( 
    coord=coord, para=para, solver=solver, case=testcase
)

# %%
integrate_fn = solver.trajectory(start_with_input=True)

# %%
final, trajectory = integrate_fn(block)

final = jax.block_until_ready(final)
trajectory = jax.block_until_ready(trajectory)

# %%
diag_tools = tf.discretization.DiagnosticTools.from_model(
    solver=solver, 
    coord=coord,
    para=para,
    jit=True
)

trajectory_tools = tf.discretization.DiagnosticTools.from_model(
    solver=solver,
    coord=coord,
    para=para,
    trajectory=True,
    jit=True
)

# %%
final = diag_tools.cast_state(final)

diag_vars = diag_tools.compute(
    coord=coord,
    para=para,
    state=final,
    static=block.static,
    equaux=block.equaux,
    to_ugrid=True,
    use_dual=True
)

# %%
# %matplotlib inline

_, da = diag_vars["gz"].to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="inverse_distance_weighted",
    with_reshaped_da=True
)

lon_np = da["face_lon"]
lat_np = da["face_lat"]
data = da.data

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

_, da = diag_vars["u"].to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="inverse_distance_weighted",
    with_reshaped_da=True
)

lon_np = da["face_lon"]
lat_np = da["face_lat"]
data = da.data

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
trajectory = trajectory_tools.cast_state(trajectory)

trajectory_vars = trajectory_tools.compute_along_trajectory(
    coord=coord,
    para=para,
    trajectory=trajectory,
    static=block.static,
    equaux=block.equaux,
    to_ugrid=True
)

# %%
# Increase animation embed limit (e.g., to 50 MB)
plt.rcParams["animation.embed_limit"] = 200

# Get number of time steps
n_frames = trajectory_vars["gz"].data.shape[0]

fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")

_, da = trajectory_vars["gz"].to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="inverse_distance_weighted",
    with_reshaped_da=True,
)
lon_np = np.asarray(da["face_lon"])
lat_np = np.asarray(da["face_lat"])
extent = [lon_np.min(), lon_np.max(), lat_np.min(), lat_np.max()]

img = ax.imshow(
    da.isel(time=0).data,
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
cbar.set_label(f"{da.attrs['name']} at primal cell ({da.attrs['units']})")
title = ax.set_title("Trajectory Step: 0")

# Initial wind (quiver, updatable)
_, u_da = trajectory_vars["u"].to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="nearest_neighbor",
    with_reshaped_da=True,
)
_, v_da = trajectory_vars["v"].to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="nearest_neighbor",
    with_reshaped_da=True,
)
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
    img.set_data(da.isel(time=frame).data)
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
_, u_da = diag_vars["u"].to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="nearest_neighbor",
    with_reshaped_da=True,
)
_, v_da = diag_vars["v"].to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="nearest_neighbor",
    with_reshaped_da=True,
)

lon = u_da["face_lon"]
lat = u_da["face_lat"]
u = u_da.data
v = v_da.data
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
diag_vars["dundt"].to_uxarray().plot(
    height=500,
    width=1000,
    projection=ccrs.Robinson(),
    backend="matplotlib",
    pixel_ratio=0.5,
    cmap="plasma",
    # rasterize=True,
    features=["borders", "coastline"],
)
