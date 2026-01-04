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
#     display_name: terraflow
#     language: python
#     name: python3
# ---

# %%
import jax.numpy as jnp

import terraflow.util.device as device

grist_path = "../../data/GRIST_INPUT/uniform-pl-20211229/"
grist_0d_file = grist_path + "grist.grid_file.g6.ccvt.0d.nc"
grist_2d_file = grist_path + "grist.grid_file.g6.ccvt.2d.nc"

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
testcase = terraflow.test_case.swe.MountainZonalFlowTestCase(para)

state_cls, static_cls = terraflow.dynamics.ShallowWaterVariables.from_model(coord)

state = state_cls.from_testcase(coord, para, case=testcase)

static = static_cls.from_testcase(coord, para, case=testcase)


# %%
equation = terraflow.dynamics.ShallowWaterEquation.from_model(
    coord=coord,
    para=para,
    method=method,
    static=static,
    form="vector_invariant",
    ode_kind="explicit",
)

# %%
equation.dhdt.div_mfx.flux_recon_at_primal_edge.central_difference()
equation.dundt.coriolis.flux_recon_at_primal_edge.central_difference()
equation.dundt.coriolis.pv_at_dual_edge.interp.upwind(3, use_mbs=True)

# %%
import datetime

solver = terraflow.solver.ExRungeKutta3Solver(
    equation=equation,
    dt=datetime.timedelta(seconds=60.0),                                                     
    save_interval=datetime.timedelta(seconds=3600.0*6),
    dev=dev,
)

# %%
import jax

integrate_fn = jax.jit(solver.trajectory(
    equation=equation,
    start_time="0001-01-01T00:00:00",
    end_time="0001-02-21T00:00:00",
    start_with_input=True,
    output_fields=True,
))

# %%
final, trajectory = integrate_fn(state)

# %%
diag_tools = terraflow.discretization.DiagnosticToolsBuilder.from_equation(equation, jit=True)

trajectory_tools = terraflow.discretization.DiagnosticToolsBuilder.from_equation(equation, trajectory=True, jit=True)

# %%
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

_, da = diag_tools.thickness_at_primal_cell(final).to_structured_grid(
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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm 
import cartopy.crs as ccrs
import cartopy.feature as cfeature

_, ds = diag_tools.uv_at_primal_cell(final).to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="inverse_distance_weighted",
    with_reshaped_ds=True
)

lon_np = ds["face_lon"]
lat_np = ds["face_lat"]
data = ds["u_pc"].data

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
ax.set_title(f"Remapped {ds['u_pc'].attrs['name']} on Lat-Lon Grid")
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
cbar.set_label(f"{ds['u_pc'].attrs['name']} at primal cell ({ds['u_pc'].attrs['units']})")
plt.tight_layout()

# %%
from matplotlib import animation
from IPython.display import HTML

# Increase animation embed limit (e.g., to 50 MB)
plt.rcParams["animation.embed_limit"] = 200

# Get number of time steps
n_frames = trajectory.to_arrays().h_pc.shape[0]

fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")

trajectory_h_pc = trajectory_tools.thickness_at_primal_cell(trajectory)
_, da = trajectory_h_pc.to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="inverse_distance_weighted",
    with_reshaped_da=True,
)
lon_np = np.asarray(da["face_lon"])
lat_np = np.asarray(da["face_lat"])
extent = [lon_np.min(), lon_np.max(), lat_np.min(), lat_np.max()]

img = ax.imshow(
    da.isel(n_time=0).data,
    origin="lower",
    extent=extent,
    transform=ccrs.PlateCarree(),
    cmap="plasma",
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
trajectory_uv_pc = trajectory_tools.uv_at_primal_cell(trajectory)
_, ds = trajectory_uv_pc.to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="nearest_neighbor",
    with_reshaped_ds=True,
)
u_0 = ds["u_pc"].isel(n_time=0).data
v_0 = ds["v_pc"].isel(n_time=0).data

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
    img.set_data(da.isel(n_time=frame).data)
    u = ds["u_pc"].isel(n_time=frame).data[::stride, ::stride]
    v = ds["v_pc"].isel(n_time=frame).data[::stride, ::stride]
    qv.set_UVC(u, v)
    title.set_text(f"Trajectory Step: {frame}")
    return img, qv, title

# Cartopy axes generally do not support blitting; keep blit=False
anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
anim.save("thickness_uv.gif", writer="pillow", dpi=120, fps=5)
plt.close(fig)
HTML(anim.to_jshtml())

# %%
from matplotlib import animation
from IPython.display import HTML

# Increase animation embed limit (e.g., to 50 MB)
plt.rcParams["animation.embed_limit"] = 200

# Get number of time steps
n_frames = trajectory.to_arrays().h_pc.shape[0]

fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")

trajectory_data = trajectory_tools.pv_at_dual_cell(trajectory)
_, da = trajectory_data.to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="inverse_distance_weighted",
    with_reshaped_da=True,
)
lon_np = np.asarray(da["face_lon"])
lat_np = np.asarray(da["face_lat"])
extent = [lon_np.min(), lon_np.max(), lat_np.min(), lat_np.max()]

img = ax.imshow(
    da.isel(n_time=0).data,
    origin="lower",
    extent=extent,
    transform=ccrs.PlateCarree(),
    cmap="plasma",
    interpolation="nearest",
    animated=True,
)

cbar = plt.colorbar(img, ax=ax, orientation="vertical", pad=0.04)
cbar.set_label(f"{da.attrs['name']} at primal cell ({da.attrs['units']})")
title = ax.set_title("Trajectory Step: 0")

def update(frame):
    img.set_data(da.isel(n_time=frame).data)
    title.set_text(f"Trajectory Step: {frame}")
    return img, title

# Cartopy axes generally do not support blitting; keep blit=False
anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
# anim.save(f"{da.attrs['name']}.mp4", writer="ffmpeg", dpi=120, fps=5)
plt.close(fig)
HTML(anim.to_jshtml())

# %%
# %matplotlib inline
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import cartopy.feature as cfeature
import numpy as np

_, ds = diag_tools.uv_at_primal_cell(final).to_structured_grid(
    east_west_resolution=1.0,
    south_north_resolution=1.0,
    remap_method="nearest_neighbor",
    with_reshaped_ds=True,
)

lon = ds["face_lon"]
lat = ds["face_lat"]
u = ds["u_pc"]
v = ds["v_pc"]
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
    regrid_shape=20, # sample every 20 points, to avoid overcrowding
    scale=500, # ajust arrow length
    width=0.003, # ajust arrow width
    color="black",
)
plt.tight_layout()

plt.show()

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
    cmap="plasma",
    levels=20,
    alpha=0.7
)
plt.colorbar(filled, ax=ax, orientation="vertical", pad=0.04, label="Wind Speed (m/s)")
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
import cartopy.crs as ccrs

diag_tools.dundt(final).to_uxarray().plot(
    height=500,
    width=1000,
    projection=ccrs.Robinson(),
    backend="matplotlib",
    pixel_ratio=0.5,
    cmap="plasma",
    # rasterize=True,
    features=["borders", "coastline"],
)
