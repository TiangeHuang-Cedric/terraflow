#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import terraflow.core as tf

tf.util.datetime.perf_start("program")

grist_path = "../../../data/GRIST_INPUT/uniform-pl-20211229/"
grist_0d_file = grist_path + "grist.grid_file.g6.ccvt.0d.nc"
grist_2d_file = grist_path + "grist.grid_file.g6.ccvt.2d.nc"

dev = tf.util.device.CPU(float_type=jnp.float64, int_type=jnp.int64)

para = tf.planet.PlanetParameterSet.from_si(
    planet="earth", component="atmosphere", dev=dev
)

mesh = tf.mesh.SphericalCentroidVoronoiTessellations.from_grist(
    grist_0d_file=grist_0d_file,
    grist_2d_file=grist_2d_file,
    n_dual=3,
    dev=dev
)

vert = tf.vert_coord.HybridVertCoordinate.cam_l30(dev=dev)

coord = tf.layout.CoordinateSystem(hori=mesh, vert=vert)

method = tf.operator.FiniteVolumeMethod()

config = tf.util.model_config.ModelConfig(
    pv_adv_scheme="pure_upwind",
    upwind_order_pv=3,
    pt_adv_scheme="upwind",
    upwind_order=3,
    use_smag_damp=True,
    use_sponge_layer=True,
)

equation = tf.dynamics.HydrostaticEquation.build_with_config(
    config=config,
    coord=coord,
    para=para,
    method=method,
    form="vector_invariant",
    ode_kind="explicit",
    ideal_dry_core=False,
)

static_reader = tf.readers.topo.ETOPO1Reader(
    topo_file="../../../data/topo/ETOPO1_Ice_g_gmt4.grd",
    para=para,
    cfg=tf.readers.TopographyReaderConfig(
        regrid_type="fill",
        fill_backend="scipy",
        fill_split_with_chunks=True,
        fill_lon_chunk_size=360,
        fill_lat_chunk_size=180,
        use_zs_grad_filter=True,
        zs_grad_filter_type="laplacian2",
    )
)

bkg_reader = tf.readers.bkg.ERA5Reader(
    bkg_file="../../../data/bkg/era5_2022060100.nc",
    para=para,
    cfg=tf.readers.BackgroundReaderConfig(
        hori_interp_backend="esmf",
        hori_interp_method="conserve_1st",
        interp_at_midpoint=True,
        vert_interp_backend="jax",
        vert_interp_type="native",
    )
)

time_manager = tf.util.datetime.DateTimeManager.from_reader(
    reader=bkg_reader,
    run_time="12 days",
)

solver = tf.solver.ExRungeKutta3Solver(
    equation=equation,
    time_manager=time_manager,
    save_interval="6 hours",
    dt_dyn="300 seconds",
    dt_adv="600 seconds",
    dev=dev,
)

block_cls = equation.build_block_cls(coord=coord)

block = block_cls.from_reader(
    static_reader=static_reader,
    bkg_reader=bkg_reader,
    coord=coord,
    para=para,
    solver=solver,
)

era5_1deg_lon = jnp.arange(0, 359 + 1.0, 1.0, dtype=dev.float_type)
era5_1deg_lat = jnp.arange(90, -90 - 1.0, -1.0, dtype=dev.float_type)

history = tf.util.history.UnstructuredHistoryset(
    solver=solver,
    coord=coord,
    equation=equation,
    case_name="terraflow" + "." + mesh.glevel,
    output_h1=False,
    output_h2=False,
    h2_opt={
        "hori_regrid_method": "conserve_1st",
        "hori_regrid_backend": "esmf",
        "vert_interp": True,
        "lon": era5_1deg_lon,
        "lat": era5_1deg_lat,
        "plev": bkg_reader.lev,
        "plev_unit": "Pa",
    }
).setup(block)

integrate_fn = tf.solver.compiled_with_block(
    block=block,
    fun=solver.io_callback(history=history),
)

tf.util.datetime.perf_start("integration")

final = integrate_fn(block)
jax.block_until_ready(final)

tf.util.datetime.perf_end("integration")

tf.util.post_process.CDOPostProcesser.remap_history(
    history=history,
    res=1.0,
    remap=True,
    use_weight=True,
)

tf.util.datetime.perf_end("program")
