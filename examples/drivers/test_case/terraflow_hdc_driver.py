#!/usr/bin/env python3

import jax.numpy as jnp
import terraflow.core as tf

tf.util.datetime.perf_start("program")

grist_path = "../../../data/GRIST_INPUT/uniform-pl-20211229/"
grist_0d_file = grist_path + "grist.grid_file.g4.ccvt.0d.nc"
grist_2d_file = grist_path + "grist.grid_file.g4.ccvt.2d.nc"

dev = tf.util.device.CPU(
    float_type=jnp.float64, 
    int_type=jnp.int64,
    use_async_io=False, 
    proc_io_stride=0
)

para = tf.planet.PlanetParameterSet.from_si(
    planet="earth", component="atmosphere", dev=dev
)

vert_coord = tf.vert_coord.HybridVertCoordinate.test_l26(dev=dev)

mesh = tf.mesh.SCVTMesh.from_grist(
    mesh_0d_file=grist_0d_file,
    mesh_2d_file=grist_2d_file,
    para=para,
    vert_coord=vert_coord,
    hori_hw=3,
    lev_hw=3,
    dev=dev
)

test_case = tf.test_case.baroclinic.MountainWaveTestCase(para)

config = tf.model.UserConfig(
    pv_adv_scheme="upwind",
    upwind_order_pv=3,
    pt_adv_scheme="upwind",
    upwind_order=3,
    run_time="30 days",
    dt_dyn="300 seconds",
    dt_adv="300 seconds",
    save_interval="6 hours"
)

model = tf.model.build_hydrostatic_model(
    config=config,
    para=para,
    mesh=mesh,
    vert_coord=vert_coord,
    method="finite_volume_method",
    form="vector_invariant",
    staggered_kind="triskc",
    test_case=test_case,
    dev=dev,
    ideal_dry_core=True,
    restart=False
)

history = model.build_history(
    mode="scvt",
    case_name="terraflow.hdc" + ".g" + str(mesh.glevel),
    output_h1=False,
    output_h2=False,
    h2_opt={
        "hori_regrid_method": "conserve_1st",
        "hori_regrid_backend": "esmf",
        "vert_interp": False,
    }
)

integrate_fn = tf.model.compiled_with_model(
    model=model,
    fun=model.io_callback(history=history),
)

tf.util.datetime.perf_start("integration")

final = integrate_fn(model)
tf.model.block_until_ready(final)

tf.util.datetime.perf_end("integration")

if dev.is_root():
    tf.util.post_process.CDOPostProcesser.remap_history(
        history=history,
        res=1.0,
        remap=True,
        use_weight=True,
    )

tf.util.datetime.perf_end("program")
