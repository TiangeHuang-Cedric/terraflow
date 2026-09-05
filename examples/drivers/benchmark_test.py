#!/usr/bin/env python3

import time
import jax
import terraflow.core as tf

grist_path = "../../data/GRIST_INPUT/uniform-pl-20211229/"
grist_0d_file = grist_path + "grist.grid_file.g4.ccvt.0d.nc"
grist_2d_file = grist_path + "grist.grid_file.g4.ccvt.2d.nc"

dev = tf.util.device.CPU(
    float_type="r8",
    int_type="i4",
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
    run_time="1 hours",
    dt_dyn="300 seconds",
    dt_adv="300 seconds",
    save_interval="300 seconds"
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
    ideal_dry_core=False,
    restart=False
)

integrate_fn = tf.model.compiled_with_model(
    model=model,
    fun=model.integrate(),
)

jax.profiler.start_trace("../../.profile_data", create_perfetto_trace=True)

if dev.is_root():
    tf.util.log.logger.info("Start integrate.")

t0 = time.perf_counter()
with jax.profiler.StepTraceAnnotation("integrate", step_num=0):
    final = integrate_fn(model)
    final = tf.model.block_until_ready(final)
t1 = time.perf_counter()
tf.util.log.logger.info(f"Time cost: {(t1 - t0):.6f} s")

if dev.is_root():
    tf.util.log.logger.success("Successfully run integration.")

jax.profiler.stop_trace()
