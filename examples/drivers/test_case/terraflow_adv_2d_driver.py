#!/usr/bin/env python3

import terraflow.core as tf

tf.util.datetime.perf_start("program")

grist_path = "../../../data/GRIST_INPUT/uniform-pl-20211229/"
grist_0d_file = grist_path + "grist.grid_file.g6.ccvt.0d.nc"
grist_2d_file = grist_path + "grist.grid_file.g6.ccvt.2d.nc"

dev = tf.util.device.CPU(
    backend="jax",
    float_type="r8",
    int_type="i4",
    use_async_io=False,
    proc_io_stride=0
)

para = tf.planet.PlanetParameterSet.from_si(
    planet="earth", component="atmosphere", dev=dev
)

test_case = tf.test_case.adv.SolidRotationTestCase(para)

mesh = tf.mesh.SCVTMesh.from_grist(
    mesh_0d_file=grist_0d_file,
    mesh_2d_file=grist_2d_file,
    para=para,
    hori_hw=3,
    dev=dev
)

config = tf.model.UserConfig(
    bg_adv_scheme="upwind",
    upwind_order_bg=3,
    adv_scheme="upwind",
    upwind_order=3,
    run_time="24 days",
    dt_dyn="300 seconds",
    dt_adv="600 seconds",
    save_interval="6 hours"
)

model = tf.model.build_adv_2d_model(
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

history = model.build_history(
    mode="scvt",
    case_name="terraflow.adv" + ".g" + str(mesh.glevel),
    output_h1=False,
    output_h2=False,
)

integrate_fn = model.io_callback(history=history, with_jit=True)

tf.util.datetime.perf_start("integration")

final = integrate_fn(model)
dev.block_until_ready(final)

tf.util.datetime.perf_end("integration")

if dev.is_root():
    tf.util.post_process.CDOPostProcesser.remap_history(
        history=history,
        res=1.0,
        remap=True,
        use_weight=True,
    )

tf.util.datetime.perf_end("program")
