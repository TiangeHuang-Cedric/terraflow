#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import terraflow.core as tf

tf.util.datetime.perf_start("program")

grist_path = "../../data/GRIST_INPUT/uniform-pl-20211229/"
grist_0d_file = grist_path + "grist.grid_file.g4.ccvt.0d.nc"
grist_2d_file = grist_path + "grist.grid_file.g4.ccvt.2d.nc"

dev = tf.util.device.CPU(float_type=jnp.float64, int_type=jnp.int64)

mesh = tf.mesh.SphericalCentroidVoronoiTessellations.from_grist(
    grist_0d_file=grist_0d_file,
    grist_2d_file=grist_2d_file,
    n_dual=3,
    dev=dev
)

vert = tf.vert_coord.HybridVertCoordinate.test_l26(dev=dev)

coord = tf.layout.CoordinateSystem(hori=mesh, vert=vert)

para = tf.planet.PlanetParameterSet.from_si(
    planet="earth", component="atmosphere", dev=dev
)

method = tf.operator.FiniteVolumeMethod()

testcase = tf.test_case.baroclinic.MountainWaveTestCase(para)

equation = tf.dynamics.DryHydrostaticEquation.from_model(
    coord=coord,
    para=para,
    method=method,
    form="vector_invariant",
    ode_kind="explicit",
)

equation.space_operators.prepare.pv.interp.pure_upwind(3)
equation.space_operators.dptdt.div_mft.adv_scheme.upwind(3)
equation.space_operators.dptdt.adv_vert.adv_scheme.upwind(3)

equation.build_default()

time_manager = tf.util.datetime.DateTimeManager(
    start_time="2000-01-01T00:00:00",
    end_time="2000-01-31T00:00:00",
)

solver = tf.solver.ExRungeKutta3Solver(
    equation=equation,
    time_manager=time_manager,
    save_interval="6 hours",
    dt_dyn="300 seconds",
    dev=dev,
)

block_cls = equation.build_block_cls(coord=coord)

block = block_cls.from_testcase(
    coord=coord, para=para, solver=solver, case=testcase
)

history = tf.util.history.UnstructuredHistoryset(
    solver=solver,
    coord=coord,
    equation=equation,
    case_name="terraflow" + "." + mesh.glevel,
    output_h1=False,
    output_h2=False,
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
