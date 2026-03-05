from loguru import logger
import time
import datetime
import jax
import jax.numpy as jnp
import terraflow as tf

t0 = time.perf_counter()

grist_path = "../../data/GRIST_INPUT/uniform-pl-20211229/"
grist_0d_file = grist_path + "grist.grid_file.g6.ccvt.0d.nc"
grist_2d_file = grist_path + "grist.grid_file.g6.ccvt.2d.nc"

dev = tf.util.device.CPU(float_type=jnp.float64, int_type=jnp.int64)

mesh = tf.mesh.OrthogonalDualMesh.from_grist(
    grist_0d_file=grist_0d_file,
    grist_2d_file=grist_2d_file,
    n_dual=3,
    dev=dev
)

vert = tf.vert_coord.OneLayerVertCoordinate()

coord = tf.layout.CoordinateSystem(hori=mesh, vert=vert)

para = tf.planet.PlanetParameterSet.from_si(
    planet="earth", component="atmosphere", dev=dev
)

method = tf.operator.FiniteVolumeMethod()

testcase = tf.test_case.swe.MountainZonalFlowTestCase(para)

time_manager = tf.util.datetime.DateTimeManager(
    start_time="0001-01-01T00:00:00",
    end_time="0001-01-21T00:00:00",
    save_interval="6 hours",
    dt_dyn=datetime.timedelta(seconds=300.0),
)

equation = tf.dynamics.ShallowWaterEquation.from_model(
    coord=coord,
    para=para,
    method=method,
    form="vector_invariant",
    ode_kind="explicit",
)

equation.aux.dmg.recon_at_edge.central_difference()
equation.dundt.coriolis.pv.interp.pure_upwind(3)

equation.build_default()

block = equation.build_block_from_testcase(
    coord=coord, para=para, case=testcase
)

block = equation.prepare(block)

history = tf.util.history.UnstructuredHistoryset(
    coord=coord,
    time_manager=time_manager,
    case_name="terraflow.swm",
    output_h1=False,
    output_h2=False,
).setup(block)

solver = tf.solver.ExRungeKutta3Solver(
    equation=equation,
    time_manager=time_manager,
    dev=dev,
)

integrate_fn = jax.jit(solver.io_callback(history=history,))

logger.info("Start integrating...")
t1 = time.perf_counter()
final = integrate_fn(block)
jax.block_until_ready(final)
t2 = time.perf_counter()
logger.success(f"Integration time cost {t2 - t1} seconds.")

logger.success(f"Total time cost {t2 - t0} seconds.")
