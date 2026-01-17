from loguru import logger
import jax
import time
import datetime
import jax.numpy as jnp
import terraflow as tf
from terraflow.util.datetime import DateTimeManager
import terraflow.util.history as history
import terraflow.util.device as device
from terraflow.discretization import DiagnosticTools

grist_path = "../../data/GRIST_INPUT/uniform-pl-20211229/"
grist_0d_file = grist_path + "grist.grid_file.g6.ccvt.0d.nc"
grist_2d_file = grist_path + "grist.grid_file.g6.ccvt.2d.nc"


dev = device.CPU(float_type=jnp.float64, int_type=jnp.int64)


mesh = tf.mesh.OrthogonalDualMesh.from_grist(
    grist_0d_file, grist_2d_file, n_dual=3, dev=dev
)

vert = tf.vert_coord.OneLayerVertCoordinate()

layout = tf.layout.orthogonal_dual_mesh.TriSKBasedCGridStaggeringLayout()

coord = tf.layout.CoordinateSystem(hori=mesh, vert=vert, layout=layout)

para = tf.planet.PlanetParameterSet.from_si(
    planet="earth", component="atmosphere", dev=dev
)

method = tf.operator.FiniteVolumeMethod()

testcase = tf.test_case.swe.MountainZonalFlowTestCase(para)

state_cls, static_cls = tf.dynamics.ShallowWaterVariables.from_model(coord)

state = state_cls.from_testcase(coord, para, case=testcase)

static = static_cls.from_testcase(coord, para, case=testcase)


time_manager = DateTimeManager(
    start_time="0001-01-01T00:00:00",
    end_time="0001-01-21T00:00:00",
    save_interval="6 hours",
)


equation = tf.dynamics.ShallowWaterEquation.from_model(
    coord=coord,
    para=para,
    method=method,
    static=static,
    form="vector_invariant",
    ode_kind="explicit",
)

equation.dhdt.div_mfx.flux_recon_at_primal_edge.central_difference()
equation.dundt.coriolis.flux_recon_at_primal_edge.central_difference()
equation.dundt.coriolis.pv_at_dual_edge.interp.pure_upwind(3, use_mbs=True)


diag_tool = DiagnosticTools.from_equation(
    equation=equation, jit=True
)

h0 = history.UnstructuredHistory(
    coord=coord,
    time_manager=time_manager,
    file_path="terraflow.nc",
    diag_tool=diag_tool,
).setup(state, static)


solver = tf.solver.ExRungeKutta3Solver(
    equation=equation,
    dt=datetime.timedelta(seconds=300.0),
    time_manager=time_manager,
    dev=dev,
)


integrate_fn = jax.jit(solver.io_callback(
    equation=equation,
    history_file=h0,
    output_fields=True,
))

logger.info("Start integrating...")
t0 = time.perf_counter()
final = integrate_fn(state)
jax.block_until_ready(final)
t1 = time.perf_counter()
logger.success(f"Integration took {t1 - t0} seconds.")
