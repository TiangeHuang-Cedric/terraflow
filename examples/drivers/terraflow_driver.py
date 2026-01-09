from terraflow.discretization import DiagnosticTools
import jax
import datetime
import jax.numpy as jnp
import terraflow
from terraflow.util.datetime import DateTimeManager
import terraflow.util.history as history
import terraflow.util.device as device

grist_path = "../../data/GRIST_INPUT/uniform-pl-20211229/"
grist_0d_file = grist_path + "grist.grid_file.g4.ccvt.0d.nc"
grist_2d_file = grist_path + "grist.grid_file.g4.ccvt.2d.nc"


dev = device.CPU(float_type=jnp.float64, int_type=jnp.int64)


mesh = terraflow.mesh.OrthogonalDualMesh.from_grist(
    grist_0d_file, grist_2d_file, n_dual=3, dev=dev
)

vert = terraflow.vert_coord.OneLayerVertCoordinate()

layout = terraflow.layout.orthogonal_dual_mesh.TriSKBasedCGridStaggeringLayout()

coord = terraflow.layout.CoordinateSystem(hori=mesh, vert=vert, layout=layout)

para = terraflow.planet.PlanetParameterSet.from_si(
    planet="earth", component="atmosphere", dev=dev
)

method = terraflow.discretization.FiniteVolumeMethod()


testcase = terraflow.test_case.swe.MountainZonalFlowTestCase(para)

state_cls, static_cls = terraflow.dynamics.ShallowWaterVariables.from_model(
    coord)

state = state_cls.from_testcase(coord, para, case=testcase)

static = static_cls.from_testcase(coord, para, case=testcase)


time_manager = DateTimeManager(
    start_time="0001-01-01T00:00:00",
    end_time="0001-01-21T00:00:00",
    save_interval="6 hours",
)


equation = terraflow.dynamics.ShallowWaterEquation.from_model(
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

h0 = history.Unstructured2StructuredHistory(
    coord=coord,
    time_manager=time_manager,
    file_path="terraflow.nc",
    diag_tool=diag_tool,
).setup(state, static)


solver = terraflow.solver.ExRungeKutta3Solver(
    equation=equation,
    dt=datetime.timedelta(seconds=120.0),
    save_interval=datetime.timedelta(seconds=3600.0*6),
    dev=dev,
)


integrate_fn = jax.jit(solver.io_callback(
    equation=equation,
    time_manager=time_manager,
    history_file=h0,
    output_fields=True,
))


final = integrate_fn(state)
