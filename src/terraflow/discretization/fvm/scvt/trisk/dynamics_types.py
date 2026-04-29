from typing import (
    Self,
    cast,
    Dict,
    Optional,
    TypeVar,
    Generic,
    Union,
)
import equinox as eqx
import jax.numpy as jnp

from terraflow.util import typing, formula, log
from terraflow.planet import PlanetParameterSet
from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.layout.abstract import (
    CoordinateSystem,
    Basis,
    MeshEntity,
    VertLoc,
)
from terraflow.layout.field import ArrayField, StaggeredFieldMetaData, NumericMetaData
from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
from terraflow.test_case.abstract import AbstractTestCase
from terraflow.test_case.swe import (
    AbstractShallowWaterEquationTestCase,
    MountainZonalFlowTestCase
)
from terraflow.test_case.baroclinic import (
    AbstractBaroclinicEquationTestCase,
    SteadyStateTestCase,
    MountainWaveTestCase,
    RossbyHaurwitzWaveTestCase
)
from terraflow.test_case.adv import (
    AbstractAdvectionEquation2DTestCase,
    SolidRotationTestCase
)
from terraflow.dynamics.adv.types import AbstractAdvectionEquationSet2D
from terraflow.solver.abstract import AbstractSolver
from terraflow.discretization.abstract import (
    AbstractEquationMode,
    AbstractEquationSet,
    AbstractBlock,
    AbstractFields,
    AbstractEquationTermAuxiliaryDataset
)
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at
from terraflow.discretization.fvm.scvt.trisk.tracer import Tracer, default_tracer_meta
from terraflow.discretization.fvm.scvt.trisk.common.pgf.types import PressureGradientForceDataset
from terraflow.discretization.fvm.scvt.trisk.common.coriolis.types import CoriolisForceAtEdgeData
from terraflow.discretization.fvm.scvt.trisk.common.adv.types import AdvectionAuxiliaryDataset
from terraflow.discretization.fvm.scvt.trisk.common.adv.batch import AdvBatchType

Int = typing.Int
Float = typing.Float
Array = typing.Array
PyTree = typing.PyTree

FDT = ArrayField[StaggeredFieldMetaData]
NDT = ArrayField[NumericMetaData]

VM = TypeVar("VM", OneLayerVertCoordinate, HybridVertCoordinate)
VLO = TypeVar("VLO", OneLayerVertCoordLayout, HybridVertCoordLayout)
ES = TypeVar("ES", bound=AbstractEquationSet)

CoordSys = CoordinateSystem[
    SphericalCentroidVoronoiTessellations,
    VM,
    TriSKBasedCGridStaggeringLayout,
    VLO
]

CoordSysOneLayer = CoordinateSystem[
    SphericalCentroidVoronoiTessellations,
    OneLayerVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    OneLayerVertCoordLayout
]

CoordSysHybrid = CoordinateSystem[
    SphericalCentroidVoronoiTessellations,
    HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    HybridVertCoordLayout
]

NLEVP = "ilev"
NLEV = "lev"
NCELL = "cell"
NEDGE = "edge"
NVTX = "vtx"

Element = Union[
    "State",
    "Tendency",
    "Static",
    "Auxiliary",
    AdvBatchType,
    dict[str, AdvBatchType],
]


class Block(
    AbstractBlock[
        SphericalCentroidVoronoiTessellations,
        VM,
        TriSKBasedCGridStaggeringLayout,
        VLO,
        "State", "Tendency", "Static", "Auxiliary", AdvBatchType, Tracer, "EquationTermAuxDataset"
    ],
    Generic[VM, VLO]
):

    @classmethod
    def from_model(
        cls,
        coord: CoordSys,
        para: PlanetParameterSet,
        solver: AbstractSolver[ES],
        tracers: Optional[Tracer] = None,
        test_case: Optional[AbstractTestCase] = None,
    ) -> Self:
        equation = solver.equation
        mode: AbstractEquationMode = equation.mode
        # ================================================
        state = State.from_coord(coord=coord, equation=equation)
        static = Static.from_coord(coord=coord, equation=equation)
        tend = Tendency.from_coord(coord=coord, equation=equation)
        aux = Auxiliary.from_coord(coord=coord, equation=equation)
        # ================================================
        # Adv init
        dt_dyn = solver.time_manager.dt_dyn
        dt_dyn_seconds = dt_dyn.total_seconds()
        if mode.is_barotropic():
            adv_batch_bg = AdvBatchType.empty(mode)
        elif mode.is_advection2d():
            adv_batch_bg = AdvBatchType.from_coord(
                name="bg",
                batch_hori_loc=MeshEntity.Cell,
                batch_vert_loc="",
                coord=coord,
                equation=equation,
                dt=dt_dyn_seconds,
                dts_substep=solver.calc_dts_substep(dt_dyn),
                dt_dyn=dt_dyn_seconds,
            )
        elif mode.is_baroclinic() or mode.is_advection3d():
            adv_batch_bg = AdvBatchType.from_coord(
                name="bg",
                batch_hori_loc=MeshEntity.Cell,
                batch_vert_loc=VertLoc.Cell,
                coord=coord,
                equation=equation,
                dt=dt_dyn_seconds,
                dts_substep=solver.calc_dts_substep(dt_dyn),
                dt_dyn=dt_dyn_seconds,
            )
        else:
            raise ValueError(
                f"Invalid equation mode {mode} for background batch metadata generation."
            )
        if mode.is_baroclinic():
            adv_batch_pt = AdvBatchType.from_coord(
                name="pt",
                batch_hori_loc=MeshEntity.Cell,
                batch_vert_loc=VertLoc.Cell,
                coord=coord,
                equation=equation,
                dt=dt_dyn_seconds,
                dts_substep=solver.calc_dts_substep(dt_dyn),
                dt_dyn=dt_dyn_seconds,
            )
        else:
            adv_batch_pt = AdvBatchType.empty(mode=mode)
        if tracers is not None:
            # Parse batch information from tracers_meta
            batch_names: list[str] = []
            batch_idx: dict[str, list[int]] = {}
            for itra, tracer_meta in enumerate(tracers.tracers_meta.values()):
                bname = tracer_meta.batch_name
                if bname in batch_names:
                    batch_idx[bname].append(itra)
                    continue
                batch_names.append(bname)
                batch_idx[bname] = [itra]

            # Initialize adv batches for tracers
            adv_batches: dict[str, AdvBatchType] = {}
            for bname in batch_names:
                dts_substep = solver.calc_dts_substep(tracers.batch_dts[bname])
                adv_batches[bname] = AdvBatchType.from_coord(
                    name=bname,
                    batch_hori_loc=MeshEntity.Cell,
                    batch_vert_loc=VertLoc.Cell if mode.is_baroclinic() or mode.is_advection3d() else "",
                    coord=coord,
                    equation=equation,
                    dt=tracers.batch_dts_seconds[bname],
                    dts_substep=dts_substep,
                    dt_dyn=dt_dyn.total_seconds(),
                    idx=jnp.array(batch_idx[bname])
                )
        else:
            # if tracers is not provided, initialize an empty adv_batches
            adv_batches = {}
            tracers = Tracer.empty(mode=mode)
        # ================================================
        return cls(
            mode=mode,
            old=jnp.array(0, dtype=jnp.int32),
            new=jnp.array(1, dtype=jnp.int32),
            _coord=coord,
            para=para,
            state=state,
            tend=tend,
            static=static,
            aux=aux,
            adv_batch_bg=adv_batch_bg,
            adv_batch_pt=adv_batch_pt,
            adv_batches=adv_batches,
            tracers=tracers,
            equaux=EquationTermAuxDataset.empty(),
            test_case=test_case,
        )

    @classmethod
    def from_testcase(
        cls,
        coord: CoordSys,
        para: PlanetParameterSet,
        solver: AbstractSolver[ES],
        *,
        case: Optional[AbstractTestCase] = None,
    ) -> Self:
        # =======================================================
        equation = solver.equation
        if equation.mode.is_baroclinic():
            # ==============================================
            # Baroclinic case
            if case is None:
                case = SteadyStateTestCase(para)
            elif not isinstance(case, AbstractBaroclinicEquationTestCase):
                raise ValueError(
                    "Invalid test case for baroclinic equation"
                )
            if isinstance(case, SteadyStateTestCase):
                block = baroclinic_steady_state_case_init_fn(
                    coord=coord,
                    para=para,
                    solver=solver,
                    test_case=case
                )
            elif isinstance(case, MountainWaveTestCase):
                block = baroclinic_mountain_wave_case_init_fn(
                    coord=coord,
                    para=para,
                    solver=solver,
                    test_case=case
                )
            elif isinstance(case, RossbyHaurwitzWaveTestCase):
                block = baroclinic_rossby_wave_case_init_fn(
                    coord=coord,
                    para=para,
                    solver=solver,
                    test_case=case
                )
            else:
                raise NotImplementedError(
                    f"Initialization for case {type(case)} is not implemented yet."
                )
            # ==============================================
        elif equation.mode.is_advection():
            # ==============================================
            # Advection case
            if isinstance(equation, AbstractAdvectionEquationSet2D):
                if case is None:
                    case = SolidRotationTestCase(para)
                elif not isinstance(case, AbstractAdvectionEquation2DTestCase):
                    raise ValueError(
                        "Invalid test case for advection equation"
                    )
                block = advection2d_case_init_fn(
                    coord=coord,
                    para=para,
                    solver=solver,
                    test_case=case
                )
            else:
                raise NotImplementedError(
                    "Unvalid equation type for advection testcase initialization"
                )
            # ==============================================
        elif equation.mode.is_barotropic():
            # ==============================================
            # Barotropic case
            if case is None:
                case = MountainZonalFlowTestCase(para)
            elif not isinstance(case, AbstractShallowWaterEquationTestCase):
                raise ValueError(
                    "Invalid test case for shallow water equation"
                )
            block = barotropic_case_init_fn(
                coord=coord,
                para=para,
                solver=solver,
                test_case=case
            )
            # ==============================================
        else:
            raise ValueError(
                f"Invalid equation mode for testcase initialization. Invalid mode={equation.mode}"
            )
        log.logger.success("Initialized block from testcase.")
        # =======================================================
        return solver.equation.prepare_block(block)

    def testcase_set_un(self, itime: Int[Array, ""] | int) -> Self:
        if self.test_case is None:
            raise ValueError(
                "Test case is not provided for the block. Cannot set un.")
        else:
            if isinstance(self.test_case, AbstractAdvectionEquation2DTestCase):
                un = init_un_in_barotropic_case(
                    coord=self.coord,
                    mode=self.mode,
                    test_case=self.test_case
                )
                coord = cast(CoordSys, self.coord)
                return tree_at(
                    coord=coord,
                    where=lambda b: b.state.un,
                    pytree=self,
                    replace=un,
                    at_dims=0,
                    at_idxes=itime
                )
            else:
                raise NotImplementedError(
                    f"Initialization of un for test case {type(self.test_case)} is not implemented yet."
                )

    def diagnose_uv(self, itime_str: str) -> Self:
        # =======================================================
        # Diagnose u and v from un
        itime = self.get_time_idx(itime_str=itime_str)
        un = self.state.un.unwrap()[itime, :, :]
        uv = self.coord.mesh.reconstruct_vector(
            normal_u_at_edge=un,
        )
        # =======================================================
        return tree_at(
            coord=cast(CoordSys, self.coord),
            where=lambda b: (b.aux.u.data, b.aux.v.data),
            pytree=self,
            replace=(uv["u"], uv["v"]),
        )

    def diagnose_total_mass(
        self,
        itime_str: str
    ) -> Float[Array, ""]:
        # =============================================
        itime = self.get_time_idx(itime_str=itime_str)
        mesh = self.coord.mesh
        dmg = self.state.dmg.unwrap()[itime, :, :]
        primal_area = mesh.primal.cell.area
        # =============================================
        tm = jnp.array(0.0, dtype=jnp.float64)
        return tm + jnp.sum(dmg * primal_area[None, :])

    def diagnose_total_water_vapor(
        self,
        itime: Int[Array, ""] | int
    ) -> Float[Array, ""]:
        # TODO: When processing moist dycore, this should be modified
        raise NotImplementedError

    def diagnose_total_energy(
        self,
        itime_str: str
    ) -> tuple[
        Float[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
        Float[Array, ""]
    ]:
        # =============================================
        float_type = self.dev.float_type
        itime = self.get_time_idx(itime_str=itime_str)
        mesh = self.coord.mesh
        edp_len = mesh.primal.edge.length
        edd_len = mesh.dual.edge.length
        gzs = self.static.gzs.unwrap()
        dmg = self.state.dmg.unwrap()[itime, :, :]
        dmg_e = self.state.dmg_e.unwrap()[itime, :, :]
        un = self.state.un.unwrap()[itime, :, :]
        tv = self.aux.tv.unwrap()
        g = self.para.gravity.unwrap()
        cpd = self.para.cpd.unwrap()
        # Weighted area
        primal_area = mesh.primal.cell.area
        kite_area = edp_len * edd_len * 0.5
        # =============================================
        te_ke = jnp.array(0.0, dtype=float_type)
        te_ie = jnp.array(0.0, dtype=float_type)
        te_pe = jnp.array(0.0, dtype=float_type)
        # =============================================
        # Total energy due to kinetic energy
        # TODO: nh dynamics kinetic energy
        # Kinetic energy at edge
        te_ke = te_ke + jnp.sum(0.5 * un**2 * dmg_e * kite_area[None, :])
        # =============================================
        # Total energy due to internal energy
        if self.mode.is_baroclinic():
            te_ie = te_ie + jnp.sum(dmg * cpd * tv * primal_area[None, :])
        # =============================================
        # Total energy due to potential energy
        if self.mode.is_baroclinic():
            mgs = self.state.mgs.unwrap()[itime, :]
            te_pe = te_pe + jnp.sum(gzs * mgs * primal_area)
        else:
            te_pe = te_pe + jnp.sum(
                (dmg.squeeze()**2 * g * 0.5 + dmg.squeeze()*gzs) * primal_area,
            )
        # =============================================
        return te_ke + te_ie + te_pe, te_ke, te_ie, te_pe

    def diagnose_total_potential_enstrophy(self, itime_str: str) -> Float[Array, ""]:
        # =======================================================
        mesh = self.coord.mesh
        itime = self.get_time_idx(itime_str=itime_str)
        edp_len = mesh.primal.edge.length
        edd_len = mesh.dual.edge.length
        pv_e = self.aux.pv_e.unwrap()
        dmg_e = self.state.dmg_e.unwrap()[itime, :]
        # Weighted area
        kite_area = edp_len * edd_len * 0.5
        # =======================================================
        # Total potential enstrophy
        return jnp.sum(dmg_e * pv_e**2 * kite_area[None, :])

    def diagnose_total_potential_temperature(self, itime_str: str) -> Float[Array, ""]:
        # =======================================================
        float_type = self.dev.float_type
        itime = self.get_time_idx(itime_str=itime_str)
        mesh = self.coord.mesh
        dmg = self.state.dmg.unwrap()[itime, :, :]
        pt = self.state.pt.unwrap()[itime, :, :]
        # Weighted area
        primal_area = mesh.primal.cell.area
        # =======================================================
        # Total potential temperature
        if self.mode.is_baroclinic():
            return jnp.sum(dmg * pt * primal_area[None, :])
        else:
            return jnp.array(0.0, dtype=float_type)

    def diagnose_total_tracers(self, itime_str: str) -> Float[Array, ""]:
        # =======================================================
        float_type = self.dev.float_type
        mesh = self.coord.mesh
        # Weighted area
        primal_area = mesh.primal.cell.area
        # =======================================================
        tq = jnp.array(0.0, dtype=float_type)
        for batch in self.adv_batches.values():
            dmg = batch.m.unwrap()
            itime = batch.get_time_idx(itime_str)
            idx = batch.idx
            q = self.tracers.q.unwrap()[itime, idx, :, :]
            tq = tq + jnp.sum(dmg * q * primal_area[None, :])
        # =======================================================
        return tq

    def diagnose4barotropic(self, itime_str: str) -> Self:
        # =======================================================
        # Diagnose u and v at primal cell
        self = self.diagnose_uv(itime_str=itime_str)
        # =======================================================
        # Diagnose total mass, total energy, total potential enstrophy
        # =========================================
        # Total mass
        tm = self.diagnose_total_mass(itime_str=itime_str)
        # =========================================
        # Total energy
        te, te_ke, _, te_pe = self.diagnose_total_energy(itime_str=itime_str)
        # =========================================
        # Total potential enstrophy
        tpe = self.diagnose_total_potential_enstrophy(itime_str=itime_str)
        # =======================================================
        # Wrap diagnosed variables into block
        itime = self.get_time_idx(itime_str=itime_str)
        coord = cast(CoordSys, self.coord)
        if self.mode.is_barotropic():
            return tree_at(
                coord=coord,
                where=lambda b: (
                    b.state.tm, b.state.te, b.state.tpe,
                    b.state.te_ke, b.state.te_pe
                ),
                pytree=self,
                replace=(tm, te, tpe, te_ke, te_pe),
                at_dims=(0,) * 5,
                at_idxes=(itime,) * 5
            )
        else:
            raise ValueError(
                f"Invalid equation mode for barotropic diagnosis. Invalid mode={self.mode}"
            )

    def diagnose4baroclinic(self, itime_str: str) -> Self:
        # =======================================================
        # Diagnose u and v at primal cell
        self = self.diagnose_uv(itime_str=itime_str)
        # =======================================================
        # Diagnose total mass, total energy, total potential enstrophy
        # =========================================
        # Total mass
        tm = self.diagnose_total_mass(itime_str=itime_str)
        # =========================================
        # TODO: When processing moist dycore, this should be modified
        # Total water vapor
        # =========================================
        # Total energy
        out = self.diagnose_total_energy(itime_str=itime_str)
        te, te_ke, te_ie, te_pe = out
        # =========================================
        # Total potential enstrophy
        tpe = self.diagnose_total_potential_enstrophy(itime_str=itime_str)
        # =========================================
        # Total potential temperature
        tpt = self.diagnose_total_potential_temperature(itime_str=itime_str)
        # =========================================
        # Wrap diagnosed variables into block
        itime = self.get_time_idx(itime_str=itime_str)
        coord = cast(CoordSys, self.coord)
        if self.mode.is_baroclinic():
            return tree_at(
                coord=coord,
                where=lambda b: (
                    b.state.tm, b.state.te, b.state.tpe, b.state.tpt,
                    b.state.te_ke, b.state.te_ie, b.state.te_pe
                ),
                pytree=self,
                replace=(tm, te, tpe, tpt, te_ke, te_ie, te_pe),
                at_dims=(0,) * 7,
                at_idxes=(itime,) * 7
            )
        else:
            raise ValueError(
                f"Invalid equation mode for baroclinic diagnosis. Invalid mode={self.mode}"
            )

    def diagnose4advection(self, itime_str: str) -> Self:
        # =======================================================
        # Diagnose u and v at primal cell
        self = self.diagnose_uv(itime_str=itime_str)
        # =======================================================
        # Diagnose total mass, total tracers
        # =========================================
        # Total mass
        tm = self.diagnose_total_mass(itime_str=itime_str)
        # =========================================
        # Total tracers
        tq = self.diagnose_total_tracers(itime_str=itime_str)
        # =======================================================
        # Wrap diagnosed variables into block
        itime = self.get_time_idx(itime_str=itime_str)
        coord = cast(CoordSys, self.coord)
        if self.mode.is_advection():
            return tree_at(
                coord=coord,
                where=lambda b: (b.state.tm, b.state.tq),
                pytree=self,
                replace=(tm, tq),
                at_dims=(0, 0),
                at_idxes=(itime, itime),
            )
        else:
            raise ValueError(
                f"Invalid equation mode for advection diagnosis. Invalid mode={self.mode}"
            )

    def diagnose(self, itime_str: str,) -> Self:
        if self.mode.is_barotropic():
            return self.diagnose4barotropic(itime_str=itime_str)
        elif self.mode.is_baroclinic():
            return self.diagnose4baroclinic(itime_str=itime_str)
        elif self.mode.is_advection():
            return self.diagnose4advection(itime_str=itime_str)
        else:
            raise ValueError(
                f"Invalid equation mode for diagnosis. Invalid mode={self.mode}"
            )

    def reset(self, pass_tag: str) -> Self:
        return eqx.tree_at(
            lambda b: b.tend,
            self,
            replace=self.tend.reset(pass_tag=pass_tag),
        )


class State(AbstractFields[StaggeredFieldMetaData]):
    mode: AbstractEquationMode = eqx.field(static=True)
    fields: Dict[str, StaggeredFieldMetaData] = eqx.field(static=True)
    un: FDT
    gz: FDT
    gz_lev: FDT
    dmg: FDT
    dmg_e: FDT
    dmg_vtx: FDT
    dmg_lev: FDT
    mg: FDT
    mg_lev: FDT
    mgs: FDT
    pt: FDT
    ph: FDT
    ph_lev: FDT
    phs: FDT
    # Nonhydrostatic variables
    p: FDT
    p_lev: FDT
    tm: NDT
    te: NDT
    te_ke: NDT
    te_ie: NDT
    te_pe: NDT
    tpe: NDT
    tpt: NDT
    tq: NDT

    @staticmethod
    def default_meta(mode: AbstractEquationMode) -> dict[str, StaggeredFieldMetaData | NumericMetaData]:
        return default_state_meta(mode=mode)


class Tendency(AbstractFields[StaggeredFieldMetaData]):
    mode: AbstractEquationMode = eqx.field(static=True)
    fields: Dict[str, StaggeredFieldMetaData] = eqx.field(static=True)
    dgzdt: FDT
    dgzdt_dmf: FDT
    dmgsdt: FDT
    dmgsdt_dmf: FDT
    dptdt: FDT
    dptdt_dptfn: FDT
    dptdt_ptvf: FDT
    dundt: FDT
    dundt_coriolis: FDT
    dundt_grad_ke: FDT
    dundt_pgf: FDT
    dundt_wedundeta: FDT

    @staticmethod
    def default_meta(mode: AbstractEquationMode) -> dict[str, StaggeredFieldMetaData]:
        return default_tendency_meta(mode=mode)

    def reset(self, pass_tag: str) -> Self:
        if self.mode.is_barotropic():
            if pass_tag == "all_pass":
                return eqx.tree_at(
                    lambda b: (b.dgzdt.data, b.dundt.data),
                    self,
                    (
                        jnp.zeros_like(self.dgzdt.data),
                        jnp.zeros_like(self.dundt.data)
                    )
                )
            elif pass_tag == "forward_pass":
                return eqx.tree_at(
                    lambda b: b.dgzdt.data,
                    self,
                    jnp.zeros_like(self.dgzdt.data)
                )
            elif pass_tag == "backward_pass":
                return eqx.tree_at(
                    lambda b: b.dundt.data,
                    self,
                    jnp.zeros_like(self.dundt.data)
                )
            else:
                raise ValueError(f"Invalid pass_tag: {pass_tag}")
        if (self.mode.is_dry_hydrostatic()):
            if pass_tag == "all_pass":
                return eqx.tree_at(
                    lambda b: (b.dmgsdt.data, b.dptdt.data, b.dundt.data),
                    self,
                    (
                        jnp.zeros_like(self.dmgsdt.data),
                        jnp.zeros_like(self.dptdt.data),
                        jnp.zeros_like(self.dundt.data)
                    )
                )
            elif pass_tag == "forward_pass":
                return eqx.tree_at(
                    lambda b: (b.dmgsdt.data, b.dptdt.data),
                    self,
                    (
                        jnp.zeros_like(self.dmgsdt.data),
                        jnp.zeros_like(self.dptdt.data)
                    )
                )
            elif pass_tag == "backward_pass":
                return eqx.tree_at(
                    lambda b: b.dundt.data,
                    self,
                    jnp.zeros_like(self.dundt.data)
                )
            else:
                raise ValueError(f"Invalid pass_tag: {pass_tag}")
        else:
            raise NotImplementedError


class Static(AbstractFields[StaggeredFieldMetaData]):
    mode: AbstractEquationMode = eqx.field(static=True)
    fields: Dict[str, StaggeredFieldMetaData] = eqx.field(static=True)
    gzs: FDT
    ref_ps: FDT

    @staticmethod
    def default_meta(mode: AbstractEquationMode,) -> dict[str, StaggeredFieldMetaData]:
        return default_static_meta(mode=mode)


class Auxiliary(AbstractFields[StaggeredFieldMetaData]):
    mode: AbstractEquationMode = eqx.field(static=True)
    fields: Dict[str, StaggeredFieldMetaData] = eqx.field(static=True)
    u: FDT
    v: FDT
    un_lev: FDT
    ut: FDT
    div: FDT
    vor: FDT
    pv_vtx: FDT
    pv_e: FDT
    ke: FDT
    mfn: FDT
    mft: FDT
    mfz_lev: FDT
    mfz_e_lev: FDT
    dmf: FDT
    t: FDT
    tv: FDT
    rhod: FDT

    @staticmethod
    def default_meta(mode: AbstractEquationMode) -> dict[str, StaggeredFieldMetaData]:
        return default_auxiliary_meta(mode=mode)


class EquationTermAuxDataset(AbstractEquationTermAuxiliaryDataset):
    pgf: PressureGradientForceDataset
    coriolis: CoriolisForceAtEdgeData
    adv: AdvectionAuxiliaryDataset

    @classmethod
    def empty(cls) -> Self:
        pgf = PressureGradientForceDataset.empty()
        coriolis = CoriolisForceAtEdgeData.empty()
        adv = AdvectionAuxiliaryDataset.empty()
        return cls(
            pgf=pgf,
            coriolis=coriolis,
            adv=adv
        )


FD = ArrayField[StaggeredFieldMetaData]


def barotropic_case_init_fn(
    coord: CoordSysOneLayer,
    para: PlanetParameterSet,
    solver: AbstractSolver,
    *,
    test_case: AbstractShallowWaterEquationTestCase,
) -> Block:
    g = para.gravity.unwrap()
    block = Block.from_model(
        coord=coord,
        para=para,
        solver=solver,
        test_case=test_case
    )
    mode = solver.equation.mode
    un = init_un_in_barotropic_case(
        coord=coord, mode=mode, test_case=test_case
    )
    gz = init_z_in_barotropic_case(
        coord=coord, mode=mode, test_case=test_case
    ) * g
    gzs = init_zs_in_barotropic_case(
        coord=coord, mode=mode, test_case=test_case
    ) * g
    block = tree_at(
        coord=coord,
        where=lambda b: (b.state.un, b.state.gz),
        pytree=block,
        replace=(un, gz),
        at_dims=(0, 0),
        at_idxes=(0, 0)
    )
    block = tree_at(
        coord=coord,
        where=lambda b: b.static.gzs,
        pytree=block,
        replace=gzs
    )
    return block


def baroclinic_steady_state_case_init_fn(
    coord: CoordSysHybrid,
    para: PlanetParameterSet,
    solver: AbstractSolver,
    *,
    test_case: SteadyStateTestCase,
) -> Block:
    full_kds = coord.vert.full_kds
    full_kde = coord.vert.full_kde
    block = Block.from_model(
        coord=coord,
        para=para,
        solver=solver,
        test_case=test_case
    )
    mode = solver.equation.mode
    gzs = init_gzs_in_baroclinic_case(
        coord=coord, mode=mode, test_case=test_case
    )
    gz = init_gz_in_baroclinic_case(
        coord=coord, mode=mode, test_case=test_case
    )
    gz_lev = init_gz_lev_in_baroclinic_case(
        coord=coord, mode=mode, test_case=test_case
    )
    mgs = init_mgs_in_baroclinic_case(
        coord=coord, mode=mode, test_case=test_case, gzs=gzs
    )
    mg = coord.vert.calc_mg(mgs, ks=full_kds, ke=full_kde)
    t = init_t_in_baroclinic_case(
        coord=coord, mode=mode, test_case=test_case
    )
    pt = formula.dry_potential_temperature(t=t, p=mg, para=para)
    un = init_un_in_baroclinic_case(
        coord=coord, mode=mode, test_case=test_case
    )
    ref_ps = mgs
    block = tree_at(
        coord=coord,
        where=lambda b: (
            b.state.mgs, b.state.mg,
            b.state.gz, b.state.gz_lev,
            b.state.un, b.state.pt
        ),
        pytree=block,
        replace=(mgs, mg, gz, gz_lev, un, pt),
        at_dims=(0,) * 6,
        at_idxes=(0,) * 6
    )
    block = tree_at(
        coord=coord,
        where=lambda b: (b.static.gzs, b.static.ref_ps),
        pytree=block,
        replace=(gzs, ref_ps),
    )
    block = tree_at(
        coord=coord,
        where=lambda b: b.aux.t,
        pytree=block,
        replace=t,
    )
    return block


def baroclinic_mountain_wave_case_init_fn(
    coord: CoordSysHybrid,
    para: PlanetParameterSet,
    solver: AbstractSolver,
    *,
    test_case: MountainWaveTestCase,
) -> Block:
    full_kds = coord.vert.full_kds
    full_kde = coord.vert.full_kde
    block = Block.from_model(
        coord=coord,
        para=para,
        solver=solver,
        test_case=test_case
    )
    mode = solver.equation.mode
    gzs = init_gzs_in_baroclinic_case(
        coord=coord, mode=mode, test_case=test_case
    )
    mgs = init_mgs_in_baroclinic_case(
        coord=coord, mode=mode, test_case=test_case, gzs=gzs
    )
    mg = coord.vert.calc_mg(mgs=mgs, ks=full_kds, ke=full_kde)
    pt = init_pt_in_baroclinic_case(
        coord=coord, mode=mode, test_case=test_case, mg=mg
    )
    un = init_un_in_baroclinic_case(
        coord=coord, mode=mode, test_case=test_case
    )
    ref_ps = mgs
    block = tree_at(
        coord=coord,
        where=lambda b: (
            b.state.mgs, b.state.mg, b.state.un, b.state.pt
        ),
        pytree=block,
        replace=(mgs, mg, un, pt),
        at_dims=(0,) * 4,
        at_idxes=(0,) * 4
    )
    block = tree_at(
        coord=coord,
        where=lambda b: (b.static.gzs, b.static.ref_ps),
        pytree=block,
        replace=(gzs, ref_ps)
    )
    return block


def baroclinic_rossby_wave_case_init_fn(
    coord: CoordSysHybrid,
    para: PlanetParameterSet,
    solver: AbstractSolver,
    *,
    test_case: RossbyHaurwitzWaveTestCase,
) -> Block:
    full_kds = coord.vert.full_kds
    full_kde = coord.vert.full_kde
    half_kds = coord.vert.half_kds
    half_kde = coord.vert.half_kde
    block = Block.from_model(
        coord=coord,
        para=para,
        solver=solver,
        test_case=test_case
    )
    mode = solver.equation.mode
    un = init_un_in_baroclinic_case(
        coord=coord, mode=mode, test_case=test_case
    )
    gzs = init_gzs_in_baroclinic_case(
        coord=coord, mode=mode, test_case=test_case
    )
    mgs = init_mgs_in_baroclinic_case(
        coord=coord, mode=mode, test_case=test_case, gzs=gzs
    )
    mg = coord.vert.calc_mg(mgs=mgs, ks=full_kds, ke=full_kde)
    mg_lev = coord.vert.calc_mg_lev(mgs=mgs, ks=half_kds, ke=half_kde)
    pt = init_pt_in_baroclinic_case(
        coord=coord, mode=mode, test_case=test_case, mg=mg
    )
    gz = init_gz_in_baroclinic_case(
        coord=coord, mode=mode, test_case=test_case, mg=mg
    )
    gz_lev = init_gz_lev_in_baroclinic_case(
        coord=coord, mode=mode, test_case=test_case, mg=mg_lev
    )
    ref_ps = mgs
    block = tree_at(
        coord=coord,
        where=lambda b: (
            b.state.mgs, b.state.mg, b.state.mg_lev,
            b.state.gz, b.state.gz_lev,
            b.state.un, b.state.pt
        ),
        pytree=block,
        replace=(mgs, mg, mg_lev, gz, gz_lev, un, pt),
        at_dims=(0,) * 7,
        at_idxes=(0,) * 7
    )
    block = tree_at(
        coord=coord,
        where=lambda b: (b.static.gzs, b.static.ref_ps),
        pytree=block,
        replace=(gzs, ref_ps)
    )
    return block


def advection2d_case_init_fn(
    coord: CoordSysOneLayer,
    para: PlanetParameterSet,
    solver: AbstractSolver,
    *,
    test_case: AbstractAdvectionEquation2DTestCase,
) -> Block:
    mode = solver.equation.mode
    block = Block.from_model(
        coord=coord,
        para=para,
        solver=solver,
        tracers=Tracer.add_tracers(
            mode=mode,
            coord=coord,
            tracers_meta=test_case.init_tracers_meta(
                dt_adv=solver.time_manager.dt_adv
            )
        ),
        test_case=test_case
    )
    dmg = init_background_in_advection2d_case(
        coord=coord, mode=mode, test_case=test_case
    )
    un = init_un_in_barotropic_case(
        coord=coord, mode=mode, test_case=test_case
    )
    q = init_tracers_in_advection2d_case(
        coord=coord, mode=mode, test_case=test_case
    )
    block = tree_at(
        coord=coord,
        where=lambda b: (b.state.dmg, b.state.un),
        pytree=block,
        replace=(dmg, un),
        at_dims=(0, 0),
        at_idxes=(0, 0)
    )
    return tree_at(
        coord=coord,
        where=lambda b: b.tracers.q,
        pytree=block,
        replace=q,
        at_dims=0,
        at_idxes=0,
    )


def init_z_in_barotropic_case(
    coord: CoordSys,
    mode: AbstractEquationMode,
    test_case: AbstractShallowWaterEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=mode)["gz"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for gz variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    if horizontal_coord is None:
        raise ValueError(
            "Horizontal coordinate information is missing for gz variable."
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    return test_case.init_geopotential_height(
        lon, lat
    ).reshape((1,) + lon.shape)  # add leading axis for vmap


def init_un_in_barotropic_case(
    coord: CoordSys,
    mode: AbstractEquationMode,
    test_case: AbstractShallowWaterEquationTestCase | AbstractAdvectionEquation2DTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=mode)["un"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for un variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    if horizontal_coord is None:
        raise ValueError(
            "Horizontal coordinate information is missing for un variable."
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    cart = horizontal_coord["cart"]
    nr = horizontal_coord["nr_orient"]
    return test_case.init_normal_velocity(
        lon, lat, cart, nr
    ).reshape((1,) + lon.shape)  # add leading axis for vmap


def init_zs_in_barotropic_case(
    coord: CoordSys,
    mode: AbstractEquationMode,
    test_case: AbstractShallowWaterEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_static_meta(mode=mode)["gzs"]
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    if horizontal_coord is None:
        raise ValueError(
            "Horizontal coordinate information is missing for gzs variable."
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    return test_case.init_topography(lon, lat)


def init_mgs_in_baroclinic_case(
    coord: CoordSys,
    mode: AbstractEquationMode,
    test_case: AbstractBaroclinicEquationTestCase,
    gzs: Array,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=mode)["mgs"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for mgs variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    if horizontal_coord is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    return test_case.init_mgs(lon, lat, gzs)


def init_t_in_baroclinic_case(
    coord: CoordSys,
    mode: AbstractEquationMode,
    test_case: AbstractBaroclinicEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_auxiliary_meta(mode=mode)["t"]
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    vertical_coord = meta.get_vert_position(coord).vertical_coord
    if horizontal_coord is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    if vertical_coord is None:
        raise ValueError(
            f"Vertical coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    eta = vertical_coord
    return test_case.init_t(lon, lat, eta)


def init_pt_in_baroclinic_case(
    coord: CoordSys,
    mode: AbstractEquationMode,
    test_case: AbstractBaroclinicEquationTestCase,
    mg: Array,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=mode)["pt"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for pt variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    vertical_coord = meta.get_vert_position(coord).vertical_coord
    if horizontal_coord is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    if vertical_coord is None:
        raise ValueError(
            f"Vertical coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    eta = vertical_coord
    return test_case.init_pt(lon, lat, eta, mg)


def init_un_in_baroclinic_case(
    coord: CoordSys,
    mode: AbstractEquationMode,
    test_case: AbstractBaroclinicEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=mode)["un"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for un variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    vertical_coord = meta.get_vert_position(coord).vertical_coord
    if horizontal_coord is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    if vertical_coord is None:
        raise ValueError(
            f"Vertical coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    eta = vertical_coord
    cart = horizontal_coord["cart"]
    nr = horizontal_coord["nr_orient"]
    return test_case.init_normal_velocity(
        lon, lat, eta, cart, nr
    )


def init_gzs_in_baroclinic_case(
    coord: CoordSys,
    mode: AbstractEquationMode,
    test_case: AbstractBaroclinicEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_static_meta(mode=mode)["gzs"]
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    if horizontal_coord is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    return test_case.init_gzs(lon, lat)


def init_gz_in_baroclinic_case(
    coord: CoordSys,
    mode: AbstractEquationMode,
    test_case: AbstractBaroclinicEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=mode)["gz"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for gz variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    vertical_coord = meta.get_vert_position(coord).vertical_coord
    if horizontal_coord is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    if vertical_coord is None:
        raise ValueError(
            f"Vertical coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    eta = vertical_coord
    return test_case.init_gz(lon, lat, eta, *args, **kwargs)


def init_gz_lev_in_baroclinic_case(
    coord: CoordSys,
    mode: AbstractEquationMode,
    test_case: AbstractBaroclinicEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=mode)["gz_lev"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for gz_lev variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    vertical_coord = meta.get_vert_position(coord).vertical_coord
    if horizontal_coord is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    if vertical_coord is None:
        raise ValueError(
            f"Vertical coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    eta = vertical_coord
    return test_case.init_gz(lon, lat, eta, *args, **kwargs)


def init_background_in_advection2d_case(
    coord: CoordSysOneLayer,
    mode: AbstractEquationMode,
    test_case: AbstractAdvectionEquation2DTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=mode)["dmg"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for dmg variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    if horizontal_coord is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    return test_case.init_background(lon, lat, *args, **kwargs)


def init_tracers_in_advection2d_case(
    coord: CoordSysOneLayer,
    mode: AbstractEquationMode,
    test_case: AbstractAdvectionEquation2DTestCase,
    *args,
    **kwargs
) -> Array:
    # set to 1 just for return metadata
    meta = default_tracer_meta(mode, ntracers=1)["q"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for tracer variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    if horizontal_coord is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    q = test_case.init_tracers(lon, lat, *args, **kwargs)
    ntracers = q.shape[0]
    return q.reshape((ntracers, 1,) + lon.shape)


def default_state_meta(mode: AbstractEquationMode) -> dict[str, StaggeredFieldMetaData | NumericMetaData]:
    meta: dict[str, StaggeredFieldMetaData | NumericMetaData] = {}
    if mode.is_barotropic() or mode.is_advection2d():
        meta["un"] = StaggeredFieldMetaData(
            name="un",
            dims=(NEDGE,),
            long_name="Normal velocity at edge",
            description="Normal velocity at edge",
            units="m/s",
            hori_location=MeshEntity.Edge,
            vert_location="",
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=True,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
    if mode.is_baroclinic() or mode.is_advection3d():
        meta["un"] = StaggeredFieldMetaData(
            name="un",
            dims=(NLEV, NEDGE,),
            long_name="Normal velocity at edge",
            description="Normal velocity at edge",
            units="m/s",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell,
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=True,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
    if mode.is_barotropic():
        meta["gz"] = StaggeredFieldMetaData(
            name="gz",
            dims=(NCELL,),
            long_name="Geopotential at primal cell center",
            description="Geopotential at primal cell center",
            units="m2 / s2",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=True,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
    elif mode.is_baroclinic() or mode.is_advection3d():
        meta["gz"] = StaggeredFieldMetaData(
            name="gz",
            dims=(NLEV, NCELL,),
            long_name="Geopotential at primal cell center",
            description="Geopotential at primal cell center",
            units="m2 / s2",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell if mode.baroclinic else "",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0" if (not mode.is_advection3d()) else ""
        )
    if mode.is_baroclinic():
        meta["gz_lev"] = StaggeredFieldMetaData(
            name="gz_lev",
            dims=(NLEVP, NCELL,),
            long_name="Geopotential at primal cell center and vertical level",
            description="Geopotential at primal cell center and vertical level",
            units="m2 / s2",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Lev,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
        meta["mg"] = StaggeredFieldMetaData(
            name="mg",
            dims=(NLEV, NCELL,),
            long_name="Dry air weight at primal cell center and full level",
            description="Dry air weight at primal cell center and full level",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1"
        )
        meta["mg_lev"] = StaggeredFieldMetaData(
            name="mg_lev",
            dims=(NLEVP, NCELL,),
            long_name="Dry air weight at primal cell center and half level",
            description="Dry air weight at primal cell center and half level",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Lev,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1"
        )
        meta["mgs"] = StaggeredFieldMetaData(
            name="mgs",
            dims=(NCELL,),
            long_name="Dry air weight on surface at primal cell center and full level",
            description="Dry air weight on surface at primal cell center and full level",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Sfc,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=True,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1"
        )
        meta["dmg"] = StaggeredFieldMetaData(
            name="dmg",
            dims=(NLEV, NCELL,),
            long_name="Difference of dry air weight at primal cell center between the neighboring half levels",
            description="Difference of dry air weight at primal cell center between the neighboring half levels",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1"
        )
        meta["dmg_e"] = StaggeredFieldMetaData(
            name="dmg_e",
            dims=(NLEV, NEDGE,),
            long_name="Difference of dry air weight at edge between the neighboring half levels",
            description="Difference of dry air weight at edge between the neighboring half levels",
            units="Pa",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1"
        )
        meta["dmg_vtx"] = StaggeredFieldMetaData(
            name="dmg_vtx",
            dims=(NLEV, NVTX,),
            long_name="Difference of dry air weight at dual cell center between the neighboring half levels",
            description="Difference of dry air weight at dual cell center between the neighboring half levels",
            units="Pa",
            hori_location=MeshEntity.Vtx,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_vtx lat_vtx",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1"
        )
        meta["dmg_lev"] = StaggeredFieldMetaData(
            name="dmg_lev",
            dims=(NLEVP, NCELL,),
            long_name="Difference of dry air weight at primal cell center between the neighboring full levels",
            description="Difference of dry air weight at primal cell center between the neighboring full levels",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Lev,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1"
        )
        meta["pt"] = StaggeredFieldMetaData(
            name="pt",
            dims=(NLEV, NCELL,),
            long_name="Potential temperature at primal cell center and vertical level",
            description="Potential temperature at primal cell center and vertical level",
            units="K",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=True,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
    if mode.is_advection3d():
        meta["dmg"] = StaggeredFieldMetaData(
            name="dmg",
            dims=(NLEV, NCELL,),
            long_name="Difference of dry air weight at primal cell center between the neighboring half levels",
            description="Difference of dry air weight at primal cell center between the neighboring half levels",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1"
        )
    if mode.is_barotropic() or mode.is_advection2d():
        meta["dmg"] = StaggeredFieldMetaData(
            name="dmg",
            dims=(NCELL,),
            long_name="Fluid thickness at primal cell",
            description="Fluid thickness at primal cell",
            units="m",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1"
        )
    if mode.is_barotropic():
        meta["dmg_e"] = StaggeredFieldMetaData(
            name="dmg_e",
            dims=(NEDGE,),
            long_name="Fluid thickness at edge",
            description="Fluid thickness at edge",
            units="m",
            hori_location=MeshEntity.Edge,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1"
        )
    if mode.is_barotropic():
        meta["dmg_vtx"] = StaggeredFieldMetaData(
            name="dmg_vtx",
            dims=(NVTX,),
            long_name="Fluid thickness at dual cell center",
            description="Fluid thickness at dual cell center",
            units="m",
            hori_location=MeshEntity.Vtx,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_vtx lat_vtx",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1"
        )
    if mode.is_baroclinic():
        meta["ph"] = StaggeredFieldMetaData(
            name="ph",
            dims=(NLEV, NCELL,),
            long_name="Hydrostatic pressure at primal cell center",
            description="Hydrostatic pressure at primal cell center",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
        meta["ph_lev"] = StaggeredFieldMetaData(
            name="ph_lev",
            dims=(NLEVP, NCELL,),
            long_name="Hydrostatic pressure at primal cell center and vertical level",
            description="Hydrostatic pressure at primal cell center and vertical level",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Lev,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1"
        )
        meta["phs"] = StaggeredFieldMetaData(
            name="phs",
            dims=(NCELL,),
            long_name="Surface hydrostatic pressure at primal cell center",
            description="Surface hydrostatic pressure at primal cell center",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Sfc,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
        meta["p"] = StaggeredFieldMetaData(
            name="p",
            dims=(NLEV, NCELL,),
            long_name="Pressure at primal cell center and vertical level",
            description="Pressure at primal cell center and vertical level",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0" if mode.nonhydrostatic else ""
        )
        meta["p_lev"] = StaggeredFieldMetaData(
            name="p_lev",
            dims=(NLEVP, NCELL),
            long_name="Pressure at primal cell center and vertical level",
            description="Pressure at primal cell center and vertical level",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Lev,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0" if mode.nonhydrostatic else ""
        )
    if mode.is_barotropic() or mode.is_baroclinic() or mode.is_advection():
        meta["tm"] = NumericMetaData(
            name="tm",
            long_name="Total mass",
            description="Total mass",
            units="kg",
            is_static=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
    if mode.is_barotropic() or mode.is_baroclinic():
        meta["tpe"] = NumericMetaData(
            name="tpe",
            long_name="Total potential vorticity enstrophy",
            description="Total potential vorticity enstrophy",
            units="J/kg",
            is_static=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
        meta["te"] = NumericMetaData(
            name="te",
            long_name="Total energy",
            description="Total energy",
            is_static=False,
            is_expand_dims=True,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
        meta["te_ke"] = NumericMetaData(
            name="te_ke",
            long_name="Total energy due to kinetic energy",
            description="Total energy due to kinetic energy",
            units="J/kg",
            is_static=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output=""
        )
        meta["te_pe"] = NumericMetaData(
            name="te_pe",
            long_name="Total energy due to potential energy",
            description="Total energy due to potential energy",
            units="J/kg",
            is_static=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output=""
        )
    if mode.is_baroclinic():
        meta["te_ie"] = NumericMetaData(
            name="te_ie",
            long_name="Total energy due to internal energy",
            description="Total energy due to internal energy",
            units="J/kg",
            is_static=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output=""
        )
        meta["tpt"] = NumericMetaData(
            name="tpt",
            long_name="Total potential temperature",
            description="Total potential temperature",
            units="K",
            is_static=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
    if mode.is_advection():
        meta["tq"] = NumericMetaData(
            name="tq",
            long_name="Total tracer dry mixing ratio",
            description="Total tracer dry mixing ratio",
            units="m3 kg/kg",
            is_static=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
    return meta


def default_tendency_meta(mode: AbstractEquationMode) -> dict[str, StaggeredFieldMetaData]:
    meta: dict[str, StaggeredFieldMetaData] = {}
    if mode.is_barotropic():
        meta["dgzdt"] = StaggeredFieldMetaData(
            name="dgzdt",
            dims=(NCELL,),
            long_name="Tendency of geopotential",
            description="Tendency of geopotential calculated from state variables",
            units="m2 / s3",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
        meta["dgzdt_dmf"] = StaggeredFieldMetaData(
            name="dgzdt_dmf",
            dims=(NCELL,),
            long_name="Tendency of geopotential due to mass flux divergence",
            description="Tendency of geopotential due to mass flux divergence calculated from state variables",
            units="m2 / s3",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.output_h1_tend else ""
        )
    if mode.is_baroclinic():
        meta["dmgsdt"] = StaggeredFieldMetaData(
            name="dmgsdt",
            dims=(NCELL,),
            long_name="Tendency of dry air weight at primal cell center and full level",
            description="Tendency of dry air weight at primal cell center and full level",
            units="Pa / s",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Sfc,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
        meta["dmgsdt_dmf"] = StaggeredFieldMetaData(
            name="dmgsdt_dmf",
            dims=(NCELL,),
            long_name="Tendency of dry air weight at primal cell center and full level due to mass flux divergence",
            description="Tendency of dry air weight at primal cell center and full level due to mass flux divergence calculated from state variables",
            units="Pa / s",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Sfc,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.output_h1_tend else ""
        )
        meta["dptdt"] = StaggeredFieldMetaData(
            name="dptdt",
            dims=(NLEV, NCELL,),
            long_name="Tendency of potential temperature at primal cell center and vertical level",
            description="Tendency of potential temperature at primal cell center and vertical level",
            units="K / s",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
        meta["dptdt_dptfn"] = StaggeredFieldMetaData(
            name="dptdt_dptfn",
            dims=(NLEV, NCELL,),
            long_name="Tendency of potential temperature at primal cell center and vertical level due to horizontal advection",
            description="Tendency of potential temperature at primal cell center and vertical level due to horizontal advection",
            units="K / s",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.output_h1_tend else ""
        )
        meta["dptdt_ptvf"] = StaggeredFieldMetaData(
            name="dptdt_ptvf",
            dims=(NLEV, NCELL,),
            long_name="Tendency of potential temperature at primal cell center and vertical level due to vertical advection",
            description="Tendency of potential temperature at primal cell center and vertical level due to vertical advection",
            units="K / s",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.output_h1_tend else ""
        )
    if mode.is_barotropic() or mode.is_baroclinic():
        meta["dundt"] = StaggeredFieldMetaData(
            name="dundt",
            dims=(NLEV, NEDGE,) if mode.is_baroclinic() else (NEDGE,),
            long_name="Tendency of normal velocity at edge",
            description="Tendency of normal velocity at edge",
            units="m / s2",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
        meta["dundt_coriolis"] = StaggeredFieldMetaData(
            name="dundt_coriolis",
            dims=(NLEV, NEDGE,) if mode.is_baroclinic() else (NEDGE,),
            long_name="Tendency of normal velocity at edge due to Coriolis force",
            description="Tendency of normal velocity at edge due to Coriolis force",
            units="m / s2",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.output_h1_tend else ""
        )
        meta["dundt_grad_ke"] = StaggeredFieldMetaData(
            name="dundt_grad_ke",
            dims=(NLEV, NEDGE,) if mode.is_baroclinic() else (NEDGE,),
            long_name="Tendency of normal velocity at edge due to gradient of kinetic energy",
            description="Tendency of normal velocity at edge due to gradient of kinetic energy",
            units="m / s2",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.output_h1_tend else ""
        )
        meta["dundt_pgf"] = StaggeredFieldMetaData(
            name="dundt_pgf",
            dims=(NLEV, NEDGE,) if mode.is_baroclinic() else (NEDGE,),
            long_name="Tendency of normal velocity at edge due to pressure gradient force",
            description="Tendency of normal velocity at edge due to pressure gradient force",
            units="m / s2",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.output_h1_tend else ""
        )
    if mode.is_baroclinic():
        meta["dundt_wedundeta"] = StaggeredFieldMetaData(
            name="dundt_wedundeta",
            dims=(NLEV, NEDGE,),
            long_name="Tendency of normal velocity at edge due to vertical advection",
            description="Tendency of normal velocity at edge due to vertical advection",
            units="m / s2",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell,
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.output_h1_tend else ""
        )
    return meta


def default_static_meta(mode: AbstractEquationMode,) -> dict[str, StaggeredFieldMetaData]:
    meta: dict[str, StaggeredFieldMetaData] = {}
    if mode.is_barotropic() or mode.is_baroclinic():
        meta["gzs"] = StaggeredFieldMetaData(
            name="gzs",
            dims=(NCELL,),
            long_name="Surface geopotential at primal cell center",
            description="Surface geopotential at primal cell center",
            units="m2 / s2",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Sfc,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_static=True,
            output="h0"
        )
    if mode.is_baroclinic():
        meta["ref_ps"] = StaggeredFieldMetaData(
            name="ref_ps",
            dims=(NCELL,),
            long_name="Reference surface pressure at primal cell center",
            description="Reference surface pressure at primal cell center",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Sfc,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_static=False,
            output="h1"
        )
    return meta


def default_auxiliary_meta(mode: AbstractEquationMode) -> dict[str, StaggeredFieldMetaData]:
    meta: dict[str, StaggeredFieldMetaData] = {}
    if mode.is_barotropic() or mode.is_advection2d():
        meta["u"] = StaggeredFieldMetaData(
            name="u",
            dims=(NCELL,),
            long_name="Zonal velocity at primal cell center",
            description="Zonal velocity at primal cell center reconstructed from normal velocity at dual edge",
            units="m/s",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h0"
        )
        meta["v"] = StaggeredFieldMetaData(
            name="v",
            dims=(NCELL,),
            long_name="Meridional velocity at primal cell center",
            description="Meridional velocity at primal cell center reconstructed from normal velocity at dual edge",
            units="m/s",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h0"
        )
    if mode.is_baroclinic() or mode.is_advection3d():
        meta["u"] = StaggeredFieldMetaData(
            name="u",
            dims=(NLEV, NCELL,),
            long_name="Zonal velocity at primal cell center",
            description="Zonal velocity at primal cell center reconstructed from normal velocity at dual edge",
            units="m/s",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h0"
        )
        meta["v"] = StaggeredFieldMetaData(
            name="v",
            dims=(NLEV, NCELL,),
            long_name="Meridional velocity at primal cell center",
            description="Meridional velocity at primal cell center reconstructed from normal velocity at dual edge",
            units="m/s",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h0"
        )
    if mode.is_baroclinic():
        meta["un_lev"] = StaggeredFieldMetaData(
            name="un_lev",
            dims=(NLEVP, NEDGE,),
            long_name="Normal velocity at edge and half level",
            description="Normal velocity at edge and half level",
            units="m/s",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Lev,
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=True,
            output=""
        )
    if mode.is_barotropic() or mode.is_baroclinic():
        meta["ut"] = StaggeredFieldMetaData(
            name="ut",
            dims=(NLEV, NEDGE,) if mode.is_baroclinic() else (NEDGE,),
            long_name="Tangential velocity at edge",
            description="Tangential velocity at edge reconstructed from normal velocity at dual edge",
            units="m/s",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Scalar,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
        meta["div"] = StaggeredFieldMetaData(
            name="div",
            dims=(NLEV, NCELL,) if mode.is_baroclinic() else (NCELL,),
            long_name="Divergence at primal cell center",
            description="Divergence at primal cell center",
            units="1/s",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h0"
        )
        meta["vor"] = StaggeredFieldMetaData(
            name="vor",
            dims=(NLEV, NVTX,) if mode.is_baroclinic() else (NVTX,),
            long_name="Relative vorticity at dual cell center",
            description="Relative vorticity at dual cell center",
            units="1/s",
            hori_location=MeshEntity.Vtx,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Scalar,
            coordinates="lon_vtx lat_vtx",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h0"
        )
        meta["pv_vtx"] = StaggeredFieldMetaData(
            name="pv_vtx",
            dims=(NLEV, NVTX,) if mode.is_baroclinic() else (NVTX,),
            long_name="Potential vorticity at dual cell center",
            description="Potential vorticity at dual cell center",
            units="Pa-1 s-1",
            hori_location=MeshEntity.Vtx,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Scalar,
            coordinates="lon_vtx lat_vtx",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
        meta["pv_e"] = StaggeredFieldMetaData(
            name="pv_e",
            dims=(NLEV, NEDGE,) if mode.is_baroclinic() else (NEDGE,),
            long_name="Potential vorticity at edge",
            description="Potential vorticity at edge",
            units="Pa-1 s-1",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Scalar,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
        meta["ke"] = StaggeredFieldMetaData(
            name="ke",
            dims=(NLEV, NCELL,) if mode.is_baroclinic() else (NCELL,),
            long_name="Kinetic energy at primal cell center",
            description="Kinetic energy at primal cell center",
            units="m2 / s2",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
        meta["mfn"] = StaggeredFieldMetaData(
            name="mfn",
            dims=(NLEV, NEDGE,) if mode.is_baroclinic() else (NEDGE,),
            long_name="Mass flux at edge midpoint",
            description="Mass flux at edge midpoint",
            units="kg / (m s)",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Scalar,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
    if mode.is_baroclinic():
        meta["mft"] = StaggeredFieldMetaData(
            name="mft",
            dims=(NLEV, NEDGE,),
            long_name="Mass flux at edge midpoint reconstructed from tangential velocity",
            description="Mass flux at edge midpoint reconstructed from tangential velocity",
            units="kg / (m s)",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
        meta["mfz_lev"] = StaggeredFieldMetaData(
            name="mfz_lev",
            dims=(NLEVP, NCELL,),
            long_name="Vertical mass flux at primal cell center and half level",
            description="Vertical mass flux at primal cell center and half level",
            units="kg / (m s)",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Lev,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
        meta["mfz_e_lev"] = StaggeredFieldMetaData(
            name="mfz_e_lev",
            dims=(NLEVP, NEDGE,),
            long_name="Vertical mass flux at edge and half level",
            description="Vertical mass flux at edge and half level",
            units="kg / (m s)",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Lev,
            basis=Basis.Scalar,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output=""
        )
    if mode.is_barotropic() or mode.is_baroclinic():
        meta["dmf"] = StaggeredFieldMetaData(
            name="dmf",
            dims=(NLEV, NCELL,) if mode.is_baroclinic() else (NCELL,),
            long_name="Divergence of mass flux at primal cell center",
            description="Divergence of mass flux at primal cell center",
            units="kg / (m s)",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
    if mode.is_baroclinic():
        meta["t"] = StaggeredFieldMetaData(
            name="t",
            dims=(NLEV, NCELL,),
            long_name="Temperature at primal cell center",
            description="Temperature at primal cell center",
            units="K",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h0"
        )
        meta["tv"] = StaggeredFieldMetaData(
            name="tv",
            dims=(NLEV, NCELL,),
            long_name="Virtual temperature at primal cell center",
            description="Virtual temperature at primal cell center",
            units="K",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
        meta["rhod"] = StaggeredFieldMetaData(
            name="rhod",
            dims=(NLEV, NCELL,),
            long_name="Density of dry air at primal cell center",
            description="Density of dry air at primal cell center",
            units="kg / m3",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
    return meta
