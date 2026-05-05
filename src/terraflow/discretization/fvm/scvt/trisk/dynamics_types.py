from __future__ import annotations
from typing import (
    Self,
    cast,
    Dict,
    Optional,
    TypeVar,
    Generic,
    TYPE_CHECKING
)
import equinox as eqx
import jax
import jax.numpy as jnp

from terraflow.util import log
from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.layout.abstract import (
    CoordinateSystem,
    MeshEntity,
    VertLoc,
)
from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
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
    AbstractAdvectionEquation3DTestCase,
    SolidRotationTestCase,
    DCMIP12TestCase
)
from terraflow.solver.abstract import AbstractSolver
from terraflow.discretization.abstract import (
    AbstractEquationMode,
    AbstractEquationSet,
    AbstractBlock,
    AbstractFields,
    AbstractEquationTermAuxiliaryDataset
)
from terraflow.layout.field import ArrayField, StaggeredFieldMetaData, NumericMetaData
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at
from terraflow.discretization.fvm.scvt.trisk.tracer import Tracer
from terraflow.discretization.fvm.scvt.trisk.common.pgf.types import PressureGradientForceDataset
from terraflow.discretization.fvm.scvt.trisk.common.coriolis.types import CoriolisForceAtEdgeData
from terraflow.discretization.fvm.scvt.trisk.common.adv.types import AdvectionAuxiliaryDataset
from terraflow.discretization.fvm.scvt.trisk.common.adv.batch import AdvBatchType
from .element_meta import (
    default_state_meta,
    default_tendency_meta,
    default_static_meta,
    default_auxiliary_meta
)
from .test_case.swm import barotropic_case_init_fn
from .test_case.baroclinic import (
    baroclinic_steady_state_case_init_fn,
    baroclinic_mountain_wave_case_init_fn,
    baroclinic_rossby_wave_case_init_fn
)
from .test_case.adv import (
    advection2d_case_init_fn,
    advection2d_case_set_uv_fn,
    dcmip12_case_init_fn,
    dcmip12_case_set_uv_fn
)

VM = TypeVar("VM", OneLayerVertCoordinate, HybridVertCoordinate)
VLO = TypeVar("VLO", OneLayerVertCoordLayout, HybridVertCoordLayout)
FDT = ArrayField[StaggeredFieldMetaData]
NDT = ArrayField[NumericMetaData]

NLEVP = "ilev"
NLEV = "lev"
NCELL = "cell"
NEDGE = "edge"
NVTX = "vtx"

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

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.planet import PlanetParameterSet
    from terraflow.test_case.abstract import AbstractTestCase

    Int = typing.Int
    Float = typing.Float
    Array = typing.Array
    PyTree = typing.PyTree

    ES = TypeVar("ES", bound=AbstractEquationSet)


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
        dt_dyn = solver.dt_dyn
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
        float_type = coord.dev.float_type
        return cls(
            elapsed_seconds=jnp.array(0.0, dtype=float_type),
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
        mode: AbstractEquationMode = equation.mode
        if mode.is_baroclinic():
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
        elif mode.is_advection():
            # ==============================================
            # Advection case
            if mode.is_advection2d():
                if case is None:
                    case = SolidRotationTestCase(para)
                elif not isinstance(case, AbstractAdvectionEquation2DTestCase):
                    raise ValueError(
                        "Invalid test case for advection equation"
                    )
                block = advection2d_case_init_fn(
                    time_in_seconds=jnp.array(0.0, dtype=coord.dev.float_type),
                    coord=coord,
                    para=para,
                    solver=solver,
                    test_case=case
                )
            elif mode.is_advection3d():
                if case is None:
                    case = DCMIP12TestCase(para)
                elif not isinstance(case, AbstractAdvectionEquation3DTestCase):
                    raise ValueError(
                        "Invalid test case for 3D advection equation"
                    )
                if isinstance(case, DCMIP12TestCase):
                    block = dcmip12_case_init_fn(
                        time_in_seconds=jnp.array(
                            0.0,
                            dtype=coord.dev.float_type
                        ),
                        coord=coord,
                        para=para,
                        solver=solver,
                        test_case=case
                    )
                else:
                    raise NotImplementedError(
                        f"Initialization for case {type(case)} is not implemented yet."
                    )
            else:
                raise NotImplementedError(
                    "Unvalid equation type for advection testcase initialization"
                )
            # ==============================================
        elif mode.is_barotropic():
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

    def testcase_set_uv(self, itime: Int[Array, ""] | int) -> Self:
        dt_dyn = self.adv_batch_bg.dt
        elapsed_seconds = self.elapsed_seconds + dt_dyn
        self = eqx.tree_at(
            lambda b: b.elapsed_seconds,
            self,
            elapsed_seconds
        )
        if self.test_case is None:
            raise ValueError(
                "Test case is not provided for the block. Cannot set un."
            )
        else:
            if isinstance(self.test_case, AbstractAdvectionEquation2DTestCase):
                self = cast(
                    Self,
                    advection2d_case_set_uv_fn(
                        block=self,
                        test_case=self.test_case,
                        itime=itime
                    )
                )
            elif isinstance(self.test_case, AbstractAdvectionEquation3DTestCase):
                if isinstance(self.test_case, DCMIP12TestCase):
                    self = cast(
                        Self,
                        dcmip12_case_set_uv_fn(
                            block=self,
                            test_case=self.test_case,
                            itime=itime
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"Initialization of un for test case {type(self.test_case)} is not implemented yet."
                    )
            else:
                raise NotImplementedError(
                    f"Initialization of un for test case {type(self.test_case)} is not implemented yet."
                )
            return self

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

    def set_new_tracer_mass4batches(self) -> Self:
        # ==========================================
        # Set new tracer mass for current batch
        m_new = self.adv_batch_bg.m.unwrap()
        for batch in self.adv_batches.values():

            def replace_tracer_mass(b: Self) -> Self:
                mq_new = b.tracers.mq.unwrap()
                q_new = b.tracers.q.unwrap()[batch.new, batch.idx, :, :]
                mq_new = mq_new.at[batch.new, batch.idx, :, :].set(
                    m_new * q_new
                )
                return eqx.tree_at(
                    lambda b: b.tracers.mq.data,
                    b,
                    mq_new,
                )

            def do_nothing(b: Self) -> Self:
                return b

            self = jax.lax.cond(
                pred=batch.is_run_tracer_istep(),
                true_fun=replace_tracer_mass,
                false_fun=do_nothing,
                operand=self
            )
        # ==========================================
        return self

    def set_new_mass4batches(self) -> Self:
        # ==========================================
        # Set new mass for current batch
        m_new = self.adv_batch_bg.m.unwrap()
        for batch_name, batch in self.adv_batches.items():
            batch: AdvBatchType = batch.set_new_mass(m_new=m_new)
            self = eqx.tree_at(
                lambda b: b.adv_batches[batch_name],
                self,
                batch
            )
        # ==========================================
        return self


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
    we_lev: FDT
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
