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
from terraflow.test_case.swe import (
    ShallowWaterEquationTestCase,
    MountainZonalFlowTestCase
)
from terraflow.test_case.baroclinic import (
    BaroclinicEquationTestCase,
    SteadyStateTestCase,
    MountainWaveTestCase,
    RossbyHaurwitzWaveTestCase
)
from terraflow.discretization.abstract import (
    AbstractEquationMode,
    AbstractEquationSet,
    AbstractBlock,
    AbstractFields,
    AbstractEquationTermAuxiliaryDataset
)
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at
from terraflow.discretization.fvm.scvt.trisk.common.pgf.types import PressureGradientForceDataset
from terraflow.discretization.fvm.scvt.trisk.common.coriolis.types import CoriolisForceAtEdgeData
from terraflow.discretization.fvm.scvt.trisk.common.adv.types import AdvectionAuxiliaryDataset
from terraflow.discretization.fvm.scvt.trisk.common.adv.batch import AdvBatchType

Int = typing.Int
Array = typing.Array
PyTree = typing.PyTree

FDT = ArrayField[StaggeredFieldMetaData]

VM = TypeVar("VM", OneLayerVertCoordinate, HybridVertCoordinate)
VLO = TypeVar("VLO", OneLayerVertCoordLayout, HybridVertCoordLayout)

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
    AdvBatchType
]


class Block(
    AbstractBlock[
        SphericalCentroidVoronoiTessellations,
        VM,
        TriSKBasedCGridStaggeringLayout,
        VLO,
        "State", "Tendency", "Static", "Auxiliary", AdvBatchType, "EquationTermAuxDataset"
    ],
    Generic[VM, VLO]
):
    mode: AbstractEquationMode = eqx.field(static=True)
    old: Array
    new: Array
    _coord: CoordSys
    para: PlanetParameterSet
    state: "State"
    tend: "Tendency"
    static: "Static"
    aux: "Auxiliary"
    adv_batch_pt: AdvBatchType
    equaux: "EquationTermAuxDataset"

    @classmethod
    def from_testcase(
        cls,
        coord: CoordSys,
        para: PlanetParameterSet,
        equation: AbstractEquationSet,
        *,
        case: Optional[
            BaroclinicEquationTestCase | ShallowWaterEquationTestCase
        ] = None,
    ):
        # =======================================================
        if equation.mode.baroclinic:
            # ==============================================
            # Baroclinic case
            if case is None:
                case = SteadyStateTestCase(para)
            elif not isinstance(case, BaroclinicEquationTestCase):
                raise ValueError(
                    "Invalid test case for baroclinic equation"
                )
            if isinstance(case, SteadyStateTestCase):
                elements = baroclinic_steady_state_case_init_fn(
                    coord=coord, para=para, equation=equation, case=case
                )
            elif isinstance(case, MountainWaveTestCase):
                elements = baroclinic_mountain_wave_case_init_fn(
                    coord=coord, para=para, equation=equation, case=case
                )
            elif isinstance(case, RossbyHaurwitzWaveTestCase):
                elements = baroclinic_rossby_wave_case_init_fn(
                    coord=coord, para=para, equation=equation, case=case
                )
            else:
                raise NotImplementedError(
                    f"Initialization for case {type(case)} is not implemented yet."
                )
            # ==============================================
        else:
            # ==============================================
            # Barotropic case
            if case is None:
                case = MountainZonalFlowTestCase(para)
            elif not isinstance(case, ShallowWaterEquationTestCase):
                raise ValueError(
                    "Invalid test case for shallow water equation"
                )
            elements = barotropic_case_init_fn(
                coord=coord, para=para, equation=equation, case=case
            )
            # ==============================================
        log.logger.success("Initialized block from testcase.")
        # =======================================================
        state = cast(State, elements["state"])
        tend = cast(Tendency, elements["tend"])
        static = cast(Static, elements["static"])
        aux = cast(Auxiliary, elements["aux"])
        adv_batch_pt = cast(AdvBatchType, elements["adv_batch_pt"])
        return cls(
            mode=equation.mode,
            old=jnp.array(0, dtype=jnp.int32),
            new=jnp.array(1, dtype=jnp.int32),
            _coord=coord,
            para=para,
            state=state,
            tend=tend,
            static=static,
            aux=aux,
            adv_batch_pt=adv_batch_pt,
            equaux=EquationTermAuxDataset.empty()
        )

    def diagnose(self, itime: Int[Array, ""] | int) -> "Block":
        state = self.state
        un = state.un.unwrap()[itime, :, :]
        # =======================================================
        # Diagnose u and v from un
        uv = self.coord.mesh.reconstruct_vector(
            normal_u_at_edge=un,
        )
        block: "Block" = tree_at(
            coord=cast(CoordSys, self.coord),
            where=lambda b: (b.aux.u.data, b.aux.v.data),
            pytree=self,
            replace=(uv["u"], uv["v"]),
        )
        # =======================================================
        # Diagnose total mass, total energy, total potential enstrophy
        mesh = block.coord.mesh
        edp_len = mesh.primal.edge.length
        edd_len = mesh.dual.edge.length
        gzs = block.static.gzs.unwrap()
        dmg = state.dmg.unwrap()[itime, :, :]
        un = state.un.unwrap()[itime, :, :]
        tv = block.aux.tv.unwrap()
        pv_e = block.aux.pv_e.unwrap()
        g = block.para.gravity.unwrap()
        cpd = block.para.cpd.unwrap()

        # Weighted area
        primal_area = mesh.primal.cell.area
        kite_area = edp_len * edd_len * 0.5

        tm = jnp.array(0.0, dtype=jnp.float64)
        te = jnp.array(0.0, dtype=jnp.float64)
        tpe = jnp.array(0.0, dtype=jnp.float64)
        tpt = jnp.array(0.0, dtype=jnp.float64)
        tqv = jnp.array(0.0, dtype=jnp.float64)
        te_ke = jnp.array(0.0, dtype=jnp.float64)
        te_ie = jnp.array(0.0, dtype=jnp.float64)
        te_pe = jnp.array(0.0, dtype=jnp.float64)

        # =========================================
        # Total mass
        tm = tm + jnp.sum(dmg * primal_area[None, :])
        # =========================================
        # TODO: When processing moist dycore, this should be modified
        # Total water vapor
        # =========================================
        # Total energy
        # =================================
        # Total energy due to kinetic energy
        # TODO: nh dynamics kinetic energy
        # Kinetic energy at edge
        edd_v = mesh.dual.edge.endpoints
        cell_0 = edd_v[:, 0]
        cell_1 = edd_v[:, 1]
        dmg_e = 0.5 * (dmg[:, cell_0] + dmg[:, cell_1])
        te_ke = te_ke + jnp.sum(0.5 * un**2 * dmg_e * kite_area[None, :])
        # ==================================
        # Total energy due to internal energy
        if self.mode.baroclinic:
            te_ie = te_ie + jnp.sum(dmg * cpd * tv * primal_area[None, :])
        # ==================================
        # Total energy due to potential energy
        if self.mode.baroclinic:
            mgs = state.mgs.unwrap()[itime, :]
            te_pe = te_pe + jnp.sum(gzs * mgs * primal_area)
        else:
            te_pe = te_pe + jnp.sum(
                (dmg.squeeze()**2 * g * 0.5 + dmg.squeeze()*gzs) * primal_area,
            )
        # ==================================
        te = te_ke + te_ie + te_pe
        # =========================================
        # Total potential enstrophy
        tpe = tpe + jnp.sum(dmg_e * pv_e**2 * kite_area[None, :])
        # =========================================
        # Total potential temperature
        if self.mode.baroclinic:
            pt = state.pt.unwrap()[itime, :, :]
            tpt = tpt + jnp.sum(dmg * pt * primal_area[None, :])
        # =========================================
        # Wrap diagnosed variables into block
        if self.mode.baroclinic:
            return tree_at(
                coord=block.coord,
                where=lambda b: (
                    b.state.tm, b.state.te, b.state.tpe, b.state.tpt,
                    b.state.te_ke, b.state.te_ie, b.state.te_pe
                ),
                pytree=block,
                replace=(tm, te, tpe, tpt, te_ke, te_ie, te_pe),
                at_dims=(0,) * 7,
                at_idxes=(itime,) * 7
            )
        else:
            return tree_at(
                coord=block.coord,
                where=lambda b: (
                    b.state.tm, b.state.te, b.state.tpe,
                    b.state.te_ke, b.state.te_pe
                ),
                pytree=block,
                replace=(tm, te, tpe, te_ke, te_pe),
                at_dims=(0,) * 5,
                at_idxes=(itime,) * 5
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
    tm: FDT
    te: FDT
    te_ke: FDT
    te_ie: FDT
    te_pe: FDT
    tpe: FDT
    tpt: FDT

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
        if not self.mode.baroclinic:
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
        if (
            self.mode.baroclinic
            and self.mode.hydrostatic
            and not self.mode.nonhydrostatic
            and self.mode.ideal_dry_core
            and not self.mode.advection
        ):
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


def barotropic_block_element_init_fn(
    coord: CoordSysOneLayer,
    equation: AbstractEquationSet,
) -> Dict[str, Element]:
    state = State.from_coord(coord=coord, equation=equation)
    static = Static.from_coord(coord=coord, equation=equation)
    tend = Tendency.from_coord(coord=coord, equation=equation)
    aux = Auxiliary.from_coord(coord=coord, equation=equation)
    adv_batch_pt = AdvBatchType.empty(mode=equation.mode)
    return {
        "state": state,
        "static": static,
        "tend": tend,
        "aux": aux,
        "adv_batch_pt": adv_batch_pt
    }


def barotropic_case_init_fn(
    coord: CoordSysOneLayer,
    para: PlanetParameterSet,
    equation: AbstractEquationSet,
    *,
    case: ShallowWaterEquationTestCase,
) -> Dict[str, Element]:
    g = para.gravity.unwrap()
    elements = barotropic_block_element_init_fn(
        coord=coord, equation=equation
    )
    un = init_un_in_barotropic_case(
        coord=coord, equation=equation, test=case
    )
    gz = init_z_in_barotropic_case(
        coord=coord, equation=equation, test=case
    ) * g
    gzs = init_zs_in_barotropic_case(
        coord=coord, equation=equation, test=case
    ) * g
    elements["state"] = tree_at(
        coord=coord,
        where=lambda s: (s.un, s.gz),
        pytree=elements["state"],
        replace=(un, gz),
        at_dims=(0, 0),
        at_idxes=(0, 0)
    )
    elements["static"] = tree_at(
        coord=coord,
        where=lambda s: s.gzs,
        pytree=elements["static"],
        replace=gzs
    )
    return elements


def baroclinic_block_element_init_fn(
    coord: CoordSysHybrid,
    equation: AbstractEquationSet,
) -> Dict[str, Element]:
    state = State.from_coord(coord=coord, equation=equation)
    tend = Tendency.from_coord(coord=coord, equation=equation)
    static = Static.from_coord(coord=coord, equation=equation)
    aux = Auxiliary.from_coord(coord=coord, equation=equation)
    adv_batch_pt = AdvBatchType.from_coord(
        name="pt",
        batch_hori_loc=MeshEntity.Cell,
        batch_vert_loc=VertLoc.Cell,
        coord=coord,
        equation=equation
    )
    return {
        "state": state,
        "static": static,
        "tend": tend,
        "aux": aux,
        "adv_batch_pt": adv_batch_pt
    }


def baroclinic_steady_state_case_init_fn(
    coord: CoordSysHybrid,
    para: PlanetParameterSet,
    equation: AbstractEquationSet,
    *,
    case: SteadyStateTestCase,
) -> Dict[str, Element]:
    full_kds = coord.vert.full_kds
    full_kde = coord.vert.full_kde
    elements = baroclinic_block_element_init_fn(
        coord=coord, equation=equation
    )
    gzs = init_gzs_in_baroclinic_case(
        coord=coord, equation=equation, test=case
    )
    gz = init_gz_in_baroclinic_case(
        coord=coord, equation=equation, test=case
    )
    gz_lev = init_gz_lev_in_case(
        coord=coord, equation=equation, test=case
    )
    mgs = init_mgs_in_baroclinic_case(
        coord=coord, equation=equation, test=case, gzs=gzs
    )
    mg = coord.vert.calc_mg(mgs, ks=full_kds, ke=full_kde)
    t = init_t_in_baroclinic_case(
        coord=coord, equation=equation, test=case
    )
    pt = formula.dry_potential_temperature(t=t, p=mg, para=para)
    un = init_un_in_baroclinic_case(
        coord=coord, equation=equation, test=case
    )
    ref_ps = mgs
    elements["state"] = tree_at(
        coord=coord,
        where=lambda s: (
            s.mgs, s.mg,
            s.gz, s.gz_lev,
            s.un, s.pt
        ),
        pytree=elements["state"],
        replace=(mgs, mg, gz, gz_lev, un, pt),
        at_dims=(0,) * 6,
        at_idxes=(0,) * 6
    )
    elements["static"] = tree_at(
        coord=coord,
        where=lambda s: (s.gzs, s.ref_ps),
        pytree=elements["static"],
        replace=(gzs, ref_ps),
    )
    elements["aux"] = tree_at(
        coord=coord,
        where=lambda a: a.t,
        pytree=elements["aux"],
        replace=t,
    )
    return elements


def baroclinic_mountain_wave_case_init_fn(
    coord: CoordSysHybrid,
    para: PlanetParameterSet,
    equation: AbstractEquationSet,
    *,
    case: MountainWaveTestCase,
) -> Dict[str, Element]:
    full_kds = coord.vert.full_kds
    full_kde = coord.vert.full_kde
    elements = baroclinic_block_element_init_fn(
        coord=coord, equation=equation
    )
    gzs = init_gzs_in_baroclinic_case(
        coord=coord, equation=equation, test=case
    )
    mgs = init_mgs_in_baroclinic_case(
        coord=coord, equation=equation, test=case, gzs=gzs
    )
    mg = coord.vert.calc_mg(mgs=mgs, ks=full_kds, ke=full_kde)
    pt = init_pt_in_baroclinic_case(
        coord=coord, equation=equation, test=case, mg=mg
    )
    un = init_un_in_baroclinic_case(
        coord=coord, equation=equation, test=case
    )
    ref_ps = mgs
    elements["state"] = tree_at(
        coord=coord,
        where=lambda s: (
            s.mgs, s.mg, s.un, s.pt
        ),
        pytree=elements["state"],
        replace=(mgs, mg, un, pt),
        at_dims=(0,) * 4,
        at_idxes=(0,) * 4
    )
    elements["static"] = tree_at(
        coord=coord,
        where=lambda s: (s.gzs, s.ref_ps),
        pytree=elements["static"],
        replace=(gzs, ref_ps)
    )
    return elements


def baroclinic_rossby_wave_case_init_fn(
    coord: CoordSysHybrid,
    para: PlanetParameterSet,
    equation: AbstractEquationSet,
    *,
    case: RossbyHaurwitzWaveTestCase,
) -> Dict[str, Element]:
    full_kds = coord.vert.full_kds
    full_kde = coord.vert.full_kde
    half_kds = coord.vert.half_kds
    half_kde = coord.vert.half_kde
    elements = baroclinic_block_element_init_fn(
        coord=coord, equation=equation
    )
    un = init_un_in_baroclinic_case(
        coord=coord, equation=equation, test=case
    )
    gzs = init_gzs_in_baroclinic_case(
        coord=coord, equation=equation, test=case
    )
    mgs = init_mgs_in_baroclinic_case(
        coord=coord, equation=equation, test=case, gzs=gzs
    )
    mg = coord.vert.calc_mg(mgs=mgs, ks=full_kds, ke=full_kde)
    mg_lev = coord.vert.calc_mg_lev(mgs=mgs, ks=half_kds, ke=half_kde)
    pt = init_pt_in_baroclinic_case(
        coord=coord, equation=equation, test=case, mg=mg
    )
    gz = init_gz_in_baroclinic_case(
        coord=coord, equation=equation, test=case, mg=mg
    )
    gz_lev = init_gz_lev_in_case(
        coord=coord, equation=equation, test=case, mg=mg_lev
    )
    ref_ps = mgs
    elements["state"] = tree_at(
        coord=coord,
        where=lambda s: (
            s.mgs, s.mg, s.mg_lev,
            s.gz, s.gz_lev,
            s.un, s.pt
        ),
        pytree=elements["state"],
        replace=(mgs, mg, mg_lev, gz, gz_lev, un, pt),
        at_dims=(0,) * 7,
        at_idxes=(0,) * 7
    )
    elements["static"] = tree_at(
        coord=coord,
        where=lambda s: (s.gzs, s.ref_ps),
        pytree=elements["static"],
        replace=(gzs, ref_ps)
    )
    return elements


def init_z_in_barotropic_case(
    coord: CoordSys,
    equation: AbstractEquationSet,
    test: ShallowWaterEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=equation.mode)["gz"]
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
    return test.init_geopotential_height(
        lon, lat
    ).reshape((1,) + lon.shape)  # add leading axis for vmap


def init_un_in_barotropic_case(
    coord: CoordSys,
    equation: AbstractEquationSet,
    test: ShallowWaterEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=equation.mode)["un"]
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
    return test.init_normal_velocity(
        lon, lat, cart, nr
    ).reshape((1,) + lon.shape)  # add leading axis for vmap


def init_zs_in_barotropic_case(
    coord: CoordSys,
    equation: AbstractEquationSet,
    test: ShallowWaterEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_static_meta(mode=equation.mode)["gzs"]
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    if horizontal_coord is None:
        raise ValueError(
            "Horizontal coordinate information is missing for gzs variable."
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    return test.init_topography(lon, lat)


def init_mgs_in_baroclinic_case(
    coord: CoordSys,
    equation: AbstractEquationSet,
    test: BaroclinicEquationTestCase,
    gzs: Array,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=equation.mode)["mgs"]
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
    return test.init_mgs(lon, lat, gzs)


def init_t_in_baroclinic_case(
    coord: CoordSys,
    equation: AbstractEquationSet,
    test: BaroclinicEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_auxiliary_meta(mode=equation.mode)["t"]
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
    return test.init_t(lon, lat, eta)


def init_pt_in_baroclinic_case(
    coord: CoordSys,
    equation: AbstractEquationSet,
    test: BaroclinicEquationTestCase,
    mg: Array,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=equation.mode)["pt"]
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
    return test.init_pt(lon, lat, eta, mg)


def init_un_in_baroclinic_case(
    coord: CoordSys,
    equation: AbstractEquationSet,
    test: BaroclinicEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=equation.mode)["un"]
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
    return test.init_normal_velocity(
        lon, lat, eta, cart, nr
    )


def init_gzs_in_baroclinic_case(
    coord: CoordSys,
    equation: AbstractEquationSet,
    test: BaroclinicEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_static_meta(mode=equation.mode)["gzs"]
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    if horizontal_coord is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    return test.init_gzs(lon, lat)


def init_gz_in_baroclinic_case(
    coord: CoordSys,
    equation: AbstractEquationSet,
    test: BaroclinicEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=equation.mode)["gz"]
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
    return test.init_gz(lon, lat, eta, *args, **kwargs)


def init_gz_lev_in_case(
    coord: CoordSys,
    equation: AbstractEquationSet,
    test: BaroclinicEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=equation.mode)["gz_lev"]
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
    return test.init_gz(lon, lat, eta, *args, **kwargs)


def default_state_meta(mode: AbstractEquationMode) -> dict[str, StaggeredFieldMetaData | NumericMetaData]:
    meta: dict[str, StaggeredFieldMetaData | NumericMetaData] = {}
    meta["un"] = StaggeredFieldMetaData(
        name="un",
        dims=(NLEV, NEDGE,) if mode.baroclinic else (NEDGE,),
        long_name="Normal velocity at edge",
        description="Normal velocity at edge",
        units="m/s",
        hori_location=MeshEntity.Edge,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
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
    meta["gz"] = StaggeredFieldMetaData(
        name="gz",
        dims=(NLEV, NCELL,) if mode.baroclinic else (NCELL,),
        long_name="Geopotential at primal cell center",
        description="Geopotential at primal cell center",
        units="m2 / s2",
        hori_location=MeshEntity.Cell,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
        basis=Basis.Scalar,
        coordinates="lon_cell lat_cell",
        staggered_kind="C",
        mesh_name="primal",
        is_prognostic=False if mode.baroclinic else True,
        is_expand_dims=True,
        expand_axis=0,
        expand_axis_length=mode.state_tile_depth,
        output="h0"
    )
    if mode.baroclinic:
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
            vert_location=VertLoc.Cell if mode.baroclinic else "",
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
            vert_location=VertLoc.Lev if mode.baroclinic else "",
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
            vert_location=VertLoc.Cell if mode.baroclinic else "",
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
            vert_location=VertLoc.Cell if mode.baroclinic else "",
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
            vert_location=VertLoc.Cell if mode.baroclinic else "",
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
            vert_location=VertLoc.Lev if mode.baroclinic else "",
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
    if not mode.baroclinic:
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
        meta["dmg_e"] = StaggeredFieldMetaData(
            name="dmg_e",
            dims=(NEDGE,),
            long_name="Fluid thickness at edge midpoint",
            description="Fluid thickness at edge midpoint",
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
    if mode.baroclinic:
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
    if mode.baroclinic:
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
    return meta


def default_tendency_meta(mode: AbstractEquationMode) -> dict[str, StaggeredFieldMetaData]:
    meta: dict[str, StaggeredFieldMetaData] = {}
    if not mode.baroclinic:
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
    if mode.baroclinic:
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
    meta["dundt"] = StaggeredFieldMetaData(
        name="dundt",
        dims=(NLEV, NEDGE,) if mode.baroclinic else (NEDGE,),
        long_name="Tendency of normal velocity at edge",
        description="Tendency of normal velocity at edge",
        units="m / s2",
        hori_location=MeshEntity.Edge,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
        basis=Basis.Normal,
        coordinates="lon_edge lat_edge",
        staggered_kind="C",
        mesh_name="primal",
        is_prognostic=False,
        output="h1"
    )
    meta["dundt_coriolis"] = StaggeredFieldMetaData(
        name="dundt_coriolis",
        dims=(NLEV, NEDGE,) if mode.baroclinic else (NEDGE,),
        long_name="Tendency of normal velocity at edge due to Coriolis force",
        description="Tendency of normal velocity at edge due to Coriolis force",
        units="m / s2",
        hori_location=MeshEntity.Edge,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
        basis=Basis.Normal,
        coordinates="lon_edge lat_edge",
        staggered_kind="C",
        mesh_name="primal",
        is_prognostic=False,
        output="h1" if mode.output_h1_tend else ""
    )
    meta["dundt_grad_ke"] = StaggeredFieldMetaData(
        name="dundt_grad_ke",
        dims=(NLEV, NEDGE,) if mode.baroclinic else (NEDGE,),
        long_name="Tendency of normal velocity at edge due to gradient of kinetic energy",
        description="Tendency of normal velocity at edge due to gradient of kinetic energy",
        units="m / s2",
        hori_location=MeshEntity.Edge,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
        basis=Basis.Normal,
        coordinates="lon_edge lat_edge",
        staggered_kind="C",
        mesh_name="primal",
        is_prognostic=False,
        output="h1" if mode.output_h1_tend else ""
    )
    meta["dundt_pgf"] = StaggeredFieldMetaData(
        name="dundt_pgf",
        dims=(NLEV, NEDGE,) if mode.baroclinic else (NEDGE,),
        long_name="Tendency of normal velocity at edge due to pressure gradient force",
        description="Tendency of normal velocity at edge due to pressure gradient force",
        units="m / s2",
        hori_location=MeshEntity.Edge,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
        basis=Basis.Normal,
        coordinates="lon_edge lat_edge",
        staggered_kind="C",
        mesh_name="primal",
        is_prognostic=False,
        output="h1" if mode.output_h1_tend else ""
    )
    if mode.baroclinic:
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
    if mode.baroclinic:
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
    meta["u"] = StaggeredFieldMetaData(
        name="u",
        dims=(NLEV, NCELL,) if mode.baroclinic else (NCELL,),
        long_name="Zonal velocity at primal cell center",
        description="Zonal velocity at primal cell center reconstructed from normal velocity at dual edge",
        units="m/s",
        hori_location=MeshEntity.Cell,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
        basis=Basis.Scalar,
        coordinates="lon_cell lat_cell",
        staggered_kind="C",
        mesh_name="primal",
        is_prognostic=False,
        output="h0"
    )
    meta["v"] = StaggeredFieldMetaData(
        name="v",
        dims=(NLEV, NCELL,) if mode.baroclinic else (NCELL,),
        long_name="Meridional velocity at primal cell center",
        description="Meridional velocity at primal cell center reconstructed from normal velocity at dual edge",
        units="m/s",
        hori_location=MeshEntity.Cell,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
        basis=Basis.Scalar,
        coordinates="lon_cell lat_cell",
        staggered_kind="C",
        mesh_name="primal",
        is_prognostic=False,
        output="h0"
    )
    if mode.baroclinic:
        meta["un_lev"] = StaggeredFieldMetaData(
            name="un_lev",
            dims=(NLEVP, NEDGE,) if mode.baroclinic else (NEDGE,),
            long_name="Normal velocity at edge and half level",
            description="Normal velocity at edge and half level",
            units="m/s",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Lev if mode.baroclinic else "",
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=True,
            output=""
        )
    meta["ut"] = StaggeredFieldMetaData(
        name="ut",
        dims=(NLEV, NEDGE,) if mode.baroclinic else (NEDGE,),
        long_name="Tangential velocity at edge",
        description="Tangential velocity at edge reconstructed from normal velocity at dual edge",
        units="m/s",
        hori_location=MeshEntity.Edge,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
        basis=Basis.Scalar,
        coordinates="lon_edge lat_edge",
        staggered_kind="C",
        mesh_name="primal",
        is_prognostic=False,
        output="h1"
    )
    meta["div"] = StaggeredFieldMetaData(
        name="div",
        dims=(NLEV, NCELL,) if mode.baroclinic else (NCELL,),
        long_name="Divergence at primal cell center",
        description="Divergence at primal cell center",
        units="1/s",
        hori_location=MeshEntity.Cell,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
        basis=Basis.Scalar,
        coordinates="lon_cell lat_cell",
        staggered_kind="C",
        mesh_name="primal",
        is_prognostic=False,
        output="h0"
    )
    meta["vor"] = StaggeredFieldMetaData(
        name="vor",
        dims=(NLEV, NVTX,) if mode.baroclinic else (NVTX,),
        long_name="Relative vorticity at dual cell center",
        description="Relative vorticity at dual cell center",
        units="1/s",
        hori_location=MeshEntity.Vtx,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
        basis=Basis.Scalar,
        coordinates="lon_vtx lat_vtx",
        staggered_kind="C",
        mesh_name="primal",
        is_prognostic=False,
        output="h0"
    )
    meta["pv_vtx"] = StaggeredFieldMetaData(
        name="pv_vtx",
        dims=(NLEV, NVTX,) if mode.baroclinic else (NVTX,),
        long_name="Potential vorticity at dual cell center",
        description="Potential vorticity at dual cell center",
        units="Pa-1 s-1",
        hori_location=MeshEntity.Vtx,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
        basis=Basis.Scalar,
        coordinates="lon_vtx lat_vtx",
        staggered_kind="C",
        mesh_name="primal",
        is_prognostic=False,
        output="h1"
    )
    meta["pv_e"] = StaggeredFieldMetaData(
        name="pv_e",
        dims=(NLEV, NEDGE,) if mode.baroclinic else (NEDGE,),
        long_name="Potential vorticity at edge",
        description="Potential vorticity at edge",
        units="Pa-1 s-1",
        hori_location=MeshEntity.Edge,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
        basis=Basis.Scalar,
        coordinates="lon_edge lat_edge",
        staggered_kind="C",
        mesh_name="primal",
        is_prognostic=False,
        output="h1"
    )
    meta["ke"] = StaggeredFieldMetaData(
        name="ke",
        dims=(NLEV, NCELL,) if mode.baroclinic else (NCELL,),
        long_name="Kinetic energy at primal cell center",
        description="Kinetic energy at primal cell center",
        units="m2 / s2",
        hori_location=MeshEntity.Cell,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
        basis=Basis.Scalar,
        coordinates="lon_cell lat_cell",
        staggered_kind="C",
        mesh_name="primal",
        is_prognostic=False,
        output="h1"
    )
    meta["mfn"] = StaggeredFieldMetaData(
        name="mfn",
        dims=(NLEV, NEDGE,) if mode.baroclinic else (NEDGE,),
        long_name="Mass flux at edge midpoint",
        description="Mass flux at edge midpoint",
        units="kg / (m s)",
        hori_location=MeshEntity.Edge,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
        basis=Basis.Scalar,
        coordinates="lon_edge lat_edge",
        staggered_kind="C",
        mesh_name="primal",
        is_prognostic=False,
        output="h1"
    )
    if mode.baroclinic:
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
    meta["dmf"] = StaggeredFieldMetaData(
        name="dmf",
        dims=(NLEV, NCELL,) if mode.baroclinic else (NCELL,),
        long_name="Divergence of mass flux at primal cell center",
        description="Divergence of mass flux at primal cell center",
        units="kg / (m s)",
        hori_location=MeshEntity.Cell,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
        basis=Basis.Scalar,
        coordinates="lon_cell lat_cell",
        staggered_kind="C",
        mesh_name="primal",
        is_prognostic=False,
        output="h1"
    )
    if mode.baroclinic:
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
