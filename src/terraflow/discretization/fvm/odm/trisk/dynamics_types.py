from typing import Self, List, Dict, Optional, TypeVar,  Generic
from loguru import logger
import equinox as eqx
import jax.numpy as jnp

from terraflow.util import typing, formula
from terraflow.planet import PlanetParameterSet
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.layout.abstract import (
    CoordinateSystem,
    Basis,
    MeshEntity,
    VertLoc,
)
from terraflow.layout.field import ArrayField, StaggeredFieldMetaData
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
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
)
from terraflow.discretization.abstract import (
    AbstractEquationMode,
    AbstractEquationSet,
    AbstractBlock,
    AbstractFields,
    AbstractEquationTermAuxiliaryDataset
)
from terraflow.discretization.fvm.odm.trisk.common.pgf.types import PressureGradientForceDataset
from terraflow.discretization.fvm.odm.trisk.common.coriolis.types import CoriolisForceAtEdgeData
from terraflow.discretization.fvm.odm.trisk.common.adv.types import AdvectionAuxiliaryDataset
from terraflow.discretization.fvm.odm.trisk.common.adv.batch import AdvBatchType

Array = typing.Array
FDT = ArrayField[StaggeredFieldMetaData]

VM = TypeVar("VM", OneLayerVertCoordinate, HybridVertCoordinate)
VLO = TypeVar("VLO", OneLayerVertCoordLayout, HybridVertCoordLayout)

CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    VM,
    TriSKBasedCGridStaggeringLayout,
    VLO
]

NLEVP = "ilev"
NLEV = "lev"
NCELL = "cell"
NEDGE = "edge"
NVTX = "vtx"


class Block(
    AbstractBlock[
        OrthogonalDualMesh,
        VM,
        TriSKBasedCGridStaggeringLayout,
        VLO,
        "State", "Tendency", "Static", "Auxiliary", AdvBatchType, "EquationTermAuxDataset"
    ],
    Generic[VM, VLO]
):
    mode: AbstractEquationMode = eqx.field(static=True)
    old: int
    new: int
    _coord: CoordSys
    para: PlanetParameterSet
    state: List["State"]
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
        total_steps: int = 3
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
                state, static, tend, aux, adv_batch_pt = baroclinic_steady_state_case_init_fn(
                    coord=coord, para=para, equation=equation, case=case
                )
            elif isinstance(case, MountainWaveTestCase):
                state, static, tend, aux, adv_batch_pt = baroclinic_mountain_wave_case_init_fn(
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
            state, static, tend, aux, adv_batch_pt = barotropic_case_init_fn(
                coord=coord, para=para, equation=equation, case=case
            )
            # ==============================================
        logger.success("Initialized block from testcase.")
        # =======================================================
        return cls(
            mode=equation.mode,
            old=0,
            new=1,
            _coord=coord,
            para=para,
            state=[state,] * total_steps,  # for time integration stages
            tend=tend,
            static=static,
            aux=aux,
            adv_batch_pt=adv_batch_pt,
            equaux=EquationTermAuxDataset.empty()
        )

    def diagnose(self, itime: int) -> Self:
        state = self.get_state(itime)
        uv = self.coord.mesh.reconstruct_vector(
            normal_u_at_edge=state.un.data,
        )
        self = eqx.tree_at(
            lambda b: (b.aux.u.data, b.aux.v.data),
            self,
            (uv["u"], uv["v"])
        )
        return self

    def reset(self, pass_tag: str) -> Self:
        return eqx.tree_at(
            lambda b: b.tend,
            self,
            replace=self.tend.reset(pass_tag=pass_tag),
        )


def barotropic_block_element_init_fn(
    coord: CoordSys,
    equation: AbstractEquationSet,
) -> tuple["State", "Static", "Tendency", "Auxiliary", "AdvBatchType"]:
    state = State.from_coord(coord=coord, equation=equation)
    static = Static.from_coord(coord=coord, equation=equation)
    tend = Tendency.from_coord(coord=coord, equation=equation)
    aux = Auxiliary.from_coord(coord=coord, equation=equation)
    adv_batch_pt = AdvBatchType.empty(mode=equation.mode)
    return state, static, tend, aux, adv_batch_pt


def barotropic_case_init_fn(
    coord: CoordSys,
    para: PlanetParameterSet,
    equation: AbstractEquationSet,
    *,
    case: ShallowWaterEquationTestCase,
) -> tuple["State", "Static", "Tendency", "Auxiliary", "AdvBatchType"]:
    g = para.gravity.unwrap()
    state, static, tend, aux, adv_batch_pt = barotropic_block_element_init_fn(
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
    state = eqx.tree_at(
        lambda s: (s.un.data, s.gz.data),
        state,
        (un, gz)
    )
    static = eqx.tree_at(
        lambda s: s.gzs.data,
        static,
        gzs
    )
    return state, static, tend, aux, adv_batch_pt


def baroclinic_block_element_init_fn(
    coord: CoordSys,
    equation: AbstractEquationSet,
) -> tuple["State", "Static", "Tendency", "Auxiliary", "AdvBatchType"]:
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
    return state, static, tend, aux, adv_batch_pt


def baroclinic_steady_state_case_init_fn(
    coord: CoordSys,
    para: PlanetParameterSet,
    equation: AbstractEquationSet,
    *,
    case: SteadyStateTestCase,
) -> tuple["State", "Static", "Tendency", "Auxiliary", "AdvBatchType"]:
    state, static, tend, aux, adv_batch_pt = baroclinic_block_element_init_fn(
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
    mg = coord.vert.calc_mg(mgs)
    t = init_t_in_baroclinic_case(
        coord=coord, equation=equation, test=case
    )
    pt = formula.dry_potential_temperature(t=t, p=mg, para=para)
    un = init_un_in_baroclinic_case(
        coord=coord, equation=equation, test=case
    )
    ref_ps = mgs
    state = eqx.tree_at(
        lambda s: (
            s.mgs.data, s.mg.data,
            s.gz.data, s.gz_lev.data,
            s.un.data, s.pt.data
        ),
        state,
        (mgs, mg, gz, gz_lev, un, pt)
    )
    static = eqx.tree_at(
        lambda s: (s.gzs.data, s.ref_ps.data),
        static,
        (gzs, ref_ps)
    )
    aux = eqx.tree_at(
        lambda a: a.t.data,
        aux,
        t
    )
    return state, static, tend, aux, adv_batch_pt


def baroclinic_mountain_wave_case_init_fn(
    coord: CoordSys,
    para: PlanetParameterSet,
    equation: AbstractEquationSet,
    *,
    case: MountainWaveTestCase,
) -> tuple["State", "Static", "Tendency", "Auxiliary", "AdvBatchType"]:
    state, static, tend, aux, adv_batch_pt = baroclinic_block_element_init_fn(
        coord=coord, equation=equation
    )
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
    gzs = init_gzs_in_baroclinic_case(
        coord=coord, equation=equation, test=case
    )
    mgs = init_mgs_in_baroclinic_case(
        coord=coord, equation=equation, test=case, gzs=gzs
    )
    mg = coord.vert.calc_mg(mgs)
    pt = init_pt_in_baroclinic_case(
        coord=coord, equation=equation, test=case, p=mg
    )
    un = init_un_in_baroclinic_case(
        coord=coord, equation=equation, test=case
    )
    ref_ps = mgs
    state = eqx.tree_at(
        lambda s: (
            s.mgs.data, s.mg.data, s.un.data, s.pt.data
        ),
        state,
        (mgs, mg, un, pt)
    )
    static = eqx.tree_at(
        lambda s: (s.gzs.data, s.ref_ps.data),
        static,
        (gzs, ref_ps)
    )
    return state, static, tend, aux, adv_batch_pt


def init_z_in_barotropic_case(
    coord: CoordSys,
    equation: AbstractEquationSet,
    test: ShallowWaterEquationTestCase,
) -> Array:
    meta = default_state_meta(mode=equation.mode)["gz"]
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
) -> Array:
    meta = default_state_meta(mode=equation.mode)["un"]
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
) -> Array:
    meta = default_static_meta(mode=equation.mode)
    horizontal_coord = meta["gzs"].get_hori_position(coord).horizontal_coord
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
    gzs: Array
) -> Array:
    meta = default_state_meta(mode=equation.mode)["mgs"]
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
    test: BaroclinicEquationTestCase
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
    p: Array
) -> Array:
    meta = default_state_meta(mode=equation.mode)["pt"]
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
    return test.init_pt(lon, lat, eta, p)


def init_un_in_baroclinic_case(
    coord: CoordSys,
    equation: AbstractEquationSet,
    test: BaroclinicEquationTestCase
) -> Array:
    meta = default_state_meta(mode=equation.mode)["un"]
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
    test: BaroclinicEquationTestCase
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
    test: BaroclinicEquationTestCase
) -> Array:
    meta = default_state_meta(mode=equation.mode)["gz"]
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
    return test.init_gz(lon, lat, eta)


def init_gz_lev_in_case(
    coord: CoordSys,
    equation: AbstractEquationSet,
    test: BaroclinicEquationTestCase
) -> Array:
    meta = default_state_meta(mode=equation.mode)["gz_lev"]
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
    return test.init_gz(lon, lat, eta)


def default_state_meta(mode: AbstractEquationMode) -> dict[str, StaggeredFieldMetaData]:
    meta: dict[str, StaggeredFieldMetaData] = {}
    meta["un"] = StaggeredFieldMetaData(
        name="un",
        dims=(NLEV, NEDGE,) if mode.baroclinic else (NEDGE,),
        long_name="Normal velocity at dual edge",
        description="Normal velocity at dual edge",
        units="m/s",
        hori_location=MeshEntity.Edge,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
        basis=Basis.Normal,
        coordinates="lon_edge lat_edge",
        staggered_kind="C",
        mesh_name="primal",
        is_prognostic=True,
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
            output="h1"
        )
        meta["dmg_e"] = StaggeredFieldMetaData(
            name="dmg_e",
            dims=(NEDGE,),
            long_name="Fluid thickness at primal edge midpoint",
            description="Fluid thickness at primal edge midpoint",
            units="m",
            hori_location=MeshEntity.Edge,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
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
            output="h0" if mode.nonhydrostatic else ""
        )
    return meta


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

    @staticmethod
    def default_meta(mode: AbstractEquationMode) -> dict[str, StaggeredFieldMetaData]:
        return default_state_meta(mode=mode)


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
        long_name="Tendency of normal velocity at dual edge",
        description="Tendency of normal velocity at dual edge",
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
        long_name="Tendency of normal velocity at dual edge due to Coriolis force",
        description="Tendency of normal velocity at dual edge due to Coriolis force",
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
        long_name="Tendency of normal velocity at dual edge due to gradient of kinetic energy",
        description="Tendency of normal velocity at dual edge due to gradient of kinetic energy",
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
        long_name="Tendency of normal velocity at dual edge due to pressure gradient force",
        description="Tendency of normal velocity at dual edge due to pressure gradient force",
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
            long_name="Tendency of normal velocity at dual edge due to vertical advection",
            description="Tendency of normal velocity at dual edge due to vertical advection",
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


class Static(AbstractFields[StaggeredFieldMetaData]):
    mode: AbstractEquationMode = eqx.field(static=True)
    fields: Dict[str, StaggeredFieldMetaData] = eqx.field(static=True)
    gzs: FDT
    ref_ps: FDT

    @staticmethod
    def default_meta(mode: AbstractEquationMode,) -> dict[str, StaggeredFieldMetaData]:
        return default_static_meta(mode=mode)


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
    meta["ut"] = StaggeredFieldMetaData(
        name="ut",
        dims=(NLEV, NEDGE,) if mode.baroclinic else (NEDGE,),
        long_name="Tangential velocity at dual edge",
        description="Tangential velocity at dual edge reconstructed from normal velocity at dual edge",
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
    meta["coriolis"] = StaggeredFieldMetaData(
        name="coriolis",
        dims=(NLEV, NEDGE,) if mode.baroclinic else (NEDGE,),
        long_name="Nonlinear Coriolis force at dual edge",
        description="Nonlinear Coriolis force at dual edge",
        units="m/s2",
        hori_location=MeshEntity.Edge,
        vert_location=VertLoc.Cell if mode.baroclinic else "",
        basis=Basis.Normal,
        coordinates="lon_edge lat_edge",
        staggered_kind="C",
        mesh_name="primal",
        is_prognostic=False,
        output=""
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
        output="h1"
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
        long_name="Mass flux at dual edge midpoint",
        description="Mass flux at dual edge midpoint",
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
            long_name="Mass flux at dual edge midpoint reconstructed from tangential velocity",
            description="Mass flux at dual edge midpoint reconstructed from tangential velocity",
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


class Auxiliary(AbstractFields[StaggeredFieldMetaData]):
    mode: AbstractEquationMode = eqx.field(static=True)
    fields: Dict[str, StaggeredFieldMetaData] = eqx.field(static=True)
    u: FDT
    v: FDT
    ut: FDT
    coriolis: FDT
    vor: FDT
    pv_vtx: FDT
    pv_e: FDT
    ke: FDT
    mfn: FDT
    mft: FDT
    mfz_lev: FDT
    dmf: FDT
    t: FDT
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
