from typing import Dict, Optional, Self
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import typing
from terraflow.planet import PlanetParameterSet
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.vert_coord import OneLayerVertCoordinate
from terraflow.layout.abstract import (
    CoordinateSystem,
    Basis,
    MeshEntity
)
from terraflow.layout.field import ArrayField, Field, StaggeredFieldMetaData
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.discretization.abstract import (
    AbstractFields,
    AbstractBlock,
    AbstractEquationTermAuxiliaryDataset
)
from terraflow.discretization.fvm.odm.trisk.coriolis.types import CoriolisForceAtEdgeData
from terraflow.discretization.fvm.odm.trisk.adv.types import AdvectionAuxiliaryDataset
from terraflow.test_case.swe import MountainZonalFlowTestCase, ShallowWaterEquationTestCase

Array = typing.Array

FDT = ArrayField[StaggeredFieldMetaData]

Coordsys = CoordinateSystem[
    OrthogonalDualMesh,
    OneLayerVertCoordinate,
    TriSKBasedCGridStaggeringLayout
]

NCELL = "ncell"
NEDGE = "nedge"
NVTX = "nvtx"


class Block(
    AbstractBlock[
        OrthogonalDualMesh,
        OneLayerVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        "State", "Static", "Tendency", "Auxiliary", "EquationTermAuxDataset"
    ]
):
    coord: Coordsys
    para: PlanetParameterSet
    state: "State"
    tend: "Tendency"
    static: "Static"
    aux: "Auxiliary"
    equaux: "EquationTermAuxDataset"

    @classmethod
    def from_testcase(
        cls,
        coord: Coordsys,
        para: PlanetParameterSet,
        *,
        case: Optional[ShallowWaterEquationTestCase] = None,
    ) -> Self:
        if case is None:
            case = MountainZonalFlowTestCase(para)
        state = State.from_testcase(coord, para, case=case)
        static = Static.from_testcase(coord, para, case=case)
        tend = Tendency.from_coord(coord)
        aux = Auxiliary.from_coord(coord)
        equaux = EquationTermAuxDataset.empty()
        return cls(
            coord=coord, para=para,
            state=state, tend=tend, static=static,
            aux=aux, equaux=equaux
        )

    def diagnose(self) -> Self:
        uv = self.coord.mesh.reconstruct_vector(
            normal_u_at_edge=self.state.un.data,
        )
        self = eqx.tree_at(
            lambda b: (b.state.u.data, b.state.v.data),
            self,
            (uv["u"], uv["v"])
        )
        return self


def default_state_meta() -> Dict[str, StaggeredFieldMetaData]:
    """Get the default meta for shallow water state variables."""
    return {
        "z": StaggeredFieldMetaData(
            name="z",
            dims=(NCELL,),
            long_name="Geopotential height at primal cell",
            description="Geopotential height at primal cell centers",
            units="m",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            staggered_kind="C",
            mesh_name="primal",
            coordinates="lon_cell lat_cell",
            is_prognostic=True,
            output="h0"
        ),
        "un": StaggeredFieldMetaData(
            name="un",
            dims=(NEDGE,),
            long_name="normal velocity at dual edge midpoints",
            description="Normal velocity at dual edge midpoints",
            units="m/s",
            hori_location=MeshEntity.Edge,
            vert_location="",
            basis=Basis.Normal,
            staggered_kind="C",
            mesh_name="primal",
            coordinates="lon_edge lat_edge",
            is_prognostic=True,
            output="h0"
        ),
        "u": StaggeredFieldMetaData(
            name="u",
            dims=("ncell",),
            long_name="Zonal velocity at primal cell center",
            description="Zonal velocity at primal cell center reconstructed from normal velocity at dual edge",
            units="m/s",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.NormalX,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h0"
        ),
        "v": StaggeredFieldMetaData(
            name="v_cell",
            dims=("ncell",),
            long_name="Meridional velocity at primal cell center",
            description="Meridional velocity at primal cell center reconstructed from normal velocity at dual edge",
            units="m/s",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.NormalY,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h0"
        ),
    }


def init_z_in_case(
    coord: Coordsys,
    test: ShallowWaterEquationTestCase,
) -> Array:
    meta = default_state_meta()
    lon = meta["z"].get_position(coord).horizontal_coord["lon"]
    lat = meta["z"].get_position(coord).horizontal_coord["lat"]
    return test.init_geopotential_height(
        lon, lat
    ).reshape((1,) + lon.shape)  # add leading axis for vmap


def init_un_in_case(
    coord: Coordsys,
    test: ShallowWaterEquationTestCase,
) -> Array:
    meta = default_state_meta()
    lon = meta["un"].get_position(coord).horizontal_coord["lon"]
    lat = meta["un"].get_position(coord).horizontal_coord["lat"]
    cart = meta["un"].get_position(coord).horizontal_coord["cart"]
    nr = meta["un"].get_position(coord).horizontal_coord["nr_orient"]
    return test.init_normal_velocity(
        lon, lat, cart, nr
    ).reshape((1,) + lon.shape)  # add leading axis for vmap


class State(AbstractFields[StaggeredFieldMetaData]):
    z: FDT
    un: FDT
    u: FDT
    v: FDT

    @staticmethod
    def default_meta() -> dict[str, StaggeredFieldMetaData]:
        return default_state_meta()

    @classmethod
    def from_testcase(
        cls,
        coord: Coordsys,
        para: PlanetParameterSet,
        *,
        case: Optional[ShallowWaterEquationTestCase] = None,
    ) -> Self:
        if case is None:
            case = MountainZonalFlowTestCase(para)
        z = init_z_in_case(coord, case)
        un = init_un_in_case(coord, case)
        return cls.from_arrays(
            array_dict={
                "z": z,
                "un": un,
                "u": jnp.zeros_like(z),
                "v": jnp.zeros_like(z),
            },
            meta=cls.default_meta()
        )


def default_static_meta() -> Dict[str, StaggeredFieldMetaData]:
    return {
        "zs": StaggeredFieldMetaData(
            name="zs",
            dims=(NCELL,),
            long_name="topography at primal cell",
            description="Topography height at primal cell centers",
            units="m",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            staggered_kind="C",
            mesh_name="primal",
            coordinates="lon_cell lat_cell",
            is_prognostic=False,
            output=""
        )
    }


def init_topo_in_case(
    coord: Coordsys,
    test: ShallowWaterEquationTestCase,
) -> Array:
    meta = default_static_meta()
    coord_info = meta["zs"].get_position(coord)
    lon = coord_info.horizontal_coord["lon"]
    lat = coord_info.horizontal_coord["lat"]
    return test.init_topography(lon, lat)


class Static(AbstractFields[StaggeredFieldMetaData]):
    zs: FDT

    @staticmethod
    def default_meta() -> dict[str, StaggeredFieldMetaData]:
        return default_static_meta()

    @classmethod
    def from_testcase(
        cls,
        coord: Coordsys,
        para: PlanetParameterSet,
        *,
        case: Optional[ShallowWaterEquationTestCase] = None,
    ) -> Self:
        if case is None:
            case = MountainZonalFlowTestCase(para)
        return cls.from_arrays(
            array_dict={
                "zs": init_topo_in_case(coord, case)
            },
            meta=cls.default_meta()
        )


def default_tendency_meta() -> Dict[str, StaggeredFieldMetaData]:
    return {
        "dzdt": StaggeredFieldMetaData(
            name="dzdt",
            dims=(NCELL,),
            long_name="Tendency of geopotential height",
            description="Tendency of geopotential height calculated from state variables",
            units="m/s",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        ),
        "dzdt_mfn_e": StaggeredFieldMetaData(
            name="dzdt_mfn_e",
            dims=(NCELL,),
            long_name="Tendency of geopotential height due to normal mass flux divergence term",
            description="Tendency of geopotential height due to normal mass flux divergence term calculated from state variables",
            units="1/s",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        ),
        "dundt": StaggeredFieldMetaData(
            name="dundt",
            dims=(NEDGE,),
            long_name="Tendency of normal velocity",
            description="Tendency of normal velocity calculated from state variables",
            units="m/s2",
            hori_location=MeshEntity.Edge,
            vert_location="",
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        ),
        "dundt_coriolis": StaggeredFieldMetaData(
            name="dundt_coriolis",
            dims=(NEDGE,),
            long_name="Tendency of normal velocity due to coriolis term",
            description="Tendency of normal velocity due to coriolis term calculated from state variables",
            units="m/s2",
            hori_location=MeshEntity.Edge,
            vert_location="",
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        ),
        "dundt_grad_ke": StaggeredFieldMetaData(
            name="dundt_grad_ke",
            dims=(NEDGE,),
            long_name="Tendency of normal velocity due to gradient of kinetic energy term",
            description="Tendency of normal velocity due to gradient of kinetic energy term calculated from state variables",
            units="m/s2",
            hori_location=MeshEntity.Edge,
            vert_location="",
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        ),
        "dundt_pgf": StaggeredFieldMetaData(
            name="dundt_pgf",
            dims=(NEDGE,),
            long_name="Tendency of normal velocity due to pressure gradient force term",
            description="Tendency of normal velocity due to pressure gradient force term calculated from state variables",
            units="m/s2",
            hori_location=MeshEntity.Edge,
            vert_location="",
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        ),
    }


class Tendency(AbstractFields[StaggeredFieldMetaData]):
    dzdt: FDT
    dzdt_mfn_e: FDT
    dundt: FDT
    dundt_coriolis: FDT
    dundt_grad_ke: FDT
    dundt_pgf: FDT

    @staticmethod
    def default_meta() -> dict[str, StaggeredFieldMetaData]:
        return default_tendency_meta()

    @classmethod
    def from_coord(
        cls,
        coord: Coordsys,
    ) -> Self:
        meta = cls.default_meta()
        n_polg = coord.mesh.primal.cell.centroid.lon.shape[0]
        n_edge = coord.mesh.primal.edge.midpoint.lon.shape[0]
        float_type = coord.mesh.dev.float_type
        array_dict = {}
        array_dict["dzdt"] = jnp.zeros(
            (1, n_polg,), dtype=float_type
        )
        array_dict["dzdt_mfn_e"] = jnp.zeros(
            (1, n_polg,), dtype=float_type
        )
        array_dict["dundt"] = jnp.zeros(
            (1, n_edge,), dtype=float_type
        )
        array_dict["dundt_coriolis"] = jnp.zeros(
            (1, n_edge,), dtype=float_type
        )
        array_dict["dundt_grad_ke"] = jnp.zeros(
            (1, n_edge,), dtype=float_type
        )
        array_dict["dundt_pgf"] = jnp.zeros(
            (1, n_edge,), dtype=float_type
        )
        return cls.from_arrays(array_dict=array_dict, meta=meta)

    def _tend2state(self, pass_tag: str = "all_pass") -> dict[str, Array]:
        """Get a mapping from tendency variable names to state variable names.

        Args:
            pass_tag: A string indicating which tendencies to include.
                - "all_pass": include all tendencies
                - "forward_pass": include only tendencies related to forward euler step.
                   other tendencies are set to zero.
                - "backward_pass": include only tendencies related to backward euler step.
                   other tendencies are set to zero.
        Returns:
            A dictionary mapping tendency variable names to state variable names.
        """
        if pass_tag == "all_pass":
            return {
                "z": self.dzdt.data,
                "un": self.dundt.data,
                "u": jnp.empty(0),
                "v": jnp.empty(0),
            }
        elif pass_tag == "forward_pass":
            return {
                "z": self.dzdt.data,
                "un": self.zeros_like(self.dundt).data,
                "u": jnp.empty(0),
                "v": jnp.empty(0),
            }
        elif pass_tag == "backward_pass":
            return {
                "z": self.zeros_like(self.dzdt).data,
                "un": self.dundt.data,
                "u": jnp.empty(0),
                "v": jnp.empty(0),
            }
        else:
            raise ValueError(f"Invalid pass_tag: {pass_tag}")

    def to_state(self, *, pass_tag: str) -> State:
        return State.from_arrays(
            array_dict={
                key: var for key, var in self._tend2state(pass_tag=pass_tag).items()
            },
            meta=default_state_meta()
        )

    def zeros_like(
        self,
        data: ArrayField[StaggeredFieldMetaData],
    ) -> ArrayField[StaggeredFieldMetaData]:
        return Field.create(
            data=jnp.zeros_like(data.data),
            meta=data.meta,
            as_quantity=False
        )


def default_aux_meta() -> Dict[str, StaggeredFieldMetaData]:
    """Get the default metadata for shallow water auxiliary variables."""
    return {
        "ut": StaggeredFieldMetaData(
            name="ut",
            dims=(NEDGE,),
            long_name="Tangential velocity at dual edge midpoint",
            description="Tangential velocity at dual edge midpoint",
            units="m/s",
            hori_location=MeshEntity.Edge,
            vert_location="",
            basis=Basis.Tangent,
            staggered_kind="C",
            coordinates="lon_edge lat_edge",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        ),
        "z_e": StaggeredFieldMetaData(
            name="z_e",
            dims=(NEDGE,),
            long_name="Geopotential height at dual edge midpoint",
            description="Geopotential height at dual edge midpoint",
            units="m",
            hori_location=MeshEntity.Edge,
            vert_location="",
            basis=Basis.Scalar,
            staggered_kind="C",
            coordinates="lon_edge lat_edge",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        ),
        "z_vtx": StaggeredFieldMetaData(
            name="z_vtx",
            dims=(NVTX,),
            long_name="Geopotential height at dual cell center",
            description="Geopotential height at dual cell center",
            units="m",
            hori_location=MeshEntity.Vtx,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_vtx lat_vtx",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        ),
        "mfn_e": StaggeredFieldMetaData(
            name="mfn_e",
            dims=(NEDGE,),
            long_name="Mass flux at dual edge midpoint",
            description="Mass flux at dual edge midpoint",
            units="m2 /s",
            hori_location=MeshEntity.Edge,
            vert_location="",
            basis=Basis.Normal,
            staggered_kind="C",
            coordinates="lon_edge lat_edge",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        ),
        "rel_vor": StaggeredFieldMetaData(
            name="rel_vor",
            dims=(NVTX,),
            long_name="Relative vorticity at dual cell center",
            description="Relative vorticity at dual cell center",
            units="1/s",
            hori_location=MeshEntity.Vtx,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_vtx lat_vtx",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        ),
        "abs_vor": StaggeredFieldMetaData(
            name="abs_vor",
            dims=(NVTX,),
            long_name="Absolute vorticity at dual cell center",
            description="Absolute vorticity at dual cell center",
            units="1/s",
            hori_location=MeshEntity.Vtx,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_vtx lat_vtx",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        ),
        "pv_vtx": StaggeredFieldMetaData(
            name="pv_vtx",
            dims=(NVTX,),
            long_name="Potential vorticity at dual cell center",
            description=(
                "Potential vorticity at dual cell center calculated from "
                "absolute vorticity and geopotential height at dual cell center"
            ),
            units="1/(m s)",
            hori_location=MeshEntity.Vtx,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_vtx lat_vtx",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        ),
        "pv_e": StaggeredFieldMetaData(
            name="pv_e",
            dims=(NEDGE,),
            long_name="Potential vorticity at dual edge midpoint",
            description="Potential vorticity at dual edge midpoint",
            units="1/(m s)",
            hori_location=MeshEntity.Edge,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        ),
        "ke": StaggeredFieldMetaData(
            name="ke",
            dims=(NCELL,),
            long_name="Kinetic energy at primal cell center",
            description="Kinetic energy at primal cell center",
            units="m2/s2",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
    }


class Auxiliary(AbstractFields[StaggeredFieldMetaData]):
    ut: FDT
    z_e: FDT
    mfn_e: FDT
    z_vtx: FDT
    rel_vor: FDT
    abs_vor: FDT
    pv_vtx: FDT
    pv_e: FDT
    ke: FDT

    @staticmethod
    def default_meta() -> dict[str, StaggeredFieldMetaData]:
        return default_aux_meta()

    @classmethod
    def from_coord(
        cls,
        coord: Coordsys,
    ) -> Self:
        meta = cls.default_meta()
        n_polg = coord.mesh.primal.cell.centroid.lon.shape[0]
        n_edge = coord.mesh.primal.edge.midpoint.lon.shape[0]
        n_dual = coord.mesh.dual.cell.centroid.lon.shape[0]
        float_type = coord.mesh.dev.float_type
        array_dict = {}
        array_dict["ut"] = jnp.zeros((1, n_edge,), dtype=float_type)
        array_dict["z_e"] = jnp.zeros((1, n_edge,), dtype=float_type)
        array_dict["mfn_e"] = jnp.zeros((1, n_edge,), dtype=float_type)
        array_dict["z_vtx"] = jnp.zeros((1, n_dual,), dtype=float_type)
        array_dict["rel_vor"] = jnp.zeros((1, n_dual,), dtype=float_type)
        array_dict["abs_vor"] = jnp.zeros((1, n_dual,), dtype=float_type)
        array_dict["pv_vtx"] = jnp.zeros((1, n_dual,), dtype=float_type)
        array_dict["pv_e"] = jnp.zeros((1, n_edge,), dtype=float_type)
        array_dict["ke"] = jnp.zeros((1, n_polg,), dtype=float_type)
        return cls.from_arrays(array_dict=array_dict, meta=meta)


class EquationTermAuxDataset(AbstractEquationTermAuxiliaryDataset):
    coriolis: CoriolisForceAtEdgeData
    adv: AdvectionAuxiliaryDataset

    @classmethod
    def empty(cls) -> Self:
        coriolis = CoriolisForceAtEdgeData.empty()
        adv = AdvectionAuxiliaryDataset.empty()
        return cls(
            coriolis=coriolis,
            adv=adv
        )
