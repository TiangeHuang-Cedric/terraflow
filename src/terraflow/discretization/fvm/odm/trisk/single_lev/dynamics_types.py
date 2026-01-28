from tkinter import N
from typing import Dict, Optional, Self, TypeVar
import jax.numpy as jnp

from terraflow.util import typing
from terraflow.planet import PlanetParameterSet
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.vert_coord import OneLayerVertCoordinate, HybridVertCoordinate
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
)
from terraflow.test_case.swe import MountainZonalFlowTestCase, ShallowWaterEquationTestCase


VM = TypeVar("VM", OneLayerVertCoordinate, HybridVertCoordinate)

Array = typing.Array

FDT = ArrayField[StaggeredFieldMetaData]

Coordsys = CoordinateSystem[
    OrthogonalDualMesh,
    VM,
    TriSKBasedCGridStaggeringLayout
]

NCELL = "ncell"
NEDGE = "nedge"
NVTX = "nvtx"


class Block(AbstractBlock["State", "Static", "Tendency", "Auxiliary"]):
    state: "State"
    static: "Static"
    tend: "Tendency"
    aux: "Auxiliary"

    @staticmethod
    def state_cls() -> type["State"]:
        return State

    @staticmethod
    def static_cls() -> type["Static"]:
        return Static

    @staticmethod
    def tend_cls() -> type["Tendency"]:
        return Tendency

    @staticmethod
    def aux_cls() -> type["Auxiliary"]:
        return Auxiliary

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
        state: "State" = cls.state_cls().from_testcase(coord, para, case=case)
        static: "Static" = cls.static_cls().from_testcase(coord, para, case=case)
        aux: "Auxiliary" = cls.aux_cls().from_coord(coord)
        tend: "Tendency" = cls.tend_cls().from_coord(coord)
        return cls(state, static, tend, aux)


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
        )
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
        return cls.from_arrays(
            array_dict={
                "z": init_z_in_case(coord, case),
                "un": init_un_in_case(coord, case)
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
            mesh_name="primal"
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
            mesh_name="primal"
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
            mesh_name="primal"
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
            mesh_name="primal"
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
            mesh_name="primal"
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
            mesh_name="primal"
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
        n_polg = coord.mesh.n_polg
        n_edge = coord.mesh.n_edge
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

    def _tend2state(self, pass_tag: str = "all_pass") -> dict[str, ArrayField[StaggeredFieldMetaData]]:
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
                "z": self.dzdt,
                "un": self.dundt
            }
        elif pass_tag == "forward_pass":
            return {
                "z": self.dzdt,
                "un": self.zeros_like(self.dundt)
            }
        elif pass_tag == "backward_pass":
            return {
                "z": self.zeros_like(self.dzdt),
                "un": self.dundt
            }
        else:
            raise ValueError(f"Invalid pass_tag: {pass_tag}")

    def to_state(self, pass_tag: str = "all") -> State:
        return State.from_arrays(
            array_dict={
                key: var.data for key, var in self._tend2state(pass_tag=pass_tag).items()
            },
            meta=State.default_meta()
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
            mesh_name="primal"
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
            mesh_name="primal"
        ),
        "z_vtx": StaggeredFieldMetaData(
            name="z_vtx",
            dims=(NVTX,),
            long_name="Geopotential height at dual cell center",
            description="Geopotential height at dual cell center",
            units="m",
            hori_location=MeshEntity.vtx,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_vtx lat_vtx",
            staggered_kind="C",
            mesh_name="primal"
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
            mesh_name="primal"
        ),
        "rel_vor": StaggeredFieldMetaData(
            name="rel_vor",
            dims=(NVTX,),
            long_name="Relative vorticity at dual cell center",
            description="Relative vorticity at dual cell center",
            units="1/s",
            hori_location=MeshEntity.vtx,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_vtx lat_vtx",
            staggered_kind="C",
            mesh_name="primal"
        ),
        "abs_vor": StaggeredFieldMetaData(
            name="abs_vor",
            dims=(NVTX,),
            long_name="Absolute vorticity at dual cell center",
            description="Absolute vorticity at dual cell center",
            units="1/s",
            hori_location=MeshEntity.vtx,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_vtx lat_vtx",
            staggered_kind="C",
            mesh_name="primal"
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
            hori_location=MeshEntity.vtx,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_vtx lat_vtx",
            staggered_kind="C",
            mesh_name="primal"
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
            mesh_name="primal"
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
            mesh_name="primal"
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
        n_polg = coord.mesh.n_polg
        n_edge = coord.mesh.n_edge
        n_dual = coord.mesh.n_dual
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
