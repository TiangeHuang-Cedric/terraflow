from abc import abstractmethod
from typing import Dict, Optional, Self, Generic, TypeVar
import jax.numpy as jnp
import unxt as ux

from terraflow.util import typing
from terraflow.planet import PlanetParameterSet
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.vert_coord import OneLayerVertCoordinate, HybridVertCoordinate
from terraflow.layout.abstract import (
    CoordinateSystem,
    Basis,
    MeshEntity
)
from terraflow.layout.field import Field, StaggeredFieldMetaData
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.discretization.abstract import (
    AbstractVariableSet,
    AbstractFieldBasedVariableSet,
    AbstractQuantityBasedVariableSet,
    AbstractArrayBasedVariableSet,
    AbstractBlock,
    AbstractBlockFields,
    AbstractBlockQuantities,
    AbstractBlockArrays
)
from terraflow.test_case.swe import MountainZonalFlowTestCase, ShallowWaterEquationTestCase


VM = TypeVar("VM", OneLayerVertCoordinate, HybridVertCoordinate)
DT = TypeVar("DT", "Array", Field, ux.AbstractQuantity)
STATE = TypeVar(
    "STATE",
    "StateFields", "StateQuantities", "StateArrays"
)
STATIC = TypeVar(
    "STATIC", "StaticFields", "StaticQuantities", "StaticArrays"
)
TEND = TypeVar(
    "TEND",
    "TendencyFields", "TendencyQuantities", "TendencyArrays"
)
AUX = TypeVar(
    "AUX",
    "AuxiliaryFields", "AuxiliaryQuantities", "AuxiliaryArrays"
)

Array = typing.Array
ArrayWithUnits = typing.ArrayWithUnits


Coordsys = CoordinateSystem[
    OrthogonalDualMesh,
    VM,
    TriSKBasedCGridStaggeringLayout
]


class BaseBlock(AbstractBlock[STATE, STATIC, TEND, AUX], Generic[STATE, STATIC, TEND, AUX]):

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
        state: STATE = cls.state_cls().from_testcase(coord, para, case=case)
        static: STATIC = cls.static_cls().from_testcase(coord, para, case=case)
        aux: AUX = cls.aux_cls().from_coord(coord)
        tend: TEND = cls.tend_cls().from_coord(coord)
        return cls(state, static, tend, aux)


class BlockFields(
    AbstractBlockFields[
        "StateFields",
        "StaticFields",
        "TendencyFields",
        "AuxiliaryFields",
        "BlockQuantities",
        "BlockArrays"
    ],
    BaseBlock
):
    state: "StateFields"
    static: "StaticFields"
    tend: "TendencyFields"
    aux: "AuxiliaryFields"

    @staticmethod
    def state_cls() -> type["StateFields"]:
        return StateFields

    @staticmethod
    def static_cls() -> type["StaticFields"]:
        return StaticFields

    @staticmethod
    def tend_cls() -> type["TendencyFields"]:
        return TendencyFields

    @staticmethod
    def aux_cls() -> type["AuxiliaryFields"]:
        return AuxiliaryFields

    @staticmethod
    def quantity_block_type() -> type["BlockQuantities"]:
        return BlockQuantities

    @staticmethod
    def array_block_type() -> type["BlockArrays"]:
        return BlockArrays


class BlockArrays(
    AbstractBlockArrays[
        "StateArrays",
        "StaticArrays",
        "TendencyArrays",
        "AuxiliaryArrays",
        "BlockFields",
        "BlockQuantities"
    ],
    BaseBlock
):
    state: "StateArrays"
    static: "StaticArrays"
    tend: "TendencyArrays"
    aux: "AuxiliaryArrays"

    @staticmethod
    def state_cls() -> type["StateArrays"]:
        return StateArrays

    @staticmethod
    def static_cls() -> type["StaticArrays"]:
        return StaticArrays

    @staticmethod
    def tend_cls() -> type["TendencyArrays"]:
        return TendencyArrays

    @staticmethod
    def aux_cls() -> type["AuxiliaryArrays"]:
        return AuxiliaryArrays

    @staticmethod
    def quantity_block_type() -> type["BlockQuantities"]:
        return BlockQuantities

    @staticmethod
    def field_block_type() -> type["BlockFields"]:
        return BlockFields


class BlockQuantities(
    AbstractBlockQuantities[
        "StateQuantities",
        "StaticQuantities",
        "TendencyQuantities",
        "AuxiliaryQuantities",
        "BlockFields",
        "BlockArrays"
    ],
    BaseBlock
):
    state: "StateQuantities"
    static: "StaticQuantities"
    tend: "TendencyQuantities"
    aux: "AuxiliaryQuantities"

    @staticmethod
    def state_cls() -> type["StateQuantities"]:
        return StateQuantities

    @staticmethod
    def static_cls() -> type["StaticQuantities"]:
        return StaticQuantities

    @staticmethod
    def tend_cls() -> type["TendencyQuantities"]:
        return TendencyQuantities

    @staticmethod
    def aux_cls() -> type["AuxiliaryQuantities"]:
        return AuxiliaryQuantities

    @staticmethod
    def field_block_type() -> type["BlockFields"]:
        return BlockFields

    @staticmethod
    def array_block_type() -> type["BlockArrays"]:
        return BlockArrays


def default_state_meta() -> Dict[str, StaggeredFieldMetaData]:
    """Get the default meta for shallow water state variables."""
    return {
        "h_pc": StaggeredFieldMetaData(
            name="h_pc",
            dims=("nPrimal_Cell",),
            long_name="thickness at primal cell",
            description="Fluid thickness at primal cell centers",
            units="m",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            staggered_kind="C",
            mesh_name="primal",
            coordinates="lon_pc lat_pc",
        ),
        "un_e": StaggeredFieldMetaData(
            name="un_e",
            dims=("nDual_Edge",),
            long_name="normal velocity at dual edge midpoints",
            description="Normal velocity at dual edge midpoints",
            units="m/s",
            hori_location=MeshEntity.Edge,
            vert_location="",
            basis=Basis.Normal,
            staggered_kind="C",
            mesh_name="primal",
            coordinates="lon_de lat_de",
        )
    }


def init_h_in_case(
    coord: Coordsys,
    test: ShallowWaterEquationTestCase,
) -> Array:
    meta = default_state_meta()
    lon = meta["h_pc"].get_position(coord).horizontal_coord["lon"]
    lat = meta["h_pc"].get_position(coord).horizontal_coord["lat"]
    return test.init_thickness(
        lon, lat
    ).reshape((1,) + lon.shape)  # add leading axis for vmap


def init_un_in_case(
    coord: Coordsys,
    test: ShallowWaterEquationTestCase,
) -> Array:
    meta = default_state_meta()
    lon = meta["un_e"].get_position(coord).horizontal_coord["lon"]
    lat = meta["un_e"].get_position(coord).horizontal_coord["lat"]
    cart = meta["un_e"].get_position(coord).horizontal_coord["cart"]
    nr = meta["un_e"].get_position(coord).horizontal_coord["nr_orient"]
    return test.init_normal_velocity(
        lon, lat, cart, nr
    ).reshape((1,) + lon.shape)  # add leading axis for vmap


class BaseState(AbstractVariableSet, Generic[DT]):
    h_pc: DT
    un_e: DT

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
                "h_pc": init_h_in_case(coord, case),
                "un_e": init_un_in_case(coord, case)
            },
            meta=cls.default_meta()
        )


class StateArrays(
    AbstractArrayBasedVariableSet[
        "StateFields", "StateQuantities", StaggeredFieldMetaData
    ],
    BaseState[Array],
):
    @staticmethod
    def default_meta() -> dict[str, StaggeredFieldMetaData]:
        return default_state_meta()

    @staticmethod
    def quantity_varset_type() -> type["StateQuantities"]:
        return StateQuantities

    @staticmethod
    def field_varset_type() -> type["StateFields"]:
        return StateFields


class StateQuantities(
    AbstractQuantityBasedVariableSet[
        "StateFields", "StateArrays", StaggeredFieldMetaData
    ],
    BaseState[ux.AbstractQuantity],
):
    @staticmethod
    def default_meta() -> dict[str, StaggeredFieldMetaData]:
        return default_state_meta()

    @staticmethod
    def array_varset_type() -> type["StateArrays"]:
        return StateArrays

    @staticmethod
    def field_varset_type() -> type["StateFields"]:
        return StateFields


class StateFields(
    AbstractFieldBasedVariableSet[
        "StateQuantities", "StateArrays", StaggeredFieldMetaData
    ],
    BaseState[Field[Array, StaggeredFieldMetaData]]
):
    @staticmethod
    def default_meta() -> dict[str, StaggeredFieldMetaData]:
        return default_state_meta()

    @staticmethod
    def array_varset_type() -> type["StateArrays"]:
        return StateArrays

    @staticmethod
    def quantity_varset_type() -> type["StateQuantities"]:
        return StateQuantities


def default_static_meta() -> Dict[str, StaggeredFieldMetaData]:
    return {
        "topo": StaggeredFieldMetaData(
            name="hs_pc",
            dims=("nPrimal_Cell",),
            long_name="topography at primal cell",
            description="Topography height at primal cell centers",
            units="m",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            staggered_kind="C",
            mesh_name="primal",
            coordinates="lon_pc lat_pc",
        )
    }


def init_topo_in_case(
    coord: Coordsys,
    test: ShallowWaterEquationTestCase,
) -> Array:
    meta = default_static_meta()
    coord_info = meta["topo"].get_position(coord)
    lon = coord_info.horizontal_coord["lon"]
    lat = coord_info.horizontal_coord["lat"]
    return test.init_topography(lon, lat)


class BaseStatic(AbstractVariableSet, Generic[DT]):
    topo: DT

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
                "topo": init_topo_in_case(coord, case)
            },
            meta=cls.default_meta()
        )


class StaticArrays(
    AbstractArrayBasedVariableSet[
        "StaticFields", "StaticQuantities", StaggeredFieldMetaData
    ],
    BaseStatic[Array]
):
    @staticmethod
    def default_meta() -> dict[str, StaggeredFieldMetaData]:
        return default_static_meta()

    @staticmethod
    def quantity_varset_type() -> type["StaticQuantities"]:
        return StaticQuantities

    @staticmethod
    def field_varset_type() -> type["StaticFields"]:
        return StaticFields


class StaticQuantities(
    AbstractQuantityBasedVariableSet[
        "StaticFields", "StaticArrays", StaggeredFieldMetaData
    ],
    BaseStatic[ux.AbstractQuantity]
):
    @staticmethod
    def default_meta() -> dict[str, StaggeredFieldMetaData]:
        return default_static_meta()

    @staticmethod
    def array_varset_type() -> type["StaticArrays"]:
        return StaticArrays

    @staticmethod
    def field_varset_type() -> type["StaticFields"]:
        return StaticFields


class StaticFields(
    AbstractFieldBasedVariableSet[
        "StaticQuantities", "StaticArrays", StaggeredFieldMetaData
    ],
    BaseStatic[Field[Array, StaggeredFieldMetaData]]
):
    @staticmethod
    def default_meta() -> dict[str, StaggeredFieldMetaData]:
        return default_static_meta()

    @staticmethod
    def array_varset_type() -> type["StaticArrays"]:
        return StaticArrays

    @staticmethod
    def quantity_varset_type() -> type["StaticQuantities"]:
        return StaticQuantities


def default_tendency_meta() -> Dict[str, StaggeredFieldMetaData]:
    return {
        "dhdt": StaggeredFieldMetaData(
            name="dhdt",
            dims=("nPrimal_Cell",),
            long_name="Tendency of thickness",
            description="Tendency of thickness calculated from state variables",
            units="m/s",
            hori_location="cell",
            vert_location="",
            basis="scalar",
            coordinates="lon_pc lat_pc",
            staggered_kind="C",
            mesh_name="primal"
        ),
        "dhdt_mfx": StaggeredFieldMetaData(
            name="dhdt_mfx",
            dims=("nPrimal_Cell",),
            long_name="Tendency of thickness due to mass flux divergence term",
            description="Tendency of thickness due to mass flux divergence term calculated from state variables",
            units="1/s",
            hori_location="cell",
            vert_location="",
            basis="scalar",
            coordinates="lon_pc lat_pc",
            staggered_kind="C",
            mesh_name="primal"
        ),
        "dundt": StaggeredFieldMetaData(
            name="dundt",
            dims=("nDual_Edge",),
            long_name="Tendency of normal velocity",
            description="Tendency of normal velocity calculated from state variables",
            units="m/s^2",
            hori_location="edge",
            vert_location="",
            basis="normal",
            coordinates="lon_de lat_de",
            staggered_kind="C",
            mesh_name="dual"
        ),
        "dundt_coriolis": StaggeredFieldMetaData(
            name="dundt_coriolis",
            dims=("nDual_Edge",),
            long_name="Tendency of normal velocity due to coriolis term",
            description="Tendency of normal velocity due to coriolis term calculated from state variables",
            units="m/s^2",
            hori_location="edge",
            vert_location="",
            basis="scalar",
            coordinates="lon_de lat_de",
            staggered_kind="C",
            mesh_name="dual"
        ),
        "dundt_grad_ke": StaggeredFieldMetaData(
            name="dundt_grad_ke",
            dims=("nDual_Edge",),
            long_name="Tendency of normal velocity due to gradient of kinetic energy term",
            description="Tendency of normal velocity due to gradient of kinetic energy term calculated from state variables",
            units="m/s^2",
            hori_location="edge",
            vert_location="",
            basis="scalar",
            coordinates="lon_de lat_de",
            staggered_kind="C",
            mesh_name="dual"
        ),
        "dundt_pgf": StaggeredFieldMetaData(
            name="dundt_pgf",
            dims=("nDual_Edge",),
            long_name="Tendency of normal velocity due to pressure gradient force term",
            description="Tendency of normal velocity due to pressure gradient force term calculated from state variables",
            units="m/s^2",
            hori_location="edge",
            vert_location="",
            basis="scalar",
            coordinates="lon_de lat_de",
            staggered_kind="C",
            mesh_name="dual"
        ),
    }


class BaseTendency(AbstractVariableSet, Generic[DT]):
    dhdt: DT
    dhdt_mfx: DT
    dundt: DT
    dundt_coriolis: DT
    dundt_grad_ke: DT
    dundt_pgf: DT

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
        array_dict["dhdt"] = jnp.zeros(
            (1, n_polg,), dtype=float_type
        )
        array_dict["dhdt_mfx"] = jnp.zeros(
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

    @abstractmethod
    def zeros_like(self, data: DT) -> DT:
        """Create a tendency variable set with all-zero arrays."""
        raise NotImplementedError

    def _tend2state(self, pass_tag: str = "all") -> dict[str, DT]:
        """Get a mapping from tendency variable names to state variable names.

        Args:
            pass_tag: A string indicating which tendencies to include.
                - "all": include all tendencies
                - "forward_pass": include only tendencies related to forward euler step.
                   other tendencies are set to zero.
                - "backward_pass": include only tendencies related to backward euler step.
                   other tendencies are set to zero.
        Returns:
            A dictionary mapping tendency variable names to state variable names.
        """
        if pass_tag == "all":
            return {
                "h_pc": self.dhdt,
                "un_e": self.dundt
            }
        elif pass_tag == "forward_pass":
            return {
                "h_pc": self.dhdt,
                "un_e": self.zeros_like(self.dundt)
            }
        elif pass_tag == "backward_pass":
            return {
                "h_pc": self.zeros_like(self.dhdt),
                "un_e": self.dundt
            }
        else:
            raise ValueError(f"Invalid pass_tag: {pass_tag}")


class TendencyArrays(
    AbstractArrayBasedVariableSet[
        "TendencyFields", "TendencyQuantities", StaggeredFieldMetaData
    ],
    BaseTendency[Array]
):
    @staticmethod
    def default_meta() -> dict[str, StaggeredFieldMetaData]:
        return default_tendency_meta()

    @staticmethod
    def quantity_varset_type() -> type["TendencyQuantities"]:
        return TendencyQuantities

    @staticmethod
    def field_varset_type() -> type["TendencyFields"]:
        return TendencyFields

    def to_state(self, pass_tag: str) -> StateArrays:
        return StateArrays.from_arrays(
            array_dict=self._tend2state(pass_tag=pass_tag),
            meta=StateArrays.default_meta()
        )

    def zeros_like(self, data: Array) -> Array:
        return jnp.zeros_like(data)


class TendencyQuantities(
    AbstractQuantityBasedVariableSet[
        "TendencyFields", "TendencyArrays", StaggeredFieldMetaData
    ],
    BaseTendency[ux.AbstractQuantity]
):
    @staticmethod
    def default_meta() -> dict[str, StaggeredFieldMetaData]:
        return default_tendency_meta()

    @staticmethod
    def array_varset_type() -> type["TendencyArrays"]:
        return TendencyArrays

    @staticmethod
    def field_varset_type() -> type["TendencyFields"]:
        return TendencyFields

    def to_state(self, pass_tag: str) -> StateQuantities:
        return StateQuantities(
            **self._tend2state(pass_tag=pass_tag),
            meta=StateQuantities.default_meta()
        )

    def zeros_like(self, data: ux.AbstractQuantity) -> ux.AbstractQuantity:
        return data * 0.0


class TendencyFields(
    AbstractFieldBasedVariableSet[
        "TendencyQuantities", "TendencyArrays", StaggeredFieldMetaData
    ],
    BaseTendency[Field[Array, StaggeredFieldMetaData]]
):
    @staticmethod
    def default_meta() -> dict[str, StaggeredFieldMetaData]:
        return default_tendency_meta()

    @staticmethod
    def array_varset_type() -> type["TendencyArrays"]:
        return TendencyArrays

    @staticmethod
    def quantity_varset_type() -> type["TendencyQuantities"]:
        return TendencyQuantities

    def to_state(self, pass_tag: str = "all") -> StateFields:
        return StateFields(
            **self._tend2state(pass_tag=pass_tag)
        )

    def zeros_like(
        self,
        data: Field[Array, StaggeredFieldMetaData]
    ) -> Field[Array, StaggeredFieldMetaData]:
        return Field.create(
            data=jnp.zeros_like(data.data),
            meta=data.meta,
            as_quantity=False
        )


def default_aux_meta() -> Dict[str, StaggeredFieldMetaData]:
    """Get the default metadata for shallow water auxiliary variables."""
    return {
        "ut_e": StaggeredFieldMetaData(
            name="ut_e",
            dims=("nDual_Edge",),
            long_name="Tangential velocity at dual edge midpoint",
            description="Tangential velocity at dual edge midpoint",
            units="m/s",
            hori_location="edge",
            vert_location="",
            basis="tangent",
            staggered_kind="C",
            coordinates="lon_de lat_de",
            mesh_name="dual"
        ),
        "h_e": StaggeredFieldMetaData(
            name="h_e",
            dims=("nDual_Edge",),
            long_name="Thickness at dual edge midpoint",
            description="Thickness at dual edge midpoint",
            units="m",
            hori_location="edge",
            vert_location="",
            basis="scalar",
            staggered_kind="C",
            coordinates="lon_de lat_de",
            mesh_name="primal"
        ),
        "h_dc": StaggeredFieldMetaData(
            name="h_dc",
            dims=("nDual_Cell",),
            long_name="Thickness at dual cell center",
            description="Thickness at dual cell center",
            units="m",
            hori_location="cell",
            vert_location="",
            basis="scalar",
            coordinates="lon_dc lat_dc",
            staggered_kind="C",
            mesh_name="dual"
        ),
        "mfx_e": StaggeredFieldMetaData(
            name="mfx_e",
            dims=("nDual_Edge",),
            long_name="Mass flux at dual edge midpoint",
            description="Mass flux at dual edge midpoint",
            units="m^2 /s",
            hori_location="edge",
            vert_location="",
            basis="scalar",
            staggered_kind="C",
            coordinates="lon_de lat_de",
            mesh_name="primal"
        ),
        "rel_vor_dc": StaggeredFieldMetaData(
            name="rel_vor_dc",
            dims=("nDual_Cell",),
            long_name="Relative vorticity at dual cell center",
            description="Relative vorticity at dual cell center",
            units="1/s",
            hori_location="cell",
            vert_location="",
            basis="scalar",
            coordinates="lon_dc lat_dc",
            staggered_kind="C",
            mesh_name="dual"
        ),
        "abs_vor_dc": StaggeredFieldMetaData(
            name="abs_vor_dc",
            dims=("nDual_Cell",),
            long_name="Absolute vorticity at dual cell center",
            description="Absolute vorticity at dual cell center",
            units="1/s",
            hori_location="cell",
            vert_location="",
            basis="scalar",
            coordinates="lon_dc lat_dc",
            staggered_kind="C",
            mesh_name="dual"
        ),
        "pv_dc": StaggeredFieldMetaData(
            name="pv_dc",
            dims=("nDual_Cell",),
            long_name="Potential vorticity at dual cell center",
            description=(
                "Potential vorticity at dual cell center calculated from "
                "absolute vorticity and thickness at dual cell center"
            ),
            units="1/(m s)",
            hori_location="cell",
            vert_location="",
            basis="scalar",
            coordinates="lon_dc lat_dc",
            staggered_kind="C",
            mesh_name="dual"
        ),
        "pv_e": StaggeredFieldMetaData(
            name="pv_e",
            dims=("nDual_Edge",),
            long_name="Potential vorticity at dual edge midpoint",
            description="Potential vorticity at dual edge midpoint",
            units="1/(m s)",
            hori_location="edge",
            vert_location="",
            basis="scalar",
            coordinates="lon_de lat_de",
            staggered_kind="C",
            mesh_name="dual"
        ),
        "ke_pc": StaggeredFieldMetaData(
            name="ke_pc",
            dims=("nPrimal_Cell",),
            long_name="Kinetic energy at primal cell center",
            description="Kinetic energy at primal cell center",
            units="m^2/s^2",
            hori_location="cell",
            vert_location="",
            basis="scalar",
            coordinates="lon_pc lat_pc",
            staggered_kind="C",
            mesh_name="primal"
        )
    }


class BaseAuxiliary(AbstractVariableSet, Generic[DT]):
    ut_e: DT
    h_e: DT
    mfx_e: DT
    h_dc: DT
    rel_vor_dc: DT
    abs_vor_dc: DT
    pv_dc: DT
    pv_e: DT
    ke_pc: DT

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
        array_dict["ut_e"] = jnp.zeros((1, n_edge,), dtype=float_type)
        array_dict["h_e"] = jnp.zeros((1, n_edge,), dtype=float_type)
        array_dict["mfx_e"] = jnp.zeros((1, n_edge,), dtype=float_type)
        array_dict["h_dc"] = jnp.zeros((1, n_dual,), dtype=float_type)
        array_dict["rel_vor_dc"] = jnp.zeros((1, n_dual,), dtype=float_type)
        array_dict["abs_vor_dc"] = jnp.zeros((1, n_dual,), dtype=float_type)
        array_dict["pv_dc"] = jnp.zeros((1, n_dual,), dtype=float_type)
        array_dict["pv_e"] = jnp.zeros((1, n_edge,), dtype=float_type)
        array_dict["ke_pc"] = jnp.zeros((1, n_polg,), dtype=float_type)
        return cls.from_arrays(array_dict=array_dict, meta=meta)


class AuxiliaryArrays(
    AbstractArrayBasedVariableSet[
        "AuxiliaryFields", "AuxiliaryQuantities", StaggeredFieldMetaData
    ],
    BaseAuxiliary[Array]
):
    @staticmethod
    def default_meta() -> dict[str, StaggeredFieldMetaData]:
        return default_aux_meta()

    @staticmethod
    def quantity_varset_type() -> type["AuxiliaryQuantities"]:
        return AuxiliaryQuantities

    @staticmethod
    def field_varset_type() -> type["AuxiliaryFields"]:
        return AuxiliaryFields


class AuxiliaryQuantities(
    AbstractQuantityBasedVariableSet[
        "AuxiliaryFields", "AuxiliaryArrays", StaggeredFieldMetaData
    ],
    BaseAuxiliary[ux.AbstractQuantity]
):
    @staticmethod
    def default_meta() -> dict[str, StaggeredFieldMetaData]:
        return default_aux_meta()

    @staticmethod
    def array_varset_type() -> type["AuxiliaryArrays"]:
        return AuxiliaryArrays

    @staticmethod
    def field_varset_type() -> type["AuxiliaryFields"]:
        return AuxiliaryFields


class AuxiliaryFields(
    AbstractFieldBasedVariableSet[
        "AuxiliaryQuantities", "AuxiliaryArrays", StaggeredFieldMetaData
    ],
    BaseAuxiliary[Field[Array, StaggeredFieldMetaData]]
):
    @staticmethod
    def default_meta() -> dict[str, StaggeredFieldMetaData]:
        return default_aux_meta()

    @staticmethod
    def array_varset_type() -> type["AuxiliaryArrays"]:
        return AuxiliaryArrays

    @staticmethod
    def quantity_varset_type() -> type["AuxiliaryQuantities"]:
        return AuxiliaryQuantities
