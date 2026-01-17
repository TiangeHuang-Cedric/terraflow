from typing import Literal, TypeVar, Dict, Union, Optional
from plum import overload
import unxt as ux

from terraflow.util import typing
from terraflow.planet import PlanetParameterSet
from terraflow.mesh import AbstractMesh
from terraflow.vert_coord import AbstractVertCoordinate
from terraflow.layout.abstract import AbstractLayout, CoordinateSystem
from terraflow.layout.field import Field, StaggeredFieldMetaData
from terraflow.layout.abstract import Basis, MeshEntity
from terraflow.discretization.abstract import AbstractFieldBasedVariableSet
from terraflow.discretization.abstract import AbstractQuantityBasedVariableSet
from terraflow.discretization.abstract import AbstractArrayBasedVariableSet
from terraflow.test_case.swe import MountainZonalFlowTestCase, ShallowWaterEquationTestCase


HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
LO = TypeVar("LO", bound=AbstractLayout)

Array = typing.Array
ArrayWithUnits = typing.ArrayWithUnits


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
    coord: CoordinateSystem[HM, VM, LO],
    test: ShallowWaterEquationTestCase,
) -> Array:
    meta = default_state_meta()
    lon = meta["h_pc"].get_position(coord).horizontal_coord["lon"]
    lat = meta["h_pc"].get_position(coord).horizontal_coord["lat"]
    return test.init_thickness(
        lon, lat
    ).reshape((1,) + lon.shape)  # add leading axis for vmap


def init_un_in_case(
    coord: CoordinateSystem[HM, VM, LO],
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


class StateArrays(
    AbstractArrayBasedVariableSet[
        "StateFields", "StateQuantities", StaggeredFieldMetaData
    ]
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

    @property
    def h_pc(self) -> Array:
        return self.variables["h_pc"]

    @property
    def un_e(self) -> Array:
        return self.variables["un_e"]

    @classmethod
    def from_testcase(
        cls,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        *,
        test: Optional[ShallowWaterEquationTestCase] = None,
    ) -> "StateArrays":
        if test is None:
            test = MountainZonalFlowTestCase(para)
        return cls(
            variables={
                "h_pc": init_h_in_case(coord, test),
                "un_e": init_un_in_case(coord, test)
            },
            meta=cls.default_meta()
        )


class StateQuantities(
    AbstractQuantityBasedVariableSet[
        "StateFields", "StateArrays", StaggeredFieldMetaData
    ]
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

    @property
    def h_pc(self) -> ux.AbstractQuantity:
        return self.variables["h_pc"]

    @property
    def un_e(self) -> ux.AbstractQuantity:
        return self.variables["un_e"]

    @classmethod
    def from_testcase(
        cls,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        *,
        test: Optional[ShallowWaterEquationTestCase] = None,
    ) -> "StateQuantities":
        if test is None:
            test = MountainZonalFlowTestCase(para)
        meta = cls.default_meta()
        return cls(
            variables={
                "h_pc": ux.Quantity(
                    init_h_in_case(coord, test),  # type: ignore[arg-type]
                    meta["h_pc"].units
                ),
                "un_e": ux.Quantity(
                    init_un_in_case(coord, test),  # type: ignore[arg-type]
                    meta["un_e"].units
                ),
            },
            meta=meta
        )


class StateFields(
    AbstractFieldBasedVariableSet[
        "StateQuantities", "StateArrays", StaggeredFieldMetaData
    ]
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

    @property
    def h_pc(self) -> Field:
        return self.variables["h_pc"]

    @property
    def un_e(self) -> Field:
        return self.variables["un_e"]

    @classmethod
    def from_testcase(
        cls,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        *,
        case: Optional[ShallowWaterEquationTestCase] = None,
    ) -> "StateFields":
        if case is None:
            case = MountainZonalFlowTestCase(para)
        return cls(
            {
                "h_pc": Field(
                    data=init_h_in_case(coord, case),
                    meta=cls.default_meta()["h_pc"]
                ),
                "un_e": Field(
                    data=init_un_in_case(coord, case),
                    meta=cls.default_meta()["un_e"]
                )
            }
        )


State = Union[StateFields, StateQuantities, StateArrays]


@overload
def construct_state(
    coord: CoordinateSystem[HM, VM, LO],
    para: PlanetParameterSet,
    *,
    test: Optional[ShallowWaterEquationTestCase] = None,
    as_field: Literal[True],
    as_quantity: Literal[False],
    as_array: Literal[False]
) -> StateFields:
    ...


@overload
def construct_state(
    coord: CoordinateSystem[HM, VM, LO],
    para: PlanetParameterSet,
    *,
    test: Optional[ShallowWaterEquationTestCase] = None,
    as_field: Literal[False],
    as_quantity: Literal[True],
    as_array: Literal[False]
) -> StateQuantities:
    ...


@overload
def construct_state(
    coord: CoordinateSystem[HM, VM, LO],
    para: PlanetParameterSet,
    *,
    test: Optional[ShallowWaterEquationTestCase] = None,
    as_field: Literal[False],
    as_quantity: Literal[False],
    as_array: Literal[True]
) -> StateArrays:
    ...


def construct_state(
    coord: CoordinateSystem[HM, VM, LO],
    para: PlanetParameterSet,
    *,
    test: Optional[ShallowWaterEquationTestCase] = None,
    as_field: bool = False,
    as_quantity: bool = False,
    as_array: bool = False
) -> Union[StateFields, StateQuantities, StateArrays]:
    if sum([as_field, as_quantity, as_array]) != 1:
        as_field = True

    if as_field:
        return StateFields.from_testcase(coord, para, case=test)
    elif as_quantity:
        return StateQuantities.from_testcase(coord, para, test=test)
    else:  # as_array
        return StateArrays.from_testcase(coord, para, test=test)


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
    coord: CoordinateSystem[HM, VM, LO],
    test: ShallowWaterEquationTestCase,
) -> Array:
    meta = default_static_meta()
    coord_info = meta["topo"].get_position(coord)
    lon = coord_info.horizontal_coord["lon"]
    lat = coord_info.horizontal_coord["lat"]
    return test.init_topography(lon, lat)


class StaticArrays(
    AbstractArrayBasedVariableSet[
        "StaticFields", "StaticQuantities", StaggeredFieldMetaData
    ]
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

    @property
    def topo(self) -> Array:
        return self.variables["topo"]

    @classmethod
    def from_testcase(
        cls,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        *,
        test: Optional[ShallowWaterEquationTestCase] = None,
    ) -> "StaticArrays":
        if test is None:
            test = MountainZonalFlowTestCase(para)
        return cls(
            variables={
                "topo": init_topo_in_case(coord, test)
            },
            meta=cls.default_meta()
        )


class StaticQuantities(
    AbstractQuantityBasedVariableSet[
        "StaticFields", "StaticArrays", StaggeredFieldMetaData
    ]
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

    @property
    def topo(self) -> ux.AbstractQuantity:
        return self.variables["topo"]

    @classmethod
    def from_testcase(
        cls,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        *,
        test: Optional[ShallowWaterEquationTestCase] = None,
    ) -> "StaticQuantities":
        if test is None:
            test = MountainZonalFlowTestCase(para)
        meta = cls.default_meta()
        return cls(
            variables={
                "topo": ux.Quantity(
                    init_topo_in_case(coord, test),  # type: ignore[arg-type]
                    meta["topo"].units
                ),
            },
            meta=meta
        )


class StaticFields(
    AbstractFieldBasedVariableSet[
        "StaticQuantities", "StaticArrays", StaggeredFieldMetaData
    ]
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

    @property
    def topo(self) -> Field:
        return self.variables["topo"]

    @classmethod
    def from_testcase(
        cls,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        *,
        case: Optional[ShallowWaterEquationTestCase] = None,
    ) -> "StaticFields":
        if case is None:
            case = MountainZonalFlowTestCase(para)
        return cls(
            {
                "topo": Field(
                    data=init_topo_in_case(coord, case),
                    meta=cls.default_meta()["topo"]
                ),
            }
        )


Static = Union[StaticFields, StaticQuantities, StaticArrays]


@overload
def construct_static(
    coord: CoordinateSystem[HM, VM, LO],
    para: PlanetParameterSet,
    *,
    test: Optional[ShallowWaterEquationTestCase] = None,
    as_field: Literal[True],
    as_quantity: Literal[False],
    as_array: Literal[False]
) -> StaticFields:
    ...


@overload
def construct_static(
    coord: CoordinateSystem[HM, VM, LO],
    para: PlanetParameterSet,
    *,
    test: Optional[ShallowWaterEquationTestCase] = None,
    as_field: Literal[False],
    as_quantity: Literal[True],
    as_array: Literal[False]
) -> StaticQuantities:
    ...


@overload
def construct_static(
    coord: CoordinateSystem[HM, VM, LO],
    para: PlanetParameterSet,
    *,
    test: Optional[ShallowWaterEquationTestCase] = None,
    as_field: Literal[False],
    as_quantity: Literal[False],
    as_array: Literal[True]
) -> StaticArrays:
    ...


def construct_static(
    coord: CoordinateSystem[HM, VM, LO],
    para: PlanetParameterSet,
    *,
    test: Optional[ShallowWaterEquationTestCase] = None,
    as_field: bool = False,
    as_quantity: bool = False,
    as_array: bool = False
) -> Union[StaticFields, StaticQuantities, StaticArrays]:
    if sum([as_field, as_quantity, as_array]) != 1:
        as_field = True

    if as_field:
        return StaticFields.from_testcase(coord, para, case=test)
    elif as_quantity:
        return StaticQuantities.from_testcase(coord, para, test=test)
    else:  # as_array
        return StaticArrays.from_testcase(coord, para, test=test)
