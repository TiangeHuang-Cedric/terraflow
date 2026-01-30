from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Generic, Mapping, TypeVar, Type, overload
from typing import Optional, Union, List, Dict, Self, cast, Literal, Any
import dataclasses
import equinox as eqx
import unxt as ux
import jax.numpy as jnp
import xarray as xr
import uxarray as uxr

import terraflow.operator as op
from terraflow.util import typing, uxarray as tuxr
from terraflow.mesh import AbstractMesh
from terraflow.vert_coord import AbstractVertCoordinate
from terraflow.layout import CoordinateSystem, AbstractLayout
from terraflow.layout import AbstractFieldMetaData, StaggeredFieldMetaData
from terraflow.layout import Field, ArrayField, FieldLike
from terraflow.planet import PlanetParameterSet
from terraflow.solver.abstract import AbstractODE
from terraflow.manager.registry import navigate

Numeric = typing.Numeric
Array = typing.Array


HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
LO = TypeVar("LO", bound=AbstractLayout)
DM = TypeVar("DM", bound=op.AbstractDiscretizationMethod)
ESAUX = TypeVar("ESAUX", bound="AbstractEquationSetAuxiliaryTerms")
ES = TypeVar("ES", bound="AbstractEquationSet")
EQU = TypeVar("EQU", bound="AbstractEquationForm")
FMT = TypeVar("FMT", bound=AbstractFieldMetaData)
STATE = TypeVar("STATE", bound="AbstractFields")
STATIC = TypeVar("STATIC", bound="AbstractFields")
TEND = TypeVar("TEND", bound="AbstractFields")
AUX = TypeVar("AUX", bound="AbstractFields")
AUXDS = TypeVar("AUXDS", bound="AbstractEquationTermAuxiliaryDataset")
BLOCK = TypeVar("BLOCK", bound="AbstractBlock")


class DiscretizationInfo:
    planet: str
    component: str
    equset: str
    form: str
    equ: str
    ode_kind: str
    method: str
    layout: str
    horizontal_mesh: str
    vertical_coordinate: str

    def __init__(
        self,
        planet: str,
        component: str,
        equset: str,
        form: str,
        equ: str = "",
        ode_kind: str = "",
        method: str = "",
        layout: str = "",
        horizontal_mesh: str = "",
        vertical_coordinate: str = ""
    ):
        self.planet = planet
        self.component = component
        self.equset = equset
        self.form = form
        self.equ = equ
        self.ode_kind = ode_kind
        self.method = method
        self.layout = layout
        self.horizontal_mesh = horizontal_mesh
        self.vertical_coordinate = vertical_coordinate

    def copy(self) -> "DiscretizationInfo":
        return DiscretizationInfo(
            planet=self.planet,
            component=self.component,
            equset=self.equset,
            form=self.form,
            equ=self.equ,
            ode_kind=self.ode_kind,
            method=self.method,
            layout=self.layout,
            horizontal_mesh=self.horizontal_mesh,
            vertical_coordinate=self.vertical_coordinate
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "planet": self.planet,
            "component": self.component,
            "equset": self.equset,
            "form": self.form,
            "equ": self.equ,
            "ode_kind": self.ode_kind,
            "method": self.method,
            "layout": self.layout,
            "horizontal_mesh": self.horizontal_mesh,
            "vertical_coordinate": self.vertical_coordinate
        }

    def id(self) -> str:
        """
        Generate a unique identifier string for the discretization configuration.

        Note: If any of the fields are empty strings, they will be exincluded in the ID,

       Return:
            a string identifier with no consecutive "::" in the ID.
        """
        return "::".join(filter(lambda v: v != "", self.to_dict().values()))

    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        operator: op.AbstractDiscretizationMethod,
        equset_type: Type[ES],
        equ_type: Optional[Type[EQU]] = None
    ):
        """
        Create DiscretizationInfo from model components.

        Args:
            coord: Coordinate system used in the model.
            para: Planet parameter set.
            operator: Discretization operator.
            equset_type: Type of the dynamics set.
            equ_type: [Optional] Type of the dynamics form.
                If None, defaults to an empty string, meaning no specific dynamics form.

        Returns:
            An instance of DiscretizationInfo populated with the provided model information.
        """
        return cls(
            planet=para.planet,
            component=para.component,
            equset=equset_type.alias_name(),
            form=equset_type.form_name(),
            equ=equ_type.alias_name() if equ_type is not None else "",
            ode_kind=equset_type.ode_kind_alias_name(),
            method=operator.alias_name(),
            layout=coord.layout.alias_name(),
            horizontal_mesh=coord.mesh.alias_name(),
            vertical_coordinate=coord.vert.alias_name()
        )


class AbstractEquationBase(Generic[DM], ABC):
    _disc: DiscretizationInfo
    _method: DM

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: DM,
    ):
        self._disc = disc
        self._method = method
        self._built = False

    @property
    def disc(self) -> DiscretizationInfo:
        return self._disc

    @property
    def method(self) -> DM:
        return self._method

    def _navi_between_tables(
        self,
        table: str,
        id_col: str,
        id_val: str,
    ) -> Any:
        """Navigate to a specific configuration in a table."""
        return navigate(table, id_col=id_col, id_val=id_val)

    @abstractmethod
    def disc_info(self) -> Dict[str, str]:
        pass

    def disc_id(self) -> str:
        return "::".join(self.disc_info().values())


class AbstractEquationTerm(
    AbstractEquationBase[DM],
    Generic[DM, ESAUX],
    ABC
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: DM,
        aux: ESAUX
    ):
        super().__init__(disc, method)
        self._table = "equterm"
        self.aux = aux

    @abstractmethod
    def __call__(self, block: BLOCK, *args: Any, **kwargs: Any) -> BLOCK:
        raise NotImplementedError

    @abstractmethod
    def unwrap(self, block) -> Field:
        raise NotImplementedError

    def disc_info(self) -> Dict[str, str]:
        return {
            "planet": self._disc.planet,
            "component": self._disc.component,
            "equset": self._disc.equset,
            "form": self._disc.form,
            "equ": self._disc.equ,
            "ode_kind": self._disc.ode_kind,
            "method": self._disc.method,
            "layout": self._disc.layout,
            "horizontal_mesh": self._disc.horizontal_mesh,
            "vertical_coordinate": self._disc.vertical_coordinate,
        }


class AbstractEquationForm(
    AbstractEquationBase[DM],
    Generic[DM, ESAUX],
    ABC
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: DM,
        aux: ESAUX
    ):
        disc_copy = disc.copy()
        disc_copy.equ = self.alias_name()
        super().__init__(disc_copy, method)
        self._table = "equ"
        self.aux = aux

    @abstractmethod
    def __call__(self, block: BLOCK, *args: Any, **kwargs: Any) -> BLOCK:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def ode_kind_alias_name() -> str:
        raise NotImplementedError

    def disc_info(self) -> Dict[str, str]:
        return {
            "planet": self._disc.planet,
            "component": self._disc.component,
            "equset": self._disc.equset,
            "form": self._disc.form,
            "equ": self._disc.equ,
            "ode_kind": self._disc.ode_kind,
            "method": self._disc.method,
            "layout": self._disc.layout,
            "horizontal_mesh": self._disc.horizontal_mesh,
            "vertical_coordinate": self._disc.vertical_coordinate,
        }

    def navi2term(
        self,
        group: str,
        choice: str
    ) -> Callable:
        """Navigate to a specific configuration in a table within the same group."""
        id_col = "unique_id"
        disc = self.disc_id()
        id_val = disc + "::" + group + "::" + choice
        return navigate("equterm", id_col=id_col, id_val=id_val)[0]

    def navi2termgroup(
        self,
        group: str
    ) -> List[Callable]:
        id_col = "group_id"
        disc = self.disc_id()
        id_val = disc + "::" + group
        return navigate("equterm", id_col=id_col, id_val=id_val)


class AbstractContinuityEquForm(AbstractEquationForm[DM, ESAUX], ABC):
    pass


class AbstractMomentumEquForm(AbstractEquationForm[DM, ESAUX], ABC):
    pass


class AbstractThermodynamicEquForm(AbstractEquationForm[DM, ESAUX], ABC):
    pass


class AbstractEquationSetAuxiliaryTerms(AbstractEquationBase[DM]):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: DM,
    ):
        super().__init__(disc, method)
        self._table = "equaux"
        self._built = False

    @abstractmethod
    def __call__(self, block: BLOCK, pass_tag: str, *args, **kwargs) -> BLOCK:
        raise NotImplementedError

    @abstractmethod
    def build_default(self) -> Self:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, block) -> BLOCK:
        raise NotImplementedError

    def disc_info(self) -> Dict[str, str]:
        return {
            "planet": self._disc.planet,
            "component": self._disc.component,
            "equset": self._disc.equset,
            "form": self._disc.form,
            "equ": self._disc.equ,
            "ode_kind": self._disc.ode_kind,
            "method": self._disc.method,
            "layout": self._disc.layout,
            "horizontal_mesh": self._disc.horizontal_mesh,
            "vertical_coordinate": self._disc.vertical_coordinate,
        }

    def has_built(self) -> bool:
        return self._built


class AbstractEquationSet(
    AbstractEquationBase[DM],
    AbstractODE,
    Generic[DM, ESAUX],
    ABC
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: DM,
    ):
        super().__init__(disc, method)
        self._table = "equset"
        factory = self.navi2aux()
        aux: ESAUX = factory(
            disc=disc,
            method=method
        )
        self.aux = aux

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        """Return the alias name of the dynamics set."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def form_name() -> str:
        """Return the form name of the dynamics set. Default to empty string."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def ode_kind_alias_name() -> str:
        """Return the ODE kind alias name of the dynamics set. Default to empty string."""
        raise NotImplementedError

    def build_default(self) -> Self:
        self.aux.build_default()
        return self

    def prepare(self, block: BLOCK) -> BLOCK:
        self.aux.build_default()
        return self.aux.prepare(block)

    def disc_info(self) -> Dict[str, str]:
        return {
            "planet": self._disc.planet,
            "component": self._disc.component,
            "equset": self._disc.equset,
            "form": self._disc.form,
            "ode_kind": self._disc.ode_kind,
            "method": self._disc.method,
            "layout": self._disc.layout,
            "horizontal_mesh": self._disc.horizontal_mesh,
            "vertical_coordinate": self._disc.vertical_coordinate,
        }

    def navi2aux(self) -> Callable:
        """Navigate to the auxiliary terms configuration in the equaux table."""
        id_col = "unique_id"
        disc = self.disc_id()
        choice = "auxiliary_terms"
        id_val = disc + "::" + choice
        return navigate("equaux", id_col=id_col, id_val=id_val)[0]

    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        method: DM,
        *args,
        **kwargs
    ) -> Self:
        """
        Create an instance of the dynamics set from model components.

        Args:
            coord: Coordinate system used in the model.
            para: Planet parameter set.
            method: Discretization method.

        Returns:
            An instance of the dynamics set populated with the provided model information.
        """
        disc = DiscretizationInfo.from_model(
            coord, para, method, cls
        )
        return cls(disc, method, *args, **kwargs)


class AbstractVariableSet(eqx.Module, ABC):

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _wrap(self, *args, **kwargs) -> Self:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_arrays(cls, *args, **kwargs) -> Self:
        raise NotImplementedError

    @abstractmethod
    def keys(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def values(self) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def items(self) -> List[tuple[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def isel(self, index: int, axis: int) -> Self:
        """Index select along a specific dimension for all variables."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, val: float = 0.0) -> Self:
        """Reset all variables to a specific value."""
        raise NotImplementedError

    def to_netcdf(
        self,
        coord: CoordinateSystem[HM, VM, LO],
        path: str,
        format="UGRID",
        use_dual: bool = False,
        **kwargs
    ) -> None:
        """Save variable set to a NetCDF file."""
        if format == "UGRID":
            ds = self.to_xarray_with_ugrid_format(use_dual=use_dual)
            grid_ds = coord.mesh.to_ugrid(
                use_dual=use_dual, **kwargs
            ).to_xarray()
            ds = xr.merge([ds, grid_ds])
            ds.to_netcdf(path)

    def to_ugrid(
        self,
        coord: CoordinateSystem[HM, VM, LO],
        use_dual: bool = False,
        **kwargs
    ) -> tuxr.UGRIDDataset:
        """Convert variable set to a UXarray.UxGrid."""
        ds = self.to_xarray_with_ugrid_format(use_dual=use_dual)
        uxgrid = coord.mesh.to_ugrid(use_dual=use_dual, **kwargs).to_uxgrid()
        return tuxr.UGRIDDataset.from_xarray(ds, uxgrid)

    def to_uxarray(
        self,
        coord: CoordinateSystem[HM, VM, LO],
        use_dual: bool = False,
        **kwargs
    ) -> uxr.UxDataset:
        """Convert variable set to a UXarray.Dataset."""
        ds = self.to_xarray_with_ugrid_format(use_dual=use_dual)
        uxgrid = coord.mesh.to_ugrid(use_dual=use_dual, **kwargs).to_uxgrid()
        return uxr.UxDataset.from_xarray(ds, uxgrid)

    @abstractmethod
    def to_xarray_with_ugrid_format(self, use_dual: bool = False) -> xr.Dataset:
        """Convert variable set to a xarray.Dataset with UGRID format."""
        raise NotImplementedError


class AbstractFields(AbstractVariableSet, Generic[FMT]):

    def __init__(
        self,
        variables: Dict[str, ArrayField[FMT]]
    ):
        for k, v in variables.items():
            if not isinstance(v, Field):
                raise TypeError(f"Variable '{k}' is not a Field.")
            object.__setattr__(self, k, v)

    @classmethod
    def _field_names(cls) -> tuple[str, ...]:
        return tuple(
            f.name
            for f in dataclasses.fields(cls)
        )

    def __iter__(self):
        for name in self._field_names():
            yield getattr(self, name)

    def __len__(self) -> int:
        return len(self._field_names())

    def _wrap(self, **kwargs) -> Self:
        names = self._field_names()
        new_vals = {n: kwargs.get(n, getattr(self, n)) for n in names}
        for k, v in new_vals.items():
            if not isinstance(v, Field):
                raise TypeError(f"Variable '{k}' is not a Field.")
            self = eqx.tree_at(
                lambda vs: object.__getattribute__(vs, k),
                self,
                v
            )
        return self

    def reset(self, val: float = 0.0) -> Self:
        new_vars: Dict[str, ArrayField[FMT]] = {}
        for key in self.keys():
            var = object.__getattribute__(self, key)
            if isinstance(var.data, jnp.ndarray):
                new_vars[key] = Field.create(
                    data=jnp.full_like(var.data, val),
                    meta=var.meta,
                    as_quantity=var.is_quantity()
                )
            elif isinstance(var.data, ux.AbstractQuantity):
                new_array = jnp.full_like(var.data.value, val)
                new_vars[key] = Field.create(
                    data=ux.Quantity(
                        new_array, var.data.unit  # type: ignore
                    ),
                    meta=var.meta,
                    as_quantity=var.is_quantity()
                )
            else:
                raise TypeError(
                    f"Variable '{key}' is of unsupported type '{type(var)}' for reset operation."
                )
        return self._wrap(**new_vars)

    def keys(self) -> List[str]:
        return list(self._field_names())

    def values(self) -> List[ArrayField[FMT]]:
        return [getattr(self, name) for name in self._field_names()]

    def items(self) -> List[tuple[str, ArrayField[FMT]]]:
        return [(name, getattr(self, name)) for name in self._field_names()]

    def isel(self, index: int, axis: int) -> Self:
        """Index select along a specific dimension for all variables."""
        new_vars: Dict[str, ArrayField[FMT]] = {}
        for key, var in self.items():
            new_vars[key] = var.isel(index=index, axis=axis)
        return self._wrap(**new_vars)

    def copy(self) -> Self:
        return self._wrap()

    def is_staggered(self) -> bool:
        """Check if all stored variables are staggered Fields."""
        return all(
            isinstance(v.meta, StaggeredFieldMetaData)
            for v in self.values()
        )

    @staticmethod
    @abstractmethod
    def default_meta() -> Dict[str, FMT]:
        """Return default variable metadata dictionary."""
        return {}

    def default_fields(self) -> Dict[str, ArrayField[FMT]]:
        """Return default variable field dictionary."""
        data = {key: jnp.zeros(()) for key in self.default_meta().keys()}
        return {
            key: ArrayField[FMT](
                data=data[key],
                meta=field_meta
            )
            for key, field_meta in self.default_meta().items()
        }

    def existing_values(self) -> Dict[str, ArrayField[FMT]]:
        """Return existing variables."""
        if getattr(self, "variables", None) is None:
            return self.default_fields()
        else:
            if self.is_fields():
                return dict(self.items())
            else:
                raise TypeError("Variables are not stored as Fields.")

    def existing_meta(self) -> Dict[str, FMT]:
        """Return metadata of existing variables."""
        if getattr(self, "variables", None) is None:
            return self.default_meta()
        else:
            meta: Dict[str, FMT] = {}
            for key, var in self.items():
                if isinstance(var, Field):
                    meta[key] = var.meta
        return meta

    def varnames(self) -> List[str]:
        return list(self.keys())

    def is_fields(self) -> bool:
        """Check if all stored variables are Fields."""
        return all(isinstance(v, Field) for v in self.values())

    def is_arrays(self) -> bool:
        """Check if all stored variables are Arrays."""
        return all(v.is_array() for v in self.values())

    def is_quantities(self) -> bool:
        """Check if all stored variables are AbstractQuantities."""
        return all(v.is_quantity() for v in self.values())

    def unwrap(self) -> Mapping[str, Union[Array, ux.AbstractQuantity]]:
        vars: Dict[str, Union[Array, ux.AbstractQuantity]] = {}
        for key, var in self.items():
            if isinstance(var, Field):
                vars[key] = var.unwrap()
            else:
                vars[key] = var
        if all(isinstance(v, Array) for v in vars.values()):
            return cast(Dict[str, Array], vars)
        elif all(isinstance(v, ux.AbstractQuantity) for v in vars.values()):
            return cast(Dict[str, ux.AbstractQuantity], vars)
        else:
            raise TypeError(
                "Incompatible variable types after unwrapping; "
                "neither all Arrays nor all AbstractQuantities."
            )

    @classmethod
    def wrap(cls, vars_dict: Dict[str, ArrayField[FMT]]) -> Self:
        if not all(isinstance(v, Field) for v in vars_dict.values()):
            raise TypeError(
                "Incompatible variable types for wrapping, must be Fields"
            )
        return cls(variables=vars_dict)

    @classmethod
    def from_arrays(
        cls,
        array_dict: Dict[str, Array],
        meta: Dict[str, FMT]
    ) -> Self:
        new_variables: Dict[str, ArrayField[FMT]] = {}
        for key, var in array_dict.items():
            if key not in meta:
                raise KeyError(f"Metadata for variable '{key}' not found.")
            new_variables[key] = ArrayField[FMT](
                data=var,
                meta=meta[key]
            )
        return cls(variables=new_variables)

    def to_default_unit(self) -> Self:
        """Convert all parameters to default unit system.
        - If the parameter is a Field, convert to the default unit defined in metadata.
        - If the parameter is an AbstractQuantity, convert to SI units.
        """
        default = self.default_meta()
        new_variables: Dict[str, ArrayField[FMT]] = {}
        for key, var in self.items():
            if isinstance(var, Field):
                default_units = default[key].units
                if var.meta.units != default_units:
                    new_variables[key] = var.to(default_units)
                else:
                    new_variables[key] = var
            else:
                raise TypeError(f"Variable '{key}' is neither a Field.")
        if self.is_fields():
            return type(self)(variables=new_variables)
        else:
            raise TypeError(
                "Cannot convert to default unit on incompatible variable types."
                "Please check if all variables are Fields."
            )

    def decompose(self, usys: ux.AbstractUnitSystem) -> Self:
        """Convert all parameters to the specified unit system."""
        new_variables: Dict[str, ArrayField[FMT]] = {}
        for key, var in self.items():
            if isinstance(var, Field):
                new_variables[key] = var.decompose(usys)  # type: ignore
            else:
                raise TypeError(f"Variable '{key}' is not a Field.")
        if self.is_fields():
            return type(self)(variables=new_variables)
        else:
            raise TypeError(
                "Cannot convert to specified unit on incompatible variable types."
                "Please check if all variables are Fields."
            )

    def to_si(self) -> Self:
        """Convert all parameters to SI units."""
        new_variables: Dict[str, ArrayField[FMT]] = {}
        for key, var in self.items():
            if isinstance(var, Field):
                new_variables[key] = var.to_si()
            else:
                raise TypeError(f"Variable '{key}' is not a Field.")
        if self.is_fields():
            return type(self)(variables=new_variables)
        else:
            raise TypeError(
                "Cannot convert to SI units on incompatible variable types."
                "Please check if all variables are Fields."
            )

    def to_xarray_with_ugrid_format(self, use_dual: bool = False) -> xr.Dataset:
        """Get the meta needed for UGRID NetCDF Format"""
        if not self.is_staggered():
            raise TypeError(
                "Variables are not all staggered Fields, cannot convert to UGRID format."
            )
        var_ds: Dict[str, xr.DataArray] = {}
        for key, var in self.items():
            var_ds[key] = tuxr.format_terraflow2ugrid(
                var.data, var.meta.to_dict(), use_dual=use_dual
            )
        return xr.Dataset(data_vars=var_ds)


class DiagnosticVariableSet(AbstractVariableSet, Generic[FMT], ABC):
    variables: Dict[str, ArrayField[FMT]]

    def __init__(self, variables: Dict[str, ArrayField[FMT]]):
        object.__setattr__(self, "variables", variables)

    def __iter__(self):
        return iter(self.variables.items())

    def __len__(self) -> int:
        return len(self.variables)

    def __getattribute__(self, name: str) -> Any:
        if name == "variables":
            return object.__getattribute__(self, name)
        variables = object.__getattribute__(self, "variables")
        if name in variables:
            return variables[name]
        return object.__getattribute__(self, name)

    def _wrap(self, variables: Dict[str, ArrayField[FMT]]) -> Self:
        return type(self)(variables=variables)

    def keys(self) -> List[str]:
        return list(self.variables.keys())

    def values(self) -> List[ArrayField[FMT]]:
        return list(self.variables.values())

    def items(self) -> List[tuple[str, ArrayField[FMT]]]:
        return list(self.variables.items())

    def isel(self, index: int, axis: int) -> Self:
        """Index select along a specific dimension for all variables."""
        new_vars: Dict[str, ArrayField[FMT]] = {}
        for key, var in self.items():
            if isinstance(var.data, jnp.ndarray):
                data = var.data
            else:
                raise TypeError(
                    f"Variable '{key}' is of unsupported type '{type(var)}' for isel operation."
                )
            new_var = jnp.take(data, indices=index, axis=axis)
            new_vars[key] = Field(data=new_var, meta=var.meta)
        return self._wrap(new_vars)

    def reset(self, val: float = 0.0) -> Self:
        new_vars: Dict[str, ArrayField[FMT]] = {}
        for key, var in self.items():
            if isinstance(var.data, jnp.ndarray):
                new_vars[key] = Field(
                    data=jnp.full_like(var.data, val),
                    meta=var.meta
                )
            else:
                raise TypeError(
                    f"Variable '{key}' is of unsupported type '{type(var)}' for reset operation."
                )
        return self._wrap(new_vars)

    def copy(self) -> Self:
        return type(self)(variables=self.variables.copy())

    def existing_values(
        self
    ) -> Dict[str, ArrayField[FMT]]:
        """Return existing variables."""
        if getattr(self, "variables", None) is None:
            raise ValueError(
                "No variables stored, cannot retrieve existing values."
            )
        else:
            return self.variables

    def existing_meta(self) -> Dict[str, FMT]:
        """Return metadata of existing variables."""
        if getattr(self, "variables", None) is None:
            raise ValueError("No variables stored, cannot retrieve metadata.")
        else:
            return self.meta

    def is_arrays(self) -> bool:
        """Check if all stored variables are Arrays."""
        return all(isinstance(v, Array) for v in self.values())

    def is_staggered(self) -> bool:
        """Check if all stored variables are staggered Fields."""
        return all(
            isinstance(m, StaggeredFieldMetaData)
            for m in self.meta
        )

    def to_xarray_with_ugrid_format(self, use_dual: bool = False) -> xr.Dataset:
        """Get the meta needed for UGRID NetCDF Format"""
        if not self.is_staggered():
            raise TypeError(
                "Variables are not all staggered Fields, cannot convert to UGRID format."
            )
        var_ds: Dict[str, xr.DataArray] = {}
        for key, var in self.items():
            var_ds[key] = tuxr.format_terraflow2ugrid(
                var.data, self.meta[key].to_dict(), use_dual=use_dual
            )
        return xr.Dataset(data_vars=var_ds)

    @classmethod
    def from_arrays(
        cls,
        array_dict: Dict[str, Array],
        meta: Dict[str, FMT]
    ) -> Self:
        variables = {
            k: Field(data=v, meta=meta[k])
            for k, v in array_dict.items()
        }
        return cls(variables=variables)


class AbstractEquationTermAuxiliaryData(eqx.Module, ABC):
    _init: bool

    def has_initialized(self) -> bool:
        return self._init

    @classmethod
    @abstractmethod
    def empty(cls) -> Self:
        raise NotImplementedError


class AbstractEquationTermAuxiliaryDataset(eqx.Module, ABC):
    @classmethod
    @abstractmethod
    def empty(cls) -> Self:
        raise NotImplementedError


class AbstractBlock(
    ABC,
    eqx.Module,
    Generic[HM, VM, LO, STATE, STATIC, TEND, AUX, AUXDS]
):
    coord: CoordinateSystem[HM, VM, LO]
    para: PlanetParameterSet
    state: STATE
    tend: TEND
    static: STATIC
    aux: AUX
    equaux: AUXDS

    def reset(self, val: float = 0.0) -> Self:
        return eqx.tree_at(
            lambda b: (b.tend, b.aux),
            self,
            replace=(self.tend.reset(val), self.aux.reset(val))
        )

    def diagnose(self) -> Self:
        return self


class AbstractDiagnosticTools(Generic[HM, VM, LO, DM, ES, FMT], ABC):

    def __init__(self, equation: ES, jit: bool = False, trajectory: bool = False):
        self._equation = equation
        self.jit = jit
        self.trajectory = trajectory

    @abstractmethod
    def compute(
        self,
        *args,
        **kwargs
    ) -> Mapping[str, FieldLike | tuxr.UGRIDDataArray]:
        """Compute diagnostic variables based on the input state."""
        raise NotImplementedError

    @abstractmethod
    def compute_along_trajectory(
        self,
        *args,
        **kwargs
    ) -> Mapping[str, FieldLike | tuxr.UGRIDDataArray]:
        """Compute diagnostic variables along trajectories based on the input state."""
        raise NotImplementedError

    @overload
    def process_field(
        self,
        coord: CoordinateSystem[HM, VM, LO],
        field: Field,
        to_ugrid: Literal[False] = False,
        **kwargs
    ) -> Field:
        ...

    @overload
    def process_field(
        self,
        coord: CoordinateSystem[HM, VM, LO],
        field: Field,
        to_ugrid: Literal[True],
        **kwargs
    ) -> tuxr.UGRIDDataArray:
        ...

    def process_field(
        self,
        coord: CoordinateSystem[HM, VM, LO],
        field: Field,
        to_ugrid: bool = False,
        **kwargs
    ) -> Field | tuxr.UGRIDDataArray:
        """
        Process Field based on trajectory setting.

        Args:
            field: Original Field. The order of dimensions should be (time, vertical, horizontal...).

        Returns:
            Processed Field. if trajectory is True, prepend 'time' to dims. Otherwise, return Field as is.
        """
        if self.trajectory:
            dims = ("time",) + getattr(field.meta, "dims", ())
            new_meta = deepcopy(field.meta)
            setattr(new_meta, "dims", dims)
            res = Field.create(
                data=field.data,
                meta=new_meta,
                as_quantity=field.is_quantity()
            )
            if to_ugrid:
                ds = res.to_xarray_with_ugrid_format(**kwargs)
                uxgrid = coord.mesh.to_ugrid(**kwargs).to_uxgrid()
                return tuxr.UGRIDDataArray.from_xarray(ds, uxgrid)
            else:
                return res
        else:
            if to_ugrid:
                ds = field.to_xarray_with_ugrid_format(**kwargs)
                uxgrid = coord.mesh.to_ugrid(**kwargs).to_uxgrid()
                return tuxr.UGRIDDataArray.from_xarray(ds, uxgrid)
            else:
                return field

    def process_meta(
        self,
        meta: FMT,
    ) -> FMT:
        """
        Process dimension tuple based on trajectory setting.

        Args:
            meta: Original metadata. The order of dimensions should be (time, vertical, horizontal...).

        Returns:
            Processed dimension tuple. if trajectory is True, prepend 'time' to dims. Otherwise, return dims as is.
        """
        if self.trajectory:
            dims = ("time",) + getattr(meta, "dims", ())
            setattr(meta, "dims", dims)
            return meta
        else:
            return meta

    @property
    def equation(self) -> ES:
        if not hasattr(self, "_equation"):
            raise AttributeError(
                "Equation has not been set. "
                "Please using 'set_equation' method to specify the dynamics first."
            )
        return self._equation
