from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Generic, Mapping, TypeVar, Type, overload, Tuple
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
from terraflow.layout import Field, QuantityField, ArrayField, FieldLike
from terraflow.planet import PlanetParameterSet
from terraflow.solver.abstract import AbstractODE
from terraflow.manager.registry import navigate

Numeric = typing.Numeric
Array = typing.Array


HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
LO = TypeVar("LO", bound=AbstractLayout)
DM = TypeVar("DM", bound=op.AbstractDiscretizationMethod)
ESAUX = TypeVar("ESAUX", bound="AbstractEquationSetAuxiliary")
ES = TypeVar("ES", bound="AbstractEquationSet")
EQU = TypeVar("EQU", bound="AbstractEquationForm")
FMT = TypeVar("FMT", bound=AbstractFieldMetaData)
DT = TypeVar("DT", Array, ux.AbstractQuantity)
STATE = TypeVar("STATE", bound="AbstractVariableSet")
STATIC = TypeVar("STATIC", bound="AbstractVariableSet")
TEND = TypeVar("TEND", bound="AbstractVariableSet")
AUX = TypeVar("AUX", bound="AbstractVariableSet")
BLOCK = TypeVar("BLOCK", bound="AbstractBlock")
BF = TypeVar("BF", bound="AbstractBlock")
BQ = TypeVar("BQ", bound="AbstractBlock")
BA = TypeVar("BA", bound="AbstractBlock")


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


class AbstractEquationBase(Generic[HM, VM, LO, DM, BLOCK], ABC):
    _disc: DiscretizationInfo
    _coord: CoordinateSystem[HM, VM, LO]
    _para: PlanetParameterSet
    _method: DM

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        method: DM,
    ):
        self._disc = disc
        self._coord = coord
        self._para = para
        self._method = method
        self._built = False

    @property
    def coord(self) -> CoordinateSystem[HM, VM, LO]:
        return self._coord

    @property
    def mesh(self) -> HM:
        return self.coord.mesh

    @property
    def vert(self) -> VM:
        return self.coord.vert

    @property
    def layout(self) -> LO:
        return self.coord.layout

    @property
    def para(self) -> PlanetParameterSet:
        return self._para

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
    def build_default(self) -> Self:
        raise NotImplementedError

    def has_built(self) -> bool:
        return self._built

    @abstractmethod
    def disc_info(self) -> Dict[str, str]:
        pass

    def disc_id(self) -> str:
        return "::".join(self.disc_info().values())


class AbstractEquationTerm(
    AbstractEquationBase[HM, VM, LO, DM, BLOCK],
    Generic[HM, VM, LO, DM, ESAUX, BLOCK],
    ABC
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        method: DM,
        aux: ESAUX
    ):
        super().__init__(disc, coord, para, method)
        self._table = "equterm"
        self._aux = aux

    @abstractmethod
    def __call__(self, block: BLOCK, *args: Any, **kwargs: Any) -> BLOCK:
        raise NotImplementedError

    @abstractmethod
    def unwrap(self, block: BLOCK) -> Array:
        raise NotImplementedError

    @property
    def aux(self) -> ESAUX:
        return self._aux

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
    AbstractEquationBase[HM, VM, LO, DM, BLOCK],
    Generic[HM, VM, LO, DM, ESAUX, BLOCK],
    ABC
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        method: DM,
        aux: ESAUX
    ):
        disc_copy = disc.copy()
        disc_copy.equ = self.alias_name()
        super().__init__(disc_copy, coord, para, method)
        self._table = "equ"
        self._aux = aux

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

    @property
    def aux(self) -> ESAUX:
        return self._aux

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


class AbstractContinuityEquForm(AbstractEquationForm[HM, VM, LO, DM, ESAUX, BLOCK], ABC):
    pass


class AbstractMomentumEquForm(AbstractEquationForm[HM, VM, LO, DM, ESAUX, BLOCK], ABC):
    pass


class AbstractEquationSetAuxiliary(
    AbstractEquationBase[HM, VM, LO, DM, BLOCK],
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        method: DM,
    ):
        super().__init__(disc, coord, para, method)
        self._table = "equaux"

    def __call__(self, block: BLOCK) -> BLOCK:
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


class AbstractEquationSet(
    AbstractEquationBase[HM, VM, LO, DM, BLOCK],
    AbstractODE,
    Generic[HM, VM, LO, DM, ESAUX, BLOCK],
    ABC
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        method: DM
    ):
        super().__init__(disc, coord, para, method)
        self._table = "equset"
        factory = self.navi2aux()
        self._aux: ESAUX = factory(
            disc=disc,
            coord=coord,
            para=para,
            method=method
        )

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

    @property
    def aux(self) -> ESAUX:
        return self._aux

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
    def _from_model(
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
            operator: Discretization operator.

        Returns:
            An instance of the dynamics set populated with the provided model information.
        """
        disc = DiscretizationInfo.from_model(
            coord, para, method, cls
        )
        return cls(disc, coord, para, method, *args, **kwargs)


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


class AbstractRawVariableSet(AbstractVariableSet, Generic[FMT, DT]):
    meta: Dict[str, FMT] = eqx.field(static=True)

    def __init__(self, variables: Dict[str, DT], meta: Dict[str, FMT]):
        for k, v in variables.items():
            if k not in meta:
                raise KeyError(f"Metadata for variable '{k}' not found.")
            object.__setattr__(self, k, v)
        object.__setattr__(self, "meta", meta)

    def __iter__(self):
        return iter(self.values())

    def __len__(self) -> int:
        return len(self.meta)

    def _wrap(self, variables: Dict[str, DT]) -> Self:
        for k, v in variables.items():
            if k not in self.meta:
                raise KeyError(f"Metadata for variable '{k}' not found.")
            self = eqx.tree_at(
                lambda vs: object.__getattribute__(vs, k),
                self,
                v
            )
        return self

    def keys(self) -> List[str]:
        return list(self.meta.keys())

    def values(self) -> List[DT]:
        return [object.__getattribute__(self, key) for key in self.keys()]

    def items(self) -> List[Tuple[str, DT]]:
        return [(key, object.__getattribute__(self, key)) for key in self.keys()]

    def isel(self, index: int, axis: int) -> Self:
        """Index select along a specific dimension for all variables."""
        new_vars: Dict[str, Any] = {}
        for key in self.keys():
            var = object.__getattribute__(self, key)
            if isinstance(var, jnp.ndarray):
                new_vars[key] = jnp.take(var, indices=index, axis=axis)
            elif isinstance(var, ux.AbstractQuantity):
                new_array = jnp.take(var.value, indices=index, axis=axis)
                new_vars[key] = ux.Quantity(
                    new_array, var.unit)  # type: ignore
            else:
                raise TypeError(
                    f"Variable '{key}' is of unsupported type '{type(var)}' for isel operation."
                )
        return self._wrap(new_vars)

    @staticmethod
    def _convert_meta_unit(field_meta: FMT, usys: ux.AbstractUnitSystem) -> FMT:
        """Convert the unit string to the specified unit system."""
        quantity = ux.Quantity(1.0, field_meta.unit)  # type: ignore
        converted = quantity.decompose(usys)  # type: ignore
        return eqx.tree_at(
            lambda fm: fm.unit,
            field_meta,
            converted.unit.to_string()
        )

    def _convert_meta_units(self, usys: ux.AbstractUnitSystem) -> Dict[str, FMT]:
        new_meta: Dict[str, FMT] = {}
        for key, field_meta in self.meta.items():
            new_meta[key] = self._convert_meta_unit(field_meta, usys)
        return new_meta

    def _wrap_with_meta(
        self,
        variables: Dict[str, Any],
        meta: Dict[str, FMT]
    ) -> Self:
        return eqx.tree_at(
            lambda vs: (vs.variables, vs._meta),
            self,
            (variables, meta)
        )

    def copy(self) -> Self:
        return self._wrap_with_meta(
            variables=dict(self.items()),
            meta=self.meta.copy()
        )

    def is_staggered(self) -> bool:
        """Check if all stored variables are staggered Fields."""
        return all(
            isinstance(v, StaggeredFieldMetaData)
            for v in self.meta.values()
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
                var, self.meta[key].to_dict(), use_dual=use_dual
            )
        return xr.Dataset(data_vars=var_ds)


FV = TypeVar("FV", bound="AbstractFieldBasedVariableSet")
AV = TypeVar("AV", bound="AbstractArrayBasedVariableSet")
QV = TypeVar("QV", bound="AbstractQuantityBasedVariableSet")


class AbstractFieldBasedVariableSet(AbstractVariableSet, Generic[QV, AV, FMT]):

    def __init__(self, variables: Dict[str, FieldLike]):
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
        new_vals = tuple(kwargs.get(n, getattr(self, n)) for n in names)
        for k, v in new_vals:
            if not isinstance(v, Field):
                raise TypeError(f"Variable '{k}' is not a Field.")
            self = eqx.tree_at(
                lambda vs: object.__getattribute__(vs, k),
                self,
                v
            )
        return self

    def reset(self, val: float = 0.0) -> Self:
        new_vars: Dict[str, Field] = {}
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

    def values(self) -> List[FieldLike]:
        return [getattr(self, name) for name in self._field_names()]

    def items(self) -> List[tuple[str, FieldLike]]:
        return [(name, getattr(self, name)) for name in self._field_names()]

    def isel(self, index: int, axis: int) -> Self:
        """Index select along a specific dimension for all variables."""
        new_vars: Dict[str, FieldLike] = {}
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

    @staticmethod
    @abstractmethod
    def array_varset_type() -> Type[AV]:
        """Return the array-based variable set type produced by ``to_value``."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def quantity_varset_type() -> Type[QV]:
        """Return the quantity-based variable set type produced by ``to_value``."""
        raise NotImplementedError

    @overload
    def default_fields(
        self,
        as_quantity: Literal[True]
    ) -> Dict[str, QuantityField[FMT]]:
        ...

    @overload
    def default_fields(
        self,
        as_quantity: Literal[False]
    ) -> Dict[str, ArrayField[FMT]]:
        ...

    def default_fields(
        self,
        as_quantity: bool = False
    ) -> Mapping[str, FieldLike]:
        """Return default variable field dictionary."""
        if as_quantity:
            data = self.default_quantities()
            return {
                key: QuantityField[FMT](
                    data=data[key],
                    meta=field_meta
                )
                for key, field_meta in self.default_meta().items()
            }
        else:
            data = self.default_arrays()
            return {
                key: ArrayField[FMT](
                    data=data[key],
                    meta=field_meta
                )
                for key, field_meta in self.default_meta().items()
            }

    def default_quantities(self) -> Dict[str, ux.AbstractQuantity]:
        """Return default variable quantity dictionary."""
        return {
            key: ux.Quantity(
                jnp.zeros(()), field_meta.unit  # type: ignore
            )
            for key, field_meta in self.default_meta().items()
        }

    def default_arrays(self) -> Dict[str, Array]:
        """Return default variable array dictionary."""
        return {key: jnp.zeros(()) for key in self.default_meta().keys()}

    @overload
    def existing_values(
        self,
        as_quantity: Literal[True]
    ) -> Dict[str, QuantityField[FMT]]:
        ...

    @overload
    def existing_values(
        self,
        as_quantity: Literal[False]
    ) -> Dict[str, ArrayField[FMT]]:
        ...

    def existing_values(
        self,
        as_quantity: bool = False
    ) -> Mapping[str, FieldLike]:
        """Return existing variables."""
        if getattr(self, "variables", None) is None:
            return self.default_fields(as_quantity=as_quantity)
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
    def wrap(cls, vars_dict: Dict[str, FieldLike]) -> Self:
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
        new_variables: Dict[str, FieldLike] = {}
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
        new_variables: Dict[str, FieldLike] = {}
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
        new_variables: Dict[str, FieldLike] = {}
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
        new_variables: Dict[str, FieldLike] = {}
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

    def to_quantities(self, usys: ux.AbstractUnitSystem = ux.unitsystems.si) -> QV:
        """Convert all parameters to AbstractQuantities."""
        new_variables: Dict[str, ux.AbstractQuantity] = {}
        for key, var in self.items():
            if isinstance(var, Field):
                new_variables[key] = cast(
                    ux.AbstractQuantity,
                    var.to_quantity(usys).unwrap()
                )
            else:
                raise TypeError(f"Variable '{key}' is not a Field.")
        quantity_cls = self.quantity_varset_type()
        return quantity_cls(variables=new_variables, meta=self.existing_meta())

    def to_arrays(self, usys: ux.AbstractUnitSystem = ux.unitsystems.si) -> AV:
        """Convert all parameters to their numeric values in the specified unit system."""
        new_variables: Dict[str, Array] = {}
        for key, var in self.items():
            if isinstance(var, Field):
                new_variables[key] = cast(
                    Array,
                    var.to_array(usys).unwrap()
                )
            else:
                raise TypeError(f"Variable '{key}' is not a Field.")
        array_cls = self.array_varset_type()
        return array_cls(variables=new_variables, meta=self.existing_meta())

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


class AbstractQuantityBasedVariableSet(
    AbstractRawVariableSet[FMT, ux.AbstractQuantity],
    Generic[FV, AV, FMT]
):

    def __init__(
        self,
        variables: Dict[str, ux.AbstractQuantity],
        meta: Dict[str, FMT],
    ):
        super().__init__(variables=variables, meta=meta)

    @staticmethod
    @abstractmethod
    def default_meta() -> Dict[str, FMT]:
        """Return default variable metadata dictionary."""
        return {}

    @staticmethod
    @abstractmethod
    def field_varset_type() -> Type[FV]:
        """Return the field-based variable set type produced by ``to_value``."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def array_varset_type() -> Type[AV]:
        """Return the array-based variable set type produced by ``to_value``."""
        raise NotImplementedError

    def reset(self, val: float = 0.0) -> Self:
        new_vars: Dict[str, ux.AbstractQuantity] = {}
        for key in self.keys():
            var = object.__getattribute__(self, key)
            if isinstance(var, ux.AbstractQuantity):
                new_array = jnp.full_like(var.value, val)
                new_vars[key] = ux.Quantity(
                    new_array, var.unit  # type: ignore
                )
            else:
                raise TypeError(
                    f"Variable '{key}' is of unsupported type '{type(var)}' for reset operation."
                )
        return self._wrap(new_vars)

    def default_quantities(self) -> Dict[str, ux.AbstractQuantity]:
        """Return default variable quantity dictionary."""
        return {
            key: ux.Quantity(
                jnp.zeros(()), field_meta.unit  # type: ignore
            )
            for key, field_meta in self.default_meta().items()
        }

    def existing_values(self) -> Dict[str, ux.AbstractQuantity]:
        """Return existing variables."""
        if getattr(self, "variables", None) is None:
            return self.default_quantities()
        else:
            if self.is_quantities():
                return {key: value for key, value in self.items()}
            else:
                raise TypeError("Variables are not stored as Quantities.")

    def existing_meta(self) -> Dict[str, FMT]:
        """Return metadata of existing variables."""
        if getattr(self, "variables", None) is None:
            return self.default_meta()
        else:
            return self.meta

    def varnames(self) -> List[str]:
        return list(self.keys())

    def is_quantities(self) -> bool:
        """Check if all stored variables are AbstractQuantities."""
        return all(isinstance(v, ux.AbstractQuantity) for v in self.values())

    def unwrap(self) -> Dict[str, Array]:
        vars: Dict[str, Array] = {}
        for key, var in self.items():
            vars[key] = var.value
        if all(isinstance(v, Array) for v in vars.values()):
            return cast(Dict[str, Array], vars)
        else:
            raise TypeError(
                "Incompatible variable types after unwrapping, not all Arrays."
            )

    @classmethod
    def wrap(cls, vars_dict: Dict[str, ux.AbstractQuantity]) -> Self:
        if not all(isinstance(v, ux.AbstractQuantity) for v in vars_dict.values()):
            raise TypeError(
                "Incompatible variable types for wrapping, must be AbstractQuantities"
            )
        return cls(variables=vars_dict, meta=cls.default_meta())

    @classmethod
    def from_arrays(
        cls,
        array_dict: Dict[str, Array],
        meta: Dict[str, FMT]
    ) -> Self:
        new_variables: Dict[str, ux.AbstractQuantity] = {}
        for key, var in array_dict.items():
            if key not in meta:
                raise KeyError(f"Metadata for variable '{key}' not found.")
            new_variables[key] = ux.Quantity(
                var, meta[key].units  # type: ignore
            )
        return cls(variables=new_variables, meta=meta)

    def to_default_unit(self) -> Self:
        """Convert all parameters to default unit system.
        - If the parameter is a Field, convert to the default unit defined in metadata.
        - If the parameter is an AbstractQuantity, convert to SI units.
        """
        default = self.default_meta()
        new_variables: Dict[str, ux.AbstractQuantity] = {}
        existing_meta = self.existing_meta()
        for key, var in self.items():
            if isinstance(var, ux.AbstractQuantity):
                default_units = default[key].units
                if var.unit != default_units:
                    new_variables[key] = var.to(default_units)
                    existing_meta[key].units = default_units
                else:
                    new_variables[key] = var
            else:
                raise TypeError(f"Variable '{key}' is neither a Quantity.")
        if self.is_quantities():
            return self._wrap_with_meta(
                variables=new_variables,
                meta=existing_meta
            )
        else:
            raise TypeError(
                "Cannot convert to default unit on incompatible variable types."
                "Please check if all variables are AbstractQuantities."
            )

    def decompose(self, usys: ux.AbstractUnitSystem) -> Self:
        """Convert all parameters to the specified unit system."""
        new_variables: Dict[str, ux.AbstractQuantity] = {}
        existing_meta = self.existing_meta()
        for key, var in self.items():
            if isinstance(var, Field):
                new_variables[key] = var.decompose(usys)  # type: ignore
                existing_meta[key] = self._convert_meta_unit(
                    existing_meta[key], usys
                )
            else:
                raise TypeError(f"Variable '{key}' is not a Field.")
        if self.is_quantities():
            return self._wrap_with_meta(
                variables=new_variables,
                meta=existing_meta
            )
        else:
            raise TypeError(
                "Cannot convert to specified unit on incompatible variable types."
                "Please check if all variables are AbstractQuantities."
            )

    def to_si(self) -> Self:
        """Convert all parameters to SI units."""
        new_variables: Dict[str, ux.AbstractQuantity] = {}
        existing_meta = self.existing_meta()
        for key, var in self.items():
            if isinstance(var, Field):
                new_variables[key] = var.to_si().to_quantity().data
                existing_meta[key] = self._convert_meta_unit(
                    existing_meta[key], ux.unitsystems.si
                )
            else:
                raise TypeError(f"Variable '{key}' is not a Field.")
        if self.is_quantities():
            return type(self)(variables=new_variables, meta=existing_meta)
        else:
            raise TypeError(
                "Cannot convert to SI units on incompatible variable types."
                "Please check if all variables are AbstractQuantities."
            )

    def to_fields(self) -> FV:
        """Convert all parameters to Fields."""
        new_variables: Dict[str, Field] = {}
        for key, var in self.items():
            if isinstance(var, ux.AbstractQuantity):
                new_variables[key] = Field(
                    data=var,
                    meta=self.meta[key]
                )
            else:
                raise TypeError(f"Variable '{key}' is not a AbstractQuantity.")
        field_cls = self.field_varset_type()
        return field_cls(variables=new_variables)

    def to_arrays(self, usys: ux.AbstractUnitSystem = ux.unitsystems.si) -> AV:
        """Convert all parameters to their numeric values in the specified unit system."""
        new_variables: Dict[str, Array] = {}
        for key, var in self.items():
            if isinstance(var, ux.AbstractQuantity):
                new_variables[key] = jnp.asarray(var.to_value(usys))
            else:
                raise TypeError(f"Variable '{key}' is not a AbstractQuantity.")
        array_cls = self.array_varset_type()
        return array_cls(variables=new_variables, meta=self.existing_meta())


def array_convert_from_to_units(
    array: Array,
    from_unit: str,
    to_unit: str
) -> Array:
    quantity = ux.Quantity(array, from_unit)  # type: ignore
    converted = quantity.to(to_unit)
    return converted.value


class AbstractArrayBasedVariableSet(
    AbstractRawVariableSet[FMT, Array],
    Generic[FV, QV, FMT]
):

    def __init__(
        self,
        variables: Dict[str, Array],
        meta: Dict[str, FMT]
    ):
        super().__init__(variables=variables, meta=meta)

    def default_array_dict(self) -> Dict[str, Array]:
        """Return default variable array dictionary."""
        return {
            key: jnp.zeros(())
            for key in self.default_meta().keys()
        }

    @staticmethod
    @abstractmethod
    def default_meta() -> Dict[str, FMT]:
        """Return default variable metadata dictionary."""
        return {}

    @staticmethod
    @abstractmethod
    def field_varset_type() -> Type[FV]:
        """Return the field-based variable set type produced by ``to_value``."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def quantity_varset_type() -> Type[QV]:
        """Return the quantity-based variable set type produced by ``to_fields``."""
        raise NotImplementedError

    def reset(self, val: float = 0.0) -> Self:
        new_vars: Dict[str, Array] = {}
        for key in self.keys():
            var = object.__getattribute__(self, key)
            if isinstance(var, jnp.ndarray):
                new_vars[key] = jnp.full_like(var, val)
            else:
                raise TypeError(
                    f"Variable '{key}' is of unsupported type '{type(var)}' for reset operation."
                )
        return self._wrap(new_vars)

    def existing_values(self) -> Dict[str, Array]:
        """Return existing variables."""
        if getattr(self, "variables", None) is None:
            return self.default_array_dict()
        else:
            if self.is_arrays():
                return {k: v for k, v in self.values()}
            else:
                raise TypeError("Variables are not stored as Arrays.")

    def existing_meta(self) -> Dict[str, FMT]:
        """Return metadata of existing variables."""
        if getattr(self, "variables", None) is None:
            return self.default_meta()
        else:
            return self.meta

    def varnames(self) -> List[str]:
        return list(self.keys())

    def is_arrays(self) -> bool:
        """Check if all stored variables are Arrays."""
        return all(isinstance(v, Array) for v in self.values())

    def unwrap(self) -> Dict[str, Array]:
        if not all(isinstance(v, Array) for v in self.values()):
            raise TypeError(
                "Incompatible variable types after unwrapping, not all Arrays."
            )
        return {k: v for k, v in self.items()}

    @classmethod
    def wrap(cls, vars_dict: Dict[str, Array]) -> Self:
        if not all(isinstance(v, Array) for v in vars_dict.values()):
            raise TypeError(
                "Incompatible variable types for wrapping, must be Arrays"
            )
        return cls(variables=vars_dict, meta=cls.default_meta())

    @classmethod
    def from_arrays(
        cls,
        array_dict: Dict[str, Array],
        meta: Dict[str, FMT]
    ) -> Self:
        return cls(variables=array_dict, meta=meta)

    def to_default_unit(self) -> Self:
        """Convert all parameters to default unit system.
        - If the parameter is a Field, convert to the default unit defined in metadata.
        - If the parameter is an AbstractQuantity, convert to SI units.
        """
        default = self.default_meta()
        meta = self.existing_meta()
        new_variables: Dict[str, Array] = {}
        for key, var in self.items():
            new_variables[key] = array_convert_from_to_units(
                array=var,
                from_unit=meta[key].units,
                to_unit=default[key].units
            )
        if self.is_arrays():
            return self._wrap(new_variables)
        else:
            raise TypeError(
                "Cannot convert to default unit on incompatible variable types."
                "Please check if all variables are Arrays."
            )

    def decompose(self, usys: ux.AbstractUnitSystem) -> Self:
        """Convert all parameters to the specified unit system."""
        new_variables: Dict[str, Array] = {}
        existing_meta = self.existing_meta()
        for key, var in self.items():
            if isinstance(var, Array):
                new_variables[key] = ux.Quantity(
                    var, existing_meta[key].units  # type: ignore
                ).decompose(usys).value  # type: ignore
                existing_meta[key] = self._convert_meta_unit(
                    existing_meta[key], usys
                )
            else:
                raise TypeError(f"Variable '{key}' is not a Field.")
        if self.is_arrays():
            return self._wrap_with_meta(new_variables, existing_meta)
        else:
            raise TypeError(
                "Cannot convert to specified unit on incompatible variable types."
                "Please check if all variables are Arrays."
            )

    def to_si(self) -> Self:
        """Convert all parameters to SI units."""
        return self.decompose(ux.unitsystems.si)

    def to_fields(
        self,
        usys: ux.AbstractUnitSystem = ux.unitsystems.si,
        as_quantity: bool = True
    ) -> FV:
        """Convert all parameters to Fields."""
        new_variables: Dict[str, Field] = {}
        existing_meta = self.existing_meta()
        for key, var in self.items():
            if isinstance(var, Array):
                new_variables[key] = Field(
                    data=ux.Quantity(
                        var, existing_meta[key].units  # type: ignore
                    ),
                    meta=existing_meta[key]
                ).decompose(usys)
            else:
                raise TypeError(f"Variable '{key}' is not a Array.")
            if as_quantity:
                new_variables[key] = new_variables[key].to_quantity(usys)
            else:
                new_variables[key] = new_variables[key].to_array(usys)
        field_cls = self.field_varset_type()
        return field_cls(variables=new_variables)

    def to_quantities(self, usys: ux.AbstractUnitSystem = ux.unitsystems.si) -> QV:
        """Convert all parameters to their numeric values in the specified unit system."""
        new_variables: Dict[str, ux.AbstractQuantity] = {}
        existing_meta = self.existing_meta()
        for key, var in self.items():
            if isinstance(var, Array):
                new_variables[key] = ux.Quantity(
                    var, existing_meta[key].unit  # type: ignore
                ).to(usys)
            else:
                raise TypeError(f"Variable '{key}' is not a Array.")
        quantity_cls = self.quantity_varset_type()
        return quantity_cls(variables=new_variables, meta=existing_meta)


class DiagnosticVariableSet(AbstractVariableSet, Generic[FMT], ABC):
    variables: Dict[str, Array]
    meta: Dict[str, FMT]

    def __init__(self, variables: Dict[str, Array], meta: Dict[str, FMT]):
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "meta", meta)

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

    def _wrap(self, variables: Dict[str, Array]) -> Self:
        return type(self)(variables=variables, meta=self.meta)

    def keys(self) -> List[str]:
        return list(self.variables.keys())

    def values(self) -> List[Array]:
        return list(self.variables.values())

    def items(self) -> List[tuple[str, Array]]:
        return list(self.variables.items())

    def isel(self, index: int, axis: int) -> Self:
        """Index select along a specific dimension for all variables."""
        new_vars: Dict[str, Array] = {}
        for key, var in self.items():
            new_vars[key] = jnp.take(var, indices=index, axis=axis)
        return self._wrap(new_vars)

    def reset(self, val: float = 0.0) -> Self:
        new_vars: Dict[str, Array] = {}
        for key, var in self.items():
            if isinstance(var, Array):
                new_vars[key] = jnp.full_like(var, val)
            else:
                raise TypeError(
                    f"Variable '{key}' is of unsupported type '{type(var)}' for reset operation."
                )
        return self._wrap(new_vars)

    def copy(self) -> Self:
        return type(self)(
            variables={k: v for k, v in self.items()},
            meta=self.meta
        )

    def existing_values(
        self
    ) -> Dict[str, Array]:
        """Return existing variables."""
        if getattr(self, "variables", None) is None:
            raise ValueError(
                "No variables stored, cannot retrieve existing values."
            )
        else:
            if self.is_arrays():
                return self.variables
            else:
                raise TypeError("Variables are not stored as Arrays.")

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
                var, self.meta[key].to_dict(), use_dual=use_dual
            )
        return xr.Dataset(data_vars=var_ds)

    @classmethod
    def from_arrays(
        cls,
        array_dict: Dict[str, Array],
        meta: Dict[str, FMT]
    ) -> Self:
        return cls(variables=array_dict, meta=meta)


class AbstractBlock(ABC, eqx.Module, Generic[STATE, STATIC, TEND, AUX]):
    state: STATE
    static: STATIC
    tend: TEND
    aux: AUX

    def __init__(self, state: STATE, static: STATIC, tend: TEND, aux: AUX):
        self.state = state
        self.static = static
        self.tend = tend
        self.aux = aux

    def copy(self) -> Self:
        return type(self)(
            state=self.state,
            static=self.static,
            tend=self.tend,
            aux=self.aux
        )

    @staticmethod
    @abstractmethod
    def state_cls(*args, **kwargs) -> Type[STATE]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def static_cls(*args, **kwargs) -> Type[STATIC]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def tend_cls(*args, **kwargs) -> Type[TEND]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def aux_cls(*args, **kwargs) -> Type[AUX]:
        raise NotImplementedError

    def reset(self, val: float = 0.0) -> Self:
        return eqx.tree_at(
            lambda b: (b.tend, b.aux),
            self,
            replace=(self.tend.reset(val), self.aux.reset(val))
        )


class AbstractBlockFields(AbstractBlock, Generic[STATE, STATIC, TEND, AUX, BQ, BA]):

    @staticmethod
    @abstractmethod
    def quantity_block_type() -> Type[BQ]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def array_block_type() -> Type[BA]:
        raise NotImplementedError

    def to_quantities(self) -> BQ:
        return self.quantity_block_type()(
            state=self.state.to_quantities(),
            static=self.static.to_quantities(),
            tend=self.tend.to_quantities(),
            aux=self.aux.to_quantities()
        )

    def to_arrays(self) -> BA:
        return self.array_block_type()(
            state=self.state.to_arrays(),
            static=self.static.to_arrays(),
            tend=self.tend.to_arrays(),
            aux=self.aux.to_arrays()
        )


class AbstractBlockQuantities(AbstractBlock, Generic[STATE, STATIC, TEND, AUX, BF, BA]):

    @staticmethod
    @abstractmethod
    def field_block_type() -> Type[BF]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def array_block_type() -> Type[BA]:
        raise NotImplementedError

    def to_fields(self) -> BF:
        return self.field_block_type()(
            state=self.state.to_fields(),
            static=self.static.to_fields(),
            tend=self.tend.to_fields(),
            aux=self.aux.to_fields()
        )

    def to_arrays(self) -> BA:
        return self.array_block_type()(
            state=self.state.to_arrays(),
            static=self.static.to_arrays(),
            tend=self.tend.to_arrays(),
            aux=self.aux.to_arrays()
        )


class AbstractBlockArrays(AbstractBlock, Generic[STATE, STATIC, TEND, AUX, BF, BQ]):

    @staticmethod
    @abstractmethod
    def field_block_type() -> Type[BF]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def quantity_block_type() -> Type[BQ]:
        raise NotImplementedError

    def to_fields(self) -> BF:
        return self.field_block_type()(
            state=self.state.to_fields(),
            static=self.static.to_fields(),
            tend=self.tend.to_fields(),
            aux=self.aux.to_fields()
        )

    def to_quantities(self) -> BQ:
        return self.quantity_block_type()(
            state=self.state.to_quantities(),
            static=self.static.to_quantities(),
            tend=self.tend.to_quantities(),
            aux=self.aux.to_quantities()
        )


class AbstractDiagnosticTools(Generic[HM, VM, LO, DM, ES, FMT], ABC):

    def __init__(self, equation: ES, jit: bool = False, trajectory: bool = False):
        self._equation = equation
        self.jit = jit
        self.trajectory = trajectory

    @abstractmethod
    def to_diag(self, state, tend, aux) -> DiagnosticVariableSet[FMT]:
        """Convert input variable set to diagnostic variable set."""
        raise NotImplementedError

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
        field: Field,
        to_ugrid: Literal[False] = False
    ) -> Field:
        ...

    @overload
    def process_field(
        self,
        field: Field,
        to_ugrid: Literal[True]
    ) -> tuxr.UGRIDDataArray:
        ...

    def process_field(
        self,
        field: Field,
        to_ugrid: bool = False
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
                ds = res.to_xarray_with_ugrid_format()
                uxgrid = self.coord.mesh.to_ugrid().to_uxgrid()
                return tuxr.UGRIDDataArray.from_xarray(ds, uxgrid)
            else:
                return res
        else:
            if to_ugrid:
                ds = field.to_xarray_with_ugrid_format()
                uxgrid = self.coord.mesh.to_ugrid().to_uxgrid()
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
    def coord(self) -> CoordinateSystem[HM, VM, LO]:
        return self.equation.coord

    @property
    def mesh(self) -> HM:
        return self.equation.coord.mesh

    @property
    def vert(self) -> VM:
        return self.equation.coord.vert

    @property
    def layout(self) -> LO:
        return self.equation.coord.layout

    @property
    def method(self) -> DM:
        return self.equation.method

    @property
    def para(self) -> PlanetParameterSet:
        return self.equation.para

    @property
    def equation(self) -> ES:
        if not hasattr(self, "_equation"):
            raise AttributeError(
                "Equation has not been set. "
                "Please using 'set_equation' method to specify the dynamics first."
            )
        return self._equation
