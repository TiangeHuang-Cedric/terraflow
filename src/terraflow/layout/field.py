from abc import ABC, abstractmethod
from typing import Union, TypeVar, Literal, Dict, Any
from typing import Self, Generic, overload, cast
import numpy as np
import unxt as ux
import equinox as eqx
import jax.numpy as jnp
import xarray as xr
import uxarray as uxr

import terraflow.util.uxarray as tuxr
from terraflow.util import typing
from terraflow.mesh.abstract import AbstractMesh
from terraflow.vert_coord.abstract import AbstractVertCoordinate
from terraflow.layout.abstract import AbstractLayout
from terraflow.layout.abstract import CoordinateInfo
from terraflow.layout.abstract import CoordinateSystem


Array = typing.Array
ArrayWithUnits = typing.ArrayWithUnits

HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
HLO = TypeVar("HLO", bound=AbstractLayout)
VLO = TypeVar("VLO", bound=AbstractLayout)
FDT = TypeVar("FDT", Array, ux.AbstractQuantity)
FMT = TypeVar("FMT", bound="AbstractMetaData")


def unit_convert(from_unit: str, usys: ux.AbstractUnitSystem) -> str:
    """Convert the unit string to the specified unit system."""
    quantity = ux.Quantity(1.0, from_unit)  # type: ignore
    converted = quantity.decompose(usys)  # type: ignore
    return converted.unit.to_string()


class AbstractMetaData(eqx.Module, ABC):
    """Abstract metadata for AbstractField."""
    name: str = eqx.field(static=True, default="")
    long_name: str = eqx.field(static=True, default="")
    description: str = eqx.field(static=True, default="")
    units: str = eqx.field(static=True, default="")
    is_static: bool = eqx.field(static=True, default=False)
    is_expand_dims: bool = eqx.field(static=True, default=False)
    expand_axis: int = eqx.field(static=True, default=0)
    expand_axis_length: int = eqx.field(static=True, default=1)
    output: str = eqx.field(static=True, default="")

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if the metadata indicates an empty field."""
        raise NotImplementedError

    @abstractmethod
    def wrap_unit(self, usys: Union[ux.AbstractUnitSystem, str]) -> Self:
        """Return a copy of the metadata with its unit converted to ``usys``."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create(cls, **kwargs) -> Self:
        """Create a new FieldMetaData instance."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, meta: dict) -> Self:
        """Create a new FieldMetaData instance from a dictionary."""
        raise NotImplementedError

    @abstractmethod
    def zeros_like(self, coord: CoordinateSystem[HM, VM, HLO, VLO]) -> Any:
        """Create a zero-filled array with the same shape as the specified layout position."""
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert the FieldMetaData instance to a dictionary."""
        raise NotImplementedError


class EmptyMetaData(AbstractMetaData):

    def is_empty(self) -> bool:
        return True

    def wrap_unit(self, usys: Union[ux.AbstractUnitSystem, str]) -> Self:
        """Return a copy of the metadata with its unit converted to ``usys``."""
        raise ValueError(
            "Cannot wrap unit when metadata is empty."
        )

    @classmethod
    def create(cls, **kwargs) -> Self:
        """Create a new FieldMetaData instance."""
        return cls()

    @classmethod
    def from_dict(cls, meta: dict) -> Self:
        """Create a new FieldMetaData instance from a dictionary."""
        raise ValueError(
            "Cannot create from dict when metadata is empty."
        )

    def zeros_like(self, coord: CoordinateSystem[HM, VM, HLO, VLO]) -> Any:
        """Create a zero-filled array with the same shape as the specified layout position."""
        raise ValueError(
            "Cannot create zeros_like when metadata is empty."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the FieldMetaData instance to a dictionary."""
        raise ValueError(
            "Cannot convert to dict when metadata is empty."
        )


class NumericMetaData(AbstractMetaData):
    """Metadata for numeric fields that can be directly converted to arrays."""

    @classmethod
    def create(cls, **kwargs) -> Self:
        """Create a new NumericMetaData instance."""
        return cls(
            name=kwargs.get("name", ""),
            long_name=kwargs.get("long_name", ""),
            description=kwargs.get("description", ""),
            units=kwargs.get("units", ""),
            is_static=kwargs.get("is_static", False),
            is_expand_dims=kwargs.get("is_expand_dims", False),
            expand_axis=kwargs.get("expand_axis", 0),
            expand_axis_length=kwargs.get("expand_axis_length", 1),
            output=kwargs.get("output", ""),
        )

    @classmethod
    def from_dict(cls, meta: dict) -> Self:
        """Create a new NumericMetaData instance from a dictionary."""
        return cls(
            name=meta.get("name", ""),
            long_name=meta.get("long_name", ""),
            description=meta.get("description", ""),
            units=meta.get("units", ""),
            is_static=meta.get("is_static", False),
            is_expand_dims=meta.get("is_expand_dims", False),
            expand_axis=meta.get("expand_axis", 0),
            expand_axis_length=meta.get("expand_axis_length", 1),
            output=meta.get("output", ""),
        )

    def wrap_unit(self, usys: Union[ux.AbstractUnitSystem, str]) -> Self:
        """Return a copy of the metadata with its unit converted to ``usys``."""
        if isinstance(usys, str):
            units = usys
        else:
            units = unit_convert(self.units, usys)

        return type(self)(
            name=self.name,
            long_name=self.long_name,
            description=self.description,
            units=units,
            is_static=self.is_static,
            is_expand_dims=self.is_expand_dims,
            expand_axis=self.expand_axis,
            expand_axis_length=self.expand_axis_length,
            output=self.output,
        )

    def is_empty(self) -> bool:
        """Check if the metadata indicates an empty field."""
        return self.name == "" and self.units == ""

    def zeros_like(self, coord: CoordinateSystem[HM, VM, HLO, VLO]) -> "ArrayField[Self]":
        """Create a zero-filled array with the same shape as the specified layout position."""
        if self.is_expand_dims:
            data = jnp.zeros(
                (self.expand_axis_length,),
                dtype=coord.dev.float_type
            )
        else:
            data = jnp.array(0.0, dtype=coord.dev.float_type)
        return ArrayField.create(
            data=data,
            meta=self,
            as_quantity=False
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "long_name": self.long_name,
            "description": self.description,
            "units": self.units,
            "is_static": self.is_static,
            "is_expand_dims": self.is_expand_dims,
            "expand_axis": self.expand_axis,
            "expand_axis_length": self.expand_axis_length,
            "output": self.output,
        }


class FieldMetaData(AbstractMetaData):
    """Metadata for AbstractField."""
    name: str = eqx.field(static=True, default="")
    dims: tuple[str, ...] = eqx.field(static=True, default=())
    long_name: str = eqx.field(static=True, default="")
    description: str = eqx.field(static=True, default="")
    units: str = eqx.field(static=True, default="")
    hori_location: str = eqx.field(static=True, default="")
    vert_location: str = eqx.field(static=True, default="")
    basis: str = eqx.field(static=True, default="")
    coordinates: str = eqx.field(static=True, default="")
    is_prognostic: bool = eqx.field(static=True, default=False)
    is_static: bool = eqx.field(static=True, default=False)
    is_expand_dims: bool = eqx.field(static=True, default=False)
    expand_axis: int = eqx.field(static=True, default=0)
    expand_axis_length: int = eqx.field(static=True, default=1)
    output: str = eqx.field(static=True, default="")

    def is_empty(self) -> bool:
        """Check if the metadata indicates an empty field."""
        return self.name == "" and self.dims == () and self.units == ""

    def wrap_unit(self, usys: Union[ux.AbstractUnitSystem, str]) -> Self:
        """Return a copy of the metadata with its unit converted to ``usys``."""
        if isinstance(usys, str):
            units = usys
        else:
            units = unit_convert(self.units, usys)

        # All fields in FieldMetaData are marked static, so we can't rely on
        # ``eqx.tree_at`` to update ``unit``. Create a new metadata instance
        # instead to avoid PyTree structure mismatches when used inside
        # ``Field._wrap_with_unit``.
        return type(self)(
            name=self.name,
            dims=self.dims,
            long_name=self.long_name,
            description=self.description,
            units=units,
            hori_location=self.hori_location,
            vert_location=self.vert_location,
            basis=self.basis,
            coordinates=self.coordinates,
            is_prognostic=self.is_prognostic,
            is_static=self.is_static,
            is_expand_dims=self.is_expand_dims,
            expand_axis=self.expand_axis,
            expand_axis_length=self.expand_axis_length,
            output=self.output,
        )

    def zeros_like(self, coord: CoordinateSystem[HM, VM, HLO, VLO]) -> "ArrayField[Self]":
        """Create a zero-filled array with the same shape as the specified layout position."""
        data = coord.zeros_like(
            hori_loc=self.hori_location,
            vert_loc=self.vert_location,
            basis=self.basis,
            is_expand_dims=self.is_expand_dims,
            expand_axis=self.expand_axis,
            expand_axis_length=self.expand_axis_length,
        )
        return ArrayField.create(
            data=data,
            meta=self,
            as_quantity=False
        )

    def get_hori_position(self, coord: CoordinateSystem[HM, VM, HLO, VLO]) -> CoordinateInfo:
        return coord.get_hori_layout(
            loc=self.hori_location, basis=self.basis
        )

    def get_vert_position(self, coord: CoordinateSystem[HM, VM, HLO, VLO]) -> CoordinateInfo:
        return coord.get_vert_layout(
            loc=self.vert_location
        )

    @classmethod
    def create(cls, **kwargs) -> Self:
        """Create a new FieldMetaData instance."""
        return cls(
            name=kwargs.get("name", ""),
            dims=kwargs.get("dims", ()),
            long_name=kwargs.get("long_name", ""),
            description=kwargs.get("description", ""),
            units=kwargs.get("units", ""),
            hori_location=kwargs.get("hori_location", ""),
            vert_location=kwargs.get("vert_location", ""),
            basis=kwargs.get("basis", ""),
            coordinates=kwargs.get("coordinates", ""),
            is_prognostic=kwargs.get("is_prognostic", False),
            is_static=kwargs.get("is_static", False),
            is_expand_dims=kwargs.get("is_expand_dims", False),
            expand_axis=kwargs.get("expand_axis", 0),
            expand_axis_length=kwargs.get("expand_axis_length", 1),
            output=kwargs.get("output", ""),
        )

    @classmethod
    def from_dict(cls, meta: dict) -> Self:
        """Create a new FieldMetaData instance from a dictionary."""
        return cls(
            name=meta.get("name", ""),
            dims=tuple(meta.get("dims", ())),
            long_name=meta.get("long_name", ""),
            description=meta.get("description", ""),
            units=meta.get("units", ""),
            hori_location=meta.get("hori_location", ""),
            vert_location=meta.get("vert_location", ""),
            basis=meta.get("basis", ""),
            coordinates=meta.get("coordinates", ""),
            is_prognostic=meta.get("is_prognostic", False),
            is_static=meta.get("is_static", False),
            is_expand_dims=meta.get("is_expand_dims", False),
            expand_axis=meta.get("expand_axis", 0),
            expand_axis_length=meta.get("expand_axis_length", 1),
            output=meta.get("output", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the FieldMetaData instance to a dictionary."""
        return {
            "name": self.name,
            "dims": self.dims,
            "long_name": self.long_name,
            "description": self.description,
            "units": self.units,
            "hori_location": self.hori_location,
            "vert_location": self.vert_location,
            "basis": self.basis,
            "coordinates": self.coordinates,
            "is_prognostic": self.is_prognostic,
            "is_static": self.is_static,
            "is_expand_dims": self.is_expand_dims,
            "expand_axis": self.expand_axis,
            "expand_axis_length": self.expand_axis_length,
            "output": self.output,
        }


class StaggeredFieldMetaData(FieldMetaData):
    """Metadata for staggered AbstractField."""
    staggered_kind: str = eqx.field(static=True, default="")
    mesh_name: str = eqx.field(static=True, default="")

    def wrap_unit(self, usys: Union[ux.AbstractUnitSystem, str]) -> Self:
        """Return a copy of the metadata with its unit converted to ``usys``."""
        if isinstance(usys, str):
            units = usys
        else:
            units = unit_convert(self.units, usys)

        return type(self)(
            name=self.name,
            dims=self.dims,
            long_name=self.long_name,
            description=self.description,
            units=units,
            hori_location=self.hori_location,
            vert_location=self.vert_location,
            basis=self.basis,
            coordinates=self.coordinates,
            staggered_kind=self.staggered_kind,
            mesh_name=self.mesh_name,
            is_prognostic=self.is_prognostic,
            is_static=self.is_static,
            is_expand_dims=self.is_expand_dims,
            expand_axis=self.expand_axis,
            expand_axis_length=self.expand_axis_length,
            output=self.output,
        )

    @classmethod
    def create(cls, **kwargs) -> Self:
        """Create a new StaggeredFieldMetaData instance."""
        return cls(
            name=kwargs.get("name", ""),
            dims=kwargs.get("dims", ()),
            long_name=kwargs.get("long_name", ""),
            description=kwargs.get("description", ""),
            units=kwargs.get("units", ""),
            hori_location=kwargs.get("hori_location", ""),
            vert_location=kwargs.get("vert_location", ""),
            basis=kwargs.get("basis", ""),
            coordinates=kwargs.get("coordinates", ""),
            staggered_kind=kwargs.get("staggered_kind", ""),
            mesh_name=kwargs.get("mesh_name", ""),
            is_prognostic=kwargs.get("is_prognostic", False),
            is_static=kwargs.get("is_static", False),
            is_expand_dims=kwargs.get("is_expand_dims", False),
            expand_axis=kwargs.get("expand_axis", 0),
            expand_axis_length=kwargs.get("expand_axis_length", 1),
            output=kwargs.get("output", ""),
        )

    @classmethod
    def from_dict(cls, meta: dict) -> Self:
        """Create a new StaggeredFieldMetaData instance from a dictionary."""
        return cls(
            name=meta.get("name", ""),
            dims=tuple(meta.get("dims", ())),
            long_name=meta.get("long_name", ""),
            description=meta.get("description", ""),
            units=meta.get("units", ""),
            hori_location=meta.get("hori_location", ""),
            vert_location=meta.get("vert_location", ""),
            basis=meta.get("basis", ""),
            coordinates=meta.get("coordinates", ""),
            staggered_kind=meta.get("staggered_kind", ""),
            mesh_name=meta.get("mesh_name", ""),
            is_prognostic=meta.get("is_prognostic", False),
            is_static=meta.get("is_static", False),
            is_expand_dims=meta.get("is_expand_dims", False),
            expand_axis=meta.get("expand_axis", 0),
            expand_axis_length=meta.get("expand_axis_length", 1),
            output=meta.get("output", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the StaggeredFieldMetaData instance to a dictionary."""
        return {
            **super().to_dict(),
            "staggered_kind": self.staggered_kind,
            "mesh_name": self.mesh_name,
        }


class Field(eqx.Module, Generic[FDT, FMT]):
    """
    AbstractField class that wraps JAX arrays and provides arithmetic operations.

    This class is designed to work with JAX's functional programming model
    and equinox's immutable Module system. All operations return new AbstractField
    instances rather than modifying in place.
    """
    data: FDT = eqx.field()
    _meta: FMT = eqx.field(static=True)

    def __init__(self, data: FDT, meta: FMT):
        if isinstance(data, ux.AbstractQuantity):
            validated_data = cast(ux.AbstractQuantity, data)
        elif isinstance(data, (jnp.ndarray, np.ndarray)):
            validated_data = cast(Array, data)
        else:
            raise TypeError(
                "data must be either a JAX-like array or an unxt.AbstractQuantity"
            )
        if isinstance(meta, FieldMetaData):
            validated_meta = cast(FieldMetaData, meta)
        elif isinstance(meta, StaggeredFieldMetaData):
            validated_meta = cast(StaggeredFieldMetaData, meta)
        elif isinstance(meta, NumericMetaData):
            validated_meta = cast(NumericMetaData, meta)
        elif isinstance(meta, EmptyMetaData):
            validated_meta = cast(EmptyMetaData, meta)
        else:
            raise TypeError(
                "meta must be either a FieldMetaData, StaggeredFieldMetaData, NumericMetaData, or EmptyMetaData"
            )
        object.__setattr__(self, 'data', cast(FDT, validated_data))
        object.__setattr__(self, '_meta', validated_meta)

    def __getattr__(self, name: str):
        """
        Delegate attribute access to the underlying data array.
        This allows direct access to JAX array methods without explicit .data access.
        """
        # Avoid infinite recursion by using object.__getattribute__
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        try:
            data = object.__getattribute__(self, "data")
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Get the attribute from the data array
        attr = getattr(data, name)

        # If it's a method, wrap it to return AbstractField instances
        if callable(attr):
            def wrapped_method(*args, **kwargs):
                result = attr(*args, **kwargs)
                # If result is an array, wrap it in a AbstractField
                if isinstance(result, (jnp.ndarray, np.ndarray)):
                    return self._wrap(result)
                return result
            return wrapped_method

        return attr

    # --- Python / NumPy Protocol ---
    def __array__(self, dtype=None):
        """Support numpy array conversion."""
        return np.asarray(self.data, dtype=dtype)

    def __jax_array__(self):
        """Support JAX array protocol."""
        return self.data

    def __repr__(self):
        info = f"name='{self.meta.name}'" if self.meta.name else "unnamed"
        shape = f"shape={self.data.shape}" if hasattr(
            self.data, 'shape'
        ) else ""
        return f"Field({info}, {shape})"

    def __len__(self) -> int:
        """Return length of the data array."""
        return len(self.data)

    def __getitem__(self, index) -> Self:
        """Support indexing, returns a new AbstractField with indexed data."""
        return self._wrap(self.data[index])

    # --- Arithmetic Operations ---
    def _coerce(self, other):
        """Extract data from AbstractField or return as-is."""
        if isinstance(other, Field):
            return other.data
        return other

    def _wrap(self, value) -> Self:
        """Wrap a value in a new AbstractField instance with same metadata."""
        if self.is_quantity():
            if isinstance(value, ux.AbstractQuantity):
                array_value = value.to(self.meta.units)
            else:
                array_value = ux.Quantity(
                    cast(Array, jnp.asarray(value)),  # type: ignore[arg-type]
                    self.meta.units
                )
        else:
            if isinstance(value, ux.AbstractQuantity):
                array_value = value.to(self.meta.units).value
            else:
                array_value = jnp.asarray(value)

        return eqx.tree_at(
            lambda f: f.data,
            self,
            cast(FDT, array_value)
        )

    def _wrap_with_unit(
        self,
        value,
        unit: Union[ux.AbstractUnitSystem, str]
    ) -> Self:
        """Wrap a value in a new AbstractField instance with specified unit."""
        # Handle unxt.Quantity objects by extracting their value
        if isinstance(self.meta, EmptyMetaData):
            raise ValueError(
                "Cannot wrap with unit when metadata is empty."
            )
        meta = self.meta.wrap_unit(unit)
        instance = type(self)(
            data=self.data,
            meta=meta,
        )
        return instance._wrap(value)

    def __add__(self, other) -> Self:
        return self._wrap(self.data + self._coerce(other))

    def __radd__(self, other) -> Self:
        return self._wrap(self._coerce(other) + self.data)

    def __sub__(self, other) -> Self:
        return self._wrap(self.data - self._coerce(other))

    def __rsub__(self, other) -> Self:
        return self._wrap(self._coerce(other) - self.data)

    def __mul__(self, other) -> Self:
        return self._wrap(self.data * self._coerce(other))

    def __rmul__(self, other) -> Self:
        return self._wrap(self._coerce(other) * self.data)

    def __truediv__(self, other) -> Self:
        return self._wrap(self.data / self._coerce(other))

    def __rtruediv__(self, other) -> Self:
        return self._wrap(self._coerce(other) / self.data)

    def __pow__(self, other) -> Self:
        return self._wrap(self.data ** self._coerce(other))

    def __neg__(self) -> Self:
        return self._wrap(-self.data)

    def __pos__(self) -> Self:
        return self._wrap(+self.data)

    def __abs__(self) -> Self:
        return self._wrap(jnp.abs(self.data))  # type: ignore

    def isel(self, index: int, axis: int) -> Self:
        """Index along a specified axis."""
        if isinstance(self.data, Array):
            return self._wrap(jnp.take(self.data, indices=index, axis=axis))
        elif isinstance(self.data, ux.AbstractQuantity):
            new_data = jnp.take(
                jnp.asarray(self.data.value),
                indices=index,
                axis=axis
            )
            quantity_data = ux.Quantity(
                new_data, self.data.unit)  # type: ignore
            return self._wrap(quantity_data)
        else:
            raise TypeError(
                "data must be either a JAX-like array or an unxt.AbstractQuantity"
            )

    def reshape(self, new_shape: tuple[int, ...]) -> Self:
        """Reshape the underlying data array."""
        if isinstance(self.data, Array):
            return self._wrap(jnp.reshape(self.data, new_shape))
        elif isinstance(self.data, ux.AbstractQuantity):
            new_data = jnp.reshape(
                jnp.asarray(self.data.value),
                new_shape
            )
            quantity_data = ux.Quantity(
                new_data, self.data.unit)  # type: ignore
            return self._wrap(quantity_data)
        else:
            raise TypeError(
                "data must be either a JAX-like array or an unxt.AbstractQuantity"
            )

    def zeros_like(self) -> Self:
        """Return a new AbstractField with the same shape and metadata but filled with zeros."""
        if isinstance(self.data, Array):
            zero_data = jnp.zeros_like(self.data)
            return self._wrap(zero_data)
        elif isinstance(self.data, ux.AbstractQuantity):
            zero_array = jnp.zeros_like(jnp.asarray(self.data.value))
            zero_quantity = ux.Quantity(
                zero_array, self.data.unit)  # type: ignore
            return self._wrap(zero_quantity)
        else:
            raise TypeError(
                "data must be either a JAX-like array or an unxt.AbstractQuantity"
            )

    @overload
    @classmethod
    def empty(
        cls,
        as_quantity: Literal[False] = False
    ) -> "ArrayField[FMT]":
        ...

    @overload
    @classmethod
    def empty(
        cls,
        as_quantity: Literal[True]
    ) -> "QuantityField[FMT]":
        ...

    @classmethod
    def empty(
        cls,
        as_quantity: bool = False
    ) -> Union["ArrayField[FMT]", "QuantityField[FMT]"]:
        """Create an empty AbstractField instance with default metadata."""
        meta = EmptyMetaData()
        if as_quantity:
            quantity_data = ux.Quantity(
                [], meta.units  # type: ignore
            )
            return QuantityField(data=quantity_data, meta=meta)
        else:
            return ArrayField(data=jnp.array([]), meta=meta)

    def is_empty(self) -> bool:
        """Check if the field is empty (i.e., has no data)."""
        if not (isinstance(self.data, Array) or isinstance(self.data, ux.AbstractQuantity)):
            raise TypeError(
                "data must be either a JAX-like array or an unxt.AbstractQuantity"
            )
        if isinstance(self.data, Array) and self.meta.is_empty():
            return self.data.size == 0
        elif isinstance(self.data, ux.AbstractQuantity) and self.meta.is_empty():
            return self.data.value.size == 0
        else:
            return False

    @overload
    @classmethod
    def create(
        cls,
        data: Union[Array, ux.AbstractQuantity],
        meta: FMT,
        as_quantity: Literal[False]
    ) -> "ArrayField[FMT]":
        ...

    @overload
    @classmethod
    def create(
        cls,
        data: Union[Array, ux.AbstractQuantity],
        meta: FMT,
        as_quantity: Literal[True]
    ) -> "QuantityField[FMT]":
        ...

    @classmethod
    def create(
        cls,
        data: Union[Array, ux.AbstractQuantity],
        meta: FMT,
        as_quantity: bool = False
    ) -> Union["ArrayField[FMT]", "QuantityField[FMT]"]:
        """Create a new AbstractField instance with the same metadata."""
        if as_quantity:
            if isinstance(data, ux.AbstractQuantity):
                new_data = data.to(meta.units)
                return QuantityField(data=new_data, meta=meta)
            else:
                quantity_data = ux.Quantity(
                    cast(Array, jnp.asarray(data)),  # type: ignore[arg-type]
                    meta.units
                )
                return QuantityField(data=quantity_data, meta=meta)
        else:
            if isinstance(data, ux.AbstractQuantity):
                array_data = data.to(meta.units).value
                return ArrayField(data=jnp.asarray(array_data), meta=meta)
            else:
                return ArrayField(data=jnp.asarray(data), meta=meta)

    @property
    def meta(self) -> FMT:
        return self._meta

    @overload
    def update(
        self,
        data: Array,
        unit: Union[str, ux.AbstractUnitSystem] = ""
    ) -> Self: ...

    @overload
    def update(
        self,
        data: ux.AbstractQuantity,
        unit: Union[str, ux.AbstractUnitSystem] = ""
    ) -> Self: ...

    def update(
        self,
        data: Union[Array, ux.AbstractQuantity],
        unit: Union[str, ux.AbstractUnitSystem] = ""
    ) -> Self:
        """Update the field data and unit."""
        if unit:
            return self._wrap_with_unit(data, unit)
        else:
            return self._wrap(data)  # type: ignore

    def is_array(self) -> bool:
        """Check if the underlying data is a raw array (without units)."""
        return isinstance(self.data, (jnp.ndarray, np.ndarray))

    def is_quantity(self) -> bool:
        """Check if the underlying data is an AbstractQuantity."""
        return isinstance(self.data, ux.AbstractQuantity)

    def unwrap(self) -> FDT:
        """Unwrap and return the underlying data array."""
        if self.is_quantity() or self.is_array():
            return cast(FDT, self.data)
        else:
            raise TypeError(
                "Value is neither a Quantity nor a JAX array."
            )

    def to(self, unit: str) -> Self:
        """Convert the parameter value to the specified unit and return it."""
        if isinstance(self.data, ux.AbstractQuantity):
            new_data = self.data.to(unit)
            return self.update(new_data, unit)
        else:
            new_data = ux.Quantity(
                jnp.asarray(self.data),  # type: ignore[arg-type]
                self.meta.units
            ).to(unit)
            return self.update(new_data, unit)

    def decompose(self, usys: ux.AbstractUnitSystem) -> Self:
        """Decompose the parameter value and return it."""
        if isinstance(self.data, ux.AbstractQuantity):
            new_data = self.data.decompose(usys)  # type: ignore
            return self.update(new_data, usys)
        else:
            new_data = ux.Quantity(
                jnp.asarray(self.data),  # type: ignore[arg-type]
                self.meta.units
            ).decompose(usys)  # type: ignore[arg-type]
            return self.update(new_data, usys)

    def to_si(self) -> Self:
        """Convert the parameter value to SI unit and return it."""
        if isinstance(self.data, ux.Quantity):
            new_data = self.decompose(ux.unitsystems.si).data
        else:
            new_data = ux.Quantity(
                jnp.asarray(self.data),  # type: ignore[arg-type]
                self.meta.units
            ).decompose(ux.unitsystems.si)  # type: ignore[arg-type]
        return self.update(new_data, ux.unitsystems.si)

    def to_quantity(
        self,
        usys: ux.AbstractUnitSystem = ux.unitsystems.si
    ) -> "QuantityField[FMT]":
        """Convert the parameter value to its Quantity representation in the specified unit system."""
        if isinstance(self.data, ux.Quantity):
            new_data = self.data.decompose(usys)  # type: ignore[arg-type]
        elif isinstance(self.data, (jnp.ndarray, np.ndarray)):
            base = ux.Quantity(  # type: ignore[arg-type]
                jnp.asarray(self.data),  # type: ignore[arg-type]
                self.meta.units
            )
            new_data = base.decompose(usys)  # type: ignore[arg-type]
        else:
            raise TypeError(
                "Value is neither a Quantity nor a JAX array."
            )
        if isinstance(self.meta, EmptyMetaData):
            raise ValueError(
                "Cannot convert to quantity when metadata is empty."
            )
        meta = self.meta.wrap_unit(usys)
        return QuantityField(data=new_data, meta=meta)

    def to_array(
        self,
        usys: ux.AbstractUnitSystem = ux.unitsystems.si
    ) -> "ArrayField[FMT]":
        """Convert the parameter value to its numeric value in the specified unit system."""
        if isinstance(self.data, ux.AbstractQuantity):
            new_data = self.data.decompose(
                usys  # type: ignore[arg-type]
            ).value
        elif isinstance(self.data, (jnp.ndarray, np.ndarray)):
            base = ux.Quantity(
                jnp.asarray(self.data),  # type: ignore[arg-type]
                self.meta.units
            )
            new_data = base.decompose(usys).value  # type: ignore[arg-type]
        else:
            raise TypeError(
                "Value is neither a Quantity nor a JAX array."
            )
        if isinstance(self.meta, EmptyMetaData):
            raise ValueError(
                "Cannot convert to array when metadata is empty."
            )
        meta = self.meta.wrap_unit(usys)
        return ArrayField(data=jnp.asarray(new_data), meta=meta)

    def get_hori_position(self, coord: CoordinateSystem[HM, VM, HLO, VLO]) -> CoordinateInfo:
        if not isinstance(self.meta, FieldMetaData):
            raise TypeError(
                "meta must be an instance of FieldMetaData to get horizontal position"
            )
        return self.meta.get_hori_position(coord)

    def get_vert_position(self, coord: CoordinateSystem[HM, VM, HLO, VLO]) -> CoordinateInfo:
        if not isinstance(self.meta, FieldMetaData):
            raise TypeError(
                "meta must be an instance of FieldMetaData to get vertical position"
            )
        return self.meta.get_vert_position(coord)

    def to_ugrid(
        self,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        use_dual: bool = False,
        **kwargs
    ) -> tuxr.UGRIDDataArray:
        """Convert the Field to a UgridDataArray."""
        ds = self.to_xarray_with_ugrid_format(use_dual=use_dual)
        ugrid = coord.mesh.to_ugrid(use_dual=use_dual, **kwargs).to_uxgrid()
        return tuxr.UGRIDDataArray.from_xarray(ds, ugrid)

    def to_uxarray(
        self,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        use_dual: bool = False,
        **kwargs
    ) -> uxr.UxDataArray:
        """Convert the Field to an Uxarray.DataArray."""
        ds = self.to_xarray_with_ugrid_format(use_dual=use_dual)
        uxgrid = coord.mesh.to_ugrid(use_dual=use_dual, **kwargs).to_uxgrid()
        return uxr.UxDataArray.from_xarray(ds, uxgrid)

    def to_xarray_with_ugrid_format(self, use_dual: bool = False) -> xr.DataArray:
        """Convert the Field to an xarray.Dataset with UGRID format."""
        if isinstance(self.meta, EmptyMetaData):
            raise ValueError(
                "Cannot convert to xarray when metadata is empty."
            )
        return tuxr.format_terraflow2ugrid(
            self.data, self.meta.to_dict(), use_dual=use_dual
        )


# Convenience aliases for common specializations
QuantityField = Field[ux.AbstractQuantity, FMT]
ArrayField = Field[Array, FMT]
FieldLike = QuantityField | ArrayField
