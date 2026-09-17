from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar, Dict, Any, List
from typing import Self, Generic, Type, Optional
import pint


from terraflow.util import typing
from terraflow.util import equinox as eqx
from terraflow.util.backend import AbstractBackend
from terraflow.util.planet import unit_registry


if TYPE_CHECKING:
    from terraflow.util.device import AbstractDevice


Array = typing.Array
ArrayWithUnits = typing.ArrayWithUnits

FMT = TypeVar("FMT", bound="AbstractMeta")


def unit_convert(from_unit: str, unit: str = "SI") -> str:
    """Convert the unit string to the specified unit or unit system."""
    quantity = unit_registry.Quantity(1.0, from_unit)
    if unit.lower() == "si":
        converted = quantity.to_base_units()
    else:
        converted = quantity.to(unit)
    return str(converted.units)


class AbstractMeta(eqx.Module, ABC):
    """Abstract metadata for AbstractField."""
    name           : str      = eqx.field(static=True, default=""   )
    long_name      : str      = eqx.field(static=True, default=""   )
    units          : str      = eqx.field(static=True, default=""   )
    is_static      : bool     = eqx.field(static=True, default=False)
    is_expand_dim0 : bool     = eqx.field(static=True, default=False)
    expand_dim0_len: int      = eqx.field(static=True, default=1    )
    output         : str      = eqx.field(static=True, default=""   )
    backend        : AbstractBackend  = eqx.field(static=True, default=None )
    
    def is_empty(self) -> bool:
        """Check if the metadata indicates an empty field."""
        return self.name == "" and self.long_name == "" and self.units == ""

    def wrap_unit(self, unit: str) -> Self:
        """Return a copy of the metadata with its unit converted to ``unit``."""
        meta = self.to_dict()
        if unit.lower() == "si":
            meta["units"] = unit_convert(self.units, unit)
        else:
            meta["units"] = unit
        return type(self).from_dict(meta)

    @classmethod
    def from_dict(cls, meta: dict) -> Self:
        """Create a new FieldMetaData instance from a dictionary."""
        return cls(**meta)

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert the FieldMetaData instance to a dictionary."""
        raise NotImplementedError


class ScalarFieldMeta(AbstractMeta):
    """Metadata for numeric fields that can be directly converted to arrays."""
    name           : str     = eqx.field(static=True, default=""   )
    long_name      : str     = eqx.field(static=True, default=""   )
    units          : str     = eqx.field(static=True, default=""   )
    is_static      : bool    = eqx.field(static=True, default=False)
    is_expand_dim0 : bool    = eqx.field(static=True, default=False)
    expand_dim0_len: int     = eqx.field(static=True, default=1    )
    output         : str     = eqx.field(static=True, default=""   )
    backend        : AbstractBackend = eqx.field(static=True, default=None )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "long_name": self.long_name,
            "units": self.units,
            "is_static": self.is_static,
            "is_expand_dim0": self.is_expand_dim0,
            "expand_dim0_len": self.expand_dim0_len,
            "output": self.output,
            "backend": self.backend,
        }


class AbstractField(eqx.Module, Generic[FMT]):
    """
    AbstractField class that wraps JAX arrays and provides arithmetic operations.

    This class is designed to work with JAX's functional programming model
    and equinox's immutable Module system. All operations return new AbstractField
    instances rather than modifying in place.
    """
    d      : Array
    meta   : FMT     = eqx.field(static=True)

    def __jax_array__(self):
        """Support JAX array protocol."""
        return self.d

    def __repr__(self):
        info = f"name='{self.meta.name}'" if self.meta.name else "unnamed"
        shape = f"shape={self.d.shape}" if hasattr(
            self.d, 'shape'
        ) else ""
        return f"Field({info}, {shape})"

    def __len__(self) -> int:
        """Return length of the data array."""
        return len(self.d)

    def __getitem__(self, index) -> Self:
        """Support indexing, returns a new AbstractField with indexed data."""
        return self._wrap(self.d[index])

    def _wrap(self, value: ArrayWithUnits) -> Self:
        """Wrap a value in a new AbstractField instance with same metadata."""
        if isinstance(value, pint.Quantity):
            array_value = value.to(self.meta.units).magnitude
            array_value = self.meta.backend.from_dlpack(array_value)
        else:
            array_value = self.meta.backend.from_dlpack(value      )

        return self.meta.backend.tree_at(
            lambda f: f.d,
            self,
            array_value
        )

    def _wrap_with_unit(
        self,
        value: ArrayWithUnits,
        unit: str
    ) -> Self:
        """Wrap a value in a new AbstractField instance with specified unit."""
        meta = self.meta.wrap_unit(unit)
        instance = type(self)(
            d=self.d,
            meta=meta,
        )
        return instance._wrap(value)

    def isel(self, index: int, axis: int) -> Self:
        """Index along a specified axis."""
        backend = self.meta.backend
        if backend is not None:
            return self._wrap(backend.take(self.d, indices=index, axis=axis))
        else:
            raise ValueError(f"{self.meta.name} is an empty field!")

    def reshape(self, new_shape: tuple[int, ...]) -> Self:
        """Reshape the underlying data array."""
        backend = self.meta.backend
        if backend is not None:
            return self._wrap(backend.reshape(self.d, new_shape))
        else:
            raise ValueError(f"{self.meta.name} is an empty field!")

    def zeros_like(self) -> Self:
        """Return a new AbstractField with the same shape and metadata but filled with zeros."""
        backend = self.meta.backend
        if backend is not None:
            zero_data = backend.zeros_like(self.d)
        else:
            raise ValueError(f"{self.meta.name} is an empty field!")
        return self._wrap(zero_data)
    
    @staticmethod
    @abstractmethod
    def get_meta_cls() -> Type[FMT]:
        raise NotImplementedError
    
    @classmethod
    def empty(cls) -> Self:
        """Create an empty AbstractField instance with default metadata."""
        meta = cls.get_meta_cls()()
        return cls(d=None, meta=meta)

    def is_empty(self) -> bool:
        """Check if the field is empty (i.e., has no data)."""
        if (self.meta.is_empty()):
            if self.d.size != 0:
                raise ValueError(f"Inconsistent for field{self.meta.name}: meta is empty when data is not empty!")
            return True
        else:
            if self.d.size == 0:
                raise ValueError(f"Inconsistent for field{self.meta.name}: meta is not empty when data is empty!")
            return False

    @classmethod
    def create(
        cls,
        data: ArrayWithUnits,
        meta: FMT,
    ) -> Self:
        """Create a new AbstractField instance with the same metadata."""
        backend = meta.backend
        if isinstance(data, pint.Quantity):
            array_data = data.to(meta.units).magnitude
            return cls(d=meta.backend.from_dlpack(array_data), meta=meta)
        elif backend is not None:
            return cls(d=meta.backend.from_dlpack(data      ), meta=meta)
        else:
            return cls(d=meta.backend.asarray(data), meta=meta)

    def update(
        self,
        data: ArrayWithUnits,
        unit: str = ""
    ) -> Self:
        """Update the field data and unit."""
        if unit:
            return self._wrap_with_unit(data, unit)
        else:
            return self._wrap(data)

    def unwrap(self) -> Array:
        """Unwrap and return the underlying data array."""
        return self.d

    def to(self, unit: str) -> Self:
        """Convert the field value to the specified unit and return it."""
        quantity = unit_registry.Quantity(self.d, self.meta.units)
        new_data = quantity.to(unit).magnitude
        return self.update(new_data, unit)

    def to_base_units(self) -> Self:
        """Convert the field value to base units and return it."""
        quantity = unit_registry.Quantity(self.d, self.meta.units)
        new_data = quantity.to_base_units()
        return self.update(new_data.magnitude, str(new_data.units))

    def to_reduced_units(self) -> Self:
        """Convert the field value to reduced units and return it."""
        quantity = unit_registry.Quantity(self.d, self.meta.units)
        new_data = quantity.to_reduced_units()
        return self.update(new_data.magnitude, str(new_data.units))

    def uconvert(self, usys: str = "SI") -> Self:
        """Compatibility wrapper for previous unit-system conversion calls."""
        if isinstance(usys, str) and usys.lower() == "si":
            return self.to_base_units()
        raise ValueError(
            "AbstractField.uconvert only supports 'SI' after migrating to pint; "
            "use to(unit), to_base_units(), or to_reduced_units()."
        )

    def to_si(self) -> Self:
        """Convert the field value to SI base units and return it."""
        return self.to_base_units()


class ScalarField(AbstractField[ScalarFieldMeta]):
    d   : Array           = eqx.field(default=0)
    meta: ScalarFieldMeta = eqx.field(default_factory=lambda: ScalarFieldMeta(), static=True)
     
    @staticmethod
    def get_meta_cls() -> type[ScalarFieldMeta]:
        return ScalarFieldMeta
    
    @classmethod
    def initialize(
        cls, 
        name: str,
        long_name: str,
        units: str,
        dev: AbstractDevice,
        is_static: bool = False,
        is_expand_dim0: bool = False,
        expand_dim0_len: int = 1,
        output: str = ""
    ) -> Self:
        d    = dev.backend.zeros(expand_dim0_len, device=dev(), dtype=dev.float_type)
        meta = ScalarFieldMeta(
            name=name,
            long_name=long_name,
            units=units,
            is_static=is_static,
            is_expand_dim0=is_expand_dim0,
            expand_dim0_len=expand_dim0_len,
            output=output,
            backend=dev.backend,
        )
        return cls(d=d, meta=meta)
    
    def to_dict(
        self, 
        dim0_idx: Optional[int] = None
    ) -> dict[str, Array]:
        res: dict[str, Array] = {}
        var = self.d
        if isinstance(self.meta, ScalarFieldMeta):
            if self.meta.is_expand_dim0:
                if dim0_idx is not None:
                    var = var[dim0_idx, ...]
                else:
                    raise ValueError("Must provide dim0_idx!")
        res[self.meta.name] = var
        return res


class AbstractFields(eqx.Module, ABC):

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _wrap(self, *args, **kwargs) -> Self:
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
    def to_dict(self) -> dict[str, Array]:
        raise NotImplementedError
