from typing import Self, Dict, Union
import unxt as ux
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import typing, device

Numeric = typing.Numeric
Array = typing.Array
ArrayWithUnits = typing.ArrayWithUnits


class PlanetParameter(eqx.Module):
    """Planet parameter storing physical constants.

    All fields are static because these are immutable constants.
    The numeric value is stored separately from units to avoid JAX array issues.
    """
    name: str = eqx.field(static=True)
    description: str = eqx.field(static=True)
    unit: str = eqx.field(static=True)
    data: ArrayWithUnits

    def __init__(
        self,
        name: str,
        description: str,
        unit: str,
        value: Numeric | ux.AbstractQuantity,
        create_with_unit: bool = True,
    ):
        self.name = name
        self.description = description
        self.unit = unit

        # Extract numeric value and store as plain float
        if create_with_unit:
            if isinstance(value, ux.Quantity):
                self.data = value.to(unit)
            else:
                value_arr = jnp.asarray(value)
                self.data = ux.Quantity(value_arr, unit)  # type: ignore
        else:
            if isinstance(value, ux.Quantity):
                self.data = value.value
            else:
                self.data = jnp.asarray(value)

    def __repr__(self):
        return (
            f"PlanetParameter(name={self.name}, "
            f"description={self.description}, "
            f"unit={self.unit}, "
            f"value={self.data})"
        )

    def _unit_convert(self, usys: ux.AbstractUnitSystem | str) -> str:
        """Convert its unit to a specified unit system."""
        quantity = ux.Quantity(1.0, self.unit)  # type: ignore
        si_quantity = quantity.decompose(usys)  # type: ignore
        return str(si_quantity.unit)

    def copy(self) -> "PlanetParameter":
        """Create a copy of the parameter."""
        import copy
        return copy.deepcopy(self)

    def unwrap(self) -> Array:
        """Return the raw value without unit."""
        if isinstance(self.data, ux.AbstractQuantity):
            return self.data.value
        return self.data

    def to(self, unit: str) -> Self:
        """Convert the parameter value to the specified unit and return it."""
        if isinstance(self.data, ux.AbstractQuantity):
            self.data = self.data.to(unit)
        else:
            value = ux.Quantity(self.data, self.unit)  # type: ignore
            self.data = value.to(unit).value
        self.unit = unit
        return self

    def decompose(self, usys: ux.AbstractUnitSystem) -> Self:
        """Decompose the parameter value and return it."""
        if isinstance(self.data, ux.AbstractQuantity):
            self.data = self.data.decompose(usys)  # type: ignore
        else:
            self.data = ux.Quantity(
                self.data, self.unit  # type: ignore
            ).decompose(usys).value  # type: ignore
        self.unit = self._unit_convert(usys)
        return self

    def to_unit(self) -> Self:
        """Convert the parameter value to its internal unit and return it."""
        if not isinstance(self.data, ux.AbstractQuantity):
            self.data = ux.Quantity(
                self.data, self.unit  # type: ignore
            )
        return self

    def to_si(self) -> Self:
        """Convert the parameter value to SI unit and return it."""
        if isinstance(self.data, ux.AbstractQuantity):
            self.data = self.data.decompose(ux.unitsystems.si)  # type: ignore
        else:
            self.data = ux.Quantity(
                self.data, self.unit  # type: ignore
            ).decompose(ux.unitsystems.si).value  # type: ignore
        self.unit = self._unit_convert(ux.unitsystems.si)
        return self

    def to_value(self) -> Self:
        """Convert the parameter value to its raw float representation."""
        if isinstance(self.data, ux.AbstractQuantity):
            self.data = self.data.value
        return self

    def to_quantity(self) -> Self:
        """Convert the parameter value to a Quantity object."""
        if not isinstance(self.data, ux.AbstractQuantity):
            self.data = ux.Quantity(
                self.data, self.unit  # type: ignore
            )
        return self


class PlanetParameterSet(eqx.Module):
    """Base class for planet parameter set in equations.

    All fields are static because PlanetParameter now stores only plain floats internally.
    The Quantity objects are created on-demand via properties.
    """
    planet: str = eqx.field(static=True)
    component: str = eqx.field(static=True)
    parameters: Dict[str, PlanetParameter]
    default: Dict[str, PlanetParameter]

    def __init__(
        self,
        planet: str,
        component: str,
        parameters: Dict[str, PlanetParameter] = {},
        default: Dict[str, PlanetParameter] = {}
    ):
        self.planet = planet
        self.component = component
        self.parameters = parameters
        if default:
            self.default = default
        else:
            import copy
            self.default = copy.deepcopy(parameters)

    def __getitem__(self, name: str) -> PlanetParameter:
        param = self.parameters.get(name)
        if param is None:
            raise KeyError(
                f"Parameter '{name}' not found in the parameter set."
            )
        return param

    def __iter__(self):
        return iter(self.parameters)

    def __len__(self):
        return len(self.parameters)

    @classmethod
    def from_dict(cls, parasets: Dict[str, Union[str, Dict[str, PlanetParameter]]]) -> Self:
        planet = parasets.get("planet")
        component = parasets.get("component")
        parameters = parasets.get("parameters")
        if isinstance(planet, str) and isinstance(component, str) and isinstance(parameters, dict):
            return cls(planet, component, parameters)
        else:
            raise ValueError(
                f"Invalid dictionary format {parasets} for PlanetParameterSet."
            )

    @classmethod
    def from_si(cls, planet: str, component: str, dev: device.AbstractDevice) -> Self:
        """Create a PlanetParameterSet for Earth with specified component."""
        if planet not in ["earth"]:
            raise ValueError(
                f"Planet '{planet}' is not supported."
            )
        if component not in ["atmosphere", "ocean"]:
            raise ValueError(
                f"Component '{component}' is not supported for Earth."
            )
        if planet == "earth" and component == "atmosphere":
            parasets = earth_atmosphere_parameters(dev)
        else:
            raise NotImplementedError(
                f"Component '{component}' is not implemented yet."
            )
        return cls.from_dict(parasets)

    def parameter_names(self) -> list[str]:
        return list(self.parameters.keys())

    def keys(self) -> list[str]:
        return list(self.parameters.keys())

    def values(self) -> list[PlanetParameter]:
        return list(self.parameters.values())

    def items(self) -> list[tuple[str, PlanetParameter]]:
        return list(self.parameters.items())

    def get(self, name: str, default=None):
        return self.parameters.get(name, default)

    def add(self, param: PlanetParameter) -> Self:
        """Add a new parameter to the set."""
        name = getattr(param, "name", None)
        if not name:
            raise Exception("Parameter must have a name.")
        self.parameters[name] = param
        return self

    def copy(self) -> Self:
        """Create a copy of the parameter set."""
        import copy
        return copy.deepcopy(self)

    def decompose(self, usys: ux.AbstractUnitSystem) -> Self:
        """Decompose all parameters to the specified unit system and return it."""
        for key, param in zip(self.keys(), self.values()):
            if not isinstance(param, str):
                self.parameters[key] = param.decompose(usys)
        return self

    def to_unit(self) -> Self:
        """Convert all parameters to its unit system."""
        for key, param in zip(self.keys(), self.values()):
            if not isinstance(param, str):
                self.parameters[key] = param.to_unit()
        return self

    def to_si(self) -> Self:
        """Convert all parameters to SI units."""
        for key, param in zip(self.keys(), self.values()):
            if not isinstance(param, str):
                self.parameters[key] = param.to_si()
        return self

    def to_values(self) -> Self:
        """Convert all parameters to its raw float representation."""
        for key, param in zip(self.keys(), self.values()):
            if not isinstance(param, str):
                self.parameters[key] = param.to_value()
        return self

    def to_quantities(self) -> Self:
        """Convert all parameters to Quantity objects."""
        for key, param in zip(self.keys(), self.values()):
            if not isinstance(param, str):
                self.parameters[key] = param.to_quantity()
        return self

    def to_default(self) -> Self:
        """Convert all parameters to the default unit system."""
        default = self.default
        for key, param in zip(self.keys(), self.values()):
            if not isinstance(param, str):
                default_param = default[key]
                if isinstance(default_param, PlanetParameter):
                    default_unit = default_param.unit
                    self.parameters[key] = param.to(default_unit)
        return self

    @property
    def gravity(self) -> PlanetParameter:
        return self["gravity"]

    @property
    def radius(self) -> PlanetParameter:
        return self["radius"]

    @property
    def omega(self) -> PlanetParameter:
        return self["omega"]


def earth_atmosphere_parameters(dev: device.AbstractDevice) -> Dict[str, Union[str, Dict[str, PlanetParameter]]]:
    return {
        "planet": "earth",
        "component": "atm",
        "parameters": {
            "gravity": PlanetParameter(
                name="gravity",
                description="Gravity Acceleration of Earth",
                unit="m/s**2",
                value=jnp.array(9.80616, dtype=dev.float_type)
            ),
            "radius": PlanetParameter(
                name="radius",
                description="Radius of Earth",
                unit="m",
                value=jnp.array(6.37122e6, dtype=dev.float_type)
            ),
            "omega": PlanetParameter(
                name="omega",
                description="Angular Velocity of Earth's Rotation",
                unit="rad/s",
                value=jnp.array(7.2921e-5, dtype=dev.float_type)
            )
        }
    }
