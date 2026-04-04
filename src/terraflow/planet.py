from typing import Self, Dict, Union, Optional, List
import unxt as ux
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import typing, device, log

Numeric = typing.Numeric
Array = typing.Array
ArrayWithUnits = typing.ArrayWithUnits

kb: float = 1.380649e-23        # Boltzmann constant (J K-1)
na: float = 6.02214076e23       # Avogadro constant (mol-1)
ru: float = kb * na             # Universal gas constant (J K-1 mol-1)
amu: float = 1.0 / (1000 * na)  # Atomic mass unit (kg) ~ 1.6605390672e-27
m_o2: float = 32.00e-3          # O2 molar mass (kg mol-1)
m_n2: float = 28.01e-3          # N2 molar mass (kg mol-1)
m_ar: float = 39.94e-3          # Ar molar mass (kg mol-1)
m_co2: float = 44.01e-3         # CO2 molar mass (kg mol-1)
m_h2o: float = 18.02e-3         # H2O molar mass (kg mol-1)


class GasType:
    name: str = ""
    n_atom: int = 0
    dof: float = 0.0  # Degrees of Freedom
    m_ratio: float = 0.0  # Mass ratio (1)
    v_ratio: float = 0.0  # Volume ratio or mole ratio (1)
    m: float = 0.0  # Molecular mass (kg/mol)
    r: float = 0.0  # Specific gas constant (J/(kg*K))
    cp: float = 0.0  # Specific heat capacity at constant pressure (J/(kg*K))
    cv: float = 0.0  # Specific heat capacity at constant volume (J/(kg*K))
    gamma: float = 0.0  # Adiabatic index (cp / cv)
    kappa: float = 0.0  # Poisson constant (r / cp)
    l: float = 0.0  # Latent heat rate (J/kg)

    def __init__(
        self,
        name: str,
        n_atom: int,
        m: float,
        v_ratio: Optional[float] = None,
        dof: Optional[float] = None
    ):
        self.name = name
        self.n_atom = n_atom
        self.m = m
        self.r = ru / m
        if dof is not None:
            dof_ = dof
        else:
            if n_atom == 1:
                dof_ = 3.0
            elif n_atom == 2:
                dof_ = 5.0
            else:
                raise ValueError(
                    f"Cannot infer degrees of freedom for {name} with n_atom={n_atom}."
                )
        self.cv = dof_ * 0.5 * self.r
        self.cp = self.cv + self.r
        if v_ratio is not None:
            self.v_ratio = v_ratio

        self.gamma = self.cp / self.cv
        self.kappa = self.r / self.cp

    @classmethod
    def from_gases(
        cls,
        name: str,
        gases: List[Self]
    ) -> Self:
        self = cls.__new__(cls)
        # Calculate mean molecular mass
        sum_ratio = sum(g.v_ratio for g in gases)
        self.m = sum(g.v_ratio * g.m for g in gases) / sum_ratio
        self.r = ru / self.m

        # Calculate mass mixing ratio from volume mixing ratio
        for g in gases:
            g.m_ratio = (g.v_ratio * g.m) / self.m

        # Calculate cp and cv
        sum_ratio = sum(g.m_ratio for g in gases)
        self.cv = sum(g.m_ratio * g.cv for g in gases) / sum_ratio
        self.cp = self.cv + self.r
        self.gamma = self.cp / self.cv
        self.kappa = self.r / self.cp
        return self


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
        data: Numeric | ux.AbstractQuantity,
        create_with_unit: bool,
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "unit", unit)

        # Extract numeric value and store as plain float
        if create_with_unit:
            if isinstance(data, ux.AbstractQuantity):
                object.__setattr__(self, "data", data.to(unit))
            else:
                value_arr = jnp.asarray(data)
                object.__setattr__(self, "data", ux.Quantity(
                    value_arr, unit))  # type: ignore
        else:
            if isinstance(data, ux.AbstractQuantity):
                object.__setattr__(self, "data", jnp.asarray(data.value))
            else:
                object.__setattr__(self, "data", jnp.asarray(data))

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

    def update(
        self,
        data: ArrayWithUnits,
        unit: Optional[str] = ""
    ) -> "PlanetParameter":
        if unit:
            return self._wrap_with_unit(data, unit)
        else:
            return self._wrap(data)

    def _wrap(self, data: ArrayWithUnits) -> "PlanetParameter":
        """Wrap the raw value with unit if needed."""
        if isinstance(self.data, ux.AbstractQuantity):
            create_with_unit = True
            if isinstance(data, ux.AbstractQuantity):
                data = data.to(self.unit)
            elif isinstance(data, Array):
                data = ux.Quantity(data, self.unit)  # type: ignore
            else:
                raise ValueError(
                    f"Data should be an array or quantity, but got {type(data)}."
                )
        elif isinstance(self.data, Array):
            create_with_unit = False
            if isinstance(data, ux.AbstractQuantity):
                data = jnp.asarray(data.value)
            elif isinstance(data, Array):
                data = jnp.asarray(data)
            else:
                raise ValueError(
                    f"Data should be an array or quantity, but got {type(data)}."
                )
        else:
            raise ValueError(
                f"Data should be an array or quantity, but got {type(self.data)}."
            )
        return PlanetParameter(
            name=self.name,
            description=self.description,
            unit=self.unit,
            data=data,
            create_with_unit=create_with_unit
        )

    def _wrap_with_unit(self, data: ArrayWithUnits, unit: str) -> "PlanetParameter":
        """Wrap the raw value with unit."""
        if isinstance(self.data, ux.AbstractQuantity):
            create_with_unit = True
            if isinstance(data, ux.AbstractQuantity):
                data = data.to(unit)
            elif isinstance(data, Array):
                data = ux.Quantity(data, unit)  # type: ignore
            else:
                raise ValueError(
                    f"Data should be an array or quantity, but got {type(data)}."
                )
        elif isinstance(self.data, Array):
            create_with_unit = False
            if isinstance(data, ux.AbstractQuantity):
                data = jnp.asarray(data.value)
            elif isinstance(data, Array):
                data = jnp.asarray(data)
            else:
                raise ValueError(
                    f"Data should be an array or quantity, but got {type(data)}."
                )
        else:
            raise ValueError(
                f"Data should be an array or quantity, but got {type(self.data)}."
            )
        return PlanetParameter(
            name=self.name,
            description=self.description,
            unit=unit,
            data=data,
            create_with_unit=create_with_unit
        )

    def unwrap(self) -> Array:
        """Return the raw value without unit."""
        if isinstance(self.data, ux.AbstractQuantity):
            return jnp.asarray(self.data.value)
        return self.data

    def to(self, unit: str) -> "PlanetParameter":
        """Convert the parameter value to the specified unit and return it."""
        if isinstance(self.data, ux.AbstractQuantity):
            data = self.data.to(unit)
        elif isinstance(self.data, Array):
            value = ux.Quantity(self.data, self.unit)  # type: ignore
            data = value.to(unit).value
        else:
            raise ValueError(
                f"Data should be an array or quantity, but got {type(self.data)}."
            )
        return self.update(data=data, unit=unit)

    def decompose(self, usys: ux.AbstractUnitSystem) -> "PlanetParameter":
        """Decompose the parameter value and return it."""
        if isinstance(self.data, ux.AbstractQuantity):
            data = self.data.decompose(usys)  # type: ignore
        elif isinstance(self.data, Array):
            data = ux.Quantity(
                self.data, self.unit  # type: ignore
            ).decompose(usys).value  # type: ignore
        else:
            raise ValueError(
                f"Data should be an array or quantity, but got {type(self.data)}."
            )
        unit = self._unit_convert(usys)
        return self.update(data=data, unit=unit)

    def to_si(self) -> "PlanetParameter":
        """Convert the parameter value to SI unit and return it."""
        if isinstance(self.data, ux.AbstractQuantity):
            data = self.data.decompose(ux.unitsystems.si)  # type: ignore
        elif isinstance(self.data, Array):
            data = ux.Quantity(
                self.data, self.unit  # type: ignore
            ).decompose(ux.unitsystems.si).value  # type: ignore
        else:
            raise ValueError(
                f"Data should be an array or quantity, but got {type(self.data)}."
            )
        unit = self._unit_convert(ux.unitsystems.si)
        return self.update(data=data, unit=unit)

    def to_value(self) -> Self:
        """Convert the parameter value to its raw float representation."""
        if isinstance(self.data, ux.AbstractQuantity):
            data = jnp.asarray(self.data.value)
        elif isinstance(self.data, Array):
            data = self.data
        else:
            raise ValueError(
                f"Data should be an array or quantity, but got {type(self.data)}."
            )
        return eqx.tree_at(
            lambda x: x.data,
            self,
            data
        )

    def to_quantity(self) -> Self:
        """Convert the parameter value to a Quantity object."""
        if isinstance(self.data, Array):
            data = ux.Quantity(
                self.data, self.unit  # type: ignore
            )
        elif isinstance(self.data, ux.AbstractQuantity):
            data = self.data
        else:
            raise ValueError(
                f"Data should be an array or quantity, but got {type(self.data)}."
            )
        return eqx.tree_at(
            lambda x: x.data,
            self,
            data
        )


class PlanetParameterSet(eqx.Module):
    """Base class for planet parameter set in equations.

    All fields are static because PlanetParameter now stores only plain floats internally.
    The Quantity objects are created on-demand via properties.
    """
    planet: str = eqx.field(static=True)
    component: str = eqx.field(static=True)
    parameters: Dict[str, PlanetParameter]
    default: Dict[str, PlanetParameter]
    dev: device.AbstractDevice = eqx.field(static=True)

    def __init__(
        self,
        planet: str,
        component: str,
        parameters: Dict[str, PlanetParameter],
        dev: device.AbstractDevice,
        default: Dict[str, PlanetParameter] = {},
    ):
        object.__setattr__(self, "planet", planet)
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "dev", dev)
        if default:
            object.__setattr__(self, "default", default)
        else:
            import copy
            object.__setattr__(self, "default", copy.deepcopy(parameters))

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
    def from_dict(cls, parasets: Dict[str, Union[str, Dict[str, PlanetParameter]]], dev: device.AbstractDevice) -> Self:
        planet = parasets.get("planet")
        component = parasets.get("component")
        parameters = parasets.get("parameters")
        if planet is None:
            raise ValueError(
                f"Planet name is missing in the input: {parasets}"
            )
        if component is None:
            raise ValueError(
                f"Component name is missing in the input: {parasets}"
            )
        if parameters is None:
            raise ValueError(
                f"Parameters dictionary is missing in the input: {parasets}"
            )
        if isinstance(parameters, str):
            raise ValueError(
                f"Parameters should be a dictionary of PlanetParameter, but got string: {parameters}"
            )
        for k, v in parameters.items():
            if not isinstance(v, PlanetParameter):
                raise ValueError(
                    f"All parameters should be of type PlanetParameter, but got {type(v)} in {parameters}"
                )
            if isinstance(v.data, Array):
                data = v.data
            elif isinstance(v.data, ux.AbstractQuantity):
                data = v.data.value
            else:
                raise ValueError(
                    f"Parameter value should be a numeric type or array, but got {type(v.data)} in {parameters}"
                )
            if isinstance(data, (jnp.ndarray, float)):
                data = jnp.asarray(data, dtype=dev.float_type)
            elif isinstance(data, (jnp.ndarray, int)):
                data = jnp.asarray(data, dtype=dev.int_type)
            else:
                raise ValueError(
                    f"Parameter value should be a numeric type or array, but got {type(v.data)} in {parameters}"
                )
            parameters[k] = v.update(data)
        if isinstance(planet, str) and isinstance(component, str) and isinstance(parameters, dict):
            return cls(planet=planet, component=component, parameters=parameters, dev=dev)
        else:
            raise ValueError(
                f"Invalid dictionary format {parasets} for PlanetParameterSet."
            )

    @classmethod
    def from_si(cls, planet: str, component: str, dev: device.AbstractDevice, create_with_unit: bool = False) -> Self:
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
            parasets = earth_atmosphere_parameters(
                dev=dev, create_with_unit=create_with_unit
            )
        else:
            raise NotImplementedError(
                f"Component '{component}' is not implemented yet."
            )
        log.logger.success(
            f"Initialized planet parameters for {planet} - {component} in SI units."
        )
        return cls.from_dict(parasets=parasets, dev=dev)

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

    def update(self, parameters: Dict[str, PlanetParameter]) -> Self:
        """Update the parameter set with new parameters."""
        return eqx.tree_at(
            lambda x: x.parameters,
            self,
            parameters
        )

    def decompose(self, usys: ux.AbstractUnitSystem) -> Self:
        """Decompose all parameters to the specified unit system and return it."""
        parameters = {}
        for key, param in zip(self.keys(), self.values()):
            if not isinstance(param, str):
                parameters[key] = param.decompose(usys)
        return self.update(parameters=parameters)

    def to_si(self) -> Self:
        """Convert all parameters to SI units."""
        parameters = {}
        for key, param in zip(self.keys(), self.values()):
            if not isinstance(param, str):
                parameters[key] = param.to_si()
        return self.update(parameters=parameters)

    def to_values(self) -> Self:
        """Convert all parameters to its raw float representation."""
        parameters = {}
        for key, param in zip(self.keys(), self.values()):
            if not isinstance(param, str):
                parameters[key] = param.to_value()
        return self.update(parameters=parameters)

    def to_quantities(self) -> Self:
        """Convert all parameters to Quantity objects."""
        parameters = {}
        for key, param in zip(self.keys(), self.values()):
            if not isinstance(param, str):
                parameters[key] = param.to_quantity()
        return self.update(parameters=parameters)

    def to_default(self) -> Self:
        """Convert all parameters to the default unit system."""
        default = self.default
        parameters = {}
        for key, param in zip(self.keys(), self.values()):
            if not isinstance(param, str):
                default_param = default[key]
                if isinstance(default_param, PlanetParameter):
                    default_unit = default_param.unit
                    parameters[key] = param.to(default_unit)
        return self.update(parameters=parameters)

    @property
    def gravity(self) -> PlanetParameter:
        return self["gravity"]

    @property
    def radius(self) -> PlanetParameter:
        return self["radius"]

    @property
    def omega(self) -> PlanetParameter:
        return self["omega"]

    @property
    def rd(self) -> PlanetParameter:
        return self["rd"]

    @property
    def rv(self) -> PlanetParameter:
        return self["rv"]

    @property
    def cpd(self) -> PlanetParameter:
        return self["cpd"]

    @property
    def cpv(self) -> PlanetParameter:
        return self["cpv"]

    @property
    def cvd(self) -> PlanetParameter:
        return self["cvd"]

    @property
    def cvv(self) -> PlanetParameter:
        return self["cvv"]

    @property
    def lv(self) -> PlanetParameter:
        return self["lv"]

    @property
    def lapse_rate(self) -> PlanetParameter:
        return self["lapse_rate"]

    @property
    def rd_o_rv(self) -> PlanetParameter:
        return self["rd_o_rv"]

    @property
    def rv_o_rd(self) -> PlanetParameter:
        return self["rv_o_rd"]

    @property
    def p0(self) -> PlanetParameter:
        return self["p0"]

    @property
    def rd_o_g(self) -> PlanetParameter:
        return self["rd_o_g"]

    @property
    def rd_o_cpd(self) -> PlanetParameter:
        return self["rd_o_cpd"]

    @property
    def cpd_o_cvd(self) -> PlanetParameter:
        return self["cpd_o_cvd"]

    @property
    def cvd_o_cpd(self) -> PlanetParameter:
        return self["cvd_o_cpd"]

    @property
    def pk0(self) -> PlanetParameter:
        return self["pk0"]


def earth_atmosphere_parameters(
    dev: device.AbstractDevice,
    create_with_unit: bool
) -> Dict[str, Union[str, Dict[str, PlanetParameter]]]:
    gases_list = [
        GasType(
            name="n2", n_atom=2, m=m_n2, v_ratio=0.7808, dof=5.00607939581368
        ),
        GasType(
            name="o2", n_atom=2, m=m_o2, v_ratio=0.2095, dof=5.07260684628201
        ),
        GasType(
            name="ar", n_atom=1, m=m_ar, v_ratio=0.0093, dof=3.02464704198518
        ),
        GasType(
            name="co2", n_atom=3, m=m_co2, v_ratio=0.000385, dof=6.93407411251379
        ),
        GasType(
            name="h2o", n_atom=3, m=m_h2o, v_ratio=999.0, dof=6.0
        )
    ]
    major_gas = GasType.from_gases(name="dry_air", gases=gases_list[:-1])
    minor_gas = GasType.from_gases(name="water_vapor", gases=[gases_list[-1]])

    # Testing fixed constants
    major_gas.r = 287.0
    major_gas.cp = 1004.5
    major_gas.cv = 717.0
    major_gas.gamma = major_gas.cp / major_gas.cv
    major_gas.kappa = major_gas.r / major_gas.cp
    minor_gas.r = 461.5
    minor_gas.cp = 1810.0
    minor_gas.cv = 1348.5
    minor_gas.gamma = minor_gas.cp / minor_gas.cv
    minor_gas.kappa = minor_gas.r / minor_gas.cp
    minor_gas.l = 2.5e6

    consts: Dict[str, Union[str, Dict[str, PlanetParameter]]] = {}
    consts["planet"] = "earth"
    consts["component"] = "atm"
    parameters: Dict[str, PlanetParameter] = {}
    parameters["gravity"] = PlanetParameter(
        name="gravity",
        description="Gravity Acceleration of Earth",
        unit="m/s**2",
        data=jnp.array(9.80616, dtype=dev.float_type),
        create_with_unit=create_with_unit
    )
    parameters["radius"] = PlanetParameter(
        name="radius",
        description="Radius of Earth",
        unit="m",
        data=jnp.array(6.37122e6, dtype=dev.float_type),
        create_with_unit=create_with_unit
    )
    parameters["omega"] = PlanetParameter(
        name="omega",
        description="Angular Velocity of Earth's Rotation",
        unit="rad/s",
        data=jnp.array(7.2921e-5, dtype=dev.float_type),
        create_with_unit=create_with_unit
    )
    parameters["rd"] = PlanetParameter(
        name="rd",
        description="Gas Constant for Dry Air",
        unit="J/(kg*K)",
        data=jnp.array(major_gas.r, dtype=dev.float_type),
        create_with_unit=create_with_unit
    )
    parameters["rv"] = PlanetParameter(
        name="rv",
        description="Gas Constant for Moist Air",
        unit="J/(kg*K)",
        data=jnp.array(minor_gas.r, dtype=dev.float_type),
        create_with_unit=create_with_unit
    )
    parameters["cpd"] = PlanetParameter(
        name="cpd",
        description="Specific heat capacity under constant pressure",
        unit="J/(kg*K)",
        data=jnp.array(major_gas.cp, dtype=dev.float_type),
        create_with_unit=create_with_unit
    )
    parameters["cpv"] = PlanetParameter(
        name="cpv",
        description="Specific heat capacity of moist air under constant pressure",
        unit="J/(kg*K)",
        data=jnp.array(minor_gas.cp, dtype=dev.float_type),
        create_with_unit=create_with_unit
    )
    parameters["cvd"] = PlanetParameter(
        name="cvd",
        description="Specific heat capacity under constant volume",
        unit="J/(kg*K)",
        data=jnp.array(major_gas.cv, dtype=dev.float_type),
        create_with_unit=create_with_unit
    )
    parameters["cvv"] = PlanetParameter(
        name="cvv",
        description="Specific heat capacity of moist air under constant volume",
        unit="J/(kg*K)",
        data=jnp.array(minor_gas.cv, dtype=dev.float_type),
        create_with_unit=create_with_unit
    )
    parameters["lv"] = PlanetParameter(
        name="lv",
        description="Latent heat rate of moist air",
        unit="J/kg",
        data=jnp.array(minor_gas.l, dtype=dev.float_type),
        create_with_unit=create_with_unit
    )
    parameters["lapse_rate"] = PlanetParameter(
        name="lapse_rate",
        description="Lapse rate of temperature",
        unit="K/m",
        data=jnp.array(0.0065, dtype=dev.float_type),
        create_with_unit=create_with_unit
    )
    parameters["rd_o_rv"] = PlanetParameter(
        name="rd_o_rv",
        description="Constant rd / rv",
        unit="1",
        data=parameters["rd"].unwrap() / parameters["rv"].unwrap(),
        create_with_unit=create_with_unit
    )
    parameters["rv_o_rd"] = PlanetParameter(
        name="rv_o_rd",
        description="Constant rv / rd",
        unit="1",
        data=parameters["rv"].unwrap() / parameters["rd"].unwrap(),
        create_with_unit=create_with_unit
    )
    parameters["p0"] = PlanetParameter(
        name="p0",
        description="Reference Pressure",
        unit="Pa",
        data=jnp.array(1.0e5, dtype=dev.float_type),
        create_with_unit=create_with_unit
    )
    parameters["rd_o_g"] = PlanetParameter(
        name="rd_o_g",
        description="Constant rd / g",
        unit="kg/(m*s**2*K)",
        data=parameters["rd"].unwrap() / parameters["gravity"].unwrap(),
        create_with_unit=create_with_unit
    )
    parameters["rd_o_cpd"] = PlanetParameter(
        name="rd_o_cpd",
        description="Constant rd / cpd",
        unit="1",
        data=parameters["rd"].unwrap() / parameters["cpd"].unwrap(),
        create_with_unit=create_with_unit
    )
    parameters["cpd_o_cvd"] = PlanetParameter(
        name="cpd_o_cvd",
        description="Constant cpd / cvd",
        unit="1",
        data=parameters["cpd"].unwrap() / parameters["cvd"].unwrap(),
        create_with_unit=create_with_unit
    )
    parameters["cvd_o_cpd"] = PlanetParameter(
        name="cvd_o_cpd",
        description="Constant cvd / cpd",
        unit="1",
        data=parameters["cvd"].unwrap() / parameters["cpd"].unwrap(),
        create_with_unit=create_with_unit
    )
    parameters["pk0"] = PlanetParameter(
        name="pk0",
        description="Constant p0 ** (rd / cpd)",
        unit="Pa",
        data=parameters["p0"].unwrap() ** (parameters["rd_o_cpd"].unwrap()),
        create_with_unit=create_with_unit
    )
    return {
        "planet": consts["planet"],
        "component": consts["component"],
        "parameters": parameters
    }
