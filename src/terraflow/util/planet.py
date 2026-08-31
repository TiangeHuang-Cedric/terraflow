from typing import Self, Dict, Union, Optional, List
import unxt as ux

from terraflow.util import log
from terraflow.util.device import AbstractDevice, CPU


kb   : float = 1.380649e-23        # Boltzmann constant (J K-1)
na   : float = 6.02214076e23       # Avogadro constant (mol-1)
ru   : float = kb * na             # Universal gas constant (J K-1 mol-1)
amu  : float = 1.0 / (1000 * na)   # Atomic mass unit (kg) ~ 1.6605390672e-27
m_o2 : float = 32.00e-3            # O2 molar mass (kg mol-1)
m_n2 : float = 28.01e-3            # N2 molar mass (kg mol-1)
m_ar : float = 39.94e-3            # Ar molar mass (kg mol-1)
m_co2: float = 44.01e-3            # CO2 molar mass (kg mol-1)
m_h2o: float = 18.02e-3            # H2O molar mass (kg mol-1)


class GasType:
    name   : str = ""
    n_atom : int = 0
    dof    : float = 0.0  # Degrees of Freedom
    m_ratio: float = 0.0  # Mass ratio (1)
    v_ratio: float = 0.0  # Volume ratio or mole ratio (1)
    m      : float = 0.0  # Molecular mass (kg/mol)
    r      : float = 0.0  # Specific gas constant (J/(kg*K))
    cp     : float = 0.0  # Specific heat capacity at constant pressure (J/(kg*K))
    cv     : float = 0.0  # Specific heat capacity at constant volume (J/(kg*K))
    gamma  : float = 0.0  # Adiabatic index (cp / cv)
    kappa  : float = 0.0  # Poisson constant (r / cp)
    l      : float = 0.0  # Latent heat rate (J/kg)

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


class PlanetParameter:
    """Planet parameter storing physical constants.

    All fields are static because these are immutable constants.
    The numeric value is stored separately from units to avoid JAX array issues.
    """
    name       : str
    description: str
    unit       : str
    d          : float

    def __init__(
        self,
        name: str,
        description: str,
        unit: str,
        data: float | ux.AbstractQuantity
    ):
        self.name        = name
        self.description = description
        self.unit        = unit

        # Extract numeric value and store as plain float
        if isinstance(data, ux.AbstractQuantity):
            self.d = float(data.ustrip(unit))
        else:
            self.d = float(data)

    def __repr__(self):
        return (
            f"PlanetParameter(name={self.name}, "
            f"description={self.description}, "
            f"unit={self.unit}, "
            f"value={self.d})"
        )

    def _unit_convert(self, usys: ux.AbstractUnitSystem | str) -> str:
        """Convert its unit to a specified unit system."""
        quantity: ux.AbstractQuantity = ux.Quantity(1.0, self.unit)
        si_quantity = quantity.uconvert(usys)
        return str(si_quantity.unit)

    def copy(self) -> "PlanetParameter":
        """Create a copy of the parameter."""
        import copy
        return copy.deepcopy(self)

    def unwrap(self) -> float:
        """Return the raw value without unit."""
        return self.d

    def to(self, unit: str):
        """Convert the parameter value to the specified unit and return it."""
        value: ux.AbstractQuantity = ux.Quantity(self.d, self.unit)
        self.d    = float(value.ustrip(unit))
        self.unit = unit

    def uconvert(self, usys: ux.AbstractUnitSystem):
        """Decompose the parameter value and return it."""
        quantity: ux.AbstractQuantity = ux.Quantity(self.d, self.unit)
        self.d = float(quantity.ustrip(usys))
        self.unit = self._unit_convert(usys)

    def to_si(self):
        """Convert the parameter value to SI unit and return it."""
        quantity: ux.AbstractQuantity = ux.Quantity(self.d, self.unit)
        self.d = float(quantity.ustrip(ux.unitsystems.si))
        self.unit = self._unit_convert(ux.unitsystems.si)


class PlanetParameterSet:
    """Base class for planet parameter set in equations.

    All fields are static because PlanetParameter now stores only plain floats internally.
    The Quantity objects are created on-demand via properties.
    """
    planet    : str
    component : str
    parameters: Dict[str, PlanetParameter]
    default   : Dict[str, PlanetParameter]

    def __init__(
        self,
        planet: str,
        component: str,
        parameters: Dict[str, PlanetParameter]
    ):
        self.planet = planet
        self.component = component
        self.parameters = parameters

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
        planet     = parasets.get("planet")
        component  = parasets.get("component")
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
        if isinstance(planet, str) and isinstance(component, str) and isinstance(parameters, dict):
            return cls(planet=planet, component=component, parameters=parameters)
        else:
            raise ValueError(
                f"Invalid dictionary format {parasets} for PlanetParameterSet."
            )

    @classmethod
    def from_si(
        cls, 
        planet: str, 
        component: str, 
        dev: AbstractDevice = CPU()
    ) -> Self:
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
            parasets = earth_atmosphere_parameters()
        else:
            raise NotImplementedError(
                f"Component '{component}' is not implemented yet."
            )
        if dev.proc.is_root():
            log.logger.success(
                f"Initialized planet parameters for {planet} - {component} in SI units."
            )
        return cls.from_dict(parasets=parasets)

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

    def uconvert(self, usys: ux.AbstractUnitSystem):
        """Decompose all parameters to the specified unit system and return it."""
        for key, param in zip(self.keys(), self.values()):
            if not isinstance(param, str):
                self.parameters[key].uconvert(usys)

    def to_si(self):
        """Convert all parameters to SI units."""
        for key, param in zip(self.keys(), self.values()):
            if not isinstance(param, str):
                self.parameters[key].to_si()

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


def earth_atmosphere_parameters() -> Dict[str, Union[str, Dict[str, PlanetParameter]]]:
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
        data=9.80616
    )
    parameters["radius"] = PlanetParameter(
        name="radius",
        description="Radius of Earth",
        unit="m",
        data=6.37122e6
    )
    parameters["omega"] = PlanetParameter(
        name="omega",
        description="Angular Velocity of Earth's Rotation",
        unit="rad/s",
        data=7.2921e-5
    )
    parameters["rd"] = PlanetParameter(
        name="rd",
        description="Gas Constant for Dry Air",
        unit="J/(kg*K)",
        data=major_gas.r
    )
    parameters["rv"] = PlanetParameter(
        name="rv",
        description="Gas Constant for Moist Air",
        unit="J/(kg*K)",
        data=minor_gas.r
    )
    parameters["cpd"] = PlanetParameter(
        name="cpd",
        description="Specific heat capacity under constant pressure",
        unit="J/(kg*K)",
        data=major_gas.cp
    )
    parameters["cpv"] = PlanetParameter(
        name="cpv",
        description="Specific heat capacity of moist air under constant pressure",
        unit="J/(kg*K)",
        data=minor_gas.cp
    )
    parameters["cvd"] = PlanetParameter(
        name="cvd",
        description="Specific heat capacity under constant volume",
        unit="J/(kg*K)",
        data=major_gas.cv
    )
    parameters["cvv"] = PlanetParameter(
        name="cvv",
        description="Specific heat capacity of moist air under constant volume",
        unit="J/(kg*K)",
        data=minor_gas.cv
    )
    parameters["lv"] = PlanetParameter(
        name="lv",
        description="Latent heat rate of moist air",
        unit="J/kg",
        data=minor_gas.l
    )
    parameters["lapse_rate"] = PlanetParameter(
        name="lapse_rate",
        description="Lapse rate of temperature",
        unit="K/m",
        data=0.0065
    )
    parameters["rd_o_rv"] = PlanetParameter(
        name="rd_o_rv",
        description="Constant rd / rv",
        unit="1",
        data=parameters["rd"].unwrap() / parameters["rv"].unwrap()
    )
    parameters["rv_o_rd"] = PlanetParameter(
        name="rv_o_rd",
        description="Constant rv / rd",
        unit="1",
        data=parameters["rv"].unwrap() / parameters["rd"].unwrap()
    )
    parameters["p0"] = PlanetParameter(
        name="p0",
        description="Reference Pressure",
        unit="Pa",
        data=1.0e5
    )
    parameters["rd_o_g"] = PlanetParameter(
        name="rd_o_g",
        description="Constant rd / g",
        unit="kg/(m*s**2*K)",
        data=parameters["rd"].unwrap() / parameters["gravity"].unwrap()
    )
    parameters["rd_o_cpd"] = PlanetParameter(
        name="rd_o_cpd",
        description="Constant rd / cpd",
        unit="1",
        data=parameters["rd"].unwrap() / parameters["cpd"].unwrap()
    )
    parameters["cpd_o_cvd"] = PlanetParameter(
        name="cpd_o_cvd",
        description="Constant cpd / cvd",
        unit="1",
        data=parameters["cpd"].unwrap() / parameters["cvd"].unwrap()
    )
    parameters["cvd_o_cpd"] = PlanetParameter(
        name="cvd_o_cpd",
        description="Constant cvd / cpd",
        unit="1",
        data=parameters["cvd"].unwrap() / parameters["cpd"].unwrap()
    )
    parameters["pk0"] = PlanetParameter(
        name="pk0",
        description="Constant p0 ** (rd / cpd)",
        unit="Pa",
        data=parameters["p0"].unwrap() ** (parameters["rd_o_cpd"].unwrap())
    )
    return {
        "planet": consts["planet"],
        "component": consts["component"],
        "parameters": parameters
    }
