from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import (
    Callable,
    Generic,
    Mapping,
    TypeVar,
    Type,
    overload,
    Optional,
    Union,
    List,
    Dict,
    Self,
    cast,
    Literal,
    Any,
    get_type_hints,
    get_origin,
    get_args,
    TYPE_CHECKING
)
from dataclasses import dataclass
import datetime
import equinox as eqx
import types
import unxt as ux
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
import uxarray as uxr

from terraflow.util import typing, device, model_config, uxarray as tuxr
import terraflow.operator as op
from terraflow.mesh.abstract import AbstractMesh
from terraflow.vert_coord.abstract import AbstractVertCoordinate
from terraflow.layout.abstract import CoordinateSystem, AbstractLayout
from terraflow.layout.field import (
    AbstractMetaData,
    NumericMetaData,
    StaggeredFieldMetaData,
    Field,
    ArrayField,
    FieldLike
)
from terraflow.planet import PlanetParameterSet
from terraflow.solver.abstract import AbstractODE
from terraflow.manager.registry import navigate

if TYPE_CHECKING:
    from terraflow.prepare.readers.abstract import AbstractReader
    from terraflow.test_case.abstract import AbstractTestCase
    from terraflow.solver.abstract import AbstractSolver


Int = typing.Int
Float = typing.Float
Numeric = typing.Numeric
Array = typing.Array

HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
HLO = TypeVar("HLO", bound=AbstractLayout)
VLO = TypeVar("VLO", bound=AbstractLayout)
DM = TypeVar("DM", bound=op.AbstractDiscretizationMethod)
PSO = TypeVar("PSO", bound="AbstractPrepareSpaceOperators")
SOP = TypeVar("SOP", bound="AbstractSpaceOperators")
ES = TypeVar("ES", bound="AbstractEquationSet")
EQU = TypeVar("EQU", bound="AbstractEquationForm")
FMT = TypeVar("FMT", bound=AbstractMetaData)
STATE = TypeVar("STATE", bound="AbstractFields")
STATIC = TypeVar("STATIC", bound="AbstractFields")
TEND = TypeVar("TEND", bound="AbstractFields")
AUX = TypeVar("AUX", bound="AbstractFields")
ADVBATCH = TypeVar("ADVBATCH", bound="AbstractAdvBatchType")
TRACERS = TypeVar("TRACERS", bound="AbstractFields")
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
        coord: CoordinateSystem[HM, VM, HLO, VLO],
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
            layout=coord.hori_layout.alias_name(),
            horizontal_mesh=coord.mesh.alias_name(),
            vertical_coordinate=coord.vert.alias_name()
        )


@dataclass(frozen=True)
class EquationMode(ABC):
    baroclinic: bool
    hydrostatic: bool
    nonhydrostatic: bool
    ideal_dry_core: bool
    advection2d: bool
    advection3d: bool
    output_h1_tend: bool
    state_tile_depth: int

    def __init__(
        self,
        *,
        baroclinic: bool = False,
        hydrostatic: bool = False,
        nonhydrostatic: bool = False,
        ideal_dry_core: bool = False,
        advection2d: bool = False,
        advection3d: bool = False,
        **kwargs,
    ):
        object.__setattr__(
            self, "baroclinic", baroclinic
        )
        object.__setattr__(
            self, "hydrostatic", hydrostatic
        )
        object.__setattr__(
            self, "nonhydrostatic", nonhydrostatic
        )
        object.__setattr__(
            self, "ideal_dry_core", ideal_dry_core
        )
        object.__setattr__(
            self, "advection2d", advection2d
        )
        object.__setattr__(
            self, "advection3d", advection3d
        )
        object.__setattr__(
            self, "output_h1_tend",
            kwargs.get("output_h1_tend", False)
        )
        object.__setattr__(
            self, "state_tile_depth",
            kwargs.get("state_tile_depth", 3)
        )

    @classmethod
    def from_swm(cls, **kwargs):
        return cls(
            baroclinic=False,
            hydrostatic=False,
            nonhydrostatic=False,
            ideal_dry_core=False,
            advection2d=False,
            advection3d=False,
            **kwargs
        )

    @classmethod
    def from_hdc(cls, ideal_dry_core: bool, **kwargs):
        return cls(
            baroclinic=True,
            hydrostatic=True,
            nonhydrostatic=False,
            ideal_dry_core=ideal_dry_core,
            advection2d=False,
            advection3d=False,
            **kwargs
        )

    @classmethod
    def from_adv2d(cls, **kwargs):
        return cls(
            baroclinic=False,
            hydrostatic=False,
            nonhydrostatic=False,
            ideal_dry_core=False,
            advection2d=True,
            advection3d=False,
            **kwargs
        )

    @classmethod
    def from_adv3d(cls, **kwargs):
        return cls(
            baroclinic=False,
            hydrostatic=False,
            nonhydrostatic=False,
            ideal_dry_core=False,
            advection2d=False,
            advection3d=True,
            **kwargs
        )

    def is_barotropic(self) -> bool:
        return (
            not self.baroclinic
            and not self.hydrostatic
            and not self.nonhydrostatic
            and not self.ideal_dry_core
            and not self.advection2d
            and not self.advection3d
        )

    def is_baroclinic(self) -> bool:
        return self.baroclinic

    def is_hydrostatic(self) -> bool:
        return self.is_dry_hydrostatic() or self.is_moist_hydrostatic()

    def is_nonhydrostatic(self) -> bool:
        return self.is_dry_nonhydrostatic() or self.is_moist_nonhydrostatic()

    def is_dry(self) -> bool:
        return self.is_dry_hydrostatic() or self.is_dry_nonhydrostatic()

    def is_moist(self) -> bool:
        return self.is_moist_hydrostatic() or self.is_moist_nonhydrostatic()

    def is_dry_hydrostatic(self) -> bool:
        return (
            self.baroclinic
            and self.hydrostatic
            and not self.nonhydrostatic
            and self.ideal_dry_core
            and not self.advection2d
            and not self.advection3d
        )

    def is_dry_nonhydrostatic(self) -> bool:
        return (
            self.baroclinic
            and not self.hydrostatic
            and self.nonhydrostatic
            and self.ideal_dry_core
            and not self.advection2d
            and not self.advection3d
        )

    def is_moist_hydrostatic(self) -> bool:
        return (
            self.baroclinic
            and self.hydrostatic
            and not self.nonhydrostatic
            and not self.ideal_dry_core
            and not self.advection2d
            and not self.advection3d
        )

    def is_moist_nonhydrostatic(self) -> bool:
        return (
            self.baroclinic
            and not self.hydrostatic
            and self.nonhydrostatic
            and not self.ideal_dry_core
            and not self.advection2d
            and not self.advection3d
        )

    def is_advection(self) -> bool:
        return self.advection2d or self.advection3d

    def is_advection2d(self) -> bool:
        return (
            not self.baroclinic
            and not self.hydrostatic
            and not self.nonhydrostatic
            and not self.ideal_dry_core
            and self.advection2d
            and not self.advection3d
        )

    def is_advection3d(self) -> bool:
        return (
            not self.baroclinic
            and not self.hydrostatic
            and not self.nonhydrostatic
            and not self.ideal_dry_core
            and not self.advection2d
            and self.advection3d
        )

    def is_2d_model(self) -> bool:
        return self.is_barotropic() or self.is_advection2d()

    def is_3d_model(self) -> bool:
        return self.is_baroclinic() or self.is_advection3d()


class AbstractEquationBase(Generic[DM], ABC):
    _disc: DiscretizationInfo
    _method: DM
    _mode: EquationMode
    _cfg: Dict[str, Any]
    _built: bool

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: DM,
        mode: EquationMode
    ):
        self._disc = disc
        self._method = method
        self._mode = mode
        self._cfg = {}
        self._built = False

    @abstractmethod
    def config2dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_with_config(self, cfg: model_config.ModelConfig) -> Self:
        raise NotImplementedError

    @abstractmethod
    def prepare_block(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @property
    def disc(self) -> DiscretizationInfo:
        return self._disc

    @property
    def method(self) -> DM:
        return self._method

    @property
    def mode(self) -> EquationMode:
        return self._mode

    def has_built(self) -> bool:
        return self._built

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
    Generic[DM],
    ABC
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: DM,
        mode: EquationMode
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )
        self._table = "equterm"

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        raise NotImplementedError

    def _update_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in cfg.items():
            self._cfg[self.alias_name() + "." + k] = v
        return self._cfg

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


class AbstractDynamicalEquationTerm(
    AbstractEquationTerm[DM],
    Generic[DM],
    ABC
):
    @abstractmethod
    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
        *args: Any,
        **kwargs: Any
    ) -> BLOCK:
        raise NotImplementedError


class AbstractAdvectionEquationTerm(
    AbstractEquationTerm[DM],
    Generic[DM],
    ABC
):
    @abstractmethod
    def __call__(self, block: BLOCK, *args: Any, **kwargs: Any) -> BLOCK:
        raise NotImplementedError

    @abstractmethod
    def calc_at_batch(
        self,
        batch: AbstractAdvBatchType,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
        *args: Any,
        **kwargs: Any
    ) -> tuple[AbstractAdvBatchType, BLOCK]:
        raise NotImplementedError


class AbstractEquationForm(
    AbstractEquationBase[DM],
    Generic[DM],
    ABC
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: DM,
        mode: EquationMode
    ):
        disc_copy = disc.copy()
        disc_copy.equ = self.alias_name()
        super().__init__(
            disc=disc_copy,
            method=method,
            mode=mode
        )
        self._table = "equ"

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def ode_kind_alias_name() -> str:
        raise NotImplementedError

    def _update_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in cfg.items():
            self._cfg[self.alias_name() + "." + k] = v
        return self._cfg

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


class AbstractDynamicalEquationForm(
    AbstractEquationForm[DM],
    Generic[DM],
    ABC
):
    @abstractmethod
    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
        *args: Any,
        **kwargs: Any
    ) -> BLOCK:
        raise NotImplementedError


class AbstractAdvectionEquationForm(
    AbstractEquationForm[DM],
    Generic[DM],
    ABC
):

    @abstractmethod
    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
        *args: Any,
        **kwargs: Any
    ) -> BLOCK:
        raise NotImplementedError


class AbstractContinuityEquForm(
    AbstractDynamicalEquationForm[DM],
    Generic[DM],
    ABC
):
    pass


class AbstractMomentumEquForm(
        AbstractDynamicalEquationForm[DM],
        Generic[DM],
        ABC):
    pass


class AbstractThermodynamicEquForm(
    AbstractDynamicalEquationForm[DM],
    Generic[DM],
    ABC
):
    pass


class AbstractTracerEquForm(
    AbstractAdvectionEquationForm[DM],
    Generic[DM],
    ABC
):
    pass


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


class AbstractEquationOperator(
    AbstractEquationBase[DM],
    Generic[DM],
    ABC
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: DM,
        mode: EquationMode
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )

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


class AbstractDynamicalEquationOperator(
    AbstractEquationOperator[DM],
    Generic[DM],
    ABC
):
    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        *args: Any,
        **kwargs: Any
    ) -> BLOCK:
        raise NotImplementedError


class AbstractAdvectionEquationOperator(
    AbstractEquationOperator[DM],
    Generic[DM],
    ABC
):
    def __call__(
        self,
        batch: "AbstractAdvBatchType",
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        *args: Any,
        **kwargs: Any
    ) -> tuple["AbstractAdvBatchType", BLOCK]:
        raise NotImplementedError


class AbstractPrepareSpaceOperators(
    AbstractEquationBase[DM],
    Generic[DM],
    ABC
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: DM,
        mode: EquationMode
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )
        self._table = "equaux"

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        *args, **kwargs
    ) -> BLOCK:
        raise NotImplementedError

    @abstractmethod
    def first_call(self, block: BLOCK, *args, **kwargs) -> BLOCK:
        """First call to the space operators, which will be called in the prepare step."""
        raise NotImplementedError

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


class AbstractSpaceOperators(
    AbstractEquationBase[DM],
    Generic[DM, PSO],
    ABC
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: DM,
        mode: EquationMode,
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )
        self._table = "equaux"
        factory = self.navi2prepare_ops()
        operators_prepare: PSO = factory(
            disc=disc,
            method=method,
            mode=mode
        )
        self._operators_prepare = operators_prepare

    @abstractmethod
    def __call__(
        self,
        block: BLOCK,
        old_idx: int,
        star_idx: Int[Array, ""] | int,
        new_idx: int,
        dt: Float[Array, ""] | float,
        substep: int,
        *args,
        **kwargs
    ) -> BLOCK:
        raise NotImplementedError

    @abstractmethod
    def diagnose(self, block: BLOCK, *args, **kwargs) -> BLOCK:
        """Optional diagnostic function to be called after space operators."""
        raise NotImplementedError

    @abstractmethod
    def update_state(
        self,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
        *args,
        **kwargs
    ) -> BLOCK:
        """Update the state of the block after applying space operators."""
        raise NotImplementedError

    @abstractmethod
    def advection_stepper(self, block: BLOCK, *args, **kwargs) -> BLOCK:
        """Return the advection stepper function to be called in the advection step."""
        raise NotImplementedError

    @property
    def prepare(self) -> PSO:
        return self._operators_prepare

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

    def navi2prepare_ops(self) -> Callable:
        """Navigate to the prepare space operators configuration in the equaux table."""
        id_col = "unique_id"
        disc = self.disc_id()
        choice = "prepare_space_operators"
        id_val = disc + "::" + choice
        return navigate("equaux", id_col=id_col, id_val=id_val)[0]


class AbstractEquationSet(
    AbstractEquationBase[DM],
    AbstractODE,
    Generic[DM, SOP],
    ABC
):
    _space_operators: SOP

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: DM,
        mode: EquationMode
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )
        self._table = "equset"

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

    @abstractmethod
    def build_block_cls(self, coord: CoordinateSystem) -> Any:
        """Build the initial block from a testcase."""
        raise NotImplementedError

    @abstractmethod
    def build_default(self, *args, **kwargs) -> Self:
        raise NotImplementedError

    @property
    def space_operators(self) -> SOP:
        return self._space_operators

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

    def config2ncattrs(self) -> Dict[str, Any]:
        """Convert the configuration dictionary to attributes for netCDF file creation."""
        cleaned: Dict[str, Any] = {}
        for k, v in self._cfg.items():
            if isinstance(v, bool):
                cleaned[k] = np.int32(v)
                continue
            arr = np.asarray(v)
            if arr.shape == () and arr.dtype.kind in ("i", "u"):
                cleaned[k] = np.int32(arr)
            else:
                cleaned[k] = v
        return cleaned

    def navi2space_ops(self) -> Callable:
        """Navigate to the space operators configuration in the equaux table."""
        id_col = "unique_id"
        disc = self.disc_id()
        choice = "space_operators"
        id_val = disc + "::" + choice
        return navigate("equaux", id_col=id_col, id_val=id_val)[0]

    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
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
    def reset(self) -> Self:
        """Reset all variables to a specific value."""
        raise NotImplementedError

    def to_netcdf(
        self,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        path: str,
        format="UGRID",
        use_dual: bool = False,
        expand_idx: Optional[int] = None,
        **kwargs
    ) -> None:
        """Save variable set to a NetCDF file."""
        if format == "UGRID":
            ds = self.to_xarray_with_ugrid_format(
                use_dual=use_dual,
                expand_idx=expand_idx
            )
            grid_ds = coord.mesh.to_ugrid(
                use_dual=use_dual,
                **kwargs
            ).to_xarray()
            ds = xr.merge([ds, grid_ds])
            ds.to_netcdf(path)

    def to_ugrid(
        self,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        use_dual: bool = False,
        expand_idx: Optional[int] = None,
        **kwargs
    ) -> tuxr.UGRIDDataset:
        """Convert variable set to a UXarray.UxGrid."""
        ds = self.to_xarray_with_ugrid_format(
            use_dual=use_dual,
            expand_idx=expand_idx
        )
        uxgrid = coord.mesh.to_ugrid(
            use_dual=use_dual,
            **kwargs
        ).to_uxgrid()
        return tuxr.UGRIDDataset.from_xarray(ds, uxgrid)

    def to_uxarray(
        self,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        use_dual: bool = False,
        expand_idx: Optional[int] = None,
        **kwargs
    ) -> uxr.UxDataset:
        """Convert variable set to a UXarray.Dataset."""
        ds = self.to_xarray_with_ugrid_format(
            use_dual=use_dual,
            expand_idx=expand_idx
        )
        uxgrid = coord.mesh.to_ugrid(
            use_dual=use_dual,
            **kwargs
        ).to_uxgrid()
        return uxr.UxDataset.from_xarray(ds, uxgrid)

    @abstractmethod
    def to_xarray_with_ugrid_format(
        self,
        use_dual: bool = False,
        expand_idx: Optional[int] = None
    ) -> xr.Dataset:
        """Convert variable set to a xarray.Dataset with UGRID format."""
        raise NotImplementedError


class AbstractFields(AbstractVariableSet, Generic[FMT]):
    mode: EquationMode = eqx.field(static=True)
    fields: Dict[str, FMT] = eqx.field(static=True)

    def __init__(
        self,
        mode: EquationMode,
        **fields: ArrayField[FMT] | ArrayField[NumericMetaData] | Any,
    ):
        object.__setattr__(self, "mode", mode)
        exist_fields: Dict[str, FMT | NumericMetaData] = {}
        for k, v in fields.items():
            if isinstance(v, Field):
                if not v.is_empty():
                    exist_fields[k] = v.meta
                object.__setattr__(self, k, v)
            else:
                raise TypeError(f"Variable '{k}' is not a Field.")
        object.__setattr__(self, "fields", exist_fields)

    @classmethod
    def get_all_fields(cls) -> tuple[str, ...]:
        def is_field_type(tp: Any) -> bool:
            origin = get_origin(tp)
            if origin is None:
                if tp is Field:
                    return True
                if isinstance(tp, type) and issubclass(tp, Field):
                    return True
                return False
            if origin in (Union, types.UnionType):
                return any(is_field_type(arg) for arg in get_args(tp))
            if origin is Literal:
                return False
            return isinstance(origin, type) and issubclass(origin, Field)

        annotations: Dict[str, Any] = {}
        for base in reversed(cls.__mro__):
            if base is object:
                continue
            try:
                annotations.update(get_type_hints(base, include_extras=True))
            except Exception:
                annotations.update(getattr(base, "__annotations__", {}))

        return tuple(
            name for name, tp in annotations.items() if is_field_type(tp)
        )

    def __iter__(self):
        for name in self.existing_meta().keys():
            yield getattr(self, name)

    def __len__(self) -> int:
        return len(self.existing_meta().keys())

    def _wrap(self, **kwargs: ArrayField[FMT]) -> Self:
        names = self.existing_meta().keys()
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

    def reset(self, *args, **kwargs) -> Self:
        new_vars: Dict[str, ArrayField[FMT]] = {}
        for key in self.keys():
            var = object.__getattribute__(self, key)
            if var.meta.is_prognostic:
                continue
            if isinstance(var.data, jnp.ndarray):
                new_vars[key] = Field.create(
                    data=jnp.zeros_like(var.data),
                    meta=var.meta,
                    as_quantity=var.is_quantity()
                )
            elif isinstance(var.data, ux.AbstractQuantity):
                new_array = jnp.zeros_like(jnp.asarray(var.data.value))
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
        return list(self.existing_meta().keys())

    def values(self) -> List[ArrayField[FMT]]:
        return [getattr(self, name) for name in self.existing_meta().keys()]

    def items(self) -> List[tuple[str, ArrayField[FMT]]]:
        return [(name, getattr(self, name)) for name in self.existing_meta().keys()]

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
    def default_meta(
        mode: EquationMode,
        *args,
        **kwargs
    ) -> Dict[str, FMT | NumericMetaData] | Dict[str, FMT]:
        """Return default variable metadata dictionary."""
        return {}

    @classmethod
    def default_fields(
        cls,
        mode: EquationMode,
        *args,
        **kwargs
    ) -> Union[
        Dict[
            str,
            ArrayField[FMT] | ArrayField[NumericMetaData]
        ],
        Dict[str, ArrayField[FMT]]
    ]:
        """Return default variable field dictionary."""
        fields = {}
        for key, field_meta in cls.default_meta(mode=mode).items():
            if isinstance(field_meta, NumericMetaData):
                fields[key] = ArrayField[NumericMetaData](
                    data=jnp.zeros(()),
                    meta=field_meta
                )
            else:
                fields[key] = ArrayField[FMT](
                    data=jnp.zeros(()),
                    meta=field_meta
                )
        return fields

    def existing_values(self) -> Dict[str, ArrayField[FMT]]:
        """Return existing variables."""
        if self.is_fields():
            return dict(self.items())
        else:
            raise TypeError("Variables are not stored as Fields.")

    def existing_meta(self) -> Dict[str, FMT]:
        """Return metadata of existing variables."""
        return self.fields

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
    def from_arrays(
        cls,
        mode: EquationMode,
        array_dict: Dict[str, Array],
        meta: Dict[str, FMT],
        *args,
        **kwargs
    ) -> Self:
        new_variables: Dict[str, ArrayField[FMT]] = {}
        for key, var in array_dict.items():
            if key not in meta:
                raise KeyError(f"Metadata for variable '{key}' not found.")
            new_variables[key] = ArrayField[FMT](
                data=var,
                meta=meta[key]
            )
        return cls(mode=mode, **new_variables)

    @classmethod
    def from_coord(
        cls,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        equation: AbstractEquationSet,
        *args,
        **kwargs
    ) -> Self:
        meta = cls.default_meta(mode=equation.mode)
        exist_fields: Dict[
            str,
            ArrayField[FMT] | ArrayField[NumericMetaData]
        ] = {}
        for field_name in cls.get_all_fields():
            if field_name not in meta:
                exist_fields[field_name] = ArrayField.empty()
            else:
                exist_fields[field_name] = meta[field_name].zeros_like(coord)
        return cls(mode=equation.mode, **exist_fields)

    def zeros_like(self) -> Self:
        new_variables: Dict[str, ArrayField[FMT]] = {}
        for key, var in self.items():
            if isinstance(var, Field):
                new_variables[key] = var.zeros_like()
            else:
                raise TypeError(f"Variable '{key}' is not a Field.")
        return self._wrap(**new_variables)

    @classmethod
    def empty(cls, mode: EquationMode) -> Self:
        exist_fields: Dict[str, ArrayField[FMT]] = {}
        for field_name in cls.get_all_fields():
            exist_fields[field_name] = ArrayField.empty()
        return cls(mode=mode, **exist_fields)

    def empty_like(self) -> Self:
        return self.empty(mode=self.mode)

    def to_default_unit(self) -> Self:
        """Convert all parameters to default unit system.
        - If the parameter is a Field, convert to the default unit defined in metadata.
        - If the parameter is an AbstractQuantity, convert to SI units.
        """
        default = self.default_meta(mode=self.mode)
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
            return self._wrap(**new_variables)
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
            return self._wrap(**new_variables)
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
            return self._wrap(**new_variables)
        else:
            raise TypeError(
                "Cannot convert to SI units on incompatible variable types."
                "Please check if all variables are Fields."
            )

    def to_xarray_with_ugrid_format(
        self,
        use_dual: bool = False,
        expand_idx: Optional[int] = None
    ) -> xr.Dataset:
        """Get the meta needed for UGRID NetCDF Format"""
        var_ds: Dict[str, xr.DataArray] = {}
        for var in self.values():
            if isinstance(var.meta, StaggeredFieldMetaData):
                var_ds[var.meta.name] = tuxr.format_terraflow2ugrid(
                    var.data,
                    var.meta.to_dict(),
                    use_dual=use_dual,
                    expand_idx=expand_idx
                )
            elif isinstance(var.meta, NumericMetaData):
                continue
            else:
                raise TypeError(
                    f"Variable '{var.meta.name}' metadata is not of type StaggeredFieldMetaData, "
                    "which is required for UGRID format conversion."
                )
        return xr.Dataset(data_vars=var_ds)


class AbstractAdvBatchType(AbstractFields[FMT], Generic[FMT]):
    name: str = eqx.field(static=True)
    hori_loc: str = eqx.field(static=True)
    vert_loc: str = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    dts_substep: Float[Array, "nsubsteps"]
    nsubsteps: int = eqx.field(static=True)
    nsteps: int = eqx.field(static=True)
    istep: Array
    idx: Int[Array, "ntracers_in_batch"]
    old: Int[Array, ""]
    new: Int[Array, ""]

    def __init__(
        self,
        name: str,
        batch_hori_loc: str,
        batch_vert_loc: str,
        mode: EquationMode,
        dt: float,
        dts_substep: Float[Array, "nsubsteps"],
        dt_dyn: float,
        idx: Optional[Int[Array, "ntracers_in_batch"]] = None,
        **fields: ArrayField[FMT]
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "hori_loc", batch_hori_loc)
        object.__setattr__(self, "vert_loc", batch_vert_loc)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "dts_substep", dts_substep)
        if dts_substep.shape == ():
            object.__setattr__(self, "nsubsteps", 0)
        else:
            object.__setattr__(self, "nsubsteps", dts_substep.shape[0])
        object.__setattr__(self, "nsteps", int(dt / dt_dyn))
        object.__setattr__(self, "istep", jnp.array(0))
        if idx is None:
            idx = jnp.array([])
        object.__setattr__(self, "idx", idx)
        object.__setattr__(self, "old", jnp.array(0))
        object.__setattr__(self, "new", jnp.array(1))
        super().__init__(mode=mode, **fields)

    @staticmethod
    @abstractmethod
    def default_meta(
        name: str,
        batch_hori_loc: str,
        batch_vert_loc: str,
        mode: EquationMode,
    ) -> Dict[str, FMT]:
        """Return default variable metadata dictionary."""
        raise NotImplementedError

    @classmethod
    def default_fields(
        cls,
        name: str,
        batch_hori_loc: str,
        batch_vert_loc: str,
        mode: EquationMode,
    ) -> Dict[str, ArrayField[FMT]]:
        """Return default variable field dictionary."""
        return {
            key: ArrayField[FMT](
                data=jnp.zeros(()),
                meta=field_meta
            )
            for key, field_meta in cls.default_meta(
                name=name,
                batch_hori_loc=batch_hori_loc,
                batch_vert_loc=batch_vert_loc,
                mode=mode
            ).items()
        }

    @classmethod
    def from_arrays(
        cls,
        name: str,
        batch_hori_loc: str,
        batch_vert_loc: str,
        mode: EquationMode,
        array_dict: Dict[str, Array],
        meta: Dict[str, FMT],
        dt: float,
        dts_substep: Float[Array, "nsubsteps"],
        dt_dyn: float,
        idx: Optional[Int[Array, "ntracers_in_batch"]] = None
    ) -> Self:
        new_variables: Dict[str, ArrayField[FMT]] = {}
        for key, var in array_dict.items():
            if key not in meta:
                raise KeyError(f"Metadata for variable '{key}' not found.")
            new_variables[key] = ArrayField[FMT](
                data=var,
                meta=meta[key]
            )
        return cls(
            name=name,
            batch_hori_loc=batch_hori_loc,
            batch_vert_loc=batch_vert_loc,
            mode=mode,
            dt=dt,
            dts_substep=dts_substep,
            dt_dyn=dt_dyn,
            idx=idx,
            **new_variables,
        )

    @classmethod
    def from_coord(
        cls,
        name: str,
        batch_hori_loc: str,
        batch_vert_loc: str,
        coord: CoordinateSystem,
        equation: AbstractEquationSet,
        dt: float,
        dts_substep: Float[Array, "nsubsteps"],
        dt_dyn: float,
        idx: Optional[Int[Array, "ntracers_in_batch"]] = None,
    ) -> Self:
        meta = cls.default_meta(
            name=name,
            batch_hori_loc=batch_hori_loc,
            batch_vert_loc=batch_vert_loc,
            mode=equation.mode
        )
        exist_fields: Dict[str, ArrayField[FMT]] = {}
        for _cls_name in cls.get_all_fields():
            field_name = name + "_" + _cls_name
            if field_name not in [m.name for m in meta.values()]:
                exist_fields[_cls_name] = Field.empty()
            else:
                exist_fields[_cls_name] = meta[_cls_name].zeros_like(coord)
        return cls(
            name=name,
            batch_hori_loc=batch_hori_loc,
            batch_vert_loc=batch_vert_loc,
            mode=equation.mode,
            dt=dt,
            dts_substep=dts_substep,
            dt_dyn=dt_dyn,
            idx=idx,
            **exist_fields
        )

    @classmethod
    def empty(cls, mode: EquationMode) -> Self:
        exist_fields: Dict[str, ArrayField[FMT]] = {}
        for _cls_name in cls.get_all_fields():
            exist_fields[_cls_name] = Field.empty()
        return cls(
            name="",
            batch_hori_loc="",
            batch_vert_loc="",
            mode=mode,
            dt=0.0,
            dts_substep=jnp.zeros(()),
            dt_dyn=1.0,
            idx=None,
            **exist_fields
        )

    def to_default_unit(self) -> Self:
        """Convert all parameters to default unit system.
        - If the parameter is a Field, convert to the default unit defined in metadata.
        - If the parameter is an AbstractQuantity, convert to SI units.
        """
        default = self.default_meta(
            name=self.name,
            batch_hori_loc=self.hori_loc,
            batch_vert_loc=self.vert_loc,
            mode=self.mode
        )
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
            return self._wrap(**new_variables)
        else:
            raise TypeError(
                "Cannot convert to default unit on incompatible variable types."
                "Please check if all variables are Fields."
            )

    def swap_time_indices(self) -> Self:
        return eqx.tree_at(
            lambda b: (b.old, b.new),
            self,
            (self.new, self.old)
        )

    def get_time_idx(self, itime_str: str) -> Int[Array, ""]:
        if itime_str == "old":
            return self.old
        elif itime_str == "new":
            return self.new
        else:
            raise ValueError(
                f"Invalid time index '{itime_str}'. Expected 'old' or 'new'."
            )

    def is_run_tracer_istep(self) -> Int[Array, ""]:
        """Determine whether to run tracer."""
        return self.istep == (self.nsteps - 1)

    def update_istep(self) -> Self:
        istep = self.istep
        nsteps = self.nsteps

        def not_reset_istep(batch: Self):
            return eqx.tree_at(
                where=lambda b: b.istep,
                pytree=batch,
                replace=istep + 1
            )

        def reset_istep(batch: Self):
            return eqx.tree_at(
                where=lambda b: b.istep,
                pytree=batch,
                replace=jnp.array(0, dtype=istep.dtype)
            )

        return jax.lax.cond(
            pred=istep == nsteps - 1,
            true_fun=reset_istep,
            false_fun=not_reset_istep,
            operand=self
        )


class TracerMetaData:
    batch_name: str = eqx.field(static=True, default="")
    dt: datetime.timedelta = eqx.field(
        static=True,
        default=datetime.timedelta(0)
    )
    dt_seconds: float = eqx.field(static=True, default=0.0)
    name: str = eqx.field(static=True, default="")
    long_name: str = eqx.field(static=True, default="")
    unit: str = eqx.field(static=True, default="")

    def __init__(
        self,
        batch_name: str,
        dt: datetime.timedelta,
        name: str,
        long_name: str,
        unit: Optional[str] = None,
    ):
        object.__setattr__(self, "batch_name", batch_name)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "dt_seconds", dt.total_seconds())
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "long_name", long_name)
        if unit is None:
            unit = ""
        object.__setattr__(self, "unit", unit)

    @classmethod
    def from_reader(
        cls,
        mode: EquationMode,
        para: PlanetParameterSet,
        reader: AbstractReader,
        dt_adv: datetime.timedelta
    ) -> Dict[str, Self]:
        if mode.ideal_dry_core:
            return {}

        if para.planet == "earth":
            if reader.reader_type == "era5":
                meta: Dict[str, Self] = {}
                meta["qv"] = cls(
                    batch_name="moist",
                    dt=dt_adv,
                    name="qv",
                    long_name="Water vapor",
                    unit="kg kg-1"
                )
                meta["qc"] = cls(
                    batch_name="moist",
                    dt=dt_adv,
                    name="qc",
                    long_name="Cloud water",
                    unit="kg kg-1"
                )
                meta["qi"] = cls(
                    batch_name="moist",
                    dt=dt_adv,
                    name="qi",
                    long_name="Cloud ice",
                    unit="kg kg-1"
                )
                meta["qr"] = cls(
                    batch_name="moist",
                    dt=dt_adv,
                    name="qr",
                    long_name="Rain water",
                    unit="kg kg-1"
                )
                meta["qs"] = cls(
                    batch_name="moist",
                    dt=dt_adv,
                    name="qs",
                    long_name="Snow",
                    unit="kg kg-1"
                )
                return meta
            else:
                raise NotImplementedError(
                    f"Reader type '{reader.reader_type}' not supported for TracerMetaData extraction."
                )
        else:
            raise NotImplementedError(
                f"Planet '{para.planet}' not supported for TracerMetaData extraction."
            )


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

    def reset(self) -> Self:
        new_vars: Dict[str, ArrayField[FMT]] = {}
        for key, var in self.items():
            if isinstance(var.data, jnp.ndarray):
                new_vars[key] = Field(
                    data=jnp.zeros_like(var.data),
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

    def to_xarray_with_ugrid_format(
        self,
        use_dual: bool = False,
        expand_idx: Optional[int] = None
    ) -> xr.Dataset:
        """Get the meta needed for UGRID NetCDF Format"""
        var_ds: Dict[str, xr.DataArray] = {}
        for key, var in self.items():
            if isinstance(var.meta, StaggeredFieldMetaData):
                var_ds[key] = tuxr.format_terraflow2ugrid(
                    var.data,
                    self.meta[key].to_dict(),
                    use_dual=use_dual,
                    expand_idx=expand_idx
                )
            elif isinstance(var.meta, NumericMetaData):
                continue
            else:
                raise TypeError(
                    f"Variable '{key}' metadata is not of type StaggeredFieldMetaData, "
                    "which is required for UGRID format conversion."
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


class AbstractBlock(
    ABC,
    eqx.Module,
    Generic[
        HM, VM, HLO, VLO,
        STATE, TEND, STATIC, AUX, ADVBATCH, TRACERS, AUXDS
    ]
):
    elapsed_seconds: Float[Array, ""]
    mode: EquationMode = eqx.field(static=True)
    old: Array
    new: Array
    _coord: CoordinateSystem[HM, VM, HLO, VLO]
    para: PlanetParameterSet
    state: STATE
    tend: TEND
    static: STATIC
    aux: AUX
    adv_batch_bg: ADVBATCH
    adv_batch_pt: ADVBATCH
    adv_batches: dict[str, ADVBATCH]
    tracers: TRACERS
    equaux: AUXDS
    test_case: AbstractTestCase | None = eqx.field(default=None, static=True)

    @classmethod
    @abstractmethod
    def from_testcase(
        cls,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        para: PlanetParameterSet,
        solver: AbstractSolver,
        *,
        case: Any = None,
    ) -> Self:
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs) -> Self:
        raise NotImplementedError

    @property
    def dev(self) -> device.AbstractDevice:
        return self.coord.dev

    @property
    def coord(self) -> CoordinateSystem[HM, VM, HLO, VLO]:
        return self._coord

    def diagnose(self, itime_str: str) -> Self:
        return self

    def swap_time_indices(self) -> Self:
        return eqx.tree_at(
            lambda b: (b.old, b.new),
            self,
            (self.new, self.old)
        )

    def swap_batches_time_indices(self) -> Self:
        for batch_name, batch in self.adv_batches.items():

            def swap_batch_time_indices(block: Self) -> Self:
                return eqx.tree_at(
                    lambda b: b.adv_batches[batch_name],
                    block,
                    batch.swap_time_indices()
                )

            def do_nothing(block: Self) -> Self:
                return block

            self = jax.lax.cond(
                pred=batch.is_run_tracer_istep(),
                true_fun=swap_batch_time_indices,
                false_fun=do_nothing,
                operand=self
            )

        return self

    def get_time_idx(self, itime_str: str) -> Int[Array, ""]:
        if itime_str == "old":
            return self.old
        elif itime_str == "new":
            return self.new
        else:
            raise ValueError(
                f"Invalid time index '{itime_str}'. Expected 'old' or 'new'."
            )

    def update_batches_istep(self) -> Self:
        for batch_name, batch in self.adv_batches.items():
            batch = batch.update_istep()
            self = eqx.tree_at(
                lambda b: b.adv_batches[batch_name],
                self,
                batch
            )
        return self


class AbstractDiagnosticTools(Generic[HM, VM, HLO, VLO, DM, ES, FMT], ABC):

    def __init__(
        self,
        solver: AbstractSolver,
        jit: bool = False,
        trajectory: bool = False
    ):
        self.solver = solver
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
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        field: Field,
        to_ugrid: Literal[False] = False,
        **kwargs
    ) -> Field:
        ...

    @overload
    def process_field(
        self,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        field: Field,
        to_ugrid: Literal[True],
        **kwargs
    ) -> tuxr.UGRIDDataArray:
        ...

    def process_field(
        self,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
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
        return self.solver.equation

    @staticmethod
    def expand_dims_on_state(
        state: STATE,
        expand_axis: int = 0,
        expand_axis_length: int = 3
    ) -> STATE:
        def wrap(leaf):
            if isinstance(leaf, Array) and leaf.size != 0:
                res = jnp.expand_dims(leaf, axis=expand_axis)
                tile_shape = [1] * res.ndim
                tile_shape[expand_axis] = int(expand_axis_length)
                return jnp.tile(res, tuple(tile_shape))
            return leaf
        return jax.tree_util.tree_map(f=wrap, tree=state)


if TYPE_CHECKING:
    from terraflow.discretization.fvm.scvt.trisk.single_lev.diag import DiagnosticsTools as fvm_scvt_trisk_1lev_diag
    from terraflow.dynamics.shallow_water.types import ExVif_SWE_SCVT_TriSKCGrid_FVM


DIAG = TypeVar("DIAG", bound=AbstractDiagnosticTools)


class DiagnosticToolsBuilder(Generic[DIAG]):

    @classmethod
    def select(
        cls,
        coord: CoordinateSystem,
        para: PlanetParameterSet,
        equation: AbstractEquationSet,
    ) -> Callable[..., DIAG]:
        disc = DiscretizationInfo.from_model(
            coord=coord,
            para=para,
            operator=equation.method,
            equset_type=type(equation)
        )
        diag_factory = navigate(
            "diag",
            id_col="unique_id",
            id_val=disc.id(),
            attr="factory"
        )[0]
        return cast(Callable[..., DIAG], diag_factory)

    @overload
    @classmethod
    def from_model(
        cls,
        solver: AbstractSolver[ExVif_SWE_SCVT_TriSKCGrid_FVM],
        coord: CoordinateSystem,
        para: PlanetParameterSet,
        jit: bool = False,
        trajectory: bool = False,
        **kwargs,
    ) -> "fvm_scvt_trisk_1lev_diag":
        ...

    @overload
    @classmethod
    def from_model(
        cls,
        solver: AbstractSolver[ES],
        coord: CoordinateSystem,
        para: PlanetParameterSet,
        jit: bool = False,
        trajectory: bool = False,
        **kwargs,
    ) -> AbstractDiagnosticTools:
        ...

    @classmethod
    def from_model(
        cls,
        solver: AbstractSolver[ES],
        coord: CoordinateSystem,
        para: PlanetParameterSet,
        jit: bool = False,
        trajectory: bool = False,
        **kwargs,
    ) -> AbstractDiagnosticTools:
        diag_factory = cls.select(
            coord=coord, para=para, equation=solver.equation
        )
        return diag_factory(
            solver=solver,
            jit=jit,
            trajectory=trajectory,
            **kwargs
        )
