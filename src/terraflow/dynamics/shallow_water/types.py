from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Literal,
    Self,
    overload,
    TypeVar,
    Generic,
    Callable,
    TYPE_CHECKING,
)
import inspect

from terraflow.util import typing, log, model_config
from terraflow.planet import PlanetParameterSet
from terraflow.operator import AbstractDiscretizationMethod
from terraflow.mesh.abstract import AbstractMesh, AbstractUnstructuredMesh
from terraflow.vert_coord.abstract import AbstractVertCoordinate
from terraflow.layout.abstract import AbstractLayout, CoordinateSystem
from terraflow.solver.abstract import AbstractExplicitODE
import terraflow.discretization.abstract as disc_types

if TYPE_CHECKING:
    from terraflow.operator import FiniteVolumeMethod
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    import terraflow.discretization.fvm.scvt.trisk.single_lev.dynamics_types as fvm_scvt_trisk_vars
    import terraflow.discretization.fvm.scvt.trisk.single_lev.term as fvm_scvt_trisk_terms

    Int = typing.Int
    Float = typing.Float
    Array = typing.Array

HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
HLO = TypeVar("HLO", bound=AbstractLayout)
VLO = TypeVar("VLO", bound=AbstractLayout)
DM = TypeVar("DM", bound=AbstractDiscretizationMethod)
UHM = TypeVar("UHM", bound=AbstractUnstructuredMesh)
BLOCK = TypeVar("BLOCK", bound=disc_types.AbstractBlock)
STATE = TypeVar("STATE", bound=disc_types.AbstractFields)
PSO = TypeVar("PSO", bound=disc_types.AbstractPrepareSpaceOperators)
SOP = TypeVar("SOP", bound=disc_types.AbstractSpaceOperators)


class AbstractShallowWaterEquationSet(
    disc_types.AbstractEquationSet[DM, SOP],
    Generic[DM, SOP],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        **kwargs
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=disc_types.EquationMode.from_swm(**kwargs)
        )
        factory = self.navi2space_ops()
        space_operators: SOP = factory(
            disc=disc,
            method=method,
            mode=self.mode
        )
        self._space_operators = space_operators

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the dynamics set."""
        return "swe"


ShallowWaterEquationSet = TypeVar(
    "ShallowWaterEquationSet",
    bound=AbstractShallowWaterEquationSet
)


class AbstractShallowWaterEquationForm(
    disc_types.AbstractDynamicalEquationForm[DM],
    Generic[DM],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: disc_types.EquationMode
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )


class AbstractShallowWaterTerm(
    disc_types.AbstractDynamicalEquationTerm[DM],
    Generic[DM],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: disc_types.EquationMode
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )


class AbstractShallowWaterSpaceOperators(
    disc_types.AbstractSpaceOperators[DM, PSO],
    Generic[DM, PSO],
    ABC
):

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: disc_types.EquationMode
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )

    @abstractmethod
    def init_continuity_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM
    ) -> disc_types.AbstractContinuityEquForm:
        raise NotImplementedError

    @abstractmethod
    def init_momentum_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM
    ) -> disc_types.AbstractMomentumEquForm:
        raise NotImplementedError


class DivergenceOfMassFluxProtocol(
    AbstractShallowWaterTerm[DM],
    Generic[DM]
):
    @staticmethod
    def alias_name() -> str:
        return "dmf"


DMF = TypeVar("DMF", bound=DivergenceOfMassFluxProtocol)


class NonlinearCoriolisTermProtocol(
    AbstractShallowWaterTerm[DM],
    Generic[DM]
):
    @staticmethod
    def alias_name() -> str:
        return "nct"


NCT = TypeVar("NCT", bound=NonlinearCoriolisTermProtocol)


class GradientOfKineticEnergyProtocol(
    AbstractShallowWaterTerm[DM],
    Generic[DM]
):
    @staticmethod
    def alias_name() -> str:
        return "grad_ke"


GKE = TypeVar("GKE", bound=GradientOfKineticEnergyProtocol)


class PressureGradientForceTermProtocol(
    AbstractShallowWaterTerm[DM],
    Generic[DM]
):
    @staticmethod
    def alias_name() -> str:
        return "pgf"


PGF = TypeVar("PGF", bound=PressureGradientForceTermProtocol)


class ExContinuityEqu(
    AbstractShallowWaterEquationForm[DM],
    disc_types.AbstractContinuityEquForm[DM],
    Generic[DMF, DM],
    ABC
):
    dmf: DMF

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: disc_types.EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.dmf = self.init_div_mf()

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> BLOCK:
        return self.dmf(block, star_idx, dt, substep)

    @staticmethod
    def alias_name() -> str:
        return "dgzdt"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def config2dict(self) -> dict:
        cfg = self.dmf.config2dict()
        self._update_config(cfg=cfg)
        return self._cfg

    def build_with_config(self, cfg: model_config.ModelConfig) -> Self:
        if not self.has_built():
            self.dmf.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.dmf.prepare_block(block)
        return block

    def init_div_mf(self) -> DMF:
        factory = self.navi2term(
            group="div_of_mf",
            choice="builder"
        )
        origin = getattr(factory, '__origin__', factory)
        if inspect.isclass(origin) and issubclass(origin, DivergenceOfMassFluxProtocol):
            instance = factory(
                disc=self._disc, method=self.method, mode=self.mode
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of DivergenceOfMassFluxProtocol."
            )


class ExNormalMomentumEqu4UGRID(
    AbstractShallowWaterEquationForm[DM],
    disc_types.AbstractMomentumEquForm[DM],
    Generic[NCT, GKE, PGF, DM],
    ABC
):
    coriolis: NCT
    grad_ke: GKE
    pgf: PGF

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: disc_types.EquationMode
    ) -> None:
        super().__init__(disc=disc, method=method, mode=mode)
        self.coriolis = self.init_nonlinear_coriolis_term()
        self.grad_ke = self.init_grad_kinetic_energy_term()
        self.pgf = self.init_grad_geopotential_term()

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> BLOCK:
        block = self.coriolis(block, star_idx, dt, substep)
        block = self.grad_ke(block, star_idx, dt, substep)
        block = self.pgf(block, star_idx, dt, substep)
        return block

    @staticmethod
    def alias_name() -> str:
        return "dundt"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def config2dict(self) -> Dict[str, Any]:
        cfg = {}
        cfg.update(self.coriolis.config2dict())
        cfg.update(self.grad_ke.config2dict())
        cfg.update(self.pgf.config2dict())
        self._update_config(cfg=cfg)
        return self._cfg

    def build_with_config(self, cfg: model_config.ModelConfig) -> Self:
        if not self._built:
            self.coriolis.build_with_config(cfg=cfg)
            self.grad_ke.build_with_config(cfg=cfg)
            self.pgf.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.coriolis.prepare_block(block)
        block = self.grad_ke.prepare_block(block)
        block = self.pgf.prepare_block(block)
        return block

    def init_nonlinear_coriolis_term(self) -> NCT:
        factory = self.navi2term(
            group="nonlinear_coriolis",
            choice="builder"
        )
        if inspect.isclass(factory) and issubclass(factory, NonlinearCoriolisTermProtocol):
            instance = factory(
                disc=self._disc, method=self.method, mode=self.mode
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of NonlinearCoriolisTermProtocol."
            )

    def init_grad_kinetic_energy_term(self) -> GKE:
        factory = self.navi2term(
            group="grad_of_kinetic_energy",
            choice="builder"
        )
        if inspect.isclass(factory) and issubclass(factory, GradientOfKineticEnergyProtocol):
            instance = factory(
                disc=self._disc, method=self.method, mode=self.mode
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of GradientOfKineticEnergyTermProtocol."
            )

    def init_grad_geopotential_term(self) -> PGF:
        factory = self.navi2term(
            group="grad_of_geopotential",
            choice="builder"
        )
        if inspect.isclass(factory) and issubclass(factory, PressureGradientForceTermProtocol):
            instance = factory(
                disc=self._disc, method=self.method, mode=self.mode
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of GradientOfGeopotentialTermProtocol."
            )


class ExVifShallowWaterEquationSet4UGRIDSpaceOperators(
    AbstractShallowWaterSpaceOperators[DM, PSO],
    Generic[
        DMF,
        NCT, GKE, PGF,
        DM, PSO
    ],
    ABC
):
    dgzdt: ExContinuityEqu[DMF, DM]
    dundt: ExNormalMomentumEqu4UGRID[
        NCT, GKE, PGF, DM
    ]

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: disc_types.EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.dgzdt = self.init_continuity_equation(disc=disc, method=method)
        self.dundt = self.init_momentum_equation(disc=disc, method=method)

    def init_continuity_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> ExContinuityEqu[DMF, DM]:
        return ExContinuityEqu(disc=disc, method=method, mode=self.mode)

    def init_momentum_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> ExNormalMomentumEqu4UGRID[
        NCT, GKE, PGF, DM
    ]:
        return ExNormalMomentumEqu4UGRID(disc=disc, method=method, mode=self.mode)


class ExVifShallowWaterEquationSet4UGRID(
    AbstractShallowWaterEquationSet[DM, SOP],
    AbstractExplicitODE,
    Generic[DM, SOP],
):

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        **kwargs,
    ):
        super().__init__(disc=disc, method=method, **kwargs)

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the dynamics set."""
        return AbstractShallowWaterEquationSet.alias_name()

    @staticmethod
    def form_name() -> str:
        """Return the form name of the dynamics set."""
        return "vif"

    @staticmethod
    def ode_kind_alias_name() -> str:
        """Return the ODE kind alias name of the dynamics set."""
        return "explicit"

    @overload
    def build_block_cls(
        self,
        coord: CoordinateSystem[
            SphericalCentroidVoronoiTessellations,
            OneLayerVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            OneLayerVertCoordLayout
        ],
    ) -> type[fvm_scvt_trisk_vars.Block]:
        ...

    @overload
    def build_block_cls(
        self,
        coord: CoordinateSystem[UHM, VM, HLO, VLO],
    ) -> type[disc_types.AbstractBlock]:
        ...

    def build_block_cls(
        self,
        coord: CoordinateSystem[UHM, VM, HLO, VLO],
    ) -> type[disc_types.AbstractBlock]:
        from terraflow.operator import FiniteVolumeMethod
        from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
        from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
        from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
        from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
        if (
            isinstance(self.method, FiniteVolumeMethod) and
            isinstance(coord.mesh, SphericalCentroidVoronoiTessellations) and
            isinstance(coord.vert, OneLayerVertCoordinate) and
            isinstance(coord.hori_layout, TriSKBasedCGridStaggeringLayout) and
            isinstance(coord.vert_layout, OneLayerVertCoordLayout)
        ):
            from terraflow.discretization.fvm.scvt.trisk.single_lev.dynamics_types import Block
            return Block
        else:
            raise NotImplementedError(
                "Unsupported combination of method and coordinate system for building block. "
            )

    def config2dict(self) -> Dict[str, Any]:
        cfg = self._space_operators.config2dict()
        self._cfg.update(cfg)
        return self._cfg

    def build_default(self) -> Self:
        cfg = model_config.ModelConfig()
        self.build_with_config(cfg=cfg)
        self._cfg = self.config2dict()
        return self

    def build_with_config(self, cfg: model_config.ModelConfig) -> Self:
        if not self.has_built():
            self._space_operators.build_with_config(cfg=cfg)
            self._cfg = cfg.to_dict()
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self._space_operators.prepare_block(block)
        log.logger.success("Prepared auxiliary and data.")
        return block

    def explicit_terms(self, advection: bool = False, *args, **kwargs) -> Callable:

        if not advection:
            def _wrap_dynamics(block: BLOCK, *args, **kwargs) -> BLOCK:
                return self.space_operators(
                    block=block,
                    *args,
                    **kwargs,
                    explicit=True
                )
            return _wrap_dynamics
        else:
            def _wrap_advection(
                block: BLOCK,
                *args,
                **kwargs
            ) -> BLOCK:
                return self.space_operators.advection_stepper(
                    block,
                    *args,
                    **kwargs,
                    explicit=True
                )
            return _wrap_advection


class ShallowWaterEquationBuilder:
    @overload
    @classmethod
    def build_with_config(
        cls,
        coord: CoordinateSystem[
            SphericalCentroidVoronoiTessellations,
            OneLayerVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            OneLayerVertCoordLayout
        ],
        para: PlanetParameterSet,
        method: FiniteVolumeMethod,
        form: Literal["vector_invariant"],
        ode_kind: Literal["explicit"],
        config: model_config.ModelConfig = model_config.ModelConfig(),
        output_h1_tend: bool = False,
        state_tile_depth: int = 3,
    ) -> ExVif_SWE_SCVT_TriSKCGrid_FVM:
        ...

    @overload
    @classmethod
    def build_with_config(
        cls,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        para: PlanetParameterSet,
        method: AbstractDiscretizationMethod,
        form: str,
        ode_kind: str,
        config: model_config.ModelConfig = model_config.ModelConfig(),
        *args,
        **kwargs,
    ) -> AbstractShallowWaterEquationSet:
        ...

    @classmethod
    def build_with_config(
        cls,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        para: PlanetParameterSet,
        method: AbstractDiscretizationMethod,
        form: str,
        ode_kind: str,
        config: model_config.ModelConfig = model_config.ModelConfig(),
        *args,
        **kwargs,
    ) -> AbstractShallowWaterEquationSet:
        equation = cls.from_model(
            coord=coord,
            para=para,
            method=method,
            form=form,
            ode_kind=ode_kind,
            *args,
            **kwargs
        )
        equation.build_with_config(cfg=config)
        return equation

    @overload
    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[
            SphericalCentroidVoronoiTessellations,
            OneLayerVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            OneLayerVertCoordLayout
        ],
        para: PlanetParameterSet,
        method: FiniteVolumeMethod,
        form: Literal["vector_invariant"],
        ode_kind: Literal["explicit"],
        output_h1_tend: bool = False,
        state_tile_depth: int = 3,
    ) -> ExVif_SWE_SCVT_TriSKCGrid_FVM:
        ...

    @overload
    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        para: PlanetParameterSet,
        method: AbstractDiscretizationMethod,
        form: str,
        ode_kind: str,
        *args,
        **kwargs,
    ) -> AbstractShallowWaterEquationSet:
        ...

    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        para: PlanetParameterSet,
        method: AbstractDiscretizationMethod,
        form: str,
        ode_kind: str,
        *args,
        **kwargs,
    ) -> AbstractShallowWaterEquationSet:
        if (ode_kind == "explicit" and form == "vector_invariant"):
            return ExVifShallowWaterEquationSet4UGRID.from_model(
                coord,
                para,
                method,
                *args,
                **kwargs
            )
        else:
            raise ValueError(
                f"Kwargs ode_kind={ode_kind} and form={form} not supported."
                "Please check the avaliable options."
            )


if TYPE_CHECKING:
    ExVif_SWE_SCVT_TriSKCGrid_FVM = ExVifShallowWaterEquationSet4UGRID[
        FiniteVolumeMethod,
        fvm_scvt_trisk_terms.ExSpaceOperators,
    ]
