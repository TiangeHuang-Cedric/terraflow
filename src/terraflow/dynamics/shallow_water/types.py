from __future__ import annotations
from abc import ABC, abstractmethod
from turtle import st
from typing import (
    Literal,
    Self,
    overload,
    TypeVar,
    Generic,
    Callable,
    TYPE_CHECKING,
)
import inspect

from terraflow.util import typing, log
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
    import terraflow.discretization.fvm.scvt.trisk.dynamics_types as fvm_scvt_trisk_vars
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


class ShallowWaterEquationMode(disc_types.AbstractEquationMode):
    def __init__(self, **kwargs):
        super().__init__(
            baroclinic=False,
            hydrostatic=False,
            nonhydrostatic=False,
            ideal_dry_core=False,
            advection2d=False,
            advection3d=False,
            **kwargs
        )


class AbstractShallowWaterEquationSet(
    disc_types.AbstractEquationSet[DM, ShallowWaterEquationMode, SOP],
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
            mode=ShallowWaterEquationMode(**kwargs)
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
    disc_types.AbstractDynamicalEquationForm[DM, ShallowWaterEquationMode],
    Generic[DM],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: ShallowWaterEquationMode
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )


class AbstractShallowWaterTerm(
    disc_types.AbstractDynamicalEquationTerm[DM, ShallowWaterEquationMode],
    Generic[DM],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: ShallowWaterEquationMode
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )


class AbstractShallowWaterSpaceOperators(
    disc_types.AbstractSpaceOperators[DM, ShallowWaterEquationMode, PSO],
    Generic[DM, PSO],
    ABC
):

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: ShallowWaterEquationMode
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
    pass


DMF = TypeVar("DMF", bound=DivergenceOfMassFluxProtocol)


class NonlinearCoriolisTermProtocol(
    AbstractShallowWaterTerm[DM],
    Generic[DM]
):
    pass


NCT = TypeVar("NCT", bound=NonlinearCoriolisTermProtocol)


class GradientOfKineticEnergyProtocol(
    AbstractShallowWaterTerm[DM],
    Generic[DM]
):
    pass


GKE = TypeVar("GKE", bound=GradientOfKineticEnergyProtocol)


class PressureGradientForceTermProtocol(
    AbstractShallowWaterTerm[DM],
    Generic[DM]
):
    pass


PGF = TypeVar("PGF", bound=PressureGradientForceTermProtocol)


class ExContinuityEqu(
    AbstractShallowWaterEquationForm[DM],
    disc_types.AbstractContinuityEquForm[DM, ShallowWaterEquationMode],
    Generic[DMF, DM],
    ABC
):
    dmf: DMF

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: ShallowWaterEquationMode
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
        return "continuity_equ"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def build_default(self) -> Self:
        if not self.has_built():
            self.dmf.build_default()
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
    disc_types.AbstractMomentumEquForm[DM, ShallowWaterEquationMode],
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
        mode: ShallowWaterEquationMode
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
        return "normal_momentum_equ"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def build_default(self) -> Self:
        if not self._built:
            self.coriolis.build_default()
            self.grad_ke.build_default()
            self.pgf.build_default()
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
        mode: ShallowWaterEquationMode
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
            from terraflow.discretization.fvm.scvt.trisk.dynamics_types import Block
            return Block
        else:
            raise NotImplementedError(
                "Unsupported combination of method and coordinate system for building block. "
            )

    def build_default(self) -> Self:
        if not self.has_built():
            self._space_operators.build_default()
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
                batch: disc_types.AbstractAdvBatchType,
                block: BLOCK,
                *args,
                **kwargs
            ) -> tuple[disc_types.AbstractAdvBatchType, BLOCK]:
                return self.space_operators.advection_space_operators(
                    batch,
                    block,
                    *args,
                    **kwargs,
                    explicit=True
                )
            return _wrap_advection


class ShallowWaterEquationBuilder:
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
