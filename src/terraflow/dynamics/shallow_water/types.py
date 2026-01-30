from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal
from typing import overload, TYPE_CHECKING, TypeVar, Generic
import inspect
import equinox as eqx

from terraflow.util import typing
from terraflow.planet import PlanetParameterSet
from terraflow.operator import AbstractDiscretizationMethod
from terraflow.mesh import AbstractMesh, AbstractUnstructuredMesh
from terraflow.vert_coord import AbstractVertCoordinate
from terraflow.layout import AbstractLayout, CoordinateSystem
from terraflow.solver.abstract import AbstractExplicitODE
import terraflow.discretization.abstract as disc_types

if TYPE_CHECKING:
    from terraflow.operator import FiniteVolumeMethod
    from terraflow.mesh import OrthogonalDualMesh
    from terraflow.vert_coord import OneLayerVertCoordinate
    from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
    import terraflow.discretization.fvm.odm.trisk.single_lev.term as fvm_odm_trisk_1lev_builder


HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
LO = TypeVar("LO", bound=AbstractLayout)
DM = TypeVar("DM", bound=AbstractDiscretizationMethod)
UHM = TypeVar("UHM", bound=AbstractUnstructuredMesh)
BLOCK = TypeVar("BLOCK", bound=disc_types.AbstractBlock)
ESAUX = TypeVar("ESAUX", bound=disc_types.AbstractEquationSetAuxiliaryTerms)

Array = typing.Array


class AbstractShallowWaterEquationSet(
    disc_types.AbstractEquationSet[DM, ESAUX],
    Generic[DM, ESAUX],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ):
        super().__init__(disc=disc, method=method)

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the dynamics set."""
        return "swe"

    @abstractmethod
    def init_continuity_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        aux: ESAUX
    ) -> disc_types.AbstractContinuityEquForm:
        raise NotImplementedError

    @abstractmethod
    def init_momentum_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        aux: ESAUX
    ) -> disc_types.AbstractMomentumEquForm:
        raise NotImplementedError


ShallowWaterEquationSet = TypeVar(
    "ShallowWaterEquationSet",
    bound=AbstractShallowWaterEquationSet
)


class DivergenceOfMassFluxProtocol(
    disc_types.AbstractEquationTerm[DM, ESAUX],
    Generic[DM, ESAUX]
):
    pass


DMF = TypeVar("DMF", bound=DivergenceOfMassFluxProtocol)


class NonlinearCoriolisTermProtocol(
    disc_types.AbstractEquationTerm[DM, ESAUX],
    Generic[DM, ESAUX]
):
    pass


NCT = TypeVar("NCT", bound=NonlinearCoriolisTermProtocol)


class GradientOfKineticEnergyProtocol(
    disc_types.AbstractEquationTerm[DM, ESAUX],
    Generic[DM, ESAUX]
):
    pass


GKE = TypeVar("GKE", bound=GradientOfKineticEnergyProtocol)


class PressureGradientForceTermProtocol(
    disc_types.AbstractEquationTerm[DM, ESAUX],
    Generic[DM, ESAUX]
):
    pass


PGF = TypeVar("PGF", bound=PressureGradientForceTermProtocol)


class ExContinuityEqu(
    disc_types.AbstractContinuityEquForm[DM, ESAUX],
    Generic[DMF, DM, ESAUX],
    ABC
):
    div_mfx: DMF

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        aux: ESAUX
    ):
        super().__init__(disc, method, aux)
        self.div_mfx = self.init_div_mfx()

    def __call__(self, block: BLOCK) -> BLOCK:
        block = self.div_mfx(block)
        dzdt_mfn_e = block.tend.dzdt_mfn_e.unwrap()
        return eqx.tree_at(
            lambda b: b.tend.dzdt.data,
            block,
            dzdt_mfn_e
        )

    @staticmethod
    def alias_name() -> str:
        return "continuity_equ"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def init_div_mfx(self) -> DMF:
        factory = self.navi2term(
            group="div_of_mfx",
            choice="builder"
        )
        origin = getattr(factory, '__origin__', factory)
        if inspect.isclass(origin) and issubclass(origin, DivergenceOfMassFluxProtocol):
            instance = factory(
                disc=self._disc,
                method=self.method,
                aux=self.aux
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of DivergenceOfMassFluxProtocol."
            )


class ExNormalMomentumEqu4UGRID(
    disc_types.AbstractMomentumEquForm[DM, ESAUX],
    Generic[NCT, GKE, PGF, DM, ESAUX],
    ABC
):
    coriolis: NCT
    grad_ke: GKE
    pgf: PGF

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        aux: ESAUX
    ) -> None:
        super().__init__(disc, method, aux)
        self.coriolis = self.init_nonlinear_coriolis_term()
        self.grad_ke = self.init_grad_kinetic_energy_term()
        self.pgf = self.init_grad_geopotential_term()

    def __call__(self, block: BLOCK) -> BLOCK:
        block = self.coriolis(block)
        block = self.grad_ke(block)
        block = self.pgf(block)
        dundt_coriolis = block.tend.dundt_coriolis.unwrap()
        dundt_grad_ke = block.tend.dundt_grad_ke.unwrap()
        dundt_pgf = block.tend.dundt_pgf.unwrap()
        return eqx.tree_at(
            lambda b: b.tend.dundt.data,
            block,
            replace=(
                dundt_coriolis
                + dundt_grad_ke
                + dundt_pgf
            )
        )

    @staticmethod
    def alias_name() -> str:
        return "normal_momentum_equ"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def init_nonlinear_coriolis_term(self) -> NCT:
        factory = self.navi2term(
            group="nonlinear_coriolis",
            choice="builder"
        )
        if inspect.isclass(factory) and issubclass(factory, NonlinearCoriolisTermProtocol):
            instance = factory(
                disc=self._disc,
                method=self.method,
                aux=self.aux
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
                disc=self._disc,
                method=self.method,
                aux=self.aux
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
                disc=self._disc,
                method=self._method,
                aux=self.aux
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of GradientOfGeopotentialTermProtocol."
            )


class ExVifShallowWaterEquationSet4UGRID(
    AbstractShallowWaterEquationSet[DM, ESAUX],
    AbstractExplicitODE,
    Generic[
        DMF,
        NCT, GKE, PGF,
        ESAUX,
        DM
    ],
):
    dzdt: ExContinuityEqu[DMF, DM, ESAUX]
    dundt: ExNormalMomentumEqu4UGRID[
        NCT, GKE, PGF, DM, ESAUX
    ]

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ):
        super().__init__(disc=disc, method=method)
        self.dzdt = self.init_continuity_equation(
            disc=disc,
            method=method,
            aux=self.aux
        )
        self.dundt = self.init_momentum_equation(
            disc=disc,
            method=method,
            aux=self.aux
        )

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

    def init_continuity_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        aux: ESAUX
    ) -> ExContinuityEqu[DMF, DM, ESAUX]:
        return ExContinuityEqu(
            disc=disc,
            method=method,
            aux=aux
        )

    def init_momentum_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        aux: ESAUX
    ) -> ExNormalMomentumEqu4UGRID[
        NCT, GKE, PGF, DM, ESAUX
    ]:
        return ExNormalMomentumEqu4UGRID(
            disc=disc,
            method=method,
            aux=aux
        )

    def explicit_terms(self, block: BLOCK, pass_tag: str = "all_pass") -> BLOCK:
        block = self.aux(block, pass_tag=pass_tag)
        if pass_tag == "all_pass":
            block = self.dzdt(block)
            block = self.dundt(block)
        elif pass_tag == "forward_pass":
            block = self.dzdt(block)
        elif pass_tag == "backward_pass":
            block = self.dundt(block)
        else:
            raise ValueError(f"Unknown pass_tag: {pass_tag}")
        return block


class ShallowWaterEquationBuilder:
    @overload
    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[
            OrthogonalDualMesh,
            OneLayerVertCoordinate,
            TriSKBasedCGridStaggeringLayout
        ],
        para: PlanetParameterSet,
        method: FiniteVolumeMethod,
        form: Literal["vector_invariant"],
        ode_kind: Literal["explicit"],
    ) -> ExVif_SWE_ODM_TriSKCGrid_FVM:
        ...

    @overload
    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[HM, VM, LO],
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
        coord: CoordinateSystem[HM, VM, LO],
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
    ExVif_SWE_ODM_TriSKCGrid_FVM = ExVifShallowWaterEquationSet4UGRID[
        fvm_odm_trisk_1lev_builder.DivergenceOfMassFluxAtPrimalCell,
        fvm_odm_trisk_1lev_builder.NonlinearCoriolisTermAtDualEdge,
        fvm_odm_trisk_1lev_builder.GradientOfKineticEnergyAtDualEdge,
        fvm_odm_trisk_1lev_builder.PressureGradientForceAtDualEdge,
        fvm_odm_trisk_1lev_builder.AuxiliaryTerms,
        FiniteVolumeMethod,
    ]
