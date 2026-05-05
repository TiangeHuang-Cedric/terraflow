from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, TypeVar, Generic, Self, TYPE_CHECKING
import inspect

from terraflow.mesh.abstract import AbstractMesh, AbstractUnstructuredMesh
from terraflow.vert_coord.abstract import AbstractVertCoordinate
from terraflow.layout.abstract import AbstractLayout
from terraflow.operator import AbstractDiscretizationMethod
import terraflow.discretization.abstract as disc_types

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.util.model_config import ModelConfig
    Int = typing.Int
    Float = typing.Float
    Array = typing.Array

HM = TypeVar("HM", bound=AbstractMesh)
UHM = TypeVar("UHM", bound=AbstractUnstructuredMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
HLO = TypeVar("HLO", bound=AbstractLayout)
VLO = TypeVar("VLO", bound=AbstractLayout)
DM = TypeVar("DM", bound=AbstractDiscretizationMethod)
BLOCK = TypeVar("BLOCK", bound=disc_types.AbstractBlock)
STATE = TypeVar("STATE", bound=disc_types.AbstractFields)
PSO = TypeVar("PSO", bound=disc_types.AbstractPrepareSpaceOperators)
SOP = TypeVar("SOP", bound=disc_types.AbstractSpaceOperators)
BEM = TypeVar("BEM", bound="AbstractBaroclinicEquationMode")
HEM = TypeVar("HEM", bound="AbstractHydrostaticEquationMode")


class AbstractBaroclinicEquationMode(
    disc_types.AbstractEquationMode,
    ABC
):
    def __init__(
        self,
        hydrostatic: bool = False,
        nonhydrostatic: bool = False,
        ideal_dry_core: bool = False,
        **kwargs
    ):
        super().__init__(
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection2d=False,
            advection3d=False,
            **kwargs
        )


class AbstractBaroclinicEquationSet(
    disc_types.AbstractEquationSet[DM, BEM, SOP],
    Generic[DM, BEM, SOP],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: BEM,
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )
        factory = self.navi2space_ops()
        space_operators: SOP = factory(
            disc=disc,
            method=method,
            mode=mode
        )
        self._space_operators = space_operators

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        raise NotImplementedError


class AbstractHydrostaticEquationMode(
    AbstractBaroclinicEquationMode,
    ABC
):
    def __init__(self, ideal_dry_core: bool = False, **kwargs):
        super().__init__(
            hydrostatic=True,
            nonhydrostatic=False,
            ideal_dry_core=ideal_dry_core,
            advection=False,
            **kwargs
        )


class AbstractHydrostaticEquationSet(
    AbstractBaroclinicEquationSet[DM, HEM, SOP],
    Generic[DM, HEM, SOP],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: HEM,
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        raise NotImplementedError


class DryHydrostaticEquationMode(
    AbstractHydrostaticEquationMode,
    ABC
):
    def __init__(self, **kwargs):
        super().__init__(ideal_dry_core=True, **kwargs)


class AbstractDryHydrostaticEquationSet(
    AbstractHydrostaticEquationSet[DM, DryHydrostaticEquationMode, SOP],
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
            disc=disc, method=method, mode=DryHydrostaticEquationMode(**kwargs)
        )

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        return "dry_hdc"


class AbstractBaroclinicEquationTerm(
    disc_types.AbstractDynamicalEquationTerm[DM, BEM],
    Generic[DM, BEM],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: BEM
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )


class AbstractBaroclinicEquationForm(
    disc_types.AbstractDynamicalEquationForm[DM, BEM],
    Generic[DM, BEM],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: BEM
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )


class AbstractBaroclinicSpaceOperators(
    disc_types.AbstractSpaceOperators[DM, BEM, PSO],
    Generic[DM, BEM, PSO],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: BEM
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
        method: DM,
    ) -> disc_types.AbstractContinuityEquForm:
        raise NotImplementedError

    @abstractmethod
    def init_momentum_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> disc_types.AbstractMomentumEquForm:
        raise NotImplementedError

    @abstractmethod
    def init_thermodynamic_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> disc_types.AbstractThermodynamicEquForm:
        raise NotImplementedError


class DivergenceOfMassFluxAtSurfaceProtocol(
    AbstractBaroclinicEquationTerm[DM, BEM],
    disc_types.AbstractEquationTerm[DM, BEM],
    Generic[DM, BEM]
):
    @staticmethod
    def alias_name() -> str:
        return "dmfs"


DMFS = TypeVar("DMFS", bound=DivergenceOfMassFluxAtSurfaceProtocol)


class DivergenceOfThermodynamicMassFluxProtocol(
    AbstractBaroclinicEquationTerm[DM, BEM],
    disc_types.AbstractEquationTerm[DM, BEM],
    Generic[DM, BEM]
):
    @staticmethod
    def alias_name() -> str:
        return "dmft"


DMFT = TypeVar("DMFT", bound=DivergenceOfThermodynamicMassFluxProtocol)


class VerticalAdvectionOfThermodynamicTermProtocol(
    AbstractBaroclinicEquationTerm[DM, BEM],
    disc_types.AbstractEquationTerm[DM, BEM],
    Generic[DM, BEM]
):
    @staticmethod
    def alias_name() -> str:
        return "vadv"


VAT = TypeVar("VAT", bound=VerticalAdvectionOfThermodynamicTermProtocol)


class NonlinearCoriolisTermProtocol(
    AbstractBaroclinicEquationTerm[DM, BEM],
    disc_types.AbstractEquationTerm[DM, BEM],
    Generic[DM, BEM]
):
    @staticmethod
    def alias_name() -> str:
        return "nct"


NCT = TypeVar("NCT", bound=NonlinearCoriolisTermProtocol)


class GradientOfKineticEnergyProtocol(
    AbstractBaroclinicEquationTerm[DM, BEM],
    disc_types.AbstractEquationTerm[DM, BEM],
    Generic[DM, BEM]
):
    @staticmethod
    def alias_name() -> str:
        return "grad_ke"


GKE = TypeVar("GKE", bound=GradientOfKineticEnergyProtocol)


class VerticalAdvectionOfMomentumTermProtocol(
    AbstractBaroclinicEquationTerm[DM, BEM],
    disc_types.AbstractEquationTerm[DM, BEM],
    Generic[DM, BEM]
):
    @staticmethod
    def alias_name() -> str:
        return "vadv"


VAM = TypeVar("VAM", bound=VerticalAdvectionOfMomentumTermProtocol)


class PressureGradientForceTermProtocol(
    AbstractBaroclinicEquationTerm[DM, BEM],
    disc_types.AbstractEquationTerm[DM, BEM],
    Generic[DM, BEM]
):
    @staticmethod
    def alias_name() -> str:
        return "pgf"


PGF = TypeVar("PGF", bound=PressureGradientForceTermProtocol)


class ExContinuityEqu(
    AbstractBaroclinicEquationForm[DM, BEM],
    disc_types.AbstractContinuityEquForm[DM, BEM],
    Generic[DMFS, DM, BEM],
    ABC
):
    div_mfs: DMFS

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: BEM
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )
        self.div_mfs = self.init_div_mfs()

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> BLOCK:
        return self.div_mfs(block, star_idx, dt, substep)

    @staticmethod
    def alias_name() -> str:
        return "dmgsdt"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.div_mfs.config2dict()
        self._update_config(cfg=cfg)
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.div_mfs.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.div_mfs.prepare_block(block)
        return block

    def init_div_mfs(self) -> DMFS:
        factory = self.navi2term(
            group="div_of_mfs",
            choice="builder"
        )
        origin = getattr(factory, '__origin__', factory)
        if inspect.isclass(origin) and issubclass(origin, DivergenceOfMassFluxAtSurfaceProtocol):
            instance = factory(
                disc=self._disc,
                method=self.method,
                mode=self.mode
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of DivergenceOfMassFluxAtSurfaceProtocol."
            )


class ExThermodynamicEqu(
    AbstractBaroclinicEquationForm[DM, BEM],
    disc_types.AbstractThermodynamicEquForm[DM, BEM],
    Generic[DMFT, VAT, DM, BEM],
    ABC
):
    div_mft: DMFT
    adv_vert: VAT

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: BEM
    ) -> None:
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )
        self.div_mft = self.init_div_mft()
        self.adv_vert = self.init_adv_vert()

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> BLOCK:
        block = self.div_mft(block, star_idx, dt, substep)
        block = self.adv_vert(block, star_idx, dt, substep)
        return block

    @staticmethod
    def alias_name() -> str:
        return "dptdt"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.div_mft.config2dict()
        self._update_config(cfg=cfg)
        cfg = self.adv_vert.config2dict()
        self._update_config(cfg=cfg)
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.div_mft.build_with_config(cfg=cfg)
            self.adv_vert.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.div_mft.prepare_block(block)
        block = self.adv_vert.prepare_block(block)
        return block

    def init_div_mft(self) -> DMFT:
        factory = self.navi2term(
            group="div_of_mft",
            choice="builder"
        )
        origin = getattr(factory, '__origin__', factory)
        if inspect.isclass(origin) and issubclass(origin, DivergenceOfThermodynamicMassFluxProtocol):
            instance = factory(
                disc=self._disc,
                method=self.method,
                mode=self.mode
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of DivergenceOfThermodynamicMassFluxProtocol."
            )

    def init_adv_vert(self) -> VAT:
        factory = self.navi2term(
            group="vert_adv_th",
            choice="builder"
        )
        origin = getattr(factory, '__origin__', factory)
        if inspect.isclass(origin) and issubclass(origin, VerticalAdvectionOfThermodynamicTermProtocol):
            instance = factory(
                disc=self._disc,
                method=self.method,
                mode=self.mode
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of VerticalAdvectionOfThermodynamicTermProtocol."
            )


class ExNormalMomentumEqu4UGRID(
    AbstractBaroclinicEquationForm[DM, BEM],
    disc_types.AbstractMomentumEquForm[DM, BEM],
    Generic[NCT, GKE, VAM, PGF, DM, BEM],
    ABC
):
    coriolis: NCT
    grad_ke: GKE
    adv_vert: VAM
    pgf: PGF

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: BEM
    ) -> None:
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )
        self.coriolis = self.init_coriolis()
        self.grad_ke = self.init_grad_ke()
        self.adv_vert = self.init_vertical_advection()
        self.pgf = self.init_pgf()

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> BLOCK:
        block = self.coriolis(block, star_idx, dt, substep)
        block = self.grad_ke(block, star_idx, dt, substep)
        block = self.adv_vert(block, star_idx, dt, substep)
        block = self.pgf(block, star_idx, dt, substep)
        return block

    @staticmethod
    def alias_name() -> str:
        return "dundt"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.coriolis.config2dict()
        self._update_config(cfg=cfg)
        cfg = self.grad_ke.config2dict()
        self._update_config(cfg=cfg)
        cfg = self.adv_vert.config2dict()
        self._update_config(cfg=cfg)
        cfg = self.pgf.config2dict()
        self._update_config(cfg=cfg)
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.coriolis.build_with_config(cfg=cfg)
            self.grad_ke.build_with_config(cfg=cfg)
            self.adv_vert.build_with_config(cfg=cfg)
            self.pgf.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.coriolis.prepare_block(block)
        block = self.grad_ke.prepare_block(block)
        block = self.adv_vert.prepare_block(block)
        block = self.pgf.prepare_block(block)
        return block

    def init_coriolis(self) -> NCT:
        factory = self.navi2term(
            group="nonlinear_coriolis",
            choice="builder"
        )
        origin = getattr(factory, '__origin__', factory)
        if inspect.isclass(origin) and issubclass(origin, NonlinearCoriolisTermProtocol):
            instance = factory(
                disc=self._disc, method=self.method, mode=self.mode
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of NonlinearCoriolisTermProtocol."
            )

    def init_grad_ke(self) -> GKE:
        factory = self.navi2term(
            group="grad_of_kinetic_energy",
            choice="builder"
        )
        origin = getattr(factory, '__origin__', factory)
        if inspect.isclass(origin) and issubclass(origin, GradientOfKineticEnergyProtocol):
            instance = factory(
                disc=self._disc, method=self.method, mode=self.mode)
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of GradientOfKineticEnergyTermProtocol."
            )

    def init_vertical_advection(self) -> VAM:
        factory = self.navi2term(
            group="vert_adv",
            choice="builder"
        )
        origin = getattr(factory, '__origin__', factory)
        if inspect.isclass(origin) and issubclass(origin, VerticalAdvectionOfMomentumTermProtocol):
            instance = factory(
                disc=self._disc, method=self.method, mode=self.mode)
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of VerticalAdvectionOfMomentumTermProtocol."
            )

    def init_pgf(self) -> PGF:
        factory = self.navi2term(
            group="pgf",
            choice="builder"
        )
        origin = getattr(factory, '__origin__', factory)
        if inspect.isclass(origin) and issubclass(origin, PressureGradientForceTermProtocol):
            instance = factory(
                disc=self._disc, method=self._method, mode=self.mode)
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of PressureGradientForceTermProtocol."
            )


class ExVifBaroclinicEquationSet4UGRIDSpaceOperators(
    AbstractBaroclinicSpaceOperators[DM, BEM, PSO],
    Generic[
        DMFS,
        DMFT, VAT,
        NCT, GKE, VAM, PGF,
        DM, BEM, PSO,
    ],
):

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: BEM
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.dmgsdt = self.init_continuity_equation(
            disc=disc,
            method=method,
        )
        self.dptdt = self.init_thermodynamic_equation(
            disc=disc,
            method=method,
        )
        self.dundt = self.init_momentum_equation(
            disc=disc,
            method=method,
        )

    def init_continuity_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> ExContinuityEqu[DMFS, DM, BEM]:
        return ExContinuityEqu(disc=disc, method=method, mode=self.mode)

    def init_thermodynamic_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> ExThermodynamicEqu[DMFT, VAT, DM, BEM]:
        return ExThermodynamicEqu(disc=disc, method=method, mode=self.mode)

    def init_momentum_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> ExNormalMomentumEqu4UGRID[NCT, GKE, VAM, PGF, DM, BEM]:
        return ExNormalMomentumEqu4UGRID(disc=disc, method=method, mode=self.mode)
