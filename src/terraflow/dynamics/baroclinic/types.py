from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Self
import inspect
import equinox as eqx

from terraflow.mesh.abstract import AbstractMesh, AbstractUnstructuredMesh
from terraflow.vert_coord.abstract import AbstractVertCoordinate
from terraflow.layout.abstract import AbstractLayout
from terraflow.operator import AbstractDiscretizationMethod
import terraflow.discretization.abstract as disc_types


HM = TypeVar("HM", bound=AbstractMesh)
UHM = TypeVar("UHM", bound=AbstractUnstructuredMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
HLO = TypeVar("HLO", bound=AbstractLayout)
VLO = TypeVar("VLO", bound=AbstractLayout)
DM = TypeVar("DM", bound=AbstractDiscretizationMethod)
BLOCK = TypeVar("BLOCK", bound=disc_types.AbstractBlock)
PSO = TypeVar("PSO", bound=disc_types.AbstractPrepareSpaceOperators)
SOP = TypeVar("SOP", bound=disc_types.AbstractSpaceOperators)


class AbstractBaroclinicEquationSet(
    disc_types.AbstractEquationSet[DM, SOP],
    Generic[DM, SOP],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        factory = self.navi2space_ops()
        space_operators: SOP = factory(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection,
        )
        self._space_operators = space_operators

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        raise NotImplementedError


class AbstractHydrostaticEquationSet(
    AbstractBaroclinicEquationSet[DM, SOP],
    Generic[DM, SOP],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        ideal_dry_core: bool,
    ):
        super().__init__(
            disc=disc,
            method=method,
            hydrostatic=True,
            nonhydrostatic=False,
            ideal_dry_core=ideal_dry_core,
            advection=False
        )

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        raise NotImplementedError


class AbstractDryHydrostaticEquationSet(
    AbstractHydrostaticEquationSet[DM, SOP],
    Generic[DM, SOP],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM
    ):
        super().__init__(disc=disc, method=method, ideal_dry_core=True)

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        return "dry_hdc"


class AbstractBaroclinicEquationTerm(
    disc_types.AbstractEquationTerm[DM],
    Generic[DM],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )


class AbstractBaroclinicEquationForm(
    disc_types.AbstractEquationForm[DM],
    Generic[DM],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )


class AbstractBaroclinicSpaceOperators(
    disc_types.AbstractSpaceOperators[DM, PSO],
    Generic[DM, PSO],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
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
    AbstractBaroclinicEquationTerm[DM],
    disc_types.AbstractEquationTerm[DM],
    Generic[DM]
):
    pass


DMFS = TypeVar("DMFS", bound=DivergenceOfMassFluxAtSurfaceProtocol)


class DivergenceOfThermodynamicMassFluxProtocol(
    AbstractBaroclinicEquationTerm[DM],
    disc_types.AbstractEquationTerm[DM],
    Generic[DM]
):
    pass


DMFT = TypeVar("DMFT", bound=DivergenceOfThermodynamicMassFluxProtocol)


class VerticalAdvectionOfThermodynamicTermProtocol(
    AbstractBaroclinicEquationTerm[DM],
    disc_types.AbstractEquationTerm[DM],
    Generic[DM]
):
    pass


VAT = TypeVar("VAT", bound=VerticalAdvectionOfThermodynamicTermProtocol)


class NonlinearCoriolisTermProtocol(
    AbstractBaroclinicEquationTerm[DM],
    disc_types.AbstractEquationTerm[DM],
    Generic[DM]
):
    pass


NCT = TypeVar("NCT", bound=NonlinearCoriolisTermProtocol)


class GradientOfKineticEnergyProtocol(
    AbstractBaroclinicEquationTerm[DM],
    disc_types.AbstractEquationTerm[DM],
    Generic[DM]
):
    pass


GKE = TypeVar("GKE", bound=GradientOfKineticEnergyProtocol)


class VerticalAdvectionOfMomentumTermProtocol(
    AbstractBaroclinicEquationTerm[DM],
    disc_types.AbstractEquationTerm[DM],
    Generic[DM]
):
    pass


VAM = TypeVar("VAM", bound=VerticalAdvectionOfMomentumTermProtocol)


class PressureGradientForceTermProtocol(
    AbstractBaroclinicEquationTerm[DM],
    disc_types.AbstractEquationTerm[DM],
    Generic[DM]
):
    pass


PGF = TypeVar("PGF", bound=PressureGradientForceTermProtocol)


class ExContinuityEqu(
    AbstractBaroclinicEquationForm[DM],
    disc_types.AbstractContinuityEquForm[DM],
    Generic[DMFS, DM],
    ABC
):
    div_mfs: DMFS

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.div_mfs = self.init_div_mfs()

    def __call__(self, block: BLOCK) -> BLOCK:
        block = self.div_mfs(block)
        return eqx.tree_at(
            lambda b: b.tend.dmgsdt.data,
            block,
            replace=self.div_mfs.unwrap(block).unwrap()
        )

    @staticmethod
    def alias_name() -> str:
        return "continuity_equ"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def build_default(self) -> Self:
        if not self.has_built():
            self.div_mfs.build_default()
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
                hydrostatic=self.hydrostatic,
                nonhydrostatic=self.nonhydrostatic,
                ideal_dry_core=self.ideal_dry_core,
                advection=self.advection
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of DivergenceOfMassFluxAtSurfaceProtocol."
            )


class ExThermodynamicEqu(
    AbstractBaroclinicEquationForm[DM],
    disc_types.AbstractThermodynamicEquForm[DM],
    Generic[DMFT, VAT, DM],
    ABC
):
    div_mft: DMFT
    adv_vert: VAT

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ) -> None:
        super().__init__(
            disc=disc,
            method=method,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.div_mft = self.init_div_mft()
        self.adv_vert = self.init_adv_vert()

    def __call__(self, block: BLOCK) -> BLOCK:
        block = self.div_mft(block)
        block = self.adv_vert(block)
        return eqx.tree_at(
            lambda b: b.tend.dptdt.data,
            block,
            replace=(
                self.div_mft.unwrap(block).unwrap()
                + self.adv_vert.unwrap(block).unwrap()
            )
        )

    @staticmethod
    def alias_name() -> str:
        return "thermodynamic_equ"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def build_default(self) -> Self:
        if not self.has_built():
            self.div_mft.build_default()
            self.adv_vert.build_default()
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
                hydrostatic=self.hydrostatic,
                nonhydrostatic=self.nonhydrostatic,
                ideal_dry_core=self.ideal_dry_core,
                advection=self.advection
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
                hydrostatic=self.hydrostatic,
                nonhydrostatic=self.nonhydrostatic,
                ideal_dry_core=self.ideal_dry_core,
                advection=self.advection
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of VerticalAdvectionOfThermodynamicTermProtocol."
            )


class ExNormalMomentumEqu4UGRID(
    AbstractBaroclinicEquationForm[DM],
    disc_types.AbstractMomentumEquForm[DM],
    Generic[NCT, GKE, VAM, PGF, DM],
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
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ) -> None:
        super().__init__(
            disc=disc,
            method=method,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.coriolis = self.init_coriolis()
        self.grad_ke = self.init_grad_ke()
        self.adv_vert = self.init_vertical_advection()
        self.pgf = self.init_pgf()

    def __call__(self, block: BLOCK) -> BLOCK:
        block = self.coriolis(block)
        block = self.grad_ke(block)
        block = self.adv_vert(block)
        block = self.pgf(block)
        return eqx.tree_at(
            lambda b: b.tend.dundt.data,
            block,
            replace=(
                self.coriolis.unwrap(block).unwrap()
                + self.grad_ke.unwrap(block).unwrap()
                + self.adv_vert.unwrap(block).unwrap()
                + self.pgf.unwrap(block).unwrap()
            )
        )

    @staticmethod
    def alias_name() -> str:
        return "normal_momentum_equ"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def build_default(self) -> Self:
        if not self.has_built():
            self.coriolis.build_default()
            self.grad_ke.build_default()
            self.adv_vert.build_default()
            self.pgf.build_default()
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
                disc=self._disc,
                method=self.method,
                hydrostatic=self.hydrostatic,
                nonhydrostatic=self.nonhydrostatic,
                ideal_dry_core=self.ideal_dry_core,
                advection=self.advection
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
                disc=self._disc,
                method=self.method,
                hydrostatic=self.hydrostatic,
                nonhydrostatic=self.nonhydrostatic,
                ideal_dry_core=self.ideal_dry_core,
                advection=self.advection
            )
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
                disc=self._disc,
                method=self.method,
                hydrostatic=self.hydrostatic,
                nonhydrostatic=self.nonhydrostatic,
                ideal_dry_core=self.ideal_dry_core,
                advection=self.advection
            )
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
                disc=self._disc,
                method=self._method,
                hydrostatic=self.hydrostatic,
                nonhydrostatic=self.nonhydrostatic,
                ideal_dry_core=self.ideal_dry_core,
                advection=self.advection
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of PressureGradientForceTermProtocol."
            )


class ExVifBaroclinicEquationSet4UGRIDSpaceOperators(
    AbstractBaroclinicSpaceOperators[DM, PSO],
    Generic[
        DMFS,
        DMFT, VAT,
        NCT, GKE, VAM, PGF,
        DM, PSO,
    ],
):

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
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
    ) -> ExContinuityEqu[DMFS, DM,]:
        return ExContinuityEqu(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )

    def init_thermodynamic_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> ExThermodynamicEqu[DMFT, VAT, DM,
                            ]:
        return ExThermodynamicEqu(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )

    def init_momentum_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> ExNormalMomentumEqu4UGRID[NCT, GKE, VAM, PGF, DM]:
        return ExNormalMomentumEqu4UGRID(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self.prepare.build_default()
            self.dmgsdt.build_default()
            self.dptdt.build_default()
            self.dundt.build_default()
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.prepare.prepare_block(block)
        block = self.dmgsdt.prepare_block(block)
        block = self.dptdt.prepare_block(block)
        block = self.dundt.prepare_block(block)
        return block
