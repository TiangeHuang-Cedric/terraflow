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
from terraflow.operator import AbstractDiscretizationMethod, FiniteVolumeMethod
from terraflow.mesh.abstract import AbstractMesh, AbstractUnstructuredMesh
from terraflow.vert_coord.abstract import AbstractVertCoordinate
from terraflow.layout.abstract import AbstractLayout, CoordinateSystem
from terraflow.solver.abstract import AbstractExplicitODE
import terraflow.discretization.abstract as disc_types

if TYPE_CHECKING:
    from terraflow.operator import FiniteVolumeMethod
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
    import terraflow.discretization.fvm.scvt.trisk.dynamics_types as fvm_scvt_trisk_vars
    import terraflow.discretization.fvm.scvt.trisk.single_lev.adv as fvm_scvt_trisk_1lev_adv_terms
    import terraflow.discretization.fvm.scvt.trisk.hybrid_lev.adv as fvm_scvt_trisk_hybrid_adv_terms

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
ADVEM = TypeVar("ADVEM", "AdvectionEquation2DMode", "AdvectionEquation3DMode")


class AdvectionEquation2DMode(disc_types.AbstractEquationMode):
    def __init__(self, **kwargs):
        super().__init__(
            baroclinic=False,
            hydrostatic=False,
            nonhydrostatic=False,
            advection2d=True,
            advection3d=False,
            **kwargs
        )


class AdvectionEquation3DMode(disc_types.AbstractEquationMode):
    def __init__(self, **kwargs):
        super().__init__(
            baroclinic=False,
            hydrostatic=False,
            nonhydrostatic=False,
            advection2d=False,
            advection3d=True,
            **kwargs
        )


class AbstractAdvectionEquationSet2D(
    disc_types.AbstractEquationSet[DM, AdvectionEquation2DMode, SOP],
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
            mode=AdvectionEquation2DMode(**kwargs),
        )
        factory = self.navi2space_ops()
        space_operators: SOP = factory(
            disc=disc,
            method=method,
            mode=self.mode,
        )
        self._space_operators = space_operators

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the dynamics set."""
        return "adv"


class AbstractAdvectionEquationSet3D(
    disc_types.AbstractEquationSet[DM, AdvectionEquation3DMode, SOP],
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
            mode=AdvectionEquation3DMode(**kwargs),
        )
        factory = self.navi2space_ops()
        space_operators: SOP = factory(
            disc=disc,
            method=method,
            mode=self.mode,
        )
        self._space_operators = space_operators

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the dynamics set."""
        return "adv"


class AbstractDynamicalEquationForm(
    disc_types.AbstractDynamicalEquationForm[DM, ADVEM],
    Generic[DM, ADVEM],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: ADVEM,
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode,
        )


class AbstractAdvectionEquationForm(
    disc_types.AbstractAdvectionEquationForm[DM, ADVEM],
    Generic[DM, ADVEM],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: ADVEM,
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode,
        )


class AbstractDynamicalEquationTerm(
    disc_types.AbstractDynamicalEquationTerm[DM, ADVEM],
    Generic[DM, ADVEM],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: ADVEM,
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode,
        )


class AbstractAdvectionEquationTerm(
    disc_types.AbstractAdvectionEquationTerm[DM, ADVEM],
    Generic[DM, ADVEM],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: ADVEM,
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode,
        )


class AbstractAdvectionSpaceOperators(
    disc_types.AbstractSpaceOperators[DM, ADVEM, PSO],
    Generic[DM, ADVEM, PSO],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: ADVEM,
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode,
        )

    @abstractmethod
    def init_continuity_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> disc_types.AbstractContinuityEquForm[DM, ADVEM]:
        """Initialize the continuity equation form."""
        pass

    @abstractmethod
    def init_tracer_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> disc_types.AbstractTracerEquForm[DM, ADVEM]:
        """Initialize the tracer equation form."""
        pass


class AbstractAdvectionSpaceOperators2D(
    AbstractAdvectionSpaceOperators[DM, AdvectionEquation2DMode, PSO],
    Generic[DM, PSO],
    ABC
):
    pass


class AbstractAdvectionSpaceOperators3D(
    AbstractAdvectionSpaceOperators[DM, AdvectionEquation3DMode, PSO],
    Generic[DM, PSO],
    ABC
):
    pass


class DivergenceOfMassFluxProtocol(
    AbstractDynamicalEquationTerm[DM, ADVEM],
    Generic[DM, ADVEM]
):
    @staticmethod
    def alias_name() -> str:
        return "dmf"


DMF = TypeVar("DMF", bound=DivergenceOfMassFluxProtocol)


class VerticalAdvectionOfMassFluxProtocol(
    AbstractDynamicalEquationTerm[DM, ADVEM],
    Generic[DM, ADVEM]
):
    @staticmethod
    def alias_name() -> str:
        return "vadv"


VMF = TypeVar("VMF", bound=VerticalAdvectionOfMassFluxProtocol)


class DivergenceOfTracerFluxProtocol(
    AbstractAdvectionEquationTerm[DM, ADVEM],
    Generic[DM, ADVEM]
):
    @staticmethod
    def alias_name() -> str:
        return "dtf"


DTF = TypeVar("DTF", bound=DivergenceOfTracerFluxProtocol)


class VerticalAdvectionOfTracerProtocol(
    AbstractAdvectionEquationTerm[DM, ADVEM],
    Generic[DM, ADVEM]
):
    @staticmethod
    def alias_name() -> str:
        return "vadv"


VAT = TypeVar("VAT", bound=VerticalAdvectionOfTracerProtocol)


class Ex2DContinuityEqu(
    AbstractDynamicalEquationForm[DM, AdvectionEquation2DMode],
    disc_types.AbstractContinuityEquForm[DM, AdvectionEquation2DMode],
    Generic[DMF, DM],
    ABC
):
    dmf: DMF

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: AdvectionEquation2DMode,
    ):
        super().__init__(
            disc=disc, method=method, mode=mode,
        )
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
        return "dmdt"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def config2dict(self) -> Dict[str, Any]:
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


class Ex2DTracerEqu(
    AbstractAdvectionEquationForm[DM, AdvectionEquation2DMode],
    disc_types.AbstractTracerEquForm[DM, AdvectionEquation2DMode],
    Generic[DTF, DM],
    ABC
):
    dtf: DTF

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: AdvectionEquation2DMode,
    ):
        super().__init__(
            disc=disc, method=method, mode=mode,
        )
        self.dtf = self.init_div_tf()

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> BLOCK:
        return self.dtf(block, star_idx, dt, substep)

    def calc_at_batch(
        self,
        batch: disc_types.AbstractAdvBatchType,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> tuple[disc_types.AbstractAdvBatchType, BLOCK]:
        return self.dtf.calc_at_batch(batch, block, star_idx, dt, substep)

    @staticmethod
    def alias_name() -> str:
        return "dqdt"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.dtf.config2dict()
        self._update_config(cfg=cfg)
        return self._cfg

    def build_with_config(self, cfg: model_config.ModelConfig) -> Self:
        if not self.has_built():
            self.dtf.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.dtf.prepare_block(block)
        return block

    def init_div_tf(self) -> DTF:
        factory = self.navi2term(
            group="div_of_tf",
            choice="builder"
        )
        origin = getattr(factory, '__origin__', factory)
        if inspect.isclass(origin) and issubclass(origin, DivergenceOfTracerFluxProtocol):
            instance = factory(
                disc=self._disc, method=self.method, mode=self.mode
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of DivergenceOfTracerFluxProtocol."
            )


class Ex3DContinuityEqu(
    AbstractDynamicalEquationForm[DM, AdvectionEquation3DMode],
    disc_types.AbstractContinuityEquForm[DM, AdvectionEquation3DMode],
    Generic[DMF, VMF, DM],
    ABC
):
    dmf: DMF
    vadv: VMF

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: AdvectionEquation3DMode,
    ):
        super().__init__(
            disc=disc, method=method, mode=mode,
        )
        self.dmf = self.init_div_mf()
        self.vadv = self.init_vert_adv()

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> BLOCK:
        block = self.dmf(block, star_idx, dt, substep)
        block = self.vadv(block, star_idx, dt, substep)
        return block

    @staticmethod
    def alias_name() -> str:
        return "dmdt"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.dmf.config2dict()
        self._update_config(cfg=cfg)
        cfg = self.vadv.config2dict()
        self._update_config(cfg=cfg)
        return self._cfg

    def build_with_config(self, cfg: model_config.ModelConfig) -> Self:
        if not self.has_built():
            self.dmf.build_with_config(cfg=cfg)
            self.vadv.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.dmf.prepare_block(block)
        block = self.vadv.prepare_block(block)
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

    def init_vert_adv(self) -> VMF:
        factory = self.navi2term(
            group="vert_adv",
            choice="builder"
        )
        origin = getattr(factory, '__origin__', factory)
        if inspect.isclass(origin) and issubclass(origin, VerticalAdvectionOfMassFluxProtocol):
            instance = factory(
                disc=self._disc, method=self.method, mode=self.mode
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of VerticalAdvectionOfMassFluxProtocol."
            )


class Ex3DTracerEqu(
    AbstractAdvectionEquationForm[DM, AdvectionEquation3DMode],
    disc_types.AbstractTracerEquForm[DM, AdvectionEquation3DMode],
    Generic[DTF, VAT, DM],
    ABC
):
    dtf: DTF
    vadv: VAT

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: AdvectionEquation3DMode,
    ):
        super().__init__(
            disc=disc, method=method, mode=mode,
        )
        self.dtf = self.init_div_tf()
        self.vadv = self.init_vert_adv()

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> BLOCK:
        block = self.dtf(block, star_idx, dt, substep)
        block = self.vadv(block, star_idx, dt, substep)
        return block

    def calc_at_batch(
        self,
        batch: disc_types.AbstractAdvBatchType,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> tuple[disc_types.AbstractAdvBatchType, BLOCK]:
        batch, block = self.dtf.calc_at_batch(
            batch, block, star_idx, dt, substep
        )
        batch, block = self.vadv.calc_at_batch(
            batch, block, star_idx, dt, substep
        )
        return batch, block

    @staticmethod
    def alias_name() -> str:
        return "dqdt"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def config2dict(self) -> Dict[str, Any]:
        cfg = self.dtf.config2dict()
        self._update_config(cfg=cfg)
        cfg = self.vadv.config2dict()
        self._update_config(cfg=cfg)
        return self._cfg

    def build_with_config(self, cfg: model_config.ModelConfig) -> Self:
        if not self.has_built():
            self.dtf.build_with_config(cfg=cfg)
            self.vadv.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.dtf.prepare_block(block)
        block = self.vadv.prepare_block(block)
        return block

    def init_div_tf(self) -> DTF:
        factory = self.navi2term(
            group="div_of_tf",
            choice="builder"
        )
        origin = getattr(factory, '__origin__', factory)
        if inspect.isclass(origin) and issubclass(origin, DivergenceOfTracerFluxProtocol):
            instance = factory(
                disc=self._disc, method=self.method, mode=self.mode
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of DivergenceOfTracerFluxProtocol."
            )

    def init_vert_adv(self) -> VAT:
        factory = self.navi2term(
            group="vert_adv",
            choice="builder"
        )
        origin = getattr(factory, '__origin__', factory)
        if inspect.isclass(origin) and issubclass(origin, VerticalAdvectionOfTracerProtocol):
            instance = factory(
                disc=self._disc, method=self.method, mode=self.mode
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of VerticalAdvectionOfTracerProtocol."
            )


class ExVif2DAdvectionEquationSet4UGRIDSpaceOperators(
    AbstractAdvectionSpaceOperators2D[DM, PSO],
    Generic[
        DMF,
        DTF,
        DM, PSO
    ],
    ABC
):
    dmdt: Ex2DContinuityEqu[DMF, DM]
    dqdt: Ex2DTracerEqu[DTF, DM]

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: AdvectionEquation2DMode,
    ):
        super().__init__(
            disc=disc, method=method, mode=mode
        )
        self.dmdt = self.init_continuity_equation(disc, method)
        self.dqdt = self.init_tracer_equation(disc, method)

    def init_continuity_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM
    ) -> Ex2DContinuityEqu[DMF, DM]:
        return Ex2DContinuityEqu(disc=disc, method=method, mode=self.mode)

    def init_tracer_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM
    ) -> Ex2DTracerEqu[DTF, DM]:
        return Ex2DTracerEqu(disc=disc, method=method, mode=self.mode)


class ExVif3DAdvectionEquationSet4UGRIDSpaceOperators(
    AbstractAdvectionSpaceOperators3D[DM, PSO],
    Generic[
        DMF, VMF,
        DTF, VAT,
        DM, PSO
    ],
    ABC
):
    dmdt: Ex3DContinuityEqu[DMF, VMF, DM]
    dqdt: Ex3DTracerEqu[DTF, VAT, DM]

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: AdvectionEquation3DMode,
    ):
        super().__init__(
            disc=disc, method=method, mode=mode
        )
        self.dmdt = self.init_continuity_equation(disc, method)
        self.dqdt = self.init_tracer_equation(disc, method)

    def init_continuity_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM
    ) -> Ex3DContinuityEqu[DMF, VMF, DM]:
        return Ex3DContinuityEqu(disc=disc, method=method, mode=self.mode)

    def init_tracer_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM
    ) -> Ex3DTracerEqu[DTF, VAT, DM]:
        return Ex3DTracerEqu(disc=disc, method=method, mode=self.mode)


class ExVif2DAdvectionEquationSet4UGRID(
    AbstractAdvectionEquationSet2D[DM, SOP],
    AbstractExplicitODE,
    Generic[DM, SOP],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        **kwargs
    ):
        super().__init__(disc=disc, method=method, **kwargs)

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the dynamics set."""
        return AbstractAdvectionEquationSet2D.alias_name()

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
    ) -> type[disc_types.AbstractBlock]: ...

    def build_block_cls(
        self,
        coord: CoordinateSystem[UHM, VM, HLO, VLO],
    ) -> type[disc_types.AbstractBlock]:
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
        if advection:

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

        else:

            def _wrap_dynamics(block: BLOCK, *args, **kwargs) -> BLOCK:
                return self.space_operators(
                    block=block,
                    *args,
                    **kwargs,
                    explicit=True
                )

            return _wrap_dynamics


class ExVif3DAdvectionEquationSet4UGRID(
    AbstractAdvectionEquationSet3D[DM, SOP],
    AbstractExplicitODE,
    Generic[DM, SOP],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        **kwargs
    ):
        super().__init__(disc=disc, method=method, **kwargs)

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the dynamics set."""
        return AbstractAdvectionEquationSet3D.alias_name()

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
            HybridVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            HybridVertCoordLayout
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
        from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
        from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
        from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
        from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
        if (
            isinstance(self.method, FiniteVolumeMethod) and
            isinstance(coord.mesh, SphericalCentroidVoronoiTessellations) and
            isinstance(coord.vert, HybridVertCoordinate) and
            isinstance(coord.hori_layout, TriSKBasedCGridStaggeringLayout) and
            isinstance(coord.vert_layout, HybridVertCoordLayout)
        ):
            from terraflow.discretization.fvm.scvt.trisk.dynamics_types import Block
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

        if advection:

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

        else:

            def _wrap_dynamics(block: BLOCK, *args, **kwargs) -> BLOCK:
                return self.space_operators(
                    block,
                    *args,
                    **kwargs,
                    explicit=True
                )

            return _wrap_dynamics


class AdvectionEquation2DBuilder:
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
        method: AbstractDiscretizationMethod,
        form: Literal["vector_invariant"],
        ode_kind: Literal["explicit"],
        config: model_config.ModelConfig = model_config.ModelConfig(),
        *args,
        **kwargs,
    ) -> ExVif_ADV2D_SCVT_TriSKCGrid_FVM:
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
    ) -> AbstractAdvectionEquationSet2D:
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
    ) -> AbstractAdvectionEquationSet2D:
        equation = cls.from_model(
            coord=coord,
            para=para,
            method=method,
            form=form,
            ode_kind=ode_kind,
            *args,
            **kwargs,
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
        method: AbstractDiscretizationMethod,
        form: Literal["vector_invariant"],
        ode_kind: Literal["explicit"],
        *args,
        **kwargs,
    ) -> ExVif_ADV2D_SCVT_TriSKCGrid_FVM:
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
    ) -> AbstractAdvectionEquationSet2D:
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
    ) -> AbstractAdvectionEquationSet2D:
        if (ode_kind == "explicit" and form == "vector_invariant"):
            return ExVif2DAdvectionEquationSet4UGRID.from_model(
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


class AdvectionEquation3DBuilder:
    @overload
    @classmethod
    def build_with_config(
        cls,
        coord: CoordinateSystem[
            SphericalCentroidVoronoiTessellations,
            HybridVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            HybridVertCoordLayout
        ],
        para: PlanetParameterSet,
        method: AbstractDiscretizationMethod,
        form: Literal["vector_invariant"],
        ode_kind: Literal["explicit"],
        config: model_config.ModelConfig = model_config.ModelConfig(),
        *args,
        **kwargs,
    ) -> ExVif_ADV3D_SCVT_TriSKCGrid_FVM:
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
    ) -> AbstractAdvectionEquationSet3D:
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
    ) -> AbstractAdvectionEquationSet3D:
        equation = cls.from_model(
            coord=coord,
            para=para,
            method=method,
            form=form,
            ode_kind=ode_kind,
            *args,
            **kwargs,
        )
        equation.build_with_config(cfg=config)
        return equation

    @overload
    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[
            SphericalCentroidVoronoiTessellations,
            HybridVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            HybridVertCoordLayout
        ],
        para: PlanetParameterSet,
        method: AbstractDiscretizationMethod,
        form: Literal["vector_invariant"],
        ode_kind: Literal["explicit"],
        *args,
        **kwargs,
    ) -> ExVif_ADV3D_SCVT_TriSKCGrid_FVM:
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
    ) -> AbstractAdvectionEquationSet3D:
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
    ) -> AbstractAdvectionEquationSet3D:
        if (ode_kind == "explicit" and form == "vector_invariant"):
            return ExVif3DAdvectionEquationSet4UGRID.from_model(
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
    ExVif_ADV2D_SCVT_TriSKCGrid_FVM = ExVif2DAdvectionEquationSet4UGRID[
        FiniteVolumeMethod,
        fvm_scvt_trisk_1lev_adv_terms.ExSpaceOperators
    ]
    ExVif_ADV3D_SCVT_TriSKCGrid_FVM = ExVif3DAdvectionEquationSet4UGRID[
        FiniteVolumeMethod,
        fvm_scvt_trisk_hybrid_adv_terms.ExSpaceOperators
    ]
