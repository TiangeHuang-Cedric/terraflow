from __future__ import annotations
from typing import (
    overload,
    Literal,
    Union,
    TypeVar,
    List,
    Callable,
    TYPE_CHECKING,
)
import datetime
import jax
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import typing, device, log
from terraflow.util.datetime import DateTimeManager
from terraflow.solver.abstract import AbstractExplicitODE, AbstractSolver
from terraflow.solver.abstract import AbstractRungeKuttaSolver

EXODE = TypeVar("EXODE", bound=AbstractExplicitODE)


if TYPE_CHECKING:
    from terraflow.mesh.abstract import AbstractMesh
    from terraflow.vert_coord.abstract import AbstractVertCoordinate
    from terraflow.layout.abstract import AbstractLayout
    from terraflow.discretization.abstract import (
        AbstractFields,
        AbstractAdvBatchType,
        AbstractEquationTermAuxiliaryDataset,
        AbstractBlock
    )

    Float = typing.Float
    Int = typing.Int
    IntType = typing.IntType
    Array = typing.Array

    HM = AbstractMesh
    VM = AbstractVertCoordinate
    HLO = AbstractLayout
    VLO = AbstractLayout
    STATE = AbstractFields
    STATIC = AbstractFields
    TEND = AbstractFields
    AUX = AbstractFields
    ADVBATCH = AbstractAdvBatchType
    TRACERS = AbstractFields
    AUXDS = AbstractEquationTermAuxiliaryDataset

    BLOCK = AbstractBlock[
        HM, VM, HLO, VLO,
        STATE, STATIC, TEND, AUX, ADVBATCH, TRACERS, AUXDS
    ]

all_pass = "all_pass"
forward_pass = "forward_pass"
backward_pass = "backward_pass"


class ExRungeKutta3Solver(AbstractRungeKuttaSolver[EXODE]):
    nrk: IntType = 3    # Number of stages for RK3
    order: IntType = 3  # Order of accuracy for RK3
    dt_rk: List[datetime.timedelta]
    dt_rk_seconds: Array
    dev: device.AbstractDevice

    def __init__(
        self,
        equation: EXODE,
        time_manager: DateTimeManager,
        dev: device.AbstractDevice,
        forward_backward: bool = True
    ):
        super().__init__(
            nsubsteps=3,
            equation=equation,
            time_manager=time_manager,
            dev=dev,
        )
        self.dt_rk = self.calc_dts_substep(self.dt_dyn, as_array=False)
        self.dt_rk_seconds = jnp.asarray(
            [dt_rk.total_seconds() for dt_rk in self.dt_rk],
            dtype=dev.float_type
        )
        if equation.mode.is_advection():
            self.forward_backward = False
        else:
            self.forward_backward = forward_backward
        if self.forward_backward:
            log.logger.success(
                "Use Forward-Backward Third-order Runge-Kutta time integration solver."
            )
        else:
            log.logger.success(
                "Use Third-order Runge-Kutta time integration solver."
            )

    @overload
    def calc_dts_substep(
        self,
        dt: datetime.timedelta,
        as_array: Literal[False]
    ) -> List[datetime.timedelta]:
        ...

    @overload
    def calc_dts_substep(
        self,
        dt: datetime.timedelta,
        as_array: Literal[True] = True
    ) -> Array:
        ...

    def calc_dts_substep(
        self,
        dt: datetime.timedelta,
        as_array: bool = True
    ) -> Union[Array, List[datetime.timedelta]]:

        def get_dt_rk_substep(irk: int) -> datetime.timedelta:
            return (dt / (self.nsubsteps - irk))

        if as_array:
            return jnp.asarray(
                [
                    get_dt_rk_substep(irk).total_seconds()
                    for irk in range(self.nsubsteps)
                ],
                dtype=self.dev.float_type
            )
        else:
            return [get_dt_rk_substep(irk) for irk in range(self.nsubsteps)]

    def dynamical_stepper(self) -> Callable:
        """Third-order Runge-Kutta time stepper."""
        rhs = self.equation.explicit_terms()
        update_state_fn = self.update_state_fn()

        substep = step_forward_backward if self.forward_backward else step_all

        def rk3_step(block: BLOCK) -> BLOCK:
            block = substep(
                space_operators=rhs,
                update_state_fn=update_state_fn,
                block=block,
                old_idx=block.old,
                star_idx=block.old,
                new_idx=block.new,
                dt=self.dt_rk_seconds[0],
                substep=0,
            )
            block = substep(
                space_operators=rhs,
                update_state_fn=update_state_fn,
                block=block,
                old_idx=block.old,
                star_idx=block.new,
                new_idx=2,
                dt=self.dt_rk_seconds[1],
                substep=1,
            )
            block = substep(
                space_operators=rhs,
                update_state_fn=update_state_fn,
                block=block,
                old_idx=block.old,
                star_idx=2,
                new_idx=block.new,
                dt=self.dt_rk_seconds[2],
                substep=2,
            )
            return block

        return rk3_step

    def advection_stepper(self) -> Callable:
        d2a_coupling = self.equation._space_operators.d2a_coupling
        space_operators = self.equation.explicit_terms(advection=True)
        return ex_rk3_advection_stepper(
            d2a_coupling=d2a_coupling,
            space_operators=space_operators
        )


class ExEulerSolver(AbstractSolver[EXODE]):
    """Forward-Backward Euler time integration solver.

    This solver uses two explicit schemes that are more economical than the standard leapfrog scheme.
    For brief, It is obtained by first integrating the gravity wave terms of either the dynamics of motion
    or of the continuity dynamics forward, and then those of the other dynamics backward in time.
    Thus, this scheme could be called the forward-backward scheme.

    References:
        - Mesinger, F., & Arakawa, A. (1976). 
          Numerical methods used in atmospheric models, (Vol. 1). Geneva (Switzerland): 
          Global Atmospheric Research Programe (GARP) WMO‐ICSU Joint Organizing Committe.
    """

    def __init__(
        self,
        equation: EXODE,
        time_manager: DateTimeManager,
        dev: device.AbstractDevice,
        forward_backward: bool = True
    ):
        super().__init__(
            nsubsteps=1,
            equation=equation,
            time_manager=time_manager,
            dev=dev,
        )
        self.dt_seconds = jnp.asarray(
            self.dt_dyn.total_seconds(),
            dtype=dev.float_type
        )
        if equation.mode.is_advection():
            self.forward_backward = False
        else:
            self.forward_backward = forward_backward
        if self.forward_backward:
            log.logger.success(
                "Use Forward-Backward Euler time integration solver."
            )
        else:
            log.logger.success(
                "Use Euler time integration solver."
            )

    @overload
    def calc_dts_substep(
        self,
        dt: datetime.timedelta,
        as_array: Literal[False]
    ) -> List[datetime.timedelta]:
        ...

    @overload
    def calc_dts_substep(
        self,
        dt: datetime.timedelta,
        as_array: Literal[True] = True
    ) -> Array:
        ...

    def calc_dts_substep(
        self,
        dt: datetime.timedelta,
        as_array: bool = True
    ) -> Union[Array, List[datetime.timedelta]]:
        if as_array:
            return jnp.asarray(
                [dt.total_seconds()],
                dtype=self.dev.float_type
            )
        else:
            return [dt]

    def dynamical_stepper(self) -> Callable:
        """Forward-Backward Euler time stepper."""
        rhs = self.equation.explicit_terms()
        update_state_fn = self.update_state_fn()

        substep = step_forward_backward if self.forward_backward else step_all

        def euler_step(block: BLOCK) -> BLOCK:
            block = substep(
                space_operators=rhs,
                update_state_fn=update_state_fn,
                block=block,
                old_idx=block.old,
                star_idx=block.old,
                new_idx=block.new,
                dt=self.dt_seconds,
                substep=0,
            )
            return block

        return euler_step

    def advection_stepper(self) -> Callable:
        d2a_coupling = self.equation._space_operators.d2a_coupling
        space_operators = self.equation.explicit_terms(advection=True)
        return ex_euler_advection_stepper(
            d2a_coupling=d2a_coupling,
            space_operators=space_operators
        )


def step_all(
    space_operators: Callable,
    block: BLOCK,
    old_idx: Int[Array, ""] | int,
    star_idx: Int[Array, ""] | int,
    new_idx: Int[Array, ""] | int,
    dt: Float[Array, ""] | float,
    substep: int,
    update_state_fn: Callable
) -> BLOCK:
    block = space_operators(
        block,
        old_idx=old_idx,
        star_idx=star_idx,
        new_idx=new_idx,
        dt=dt,
        substep=substep,
        pass_tag=all_pass
    )
    return update_state_fn(
        block=block,
        old_idx=old_idx,
        new_idx=new_idx,
        dt=dt,
        substep=substep,
        pass_tag=all_pass
    )


def step_forward_backward(
    space_operators: Callable,
    update_state_fn: Callable,
    block: BLOCK,
    old_idx: Int[Array, ""] | int,
    star_idx: Int[Array, ""] | int,
    new_idx: Int[Array, ""] | int,
    dt: Float[Array, ""] | float,
    substep: int,
) -> BLOCK:
    block = space_operators(
        block=block,
        old_idx=old_idx,
        star_idx=star_idx,
        new_idx=new_idx,
        dt=dt,
        substep=substep,
        pass_tag=forward_pass,
    )
    block = update_state_fn(
        block=block,
        old_idx=old_idx,
        new_idx=new_idx,
        dt=dt,
        substep=substep,
        pass_tag=forward_pass
    )
    block = space_operators(
        block=block,
        old_idx=old_idx,
        star_idx=star_idx,
        new_idx=new_idx,
        dt=dt,
        substep=substep,
        pass_tag=backward_pass
    )
    block = update_state_fn(
        block=block,
        old_idx=old_idx,
        new_idx=new_idx,
        dt=dt,
        substep=substep,
        pass_tag=backward_pass
    )
    return block


def step_adv(
    space_operators: Callable,
    batch: AbstractAdvBatchType,
    block: BLOCK,
    old_idx: Int[Array, ""] | int,
    star_idx: Int[Array, ""] | int,
    new_idx: Int[Array, ""] | int,
    dt: Float[Array, ""] | float,
    substep: int,
) -> tuple[AbstractAdvBatchType, BLOCK]:
    return space_operators(
        batch=batch,
        block=block,
        old_idx=old_idx,
        star_idx=star_idx,
        new_idx=new_idx,
        dt=dt,
        substep=substep,
    )


def ex_rk3_advection_stepper(d2a_coupling: Callable, space_operators: Callable) -> Callable:

    substep = step_adv

    def rk3_step(block: BLOCK) -> BLOCK:

        block = d2a_coupling(block)

        for batch_name, batch in block.adv_batches.items():

            def do_nothing(
                carry_in: tuple[AbstractAdvBatchType, BLOCK]
            ) -> tuple[AbstractAdvBatchType, BLOCK]:
                return carry_in

            def run_tracers(
                carry_in: tuple[AbstractAdvBatchType, BLOCK]
            ) -> tuple[AbstractAdvBatchType, BLOCK]:
                batch, block = carry_in
                batch, block = substep(
                    space_operators=space_operators,
                    batch=batch,
                    block=block,
                    old_idx=batch.old,
                    star_idx=batch.old,
                    new_idx=batch.new,
                    dt=batch.dts_substep[0],
                    substep=0,
                )
                batch, block = substep(
                    space_operators=space_operators,
                    batch=batch,
                    block=block,
                    old_idx=batch.old,
                    star_idx=batch.new,
                    new_idx=2,
                    dt=batch.dts_substep[1],
                    substep=1,
                )
                batch, block = substep(
                    space_operators=space_operators,
                    batch=batch,
                    block=block,
                    old_idx=batch.old,
                    star_idx=2,
                    new_idx=batch.new,
                    dt=batch.dts_substep[2],
                    substep=2,
                )
                return batch, block

            batch, block = jax.lax.cond(
                batch.is_run_tracer_istep(),
                run_tracers,
                do_nothing,
                (batch, block)
            )

            batch = batch.update_istep()

            block = eqx.tree_at(
                lambda b: b.adv_batches[batch_name],
                block,
                batch
            )

        return block

    return rk3_step


def ex_euler_advection_stepper(d2a_coupling: Callable, space_operators: Callable) -> Callable:

    substep = step_adv

    def euler_step(block: BLOCK) -> BLOCK:

        block = d2a_coupling(block)

        for batch_name, batch in block.adv_batches.items():

            def do_nothing(
                carry_in: tuple[AbstractAdvBatchType, BLOCK]
            ) -> tuple[AbstractAdvBatchType, BLOCK]:
                return carry_in

            def run_tracers(
                carry_in: tuple[AbstractAdvBatchType, BLOCK]
            ) -> tuple[AbstractAdvBatchType, BLOCK]:
                batch, block = carry_in
                batch, block = substep(
                    space_operators=space_operators,
                    batch=batch,
                    block=block,
                    old_idx=batch.old,
                    star_idx=batch.old,
                    new_idx=batch.new,
                    dt=batch.dts_substep[0],
                    substep=0,
                )
                return batch, block

            batch, block = jax.lax.cond(
                batch.is_run_tracer_istep(),
                run_tracers,
                do_nothing,
                (batch, block)
            )

            batch = batch.update_istep()

            block = eqx.tree_at(
                lambda b: b.adv_batches[batch_name],
                block,
                batch
            )

        return block

    return euler_step
