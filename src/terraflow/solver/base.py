from __future__ import annotations
from typing import List, Callable, TYPE_CHECKING
import datetime
import jax.numpy as jnp

from terraflow.util import typing, device, log
from terraflow.util.datetime import DateTimeManager
from terraflow.solver.abstract import AbstractExplicitODE, AbstractSolver
from terraflow.solver.abstract import AbstractRungeKuttaSolver

if TYPE_CHECKING:
    from terraflow.discretization.abstract import AbstractBlock

    Float = typing.Float
    Int = typing.Int
    IntType = typing.IntType
    Array = typing.Array

all_pass = "all_pass"
forward_pass = "forward_pass"
backward_pass = "backward_pass"


def step_all(
    space_operators: Callable,
    block: AbstractBlock,
    old_idx: Int[Array, ""] | int,
    star_idx: Int[Array, ""] | int,
    new_idx: Int[Array, ""] | int,
    dt: float | Array,
    update_state_fn: Callable
) -> AbstractBlock:
    block = space_operators(
        block,
        old_idx=old_idx,
        star_idx=star_idx,
        new_idx=new_idx,
        pass_tag=all_pass
    )
    return update_state_fn(
        block=block,
        old_idx=old_idx,
        new_idx=new_idx,
        dt=dt,
        pass_tag=all_pass
    )


def step_forward_backward(
    space_operators: Callable,
    block: AbstractBlock,
    old_idx: Int[Array, ""] | int,
    star_idx: Int[Array, ""] | int,
    new_idx: Int[Array, ""] | int,
    dt: float | Array,
    update_state_fn: Callable
) -> AbstractBlock:
    block = space_operators(
        block=block,
        old_idx=old_idx,
        star_idx=star_idx,
        new_idx=new_idx,
        pass_tag=forward_pass
    )
    block = update_state_fn(
        block=block,
        old_idx=old_idx,
        new_idx=new_idx,
        dt=dt,
        pass_tag=forward_pass
    )
    block = space_operators(
        block=block,
        old_idx=old_idx,
        star_idx=star_idx,
        new_idx=new_idx,
        pass_tag=backward_pass
    )
    block = update_state_fn(
        block=block,
        old_idx=old_idx,
        new_idx=new_idx,
        dt=dt,
        pass_tag=backward_pass
    )
    return block


class ExRungeKutta3Solver(AbstractRungeKuttaSolver[AbstractExplicitODE]):
    nrk: IntType = 3  # Number of stages for RK3
    order: IntType = 3  # Order of accuracy for RK3
    dt_rk: List[datetime.timedelta]
    dt_rk_seconds: Array
    dev: device.AbstractDevice

    def __init__(
        self,
        equation: AbstractExplicitODE,
        time_manager: DateTimeManager,
        dev: device.AbstractDevice,
        with_jit: bool = True,
        forward_backward: bool = True
    ):
        super().__init__(
            equation=equation,
            time_manager=time_manager,
            dev=dev,
            with_jit=with_jit
        )
        self.nrk = 3
        self.dt_rk = [self.dt / (self.nrk - irk) for irk in range(self.nrk)]
        self.dt_rk_seconds = jnp.asarray(
            [dt_rk.total_seconds() for dt_rk in self.dt_rk],
            dtype=dev.float_type
        )
        self.forward_backward = forward_backward
        if self.forward_backward:
            log.logger.success(
                "Use Forward-Backward Third-order Runge-Kutta time integration solver."
            )
        else:
            log.logger.success(
                "Use Third-order Runge-Kutta time integration solver."
            )

    def stepper(self) -> Callable:
        """Third-order Runge-Kutta time stepper."""
        rhs = self.equation.explicit_terms
        update_state_fn = self.update_state_fn()

        substep = step_forward_backward if self.forward_backward else step_all

        def rk3_step(block: AbstractBlock) -> AbstractBlock:
            block = substep(
                space_operators=rhs,
                block=block,
                old_idx=block.old,
                star_idx=block.old,
                new_idx=block.new,
                dt=self.dt_rk_seconds[0],
                update_state_fn=update_state_fn
            )
            block = substep(
                space_operators=rhs,
                block=block,
                old_idx=block.old,
                star_idx=block.new,
                new_idx=2,
                dt=self.dt_rk_seconds[1],
                update_state_fn=update_state_fn
            )
            block = substep(
                space_operators=rhs,
                block=block,
                old_idx=block.old,
                star_idx=2,
                new_idx=block.new,
                dt=self.dt_rk_seconds[2],
                update_state_fn=update_state_fn
            )
            return block

        return rk3_step


class ExEulerSolver(AbstractSolver[AbstractExplicitODE]):

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
        equation: AbstractExplicitODE,
        time_manager: DateTimeManager,
        dev: device.AbstractDevice,
        with_jit: bool = True,
        forward_backward: bool = True
    ):
        super().__init__(
            equation=equation,
            time_manager=time_manager,
            dev=dev,
            with_jit=with_jit
        )
        self.dt_seconds = jnp.asarray(
            self.dt.total_seconds(),
            dtype=dev.float_type
        )
        self.forward_backward = forward_backward
        if self.forward_backward:
            log.logger.success(
                "Use Forward-Backward Euler time integration solver."
            )
        else:
            log.logger.success(
                "Use Euler time integration solver."
            )

    def stepper(self) -> Callable:
        """Forward-Backward Euler time stepper."""
        rhs = self.equation.explicit_terms
        update_state_fn = self.update_state_fn()

        substep = step_forward_backward if self.forward_backward else step_all

        def euler_step(block: AbstractBlock) -> AbstractBlock:
            return substep(
                space_operators=rhs,
                block=block,
                old_idx=block.old,
                star_idx=block.old,
                new_idx=block.new,
                dt=self.dt_seconds,
                update_state_fn=update_state_fn
            )

        return euler_step
