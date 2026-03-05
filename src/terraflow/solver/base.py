from __future__ import annotations
from typing import List, Callable, TYPE_CHECKING
import datetime
import jax
import jax.numpy as jnp
import equinox as eqx
from loguru import logger

from terraflow.util import typing, device
from terraflow.util.datetime import DateTimeManager
from terraflow.layout.field import Field

from terraflow.solver.abstract import AbstractExplicitODE, AbstractSolver
from terraflow.solver.abstract import AbstractRungeKuttaSolver

if TYPE_CHECKING:
    from terraflow.discretization.abstract import AbstractBlock


Array = typing.Array
Float = typing.FloatType
Int = typing.IntType

all_pass = "all_pass"
forward_pass = "forward_pass"
backward_pass = "backward_pass"


def update_state_forward_backward(
    old_state: typing.PyTree,
    new_state: typing.PyTree,
    tend: typing.PyTree,
    dt: float | Array,
    pass_tag: str
) -> typing.PyTree:
    old_prog = old_state.to_prognostic()
    tend2prog = tend.to_prognostic(pass_tag=pass_tag)
    new_prog = jax.tree_util.tree_map(
        lambda op, tp: op + tp * dt,
        old_prog,
        tend2prog,
        is_leaf=lambda x: isinstance(x, Array)
    )
    return new_state.wrap_from_prognostic(new_prog)


def step_all(
    space_operators: Callable,
    old_block: AbstractBlock,
    old_state: typing.PyTree,
    dt: float | Array
) -> tuple[AbstractBlock, AbstractBlock]:
    old_block = space_operators(old_block, pass_tag=all_pass)
    new_state = update_state_forward_backward(
        old_state=old_state,
        new_state=old_block.state,
        tend=old_block.tend,
        dt=dt,
        pass_tag=all_pass
    )
    new_block = eqx.tree_at(
        lambda b: b.state,
        old_block,
        new_state
    )
    new_block = new_block.reset()
    return new_block, old_block


def step_forward_backward(
    space_operators: Callable,
    old_block: AbstractBlock,
    old_state: typing.PyTree,
    dt: float | Array
) -> tuple[AbstractBlock, AbstractBlock]:
    old_block = space_operators(old_block, pass_tag=forward_pass)
    star_state = update_state_forward_backward(
        old_state=old_state,
        new_state=old_block.state,
        tend=old_block.tend,
        dt=dt,
        pass_tag=forward_pass
    )
    star_block = eqx.tree_at(
        lambda b: b.state,
        old_block,
        star_state
    )
    star_block = space_operators(star_block, pass_tag=backward_pass)
    new_state = update_state_forward_backward(
        old_state=star_block.state,
        new_state=star_block.state,
        tend=star_block.tend,
        dt=dt,
        pass_tag=backward_pass
    )
    new_block = eqx.tree_at(
        lambda b: b.state,
        star_block,
        new_state
    )
    new_block = new_block.reset()
    old_state = jax.tree_util.tree_map(
        lambda os, ns: (
            os
            if getattr(os, "meta", None) and getattr(os.meta, "is_prognostic", False)
            else ns
        ),
        old_block.state,
        new_block.state,
        is_leaf=lambda x: isinstance(x, Field)
    )
    old_block = eqx.tree_at(
        lambda b: b.state,
        star_block,
        old_state
    )
    return new_block, old_block


class ExRungeKutta3Solver(AbstractRungeKuttaSolver[AbstractExplicitODE]):
    nrk: Int = 3  # Number of stages for RK3
    order: Int = 3  # Order of accuracy for RK3
    dt_rk: List[datetime.timedelta]
    dt_rk_seconds: Array
    dev: device.AbstractDevice

    def __init__(
        self,
        equation: AbstractExplicitODE,
        time_manager: DateTimeManager,
        dev: device.AbstractDevice,
        forward_backward: bool = True
    ):
        super().__init__(equation, time_manager, dev)
        self.nrk = 3
        self.dt_rk = [self.dt / (self.nrk - irk) for irk in range(self.nrk)]
        self.dt_rk_seconds = jnp.asarray(
            [dt_rk.total_seconds() for dt_rk in self.dt_rk],
            dtype=dev.float_type
        )
        self.forward_backward = forward_backward
        if self.forward_backward:
            logger.success(
                "Use Forward-Backward Third-order Runge-Kutta time integration solver."
            )
        else:
            logger.success(
                "Use Third-order Runge-Kutta time integration solver."
            )

    def stepper(self) -> Callable:
        """Third-order Runge-Kutta time stepper."""
        rhs = self.equation.explicit_terms
        nrk = self.nrk

        def step_fn_all(old_block: AbstractBlock) -> tuple[AbstractBlock, AbstractBlock]:
            new_block = old_block.reset()
            old_state = new_block.state
            for irk in range(nrk):
                new_block, old_block = step_all(
                    space_operators=rhs,
                    old_block=new_block,
                    old_state=old_state,
                    dt=self.dt_rk_seconds[irk]
                )
            return new_block, old_block

        def step_fn_fb(old_block: AbstractBlock) -> tuple[AbstractBlock, AbstractBlock]:
            new_block = old_block.reset()
            old_state = new_block.state
            for irk in range(nrk):
                new_block, old_block = step_forward_backward(
                    space_operators=rhs,
                    old_block=new_block,
                    old_state=old_state,
                    dt=self.dt_rk_seconds[irk]
                )
            return new_block, old_block

        return step_fn_fb if self.forward_backward else step_fn_all


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
        forward_backward: bool = True
    ):
        super().__init__(equation, time_manager, dev)
        self.dt_seconds = self.dt.total_seconds()
        self.forward_backward = forward_backward
        if self.forward_backward:
            logger.success(
                "Use Forward-Backward Euler time integration solver."
            )
        else:
            logger.success(
                "Use Euler time integration solver."
            )

    def stepper(self) -> Callable:
        """Forward-Backward Euler time stepper."""
        rhs = self.equation.explicit_terms

        def step_fn_all(old_block: AbstractBlock) -> tuple[AbstractBlock, AbstractBlock]:
            new_block = old_block.reset()
            return step_all(
                space_operators=rhs,
                old_block=new_block,
                old_state=new_block.state,
                dt=self.dt_seconds
            )

        def step_fn_fb(old_block: AbstractBlock) -> tuple[AbstractBlock, AbstractBlock]:
            new_block = old_block.reset()
            return step_forward_backward(
                space_operators=rhs,
                old_block=new_block,
                old_state=new_block.state,
                dt=self.dt_seconds
            )

        return step_fn_fb if self.forward_backward else step_fn_all
