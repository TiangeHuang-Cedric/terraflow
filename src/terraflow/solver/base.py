from __future__ import annotations
from typing import List, Callable, TYPE_CHECKING
import datetime
import jax
import jax.numpy as jnp
import equinox as eqx

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


def update_block_forward_backward(
    block: typing.PyTree,
    old_state: typing.PyTree,
    tend: typing.PyTree,
    dt: float | Array,
    pass_tag: str
) -> typing.PyTree:
    tend2state = tend.to_state(pass_tag=pass_tag)
    new_state = jax.tree_util.tree_map(
        lambda s, t: (
            s + t * dt
            if getattr(s, "meta", None) and getattr(s.meta, "is_prognostic", False)
            else s
        ),
        old_state,
        tend2state,
        is_leaf=lambda x: isinstance(x, Field)
    )
    return eqx.tree_at(
        lambda b: b.state,
        block,
        replace=new_state
    )


def step_all(
    space_operators: Callable,
    block: AbstractBlock,
    old_state: typing.PyTree,
    dt: float | Array
) -> AbstractBlock:
    block = space_operators(block, pass_tag=all_pass)
    new_block = update_block_forward_backward(
        block=block,
        old_state=old_state,
        tend=block.tend,
        dt=dt,
        pass_tag=all_pass
    )
    return new_block


def step_forward_backward(
    space_operators: Callable,
    block: AbstractBlock,
    old_state: typing.PyTree,
    dt: float | Array
) -> AbstractBlock:
    block = space_operators(block, pass_tag=forward_pass)
    star_block = update_block_forward_backward(
        block=block,
        old_state=old_state,
        tend=block.tend,
        dt=dt,
        pass_tag=forward_pass
    )
    star_block = space_operators(star_block, pass_tag=backward_pass)
    new_block = update_block_forward_backward(
        block=star_block,
        old_state=star_block.state,
        tend=star_block.tend,
        dt=dt,
        pass_tag=backward_pass
    )
    return new_block


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

    def stepper(self) -> Callable:
        """Third-order Runge-Kutta time stepper."""
        rhs = self.equation.explicit_terms
        nrk = self.nrk

        def step_fn_all(block: AbstractBlock) -> AbstractBlock:
            block = block.reset()
            old_state = block.state
            for irk in range(nrk):
                block = step_all(
                    space_operators=rhs,
                    block=block,
                    old_state=old_state,
                    dt=self.dt_rk_seconds[irk]
                )
            return block

        def step_fn_fb(block: AbstractBlock) -> AbstractBlock:
            block = block.reset()
            old_state = block.state
            for irk in range(nrk):
                block = step_forward_backward(
                    space_operators=rhs,
                    block=block,
                    old_state=old_state,
                    dt=self.dt_rk_seconds[irk]
                )
            return block

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

    def stepper(self) -> Callable:
        """Forward-Backward Euler time stepper."""
        rhs = self.equation.explicit_terms

        def step_fn_all(block: AbstractBlock) -> AbstractBlock:
            block = block.reset()
            return step_all(
                space_operators=rhs,
                block=block,
                old_state=block.state,
                dt=self.dt_seconds
            )

        def step_fn_fb(block: AbstractBlock) -> AbstractBlock:
            block = block.reset()
            return step_forward_backward(
                space_operators=rhs,
                block=block,
                old_state=block.state,
                dt=self.dt_seconds
            )

        return step_fn_fb if self.forward_backward else step_fn_all
