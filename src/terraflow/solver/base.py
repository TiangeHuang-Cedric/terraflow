from __future__ import annotations
from typing import List, Callable, TYPE_CHECKING
import datetime
import tree_math
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import typing, device
from terraflow.util.datetime import DateTimeManager

from terraflow.solver.abstract import AbstractExplicitODE, AbstractSolver
from terraflow.solver.abstract import AbstractRungeKuttaSolver

if TYPE_CHECKING:
    from terraflow.discretization.abstract import AbstractBlock


Array = typing.Array
Float = typing.FloatType
Int = typing.IntType


class ExRungeKutta3Solver(AbstractRungeKuttaSolver[AbstractExplicitODE]):
    nrk: Int = 3  # Number of stages for RK3
    order: Int = 3  # Order of accuracy for RK3
    dt_rk: List[datetime.timedelta]
    dt_rk_seconds: Array
    dev: device.AbstractDevice

    def __init__(
        self,
        equation: AbstractExplicitODE,
        dt: datetime.timedelta,
        time_manager: DateTimeManager,
        dev: device.AbstractDevice
    ):
        super().__init__(equation, dt, time_manager, dev)
        self.nrk = 3
        self.dt_rk = [dt / (self.nrk - irk) for irk in range(self.nrk)]
        self.dt_rk_seconds = jnp.asarray(
            [dt_rk.total_seconds() for dt_rk in self.dt_rk],
            dtype=dev.float_type
        )

    def stepper(self, equation: AbstractExplicitODE) -> Callable:
        """Third-order Runge-Kutta time stepper."""
        rhs = equation.explicit_terms
        nrk = self.nrk
        dt_rk_seconds = self.dt_rk_seconds
        update = tree_math.wrap(tree_math.unwrap(self.update))

        def step_fn(block: AbstractBlock) -> AbstractBlock:
            block = block.reset()
            states0 = block.state

            for irk in range(nrk):
                block = rhs(block)
                tend = block.tend.to_state(pass_tag="all")
                state_new = update(
                    old_state=states0,
                    tend=tend,
                    dt=dt_rk_seconds[irk]
                )
                block = eqx.tree_at(
                    lambda b: b.state,
                    block,
                    replace=state_new
                )
            return block

        return step_fn


class ExForwardBackwardEulerSolver(AbstractSolver[AbstractExplicitODE]):

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

    def stepper(self, equation: AbstractExplicitODE) -> Callable:
        """Forward-Backward Euler time stepper."""
        rhs = equation.explicit_terms
        dt_seconds = self.dt.total_seconds()
        update = tree_math.wrap(tree_math.unwrap(self.update))

        def forward_backward_stepper(
            block: AbstractBlock,
            pass_tag: str
        ) -> AbstractBlock:
            block = rhs(block, pass_tag=pass_tag)
            tend = block.tend.to_state(pass_tag=pass_tag)
            new_state = update(
                old_state=block.state,
                tend=tend,
                dt=dt_seconds
            )
            return eqx.tree_at(
                lambda b: b.state,
                block,
                replace=new_state
            )

        def step_fn(block: AbstractBlock) -> AbstractBlock:
            # Forward Euler for explicit terms
            block = block.reset()
            star_block = forward_backward_stepper(
                block,
                pass_tag="forward_pass"
            )
            # Backward Euler for explicit terms
            new_block = forward_backward_stepper(
                star_block,
                pass_tag="backward_pass"
            )
            return new_block

        return step_fn
