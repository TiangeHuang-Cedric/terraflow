from typing import List, Callable
import datetime
import tree_math
import jax.numpy as jnp

from terraflow.util import typing, device
from terraflow.solver.abstract import AbstractExplicitODE, AbstractSolver
from terraflow.solver.abstract import AbstractRungeKuttaSolver


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
        dt: datetime.timedelta,
        save_interval: datetime.timedelta,
        dev: device.AbstractDevice
    ):
        super().__init__(dt, save_interval, dev)
        self.nrk = 3
        self.dt_rk = [dt / (self.nrk - irk) for irk in range(self.nrk)]
        self.dt_rk_seconds = jnp.asarray(
            [dt_rk.total_seconds() for dt_rk in self.dt_rk],
            dtype=dev.float_type
        )

    def stepper(self, equation: AbstractExplicitODE) -> Callable:
        """Third-order Runge-Kutta time stepper."""
        rhs = tree_math.unwrap(equation.explicit_terms)
        nrk = self.nrk
        dt_rk_seconds = self.dt_rk_seconds

        def step_fn(states: tree_math.Vector) -> tree_math.Vector:
            states0 = states
            for irk in range(nrk):
                tendency = rhs(states)
                states = states0 + tendency * dt_rk_seconds[irk]
            return states

        return tree_math.wrap(step_fn)


class ExForwardBackwardEulerSolver(AbstractSolver[AbstractExplicitODE]):

    """Forward-Backward Euler time integration solver.

    This solver used two expicit schemes that are economical than the standard leapfrog shceme.
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
        forward_rhs = tree_math.unwrap(
            lambda x: equation.explicit_terms(x, pass_tag="forward_pass")
        )
        backward_rhs = tree_math.unwrap(
            lambda x: equation.explicit_terms(x, pass_tag="backward_pass")
        )
        dt_seconds = self.dt.total_seconds()

        def step_fn(states: tree_math.Vector) -> tree_math.Vector:
            # Forward Euler for explicit terms
            tendency_forward = forward_rhs(states)
            star_states = states + tendency_forward * dt_seconds
            # Backward Euler for explicit terms
            tendency_backward = backward_rhs(star_states)
            new_states = star_states + tendency_backward * dt_seconds
            return new_states

        return tree_math.wrap(step_fn)
