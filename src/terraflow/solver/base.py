from typing import List, Callable
import datetime
import tree_math
import jax.numpy as jnp

from terraflow.util import typing, device
from terraflow.solver.abstract import AbstractExplicitODE
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
            # TODO: Check if states0 points to the same memory as states
            # states0 = copy.deepcopy(states)
            states0 = states
            for irk in range(nrk):
                tendency = rhs(states)
                states = states0 + tendency * dt_rk_seconds[irk]
            return states

        return tree_math.wrap(step_fn)


class ForwardBackwardEulerSolver(AbstractExplicitODE):
    pass
