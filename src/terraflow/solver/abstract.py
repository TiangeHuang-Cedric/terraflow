from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Generic, Tuple
from typing import Literal, overload, TYPE_CHECKING
import datetime
import jax

from terraflow.util import typing, device

if TYPE_CHECKING:
    from terraflow.discretization.abstract import AbstractFieldBasedVariableSet
    from terraflow.discretization.abstract import AbstractArrayBasedVariableSet


class AbstractODE(ABC):
    pass


class AbstractExplicitODE(AbstractODE):

    @abstractmethod
    def explicit_terms(self, *args, **kwargs) -> typing.PyTree:
        raise NotImplementedError


class AbstractImplicitExplicitODE(AbstractODE):

    @abstractmethod
    def explicit_terms(self, *args, **kwargs) -> typing.PyTree:
        raise NotImplementedError

    @abstractmethod
    def implicit_terms(self, *args, **kwargs) -> typing.PyTree:
        raise NotImplementedError


def repeated(
    fn: Callable,
    steps: int,
    scan_fn: Callable = jax.lax.scan
) -> Callable:
    """Returns a version of fn() that is repeatedly applied `steps` times."""
    if steps == 1:
        return fn

    def f_repeated(x_initial: tuple[typing.PyTree]) -> typing.PyTree:
        def g(x, _): return (fn(x), None)
        x_final, _ = scan_fn(g, x_initial, xs=None, length=steps)
        return x_final
    return f_repeated


class MultiStepWithFieldsOutput:

    def __init__(
        self,
        step: Callable,
        outer_steps: int,
        outer_scan_fn: Callable = jax.lax.scan,
    ):
        self.step = step
        self.outer_steps = outer_steps
        self.outer_scan_fn = outer_scan_fn

    @overload
    def __call__(
        self,
        x: terraflow.equation.shallow_water.vars.unstructure.StateFields,
    ) -> Tuple[
        terraflow.equation.shallow_water.vars.unstructure.StateFields, terraflow.equation.shallow_water.vars.unstructure.StateFields]:
        ...

    @overload
    def __call__(
        self,
        x: AbstractFieldBasedVariableSet,
    ) -> Tuple[AbstractFieldBasedVariableSet, AbstractFieldBasedVariableSet]:
        ...

    def __call__(
        self,
        x: AbstractFieldBasedVariableSet,
    ) -> Tuple[AbstractFieldBasedVariableSet, AbstractFieldBasedVariableSet]:
        """Returns a version of this MultiStep that outputs fields."""
        x_array = x.to_arrays()
        final_array, trajectory_array = self.outer_scan_fn(
            self.step, x_array, xs=None, length=self.outer_steps
        )
        final = final_array.to_fields()
        trajectory = trajectory_array.to_fields()
        return final, trajectory


class MultiStepWithArrayOutput:

    def __init__(
        self,
        step: Callable,
        outer_steps: int,
        outer_scan_fn: Callable = jax.lax.scan,
    ):
        self.step = step
        self.outer_steps = outer_steps
        self.outer_scan_fn = outer_scan_fn

    @overload
    def __call__(
        self,
        x: terraflow.equation.shallow_water.vars.unstructure.StateFields,
    ) -> Tuple[
        terraflow.equation.shallow_water.vars.unstructure.StateArrays, terraflow.equation.shallow_water.vars.unstructure.StateArrays]:
        ...

    @overload
    def __call__(
        self,
        x: AbstractFieldBasedVariableSet,
    ) -> Tuple[AbstractArrayBasedVariableSet, AbstractArrayBasedVariableSet]:
        ...

    def __call__(
        self,
        x: AbstractFieldBasedVariableSet,
    ) -> Tuple[AbstractArrayBasedVariableSet, AbstractArrayBasedVariableSet]:
        """Returns a version of this MultiStep that outputs arrays."""
        x_array = x.to_arrays()
        return self.outer_scan_fn(
            self.step, x_array, xs=None, length=self.outer_steps
        )


@overload
def trajectory_from_step(
    step_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    *,
    start_with_input: bool = False,
    post_process_fn: Callable = lambda x: x,
    outer_scan_fn: Callable = jax.lax.scan,
    inner_scan_fn: Callable = jax.lax.scan,
    output_fields: Literal[True],
) -> MultiStepWithFieldsOutput:
    ...


@overload
def trajectory_from_step(
    step_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    *,
    start_with_input: bool = False,
    post_process_fn: Callable = lambda x: x,
    outer_scan_fn: Callable = jax.lax.scan,
    inner_scan_fn: Callable = jax.lax.scan,
    output_fields: Literal[False],
) -> MultiStepWithArrayOutput:
    ...


def trajectory_from_step(
    step_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    *,
    start_with_input: bool = False,
    post_process_fn: Callable = lambda x: x,
    outer_scan_fn: Callable = jax.lax.scan,
    inner_scan_fn: Callable = jax.lax.scan,
    output_fields: bool = True,
) -> MultiStepWithArrayOutput | MultiStepWithFieldsOutput:
    """Returns a function that accumulates repeated applications of `step_fn`.

    Compute a trajectory by repeatedly calling `step_fn()`
    `outer_steps * inner_steps` times.

    Args:
      step_fn: function that takes a state and returns state after one time step.
      outer_steps: number of steps to save in the generated trajectory.
      inner_steps: number of repeated calls to step_fn() between saved steps.
      start_with_input: if True, output the trajectory at steps [0, ..., steps-1]
        instead of steps [1, ..., steps].
      post_process_fn: function to apply to trajectory outputs.
      outer_scan_fn: scan function to use for outer (saved) steps.
      inner_scan_fn: scan function to use for inner (unsaved) steps.

    Returns:
      A function that takes an initial state and returns a tuple consisting of:
        (1) the final frame of the trajectory.
        (2) trajectory of length `outer_steps` representing time evolution.
    """
    if inner_steps != 1:
        step_fn = repeated(step_fn, inner_steps, inner_scan_fn)

    def step(carry_in, _):
        carry_out = step_fn(carry_in)
        frame = carry_in if start_with_input else carry_out
        return carry_out, post_process_fn(frame)

    if output_fields:
        return MultiStepWithFieldsOutput(step, outer_steps, outer_scan_fn)
    else:
        return MultiStepWithArrayOutput(step, outer_steps, outer_scan_fn)


ODE = TypeVar("ODE", bound=AbstractODE)


class AbstractSolver(Generic[ODE], ABC):
    dt: datetime.timedelta
    save_interval: datetime.timedelta

    def __init__(
        self,
        dt: datetime.timedelta,
        save_interval: datetime.timedelta,
        dev: device.AbstractDevice
    ):
        self.dt = dt
        self.save_interval = save_interval
        self.dev = dev

    @abstractmethod
    def stepper(self, equation: ODE) -> Callable:
        raise NotImplementedError

    @overload
    def trajectory(
        self,
        equation: ODE,
        start_time: datetime.datetime | str,
        end_time: datetime.datetime | str,
        *,
        start_with_input: bool = False,
        post_process_fn: Callable = lambda x: x,
        outer_scan_fn: Callable = jax.lax.scan,
        inner_scan_fn: Callable = jax.lax.scan,
        output_fields: Literal[True],
    ) -> MultiStepWithFieldsOutput:
        ...

    @overload
    def trajectory(
        self,
        equation: ODE,
        start_time: datetime.datetime | str,
        end_time: datetime.datetime | str,
        *,
        start_with_input: bool = False,
        post_process_fn: Callable = lambda x: x,
        outer_scan_fn: Callable = jax.lax.scan,
        inner_scan_fn: Callable = jax.lax.scan,
        output_fields: Literal[False],
    ) -> MultiStepWithArrayOutput:
        ...

    def trajectory(
        self,
        equation: ODE,
        start_time: datetime.datetime | str,
        end_time: datetime.datetime | str,
        *,
        start_with_input: bool = False,
        post_process_fn: Callable = lambda x: x,
        outer_scan_fn=jax.lax.scan,
        inner_scan_fn=jax.lax.scan,
        output_fields: bool = True,
    ) -> MultiStepWithFieldsOutput | MultiStepWithArrayOutput:
        """Solve the ODE from start_time to end_time.

        Args:
            start_time: Initial time.
            end_time: Final time.
            equation: Equation to solve the ODE.
        """
        if isinstance(start_time, str):
            start_time = datetime.datetime.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = datetime.datetime.fromisoformat(end_time)

        outer_steps = int((end_time - start_time) / self.save_interval)
        inner_steps = int((self.save_interval) / self.dt)

        stepper = self.stepper(equation)

        return trajectory_from_step(
            stepper,
            outer_steps=outer_steps,
            inner_steps=inner_steps,
            start_with_input=start_with_input,
            post_process_fn=post_process_fn,
            outer_scan_fn=outer_scan_fn,
            inner_scan_fn=inner_scan_fn,
            output_fields=output_fields,
        )


class AbstractRungeKuttaSolver(AbstractSolver[ODE]):
    pass
