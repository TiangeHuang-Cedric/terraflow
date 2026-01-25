from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Generic, Tuple
from typing import Literal, overload, Self, TYPE_CHECKING
import datetime
import jax

from terraflow.util import typing, device, history
from terraflow.util.datetime import DateTimeManager

if TYPE_CHECKING:
    from terraflow.discretization.abstract import (
        AbstractBlockFields,
        AbstractBlockArrays,
        AbstractFieldBasedVariableSet,
        AbstractArrayBasedVariableSet
    )
    import terraflow.discretization.fvm.odm.trisk.single_lev.dynamics_types as fvm_odm_trisk_1lev_vars


Array = typing.Array


class AbstractODE(ABC):

    @abstractmethod
    def build_default(self) -> Self:
        raise NotImplementedError


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


class HistoryMultiStepWithFieldsOutput:

    def __init__(
        self,
        step: Callable,
        repeated_step: Callable,
        history_file: history.AbstractHistory,
        outer_steps: int,
        outer_scan_fn: Callable = jax.lax.scan,
    ):
        self.step = step
        self.repeated_step = repeated_step
        self.history_file = history_file
        self.outer_steps = outer_steps
        self.outer_scan_fn = outer_scan_fn

    @overload
    def __call__(
        self,
        block: fvm_odm_trisk_1lev_vars.BlockFields,
    ) -> fvm_odm_trisk_1lev_vars.BlockFields:
        ...

    @overload
    def __call__(
        self,
        block: AbstractBlockFields,
    ) -> AbstractBlockFields:
        ...

    def __call__(
        self,
        block: AbstractBlockFields,
    ) -> AbstractBlockFields:
        """Returns a version of this MultiStep that outputs fields."""
        block_array = block.to_arrays()
        self.history_file.output_prepare(0)
        block_array = self.step(block_array)
        self.history_file.output(block_array, 0)
        block_array = block_array.reset()
        final_array, _ = self.outer_scan_fn(
            self.repeated_step, (block_array, 1), xs=None, length=self.outer_steps
        )[0]
        return final_array.to_fields()


class HistoryMultiStepWithArrayOutput:

    def __init__(
        self,
        step: Callable,
        repeated_step: Callable,
        history_file: history.AbstractHistory,
        outer_steps: int,
        outer_scan_fn: Callable = jax.lax.scan,
    ):
        self.step = step
        self.repeated_step = repeated_step
        self.history_file = history_file
        self.outer_steps = outer_steps
        self.outer_scan_fn = outer_scan_fn

    @overload
    def __call__(
        self,
        block: fvm_odm_trisk_1lev_vars.BlockFields,
    ) -> fvm_odm_trisk_1lev_vars.BlockArrays:
        ...

    @overload
    def __call__(
        self,
        block: AbstractBlockFields,
    ) -> AbstractBlockArrays:
        ...

    def __call__(
        self,
        block: AbstractBlockFields,
    ) -> AbstractBlockArrays:
        """Returns a version of this MultiStep that outputs arrays."""
        block_array = block.to_arrays()
        self.history_file.output_prepare(0)
        block_array = self.step(block_array)
        self.history_file.output(block_array, 0)
        block_array = block_array.reset()
        final_array, _ = self.outer_scan_fn(
            self.repeated_step, (block_array, 1), xs=None, length=self.outer_steps
        )[0]
        return final_array


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
        x: fvm_odm_trisk_1lev_vars.BlockFields,
    ) -> Tuple[fvm_odm_trisk_1lev_vars.BlockFields, fvm_odm_trisk_1lev_vars.StateFields]:
        ...

    @overload
    def __call__(
        self,
        x: AbstractBlockFields,
    ) -> Tuple[AbstractBlockFields, AbstractFieldBasedVariableSet]:
        ...

    def __call__(
        self,
        x: AbstractBlockFields,
    ) -> Tuple[AbstractBlockFields, AbstractFieldBasedVariableSet]:
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
        x: fvm_odm_trisk_1lev_vars.BlockFields,
    ) -> Tuple[fvm_odm_trisk_1lev_vars.BlockArrays, fvm_odm_trisk_1lev_vars.StateArrays]:
        ...

    @overload
    def __call__(
        self,
        x: AbstractBlockFields,
    ) -> Tuple[AbstractBlockArrays, AbstractArrayBasedVariableSet]:
        ...

    def __call__(
        self,
        x: AbstractBlockFields,
    ) -> Tuple[AbstractBlockArrays, AbstractArrayBasedVariableSet]:
        """Returns a version of this MultiStep that outputs arrays."""
        x_array = x.to_arrays()
        return self.outer_scan_fn(
            self.step, x_array, xs=None, length=self.outer_steps
        )


def io_callback(
    step_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    *,
    outer_scan_fn: Callable = jax.lax.scan,
    inner_scan_fn: Callable = jax.lax.scan,
    output_fields: bool = True,
    history_file: history.AbstractHistory
) -> HistoryMultiStepWithArrayOutput | HistoryMultiStepWithFieldsOutput:
    if inner_steps != 1:
        repeated_step_fn = repeated(
            fn=step_fn,
            steps=inner_steps,
            scan_fn=inner_scan_fn
        )
    else:
        raise ValueError(f"Unsupported format: {format}")

    def step(carry_in, _):
        old_block, save_idx = carry_in
        history_file.output_prepare(save_idx)
        new_block = repeated_step_fn(old_block)
        history_file.output(new_block, save_idx)
        return (new_block, save_idx + 1), None

    if output_fields:
        return HistoryMultiStepWithFieldsOutput(
            step=step_fn,
            repeated_step=step,
            history_file=history_file,
            outer_steps=outer_steps,
            outer_scan_fn=outer_scan_fn
        )
    else:
        return HistoryMultiStepWithArrayOutput(
            step=step_fn,
            repeated_step=step,
            history_file=history_file,
            outer_steps=outer_steps,
            outer_scan_fn=outer_scan_fn
        )


def trajectory_from_step(
    step_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    *,
    start_with_input: bool = False,
    post_process_fn: Callable = lambda x: x.state,
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

    def step(old_block, _):
        new_block = step_fn(old_block)
        frame = old_block.state if start_with_input else new_block.state
        return new_block, post_process_fn(frame)

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
        equation: ODE,
        dt: datetime.timedelta,
        time_manager: DateTimeManager,
        dev: device.AbstractDevice
    ):
        equation.build_default()
        self.equation = equation
        self.time_manager = time_manager
        self.dt = dt
        self.dev = dev

    @abstractmethod
    def stepper(self, equation: ODE) -> Callable:
        raise NotImplementedError

    @overload
    def trajectory(
        self,
        equation: ODE,
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
        *,
        start_with_input: bool = False,
        post_process_fn: Callable = lambda x: x,
        outer_scan_fn=jax.lax.scan,
        inner_scan_fn=jax.lax.scan,
        output_fields: bool = True,
    ) -> MultiStepWithFieldsOutput | MultiStepWithArrayOutput:
        """Solve the ODE from start_time to end_time.

        Args:
            equation: Equation to solve the ODE.
            time_manager: DateTimeManager managing start and end times.
        """
        start_time = self.time_manager.start_time
        end_time = self.time_manager.end_time
        save_interval = self.time_manager.save_interval
        dt = self.dt

        outer_steps = int((end_time - start_time) / save_interval) + 1
        inner_steps = int((save_interval) / dt)

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

    @overload
    def io_callback(
        self,
        equation: ODE,
        history_file: history.AbstractHistory,
        *,
        outer_scan_fn: Callable = jax.lax.scan,
        inner_scan_fn: Callable = jax.lax.scan,
        output_fields: Literal[False],
    ) -> HistoryMultiStepWithArrayOutput:
        ...

    @overload
    def io_callback(
        self,
        equation: ODE,
        history_file: history.AbstractHistory,
        *,
        outer_scan_fn: Callable = jax.lax.scan,
        inner_scan_fn: Callable = jax.lax.scan,
        output_fields: Literal[True],
    ) -> HistoryMultiStepWithFieldsOutput:
        ...

    def io_callback(
        self,
        equation: ODE,
        history_file: history.AbstractHistory,
        *,
        outer_scan_fn: Callable = jax.lax.scan,
        inner_scan_fn: Callable = jax.lax.scan,
        output_fields: bool = True,
    ) -> HistoryMultiStepWithArrayOutput | HistoryMultiStepWithFieldsOutput:
        """Solve the ODE from start_time to end_time.

        Args:
            equation: Equation to solve the ODE.
            time_manager: DateTimeManager managing start and end times.
        """
        start_time = self.time_manager.start_time
        end_time = self.time_manager.end_time
        save_interval = self.time_manager.save_interval
        dt = self.dt

        outer_steps = int((end_time - start_time) / save_interval)
        inner_steps = int((save_interval) / dt)

        stepper = self.stepper(equation)

        return io_callback(
            stepper,
            outer_steps,
            inner_steps,
            outer_scan_fn=outer_scan_fn,
            inner_scan_fn=inner_scan_fn,
            output_fields=output_fields,
            history_file=history_file
        )


class AbstractRungeKuttaSolver(AbstractSolver[ODE]):
    pass
