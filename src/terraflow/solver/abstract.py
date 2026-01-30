from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Generic, Tuple
from typing import overload, Self, TYPE_CHECKING
import datetime
import jax

from terraflow.util import typing, device, history as hy
from terraflow.util.datetime import DateTimeManager

if TYPE_CHECKING:
    from terraflow.discretization.abstract import (
        AbstractBlock,
        AbstractFields,
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
    """
    Returns a version of fn() that is repeatedly applied `steps` times.
    The returned function only returns the final output.
    """
    if steps == 1:
        return fn

    def f_repeated(x_initial: tuple[typing.PyTree]) -> typing.PyTree:
        def g(x, _): return (fn(x), None)
        x_final, _ = scan_fn(g, x_initial, xs=None, length=steps)
        return x_final
    return f_repeated


def repeated_with_inout(
    fn: Callable,
    steps: int,
    scan_fn: Callable = jax.lax.scan
) -> Callable:
    """
    Returns a version of fn() that is repeatedly applied `steps` times.
    The returned function returns both the final output and the input
    from the previous step.
    """
    if steps == 1:
        return fn

    def f_repeated(carry_in: tuple[typing.PyTree, typing.PyTree]) -> tuple[typing.PyTree, typing.PyTree]:
        def g(carry, _):
            old_block, new_block = carry
            new_block = fn(old_block)
            return (new_block, old_block), None
        (new_block, old_block), _ = scan_fn(
            g, carry_in, xs=None, length=steps
        )
        return new_block, old_block
    return f_repeated


class MultiStepWithHistory:

    def __init__(
        self,
        step: Callable,
        repeated_step: Callable,
        history: hy.AbstractHistoryset,
        outer_steps: int,
        outer_scan_fn: Callable = jax.lax.scan,
    ):
        self.step = step
        self.repeated_step = repeated_step
        self.history = history
        self.outer_steps = outer_steps
        self.outer_scan_fn = outer_scan_fn

    @overload
    def __call__(
        self,
        block: fvm_odm_trisk_1lev_vars.Block,
    ) -> fvm_odm_trisk_1lev_vars.Block:
        ...

    @overload
    def __call__(
        self,
        block: AbstractBlock,
    ) -> AbstractBlock:
        ...

    def __call__(
        self,
        block: AbstractBlock,
    ) -> AbstractBlock:
        """Returns a version of this MultiStep that outputs fields."""
        old_block = block.diagnose()
        self.history.output_prepare(0)
        new_block = self.step(old_block)
        self.history.output(
            old_block=old_block,
            new_block=new_block,
            save_idx=0
        )
        final, _ = self.outer_scan_fn(
            self.repeated_step, (new_block, 1), xs=None, length=self.outer_steps
        )[0]
        return final


class MultiStep:

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
        x: fvm_odm_trisk_1lev_vars.Block,
    ) -> Tuple[fvm_odm_trisk_1lev_vars.Block, fvm_odm_trisk_1lev_vars.State]:
        ...

    @overload
    def __call__(
        self,
        x: AbstractBlock,
    ) -> Tuple[AbstractBlock, AbstractFields]:
        ...

    def __call__(
        self,
        x: AbstractBlock,
    ) -> Tuple[AbstractBlock, AbstractFields]:
        """Returns a version of this MultiStep that outputs fields."""
        final, trajectory = self.outer_scan_fn(
            self.step, x, xs=None, length=self.outer_steps
        )
        return final, trajectory


def io_callback(
    step_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    *,
    outer_scan_fn: Callable = jax.lax.scan,
    inner_scan_fn: Callable = jax.lax.scan,
    history: hy.AbstractHistoryset
) -> MultiStepWithHistory:
    if inner_steps != 1:
        repeated_step_fn = repeated_with_inout(
            fn=step_fn,
            steps=inner_steps,
            scan_fn=inner_scan_fn
        )
    else:
        raise ValueError(f"Unsupported format: {format}")

    def step(carry_in, _):
        block, save_idx = carry_in
        history.output_prepare(save_idx)
        new_block, old_block = repeated_step_fn((block, block))
        old_block = old_block.diagnose()
        history.output(
            old_block=old_block,
            new_block=new_block,
            save_idx=save_idx
        )
        return (new_block, save_idx + 1), None

    return MultiStepWithHistory(
        step=step_fn,
        repeated_step=step,
        history=history,
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
) -> MultiStep:
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

    return MultiStep(step, outer_steps, outer_scan_fn)


ODE = TypeVar("ODE", bound=AbstractODE)


class AbstractSolver(Generic[ODE], ABC):
    dt: datetime.timedelta
    save_interval: datetime.timedelta

    def __init__(
        self,
        equation: ODE,
        time_manager: DateTimeManager,
        dev: device.AbstractDevice
    ):
        self.equation = equation
        self.time_manager = time_manager
        self.dt = time_manager.dt_dyn
        self.dev = dev

    @abstractmethod
    def stepper(self, *args, **kwargs) -> Callable:
        raise NotImplementedError

    def trajectory(
        self,
        *,
        start_with_input: bool = False,
        post_process_fn: Callable = lambda x: x,
        outer_scan_fn=jax.lax.scan,
        inner_scan_fn=jax.lax.scan,
    ) -> MultiStep:
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

        stepper = self.stepper()

        return trajectory_from_step(
            stepper,
            outer_steps=outer_steps,
            inner_steps=inner_steps,
            start_with_input=start_with_input,
            post_process_fn=post_process_fn,
            outer_scan_fn=outer_scan_fn,
            inner_scan_fn=inner_scan_fn,
        )

    def io_callback(
        self,
        history: hy.AbstractHistoryset,
        *,
        outer_scan_fn: Callable = jax.lax.scan,
        inner_scan_fn: Callable = jax.lax.scan,
    ) -> MultiStepWithHistory:
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

        stepper = self.stepper()

        return io_callback(
            stepper,
            outer_steps,
            inner_steps,
            outer_scan_fn=outer_scan_fn,
            inner_scan_fn=inner_scan_fn,
            history=history
        )


class AbstractRungeKuttaSolver(AbstractSolver[ODE]):
    pass
