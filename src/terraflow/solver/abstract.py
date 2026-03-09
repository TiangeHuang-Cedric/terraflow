from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    Callable,
    TypeVar,
    Generic,
    Tuple,
    overload,
    Self,
    TYPE_CHECKING,
    Optional
)
from loguru import logger
import datetime
import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp
from jax._src import pjit

from terraflow.util import typing, device, history as hy
from terraflow.util.datetime import DateTimeManager

if TYPE_CHECKING:
    from terraflow.discretization.abstract import (
        AbstractBlock,
        AbstractFields,
    )
    import terraflow.discretization.fvm.odm.trisk.dynamics_types as fvm_odm_trisk_vars


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

    def f_repeated(x_initial: typing.PyTree) -> typing.PyTree:

        def g(old_x, _):
            new_x, _ = fn(old_x)
            return (new_x, None)

        new_x_final, _ = scan_fn(
            g,
            x_initial,
            xs=None,
            length=steps
        )

        return new_x_final

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

    def f_repeated(x_initial: typing.PyTree) -> tuple[typing.PyTree, typing.PyTree]:
        carry_in = (x_initial, x_initial)

        def g(carry, _):
            old_block, _ = carry
            new_block, old_block = fn(old_block)
            return (new_block, old_block), None

        (new_block, old_block), _ = scan_fn(
            g, carry_in, xs=None, length=steps
        )

        return new_block, old_block

    return f_repeated


class MultiStepWithHistory:

    def __init__(
        self,
        step_fn: Callable,
        inner_step_fn: Callable,
        history: hy.abstract.AbstractHistoryset,
        outer_steps: int,
        outer_scan_fn: Callable = jax.lax.scan,
    ):
        self.step_fn = step_fn
        self.inner_step_fn = inner_step_fn
        self.history = history
        self.outer_steps = outer_steps
        self.outer_scan_fn = outer_scan_fn

    @overload
    def __call__(
        self,
        block: fvm_odm_trisk_vars.Block,
    ) -> fvm_odm_trisk_vars.Block:
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
        self.history.output_prepare(0)
        new_block, old_block = self.step_fn(block)
        old_block = old_block.diagnose()
        self.history.output(
            block=old_block,
            save_idx=0,
            first_call=True
        )
        final, _ = self.outer_scan_fn(
            self.inner_step_fn,
            (new_block, 1),
            xs=None,
            length=self.outer_steps
        )[0]
        return final


class MultiStepWithTrajectory:

    def __init__(
        self,
        step_fn: Callable,
        inner_step_fn: Callable,
        outer_steps: int,
        start_with_input: bool = False,
        post_process_fn: Callable = lambda x: x.state,
        outer_scan_fn: Callable = jax.lax.scan,
    ):
        self.step_fn = step_fn
        self.inner_step_fn = inner_step_fn
        self.outer_steps = outer_steps
        self.start_with_input = start_with_input
        self.post_process_fn = post_process_fn
        self.outer_scan_fn = outer_scan_fn

    @overload
    def __call__(
        self,
        block: fvm_odm_trisk_vars.Block,
    ) -> Tuple[fvm_odm_trisk_vars.State, fvm_odm_trisk_vars.State]:
        ...

    @overload
    def __call__(
        self,
        block: AbstractBlock,
    ) -> Tuple[AbstractFields, AbstractFields]:
        ...

    def __call__(
        self,
        block: AbstractBlock,
    ) -> Tuple[AbstractFields, AbstractFields]:
        new_block, old_block = self.step_fn(block)
        _, trajectory = self.outer_scan_fn(
            self.inner_step_fn,
            new_block,
            xs=None,
            length=self.outer_steps
        )
        if self.start_with_input:
            trajectory = jax.tree_util.tree_map(
                lambda o, t: jnp.concatenate(
                    [jnp.expand_dims(o, axis=0), t],
                    axis=0
                ),
                self.post_process_fn(old_block),
                trajectory,
                is_leaf=lambda x: isinstance(x, Array)
            )
        final = jax.tree_util.tree_map(
            lambda s: s[-1, ...],
            trajectory
        )
        return final, trajectory


class MultiStepWithIntegration:

    def __init__(
        self,
        step_fn: Callable,
        inner_step_fn: Callable,
        outer_steps: int,
        outer_scan_fn: Callable = jax.lax.scan,
    ):
        self.step_fn = step_fn
        self.inner_step_fn = inner_step_fn
        self.outer_steps = outer_steps
        self.outer_scan_fn = outer_scan_fn

    @overload
    def __call__(
        self,
        block: fvm_odm_trisk_vars.Block,
    ) -> fvm_odm_trisk_vars.Block:
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
        new_block, _ = self.step_fn(block)
        _, final_block = self.outer_scan_fn(
            self.inner_step_fn,
            (new_block, new_block),
            xs=None,
            length=self.outer_steps
        )[0]
        return final_block.diagnose()


def io_callback(
    step_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    *,
    with_jit: bool = True,
    outer_scan_fn: Callable = jax.lax.scan,
    inner_scan_fn: Callable = jax.lax.scan,
    history: hy.abstract.AbstractHistoryset
) -> MultiStepWithHistory | pjit.JitWrapped:
    if inner_steps != 1:
        repeated_step_fn = repeated_with_inout(
            fn=step_fn,
            steps=inner_steps,
            scan_fn=inner_scan_fn
        )
    else:
        repeated_step_fn = step_fn

    def inner_step_fn(carry_in, _):
        block, save_idx = carry_in
        history.output_prepare(save_idx)
        new_block, old_block = repeated_step_fn(block)
        old_block = old_block.diagnose()
        history.output(
            block=old_block,
            save_idx=save_idx
        )
        return (new_block, save_idx + 1), None

    return_fn = MultiStepWithHistory(
        step_fn=step_fn,
        inner_step_fn=inner_step_fn,
        history=history,
        outer_steps=outer_steps,
        outer_scan_fn=outer_scan_fn
    )

    if with_jit:
        return jax.jit(return_fn)
    else:
        return return_fn


def trajectory_from_step(
    step_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    *,
    with_jit: bool = True,
    start_with_input: bool = False,
    post_process_fn: Callable = lambda x: x.state,
    outer_scan_fn: Callable = jax.lax.scan,
    inner_scan_fn: Callable = jax.lax.scan,
) -> MultiStepWithTrajectory | pjit.JitWrapped:
    """Returns a function that accumulates repeated applications of `step_fn`.

    Compute a trajectory by repeatedly calling `step_fn()`
    `outer_steps * inner_steps` times.

    Args:
      step_fn: function that takes a state and returns state after one time step.
      outer_steps: number of steps to save in the generated trajectory.
      inner_steps: number of repeated calls to step_fn() between saved steps.
      start_with_input: if True, output the trajectory at steps [0, ..., steps]
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
        repeated_step_fn = repeated_with_inout(
            fn=step_fn,
            steps=inner_steps,
            scan_fn=inner_scan_fn
        )
    else:
        repeated_step_fn = step_fn

    def inner_step_fn(block, _):
        new_block, old_block = repeated_step_fn(block)
        old_block = old_block.diagnose()
        return new_block, post_process_fn(old_block)

    return_fn = MultiStepWithTrajectory(
        step_fn=step_fn,
        inner_step_fn=inner_step_fn,
        outer_steps=outer_steps,
        start_with_input=start_with_input,
        post_process_fn=post_process_fn,
        outer_scan_fn=outer_scan_fn
    )

    if with_jit:
        return jax.jit(return_fn)
    else:
        return return_fn


def integrate(
    step_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    *,
    with_jit: bool = True,
    outer_scan_fn: Callable = jax.lax.scan,
    inner_scan_fn: Callable = jax.lax.scan,
) -> MultiStepWithIntegration | pjit.JitWrapped:
    if inner_steps != 1:
        repeated_step_fn = repeated_with_inout(
            fn=step_fn,
            steps=inner_steps,
            scan_fn=inner_scan_fn
        )
    else:
        repeated_step_fn = step_fn

    def inner_step_fn(carry_in, _):
        old_block, _ = carry_in
        new_block, old_block = repeated_step_fn(old_block)
        return (new_block, old_block), None

    return_fn = MultiStepWithIntegration(
        step_fn=step_fn,
        inner_step_fn=inner_step_fn,
        outer_steps=outer_steps,
        outer_scan_fn=outer_scan_fn
    )

    if with_jit:
        return jax.jit(return_fn)
    else:
        return return_fn


ODE = TypeVar("ODE", bound=AbstractODE)


def _get_final_data_with_history(
    field_name: str,
    history_filepath: str
) -> np.ndarray:
    ds = xr.open_dataset(history_filepath)
    h_final = np.asarray(ds[field_name].data[-1, ...])
    ds.close()
    return h_final


def _get_final_data_with_trajectory(
    field_name: str,
    trajectory_final: AbstractFields
) -> np.ndarray:
    t_final = np.asarray(
        getattr(trajectory_final, field_name).data,
    ).squeeze()
    return t_final


def _get_final_data_with_integrated(
    field_name: str,
    integrated: AbstractBlock
) -> np.ndarray:
    i_final = np.asarray(
        getattr(integrated.state, field_name).data,
    ).squeeze()
    return i_final


def diagnose_precision_with_history_trajectory(
    field_name: str,
    history_filepath: str,
    trajectory_final: AbstractFields,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> bool:
    h_final = _get_final_data_with_history(
        field_name=field_name, history_filepath=history_filepath
    )
    t_final = _get_final_data_with_trajectory(
        field_name=field_name, trajectory_final=trajectory_final
    )
    logger.info(
        f"History final {field_name} "
        f"max: {np.max(h_final)}, min: {np.min(h_final)}, mean: {np.mean(h_final)}"
    )
    logger.info(
        f"Trajectory final {field_name} "
        f"max: {np.max(t_final)}, min: {np.min(t_final)}, mean: {np.mean(t_final)}"
    )
    assert np.allclose(
        h_final,
        t_final,
        rtol=rtol,
        atol=atol
    ), "Final data from history and trajectory do not match within tolerance."
    assert (
        h_final == t_final
    ).all(), "Final data from history and trajectory do not match exactly."
    return True


def diagnose_precision_with_history_integrated(
    field_name: str,
    history_filepath: str,
    integrated: AbstractBlock,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> bool:
    h_final = _get_final_data_with_history(
        field_name=field_name, history_filepath=history_filepath
    )
    i_final = _get_final_data_with_integrated(
        field_name=field_name, integrated=integrated
    )
    logger.info(
        f"History final {field_name} "
        f"max: {np.max(h_final)}, min: {np.min(h_final)}, mean: {np.mean(h_final)}"
    )
    logger.info(
        f"Integrated final {field_name} "
        f"max: {np.max(i_final)}, min: {np.min(i_final)}, mean: {np.mean(i_final)}"
    )
    assert np.allclose(
        h_final,
        i_final,
        rtol=rtol,
        atol=atol
    ), "Final data from history and integrated do not match within tolerance."
    assert (
        h_final == i_final
    ).all(), "Final data from history and integrated do not match exactly."
    return True


def diagnose_precision_with_trajectory_integrated(
    field_name: str,
    trajectory_final: AbstractFields,
    integrated: AbstractBlock,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> bool:
    t_final = _get_final_data_with_trajectory(
        field_name=field_name, trajectory_final=trajectory_final
    )
    i_final = _get_final_data_with_integrated(
        field_name=field_name, integrated=integrated
    )
    logger.info(
        f"Trajectory final {field_name} "
        f"max: {np.max(t_final)}, min: {np.min(t_final)}, mean: {np.mean(t_final)}"
    )
    logger.info(
        f"Integrated final {field_name} "
        f"max: {np.max(i_final)}, min: {np.min(i_final)}, mean: {np.mean(i_final)}"
    )
    assert np.allclose(
        t_final,
        i_final,
        rtol=rtol,
        atol=atol
    ), "Final data from trajectory and integrated do not match within tolerance."
    assert (
        t_final == i_final
    ).all(), "Final data from trajectory and integrated do not match exactly."
    return True


class AbstractSolver(Generic[ODE], ABC):
    dt: datetime.timedelta
    save_interval: datetime.timedelta

    def __init__(
        self,
        equation: ODE,
        time_manager: DateTimeManager,
        dev: device.AbstractDevice,
        with_jit: bool = True
    ):
        self.equation = equation
        self.time_manager = time_manager
        self.dt = time_manager.dt_dyn
        self.dev = dev
        self.with_jit = with_jit

    @abstractmethod
    def stepper(self, *args, **kwargs) -> Callable:
        raise NotImplementedError

    def io_callback(
        self,
        history: hy.abstract.AbstractHistoryset,
        *,
        outer_scan_fn: Callable = jax.lax.scan,
        inner_scan_fn: Callable = jax.lax.scan,
    ) -> MultiStepWithHistory | pjit.JitWrapped:
        """Solve the ODE from start_time to end_time.

        Args:
            history: Historyset to save the trajectory.
            outer_scan_fn: scan function to use for outer (saved) steps.
            inner_scan_fn: scan function to use for inner (unsaved) steps.

        Returns:
            A function that takes an initial state and returns the final state after
            solving the ODE from start_time to end_time, while saving intermediate
            states to history.
        """
        start_time = self.time_manager.start_time
        end_time = self.time_manager.end_time
        save_interval = self.time_manager.save_interval
        dt = self.dt

        outer_steps = int((end_time - start_time) / save_interval)
        inner_steps = int((save_interval) / dt)

        if self.with_jit:
            stepper = jax.jit(self.stepper())
        else:
            stepper = self.stepper()

        return io_callback(
            stepper,
            outer_steps,
            inner_steps,
            with_jit=self.with_jit,
            outer_scan_fn=outer_scan_fn,
            inner_scan_fn=inner_scan_fn,
            history=history
        )

    def trajectory(
        self,
        *,
        start_with_input: bool = False,
        post_process_fn: Callable = lambda x: x.state,
        outer_scan_fn=jax.lax.scan,
        inner_scan_fn=jax.lax.scan,
    ) -> MultiStepWithTrajectory | pjit.JitWrapped:
        """Solve the ODE from start_time to end_time.

        Args:
            start_with_input: if True, output the trajectory at steps [0, ..., steps]
                instead of steps [1, ..., steps].
            post_process_fn: function to apply to trajectory outputs.
            outer_scan_fn: scan function to use for outer (saved) steps.
            inner_scan_fn: scan function to use for inner (unsaved) steps.

        Returns:
            A function that takes an initial state and returns a tuple consisting of:
                (1) the final frame of the trajectory.
                (2) trajectory of length `outer_steps` representing time evolution.
        """
        start_time = self.time_manager.start_time
        end_time = self.time_manager.end_time
        save_interval = self.time_manager.save_interval
        dt = self.dt

        outer_steps = int((end_time - start_time) / save_interval)
        inner_steps = int((save_interval) / dt)

        if self.with_jit:
            stepper = jax.jit(self.stepper())
        else:
            stepper = self.stepper()

        return trajectory_from_step(
            stepper,
            outer_steps=outer_steps,
            inner_steps=inner_steps,
            with_jit=self.with_jit,
            start_with_input=start_with_input,
            post_process_fn=post_process_fn,
            outer_scan_fn=outer_scan_fn,
            inner_scan_fn=inner_scan_fn,
        )

    def integrate(
        self,
        *,
        outer_scan_fn=jax.lax.scan,
        inner_scan_fn=jax.lax.scan,
    ) -> MultiStepWithIntegration | pjit.JitWrapped:
        """Integrate the ODE from start_time to end_time.

        Args:
            outer_scan_fn: scan function to use for outer (saved) steps.
            inner_scan_fn: scan function to use for inner (unsaved) steps.

        Returns:
            A function that takes an initial block and returns the final block after
            integrating the ODE from start_time to end_time.
        """
        start_time = self.time_manager.start_time
        end_time = self.time_manager.end_time
        save_interval = self.time_manager.save_interval
        dt = self.dt

        outer_steps = int((end_time - start_time) / save_interval)
        inner_steps = int((save_interval) / dt)

        if self.with_jit:
            stepper = jax.jit(self.stepper())
        else:
            stepper = self.stepper()

        return integrate(
            stepper,
            outer_steps=outer_steps,
            inner_steps=inner_steps,
            with_jit=self.with_jit,
            outer_scan_fn=outer_scan_fn,
            inner_scan_fn=inner_scan_fn,
        )

    def diagnose_precision(
        self,
        field_name: str,
        history_filepath: Optional[str] = None,
        trajectory_final: Optional[AbstractFields] = None,
        integrated: Optional[AbstractBlock] = None,
    ) -> bool:
        if (
            history_filepath is not None and
            trajectory_final is not None and
            integrated is None
        ):
            return diagnose_precision_with_history_trajectory(
                field_name=field_name,
                history_filepath=history_filepath,
                trajectory_final=trajectory_final
            )
        elif (
            history_filepath is not None and
            trajectory_final is None and
            integrated is not None
        ):
            return diagnose_precision_with_history_integrated(
                field_name=field_name,
                history_filepath=history_filepath,
                integrated=integrated
            )
        elif (
            history_filepath is None
            and trajectory_final is not None
            and integrated is not None
        ):
            return diagnose_precision_with_trajectory_integrated(
                field_name=field_name,
                trajectory_final=trajectory_final,
                integrated=integrated
            )
        else:
            raise ValueError(
                "Invalid combination of inputs for precision diagnosis. "
                "Provide either (1) history_filepath and trajectory, "
                "or (2) history_filepath and integrated, "
                "or (3) trajectory and integrated."
            )


class AbstractRungeKuttaSolver(AbstractSolver[ODE]):
    pass
