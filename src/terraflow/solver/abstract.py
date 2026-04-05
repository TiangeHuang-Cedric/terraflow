from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    Callable,
    TypeVar,
    Generic,
    Tuple,
    Self,
    TYPE_CHECKING,
    Optional,
    Any
)
import datetime
import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp
from jax._src import pjit

from terraflow.util import typing, log, device
from terraflow.util.history.abstract import AbstractHistoryset
from terraflow.util.datetime import DateTimeManager

if TYPE_CHECKING:
    from terraflow.discretization.abstract import (
        AbstractBlock,
        AbstractFields,
    )


Array = typing.Array


class AbstractODE(ABC):
    _space_operators: Any

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


ODE = TypeVar("ODE", bound=AbstractODE)


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

    def update_state_fn(self) -> Callable:
        return self.equation._space_operators.update_state

    def operator_prepare_fn(self) -> Callable:
        return self.equation._space_operators.prepare.first_call

    def diagnose_fn(self) -> Callable:
        return self.equation._space_operators.diagnose

    def terraflow_op_first_call(self) -> Callable:

        def first_call_fn(
            block: AbstractBlock,
            with_diag: bool = True
        ) -> AbstractBlock:
            block = self.operator_prepare_fn()(block)
            if with_diag:
                # when with_diag is True,
                # we will apply the diagnose function to the old block.
                block = self.diagnose_fn()(
                    block,
                    itime=block.old
                )
            return block

        return first_call_fn

    def terraflow_run(self) -> Callable:

        def run_fn(block: AbstractBlock) -> AbstractBlock:
            block = self.stepper()(block)
            block = block.swap_time_indices()
            return block

        return run_fn

    def terraflow_diagnose(self) -> Callable:
        """Returns a function that applies diagnose_fn to a block during integration."""

        def diagnose_fn(block: AbstractBlock) -> AbstractBlock:
            block = self.diagnose_fn()(
                block,
                itime=block.new
            )
            return block

        return diagnose_fn

    def io_callback(
        self,
        history: AbstractHistoryset,
        *,
        use_scan: bool = False,
        outer_scan_fn: Callable = jax.lax.scan,
        inner_scan_fn: Callable = jax.lax.scan,
    ) -> Callable[[AbstractBlock], AbstractBlock] | pjit.JitWrapped:
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

        if use_scan:
            return_fn = io_callback_in_scan(
                history=history,
                run_fn=self.terraflow_run(),
                operator_prepare_fn=self.terraflow_op_first_call(),
                diagnose_fn=self.terraflow_diagnose(),
                outer_steps=outer_steps,
                inner_steps=inner_steps,
                outer_scan_fn=outer_scan_fn,
                inner_scan_fn=inner_scan_fn,
            )
        else:
            log_every_time = self.time_manager.log_every
            log_steps = int(log_every_time / dt)
            return_fn = io_callback_in_while_loop(
                history=history,
                run_fn=self.terraflow_run(),
                op_first_call_fn=self.terraflow_op_first_call(),
                diagnose_fn=self.terraflow_diagnose(),
                total_timesteps=outer_steps * inner_steps,
                save_steps=inner_steps,
                log_steps=log_steps
            )

        if self.with_jit:
            return jax.jit(return_fn)
        else:
            return return_fn

    def trajectory(
        self,
        *,
        start_with_input: bool = False,
        post_process_fn: Callable = lambda x: x.get_state(x.new),
        outer_scan_fn=jax.lax.scan,
        inner_scan_fn=jax.lax.scan,
    ) -> Callable[[AbstractBlock], tuple[AbstractFields, ...]] | pjit.JitWrapped:
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

        return trajectory_in_scan(
            run_fn=self.terraflow_run(),
            op_first_call_fn=self.terraflow_op_first_call(),
            diagnose_fn=self.terraflow_diagnose(),
            outer_steps=outer_steps,
            inner_steps=inner_steps,
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
    ) -> Callable[[AbstractBlock], AbstractBlock] | pjit.JitWrapped:
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

        return integrate_in_scan(
            run_fn=self.terraflow_run(),
            op_first_call_fn=self.terraflow_op_first_call(),
            diagnose_fn=self.terraflow_diagnose(),
            outer_steps=outer_steps,
            inner_steps=inner_steps,
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


def io_callback_in_while_loop(
    history: AbstractHistoryset,
    run_fn: Callable,
    op_first_call_fn: Callable,
    diagnose_fn: Callable,
    total_timesteps: int,
    save_steps: int,
    log_steps: int,
) -> Callable[[AbstractBlock], AbstractBlock]:
    return wrap_fn_for_while_loop(
        history=history,
        run_fn=run_fn,
        op_first_call_fn=op_first_call_fn,
        diagnose_fn=diagnose_fn,
        total_timesteps=total_timesteps,
        save_steps=save_steps,
        log_steps=log_steps
    )


def io_callback_in_scan(
    run_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    operator_prepare_fn: Callable,
    diagnose_fn: Callable,
    history: AbstractHistoryset,
    *,
    post_process_fn: Callable = lambda x: x.get_state(x.new),
    outer_scan_fn: Callable = jax.lax.scan,
    inner_scan_fn: Callable = jax.lax.scan,
) -> Callable[[AbstractBlock], AbstractBlock]:
    return wrap_fn_for_scan(
        history=history,
        run_fn=run_fn,
        op_first_call_fn=operator_prepare_fn,
        diagnose_fn=diagnose_fn,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        post_process_fn=post_process_fn,
        outer_scan_fn=outer_scan_fn,
        inner_scan_fn=inner_scan_fn,
    )


def trajectory_in_scan(
    run_fn: Callable,
    op_first_call_fn: Callable,
    diagnose_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    *,
    start_with_input: bool = False,
    post_process_fn: Callable = lambda x: x.get_state(x.new),
    outer_scan_fn: Callable = jax.lax.scan,
    inner_scan_fn: Callable = jax.lax.scan,
) -> Callable[[AbstractBlock], Tuple[AbstractFields, AbstractFields]]:
    """Returns a function that accumulates repeated applications of `step_fn`.

    Compute a trajectory by repeatedly calling `step_fn()`
    `outer_steps * inner_steps` times.

    Args:
      run_fn: function that takes a state and returns state after one time step.
      op_first_call_fn: function to prepare the block for the first call to run_fn.
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
    outer_fn = wrap_fn_for_scan(
        run_fn=run_fn,
        op_first_call_fn=op_first_call_fn,
        diagnose_fn=diagnose_fn,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        start_with_input=start_with_input,
        post_process_fn=post_process_fn,
        outer_scan_fn=outer_scan_fn,
        inner_scan_fn=inner_scan_fn,
    )

    def return_fn(block: AbstractBlock) -> Tuple[AbstractFields, AbstractFields]:
        _, trajectory = outer_fn(block)
        if start_with_input:
            trajectory = jax.tree_util.tree_map(
                lambda o, t: jnp.concatenate(
                    [jnp.expand_dims(o, axis=0), t],
                    axis=0
                ),
                block.get_state(0),
                trajectory,
                is_leaf=lambda x: isinstance(x, Array)
            )
        final = jax.tree_util.tree_map(
            lambda s: s[-1, ...],
            trajectory
        )
        return final, trajectory

    return return_fn


def integrate_in_scan(
    run_fn: Callable,
    op_first_call_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    diagnose_fn: Callable,
    *,
    outer_scan_fn: Callable = jax.lax.scan,
    inner_scan_fn: Callable = jax.lax.scan,
) -> Callable[[AbstractBlock], AbstractBlock]:
    outer_fn = wrap_fn_for_scan(
        run_fn=run_fn,
        op_first_call_fn=op_first_call_fn,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        outer_scan_fn=outer_scan_fn,
        inner_scan_fn=inner_scan_fn,
    )

    def return_fn(block: AbstractBlock) -> AbstractBlock:
        final, _ = outer_fn(block)
        final = diagnose_fn(final)
        return final

    return return_fn


def output(
    history: AbstractHistoryset,
    block: AbstractBlock,
    save_idx: int,
    first_call: bool = False
) -> None:
    history.output(
        block=block,
        itime=block.new,
        save_idx=save_idx,
        first_call=first_call
    )


def wrap_fn_for_scan(
    run_fn: Callable,
    op_first_call_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    diagnose_fn: Optional[Callable] = None,
    start_with_input: bool = False,
    post_process_fn: Optional[Callable] = None,
    history: Optional[AbstractHistoryset] = None,
    outer_scan_fn: Callable = jax.lax.scan,
    inner_scan_fn: Callable = jax.lax.scan,
) -> Callable:
    diagnose_fn = diagnose_fn if diagnose_fn is not None else lambda block: block
    post_process_fn = post_process_fn if post_process_fn is not None else lambda block: None

    repeated_step_fn = repeated(
        fn=run_fn,
        steps=inner_steps,
        scan_fn=inner_scan_fn
    )

    if history is None:
        def outer_step_fn_without_history(block, _):
            block = repeated_step_fn(block)
            block = diagnose_fn(block)
            return block, post_process_fn(block)

        outer_step_fn = outer_step_fn_without_history
    else:
        def outer_step_fn_with_history(carry_in, _):
            block, save_idx = carry_in
            history.output_prepare(save_idx)
            block = repeated_step_fn(block)
            block = diagnose_fn(block)
            output(
                history=history,
                block=block,
                save_idx=save_idx
            )
            return (block, save_idx + 1), None

        outer_step_fn = outer_step_fn_with_history

    if history is None:
        def return_fn_without_history(block: AbstractBlock) -> tuple[AbstractBlock, Any]:
            if start_with_input:
                block = op_first_call_fn(block, with_diag=True)
            else:
                block = op_first_call_fn(block, with_diag=False)
            block = run_fn(block)
            block, trajectory = outer_scan_fn(
                outer_step_fn,
                block,
                xs=None,
                length=outer_steps
            )
            return block, trajectory

        return return_fn_without_history
    else:
        def return_fn_with_history(block: AbstractBlock) -> AbstractBlock:
            history.output_prepare(0)
            block = op_first_call_fn(block, with_diag=True)
            history.output(
                block=block,
                itime=block.old,
                save_idx=0,
                first_call=True
            )
            block = run_fn(block)
            carry = (block, 1)
            final_carry, _ = outer_scan_fn(
                outer_step_fn,
                carry,
                xs=None,
                length=outer_steps
            )
            block = final_carry[0]
            return block

        return return_fn_with_history


def wrap_fn_for_while_loop(
    run_fn: Callable,
    op_first_call_fn: Callable,
    total_timesteps: int,
    save_steps: int,
    diagnose_fn: Optional[Callable] = None,
    history: Optional[AbstractHistoryset] = None,
    log_steps: Optional[int] = None,
) -> Callable[[AbstractBlock], AbstractBlock]:
    diagnose_fn = diagnose_fn if diagnose_fn is not None else lambda block: block

    def while_loop_cond_fn(carry):
        _, itimestep = carry
        return itimestep <= total_timesteps

    if history is None or log_steps is None:
        def process_on_save_step(block):
            block = diagnose_fn(block)
            return block

        def while_loop_body_fn(carry) -> tuple[AbstractBlock, int]:
            block, itimestep = carry
            block = run_fn(block)
            block = jax.lax.cond(
                (itimestep % save_steps) == 0,
                process_on_save_step,
                lambda block: block,
                block,
            )
            block = diagnose_fn(block)
            return block, itimestep + 1
    else:

        def output_on_log_step(itimestep):
            history.output_diag(itimestep=itimestep)

        def output_on_save_step(block, save_idx):
            block = diagnose_fn(block)
            output(
                history=history,
                block=block,
                save_idx=save_idx
            )

        def while_loop_body_fn(carry) -> tuple[AbstractBlock, int]:
            block, itimestep = carry
            block = run_fn(block)
            jax.lax.cond(
                itimestep % log_steps == 0,
                output_on_log_step,
                lambda _: None,
                itimestep
            )
            jax.lax.cond(
                (itimestep % save_steps) == 0,
                output_on_save_step,
                lambda _, __: None,
                block,
                itimestep // save_steps
            )
            return block, itimestep + 1

    if history is None:
        def return_fn(block: AbstractBlock) -> AbstractBlock:
            block = op_first_call_fn(block, with_diag=False)
            block = run_fn(block)
            carry = (block, 1)
            final_carry = jax.lax.while_loop(
                cond_fun=while_loop_cond_fn,
                body_fun=while_loop_body_fn,
                init_val=carry
            )
            block = final_carry[0]
            return block
    else:
        def return_fn(block: AbstractBlock) -> AbstractBlock:
            history.output_prepare(0)
            block = op_first_call_fn(block, with_diag=True)
            history.output(
                block=block,
                itime=block.old,
                save_idx=0,
                first_call=True
            )
            block = run_fn(block)
            carry = (block, 1)
            final_carry = jax.lax.while_loop(
                cond_fun=while_loop_cond_fn,
                body_fun=while_loop_body_fn,
                init_val=carry
            )
            block = final_carry[0]
            return block

    return return_fn


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

    def f_repeated(x_initial: AbstractBlock) -> AbstractBlock:

        def g(old_x, _):
            new_x = fn(old_x)
            return (new_x, None)

        new_x_final, _ = scan_fn(
            g,
            x_initial,
            xs=None,
            length=steps
        )

        return new_x_final

    return f_repeated


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
    log.logger.info(
        f"History final {field_name} "
        f"max: {np.max(h_final)}, min: {np.min(h_final)}, mean: {np.mean(h_final)}"
    )
    log.logger.info(
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
    log.logger.info(
        f"History final {field_name} "
        f"max: {np.max(h_final)}, min: {np.min(h_final)}, mean: {np.mean(h_final)}"
    )
    log.logger.info(
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
    log.logger.info(
        f"Trajectory final {field_name} "
        f"max: {np.max(t_final)}, min: {np.min(t_final)}, mean: {np.mean(t_final)}"
    )
    log.logger.info(
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
