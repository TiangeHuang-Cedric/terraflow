from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    overload,
    Tuple,
    Union,
    Any,
    Optional,
    TypeVar,
    Generic,
    Self,
    Dict,
    Callable
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp
from jax._src import pjit
import equinox as eqx


from terraflow.util import log, typing
from terraflow.util.datetime import DateTimeManager
from terraflow.dynamics.abstract import (
    AbstractBlock,
    AbstractFields,
)

BLOCK  = TypeVar("BLOCK" , bound=AbstractBlock  )
FIELDS = TypeVar("FIELDS", bound=AbstractFields )
MODEL  = TypeVar("MODEL" , bound="AbstractModel")

INT   = typing.Int
Array = typing.Array


if TYPE_CHECKING:
    from terraflow.util.history.abstract import AbstractHistoryset
    from terraflow.model.config import Const, UserConfig
    


@overload
def compiled_with_model(
    model: MODEL,
    fun: Callable[[MODEL], MODEL],
    with_jit: bool = True
) -> Callable[[MODEL], MODEL]:
    ...


@overload
def compiled_with_model(
    model: MODEL,
    fun: Callable[[MODEL], Tuple[FIELDS, ...]],
    with_jit: bool = True
) -> Callable[[MODEL], Tuple[FIELDS, ...]]:
    ...


def compiled_with_model(
    model: MODEL,
    fun: Callable[[MODEL], Union[MODEL, Tuple[FIELDS, ...]]],
    with_jit: bool = True
) -> Callable[[MODEL], Union[MODEL, Tuple[FIELDS, ...]]]:
    import jax
    if with_jit:
        return jax.jit(fun).lower(model).compile()
    else:
        return fun


@overload
def block_until_ready(model: Dict[str, Array]) -> Dict[str, Array]:
    ...


@overload
def block_until_ready(model: MODEL) -> MODEL:
    ...


def block_until_ready(model: MODEL | Dict[str, Array]) -> MODEL | Dict[str, Array]:
    return jax.block_until_ready(model)


@dataclass
class ModelMode(ABC):
    baroclinic    : bool = field(default=False)
    hydrostatic   : bool = field(default=False)
    nonhydrostatic: bool = field(default=False)
    ideal_dry_core: bool = field(default=False)
    advection2d   : bool = field(default=False)
    advection3d   : bool = field(default=False)
    restart       : bool = field(default=False)

    @classmethod
    def from_swm(cls, restart: bool = False):
        return cls(
            baroclinic=False,
            hydrostatic=False,
            nonhydrostatic=False,
            ideal_dry_core=False,
            advection2d=False,
            advection3d=False,
            restart=restart
        )

    @classmethod
    def from_hdc(cls, ideal_dry_core: bool, restart: bool = False):
        return cls(
            baroclinic=True,
            hydrostatic=True,
            nonhydrostatic=False,
            ideal_dry_core=ideal_dry_core,
            advection2d=False,
            advection3d=False,
            restart=restart
        )

    @classmethod
    def from_adv2d(cls, restart: bool = False):
        return cls(
            baroclinic=False,
            hydrostatic=False,
            nonhydrostatic=False,
            ideal_dry_core=False,
            advection2d=True,
            advection3d=False,
            restart=restart
        )

    @classmethod
    def from_adv3d(cls, restart: bool = False):
        return cls(
            baroclinic=False,
            hydrostatic=False,
            nonhydrostatic=False,
            ideal_dry_core=False,
            advection2d=False,
            advection3d=True,
            restart=restart
        )

    def is_barotropic(self) -> bool:
        return (
            not self.baroclinic
            and not self.hydrostatic
            and not self.nonhydrostatic
            and not self.ideal_dry_core
            and not self.advection2d
            and not self.advection3d
        )

    def is_baroclinic(self) -> bool:
        return self.baroclinic

    def is_hydrostatic(self) -> bool:
        return self.is_dry_hydrostatic() or self.is_moist_hydrostatic()

    def is_nonhydrostatic(self) -> bool:
        return self.is_dry_nonhydrostatic() or self.is_moist_nonhydrostatic()

    def is_dry(self) -> bool:
        return self.is_dry_hydrostatic() or self.is_dry_nonhydrostatic()

    def is_moist(self) -> bool:
        return self.is_moist_hydrostatic() or self.is_moist_nonhydrostatic()

    def is_dry_hydrostatic(self) -> bool:
        return (
            self.baroclinic
            and self.hydrostatic
            and not self.nonhydrostatic
            and self.ideal_dry_core
            and not self.advection2d
            and not self.advection3d
        )

    def is_dry_nonhydrostatic(self) -> bool:
        return (
            self.baroclinic
            and not self.hydrostatic
            and self.nonhydrostatic
            and self.ideal_dry_core
            and not self.advection2d
            and not self.advection3d
        )

    def is_moist_hydrostatic(self) -> bool:
        return (
            self.baroclinic
            and self.hydrostatic
            and not self.nonhydrostatic
            and not self.ideal_dry_core
            and not self.advection2d
            and not self.advection3d
        )

    def is_moist_nonhydrostatic(self) -> bool:
        return (
            self.baroclinic
            and not self.hydrostatic
            and self.nonhydrostatic
            and not self.ideal_dry_core
            and not self.advection2d
            and not self.advection3d
        )

    def is_advection(self) -> bool:
        return self.advection2d or self.advection3d

    def is_advection2d(self) -> bool:
        return (
            not self.baroclinic
            and not self.hydrostatic
            and not self.nonhydrostatic
            and not self.ideal_dry_core
            and self.advection2d
            and not self.advection3d
        )

    def is_advection3d(self) -> bool:
        return (
            not self.baroclinic
            and not self.hydrostatic
            and not self.nonhydrostatic
            and not self.ideal_dry_core
            and not self.advection2d
            and self.advection3d
        )

    def is_2d_model(self) -> bool:
        return self.is_barotropic() or self.is_advection2d()

    def is_3d_model(self) -> bool:
        return self.is_baroclinic() or self.is_advection3d()
    
    
class AbstractModel(eqx.Module, ABC, Generic[BLOCK, FIELDS]):
    mode        : ModelMode       = eqx.field(static=True)
    const       : Const           = eqx.field(static=True)
    cfg         : UserConfig      = eqx.field(static=True)
    time_manager: DateTimeManager = eqx.field(static=True)
    block       : BLOCK

    @abstractmethod
    def terraflow_run(self, block: BLOCK) -> BLOCK:
        raise NotImplementedError

    @abstractmethod
    def terraflow_diagnose(self, block: BLOCK) -> BLOCK:
        raise NotImplementedError
    
    @abstractmethod
    def build_history(self, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    def io_callback(
        self,
        history: AbstractHistoryset,
        *,
        use_scan: bool = False,
        outer_scan_fn: Callable = jax.lax.scan,
        inner_scan_fn: Callable = jax.lax.scan,
    ) -> Callable[[Self], Self] | pjit.JitWrapped:
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
        dt_dyn = self.time_manager.dt_dyn

        outer_steps = int((end_time - start_time) / save_interval)
        inner_steps = int((save_interval) / dt_dyn)

        if use_scan:
            return io_callback_in_scan(
                history=history,
                run_fn=self.terraflow_run,
                diagnose_fn=self.terraflow_diagnose,
                outer_steps=outer_steps,
                inner_steps=inner_steps,
                outer_scan_fn=outer_scan_fn,
                inner_scan_fn=inner_scan_fn,
            )
        else:
            log_every_time = self.time_manager.log_every
            log_steps = int(log_every_time / dt_dyn)
            return io_callback_in_while_loop(
                history=history,
                run_fn=self.terraflow_run,
                diagnose_fn=self.terraflow_diagnose,
                total_timesteps=outer_steps * inner_steps,
                save_steps=inner_steps,
                log_steps=log_steps
            )

    def trajectory(
        self,
        *,
        start_with_input: bool = False,
        post_process_fn: Callable = (
            lambda x: x.trajectory_post_process()
        ),
        outer_scan_fn=jax.lax.scan,
        inner_scan_fn=jax.lax.scan,
    ) -> Callable[[Self], tuple[Self, dict[str, Array]]]:
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
        dt_dyn = self.time_manager.dt_dyn

        outer_steps = int((end_time - start_time) / save_interval)
        inner_steps = int((save_interval) / dt_dyn)

        return trajectory_in_scan(
            run_fn=self.terraflow_run,
            diagnose_fn=self.terraflow_diagnose,
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
        total_scan_fn=jax.lax.scan,
    ) -> Callable[[Self], Self]:
        """Integrate the ODE from start_time to end_time.

        Args:
            total_scan_fn: scan function to use for the integration.

        Returns:
            A function that takes an initial block and returns the final block after
            integrating the ODE from start_time to end_time.
        """
        start_time = self.time_manager.start_time
        end_time = self.time_manager.end_time
        dt_dyn = self.time_manager.dt_dyn

        total_steps = int((end_time - start_time) / dt_dyn)

        return integrate_in_scan(
            run_fn=self.terraflow_run,
            diagnose_fn=self.terraflow_diagnose,
            total_steps=total_steps,
            total_scan_fn=total_scan_fn,
        )

    def diagnose_precision(
        self,
        field_name: str,
        history_filepath: Optional[str] = None,
        trajectory_final: Optional[FIELDS] = None,
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


def io_callback_in_while_loop(
    history: AbstractHistoryset,
    run_fn: Callable,
    diagnose_fn: Callable,
    total_timesteps: int,
    save_steps: int,
    log_steps: int,
) -> Callable[[MODEL], MODEL]:
    return wrap_fn_for_while_loop(
        history=history,
        run_fn=run_fn,
        diagnose_fn=diagnose_fn,
        total_timesteps=total_timesteps,
        save_steps=save_steps,
        log_steps=log_steps
    )


def io_callback_in_scan(
    run_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    diagnose_fn: Callable,
    history: AbstractHistoryset,
    *,
    post_process_fn: Callable = lambda x: x.get_state(x.old),
    outer_scan_fn: Callable = jax.lax.scan,
    inner_scan_fn: Callable = jax.lax.scan,
) -> Callable[[MODEL], MODEL]:
    return wrap_fn_for_split_scan(
        history=history,
        run_fn=run_fn,
        diagnose_fn=diagnose_fn,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        post_process_fn=post_process_fn,
        outer_scan_fn=outer_scan_fn,
        inner_scan_fn=inner_scan_fn,
    )


def trajectory_in_scan(
    run_fn: Callable,
    diagnose_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    *,
    start_with_input: bool = False,
    post_process_fn: Callable = (
        lambda x: x.trajectory_post_process()
    ),
    outer_scan_fn: Callable = jax.lax.scan,
    inner_scan_fn: Callable = jax.lax.scan,
) -> Callable[[MODEL], Tuple[MODEL, dict[str, Array]]]:
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
    outer_fn = wrap_fn_for_split_scan(
        run_fn=run_fn,
        diagnose_fn=diagnose_fn,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        post_process_fn=post_process_fn,
        outer_scan_fn=outer_scan_fn,
        inner_scan_fn=inner_scan_fn,
    )

    def return_fn(old_model: MODEL) -> Tuple[MODEL, dict[str, Array]]:
        old_block = old_model.block
        final_model, trajectory = outer_fn(old_model)
        if start_with_input:
            trajectory = jax.tree_util.tree_map(
                lambda o, t: jnp.concatenate(
                    [jnp.expand_dims(o, axis=0), t],
                    axis=0
                ),
                old_block.trajectory_post_process(),
                trajectory,
                is_leaf=lambda x: isinstance(x, Array)
            )
        return final_model, trajectory

    return return_fn


def integrate_in_scan(
    run_fn: Callable,
    diagnose_fn: Callable,
    total_steps: int,
    *,
    total_scan_fn: Callable = jax.lax.scan,
) -> Callable[[MODEL], MODEL]:
    return wrap_fn_for_scan(
        run_fn=run_fn,
        diagnose_fn=diagnose_fn,
        total_steps=total_steps,
        total_scan_fn=total_scan_fn,
    )


def output(
    history: AbstractHistoryset,
    block: AbstractBlock,
    save_idx: int,
    first_call: bool = False
) -> None:
    history.output(
        block=block,
        itime_str="old",
        save_idx=save_idx,
        first_call=first_call
    )


@overload
def wrap_fn_for_split_scan(
    run_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    history: AbstractHistoryset,
    diagnose_fn: Optional[Callable] = None,
    post_process_fn: Optional[Callable] = None,
    outer_scan_fn: Callable = jax.lax.scan,
    inner_scan_fn: Callable = jax.lax.scan,
) -> Callable[[MODEL], MODEL]:
    ...

@overload
def wrap_fn_for_split_scan(
    run_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    history: None = None,
    diagnose_fn: Optional[Callable] = None,
    post_process_fn: Optional[Callable] = None,
    outer_scan_fn: Callable = jax.lax.scan,
    inner_scan_fn: Callable = jax.lax.scan,
) -> Callable[[MODEL], tuple[MODEL, Any]]:
    ...

def wrap_fn_for_split_scan(
    run_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    history: Optional[AbstractHistoryset] = None,
    diagnose_fn: Optional[Callable] = None,
    post_process_fn: Optional[Callable] = None,
    outer_scan_fn: Callable = jax.lax.scan,
    inner_scan_fn: Callable = jax.lax.scan,
) -> Callable[[MODEL], Union[MODEL, tuple[MODEL, Any]]]:
    diagnose_fn = diagnose_fn if diagnose_fn is not None else lambda block: block
    post_process_fn = post_process_fn if post_process_fn is not None else lambda block: None

    repeated_step_fn = repeated(
        fn=run_fn,
        steps=inner_steps,
        scan_fn=inner_scan_fn
    )

    if history is None:
        def outer_step_fn_without_history(block: AbstractBlock, _):
            block = repeated_step_fn(block)
            block = diagnose_fn(block)
            return block, post_process_fn(block)

        outer_step_fn = outer_step_fn_without_history
    else:
        def outer_step_fn_with_history(carry_in: tuple[AbstractBlock, int], _):
            block, save_idx = carry_in
            history.output_prepare(save_interval_idx=save_idx)
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
        def return_fn_without_history(model: MODEL) -> tuple[MODEL, Any]:
            block = model.block
            block, trajectory = outer_scan_fn(
                outer_step_fn,
                block,
                xs=None,
                length=outer_steps
            )
            model = eqx.tree_at(
                where=lambda m: m.block,
                pytree=model,
                replace=block                
            )
            return model, trajectory

        return return_fn_without_history
    else:
        def return_fn_with_history(model: MODEL) -> MODEL:
            block = model.block
            history.output_prepare(save_interval_idx=0)
            history.output(
                block=block,
                itime_str="old",
                save_idx=0,
                first_call=True
            )
            carry = (block, 1)
            final_carry, _ = outer_scan_fn(
                outer_step_fn,
                carry,
                xs=None,
                length=outer_steps
            )
            block = final_carry[0]
            return eqx.tree_at(
                where=lambda m: m.block,
                pytree=model,
                replace=block                
            )

        return return_fn_with_history


def wrap_fn_for_scan(
    run_fn: Callable,
    total_steps: int,
    diagnose_fn: Optional[Callable] = None,
    total_scan_fn: Callable = jax.lax.scan,
) -> Callable[[MODEL], MODEL]:
    diagnose_fn = diagnose_fn if diagnose_fn is not None else lambda block: block

    repeated_step_fn = repeated(
        fn=run_fn,
        steps=total_steps,
        scan_fn=total_scan_fn
    )

    def wrap_fn(block):
        block = repeated_step_fn(block)
        block = diagnose_fn(block)
        return block

    def return_fn(model: MODEL) -> MODEL:
        block = model.block
        block = wrap_fn(block)
        return eqx.tree_at(
            where=lambda m: m.block,
            pytree=model,
            replace=block                
        )

    return return_fn


def wrap_fn_for_while_loop(
    run_fn: Callable,
    total_timesteps: int,
    save_steps: int,
    diagnose_fn: Optional[Callable] = None,
    history: Optional[AbstractHistoryset] = None,
    log_steps: Optional[int] = None,
) -> Callable[[MODEL], MODEL]:
    # itimestep starts from 0, and at the beginning of the loop, block is at time level 0 (initial condition).
    diagnose_fn = diagnose_fn if diagnose_fn is not None else lambda block: block

    def while_loop_cond_fn(carry: tuple[BLOCK, int]) -> bool:
        # While itimestep starts from 0, the loop condition should be itimestep <= total_timesteps,
        # because we want to run the loop body for itimestep = total_timesteps,
        # which will has the final time level state and diag variables.
        _, itimestep = carry
        return itimestep <= total_timesteps

    if history is None or log_steps is None:
        def process_on_save_step(block: BLOCK) -> BLOCK:
            block = diagnose_fn(block)
            return block

        def while_loop_body_fn_1(carry: tuple[BLOCK, int]) -> tuple[BLOCK, int]:
            block, itimestep = carry
            block = run_fn(block)
            block = jax.lax.cond(
                itimestep % save_steps == 0,
                process_on_save_step,
                lambda block: block,
                block,
            )
            return block, itimestep + 1
    else:

        def output_on_log_step(itimestep: int):
            history.output_diag(itimestep=itimestep)

        def output_on_save_step(block: BLOCK, save_idx: int):
            block = diagnose_fn(block)
            output(
                history=history,
                block=block,
                save_idx=save_idx
            )

        def while_loop_body_fn_2(carry: tuple[BLOCK, int]) -> tuple[BLOCK, int]:
            block, itimestep = carry
            jax.lax.cond(
                itimestep % log_steps == 0,
                output_on_log_step,
                lambda _: None,
                itimestep
            )
            block = run_fn(block)
            jax.lax.cond(
                itimestep % save_steps == 0,
                output_on_save_step,
                lambda _, __: None,
                block,
                itimestep // save_steps
            )
            return block, itimestep + 1

    if history is None:
        def return_fn(model: MODEL) -> MODEL:
            block = model.block
            carry = (block, 1)
            final_carry = jax.lax.while_loop(
                cond_fun=while_loop_cond_fn,
                body_fun=while_loop_body_fn_1,
                init_val=carry
            )
            block = final_carry[0]
            return eqx.tree_at(
                where=lambda m: m.block,
                pytree=model,
                replace=block
            )
    else:
        def return_fn(model: MODEL) -> MODEL:
            block = model.block 
            history.output_prepare(save_interval_idx=0)
            history.output(
                block=block,
                itime_str="old",
                save_idx=0,
                first_call=True
            )
            carry = (block, 1)
            final_carry = jax.lax.while_loop(
                cond_fun=while_loop_cond_fn,
                body_fun=while_loop_body_fn_2,
                init_val=carry
            )
            block = final_carry[0]
            return eqx.tree_at(
                where=lambda m: m.block,
                pytree=model,
                replace=block
            )

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

    def f_repeated(x_initial: BLOCK) -> BLOCK:

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
    h_final = np.asarray(ds[field_name].d[-1, ...])
    ds.close()
    return h_final


def _get_final_data_with_trajectory(
    field_name: str,
    trajectory_final: AbstractFields
) -> np.ndarray:
    t_final = np.asarray(
        getattr(trajectory_final, field_name).d,
    ).squeeze()
    return t_final


def _get_final_data_with_integrated(
    field_name: str,
    integrated: AbstractBlock
) -> np.ndarray:
    i_final = np.asarray(
        getattr(integrated.state, field_name).d,
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