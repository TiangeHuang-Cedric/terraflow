from __future__ import annotations
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, Sequence, TYPE_CHECKING
import os
import jax
import jax.experimental
import jax.numpy as jnp
from jax._src import config as jax_config
import numpy as np
import equinox as eqx
import quadax


from terraflow.util.parallel_types import MPI, Proc
from terraflow.util.backend.abstract import AbstractBackend
from terraflow.util.equinox import install_pytree_registrar


if TYPE_CHECKING:
    from jax import Array
    from terraflow.util.typing import Number


_TYPE_REMAPPING = {
    "r4": jnp.float32,
    "r8": jnp.float64,
    "i4": jnp.int32,
    "i8": jnp.int64,
}

_DTYPE_JAX2NUMP = {
    jnp.float32: np.float32,
    jnp.float64: np.float64,
    jnp.int32  : np.int32,
    jnp.int64  : np.int64
}

_REGISTERED_STRUCT_MODULES: set[type] = set()


def register_struct_module_class(cls: type) -> None:
    if cls in _REGISTERED_STRUCT_MODULES:
        return
    try:
        jax.tree_util.register_pytree_node_class(cls)
    except ValueError as exc:
        message = str(exc).lower()
        if "duplicate" not in message and "already registered" not in message:
            raise
    _REGISTERED_STRUCT_MODULES.add(cls)


def enable_struct_pytrees() -> None:
    """Register backend-neutral struct modules as JAX pytrees."""
    install_pytree_registrar(register_struct_module_class)


def has_slurm() -> bool:
    return all(
        var in os.environ
        for var in (
            "SLURM_JOB_ID",
            "SLURM_STEP_NODELIST",
            "SLURM_NTASKS",
            "SLURM_PROCID",
            "SLURM_LOCALID",
        )
    )


def has_openmpi() -> bool:
    return "OMPI_MCA_orte_hnp_uri" in os.environ


def cluster_detection_method() -> str | None:
    method = os.environ.get("TERRAFLOW_CLUSTER_DETECTION_METHOD")
    if method:
        return method
    if MPI.COMM_WORLD.Get_size() > 1:
        return "mpi4py"
    if has_slurm() or has_openmpi():
        return None
    return None


class JaxBackend(AbstractBackend):
    """JAX backend implementation."""

    name = "jax"

    def __init__(self, device_string: str, ref: Any, float_type: str = "r8", int_type: str = "i4"):
        enable_struct_pytrees()
        self.float_type = _TYPE_REMAPPING[float_type]
        self.int_type   = _TYPE_REMAPPING[int_type  ]
        if self.float_type is jnp.float64:
            jax.config.update("jax_enable_x64", True)
        self.device_string = device_string
        self.ref = ref

    def __call__(self) -> Any:
        return self.ref
    
    def is_distributed(self) -> bool:
        return jax.distributed.is_initialized()
   
    @staticmethod 
    def distributed_initialize(proc: Proc, use_backend_distributed: bool):
        if not use_backend_distributed:
            return
        if proc.np_model <= 1 or jax.distributed.is_initialized():
            return
        if not proc.is_model():
            return

        jax.distributed.initialize(
            cluster_detection_method=cluster_detection_method(),
        )
    
    def local_devices(self, backend: str | None = None) -> list[Any]:
        backend = self.device_string if backend is None else backend
        return list(jax.local_devices(backend=backend))

    def is_array(self, value: Any) -> bool:
        return isinstance(value, jax.Array)

    def astype(self, x: Array, dtype: Any) -> Array:
        return x.astype(dtype)
    
    @property
    def int32(self) -> jnp.dtype:
        return jnp.int32
    
    @property
    def int64(self) -> jnp.dtype:
        return jnp.int64
    
    @property
    def float32(self) -> jnp.dtype:
        return jnp.float32
    
    @property
    def float64(self) -> jnp.dtype:
        return jnp.float64

    @property
    def pi(self) -> Any:
        return jnp.pi
    
    def to_numpy_dtype(self, dtype: jnp.dtype) -> np.dtype:
        return _DTYPE_JAX2NUMP[dtype]
    
    def exp(self, x: Array) -> Array:
        return jnp.exp(x)
    
    def log(self, x: Array) -> Array:
        return jnp.log(x)
    
    def sin(self, x: Array) -> Array:
        return jnp.sin(x)
    
    def cos(self, x: Array) -> Array:
        return jnp.cos(x)
    
    def tan(self, x: Array) -> Array:
        return jnp.tan(x)
    
    def arcsin(self, x: Array) -> Array:
        return jnp.arcsin(x)
    
    def arccos(self, x: Array) -> Array:
        return jnp.arccos(x)
    
    def arctan(self, x: Array) -> Array:
        return jnp.arctan(x)

    def sqrt(self, x: Array) -> Array:
        return jnp.sqrt(x)
    
    def rint(self, x: Array) -> Array:
        return jnp.rint(x)
    
    def abs(self, x: Array) -> Array:
        return jnp.abs(x)
    
    def sign(self, x: Array) -> Array:
        return jnp.sign(x)
    
    def argmax(self, a: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return jnp.argmax(a, axis=axis, keepdims=keepdims)
    
    def argmin(self, a: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return jnp.argmin(a, axis=axis, keepdims=keepdims)
    
    def max(self, a: Array, axis: Any | None = None) -> Array:
        return jnp.max(a, axis)
    
    def min(self, a: Array, axis: Any | None = None) -> Array:
        return jnp.min(a, axis)
    
    def maximum(self, x: Array, y: Array) -> Array:
        return jnp.maximum(x, y)
    
    def minimum(self, x: Array, y: Array) -> Array:
        return jnp.minimum(x, y)
    
    def eps(self, dtype: jnp.dtype) -> Any:
        return jnp.finfo(dtype).eps
    
    def any(self, a: Array, axis: Any | None = None) -> Array:
        return jnp.any(a, axis=axis)
    
    def all(self, a: Array, axis: Any | None = None) -> Array:
        return jnp.all(a, axis=axis)

    def count_nonzero(
        self,
        a: Array,
        axis: Any | None = None,
        keepdims: bool = False,
    ) -> Array:
        return jnp.count_nonzero(a, axis=axis, keepdims=keepdims)
    
    def allclose(
        self, 
        a: Array, 
        b: Array, 
        rtol: float = 0.00001, 
        atol: float = 1e-8,
        equal_nan: bool = False
    ) -> Array:
        return jnp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def norm(self, x: Array, axis: Any | None = None, keepdims: bool = False) -> Array:
        return jnp.linalg.norm(x, axis=axis, keepdims=keepdims)

    def svd(self, a: Array, full_matrices: bool = True) -> Any:
        return jnp.linalg.svd(a, full_matrices=full_matrices)

    def pinv(self, a: Array, rcond: Any | None = None) -> Array:
        if rcond is None:
            return jnp.linalg.pinv(a)
        return jnp.linalg.pinv(a, rcond=rcond)

    def diag(self, v: Array) -> Array:
        return jnp.diag(v)

    def det(self, a: Array) -> Array:
        return jnp.linalg.det(a)

    def dot(self, a: Array, b: Array) -> Array:
        return jnp.dot(a, b)

    def clip(self, a: Array, a_min: Any | None = None, a_max: Any | None = None) -> Array:
        return jnp.clip(a, a_min, a_max)

    def mod(self, x: Array, y: Any) -> Array:
        return jnp.mod(x, y)

    def searchsorted(self, a: Array, v: Array, side: str = "left") -> Array:
        return jnp.searchsorted(a, v, side=side)

    def filter_ordered_unique(
        self,
        x: Array,
        size: int | None = None,
        fill_value: int = -1,
        axis: int = 0,
    ) -> Array:
        x_last = jnp.moveaxis(x, axis, -1)
        leading_shape = x_last.shape[:-1]
        length = x_last.shape[-1]
        size_local = size if size is not None else length

        def process_1d_array(arr: Array) -> Array:
            def body(i, carry):
                out, count, seen = carry
                val = arr[i]
                is_valid = val != fill_value
                exists = jnp.any(seen == val)
                is_new = jnp.logical_and(is_valid, jnp.logical_not(exists))
                room = count < size_local
                do_write = jnp.logical_and(is_new, room)
                out = jax.lax.cond(
                    do_write,
                    lambda o: o.at[count].set(val),
                    lambda o: o,
                    out,
                )
                seen = jax.lax.cond(
                    do_write,
                    lambda s: s.at[count].set(val),
                    lambda s: s,
                    seen,
                )
                count = jax.lax.cond(
                    do_write,
                    lambda c: c + 1,
                    lambda c: c,
                    count,
                )
                return out, count, seen

            init_out = jnp.full((size_local,), fill_value)
            init_seen = jnp.full((size_local,), fill_value)
            out, _, _ = jax.lax.fori_loop(
                0, length, body, (init_out, 0, init_seen)
            )
            return out

        x_flat = x_last.reshape((-1, length))
        out_flat = jax.vmap(process_1d_array, in_axes=0, out_axes=0)(x_flat)
        out_last = out_flat.reshape(leading_shape + (size_local,))
        return jnp.moveaxis(out_last, -1, axis)
    
    def arange(
        self, 
        start: Number, 
        stop: Number | None = None, 
        step: Number | None = None, 
        dtype: jnp.dtype | None = None, 
        device: Any | None = None
    ) -> Array:
        return jnp.arange(start, stop=stop, step=step, dtype=dtype, device=device)
    
    def full(
        self, 
        shape: Any, 
        fill_value: Array, 
        dtype: jnp.dtype | None = None, 
        device: Any | None = None
    ):
        return jnp.full(shape, fill_value, dtype=dtype, device=device)
    
    def full_like(
        self, 
        a: Array, 
        fill_value: Array, 
        dtype: jnp.dtype | None = None, 
        device: Any | None = None
    ) -> Array:
        return jnp.full_like(a, fill_value, dtype=dtype, device=device)
    
    def rad2deg(self, x: Array) -> Array:
        return jnp.rad2deg(x)
    
    def deg2rad(self, x: Array) -> Array:
        return jnp.deg2rad(x)
    
    def array(self, object: Any, dtype: jnp.dtype | None = None, *, device: Any | None = None) -> Array:
        return jnp.array(object, dtype, device=device)

    def asarray(self, value: Any, *, dtype: Any | None = None, device: Any | None = None) -> jax.Array:
        kwargs: dict[str, Any] = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if device is not None:
            kwargs["device"] = device
        return jnp.asarray(value, **kwargs)

    def zeros(self, shape: Any, *, dtype: Any | None = None, device: Any | None = None) -> jax.Array:
        kwargs: dict[str, Any] = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if device is not None:
            kwargs["device"] = device
        return jnp.zeros(shape, **kwargs)

    def zeros_like(self, value: Array) -> Array:
        return jnp.zeros_like(value)

    def ones_like(self, value: Array) -> Array:
        return jnp.ones_like(value)

    def flip(self, value: Array, axis: Any | None = None) -> Array:
        return jnp.flip(value, axis=axis)

    def isnan(self, value: Array) -> Array:
        return jnp.isnan(value)

    def isclose(
        self,
        a: Array,
        b: Array,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> Array:
        return jnp.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def logical_and(self, x: Array, y: Array) -> Array:
        return jnp.logical_and(x, y)

    def logical_or(self, x: Array, y: Array) -> Array:
        return jnp.logical_or(x, y)

    def logical_not(self, x: Array) -> Array:
        return jnp.logical_not(x)

    def cross(self, a: Array, b: Array, axis: int = -1) -> Array:
        return jnp.cross(a, b, axis=axis)

    def empty(self) -> Array:
        return jnp.array([])

    def take(self, value: Array, indices: Any, axis: int) -> Array:
        return jnp.take(value, indices=indices, axis=axis)

    def slice_update(self, value: Array, index: Any, replacement: Any) -> Array:
        return value.at[index].set(replacement)

    def halo_update_axis0(self, value: Array, send_idx: Any, recv_idx: Any) -> Array:
        return value.at[recv_idx, ...].set(jnp.take(value, indices=send_idx, axis=0))

    def halo_update_axis1_2d(self, value: Array, send_idx: Any, recv_idx: Any, n_lev: int) -> Array:
        return value.at[:n_lev, recv_idx].set(jnp.take(value[:n_lev, :], indices=send_idx, axis=1))

    def halo_update_axis1_3d(self, value: Array, i0: int, send_idx: Any, recv_idx: Any, n_lev: int) -> Array:
        return value.at[i0, :n_lev, recv_idx].set(jnp.take(value[i0, :n_lev, :], indices=send_idx, axis=1))

    def reshape(self, value: Array, shape: tuple[int, ...]) -> Array:
        return jnp.reshape(value, shape)

    def concatenate(self, values: list[Array], *, axis: int = 0) -> Array:
        return jnp.concatenate(values, axis=axis)

    def expand_dims(self, value: Array, *, axis: int) -> Array:
        return jnp.expand_dims(value, axis=axis)
    
    def tile(self, A: Array, reps: Any) -> Array:
        return jnp.tile(A, reps)
    
    def from_dlpack(self, value: Any) -> Array:
        return jnp.from_dlpack(value)
    
    def stack(self, values: Sequence[Array] | Array, axis: int = 0) -> Array:
        return jnp.stack(values, axis=axis)
    
    def where(self, condition: Array, input: Array, other: Array):
        return jnp.where(condition, input, other)
    
    def sum(
        self,
        a: Array,
        axis: Any | None = None,
        keepdim: bool = False,
        dtype: Any | None = None,
    ) -> Array:
        return jnp.sum(a, axis=axis, keepdims=keepdim, dtype=dtype)
    
    def cumulative_sum(
        self, 
        x: Array, 
        axis: Any | None = None, 
        include_initial: bool = False
    ) -> Array:
        return jnp.cumulative_sum(x, axis=axis, include_initial=include_initial)
    
    def cumsum(self, x: Array, axis: Any | None = None) -> Array:
        return jnp.cumsum(x, axis=axis)

    def vert_linear_interp(
        self,
        x1: Array,
        data1: Array,
        x2: Array,
        allow_extrap: bool = False,
    ) -> Array:
        if allow_extrap:
            left = "extrapolate"
            right = "extrapolate"
        else:
            left = None
            right = None        
        hori_shape = data1.shape[1:]
        nlev_x1 = data1.shape[0]
        ncol = _shape_prod(hori_shape) if hori_shape else 1

        data1_cols = data1.reshape(nlev_x1, ncol)

        if x1.ndim == 1:
            x1_cols = jnp.broadcast_to(x1[:, None], (nlev_x1, ncol))
        else:
            x1_cols = x1.reshape(nlev_x1, ncol)

        # Perform interpolation along the horizontal axis for each point
        if x2.ndim == 1 and x2.shape[0] != ncol:
            # If x2 is 1D but does not match the horizontal shape of data1,
            # evaluate interpolation for each column of data1 separately.
            interp_func = jax.vmap(
                lambda x1, data1: jnp.interp(
                    x2,
                    x1,
                    data1,
                    left=left,
                    right=right
                ),
                in_axes=(1, 1),
                out_axes=1
            )
            res_cols = interp_func(x1_cols, data1_cols)
            res_cols = res_cols.reshape((x2.shape[0],) + hori_shape)
        elif x2.ndim == 1 and x2.shape[0] == ncol:
            # If x2 is 1D and matches the horizontal shape of data1,
            # interpolate each column of data1 to the same x2 coordinates.
            x2_ = x2.reshape(1, -1)
            interp_func = jax.vmap(
                lambda x1, data1, x2: jnp.interp(
                    x2,
                    x1,
                    data1,
                    left=left,
                    right=right
                ),
                in_axes=(1, 1, 1),
                out_axes=1
            )
            res_cols = interp_func(x1_cols, data1_cols, x2_)
            res_cols = res_cols.reshape(hori_shape)
        elif x2.ndim >= 2 and _shape_prod(x2.shape[1:]) == ncol:
            # If x2 is N-Dimensional and matches the horizontal shape of data1,
            # interpolate each column to its corresponding x2 coordinates.
            interp_func = jax.vmap(
                lambda x1, data1, x2: jnp.interp(
                    x2,
                    x1,
                    data1,
                    left=left,
                    right=right
                ),
                in_axes=(1, 1, 1),
                out_axes=1
            )
            res_cols = interp_func(x1_cols, data1_cols, x2)
            res_cols = res_cols.reshape((x2.shape[0],) + hori_shape)
        else:
            raise ValueError(
                "x2 must be either 1D or N-Dimensional array matching data1 horizontal shape."
            )

        return res_cols

    def vert_log_linear_interp(
        self,
        x1: Array,
        data1: Array,
        x2: Array,
        allow_extrap: bool = False,
    ) -> Array:
        return self.vert_linear_interp(
            x1=jnp.log(x1),
            data1=data1,
            x2=jnp.log(x2),
            allow_extrap=allow_extrap
        )

    def quadgk(
        self,
        fun: Callable[..., Array],
        interval: Array,
        args: tuple = (),
        full_output: bool = False,
        epsabs: Any | None = None,
        epsrel: Any | None = None,
        max_ninter: int = 50,
        order: int = 21,
        norm: Any = jnp.inf,
    ) -> Any:
        return quadax.quadgk(
            fun,
            interval,
            args=args,
            full_output=full_output,
            epsabs=epsabs,
            epsrel=epsrel,
            max_ninter=max_ninter,
            order=order,
            norm=norm,
        )
    
    def vmap(
        self,
        fun: Callable,
        in_axes: Any = 0,
        out_axes: Any = 0,
        axis_name: Any | None = None,
        axis_size: int | None = None,
        spmd_axis_name: Any | None = None,
    ) -> Callable:
        return jax.vmap(
            fun,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
        )
    
    def broadcast_to(self, array: Array, shape: Any, *, sharding: Any | None = None) -> Array:
        if sharding is None:
            return jnp.broadcast_to(array, shape)
        return jnp.broadcast_to(array, shape, out_sharding=sharding)
    
    def compile(self, fun: Callable[[Any], Any], model: Any, *, with_jit: bool = True, optimization_level = "O2") -> Callable[[Any], Any]:
        if with_jit:
            if isinstance(optimization_level, str):
                try:
                    optimization_level = jax_config.EffortLevel(optimization_level.upper()).value
                except ValueError as exc:
                    raise ValueError(
                        "optimization_level must be one of 'UNKNOWN', 'O0', 'O1', 'O2', or 'O3'."
                    ) from exc
            return jax.jit(
                fun, 
                compiler_options={"optimization_level": optimization_level}
            ).lower(model).compile()
        return fun

    def block_until_ready(self, value: Any) -> Any:
        return jax.block_until_ready(value)

    def clear_caches(self) -> None:
        jax.clear_caches()

    def profile_scope(self, name: str) -> AbstractContextManager[Any]:
        return jax.named_scope(name)

    def io_callback(
        self,
        callback: Callable[..., Any],
        result_shape_dtypes: Any,
        *args: Any,
        ordered: bool = False,
        **kwargs: Any,
    ) -> Any:
        return jax.experimental.io_callback(
            callback,
            result_shape_dtypes,
            *args,
            ordered=ordered,
            **kwargs,
        )

    def allreduce(
        self,
        value: Any,
        *,
        op: Any,
        comm: Any,
    ) -> Any:
        if comm is None or comm == MPI.COMM_NULL or comm.size == 1:
            return value

        import mpi4jax

        return mpi4jax.allreduce(value, op=op, comm=comm)

    def sendrecv(
        self,
        sendbuf: Any,
        recvbuf: Any,
        *,
        source: int,
        dest: int,
        sendtag: int,
        recvtag: int,
        comm: Any,
    ) -> Any:
        if comm is None or comm == MPI.COMM_NULL or comm.size == 1:
            return sendbuf

        import mpi4jax

        return mpi4jax.sendrecv(
            sendbuf,
            recvbuf,
            source=source,
            dest=dest,
            sendtag=sendtag,
            recvtag=recvtag,
            comm=comm,
        )

    def tree_at(
        self,
        where: Callable[[Any], Any],
        pytree: Any,
        replace: Any,
        is_leaf: Callable[[Any], bool] | None = None,
        at_dims: int | Sequence[int | None] | None = None,
        at_idxes: Any = None,
    ) -> Any:
        if at_dims is None and at_idxes is None:
            return eqx.tree_at(
                where=where,
                pytree=pytree,
                replace=replace,
                is_leaf=is_leaf,
            )

        fields = where(pytree)

        if self.is_array(fields):
            if not isinstance(at_dims, int) or at_idxes is None:
                raise ValueError("Size dismatch between where function and at_dims or at_idxes!")
            indexer: list[Any] = [slice(None)] * fields.ndim
            indexer[at_dims] = at_idxes
            fields_rep = fields.at[tuple(indexer)].set(replace)
            return eqx.tree_at(
                where=where,
                pytree=pytree,
                replace=fields_rep,
                is_leaf=is_leaf,
            )

        if isinstance(at_dims, int) or at_dims is None:
            raise ValueError("Size dismatch between where function and at_dims!")
        if isinstance(at_idxes, int) or at_idxes is None:
            raise ValueError("Size dismatch between where function and at_idxes!")
        if not isinstance(replace, (tuple, list)):
            raise ValueError("replace must be a tuple or list when where returns multiple values.")

        def replace_fn(i: int) -> Any:
            field = fields[i]
            at_dim = at_dims[i]
            at_idx = at_idxes[i]
            if at_dim is None and at_idx is None:
                return replace[i]
            if at_dim is not None and at_idx is not None:
                indexer = [slice(None)] * field.ndim
                indexer[at_dim] = at_idx
                return field.at[tuple(indexer)].set(replace[i])
            raise ValueError("at_dims and at_idxes must both be None or both be provided.")

        fields_rep = tuple(replace_fn(i) for i, _ in enumerate(fields))
        return eqx.tree_at(
            where=where,
            pytree=pytree,
            replace=fields_rep,
            is_leaf=is_leaf,
        )

    def error_if(
        self,
        x: Any,
        pred: Any,
        msg: str,
    ) -> Any:
        return eqx.error_if(
            x=x,
            pred=pred,
            msg=msg,
            on_error="nan" if self.is_metal() else "raise",
        )
        
    def tree_map(self, f: Callable, tree: Any, *rest: Any, is_leaf: Callable[[Any], bool]) -> Any:
        return jax.tree_util.tree_map(
            f,
            tree,
            *rest,
            is_leaf=is_leaf
        )

    def scan(
        self,
        fn: Callable[[Any, Any], tuple[Any, Any]],
        init: Any,
        xs: Any = None,
        length: int | None = None,
    ) -> tuple[Any, Any]:
        return jax.lax.scan(fn, init, xs=xs, length=length)

    def cond(
        self,
        pred: Any,
        true_fun: Callable[..., Any],
        false_fun: Callable[..., Any],
        *operands: Any,
    ) -> Any:
        return jax.lax.cond(pred, true_fun, false_fun, *operands)

    def while_loop(
        self,
        cond_fun: Callable[[Any], Any],
        body_fun: Callable[[Any], Any],
        init_val: Any,
    ) -> Any:
        return jax.lax.while_loop(cond_fun=cond_fun, body_fun=body_fun, init_val=init_val)


def _shape_prod(shape: tuple[int, ...]) -> int:
    out = 1
    for dim in shape:
        out *= dim
    return out
