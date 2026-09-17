from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any
import numpy as np


if TYPE_CHECKING:
    from terraflow.util.parallel_types import Proc


def as_bool(value: Any) -> bool:
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


class AbstractBackend(ABC):
    """Array backend interface used by TerraFlow devices."""

    name: str
    device_string: str
    float_type: Any
    int_type: Any
    
    @abstractmethod
    def __init__(self, float_type: str = "r8", int_type: str = "i4"):
        raise NotImplementedError

    @abstractmethod
    def __call__(self) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def is_distributed(self) -> bool:
        raise NotImplementedError
   
    @staticmethod 
    @abstractmethod
    def distributed_initialize(proc: Proc, use_backend_distributed: bool):
        raise NotImplementedError
    
    @abstractmethod
    def local_devices(self, backend: str | None = None) -> list[Any]:
        raise NotImplementedError
        
    def is_cpu(self) -> bool:
        return self.device_string == "cpu"

    def is_metal(self) -> bool:
        return self.device_string == "mps"
    
    def is_gpu(self) -> bool:
        return self.device_string == "gpu"
    
    def is_tpu(self) -> bool:
        return self.device_string == "tpu"

    @abstractmethod
    def is_array(self, value: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def astype(self, x: Any, dtype: Any) -> Any:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def int32(self) -> Any:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def int64(self) -> Any:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def float32(self) -> Any:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def float64(self) -> Any:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def pi(self) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def to_numpy_dtype(self, dtype: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def exp(self, x: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def log(self, x: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def sin(self, x: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def cos(self, x: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def tan(self, x: Any) -> Any:
        raise NotImplementedError
    
    def asin(self, x: Any) -> Any:
        return self.arcsin(x)
    
    def acos(self, x: Any) -> Any:
        return self.arccos(x)
    
    def atan(self, x: Any) -> Any:
        return self.arctan(x)
    
    @abstractmethod
    def arcsin(self, x: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def arccos(self, x: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def arctan(self, x: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def sqrt(self, x: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def rint(self, x: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def abs(self, x: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def sign(self, x: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def argmax(self, a: Any, axis: int | None = None, keepdims: bool = False) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def argmin(self, a: Any, axis: int | None = None, keepdims: bool = False) -> Any:
        raise NotImplementedError    
    
    @abstractmethod
    def max(self, a: Any, axis: Any | None = None) -> Any:
        raise NotImplementedError
    
    @abstractmethod 
    def min(self, a: Any, axis: Any | None = None) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def maximum(self, x: Any, y: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def minimum(self, x: Any, y: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def eps(self, dtype: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def any(self, a: Any, axis: Any | None = None) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def all(self, a: Any, axis: Any | None = None) -> Any:
        raise NotImplementedError 

    @abstractmethod
    def count_nonzero(
        self,
        a: Any,
        axis: Any | None = None,
        keepdims: bool = False,
    ) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def allclose(
        self, 
        a: Any, 
        b: Any, 
        rtol: float = 0.00001, 
        atol: float = 1e-8,
        equal_nan: bool = False
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def norm(self, x: Any, axis: Any | None = None, keepdims: bool = False) -> Any:
        raise NotImplementedError

    @abstractmethod
    def svd(self, a: Any, full_matrices: bool = True) -> Any:
        raise NotImplementedError

    @abstractmethod
    def pinv(self, a: Any, rcond: Any | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def diag(self, v: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def det(self, a: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def dot(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def clip(self, a: Any, a_min: Any | None = None, a_max: Any | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def mod(self, x: Any, y: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def searchsorted(self, a: Any, v: Any, side: str = "left") -> Any:
        raise NotImplementedError

    @abstractmethod
    def filter_ordered_unique(
        self,
        x: Any,
        size: int | None = None,
        fill_value: int = -1,
        axis: int = 0,
    ) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def arange(
        self, 
        start: Any, 
        stop: Any | None = None, 
        step: Any | None = None, 
        dtype: Any | None = None, 
        device: Any | None = None
    ):
        raise NotImplementedError
    
    @abstractmethod
    def full(
        self, 
        shape: Any, 
        fill_value: Any, 
        dtype: Any | None = None, 
        device: Any | None = None
    ) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def full_like(
        self, 
        a: Any, 
        fill_value: Any, 
        dtype: Any | None = None, 
        device: Any | None = None
    ) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def array(self, object: Any, dtype: Any | None = None, *, device: Any | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def asarray(self, value: Any, *, dtype: Any | None = None, device: Any | None = None) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def rad2deg(self, x: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def deg2rad(self, x: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def zeros(self, shape: Any, *, dtype: Any | None = None, device: Any | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def zeros_like(self, value: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def ones_like(self, value: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def flip(self, value: Any, axis: Any | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def isnan(self, value: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def isclose(
        self,
        a: Any,
        b: Any,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def logical_and(self, x: Any, y: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def logical_or(self, x: Any, y: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def logical_not(self, x: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def cross(self, a: Any, b: Any, axis: int = -1) -> Any:
        raise NotImplementedError

    @abstractmethod
    def empty(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def take(self, value: Any, indices: Any, axis: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def slice_update(self, value: Any, index: Any, replacement: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def halo_update_axis0(self, value: Any, send_idx: Any, recv_idx: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def halo_update_axis1_2d(self, value: Any, send_idx: Any, recv_idx: Any, n_lev: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def halo_update_axis1_3d(self, value: Any, i0: int, send_idx: Any, recv_idx: Any, n_lev: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def reshape(self, value: Any, shape: tuple[int, ...]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def concatenate(self, values: list[Any], *, axis: int = 0) -> Any:
        raise NotImplementedError

    @abstractmethod
    def expand_dims(self, value: Any, *, axis: int) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def tile(self, A: Any, reps: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def from_dlpack(self, value: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def stack(self, values: list[Any], axis: int = 0) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def where(self, condition: Any, input: Any, other: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def sum(
        self,
        a: Any,
        axis: Any | None = None,
        keepdim: bool = False,
        dtype: Any | None = None,
    ) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def cumulative_sum(self, x: Any, axis: Any | None = None, include_initial: bool = False) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def cumsum(self, x: Any, axis: Any | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def vert_linear_interp(
        self,
        x1: Any,
        data1: Any,
        x2: Any,
        allow_extrap: bool = False,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def vert_log_linear_interp(
        self,
        x1: Any,
        data1: Any,
        x2: Any,
        allow_extrap: bool = False,
    ) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def quadgk(
        self,
        fun: Callable[..., Any],
        interval: Any,
        args: tuple = (),
        full_output: bool = False,
        epsabs: Any | None = None,
        epsrel: Any | None = None,
        max_ninter: int = 50,
        order: int = 21,
        norm: Any = np.inf,
    ) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def vmap(
        self, 
        fun: Callable,
        in_axes: Any = 0,
        out_axes: Any = 0,
        axis_name: Any | None = None,
        axis_size: int | None = None,
        spmd_axis_name: Any | None = None,        
    ) -> Callable:
        raise NotImplementedError
    
    @abstractmethod
    def broadcast_to(self, array: Any, shape: Any, *, sharding: Any | None = None) -> Any:
        raise NotImplementedError
    
    def to_numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value)
    
    def compile(self, fun: Callable[[Any], Any], model: Any, *, with_jit: bool = True) -> Callable[[Any], Any]:
        return fun

    def block_until_ready(self, value: Any) -> Any:
        return value

    @abstractmethod
    def clear_caches(self) -> None:
        raise NotImplementedError()

    def profile_scope(self, name: str) -> AbstractContextManager[Any]:
        return nullcontext()

    @abstractmethod
    def io_callback(
        self,
        callback: Callable[..., Any],
        result_shape_dtypes: Any,
        *args: Any,
        ordered: bool = False,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def allreduce(
        self,
        value: Any,
        *,
        op: Any,
        comm: Any,
    ) -> Any:
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
    def tree_at(
        self,
        where: Callable[[Any], Any],
        pytree: Any,
        replace: Any,
        is_leaf: Callable[[Any], bool] | None = None,
        at_dims: int | Sequence[int | None] | None = None,
        at_idxes: Any = None,
    ) -> Any:
        raise NotImplementedError()

    def prepare_tree_at(
        self,
        where: Callable[[Any], Any],
        pytree: Any,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> Any:
        raise NotImplementedError()

    def apply_tree_at(
        self,
        plan: Any,
        pytree: Any,
        replace: Any,
        *,
        at_dims: int | Sequence[int | None] | None = None,
        at_idxes: Any = None,
    ) -> Any:
        raise NotImplementedError()

    def tree_at_many(
        self,
        pytree: Any,
        updates: dict[Callable[[Any], Any], Any] | Sequence[tuple[Callable[[Any], Any], Any]],
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def error_if(
        self,
        x: Any,
        pred: Any,
        msg: str
    ) -> Any:
        raise NotImplementedError()
    
    def tree_map(self, f: Callable, tree: Any, *rest: Any, is_leaf: Callable[[Any], bool]) -> Any:
        raise NotImplementedError

    def scan(
        self,
        fn: Callable[[Any, Any], tuple[Any, Any]],
        init: Any,
        xs: Any = None,
        length: int | None = None,
    ) -> tuple[Any, Any]:
        if xs is None:
            if length is None:
                raise ValueError("length must be provided when xs is None")
            iterator = [None] * length
        elif isinstance(xs, (tuple, list)):
            scan_length = length if length is not None else xs[0].shape[0]
            iterator = (
                type(xs)(item[i] for item in xs)
                for i in range(scan_length)
            )
        else:
            iterator = xs

        carry = init
        outputs = []
        has_output = False
        for x in iterator:
            carry, y = fn(carry, x)
            if y is not None:
                outputs.append(y)
                has_output = True

        if not has_output:
            return carry, None
        return carry, self.stack(outputs)

    def cond(
        self,
        pred: Any,
        true_fun: Callable[..., Any],
        false_fun: Callable[..., Any],
        *operands: Any,
    ) -> Any:
        if as_bool(pred):
            return true_fun(*operands)
        return false_fun(*operands)

    def while_loop(
        self,
        cond_fun: Callable[[Any], Any],
        body_fun: Callable[[Any], Any],
        init_val: Any,
    ) -> Any:
        value = init_val
        while as_bool(cond_fun(value)):
            value = body_fun(value)
        return value
