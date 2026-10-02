from __future__ import annotations
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, fields, is_dataclass
import math
import os
import socket
from typing import Any, NamedTuple, TYPE_CHECKING
import numpy as np
import torch
from torch.utils._pytree import tree_map
from torchquad import GaussLegendre


from terraflow.util.backend.abstract import AbstractBackend, as_bool
from terraflow.util.equinox import replace as module_replace
from terraflow.util.parallel_types import MPI, Proc


if TYPE_CHECKING:
    from torch import Tensor, dtype, Device
    from terraflow.util.typing import Number


_TYPE_REMAPPING = {
    "r4": torch.float32,
    "r8": torch.float64,
    "i4": torch.int32,
    "i8": torch.int64,
}

_DTYPE_TORCH2NUMP = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32  : np.int32,
    torch.int64  : np.int64
}


def repair_torchinductor_precompiled_headers() -> None:
    """Drop stale TorchInductor PCH files that clang refuses to use."""
    try:
        import torch._inductor.codecache as codecache
    except Exception:
        return

    cache_dir = getattr(codecache, "cache_dir", None)
    if not callable(cache_dir):
        return

    try:
        header_dir = os.path.join(str(cache_dir()), "precompiled_headers")
    except Exception:
        return

    if not os.path.isdir(header_dir):
        return

    repaired = False
    try:
        filenames = os.listdir(header_dir)
    except OSError:
        return

    for filename in filenames:
        if not filename.endswith(".h"):
            continue
        header_path = os.path.join(header_dir, filename)
        pch_path = header_path + ".pch"
        if not os.path.isfile(pch_path):
            continue
        try:
            if os.path.getmtime(header_path) > os.path.getmtime(pch_path):
                os.remove(pch_path)
                repaired = True
        except OSError:
            continue

    if repaired:
        precompile_header = getattr(codecache, "_precompile_header", None)
        cache_clear = getattr(precompile_header, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()


def local_cuda_device_id(proc: Proc) -> int | None:
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return None

    local_proc_id = local_process_id(proc)
    if device_count == 1:
        return 0
    if local_proc_id is None:
        return torch.cuda.current_device()
    if local_proc_id >= device_count:
        raise RuntimeError(
            f"Local rank {local_proc_id} cannot be mapped to {device_count} visible CUDA device(s). "
            "Set CUDA_VISIBLE_DEVICES per node or launch one process per visible GPU."
        )
    return local_proc_id


def local_process_id(proc: Proc) -> int | None:
    for name in (
        "LOCAL_RANK",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "MPI_LOCALRANKID",
        "MV2_COMM_WORLD_LOCAL_RANK",
    ):
        value = os.environ.get(name)
        if value is not None:
            return int(value)

    if (
        MPI.Is_initialized()
        and proc.comm_model is not None
        and proc.comm_model != MPI.COMM_NULL
        and proc.comm_model.Get_size() > 1
    ):
        local_comm = proc.comm_model.Split_type(MPI.COMM_TYPE_SHARED)
        local_rank = local_comm.Get_rank()
        local_comm.Free()
        return int(local_rank)

    return None


def select_cuda_device(proc: Proc) -> int:
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise RuntimeError("PyTorch CUDA distributed communication requested, but no CUDA device is visible.")

    local_rank = local_process_id(proc)
    if device_count == 1:
        return 0
    if local_rank is None:
        return torch.cuda.current_device()
    if local_rank >= device_count:
        raise RuntimeError(
            f"Local rank {local_rank} cannot be mapped to {device_count} visible CUDA device(s). "
            "Set CUDA_VISIBLE_DEVICES per node or launch one process per visible GPU."
        )
    return local_rank


def open_ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def torch_reduce_op(op: Any) -> Any:
    if op == MPI.SUM:
        return torch.distributed.ReduceOp.SUM
    if op == MPI.MAX:
        return torch.distributed.ReduceOp.MAX
    if op == MPI.MIN:
        return torch.distributed.ReduceOp.MIN
    if op == MPI.PROD:
        return torch.distributed.ReduceOp.PRODUCT
    raise NotImplementedError(f"Unsupported torch.distributed reduction op: {op!r}")


def use_torch_distributed_cuda(value: Any) -> bool:
    if not isinstance(value, torch.Tensor) or not value.is_cuda:
        return False
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return False
    backend = str(torch.distributed.get_backend()).lower()
    return "nccl" in backend


def use_torch_distributed_gloo_cpu(sendbuf: Any, recvbuf: Any) -> bool:
    if not isinstance(sendbuf, torch.Tensor) or not isinstance(recvbuf, torch.Tensor):
        return False
    if sendbuf.device.type != "cpu" or recvbuf.device.type != "cpu":
        return False
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return False
    backend = str(torch.distributed.get_backend()).lower()
    return "gloo" in backend


def use_torch_distributed_gloo_tensor(value: Any) -> bool:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        return False
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return False
    backend = str(torch.distributed.get_backend()).lower()
    return "gloo" in backend


def is_symbolic_bool(value: Any) -> bool:
    sym_bool_type = getattr(torch, "SymBool", None)
    return sym_bool_type is not None and isinstance(value, sym_bool_type)


def is_symbolic_number(value: Any) -> bool:
    symbolic_types = tuple(
        typ
        for typ in (
            getattr(torch, "SymInt", None),
            getattr(torch, "SymFloat", None),
        )
        if typ is not None
    )
    return bool(symbolic_types) and isinstance(value, symbolic_types)


def unary_math_or_torch(
    value: Any,
    torch_op: Callable[[torch.Tensor], torch.Tensor],
    scalar_op: Callable[[Any], Any],
    *,
    device: torch.device,
) -> Any:
    if isinstance(value, torch.Tensor):
        return torch_op(value)
    if isinstance(value, (int, float, np.number)) or is_symbolic_number(value):
        return scalar_op(value)
    return torch_op(torch.as_tensor(value, device=device))


def logical_binary(
    x: Any,
    y: Any,
    tensor_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    scalar_op: Callable[[Any, Any], Any],
    *,
    device: torch.device,
) -> Any:
    x_is_tensor = isinstance(x, torch.Tensor)
    y_is_tensor = isinstance(y, torch.Tensor)

    if not x_is_tensor and not y_is_tensor:
        return scalar_op(x, y)

    if x_is_tensor and y_is_tensor:
        return tensor_op(x, y)

    if x_is_tensor:
        if is_symbolic_bool(y):
            return x & y
        y = torch.as_tensor(y, dtype=torch.bool, device=x.device)
        return tensor_op(x, y)

    if is_symbolic_bool(x):
        return x & y
    x = torch.as_tensor(x, dtype=torch.bool, device=y.device if y_is_tensor else device)
    return tensor_op(x, y)


def logical_not(value: Any, *, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return torch.logical_not(value)
    if is_symbolic_bool(value):
        return ~value
    if isinstance(value, bool):
        return not value
    return torch.logical_not(torch.as_tensor(value, dtype=torch.bool, device=device))


def isclose_operands(a: Any, b: Any, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    a_tensor = a if isinstance(a, torch.Tensor) else None
    b_tensor = b if isinstance(b, torch.Tensor) else None

    if a_tensor is None and b_tensor is None:
        a_tensor_opt = torch.as_tensor(a, device=device)
        b_tensor_opt = torch.as_tensor(b, device=device)
    elif a_tensor is None and b_tensor is not None:
        a_tensor_opt = torch.as_tensor(a, device=b_tensor.device)
        b_tensor_opt = b_tensor
    elif b_tensor is None and a_tensor is not None:
        a_tensor_opt = a_tensor
        b_tensor_opt = torch.as_tensor(b, device=a_tensor.device)
    elif a_tensor is not None and b_tensor is not None:
        a_tensor_opt = a_tensor
        b_tensor_opt = b_tensor
    else:
        raise ValueError("Unsupported a_tensor and b_tensor!")

    dtype = torch.promote_types(a_tensor_opt.dtype, b_tensor_opt.dtype)
    return a_tensor_opt.to(dtype=dtype), b_tensor_opt.to(dtype=dtype)


class TorchQuadratureInfo(NamedTuple):
    err: Any
    neval: Any
    status: Any
    info: Any


class _LeafWrapper:
    """Unique wrapper used to disambiguate repeated scalar leaves in tree_at."""

    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value


@dataclass(frozen=True)
class TreeAtPlan:
    """Resolved structural paths for a ``tree_at`` selector."""

    paths: tuple[tuple[Any, ...], ...]

    @property
    def num_targets(self) -> int:
        return len(self.paths)


def normalize_index(index: Any, ndim: int) -> tuple[Any, ...]:
    index_parts = index if isinstance(index, tuple) else (index,)
    if index_parts.count(Ellipsis) > 1:
        raise ValueError("Only one ellipsis is supported in tensor.at indexing.")
    if Ellipsis in index_parts:
        ellipsis_pos = index_parts.index(Ellipsis)
        missing = ndim - (len(index_parts) - 1)
        if missing < 0:
            raise ValueError("Too many indices for tensor.")
        index_parts = (
            index_parts[:ellipsis_pos]
            + (slice(None),) * missing
            + index_parts[ellipsis_pos + 1:]
        )
    if len(index_parts) > ndim:
        raise ValueError("Too many indices for tensor.")
    return index_parts + (slice(None),) * (ndim - len(index_parts))


def positive_slice_indices(index: slice, size: int) -> tuple[int, int, int]:
    step = 1 if index.step is None else int(index.step)
    if step <= 0:
        raise ValueError("Negative-step slices are not supported by the functional .at path.")

    if index.start is None:
        start = 0
    else:
        start = int(index.start)
        if start < 0:
            start += size
        start = min(max(start, 0), size)

    if index.stop is None:
        stop = size
    else:
        stop = int(index.stop)
        if stop < 0:
            stop += size
        stop = min(max(stop, 0), size)

    return start, stop, step


_SLICE_UPDATE_KIND_SLICE = 0
_SLICE_UPDATE_KIND_INT = 1


def decode_static_index(
    kinds: list[int],
    starts: list[int],
    stops: list[int],
) -> tuple[Any, ...]:
    indexer: list[Any] = []
    for kind, start, stop in zip(kinds, starts, stops):
        if kind == _SLICE_UPDATE_KIND_SLICE:
            indexer.append(slice(int(start), int(stop)))
        elif kind == _SLICE_UPDATE_KIND_INT:
            indexer.append(int(start))
        else:
            raise ValueError(f"Unsupported slice update index kind: {kind}.")
    return tuple(indexer)


def encode_static_index(
    index: tuple[Any, ...],
    shape: Sequence[int],
) -> tuple[list[int], list[int], list[int]]:
    kinds: list[int] = []
    starts: list[int] = []
    stops: list[int] = []
    for dim, item in enumerate(index):
        if isinstance(item, slice):
            start, stop, step = positive_slice_indices(item, shape[dim])
            if step != 1:
                raise ValueError("Only contiguous slice updates use the custom op path.")
            kinds.append(_SLICE_UPDATE_KIND_SLICE)
            starts.append(start)
            stops.append(stop)
            continue

        if not isinstance(item, torch.Tensor) and (isinstance(item, int) or hasattr(item, "__index__")):
            idx = int(item)
            if idx < 0:
                idx += shape[dim]
            if idx < 0 or idx >= shape[dim]:
                raise IndexError("Index is out of bounds for the custom slice update path.")
            kinds.append(_SLICE_UPDATE_KIND_INT)
            starts.append(idx)
            stops.append(idx + 1)
            continue

        raise TypeError("Only static integer and contiguous slice indices use the custom op path.")
    return kinds, starts, stops


def slice_update_impl(
    value: torch.Tensor,
    replacement: torch.Tensor,
    kinds: list[int],
    starts: list[int],
    stops: list[int],
) -> torch.Tensor:
    result = value.clone()
    result[decode_static_index(kinds, starts, stops)] = replacement
    return result


def as_take_index(index: torch.Tensor, *, value: torch.Tensor, dim: int) -> torch.Tensor:
    index = index.to(device=value.device, dtype=torch.int64).reshape(-1)
    return torch.remainder(index, value.shape[dim])


def as_update_index(index: torch.Tensor, *, value: torch.Tensor, dim: int) -> torch.Tensor:
    index = index.to(device=value.device, dtype=torch.int64).reshape(-1)
    return torch.where(index < 0, index + value.shape[dim], index)


def is_full_slice(index: Any) -> bool:
    return (
        isinstance(index, slice)
        and index.start is None
        and index.stop is None
        and index.step is None
    )


def slice_stop(index: Any, size: int) -> int | None:
    if not isinstance(index, slice):
        return None
    start, stop, step = positive_slice_indices(index, size)
    if start != 0 or step != 1:
        return None
    return stop


def halo_update_axis0_impl(
    value: torch.Tensor,
    send_idx: torch.Tensor,
    recv_idx: torch.Tensor,
) -> torch.Tensor:
    send_idx = as_take_index(send_idx, value=value, dim=0)
    recv_idx = as_update_index(recv_idx, value=value, dim=0)
    result = value.clone()
    replacement = torch.index_select(value, dim=0, index=send_idx)
    result.index_copy_(0, recv_idx, replacement)
    return result


def halo_update_axis1_2d_impl(
    value: torch.Tensor,
    send_idx: torch.Tensor,
    recv_idx: torch.Tensor,
    n_lev: int,
) -> torch.Tensor:
    send_idx = as_take_index(send_idx, value=value, dim=1)
    recv_idx = as_update_index(recv_idx, value=value, dim=1)
    n_lev = int(n_lev)
    result = value.clone()
    replacement = torch.index_select(value[:n_lev, :], dim=1, index=send_idx)
    result[:n_lev, :].index_copy_(1, recv_idx, replacement)
    return result


def halo_update_axis1_3d_impl(
    value: torch.Tensor,
    i0: int,
    send_idx: torch.Tensor,
    recv_idx: torch.Tensor,
    n_lev: int,
) -> torch.Tensor:
    i0 = int(i0)
    n_lev = int(n_lev)
    send_idx = as_take_index(send_idx, value=value, dim=2)
    recv_idx = as_update_index(recv_idx, value=value, dim=2)
    result = value.clone()
    source = value[i0, :n_lev, :]
    replacement = torch.index_select(source, dim=1, index=send_idx)
    result[i0, :n_lev, :].index_copy_(1, recv_idx, replacement)
    return result


@torch.library.custom_op("terraflow::slice_update", mutates_args=())
def slice_update(
    value: torch.Tensor,
    replacement: torch.Tensor,
    kinds: list[int],
    starts: list[int],
    stops: list[int],
) -> torch.Tensor:
    return slice_update_impl(value, replacement, kinds, starts, stops)


@slice_update.register_fake
def slice_update_fake(
    value: torch.Tensor,
    replacement: torch.Tensor,
    kinds: list[int],
    starts: list[int],
    stops: list[int],
) -> torch.Tensor:
    return torch.empty_like(value)


@torch.library.custom_op("terraflow::halo_update_axis0", mutates_args=())
def halo_update_axis0(
    value: torch.Tensor,
    send_idx: torch.Tensor,
    recv_idx: torch.Tensor,
) -> torch.Tensor:
    return halo_update_axis0_impl(value, send_idx, recv_idx)


@halo_update_axis0.register_fake
def halo_update_axis0_fake(
    value: torch.Tensor,
    send_idx: torch.Tensor,
    recv_idx: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(value)


@torch.library.custom_op("terraflow::halo_update_axis1_2d", mutates_args=())
def halo_update_axis1_2d(
    value: torch.Tensor,
    send_idx: torch.Tensor,
    recv_idx: torch.Tensor,
    n_lev: int,
) -> torch.Tensor:
    return halo_update_axis1_2d_impl(value, send_idx, recv_idx, n_lev)


@halo_update_axis1_2d.register_fake
def halo_update_axis1_2d_fake(
    value: torch.Tensor,
    send_idx: torch.Tensor,
    recv_idx: torch.Tensor,
    n_lev: int,
) -> torch.Tensor:
    return torch.empty_like(value)


@torch.library.custom_op("terraflow::halo_update_axis1_3d", mutates_args=())
def halo_update_axis1_3d(
    value: torch.Tensor,
    i0: int,
    send_idx: torch.Tensor,
    recv_idx: torch.Tensor,
    n_lev: int,
) -> torch.Tensor:
    return halo_update_axis1_3d_impl(value, i0, send_idx, recv_idx, n_lev)


@halo_update_axis1_3d.register_fake
def halo_update_axis1_3d_fake(
    value: torch.Tensor,
    i0: int,
    send_idx: torch.Tensor,
    recv_idx: torch.Tensor,
    n_lev: int,
) -> torch.Tensor:
    return torch.empty_like(value)


def index_set(
    tensor: torch.Tensor,
    index: tuple[Any, ...],
    replacement: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    if dim == 0:
        try:
            kinds, starts, stops = encode_static_index(index, tensor.shape)
            return slice_update(tensor, replacement, kinds, starts, stops)
        except (TypeError, ValueError, IndexError):
            pass
    return index_set_fallback(tensor, index, replacement, dim=dim)


@torch.compiler.disable(recursive=True)
def index_set_fallback(
    tensor: torch.Tensor,
    index: tuple[Any, ...],
    replacement: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    if not index:
        return torch.broadcast_to(replacement, tensor.shape)

    head, *tail = index
    tail_tuple = tuple(tail)
    if isinstance(head, slice):
        start, stop, step = positive_slice_indices(head, tensor.shape[dim])
        selected = slice_tensor(tensor, dim=dim, start=start, stop=stop, step=step)
        updated = index_set_fallback(
            selected,
            tail_tuple,
            replacement,
            dim=dim + 1,
        )
        return torch.slice_scatter(tensor, updated, dim=dim, start=start, end=stop, step=step)

    if not isinstance(head, torch.Tensor) and (isinstance(head, int) or hasattr(head, "__index__")):
        selected = torch.select(tensor, dim=dim, index=int(head))
        updated = index_set_fallback(
            selected,
            tail_tuple,
            replacement,
            dim=dim,
        )
        return torch.select_scatter(tensor, updated, dim=dim, index=int(head))

    index_tensor = as_index_tensor(head, tensor=tensor, dim=dim)
    selected = torch.index_select(tensor, dim=dim, index=index_tensor)
    updated = index_set_fallback(
        selected,
        tail_tuple,
        replacement,
        dim=dim + 1,
    )
    return scatter_along_dim(tensor, updated, index_tensor, dim=dim)


def as_index_tensor(index: Any, *, tensor: torch.Tensor, dim: int) -> torch.Tensor:
    if isinstance(index, torch.Tensor):
        index_tensor = index.to(device=tensor.device)
    elif isinstance(index, (list, tuple)):
        index_tensor = torch.as_tensor(index, device=tensor.device)
    else:
        raise TypeError("Unsupported index type for the functional .at path.")

    if index_tensor.dtype == torch.bool:
        if index_tensor.ndim != 1:
            raise ValueError("Only one-dimensional boolean indices are supported by the functional .at path.")
        if index_tensor.shape[0] != tensor.shape[dim]:
            raise ValueError("Boolean index shape does not match indexed dimension.")
        index_tensor = torch.nonzero(index_tensor, as_tuple=False).flatten()
    else:
        index_tensor = index_tensor.to(dtype=torch.int64)
        if index_tensor.ndim == 0:
            index_tensor = index_tensor.reshape(1)
        if index_tensor.ndim != 1:
            raise ValueError("Only scalar or one-dimensional integer indices are supported by the functional .at path.")
        index_tensor = torch.where(index_tensor < 0, index_tensor + tensor.shape[dim], index_tensor)
    return index_tensor


def scatter_along_dim(
    tensor: torch.Tensor,
    updated: torch.Tensor,
    index: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    index_shape = [1] * updated.ndim
    index_shape[dim] = index.shape[0]
    scatter_index = index.reshape(index_shape).expand_as(updated)
    return torch.scatter(tensor, dim=dim, index=scatter_index, src=updated)


def slice_tensor(
    tensor: torch.Tensor,
    *,
    dim: int,
    start: int,
    stop: int,
    step: int,
) -> torch.Tensor:
    indexer: list[Any] = [slice(None)] * tensor.ndim
    indexer[dim] = slice(start, stop, step)
    return tensor[tuple(indexer)]


def shape_prod(shape: Sequence[int]) -> int:
    out = 1
    for dim in shape:
        out *= dim
    return out


def tree_at_cache_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return ("value", value)
    if isinstance(value, tuple):
        return ("tuple", tuple(tree_at_cache_value(item) for item in value))
    return ("id", id(value))


def callable_cache_value(fn: Callable[[Any], Any] | None) -> Any:
    if fn is None:
        return None
    code = getattr(fn, "__code__", None)
    if code is None:
        return ("id", id(fn))
    closure = getattr(fn, "__closure__", None) or ()
    closure_key = tuple(
        tree_at_cache_value(cell.cell_contents)
        for cell in closure
    )
    globals_dict = getattr(fn, "__globals__", {})
    global_key = tuple(
        (name, tree_at_cache_value(globals_dict[name]))
        for name in code.co_names
        if name in globals_dict
    )
    return (code, closure_key, global_key)


def is_numpy_leaf(value: Any) -> bool:
    return isinstance(value, (np.ndarray, np.generic))


class TorchBackend(AbstractBackend):
    """PyTorch eager backend implementation."""

    name = "torch"

    def __init__(
        self,
        device_string: str,
        ref: torch.device | None = None,
        float_type: str = "r8",
        int_type: str = "i4",
        functional_updates: bool = False,
    ):
        self.float_type = _TYPE_REMAPPING[float_type]
        self.int_type   = _TYPE_REMAPPING[int_type  ]
        if self.float_type is torch.float64:
            torch.set_default_dtype(torch.float64)
        self.device_string = device_string
        self.ref = ref if ref is not None else torch.device(device_string)
        self.functional_updates = functional_updates
        self._tree_at_plan_cache: dict[Any, TreeAtPlan] = {}
        self._tree_at_cache_hits   = 0
        self._tree_at_cache_misses = 0

    def __call__(self) -> torch.device:
        return self.ref
    
    def is_distributed(self) -> bool:
        return torch.distributed.is_initialized()
    
    @staticmethod
    def distributed_initialize(
        proc: Proc,
        use_backend_distributed: bool,
        backend: str | None = None,
    ):
        if not use_backend_distributed:
            return
        if proc.np_model <= 1 or torch.distributed.is_initialized():
            return
        if not proc.is_model():
            return
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is not available in this PyTorch build.")

        backend = (backend or ("nccl" if torch.cuda.is_available() else "gloo")).lower()
        if backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("NCCL distributed communication requested, but no CUDA device is visible.")
        if backend == "nccl" and not torch.distributed.is_nccl_available():
            raise RuntimeError("CUDA distributed communication requires a PyTorch build with NCCL support.")
        is_gloo_available = getattr(torch.distributed, "is_gloo_available", lambda: False)
        if backend == "gloo" and not is_gloo_available():
            raise RuntimeError("CPU distributed communication requires a PyTorch build with Gloo support.")
        if backend not in ("gloo", "nccl"):
            raise ValueError(f"Unsupported torch.distributed backend: {backend!r}.")

        if backend == "nccl":
            local_device = select_cuda_device(proc)
            torch.cuda.set_device(local_device)
            os.environ.setdefault("LOCAL_RANK", str(local_device))

        os.environ.setdefault("RANK", str(proc.id_model))
        os.environ.setdefault("WORLD_SIZE", str(proc.np_model))

        master_addr = os.environ.get("MASTER_ADDR")
        if master_addr is None:
            host = (proc.hostname or socket.getfqdn()) if proc.id_model == 0 else None
            master_addr = proc.comm_model.bcast(host, root=0)
            os.environ["MASTER_ADDR"] = str(master_addr)

        master_port = os.environ.get("MASTER_PORT")
        if master_port is None:
            port = open_ephemeral_port() if proc.id_model == 0 else None
            master_port = proc.comm_model.bcast(port, root=0)
            os.environ["MASTER_PORT"] = str(master_port)

        torch.distributed.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=int(proc.id_model),
            world_size=int(proc.np_model),
        )

    def local_devices(self, backend: str | None = None) -> list[torch.device]:
        device = self.device_string if backend is None else backend
        if device == "cpu":
            return [torch.device("cpu")]
        if device in ("gpu", "cuda"):
            return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        if device.startswith("cuda:"):
            device_index = int(device.split(":", maxsplit=1)[1])
            if 0 <= device_index < torch.cuda.device_count():
                return [torch.device(device)]
            return []
        if device == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return [torch.device("mps")]
            return []
        return [torch.device(device)]

    def is_array(self, value: Any) -> bool:
        return isinstance(value, torch.Tensor)

    def astype(self, x: Tensor, dtype: Any) -> Tensor:
        return self.asarray(x).to(dtype=dtype)
    
    @property
    def int32(self) -> dtype:
        return torch.int32
    
    @property
    def int64(self) -> dtype:
        return torch.int64
    
    @property
    def float32(self) -> dtype:
        return torch.float32
    
    @property
    def float64(self) -> dtype:
        return torch.float64
    
    @property
    def pi(self) -> Any:
        return torch.pi
    
    def to_numpy_dtype(self, dtype: dtype) -> np.dtype:
        return _DTYPE_TORCH2NUMP[dtype]
    
    def exp(self, x: Any) -> Any:
        return unary_math_or_torch(x, torch.exp, math.exp, device=self.ref)
    
    def log(self, x: Any) -> Any:
        return unary_math_or_torch(x, torch.log, math.log, device=self.ref)
    
    def sin(self, x: Any) -> Any:
        return unary_math_or_torch(x, torch.sin, math.sin, device=self.ref)
    
    def cos(self, x: Any) -> Any:
        return unary_math_or_torch(x, torch.cos, math.cos, device=self.ref)
    
    def tan(self, x: Any) -> Any:
        return unary_math_or_torch(x, torch.tan, math.tan, device=self.ref)
    
    def arcsin(self, x: Any) -> Any:
        return unary_math_or_torch(x, torch.arcsin, math.asin, device=self.ref)
    
    def arccos(self, x: Any) -> Any:
        return unary_math_or_torch(x, torch.arccos, math.acos, device=self.ref)
    
    def arctan(self, x: Any) -> Any:
        return unary_math_or_torch(x, torch.arctan, math.atan, device=self.ref)

    def sqrt(self, x: Any) -> Any:
        return unary_math_or_torch(x, torch.sqrt, math.sqrt, device=self.ref)
    
    def rint(self, x: Tensor) -> Tensor:
        return torch.round(self.asarray(x))
    
    def max(self, a: Tensor, axis: int | None = None) -> Tensor:
        if axis is None:
            return torch.amax(a)
        return torch.amax(a, dim=axis)
    
    def min(self, a: Tensor, axis: Any | None = None) -> Tensor:
        if axis is None:
            return torch.amin(a)
        return torch.amin(a, dim=axis)
    
    def maximum(self, x: Tensor, y: Tensor) -> Tensor:
        if not self.is_array(x) and not self.is_array(y):
            x = self.asarray(x)
            y = self.asarray(y, dtype=x.dtype, device=x.device)
        elif not self.is_array(x):
            x = self.asarray(x, dtype=y.dtype, device=y.device)
        elif not self.is_array(y):
            y = self.asarray(y, dtype=x.dtype, device=x.device)
        return torch.maximum(x, y)
    
    def minimum(self, x: Tensor, y: Tensor) -> Tensor:
        if not self.is_array(x) and not self.is_array(y):
            x = self.asarray(x)
            y = self.asarray(y, dtype=x.dtype, device=x.device)
        elif not self.is_array(x):
            x = self.asarray(x, dtype=y.dtype, device=y.device)
        elif not self.is_array(y):
            y = self.asarray(y, dtype=x.dtype, device=x.device)
        return torch.minimum(x, y)
    
    def eps(self, dtype: dtype) -> Any:
        return torch.finfo(dtype).eps
    
    def abs(self, x: Tensor) -> Tensor:
        return torch.abs(self.asarray(x))
    
    def sign(self, x: Tensor) -> Tensor:
        return torch.sign(self.asarray(x))
    
    def argmax(self, a: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
        return torch.argmax(a, dim=axis, keepdim=keepdims)
    
    def argmin(self, a: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
        return torch.argmin(a, dim=axis, keepdim=keepdims)
    
    def any(self, a: Tensor, axis: Any | None = None) -> Tensor:
        return torch.any(a, dim=axis)
    
    def all(self, a: Tensor, axis: Any | None = None) -> Tensor:
        return torch.all(a, dim=axis)

    def count_nonzero(
        self,
        a: Tensor,
        axis: Any | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        result = torch.count_nonzero(a, dim=axis)
        if not keepdims:
            return result

        if axis is None:
            return result.reshape((1,) * a.ndim)

        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        axes = tuple(ax if ax >= 0 else ax + a.ndim for ax in axes)
        for ax in sorted(axes):
            result = result.unsqueeze(ax)
        return result
    
    def allclose(
        self, 
        a: Tensor, 
        b: Tensor, 
        rtol: float = 0.00001, 
        atol: float = 1e-8,
        equal_nan: bool = False
    ) -> bool:
        if not self.is_array(a) and not self.is_array(b):
            a = self.asarray(a)
            b = self.asarray(b, dtype=a.dtype, device=a.device)
        elif not self.is_array(a):
            a = self.asarray(a, dtype=b.dtype, device=b.device)
        elif not self.is_array(b):
            b = self.asarray(b, dtype=a.dtype, device=a.device)
        return torch.allclose(self.asarray(a), self.asarray(b), rtol=rtol, atol=atol, equal_nan=equal_nan)

    def norm(self, x: Tensor, axis: Any | None = None, keepdims: bool = False) -> Tensor:
        return torch.linalg.norm(x, dim=axis, keepdim=keepdims)

    def svd(self, a: Tensor, full_matrices: bool = True) -> Any:
        return torch.linalg.svd(a, full_matrices=full_matrices)

    def pinv(self, a: Tensor, rcond: Any | None = None) -> Tensor:
        if rcond is None:
            return torch.linalg.pinv(a)
        return torch.linalg.pinv(a, rtol=rcond)

    def diag(self, v: Tensor) -> Tensor:
        return torch.diag(v)

    def det(self, a: Tensor) -> Tensor:
        return torch.linalg.det(a)

    def dot(self, a: Tensor, b: Tensor) -> Tensor:
        if a.ndim == 1 and b.ndim == 1:
            return torch.dot(a, b)
        return torch.matmul(a, b)

    def clip(self, a: Tensor, a_min: Any | None = None, a_max: Any | None = None) -> Tensor:
        return torch.clip(self.asarray(a), min=a_min, max=a_max)

    def mod(self, x: Tensor, y: Any) -> Tensor:
        return torch.remainder(self.asarray(x), y)

    def searchsorted(self, a: Tensor, v: Tensor, side: str = "left") -> Tensor:
        return torch.searchsorted(self.asarray(a), self.asarray(v), right=side == "right")

    def filter_ordered_unique(
        self,
        x: Tensor,
        size: int | None = None,
        fill_value: int = -1,
        axis: int = 0,
    ) -> Tensor:
        x = self.asarray(x)
        x_last = torch.movedim(x, axis, -1)
        leading_shape = x_last.shape[:-1]
        length = x_last.shape[-1]
        size_local = size if size is not None else length

        x_flat = x_last.reshape((-1, length))
        out = torch.full(
            (x_flat.shape[0], size_local),
            fill_value,
            dtype=x.dtype,
            device=x.device,
        )
        seen = torch.full_like(out, fill_value)
        count = torch.zeros(x_flat.shape[0], dtype=torch.int64, device=x.device)

        for i in range(length):
            val = x_flat[:, i]
            is_valid = val != fill_value
            exists = torch.any(seen == val[:, None], dim=-1)
            room = count < size_local
            do_write = is_valid & ~exists & room
            rows = torch.nonzero(do_write, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            cols = count[rows]
            out[rows, cols] = val[rows]
            seen[rows, cols] = val[rows]
            count[rows] += 1

        out_last = out.reshape(leading_shape + (size_local,))
        return torch.movedim(out_last, -1, axis)
    
    def arange(
        self, 
        start: Number, 
        stop: Number | None = None, 
        step: Number | None = None, 
        dtype: dtype | None = None, 
        device: Device | None = None
    ) -> Tensor:
        if stop is None:
            stop_opt  = start
            start = 0
        else:
            stop_opt = stop
        if step is None:
            step = 1
        return torch.arange(start=start, end=stop_opt, step=step, dtype=dtype, device=device or self.ref)
    
    def full(
        self, 
        shape: Any, 
        fill_value: Number, 
        dtype: dtype | None = None, 
        device: Device | None = None
    ) -> Tensor:
        return torch.full(shape, fill_value, dtype=dtype, device=device or self.ref)
    
    def full_like(
        self, 
        a: Tensor, 
        fill_value: Number, 
        dtype: dtype | None = None, 
        device: Device | None = None
    ) -> Tensor:
        return torch.full_like(a, fill_value, dtype=dtype, device=device)

    def array(self, object: Any, dtype: Any | None = None, *, device: Any | None = None) -> Any:
        return torch.asarray(object, dtype=dtype, device=device or self.ref)

    def asarray(self, value: Any, *, dtype: Any | None = None, device: Any | None = None) -> torch.Tensor:
        if self.is_array(value):
            result: Tensor = value
            if dtype is not None or device is not None:
                result = result.to(dtype=dtype or result.dtype, device=device or result.device)
            return result
        return torch.as_tensor(value, dtype=dtype, device=device or self.ref)
    
    def rad2deg(self, x: Tensor) -> Tensor:
        return torch.rad2deg(x)
    
    def deg2rad(self, x: Tensor) -> Tensor:
        return torch.deg2rad(x)

    def zeros(self, shape: Any, *, dtype: Any | None = None, device: Any | None = None) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=device or self.ref)

    def zeros_like(self, value: Tensor) -> Tensor:
        return torch.zeros_like(value)

    def ones_like(self, value: Tensor) -> Tensor:
        return torch.ones_like(value)

    def flip(self, value: Tensor, axis: Any | None = None) -> Tensor:
        if axis is None:
            axis_opt = tuple(range(value.ndim))
        elif isinstance(axis, int):
            axis_opt = (axis,)
        else:
            axis_opt = axis
        return torch.flip(value, dims=axis_opt)

    def isnan(self, value: Tensor) -> Tensor:
        return torch.isnan(value)

    def isclose(
        self,
        a: Tensor,
        b: Tensor,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> Tensor:
        a_tensor, b_tensor = isclose_operands(a, b, device=self.ref)
        return torch.isclose(a_tensor, b_tensor, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def logical_and(self, x: Any, y: Any) -> Any:
        return logical_binary(
            x,
            y,
            torch.logical_and,
            lambda a, b: a & b,
            device=self.ref,
        )

    def logical_or(self, x: Any, y: Any) -> Any:
        return logical_binary(
            x,
            y,
            torch.logical_or,
            lambda a, b: a | b,
            device=self.ref,
        )

    def logical_not(self, x: Any) -> Any:
        return logical_not(x, device=self.ref)

    def cross(self, a: Tensor, b: Tensor, axis: int = -1) -> Tensor:
        return torch.cross(self.asarray(a), self.asarray(b), dim=axis)

    def empty(self) -> Tensor:
        return torch.tensor([], device=self.ref)

    def take(self, value: Tensor, indices: Any, axis: int) -> Tensor:
        index = self.asarray(indices, dtype=torch.int64, device=value.device)
        index = index % value.shape[axis]
        return torch.index_select(value, dim=axis, index=index.reshape(-1)).reshape(
            value.shape[:axis] + index.shape + value.shape[axis + 1:]
        )

    def slice_update(self, value: Tensor, index: Any, replacement: Any) -> Tensor:
        replacement_tensor = torch.as_tensor(
            replacement,
            dtype=value.dtype,
            device=value.device,
        )
        index = normalize_index(index, value.ndim)

        if self.functional_updates:
            return index_set(value, index, replacement_tensor, dim=0)

        if value.ndim >= 1 and isinstance(index[0], torch.Tensor) and all(
            is_full_slice(item) for item in index[1:]
        ):
            recv_idx = as_update_index(index[0], value=value, dim=0)
            value.index_copy_(0, recv_idx, replacement_tensor)
            return value

        if value.ndim == 2 and isinstance(index[1], torch.Tensor):
            n_lev = slice_stop(index[0], value.shape[0])
            if n_lev is not None:
                recv_idx = as_update_index(index[1], value=value, dim=1)
                value[:n_lev, :].index_copy_(1, recv_idx, replacement_tensor)
                return value

        if value.ndim == 3 and isinstance(index[0], int) and isinstance(index[2], torch.Tensor):
            n_lev = slice_stop(index[1], value.shape[1])
            if n_lev is not None:
                recv_idx = as_update_index(index[2], value=value, dim=2)
                value[int(index[0]), :n_lev, :].index_copy_(1, recv_idx, replacement_tensor)
                return value

        value[index] = replacement_tensor
        return value

    def halo_update_axis0(self, value: Tensor, send_idx: Any, recv_idx: Any) -> Tensor:
        send_idx_tensor = self.asarray(send_idx, dtype=torch.int64, device=value.device)
        recv_idx_tensor = self.asarray(recv_idx, dtype=torch.int64, device=value.device)
        return halo_update_axis0(value, send_idx_tensor, recv_idx_tensor)

    def halo_update_axis1_2d(self, value: Tensor, send_idx: Any, recv_idx: Any, n_lev: int) -> Tensor:
        send_idx_tensor = self.asarray(send_idx, dtype=torch.int64, device=value.device)
        recv_idx_tensor = self.asarray(recv_idx, dtype=torch.int64, device=value.device)
        return halo_update_axis1_2d(value, send_idx_tensor, recv_idx_tensor, int(n_lev))

    def halo_update_axis1_3d(self, value: Tensor, i0: int, send_idx: Any, recv_idx: Any, n_lev: int) -> Tensor:
        send_idx_tensor = self.asarray(send_idx, dtype=torch.int64, device=value.device)
        recv_idx_tensor = self.asarray(recv_idx, dtype=torch.int64, device=value.device)
        return halo_update_axis1_3d(value, int(i0), send_idx_tensor, recv_idx_tensor, int(n_lev))

    def reshape(self, value: Tensor, shape: tuple[int, ...]) -> Tensor:
        return torch.reshape(value, shape)

    def concatenate(self, values: list[Tensor], *, axis: int = 0) -> Tensor:
        return torch.cat(values, dim=axis)

    def expand_dims(self, value: Tensor, *, axis: int) -> Tensor:
        return torch.unsqueeze(value, dim=axis)
    
    def tile(self, A: Tensor, reps: Any) -> Tensor:
        return torch.tile(A, reps)

    def to_numpy(self, value: Tensor) -> np.ndarray:
        if self.is_array(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)
    
    def from_dlpack(self, value: Any) -> Tensor:
        return torch.from_dlpack(value)
    
    def stack(self, values: list[Tensor] | tuple[Tensor], axis: int = 0) -> Tensor:
        return torch.stack(values, dim=axis)
    
    def where(self, condition: Tensor, input: Tensor, other: Tensor) -> Tensor:
        if not self.is_array(condition):
            condition = self.asarray(condition, dtype=torch.bool)
        if not self.is_array(input) and not self.is_array(other):
            input = self.asarray(input)
            other = self.asarray(other, dtype=input.dtype, device=input.device)
        elif not self.is_array(input):
            input = self.asarray(input, dtype=other.dtype, device=other.device)
        elif not self.is_array(other):
            other = self.asarray(other, dtype=input.dtype, device=input.device)
        return torch.where(condition, input, other)
    
    def sum(
        self,
        a: Tensor,
        axis: Any | None = None,
        keepdim: bool = False,
        dtype: Any | None = None,
    ) -> Tensor:
        return torch.sum(a, dim=axis, keepdim=keepdim, dtype=dtype)
    
    def cumulative_sum(
        self, 
        x: Tensor, 
        axis: Any | None = None, 
        include_initial: bool = False
    ) -> Tensor:
        axis_opt = 0 if axis is None else axis
        if include_initial:
            res = torch.cumsum(x, dim=axis_opt)
            zero_shape = list(x.shape)
            zero_shape[axis_opt] = 1
            zero_tensor = torch.zeros(zero_shape, dtype=x.dtype, device=x.device)
            return torch.cat([zero_tensor, res], dim=axis_opt)
        else:
            return torch.cumsum(x, dim=axis_opt)
        
    def cumsum(self, x: Any, axis: Any | None = None) -> Any:
        axis_opt = 0 if axis is None else axis 
        return torch.cumsum(x, dim=axis_opt)

    def vert_linear_interp(
        self,
        x1: Tensor,
        data1: Tensor,
        x2: Tensor,
        allow_extrap: bool = False,
    ) -> Tensor:
        x1 = self.asarray(x1)
        data1 = self.asarray(data1, device=x1.device)
        x2 = self.asarray(x2, dtype=x1.dtype, device=x1.device)

        if data1.ndim < 1:
            raise ValueError("data1 must have at least 1 dimension")

        nlev_x1 = data1.shape[0]
        if nlev_x1 < 2:
            raise ValueError("nlev_x1 must be >= 2")

        hori_shape = data1.shape[1:]
        ncol = shape_prod(hori_shape) if hori_shape else 1
        data1_cols = data1.reshape(nlev_x1, ncol)

        if x1.ndim == 1:
            if x1.shape[0] != nlev_x1:
                raise ValueError("x1 length must match data1.shape[0]")
            x1_cols = torch.broadcast_to(x1[:, None], (nlev_x1, ncol))
        else:
            if x1.shape[0] != nlev_x1 or x1.shape[1:] != hori_shape:
                raise ValueError("x1 shape must be (nlev_x1,) or match data1 shape")
            x1_cols = x1.reshape(nlev_x1, ncol)

        if x2.ndim == 0:
            x2_cols = torch.broadcast_to(x2.reshape(1, 1), (1, ncol))
            out_shape = hori_shape
        elif x2.shape == hori_shape:
            x2_cols = x2.reshape(1, ncol)
            out_shape = hori_shape
        elif x2.ndim == 1:
            x2_cols = torch.broadcast_to(x2[:, None], (x2.shape[0], ncol))
            out_shape = (x2.shape[0],) + hori_shape
        elif x2.shape[1:] == hori_shape:
            x2_cols = x2.reshape(x2.shape[0], ncol)
            out_shape = x2.shape
        else:
            raise ValueError("x2 shape is incompatible with data1 horizontal shape")

        i2 = torch.searchsorted(
            x1_cols.T.contiguous(),
            x2_cols.T.contiguous(),
            right=True,
        ).T
        in_range = (i2 > 0) & (i2 < nlev_x1)
        below = i2 == 0

        if allow_extrap:
            i1_idx = torch.where(in_range, i2 - 1, torch.where(below, 0, nlev_x1 - 2))
            i2_idx = torch.where(in_range, i2, torch.where(below, 1, nlev_x1 - 1))
        else:
            i1_idx = torch.where(in_range, i2 - 1, torch.where(below, 0, nlev_x1 - 1))
            i2_idx = torch.where(in_range, i2, torch.where(below, 0, nlev_x1 - 1))

        x_lo = torch.gather(x1_cols, dim=0, index=i1_idx)
        x_hi = torch.gather(x1_cols, dim=0, index=i2_idx)
        data_lo = torch.gather(data1_cols, dim=0, index=i1_idx)
        data_hi = torch.gather(data1_cols, dim=0, index=i2_idx)

        same = i1_idx == i2_idx
        denom = torch.where(same, torch.ones_like(x_hi), x_hi - x_lo)
        half = torch.full_like(x2_cols, 0.5)
        wgt1 = torch.where(same, half, (x_hi - x2_cols) / denom)
        wgt2 = torch.where(same, half, (x2_cols - x_lo) / denom)

        return (data_lo * wgt1 + data_hi * wgt2).reshape(out_shape)

    def vert_log_linear_interp(
        self,
        x1: Tensor,
        data1: Tensor,
        x2: Tensor,
        allow_extrap: bool = False,
    ) -> Tensor:
        x1 = self.asarray(x1)
        x2 = self.asarray(x2, dtype=x1.dtype, device=x1.device)
        return self.vert_linear_interp(
            x1=torch.log(x1),
            data1=data1,
            x2=torch.log(x2),
            allow_extrap=allow_extrap,
        )

    def quadgk(
        self,
        fun: Callable[..., Tensor],
        interval: Tensor,
        args: tuple = (),
        full_output: bool = False,
        epsabs: Any | None = None,
        epsrel: Any | None = None,
        max_ninter: int = 50,
        order: int = 21,
        norm: Any = float("inf"),
    ) -> Any:
        def points_and_weights(
            order: int,
            *,
            dtype: torch.dtype,
            device: torch.device,
        ) -> tuple[Tensor, Tensor]:
            roots_np, weights_np = GaussLegendre()._cached_points_and_weights(order)
            roots = torch.as_tensor(roots_np, dtype=dtype, device=device)
            weights = torch.as_tensor(weights_np, dtype=dtype, device=device)
            return roots, weights

        def fixed_gauss_legendre_quad(
            fun: Callable[..., Tensor],
            a: Tensor,
            b: Tensor,
            args: tuple,
            order: int,
            ninter: int,
        ) -> Tensor:
            roots, weights = points_and_weights(order, dtype=a.dtype, device=a.device)
            edges = torch.linspace(0.0, 1.0, ninter + 1, dtype=a.dtype, device=a.device)
            delta = b - a
            left = a[..., None] + delta[..., None] * edges[:-1]
            right = a[..., None] + delta[..., None] * edges[1:]
            center = 0.5 * (left + right)
            half_width = 0.5 * (right - left)
            points = center[..., None] + half_width[..., None] * roots

            flat_points = points.reshape(-1)
            values = self.asarray(fun(flat_points, *args), device=a.device)
            point_shape = points.shape
            values = values.reshape(point_shape + values.shape[1:])

            extra_dims = values.ndim - len(point_shape)
            weights_shape = (1,) * (len(point_shape) - 1) + (order,) + (1,) * extra_dims
            weighted = values * weights.reshape(weights_shape)
            interval_values = torch.sum(weighted, dim=len(point_shape) - 1)

            half_width_shape = half_width.shape + (1,) * extra_dims
            interval_values = interval_values * half_width.reshape(half_width_shape)
            return torch.sum(interval_values, dim=half_width.ndim - 1)

        interval_tensor = self.asarray(interval)
        if interval_tensor.shape[-1] != 2:
            raise ValueError("quadgk expects interval with the last dimension of size 2.")
        if not torch.is_floating_point(interval_tensor):
            interval_tensor = interval_tensor.to(dtype=torch.get_default_dtype())

        if order < 1:
            raise ValueError("order must be a positive integer.")
        if max_ninter < 1:
            raise ValueError("max_ninter must be a positive integer.")

        a = interval_tensor[..., 0]
        b = interval_tensor[..., 1]
        value = fixed_gauss_legendre_quad(fun, a, b, args, order, max_ninter)

        err_order = max(1, (order + 1) // 2)
        if err_order == order:
            err = torch.zeros_like(value)
        else:
            err_value = fixed_gauss_legendre_quad(fun, a, b, args, err_order, max_ninter)
            err = torch.abs(value - err_value)

        neval = torch.as_tensor(order * max_ninter, dtype=torch.int32, device=interval_tensor.device)
        status = torch.zeros_like(neval)
        if epsabs is not None or epsrel is not None:
            abs_tol = self.asarray(0.0 if epsabs is None else epsabs, dtype=value.dtype, device=value.device)
            rel_tol = self.asarray(0.0 if epsrel is None else epsrel, dtype=value.dtype, device=value.device)
            if callable(norm):
                scale = self.asarray(norm(value), dtype=value.dtype, device=value.device)
            else:
                scale = torch.abs(value)
            tol = torch.maximum(abs_tol, rel_tol * scale)
            status = torch.where(torch.all(err <= tol), status, torch.ones_like(status))

        extra_info = None
        if full_output:
            extra_info = {
                "backend": "torchquad.GaussLegendre",
                "ninter": torch.as_tensor(max_ninter, dtype=torch.int32, device=interval_tensor.device),
                "order": torch.as_tensor(order, dtype=torch.int32, device=interval_tensor.device),
            }
        return value, TorchQuadratureInfo(err=err, neval=neval, status=status, info=extra_info)
    
    def vmap(
        self,
        fun: Callable,
        in_axes: Any = 0,
        out_axes: Any = 0,
        axis_name: Any | None = None,
        axis_size: int | None = None,
        spmd_axis_name: Any | None = None,
    ) -> Callable:
        if axis_name is not None or axis_size is not None or spmd_axis_name is not None:
            raise NotImplementedError("axis_name, axis_size, and spmd_axis_name are not supported by torch.vmap.")
        return torch.vmap(fun, in_dims=in_axes, out_dims=out_axes)
    
    def broadcast_to(self, array: Tensor, shape: Any, *, sharding: Any | None = None) -> Tensor:
        if sharding is not None:
            raise NotImplementedError("sharding is not supported by torch.broadcast_to.")
        return torch.broadcast_to(array, shape)
    
    def compile(self, fun: Callable[[Any], Any], model: Any, *, with_jit: bool = True, **kwargs) -> Callable[[Any], Any]:
        if not with_jit:
            return fun
        if not hasattr(torch, "compile"):
            return fun
        repair_torchinductor_precompiled_headers()

        try:
            import torch._dynamo.config as dynamo_config
            import torch.fx.experimental._config as fx_config
            if hasattr(dynamo_config, "recompile_limit"):
                dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 64)
            if hasattr(fx_config, "use_duck_shape"):
                fx_config.use_duck_shape = False
        except Exception:
            pass

        return torch.compile(
            fun,
            backend="inductor",
            mode="default",
            fullgraph=False,
            dynamic=False,
        )

    def block_until_ready(self, value: Any) -> Any:
        if self.device_string == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.ref)
        elif self.device_string == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
        return value

    def clear_caches(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (
            hasattr(torch, "mps")
            and hasattr(torch.mps, "empty_cache")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            torch.mps.empty_cache()

    def profile_scope(self, name: str) -> AbstractContextManager[Any]:
        if torch.compiler.is_compiling():
            return nullcontext()
        return torch.profiler.record_function(name)

    def io_callback(
        self,
        callback: Callable[..., Any],
        result_shape_dtypes: Any,
        *args: Any,
        ordered: bool = False,
        **kwargs: Any,
    ) -> Any:
        for arg in args:
            self.block_until_ready(arg)
        result = callback(*args, **kwargs)
        if result_shape_dtypes is None or result is None:
            return result
        if isinstance(result, (tuple, list)):
            return type(result)(self.asarray(item) for item in result)
        return self.asarray(result)

    @torch.compiler.disable(recursive=True)
    def allreduce(
        self,
        value: Any,
        *,
        op: Any,
        comm: Any,
    ) -> Tensor:
        if comm is None or comm == MPI.COMM_NULL or comm.size == 1:
            return value

        if use_torch_distributed_cuda(value):
            reduced = torch.empty(value.shape, dtype=value.dtype, device=value.device)
            reduced.copy_(value)
            torch.distributed.all_reduce(reduced, op=torch_reduce_op(op))
            return reduced

        if use_torch_distributed_gloo_tensor(value):
            reduced = value.contiguous().clone()
            torch.distributed.all_reduce(reduced, op=torch_reduce_op(op))
            return reduced

        value_array = self.to_numpy(value)
        value_shape = value_array.shape
        send_array = np.ascontiguousarray(value_array.reshape(-1))
        recv_array = np.empty_like(send_array)
        comm.Allreduce(send_array, recv_array, op=op)
        recv_array = recv_array.reshape(value_shape)
        if self.is_array(value):
            return self.asarray(recv_array, dtype=value.dtype, device=value.device)
        return self.asarray(recv_array)

    @torch.compiler.disable(recursive=True)
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
    ) -> Tensor:
        if comm is None or comm == MPI.COMM_NULL or comm.size == 1:
            return sendbuf

        if use_torch_distributed_gloo_cpu(sendbuf, recvbuf):
            send_tensor = sendbuf.contiguous()
            recv_tensor = torch.empty(
                recvbuf.shape,
                dtype=recvbuf.dtype,
                device=recvbuf.device,
            )
            recv_work = torch.distributed.irecv(
                recv_tensor,
                src=int(source),
                tag=int(recvtag),
            )
            send_work = torch.distributed.isend(
                send_tensor,
                dst=int(dest),
                tag=int(sendtag),
            )
            if send_work is not None:
                send_work.wait()
            if recv_work is not None:
                recv_work.wait()
            return recv_tensor

        if use_torch_distributed_cuda(sendbuf) and use_torch_distributed_cuda(recvbuf):
            send_tensor = sendbuf.contiguous()
            recv_tensor = torch.empty(
                recvbuf.shape,
                dtype=recvbuf.dtype,
                device=recvbuf.device,
            )
            work_items = torch.distributed.batch_isend_irecv(
                [
                    torch.distributed.P2POp(
                        torch.distributed.irecv,
                        recv_tensor,
                        int(source),
                        tag=0,
                    ),
                    torch.distributed.P2POp(
                        torch.distributed.isend,
                        send_tensor,
                        int(dest),
                        tag=0,
                    ),
                ]
            )
            for work in work_items:
                work.wait()
            return recv_tensor

        send_array = np.ascontiguousarray(self.to_numpy(sendbuf))
        recv_template = self.to_numpy(recvbuf)
        recv_array = np.empty_like(recv_template)
        comm.Sendrecv(
            send_array,
            dest=int(dest),
            sendtag=sendtag,
            recvbuf=recv_array,
            source=int(source),
            recvtag=recvtag,
        )
        if self.is_array(recvbuf):
            return self.asarray(recv_array, dtype=recvbuf.dtype, device=recvbuf.device)
        return self.asarray(recv_array)

    @torch.compiler.disable(recursive=True)
    def tree_at(
        self,
        where: Callable[[Any], Any],
        pytree: Any,
        replace: Any,
        is_leaf: Callable[[Any], bool] | None = None,
        at_dims: int | Sequence[int | None] | None = None,
        at_idxes: Any = None,
    ) -> Any:
        plan = self.prepare_tree_at(where, pytree, is_leaf=is_leaf)
        return self.apply_tree_at(
            plan,
            pytree,
            replace,
            at_dims=at_dims,
            at_idxes=at_idxes,
        )

    @torch.compiler.disable(recursive=True)
    def prepare_tree_at(
        self,
        where: Callable[[Any], Any],
        pytree: Any,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> TreeAtPlan:
        cache_key = None if torch.compiler.is_compiling() else _tree_at_cache_key(pytree, where, is_leaf)
        if cache_key is not None:
            plan = self._tree_at_plan_cache.get(cache_key)
            if plan is not None:
                self._tree_at_cache_hits += 1
                return plan

        self._tree_at_cache_misses += 1
        shadow_tree = self._wrap_tree_leaves(pytree, is_leaf=is_leaf)
        shadow_targets = where(shadow_tree)
        shadow_target_values = self._normalize_tree_at_targets(shadow_targets)
        target_ids = tuple(id(target) for target in shadow_target_values)
        paths_by_id = self._collect_paths_by_id(shadow_tree, set(target_ids), is_leaf=is_leaf)

        path_list: list[tuple[Any, ...]] = []
        for target_id in target_ids:
            found_paths = paths_by_id.get(target_id, ())
            if len(found_paths) != 1:
                raise ValueError(
                    "TorchBackend.tree_at only supports unambiguous dataclass/Module field selection."
                )
            path_list.append(found_paths[0])
        plan = TreeAtPlan(tuple(path_list))
        self._validate_unique_paths(plan.paths)

        if cache_key is not None:
            self._tree_at_plan_cache[cache_key] = plan
        return plan

    @torch.compiler.disable(recursive=True)
    def apply_tree_at(
        self,
        plan: TreeAtPlan,
        pytree: Any,
        replace: Any,
        *,
        at_dims: int | Sequence[int | None] | None = None,
        at_idxes: Any = None,
    ) -> Any:
        final_replacements = self._tree_at_replacements(
            plan,
            pytree,
            replace,
            at_dims=at_dims,
            at_idxes=at_idxes,
        )
        return self._replace_paths(pytree, tuple(zip(plan.paths, final_replacements)))

    @torch.compiler.disable(recursive=True)
    def tree_at_many(
        self,
        pytree: Any,
        updates: dict[Callable[[Any], Any], Any] | Sequence[tuple[Callable[[Any], Any], Any]],
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> Any:
        items = updates.items() if isinstance(updates, dict) else updates
        path_updates: list[tuple[tuple[Any, ...], Any]] = []
        for where, replace in items:
            plan = self.prepare_tree_at(where, pytree, is_leaf=is_leaf)
            replacements = self._normalize_tree_at_replacements(replace, plan.num_targets)
            path_updates.extend(zip(plan.paths, replacements))
        self._validate_unique_paths(tuple(path for path, _ in path_updates))
        return self._replace_paths(pytree, tuple(path_updates))

    @torch.compiler.disable(recursive=True)
    def tree_replace(self, pytree: Any, **updates: Any) -> Any:
        if not is_dataclass(pytree) or isinstance(pytree, type):
            raise TypeError("tree_replace expects a dataclass or Module instance.")
        return self._replace_dataclass(pytree, updates)

    def tree_at_cache_info(self) -> dict[str, int]:
        return {
            "hits": self._tree_at_cache_hits,
            "misses": self._tree_at_cache_misses,
            "size": len(self._tree_at_plan_cache),
        }

    def clear_tree_at_cache(self) -> None:
        self._tree_at_plan_cache.clear()
        self._tree_at_cache_hits = 0
        self._tree_at_cache_misses = 0

    def cond(
        self,
        pred: Any,
        true_fun: Callable[..., Any],
        false_fun: Callable[..., Any],
        *operands: Any,
    ) -> Any:
        if isinstance(pred, torch.Tensor) and pred.ndim != 0:
            raise ValueError("TorchBackend.cond expects a scalar predicate.")
        if pred:
            return true_fun(*operands)
        return false_fun(*operands)

    def error_if(
        self,
        x: Any,
        pred: Any,
        msg: str
    ) -> Any:
        if torch.compiler.is_compiling():
            return x

        if not as_bool(pred):
            return x
        raise ValueError(msg)
    
    def tree_map(self, f: Callable, tree: Any, *rest: Any, is_leaf: Callable[[Any], bool]) -> Any:
        return tree_map(f, tree, *rest, is_leaf=is_leaf)

    def _index_replace(self, value: Any, at_dim: int | None, at_idx: Any, replacement: Any) -> Any:
        if at_dim is None and at_idx is None:
            return replacement
        if at_dim is None or at_idx is None:
            raise ValueError("at_dims and at_idxes must both be None or both be provided.")
        indexer: list[Any] = [slice(None)] * value.ndim
        indexer[at_dim] = at_idx
        return self.slice_update(value, tuple(indexer), replacement)

    def _normalize_tree_at_targets(self, targets: Any) -> tuple[Any, ...]:
        if isinstance(targets, (tuple, list)):
            return tuple(targets)
        return (targets,)

    def _normalize_tree_at_replacements(self, replace: Any, count: int) -> tuple[Any, ...]:
        if count == 1:
            return (replace,)
        if not isinstance(replace, (tuple, list)):
            raise ValueError("replace must be a tuple or list when where returns multiple values.")
        replacements = tuple(replace)
        if len(replacements) != count:
            raise ValueError("where and replace must have the same number of values.")
        return replacements

    def _tree_at_replacements(
        self,
        plan: TreeAtPlan,
        pytree: Any,
        replace: Any,
        *,
        at_dims: int | Sequence[int | None] | None,
        at_idxes: Any,
    ) -> tuple[Any, ...]:
        replace_values = self._normalize_tree_at_replacements(replace, plan.num_targets)
        if at_dims is None and at_idxes is None:
            return replace_values

        target_values = tuple(self._get_path(pytree, path) for path in plan.paths)
        if plan.num_targets == 1:
            if not isinstance(at_dims, int) or at_idxes is None:
                raise ValueError("Size dismatch between where function and at_dims or at_idxes!")
            return (
                self._index_replace(target_values[0], at_dims, at_idxes, replace),
            )

        if isinstance(at_dims, int) or at_dims is None:
            raise ValueError("Size dismatch between where function and at_dims!")
        if isinstance(at_idxes, int) or at_idxes is None:
            raise ValueError("Size dismatch between where function and at_idxes!")
        if len(at_dims) != plan.num_targets or len(at_idxes) != plan.num_targets:
            raise ValueError("where, replace, at_dims, and at_idxes must have the same length.")

        return tuple(
            self._index_replace(target, at_dim, at_idx, replacement)
            if at_dim is not None or at_idx is not None
            else replacement
            for target, replacement, at_dim, at_idx in zip(
                target_values,
                replace_values,
                at_dims,
                at_idxes,
            )
        )

    def _wrap_tree_leaves(self, value: Any, *, is_leaf: Callable[[Any], bool] | None = None) -> Any:
        if is_leaf is not None and is_leaf(value):
            return _LeafWrapper(value)
        if self.is_array(value) or is_numpy_leaf(value):
            return _LeafWrapper(value)
        if is_dataclass(value) and not isinstance(value, type):
            return self._replace_dataclass(
                value,
                {
                    item.name: self._wrap_tree_leaves(getattr(value, item.name), is_leaf=is_leaf)
                    for item in fields(value)
                },
            )
        if isinstance(value, tuple):
            return tuple(self._wrap_tree_leaves(item, is_leaf=is_leaf) for item in value)
        if isinstance(value, list):
            return [self._wrap_tree_leaves(item, is_leaf=is_leaf) for item in value]
        if isinstance(value, dict):
            return {
                key: self._wrap_tree_leaves(item, is_leaf=is_leaf)
                for key, item in value.items()
            }
        return _LeafWrapper(value)

    def _collect_paths_by_id(
        self,
        value: Any,
        target_ids: set[int],
        *,
        is_leaf: Callable[[Any], bool] | None = None,
        path: tuple[Any, ...] = (),
        out: dict[int, list[tuple[Any, ...]]] | None = None,
    ) -> dict[int, list[tuple[Any, ...]]]:
        if out is None:
            out = {target_id: [] for target_id in target_ids}
        if isinstance(value, _LeafWrapper):
            value_id = id(value)
            if value_id in target_ids:
                out[value_id].append(path)
            return out
        if is_leaf is not None and is_leaf(value):
            value_id = id(value)
            if value_id in target_ids:
                out[value_id].append(path)
            return out
        if self.is_array(value) or is_numpy_leaf(value):
            value_id = id(value)
            if value_id in target_ids:
                out[value_id].append(path)
            return out
        if is_dataclass(value) and not isinstance(value, type):
            value_id = id(value)
            if value_id in target_ids:
                out[value_id].append(path)
            for item in fields(value):
                self._collect_paths_by_id(
                    getattr(value, item.name),
                    target_ids,
                    is_leaf=is_leaf,
                    path=path + (item.name,),
                    out=out,
                )
            return out
        if isinstance(value, tuple):
            for i, child in enumerate(value):
                self._collect_paths_by_id(
                    child,
                    target_ids,
                    is_leaf=is_leaf,
                    path=path + (i,),
                    out=out,
                )
            return out
        if isinstance(value, list):
            for i, child in enumerate(value):
                self._collect_paths_by_id(
                    child,
                    target_ids,
                    is_leaf=is_leaf,
                    path=path + (i,),
                    out=out,
                )
            return out
        if isinstance(value, dict):
            for key, child in value.items():
                self._collect_paths_by_id(
                    child,
                    target_ids,
                    is_leaf=is_leaf,
                    path=path + (key,),
                    out=out,
                )
            return out
        return out

    def _find_paths_by_id(self, value: Any, target_id: int, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        if id(value) == target_id:
            return [path]
        if isinstance(value, _LeafWrapper):
            return []
        if self.is_array(value) or is_numpy_leaf(value):
            return []
        if is_dataclass(value) and not isinstance(value, type):
            paths: list[tuple[Any, ...]] = []
            for item in fields(value):
                paths.extend(self._find_paths_by_id(getattr(value, item.name), target_id, path + (item.name,)))
            return paths
        if isinstance(value, tuple):
            paths = []
            for i, child in enumerate(value):
                paths.extend(self._find_paths_by_id(child, target_id, path + (i,)))
            return paths
        if isinstance(value, list):
            paths = []
            for i, child in enumerate(value):
                paths.extend(self._find_paths_by_id(child, target_id, path + (i,)))
            return paths
        if isinstance(value, dict):
            paths = []
            for key, child in value.items():
                paths.extend(self._find_paths_by_id(child, target_id, path + (key,)))
            return paths
        return []

    def _get_path(self, value: Any, path: tuple[Any, ...]) -> Any:
        result = value
        for key in path:
            if is_dataclass(result) and not isinstance(result, type):
                result = getattr(result, key)
            elif isinstance(result, (tuple, list, dict)):
                result = result[key]
            else:
                raise ValueError(
                    "TorchBackend.tree_at path can only traverse dataclass, tuple, list, or dict containers."
                )
        return result

    def _validate_unique_paths(self, paths: tuple[tuple[Any, ...], ...]) -> None:
        if len(paths) != len(set(paths)):
            raise ValueError("TorchBackend.tree_at does not support duplicate target paths.")

    def _replace_path(self, value: Any, path: tuple[Any, ...], replacement: Any) -> Any:
        if not path:
            return replacement

        key = path[0]
        remaining = path[1:]
        if is_dataclass(value) and not isinstance(value, type):
            child = getattr(value, key)
            return self._replace_dataclass(value, {key: self._replace_path(child, remaining, replacement)})
        if isinstance(value, tuple):
            items = list(value)
            items[key] = self._replace_path(items[key], remaining, replacement)
            return tuple(items)
        if isinstance(value, list):
            items = list(value)
            items[key] = self._replace_path(items[key], remaining, replacement)
            return items
        if isinstance(value, dict):
            items = dict(value)
            items[key] = self._replace_path(items[key], remaining, replacement)
            return items
        raise ValueError("TorchBackend.tree_at can only replace inside dataclass, tuple, list, or dict containers.")

    def _replace_paths(self, value: Any, updates: tuple[tuple[tuple[Any, ...], Any], ...]) -> Any:
        if not updates:
            return value

        root_updates = [replacement for path, replacement in updates if not path]
        if root_updates:
            if len(updates) != 1:
                raise ValueError("Cannot replace the tree root and nested fields in the same update.")
            return root_updates[0]

        grouped: dict[Any, list[tuple[tuple[Any, ...], Any]]] = {}
        for path, replacement in updates:
            key = path[0]
            grouped.setdefault(key, []).append((path[1:], replacement))

        if is_dataclass(value) and not isinstance(value, type):
            replacements = {
                key: self._replace_paths(getattr(value, key), tuple(child_updates))
                for key, child_updates in grouped.items()
            }
            return self._replace_dataclass(value, replacements)
        if isinstance(value, tuple):
            items = list(value)
            for key, child_updates in grouped.items():
                items[key] = self._replace_paths(items[key], tuple(child_updates))
            return tuple(items)
        if isinstance(value, list):
            items = list(value)
            for key, child_updates in grouped.items():
                items[key] = self._replace_paths(items[key], tuple(child_updates))
            return items
        if isinstance(value, dict):
            items = dict(value)
            for key, child_updates in grouped.items():
                items[key] = self._replace_paths(items[key], tuple(child_updates))
            return items
        raise ValueError("TorchBackend.tree_at can only replace inside dataclass, tuple, list, or dict containers.")

    @torch.compiler.disable(recursive=True)
    def _replace_dataclass(self, value: Any, replacements: dict[str, Any]) -> Any:
        if not torch.compiler.is_compiling():
            return module_replace(value, **replacements)

        new_value = object.__new__(type(value))
        for item in fields(value):
            replacement = replacements.get(item.name, getattr(value, item.name))
            object.__setattr__(new_value, item.name, replacement)
        return new_value


def _tree_at_cache_key(
    pytree: Any,
    where: Callable[[Any], Any],
    is_leaf: Callable[[Any], bool] | None,
) -> Any | None:
    where_key = callable_cache_value(where)
    if where_key is None:
        return None
    return (type(pytree), where_key, callable_cache_value(is_leaf))
