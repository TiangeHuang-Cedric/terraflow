import os
import sys
import platform
import numpy as np
from typing import Any


from terraflow.util.typing import FloatType, IntType, Array
from terraflow.util.parallel_types import MPI, MPIComm, Proc
from terraflow.util.backend import AbstractBackend


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


class AbstractDevice:
    """Base class for computational devices."""
    proc: Proc
    backend: AbstractBackend

    def __init__(self, proc: Proc, backend: AbstractBackend):
        self.proc = proc
        self.backend = backend

    def __call__(self):
        """Return the backend device architecture."""
        return self.backend()
    
    @property
    def int_type(self) -> IntType:
        return self.backend.int_type
    
    @property
    def float_type(self) -> FloatType:
        return self.backend.float_type
    
    def block_until_ready(self, value: Any) -> Any:
        return self.backend.block_until_ready(value)
    
    def to_numpy(self, value: Array) -> np.ndarray:
        return self.backend.to_numpy(value)
    
    def to_host(self, value: Array) -> np.ndarray:
        return self.to_numpy(value)
    
    def to_backend(self, value: np.ndarray | Array) -> Array:
        return self.backend.asarray(value, device=self())
    
    def is_root(self) -> bool:
        return self.proc.is_root()
    
    def is_distributed(self) -> bool:
        return self.backend.is_distributed()
    
    def barrier(self) -> None:
        if self.proc.comm_model is not None and self.proc.comm_model != MPI.COMM_NULL:
            self.proc.comm_model.Barrier()        
        
    def is_apple_sys(self) -> bool:
        if sys.platform == "darwin":
            arch = platform.machine()
            if arch == "arm64":
                return True
            else:
                return False
        else:
            return False
        
    def is_cpu(self) -> bool:
        return self.backend.is_cpu()

    def is_metal(self) -> bool:
        return self.backend.is_metal()
    
    def is_gpu(self) -> bool:
        return self.backend.is_gpu()
    
    def is_tpu(self) -> bool:
        return self.backend.is_tpu()

class CPU(AbstractDevice):
    """CPU device for computations."""

    def __init__(
        self,
        backend: str, 
        float_type: str = "r8", 
        int_type: str = "i4",
        comm: MPIComm | None = None,
        use_mpi: bool | None = None,
        use_async_io: bool = False,
        proc_io_stride: int = 0,
        use_backend_distributed: bool = False,
    ):  
        proc = Proc(
            comm=comm,
            use_mpi=use_mpi,
            use_async_io=use_async_io,
            proc_io_stride=proc_io_stride
        )
        if backend == "jax":
            import jax
            from terraflow.util.backend.jax_backend import JaxBackend
            JaxBackend.distributed_initialize(
                proc=proc,
                use_backend_distributed=use_backend_distributed
            )
            if self.is_apple_sys():
                jax.config.update("jax_platforms", "cpu")
            ref = jax.local_devices(backend="cpu")[0]
            backend_impl = JaxBackend("cpu", ref, float_type=float_type, int_type=int_type)
        elif backend == "torch":
            from terraflow.util.backend.torch_backend import TorchBackend
            TorchBackend.distributed_initialize(
                proc=proc,
                use_backend_distributed=use_backend_distributed,
                backend="gloo",
            )
            backend_impl = TorchBackend("cpu", float_type=float_type, int_type=int_type)
        else:
            raise ValueError(f"Unsupported backend {backend}!") 
        super().__init__(
            proc,
            backend_impl
        )


class GPU(AbstractDevice):
    """GPU device for computations."""

    def __init__(
        self, 
        backend: str,
        float_type: str = "r8", 
        int_type: str = "i4",
        comm: MPIComm | None = None,
        use_mpi: bool | None = None,
        use_async_io: bool = False,
        proc_io_stride: int = 0,
        use_backend_distributed: bool = False,
    ):
        proc = Proc(
            comm=comm,
            use_mpi=use_mpi,
            use_async_io=use_async_io,
            proc_io_stride=proc_io_stride
        )
        if backend == "jax":
            import jax
            from terraflow.util.backend.jax_backend import JaxBackend
            JaxBackend.distributed_initialize(
                proc=proc,
                use_backend_distributed=use_backend_distributed
            )
            if (
                os.environ.get("CUDA_VISIBLE_DEVICES")
                or os.environ.get("JAX_CUDA_VISIBLE_DEVICES")
                or use_backend_distributed
            ):
                pass
            else:
                local_proc_id = local_process_id(proc)
                if local_proc_id is not None:
                    jax.config.update("jax_cuda_visible_devices", str(local_proc_id))
            if self.is_apple_sys():
                jax.config.update("jax_platforms", "mps,cpu")
                device_string = "mps"
            else:
                device_string = "gpu"
            ref = jax.local_devices(backend=device_string)[0]
            backend_impl = JaxBackend(device_string, ref, float_type=float_type, int_type=int_type)
        elif backend == "torch":
            import torch
            from terraflow.util.backend.torch_backend import TorchBackend
            from terraflow.util.backend.torch_backend import local_cuda_device_id
            TorchBackend.distributed_initialize(
                proc=proc,
                use_backend_distributed=use_backend_distributed,
                backend="nccl",
            )
            if (
                self.is_apple_sys()
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                ref = torch.device("mps")
                device_string = "mps"
            elif torch.cuda.is_available():
                local_device_id = local_cuda_device_id(proc)
                ref = torch.device(f"cuda:{local_device_id}" if local_device_id is not None else "cuda")
                torch.cuda.set_device(ref)
                device_string = "cuda"
            else:
                raise RuntimeError("No PyTorch GPU backend is available.")
            backend_impl = TorchBackend(device_string, ref, float_type=float_type, int_type=int_type)
        else:
            raise ValueError(f"Unsupported backend {backend}!")
        super().__init__(
            proc,
            backend_impl
        )


class TPU(AbstractDevice):
    """TPU device for computations."""

    def __init__(
        self, 
        float_type: str = "r8", 
        int_type: str = "i4",
        comm: MPIComm | None = None,
        use_mpi: bool | None = None,
        use_async_io: bool = False,
        proc_io_stride: int = 0
    ):
        import jax
        from terraflow.util.backend.jax_backend import JaxBackend
        proc = Proc(
            comm=comm,
            use_mpi=use_mpi,
            use_async_io=use_async_io,
            proc_io_stride=proc_io_stride
        )
        JaxBackend.distributed_initialize(
            proc=proc, 
            use_backend_distributed=True
        )
        ref = jax.local_devices(backend="tpu")[0]
        backend_impl = JaxBackend("tpu", ref, float_type=float_type, int_type=int_type)     
        super().__init__(
            proc,
            backend_impl
        )
