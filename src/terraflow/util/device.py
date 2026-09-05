import jax
import jax.numpy as jnp
import os
import sys
import platform
from mpi4py import MPI


from terraflow.util.parallel_types import Proc


TYPE_REMAPPING = {
    "r4": jnp.float32,
    "r8": jnp.float64,
    "i4": jnp.int32,
    "i8": jnp.int64,
}


class Backend:
    """Base class for computational backends."""
    device_string: str

    def __init__(self):
        raise NotImplementedError("Children class must implement __init__")

    def __call__(self):
        raise NotImplementedError("Children class must implement __call__")


class JaxBackend(Backend):
    """JAX backend for computations."""

    def __init__(self, device_string: str, ref):
        self.device_string = device_string
        self.ref = ref

    def __call__(self):
        return self.ref


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


def distributed_initialize(
    proc: Proc,
    use_jax_distributed: bool
) -> None:
    if not use_jax_distributed:
        return
    if proc.np_model <= 1 or jax.distributed.is_initialized():
        return
    if not proc.is_model():
        return

    jax.distributed.initialize(
        cluster_detection_method=cluster_detection_method(),
    )


def local_process_id(proc: Proc) -> int | None:
    for name in (
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
    float_type: type
    int_type: type
    proc: Proc
    backend: Backend

    def __init__(self, float_type: type, int_type: type, proc: Proc, backend: Backend):
        if float_type is jnp.float64:
            jax.config.update("jax_enable_x64", True)
        self.float_type = float_type
        self.int_type = int_type
        self.proc = proc
        self.backend = backend

    def __call__(self):
        """Return the JAX device architecture."""
        return self.backend()
    
    def is_root(self) -> bool:
        return self.proc.is_root()
    
    def is_distributed(self) -> bool:
        return jax.distributed.is_initialized()
    
    def barrier(self) -> None:
        if self.proc.comm_model is not None and self.proc.comm_model != MPI.COMM_NULL:
            self.proc.comm_model.Barrier()        
    
    def shutdown(self) -> None:
        self.barrier()
        if self.is_distributed():
            return jax.distributed.shutdown()
        
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
        return self.backend.device_string == "cpu"

    def is_metal(self) -> bool:
        return self.backend.device_string == "mps"
    
    def is_gpu(self) -> bool:
        return self.backend.device_string == "gpu"
    
    def is_tpu(self) -> bool:
        return self.backend.device_string == "tpu"

class CPU(AbstractDevice):
    """CPU device for computations."""

    def __init__(
        self, 
        float_type: type | str = jnp.float64, 
        int_type: type | str = jnp.int32,
        comm: MPI.Comm | None = None,
        use_async_io: bool = False,
        proc_io_stride: int = 0,
        use_jax_distributed: bool = False
    ):
        float_type_opt = TYPE_REMAPPING[float_type] if isinstance(float_type, str) else float_type
        int_type_opt   = TYPE_REMAPPING[int_type  ] if isinstance(int_type  , str) else int_type
        proc = Proc(
            comm=comm,
            use_async_io=use_async_io,
            proc_io_stride=proc_io_stride
        )
        distributed_initialize(
            proc=proc, 
            use_jax_distributed=use_jax_distributed
        )
        if self.is_apple_sys():
            jax.config.update("jax_platforms", "cpu")
        ref = jax.local_devices(backend="cpu")[0]
        super().__init__(
            float_type_opt, 
            int_type_opt, 
            proc,
            JaxBackend("cpu", ref)
        )


class GPU(AbstractDevice):
    """GPU device for computations."""

    def __init__(
        self, 
        float_type: type | str = jnp.float32, 
        int_type: type | str = jnp.int32,
        comm: MPI.Comm | None = None,
        use_async_io: bool = False,
        proc_io_stride: int = 0,
        use_jax_distributed: bool = False
    ):
        float_type_opt = TYPE_REMAPPING[float_type] if isinstance(float_type, str) else float_type
        int_type_opt   = TYPE_REMAPPING[int_type  ] if isinstance(int_type  , str) else int_type
        proc = Proc(
            comm=comm,
            use_async_io=use_async_io,
            proc_io_stride=proc_io_stride
        )
        distributed_initialize(
            proc=proc, 
            use_jax_distributed=use_jax_distributed
        )
        if (
            os.environ.get("CUDA_VISIBLE_DEVICES")
            or os.environ.get("JAX_CUDA_VISIBLE_DEVICES")
            or use_jax_distributed
        ):
            pass
        else:
            local_proc_id = local_process_id(proc)
            if local_proc_id is not None:
                jax.config.update("jax_cuda_visible_devices", str(local_proc_id))
        if self.is_apple_sys():
            jax.config.update("jax_platforms", "mps,cpu")
            backend = "mps"
        else:
            backend = "gpu"
        ref = jax.local_devices(backend=backend)[0]
        super().__init__(
            float_type_opt, 
            int_type_opt, 
            proc,
            JaxBackend(backend, ref)
        )


class TPU(AbstractDevice):
    """TPU device for computations."""

    def __init__(
        self, 
        float_type: type | str = jnp.float32, 
        int_type: type | str = jnp.int32,
        comm: MPI.Comm | None = None,
        use_async_io: bool = False,
        proc_io_stride: int = 0
    ):
        float_type_opt = TYPE_REMAPPING[float_type] if isinstance(float_type, str) else float_type
        int_type_opt   = TYPE_REMAPPING[int_type  ] if isinstance(int_type  , str) else int_type
        proc = Proc(
            comm=comm,
            use_async_io=use_async_io,
            proc_io_stride=proc_io_stride
        )
        distributed_initialize(
            proc=proc, 
            use_jax_distributed=True
        )
        ref = jax.local_devices(backend="tpu")[0]
        super().__init__(
            float_type_opt, 
            int_type_opt, 
            proc,
            JaxBackend("tpu", ref),
        )
