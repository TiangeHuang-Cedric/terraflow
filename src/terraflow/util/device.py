import jax
import jax.numpy as jnp
import os
from mpi4py import MPI


from terraflow.util.parallel_types import Proc


class Backend:
    """Base class for computational backends."""

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
    if has_slurm() or has_openmpi():
        return None
    if MPI.COMM_WORLD.Get_size() > 1:
        return "mpi4py"
    return None


def distributed_initialize(
    proc: Proc,
) -> None:
    if proc.np_model <= 1 or jax.distributed.is_initialized():
        return
    if not proc.is_model():
        return

    jax.distributed.initialize(
        cluster_detection_method=cluster_detection_method(),
    )


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


class CPU(AbstractDevice):
    """CPU device for computations."""

    def __init__(
        self, 
        float_type: type = jnp.float64, 
        int_type: type = jnp.int64,
        comm: MPI.Comm | None = None,
        use_async_io: bool = False,
        proc_io_stride: int = 0
    ):
        proc = Proc(
            comm=comm,
            use_async_io=use_async_io,
            proc_io_stride=proc_io_stride
        )
        distributed_initialize(proc)
        ref = jax.local_devices(backend="cpu")[0]
        super().__init__(
            float_type, 
            int_type, 
            proc,
            JaxBackend("cpu", ref)
        )


class GPU(AbstractDevice):
    """GPU device for computations."""

    def __init__(
        self, 
        float_type: type = jnp.float32, 
        int_type: type = jnp.int32,
        comm: MPI.Comm | None = None,
        use_async_io: bool = False,
        proc_io_stride: int = 0
    ):
        proc = Proc(
            comm=comm,
            use_async_io=use_async_io,
            proc_io_stride=proc_io_stride
        )
        distributed_initialize(proc)
        ref = jax.local_devices(backend="gpu")[0]
        super().__init__(
            float_type, 
            int_type, 
            proc,
            JaxBackend("gpu", ref)
        )


class TPU(AbstractDevice):
    """TPU device for computations."""

    def __init__(
        self, 
        float_type: type = jnp.float32, 
        int_type: type = jnp.int32,
        comm: MPI.Comm | None = None,
        use_async_io: bool = False,
        proc_io_stride: int = 0
    ):
        proc = Proc(
            comm=comm,
            use_async_io=use_async_io,
            proc_io_stride=proc_io_stride
        )
        distributed_initialize(proc)
        ref = jax.local_devices(backend="tpu")[0]
        super().__init__(
            float_type, 
            int_type, 
            proc,
            JaxBackend("tpu", ref),
        )
