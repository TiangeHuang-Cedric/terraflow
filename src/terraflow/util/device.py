import jax
import jax.numpy as jnp
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
        ref = jax.devices("cpu")[0]
        super().__init__(
            float_type, 
            int_type, 
            Proc(
                comm=comm,
                use_async_io=use_async_io,
                proc_io_stride=proc_io_stride
            ),
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
        ref = jax.devices("gpu")[0]
        super().__init__(
            float_type, 
            int_type, 
            Proc(
                comm=comm,
                use_async_io=use_async_io,
                proc_io_stride=proc_io_stride
            ),
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
        ref = jax.devices("tpu")[0]
        super().__init__(
            float_type, 
            int_type, 
            Proc(
                comm=comm,
                use_async_io=use_async_io,
                proc_io_stride=proc_io_stride
            ),
            JaxBackend("tpu", ref),
        )