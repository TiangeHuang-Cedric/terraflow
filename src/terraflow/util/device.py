import jax
import jax.numpy as jnp


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
    backend: Backend

    def __init__(self, float_type: type, int_type: type, backend: Backend):
        if float_type is jnp.float64:
            jax.config.update("jax_enable_x64", True)
        self.float_type = float_type
        self.int_type = int_type
        self.backend = backend

    def __call__(self):
        """Return the JAX device architecture."""
        return self.backend()


class CPU(AbstractDevice):
    """CPU device for computations."""

    def __init__(self, float_type: type = jnp.float64, int_type: type = jnp.int64):
        ref = jax.devices("cpu")[0]
        super().__init__(float_type, int_type, JaxBackend("cpu", ref))


class GPU(AbstractDevice):
    """GPU device for computations."""

    def __init__(self, float_type: type = jnp.float32, int_type: type = jnp.int32):
        ref = jax.devices("gpu")[0]
        super().__init__(float_type, int_type, JaxBackend("gpu", ref))


class TPU(AbstractDevice):
    """TPU device for computations."""

    def __init__(self, float_type: type = jnp.float32, int_type: type = jnp.int32):
        ref = jax.devices("tpu")[0]
        super().__init__(float_type, int_type, JaxBackend("tpu", ref))


class Distributed(AbstractDevice):
    """Base class for distributed devices."""

    def __init__(self, singe_device: AbstractDevice):
        if isinstance(singe_device, CPU):
            return DistributedCPU(singe_device.float_type, singe_device.int_type)
        elif isinstance(singe_device, GPU):
            return DistributedGPU(singe_device.float_type, singe_device.int_type)
        elif isinstance(singe_device, TPU):
            return DistributedTPU(singe_device.float_type, singe_device.int_type)
        else:
            raise TypeError("Unsupported device type for distributed device.")


class DistributedCPU(CPU):
    """Distributed CPU device for computations."""

    def __init__(self, float_type: type = jnp.float64, int_type: type = jnp.int64):
        ref = jax.devices("cpu")
        AbstractDevice.__init__(self, float_type, int_type, JaxBackend("cpu", ref))


class DistributedGPU(GPU):
    """Distributed GPU device for computations."""

    def __init__(self, float_type: type = jnp.float32, int_type: type = jnp.int32):
        ref = jax.devices("gpu")
        AbstractDevice.__init__(self, float_type, int_type, JaxBackend("gpu", ref))


class DistributedTPU(TPU):
    """Distributed TPU device for computations."""

    def __init__(self, float_type: type = jnp.float32, int_type: type = jnp.int32):
        ref = jax.devices("tpu")
        AbstractDevice.__init__(self, float_type, int_type, JaxBackend("tpu", ref))
