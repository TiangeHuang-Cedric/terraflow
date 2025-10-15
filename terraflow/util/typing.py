import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray | np.ndarray
Float = float | jnp.float32 | jnp.float64 | np.float32 | np.float64
Int = int | jnp.int32 | jnp.int64 | np.int32 | np.int64
Bool = bool | jnp.bool | np.bool
Str = str | np.str_
NoneType = type(None)
