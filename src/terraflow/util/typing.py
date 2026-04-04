import datetime
import jax.numpy as jnp
import numpy as np
import unxt as ux
import jaxtyping
from typing import TypeVar
import os

# Enable runtime type checking for unxt
os.environ["UNXT_ENABLE_RUNTIME_TYPECHECKING"] = "beartype.beartype"


Array = jaxtyping.Array
ArrayWithUnits = jaxtyping.Array | ux.AbstractQuantity
Float = jaxtyping.Float
Int = jaxtyping.Int
FloatType = float | jnp.float32 | jnp.float64 | np.float32 | np.float64
IntType = int | jnp.int32 | jnp.int64 | np.int32 | np.int64
Numeric = FloatType | IntType
Bool = bool | jnp.bool | np.bool
Str = str | np.str_
NoneType = type(None)

DateTime = TypeVar("DateTime", bound=datetime.datetime)

PyTree = jaxtyping.PyTree
