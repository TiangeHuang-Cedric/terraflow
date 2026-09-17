import datetime
import numpy as np
import jaxtyping
import pint
from typing import TypeVar, Any


Array = Any
ArrayWithUnits = Any | pint.Quantity
Float = jaxtyping.Float
Int = jaxtyping.Int
DTypeLike = jaxtyping.DTypeLike
FloatType = float | jaxtyping.Float16 | jaxtyping.Float32 | jaxtyping.Float64
IntType = int | jaxtyping.Int16 | jaxtyping.Int32 | jaxtyping.Int64 
Number = FloatType | IntType
Bool = bool | jaxtyping.Bool
Str = str | np.str_
NoneType = type(None)

DateTime = TypeVar("DateTime", bound=datetime.datetime)

PyTree = jaxtyping.PyTree
