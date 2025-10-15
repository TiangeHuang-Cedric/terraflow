import jax
import pytest

from terraflow.util import math

math.rotate_nonnegative(jnp.array([1, 2, 3, -1, 4, 5]), axis=0)