import jax
import jax.numpy as jnp


def filter_ordered_unique(
    x: jnp.ndarray,
    size: int | None = None,
    fill_value: int = -1,
    axis: int = 0
):
    """
    Filter ordered unique elements along a specified axis.

    Args:
        x: Input array.
        size: Desired size of the output array along the specified axis. 
              If None, uses the full length of the axis.
        fill_value: Value used to fill unused positions in the output array 
                    (Also used as a sentinel value). Default is -1.
        axis: Axis along which to filter unique elements. Default is 0.

    Returns:
        jnp.ndarray: Array with unique elements which exclude fill_value first and 
                     ordered following along the specified axis, padding with fill_values if necessary.

    """
    # Move the target axis to the last dimension to batch all preceding dims.
    x_last = jnp.moveaxis(x, axis, -1)
    leading_shape = x_last.shape[:-1]
    length = x_last.shape[-1]

    # If size is not given, fall back to the full length (static for JIT).
    size_local = size if size is not None else length

    def process_1d_array(arr: jnp.ndarray) -> jnp.ndarray:
        def body(i, carry):
            out, count, seen = carry
            val = arr[i]
            is_valid = val != fill_value
            exists = jnp.any(seen == val)
            is_new = jnp.logical_and(is_valid, jnp.logical_not(exists))
            room = count < size_local
            do_write = jnp.logical_and(is_new, room)
            out = jax.lax.cond(
                do_write,
                lambda o: o.at[count].set(val),
                lambda o: o,
                out,
            )
            seen = jax.lax.cond(
                do_write,
                lambda s: s.at[count].set(val),
                lambda s: s,
                seen,
            )
            count = jax.lax.cond(
                do_write,
                lambda c: c + 1,
                lambda c: c,
                count,
            )
            return out, count, seen

        init_out = jnp.full((size_local,), fill_value)
        init_seen = jnp.full((size_local,), fill_value)
        out, _, _ = jax.lax.fori_loop(
            0, length, body, (init_out, 0, init_seen)
        )
        return out

    # Batch over all leading dims flattened, then reshape back.
    x_flat = x_last.reshape((-1, length))
    out_flat = jax.vmap(process_1d_array, in_axes=0, out_axes=0)(x_flat)
    out_last = out_flat.reshape(leading_shape + (size_local,))
    return jnp.moveaxis(out_last, -1, axis)
