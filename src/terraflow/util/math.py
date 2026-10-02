import terraflow.util.typing as typing
from terraflow.util.backend import AbstractBackend

Float = typing.Float
FloatType = typing.FloatType
Array = typing.Array
ArrayWithUnits = typing.ArrayWithUnits


def roll_nonnegative_vector(bkd: AbstractBackend, x: Array, shift: int) -> Array:
    """
    Roll the elements of the input vector by a specified shift,
    while keeping the relative order of non-negative elements.
    Negative elements are treated as invalid and remain in their positions.
    Args:
        x: Input vector.
        shift (int): Number of positions to roll the elements.
    Returns:
        Rolled array of the same shape as input.
    """
    n = bkd.sum(x >= 0, axis=-1, keepdim=False)
    length = x.shape[0]
    idx = bkd.arange(length)
    valid_len = bkd.maximum(n, 1)
    mask = idx < n
    rotate_idx = bkd.where(mask, (idx + shift) % valid_len, idx)
    rotated = bkd.take(x, indices=bkd.asarray(rotate_idx), axis=0)
    return bkd.where(n > 0, rotated, x)


def roll_nonnegative(bkd: AbstractBackend, x: Array, shift: int) -> Array:
    """
    Roll the elements of the input array by a specified shift along
    the last axis, while keeping the relative order of non-negative elements.
    Negative elements are treated as invalid and remain in their positions.
    Args:
        x:
        shift:
    Returns:
        Rolled array of the same shape as input.
    """
    return bkd.vmap(lambda x_i: roll_nonnegative_vector(bkd, x_i, shift))(x)


def rotate_nonnegative_vector(bkd: AbstractBackend, x: Array) -> Array:
    """
    Rotate the elements of the input vector to the left by one position,
    while keeping the relative order of non-negative elements.
    Negative elements are treated as invalid and remain in their positions.
    Args:
        x: Input vector.
    Returns:
        Rotated array of the same shape as input.
    """
    n = bkd.sum(x >= 0, axis=-1, keepdim=False)
    length = x.shape[0]
    idx = bkd.arange(length)
    valid_len = bkd.maximum(n, 1)
    mask = idx < n
    rotate_idx = bkd.where(mask, (idx-1) % valid_len, idx)
    rotated = bkd.take(x, indices=bkd.asarray(rotate_idx), axis=0)
    return bkd.where(n > 0, rotated, x)


def rotate_nonnegative(bkd: AbstractBackend, x: Array) -> Array:
    """
    Rotate the elements of the input array to the left by one position along
    the last axis, while keeping the relative order of non-negative elements.
    Negative elements are treated as invalid and remain in their positions.
    Args:
        x: Input array.
    Returns:
        Rotated array of the same shape as input.
    """
    return bkd.vmap(lambda x_i: rotate_nonnegative_vector(bkd, x_i))(x)


def normalize(bkd: AbstractBackend, p: Array, axis: int = -1) -> Array:
    """
    Normalize a vector to have unit length.
    Args:
        p: Input vector of shape (..., 3).
    Returns:
        Normalized vector of the same shape as input.
    """
    norm: Array = bkd.norm(p, axis=axis, keepdims=True)
    denom: Array = bkd.where(norm == 0, 1.0, norm)
    return p / denom


def pinv_svd(bkd: AbstractBackend, A: Float[Array, "m n"], rtol=1e-15):
    """Compute the Moore-Penrose pseudoinverse of matrix A using SVD.

    Args:
        A: Float[Array, "m n"], input matrix to be pseudoinverted.
        rtol: float, relative tolerance for small singular values.

    Returns:
        A_pinv: Float[Array, "n m"], pseudoinverse of A.
    """
    U, S, VT = bkd.svd(A, full_matrices=False)
    tol = rtol * bkd.max(S)
    S_inv = bkd.pinv(bkd.diag(S), rcond=tol)
    A_pinv = VT.T @ S_inv @ U.T
    return A_pinv


def exp_two_values(
    bkd: AbstractBackend,
    val1: Float[Array, "..."],
    val2: Float[Array, "..."],
    x0: Float[Array, "..."],
    x1: Float[Array, "..."],
    x: Float[Array, "..."]
) -> Float[Array, "..."]:
    eps = bkd.eps(val1.dtype)
    eps_array = bkd.asarray(eps, dtype=val1.dtype)
    w = bkd.exp((x - x0)**2 * bkd.log(eps_array) / (x1 - x0)**2)
    return w * val1 + (1 - w) * val2


def sign(bkd: AbstractBackend, x: Array) -> Array:
    one = bkd.ones_like(x)
    return bkd.where(x >= 0, one, -one)


def tridiag_thomas(
    bkd: AbstractBackend,
    a: Float[Array, "nlev ncell"],
    b: Float[Array, "nlev ncell"],
    c: Float[Array, "nlev ncell"],
    d: Float[Array, "nlev ncell"]
) -> Float[Array, "nlev ncell"]:
    nlev = d.shape[0]
    cp = bkd.zeros_like(c)
    dp = bkd.zeros_like(d)

    denom = b[0, :]
    cp = bkd.slice_update(cp, (0, slice(None)), c[0, :] / denom)
    dp = bkd.slice_update(dp, (0, slice(None)), d[0, :] / denom)
    for k in range(1, nlev - 1):
        denom = b[k, :] - a[k, :] * cp[k - 1, :]
        cp = bkd.slice_update(cp, (k, slice(None)), c[k, :] / denom)
    for k in range(1, nlev):
        denom = b[k, :] - a[k, :] * cp[k - 1, :]
        dp = bkd.slice_update(
            dp,
            (k, slice(None)),
            (d[k, :] - a[k, :] * dp[k - 1, :]) / denom
        )

    x = bkd.zeros_like(d)
    x = bkd.slice_update(x, (-1, slice(None)), dp[-1, :])
    for k in range(nlev - 2, -1, -1):
        x = bkd.slice_update(
            x,
            (k, slice(None)),
            dp[k, :] - cp[k, :] * x[k + 1, :]
        )
    return x