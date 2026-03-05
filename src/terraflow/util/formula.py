from terraflow.util import typing
from terraflow.planet import PlanetParameterSet

Float = typing.Float
Array = typing.Array


def dry_potential_temperature(
    t: Float[Array, "..."],
    p: Float[Array, "..."],
    para: PlanetParameterSet
) -> Float[Array, "..."]:
    """Calculate dry potential temperature.

    Args:
        t: Temperature [K].
        p: Pressure [Pa].

    Returns:
        Dry potential temperature [K].
    """
    p0 = para.p0.unwrap()
    rd = para.rd.unwrap()
    cp = para.cp.unwrap()
    return t * (p0 / p) ** (rd / cp)
