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
    rd_o_cpd = para.rd_o_cpd.unwrap()
    p0 = para.p0.unwrap()
    return t * (p0 / p) ** rd_o_cpd


def temperature(
    pt: Float[Array, "..."],
    p: Float[Array, "..."],
    qv: Float[Array, "..."],
    para: PlanetParameterSet
) -> Float[Array, "..."]:
    """Calculate temperature.

    Args:
        pt: Modified potential temperature (K).
        p: Full pressure (Pa)
        qv: Dry mixing ratio of water vapor (kg kg-1)

    Returns:
        Temperature [K].
    """
    rd_o_cpd = para.rd_o_cpd.unwrap()
    rv_o_rd = para.rv_o_rd.unwrap()
    p0 = para.p0.unwrap()
    return pt * (p / p0) ** (rd_o_cpd) / (1 + rv_o_rd * qv)
