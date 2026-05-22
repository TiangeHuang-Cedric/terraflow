from terraflow.util import typing
from terraflow.planet import PlanetParameterSet

Float = typing.Float
Array = typing.Array


def wet_mixing_ratio(
    qv: Float[Array, "..."],
    qm: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Calculate wet mixing ratio of water vapor.

    Args:
        qv: Dry mixing ratio of water vapor (kg kg-1)
        qm: Total dry mixing ratio of water vapor and its condensate (kg kg-1)

    Returns:
        Wet mixing ratio of water vapor (kg kg-1)
    """
    return qv / (1 + qm)


def dry_mixing_ratio(
    qv: Float[Array, "..."],
    qm: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Calculate dry mixing ratio of water vapor.

    Args:
        qv: Specific humidity or wet mixing ratio of water vapor (kg kg-1)
        qm: Total dry mixing ratio of water vapor and its condensate (kg kg-1)

    Returns:
        Dry mixing ratio of water vapor (kg kg-1)
    """
    return qv * (1 + qm)


def dry_potential_temperature(
    t: Float[Array, "..."],
    p: Float[Array, "..."],
    para: PlanetParameterSet
) -> Float[Array, "..."]:
    """Calculate dry potential temperature.

    Args:
        t: Temperature [K].
        p: Pressure [Pa].
        para: PlanetParameterSet

    Returns:
        Dry potential temperature [K].
    """
    rd_o_cpd = para.rd_o_cpd.unwrap()
    p0 = para.p0.unwrap()
    return t * (p0 / p) ** rd_o_cpd


def potential_temperature(
    t: Float[Array, "..."],
    p: Float[Array, "..."],
    para: PlanetParameterSet
) -> Float[Array, "..."]:
    """Calculate potential temperature.

    Args:
        t: Temperature [K].
        p: Full pressure [Pa].
        para: PlanetParameterSet

    Returns:
        Potential temperature [K].
    """
    rd_o_cpd = para.rd_o_cpd.unwrap()
    p0 = para.p0.unwrap()
    return t * (p0 / p) ** rd_o_cpd


def modified_potential_temperature(
    t: Float[Array, "..."],
    p: Float[Array, "..."],
    qv: Float[Array, "..."],
    para: PlanetParameterSet
) -> Float[Array, "..."]:
    """Calculate modified potential temperature.

    Args:
        t: Temperature [K].
        p: Full Pressure [Pa].
        qv: Dry mixing ratio of water vapor (kg kg-1)
        para: PlanetParameterSet

    Returns:
        Modified potential temperature [K].
    """
    rd_o_cpd = para.rd_o_cpd.unwrap()
    rv_o_rd = para.rv_o_rd.unwrap()
    p0 = para.p0.unwrap()
    return t * (p0 / p) ** rd_o_cpd * (1 + rv_o_rd * qv)


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
        para: PlanetParameterSet

    Returns:
        Temperature [K].
    """
    rd_o_cpd = para.rd_o_cpd.unwrap()
    rv_o_rd = para.rv_o_rd.unwrap()
    p0 = para.p0.unwrap()
    return pt * (p / p0) ** rd_o_cpd / (1 + rv_o_rd * qv)


def temperature_from_virtual_temperature_and_wet_mixing_ratio(
    tv: Float[Array, "..."],
    qv: Float[Array, "..."],
    para: PlanetParameterSet
) -> Float[Array, "..."]:
    """Calculate temperature from virtual temperature and wet mixing ratio.

    Args:
        tv: Virtual temperature [K].
        qv: Wet mixing ratio of water vapor (kg kg-1)
        para: PlanetParameterSet

    Returns:
        Temperature [K].
    """
    rv_o_rd = para.rv_o_rd.unwrap()
    return tv / (1 + (rv_o_rd - 1) * qv)


def temperature_from_virtual_temperature_and_dry_mixing_ratio(
    tv: Float[Array, "..."],
    qv: Float[Array, "..."],
    qm: Float[Array, "..."],
    para: PlanetParameterSet
) -> Float[Array, "..."]:
    """Calculate temperature from virtual temperature and dry mixing ratio.

    Args:
        tv: Virtual temperature [K].
        qv: Dry mixing ratio of water vapor (kg kg-1)
        qm: Total dry mixing ratio of water vapor and its condensate (kg kg-1)
        para: PlanetParameterSet

    Returns:
        Temperature [K].
    """
    rv_o_rd = para.rv_o_rd.unwrap()
    return tv / (1 + rv_o_rd * qv) * (1 + qm)


def temperature_from_density_and_wet_mixing_ratio(
    rho: Float[Array, "..."],
    p: Float[Array, "..."],
    qv: Float[Array, "..."],
    para: PlanetParameterSet
) -> Float[Array, "..."]:
    """Calculate temperature from density and wet mixing ratio.

    Args:
        rho: Air Density [kg m-3].
        p: Full pressure [Pa].
        qv: Wet mixing ratio of water vapor (kg kg-1)
        para: PlanetParameterSet

    Returns:
        Temperature [K].
    """
    rd = para.rd.unwrap()
    rv_o_rd = para.rv_o_rd.unwrap()
    return p / rd / rho / (1 + (rv_o_rd - 1) * qv)


def virtual_temperature_from_dry_mixing_ratio(
    t: Float[Array, "..."],
    qv: Float[Array, "..."],
    qm: Float[Array, "..."],
    para: PlanetParameterSet
) -> Float[Array, "..."]:
    """Calculate virtual temperature from dry mixing ratio.
    See (16) in Lauritzen et al. (2018) for details.

    Args:
        t: Temperature [K].
        qv: Dry mixing ratio of water vapor (kg kg-1)
        qm: Total dry mixing ratio of water vapor and its condensate (kg kg-1)
        para: PlanetParameterSet

    Returns:
        Virtual temperature [K].
    """
    rv_o_rd = para.rv_o_rd.unwrap()
    return t * (1 + rv_o_rd * qv) / (1 + qm)


def virtual_temperature_from_wet_mixing_ratio(
    t: Float[Array, "..."],
    qv: Float[Array, "..."],
    para: PlanetParameterSet
) -> Float[Array, "..."]:
    """Calculate virtual temperature from wet mixing ratio.

    Args:
        t: Temperature [K].
        qv: Dry mixing ratio of water vapor (kg kg-1)
        para: PlanetParameterSet

    Returns:
        Virtual temperature [K].
    """
    rv_o_rd = para.rv_o_rd.unwrap()
    return t * (1 + (rv_o_rd - 1) * qv)


def virtual_temperature_from_density(
    p: Float[Array, "..."],
    rho: Float[Array, "..."],
    para: PlanetParameterSet
) -> Float[Array, "..."]:
    """Calculate virtual temperature from density and pressure.

    Args:
        p: Full pressure [Pa].
        rho: Air Density [kg m-3].
        para: PlanetParameterSet

    Returns:
        Virtual temperature [K].
    """
    rd = para.rd.unwrap()
    return p / rd / rho


def virtual_temperature_from_modified_potential_temperature(
    pt: Float[Array, "..."],
    pk: Float[Array, "..."],
    qm: Float[Array, "..."],
    para: PlanetParameterSet
) -> Float[Array, "..."]:
    """Calculate virtual temperature from modified potential temperature.

    Args:
        pt: Modified potential temperature (K).
        pk: p**(rd/cpd)
        qm: Total dry mixing ratio of water vapor and its condensate (kg kg-1)
        para: PlanetParameterSet

    Returns:
        Virtual temperature [K].
    """
    pk0 = para.pk0.unwrap()
    return pt * pk / pk0 / (1 + qm)


def virtual_potential_temperature(
    tv: Float[Array, "..."],
    p: Float[Array, "..."],
    para: PlanetParameterSet
) -> Float[Array, "..."]:
    """Calculate virtual potential temperature.

    Args:
        tv: Virtual temperature [K].
        p: Full pressure [Pa].
        para: PlanetParameterSet

    Returns:
        Virtual potential temperature [K].
    """
    rd_o_cpd = para.rd_o_cpd.unwrap()
    p0 = para.p0.unwrap()
    return tv * (p0 / p) ** rd_o_cpd


def dry_air_density(
    t: Float[Array, "..."],
    p: Float[Array, "..."],
    para: PlanetParameterSet
) -> Float[Array, "..."]:
    """Calculate dry air density.

    Args:
        t: Temperature [K].
        p: Full pressure [Pa].
        para: PlanetParameterSet

    Returns:
        Dry air density [kg m-3].
    """
    rd = para.rd.unwrap()
    return p / rd / t


def moist_air_density(
    t: Float[Array, "..."],
    p: Float[Array, "..."],
    qv: Float[Array, "..."],
    qm: Float[Array, "..."],
    para: PlanetParameterSet
) -> Float[Array, "..."]:
    """Calculate moist air density.

    Args:
        t: Temperature [K].
        p: Full pressure [Pa].
        qv: Dry mixing ratio of water vapor (kg kg-1)
        qm: Total dry mixing ratio of water vapor and its condensate (kg kg-1)
        para: PlanetParameterSet

    Returns:
        Moist air density [kg m-3].
    """
    rd = para.rd.unwrap()
    return p / rd / virtual_temperature_from_dry_mixing_ratio(t, qv, qm, para)
