import numpy as np
import xarray as xr
from typing import Optional
import healpix as hp
import easygems.healpix as egh
import easygems.remap as egr


def terraflow2healpix(
    field: xr.DataArray,
    from_lon: np.ndarray,
    from_lat: np.ndarray,
    order: int,
    weights_input: Optional[xr.Dataset] = None,
):
    """Remap data from TerraFlow grid to Healpix grid.

    Args:
        from_lon (ndarray): Longitudes of the TerraFlow grid points.
        from_lat (ndarray): Latitudes of the TerraFlow grid points.
        order (int): Order of Healpix grid.
        nside (int): Nside of Healpix grid.
        weights (ndarray): Data values on the TerraFlow grid points.

    Returns:
        ndarray: Remapped data on the Healpix grid.
    """
    # Convert coordinates to NumPy arrays and avoid exact pole values that
    # cause stereographic projections (used by QHull inside easygems) to
    # produce infinities.
    from_lon = np.asarray(from_lon, dtype=float)
    from_lat = np.asarray(from_lat, dtype=float)

    pole_mask = np.isclose(np.abs(from_lat), 90.0)
    if np.any(pole_mask):
        from_lat = from_lat.copy()
        eps_deg = 1e-4  # ~11m on the sphere; large enough to avoid infinities
        from_lat[pole_mask & (from_lat > 0.0)] = 90.0 - eps_deg
        from_lat[pole_mask & (from_lat < 0.0)] = -90.0 + eps_deg

    # Get pixel lon and lat from TerraFlow grid points
    nside = hp.order2nside(order)
    hp_lon, hp_lat = hp.pix2ang(
        nside,
        np.arange(hp.nside2npix(nside)),
        lonlat=True,
        nest=True,
    )

    # Compute remapping weights
    weights = egr.compute_weights_delaunay(
        points=(from_lon, from_lat),
        xi=(hp_lon, hp_lat)
    ) if weights_input is None else weights_input

    return egr.apply_weights(field, weights.src_idx, weights.weights, weights.valid)


def plot_healpix_grid(data: np.ndarray):
    """Plot Healpix grid.

    Args:
        data (np.ndarray): Input data array.
    """
    return egh.healpix_show(data)
