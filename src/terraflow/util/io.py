from __future__ import annotations
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    import numpy as np
    from terraflow.util import typing

    Number = typing.Numeric


def load_netcdf(file_path: str) -> xr.Dataset:
    """Load a NetCDF file into an xarray Dataset."""
    return xr.open_dataset(file_path)


def get_string(dataset: xr.Dataset, var_name: str) -> str:
    """Get the string of a variable from the xarray Dataset."""
    return dataset[var_name].item()


def get_scalar(dataset: xr.Dataset, var_name: str) -> Number:
    """Get the items of a variable from the xarray Dataset."""
    return dataset[var_name].item()


def get_array(dataset: xr.Dataset, var_name: str) -> np.ndarray:
    """Get a variable from the xarray Dataset."""
    return dataset[var_name].values


def save_netcdf(dataset: xr.Dataset, file_path: str):
    """Save an xarray Dataset to a NetCDF file."""
    dataset.to_netcdf(file_path)
