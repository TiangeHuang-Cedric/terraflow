#!/usr/bin/env python3

"""Use 1D linear interpolation from model levels to pressure levels.

Example:
    python tools/plot_ml2pl.py \
        --input examples/drivers/terraflow.g4.h0.nc \
        --pressure-var p \
        --lev-dim lev \
        --levels 1000,850,700,500 \
        --levels-unit hPa \
    --output output_plev.nc
"""

from __future__ import annotations
import argparse
from loguru import logger
import numpy as np
import xarray as xr
from xarray.coders import CFDatetimeCoder
import unxt as ux
import jax
import jax.numpy as jnp
import terraflow as tf


LON_NAME = ["lon", "ilon", "lon_cell", "lon_edge", "lon_vtx"]
LAT_NAME = ["lat", "ilat", "lat_cell", "lat_edge", "lat_vtx"]


def parse_levels(levels_text: str, unit: str) -> np.ndarray:
    values = np.array(
        [float(v.strip()) for v in levels_text.split(",") if v.strip()]
    )
    if values.size == 0:
        raise ValueError(
            "--levels must not be empty, e.g. --levels 1000,850,500"
        )
    if unit.lower() == "hpa":
        values = values * 100.0
    return values.astype(float)


def infer_pressure_unit(unit_text: str | None) -> str:
    if unit_text is None:
        return "Pa"
    text = unit_text.strip().lower()
    if text in {"pa", "pascal", "pascals"}:
        return "Pa"
    if text in {"hpa", "hectopascal", "hectopascals", "mbar", "mb"}:
        return "hPa"
    return "Pa"


def interp_hybrid2pressure(
    field: xr.DataArray,
    mlev: jnp.ndarray,
    plev: jnp.ndarray,
) -> xr.DataArray:
    field_out = jax.vmap(
        lambda x, mlev: tf.vert_coord.hybrid_coord.interp2plev(
            x,
            mlev,
            plev,
            left="extrapolate",
            right="extrapolate"
        ),
        in_axes=(0, 0),
        out_axes=0
    )(
        jnp.array(field.values),
        mlev
    )
    return xr.DataArray(
        field_out,
        dims=field.dims,
        attrs=field.attrs,
    )


def set_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interpolate model levels to pressure levels with 1D linear interpolation."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input netCDF file"
    )
    parser.add_argument(
        "--var",
        default=None,
        help="Optional variable name to interpolate. If omitted, all non-scalar variables will be processed.",
    )
    parser.add_argument(
        "--pres-varname",
        default="ph",
        help="Existing hydrostatic pressure variable name (must include vertical dim, e.g. lev)",
    )
    parser.add_argument(
        "--levels",
        default=[1000, 925, 850, 700, 500, 300, 200, 100],
        help="Target pressure levels as comma-separated values, e.g. 1000,925,850,700,500 (default unit hPa)",
    )
    parser.add_argument(
        "--levels-unit",
        default="hPa",
        choices=["hPa", "Pa", "HPA", "PA"],
        help="Unit of --levels, default is hPa",
    )
    parser.add_argument(
        "--pres-unit",
        default="Pa",
        choices=["hPa", "Pa", "HPA", "PA", None],
        help="Unit of hydrostatic pressure variable; default infers from pressure-var attrs['units']",
    )
    parser.add_argument(
        "--lev-dim",
        default="lev",
        help="Vertical dimension name, default lev"
    )
    parser.add_argument(
        "--complevel",
        type=int,
        default=4,
        help="Compression level for output netCDF (0-9), default 4"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output netCDF file path. Default: <input_stem>_plev.nc",
    )
    return parser


def parse_hydrostatic_pressure(args: argparse.Namespace) -> jnp.ndarray:
    ds = xr.open_dataset(args.input)
    pres_da = ds[args.pres_varname]
    ds.close()
    if args.lev_dim not in pres_da.dims:
        raise ValueError(
            f"Pressure variable '{args.pres_varname}' does not contain vertical dimension '{args.lev_dim}'"
        )

    pres_unit = (
        infer_pressure_unit(args.pres_unit)
        if args.pres_unit
        else infer_pressure_unit(pres_da.attrs.get("units"))
    )
    mlev: ux.Quantity = ux.Quantity(
        jnp.array(pres_da.values),
        unit=pres_unit
    )
    return mlev.decompose(ux.unitsystems.si).value


def parse_target_levels(args: argparse.Namespace) -> jnp.ndarray:
    if isinstance(args.levels, str):
        level_values = parse_levels(args.levels, args.levels_unit)
    else:
        level_values = np.array(args.levels, dtype=float)
    plev = ux.Quantity(
        jnp.array(level_values),
        unit="Pa" if args.levels_unit.lower() == "pa" else "hPa"
    )
    logger.info(
        f"Target levels (plev) shape: {plev.shape}, values: {plev.value}, unit: {plev.unit}"
    )
    return plev.decompose(ux.unitsystems.si).value


def interp_run(
    args: argparse.Namespace,
    mlev: jnp.ndarray,
    plev: jnp.ndarray
) -> dict[str, xr.DataArray]:
    ds = xr.open_dataset(args.input)
    if args.pres_varname not in ds:
        raise KeyError(
            f"Hydrostatic pressure variable '{args.pres_varname}' not found in input file"
        )

    if args.var is not None:
        if args.var not in ds:
            raise KeyError(f"Variable '{args.var}' not found in input file")
        candidate_names = [args.var]
    else:
        candidate_names = [name for name in ds.data_vars.keys()]

    out_vars: dict[str, xr.DataArray] = {}
    for var_name in candidate_names:
        if var_name == args.pres_varname:
            logger.warning(f"Skipping pressure variable itself: {var_name}")
            continue
        if var_name == args.lev_dim:
            logger.warning(
                f"Skipping vertical coordinate variable itself: {var_name}"
            )
            continue

        field = ds[var_name]
        if field.ndim == 0:
            logger.warning(f"Skipping scalar variable: {var_name}")
            out_vars[str(var_name)] = field
            continue
        if args.lev_dim not in field.dims:
            logger.warning(
                f"Skipping variable without vertical dimension '{args.lev_dim}': {var_name}"
            )
            continue

        logger.info(f"Interpolating variable: {var_name}")
        interp_da = interp_hybrid2pressure(
            field=field,
            mlev=mlev,
            plev=plev,
        )
        out_vars[str(var_name)] = interp_da
    ds.close()
    return out_vars


def output(
    args: argparse.Namespace,
    out_vars: dict[str, xr.DataArray],
    plev: jnp.ndarray
) -> None:
    output_path = args.output
    if output_path is None:
        input_path = args.input
        if input_path.lower().endswith(".nc"):
            output_path = f"{input_path[:-3]}.plev.nc"
        else:
            output_path = f"{input_path}.plev.nc"
    ds_input = xr.open_dataset(
        args.input,
        decode_times=CFDatetimeCoder(use_cftime=True)
    )
    ds = xr.Dataset(out_vars, attrs=ds_input.attrs)

    # Keep original coordinates that do not depend on the old vertical dimension.
    for coord_name, coord_da in ds_input.coords.items():
        if args.lev_dim not in coord_da.dims:
            ds = ds.assign_coords({coord_name: coord_da})

    ds = ds.assign_coords({args.lev_dim: plev})
    ds[args.lev_dim].attrs["long_name"] = "Pressure Level"
    ds[args.lev_dim].attrs["description"] = "Pressure levels interpolated from model levels"
    ds[args.lev_dim].attrs["units"] = "Pa"

    comp = dict(zlib=True, complevel=args.complevel)
    encoding = {v: comp for v in ds.data_vars}

    ds_input.close()
    ds.to_netcdf(output_path, engine="netcdf4", encoding=encoding)
    logger.success(f"Interpolation finished. Saved file: {output_path}")


parser = set_parser()

args = parser.parse_args()

mlev = parse_hydrostatic_pressure(args)
plev = parse_target_levels(args)

out_vars = interp_run(args, mlev, plev)

output(args, out_vars, plev)
