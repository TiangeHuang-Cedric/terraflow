#!/usr/bin/env python3

"""
This script is used to remap a NetCDF file to a specified resolution using CDO.
It can also generate a weight file if required.
Usage: python Fluida2Latlon.py -N 1 -w input.nc output.nc
"""
import argparse
import pathlib
from loguru import logger
from terraflow.util.post_process import CDOPostProcesser as cdo


def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputfile",
        type=str,
        help="Path to the input NetCDF file."
    )
    parser.add_argument(
        "outputfile",
        nargs="?",
        type=str,
        help="Path to the Output file"
    )
    parser.add_argument(
        "--Number",
        "-N",
        type=float,
        default=1,
        required=False,
        help="Resolotion to be interpolated to, e.g., 1 represents for 1d * 1d.",
    )
    parser.add_argument(
        "--use-weight",
        "-w",
        action="store_true",
        help="If present, generate weight file before remapping.",
    )
    return parser.parse_args()


def main():
    args = parse_commandline()
    infile = args.inputfile
    filehead = pathlib.Path(infile).with_suffix("")
    res = float(args.Number)
    res = int(res) if res.is_integer() else res
    if args.outputfile is not None:
        outfile = args.outputfile
    else:
        outfile = f"{filehead}.{res}×{res}.nc"
    use_weight = args.use_weight

    logger.info(
        f"Remapping {infile} to {outfile} with resolution {res}×{res}."
    )
    if use_weight:
        logger.info(
            "Remapping will be done using generated weight."
        )

    logger.info(
        "Starting remapping..."
    )

    try:
        cdo.remap_netcdf(
            infile=infile,
            outfile=outfile,
            res=res,
            use_weight=use_weight,
            silent=False
        )
        logger.info("Remapping completed successfully.")
    except Exception:
        cdo.run([
            "ncks",
            "-O", "-x", "-v", "lon_vtx,lat_vtx", infile, "tmp.nc"
        ])
        cdo.remap_netcdf(
            infile="tmp.nc",
            outfile=outfile,
            res=res,
            use_weight=use_weight,
            silent=False
        )
        cdo.run(["rm", "tmp.nc"])
        logger.info(
            "Remapping completed successfully after removing vertex coordinates."
        )


if __name__ == "__main__":
    main()
