#!/usr/bin/env python3

"""
This script is used to remap a NetCDF file to a specified resolution using CDO.
It can also generate a weight file if required.
Usage: python Fluida2Latlon.py -N 1 -w input.nc output.nc
"""
import argparse
import pathlib
import subprocess
from loguru import logger


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


def run(cmd, silent=False):
    stdout = subprocess.DEVNULL if silent else None
    stderr = subprocess.DEVNULL if silent else None
    subprocess.run(cmd, check=True, stdout=stdout, stderr=stderr)


def cdo_remap(infile, outfile, res, use_weight, silent=False):
    if use_weight:
        run(["cdo", f"gendis,global_{res}",
            infile, "weight.nc"], silent=silent)
        run(["cdo", f"remap,global_{res},weight.nc",
            infile, outfile], silent=silent)
        pathlib.Path("weight.nc").unlink(missing_ok=True)
        pathlib.Path("tmp.nc").unlink(missing_ok=True)
    else:
        run(["cdo", f"remapdis,global_{res}", infile, outfile], silent=silent)


def main():
    args = parse_commandline()
    infile = args.inputfile
    filehead = pathlib.Path(infile).with_suffix("")
    res = args.Number
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
        cdo_remap(infile, outfile, res, use_weight, silent=False)
        logger.info("Remapping completed successfully.")
    except Exception:
        run([
            "ncks",
            "-O", "-x", "-v", "lon_vtx,lat_vtx", infile, "tmp.nc"
        ])
        cdo_remap("tmp.nc", outfile, res, use_weight, silent=False)
        run(["rm", "tmp.nc"])
        logger.info(
            "Remapping completed successfully after removing vertex coordinates."
        )


if __name__ == "__main__":
    main()
