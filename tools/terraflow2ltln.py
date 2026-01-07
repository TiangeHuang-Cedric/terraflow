#!/usr/bin/env python3

"""
This script is used to remap a NetCDF file to a specified resolution using CDO.
It can also generate a weight file if required.
Usage: python Fluida2Latlon.py -N 1 -w input.nc output.nc
"""
import argparse
import pathlib
import subprocess


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


def run(cmd):
    subprocess.run(cmd, check=True)


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

    if use_weight:
        run(["cdo", f"gendis,global_{res}", infile, "weight.nc"])
        run(["cdo", f"remap,global_{res},weight.nc", infile, outfile])
        pathlib.Path("weight.nc").unlink(missing_ok=True)
    else:
        run(["cdo", f"remapdis,global_{res}", infile, outfile])


if __name__ == "__main__":
    main()
