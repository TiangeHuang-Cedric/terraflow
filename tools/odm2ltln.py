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
    parser.add_argument(
        "--compression-backend",
        choices=["cdo", "nccopy", "nco", "none"],
        default="nco",
        help="Compression backend to use after remapping. Defaults to nco.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="Compression level in [1, 9]. Defaults to 4.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable compression.",
    )
    parser.add_argument(
        "--out-format",
        type=str,
        default="nc4c",
        help="CDO output format passed to -f. Defaults to nc4c.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="F32",
        help="CDO output precision passed to -b. Defaults to F32.",
    )
    parser.add_argument(
        "--no-low-memory",
        action="store_true",
        help="Disable CDO low-memory mode -L.",
    )
    parser.add_argument(
        "--nccopy-chunk-spec",
        type=str,
        default=None,
        help="Chunk spec for nccopy -c, only used when compression backend is nccopy.",
    )
    parser.add_argument(
        "--nco-ppc",
        type=str,
        default=None,
        help="NCO --ppc option, only used when compression backend is nco.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Suppress command outputs.",
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
    compress = not args.no_compress
    compression_level = args.compression_level
    compression_backend = args.compression_backend
    out_format = args.out_format
    precision = args.precision
    low_memory = not args.no_low_memory
    nccopy_chunk_spec = args.nccopy_chunk_spec
    nco_ppc = args.nco_ppc
    silent = args.silent

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
            silent=silent,
            compress=compress,
            compression_level=compression_level,
            out_format=out_format,
            precision=precision,
            low_memory=low_memory,
            compression_backend=compression_backend,
            nccopy_chunk_spec=nccopy_chunk_spec,
            nco_ppc=nco_ppc
        )
        logger.info("Remapping completed successfully.")
    except Exception:
        cdo.run(
            [
                "ncks",
                "-O", "-x", "-v", "lon_vtx,lat_vtx", infile, "tmp.nc"
            ],
            silent=silent
        )
        cdo.remap_netcdf(
            infile="tmp.nc",
            outfile=outfile,
            res=res,
            use_weight=use_weight,
            silent=silent,
            compress=compress,
            compression_level=compression_level,
            out_format=out_format,
            precision=precision,
            low_memory=low_memory,
            compression_backend=compression_backend,
            nccopy_chunk_spec=nccopy_chunk_spec,
            nco_ppc=nco_ppc
        )
        cdo.run(["rm", "tmp.nc"], silent=silent)
        logger.info(
            "Remapping completed successfully after removing vertex coordinates."
        )


if __name__ == "__main__":
    main()
