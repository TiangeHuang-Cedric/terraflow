#!/usr/bin/env python3

"""Small helper to convert between .ipynb and .py using jupytext.

Usage:
  python jupytext_convert.py --from path/to/file.ipynb --to path/to/file.py
  python jupytext_convert.py --from notebook.ipynb   # will create notebook.py

If jupytext is installed it will use its Python API; otherwise it will try
to call the `jupytext` CLI. If neither is available the script exits with a
helpful message.

This script chooses sensible defaults:
- when producing .py it uses the percent format (cells marked with # %% )
- when producing .ipynb it writes the standard notebook format

"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Optional, Any
from loguru import logger


def get_destination_format(path: str) -> str:
    if path.endswith("py"):
        return "ipynb"
    return "py:percent"


def convert_with_cli(args) -> None:
    notebooks = args.notebooks
    fmt = get_destination_format(notebooks)
    kwargs = parse_jupytext_options(args)
    if "--output" in kwargs:
        out_file = kwargs["--output"]
    else:
        out_file = os.path.splitext(notebooks)[0] + "." + fmt.split(":")[0]
    if os.path.exists(out_file) and not args.force:
        raise RuntimeError(
            f"Output file {out_file} already exists. Use --force to overwrite."
        )
    if hasattr(args, "sync"):
        cmd = ["jupytext", notebooks]
    else:
        cmd = ["jupytext", "--to", fmt, notebooks]
    for k, v in kwargs.items():
        if v == "":
            cmd += [k]
        else:
            cmd += [k, v]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        raise
    except FileNotFoundError as exc:
        raise RuntimeError("jupytext CLI not found") from exc


def command_line_interface(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert between .ipynb and .py using jupytext"
    )
    p.add_argument(
        dest="notebooks",
        type=str,
        help="One or more notebook(s). Notebook is read from stdin when this argument is empty. (default: None)"
    )
    p.add_argument(
        "--output",
        "-o",
        dest="OUTPUT",
        type=str,
        required=False,
        help="Destination file. "
             "Defaults to the original file, with prefix/suffix/extension changed according to the destination format. "
             "Use '-' to print the notebook on stdout. (default: None)",
    )
    p.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing destination without asking"
    )
    p.add_argument(
        "--update",
        required=False,
        default=False,
        action="store_true",
        help="Preserve the output cells when the destination notebook is an .ipynb file that already exists (default: False)"
    )
    p.add_argument(
        "--sync",
        "-s",
        required=False,
        default=False,
        action="store_true",
        help="Synchronize the content of the paired representations of the given notebook. "
             "Input cells are taken from the file that was last modified, "
             "and outputs are read from the ipynb file, if present. (default: False)"
    )
    return p.parse_args(argv)


def parse_jupytext_options(args: argparse.Namespace) -> dict[str, Any]:
    options = {}
    if args.OUTPUT:
        options["--output"] = args.OUTPUT
    if args.update:
        options["--update"] = ""
    if args.sync:
        options["--sync"] = ""
    return options


def batch_sync_directory(base_dir: str = "examples/notebook") -> int:
    """For each .py file under base_dir, run jupytext --set-formats py,ipynb <file>
    and then jupytext <file> --sync.

    Returns 0 if all succeeded, otherwise returns number of failures.
    """
    if not os.path.isdir(base_dir):
        logger.error("Base directory not found: {}", base_dir)
        return 1

    failures = 0
    processed = 0
    for root, _dirs, files in os.walk(base_dir):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            py_path = os.path.join(root, fn)
            # ensure a corresponding .ipynb exists; if not, create one
            ipynb_path = os.path.splitext(py_path)[0] + ".ipynb"
            if not os.path.exists(ipynb_path):
                try:
                    logger.info(
                        "No corresponding ipynb for {}; creating {}", py_path, ipynb_path
                    )
                    subprocess.check_call(
                        ["jupytext", "--to", "ipynb", py_path])

                except FileNotFoundError:
                    logger.error(
                        "jupytext CLI not found. Please install jupytext."
                    )
                    return 2
                except subprocess.CalledProcessError as e:
                    logger.error(
                        "Failed to create ipynb for {}: {}", py_path, e)
                    failures += 1
                    processed += 1
                    continue

            # run set-formats
            try:
                logger.info("Setting formats for {}", py_path)
                subprocess.check_call(
                    ["jupytext", "--set-formats", "py,ipynb", py_path])
            except FileNotFoundError:
                logger.error(
                    "jupytext CLI not found. Please install jupytext.")
                return 2
            except subprocess.CalledProcessError as e:
                logger.error("Failed to set formats for {}: {}", py_path, e)
                failures += 1
                processed += 1
                continue

            # run sync
            try:
                logger.info("Syncing {}", py_path)
                subprocess.check_call(
                    ["jupytext", py_path, "--update", "--sync"])
                processed += 1
            except subprocess.CalledProcessError as e:
                logger.error("Failed to sync {}: {}", py_path, e)
                failures += 1
            except FileNotFoundError:
                logger.error(
                    "jupytext CLI not found. Please install jupytext.")
                return 2

    logger.info("Processed {} files, {} failures", processed, failures)
    return 0 if failures == 0 else failures


def main(argv: Optional[list[str]] = None) -> int:
    # If called with no argv (normal script invocation) and no CLI args provided,
    # run the batch job for examples/notebook as requested.
    if argv is None and len(sys.argv) == 1:
        logger.info(
            "No arguments provided; running batch set-formats and sync for examples/notebook"
        )
        return batch_sync_directory("examples/notebook")

    args = command_line_interface(argv)

    # Try CLI
    try:
        convert_with_cli(args)
        logger.success(f"Converted {args.notebooks} using jupytext CLI.")
        return 0
    except Exception as e:
        logger.error(f"Failed to convert using jupytext CLI: {e}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
