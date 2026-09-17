"""Install or validate ESMPy against an externally built ESMF."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import sys

from build_common import run


def require_file(value: str | None, name: str) -> Path:
    if not value:
        raise SystemExit(f"{name} is required.")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"{name} does not exist or is not a file: {path}")
    return path


def require_esmpy_source() -> Path:
    source_value = os.environ.get("ESMPY_SOURCE")
    if not source_value:
        raise SystemExit(
            "ESMPY_SOURCE is required when ESMF_DIR is not set. "
            "Point it at the ESMF source tree's src/addon/esmpy directory."
        )
    source = Path(source_value).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"ESMPy source directory does not exist: {source}")
    if not (source / "setup.py").is_file() and not (
        source / "pyproject.toml"
    ).is_file():
        raise SystemExit(f"ESMPy source directory is not installable: {source}")
    return source


def import_probe(env: dict[str, str]) -> None:
    run(
        [
            sys.executable,
            "-c",
            (
                "import os, esmpy; "
                "m = esmpy.Manager(debug=False); "
                "print('ESMPy import probe:', "
                "'ESMFMKFILE=', os.environ.get('ESMFMKFILE'), "
                "'local_pet=', m.local_pet, 'pet_count=', m.pet_count)"
            ),
        ],
        env,
    )


def mpi_probe(env: dict[str, str]) -> None:
    mpiexec = os.environ.get("MPIEXEC") or shutil.which("mpiexec")
    if not mpiexec:
        raise SystemExit("mpiexec was not found. Set MPIEXEC or load your MPI module.")

    nproc = os.environ.get("ESMPY_MPI_PROCS", "2")
    run(
        [
            mpiexec,
            "-n",
            nproc,
            sys.executable,
            "-c",
            (
                "import os, esmpy; "
                "m = esmpy.Manager(debug=False); "
                "print('ESMPy MPI probe:', os.getpid(), "
                "'local_pet=', m.local_pet, 'pet_count=', m.pet_count, flush=True)"
            ),
        ],
        env,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install or validate ESMPy against an externally built ESMF."
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--check-mpi", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    esmfmfile = require_file(env.get("ESMFMKFILE"), "ESMFMKFILE")
    env["ESMFMKFILE"] = str(esmfmfile)

    if not args.check_only:
        source = require_esmpy_source()
        install_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            "--no-deps",
        ]
        if env.get("ESMPY_BUILD_ISOLATION", "0") not in {
            "1",
            "true",
            "TRUE",
            "yes",
            "YES",
        }:
            install_cmd.append("--no-build-isolation")
        install_cmd.append(str(source))

        run([sys.executable, "-m", "pip", "uninstall", "-y", "esmpy", "ESMF"], env)
        run(install_cmd, env)

    import_probe(env)
    if args.check_mpi:
        mpi_probe(env)


if __name__ == "__main__":
    main()
