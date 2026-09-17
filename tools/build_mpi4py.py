"""Build mpi4py from source against the selected MPI C compiler."""

from __future__ import annotations
import os
import sys

from build_common import require_command, run


def import_probe(env: dict[str, str]) -> None:
    run(
        [
            sys.executable,
            "-c",
            (
                "import mpi4py.MPI as MPI; "
                "print('mpi4py import probe:', "
                "'version=', MPI.Get_version(), "
                "'rank=', MPI.COMM_WORLD.Get_rank(), "
                "'size=', MPI.COMM_WORLD.Get_size())"
            ),
        ],
        env,
    )


def main() -> None:
    env = os.environ.copy()
    mpicc = require_command(("MPICC",), "mpicc")
    mpi4py_spec = os.environ.get("MPI4PY_SPEC", "mpi4py>=4.1.2,<5")

    build_env = env.copy()
    build_env["MPICC"] = mpicc

    run([sys.executable, "-m", "pip", "uninstall", "-y", "mpi4py"], env)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-binary=mpi4py",
            "--force-reinstall",
            mpi4py_spec,
        ],
        build_env,
    )
    import_probe(build_env)


if __name__ == "__main__":
    main()
