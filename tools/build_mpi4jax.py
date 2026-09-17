"""Build mpi4jax from source against an existing mpi4py installation."""

from __future__ import annotations
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from build_common import require_command, run


def output(command: list[str], env: dict[str, str]) -> str:
    return subprocess.check_output(command, env=env, text=True).strip()


def mpi_wrapper_compiler(wrapper: str, env: dict[str, str]) -> str:
    for args in (["--showme:command"], ["-show"]):
        try:
            shown = output([wrapper, *args], env)
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        parts = shlex.split(shown)
        if parts:
            return parts[0]

    return wrapper


def gcc_libstdcxx_dir(mpicxx: str, env: dict[str, str]) -> Path | None:
    compiler = mpi_wrapper_compiler(mpicxx, env)
    try:
        libstdcxx = output([compiler, "-print-file-name=libstdc++.6.dylib"], env)
    except (subprocess.CalledProcessError, FileNotFoundError):
        libstdcxx = ""

    path = Path(libstdcxx).expanduser()
    if path.is_file():
        return path.resolve().parent

    candidate = (Path(compiler).expanduser().resolve().parent / ".." / "lib").resolve()
    if (candidate / "libstdc++.6.dylib").is_file():
        return candidate

    return None


def prepend_path(env: dict[str, str], name: str, path: Path) -> None:
    current = env.get(name)
    env[name] = f"{path}{os.pathsep}{current}" if current else str(path)


def configure_mpi4jax_build_env(env: dict[str, str], mpicxx: str) -> dict[str, str]:
    build_env = env.copy()
    build_env["MPICC"] = mpicxx
    build_env["CC"] = mpicxx
    build_env["CXX"] = mpicxx

    if sys.platform == "darwin":
        lib_dir = gcc_libstdcxx_dir(mpicxx, build_env)
        if lib_dir is not None:
            prepend_path(build_env, "DYLD_LIBRARY_PATH", lib_dir)
            ldflags = build_env.get("LDFLAGS", "")
            build_env["LDFLAGS"] = (
                f"{ldflags} -L{lib_dir} -Wl,-rpath,{lib_dir} -lstdc++"
            ).strip()
            print(f"Using GCC C++ runtime: {lib_dir}", flush=True)
        else:
            print(
                "Warning: libstdc++.6.dylib was not found from mpicxx; "
                "mpi4jax may fail to import on macOS.",
                flush=True,
            )

    return build_env


def import_probe(env: dict[str, str]) -> None:
    run(
        [
            sys.executable,
            "-c",
            (
                "import mpi4py.MPI as MPI, mpi4jax; "
                "print('mpi4jax import probe:', "
                "'mpi4jax=', mpi4jax.__version__, "
                "'rank=', MPI.COMM_WORLD.Get_rank(), "
                "'size=', MPI.COMM_WORLD.Get_size())"
            ),
        ],
        env,
    )


def otool_probe(env: dict[str, str]) -> None:
    if sys.platform != "darwin" or not shutil.which("otool"):
        return

    code = (
        "import pathlib, sys; "
        "site = pathlib.Path(sys.executable).parents[1] / 'lib' / "
        "f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'; "
        "print(next(site.glob('mpi4jax/_src/xla_bridge/mpi_xla_bridge_cpu*.so')))"
    )
    shared_object = output([sys.executable, "-c", code], env)
    run(["otool", "-L", shared_object], env)


def main() -> None:
    env = os.environ.copy()
    mpicxx = require_command(("MPICXX", "MPI_CXX"), "mpicxx")

    mpi4jax_spec = os.environ.get("MPI4JAX_SPEC", "mpi4jax==0.9.1")
    mpi4jax_index = os.environ.get("MPI4JAX_PIP_INDEX_URL", "https://pypi.org/simple")

    run([sys.executable, "-m", "pip", "uninstall", "-y", "mpi4jax"], env)

    mpi4jax_env = configure_mpi4jax_build_env(env, mpicxx)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--index-url",
            mpi4jax_index,
            "--no-deps",
            "--no-build-isolation",
            "--no-cache-dir",
            "--no-binary=mpi4jax",
            "--force-reinstall",
            mpi4jax_spec,
        ],
        mpi4jax_env,
    )

    import_probe(mpi4jax_env)
    otool_probe(mpi4jax_env)


if __name__ == "__main__":
    main()
