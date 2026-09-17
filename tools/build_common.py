"""Small helpers shared by the local build scripts."""

from __future__ import annotations
import os
import shutil
import subprocess


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def require_command(env_names: tuple[str, ...], fallback_name: str) -> str:
    for name in env_names:
        value = os.environ.get(name)
        if value:
            return value

    value = shutil.which(fallback_name)
    if value:
        return value

    joined = " or ".join(env_names)
    raise SystemExit(
        f"{fallback_name} was not found. Set {joined} or load your MPI module."
    )
