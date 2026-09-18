from __future__ import annotations
from importlib import import_module
import os
from typing import Any, Optional, TypeAlias
from dataclasses import dataclass, field


from terraflow.util.serial_proc import MPI as SERIAL_MPI


def is_env_gt_one(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    try:
        return int(value) > 1
    except ValueError:
        return False


def is_mpi_launcher() -> bool:
    if any(
        is_env_gt_one(name)
        for name in (
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
            "PMIX_SIZE",
            "MV2_COMM_WORLD_SIZE",
            "SLURM_NTASKS",
        )
    ):
        return True
    return any(
        name in os.environ
        for name in (
            "OMPI_COMM_WORLD_RANK",
            "PMI_RANK",
            "PMIX_RANK",
            "MV2_COMM_WORLD_RANK",
            "SLURM_PROCID",
        )
    )


class MPIProxy:
    def __init__(self) -> None:
        self._impl: Any = SERIAL_MPI
        self._using_mpi4py = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)

    def use_mpi4py(self) -> Any:
        if not self._using_mpi4py:
            self._impl = import_module("mpi4py.MPI")
            self._using_mpi4py = True
        return self._impl

    def use_serial(self) -> Any:
        if not self._using_mpi4py:
            self._impl = SERIAL_MPI
        return self._impl

    def is_mpi4py(self) -> bool:
        return self._using_mpi4py


MPI = MPIProxy()


def configure_mpi(*, comm: Any | None, use_mpi: bool | None) -> None:
    if use_mpi is True or comm is not None:
        MPI.use_mpi4py()
        return
    if use_mpi is False:
        MPI.use_serial()
        return
    if is_mpi_launcher():
        MPI.use_mpi4py()
    else:
        MPI.use_serial()

MPIComm         : TypeAlias = Any
MPIGroup        : TypeAlias = Any
MPIRequest      : TypeAlias = Any
MPIDistgraphComm: TypeAlias = Any


PROC_TYPE_MODEL = 1
PROC_TYPE_IO    = 2


@dataclass
class Proc:
    # Top MPI communicator
    comm      : MPIComm  = field(default_factory=lambda: MPI.COMM_NULL)
    # MPI communicator for model computation
    comm_model: MPIComm  = field(default_factory=lambda: MPI.COMM_NULL)
    # MPI communicator for IO
    comm_io   : MPIComm  = field(default_factory=lambda: MPI.COMM_NULL)
    group     : MPIGroup = field(default_factory=lambda: MPI.GROUP_NULL)
    # MPI process ID for model computation
    id_model  : int       = MPI.PROC_NULL
    # MPI process ID for IO
    id_io     : int       = MPI.PROC_NULL
    # Number of model processes
    np_model  : int       = 0
    # Number of IO processes
    np_io     : int       = 0
    # Process type (model, io, etc.)
    type      : int       = PROC_TYPE_MODEL
    # Nest domain index (root domain is 1)
    idom      : int       = 1
    hostname  : str       = ""

    def __init__(
        self,
        *,
        proc: Optional["Proc"] = None,
        comm: MPIComm | None = None,
        use_mpi: bool | None = None,
        use_async_io: bool = False,
        proc_io_stride: int = 0,
    ):
        if proc is not None:
            self.comm       = proc.comm
            self.comm_model = proc.comm_model
            self.comm_io    = proc.comm_io
            self.group      = proc.group
            self.id_model   = proc.id_model
            self.id_io      = proc.id_io
            self.np_model   = proc.np_model
            self.np_io      = proc.np_io
            self.type       = proc.type
            self.idom       = proc.idom
            self.hostname   = proc.hostname
        else:
            configure_mpi(comm=comm, use_mpi=use_mpi)
            if comm is not None:
                self.comm = comm
            else:
                if not MPI.Is_initialized():
                    MPI.Init_thread(required=MPI.THREAD_SERIALIZED)
                self.comm = MPI.COMM_WORLD

            if use_async_io:
                key = self.comm.rank
                color = key % proc_io_stride
                new_comm = self.comm.Split(color, key)
                self.comm_io = self.comm
                if color == 0:
                    self.type = PROC_TYPE_IO
                else:
                    self.comm_model = new_comm
                    self.type = PROC_TYPE_MODEL
            else:
                self.comm_model = self.comm
                self.comm_io = self.comm
                self.type = PROC_TYPE_MODEL

            if self.type == PROC_TYPE_MODEL:
                self.group = self.comm_model.group
                self.np_model = self.comm_model.size
                self.id_model = self.comm_model.rank
                self.hostname = MPI.Get_processor_name()

            self.np_io = self.comm_io.size
            self.id_io = self.comm_io.rank

    def is_root(self) -> bool:
        return self.id_model == 0

    def is_model(self) -> bool:
        return self.type == PROC_TYPE_MODEL

    def is_io(self) -> bool:
        return self.type == PROC_TYPE_IO
