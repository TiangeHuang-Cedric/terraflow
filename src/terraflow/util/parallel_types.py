from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field
from mpi4py import MPI

PROC_TYPE_MODEL = 1
PROC_TYPE_IO = 2


@dataclass
class Proc:
    # Top MPI communicator
    comm      : MPI.Comm  = field(default_factory=lambda: MPI.COMM_NULL)
    # MPI communicator for model computation
    comm_model: MPI.Comm  = field(default_factory=lambda: MPI.COMM_NULL)
    # MPI communicator for IO
    comm_io   : MPI.Comm  = field(default_factory=lambda: MPI.COMM_NULL)
    group     : MPI.Group = field(default_factory=lambda: MPI.GROUP_NULL)
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
        comm: MPI.Comm | None = None,
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
