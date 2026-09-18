from __future__ import annotations

import socket
from typing import Any


class SerialGroup:
    pass


class SerialRequest:
    def Wait(self) -> None:
        return None


class SerialComm:
    rank = 0
    size = 1

    def __init__(self) -> None:
        self.group = SerialGroup()

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return self.size

    def Barrier(self) -> None:
        return None

    def Split(self, color: int, key: int) -> "SerialComm":
        return self

    def Split_type(self, split_type: int) -> "SerialComm":
        return self

    def Free(self) -> None:
        return None

    def Allreduce(self, sendbuf: Any, recvbuf: Any, op: Any = None) -> None:
        recvbuf[...] = sendbuf
        return None

    def allreduce(self, sendobj: Any, op: Any = None) -> Any:
        return sendobj

    def bcast(self, obj: Any, root: int = 0) -> Any:
        if root != 0:
            raise ValueError("Serial communicator only supports root=0.")
        return obj

    def gather(self, sendobj: Any, root: int = 0) -> list[Any] | None:
        if root != 0:
            raise ValueError("Serial communicator only supports root=0.")
        return [sendobj]

    def allgather(self, sendobj: Any) -> list[Any]:
        return [sendobj]

    def alltoall(self, sendobj: Any) -> Any:
        if isinstance(sendobj, list) and len(sendobj) == 1:
            return [sendobj[0]]
        return sendobj

    def Create_dist_graph_adjacent(
        self,
        sources: list[int],
        destinations: list[int],
        reorder: bool = False,
    ) -> "SerialDistgraphcomm":
        return SerialDistgraphcomm()


class SerialNullComm(SerialComm):
    rank = -1
    size = 0


class SerialDistgraphcomm(SerialComm):
    pass


class SerialMPI:
    Comm          = SerialComm
    Intracomm     = SerialComm
    Distgraphcomm = SerialDistgraphcomm
    Group         = SerialGroup
    Request       = SerialRequest

    COMM_WORLD   = SerialComm()
    COMM_NULL    = SerialNullComm()
    GROUP_NULL   = SerialGroup()
    REQUEST_NULL = SerialRequest()

    SUM  = "sum"
    MAX  = "max"
    MIN  = "min"
    PROD = "prod"

    PROC_NULL         = -1
    THREAD_SERIALIZED = 0
    COMM_TYPE_SHARED  = 1

    @staticmethod
    def Is_initialized() -> bool:
        return True

    @staticmethod
    def Init_thread(required: int | None = None) -> tuple[int, int] | int:
        return required if required is not None else SerialMPI.THREAD_SERIALIZED

    @staticmethod
    def Get_processor_name() -> str:
        return socket.gethostname()


MPI = SerialMPI()
