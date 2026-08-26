from dataclasses import dataclass, field
from mpi4py import MPI
import numpy as np
import jax.numpy as jnp

from terraflow.util.typing import Array
from terraflow.util.parallel_types import Proc


@dataclass
class Domain:
    name   : str = ''
    n_cell : int = 0
    n_vtx  : int = 0
    n_edge : int = 0
    cell_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    vtx_id : np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    edge_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))


@dataclass
class CommonSite:
    n_cell        : int = 0
    n_vtx         : int = 0
    n_edge        : int = 0
    req_n_cell    : MPI.Request = field(default_factory=lambda: MPI.REQUEST_NULL)
    req_n_vtx     : MPI.Request = field(default_factory=lambda: MPI.REQUEST_NULL)
    req_n_edge    : MPI.Request = field(default_factory=lambda: MPI.REQUEST_NULL)
    cell_buf      : Array = field(default_factory=lambda: jnp.empty(0, dtype=jnp.int32))
    vtx_buf       : Array = field(default_factory=lambda: jnp.empty(0, dtype=jnp.int32))
    edge_buf      : Array = field(default_factory=lambda: jnp.empty(0, dtype=jnp.int32))
    global_cell_id: Array = field(default_factory=lambda: jnp.empty(0, dtype=jnp.int32))
    global_vtx_id : Array = field(default_factory=lambda: jnp.empty(0, dtype=jnp.int32))
    global_edge_id: Array = field(default_factory=lambda: jnp.empty(0, dtype=jnp.int32))
    local_cell_id : Array = field(default_factory=lambda: jnp.empty(0, dtype=jnp.int32))
    local_vtx_id  : Array = field(default_factory=lambda: jnp.empty(0, dtype=jnp.int32))
    local_edge_id : Array = field(default_factory=lambda: jnp.empty(0, dtype=jnp.int32))


@dataclass
class SCVTProc(Proc):
    comm_graph     : MPI.Distgraphcomm = field(default_factory=lambda: MPI.Distgraphcomm())
    indegree       : int               = 0
    outdegree      : int               = 0
    n_cell         : int               = 0
    n_vtx          : int               = 0
    n_edge         : int               = 0
    n_cell_inner   : int               = 0
    n_vtx_inner    : int               = 0
    n_edge_inner   : int               = 0
    n_cell_cmpt    : int               = 0
    n_vtx_cmpt     : int               = 0
    n_edge_cmpt    : int               = 0
    n_cell_bdry    : list              = field(default_factory=list)
    n_vtx_bdry     : list              = field(default_factory=list)
    n_edge_bdry    : list              = field(default_factory=list)
    n_cell_halo    : list              = field(default_factory=list)
    n_vtx_halo     : list              = field(default_factory=list)
    n_edge_halo    : list              = field(default_factory=list)
    hori_hw        : int               = 0
    nngb           : int               = 0
    ngb            : list              = field(default_factory=list)
    recv_nb        : list              = field(default_factory=list)
    send_nb        : list              = field(default_factory=list)
    send           : list[CommonSite]  = field(default_factory=list)
    recv           : list[CommonSite]  = field(default_factory=list)
    cell_proc_idmap: np.ndarray        = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    g2l_cell_id    : np.ndarray        = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    l2g_cell_id    : np.ndarray        = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    g2l_vtx_id     : np.ndarray        = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    l2g_vtx_id     : np.ndarray        = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    g2l_edge_id    : np.ndarray        = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    l2g_edge_id    : np.ndarray        = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    inner          : Domain            = field(default_factory=Domain)
    bdry           : list[Domain]      = field(default_factory=list)
    cmpt           : Domain            = field(default_factory=Domain)
    halo           : list[Domain]      = field(default_factory=list)

    def __init__(
        self,
        *,
        proc: Proc = Proc(),
        comm: MPI.Comm | None = None,
        hori_hw: int = 2,
    ):
        super().__init__(
            comm=comm,
            proc=proc,
        )
        self.comm_graph = MPI.Distgraphcomm()
        self.hori_hw = hori_hw
        self.indegree = 0
        self.outdegree = 0
        self.nngb = 0
        self.ngb = []
        self.recv_nb = []
        self.send_nb = []
        self.recv = []
        self.send = []
        self.n_cell_bdry = [0] * hori_hw
        self.n_vtx_bdry = [0] * hori_hw
        self.n_edge_bdry = [0] * hori_hw
        self.n_cell_halo = [0] * hori_hw
        self.n_vtx_halo = [0] * hori_hw
        self.n_edge_halo = [0] * hori_hw
        self.inner = Domain(name="inner")
        self.cmpt = Domain(name="cmpt")
        self.bdry = [
            Domain(name=f"bdry{layer}")
            for layer in range(1, hori_hw + 1)
        ]
        self.halo = [
            Domain(name=f"halo{layer}")
            for layer in range(1, hori_hw + 1)
        ]
