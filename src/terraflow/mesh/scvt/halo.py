from __future__ import annotations
from typing import TYPE_CHECKING
from mpi4py import MPI


from terraflow.util import typing
import terraflow.util.equinox as eqx
from terraflow.mesh.abstract import AbstractHalo


if TYPE_CHECKING:
    from terraflow.util.device import AbstractDevice
    from terraflow.util.backend import AbstractBackend
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
    from terraflow.mesh.scvt.parallel_types import SCVTProc

IntType = typing.IntType
INT     = typing.Int
FLT     = typing.Float
Array   = typing.Array


class SCVTHalo(AbstractHalo):
    backend         : AbstractBackend       = eqx.field(static=True)
    comm            : MPI.Distgraphcomm = eqx.field(static=True)
    ngb             : list[IntType] = eqx.field(static=True) 
    nngb            : IntType       = eqx.field(static=True)
    full_nlev       : IntType       = eqx.field(static=True) 
    half_nlev       : IntType       = eqx.field(static=True)
    max_send_n_cell : IntType       = eqx.field(static=True)
    max_send_n_vtx  : IntType       = eqx.field(static=True)
    max_send_n_edge : IntType       = eqx.field(static=True)
    max_recv_n_cell : IntType       = eqx.field(static=True)
    max_recv_n_vtx  : IntType       = eqx.field(static=True)
    max_recv_n_edge : IntType       = eqx.field(static=True)
    
    send_n_cell     : list[IntType] = eqx.field(static=True)
    send_n_vtx      : list[IntType] = eqx.field(static=True)
    send_n_edge     : list[IntType] = eqx.field(static=True)
    send_cell_id    : INT[Array, "nngb max_send_n_cell nngb"]
    send_vtx_id     : INT[Array, "nngb max_send_n_vtx" ]
    send_edge_id    : INT[Array, "nngb max_send_n_edge"]

    recv_n_cell     : list[IntType] = eqx.field(static=True)
    recv_n_vtx      : list[IntType] = eqx.field(static=True)
    recv_n_edge     : list[IntType] = eqx.field(static=True)
    recv_cell_id    : INT[Array, "nngb max_recv_n_cell"]
    recv_vtx_id     : INT[Array, "nngb max_recv_n_vtx" ]
    recv_edge_id    : INT[Array, "nngb max_recv_n_edge"]

    send_cell_buf   : FLT[Array, "nngb half_nlev max_send_n_cell"]
    send_vtx_buf    : FLT[Array, "nngb half_nlev max_send_n_vtx "]
    send_edge_buf   : FLT[Array, "nngb half_nlev max_send_n_edge"]
    recv_cell_buf   : FLT[Array, "nngb half_nlev max_recv_n_cell"]
    recv_vtx_buf    : FLT[Array, "nngb half_nlev max_recv_n_vtx "]
    recv_edge_buf   : FLT[Array, "nngb half_nlev max_recv_n_edge"]

    def __init__(
        self,
        mesh: SCVTMesh,
        proc: SCVTProc,
        dev: AbstractDevice
    ):
        bkd = mesh.dev.backend
        max_send_n_cell = 0
        max_send_n_vtx  = 0
        max_send_n_edge = 0
        max_recv_n_cell = 0
        max_recv_n_vtx  = 0
        max_recv_n_edge = 0
        for nb in range(proc.nngb):
            send = proc.send[nb]
            recv = proc.recv[nb]
            max_send_n_cell = max(max_send_n_cell, send.n_cell)
            max_send_n_vtx  = max(max_send_n_vtx , send.n_vtx )
            max_send_n_edge = max(max_send_n_edge, send.n_edge)
            max_recv_n_cell = max(max_recv_n_cell, recv.n_cell)
            max_recv_n_vtx  = max(max_recv_n_vtx , recv.n_vtx )
            max_recv_n_edge = max(max_recv_n_edge, recv.n_edge)

        self.backend         = bkd
        self.comm            = proc.comm_graph
        self.nngb            = proc.nngb
        self.ngb             = proc.ngb
        self.full_nlev       = mesh.full_kme - mesh.full_kms + 1
        self.half_nlev       = mesh.half_kme - mesh.half_kms + 1
        
        self.max_send_n_cell = max_send_n_cell
        self.max_send_n_vtx  = max_send_n_vtx
        self.max_send_n_edge = max_send_n_edge
        self.max_recv_n_cell = max_recv_n_cell
        self.max_recv_n_vtx  = max_recv_n_vtx
        self.max_recv_n_edge = max_recv_n_edge
        
        send_n_cell     = bkd.zeros(proc.nngb, device=dev(), dtype=dev.int_type)
        send_n_vtx      = bkd.zeros(proc.nngb, device=dev(), dtype=dev.int_type)
        send_n_edge     = bkd.zeros(proc.nngb, device=dev(), dtype=dev.int_type)
        send_cell_id    = bkd.zeros((proc.nngb, self.max_send_n_cell), device=dev(), dtype=dev.int_type)
        send_vtx_id     = bkd.zeros((proc.nngb, self.max_send_n_vtx ), device=dev(), dtype=dev.int_type)
        send_edge_id    = bkd.zeros((proc.nngb, self.max_send_n_edge), device=dev(), dtype=dev.int_type)
    
        recv_n_cell     = bkd.zeros(proc.nngb, device=dev(), dtype=dev.int_type)
        recv_n_vtx      = bkd.zeros(proc.nngb, device=dev(), dtype=dev.int_type)
        recv_n_edge     = bkd.zeros(proc.nngb, device=dev(), dtype=dev.int_type)
        recv_cell_id    = bkd.zeros((proc.nngb, self.max_recv_n_cell), device=dev(), dtype=dev.int_type)
        recv_vtx_id     = bkd.zeros((proc.nngb, self.max_recv_n_vtx ), device=dev(), dtype=dev.int_type)
        recv_edge_id    = bkd.zeros((proc.nngb, self.max_recv_n_edge), device=dev(), dtype=dev.int_type)
        
        kms, kme = mesh.half_kms, mesh.half_kme
        send_cell_buf = bkd.zeros((proc.nngb, kme-kms+1, self.max_send_n_cell), device=dev(), dtype=dev.float_type)
        send_vtx_buf  = bkd.zeros((proc.nngb, kme-kms+1, self.max_send_n_vtx ), device=dev(), dtype=dev.float_type)
        send_edge_buf = bkd.zeros((proc.nngb, kme-kms+1, self.max_send_n_edge), device=dev(), dtype=dev.float_type)
        
        recv_cell_buf = bkd.zeros((proc.nngb, kme-kms+1, self.max_recv_n_cell), device=dev(), dtype=dev.float_type)
        recv_vtx_buf  = bkd.zeros((proc.nngb, kme-kms+1, self.max_recv_n_vtx ), device=dev(), dtype=dev.float_type)
        recv_edge_buf = bkd.zeros((proc.nngb, kme-kms+1, self.max_recv_n_edge), device=dev(), dtype=dev.float_type)
        
        for nb in range(proc.nngb):
            send = proc.send[nb]
            recv = proc.recv[nb]
            
            send_n_cell = send_n_cell.at[nb].set(send.n_cell)
            send_n_vtx  = send_n_vtx .at[nb].set(send.n_vtx )
            send_n_edge = send_n_edge.at[nb].set(send.n_edge)
            
            recv_n_cell = recv_n_cell.at[nb].set(recv.n_cell)
            recv_n_vtx  = recv_n_vtx .at[nb].set(recv.n_vtx )
            recv_n_edge = recv_n_edge.at[nb].set(recv.n_edge)
            
            n_cell = send_n_cell[nb]
            n_vtx  = send_n_vtx [nb]
            n_edge = send_n_edge[nb]
            send_cell_id = send_cell_id.at[nb,:n_cell].set(send.local_cell_id)
            send_vtx_id  = send_vtx_id .at[nb,:n_vtx ].set(send.local_vtx_id )
            send_edge_id = send_edge_id.at[nb,:n_edge].set(send.local_edge_id)
            
            n_cell = recv_n_cell[nb]
            n_vtx  = recv_n_vtx [nb]
            n_edge = recv_n_edge[nb]
            recv_cell_id = recv_cell_id.at[nb,:n_cell].set(recv.local_cell_id)
            recv_vtx_id  = recv_vtx_id .at[nb,:n_vtx ].set(recv.local_vtx_id )
            recv_edge_id = recv_edge_id.at[nb,:n_edge].set(recv.local_edge_id)
            
        self.send_n_cell  = send_n_cell.tolist()
        self.send_n_vtx   = send_n_vtx .tolist()
        self.send_n_edge  = send_n_edge.tolist()
        self.send_cell_id = send_cell_id
        self.send_vtx_id  = send_vtx_id
        self.send_edge_id = send_edge_id
        
        self.recv_n_cell  = recv_n_cell.tolist()
        self.recv_n_vtx   = recv_n_vtx .tolist()
        self.recv_n_edge  = recv_n_edge.tolist()
        self.recv_cell_id = recv_cell_id
        self.recv_vtx_id  = recv_vtx_id
        self.recv_edge_id = recv_edge_id
        
        self.send_cell_buf = send_cell_buf
        self.send_vtx_buf  = send_vtx_buf
        self.send_edge_buf = send_edge_buf
        
        self.recv_cell_buf = recv_cell_buf
        self.recv_vtx_buf  = recv_vtx_buf
        self.recv_edge_buf = recv_edge_buf
