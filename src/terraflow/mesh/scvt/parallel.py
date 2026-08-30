from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
import mpi4jax
from mpi4py import MPI

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.mesh.scvt.field import SCVTFieldMeta
    from terraflow.mesh.scvt.halo import SCVTHalo
    
    FLT   = typing.Float
    Array = typing.Array

TAGS = {"cell": 1, "vtx": 2, "edge": 3, "lev": 1, "lev_vtx": 2, "lev_edge": 3}


def global_sum(comm: MPI.Comm, x: Array) -> Array:
    if comm is None or comm == MPI.COMM_NULL or comm.size == 1:
        return x

    return mpi4jax.allreduce(x, op=MPI.SUM, comm=comm)


def fill_field1d_array_halo(
    halo: SCVTHalo,
    f: FLT[Array, "npoint"],
    meta: SCVTFieldMeta
) -> Array:
    loc = meta.loc
    tag = TAGS[loc]

    for inb in range(halo.nngb):
        nb = halo.ngb[inb]
        if loc == "cell" or loc == "lev":
            n_cell   = halo.send_n_cell  [inb            ]
            send_idx = halo.send_cell_id [inb,    :n_cell]
            send_buf = halo.send_cell_buf[inb, 0, :n_cell]
        elif loc == "vtx" or loc == "lev_vtx":
            n_vtx    = halo.send_n_vtx   [inb            ]
            send_idx = halo.send_vtx_id  [inb,    :n_vtx ]
            send_buf = halo.send_vtx_buf [inb, 0, :n_vtx ]
        elif loc == "edge" or loc == "lev_edge":
            n_edge   = halo.send_n_edge  [inb]
            send_idx = halo.send_edge_id [inb,    :n_edge]
            send_buf = halo.send_edge_buf[inb, 0, :n_edge]
        else:
            raise ValueError(f"Unsupported location ({loc}) for field: {meta.name}!")

        if loc == "cell" or loc == "lev":
            n_cell   = halo.recv_n_cell  [inb            ]
            recv_idx = halo.recv_cell_id [inb,    :n_cell]
            recv_buf = halo.recv_cell_buf[inb, 0, :n_cell]
        elif loc == "vtx" or loc == "lev_vtx":
            n_vtx    = halo.recv_n_vtx   [inb            ]
            recv_idx = halo.recv_vtx_id  [inb,    :n_vtx ]
            recv_buf = halo.recv_vtx_buf [inb, 0, :n_vtx ]
        elif loc == "edge" or loc == "lev_edge":
            n_edge   = halo.recv_n_edge  [inb            ]
            recv_idx = halo.recv_edge_id [inb,    :n_edge]
            recv_buf = halo.recv_edge_buf[inb, 0, :n_edge]
        else:
            raise ValueError(f"Unsupported location ({loc}) for field: {meta.name}!")

        arr = f[send_idx]
        send_buf = send_buf.at[...].set(arr)
        recv_buf = mpi4jax.sendrecv(
            send_buf,
            recv_buf,
            source=nb,
            dest=nb,
            sendtag=tag,
            recvtag=tag,
            comm=halo.comm,
        )

        f = f.at[recv_idx].set(recv_buf)

    return f


def fill_field2d_array_halo(
    halo: SCVTHalo,
    f: FLT[Array, "nlev npoint"],
    meta: SCVTFieldMeta
) -> Array:
    loc = meta.loc
    tag = TAGS[loc]

    for inb in range(halo.nngb):
        nb = halo.ngb[inb]
        if loc == "cell" or loc == "lev":
            n_cell   = halo.send_n_cell[inb]
            n_lev = halo.full_nlev if loc == "cell" else halo.half_nlev
            send_idx = halo.send_cell_id [inb,         :n_cell]
            send_buf = halo.send_cell_buf[inb, :n_lev, :n_cell]
        elif loc == "vtx" or loc == "lev_vtx":
            n_vtx    = halo.send_n_vtx[inb]
            n_lev = halo.full_nlev if loc == "vtx"  else halo.half_nlev
            send_idx = halo.send_vtx_id  [inb,         :n_vtx ]
            send_buf = halo.send_vtx_buf [inb, :n_lev, :n_vtx ]
        elif loc == "edge" or loc == "lev_edge":
            n_edge   = halo.send_n_edge[inb]
            n_lev = halo.full_nlev if loc == "edge" else halo.half_nlev
            send_idx = halo.send_edge_id [inb,         :n_edge]
            send_buf = halo.send_edge_buf[inb, :n_lev, :n_edge]
        else:
            raise ValueError(f"Unsupported location ({loc}) for field: {meta.name}!")

        if loc == "cell" or loc == "lev":
            n_cell   = halo.recv_n_cell[inb]
            n_lev = halo.full_nlev if loc == "cell" else halo.half_nlev
            recv_idx = halo.recv_cell_id [inb,         :n_cell]
            recv_buf = halo.recv_cell_buf[inb, :n_lev, :n_cell]
        elif loc == "vtx" or loc == "lev_vtx":
            n_vtx    = halo.recv_n_vtx[inb]
            n_lev = halo.full_nlev if loc == "vtx"  else halo.half_nlev
            recv_idx = halo.recv_vtx_id  [inb,         :n_vtx ]
            recv_buf = halo.recv_vtx_buf [inb, :n_lev, :n_vtx ]
        elif loc == "edge" or loc == "lev_edge":
            n_edge   = halo.recv_n_edge[inb]
            n_lev = halo.full_nlev if loc == "edge" else halo.half_nlev
            recv_idx = halo.recv_edge_id [inb,         :n_edge]
            recv_buf = halo.recv_edge_buf[inb, :n_lev, :n_edge]
        else:
            raise ValueError(f"Unsupported location ({loc}) for field: {meta.name}!")

        arr = f[:n_lev, :][:, send_idx]
        send_buf = send_buf.at[...].set(arr)
        recv_buf = mpi4jax.sendrecv(
            send_buf,
            recv_buf,
            source=nb,
            dest=nb,
            sendtag=tag,
            recvtag=tag,
            comm=halo.comm,
        )

        f_lev = f[:n_lev, :]
        f_lev = f_lev.at[:, recv_idx].set(recv_buf)
        f = f.at[:n_lev, :].set(f_lev)

    return f


def fill_field3d_array_halo(
    halo: SCVTHalo,
    f: FLT[Array, "ndim0, nlev npoint"],
    i0: int,
    meta: SCVTFieldMeta
) -> Array:
    loc = meta.loc
    tag = TAGS[loc]

    for inb in range(halo.nngb):
        nb = halo.ngb[inb]
        if loc == "cell" or loc == "lev":
            n_cell   = halo.send_n_cell[inb]
            n_lev = halo.full_nlev if loc == "cell" else halo.half_nlev
            send_idx = halo.send_cell_id [inb,         :n_cell]
            send_buf = halo.send_cell_buf[inb, :n_lev, :n_cell]
        elif loc == "vtx" or loc == "lev_vtx":
            n_vtx    = halo.send_n_vtx[inb]
            n_lev = halo.full_nlev if loc == "vtx"  else halo.half_nlev
            send_idx = halo.send_vtx_id  [inb,         :n_vtx ]
            send_buf = halo.send_vtx_buf [inb, :n_lev, :n_vtx ]
        elif loc == "edge" or loc == "lev_edge":
            n_edge   = halo.send_n_edge[inb]
            n_lev = halo.full_nlev if loc == "edge" else halo.half_nlev
            send_idx = halo.send_edge_id [inb,         :n_edge]
            send_buf = halo.send_edge_buf[inb, :n_lev, :n_edge]
        else:
            raise ValueError(f"Unsupported location ({loc}) for field: {meta.name}!")

        if loc == "cell" or loc == "lev":
            n_cell   = halo.recv_n_cell[inb]
            n_lev    = halo.full_nlev if loc == "cell" else halo.half_nlev
            recv_idx = halo.recv_cell_id [inb,         :n_cell]
            recv_buf = halo.recv_cell_buf[inb, :n_lev, :n_cell]
        elif loc == "vtx" or loc == "lev_vtx":
            n_vtx    = halo.recv_n_vtx[inb]
            n_lev    = halo.full_nlev if loc == "vtx"  else halo.half_nlev
            recv_idx = halo.recv_vtx_id  [inb,         :n_vtx ]
            recv_buf = halo.recv_vtx_buf [inb, :n_lev, :n_vtx ]
        elif loc == "edge" or loc == "lev_edge":
            n_edge   = halo.recv_n_edge[inb]
            n_lev    = halo.full_nlev if loc == "edge" else halo.half_nlev
            recv_idx = halo.recv_edge_id [inb,         :n_edge]
            recv_buf = halo.recv_edge_buf[inb, :n_lev, :n_edge]
        else:
            raise ValueError(f"Unsupported location ({loc}) for field: {meta.name}!")

        arr = f[i0, :n_lev, :][:, send_idx]
        send_buf = send_buf.at[...].set(arr)
        recv_buf = mpi4jax.sendrecv(
            send_buf,
            recv_buf,
            source=nb,
            dest=nb,
            sendtag=tag,
            recvtag=tag,
            comm=halo.comm,
        )

        f_i0 = f[i0]
        f_i0 = f_i0.at[:n_lev, recv_idx].set(recv_buf)
        f = f.at[i0].set(f_i0)

    return f


def fill_raw_array_halo(
    halo: SCVTHalo,
    f: Array,
    loc: str
) -> Array:
    tag = TAGS[loc]
    f_shape = f.shape

    for inb in range(halo.nngb):
        nb = halo.ngb[inb]
        if loc == "cell" or loc == "lev":
            n_cell   = halo.send_n_cell[inb]
            send_idx = halo.send_cell_id [inb, :n_cell]
            send_buf = jnp.zeros((n_cell,) + f_shape[1:], dtype=f.dtype)
        elif loc == "vtx" or loc == "lev_vtx":
            n_vtx    = halo.send_n_vtx[inb]
            send_idx = halo.send_vtx_id  [inb, :n_vtx ]
            send_buf = jnp.zeros((n_vtx, ) + f_shape[1:], dtype=f.dtype)
        elif loc == "edge" or loc == "lev_edge":
            n_edge   = halo.send_n_edge[inb]
            send_idx = halo.send_edge_id [inb, :n_edge]
            send_buf = jnp.zeros((n_edge,) + f_shape[1:], dtype=f.dtype)
        else:
            raise ValueError(f"Unsupported location ({loc}) for array!")

        if loc == "cell" or loc == "lev":
            n_cell   = halo.recv_n_cell[inb]
            recv_idx = halo.recv_cell_id [inb, :n_cell]
            recv_buf = jnp.zeros((n_cell,) + f_shape[1:], dtype=f.dtype)
        elif loc == "vtx" or loc == "lev_vtx":
            n_vtx    = halo.recv_n_vtx[inb]
            recv_idx = halo.recv_vtx_id  [inb, :n_vtx ]
            recv_buf = jnp.zeros((n_vtx ,) + f_shape[1:], dtype=f.dtype)
        elif loc == "edge" or loc == "lev_edge":
            n_edge   = halo.recv_n_edge[inb]
            recv_idx = halo.recv_edge_id [inb, :n_edge]
            recv_buf = jnp.zeros((n_edge,) + f_shape[1:], dtype=f.dtype)
        else:
            raise ValueError(f"Unsupported location ({loc}) for array!")

        send_buf = send_buf.at[...].set(jnp.take(f, send_idx, axis=0))
        recv_buf = mpi4jax.sendrecv(
            send_buf,
            recv_buf,
            source=nb,
            dest=nb,
            sendtag=tag,
            recvtag=tag,
            comm=halo.comm,
        )

        f = f.at[recv_idx, ...].set(recv_buf)

    return f
