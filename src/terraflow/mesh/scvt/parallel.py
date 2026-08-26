from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
import mpi4jax

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.mesh.scvt.field import SCVTFieldMeta
    from terraflow.mesh.scvt.halo import SCVTHalo
    
    FLT   = typing.Float
    Array = typing.Array

TAGS = {"cell": 1, "vtx": 2, "edge": 3, "lev": 1, "lev_vtx": 2, "lev_edge": 3}


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
            send_idx = halo.send_cell_id [inb, :n_cell   ]
            send_buf = halo.send_cell_buf[inb, :n_cell, 0]
        elif loc == "vtx" or loc == "lev_vtx":
            n_vtx    = halo.send_n_vtx   [inb            ]
            send_idx = halo.send_vtx_id  [inb, :n_vtx    ]
            send_buf = halo.send_vtx_buf [inb, :n_vtx , 0]
        elif loc == "edge" or loc == "lev_edge":
            n_edge   = halo.send_n_edge  [inb]
            send_idx = halo.send_edge_id [inb, :n_edge]
            send_buf = halo.send_edge_buf[inb, :n_edge, 0]
        else:
            raise ValueError(f"Unsupported location ({loc}) for field: {meta.name}!")

        if loc == "cell" or loc == "lev":
            n_cell   = halo.recv_n_cell  [inb            ]
            recv_idx = halo.recv_cell_id [inb, :n_cell   ]
            recv_buf = halo.recv_cell_buf[inb, :n_cell, 0]
        elif loc == "vtx" or loc == "lev_vtx":
            n_vtx    = halo.recv_n_vtx   [inb            ]
            recv_idx = halo.recv_vtx_id  [inb, :n_vtx    ]
            recv_buf = halo.recv_vtx_buf [inb, :n_vtx , 0]
        elif loc == "edge" or loc == "lev_edge":
            n_edge   = halo.recv_n_edge  [inb            ]
            recv_idx = halo.recv_edge_id [inb, :n_edge   ]
            recv_buf = halo.recv_edge_buf[inb, :n_edge, 0]
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
            send_idx = halo.send_cell_id [inb, :n_cell]
            send_buf = halo.send_cell_buf[inb, :n_cell, :n_lev]
        elif loc == "vtx" or loc == "lev_vtx":
            n_vtx    = halo.send_n_vtx[inb]
            n_lev = halo.full_nlev if loc == "vtx"  else halo.half_nlev
            send_idx = halo.send_vtx_id  [inb, :n_vtx ]
            send_buf = halo.send_vtx_buf [inb, :n_vtx , :n_lev]
        elif loc == "edge" or loc == "lev_edge":
            n_edge   = halo.send_n_edge[inb]
            n_lev = halo.full_nlev if loc == "edge" else halo.half_nlev
            send_idx = halo.send_edge_id [inb, :n_edge]
            send_buf = halo.send_edge_buf[inb, :n_edge, :n_lev]
        else:
            raise ValueError(f"Unsupported location ({loc}) for field: {meta.name}!")

        if loc == "cell" or loc == "lev":
            n_cell   = halo.recv_n_cell[inb]
            n_lev = halo.full_nlev if loc == "cell" else halo.half_nlev
            recv_idx = halo.recv_cell_id [inb, :n_cell]
            recv_buf = halo.recv_cell_buf[inb, :n_cell, :n_lev]
        elif loc == "vtx" or loc == "lev_vtx":
            n_vtx    = halo.recv_n_vtx[inb]
            n_lev = halo.full_nlev if loc == "vtx"  else halo.half_nlev
            recv_idx = halo.recv_vtx_id  [inb, :n_vtx ]
            recv_buf = halo.recv_vtx_buf [inb, :n_vtx , :n_lev]
        elif loc == "edge" or loc == "lev_edge":
            n_edge   = halo.recv_n_edge[inb]
            n_lev = halo.full_nlev if loc == "edge" else halo.half_nlev
            recv_idx = halo.recv_edge_id [inb, :n_edge]
            recv_buf = halo.recv_edge_buf[inb, :n_edge, :n_lev]
        else:
            raise ValueError(f"Unsupported location ({loc}) for field: {meta.name}!")

        arr = jnp.swapaxes(f[:n_lev, send_idx], 0, 1)
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

        f = f.at[:n_lev, recv_idx].set(jnp.swapaxes(recv_buf, 0, 1))

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
            send_idx = halo.send_cell_id [inb, :n_cell]
            send_buf = halo.send_cell_buf[inb, :n_cell, :n_lev]
        elif loc == "vtx" or loc == "lev_vtx":
            n_vtx    = halo.send_n_vtx[inb]
            n_lev = halo.full_nlev if loc == "vtx"  else halo.half_nlev
            send_idx = halo.send_vtx_id  [inb, :n_vtx ]
            send_buf = halo.send_vtx_buf [inb, :n_vtx , :n_lev]
        elif loc == "edge" or loc == "lev_edge":
            n_edge   = halo.send_n_edge[inb]
            n_lev = halo.full_nlev if loc == "edge" else halo.half_nlev
            send_idx = halo.send_edge_id [inb, :n_edge]
            send_buf = halo.send_edge_buf[inb, :n_edge, :n_lev]
        else:
            raise ValueError(f"Unsupported location ({loc}) for field: {meta.name}!")

        if loc == "cell" or loc == "lev":
            n_cell   = halo.recv_n_cell[inb]
            n_lev = halo.full_nlev if loc == "cell" else halo.half_nlev
            recv_idx = halo.recv_cell_id [inb, :n_cell]
            recv_buf = halo.recv_cell_buf[inb, :n_cell, :n_lev]
        elif loc == "vtx" or loc == "lev_vtx":
            n_vtx    = halo.recv_n_vtx[inb]
            n_lev = halo.full_nlev if loc == "vtx"  else halo.half_nlev
            recv_idx = halo.recv_vtx_id  [inb, :n_vtx ]
            recv_buf = halo.recv_vtx_buf [inb, :n_vtx , :n_lev]
        elif loc == "edge" or loc == "lev_edge":
            n_edge   = halo.recv_n_edge[inb]
            n_lev = halo.full_nlev if loc == "edge" else halo.half_nlev
            recv_idx = halo.recv_edge_id [inb, :n_edge]
            recv_buf = halo.recv_edge_buf[inb, :n_edge, :n_lev]
        else:
            raise ValueError(f"Unsupported location ({loc}) for field: {meta.name}!")

        arr = jnp.swapaxes(f[i0, :n_lev, send_idx], 0, 1)
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

        f = f.at[i0, :n_lev, recv_idx].set(jnp.swapaxes(recv_buf, 0, 1))

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
