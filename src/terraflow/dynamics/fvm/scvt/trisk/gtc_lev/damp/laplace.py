from __future__ import annotations
from typing import TYPE_CHECKING


from terraflow.mesh.scvt import operators as mesh_ops
from terraflow.mesh.scvt.parallel import fill_field1d_array_halo


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block

    Float = typing.Float
    Array = typing.Array


def laplace_damp_1d(
    block: Block,
    f: Float[Array, "npoints"],
    order: int,
    coef: float,
) -> tuple[Block, Float[Array, "npoints"]]:
    c0 = (-1)**(order / 2) * coef
    mesh = block.mesh
    fn  = block.aux.fn_1d.unwrap()
    df  = block.aux.df_1d.unwrap()
    # =====================================
    g = f
    for _ in range((order - 2) // 2):
        fn = mesh_ops.grad_operator_1d(
            mesh=mesh,
            f=g,       # in
            grad=fn,    # out
            f_loc=block.aux.df_1d.meta.loc
        )
        g = mesh_ops.div_operator_1d(
            mesh=mesh,
            f=fn,       # in
            div=g,     # out
            div_loc=block.aux.df_1d.meta.loc
        )
        g = fill_field1d_array_halo(
            halo=mesh.halo,
            f=g,
            meta=block.aux.df_1d.meta
        )
    fn = mesh_ops.grad_operator_1d(
        mesh=mesh,
        f=g,      # in
        grad=fn,    # out
        f_loc=block.aux.df_1d.meta.loc
    )
    fn = c0 * fn
    # TODO: add additional damping when order is larger than 2
    df = mesh_ops.div_operator_1d(
        mesh=mesh,
        f=fn,     # in
        div=df,  # out
        div_loc=block.aux.df_1d.meta.loc
    )
    # =====================================
    f = fill_field1d_array_halo(
        halo=mesh.halo,
        f=f - df,
        meta=block.aux.df_1d.meta
    )
    return block, f
