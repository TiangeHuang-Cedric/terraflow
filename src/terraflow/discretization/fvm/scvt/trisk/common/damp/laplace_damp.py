from __future__ import annotations
from typing import TYPE_CHECKING

from terraflow.mesh.scvt import operators as mesh_ops

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as HybridLevBlock

    Float = typing.Float
    Array = typing.Array
    BLOCK = HybridLevBlock


def laplace_damp_1d(
    block: BLOCK,
    f_1d: Float[Array, "npoints"],
    order: int,
    coef: float,
) -> tuple[BLOCK, Float[Array, "npoints"]]:
    c0 = (-1)**(order / 2) * coef
    para = block.para
    coord = block.coord
    f = block.aux.f_1d.unwrap()
    df = block.aux.df_1d.unwrap()
    # =====================================
    g = f_1d.reshape(1, -1)
    f = f.reshape(1, -1)
    df = df.reshape(1, -1)
    if (order - 2) // 2 > 0:
        for _ in range((order - 2) // 2):
            f = mesh_ops.grad_at_dual_edge(
                x_at_primal_cell=g,  # in
                x_at_dual_edge=f,    # out
                coord=coord,
                para=para
            )
            g = mesh_ops.div_at_primal_cell(
                x_at_primal_edge=f,  # in
                x_at_primal_cell=g,  # out
                coord=coord,
                para=para
            )
    f = mesh_ops.grad_at_dual_edge(
        x_at_primal_cell=g,  # in
        x_at_dual_edge=f,    # out
        coord=coord,
        para=para,
    )
    f = c0 * f
    # TODO: add additional damping when order is larger than 2
    df = mesh_ops.div_at_primal_cell(
        x_at_primal_edge=f,   # in
        x_at_primal_cell=df,  # out
        coord=coord,
        para=para,
    ).reshape(-1)
    # =====================================
    return block, f_1d - df
