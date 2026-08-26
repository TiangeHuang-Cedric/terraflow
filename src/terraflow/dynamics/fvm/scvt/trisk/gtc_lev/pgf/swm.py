from __future__ import annotations
from typing import TYPE_CHECKING

from terraflow.util import typing
from terraflow.util.pytree import tree_at

if TYPE_CHECKING:
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block

    Int = typing.Int
    Float = typing.Float
    Array = typing.Array


def pgf_swm_run(block: Block, star_idx: Int[Array, ""] | int) -> Block:
    mesh = block.mesh
    ids, ide = mesh.edge_ids, mesh.edge_ide
    kds, kde = mesh.full_kds, mesh.full_kde
    gz        = block.state.gz       .unwrap()[star_idx, :, :]
    dundt     = block.tend .dundt    .unwrap()
    dundt_pgf = block.tend .dundt_pgf.unwrap()
    # ==========================================================
    v1 = mesh.cell_of_edge[ids:ide+1, 0]
    v2 = mesh.cell_of_edge[ids:ide+1, 1]
    g1 = gz[kds:kde+1, v1]
    g2 = gz[kds:kde+1, v2]
    tmp = -(g2-g1) * mesh.edpnr_eddtg[ids:ide+1] / mesh.de[ids:ide+1]
    dundt     = dundt    .at[kds:kde+1, ids:ide+1].set(
        dundt[kds:kde+1, ids:ide+1] + tmp
    )
    dundt_pgf = dundt_pgf.at[kds:kde+1, ids:ide+1].set(tmp)
    # ==========================================================
    return tree_at(
        where=lambda b: (b.tend.dundt.d, b.tend.dundt_pgf.d),
        pytree=block,
        replace=(dundt, dundt_pgf)
    )    

