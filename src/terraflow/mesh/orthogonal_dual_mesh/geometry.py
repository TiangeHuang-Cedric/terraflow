import jax.numpy as jnp
from jax import lax, debug, jit

import terraflow.util.math as math
from terraflow.mesh.orthogonal_dual_mesh.base import RefPrimalCell, RefDualCell
from terraflow.mesh.orthogonal_dual_mesh.base import RefPrimalEdge, RefDualEdge


@jit
def spherical_plg_kite_areas(
    cell: RefPrimalCell,
    edge: RefPrimalEdge,
    dual: RefDualCell
) -> jnp.ndarray:
    """
    Caculate spherical kite areas for a polygon.
    Args:
        cell (PrimalCell | GristPrimalCell): The cell element to calculate kite areas for.
        edge (PrimalEdge | GristPrimalEdge): The edge element associated with the cell.
        dual (DualCell | GristDualCell): The dual cell element associated with the cell.
    Returns:
        jnp.ndarray: Array of shape (maxnnb,) representing the kite areas.
    """
    mask = cell.valid_neighbors()  # (n_cell, maxnnb) Boolean
    pc = cell.centroid.cartesian  # (n_cell, 3)
    pc_tile = jnp.expand_dims(pc, axis=1)
    # (n_cell, nnb, 3)
    pc_tile = jnp.tile(pc_tile, (1, cell.max_neighbors, 1))
    ed1 = cell.edge  # (n_cell, maxnnb)
    ed2 = math.roll_nonnegative(ed1, 1)  # (n_cell, maxnnb)
    trv = cell.vertex  # (n_cell, maxnnb)
    ed_cc = edge.midpoint.cartesian  # (n_edge, 3)
    ped1_tile = jnp.take(ed_cc, ed1, axis=0)  # (n_cell, maxnnb, 3)
    ped2_tile = jnp.take(ed_cc, ed2, axis=0)  # (n_cell, maxnnb, 3)
    pv = dual.centroid.cartesian  # (n_dual_cell, 3)
    pv_tile = jnp.take(pv, trv, axis=0)  # (n_cell, maxnnb, 3)
    area1 = math.spherical_triangle_area(
        pc_tile, ped1_tile, pv_tile, axis=-1)
    area2 = math.spherical_triangle_area(
        pc_tile, pv_tile, ped2_tile, axis=-1)
    kite_area = area1 + area2  # (n_cell, maxnnb)
    kite_area = jnp.where(mask, kite_area, 0.0)
    cell.check_kite_areas(kite_area)
    return kite_area.astype(cell.area.dtype)


@jit
def spherical_tri_kite_areas(
    cell: RefDualCell,
    edge: RefDualEdge,
    dual: RefPrimalCell
) -> jnp.ndarray:
    """
    Caculate spherical kite areas for a triangle.
    Args:
        cell (DualCell | GristDualCell): The cell element to calculate kite areas for.
        edge (DualEdge | GristDualEdge): The edge element associated with the cell.
        dual (PrimalCell | GristPrimalCell): The dual cell element associated with the cell.
    """
    mask = cell.valid_neighbors()  # (n_cell, maxnnb) Boolean
    pc = cell.centroid.cartesian  # (n_cell, 3)
    pc_tile = jnp.expand_dims(pc, axis=1)
    # (n_cell, nnb, 3)
    pc_tile = jnp.tile(pc_tile, (1, cell.max_neighbors, 1))
    ed1 = math.roll_nonnegative(cell.edge, -1)  # (n_cell, maxnnb)
    ed2 = cell.edge  # (n_cell, maxnnb)
    trv = cell.vertex  # (n_cell, maxnnb)
    ed_cc = edge.midpoint.cartesian  # (n_edge, 3)
    ped1_tile = jnp.take(ed_cc, ed1, axis=0)  # (n_cell, maxnnb, 3)
    ped2_tile = jnp.take(ed_cc, ed2, axis=0)  # (n_cell, maxnnb, 3)
    pv = dual.centroid.cartesian  # (n_dual_cell, 3)
    pv_tile = jnp.take(pv, trv, axis=0)  # (n_cell, maxnnb, 3)
    area1 = math.spherical_triangle_area(
        pc_tile, ped1_tile, pv_tile, axis=-1)
    area2 = math.spherical_triangle_area(
        pc_tile, pv_tile, ped2_tile, axis=-1)
    kite_area = area1 + area2  # (n_cell, maxnnb)
    kite_area = jnp.where(mask, kite_area, 0.0)
    cell.check_kite_areas(kite_area)
    return kite_area.astype(cell.area.dtype)


@jit
def correct_normal_vectors(
    cell: RefPrimalCell | RefDualCell,
    edge: RefPrimalEdge | RefDualEdge
) -> jnp.ndarray:
    r"""
    Correct the normal vectors of edges:
    For a polygon (PrimalCell):
                  nb1,ed1
                tr1-----tr6
        nb2,ed2 /          \ nb6,ed6
              tr2  cell  tr5
        nb3,ed3 \          / nb5,ed5
                tr3-----tr4
                  nb4,ed

    For a triangle (DualCell):
                 local_v1
                   /\
             ed1  /  \  ed3
            p1   /    \   p3
                /  p0  \
               /        \
             v2 ---------v3
                  ed2 p
    Args:
        cell (Cell): The cell element to correct normal vectors for.
        edge (Edge): The edge element associated with the cell.
    Returns:
        jnp.ndarray: Corrected normal vectors for the edges of the cell.
    Raises:
        ValueError: If there are issues with the vector calculations.
    """
    mask = cell.valid_neighbors()  # (n_cell, maxnnb) Boolean
    p0 = cell.centroid.cartesian  # (n_cell, 3)
    p0_tile = jnp.expand_dims(p0, axis=1)
    p0_tile = jnp.tile(
        p0_tile,
        (1, cell.max_neighbors, 1)
    )  # (n_cell, maxnnb, 3)
    global_idx_plgnb = cell.neighbors  # (n_cell, maxnnb)
    p1_tile = jnp.take(p0, global_idx_plgnb, axis=0)  # (n_cell, maxnnb, 3)
    global_idx_plged = cell.edge  # (n_cell, maxnnb)
    nr_ed01 = jnp.take(
        edge.nr,
        global_idx_plged,
        axis=0
    )  # (n_cell, maxnnb, 3)
    product = jnp.sum(
        (p1_tile - p0_tile) * (p1_tile + p0_tile),
        dtype=jnp.float64,
        axis=-1
    ).astype(p0.dtype)  # (n_cell, maxnnb)
    product = jnp.where(mask, product, 0.0)

    def error_msg_1(_):
        debug.print("Vector has problem in calculating polygon nr.")

    def do_nothing(_):
        return None

    lax.cond(
        jnp.any((product > jnp.finfo(p0.dtype).eps * 10) == True),
        error_msg_1,
        do_nothing,
        None
    )

    tmp = jnp.sum(
        (p1_tile - p0_tile) * nr_ed01,
        dtype=jnp.float64,
        axis=-1
    ).astype(p0.dtype)  # (n_cell, maxnnb)
    # To prevent non-zero values in padding values in nr
    tmp = jnp.where(mask, tmp, 0.0)
    nr = jnp.where(
        tmp > jnp.finfo(p0.dtype).eps * 100,
        1,
        0
    )  # (n_cell, maxnnb)
    nr = jnp.where(tmp < -jnp.finfo(p0.dtype).eps * 100, -1, nr)

    def error_msg_2(_):
        debug.print(
            "There are non-zeros in padding values of calculated polygon nr."
        )

    lax.cond(
        jnp.sum(jnp.abs(jnp.where(mask == False, nr, 0))) > 0,
        error_msg_2,
        do_nothing,
        None
    )
    return jnp.where(mask, nr, 0).astype(cell.v.dtype)
