from __future__ import annotations
from typing import TYPE_CHECKING

from terraflow.util import math, sphere_geometry
from terraflow.mesh.scvt.parallel import fill_raw_array_halo as fill_halo

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.util.device import AbstractDevice
    from terraflow.util.planet import PlanetParameterSet
    from terraflow.mesh.scvt.mesh_type import SCVTMesh

    FloatType = typing.FloatType
    Array = typing.Array


def calc_kite_area(para: PlanetParameterSet, mesh: SCVTMesh, dev: AbstractDevice) -> SCVTMesh:
    kite_area_cell = calc_kite_area_cell(para, mesh, dev)
    kite_area_vtx  = calc_kite_area_vtx (para, mesh, dev)
    return dev.backend.tree_at(
        where=lambda mesh: (mesh.kite_area_cell, mesh.kite_area_vtx),
        pytree=mesh,
        replace=(kite_area_cell, kite_area_vtx)
    )


def correct_normal_vectors(para: PlanetParameterSet, mesh: SCVTMesh, dev: AbstractDevice) -> SCVTMesh:
    cell_nr = calc_nr_cell(para, mesh, dev)
    vtx_nr  = calc_nr_vtx (para, mesh, dev)
    edptg_eddnr = calc_edptg_eddnr(mesh)
    edpnr_eddtg = calc_edpnr_eddtg(mesh)
    edp12_edptg = calc_edp12_edptg(mesh)
    edp12_eddnr = calc_edp12_eddnr(mesh)
    return dev.backend.tree_at(
        where=lambda mesh: (mesh.cell_nr, mesh.vtx_nr,
                            mesh.edptg_eddnr, mesh.edpnr_eddtg,
                            mesh.edp12_edptg, mesh.edp12_eddnr),
        pytree=mesh,
        replace=(cell_nr, vtx_nr, edptg_eddnr, edpnr_eddtg, edp12_edptg, edp12_eddnr)
    )


def check_kite_area_cell(
    radius: FloatType,
    kite_area: Array,
    area: Array,
    dev: AbstractDevice
) -> Array:
    """Check if the sum of cell kite areas matches the cell area."""
    bkd = dev.backend
    return bkd.error_if(
        x=kite_area,
        pred=bkd.allclose(
            bkd.sum(kite_area / radius ** 2, dtype=area.dtype, axis=-1),
            area / radius ** 2,
            rtol=1e-5,
            atol=1e-5
        ) * False,
        msg="Sum of kite_area_cell does not match area_cell.",
    )


def check_kite_area_vtx(
    radius: FloatType,
    kite_area: Array,
    area: Array,
    dev: AbstractDevice
) -> Array:
    """Check if the sum of vtx kite areas matches the vtx area."""
    bkd = dev.backend
    return bkd.error_if(
        x=kite_area,
        pred=bkd.allclose(
            bkd.sum(kite_area / radius ** 2, dtype=area.dtype, axis=-1),
            area / radius ** 2,
            rtol=1e-5,
            atol=1e-5
        ) * False,
        msg="Sum of kite_area_vtx does not match area_vtx.",
    )


def tile_det(bkd, p1: Array, p2: Array, p3: Array) -> Array:
    return bkd.sum(p1 * bkd.cross(p2, p3, axis=-1), axis=-1)


def calc_kite_area_cell(para: PlanetParameterSet, mesh: SCVTMesh, dev: AbstractDevice) -> Array:
    bkd = dev.backend
    radius = para.radius.unwrap()
    ids, ide = mesh.cell_ids, mesh.cell_ide
    kite_area_cell = mesh.kite_area_cell
    area_cell = mesh.area_cell
    # Calculate kite area for cell
    mask = mesh.cell_nb[ids:ide+1, :] >= 0    # (n_cell, maxnnb) Boolean
    pc = mesh.cell_xyz[ids:ide+1, :] / radius # (n_cell, 3)
    pc_tile = bkd.expand_dims(pc, axis=1)
    pc_tile = bkd.tile(pc_tile, (1, mesh.maxnnb_cell, 1)) # (n_cell, nnb, 3)
    ed1 = mesh.cell_ed[ids:ide+1,:]      # (n_cell, maxnnb)
    ed2 = math.roll_nonnegative(bkd, ed1, 1)  # (n_cell, maxnnb)
    trv = mesh.vtx_of_cell[ids:ide+1,:]  # (n_cell, maxnnb)
    ed_cc = mesh.dual_edge_xyz / radius  # (n_edge, 3)
    ped1_tile = bkd.take(ed_cc, indices=ed1, axis=0)  # (n_cell, maxnnb, 3)
    ped2_tile = bkd.take(ed_cc, indices=ed2, axis=0)  # (n_cell, maxnnb, 3)
    pv = mesh.vtx_xyz / radius           # (n_vtx, 3)
    pv_tile = bkd.take(pv, indices=trv, axis=0)  # (n_cell, maxnnb, 3)
    area1 = sphere_geometry.spherical_triangle_area(
        bkd=bkd,
        R=radius,
        p1=pc_tile,
        p2=ped1_tile,
        p3=pv_tile,
        axis=-1
    )
    area2 = sphere_geometry.spherical_triangle_area(
        bkd=bkd,
        R=radius,
        p1=pc_tile,
        p2=pv_tile,
        p3=ped2_tile,
        axis=-1
    )
    det = tile_det(bkd, pc_tile, ped1_tile, pv_tile)
    area1 = bkd.where(det < 0, -area1, area1)
    det = tile_det(bkd, pc_tile, pv_tile, ped2_tile)
    area2 = bkd.where(det < 0, -area2, area2)
    kite_area = area1 + area2  # (n_cell, maxnnb)
    kite_area = bkd.where(mask, kite_area, 0.0)
    kite_area_cell = bkd.slice_update(kite_area_cell, (slice(ids, ide+1), slice(None)), kite_area)
    kite_area_cell = bkd.slice_update(kite_area_cell, (slice(ids, ide+1), slice(None)),
        check_kite_area_cell(
            radius=radius,
            kite_area=kite_area_cell[ids:ide+1,:],
            area=area_cell[ids:ide+1],
            dev=dev
        )
    )
    return fill_halo(mesh.halo, kite_area_cell, "cell")


def calc_kite_area_vtx(para: PlanetParameterSet, mesh: SCVTMesh, dev: AbstractDevice) -> math.ArrayWithUnits:
    bkd = dev.backend
    radius = para.radius.unwrap()
    ids, ide = mesh.vtx_ids, mesh.vtx_ide
    kite_area_vtx = mesh.kite_area_vtx
    area_vtx = mesh.area_vtx
    # Calculate kite area for vtx
    mask = mesh.vtx_nb[ids:ide+1, :] >= 0     # (n_vtx, maxnnb) Boolean
    pc = mesh.vtx_xyz[ids:ide+1, :] / radius  # (n_vtx, 3)
    pc_tile = bkd.expand_dims(pc, axis=1)
    pc_tile = bkd.tile(pc_tile, (1, mesh.maxnnb_vtx, 1)) # (n_vtx, nnb, 3)
    ed1 = math.roll_nonnegative(bkd, mesh.vtx_ed[ids:ide+1, :], -1)  # (n_vtx, maxnnb)
    ed2 = mesh.vtx_ed[ids:ide+1, :]                             # (n_vtx, maxnnb)
    trv = mesh.cell_of_vtx[ids:ide+1, :] # (n_vtx, maxnnb)
    ed_cc = mesh.dual_edge_xyz / radius  # (n_edge, 3)
    ped1_tile = bkd.take(ed_cc, indices=ed1, axis=0)  # (n_vtx, maxnnb, 3)
    ped2_tile = bkd.take(ed_cc, indices=ed2, axis=0)  # (n_vtx, maxnnb, 3)
    pv = mesh.cell_xyz / radius          # (n_cell, 3)
    pv_tile = bkd.take(pv, indices=trv, axis=0)  # (n_vtx, maxnnb, 3)
    area1 = sphere_geometry.spherical_triangle_area(
        bkd=bkd,
        R=radius,
        p1=pc_tile,
        p2=ped1_tile,
        p3=pv_tile,
        axis=-1
    )
    area2 = sphere_geometry.spherical_triangle_area(
        bkd=bkd,
        R=radius,
        p1=pc_tile,
        p2=pv_tile,
        p3=ped2_tile,
        axis=-1
    )
    det = tile_det(bkd, pc_tile, ped1_tile, pv_tile)
    area1 = bkd.where(det < 0, -area1, area1)
    det = tile_det(bkd, pc_tile, pv_tile, ped2_tile)
    area2 = bkd.where(det < 0, -area2, area2)
    kite_area = area1 + area2  # (n_vtx, maxnnb)
    kite_area = bkd.where(mask, kite_area, 0.0)
    kite_area_vtx = bkd.slice_update(kite_area_vtx, (slice(ids, ide+1), slice(None)), kite_area)
    kite_area_vtx = bkd.slice_update(kite_area_vtx, (slice(ids, ide+1), slice(None)),
        check_kite_area_vtx(
            radius=radius,
            kite_area=kite_area_vtx[ids:ide+1, :],
            area=area_vtx[ids:ide+1],
            dev=dev
        )
    )
    return fill_halo(mesh.halo, kite_area_vtx, "vtx")


def calc_nr_cell(para: PlanetParameterSet, mesh: SCVTMesh, dev: AbstractDevice) -> Array:
    r"""
    Correct the normal vectors of edges:
    For a polygon:
                  nb1,ed1
                tr1-----tr6
        nb2,ed2 /          \ nb6,ed6
              tr2  cell  tr5
        nb3,ed3 \          / nb5,ed5
                tr3-----tr4
                  nb4,ed
    """
    bkd = dev.backend
    ids, ide = mesh.cell_ids, mesh.cell_ide
    radius = para.radius.unwrap()
    cell_nr = mesh.cell_nr
    # =============================================
    mask = mesh.cell_nb[ids:ide+1, :] >=0     # (n_cell, maxnnb) Boolean
    p0 = mesh.cell_xyz[ids:ide+1, :] / radius # (n_cell, 3)
    p0_tile = bkd.expand_dims(p0, axis=1)
    p0_tile = bkd.tile(
        p0_tile,
        (1, mesh.maxnnb_cell, 1)
    )  # (n_cell, maxnnb, 3)
    global_idx_plgnb = mesh.cell_nb[ids:ide+1, :]  # (n_cell, maxnnb)
    p1_tile = bkd.take(
        mesh.cell_xyz,
        indices=global_idx_plgnb,
        axis=0
    ) / radius  # (n_cell, maxnnb, 3)
    global_idx_plged = mesh.cell_ed[ids:ide+1, :]  # (n_cell, maxnnb)
    nr_ed01 = bkd.take(
        mesh.edge_nr,
        indices=global_idx_plged,
        axis=0
    )  # (n_cell, maxnnb, 3)
    product = bkd.sum(
        (p1_tile - p0_tile) * (p1_tile + p0_tile),
        axis=-1
    )  # (n_cell, maxnnb)
    product = bkd.where(mask, product, 0.0)

    product = bkd.error_if(
        x=product,
        pred=bkd.any((product > bkd.eps(p0.dtype) * 10) == True),
        msg="Vector has problem in calculating cell nr.",
    )

    tmp = bkd.sum(
        (p1_tile - p0_tile) * nr_ed01,
        axis=-1
    )  # (n_cell, maxnnb)
    # To prevent non-zero values in padding values in nr
    tmp = bkd.where(mask, tmp, 0.0)
    nr = bkd.where(
        tmp > bkd.eps(p0.dtype) * 100,
        1,
        0
    )  # (n_cell, maxnnb)
    nr = bkd.where(tmp < -bkd.eps(p0.dtype) * 100, -1, nr)

    nr = bkd.error_if(
        x=nr,
        pred=bkd.sum(bkd.abs(bkd.where(mask == False, nr, 0))) > 0,
        msg="There are non-zeros in padding values of calculated cell nr.",
    )

    nr = bkd.astype(bkd.where(mask, nr, 0), mesh.vtx_of_cell.dtype)
    cell_nr = bkd.slice_update(cell_nr, (slice(ids, ide+1), slice(None)), nr)
    # =============================================
    return fill_halo(mesh.halo, cell_nr, "cell")


def calc_nr_vtx(para: PlanetParameterSet, mesh: SCVTMesh, dev: AbstractDevice) -> Array:
    r"""
        For a triangle:
                 local_v1
                   /\
             ed1  /  \  ed3
            p1   /    \   p3
                /  p0  \
               /        \
             v2 ---------v3
                  ed2 p
    """
    bkd = dev.backend
    ids, ide = mesh.vtx_ids, mesh.vtx_ide
    radius = para.radius.unwrap()
    vtx_nr = mesh.vtx_nr
    # =============================================
    mask = mesh.vtx_nb[ids:ide+1, :] >= 0     # (n_vtx, maxnnb) Boolean
    p0 = mesh.vtx_xyz[ids:ide+1, :] / radius  # (n_vtx, 3)
    p0_tile = bkd.expand_dims(p0, axis=1)
    p0_tile = bkd.tile(
        p0_tile,
        (1, mesh.maxnnb_vtx, 1)
    )  # (n_vtx, maxnnb, 3)
    global_idx_plgnb = mesh.vtx_nb[ids:ide+1, :]  # (n_vtx, maxnnb)
    p1_tile = bkd.take(
        mesh.vtx_xyz,
        indices=global_idx_plgnb,
        axis=0
    ) / radius # (n_vtx, maxnnb, 3)
    global_idx_plged = mesh.vtx_ed[ids:ide+1, :]  # (n_vtx, maxnnb)
    nr_ed01 = bkd.take(
        mesh.dual_edge_nr,
        indices=global_idx_plged,
        axis=0
    )  # (n_vtx, maxnnb, 3)
    product = bkd.sum(
        (p1_tile - p0_tile) * (p1_tile + p0_tile),
        axis=-1
    )  # (n_vtx, maxnnb)
    product = bkd.where(mask, product, 0.0)

    product = bkd.error_if(
        x=product,
        pred=bkd.any((product > bkd.eps(p0.dtype) * 10) == True),
        msg="Vector has problem in calculating vtx nr.",
    )

    tmp = bkd.sum(
        (p1_tile - p0_tile) * nr_ed01,
        axis=-1
    )  # (n_vtx, maxnnb)
    # To prevent non-zero values in padding values in nr
    tmp = bkd.where(mask, tmp, 0.0)
    nr = bkd.where(
        tmp > bkd.eps(p0.dtype) * 100,
        1,
        0
    )  # (n_cell, maxnnb)
    nr = bkd.where(tmp < -bkd.eps(p0.dtype) * 100, -1, nr)

    nr = bkd.error_if(
        x=nr,
        pred=bkd.sum(bkd.abs(bkd.where(mask == False, nr, 0))) > 0,
        msg="There are non-zeros in padding values of calculated vtx nr.",
    )

    nr = bkd.astype(bkd.where(mask, nr, 0), mesh.cell_of_vtx.dtype)
    vtx_nr = bkd.slice_update(vtx_nr, (slice(ids, ide+1), slice(None)), nr)
    # =============================================
    return fill_halo(mesh.halo, vtx_nr, "vtx")


def calc_edptg_eddnr(mesh: SCVTMesh) -> Array:
    """
    Return the sign between edp tangent and edd normal vectors. If they are
    the same direction, return 1; if opposite, return -1.
    """
    bkd = mesh.dev.backend
    ids, ide = mesh.edge_ids, mesh.edge_ide
    edptg_eddnr = mesh.edptg_eddnr
    edp_tg = mesh.edge_tg     [ids:ide+1,:]
    edd_nr = mesh.dual_edge_nr[ids:ide+1,:]
    tmp = bkd.astype(bkd.sign(bkd.sum(edp_tg * edd_nr, axis=-1)), edptg_eddnr.dtype)
    edptg_eddnr = bkd.slice_update(edptg_eddnr, slice(ids, ide+1), tmp)
    return fill_halo(mesh.halo, edptg_eddnr, "edge")



def calc_edpnr_eddtg(mesh: SCVTMesh) -> Array:
    """
    Return the sign between edp normal and edd tangent vectors. If they are
    the same direction, return 1; if opposite, return -1.
    """
    bkd = mesh.dev.backend
    ids, ide = mesh.edge_ids, mesh.edge_ide
    edpnr_eddtg = mesh.edpnr_eddtg
    edp_nr = mesh.edge_nr     [ids:ide+1,:]
    edd_tg = mesh.dual_edge_tg[ids:ide+1,:]
    tmp = bkd.astype(bkd.sign(bkd.sum(edp_nr * edd_tg, axis=-1)), edpnr_eddtg.dtype)
    edpnr_eddtg = bkd.slice_update(edpnr_eddtg, slice(ids, ide+1), tmp)
    return fill_halo(mesh.halo, edpnr_eddtg, "edge")



def calc_edp12_edptg(mesh: SCVTMesh) -> Array:
    """
    Return the sign between edp1 to edp2 and edp tangent vectors. If they are
    the same direction, return 1; if opposite, return -1.
    """
    bkd = mesh.dev.backend
    ids, ide = mesh.edge_ids, mesh.edge_ide
    edp12_edptg = mesh.edp12_edptg
    edp1 = mesh.vtx_of_edge[ids:ide+1, 0]  # (n_edge, )
    edp2 = mesh.vtx_of_edge[ids:ide+1, 1]  # (n_edge, )
    edp12 = mesh.vtx_xyz[edp2, :] - mesh.vtx_xyz[edp1, :]  # (n_edge, 3)
    edp_tg = mesh.edge_tg[ids:ide+1, :]
    tmp = bkd.astype(bkd.sign(bkd.sum(edp12 * edp_tg, axis=-1)), edp12_edptg.dtype)
    edp12_edptg = bkd.slice_update(edp12_edptg, slice(ids, ide+1), tmp)
    return fill_halo(mesh.halo, edp12_edptg, "edge")



def calc_edp12_eddnr(mesh: SCVTMesh) -> Array:
    """
    Return the sign between edp1 to edp2 and edd normal vectors. If they are
    the same direction, return 1; if opposite, return -1.
    """
    bkd = mesh.dev.backend
    ids, ide = mesh.edge_ids, mesh.edge_ide
    edp12_eddnr = mesh.edp12_eddnr
    edp1 = mesh.vtx_of_edge[ids:ide+1, 0]  # (n_edge, )
    edp2 = mesh.vtx_of_edge[ids:ide+1, 1]  # (n_edge, )
    edp12 = mesh.vtx_xyz[edp2, :] - mesh.vtx_xyz[edp1, :]  # (n_edge, 3)
    edd_nr = mesh.dual_edge_nr[ids:ide+1]
    tmp = bkd.astype(bkd.sign(bkd.sum(edp12 * edd_nr, axis=-1)), edp12_eddnr.dtype)
    edp12_eddnr = bkd.slice_update(edp12_eddnr, slice(ids, ide+1), tmp)
    return fill_halo(mesh.halo, edp12_eddnr, "edge")
