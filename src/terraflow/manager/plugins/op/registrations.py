from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import TypeAlias

from terraflow.discretization.fvm.odm.trisk.adv import flux_reconstruct


if TYPE_CHECKING:
    from terraflow.manager.registry import Catalog
else:
    ModelTableFactory: TypeAlias = "terraflow.manager.registry.Catalog"


def register(model: Catalog) -> None:
    from terraflow.manager.registry import Catalog

    assert isinstance(model, Catalog)

    import terraflow.mesh.orthogonal_dual_mesh.operator as op

    # Define belong key for FVM triskC ODM operators
    belong_key = "fvm::triskC::odm::1level"

    # Register base operators
    model.op.register(
        belong_key, group="div",
        factory=op.div_at_primal_cell
    )
    model.op.register(
        belong_key, group="grad",
        factory=op.grad_at_dual_edge
    )
    model.op.register(
        belong_key, group="curl",
        factory=op.curl_at_dual_cell
    )
    model.op.register(
        belong_key, group="interp",
        factory=op.interp_weighted_add_edp2plg
    )
    model.op.register(
        belong_key, group="interp",
        factory=op.interp_weighted_add_edd2dua
    )
    model.op.register(
        belong_key, group="interp",
        factory=op.interp_weighted_add_dua2plg
    )

    # Register coriolis operators
    from terraflow.discretization.fvm.odm.trisk import coriolis
    model.op.register(
        belong_key, group="coriolis",
        factory=coriolis.trisk.TriSKCoriolisForceAtEdge
    )

    # Register flux reconstruction operators
    # Register various flux reconstruction methods at primal edges
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.CentralDifference2AtEdgeOnPrimalMesh
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.WRF3AtEdgeOnPrimalMesh
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.WRF4AtEdgeOnPrimalMesh
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.HybridWRF34AtEdgeOnPrimalMesh
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.WRF5AtEdgeOnPrimalMesh
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.WRF6AtEdgeOnPrimalMesh
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.HybridWRF56AtEdgeOnPrimalMesh
    )
    # Register various flux reconstruction methods at dual edges
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.CentralDifference2AtEdgeOnDualMesh
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.WRF3AtEdgeOnDualMesh
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.WRF4AtEdgeOnDualMesh
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.HybridWRF34AtEdgeOnDualMesh
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.WRF5AtEdgeOnDualMesh
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.WRF6AtEdgeOnDualMesh
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.HybridWRF56AtEdgeOnDualMesh
    )
