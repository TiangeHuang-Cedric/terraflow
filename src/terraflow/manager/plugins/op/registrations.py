from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import TypeAlias


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
        factory=op.DivAtPrimalCell
    )
    model.op.register(
        belong_key, group="grad",
        factory=op.GradAtDualEdge
    )
    model.op.register(
        belong_key, group="curl",
        factory=op.CurlAtDualCell
    )
    model.op.register(
        belong_key, group="interp",
        factory=op.InterpWeightedAddEdp2Plg
    )
    model.op.register(
        belong_key, group="interp",
        factory=op.InterpWeightedAddEdd2Dua
    )
    model.op.register(
        belong_key, group="interp",
        factory=op.InterpWeightedAddDua2Plg
    )

    # Register coriolis operators
    from terraflow.discretization.fvm.odm.trisk import coriolis, flux_reconstruct
    model.op.register(
        belong_key, group="coriolis",
        factory=coriolis.CoriolisForceAtDualEdge
    )

    # Register flux reconstruction operators
    # Register various flux reconstruction methods at primal edges
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.CentralDifference2AtPrimalEdge
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.WRF3AtPrimalEdge
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.WRF4AtPrimalEdge
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.HybridWRF34AtPrimalEdge
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.WRF5AtPrimalEdge
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.WRF6AtPrimalEdge
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.HybridWRF56AtPrimalEdge
    )
    # Register various flux reconstruction methods at dual edges
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.CentralDifference2AtDualEdge
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.WRF3AtDualEdge
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.WRF4AtDualEdge
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.HybridWRF34AtDualEdge
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.WRF5AtDualEdge
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.WRF6AtDualEdge
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=flux_reconstruct.HybridWRF56AtDualEdge
    )
