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

    import terraflow.operator.fvm.odm as op

    # Define belong key for FVM triskC ODM operators
    belong_key = "fvm::triskC::odm::1level"

    # Register base operators
    model.op.register(
        belong_key, group="div",
        factory=op.base.div_at_primal_cell
    )
    model.op.register(
        belong_key, group="grad",
        factory=op.base.grad_at_edge
    )
    model.op.register(
        belong_key, group="curl",
        factory=op.base.curl_at_dual_cell
    )

    # Register interpolation operators
    model.op.register(
        belong_key, group="interp",
        factory=op.interp.interp_weighted_add_edp2plg
    )
    model.op.register(
        belong_key, group="interp",
        factory=op.interp.interp_weighted_add_edd2dua
    )
    model.op.register(
        belong_key, group="interp",
        factory=op.interp.interp_weighted_add_dua2plg
    )
    model.op.register(
        belong_key, group="interp",
        factory=op.interp.interp_weighted_add_pair_meeting_edges2edge
    )
    model.op.register(
        belong_key, group="interp",
        factory=op.interp.calc_trsk_weight
    )
    model.op.register(
        belong_key, group="interp",
        factory=op.interp.trisk_weighted_interpolation_operator
    )

    # Register flux reconstruction operators
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=op.flux_reconstruct.flux_reconstruct_central_difference_2nd_order
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=op.flux_reconstruct.flux_reconstruct_least_square_3rd4th_order
    )
    model.op.register(
        belong_key, group="flux_reconstruct",
        factory=op.flux_reconstruct.flux_reconstruct_least_square_5th6th_order
    )
