from __future__ import annotations
from typing import (
    Self,
    Optional,
    TypeVar,
    TYPE_CHECKING
)
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import log
from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
from terraflow.test_case.swe import (
    AbstractShallowWaterEquationTestCase,
    MountainZonalFlowTestCase
)
from terraflow.test_case.adv import (
    AbstractAdvectionEquation2DTestCase,
    SolidRotationTestCase,
)
from terraflow.discretization.fvm.scvt.trisk.base_types import AbstractBaseBlock
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at
from ..test_case.swm import barotropic_case_init_fn
from ..test_case.adv import advection2d_case_init_fn


if TYPE_CHECKING:
    from terraflow.planet import PlanetParameterSet
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.solver.abstract import AbstractSolver
    from terraflow.test_case.abstract import AbstractTestCase
    from terraflow.discretization.abstract import (
        EquationMode,
        AbstractEquationSet,
    )

    CoordSys = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        OneLayerVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        OneLayerVertCoordLayout
    ]

    ES = TypeVar("ES", bound=AbstractEquationSet)


class Block(AbstractBaseBlock[OneLayerVertCoordinate, OneLayerVertCoordLayout]):

    @classmethod
    def from_testcase(
        cls,
        coord: CoordSys,
        para: PlanetParameterSet,
        solver: AbstractSolver[ES],
        *,
        case: Optional[AbstractTestCase] = None,
    ) -> Self:
        equation = solver.equation
        mode: EquationMode = equation.mode
        # =======================================================
        if mode.is_advection2d():
            # Advection case
            if case is None:
                case = SolidRotationTestCase(para)
            elif not isinstance(case, AbstractAdvectionEquation2DTestCase):
                raise ValueError(
                    "Invalid test case for advection equation"
                )
            block = advection2d_case_init_fn(
                time_in_seconds=jnp.array(0.0, dtype=coord.dev.float_type),
                coord=coord,
                para=para,
                solver=solver,
                test_case=case
            )
        elif mode.is_barotropic():
            # Barotropic case
            if case is None:
                case = MountainZonalFlowTestCase(para)
            elif not isinstance(case, AbstractShallowWaterEquationTestCase):
                raise ValueError(
                    "Invalid test case for shallow water equation"
                )
            block = barotropic_case_init_fn(
                coord=coord,
                para=para,
                solver=solver,
                test_case=case
            )
        else:
            raise ValueError(
                f"Invalid equation mode for testcase initialization. Invalid mode={equation.mode}"
            )
        # =======================================================
        block = prepare_block_before_integration(block)
        block = solver.equation.prepare_block(block)
        # =======================================================
        log.logger.success("Initialized block from testcase.")
        return block


def prepare_block_before_integration(block: Block) -> Block:
    if block.mode.is_advection2d():
        # ===============================================
        # Background wind and mass
        m_old = block.state.dmg.unwrap()[block.old, ...]
        un_old = block.state.un.unwrap()[block.old, ...]
        block = eqx.tree_at(
            lambda b: (
                b.adv_batch_bg.m.data,
                b.adv_batch_bg.un.data
            ),
            block,
            (m_old, un_old),
        )
        # ===============================================
        # Tracer mass
        q_old = block.tracers.q.unwrap()[block.old, ...]
        mq_old = m_old[None, ...] * q_old
        block = tree_at(
            coord=block.coord,
            where=lambda b: b.tracers.mq,
            pytree=block,
            replace=mq_old,
            at_dims=0,
            at_idxes=block.old
        )
        # ===============================================
        # Wind and mass for all batches
        for batch_name in block.adv_batches.keys():
            block = eqx.tree_at(
                lambda b: (
                    b.adv_batches[batch_name].m.data,
                    b.adv_batches[batch_name].un.data,
                ),
                block,
                (m_old, un_old),
            )
        # ===============================================
    return block
