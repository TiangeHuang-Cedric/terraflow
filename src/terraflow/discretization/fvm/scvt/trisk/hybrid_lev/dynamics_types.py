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
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
from terraflow.test_case.baroclinic import (
    AbstractBaroclinicEquationTestCase,
    SteadyStateTestCase,
    MountainWaveTestCase,
    RossbyHaurwitzWaveTestCase
)
from terraflow.test_case.adv import (
    AbstractAdvectionEquation3DTestCase,
    DCMIP12TestCase
)
from terraflow.discretization.abstract import TracerMetaData
from terraflow.discretization.fvm.scvt.trisk.tracer import Tracer
from terraflow.discretization.fvm.scvt.trisk.base_types import AbstractBaseBlock
from terraflow.discretization.fvm.scvt.trisk.test_case.baroclinic import (
    baroclinic_steady_state_case_init_fn,
    baroclinic_mountain_wave_case_init_fn,
    baroclinic_rossby_wave_case_init_fn
)
from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.prepare import (
    prepare_topo,
    prepare_bkg,
)
from terraflow.discretization.fvm.scvt.trisk.test_case.adv import dcmip12_case_init_fn


if TYPE_CHECKING:
    from terraflow.prepare.readers.abstract import (
        AbstractTopographyReader,
        AbstractBackgroundReader
    )
    from terraflow.planet import PlanetParameterSet
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.test_case.abstract import AbstractTestCase
    from terraflow.solver.abstract import AbstractSolver
    from terraflow.discretization.abstract import (
        EquationMode,
        AbstractEquationSet,
    )

    CoordSys = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        HybridVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        HybridVertCoordLayout
    ]
    ES = TypeVar("ES", bound=AbstractEquationSet)


class Block(AbstractBaseBlock[HybridVertCoordinate, HybridVertCoordLayout]):

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
        if mode.is_baroclinic():
            # Baroclinic case
            if case is None:
                case = SteadyStateTestCase(para)
            elif not isinstance(case, AbstractBaroclinicEquationTestCase):
                raise ValueError(
                    "Invalid test case for baroclinic equation"
                )
            if isinstance(case, SteadyStateTestCase):
                block = baroclinic_steady_state_case_init_fn(
                    coord=coord,
                    para=para,
                    solver=solver,
                    test_case=case
                )
            elif isinstance(case, MountainWaveTestCase):
                block = baroclinic_mountain_wave_case_init_fn(
                    coord=coord,
                    para=para,
                    solver=solver,
                    test_case=case
                )
            elif isinstance(case, RossbyHaurwitzWaveTestCase):
                block = baroclinic_rossby_wave_case_init_fn(
                    coord=coord,
                    para=para,
                    solver=solver,
                    test_case=case
                )
            else:
                raise NotImplementedError(
                    f"Initialization for case {type(case)} is not implemented yet."
                )
        elif mode.is_advection3d():
            # Advection case
            if case is None:
                case = DCMIP12TestCase(para)
            elif not isinstance(case, AbstractAdvectionEquation3DTestCase):
                raise ValueError(
                    "Invalid test case for 3D advection equation"
                )
            if isinstance(case, DCMIP12TestCase):
                block = dcmip12_case_init_fn(
                    time_in_seconds=jnp.array(
                        0.0,
                        dtype=coord.dev.float_type
                    ),
                    coord=coord,
                    para=para,
                    solver=solver,
                    test_case=case
                )
            else:
                raise NotImplementedError(
                    f"Initialization for case {type(case)} is not implemented yet."
                )
        else:
            raise ValueError(
                f"Invalid equation mode for testcase initialization. Invalid mode={equation.mode}"
            )
        # =======================================================
        block = prepare_block_before_integration(block=block)
        block = solver.equation.prepare_block(block)
        # =======================================================
        log.logger.success("Initialized block from testcase.")
        return block

    @classmethod
    def from_reader(
        cls,
        topo_reader: AbstractTopographyReader,
        bkg_reader: AbstractBackgroundReader,
        coord: CoordSys,
        para: PlanetParameterSet,
        solver: AbstractSolver[ES],
    ) -> Self:
        block = cls.from_model(
            coord=coord,
            para=para,
            solver=solver,
            tracers=Tracer.add_tracers(
                mode=solver.equation.mode,
                coord=coord,
                tracers_meta=TracerMetaData.from_reader(
                    mode=solver.equation.mode,
                    para=para,
                    reader=bkg_reader,
                    dt_adv=solver.dt_adv
                )
            )
        )
        # =======================================================
        block = prepare_topo(block=block, reader=topo_reader)
        block = prepare_bkg(block=block, reader=bkg_reader)
        # =======================================================
        block = prepare_block_before_integration(block=block)
        block = solver.equation.prepare_block(block)
        # =======================================================
        log.logger.success("Initialized block from reader.")
        return block


def prepare_block_before_integration(block: Block) -> Block:
    block = eqx.tree_at(
        lambda b: b.static.ref_ps.data,
        block,
        block.state.mgs.unwrap()[0, :]
    )
    return block
