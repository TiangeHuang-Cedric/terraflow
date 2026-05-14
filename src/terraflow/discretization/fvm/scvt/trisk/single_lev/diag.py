from __future__ import annotations
from typing import (
    overload,
    Literal,
    Dict,
    Mapping,
    TYPE_CHECKING
)
import jax
import jax.numpy as jnp

from terraflow.util import typing, uxarray as tuxr
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
from terraflow.layout.abstract import (
    CoordinateSystem,
    MeshEntity,
    Basis
)
from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
from terraflow.layout.field import Field, StaggeredFieldMetaData
from terraflow.dynamics.shallow_water.types import ExVifShallowWaterEquationSet4UGRID
from terraflow.discretization.abstract import (
    AbstractDiagnosticTools,
)
from .term import ExSpaceOperators
from ..tracer import Tracer

if TYPE_CHECKING:
    from . import dynamics_types as vars
    from ..base_types import (
        State,
        Tendency,
        Static,
        Auxiliary,
        EquationTermAuxDataset
    )
    from terraflow.planet import PlanetParameterSet
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    from terraflow.solver.abstract import AbstractSolver
    Array = typing.Array

Coordsys = CoordinateSystem[
    SphericalCentroidVoronoiTessellations,
    OneLayerVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    OneLayerVertCoordLayout
]


Equation = ExVifShallowWaterEquationSet4UGRID[
    FiniteVolumeMethod,
    ExSpaceOperators,
]


class DiagnosticsTools(
    AbstractDiagnosticTools[
        SphericalCentroidVoronoiTessellations,
        OneLayerVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        OneLayerVertCoordLayout,
        FiniteVolumeMethod,
        Equation,
        StaggeredFieldMetaData
    ]
):

    def __init__(
        self,
        solver: AbstractSolver,
        jit: bool = True,
        trajectory: bool = False
    ):
        super().__init__(
            solver=solver,
            jit=jit,
            trajectory=trajectory
        )

    @staticmethod
    def cast_state(state) -> State:
        return state

    @staticmethod
    def cast_block(block) -> vars.Block:
        return block

    def uv_at_primal_cell(
        self,
        coord: Coordsys,
        state: State,
    ) -> tuple[Field, Field]:
        uv = coord.mesh.reconstruct_vector(state.un.unwrap()[0, :, :])
        u = Field.create(
            data=uv["u"],
            meta=StaggeredFieldMetaData(
                name="u_cell",
                dims=("ncell",),
                long_name="Zonal velocity at primal cell center",
                description="Zonal velocity at primal cell center reconstructed from normal velocity at dual edge",
                units="m/s",
                hori_location=MeshEntity.Cell,
                vert_location="",
                basis=Basis.NormalX,
                coordinates="lon_cell lat_cell",
                staggered_kind="C",
                mesh_name="primal"
            ),
            as_quantity=False
        )
        v = Field.create(
            data=uv["v"],
            meta=StaggeredFieldMetaData(
                name="v_cell",
                dims=("ncell",),
                long_name="Meridional velocity at primal cell center",
                description="Meridional velocity at primal cell center reconstructed from normal velocity at dual edge",
                units="m/s",
                hori_location=MeshEntity.Cell,
                vert_location="",
                basis=Basis.NormalY,
                coordinates="lon_cell lat_cell",
                staggered_kind="C",
                mesh_name="primal"
            ),
            as_quantity=False
        )
        return u, v

    @overload
    def compute(
        self,
        coord: CoordinateSystem,
        para: PlanetParameterSet,
        state: State,
        static: Static,
        equaux: EquationTermAuxDataset,
        to_ugrid: Literal[True],
        use_dual: bool = False,
    ) -> Dict[str, tuxr.UGRIDDataArray]:
        ...

    @overload
    def compute(
        self,
        coord: CoordinateSystem,
        para: PlanetParameterSet,
        state: State,
        static: Static,
        equaux: EquationTermAuxDataset,
        to_ugrid: Literal[False] = False,
        use_dual: bool = False,
    ) -> Dict[str, Field[Array, StaggeredFieldMetaData]]:
        ...

    def compute(
        self,
        coord: Coordsys,
        para: PlanetParameterSet,
        state: State,
        static: Static,
        equaux: EquationTermAuxDataset,
        to_ugrid: bool = False,
        use_dual: bool = False,
    ) -> Mapping[
        str,
        Field[Array, StaggeredFieldMetaData] | tuxr.UGRIDDataArray
    ]:
        from . import dynamics_types as vars
        from ..base_types import (
            Tendency,
            Auxiliary,
        )
        from ..common.adv.batch import AdvBatchType
        state = self.expand_dims_on_state(state)
        dt_dyn = self.solver.dt_dyn.total_seconds()
        block = vars.Block(
            elapsed_seconds=jnp.array(0),
            mode=self.equation.mode,
            _coord=coord,
            para=para,
            old=jnp.array(0),
            new=jnp.array(1),
            state=state,
            static=static,
            tend=Tendency.from_coord(
                coord=coord,
                equation=self.equation
            ),
            aux=Auxiliary.from_coord(
                coord=coord,
                equation=self.equation
            ),
            adv_batch_bg=AdvBatchType.from_coord(
                name="bg",
                batch_hori_loc=MeshEntity.Cell,
                batch_vert_loc="",
                coord=coord,
                equation=self.equation,
                dt=dt_dyn,
                dts_substep=jnp.zeros(()),
                dt_dyn=dt_dyn,
            ),
            adv_batch_pt=AdvBatchType.empty(mode=self.equation.mode),
            adv_batches={},
            tracers=Tracer.empty(mode=self.equation.mode),
            equaux=equaux
        )
        if self.jit:
            space_operators_jit = jax.jit(
                self.equation.space_operators,
                static_argnames=(
                    "old_idx",
                    "star_idx",
                    "new_idx",
                    "pass_tag",
                    "explicit",
                ),
            )
            block = space_operators_jit(
                block,
                old_idx=0,
                star_idx=0,
                new_idx=1,
                dt=jnp.array(dt_dyn),
                substep=0,
                pass_tag="all_pass",
                explicit=True
            )
        else:
            space_operators = self.equation.space_operators
            block = space_operators(
                block,
                old_idx=jnp.array(0),
                star_idx=jnp.array(0),
                new_idx=jnp.array(1),
                dt=jnp.array(dt_dyn),
                substep=0,
                pass_tag="all_pass",
                explicit=True
            )
        state = block.state
        static = block.static
        tend = block.tend
        aux = block.aux
        diag_vars = {}
        diag_vars["gz"] = self.process_field(
            coord=coord,
            field=state.gz[0, :, :],
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["un"] = self.process_field(
            coord=coord,
            field=state.un[0, :, :],
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["gzs"] = self.process_field(
            coord=coord,
            field=static.gzs,
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        u, v = self.uv_at_primal_cell(coord, state)
        diag_vars["u"] = self.process_field(
            coord=coord,
            field=u,
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["v"] = self.process_field(
            coord=coord,
            field=v,
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["ut"] = self.process_field(
            coord=coord,
            field=aux.ut,
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["mfn"] = self.process_field(
            coord=coord,
            field=aux.mfn,
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["vor"] = self.process_field(
            coord=coord,
            field=aux.vor,
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["pv_vtx"] = self.process_field(
            coord=coord,
            field=aux.pv_vtx,
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["pv_e"] = self.process_field(
            coord=coord,
            field=aux.pv_e,
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["ke"] = self.process_field(
            coord=coord,
            field=aux.ke,
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["dmg"] = self.process_field(
            coord=coord,
            field=state.dmg[0, :, :],
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["dmg_e"] = self.process_field(
            coord=coord,
            field=state.dmg_e[0, :, :],
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["dmg_vtx"] = self.process_field(
            coord=coord,
            field=state.dmg_vtx[0, :, :],
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["dmf"] = self.process_field(
            coord=coord,
            field=aux.dmf,
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["dgzdt_dmf"] = self.process_field(
            coord=coord,
            field=tend.dgzdt_dmf,
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["dundt_coriolis"] = self.process_field(
            coord=coord,
            field=tend.dundt_coriolis,
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["dundt_grad_ke"] = self.process_field(
            coord=coord,
            field=tend.dundt_grad_ke,
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["dundt_pgf"] = self.process_field(
            coord=coord,
            field=tend.dundt_pgf,
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["dgzdt"] = self.process_field(
            coord=coord,
            field=tend.dgzdt,
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        diag_vars["dundt"] = self.process_field(
            coord=coord,
            field=tend.dundt,
            to_ugrid=to_ugrid,
            use_dual=use_dual
        )
        return diag_vars

    @overload
    def compute_along_trajectory(
        self,
        coord: CoordinateSystem,
        para: PlanetParameterSet,
        trajectory: State,
        static: Static,
        equaux: EquationTermAuxDataset,
        to_ugrid: Literal[False] = False,
        use_dual: bool = False,
    ) -> Dict[str, Field[Array, StaggeredFieldMetaData]]:
        ...

    @overload
    def compute_along_trajectory(
        self,
        coord: CoordinateSystem,
        para: PlanetParameterSet,
        trajectory: State,
        static: Static,
        equaux: EquationTermAuxDataset,
        to_ugrid: Literal[True],
        use_dual: bool = False,
    ) -> Dict[str, tuxr.UGRIDDataArray]:
        ...

    def compute_along_trajectory(
        self,
        coord: CoordinateSystem,
        para: PlanetParameterSet,
        trajectory: State,
        static: Static,
        equaux: EquationTermAuxDataset,
        to_ugrid: bool = False,
        use_dual: bool = False,
    ) -> Mapping[
        str,
        Field[Array, StaggeredFieldMetaData] | tuxr.UGRIDDataArray
    ]:
        if self.jit:
            compute = jax.jit(
                self.compute,
                static_argnames=("to_ugrid", "use_dual")
            )
        else:
            compute = self.compute

        if to_ugrid:
            fields = jax.vmap(
                compute,
                in_axes=(None, None, 0, None, None, None, None),
                out_axes=0
            )(
                coord,
                para,
                trajectory,
                static,
                equaux,
                False,
                use_dual
            )
            uxgrid = coord.mesh.to_ugrid(use_dual=use_dual).to_uxgrid()
            return {
                name: tuxr.UGRIDDataArray.from_xarray(
                    da=field.to_xarray_with_ugrid_format(
                        use_dual=use_dual
                    ),
                    uxgrid=uxgrid
                )
                for name, field in fields.items()
            }

        return jax.vmap(
            compute,
            in_axes=(None, None, 0, None, None, None, None),
            out_axes=0
        )(
            coord,
            para,
            trajectory,
            static,
            equaux,
            to_ugrid,
            use_dual
        )
