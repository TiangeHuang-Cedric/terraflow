from typing import (
    overload,
    Literal,
    Dict,
    Mapping
)
import jax

from terraflow.util import typing, uxarray as tuxr
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh import OrthogonalDualMesh
from terraflow.vert_coord import OneLayerVertCoordinate
from terraflow.layout import (
    CoordinateSystem,
    Field,
    StaggeredFieldMetaData,
    MeshEntity,
    Basis
)
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
from terraflow.dynamics.shallow_water.types import ExVifShallowWaterEquationSet4UGRID
from terraflow.discretization.abstract import (
    AbstractDiagnosticTools,
)
import terraflow.discretization.fvm.odm.trisk.dynamics_types as vars
from terraflow.discretization.fvm.odm.trisk.adv.batch import AdvBatchType
from terraflow.discretization.fvm.odm.trisk.single_lev.term import (
    TendencyOfDivergenceOfMassFlux,
    NonlinearCoriolisTermAtDualEdge,
    GradientOfKineticEnergyAtDualEdge,
    PressureGradientForceAtDualEdge
)
from terraflow.discretization.fvm.odm.trisk.single_lev.aux import AuxiliaryTerms

Array = typing.Array


Coordsys = CoordinateSystem[
    OrthogonalDualMesh,
    OneLayerVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    OneLayerVertCoordLayout
]


Equation = ExVifShallowWaterEquationSet4UGRID[
    TendencyOfDivergenceOfMassFlux,
    NonlinearCoriolisTermAtDualEdge,
    GradientOfKineticEnergyAtDualEdge,
    PressureGradientForceAtDualEdge,
    AuxiliaryTerms,
    FiniteVolumeMethod,
]


class DiagnosticsTools(
    AbstractDiagnosticTools[
        OrthogonalDualMesh,
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
        equation: Equation,
        jit: bool = True,
        trajectory: bool = False
    ):
        super().__init__(
            equation, jit=jit, trajectory=trajectory
        )

    def uv_at_primal_cell(
        self,
        coord: Coordsys,
        state: vars.State,
    ) -> tuple[Field, Field]:
        uv = coord.mesh.reconstruct_vector(state.un.unwrap())
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
        coord: vars.CoordinateSystem,
        para: vars.PlanetParameterSet,
        state: vars.State,
        static: vars.Static,
        equaux: vars.EquationTermAuxDataset,
        to_ugrid: Literal[True],
        use_dual: bool = False,
    ) -> Dict[str, tuxr.UGRIDDataArray]:
        ...

    @overload
    def compute(
        self,
        coord: vars.CoordinateSystem,
        para: vars.PlanetParameterSet,
        state: vars.State,
        static: vars.Static,
        equaux: vars.EquationTermAuxDataset,
        to_ugrid: Literal[False] = False,
        use_dual: bool = False,
    ) -> Dict[str, Field[Array, StaggeredFieldMetaData]]:
        ...

    def compute(
        self,
        coord: Coordsys,
        para: vars.PlanetParameterSet,
        state: vars.State,
        static: vars.Static,
        equaux: vars.EquationTermAuxDataset,
        to_ugrid: bool = False,
        use_dual: bool = False,
    ) -> Mapping[
        str,
        Field[Array, StaggeredFieldMetaData] | tuxr.UGRIDDataArray
    ]:
        block = vars.Block(
            baroclinic=self.equation.baroclinic,
            hydrostatic=self.equation.hydrostatic,
            nonhydrostatic=self.equation.nonhydrostatic,
            ideal_dry_core=self.equation.ideal_dry_core,
            advection=self.equation.advection,
            coord=coord,
            para=para,
            state=state,
            static=static,
            tend=vars.Tendency.from_coord(coord=coord, equation=self.equation),
            aux=vars.Auxiliary.from_coord(coord=coord, equation=self.equation),
            adv_batch_pt=AdvBatchType.empty(coord=coord),
            equaux=equaux
        )
        if self.jit:
            aux_jit = jax.jit(self.equation.aux)
            explicit_terms_jit = jax.jit(self.equation.explicit_terms)
            block = aux_jit(block)
            block = explicit_terms_jit(block)
        else:
            block = self.equation.aux(block)
            block = self.equation.explicit_terms(block)
        state = block.state
        static = block.static
        tend = block.tend
        aux = block.aux
        diag_vars = {}
        diag_vars["gz"] = self.process_field(
            coord, state.gz, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["un"] = self.process_field(
            coord, state.un, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["gzs"] = self.process_field(
            coord, static.gzs, to_ugrid=to_ugrid, use_dual=use_dual
        )
        u, v = self.uv_at_primal_cell(coord, state)
        diag_vars["u"] = self.process_field(
            coord=coord, field=u, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["v"] = self.process_field(
            coord=coord, field=v, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["ut"] = self.process_field(
            coord=coord, field=aux.ut, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["mfn"] = self.process_field(
            coord=coord, field=aux.mfn, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["vor"] = self.process_field(
            coord=coord, field=aux.vor, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["pv_vtx"] = self.process_field(
            coord=coord, field=aux.pv_vtx, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["pv_e"] = self.process_field(
            coord=coord, field=aux.pv_e, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["ke"] = self.process_field(
            coord=coord, field=aux.ke, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["dmg"] = self.process_field(
            coord=coord, field=aux.dmg, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["dmg_e"] = self.process_field(
            coord=coord, field=aux.dmg_e, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["dmg_vtx"] = self.process_field(
            coord=coord, field=aux.dmg_vtx, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["dmf"] = self.process_field(
            coord=coord, field=aux.dmf, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["dgzdt_dmf"] = self.process_field(
            coord=coord, field=tend.dgzdt_dmf, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["dundt_coriolis"] = self.process_field(
            coord=coord, field=tend.dundt_coriolis, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["dundt_grad_ke"] = self.process_field(
            coord=coord, field=tend.dundt_grad_ke, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["dundt_pgf"] = self.process_field(
            coord=coord, field=tend.dundt_pgf, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["dgzdt"] = self.process_field(
            coord=coord, field=tend.dgzdt, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["dundt"] = self.process_field(
            coord=coord, field=tend.dundt, to_ugrid=to_ugrid, use_dual=use_dual
        )
        return diag_vars

    @overload
    def compute_along_trajectory(
        self,
        coord: vars.CoordinateSystem,
        para: vars.PlanetParameterSet,
        trajectory: vars.State,
        static: vars.Static,
        equaux: vars.EquationTermAuxDataset,
        to_ugrid: Literal[False] = False,
        use_dual: bool = False,
    ) -> Dict[str, Field[Array, StaggeredFieldMetaData]]:
        ...

    @overload
    def compute_along_trajectory(
        self,
        coord: vars.CoordinateSystem,
        para: vars.PlanetParameterSet,
        trajectory: vars.State,
        static: vars.Static,
        equaux: vars.EquationTermAuxDataset,
        to_ugrid: Literal[True],
        use_dual: bool = False,
    ) -> Dict[str, tuxr.UGRIDDataArray]:
        ...

    def compute_along_trajectory(
        self,
        coord: vars.CoordinateSystem,
        para: vars.PlanetParameterSet,
        trajectory: vars.State,
        static: vars.Static,
        equaux: vars.EquationTermAuxDataset,
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
