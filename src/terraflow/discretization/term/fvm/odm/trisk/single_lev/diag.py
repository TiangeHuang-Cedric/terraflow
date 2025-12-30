import xarray as xr

from terraflow.util import typing, uxarray as tux
from terraflow.layout import Field, StaggeredFieldMetaData
from terraflow.discretization import AbstractDiagnosticTools
import terraflow.dynamics.shallow_water.vars.unstructure as vars
from terraflow.mesh import OrthogonalDualMesh
from terraflow.vert_coord import OneLayerVertCoordinate
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.discretization.operator import FiniteVolumeMethod
from terraflow.dynamics.shallow_water.types import (
    ExVifShallowWaterEquationSet4UGRID
)
from terraflow.discretization.term.fvm.odm.trisk.single_lev.builder import (
    DivergenceOfMassFluxAtPrimalCell,
    NonlinearCoriolisTermAtDualEdge,
    GradientOfKineticEnergyAtDualEdge,
    PressureGradientForceAtDualEdge,
)

Array = typing.Array


class DiagnosticsTools(
    AbstractDiagnosticTools[
        OrthogonalDualMesh,
        OneLayerVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        FiniteVolumeMethod,
        ExVifShallowWaterEquationSet4UGRID[
            DivergenceOfMassFluxAtPrimalCell,
            NonlinearCoriolisTermAtDualEdge,
            GradientOfKineticEnergyAtDualEdge,
            PressureGradientForceAtDualEdge,
            OrthogonalDualMesh,
            OneLayerVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            FiniteVolumeMethod
        ]
    ]
):
    def thickness_at_primal_cell(
        self,
        states: vars.StateFields
    ) -> tux.UGRIDDataArray:
        """
        Get thickness at primal cell center.

        Args:
            states: DstateVariables containing the thickness field.

        Returns:
            Thickness at primal cell center as a QuantityField[StaggeredFieldMetaData].
        """
        h_pc = states.h_pc
        return Field.create(
            data=h_pc.to_array().data,
            meta=StaggeredFieldMetaData(
                name=h_pc.meta.name,
                dims=self.process_dims(h_pc.meta.dims),
                standard_name=h_pc.meta.standard_name,
                description=h_pc.meta.description,
                units=h_pc.meta.units,
                hori_location=h_pc.meta.hori_location,
                vert_location=h_pc.meta.vert_location,
                basis=h_pc.meta.basis,
                staggered_kind=h_pc.meta.staggered_kind,
                mesh_name=h_pc.meta.mesh_name
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)

    def normal_velocity_at_edge(
        self,
        states: vars.StateFields
    ) -> tux.UGRIDDataArray:
        """
        Get normal velocity at dual edge center.

        Args:
            states: DstateVariables containing the normal velocity field.

        Returns:
            Normal velocity at dual edge midpoint as a QuantityField[StaggeredFieldMetaData].
        """
        un_e = states.un_e
        return Field.create(
            data=un_e.to_array().data,
            meta=StaggeredFieldMetaData(
                name=un_e.meta.name,
                dims=self.process_dims((un_e.meta.dims)),
                standard_name=un_e.meta.standard_name,
                description=un_e.meta.description,
                units=un_e.meta.units,
                hori_location=un_e.meta.hori_location,
                vert_location=un_e.meta.vert_location,
                basis=un_e.meta.basis,
                staggered_kind=un_e.meta.staggered_kind,
                mesh_name=un_e.meta.mesh_name
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)

    def uv_at_primal_cell(
        self,
        states: vars.StateFields
    ) -> tux.UGRIDDataset:
        """
        Reconstruct uv vector at primal cell center from normal velocity at edge.

        Args:
            states: DstateVariables containing the normal velocity field.

        Returns:
            UV vector at primal cell center as a QuantityField[StaggeredFieldMetaData].
        """
        import terraflow.mesh.orthogonal_dual_mesh.construct as odm
        states_array = states.to_arrays()
        uv = self.process_fn(
            odm.reconstruct_vector_at_primal_cell_from_dual_edge,
            in_axes=(None, 0),
            out_axes=({"u": 0, "v": 0})
        )(self.equation.mesh, states_array.un_e)
        u_ds = Field.create(
            data=uv["u"],
            meta=StaggeredFieldMetaData(
                name="u_pc",
                dims=self.process_dims(("nPrimal_cell",)),
                standard_name="Zonal velocity at primal cell center",
                description="Zonal velocity at primal cell center reconstructed from normal velocity at dual edge",
                units="m/s",
                hori_location="cell",
                vert_location="",
                basis="x",
                staggered_kind="C",
                mesh_name="primal"
            ),
            as_quantity=False
        ).to_xarray_with_ugrid_format()
        v_ds = Field.create(
            data=uv["v"],
            meta=StaggeredFieldMetaData(
                name="v_pc",
                dims=self.process_dims(("nPrimal_cell",)),
                standard_name="Meridional velocity at primal cell center",
                description="Meridional velocity at primal cell center reconstructed from normal velocity at dual edge",
                units="m/s",
                hori_location="cell",
                vert_location="",
                basis="y",
                staggered_kind="C",
                mesh_name="primal"
            ),
            as_quantity=False
        ).to_xarray_with_ugrid_format()
        ds = xr.Dataset(
            data_vars={
                "u_pc": u_ds,
                "v_pc": v_ds
            }
        )
        uxgrid = self.coord.mesh.to_ugrid().to_uxgrid()
        return tux.UGRIDDataset.from_xarray(ds, uxgrid)

    def tangent_velocity_at_edge(
        self,
        states: vars.StateFields
    ) -> tux.UGRIDDataArray:
        """
        Calculate tangential velocity at dual edge center from normal velocity.

        Args:
            states: DstateVariables containing the normal velocity field.

        Returns:
            Tangential velocity at dual edge midpoint as a QuantityField[StaggeredFieldMetaData].
        """
        states_array = states.to_arrays()
        data = self.process_fn(
            lambda x: self.equation.dundt.coriolis(
                x, recon_tangent_velocity=True
            ),
            in_axes=(0,),
            out_axes=0
        )(states_array)
        return Field.create(
            data=data,
            meta=StaggeredFieldMetaData(
                name="ut_e",
                dims=self.process_dims(("nDual_edge",)),
                standard_name="Tangential velocity at dual edge midpoint",
                description="Tangential velocity at dual edge midpoint calculated from normal velocity",
                units="m/s",
                hori_location="edge",
                vert_location="",
                basis="tangent",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)

    def thickness_at_primal_edge(
        self,
        state: vars.StateFields
    ) -> tux.UGRIDDataArray:
        """
        Calculate thickness at primal edge center from thickness at primal cell center.

        Args:
            states: DstateVariables containing the thickness field.

        Returns:
            Thickness at primal edge center as a QuantityField[StaggeredFieldMetaData].

        """
        state_array = state.to_arrays()
        fn = self.process_fn(
            self.equation.dhdt.div_mfx.flux_recon_at_primal_edge,
            in_axes=(0, 0),
            out_axes=0
        )
        return Field.create(
            data=fn(state_array.h_pc, state_array.un_e),
            meta=StaggeredFieldMetaData(
                name="h_pe",
                dims=self.process_dims(("nPrimal_edge",)),
                standard_name="Thickness at primal edge midpoint",
                description="Thickness at primal edge midpoint calculated from thickness at primal cell center",
                units="m",
                hori_location="edge",
                vert_location="",
                basis="scalar",
                staggered_kind="C",
                mesh_name="primal"
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)

    def thickness_flux_at_primal_edge(
        self,
        state: vars.StateFields
    ) -> tux.UGRIDDataArray:
        state_array = state.to_arrays()
        fn = self.process_fn(
            self.equation.dundt.coriolis.flux_recon_at_primal_edge,
            in_axes=(0, 0),
            out_axes=0
        )
        h_pe = fn(state_array.h_pc, state_array.un_e)
        h_flux = h_pe * state_array.un_e
        return Field.create(
            data=h_flux,
            meta=StaggeredFieldMetaData(
                name="h_pe_flux",
                dims=self.process_dims(("nPrimal_edge",)),
                standard_name="Thickness flux at primal edge midpoint",
                description="Thickness flux at primal edge midpoint calculated from thickness at primal cell center",
                units="m^2/s",
                hori_location="edge",
                vert_location="",
                basis="scalar",
                staggered_kind="C",
                mesh_name="primal"
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)

    def rel_vorticity_at_dual_cell(
        self,
        state: vars.StateFields
    ) -> tux.UGRIDDataArray:
        nct = self.equation.dundt.coriolis
        fn = self.process_fn(
            nct.pv_at_dual_edge.abs_vor_at_dual_cell.relative_vorticity,
            in_axes=(0,),
            out_axes=0
        )
        state_array = state.to_arrays()
        return Field.create(
            data=fn(state_array.un_e),
            meta=StaggeredFieldMetaData(
                name="rel_vor_dc",
                dims=self.process_dims(("nDual_cell",)),
                standard_name="Relative vorticity at dual cell center",
                description=(
                    "Relative vorticity at dual cell center calculated from "
                    "normal velocity at dual edge"
                ),
                units="1/s",
                hori_location="cell",
                vert_location="",
                basis="scalar",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)

    def abs_vorticity_at_dual_cell(
        self,
        state: vars.StateFields
    ) -> tux.UGRIDDataArray:
        nct = self.equation.dundt.coriolis
        fn = self.process_fn(
            nct.pv_at_dual_edge.abs_vor_at_dual_cell,
            in_axes=(0,),
            out_axes=0
        )
        state_array = state.to_arrays()
        return Field.create(
            data=fn(state_array.un_e),
            meta=StaggeredFieldMetaData(
                name="abs_vor_dc",
                dims=self.process_dims(("nDual_cell",)),
                standard_name="Absolute vorticity at dual cell center",
                description=(
                    "Absolute vorticity at dual cell center "
                    "calculated from normal velocity at dual edge"
                ),
                units="1/s",
                hori_location="cell",
                vert_location="",
                basis="scalar",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)

    def thickness_at_dual_cell(
        self,
        state: vars.StateFields
    ) -> tux.UGRIDDataArray:
        nct = self.equation.dundt.coriolis
        fn = self.process_fn(
            nct.pv_at_dual_edge.thickness_at_dual_cell,
            in_axes=(0,),
            out_axes=0
        )
        state_array = state.to_arrays()
        return Field.create(
            data=fn(state_array.h_pc),
            meta=StaggeredFieldMetaData(
                name="h_dc",
                dims=self.process_dims(("nDual_cell",)),
                standard_name="Thickness at dual cell center",
                description=(
                    "Thickness at dual cell center calculated from "
                    "thickness at primal cell center"
                ),
                units="m",
                hori_location="cell",
                vert_location="",
                basis="scalar",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)

    def pv_at_dual_cell(
        self,
        state: vars.StateFields
    ) -> tux.UGRIDDataArray:
        nct = self.equation.dundt.coriolis
        fn = self.process_fn(
            nct.pv_at_dual_edge.pv_at_dual_cell,
            in_axes=(0, 0),
            out_axes=0
        )
        abs_vor_at_dual_cell = self.abs_vorticity_at_dual_cell(
            state
        ).data
        thickness_at_dual_cell = self.thickness_at_dual_cell(
            state
        ).data
        return Field.create(
            data=fn(abs_vor_at_dual_cell, thickness_at_dual_cell),
            meta=StaggeredFieldMetaData(
                name="pv_dc",
                dims=self.process_dims(("nDual_cell",)),
                standard_name="Potential vorticity at dual cell center",
                description=(
                    "Potential vorticity at dual cell center calculated from "
                    "absolute vorticity and thickness at dual cell center"
                ),
                units="1/(m·s)",
                hori_location="cell",
                vert_location="",
                basis="scalar",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)

    def pv_at_dual_edge(
        self,
        state: vars.StateFields
    ) -> tux.UGRIDDataArray:
        fn = self.process_fn(
            self.equation.dundt.coriolis.pv_at_dual_edge,
            in_axes=(0, 0),
            out_axes=0
        )
        state_array = state.to_arrays()
        return Field.create(
            data=fn(state_array.un_e, state_array.h_pc),
            meta=StaggeredFieldMetaData(
                name="pv_de",
                dims=self.process_dims(("nDual_edge",)),
                standard_name="Potential vorticity at dual edge center",
                description=(
                    "Potential vorticity at dual edge center calculated from "
                    "absolute vorticity and thickness at dual edge center"
                ),
                units="1/(m·s)",
                hori_location="edge",
                vert_location="",
                basis="scalar",
                staggered_kind="E",
                mesh_name="dual"
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)

    def ke_at_primal_cell(
        self,
        state: vars.StateFields
    ) -> tux.UGRIDDataArray:
        state_array = state.to_arrays()
        fn = self.process_fn(
            self.equation.dundt.grad_ke.ke_at_primal_cell,
            in_axes=(0,),
            out_axes=0
        )
        return Field.create(
            data=fn(state_array.un_e),
            meta=StaggeredFieldMetaData(
                name="ke_pc",
                dims=self.process_dims(("nPrimal_cell",)),
                standard_name="Kinetic energy at primal cell center",
                description="Kinetic energy at primal cell center calculated from normal velocity at dual edge",
                units="m²/s²",
                hori_location="cell",
                vert_location="",
                basis="scalar",
                staggered_kind="C",
                mesh_name="primal"
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)

    def dhdt(
        self,
        state: vars.StateFields
    ) -> tux.UGRIDDataArray:
        state_array = state.to_arrays()
        fn = self.process_fn(
            self.equation.dhdt,
            in_axes=(0,),
            out_axes=0
        )
        return Field.create(
            data=fn(state_array),
            meta=StaggeredFieldMetaData(
                name="dhdt",
                dims=self.process_dims(("nPrimal_cell",)),
                standard_name="Tendency of thickness",
                description="Tendency of thickness calculated from state variables",
                units="m/s",
                hori_location="cell",
                vert_location="",
                basis="scalar",
                staggered_kind="C",
                mesh_name="primal"
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)

    def dhdt_mfx(
        self,
        state: vars.StateFields
    ) -> tux.UGRIDDataArray:
        state_array = state.to_arrays()
        fn = self.process_fn(
            self.equation.dhdt.div_mfx,
            in_axes=(0,),
            out_axes=0
        )
        return Field.create(
            data=-fn(state_array),
            meta=StaggeredFieldMetaData(
                name="dhdt_div_h",
                dims=self.process_dims(("nPrimal_cell",)),
                standard_name="Tendency of thickness due to divergence term",
                description="Tendency of thickness due to divergence term calculated from state variables",
                units="1/s",
                hori_location="cell",
                vert_location="",
                basis="scalar",
                staggered_kind="C",
                mesh_name="primal"
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)

    def dundt(
        self,
        state: vars.StateFields
    ) -> tux.UGRIDDataArray:
        static_array = self.equation.static
        state_array = state.to_arrays()
        fn = self.process_fn(
            self.equation.dundt,
            in_axes=(0, None),
            out_axes=0
        )
        return Field.create(
            data=fn(state_array, static_array),
            meta=StaggeredFieldMetaData(
                name="dundt",
                dims=self.process_dims(("nDual_edge",)),
                standard_name="Tendency of normal velocity",
                description="Tendency of normal velocity calculated from state variables",
                units="m/s²",
                hori_location="edge",
                vert_location="",
                basis="normal",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)

    def dundt_coriolis(
        self,
        state: vars.StateFields
    ) -> tux.UGRIDDataArray:
        state_array = state.to_arrays()
        fn = self.process_fn(
            self.equation.dundt.coriolis,
            in_axes=(0,),
            out_axes=0
        )
        return Field.create(
            data=-fn(state_array),
            meta=StaggeredFieldMetaData(
                name="dundt_coriolis",
                dims=self.process_dims(("nDual_edge",)),
                standard_name="Tendency of normal velocity due to Coriolis term",
                description="Tendency of normal velocity due to Coriolis term calculated from state variables",
                units="m/s²",
                hori_location="edge",
                vert_location="",
                basis="scalar",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)

    def dundt_grad_ke(
        self,
        state: vars.StateFields
    ) -> tux.UGRIDDataArray:
        state_array = state.to_arrays()
        fn = self.process_fn(
            self.equation.dundt.grad_ke,
            in_axes=(0,),
            out_axes=0
        )
        return Field.create(
            data=-fn(state_array),
            meta=StaggeredFieldMetaData(
                name="dundt_grad_ke",
                dims=self.process_dims(("nDual_edge",)),
                standard_name="Tendency of normal velocity due to gradient of kinetic energy term",
                description="Tendency of normal velocity due to gradient of kinetic energy term calculated from state variables",
                units="m/s²",
                hori_location="edge",
                vert_location="",
                basis="scalar",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)

    def dundt_pgf(
        self,
        state: vars.StateFields
    ) -> tux.UGRIDDataArray:
        state_array = state.to_arrays()
        static_array = self.equation.static
        fn = self.process_fn(
            self.equation.dundt.pgf,
            in_axes=(0, None),
            out_axes=0
        )
        return Field.create(
            data=-fn(state_array, static_array),
            meta=StaggeredFieldMetaData(
                name="dundt_pgf",
                dims=self.process_dims(("nDual_edge",)),
                standard_name="Tendency of normal velocity due to pressure gradient force term",
                description="Tendency of normal velocity due to pressure gradient force term calculated from state variables",
                units="m/s²",
                hori_location="edge",
                vert_location="",
                basis="scalar",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)

    def dundt_grad_term(
        self,
        state: vars.StateFields
    ) -> tux.UGRIDDataArray:
        state_array = state.to_arrays()
        static_array = self.equation.static
        grad_ke = -self.equation.dundt.grad_ke(state_array)
        grad_phi = -self.equation.dundt.pgf(
            state_array, static_array
        )
        return Field.create(
            data=grad_ke + grad_phi,
            meta=StaggeredFieldMetaData(
                name="dundt_grad_term",
                dims=self.process_dims(("nDual_edge",)),
                standard_name="Tendency of normal velocity due to gradient term",
                description="Tendency of normal velocity due to gradient calculated from state variables",
                units="m/s²",
                hori_location="edge",
                vert_location="",
                basis="scalar",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        ).to_ugrid(self.equation.coord)
