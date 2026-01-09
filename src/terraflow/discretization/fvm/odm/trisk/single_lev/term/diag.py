from typing import overload, Literal
from numpy import var
import xarray as xr

from terraflow.util import typing, uxarray as tux
from terraflow.mesh import OrthogonalDualMesh
from terraflow.vert_coord import OneLayerVertCoordinate
from terraflow.layout import Field, StaggeredFieldMetaData
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
import terraflow.dynamics.shallow_water.vars.unstructure as vars
from terraflow.discretization.abstract import AbstractDiagnosticTools
from terraflow.discretization.abstract import DiagnosticVariableSet
from terraflow.discretization.abstract import FiniteVolumeMethod
from terraflow.dynamics.shallow_water.types import (
    ExVifShallowWaterEquationSet4UGRID
)
from terraflow.discretization.fvm.odm.trisk.single_lev.term.builder import (
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
        ],
        vars.StateFields,
        StaggeredFieldMetaData
    ]
):
    u_pc_option: bool
    v_pc_option: bool
    h_pe_option: bool
    h_de_option: bool
    mfx_pe_option: bool
    rel_vor_dc_option: bool
    abs_vor_dc_option: bool
    pv_dc_option: bool
    pv_de_option: bool
    ke_pc_option: bool
    dhdt_mfx_option: bool
    dundt_coriolis_option: bool
    dundt_grad_ke_option: bool
    dundt_pgf_option: bool
    dundt_grad_term_option: bool
    dhdt_option: bool
    dundt_option: bool

    def __init__(
        self,
        equation: ExVifShallowWaterEquationSet4UGRID[
            DivergenceOfMassFluxAtPrimalCell,
            NonlinearCoriolisTermAtDualEdge,
            GradientOfKineticEnergyAtDualEdge,
            PressureGradientForceAtDualEdge,
            OrthogonalDualMesh,
            OneLayerVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            FiniteVolumeMethod
        ],
        u_pc: bool = True,
        v_pc: bool = True,
        h_pe: bool = False,
        h_de: bool = False,
        mfx_pe: bool = False,
        rel_vor_dc: bool = False,
        abs_vor_dc: bool = False,
        pv_dc: bool = False,
        pv_de: bool = False,
        ke_pc: bool = False,
        dhdt_mfx: bool = False,
        dundt_coriolis: bool = False,
        dundt_grad_ke: bool = False,
        dundt_pgf: bool = False,
        dundt_grad_term: bool = False,
        dhdt: bool = False,
        dundt: bool = False,
        jit: bool = True,
        trajectory: bool = False
    ):
        super().__init__(
            equation, jit=jit, trajectory=trajectory
        )
        self.u_pc_option = u_pc
        self.v_pc_option = v_pc
        self.h_pe_option = h_pe
        self.h_de_option = h_de
        self.mfx_pe_option = mfx_pe
        self.rel_vor_dc_option = rel_vor_dc
        self.abs_vor_dc_option = abs_vor_dc
        self.pv_dc_option = pv_dc
        self.pv_de_option = pv_de
        self.ke_pc_option = ke_pc
        self.dhdt_mfx_option = dhdt_mfx
        self.dundt_coriolis_option = dundt_coriolis
        self.dundt_grad_ke_option = dundt_grad_ke
        self.dundt_pgf_option = dundt_pgf
        self.dundt_grad_term_option = dundt_grad_term
        self.dhdt_option = dhdt
        self.dundt_option = dundt

    @overload
    def thickness_at_primal_cell(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    @overload
    def thickness_at_primal_cell(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    def thickness_at_primal_cell(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
        """
        Get thickness at primal cell center.

        Args:
            states: DstateVariables containing the thickness field.

        Returns:
            Thickness at primal cell center as a QuantityField[StaggeredFieldMetaData].
        """
        h_pc = state.h_pc
        res = Field.create(
            data=h_pc.to_array().data,
            meta=StaggeredFieldMetaData(
                name=h_pc.meta.name,
                dims=self.process_dims(h_pc.meta.dims),
                long_name=h_pc.meta.long_name,
                description=h_pc.meta.description,
                units=h_pc.meta.units,
                hori_location=h_pc.meta.hori_location,
                vert_location=h_pc.meta.vert_location,
                basis=h_pc.meta.basis,
                coordinates=h_pc.meta.coordinates,
                staggered_kind=h_pc.meta.staggered_kind,
                mesh_name=h_pc.meta.mesh_name
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    @overload
    def normal_velocity_at_edge(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    @overload
    def normal_velocity_at_edge(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    def normal_velocity_at_edge(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
        """
        Get normal velocity at dual edge center.

        Args:
            states: DstateVariables containing the normal velocity field.

        Returns:
            Normal velocity at dual edge midpoint as a QuantityField[StaggeredFieldMetaData].
        """
        un_e = state.un_e
        res = Field.create(
            data=un_e.to_array().data,
            meta=StaggeredFieldMetaData(
                name=un_e.meta.name,
                dims=self.process_dims((un_e.meta.dims)),
                long_name=un_e.meta.long_name,
                description=un_e.meta.description,
                units=un_e.meta.units,
                hori_location=un_e.meta.hori_location,
                vert_location=un_e.meta.vert_location,
                basis=un_e.meta.basis,
                coordinates=un_e.meta.coordinates,
                staggered_kind=un_e.meta.staggered_kind,
                mesh_name=un_e.meta.mesh_name
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    @overload
    def uv_at_primal_cell(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataset:
        ...

    @overload
    def uv_at_primal_cell(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> dict[str, Field]:
        ...

    def uv_at_primal_cell(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataset | dict[str, Field]:
        """
        Reconstruct uv vector at primal cell center from normal velocity at edge.

        Args:
            states: DstateVariables containing the normal velocity field.

        Returns:
            UV vector at primal cell center as a QuantityField[StaggeredFieldMetaData].
        """
        import terraflow.mesh.orthogonal_dual_mesh.construct as odm
        states_array = state.to_arrays()
        uv = self.process_fn(
            odm.reconstruct_vector_at_primal_cell_from_dual_edge,
            in_axes=(None, 0),
            out_axes=({"u": 0, "v": 0})
        )(self.equation.mesh, states_array.un_e)
        u = Field.create(
            data=uv["u"],
            meta=StaggeredFieldMetaData(
                name="u_pc",
                dims=self.process_dims(("nPrimal_Cell",)),
                long_name="Zonal velocity at primal cell center",
                description="Zonal velocity at primal cell center reconstructed from normal velocity at dual edge",
                units="m/s",
                hori_location="cell",
                vert_location="",
                basis="x",
                coordinates="lon_pc lat_pc",
                staggered_kind="C",
                mesh_name="primal"
            ),
            as_quantity=False
        )
        v = Field.create(
            data=uv["v"],
            meta=StaggeredFieldMetaData(
                name="v_pc",
                dims=self.process_dims(("nPrimal_Cell",)),
                long_name="Meridional velocity at primal cell center",
                description="Meridional velocity at primal cell center reconstructed from normal velocity at dual edge",
                units="m/s",
                hori_location="cell",
                vert_location="",
                basis="y",
                coordinates="lon_pc lat_pc",
                staggered_kind="C",
                mesh_name="primal"
            ),
            as_quantity=False
        )
        if to_ugrid:
            u_ds = u.to_xarray_with_ugrid_format()
            v_ds = v.to_xarray_with_ugrid_format()
            ds = xr.Dataset(
                data_vars={
                    "u_pc": u_ds,
                    "v_pc": v_ds
                }
            )
            uxgrid = self.coord.mesh.to_ugrid().to_uxgrid()
            return tux.UGRIDDataset.from_xarray(ds, uxgrid)
        else:
            return {"u_pc": u, "v_pc": v}

    @overload
    def tangent_velocity_at_edge(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    @overload
    def tangent_velocity_at_edge(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    def tangent_velocity_at_edge(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
        """
        Calculate tangential velocity at dual edge center from normal velocity.

        Args:
            states: DstateVariables containing the normal velocity field.

        Returns:
            Tangential velocity at dual edge midpoint as a QuantityField[StaggeredFieldMetaData].
        """
        states_array = state.to_arrays()
        data = self.process_fn(
            lambda x: self.equation.dundt.coriolis(
                x, recon_tangent_velocity=True
            ),
            in_axes=(0,),
            out_axes=0
        )(states_array)
        res = Field.create(
            data=data,
            meta=StaggeredFieldMetaData(
                name="ut_e",
                dims=self.process_dims(("nDual_Edge",)),
                long_name="Tangential velocity at dual edge midpoint",
                description="Tangential velocity at dual edge midpoint calculated from normal velocity",
                units="m/s",
                hori_location="edge",
                vert_location="",
                basis="tangent",
                staggered_kind="C",
                coordinates="lon_de lat_de",
                mesh_name="dual"
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    @overload
    def thickness_at_primal_edge(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    @overload
    def thickness_at_primal_edge(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    def thickness_at_primal_edge(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
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
        res = Field.create(
            data=fn(state_array.h_pc, state_array.un_e),
            meta=StaggeredFieldMetaData(
                name="h_pe",
                dims=self.process_dims(("nDual_Edge",)),
                long_name="Thickness at primal edge midpoint",
                description="Thickness at primal edge midpoint calculated from thickness at primal cell center",
                units="m",
                hori_location="edge",
                vert_location="",
                basis="scalar",
                coordinates="lon_de lat_de",
                staggered_kind="C",
                mesh_name="primal"
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    @overload
    def thickness_flux_at_primal_edge(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    @overload
    def thickness_flux_at_primal_edge(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    def thickness_flux_at_primal_edge(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
        state_array = state.to_arrays()
        fn = self.process_fn(
            self.equation.dundt.coriolis.flux_recon_at_primal_edge,
            in_axes=(0, 0),
            out_axes=0
        )
        h_pe = fn(state_array.h_pc, state_array.un_e)
        h_flux = h_pe * state_array.un_e
        res = Field.create(
            data=h_flux,
            meta=StaggeredFieldMetaData(
                name="h_pe_flux",
                dims=self.process_dims(("nDual_Edge",)),
                long_name="Thickness flux at primal edge midpoint",
                description="Thickness flux at primal edge midpoint calculated from thickness at primal cell center",
                units="m^2/s",
                hori_location="edge",
                vert_location="",
                basis="scalar",
                coordinates="lon_de lat_de",
                staggered_kind="C",
                mesh_name="primal"
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    @overload
    def rel_vorticity_at_dual_cell(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    @overload
    def rel_vorticity_at_dual_cell(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    def rel_vorticity_at_dual_cell(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
        nct = self.equation.dundt.coriolis
        fn = self.process_fn(
            nct.pv_at_dual_edge.abs_vor_at_dual_cell.relative_vorticity,
            in_axes=(0,),
            out_axes=0
        )
        state_array = state.to_arrays()
        res = Field.create(
            data=fn(state_array.un_e),
            meta=StaggeredFieldMetaData(
                name="rel_vor_dc",
                dims=self.process_dims(("nDual_Cell",)),
                long_name="Relative vorticity at dual cell center",
                description=(
                    "Relative vorticity at dual cell center calculated from "
                    "normal velocity at dual edge"
                ),
                units="1/s",
                hori_location="cell",
                vert_location="",
                basis="scalar",
                coordinates="lon_dc lat_dc",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    @overload
    def abs_vorticity_at_dual_cell(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    @overload
    def abs_vorticity_at_dual_cell(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    def abs_vorticity_at_dual_cell(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
        nct = self.equation.dundt.coriolis
        fn = self.process_fn(
            nct.pv_at_dual_edge.abs_vor_at_dual_cell,
            in_axes=(0,),
            out_axes=0
        )
        state_array = state.to_arrays()
        res = Field.create(
            data=fn(state_array.un_e),
            meta=StaggeredFieldMetaData(
                name="abs_vor_dc",
                dims=self.process_dims(("nDual_Cell",)),
                long_name="Absolute vorticity at dual cell center",
                description=(
                    "Absolute vorticity at dual cell center "
                    "calculated from normal velocity at dual edge"
                ),
                units="1/s",
                hori_location="cell",
                vert_location="",
                basis="scalar",
                coordinates="lon_dc lat_dc",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    @overload
    def thickness_at_dual_cell(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    @overload
    def thickness_at_dual_cell(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    def thickness_at_dual_cell(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
        nct = self.equation.dundt.coriolis
        fn = self.process_fn(
            nct.pv_at_dual_edge.thickness_at_dual_cell,
            in_axes=(0,),
            out_axes=0
        )
        state_array = state.to_arrays()
        res = Field.create(
            data=fn(state_array.h_pc),
            meta=StaggeredFieldMetaData(
                name="h_dc",
                dims=self.process_dims(("nDual_Cell",)),
                long_name="Thickness at dual cell center",
                description=(
                    "Thickness at dual cell center calculated from "
                    "thickness at primal cell center"
                ),
                units="m",
                hori_location="cell",
                vert_location="",
                basis="scalar",
                coordinates="lon_dc lat_dc",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    @overload
    def pv_at_dual_cell(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    @overload
    def pv_at_dual_cell(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    def pv_at_dual_cell(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
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
        res = Field.create(
            data=fn(abs_vor_at_dual_cell, thickness_at_dual_cell),
            meta=StaggeredFieldMetaData(
                name="pv_dc",
                dims=self.process_dims(("nDual_Cell",)),
                long_name="Potential vorticity at dual cell center",
                description=(
                    "Potential vorticity at dual cell center calculated from "
                    "absolute vorticity and thickness at dual cell center"
                ),
                units="1/(m·s)",
                hori_location="cell",
                vert_location="",
                basis="scalar",
                coordinates="lon_dc lat_dc",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    @overload
    def pv_at_dual_edge(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    @overload
    def pv_at_dual_edge(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    def pv_at_dual_edge(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
        fn = self.process_fn(
            self.equation.dundt.coriolis.pv_at_dual_edge,
            in_axes=(0, 0, 0),
            out_axes=0
        )
        ut = self.tangent_velocity_at_edge(state).data
        state_array = state.to_arrays()
        res = Field.create(
            data=fn(state_array.un_e, ut, state_array.h_pc),
            meta=StaggeredFieldMetaData(
                name="pv_de",
                dims=self.process_dims(("nDual_Edge",)),
                long_name="Potential vorticity at dual edge center",
                description=(
                    "Potential vorticity at dual edge center calculated from "
                    "absolute vorticity and thickness at dual edge center"
                ),
                units="1/(m·s)",
                hori_location="edge",
                vert_location="",
                basis="scalar",
                coordinates="lon_de lat_de",
                staggered_kind="E",
                mesh_name="dual"
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    @overload
    def ke_at_primal_cell(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    @overload
    def ke_at_primal_cell(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    def ke_at_primal_cell(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
        state_array = state.to_arrays()
        fn = self.process_fn(
            self.equation.dundt.grad_ke.ke_at_primal_cell,
            in_axes=(0,),
            out_axes=0
        )
        res = Field.create(
            data=fn(state_array.un_e),
            meta=StaggeredFieldMetaData(
                name="ke_pc",
                dims=self.process_dims(("nPrimal_Cell",)),
                long_name="Kinetic energy at primal cell center",
                description="Kinetic energy at primal cell center calculated from normal velocity at dual edge",
                units="m²/s²",
                hori_location="cell",
                vert_location="",
                basis="scalar",
                coordinates="lon_pc lat_pc",
                staggered_kind="C",
                mesh_name="primal"
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    @overload
    def dhdt(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    @overload
    def dhdt(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    def dhdt(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
        state_array = state.to_arrays()
        fn = self.process_fn(
            self.equation.dhdt,
            in_axes=(0,),
            out_axes=0
        )
        res = Field.create(
            data=fn(state_array),
            meta=StaggeredFieldMetaData(
                name="dhdt",
                dims=self.process_dims(("nPrimal_Cell",)),
                long_name="Tendency of thickness",
                description="Tendency of thickness calculated from state variables",
                units="m/s",
                hori_location="cell",
                vert_location="",
                basis="scalar",
                coordinates="lon_pc lat_pc",
                staggered_kind="C",
                mesh_name="primal"
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    @overload
    def dhdt_mfx(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    @overload
    def dhdt_mfx(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    def dhdt_mfx(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
        state_array = state.to_arrays()
        fn = self.process_fn(
            self.equation.dhdt.div_mfx,
            in_axes=(0,),
            out_axes=0
        )
        res = Field.create(
            data=-fn(state_array),
            meta=StaggeredFieldMetaData(
                name="dhdt_div_h",
                dims=self.process_dims(("nPrimal_Cell",)),
                long_name="Tendency of thickness due to divergence term",
                description="Tendency of thickness due to divergence term calculated from state variables",
                units="1/s",
                hori_location="cell",
                vert_location="",
                basis="scalar",
                coordinates="lon_pc lat_pc",
                staggered_kind="C",
                mesh_name="primal"
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    @overload
    def dundt(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    @overload
    def dundt(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    def dundt(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
        static_array = self.equation.static
        state_array = state.to_arrays()
        fn = self.process_fn(
            self.equation.dundt,
            in_axes=(0, None),
            out_axes=0
        )
        res = Field.create(
            data=fn(state_array, static_array),
            meta=StaggeredFieldMetaData(
                name="dundt",
                dims=self.process_dims(("nDual_Edge",)),
                long_name="Tendency of normal velocity",
                description="Tendency of normal velocity calculated from state variables",
                units="m/s²",
                hori_location="edge",
                vert_location="",
                basis="normal",
                coordinates="lon_de lat_de",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    @overload
    def dundt_coriolis(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    @overload
    def dundt_coriolis(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    def dundt_coriolis(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
        state_array = state.to_arrays()
        fn = self.process_fn(
            self.equation.dundt.coriolis,
            in_axes=(0,),
            out_axes=0
        )
        res = Field.create(
            data=-fn(state_array),
            meta=StaggeredFieldMetaData(
                name="dundt_coriolis",
                dims=self.process_dims(("nDual_Edge",)),
                long_name="Tendency of normal velocity due to Coriolis term",
                description="Tendency of normal velocity due to Coriolis term calculated from state variables",
                units="m/s²",
                hori_location="edge",
                vert_location="",
                basis="scalar",
                coordinates="lon_de lat_de",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    @overload
    def dundt_grad_ke(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    @overload
    def dundt_grad_ke(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    def dundt_grad_ke(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
        state_array = state.to_arrays()
        fn = self.process_fn(
            self.equation.dundt.grad_ke,
            in_axes=(0,),
            out_axes=0
        )
        res = Field.create(
            data=-fn(state_array),
            meta=StaggeredFieldMetaData(
                name="dundt_grad_ke",
                dims=self.process_dims(("nDual_Edge",)),
                long_name="Tendency of normal velocity due to gradient of kinetic energy term",
                description="Tendency of normal velocity due to gradient of kinetic energy term calculated from state variables",
                units="m/s²",
                hori_location="edge",
                vert_location="",
                basis="scalar",
                coordinates="lon_de lat_de",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    @overload
    def dundt_pgf(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    @overload
    def dundt_pgf(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    def dundt_pgf(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
        state_array = state.to_arrays()
        static_array = self.equation.static
        fn = self.process_fn(
            self.equation.dundt.pgf,
            in_axes=(0, None),
            out_axes=0
        )
        res = Field.create(
            data=-fn(state_array, static_array),
            meta=StaggeredFieldMetaData(
                name="dundt_pgf",
                dims=self.process_dims(("nDual_Edge",)),
                long_name="Tendency of normal velocity due to pressure gradient force term",
                description="Tendency of normal velocity due to pressure gradient force term calculated from state variables",
                units="m/s²",
                hori_location="edge",
                vert_location="",
                basis="scalar",
                coordinates="lon_de lat_de",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    @overload
    def dundt_grad_term(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[True] = True
    ) -> tux.UGRIDDataArray:
        ...

    @overload
    def dundt_grad_term(
        self,
        state: vars.StateFields,
        to_ugrid: Literal[False]
    ) -> Field:
        ...

    def dundt_grad_term(
        self,
        state: vars.StateFields,
        to_ugrid: bool = True
    ) -> tux.UGRIDDataArray | Field:
        state_array = state.to_arrays()
        static_array = self.equation.static
        grad_ke = -self.equation.dundt.grad_ke(state_array)
        grad_phi = -self.equation.dundt.pgf(
            state_array, static_array
        )
        res = Field.create(
            data=grad_ke + grad_phi,
            meta=StaggeredFieldMetaData(
                name="dundt_grad_term",
                dims=self.process_dims(("nDual_Edge",)),
                long_name="Tendency of normal velocity due to gradient term",
                description="Tendency of normal velocity due to gradient calculated from state variables",
                units="m/s²",
                hori_location="edge",
                vert_location="",
                basis="scalar",
                coordinates="lon_de lat_de",
                staggered_kind="C",
                mesh_name="dual"
            ),
            as_quantity=False
        )
        if to_ugrid:
            return res.to_ugrid(self.equation.coord)
        else:
            return res

    def compute(
        self,
        states: vars.StateFields | vars.StateArrays
    ) -> DiagnosticVariableSet[StaggeredFieldMetaData]:
        if isinstance(states, vars.StateArrays):
            states_fields = states.to_fields(as_quantity=False)
        else:
            states_fields = states
        variables: dict[str, Field] = {}
        if self.u_pc_option or self.v_pc_option:
            uv_pc = self.uv_at_primal_cell(
                states_fields,
                to_ugrid=False
            )
            if self.u_pc_option:
                variables["u_pc"] = uv_pc["u_pc"]
            if self.v_pc_option:
                variables["v_pc"] = uv_pc["v_pc"]
        if self.h_pe_option:
            variables["h_pe"] = self.thickness_at_primal_edge(
                states_fields,
                to_ugrid=False
            )
        if self.h_de_option:
            variables["h_de"] = self.thickness_at_dual_cell(
                states_fields,
                to_ugrid=False
            )
        if self.mfx_pe_option:
            variables["mfx_pe"] = self.thickness_flux_at_primal_edge(
                states_fields,
                to_ugrid=False
            )
        if self.rel_vor_dc_option:
            variables["rel_vor_dc"] = self.rel_vorticity_at_dual_cell(
                states_fields,
                to_ugrid=False
            )
        if self.abs_vor_dc_option:
            variables["abs_vor_dc"] = self.abs_vorticity_at_dual_cell(
                states_fields,
                to_ugrid=False
            )
        if self.pv_dc_option:
            variables["pv_dc"] = self.pv_at_dual_cell(
                states_fields,
                to_ugrid=False
            )
        if self.pv_de_option:
            variables["pv_de"] = self.pv_at_dual_edge(
                states_fields,
                to_ugrid=False
            )
        if self.ke_pc_option:
            variables["ke_pc"] = self.ke_at_primal_cell(
                states_fields,
                to_ugrid=False
            )
        if self.dhdt_mfx_option:
            variables["dhdt_mfx"] = self.dhdt_mfx(
                states_fields,
                to_ugrid=False
            )
        if self.dundt_coriolis_option:
            variables["dundt_coriolis"] = self.dundt_coriolis(
                states_fields,
                to_ugrid=False
            )
        if self.dundt_grad_ke_option:
            variables["dundt_grad_ke"] = self.dundt_grad_ke(
                states_fields,
                to_ugrid=False
            )
        if self.dundt_pgf_option:
            variables["dundt_pgf"] = self.dundt_pgf(
                states_fields,
                to_ugrid=False
            )
        if self.dundt_grad_term_option:
            variables["dundt_grad_term"] = self.dundt_grad_term(
                states_fields,
                to_ugrid=False
            )
        if self.dhdt_option:
            variables["dhdt"] = self.dhdt(
                states_fields,
                to_ugrid=False
            )
        if self.dundt_option:
            variables["dundt"] = self.dundt(
                states_fields,
                to_ugrid=False
            )
        return DiagnosticVariableSet(variables=variables)
