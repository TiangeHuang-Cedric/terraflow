from typing import overload, Literal, Dict, Mapping
import jax

from terraflow.util import typing, uxarray as tuxr
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh import OrthogonalDualMesh
from terraflow.vert_coord import OneLayerVertCoordinate
from terraflow.layout import Field, StaggeredFieldMetaData
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
import terraflow.discretization.fvm.odm.trisk.single_lev.dynamics_types as vars
from terraflow.discretization.abstract import (
    AbstractDiagnosticTools,
    DiagnosticVariableSet
)
from terraflow.dynamics.shallow_water.types import (
    ExVifShallowWaterEquationSet4UGRID
)
from terraflow.discretization.fvm.odm.trisk.single_lev.term import (
    DivergenceOfMassFluxAtPrimalCell,
    NonlinearCoriolisTermAtDualEdge,
    GradientOfKineticEnergyAtDualEdge,
    PressureGradientForceAtDualEdge,
    AuxiliaryTerms
)

Array = typing.Array

Equation = ExVifShallowWaterEquationSet4UGRID[
    DivergenceOfMassFluxAtPrimalCell,
    NonlinearCoriolisTermAtDualEdge,
    GradientOfKineticEnergyAtDualEdge,
    PressureGradientForceAtDualEdge,
    AuxiliaryTerms,
    OrthogonalDualMesh,
    OneLayerVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    FiniteVolumeMethod,
    vars.BlockArrays
]


class DiagnosticsTools(
    AbstractDiagnosticTools[
        OrthogonalDualMesh,
        OneLayerVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        FiniteVolumeMethod,
        Equation,
        StaggeredFieldMetaData
    ]
):
    u_pc_option: bool
    v_pc_option: bool
    ut_e_option: bool
    h_e_option: bool
    h_dc_option: bool
    mfx_e_option: bool
    rel_vor_dc_option: bool
    abs_vor_dc_option: bool
    pv_dc_option: bool
    pv_e_option: bool
    ke_pc_option: bool
    dhdt_mfx_option: bool
    dundt_coriolis_option: bool
    dundt_grad_ke_option: bool
    dundt_pgf_option: bool
    dhdt_option: bool
    dundt_option: bool

    def __init__(
        self,
        equation: Equation,
        u_pc: bool = True,
        v_pc: bool = True,
        ut_e: bool = False,
        h_e: bool = False,
        h_dc: bool = False,
        mfx_e: bool = False,
        rel_vor_dc: bool = False,
        abs_vor_dc: bool = False,
        pv_dc: bool = False,
        pv_e: bool = False,
        ke_pc: bool = False,
        dhdt_mfx: bool = False,
        dundt_coriolis: bool = False,
        dundt_grad_ke: bool = False,
        dundt_pgf: bool = False,
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
        self.ut_e_option = ut_e
        self.h_e_option = h_e
        self.h_dc_option = h_dc
        self.mfx_e_option = mfx_e
        self.rel_vor_dc_option = rel_vor_dc
        self.abs_vor_dc_option = abs_vor_dc
        self.pv_dc_option = pv_dc
        self.pv_e_option = pv_e
        self.ke_pc_option = ke_pc
        self.dhdt_mfx_option = dhdt_mfx
        self.dundt_coriolis_option = dundt_coriolis
        self.dundt_grad_ke_option = dundt_grad_ke
        self.dundt_pgf_option = dundt_pgf
        self.dhdt_option = dhdt
        self.dundt_option = dundt

    def uv_at_primal_cell(
        self,
        state: vars.StateFields | vars.StateArrays,
    ) -> tuple[Field, Field]:
        import terraflow.mesh.orthogonal_dual_mesh.construct as odm
        if not isinstance(state, vars.StateArrays):
            state = state.to_arrays()
        uv = odm.reconstruct_vector_at_primal_cell_from_dual_edge(
            self.equation.mesh, state.un_e
        )
        u = Field.create(
            data=uv["u"],
            meta=StaggeredFieldMetaData(
                name="u_pc",
                dims=("nPrimal_Cell",),
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
                dims=("nPrimal_Cell",),
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
        return u, v

    def to_diag(
        self,
        state: vars.StateFields | vars.StateArrays,
        tend: vars.TendencyFields | vars.TendencyArrays,
        aux: vars.AuxiliaryFields | vars.AuxiliaryArrays,
    ) -> DiagnosticVariableSet:
        if isinstance(state, vars.StateFields):
            state = state.to_arrays()
        if isinstance(tend, vars.TendencyFields):
            tend = tend.to_arrays()
        if isinstance(aux, vars.AuxiliaryFields):
            aux = aux.to_arrays()
        diag_vars: dict[str, Array] = {}
        meta: dict[str, StaggeredFieldMetaData] = {}
        if self.u_pc_option or self.v_pc_option:
            u_pc, v_pc = self.uv_at_primal_cell(state)
            if self.u_pc_option:
                diag_vars["u_pc"] = u_pc.data
                meta["u_pc"] = u_pc.meta
            if self.v_pc_option:
                diag_vars["v_pc"] = v_pc.data
                meta["v_pc"] = v_pc.meta
        if self.ut_e_option:
            diag_vars["ut_e"] = aux.ut_e
            meta["ut_e"] = aux.meta["ut_e"]
        if self.h_e_option:
            diag_vars["h_e"] = aux.h_e
            meta["h_e"] = aux.meta["h_e"]
        if self.h_dc_option:
            diag_vars["h_dc"] = aux.h_dc
            meta["h_dc"] = aux.meta["h_dc"]
        if self.mfx_e_option:
            diag_vars["mfx_e"] = aux.mfx_e
            meta["mfx_e"] = aux.meta["mfx_e"]
        if self.rel_vor_dc_option:
            diag_vars["rel_vor_dc"] = aux.rel_vor_dc
            meta["rel_vor_dc"] = aux.meta["rel_vor_dc"]
        if self.abs_vor_dc_option:
            diag_vars["abs_vor_dc"] = aux.abs_vor_dc
            meta["abs_vor_dc"] = aux.meta["abs_vor_dc"]
        if self.pv_dc_option:
            diag_vars["pv_dc"] = aux.pv_dc
            meta["pv_dc"] = aux.meta["pv_dc"]
        if self.pv_e_option:
            diag_vars["pv_e"] = aux.pv_e
            meta["pv_e"] = aux.meta["pv_e"]
        if self.ke_pc_option:
            diag_vars["ke_pc"] = aux.ke_pc
            meta["ke_pc"] = aux.meta["ke_pc"]
        if self.dhdt_mfx_option:
            diag_vars["dhdt_mfx"] = tend.dhdt_mfx
            meta["dhdt_mfx"] = tend.meta["dhdt_mfx"]
        if self.dundt_coriolis_option:
            diag_vars["dundt_coriolis"] = tend.dundt_coriolis
            meta["dundt_coriolis"] = tend.meta["dundt_coriolis"]
        if self.dundt_grad_ke_option:
            diag_vars["dundt_grad_ke"] = tend.dundt_grad_ke
            meta["dundt_grad_ke"] = tend.meta["dundt_grad_ke"]
        if self.dundt_pgf_option:
            diag_vars["dundt_pgf"] = tend.dundt_pgf
            meta["dundt_pgf"] = tend.meta["dundt_pgf"]
        if self.dhdt_option:
            diag_vars["dhdt"] = tend.dhdt
            meta["dhdt"] = tend.meta["dhdt"]
        if self.dundt_option:
            diag_vars["dundt"] = tend.dundt
            meta["dundt"] = tend.meta["dundt"]
        return DiagnosticVariableSet(variables=diag_vars, meta=meta)

    @overload
    def compute(
        self,
        state: vars.StateFields | vars.StateArrays,
        static: vars.StaticFields | vars.StaticArrays,
        to_ugrid: Literal[True],
    ) -> Dict[str, tuxr.UGRIDDataArray]:
        ...

    @overload
    def compute(
        self,
        state: vars.StateFields | vars.StateArrays,
        static: vars.StaticFields | vars.StaticArrays,
        to_ugrid: Literal[False] = False,
    ) -> Dict[str, Field[Array, StaggeredFieldMetaData]]:
        ...

    def compute(
        self,
        state: vars.StateFields | vars.StateArrays,
        static: vars.StaticFields | vars.StaticArrays,
        to_ugrid: bool = False,
    ) -> Mapping[
        str,
        Field[Array, StaggeredFieldMetaData] | tuxr.UGRIDDataArray
    ]:
        if isinstance(state, vars.StateArrays):
            state_arrays = state
        else:
            state_arrays = state.to_arrays()
        if isinstance(static, vars.StaticArrays):
            static_arrays = static
        else:
            static_arrays = static.to_arrays()
        block = vars.BlockArrays(
            state=state_arrays,
            static=static_arrays,
            tend=vars.TendencyArrays.from_coord(self.equation.coord),
            aux=vars.AuxiliaryArrays.from_coord(self.equation.coord)
        )
        if self.jit:
            aux_jit = jax.jit(self.equation.aux)
            explicit_terms_jit = jax.jit(self.equation.explicit_terms)
            block = aux_jit(block)
            block = explicit_terms_jit(block)
        else:
            block = self.equation.aux(block)
            block = self.equation.explicit_terms(block)
        block = block.to_fields()
        state = block.state
        static = block.static
        tend = block.tend
        aux = block.aux
        diag_vars: dict[
            str,
            Field[Array, StaggeredFieldMetaData] | tuxr.UGRIDDataArray
        ] = {}
        diag_vars["h_pc"] = self.process_field(
            state.h_pc, to_ugrid=to_ugrid
        )
        diag_vars["un_e"] = self.process_field(
            state.un_e, to_ugrid=to_ugrid
        )
        diag_vars["topo"] = self.process_field(
            static.topo, to_ugrid=to_ugrid
        )
        u_pc, v_pc = self.uv_at_primal_cell(state)
        diag_vars["u_pc"] = self.process_field(
            u_pc, to_ugrid=to_ugrid
        )
        diag_vars["v_pc"] = self.process_field(
            v_pc, to_ugrid=to_ugrid
        )
        diag_vars["ut_e"] = self.process_field(
            aux.ut_e, to_ugrid=to_ugrid
        )
        diag_vars["h_e"] = self.process_field(
            aux.h_e, to_ugrid=to_ugrid
        )
        diag_vars["h_dc"] = self.process_field(
            aux.h_dc, to_ugrid=to_ugrid
        )
        diag_vars["mfx_e"] = self.process_field(
            aux.mfx_e, to_ugrid=to_ugrid
        )
        diag_vars["rel_vor_dc"] = self.process_field(
            aux.rel_vor_dc, to_ugrid=to_ugrid)
        diag_vars["abs_vor_dc"] = self.process_field(
            aux.abs_vor_dc, to_ugrid=to_ugrid
        )
        diag_vars["pv_dc"] = self.process_field(
            aux.pv_dc, to_ugrid=to_ugrid
        )
        diag_vars["pv_e"] = self.process_field(
            aux.pv_e, to_ugrid=to_ugrid
        )
        diag_vars["ke_pc"] = self.process_field(
            aux.ke_pc, to_ugrid=to_ugrid
        )
        diag_vars["dhdt_mfx"] = self.process_field(
            tend.dhdt_mfx, to_ugrid=to_ugrid
        )
        diag_vars["dundt_coriolis"] = self.process_field(
            tend.dundt_coriolis, to_ugrid=to_ugrid
        )
        diag_vars["dundt_grad_ke"] = self.process_field(
            tend.dundt_grad_ke, to_ugrid=to_ugrid
        )
        diag_vars["dundt_pgf"] = self.process_field(
            tend.dundt_pgf, to_ugrid=to_ugrid
        )
        diag_vars["dhdt"] = self.process_field(
            tend.dhdt, to_ugrid=to_ugrid
        )
        diag_vars["dundt"] = self.process_field(
            tend.dundt, to_ugrid=to_ugrid
        )
        return diag_vars

    @overload
    def compute_along_trajectory(
        self,
        trajectory: vars.StateFields,
        static: vars.StaticFields,
        to_ugrid: Literal[False] = False,
    ) -> Dict[str, Field[Array, StaggeredFieldMetaData]]:
        ...

    @overload
    def compute_along_trajectory(
        self,
        trajectory: vars.StateArrays,
        static: vars.StaticArrays,
        to_ugrid: Literal[True],
    ) -> Dict[str, tuxr.UGRIDDataArray]:
        ...

    def compute_along_trajectory(
        self,
        trajectory: vars.StateArrays | vars.StateFields,
        static: vars.StaticArrays | vars.StaticFields,
        to_ugrid: bool = False
    ) -> Mapping[
        str,
        Field[Array, StaggeredFieldMetaData] | tuxr.UGRIDDataArray
    ]:
        if isinstance(trajectory, vars.StateFields):
            trajectory = trajectory.to_arrays()
        if isinstance(static, vars.StaticFields):
            static = static.to_arrays()
        if self.jit:
            compute = jax.jit(self.compute, static_argnames=("to_ugrid",))
        else:
            compute = self.compute

        if to_ugrid:
            fields = jax.vmap(
                compute,
                in_axes=(0, None, None),
                out_axes=0
            )(
                trajectory,
                static,
                False
            )
            uxgrid = self.coord.mesh.to_ugrid().to_uxgrid()
            return {
                name: tuxr.UGRIDDataArray.from_xarray(
                    field.to_xarray_with_ugrid_format(), uxgrid
                )
                for name, field in fields.items()
            }

        return jax.vmap(
            compute,
            in_axes=(0, None, None),
            out_axes=0
        )(
            trajectory,
            static,
            to_ugrid
        )
