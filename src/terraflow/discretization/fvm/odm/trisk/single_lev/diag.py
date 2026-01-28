from typing import overload, Literal, Dict, Mapping
import jax

from terraflow.util import typing, uxarray as tuxr
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh import OrthogonalDualMesh
from terraflow.vert_coord import OneLayerVertCoordinate
from terraflow.layout import Field, StaggeredFieldMetaData, MeshEntity, Basis
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
    vars.Block
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
    u_option: bool
    v_option: bool
    ut_option: bool
    z_e_option: bool
    z_vtx_option: bool
    mfn_e_option: bool
    rel_vor_option: bool
    abs_vor_option: bool
    pv_vtx_option: bool
    pv_e_option: bool
    ke_option: bool
    dzdt_mfn_e_option: bool
    dundt_coriolis_option: bool
    dundt_grad_ke_option: bool
    dundt_pgf_option: bool
    dzdt_option: bool
    dundt_option: bool

    def __init__(
        self,
        equation: Equation,
        u: bool = True,
        v: bool = True,
        ut: bool = False,
        z_e: bool = False,
        z_vtx: bool = False,
        mfn_e: bool = False,
        rel_vor: bool = False,
        abs_vor: bool = False,
        pv_vtx: bool = False,
        pv_e: bool = False,
        ke: bool = False,
        dzdt_mfn_e: bool = False,
        dundt_coriolis: bool = False,
        dundt_grad_ke: bool = False,
        dundt_pgf: bool = False,
        dzdt: bool = False,
        dundt: bool = False,
        jit: bool = True,
        trajectory: bool = False
    ):
        super().__init__(
            equation, jit=jit, trajectory=trajectory
        )
        self.u_option = u
        self.v_option = v
        self.ut_option = ut
        self.z_e_option = z_e
        self.z_vtx_option = z_vtx
        self.mfn_e_option = mfn_e
        self.rel_vor_option = rel_vor
        self.abs_vor_option = abs_vor
        self.pv_vtx_option = pv_vtx
        self.pv_e_option = pv_e
        self.ke_option = ke
        self.dzdt_mfn_e_option = dzdt_mfn_e
        self.dundt_coriolis_option = dundt_coriolis
        self.dundt_grad_ke_option = dundt_grad_ke
        self.dundt_pgf_option = dundt_pgf
        self.dzdt_option = dzdt
        self.dundt_option = dundt

    def uv_at_primal_cell(
        self,
        state: vars.State | vars.State,
    ) -> tuple[Field, Field]:
        import terraflow.mesh.orthogonal_dual_mesh.construct as odm
        uv = odm.reconstruct_vector_at_primal_cell_from_dual_edge(
            self.equation.mesh, state.un.unwrap()
        )
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

    def to_diag(
        self,
        state: vars.State,
        tend: vars.Tendency,
        aux: vars.Auxiliary,
    ) -> DiagnosticVariableSet:
        diag_vars: dict[str, Field] = {}
        if self.u_option or self.v_option:
            u, v = self.uv_at_primal_cell(state)
            if self.u_option:
                diag_vars["u"] = u
            if self.v_option:
                diag_vars["v"] = v
        if self.ut_option:
            diag_vars["ut"] = aux.ut
        if self.z_e_option:
            diag_vars["z_e"] = aux.z_e
        if self.z_vtx_option:
            diag_vars["z_vtx"] = aux.z_vtx
        if self.mfn_e_option:
            diag_vars["mfn_e"] = aux.mfn_e
        if self.rel_vor_option:
            diag_vars["rel_vor"] = aux.rel_vor
        if self.abs_vor_option:
            diag_vars["abs_vor"] = aux.abs_vor
        if self.pv_vtx_option:
            diag_vars["pv_vtx"] = aux.pv_vtx
        if self.pv_e_option:
            diag_vars["pv_e"] = aux.pv_e
        if self.ke_option:
            diag_vars["ke"] = aux.ke
        if self.dzdt_mfn_e_option:
            diag_vars["dzdt_mfn_e"] = tend.dzdt_mfn_e
        if self.dundt_coriolis_option:
            diag_vars["dundt_coriolis"] = tend.dundt_coriolis
        if self.dundt_grad_ke_option:
            diag_vars["dundt_grad_ke"] = tend.dundt_grad_ke
        if self.dundt_pgf_option:
            diag_vars["dundt_pgf"] = tend.dundt_pgf
        if self.dzdt_option:
            diag_vars["dzdt"] = tend.dzdt
        if self.dundt_option:
            diag_vars["dundt"] = tend.dundt
        return DiagnosticVariableSet(variables=diag_vars)

    @overload
    def compute(
        self,
        state: vars.State,
        static: vars.Static,
        to_ugrid: Literal[True],
        use_dual: bool = False,
    ) -> Dict[str, tuxr.UGRIDDataArray]:
        ...

    @overload
    def compute(
        self,
        state: vars.State,
        static: vars.Static,
        to_ugrid: Literal[False] = False,
        use_dual: bool = False,
    ) -> Dict[str, Field[Array, StaggeredFieldMetaData]]:
        ...

    def compute(
        self,
        state: vars.State,
        static: vars.Static,
        to_ugrid: bool = False,
        use_dual: bool = False,
    ) -> Mapping[
        str,
        Field[Array, StaggeredFieldMetaData] | tuxr.UGRIDDataArray
    ]:
        block = vars.Block(
            state=state,
            static=static,
            tend=vars.Tendency.from_coord(self.equation.coord),
            aux=vars.Auxiliary.from_coord(self.equation.coord)
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
        diag_vars: dict[
            str,
            Field[Array, StaggeredFieldMetaData] | tuxr.UGRIDDataArray
        ] = {}
        diag_vars["z"] = self.process_field(
            state.z, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["un"] = self.process_field(
            state.un, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["zs"] = self.process_field(
            static.zs, to_ugrid=to_ugrid, use_dual=use_dual
        )
        u, v = self.uv_at_primal_cell(state)
        diag_vars["u"] = self.process_field(
            u, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["v"] = self.process_field(
            v, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["ut"] = self.process_field(
            aux.ut, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["z_e"] = self.process_field(
            aux.z_e, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["z_vtx"] = self.process_field(
            aux.z_vtx, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["mfn_e"] = self.process_field(
            aux.mfn_e, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["rel_vor"] = self.process_field(
            aux.rel_vor, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["abs_vor"] = self.process_field(
            aux.abs_vor, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["pv_vtx"] = self.process_field(
            aux.pv_vtx, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["pv_e"] = self.process_field(
            aux.pv_e, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["ke"] = self.process_field(
            aux.ke, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["dzdt_mfn_e"] = self.process_field(
            tend.dzdt_mfn_e, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["dundt_coriolis"] = self.process_field(
            tend.dundt_coriolis, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["dundt_grad_ke"] = self.process_field(
            tend.dundt_grad_ke, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["dundt_pgf"] = self.process_field(
            tend.dundt_pgf, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["dzdt"] = self.process_field(
            tend.dzdt, to_ugrid=to_ugrid, use_dual=use_dual
        )
        diag_vars["dundt"] = self.process_field(
            tend.dundt, to_ugrid=to_ugrid, use_dual=use_dual
        )
        return diag_vars

    @overload
    def compute_along_trajectory(
        self,
        trajectory: vars.State,
        static: vars.Static,
        to_ugrid: Literal[False] = False,
        use_dual: bool = False,
    ) -> Dict[str, Field[Array, StaggeredFieldMetaData]]:
        ...

    @overload
    def compute_along_trajectory(
        self,
        trajectory: vars.State,
        static: vars.Static,
        to_ugrid: Literal[True],
        use_dual: bool = False,
    ) -> Dict[str, tuxr.UGRIDDataArray]:
        ...

    def compute_along_trajectory(
        self,
        trajectory: vars.State,
        static: vars.Static,
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
                in_axes=(0, None, None, None),
                out_axes=0
            )(
                trajectory,
                static,
                False,
                use_dual
            )
            uxgrid = self.coord.mesh.to_ugrid(use_dual=use_dual).to_uxgrid()
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
            in_axes=(0, None, None, None),
            out_axes=0
        )(
            trajectory,
            static,
            to_ugrid,
            use_dual
        )
