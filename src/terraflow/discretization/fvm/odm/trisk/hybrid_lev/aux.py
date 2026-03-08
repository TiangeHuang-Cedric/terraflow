from typing import Self
import equinox as eqx
import jax.numpy as jnp

from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.orthogonal_dual_mesh import operator as mesh_op
from terraflow.layout.field import ArrayField, StaggeredFieldMetaData
from terraflow.discretization.abstract import (
    DiscretizationInfo,
    AbstractEquationSetAuxiliaryTerms,
)
from terraflow.discretization.fvm.odm.trisk import adv, base, coriolis
from terraflow.discretization.fvm.odm.trisk.adv import flux_reconstruct as fr
from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block
from terraflow.discretization.fvm.odm.trisk.hybrid_lev import operator as hybrid_op

Term = AbstractEquationSetAuxiliaryTerms[FiniteVolumeMethod]

FDT = ArrayField[StaggeredFieldMetaData]


class Temperature(Term):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        pt = block.state.pt.unwrap()
        mg = block.aux.mg.unwrap()
        exner = hybrid_op.exner_at_full_lev(
            coord=block.coord,
            para=block.para,
            mg=mg
        )
        t = pt * exner
        return eqx.tree_at(
            lambda b: b.aux.t.data,
            block,
            t
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return block


class VerticalMassFluxAtHalfLev(Term):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        dmgsdt = block.tend.dmgsdt.unwrap()
        dmgdt_lev = block.coord.vert.calc_dmgdt_lev(dmgsdt)
        dmf = block.aux.dmf.unwrap()
        mfz_lev = - dmgdt_lev - jnp.cumsum(dmf, axis=0)
        return eqx.tree_at(
            lambda b: b.aux.mfz_lev.data,
            block,
            mfz_lev
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return block


class DryAirWeight(Term):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        mgs = block.state.mgs.unwrap()
        mg_lev = block.coord.vert.calc_mg_lev(mgs)
        mg = 0.5 * (mg_lev[:-1, :] + mg_lev[1:, :])
        return eqx.tree_at(
            lambda b: (b.aux.mg_lev.data, b.aux.mg.data,),
            block,
            (mg_lev, mg)
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return block


class DeltaDryAirWeight(Term):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.recon_at_edge = fr.FluxReconstructAtEdgeOnPrimalMesh()

    def __call__(self, block: Block) -> Block:
        coord = block.coord
        un = block.state.un.unwrap()
        mg = block.aux.mg.unwrap()
        mg_lev = block.aux.mg_lev.unwrap()
        dmg = hybrid_op.delta_at_full_lev(mg_lev)
        dmg_lev = hybrid_op.delta_at_half_lev(x=mg, x_lev=mg_lev)
        dmg_e = self.recon_at_edge(
            x_at_primal_cell=dmg,
            normal_u_at_edge=un,
            coord=block.coord,
            aux=block.equaux.adv
        )
        dmg_vtx = mesh_op.interp_weighted_add_plg2dua(
            x_at_primal_cell=dmg,
            coord=coord,
        )
        return eqx.tree_at(
            lambda b: (
                b.aux.dmg.data, b.aux.dmg_lev.data,
                b.aux.dmg_e.data, b.aux.dmg_vtx.data
            ),
            block,
            replace=(dmg, dmg_lev, dmg_e, dmg_vtx)
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self.recon_at_edge.build_default()
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return self.recon_at_edge.prepare(block)


class DensityOfDryAir(Term):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        gz_lev = block.state.gz_lev.unwrap()
        dmg = block.aux.dmg.unwrap()
        dgz = hybrid_op.delta_at_full_lev(gz_lev)
        rhod = - dmg / dgz
        return eqx.tree_at(
            lambda b: b.aux.rhod.data,
            block,
            rhod
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return block


class Geopotential(Term):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        para = block.para
        gzs = block.static.gzs.unwrap()
        ph_lev = block.state.ph_lev.unwrap()
        t = block.aux.t.unwrap()
        rd = para.rd.unwrap()
        tmp = rd * t * jnp.log(ph_lev[1:, :] / ph_lev[:-1, :])
        gz_lev = gzs + jnp.cumsum(
            jnp.flip(tmp, axis=0),
            axis=0
        )
        gz_lev = jnp.concatenate(
            [gzs.reshape(1, -1), gz_lev],
            axis=0
        )
        gz_lev = jnp.flip(gz_lev, axis=0)
        gz = 0.5 * (gz_lev[:-1, :] + gz_lev[1:, :])
        return eqx.tree_at(
            lambda b: (b.state.gz_lev.data, b.state.gz.data),
            block,
            (gz_lev, gz)
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return block


class HydrostaticPressure(Term):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        mg_lev = block.aux.mg_lev.unwrap()
        mgs = block.state.mgs.unwrap()
        dmg = block.aux.dmg.unwrap()
        phs = mgs
        ph_lev = mg_lev[0, :] + jnp.cumsum(dmg, axis=0)
        ph_lev = jnp.concatenate(
            [mg_lev[0, :].reshape(1, -1), ph_lev],
            axis=0
        )
        ph = 0.5 * (ph_lev[:-1, :] + ph_lev[1:, :])
        return eqx.tree_at(
            lambda b: (b.state.phs.data, b.state.ph.data, b.state.ph_lev.data),
            block,
            (phs, ph, ph_lev)
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return block


class Pressure(Term):
    # nh
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        if self.hydrostatic:
            return eqx.tree_at(
                lambda b: (b.state.p.data, b.state.p_lev.data),
                block,
                (block.state.ph.data, block.state.ph_lev.data)
            )
        coord = block.coord
        para = block.para
        gz = block.state.gz.unwrap()
        pt = block.state.pt.unwrap()
        dmg = block.aux.dmg.unwrap()
        rd = para.rd.unwrap()
        cp = para.cp.unwrap()
        cv = para.cv.unwrap()
        gamma = cp / cv
        p0 = coord.vert.p0
        dgz = hybrid_op.delta_at_full_lev(gz)
        p = - rd * dmg * pt / (p0 * dgz)
        p = p ** gamma * p0
        return eqx.tree_at(
            lambda b: b.state.p.data,
            block,
            p
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return block


class PotentialTemperatureHorizontalAdvection(Term):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.adv_scheme = adv.HorizontalAdvectionScheme()

    def __call__(
        self,
        block: Block
    ) -> Block:
        pt = block.state.pt.unwrap()
        batch = block.adv_batch_pt
        coord = block.coord
        aux = block.equaux.adv
        batch = self.adv_scheme(
            q=pt,
            batch=batch,
            coord=coord,
            aux=aux
        )
        return eqx.tree_at(
            lambda b: b.aux.adv_batch_pt,
            block,
            batch
        )

    def prepare(self, block: Block) -> Block:
        return self.adv_scheme.prepare(block)

    def build_default(self) -> Self:
        if not self.has_built():
            self.adv_scheme.build_default()
            self._built = True
        return self

    def unwrap(
        self,
        block: Block
    ) -> tuple[FDT, FDT]:
        return (block.adv_batch_pt.qmfn, block.adv_batch_pt.qmfz)


class PotentialTemperatureVerticalAdvection(Term):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.adv_scheme = adv.VerticalAdvectionScheme()

    def __call__(
        self,
        block: Block
    ) -> Block:
        pt = block.state.pt.unwrap()
        batch = block.adv_batch_pt
        batch = self.adv_scheme(q=pt, batch=batch)
        return eqx.tree_at(
            lambda b: b.adv_batch_pt,
            block,
            batch
        )

    def prepare(self, block: Block) -> Block:
        return self.adv_scheme.prepare(block)

    def build_default(self) -> Self:
        if not self.has_built():
            self.adv_scheme.build_default()
            self._built = True
        return self

    def unwrap(
        self,
        block: Block
    ) -> tuple[FDT, FDT]:
        return (block.adv_batch_pt.qmfn, block.adv_batch_pt.qmfz)


class AuxiliaryTerms(AbstractEquationSetAuxiliaryTerms):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        baroclinic: bool,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=baroclinic,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.mg = DryAirWeight(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.dmg = DeltaDryAirWeight(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.mf = base.MassFlux(
            disc=disc,
            method=method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.dmf = base.DivergenceOfMassFluxAtCell(
            disc=disc,
            method=method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.ph = HydrostaticPressure(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.p = Pressure(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.t = Temperature(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.gz = Geopotential(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.coriolis = coriolis.CoriolisForceAtEdge(
            disc=disc,
            method=method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.vor = base.RelativeVorticityAtDualCell(
            disc=disc,
            method=method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.pv = base.PotentialVorticity(
            disc=disc,
            method=method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.ke = base.KineticEnergyAtPrimalCell(
            disc=disc,
            method=method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.rhod = DensityOfDryAir(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )

    def build_default(self) -> "AuxiliaryTerms":
        if not self.has_built():
            self.mg.build_default()
            self.dmg.build_default()
            self.mf.build_default()
            self.dmf.build_default()
            self.ph.build_default()
            self.p.build_default()
            self.t.build_default()
            self.gz.build_default()
            self.vor.build_default()
            self.coriolis.build_default()
            self.pv.build_default()
            self.ke.build_default()
            self.rhod.build_default()
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        block = self.mg.prepare(block)
        block = self.dmg.prepare(block)
        block = self.mf.prepare(block)
        block = self.dmf.prepare(block)
        block = self.ph.prepare(block)
        block = self.p.prepare(block)
        block = self.t.prepare(block)
        block = self.gz.prepare(block)
        block = self.vor.prepare(block)
        block = self.coriolis.prepare(block)
        block = self.pv.prepare(block)
        block = self.ke.prepare(block)
        block = self.rhod.prepare(block)
        return block

    def __call__(
        self,
        block: Block,
        pass_tag: str = "all_pass"
    ) -> Block:
        block = self.mg(block)
        block = self.dmg(block)
        block = self.mf(block)
        block = self.dmf(block)
        block = self.ph(block)
        block = self.p(block)
        block = self.t(block)
        block = self.gz(block)
        block = self.coriolis.recon_ut(block)
        block = self.vor(block)
        block = self.pv(block)
        block = self.coriolis(block)
        block = self.ke(block)
        block = self.rhod(block)
        return block
