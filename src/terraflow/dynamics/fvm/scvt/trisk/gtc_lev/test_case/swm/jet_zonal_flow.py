from __future__ import annotations
from typing import TYPE_CHECKING
import jax
from jax import numpy as jnp
import quadax

from terraflow.util import typing
from terraflow.util.pytree import tree_at
from terraflow.util.sphere_geometry import vector_convert_lonlat2xyz
from terraflow.mesh.scvt.parallel import fill_field2d_array_halo
from terraflow.test_case.swm import jet_zonal_flow as swm_jz


if TYPE_CHECKING:
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block


Float = typing.Float
Array = typing.Array


class JetZonalFlowTestCase(swm_jz.JetZonalFlowTestCase):
    
    def set_ic(self, block: Block) -> Block:
        block = self.init_gzs(block)
        block = self.init_un (block)
        block = self.init_gz (block)
        return block

    def init_un(self, block: Block) -> Block:
        mesh = block.mesh
        un   = block.state.un.unwrap()[0, :, :]
        # ==============================================        
        ids, ide = mesh.edge_ids, mesh.edge_ide
        u = self.u_function(mesh.dual_edge_lat[ids:ide+1])
        v = jnp.zeros_like(u)
        vec = vector_convert_lonlat2xyz(
            R=block.para.radius.unwrap(),
            u=u,
            v=v,
            xyz=mesh.dual_edge_xyz[ids:ide+1, :]
        )
        un = un.at[0, ids:ide+1].set(
            jnp.sum(vec * mesh.edge_nr[ids:ide+1, :], axis=-1)
        )
        un = fill_field2d_array_halo(
            halo=mesh.halo,
            f=un,
            meta=block.state.un.meta
        )
        # ==============================================   
        return tree_at(
            where=lambda b: b.state.un.d,
            pytree=block,
            replace=un,
            at_dims=0,
            at_idxes=0
        )

    def init_gz(self, block: Block) -> Block:
        mesh  = block.mesh
        gz    = block.state.gz.unwrap()[0, :, :]
        lat2  = self.lat2 
        alpha = self.alpha
        beta  = self.beta
        gh0   = self.gh0
        ghd   = self.ghd
        # ==============================================
        ids, ide = mesh.cell_ids, mesh.cell_ide
        
        def gh_integral(lat: Float[Array, ""]) -> Float[Array, ""]:
            val, _ = quadax.quadgk(
                self.gh_integrand,
                jnp.array([-jnp.pi / 2, lat]),
                epsabs=1.0e-12, 
                epsrel=1.0e-3
            )
            return jnp.where(jnp.isclose(lat, -jnp.pi / 2), 0.0, val)

        gz_ = jax.vmap(gh_integral)(mesh.cell_lat[ids:ide+1])
        gz = gz.at[0, ids:ide+1].set(gh0 - gz_)
        gz = gz.at[0, ids:ide+1].set(
            gz[0, ids:ide+1] + ghd * 
            jnp.cos(mesh.cell_lat[ids:ide+1]) * 
            jnp.exp(
                -(
                    jnp.where(
                        mesh.cell_lon[ids:ide+1] > jnp.pi,
                        mesh.cell_lon[ids:ide+1] - 2 * jnp.pi,
                        mesh.cell_lon[ids:ide+1]
                    ) / alpha
                )**2
            ) * 
            jnp.exp(
                -((lat2 - mesh.cell_lat[ids:ide+1]) / beta)**2
            )
        )
        gz = fill_field2d_array_halo(
            halo=mesh.halo,
            f=gz,
            meta=block.state.gz.meta
        )
        # ==============================================
        return tree_at(
            where=lambda b: b.state.gz.d,
            pytree=block,
            replace=gz,
            at_dims=0,
            at_idxes=0
        )
        
    def init_gzs(self, block: Block) -> Block:
        gzs   = block.static.gzs.unwrap()
        gzs  = jnp.zeros_like(gzs)
        # ==============================================
        return tree_at(
            where=lambda b: b.static.gzs.d,
            pytree=block,
            replace=gzs,
        )

    def u_function(self, lat: Float[Array, "nlat"]) -> Float[Array, "nlat"]:
        u_max = self.u_max
        lat0  = self.lat0
        lat1  = self.lat1
        en    = self.en
        return jnp.where(
            jnp.logical_or(lat <= lat0, lat >= lat1),
            0.0,
            u_max / en * jnp.exp(1 / (lat - lat0) / (lat - lat1))
        )


    def gh_integrand(self, lat: Float[Array, "nlat"]) -> Float[Array, "nlat"]:
        omega  = self.para.omega.unwrap()
        radius = self.para.radius.unwrap()
        u = self.u_function(lat)
        f = 2 * omega * jnp.sin(lat)
        return radius * u * (f + jnp.tan(lat) / radius * u)
