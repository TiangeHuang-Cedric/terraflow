#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import equinox as eqx
import terraflow.core as tf

tf.util.datetime.perf_start("program")

jax.config.update("jax_debug_nans", False)
jax.config.update("jax_debug_infs", False)

grist_path = "../../../data/GRIST_INPUT/uniform-pl-20211229/"
grist_0d_file = grist_path + "grist.grid_file.g5.ccvt.0d.nc"
grist_2d_file = grist_path + "grist.grid_file.g5.ccvt.2d.nc"

dev = tf.util.device.CPU(
    float_type="r8",
    int_type="i4",
    use_async_io=False, 
    proc_io_stride=0
)

para = tf.planet.PlanetParameterSet.from_si(
    planet="earth", component="atmosphere", dev=dev
)

vert_coord = tf.vert_coord.HybridVertCoordinate.cam_l30(dev=dev)

mesh = tf.mesh.SCVTMesh.from_grist(
    mesh_0d_file=grist_0d_file,
    mesh_2d_file=grist_2d_file,
    para=para,
    vert_coord=vert_coord,
    hori_hw=3,
    lev_hw=3,
    dev=dev
)

config = tf.model.UserConfig(
    pv_adv_scheme="upwind",
    upwind_order_pv=3,
    pt_adv_scheme="upwind",
    upwind_order=3,
    use_smag_damp=False,
    use_div_damp=True,
    run_time="1 hours",
    dt_dyn="300 seconds",
    dt_adv="300 seconds",
    save_interval="1 hours"
)

topo_reader = tf.readers.topo.ETOPO1Reader(
    topo_file="../../../data/topo/ETOPO1_Ice_g_gmt4.grd",
    para=para,
    cfg=tf.readers.TopographyReaderConfig(
        regrid_type="fill",
        fill_backend="scipy",
        fill_split_with_chunks=True,
        fill_lon_chunk_size=360,
        fill_lat_chunk_size=180,
        use_zs_grad_filter=True,
        zs_grad_filter_type="laplacian1",
    ),
    dev=dev
)

bkg_reader = tf.readers.bkg.ERA5Reader(
    bkg_file="../../../data/bkg/era5_2022060100.nc",
    para=para,
    cfg=tf.readers.BackgroundReaderConfig(
        hori_interp_backend="esmf",
        hori_interp_method="conserve_1st",
        interp_at_midpoint=False,
        vert_interp_backend="jax",
    ),
    dev=dev
)

model = tf.model.build_hydrostatic_model(
    config=config,
    para=para,
    mesh=mesh,
    vert_coord=vert_coord,
    method="finite_volume_method",
    form="vector_invariant",
    staggered_kind="triskc",
    bkg_reader=bkg_reader,
    topo_reader=topo_reader,
    dev=dev,
    ideal_dry_core=False,
    restart=False
)

integrate_fn = model.integrate()


def wrap_grad_fn(un, model):
    old = model.block.old
    model = eqx.tree_at(
        where=lambda m: m.block.state.un.d,
        pytree=model,
        replace_fn=lambda data: data.at[old, :, :].set(un),
    )
    final = integrate_fn(model)
    return jnp.sum(
        jnp.linalg.norm(
            final.block.state.phs.d,
            axis=-1
        )
    )


tf.util.datetime.perf_start("integration")

final, grad = jax.value_and_grad(wrap_grad_fn)(
    model.block.state.un.d[model.block.old, :, :],
    model
)
jax.block_until_ready(final)
jax.block_until_ready(grad)

print(final)
print(grad)

tf.util.datetime.perf_end("integration")

tf.util.datetime.perf_end("program")
