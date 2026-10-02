#!/usr/bin/env python3

import terraflow.core as tf

tf.util.datetime.perf_start("program")

grist_path = "../../../data/GRIST_INPUT/uniform-pl-20211229/"
grist_0d_file = grist_path + "grist.grid_file.g5.ccvt.0d.nc"
grist_2d_file = grist_path + "grist.grid_file.g5.ccvt.2d.nc"

dev = tf.util.device.CPU(
    backend="jax",
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
    dev=dev
)

config = tf.model.UserConfig(
    pv_adv_scheme="upwind",
    upwind_order_pv=3,
    pt_adv_scheme="upwind",
    upwind_order=3,
    pgf_scheme="lin97",
    use_smag_damp=False,
    use_div_damp=True,
    use_rayleigh_damp_w=False,
    use_p_damp=False,
    run_time="12 days",
    dt_dyn="300 seconds",
    dt_adv="300 seconds",
    save_interval="6 hours"
)

topo_reader = tf.readers.topo.ETOPO1Reader(
    topo_file="../../../data/topo/ETOPO1_Ice_g_gmt4.grd",
    para=para,
    cfg=tf.readers.TopographyReaderConfig(
        regrid_type="fill",
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
        prepare_regrid_gz=False
    ),
    dev=dev
)

model = tf.model.build_baroclinic_model(
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
    nonhydrostatic=True,
    ideal_dry_core=False,
    restart=False
)

era5_1deg_lon = dev.backend.arange(0, 359 + 1.0 ,  1.0, dtype=dev.float_type)
era5_1deg_lat = dev.backend.arange(90, -90 - 1.0, -1.0, dtype=dev.float_type)
era5_lev = dev.backend.array(
    [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 
    250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 
    850, 875, 900, 925, 950, 975, 1000] * 100, 
    dtype=dev.float_type
)

history = model.build_history(
    mode="scvt",
    case_name="terraflow.ndc" + ".g" + str(mesh.glevel),
    output_h1=False,
    output_h2=False,
    h2_opt={
        "hori_regrid_method": "conserve_1st",
        "hori_regrid_backend": "esmf",
        "vert_interp": False,
        "lon": era5_1deg_lon,
        "lat": era5_1deg_lat,
        "plev": era5_lev,
        "plev_unit": "Pa",
    }
)

integrate_fn = model.io_callback(history=history, with_jit=True)

tf.util.datetime.perf_start("integration")

final = integrate_fn(model)
dev.block_until_ready(final)

tf.util.datetime.perf_end("integration")

if dev.is_root():
    tf.util.post_process.CDOPostProcesser.remap_history(
        history=history,
        res=1.0,
        remap=True,
        use_weight=True,
    )

tf.util.datetime.perf_end("program")
