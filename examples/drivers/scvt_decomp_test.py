#!/usr/bin/env python3

import terraflow.core as tf


grist_path = "../../data/GRIST_INPUT/uniform-pl-20211229/"
grist_0d_file = grist_path + "grist.grid_file.g4.ccvt.0d.nc"
grist_2d_file = grist_path + "grist.grid_file.g4.ccvt.2d.nc"

dev = tf.util.device.CPU(
    float_type="r8",
    int_type="i4",
    use_async_io=False, 
    proc_io_stride=0
)

para = tf.planet.PlanetParameterSet.from_si(
    planet="earth",
    component="atmosphere",
    dev=dev
)

mesh = tf.mesh.SCVTMesh.from_grist(
    mesh_0d_file=grist_0d_file,
    mesh_2d_file=grist_2d_file,
    para=para,
    hori_hw=3,
    lev_hw=3,
    dev=dev
)

mesh.output(dev.proc)