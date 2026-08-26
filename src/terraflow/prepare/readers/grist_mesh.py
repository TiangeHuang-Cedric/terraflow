from pathlib import Path
import jax.numpy as jnp
import xarray as xr

from terraflow.util import device, log, typing
from terraflow.mesh.abstract import AbstractRefMesh

Float = typing.Float
Int   = typing.Int
Array = typing.Array
FLT   = typing.FloatType
INT   = typing.IntType
Str   = typing.Str


class GRISTRefMesh(AbstractRefMesh):
    """Reference mesh data read from GRIST 0d and 2d mesh files."""
    grist_0d_file : Str
    grist_2d_file : Str

    min_edt_dist  : FLT
    max_edt_dist  : FLT
    mean_edt_dist : FLT
    min_edp_dist  : FLT
    max_edp_dist  : FLT
    mean_edp_dist : FLT
    min_tri_area  : FLT
    max_tri_area  : FLT
    mean_tri_area : FLT
    min_plg_area  : FLT
    max_plg_area  : FLT
    mean_plg_area : FLT
    min_tri_angle : FLT
    max_tri_angle : FLT
    mean_tri_angle: FLT
    nv            : INT
    nt            : INT
    ne            : INT
    glevel        : INT
    maxnnb_plg    : INT
    maxnnb_tri    : INT

    vtx_lon       : Float[Array, "nv"]
    vtx_lat       : Float[Array, "nv"]
    tri_cc_lon    : Float[Array, "nt"]
    tri_cc_lat    : Float[Array, "nt"]
    edp_cc_lon    : Float[Array, "ne"]
    edp_cc_lat    : Float[Array, "ne"]
    edt_cc_lon    : Float[Array, "ne"]
    edt_cc_lat    : Float[Array, "ne"]
    vtx_p         : Float[Array, "nv xyz"]
    tri_cc_p      : Float[Array, "nt xyz"]
    edp_cc_p      : Float[Array, "ne xyz"]
    edt_cc_p      : Float[Array, "ne xyz"]
    vtx_ed        : Int[Array, "nv maxnnb_plg"]
    vtx_nnb       : Int[Array, "nv"]
    vtx_nb        : Int[Array, "nv maxnnb_plg"]
    vtx_tr        : Int[Array, "nv maxnnb_plg"]
    tri_ed        : Int[Array, "nt maxnnb_tri"]
    tri_nb        : Int[Array, "nt maxnnb_tri"]
    tri_v         : Int[Array, "nt maxnnb_tri"]
    edp_nr        : Float[Array, "ne xyz"]
    edp_tg        : Float[Array, "ne xyz"]
    edp_v         : Int[Array, "ne endpoint"]
    edp_len       : Float[Array, "ne"]
    edt_nr        : Float[Array, "ne xyz"]
    edt_tg        : Float[Array, "ne xyz"]
    edt_v         : Int[Array, "ne endpoint"]
    edt_len       : Float[Array, "ne"]
    plg_area      : Float[Array, "nv"]
    tri_area      : Float[Array, "nt"]

    @property
    def shape(self) -> dict[str, INT]:
        """Return the mesh shape as a dictionary."""
        return dict(
            nv=self.nv,
            nt=self.nt,
            ne=self.ne
        )

    def __init__(
        self,
        mesh_0d_file: Str,
        mesh_2d_file: Str,
        dev: device.AbstractDevice = device.CPU()
    ):
        """Initialize a GRIST reference mesh from 0d metadata and 2d mesh files."""
        arch = dev()
        float_type = dev.float_type
        int_type = dev.int_type

        ds_0d = xr.open_dataset(mesh_0d_file)
        min_edt_dist   = float(ds_0d["mesh_min_edt_dist"  ].item())
        max_edt_dist   = float(ds_0d["mesh_max_edt_dist"  ].item())
        mean_edt_dist  = float(ds_0d["mesh_mean_edt_dist" ].item())
        min_edp_dist   = float(ds_0d["mesh_min_edp_dist"  ].item())
        max_edp_dist   = float(ds_0d["mesh_max_edp_dist"  ].item())
        mean_edp_dist  = float(ds_0d["mesh_mean_edp_dist" ].item())
        min_tri_area   = float(ds_0d["mesh_min_tri_area"  ].item())
        max_tri_area   = float(ds_0d["mesh_max_tri_area"  ].item())
        mean_tri_area  = float(ds_0d["mesh_mean_tri_area" ].item())
        min_plg_area   = float(ds_0d["mesh_min_plg_area"  ].item())
        max_plg_area   = float(ds_0d["mesh_max_plg_area"  ].item())
        mean_plg_area  = float(ds_0d["mesh_mean_plg_area" ].item())
        min_tri_angle  = float(ds_0d["mesh_min_tri_angle" ].item())
        max_tri_angle  = float(ds_0d["mesh_max_tri_angle" ].item())
        mean_tri_angle = float(ds_0d["mesh_mean_tri_angle"].item())
        nv             = int  (ds_0d["mesh_nv"            ].item())
        nt             = int  (ds_0d["mesh_nt"            ].item())
        ne             = int  (ds_0d["mesh_ne"            ].item())
        glevel         = int  (ds_0d["mesh_glevel"        ].item())
        maxnnb_plg     = int  (ds_0d["mesh_maxvnb"        ].item())
        ds_0d.close()

        ds_2d = xr.open_dataset(mesh_2d_file)
        vtx_ltln_nnb = jnp.array(
            ds_2d["vtx_ltln_nnb"].values.T,
            dtype=float_type,
            device=arch
        )
        tri_cc_ltln = jnp.array(
            ds_2d["tri_cc_ltln"].values.T,
            dtype=float_type,
            device=arch
        )
        edp_cc_ltln = jnp.array(
            ds_2d["edp_cc_ltln"].values.T,
            dtype=float_type,
            device=arch
        )
        edt_cc_ltln = jnp.array(
            ds_2d["edt_cc_ltln"].values.T,
            dtype=float_type,
            device=arch
        )
        vtx_p = jnp.array(
            ds_2d["vtx_p"].values.T,
            dtype=float_type,
            device=arch
        )
        tri_cc_p = jnp.array(
            ds_2d["tri_cc_p"].values.T,
            dtype=float_type,
            device=arch
        )
        edp_cc_p = jnp.array(
            ds_2d["edp_cc_p"].values.T,
            dtype=float_type,
            device=arch
        )
        edt_cc_p = jnp.array(
            ds_2d["edt_cc_p"].values.T,
            dtype=float_type,
            device=arch
        )
        vtx_ed = jnp.array(
            ds_2d["vtx_ed"].values.T - 1,
            dtype=int_type,
            device=arch
        )
        vtx_nb = jnp.array(
            ds_2d["vtx_nb"].values.T - 1,
            dtype=int_type,
            device=arch
        )
        vtx_tr = jnp.array(
            ds_2d["vtx_tr"].values.T - 1,
            dtype=int_type,
            device=arch
        )
        tri_ed = jnp.array(
            ds_2d["tri_ed"].values.T - 1,
            dtype=int_type,
            device=arch
        )
        tri_nb = jnp.array(
            ds_2d["tri_nb"].values.T - 1,
            dtype=int_type,
            device=arch
        )
        tri_v = jnp.array(
            ds_2d["tri_v"].values.T - 1,
            dtype=int_type,
            device=arch
        )
        edp_nr = jnp.array(
            ds_2d["edp_nr"].values.T,
            dtype=float_type,
            device=arch
        )
        edp_tg = jnp.array(
            ds_2d["edp_tg"].values.T,
            dtype=float_type,
            device=arch
        )
        edp_v = jnp.array(
            ds_2d["edp_v"].values.T - 1,
            dtype=int_type,
            device=arch
        )
        edp_len_2d = jnp.array(
            ds_2d["edp_len"].values.T,
            dtype=float_type,
            device=arch
        )
        edt_nr = jnp.array(
            ds_2d["edt_nr"].values.T,
            dtype=float_type,
            device=arch
        )
        edt_tg = jnp.array(
            ds_2d["edt_tg"].values.T,
            dtype=float_type,
            device=arch
        )
        edt_v = jnp.array(
            ds_2d["edt_v"].values.T - 1,
            dtype=int_type,
            device=arch
        )
        edt_len_2d = jnp.array(
            ds_2d["edt_len"].values.T,
            dtype=float_type,
            device=arch
        )
        plg_area_2d = jnp.array(
            ds_2d["plg_area"].values.T,
            dtype=float_type,
            device=arch
        )
        tri_area_2d = jnp.array(
            ds_2d["tri_area"].values.T,
            dtype=float_type,
            device=arch
        )
        ds_2d.close()

        grist_mesh_maxnnb_tri = int(tri_nb.shape[1])
        self.grist_0d_file = str(mesh_0d_file)
        self.grist_2d_file = str(mesh_2d_file)
        self.min_edt_dist = min_edt_dist
        self.max_edt_dist = max_edt_dist
        self.mean_edt_dist = mean_edt_dist
        self.min_edp_dist = min_edp_dist
        self.max_edp_dist = max_edp_dist
        self.mean_edp_dist = mean_edp_dist
        self.min_tri_area = min_tri_area
        self.max_tri_area = max_tri_area
        self.mean_tri_area = mean_tri_area
        self.min_plg_area = min_plg_area
        self.max_plg_area = max_plg_area
        self.mean_plg_area = mean_plg_area
        self.min_tri_angle = min_tri_angle
        self.max_tri_angle = max_tri_angle
        self.mean_tri_angle = mean_tri_angle
        self.nv = nv
        self.nt = nt
        self.ne = ne
        self.glevel = glevel
        self.maxnnb_plg = maxnnb_plg
        self.maxnnb_tri = grist_mesh_maxnnb_tri
        self.vtx_lon = vtx_ltln_nnb[:, 1]
        self.vtx_lat = vtx_ltln_nnb[:, 0]
        self.tri_cc_lon = tri_cc_ltln[:, 1]
        self.tri_cc_lat = tri_cc_ltln[:, 0]
        self.edp_cc_lon = edp_cc_ltln[:, 1]
        self.edp_cc_lat = edp_cc_ltln[:, 0]
        self.edt_cc_lon = edt_cc_ltln[:, 1]
        self.edt_cc_lat = edt_cc_ltln[:, 0]
        self.vtx_p = vtx_p
        self.tri_cc_p = tri_cc_p
        self.edp_cc_p = edp_cc_p
        self.edt_cc_p = edt_cc_p
        self.vtx_ed = vtx_ed
        self.vtx_nnb = jnp.array(
            vtx_ltln_nnb[:, 2],
            dtype=int_type,
            device=arch
        )
        self.vtx_nb = vtx_nb
        self.vtx_tr = vtx_tr
        self.tri_ed = tri_ed
        self.tri_nb = tri_nb
        self.tri_v = tri_v
        self.edp_nr = edp_nr
        self.edp_tg = edp_tg
        self.edp_v = edp_v
        self.edp_len = edp_len_2d[:, 0]
        self.edt_nr = edt_nr
        self.edt_tg = edt_tg
        self.edt_v = edt_v
        self.edt_len = edt_len_2d[:, 0]
        self.plg_area = plg_area_2d[:, 0]
        self.tri_area = tri_area_2d[:, 0]

        if dev.proc.is_root():
            log.logger.info("Loading GRIST mesh from: ")
            log.logger.info(
                f"    └─grist 0d file: {Path(mesh_0d_file).absolute()}"
            )
            log.logger.info(
                f"    └─grist 2d file: {Path(mesh_2d_file).absolute()}."
            )
            log.logger.info("GRIST Mesh info: ")
            log.logger.info(
                "    └─nv={} nt={} ne={}".format(
                    self.nv,
                    self.nt,
                    self.ne
                )
            )
            log.logger.info(
                "    └─maxnnb_plg={} maxnnb_tri={} glevel={}".format(
                    self.maxnnb_plg,
                    self.maxnnb_tri,
                    self.glevel
                )
            )
