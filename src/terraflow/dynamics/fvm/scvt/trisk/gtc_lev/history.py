from __future__ import annotations
from typing import (
    TypeVar,
    Type,
    Self,
    Union,
    Generic,
    Optional,
    TYPE_CHECKING
)
from abc import abstractmethod
import time
import netCDF4 as nc4
import numpy as np


from terraflow.util import log
from terraflow.util.parallel_types import MPI, MPIComm
from terraflow.util.device import AbstractDevice
from terraflow.util.datetime import DateTimeManager
from terraflow.util.history.abstract import (
    AbstractHistoryset,
    AbstractHistory
)
from terraflow.util.history.setup.base import GlobalAttributesSetup, TimeSetup
from terraflow.util.history.setup.mesh import AbstractMeshHistorySetup
from terraflow.util.history.setup.vert import AbstractVertHistorySetup
from terraflow.util.field import ScalarField, ScalarFieldMeta
from terraflow.mesh.scvt.interp import interp_plev
from terraflow.mesh.scvt.field import (
    SCVTFields, 
    SCVTFieldLike, 
    SCVTFieldMeta,
    meta2ugrid
)


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.util.parallel_types import Proc
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
    from terraflow.mesh.scvt._esmf_regridder import ESMFRegridder
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.model.swm import ShallowWaterModel
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.model.adv import (
        Adv2DModel,
        Adv3DModel
    )
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.model.baroclinic import BaroclinicModel
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.tracer import TracerMeta
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block


    Array = typing.Array
    
    MODEL = TypeVar(
        "MODEL", 
        ShallowWaterModel, 
        Adv2DModel, 
        Adv3DModel, 
        BaroclinicModel
    )


HY = TypeVar ("HY" , bound='AbstractBaseHistory'   )
SHY = TypeVar("SHY", bound='SCVT2StructuredHistory')


STRUCTURED_HISTORY_LIST = ["h0", "h2"]

UNSTRUCTURED_SPATIAL_DIMS = {
    "cell", "vtx", "edge",
    "n_face", "n_node", "n_edge"
}

UGRID_SPATIAL_DIMS = {
    "n_face", "n_node", "n_edge"
}

LEV_DIMS = {"lev", "ilev"}

VARNAMES_USING_LOG_INTERP = {"gz", "gz_lev", "t"}

PLEV_NAMES = {"ph", "ph_lev"}


def _history_group_id(nproc: int, ngroups: int, rank: int) -> int:
    return rank // (nproc // ngroups)


def _history_group_bounds(nproc: int, ngroups: int, group_id: int) -> tuple[int, int]:
    group_count = nproc // ngroups
    group_first = group_id * group_count
    return group_first, group_count


def _history_global_range(n_global: int, ngroups: int, group_id: int) -> tuple[int, int]:
    count_base, remainder = divmod(n_global, ngroups)
    start = group_id * count_base + min(group_id, remainder)
    count = count_base + (1 if group_id < remainder else 0)
    return start, count


class SCVTHistoryIOMap:
    name: str
    n_global: int
    writer_start: int
    writer_count: int
    is_writer: bool

    def __init__(
        self,
        history: AbstractBaseHistory,
        name: str,
        n_global: int,
    ) -> None:
        self.history = history
        self.name = name
        self.n_global = int(n_global)
        group_id = _history_group_id(
            history.io_nproc,
            history.output_ngroups,
            history.io_rank
        )
        group_first, _ = _history_group_bounds(
            history.io_nproc,
            history.output_ngroups,
            group_id
        )
        self.writer_start, self.writer_count = _history_global_range(
            self.n_global,
            history.output_ngroups,
            group_id
        )
        self.is_writer = history.io_rank == group_first

    def collect(
        self,
        field_data: Array,
        meta: SCVTFieldMeta
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        local_piece = self.local_piece(field_data, meta)
        return self.history.comm_io.allgather(local_piece)

    def collect_root(
        self,
        field_data: Array,
        meta: SCVTFieldMeta
    ) -> list[tuple[np.ndarray, np.ndarray]] | None:
        local_piece = self.local_piece(field_data, meta)
        return self.history.comm_io.gather(local_piece, root=0)

    def local_piece(
        self,
        field_data: Array,
        meta: SCVTFieldMeta
    ) -> tuple[np.ndarray, np.ndarray]:
        bkd = self.history.backend
        local_data = bkd.to_numpy(field_data[..., meta.ids:meta.ide + 1])
        local_indices = self.history.owned_global_indices(meta)
        owned = self.history.source_owns_indices(self.name, local_indices)
        local_indices = local_indices[owned]
        local_data = local_data[..., owned]
        return local_indices, local_data

    def assemble_writer_range(
        self,
        pieces: list[tuple[np.ndarray, np.ndarray]],
        dtype: type[np.float32] | type[np.float64],
    ) -> np.ndarray:
        if not pieces:
            return np.empty((self.writer_count,), dtype=dtype)
        sample = pieces[0][1]
        fill_value = np.nan if np.issubdtype(dtype, np.floating) else 0
        out = np.full(
            sample.shape[:-1] + (self.writer_count,),
            fill_value,
            dtype=dtype
        )
        filled = np.zeros(self.writer_count, dtype=bool)
        writer_end = self.writer_start + self.writer_count
        for global_indices, data in pieces:
            mask = (
                (global_indices >= self.writer_start)
                & (global_indices < writer_end)
            )
            if not np.any(mask):
                continue
            pos = global_indices[mask] - self.writer_start
            if np.any(filled[pos]):
                raise RuntimeError(
                    f"SCVT history IO map for {self.name} received duplicated indices."
                )
            out[..., pos] = data[..., mask]
            filled[pos] = True
        if not np.all(filled):
            raise RuntimeError(
                f"SCVT history IO map for {self.name} has missing indices."
            )
        return out


def build_history(
    model: MODEL,
    mode: str = "scvt",
    case_name: str = "terraflow",
    file_path: str = "",
    output_h0: bool = True,
    output_h1: bool = False,
    output_h2: bool = False,
    h1_opt: dict = {"use_dual": False},
    h2_opt: dict = {},
    output_ngroups: int = 0,
    dev: Optional[AbstractDevice] = None,
    float_type: Optional[str] = None,
    int_type: Optional[str] = None,
) -> Union[UGRIDHistoryset, SCVTHistoryset]:
    if mode == "scvt":
        return SCVTHistoryset(
            model=model,
            case_name=case_name,
            file_path=file_path,
            output_h0=output_h0,
            output_h1=output_h1,
            output_h2=output_h2,
            h2_opt=h2_opt,
            output_ngroups=output_ngroups,
            dev=dev,
            float_type=float_type,
            int_type=int_type
        ).setup(model.block)
    elif mode == "ugrid":
        return UGRIDHistoryset(
            model=model,
            case_name=case_name,
            file_path=file_path,
            output_h0=output_h0,
            output_h1=output_h1,
            output_h2=output_h2,
            h2_opt=h2_opt,
            use_dual=h1_opt["use_dual"],
            output_ngroups=output_ngroups,
            dev=dev,
            float_type=float_type,
            int_type=int_type            
        ).setup(model.block)
    else:
        raise ValueError(f"Unsupported history mode {mode}!")


class AbstractBaseHistoryset(AbstractHistoryset, Generic[HY, SHY]):
    h0          : HY
    h1          : HY
    h2          : SHY
    output_h0   : bool
    output_h1   : bool
    output_h2   : bool
    time_manager: DateTimeManager
    case_name   : str
    float_type  : type[np.float32] | type[np.float64]
    int_type    : type[np.int32] | type[np.int64]
    output_ngroups: int
    built       : bool

    @abstractmethod
    def __init__(
        self,
        model: MODEL,
        case_name: str = "terraflow",
        file_path: str = "",
        output_h0: bool = True,
        output_h1: bool = False,
        output_h2: bool = False,
        h0_opt: dict = {},
        h1_opt: dict = {},
        h2_opt: dict = {},
        output_ngroups: int = 0,
        dev: Optional[AbstractDevice] = None,
        float_type: Optional[str] = None,
        int_type: Optional[str] = None,
        **kwargs
    ) -> None:
        self.built = False
        self.time_manager = model.time_manager
        self.case_name = case_name
        self.output_h0 = output_h0
        self.output_h1 = output_h1
        self.output_h2 = output_h2
        self.output_ngroups = output_ngroups
        super().__init__(
            model=model,
            case_name=case_name,
            file_path=file_path,
            dev=dev,
            float_type=float_type,
            int_type=int_type
        )
        if self.output_h0:
            self.h0 = self.history_type()(
                model=model,
                history_name="h0",
                case_name=case_name,
                file_path=file_path,
                float_type=self.float_type,
                int_type=self.int_type,
                output_ngroups=output_ngroups,
                **h0_opt,
                **kwargs,
            )
            if self.is_root():
                log.logger.info(f"History will output h0: {self.h0.file_path}")
        if self.output_h1:
            self.h1 = self.history_type()(
                model=model,
                history_name="h1",
                case_name=case_name,
                file_path=file_path,
                float_type=self.float_type,
                int_type=self.int_type,
                output_ngroups=output_ngroups,
                **h1_opt,
                **kwargs,
            )
            if self.is_root():
                log.logger.info(f"History will output h1: {self.h1.file_path}")
        if self.output_h2:
            self.h2 = self.structured_history_type()(
                model=model,
                history_name="h2",
                case_name=case_name,
                file_path=file_path,
                float_type=self.float_type,
                int_type=self.int_type,
                output_ngroups=output_ngroups,
                **h2_opt,
                **kwargs,
            )
            if self.is_root():
                log.logger.info(f"History will output h2: {self.h2.file_path}")

    def setup(self, block: Block) -> Self:
        if self.has_built():
            return self
        self.setup_h0(block)
        self.setup_h1(block)
        self.setup_h2(block)
        self.built = True
        return self

    @staticmethod
    @abstractmethod
    def history_type() -> Type[HY]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def structured_history_type() -> Type[SHY]:
        raise NotImplementedError

    def get_all_histories(self, with_h2: bool = False) -> list[str]:
        histories = []
        if self.output_h0:
            histories.append(self.h0.file_path)
        if self.output_h1:
            histories.append(self.h1.file_path)
        if self.output_h2 and with_h2:
            histories.append(self.h2.file_path)
        return histories

    def has_built(self) -> bool:
        return self.built

    def output(
        self,
        block: Block,
        itime_str: str,
        save_idx: int,
        first_call: bool = False
    ) -> None:
        """
        Write the fields and diagnostics to the history file.

        Args:
            old_block (AbstractBlock): The block before the time step (include old state).
            new_block (AbstractBlock): The block after the time step (include new state, old tend and old aux).
            save_idx (int): The index of the current save interval.
            first_call (bool): Whether the output function is called for the first time.
                If True, it will write the static fields to the history file.

        Returns:
            None
        """
        if itime_str == "old":
            self.backend.io_callback(
                self._output_old,
                None,
                block,
                save_idx,
                first_call,
                ordered=True
            )
        elif itime_str == "new":
            self.backend.io_callback(
                self._output_new,
                None,
                block,
                save_idx,
                first_call,
                ordered=True
            )
        else:
            raise ValueError(
                f"Invalid itime_str: {itime_str}, expected 'old' or 'new'."
            )

    def _output_old(
        self,
        block: Block,
        save_interval_idx: int,
        first_call: bool = False
    ) -> None:
        return self._output_in_time(
            block=block,
            itime_str="old",
            save_interval_idx=save_interval_idx,
            first_call=first_call
        )

    def _output_new(
        self,
        block: Block,
        save_interval_idx: int,
        first_call: bool = False
    ) -> None:
        return self._output_in_time(
            block=block,
            itime_str="new",
            save_interval_idx=save_interval_idx,
            first_call=first_call
        )

    def _output_in_time(
        self,
        block: Block,
        itime_str: str,
        save_interval_idx: int,
        first_call: bool = False
    ) -> None:

        self.log_write_elapsed_seconds()

        def _set_time_var(h: HY | SHY):
            if h.is_root():
                ds = nc4.Dataset(h.file_path, mode="a")
                time_var = ds.variables["time"]
                current_length = len(time_var)
                interval = save_interval_idx * self.time_manager.save_interval_value
                time_var[current_length] = interval
                ds.close()
            h.sync()

        if self.output_h0:
            self.log_start_write("h0")
            t0 = time.perf_counter()
            _set_time_var(self.h0)
            self.write_h0(block, itime_str=itime_str, first_call=first_call)
            t1 = time.perf_counter()
            self.log_done_write("h0", t1 - t0)
        if self.output_h1:
            self.log_start_write("h1")
            t0 = time.perf_counter()
            _set_time_var(self.h1)
            self.write_h1(block, itime_str=itime_str)
            t1 = time.perf_counter()
            self.log_done_write("h1", t1 - t0)
        if self.output_h2:
            self.log_start_write("h2")
            t0 = time.perf_counter()
            _set_time_var(self.h2)
            self.write_h2(block, itime_str=itime_str, first_call=first_call)
            t1 = time.perf_counter()
            self.log_done_write("h2", t1 - t0)
        self.sync()
        return None

    def setup_h0(self, block: Block) -> None:
        if self.output_h0:
            self.h0.setup(block.state)
            self.h0.setup(block.tend)
            self.h0.setup(block.static)
            self.h0.setup(block.aux)
            self.h0.setup(block.tracers)
            self.h0.setup(block.adv_batch_bg)
            self.h0.setup(block.adv_batch_pt)
            for batch in block.adv_batches.values():
                self.h0.setup(batch)
            self.h0.end_setup()

    def setup_h1(self, block: Block) -> None:
        if self.output_h1:
            self.h1.setup(block.state)
            self.h1.setup(block.tend)
            self.h1.setup(block.static)
            self.h1.setup(block.aux)
            self.h1.setup(block.adv_batch_bg)
            self.h1.setup(block.adv_batch_pt)
            for batch in block.adv_batches.values():
                self.h1.setup(batch)
            self.h1.end_setup()

    def setup_h2(self, block: Block) -> None:
        if self.output_h2:
            if self.output_h0:
                self.h2.setup(block.state)
                self.h2.setup(block.tend)
                self.h2.setup(block.static)
                self.h2.setup(block.aux)
                self.h2.setup(block.tracers)
                self.h2.setup(block.adv_batch_bg)
                self.h2.setup(block.adv_batch_pt)
                for batch in block.adv_batches.values():
                    self.h2.setup(batch)
                self.h2.end_setup()
            else:
                raise ValueError(
                    "output_h0 is False, cannot setup h2."
                )

    def write_h0(self, block: Block, itime_str: str, first_call: bool = False):
        itime = block.get_time_idx(itime_str=itime_str)
        self.h0.write_fields(block.state, dim0_idx=itime)
        self.h0.write_fields(block.tend)
        if first_call:
            self.h0.write_fields(block.static)
        self.h0.write_fields(block.aux)
        for batch in block.adv_batches.values():
            batch_idx = batch.idx
            itime_batch = batch.get_time_idx(itime_str=itime_str)
            self.h0.write_fields(
                block.tracers,
                dim0_idx=itime_batch,
                tracer_idx=batch_idx
            )
        self.h0.write_fields(block.adv_batch_bg)
        self.h0.write_fields(block.adv_batch_pt)
        for batch in block.adv_batches.values():
            self.h0.write_fields(batch)

    def write_h1(self, block: Block, itime_str: str):
        itime = block.get_time_idx(itime_str=itime_str)
        self.h1.write_fields(block.state, dim0_idx=itime)
        self.h1.write_fields(block.tend)
        self.h1.write_fields(block.static)
        self.h1.write_fields(block.aux)
        self.h1.write_fields(block.adv_batch_bg)
        self.h1.write_fields(block.adv_batch_pt)
        for batch in block.adv_batches.values():
            self.h1.write_fields(batch)

    def write_h2(self, block: Block, itime_str: str, first_call: bool = False):
        itime = block.get_time_idx(itime_str=itime_str)
        ph, ph_lev = block.get_plev()
        ph, ph_lev = self.h2.update_plev(ph=ph, ph_lev=ph_lev, itime=itime)
        if self.output_h0:
            self.h2.write_fields(
                block.state,
                dim0_idx=itime,
                ph=ph,
                ph_lev=ph_lev
            )
            self.h2.write_fields(block.tend, ph=ph, ph_lev=ph_lev)
            if first_call:
                self.h2.write_fields(block.static, ph=ph, ph_lev=ph_lev)
            self.h2.write_fields(block.aux, ph=ph, ph_lev=ph_lev)
            for batch in block.adv_batches.values():
                batch_idx = batch.idx
                itime_batch = batch.get_time_idx(itime_str=itime_str)
                self.h2.write_fields(
                    block.tracers,
                    dim0_idx=itime_batch,
                    tracer_idx=batch_idx,
                    ph=ph,
                    ph_lev=ph_lev
                )
            self.h2.write_fields(block.adv_batch_bg, ph=ph, ph_lev=ph_lev)
            self.h2.write_fields(block.adv_batch_pt, ph=ph, ph_lev=ph_lev)
            for batch in block.adv_batches.values():
                self.h2.write_fields(batch, ph=ph, ph_lev=ph_lev)
        else:
            raise ValueError(
                "output_h0 is False, cannot write h2."
            )


class UGRIDHistoryset(
    AbstractBaseHistoryset["UGRIDHistory", "SCVT2StructuredHistory"]
):

    def __init__(
        self,
        model: MODEL,
        case_name: str = "terraflow",
        file_path: str = "",
        output_h0: bool = True,
        output_h1: bool = False,
        output_h2: bool = False,
        use_dual: bool = False,
        output_ngroups: int = 0,
        dev: Optional[AbstractDevice] = None,
        float_type: Optional[str] = None,
        int_type: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(
            model=model,
            case_name=case_name,
            file_path=file_path,
            output_h0=output_h0,
            output_h1=output_h1,
            output_h2=output_h2,
            h0_opt={"use_dual": use_dual},
            h1_opt={"use_dual": use_dual},
            output_ngroups=output_ngroups,
            dev=dev,
            float_type=float_type,
            int_type=int_type,
            **kwargs
        )

    @staticmethod
    def history_type() -> Type["UGRIDHistory"]:
        return UGRIDHistory

    @staticmethod
    def structured_history_type() -> Type["SCVT2StructuredHistory"]:
        return SCVT2StructuredHistory


class SCVTHistoryset(
    AbstractBaseHistoryset["SCVTHistory", "SCVT2StructuredHistory"]
):

    def __init__(
        self,
        model: MODEL,
        case_name: str = "terraflow",
        file_path: str = "",
        output_h0: bool = True,
        output_h1: bool = False,
        output_h2: bool = False,
        output_ngroups: int = 0,
        dev: Optional[AbstractDevice] = None,
        float_type: Optional[str] = None,
        int_type: Optional[str] = None,
        **kwargs
    ) -> None:
        mode = model.mode
        with_edge = False if not mode.is_advection() and not mode.is_baroclinic() else True
        with_vtx = True if not mode.is_advection() else False
        super().__init__(
            model=model,
            case_name=case_name,
            file_path=file_path,
            output_h0=output_h0,
            output_h1=output_h1,
            output_h2=output_h2,
            h0_opt={"with_vtx": with_vtx, "with_edge": with_edge},
            h1_opt={"with_vtx": with_vtx, "with_edge": True},
            output_ngroups=output_ngroups,
            dev=dev,
            float_type=float_type,
            int_type=int_type,
            **kwargs,
        )

    @staticmethod
    def history_type() -> Type["SCVTHistory"]:
        return SCVTHistory

    @staticmethod
    def structured_history_type() -> Type["SCVT2StructuredHistory"]:
        return SCVT2StructuredHistory


class AbstractBaseHistory(AbstractHistory):
    file_path    : str
    time_manager : DateTimeManager
    float_type   : type[np.float32] | type[np.float64]
    int_type     : type[np.int32  ] | type[np.int64  ]
    nc_float_type: str
    nc_int_type  : str
    proc         : Proc
    local_mesh   : SCVTMesh
    global_mesh  : SCVTMesh
    comm_io      : MPIComm
    io_rank      : int
    io_nproc     : int
    output_ngroups: int
    grouped_io   : bool
    io_maps      : dict[str, SCVTHistoryIOMap]
    built        : bool

    @abstractmethod
    def __init__(
        self,
        model: MODEL,
        history_name: str,
        case_name: str,
        file_path: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64],
        output_ngroups: int = 0,
    ) -> None:
        super().__init__(
            model=model,
            history_name=history_name,
            case_name=case_name,
            file_path=file_path,
            float_type=float_type,
            int_type=int_type
        )
        block = model.block
        mesh  = block.mesh
        proc  = block.dev.proc
        comm  = proc.comm_model
        if not proc.is_model() or comm is None or comm == MPI.COMM_NULL:
            raise RuntimeError(
                "History output must be created on model ranks."
            )
        output_ngroups = int(output_ngroups)
        grouped_io = output_ngroups > 0
        if grouped_io:
            n_global_min = min(
                mesh.parent.n_cell,
                mesh.parent.n_vtx,
                mesh.parent.n_edge
            )
            if output_ngroups > proc.np_model:
                raise ValueError(
                    "SCVT history output_ngroups must not exceed the IO communicator size."
                )
            if output_ngroups > n_global_min:
                raise ValueError(
                    "SCVT history output_ngroups must not exceed the minimum horizontal dimension."
                )
            if proc.np_model % output_ngroups != 0:
                raise ValueError(
                    "SCVT history output_ngroups must evenly divide the IO communicator size."
                )
        else:
            output_ngroups = 1
        if proc.is_root():
            ds = nc4.Dataset(self.file_path, mode="w", format="NETCDF4")
            ds.sync()
            ds.close()
        comm.Barrier()
        self.proc           = proc
        self.local_mesh     = mesh
        self.global_mesh    = mesh.parent
        self.comm_io        = comm
        self.io_rank        = comm.rank
        self.io_nproc       = comm.size
        self.output_ngroups = output_ngroups
        self.grouped_io     = grouped_io
        self.io_maps = {
            "cell": SCVTHistoryIOMap(self, "cell", self.global_mesh.n_cell),
            "vtx" : SCVTHistoryIOMap(self, "vtx" , self.global_mesh.n_vtx ),
            "edge": SCVTHistoryIOMap(self, "edge", self.global_mesh.n_edge),
        }

    @abstractmethod
    def setup(self, *args, **kwargs) -> Self:
        raise NotImplementedError

    def end_setup(self):
        self.built = True

    def has_built(self) -> bool:
        return self.built

    def open_dataset(self) -> nc4.Dataset:
        return nc4.Dataset(self.file_path, mode="r")

    def is_root(self) -> bool:
        return self.proc.is_root()

    def sync(self) -> None:
        self.proc.comm_model.Barrier()

    def global_spatial_size(self, meta: SCVTFieldMeta) -> int:
        if meta.at_cell:
            return int(self.global_mesh.n_cell)
        if meta.at_vtx:
            return int(self.global_mesh.n_vtx)
        if meta.at_edge:
            return int(self.global_mesh.n_edge)
        raise ValueError(
            f"Unsupported field location {meta.loc} for variable {meta.name}."
        )

    def io_map(self, meta: SCVTFieldMeta) -> SCVTHistoryIOMap:
        if meta.at_cell:
            return self.io_maps["cell"]
        if meta.at_vtx:
            return self.io_maps["vtx"]
        if meta.at_edge:
            return self.io_maps["edge"]
        raise ValueError(
            f"Unsupported field location {meta.loc} for variable {meta.name}."
        )

    def owned_global_indices(self, meta: SCVTFieldMeta) -> np.ndarray:
        if meta.at_cell:
            global_id = self.local_mesh.global_cell_id
        elif meta.at_vtx:
            global_id = self.local_mesh.global_vtx_id
        elif meta.at_edge:
            global_id = self.local_mesh.global_edge_id
        else:
            raise ValueError(
                f"Unsupported field location {meta.loc} for variable {meta.name}."
            )
        global_id = self.backend.to_numpy(global_id)
        if global_id.size == 0:
            return np.arange(meta.ids, meta.ide + 1, dtype=np.int32)
        return global_id[meta.ids:meta.ide + 1]

    def source_owns_indices(self, name: str, global_indices: np.ndarray) -> np.ndarray:
        if self.io_nproc == 1:
            return np.ones(global_indices.shape, dtype=bool)
        if name == "cell":
            cell_owner = self.backend.to_numpy(self.global_mesh.cell_proc_idmap)
            return cell_owner[global_indices] == self.proc.id_model
        if name == "vtx":
            cells = self.backend.to_numpy(self.global_mesh.cell_of_vtx )[global_indices]
        elif name == "edge":
            cells = self.backend.to_numpy(self.global_mesh.cell_of_edge)[global_indices]
        else:
            raise ValueError(f"Invalid SCVT history IO map name {name}.")
        valid = cells >= 0
        cell_owner = self.backend.to_numpy(self.global_mesh.cell_proc_idmap)
        owner = np.min(
            np.where(valid, cell_owner[np.where(valid, cells, 0)], np.iinfo(np.int64).max),
            axis=1
        )
        return owner == self.proc.id_model

    def _gather_field_data(
        self,
        field_data: Array,
        meta: SCVTFieldMeta,
    ) -> np.ndarray | None:
        history_io_map = self.io_map(meta)
        pieces = history_io_map.collect_root(field_data, meta)
        if self.is_root():
            assert pieces is not None
            fill_value = np.nan
            global_data = np.full(
                pieces[0][1].shape[:-1] + (self.global_spatial_size(meta),),
                fill_value
            )
            for global_indices, data in pieces:
                global_data[..., global_indices] = data
            return global_data
        return None

    def _gather_field_pieces(
        self,
        field_data: Array,
        meta: SCVTFieldMeta
    ) -> list[tuple[np.ndarray, np.ndarray]] | None:
        history_io_map = self.io_map(meta)
        if self.grouped_io:
            return history_io_map.collect(field_data, meta)
        return history_io_map.collect_root(field_data, meta)

    def _write_spatial_range(
        self,
        ds: nc4.Dataset,
        var_name: str,
        data: np.ndarray,
        start: int,
        count: int,
        time_idx: int | None
    ) -> None:
        if count == 0:
            return None
        var = ds.variables[var_name]
        spatial_slice = slice(start, start + count)
        if time_idx is None:
            nc_idx = (slice(None),) * (var.ndim - 1) + (spatial_slice,)
        else:
            nc_idx = (time_idx,) + (slice(None),) * (var.ndim - 2) + (spatial_slice,)
        var[nc_idx] = data
        return None

    def _write_field_pieces(
        self,
        ds: nc4.Dataset,
        field: SCVTFieldLike,
        pieces: list[tuple[np.ndarray, np.ndarray]],
        time_idx: int,
        tracer_idx: Optional[list[int]],
        tracers_meta: Optional[dict[str, TracerMeta]]
    ) -> None:
        meta = field.meta
        history_io_map = self.io_map(meta)
        if not history_io_map.is_writer:
            return None
        writer_data = history_io_map.assemble_writer_range(pieces, self.float_type)
        write_time_idx = None if meta.is_static else time_idx
        if not meta.is_tracer:
            self._write_spatial_range(
                ds=ds,
                var_name=meta.name,
                data=writer_data,
                start=history_io_map.writer_start,
                count=history_io_map.writer_count,
                time_idx=write_time_idx
            )
            return None
        if tracers_meta is None:
            raise ValueError("tracers_meta is required for tracer fields.")
        if tracer_idx is None:
            raise ValueError("tracer_idx is required for tracer fields.")
        tracer_indices = set(np.asarray(tracer_idx).reshape(-1).astype(int).tolist())
        for q_idx, tracer_name in enumerate(tracers_meta.keys()):
            if q_idx not in tracer_indices:
                continue
            self._write_spatial_range(
                ds=ds,
                var_name=tracer_name,
                data=writer_data[q_idx, ...],
                start=history_io_map.writer_start,
                count=history_io_map.writer_count,
                time_idx=write_time_idx
            )
        return None

    def write_fields(
        self,
        fields: SCVTFields,
        dim0_idx: Optional[int] = None,
        tracer_idx: Optional[list[int]] = None,
        **kwargs
    ) -> None:
        io_pending: list[
            tuple[SCVTFieldLike, list[tuple[np.ndarray, np.ndarray]]]
        ] = []
        scalar_pending: list[tuple[str, np.ndarray, bool]] = []
        tracers_meta: dict[str, TracerMeta] | None = None
        if hasattr(fields, "tracers_meta"):
            tracers_meta = getattr(fields, "tracers_meta")

        for field_name in fields.keys():
            field = getattr(fields, field_name)
            if field.meta.output != self.history_name or field.meta.output == "":
                continue
            if dim0_idx is None:
                data = field.unwrap()
            else:
                data = field.unwrap()[dim0_idx, ...]
            if isinstance(field.meta, SCVTFieldMeta):
                pieces = self._gather_field_pieces(data, field.meta)
                if self.io_map(field.meta).is_writer:
                    assert pieces is not None
                    io_pending.append((field, pieces))
            elif isinstance(field.meta, ScalarFieldMeta):
                if self.is_root():
                    scalar_pending.append((
                        field.meta.name,
                        self.backend.to_numpy(data),
                        field.meta.is_static
                    ))
            else:
                raise ValueError(
                    f"Unsupported field meta data type: {type(field.meta)}"
                )

        for group_id in range(self.output_ngroups):
            writer_rank, _ = _history_group_bounds(
                self.io_nproc,
                self.output_ngroups,
                group_id
            )
            if self.io_rank == writer_rank:
                ds = nc4.Dataset(self.file_path, mode="a")
                try:
                    idx = len(ds.variables["time"]) - 1
                    for field, pieces in io_pending:
                        self._write_field_pieces(
                            ds=ds,
                            field=field,
                            pieces=pieces,
                            time_idx=idx,
                            tracer_idx=tracer_idx,
                            tracers_meta=tracers_meta
                        )
                    if self.is_root():
                        for var_name, data, is_static in scalar_pending:
                            if is_static:
                                ds.variables[var_name][...] = data
                            else:
                                ds.variables[var_name][idx] = data
                finally:
                    ds.close()
            self.sync()
        return None

    @abstractmethod
    def add_fields(self, fields: SCVTFields) -> None:
        raise NotImplementedError


class UGRIDHistory(AbstractBaseHistory):
    use_dual: bool
    attr_setup: GlobalAttributesSetup
    time_setup: TimeSetup
    mesh_setup: AbstractMeshHistorySetup
    vert_setup: AbstractVertHistorySetup

    def __init__(
        self,
        model: MODEL,
        history_name: str,
        case_name: str,
        file_path: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64],
        output_ngroups: int = 0,
        use_dual: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            history_name=history_name,
            case_name=case_name,
            file_path=file_path,
            float_type=float_type,
            int_type=int_type,
            output_ngroups=output_ngroups
        )
        self.use_dual = use_dual
        self.attr_setup = GlobalAttributesSetup(
            file_path=self.file_path,
            float_type=self.float_type,
            int_type=self.int_type,
            attrs=model.cfg.to_dict(keep_bool=False)
        )
        self.time_setup = TimeSetup(
            file_path=self.file_path,
            start_time_str=self.time_manager.start_time_str,
            save_interval_unit=self.time_manager.save_interval_unit,
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.mesh_setup = self.global_mesh.history_builder(
            file_path=self.file_path,
            output_type="ugrid",
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.vert_setup = model.block.vert_coord.history_builder(
            file_path=self.file_path,
            output_type="mlev",
            float_type=self.float_type,
            int_type=self.int_type
        )

    def setup(self, fields: SCVTFields) -> Self:
        if self.has_built():
            return self
        if self.is_root():
            if not self.attr_setup.has_built():
                self.attr_setup.setup()
            if not self.time_setup.has_built():
                self.time_setup.setup()
            if not self.mesh_setup.has_built():
                self.mesh_setup.setup(use_dual=self.use_dual)
            if not self.vert_setup.has_built():
                self.vert_setup.setup()
            if (
                self.attr_setup.has_built()
                and self.time_setup.has_built()
                and self.mesh_setup.has_built()
                and self.vert_setup.has_built()
                and not self.built
            ):
                self.add_fields(fields)
        self.sync()
        return self

    def add_fields(self, fields: SCVTFields) -> None:
        if not self.is_root():
            return None
        ds = nc4.Dataset(self.file_path, mode="a")
        var_dtype = self.nc_float_type
        for _, field in fields.items():
            if field.meta.output != self.history_name or field.meta.output == "":
                continue
            if isinstance(field.meta, SCVTFieldMeta):
                dsxr = meta2ugrid(
                    value=field.d,
                    meta=field.meta, 
                    use_dual=self.use_dual, 
                    dim0_idx=0,
                )
                var_dims = tuple(str(dim) for dim in dsxr.dims)
                if not field.meta.is_static:
                    var_dims = tuple(("time",) + var_dims)
                if not field.meta.is_tracer:
                    v = ds.createVariable(field.meta.name, var_dtype, var_dims)
                    v.mesh = dsxr.attrs.get("mesh", "")
                    v.location = dsxr.attrs.get("location", "")
                    v.coordinates = dsxr.attrs.get("coordinates", "")
                    v.units = dsxr.attrs.get("units", "")
                    v.long_name = dsxr.attrs.get("long_name", "")
                    v.description = dsxr.attrs.get("description", "")
                else:
                    if hasattr(fields, "tracers_meta"):
                        tracers_meta: dict[str, TracerMeta] = getattr(
                            fields, "tracers_meta"
                        )
                    else:
                        raise ValueError(
                            "tracers_meta is required for tracer fields."
                        )
                    for tracer_name, tracer_meta in tracers_meta.items():
                        q_name = tracer_name
                        q_long_name = (
                            dsxr.attrs.get("long_name", "")
                            + " of "
                            + tracer_meta.long_name.lower()
                        )
                        q_units = dsxr.attrs.get("units", "")
                        v = ds.createVariable(q_name, var_dtype, var_dims)
                        v.mesh = dsxr.attrs.get("mesh", "")
                        v.location = dsxr.attrs.get("location", "")
                        v.coordinates = dsxr.attrs.get("coordinates", "")
                        v.units = q_units
                        v.long_name = q_long_name
            elif isinstance(field.meta, ScalarFieldMeta):
                var_dims = ("time",) if not field.meta.is_static else ()
                v = ds.createVariable(field.meta.name, var_dtype, var_dims)
                v.units = field.meta.units
                v.long_name = field.meta.long_name
            else:
                raise ValueError(
                    f"Unsupported field meta data type: {type(field.meta)}"
                )
        ds.close()
        return None


class SCVTHistory(AbstractBaseHistory):
    file_path: str
    time_manager: DateTimeManager
    attr_setup: GlobalAttributesSetup
    time_setup: TimeSetup
    mesh_setup: AbstractMeshHistorySetup
    vert_setup: AbstractVertHistorySetup
    built: bool

    def __init__(
        self,
        model: MODEL,
        history_name: str,
        case_name: str,
        file_path: str,
        with_vtx: bool,
        with_edge: bool,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64],
        output_ngroups: int = 0,
    ) -> None:
        super().__init__(
            model=model,
            history_name=history_name,
            case_name=case_name,
            file_path=file_path,
            float_type=float_type,
            int_type=int_type,
            output_ngroups=output_ngroups
        )
        self.attr_setup = GlobalAttributesSetup(
            file_path=self.file_path,
            float_type=self.float_type,
            int_type=self.int_type,
            attrs=model.cfg.to_dict(keep_bool=False)
        )
        self.time_setup = TimeSetup(
            file_path=self.file_path,
            start_time_str=model.time_manager.start_time_str,
            save_interval_unit=model.time_manager.save_interval_unit,
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.mesh_setup = self.global_mesh.history_builder(
            file_path=self.file_path,
            output_type="unstructured",
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.vert_setup = model.block.vert_coord.history_builder(
            file_path=self.file_path,
            output_type="mlev",
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.with_vtx = with_vtx
        self.with_edge = with_edge

    def setup(self, fields: SCVTFields) -> Self:
        if self.has_built():
            return self
        if self.is_root():
            if not self.attr_setup.has_built():
                self.attr_setup.setup()
            if not self.time_setup.has_built():
                self.time_setup.setup()
            if not self.mesh_setup.has_built():
                self.mesh_setup.setup(
                    with_vtx=self.with_vtx,
                    with_edge=self.with_edge
                )
            if not self.vert_setup.has_built():
                self.vert_setup.setup()
            if (
                self.attr_setup.has_built()
                and self.time_setup.has_built()
                and self.mesh_setup.has_built()
                and self.vert_setup.has_built()
                and not self.built
            ):
                self.add_fields(fields)
        self.sync()
        return self

    def add_fields(self, fields: SCVTFields) -> None:
        if not self.is_root():
            return None
        ds = nc4.Dataset(self.file_path, mode="a")
        var_dtype = self.nc_float_type
        for _, field in fields.items():
            if field.meta.output != self.history_name or field.meta.output == "":
                continue
            if isinstance(field.meta, SCVTFieldMeta):
                var_dims = field.meta.output_dims
                if not field.meta.is_tracer:
                    v = ds.createVariable(field.meta.name, var_dtype, var_dims)
                    v.loc = field.meta.loc
                    v.coordinates = field.meta.output_coordinate
                    v.units = field.meta.units
                    v.long_name = field.meta.long_name
                else:
                    if hasattr(fields, "tracers_meta"):
                        tracers_meta: dict[str, TracerMeta] = getattr(
                            fields, "tracers_meta"
                        )
                    else:
                        raise ValueError(
                            "tracers_meta is required for tracer fields."
                        )
                    for tracer_name, tracer_meta in tracers_meta.items():
                        q_name = tracer_name
                        q_long_name = (
                            field.meta.long_name
                            + " of "
                            + tracer_meta.long_name.lower()
                        )
                        q_units = field.meta.units
                        v = ds.createVariable(q_name, var_dtype, var_dims)
                        v.loc = field.meta.loc
                        v.coordinates = field.meta.output_coordinate
                        v.units = q_units
                        v.long_name = q_long_name

            elif isinstance(field.meta, ScalarFieldMeta):
                var_dims = ("time",) if not field.meta.is_static else ()
                v = ds.createVariable(field.meta.name, var_dtype, var_dims)
                v.units = field.meta.units
                v.long_name = field.meta.long_name
            else:
                raise ValueError(
                    f"Unsupported field meta data type: {type(field.meta)}"
                )
        ds.close()
        return None


class SCVT2StructuredHistory(AbstractBaseHistory):
    file_path: str
    time_manager: DateTimeManager
    mesh: SCVTMesh
    attr_setup: GlobalAttributesSetup
    time_setup: TimeSetup
    mesh_setup: AbstractMeshHistorySetup
    vert_setup: AbstractVertHistorySetup
    structured_lon: np.ndarray
    structured_lat: np.ndarray
    structured_lev: np.ndarray
    hori_regrid_method: str
    hori_regrid_backend: str
    vert_interp: bool
    _regridders: dict[tuple[str, Optional[int]], ESMFRegridder]
    built: bool

    def __init__(
        self,
        model: MODEL,
        history_name: str,
        case_name: str,
        file_path: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64],
        hori_regrid_method: str = "conserve_1st",
        hori_regrid_backend: str = "esmf",
        vert_interp: bool = True,
        lon_west: float = -180,
        lon_east: float = 180,
        lat_south: float = -90,
        lat_north: float = 90,
        east_west_resolution: float = 1.0,
        south_north_resolution: float = 1.0,
        lon: Optional[np.ndarray | Array] = None,
        lat: Optional[np.ndarray | Array] = None,
        plev: np.ndarray | Array = None,
        plev_unit: str = "hPa",
        output_ngroups: int = 0,
    ) -> None:
        if vert_interp and model.mode.is_2d_model():
            log.logger.warning(
                "Pressure-level interpolation is unavailable for 2D models; "
                "h2 output will use model levels."
            )
            vert_interp = False
        super().__init__(
            model=model,
            history_name=history_name,
            case_name=case_name,
            file_path=file_path,
            float_type=float_type,
            int_type=int_type,
            output_ngroups=output_ngroups
        )
        self.attr_setup = GlobalAttributesSetup(
            file_path=self.file_path,
            float_type=self.float_type,
            int_type=self.int_type,
            attrs=model.cfg.to_dict(keep_bool=False)
        )
        self.time_setup = TimeSetup(
            file_path=self.file_path,
            start_time_str=model.time_manager.start_time_str,
            save_interval_unit=model.time_manager.save_interval_unit,
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.mesh_setup = self.global_mesh.history_builder(
            file_path=self.file_path,
            output_type="structured",
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.vert_setup = model.block.vert_coord.history_builder(
            file_path=self.file_path,
            output_type="plev" if vert_interp else "mlev",
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.para = model.block.para
        self.mesh = self.local_mesh
        self.hori_regrid_method = hori_regrid_method
        self.hori_regrid_backend = hori_regrid_backend
        self.vert_interp = vert_interp
        
        if plev is None:
            plev = np.array(
                [10, 20, 50, 100, 200, 300, 500, 700, 850, 925, 1000], 
                dtype=self.float_type
            )

        if lon is not None and lat is not None:
            self.structured_lon = np.array(lon , self.float_type)
            self.structured_lat = np.array(lat , self.float_type)
            self.structured_lev = np.array(plev, self.float_type)
        else:
            self.structured_lon = np.arange(
                lon_west, lon_east, east_west_resolution, dtype=self.float_type
            )
            self.structured_lat = np.arange(
                lat_south, lat_north, south_north_resolution, dtype=self.float_type
            )
            if plev_unit.lower() == "hpa" or plev_unit.lower() == "mb":
                self.structured_lev = np.array(plev, self.float_type) * 100.0
            elif plev_unit.lower() == "pa":
                self.structured_lev = np.array(plev, self.float_type)
            else:
                raise ValueError(
                    f"Unsupported plev_unit: {plev_unit}, expected 'hPa', 'Pa', or 'mb'."
                )
            self.structured_lev = np.sort(self.structured_lev)

        self._regridders = {}

    def setup(self, fields: SCVTFields) -> Self:
        if self.has_built():
            return self
        if self.is_root():
            if not self.attr_setup.has_built():
                self.attr_setup.setup()
            if not self.time_setup.has_built():
                self.time_setup.setup()
            if not self.mesh_setup.has_built():
                self.mesh_setup.setup(
                    dst_lon=self.structured_lon,
                    dst_lat=self.structured_lat,
                )
            if not self.vert_setup.has_built():
                self.vert_setup.setup(
                    lev=self.structured_lev,
                    lev_unit="Pa",
                )
            if (
                self.attr_setup.has_built()
                and self.time_setup.has_built()
                and self.mesh_setup.has_built()
                and self.vert_setup.has_built()
                and not self.built
            ):
                self.add_fields(fields)
        self.sync()
        self._prebuild_regridders(fields)
        return self

    def add_fields(self, fields: SCVTFields) -> None:
        if not self.is_root():
            return None
        ds = nc4.Dataset(self.file_path, mode="a", format="NETCDF4")
        var_dtype = self.nc_float_type
        for _, field in fields.items():
            meta = field.meta
            if meta.output not in STRUCTURED_HISTORY_LIST:
                continue
            if meta.output == "":
                continue
            if meta.name in PLEV_NAMES and self.vert_interp:
                continue

            if isinstance(meta,SCVTFieldMeta):
                var_dims = scvt2structured_var_dims(meta, self.vert_interp)
                if not meta.is_static:
                    var_dims = tuple(("time",) + var_dims)

                if not meta.is_tracer:
                    v = ds.createVariable(
                        str(meta.name), var_dtype, var_dims
                    )
                    v.setncattr("mesh", "primal")
                    v.setncattr("units", meta.units)
                    v.setncattr("long_name", meta.long_name)
                    v.setncattr("coordinates", "lon lat")
                else:
                    if hasattr(fields, "tracers_meta"):
                        tracers_meta: dict[str, TracerMeta] = getattr(
                            fields, "tracers_meta"
                        )
                    else:
                        raise ValueError(
                            "tracers_meta is required for tracer fields."
                        )
                    for tracer_name, tracer_meta in tracers_meta.items():
                        q_name = tracer_name
                        q_long_name = (
                            meta.long_name
                            + " of "
                            + tracer_meta.long_name.lower()
                        )
                        q_units = meta.units
                        v = ds.createVariable(q_name, var_dtype, var_dims)
                        v.setncattr("mesh", "primal")
                        v.setncattr("coordinates", "lon lat")
                        v.setncattr("units", q_units)
                        v.setncattr("long_name", q_long_name)
            elif isinstance(meta, ScalarFieldMeta):
                is_static = meta.is_static
                var_dims = () if is_static else ("time",)
                v = ds.createVariable(
                    str(meta.name), var_dtype, var_dims
                )
                v.setncattr("units", meta.units)
                v.setncattr("long_name", meta.long_name)
            else:
                raise ValueError(
                    f"Unsupported meta data type for variable {meta.name}: {type(meta)}."
                )
        ds.close()
        return None

    def write_fields(
        self,
        fields: SCVTFields,
        dim0_idx: Optional[int] = None,
        tracer_idx: Optional[list[int]] = None,
        ph: Optional[np.ndarray] = None,
        ph_lev: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        remapped_pending: list[tuple[SCVTFieldLike, np.ndarray, Optional[int]]] = []
        scalar_pending: list[tuple[str, np.ndarray, bool]] = []
        tracers_meta: dict[str, TracerMeta] | None = None
        if hasattr(fields, "tracers_meta"):
            tracers_meta = getattr(fields, "tracers_meta")

        for _, field in fields.items():
            meta = field.meta
            if meta.output not in STRUCTURED_HISTORY_LIST or meta.output == "":
                continue
            if isinstance(field, SCVTFieldLike):
                if field.meta.name in PLEV_NAMES and self.vert_interp:
                    continue
                if dim0_idx is None:
                    field_data = field.unwrap()
                else:
                    field_data = field.unwrap()[dim0_idx, ...]
                ndbound = self._get_field_ndbound(field.meta)
                remapped = self._remap_field(
                    field_data=self.backend.to_numpy(field_data),
                    meta=field.meta,
                    ph=ph,
                    ph_lev=ph_lev,
                )
                remapped_pending.append((field, remapped, ndbound))
            elif isinstance(field, ScalarField):
                if self.is_root():
                    if dim0_idx is None:
                        field_data = field.unwrap()
                    else:
                        field_data = field.unwrap()[dim0_idx, ...]
                    scalar_pending.append((
                        meta.name,
                        self.backend.to_numpy(field_data),
                        meta.is_static,
                    ))
            else:
                raise ValueError(
                    f"Unsupported meta data type for variable {meta.name}: {type(meta)}."
                )

        for writer_rank in range(self.io_nproc):
            if self.io_rank == writer_rank:
                ds = nc4.Dataset(self.file_path, mode="a")
                try:
                    idx = len(ds.variables["time"]) - 1
                    for field, data, ndbound in remapped_pending:
                        self._write_remapped_field(
                            ds=ds,
                            field=field,
                            data=data,
                            ndbound=ndbound,
                            time_idx=idx,
                            tracer_idx=tracer_idx,
                            tracers_meta=tracers_meta,
                        )
                    if self.is_root():
                        for var_name, data, is_static in scalar_pending:
                            if is_static:
                                ds.variables[var_name][...] = data
                            else:
                                ds.variables[var_name][idx] = data
                finally:
                    ds.close()
            self.sync()
        self.sync()
        return None

    def _write_structured_var(
        self,
        ds: nc4.Dataset,
        var_name: str,
        data: np.ndarray,
        lat_slice: slice,
        lon_slice: slice,
        time_idx: int | None,
    ) -> None:
        if (lat_slice.stop or 0) <= (lat_slice.start or 0):
            return None
        if (lon_slice.stop or 0) <= (lon_slice.start or 0):
            return None
        var = ds.variables[var_name]
        spatial_idx = (lat_slice, lon_slice)
        if time_idx is None:
            nc_idx = (slice(None),) * (var.ndim - 2) + spatial_idx
        else:
            nc_idx = (time_idx,) + (slice(None),) * (var.ndim - 3) + spatial_idx
        var[nc_idx] = data
        return None

    def _write_remapped_field(
        self,
        ds: nc4.Dataset,
        field: SCVTFieldLike,
        data: np.ndarray,
        ndbound: Optional[int],
        time_idx: int,
        tracer_idx: Optional[list[int]],
        tracers_meta: Optional[dict[str, TracerMeta]],
    ) -> None:
        meta = field.meta
        lat_slice, lon_slice = self._dst_grid_slices(meta=meta, ndbound=ndbound)
        write_time_idx = None if meta.is_static else time_idx
        if not meta.is_tracer:
            self._write_structured_var(
                ds=ds,
                var_name=meta.name,
                data=data,
                lat_slice=lat_slice,
                lon_slice=lon_slice,
                time_idx=write_time_idx,
            )
            return None
        if tracers_meta is None:
            raise ValueError("tracers_meta is required for tracer fields.")
        if tracer_idx is None:
            raise ValueError("tracer_idx is required for tracer fields.")
        tracer_indices = set(
            self.backend.to_numpy(tracer_idx).reshape(-1).astype(int).tolist()
        )
        for q_idx, tracer_name in enumerate(tracers_meta.keys()):
            if q_idx not in tracer_indices:
                continue
            self._write_structured_var(
                ds=ds,
                var_name=tracer_name,
                data=data[q_idx, ...],
                lat_slice=lat_slice,
                lon_slice=lon_slice,
                time_idx=write_time_idx,
            )
        return None

    def _get_regridder(
        self,
        meta: SCVTFieldMeta,
        ndbound: Optional[int],
    ) -> ESMFRegridder:
        from terraflow.mesh.scvt._esmf_regridder import ESMFRegridder

        loc = meta.loc
        key = (loc, ndbound)
        if key not in self._regridders:
            if self.has_built():
                available = sorted(
                    (repr(item) for item in self._regridders.keys())
                )
                raise RuntimeError(
                    "Missing prebuilt h2 ESMF regridder "
                    f"for key={key}, variable={meta.name!r}; "
                    f"available keys={available}."
                )
            lon, lat = self.structured_lon, self.structured_lat
            self._regridders[key] = ESMFRegridder(
                src=ESMFRegridder.create_scvt_mesh(
                    para=self.para,
                    mesh=self.mesh,
                    loc=loc,
                    layout="triskc",
                    with_halo=False,
                ),
                dst=ESMFRegridder.create_latlon_grid(
                    lon=np.array(lon),
                    lat=np.array(lat)
                ),
                backend=self.backend,
                ndbounds=None if ndbound is None else (ndbound,),
                method=self.hori_regrid_method,
                extrap=True,
                conserve_opt={"norm_opt": "fracarea"},
                precompute_weights=True,
            )
        return self._regridders[key]

    def _get_field_ndbound(self, meta: SCVTFieldMeta) -> Optional[int]:
        if meta.at_sfc:
            return None
        return int(meta.nlev)

    def _prebuild_regridders(self, fields: SCVTFields) -> None:
        if self.hori_regrid_backend != "esmf":
            return None
        for _, field in fields.items():
            meta = field.meta
            if meta.output not in STRUCTURED_HISTORY_LIST or meta.output == "":
                continue
            if not isinstance(field, SCVTFieldLike):
                continue
            ndbound = self._get_field_ndbound(field.meta)
            self._get_regridder(meta=field.meta, ndbound=ndbound)
        self.sync()
        return None

    def _dst_grid_slices(
        self,
        meta: SCVTFieldMeta,
        ndbound: Optional[int],
    ) -> tuple[slice, slice]:
        return self._get_regridder(meta=meta, ndbound=ndbound).dst_grid_slices()

    def _horizontal_remap(
        self,
        field_data: np.ndarray,
        meta: SCVTFieldMeta,
        ndbound: Optional[int],
    ) -> np.ndarray:
        bkd = self.backend
        regridder = self._get_regridder(meta=meta, ndbound=ndbound)
        field_data = field_data[..., meta.ids:meta.ide + 1]
        if meta.is_tracer:
            return np.stack(
                [
                    self.backend.to_numpy(
                        regridder.regrid(
                            bkd.asarray(field_data[q_idx, ...])
                        )
                    )
                    for q_idx in range(field_data.shape[0])
                ]
            )
        return self.backend.to_numpy(regridder.regrid(bkd.asarray(field_data)))

    def _remap_field(
        self,
        field_data: np.ndarray,
        meta: SCVTFieldMeta,
        ph: Optional[np.ndarray] = None,
        ph_lev: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if meta.at_sfc:
            if meta.is_tracer:
                if field_data.ndim == 3 and field_data.shape[1] == 1:
                    field_data = np.squeeze(field_data, axis=1)
                elif field_data.ndim != 2:
                    raise ValueError(
                        f"Tracer variable {meta.name} has shape {field_data.shape}, "
                        f"expected (ntracers, npoints) for {meta.loc!r}."
                    )
            else:
                if field_data.ndim == 2 and field_data.shape[0] == 1:
                    field_data = np.squeeze(field_data, axis=0)
                elif field_data.ndim != 1:
                    raise ValueError(
                        f"Variable {meta.name} has shape {field_data.shape}, "
                        f"expected (npoints,) for {meta.loc!r}."
                    )
            ndbound = None
        else:
            ndbound = self._get_field_ndbound(meta)
            # (ntracer, nlev, npoints)
            if meta.is_tracer:
                if field_data.ndim != 3:
                    raise ValueError(
                        f"Variable {meta.name} is a tracer with location "
                        f"{meta.loc}, but has shape {field_data.shape}, "
                        f"which does not match metadata dimensions {meta.dims}."
                    )
                if field_data.shape[1] != ndbound:
                    raise ValueError(
                        f"Variable {meta.name} has shape {field_data.shape}, "
                        f"expected vertical dimension {ndbound} from metadata."
                    )
            # (nlev, npoints)
            else:
                if field_data.ndim != 2:
                    raise ValueError(
                        f"Variable {meta.name} has shape {field_data.shape}, "
                        f"expected (nlev, npoints) for {meta.loc!r}."
                    )
                if field_data.shape[0] != ndbound:
                    raise ValueError(
                        f"Variable {meta.name} has shape {field_data.shape}, "
                        f"expected vertical dimension {ndbound} from metadata."
                    )
        # =================================================
        # Horizontal regridding
        if meta.name in PLEV_NAMES:
            if ph is None or ph_lev is None:
                raise ValueError(
                    f"Pressure fields ph and ph_lev are required for variable {meta.name}."
                )
            else:
                if meta.name == "ph":
                    return ph
                elif meta.name == "ph_lev":
                    return ph_lev
                else:
                    raise ValueError(
                        f"Unsupported pressure variable {meta.name}."
                    )
        hori_remapped = self._horizontal_remap(
            field_data=field_data,
            meta=meta,
            ndbound=ndbound,
        )

        if meta.at_sfc or not self.vert_interp:
            return hori_remapped
        # =================================================
        if meta.loc in ("cell", "vtx", "edge"):
            pi = ph
        elif meta.loc in ("lev", "lev_vtx", "lev_edge"):
            pi = ph_lev
        else:
            raise ValueError(
                f"Unsupported location {meta.loc} for variable {meta.name}."
            )
        # Vertical interpolation if applicable
        if meta.is_tracer:
            xi = np.moveaxis(np.asarray(hori_remapped), 1, 0)
            pi_batched = np.broadcast_to(
                np.asarray(pi)[:, None, ...],
                xi.shape,
            )
            vert_remapped = interp_plev(
                mesh=self.mesh,
                pi=pi_batched,
                xi=xi,
                po=self.structured_lev,
                xo_loc=meta.loc,
                logp=meta.name in VARNAMES_USING_LOG_INTERP,
            )
            return np.moveaxis(np.asarray(vert_remapped), 1, 0)
        vert_remapped = interp_plev(
            mesh=self.mesh,
            pi=np.asarray(pi),
            xi=np.asarray(hori_remapped),
            po=np.asarray(self.structured_lev),
            xo_loc=meta.loc,
            logp=meta.name in VARNAMES_USING_LOG_INTERP,
        )
        return np.asarray(vert_remapped)

    def update_plev(
        self,
        ph: Optional[SCVTFieldLike],
        ph_lev: Optional[SCVTFieldLike],
        itime: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if ph is None or ph_lev is None:
            return None, None
        if ph.meta.is_empty() or ph_lev.meta.is_empty():
            return None, None
        if ph.unwrap().shape[0] == 0 or ph_lev.unwrap().shape[0] == 0:
            return None, None
        ph_data = ph.unwrap()[itime, ...]
        ph_lev_data = ph_lev.unwrap()[itime, ...]
        ph_remapped = self._horizontal_remap(
            field_data=self.backend.to_numpy(ph_data),
            meta=ph.meta,
            ndbound=self._get_field_ndbound(ph.meta),
        )
        ph_lev_remapped = self._horizontal_remap(
            field_data=self.backend.to_numpy(ph_lev_data),
            meta=ph_lev.meta,
            ndbound=self._get_field_ndbound(ph_lev.meta),
        )
        return ph_remapped, ph_lev_remapped


def scvt2structured_var_dims(meta: SCVTFieldMeta, vert_interp: bool) -> tuple[str, ...]:
    # For sfc variable -> (lat, lon)
    # For variables with vertical dimension -> (lev, lat, lon)
    # For tracer 3D variables -> (ntracers, lev, lat, lon)
    # For tracer 2D variables -> (ntracers, lat, lon)
    dims = tuple(str(dim) for dim in meta.dims)

    # For sfc variable
    if len(dims) == 1:
        leading_dims = ()
    # For variables >= 2D
    elif len(dims) >= 2:
        # For variables with vertical dimension
        if vert_interp:
            if dims[-2] in LEV_DIMS:
                dims = dims[:-2] + ("lev",) + dims[-1:]

    # Process leading dimensions for spatial variables
    if len(dims) > 0 and (dims[-1] in UGRID_SPATIAL_DIMS or dims[-1] in UNSTRUCTURED_SPATIAL_DIMS):
        leading_dims = dims[:-1]
    else:
        leading_dims = dims

    # Process leading dimensions for tracer variables
    if meta.is_tracer and leading_dims[:1] == ("ntracers",):
        leading_dims = leading_dims[1:]
    return tuple(leading_dims + ("lat", "lon"))
