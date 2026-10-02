from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar
from typing import Optional
import time
import datetime


from terraflow.util import log


if TYPE_CHECKING:
    from terraflow.util.device import AbstractDevice
    from terraflow.prepare.readers.abstract import AbstractBackgroundReader
    from terraflow.prepare.readers.grist import GRISTBackgroundReader

    BKGREADER = TypeVar("BKGREADER", AbstractBackgroundReader, GRISTBackgroundReader)


time0                         : float = 0.0  # program start time in seconds
time1                         : float = 0.0  # integration start time in seconds
time_last                     : float = 0.0  # last log time in seconds
time_curr                     : float = 0.0  # current log time in seconds
time_write_last               : float = 0.0  # last write history time in seconds
time_write_curr               : float = 0.0  # current write history time in seconds
elapsed_secondes_in_real_world: float = 0.0  # elapsed secondes in real world
elapsed_seconds_in_integration: float = 0.0  # elapsed seconds in integration


def perf_start(name: str) -> None:
    global time0, time1, time_last, time_curr, time_write_last, time_write_curr
    global elapsed_secondes_in_real_world, elapsed_seconds_in_integration
    if name == "program":
        time0 = time.perf_counter()
        time_last = time0
        time_curr = time0
        elapsed_secondes_in_real_world = 0.0
    elif name == "integration":
        time1 = time.perf_counter()
        elapsed_secondes_in_real_world += time1 - time_last
        time_last = time1
        time_curr = time1
        time_write_last = time1
        time_write_curr = time1
        elapsed_seconds_in_integration = 0.0
    elif name != "":
        time_curr = time.perf_counter()
        elapsed_secondes_in_real_world += time_curr - time_last
        time_last = time_curr
    else:
        raise ValueError(f"Invalid name: {name}")


def perf_end(name: str) -> None:
    global time0, time_last, time_curr
    global elapsed_secondes_in_real_world, elapsed_seconds_in_integration
    if name == "program":
        time_curr = time.perf_counter()
        elapsed_secondes_in_real_world += time_curr - time_last
        time_last = time_curr
    elif name == "integration":
        time_curr = time.perf_counter()
        elapsed_seconds_in_integration += time_curr - time_last
        time_last = time_curr
    elif name != "":
        time_curr = time.perf_counter()
        elapsed_secondes_in_real_world += time_curr - time_last
        tmp = time_last
        time_last = time_curr
    else:
        raise ValueError(f"Invalid name: {name}")


def perf_runtime() -> float:
    global time_last, time_curr
    global elapsed_secondes_in_real_world, elapsed_seconds_in_integration
    time_curr = time.perf_counter()
    elapsed_secondes_in_real_world += time_curr - time_last
    elapsed_seconds_in_integration += time_curr - time_last
    tmp = time_last
    time_last = time_curr
    return time_curr - tmp


def perf_write_history() -> float:
    global time_last, time_curr
    global time_write_last, time_write_curr
    global elapsed_secondes_in_real_world, elapsed_seconds_in_integration
    time_write_curr = time.perf_counter()
    elapsed_secondes_in_real_world += time_write_curr - time_last
    elapsed_seconds_in_integration += time_write_curr - time_last
    tmp = time_write_last
    time_last = time_write_curr
    time_write_last = time_write_curr
    return time_write_curr - tmp


class DateTimeManager:
    start_time_str     : str
    end_time_str       : str
    dt_dyn_value       : float
    dt_dyn_unit        : str
    dt_adv_value       : float 
    dt_adv_unit        : str
    save_interval_value: float
    save_interval_unit : str
    log_every_value    : float
    log_every_unit     : str

    def __init__(
        self,
        start_time: str,
        run_time: str,
        end_time:str,
        dt_dyn: str,
        dt_adv: str,
        save_interval: str,
        log_every: str, 
        dev: AbstractDevice,
        bkg_reader: Optional[BKGREADER] = None
    ):
        # ================================================
        # Preprocess
        if start_time == "":
            start_time = "2000-01-01T00:00:00"
        if bkg_reader is not None:
            start_time = bkg_reader.time_str
        # ================================================
        # Initialize time manager
        self.dt_dyn_value = float(dt_dyn.split()[0])
        self.dt_dyn_unit  = dt_dyn.split()[1]
        
        self.dt_adv_value = float(dt_adv.split()[0])
        self.dt_adv_unit  = dt_adv.split()[1]
        
        self.save_interval_value = float(save_interval.split()[0])
        self.save_interval_unit  = save_interval.split()[1]
        
        self.log_every_value = float(log_every.split()[0])
        self.log_every_unit  = log_every.split()[1]
        
        self.start_time_str = start_time
        if end_time == "":
            if run_time == "":
                raise ValueError(
                    "Either end_time_str or run_time must be provided."
                )
            start_time_ = datetime.datetime.fromisoformat(start_time)
            run_time_value = float(run_time.split()[0])
            run_time_unit = run_time.split()[1]
            run_time_ = datetime.timedelta(**{run_time_unit: run_time_value})
            end_time_ = start_time_ + run_time_
            self.end_time_str = end_time_.isoformat()
        else:
            if run_time == "":
                if dev.is_root():
                    log.logger.warning(
                        "Both end_time_str and run_time are provided. "
                        "end_time_str will be used."
                    )
            self.end_time_str = end_time
        # =================================================
        # Diagnose time variables
        if self.end_time <= self.start_time:
            raise ValueError("End time must be larger than start time.")
        # =================================================
        # Log
        run_time_date = self.end_time - self.start_time
        if run_time_date.days >= 1:
            run_time_unit = "days"
        else:
            run_time_unit = "hours"
        if dev.is_root():
            log.logger.info(
                "Start time: {}, End time: {}, Run time: {} {}.".format(
                    self.start_time_str,
                    self.end_time_str,
                    self.run_time(unit=run_time_unit),
                    run_time_unit,
                )
            )

    @property
    def start_time(self) -> datetime.datetime:
        return datetime.datetime.fromisoformat(self.start_time_str)

    @property
    def end_time(self) -> datetime.datetime:
        return datetime.datetime.fromisoformat(self.end_time_str)
    
    @property
    def dt_dyn(self) -> datetime.timedelta:
        return datetime.timedelta(**{self.dt_dyn_unit: self.dt_dyn_value})
    
    @property
    def dt_adv(self) -> datetime.timedelta:
        return datetime.timedelta(**{self.dt_adv_unit: self.dt_adv_value})
    
    @property
    def save_interval(self) -> datetime.timedelta:
        return datetime.timedelta(**{self.save_interval_unit: self.save_interval_value})
    
    @property
    def log_every(self) -> datetime.timedelta:
        return datetime.timedelta(**{self.log_every_unit: self.log_every_value})

    def curr_time4save(self, save_interval_idx: int) -> str:
        curr_time = (
            self.start_time + self.save_interval * float(save_interval_idx)
        )
        return curr_time.isoformat()

    def curr_time4itimestep(self, itimestep: int) -> str:
        curr_time = self.start_time + self.dt_dyn * float(itimestep)
        return curr_time.isoformat()

    def run_time(self, unit: str) -> float:
        if unit not in [
            "seconds",
            "minutes",
            "hours",
            "days",
            "weeks",
        ]:
            raise ValueError(f"Invalid run_time unit: {unit}")
        if unit == "weeks":
            total_days = (
                self.end_time - self.start_time
            ).total_seconds() / 86400.0
            return float(total_days) / 7.0
        elif unit == "days":
            total_seconds = (self.end_time - self.start_time).total_seconds()
            return float(total_seconds) / 86400.0
        elif unit == "hours":
            total_seconds = (self.end_time - self.start_time).total_seconds()
            return float(total_seconds) / 3600.0
        elif unit == "minutes":
            total_seconds = (self.end_time - self.start_time).total_seconds()
            return float(total_seconds) / 60.0
        elif unit == "seconds":
            total_seconds = (self.end_time - self.start_time).total_seconds()
            return float(total_seconds)
        else:
            raise ValueError(f"Invalid run_time unit: {unit}")


def format_seconds(seconds: float, precision: int = 8) -> str:
    total_seconds = datetime.timedelta(seconds=seconds).total_seconds()
    whole = int(total_seconds)
    frac = total_seconds - whole
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    frac_str = f"{frac:.{precision}f}".split(".")[1] if precision > 0 else ""
    return f"{h}:{m:02d}:{s:02d}" + (f".{frac_str}" if precision > 0 else "")
