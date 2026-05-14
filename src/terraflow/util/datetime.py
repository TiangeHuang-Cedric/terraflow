from __future__ import annotations
from typing import TYPE_CHECKING, Self
from terraflow.util import log
from typing import Optional
import time
import datetime
import equinox as eqx

if TYPE_CHECKING:
    from terraflow.prepare.readers.abstract import AbstractBackgroundReader


time0 = 0.0  # program start time in seconds
time1 = 0.0  # integration start time in seconds
time_last = 0.0  # last log time in seconds
time_curr = 0.0  # current log time in seconds
elapsed_secondes_in_real_world = 0.0  # elapsed secondes in real world
elapsed_seconds_in_integration = 0.0  # elapsed seconds in integration


def perf_start(name: str) -> None:
    global time0, time1, time_last, time_curr
    global elapsed_secondes_in_real_world, elapsed_seconds_in_integration
    if name == "program":
        time0 = time.perf_counter()
        time_last = time0
        time_curr = time0
        elapsed_secondes_in_real_world = 0.0
        banner = [
            "████████╗███████╗██████╗ ██████╗  █████╗ ███████╗██╗      ██████╗ ██╗    ██╗",
            "╚══██╔══╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██║     ██╔═══██╗██║    ██║",
            "   ██║   █████╗  ██████╔╝██████╔╝███████║█████╗  ██║     ██║   ██║██║ █╗ ██║",
            "   ██║   ██╔══╝  ██╔══██╗██╔══██╗██╔══██║██╔══╝  ██║     ██║   ██║██║███╗██║",
            "   ██║   ███████╗██║  ██║██║  ██║██║  ██║██║     ███████╗╚██████╔╝╚███╔███╔╝",
            "   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝ ",
        ]
        log.logger.log("STAGE", "TERRAFLOW PROGRAM START...")
        print("")
        for line in banner:
            print(line)
        print("")
    elif name == "integration":
        time1 = time.perf_counter()
        elapsed_secondes_in_real_world += time1 - time_last
        time_last = time1
        time_curr = time1
        elapsed_seconds_in_integration = 0.0
        log.logger.log("STAGE", "INTEGRATION STAGE START... ")
    elif name != "":
        time_curr = time.perf_counter()
        elapsed_secondes_in_real_world += time_curr - time_last
        time_last = time_curr
        log.logger.log("STAGE", f"{name.upper()} STAGE START... ")
    else:
        raise ValueError(f"Invalid name: {name}")


def perf_end(name: str) -> None:
    global time0, time_last, time_curr
    global elapsed_secondes_in_real_world, elapsed_seconds_in_integration
    if name == "program":
        time_curr = time.perf_counter()
        elapsed_secondes_in_real_world += time_curr - time_last
        time_last = time_curr
        log.logger.log(
            "STAGE",
            f"Total time cost {time_curr - time0} seconds."
        )
    elif name == "integration":
        time_curr = time.perf_counter()
        elapsed_seconds_in_integration += time_curr - time_last
        time_last = time_curr
        log.logger.log(
            "STAGE",
            f"Integration time cost {time_curr - time1} seconds."
        )
    elif name != "":
        time_curr = time.perf_counter()
        elapsed_secondes_in_real_world += time_curr - time_last
        tmp = time_last
        time_last = time_curr
        log.logger.log(
            "STAGE",
            f"{name.capitalize()} time cost {time_curr - tmp} seconds."
        )
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


class DateTimeManager(eqx.Module):
    start_time_str: str = eqx.field(static=True)
    end_time_str: str = eqx.field(static=True)

    def __init__(
        self,
        start_time: Optional[str] = None,
        run_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ):
        # ================================================
        # Preprocess start_time
        if start_time is None:
            start_time = "2000-01-01T00:00:00"
        # ================================================
        # Initialize time manager
        self.start_time_str = start_time
        if end_time is None:
            if run_time is None:
                raise ValueError(
                    "Either end_time_str or run_time must be provided."
                )
            start_time_ = datetime.datetime.fromisoformat(start_time)
            run_time_value = int(run_time.split()[0])
            run_time_unit = run_time.split()[1]
            run_time_ = datetime.timedelta(**{run_time_unit: run_time_value})
            end_time_ = start_time_ + run_time_
            self.end_time_str = end_time_.isoformat()
        else:
            if run_time is not None:
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
        log.logger.success("Initialized date time manager. ")
        run_time_date = self.end_time - self.start_time
        if run_time_date.days >= 1:
            run_time_unit = "days"
        else:
            run_time_unit = "hours"
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

    def curr_time4save(self, save_interval: datetime.timedelta, save_interval_idx: int) -> str:
        curr_time = (
            self.start_time + save_interval * float(save_interval_idx)
        )
        return curr_time.isoformat()

    def curr_time4itimestep(self, dt_dyn: datetime.timedelta, itimestep: int) -> str:
        curr_time = self.start_time + dt_dyn * float(itimestep)
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

    @classmethod
    def from_reader(
        cls,
        reader: AbstractBackgroundReader,
        end_time: Optional[str] = None,
        run_time: Optional[str] = None,
    ) -> Self:
        start_time = reader.time_str
        return cls(
            start_time=start_time,
            end_time=end_time,
            run_time=run_time,
        )


def format_seconds(seconds: float, precision: int = 8) -> str:
    total_seconds = datetime.timedelta(seconds=seconds).total_seconds()
    whole = int(total_seconds)
    frac = total_seconds - whole
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    frac_str = f"{frac:.{precision}f}".split(".")[1] if precision > 0 else ""
    return f"{h}:{m:02d}:{s:02d}" + (f".{frac_str}" if precision > 0 else "")
