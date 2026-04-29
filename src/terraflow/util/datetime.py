from typing import Optional
import time
import datetime

from terraflow.util import log

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


class DateTimeManager:
    start_time_str: str
    end_time_str: str
    save_interval_value: float
    save_interval_unit: str
    dt_dyn: datetime.timedelta
    dt_adv: datetime.timedelta

    def __init__(
        self,
        dt_dyn: str,
        dt_adv: Optional[str] = None,
        start_time: Optional[str] = None,
        save_interval: Optional[str] = None,
        run_time: Optional[str] = None,
        end_time: Optional[str] = None,
        log_every: str = "1 hours",
    ):
        # ================================================
        # Preprocess time variables

        # Preprocess dt_dyn
        dt_dyn_value = float(dt_dyn.split()[0])
        dt_dyn_unit = dt_dyn.split()[1]

        if dt_adv is None:
            dt_adv = "0 seconds"

        # Preprocess dt_adv
        dt_adv_value = float(dt_adv.split()[0])
        dt_adv_unit = dt_adv.split()[1]

        # Preprocess start_time
        if start_time is None:
            start_time = "2000-01-01T00:00:00"

        # Preprocess save_interval
        if save_interval is None:
            save_interval_value = 0
            save_interval_unit = "seconds"
        else:
            save_interval_value = int(save_interval.split()[0])
            save_interval_unit = save_interval.split()[1]
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
        tmp = datetime.timedelta(**{save_interval_unit: save_interval_value})
        if tmp.total_seconds() == 0:
            if save_interval is not None:
                log.logger.warning(
                    "Save interval input is equal to 0. Will not save any history during integration."
                )
            save_interval_value = self.run_time(unit=save_interval_unit)
        self.save_interval_value = save_interval_value
        self.save_interval_unit = save_interval_unit
        self.dt_dyn = datetime.timedelta(**{dt_dyn_unit: dt_dyn_value})
        self.dt_adv = datetime.timedelta(**{dt_adv_unit: dt_adv_value})
        self.log_every_value = log_every.split()[0]
        self.log_every_unit = log_every.split()[1]
        # =================================================
        # Diagnose time variables
        if self.end_time <= self.start_time:
            raise ValueError("End time must be larger than start time.")
        if self.dt_adv.total_seconds() > 0.0 and self.dt_adv < self.dt_dyn:
            log.logger.warning(
                "Advection time step is smaller than dynamic time step. "
                "Will use dynamic time step as advection time step instead."
            )
            self.dt_adv = self.dt_dyn
        if self.dt_adv.total_seconds() % self.dt_dyn.total_seconds() != 0:
            raise ValueError(
                "Advection time step must be a multiple of dynamic time step."
            )
        if self.save_interval.total_seconds() > 0:
            if (self.dt_adv > self.save_interval or self.dt_dyn > self.save_interval):
                raise ValueError(
                    "Save interval must be larger than both dynamic and advection time steps."
                )
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
        if self.save_interval_value > 0:
            log.logger.info(
                "Save interval: {} {}.".format(
                    self.save_interval_value,
                    self.save_interval_unit,
                )
            )
        log.logger.info(
            "Dynamic time step: {} seconds.".format(
                self.dt_dyn.total_seconds(),
            )
        )
        if self.dt_adv.total_seconds():
            log.logger.info(
                "Advection time step: {} seconds.".format(
                    self.dt_adv.total_seconds(),
                )
            )

    @property
    def start_time(self) -> datetime.datetime:
        return datetime.datetime.fromisoformat(self.start_time_str)

    @property
    def end_time(self) -> datetime.datetime:
        return datetime.datetime.fromisoformat(self.end_time_str)

    def curr_time4save(self, save_interval_idx) -> str:
        curr_time = (
            self.start_time + self.save_interval * float(save_interval_idx)
        )
        return curr_time.isoformat()

    def curr_time4itimestep(self, itimestep) -> str:
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

    @property
    def save_interval(self) -> datetime.timedelta:
        if self.save_interval_unit not in [
            "days",
            "seconds",
            "microseconds",
            "milliseconds",
            "minutes",
            "hours",
            "weeks",
        ]:
            raise ValueError(
                f"Invalid save_interval_unit: {self.save_interval_unit}"
            )
        return datetime.timedelta(**{self.save_interval_unit: self.save_interval_value})

    @property
    def log_every(self) -> datetime.timedelta:
        log_every_value = int(self.log_every_value)
        log_every_unit = self.log_every_unit
        if log_every_unit not in [
            "days",
            "seconds",
            "microseconds",
            "milliseconds",
            "minutes",
            "hours",
            "weeks",
        ]:
            raise ValueError(
                f"Invalid log_every_unit: {log_every_unit}"
            )
        return datetime.timedelta(**{log_every_unit: log_every_value})


def format_seconds(seconds: float, precision: int = 8) -> str:
    total_seconds = datetime.timedelta(seconds=seconds).total_seconds()
    whole = int(total_seconds)
    frac = total_seconds - whole
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    frac_str = f"{frac:.{precision}f}".split(".")[1] if precision > 0 else ""
    return f"{h}:{m:02d}:{s:02d}" + (f".{frac_str}" if precision > 0 else "")
