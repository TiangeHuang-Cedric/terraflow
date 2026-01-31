from typing import Optional
from loguru import logger
import datetime


class DateTimeManager:
    start_time_str: str
    end_time_str: str
    elapsed_secondes: float
    save_interval_value: int
    save_interval_unit: str
    dt_dyn: datetime.timedelta

    def __init__(
        self,
        start_time: str,
        save_interval: str,
        dt_dyn: datetime.timedelta,
        run_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ):
        save_interval_value = int(save_interval.split()[0])
        save_interval_unit = save_interval.split()[1]
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
                logger.warning(
                    "Both end_time_str and run_time are provided. "
                    "end_time_str will be used."
                )
            self.end_time_str = end_time
        self.elapsed_secondes = 0.0
        self.save_interval_value = save_interval_value
        self.save_interval_unit = save_interval_unit
        self.dt_dyn = dt_dyn
        logger.success("Initialized date time manager. ")
        run_time_date = self.end_time - self.start_time
        if run_time_date.days >= 1:
            run_time_unit = "days"
        else:
            run_time_unit = "hours"
        logger.info(
            "Start time: {}, End time: {}, Run time: {} {}.".format(
                self.start_time_str,
                self.end_time_str,
                self.run_time(unit=run_time_unit),
                run_time_unit,
            )
        )
        logger.info(
            "Save interval: {} {}, Dynamic time step: {} seconds.".format(
                self.save_interval_value,
                self.save_interval_unit,
                self.dt_dyn.total_seconds(),
            )
        )

    @property
    def start_time(self) -> datetime.datetime:
        return datetime.datetime.fromisoformat(self.start_time_str)

    @property
    def end_time(self) -> datetime.datetime:
        return datetime.datetime.fromisoformat(self.end_time_str)

    def curr_time(self, save_interval_idx) -> str:
        curr_time = (self.start_time +
                     self.save_interval * float(save_interval_idx))
        return curr_time.isoformat()

    def run_time(self, unit: str) -> float:
        if unit not in [
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
        elif unit == "hours":
            total_seconds = (self.end_time - self.start_time).total_seconds()
            return float(total_seconds) / 3600.0
        elif unit == "minutes":
            total_seconds = (self.end_time - self.start_time).total_seconds()
            return float(total_seconds) / 60.0
        elif unit == "days":
            total_seconds = (self.end_time - self.start_time).total_seconds()
            return float(total_seconds) / 86400.0
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
