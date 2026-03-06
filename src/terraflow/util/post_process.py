"""
This module is used to post-process NetCDF files.
"""
import pathlib
import subprocess
from abc import ABC, abstractmethod
from loguru import logger
from numpy import isin

from terraflow.util import history


class AbstractPostProcessor(ABC):
    """Post-processor for NetCDF files."""

    @classmethod
    def remap_history(
        cls,
        history: history.AbstractHistoryset,
        res: float | int = 1.0,
        remap: bool = True,
        *args,
        **kwargs
    ):
        """Remap NetCDF files in a history set to a specified resolution.

        Args:
            history (history.AbstractHistoryset): History set containing NetCDF files to be remapped.
            res (float | int): Resolution to be interpolated to, e.g., 1 represents for 1d * 1d.
            remap (bool, optional): If True, perform remapping. Defaults to True.
        """
        post_processor = CDOPostProcesser()
        if isinstance(res, float):
            res = int(res) if res.is_integer() else res
        if not remap:
            logger.info(
                "Remapping is skipped. To enable remapping, set remap=True."
            )
            return
        for infile in history.get_all_histories():
            filehead = pathlib.Path(infile).with_suffix("")
            outfile = f"{filehead}.{res}×{res}.nc"
            post_processor.remap_netcdf(infile, outfile, res, *args, **kwargs)

    @staticmethod
    def run(cmd, silent=False):
        stdout = subprocess.DEVNULL if silent else None
        stderr = subprocess.DEVNULL if silent else None
        subprocess.run(cmd, check=True, stdout=stdout, stderr=stderr)

    @classmethod
    @abstractmethod
    def remap_netcdf(
        cls,
        infile: str,
        outfile: str,
        res: float,
        *args,
        **kwargs
    ):
        """Remap a NetCDF file to a specified resolution using CDO.

        Args:
            infile (str): Path to the input NetCDF file.
            outfile (str): Path to the output NetCDF file.
            res (float): Resolution to be interpolated to, e.g., 1 represents for 1d * 1d.
            use_weight (bool, optional): If True, generate weight file before remapping. Defaults to True.
            silent (bool, optional): If True, suppress output from CDO commands. Defaults to False.
        """
        raise NotImplementedError


class CDOPostProcesser(AbstractPostProcessor):
    """Post-processor for NetCDF files using CDO."""

    @classmethod
    def remap_history(
        cls,
        history: history.AbstractHistoryset,
        res: float | int = 1.0,
        remap: bool = True,
        use_weight: bool = True,
        silent: bool = False
    ):
        """Remap NetCDF files in a history set to a specified resolution.

        Args:
            history (history.AbstractHistoryset): History set containing NetCDF files to be remapped.
            res (float | int): Resolution to be interpolated to, e.g., 1 represents for 1d * 1d.
            remap (bool, optional): If True, perform remapping. Defaults to True.
            use_weight (bool, optional): If True, generate weight file before remapping. Defaults to True.
            silent (bool, optional): If True, suppress output from CDO commands. Defaults to False.
        """
        super().remap_history(
            history=history,
            res=res,
            remap=remap,
            use_weight=use_weight,
            silent=silent
        )

    @classmethod
    def remap_netcdf(
        cls,
        infile: str,
        outfile: str,
        res: float,
        use_weight: bool = True,
        silent=False
    ):
        logger.info(
            f"Remapping {infile} to {outfile} with resolution {res}×{res}."
        )
        if use_weight:
            logger.info(
                "Remapping will be done using generated weight."
            )

        logger.info("Starting remapping...")
        if use_weight:
            cls.run(
                [
                    "cdo",
                    f"gendis,global_{res}",
                    infile,
                    "weight.nc"
                ],
                silent=silent
            )
            cls.run(
                [
                    "cdo",
                    f"remap,global_{res},weight.nc",
                    infile,
                    outfile
                ],
                silent=silent
            )
            pathlib.Path("weight.nc").unlink(missing_ok=True)
            pathlib.Path("tmp.nc").unlink(missing_ok=True)
        else:
            cls.run(
                [
                    "cdo",
                    f"remapdis,global_{res}",
                    infile,
                    outfile
                ],
                silent=silent
            )
        logger.info("Remapping completed successfully.")
