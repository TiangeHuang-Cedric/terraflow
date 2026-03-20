"""
This module is used to post-process NetCDF files.
"""
import pathlib
import time
import subprocess
from abc import ABC, abstractmethod
from loguru import logger

from terraflow.util import history


class AbstractPostProcessor(ABC):
    """Post-processor for NetCDF files."""

    @classmethod
    def remap_history(
        cls,
        history: history.abstract.AbstractHistoryset,
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
        t0 = time.perf_counter()
        for infile in history.get_all_histories(with_h2=False):
            filehead = pathlib.Path(infile).with_suffix("")
            outfile = f"{filehead}.{res}×{res}.nc"
            post_processor.remap_netcdf(infile, outfile, res, *args, **kwargs)
        t1 = time.perf_counter()
        logger.success(
            f"CDO post-processing for all history files cost {t1 - t0:.2f} seconds."
        )

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

    _VALID_COMPRESSION_BACKENDS = {"cdo", "nccopy", "nco", "none"}

    @staticmethod
    def _cdo_options(
        compress: bool = True,
        compression_level: int = 4,
        out_format: str = "nc4c",
        precision: str = "F32",
        low_memory: bool = True
    ) -> list[str]:
        """Build common CDO options.

        Args:
            compress (bool, optional): Enable netCDF4 deflate compression. Defaults to True.
            compression_level (int, optional): Deflate level in [1, 9]. Defaults to 4.
            out_format (str, optional): Output format passed to ``-f``. Defaults to ``nc4c``.
            precision (str, optional): Output precision passed to ``-b``. Defaults to ``F32``.
            low_memory (bool, optional): Enable CDO low-memory mode (``-L``). Defaults to True.
        """
        level = max(1, min(9, int(compression_level)))
        options: list[str] = []
        if low_memory:
            options.append("-L")
        if out_format:
            options.extend(["-f", out_format])
        if compress:
            options.extend(["-z", f"zip_{level}"])
        if precision:
            options.extend(["-b", precision])
        return options

    @classmethod
    def _compress_with_nccopy(
        cls,
        infile: str,
        outfile: str,
        compression_level: int = 4,
        chunk_spec: str | None = None,
        silent: bool = False
    ):
        level = max(1, min(9, int(compression_level)))
        cmd = ["nccopy", "-d", str(level), "-s", "-k", "4"]
        if chunk_spec:
            cmd.extend(["-c", chunk_spec])
        cmd.extend([infile, outfile])
        cls.run(cmd, silent=silent)

    @classmethod
    def _compress_with_nco(
        cls,
        infile: str,
        outfile: str,
        compression_level: int = 4,
        ppc: str | None = None,
        silent: bool = False
    ):
        level = max(1, min(9, int(compression_level)))
        cmd = ["ncks", "-O", "-4", "-L", str(level)]
        if ppc:
            cmd.extend(["--ppc", ppc])
        cmd.extend([infile, outfile])
        cls.run(cmd, silent=silent)

    @classmethod
    def remap_history(
        cls,
        history: history.abstract.AbstractHistoryset,
        res: float | int = 1.0,
        remap: bool = True,
        use_weight: bool = True,
        silent: bool = False,
        compress: bool = True,
        compression_level: int = 4,
        out_format: str = "nc4c",
        precision: str = "F32",
        low_memory: bool = True,
        compression_backend: str = "nco",
        nccopy_chunk_spec: str | None = None,
        nco_ppc: str | None = None
    ):
        """Remap NetCDF files in a history set to a specified resolution.

        Args:
            history (history.AbstractHistoryset): History set containing NetCDF files to be remapped.
            res (float | int): Resolution to be interpolated to, e.g., 1 represents for 1d * 1d.
            remap (bool, optional): If True, perform remapping. Defaults to True.
            use_weight (bool, optional): If True, generate weight file before remapping. Defaults to True.
            silent (bool, optional): If True, suppress output from CDO commands. Defaults to False.
            compress (bool, optional): If True, write compressed netCDF4 output. Defaults to True.
            compression_level (int, optional): Compression level in [1, 9]. Defaults to 4.
            out_format (str, optional): CDO output format for ``-f``. Defaults to "nc4c".
            precision (str, optional): CDO output precision for ``-b``. Defaults to "F32".
            low_memory (bool, optional): If True, enable CDO low-memory mode ``-L``. Defaults to True.
            compression_backend (str, optional): One of "cdo", "nccopy", "nco", "none".
                Defaults to "nco".
            nccopy_chunk_spec (str | None, optional): Chunk spec for nccopy ``-c``.
                Used when compression_backend="nccopy".
            nco_ppc (str | None, optional): NCO lossy precision control passed to ``--ppc``.
                Used when compression_backend="nco".
        """
        super().remap_history(
            history=history,
            res=res,
            remap=remap,
            use_weight=use_weight,
            silent=silent,
            compress=compress,
            compression_level=compression_level,
            out_format=out_format,
            precision=precision,
            low_memory=low_memory,
            compression_backend=compression_backend,
            nccopy_chunk_spec=nccopy_chunk_spec,
            nco_ppc=nco_ppc
        )

    @classmethod
    def remap_netcdf(
        cls,
        infile: str,
        outfile: str,
        res: float,
        use_weight: bool = True,
        silent: bool = False,
        compress: bool = True,
        compression_level: int = 4,
        out_format: str = "nc4c",
        precision: str = "F32",
        low_memory: bool = True,
        compression_backend: str = "nco",
        nccopy_chunk_spec: str | None = None,
        nco_ppc: str | None = None
    ):
        backend = compression_backend.lower().strip()
        if backend not in cls._VALID_COMPRESSION_BACKENDS:
            raise ValueError(
                f"Invalid compression_backend={compression_backend!r}, "
                f"choose from {sorted(cls._VALID_COMPRESSION_BACKENDS)}"
            )

        logger.info(
            f"Remapping {infile} to {outfile} with resolution {res}×{res}."
        )
        logger.info(
            "CDO options: compress={}, compression_level={}, out_format={}, precision={}, low_memory={}, compression_backend={}",
            compress,
            compression_level,
            out_format,
            precision,
            low_memory,
            backend
        )
        if use_weight:
            logger.info(
                "Remapping will be done using generated weight."
            )

        t0 = time.perf_counter()

        logger.info("Starting post-remapping...")
        cdo_compress = compress and backend == "cdo"
        needs_post_compress = compress and backend in {"nccopy", "nco"}
        tmp_uncompressed = f"{outfile}.tmp.uncompressed.nc" if needs_post_compress else outfile

        cdo_options = cls._cdo_options(
            compress=cdo_compress,
            compression_level=compression_level,
            out_format=out_format,
            precision=precision,
            low_memory=low_memory
        )
        if use_weight:
            cls.run(
                [
                    "cdo",
                    *cdo_options,
                    f"gendis,global_{res}",
                    infile,
                    "weight.nc"
                ],
                silent=silent
            )
            cls.run(
                [
                    "cdo",
                    *cdo_options,
                    f"remap,global_{res},weight.nc",
                    infile,
                    tmp_uncompressed
                ],
                silent=silent
            )
            pathlib.Path("weight.nc").unlink(missing_ok=True)
            pathlib.Path("tmp.nc").unlink(missing_ok=True)
        else:
            cls.run(
                [
                    "cdo",
                    *cdo_options,
                    f"remapdis,global_{res}",
                    infile,
                    tmp_uncompressed
                ],
                silent=silent
            )

        t1 = time.perf_counter()
        logger.success(f"CDO Post-remapping cost {t1 - t0:.2f} seconds.")

        if needs_post_compress:
            logger.info("Start post-compression with backend: {}...", backend)
            if backend == "nccopy":
                cls._compress_with_nccopy(
                    infile=tmp_uncompressed,
                    outfile=outfile,
                    compression_level=compression_level,
                    chunk_spec=nccopy_chunk_spec,
                    silent=silent
                )
            elif backend == "nco":
                cls._compress_with_nco(
                    infile=tmp_uncompressed,
                    outfile=outfile,
                    compression_level=compression_level,
                    ppc=nco_ppc,
                    silent=silent
                )
            pathlib.Path(tmp_uncompressed).unlink(missing_ok=True)

            t2 = time.perf_counter()

            logger.success(
                f"CDO Post-compression with backend: {backend} cost {t2 - t1:.2f} seconds."
            )

        logger.success(
            f"CDO post-processing for {infile} completed successfully."
        )
