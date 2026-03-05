from loguru import logger
import sys

logger.remove()

logger.add(
    sys.stderr,
    level="INFO",
    filter=lambda r: r["level"].name == "INFO",
    format=("{time:HH:mm:ss} | "
            "<level>{level:<7}</level> | "
            "<blue>{message}</blue>"),
    colorize=True,
    backtrace=False,
    diagnose=False,
)

logger.add(
    sys.stderr,
    level="SUCCESS",
    filter=lambda r: r["level"].name == "SUCCESS",
    format=("<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<7}</level> | "
            "<green>{message}</green>"),
    colorize=True,
    backtrace=False,
    diagnose=False,
)
