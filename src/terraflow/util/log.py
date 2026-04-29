from loguru import logger
import sys

logger.remove()

logger.level("RUNTIME", no=21, color="<blue>", icon="⏱")
logger.level("STAGE", no=21, color="<magenta>", icon="⏱")

logger.add(
    sys.stderr,
    level="INFO",
    filter=lambda r: r["level"].name == "INFO",
    format=(
        "{time:HH:mm:ss} | "
        "<level>{level:<7}</level> | "
        "<white>{message}</white>"
    ),
    colorize=True,
    backtrace=False,
    diagnose=False,
)

logger.add(
    sys.stderr,
    level="SUCCESS",
    filter=lambda r: r["level"].name == "SUCCESS",
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<7}</level> | "
        "<green>{message}</green>"
    ),
    colorize=True,
    backtrace=False,
    diagnose=False,
)

logger.add(
    sys.stderr,
    level="WARNING",
    filter=lambda r: r["level"].name == "WARNING",
    format=(
        "<yellow>{time:HH:mm:ss}</yellow> | "
        "<level>{level:<7}</level> | "
        "<yellow>{message}</yellow>"
    ),
    colorize=True,
    backtrace=False,
    diagnose=False,
)

logger.add(
    sys.stderr,
    level="ERROR",
    filter=lambda r: r["level"].name == "ERROR",
    format=(
        "<red>{time:HH:mm:ss}</red> | "
        "<level>{level:<7}</level> | "
        "<red>{message}</red>"
    ),
    colorize=True,
    backtrace=False,
    diagnose=False,
)


logger.add(
    sys.stderr,
    level="RUNTIME",
    filter=lambda r: r["level"].name == "RUNTIME",
    format=(
        "<blue>{time:HH:mm:ss}</blue> | "
        "<level>{level:<7}</level> | "
        "<blue>{message}</blue>"
    ),
    colorize=True,
    backtrace=False,
    diagnose=False,
)

logger.add(
    sys.stderr,
    level="STAGE",
    filter=lambda r: r["level"].name == "STAGE",
    format=(
        "<magenta>{time:HH:mm:ss}</magenta> | "
        "<level>{level:<7}</level> | "
        "<magenta>{message}</magenta> "
        "(<magenta>{elapsed}</magenta>)"
    ),
    colorize=True,
    backtrace=False,
    diagnose=False,
)
