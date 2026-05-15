"""Central logging configuration. Call configure_logging() once at startup."""

import logging
from logging.handlers import RotatingFileHandler

from .config import DATA_DIR

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Wire up console + rotating-file logging. Idempotent."""
    global _configured
    if _configured:
        return
    _configured = True

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    try:
        file_handler = RotatingFileHandler(
            DATA_DIR / "cameralm.log",
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:
        # Console-only is acceptable if the log file can't be opened.
        pass

    # Flask's per-request access log is noise next to the app log.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
