"""Console + file logging with a consistent format.

Every entry point calls :func:`setup_logging` once.  Module-level loggers are
obtained with :func:`get_logger`, which works even if setup was never called
(useful inside tests).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Sentinel so repeated setup_logging calls from different scripts are cheap.
_CONFIGURED = False


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    quiet_libraries: bool = True,
) -> logging.Logger:
    """Configure the root logger.

    Parameters
    ----------
    level
        Root log level, e.g. ``"DEBUG"`` or :data:`logging.INFO`.
    log_file
        If given, a UTF-8 file handler is attached.  Parent directories are
        created automatically.  ``encoding="utf-8"`` matters here: the corpus
        contains Devanagari and Windows/legacy locales would otherwise raise.
    quiet_libraries
        Drop the noisy third-party loggers (transformers, urllib3, matplotlib)
        to WARNING so our own output stays readable.
    """
    global _CONFIGURED

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Reset handlers so notebooks/re-runs do not duplicate every line.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if quiet_libraries:
        for noisy in (
            "transformers",
            "urllib3",
            "matplotlib",
            "filelock",
            "sentence_transformers",
            "huggingface_hub",
            "PIL",
            "numexpr",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring a sane default on first use."""
    if not _CONFIGURED:
        # Only add a stream handler; do not force a level so tests can capture.
        if not logging.getLogger().handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
            logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
    return logging.getLogger(name)
