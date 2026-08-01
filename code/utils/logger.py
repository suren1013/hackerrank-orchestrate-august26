"""Logging setup for the project.

Provides a single `get_logger(name)` entry point that returns a configured
logger with a consistent format. Log level is read from config (which in turn
reads the LOG_LEVEL environment variable).
"""

from __future__ import annotations

import logging
import sys

from config import LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL

# Cache loggers so we don't re-add handlers on repeated calls.
_configured: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)

    if name in _configured:
        return logger

    logger.setLevel(LOG_LEVEL)

    # Only attach a handler if the logger has none (avoids duplicate output).
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        )
        logger.addHandler(handler)

    # Prevent propagation to the root logger to avoid double logging.
    logger.propagate = False

    _configured.add(name)
    return logger