"""Minimal logging setup for Kestrel runs."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for a Kestrel run."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (call :func:`setup_logging` first)."""
    return logging.getLogger(name)
