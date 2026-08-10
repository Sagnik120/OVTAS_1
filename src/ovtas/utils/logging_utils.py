"""Consistent logging setup shared by all scripts.

A single ``get_logger`` avoids every script re-implementing its own
``logging.basicConfig`` (and accidentally producing duplicate handlers
when scripts import each other).
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "ovtas", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger, configuring the root handler once."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(level)
        _CONFIGURED = True
    return logging.getLogger(name)
