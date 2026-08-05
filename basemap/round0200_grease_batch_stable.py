"""Final authority-corrected R0200 wrapper for the frozen Track F1 science."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .round0196_grease_batch_stable import (
    CAPABILITY,
    INFERENCE_CHUNK_ROWS,
    NEGATIVE_CAPABILITY,
    PATCH_CAPABILITY,
    RELOAD_TOLERANCE,
    SOURCE_DIMENSION,
    Round0196Error,
    diagnose_execution as _diagnose_execution,
    fixed_chunks,
)


ROUND_ID = "0200"


class Round0200Error(RuntimeError):
    """The corrected F1 execution is invalid."""


def diagnose_execution(value: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the unchanged R0196 scientific decision contract."""
    try:
        return _diagnose_execution(value)
    except Round0196Error as error:
        raise Round0200Error(str(error).replace("R0196", "R0200")) from error


__all__ = [
    "CAPABILITY",
    "INFERENCE_CHUNK_ROWS",
    "NEGATIVE_CAPABILITY",
    "PATCH_CAPABILITY",
    "RELOAD_TOLERANCE",
    "ROUND_ID",
    "Round0200Error",
    "SOURCE_DIMENSION",
    "diagnose_execution",
    "fixed_chunks",
]
