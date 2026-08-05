"""Authority-corrected R0199 wrapper for the frozen R0196 F1 science."""
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


ROUND_ID = "0199"


class Round0199Error(RuntimeError):
    """The authority-corrected F1 execution is invalid."""


def diagnose_execution(value: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the unchanged R0196 scientific decision contract."""
    try:
        return _diagnose_execution(value)
    except Round0196Error as error:
        raise Round0199Error(str(error).replace("R0196", "R0199")) from error


__all__ = [
    "CAPABILITY",
    "INFERENCE_CHUNK_ROWS",
    "NEGATIVE_CAPABILITY",
    "PATCH_CAPABILITY",
    "RELOAD_TOLERANCE",
    "ROUND_ID",
    "Round0199Error",
    "SOURCE_DIMENSION",
    "diagnose_execution",
    "fixed_chunks",
]
