"""Balanced-90M GPU-native graph contract."""
from __future__ import annotations

from .round0072_quality import Round0072Error


ROUND_ID = "0073"
GRAPH_RECEIPT_SCHEMA = "round0073-balanced-90m-gpu-graph-receipt-v1"


class Round0073Error(Round0072Error):
    """The balanced-90M graph contract was violated."""
