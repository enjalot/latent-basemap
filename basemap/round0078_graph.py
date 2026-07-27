"""Balanced-120M GPU-native graph contract."""
from __future__ import annotations

from .round0077_quality import Round0077Error


ROUND_ID = "0078"
GRAPH_RECEIPT_SCHEMA = "round0078-balanced-120m-gpu-graph-receipt-v1"


class Round0078Error(Round0077Error):
    """The balanced-120M graph contract was violated."""
