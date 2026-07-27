"""Balanced-120M GPU-native graph contract."""
from __future__ import annotations

from .round0081_quality import Round0081Error


ROUND_ID = "0078"
GRAPH_RECEIPT_SCHEMA = "round0078-balanced-120m-gpu-graph-receipt-v1"


class Round0078Error(Round0081Error):
    """The balanced-120M graph contract was violated."""
