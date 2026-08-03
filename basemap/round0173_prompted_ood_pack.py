"""Frozen contract for prompted U12 OOD probe staging and disjointness."""
from __future__ import annotations

from basemap.round0169_prompted_diverse import Round0169Error


ROUND_ID = "0173"
CAPABILITY = "jina-prompted-u12-ood-probe-pack-v1"
CANARY_SCHEMA = "round0173-prompt-model-canary-v1"
LANGUAGE_PROBE_SCHEMA = "round0173-prompted-language-probe-v1"
OOD_AUDIT_SCHEMA = "round0173-prompted-ood-training-disjoint-v1"


class Round0173Error(Round0169Error):
    """The registered prompted OOD probe-pack contract changed."""
