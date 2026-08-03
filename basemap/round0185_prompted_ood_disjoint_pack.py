"""Frozen contract for filtering the five R0173 U12 training overlaps."""
from __future__ import annotations


ROUND_ID = "0185"
CAPABILITY = "jina-prompted-u12-ood-probe-pack-disjoint-v2"
PACK_SCHEMA = "round0185-prompted-u12-ood-disjoint-pack-v1"
SOURCE_AUDIT_SCHEMA = "round0173-prompted-ood-training-disjoint-v1"
LANGUAGE_PROBE_SCHEMA = "round0173-prompted-language-probe-v1"
TRAINING_ROWS = 12_474_331
SOURCE_PROBE_ROWS = 1_000_000
RETAINED_PROBE_ROWS = 999_995

# These are observations accepted in Review 0173 and therefore legitimate
# inputs to this follow-up treatment.  R0185 removes exactly these families,
# then independently rescans the complete retained pack against all training
# rows; this list is not treated as proof that there are no further overlaps.
EXPECTED_REMOVALS = (
    ("arb_Arab", "corpus", 875_069, 4_949_122),
    ("arb_Arab", "corpus", 1_505_153, 4_611_329),
    ("cmn_Hani", "corpus", 849_744, 5_708_181),
    ("cmn_Hani", "corpus", 856_357, 5_653_373),
    ("tha_Thai", "corpus", 1_788_247, 11_586_019),
)


class Round0185Error(RuntimeError):
    """The registered R0185 filtered-probe contract changed."""


__all__ = [
    "CAPABILITY",
    "EXPECTED_REMOVALS",
    "LANGUAGE_PROBE_SCHEMA",
    "PACK_SCHEMA",
    "RETAINED_PROBE_ROWS",
    "ROUND_ID",
    "Round0185Error",
    "SOURCE_AUDIT_SCHEMA",
    "SOURCE_PROBE_ROWS",
    "TRAINING_ROWS",
]
