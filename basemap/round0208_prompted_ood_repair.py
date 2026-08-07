"""Frozen contract for repairing and sealing the prompted multilingual OOD reserve.

R0173 embedded the exact accepted R0108 selections for 19 in-mix languages plus
held-out Polish under the production ``Document: `` convention and then failed
its own registered acceptance: five probe corpus rows are complete stored
prompted-fp16 copies of R0168 U12 training rows.  Accepted Review 0173 blocked
the v1 capability and required a new round that "must state how it handles the
five known overlaps before observing new outputs, for example by reselection
from a preregistered reserve or by a different explicit exclusion policy".

R0208 is that round, and it takes the explicit-exclusion branch.  It performs no
embedding, so it can never introduce a new prompted row: it recomputes the
disjointness evidence over the immutable R0173 arrays, removes every offending
ordinal, equalizes the retained per-language corpus shape, and seals the
surviving ordinals as pack v2.  The R0173 arrays are never mutated; the pack
binds them plus retained-ordinal index arrays.

Identity discipline follows R0178: leakage is judged by more than one identity,
and disagreement between identities is reported rather than hidden.  Three
identities run here:

1. complete stored prompted-fp16 row bytes against the R0168 U12 matrix
   (the R0173 identity, which found the five overlaps);
2. complete stored prompted-fp16 row bytes within the pack itself
   (the within-pack repeat rule R0178 applied to its FineWeb control);
3. exact source-row identity per language dataset, resolving the U12 compact
   rows through the R0132 mapping and the R0087 global-row ranges.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


ROUND_ID = "0208"
CAPABILITY = "jina-prompted-u12-ood-probe-pack-v2"
PACK_SCHEMA = "round0208-prompted-u12-ood-probe-pack-v2"
SOURCE_ROUND_ID = "0173"
SOURCE_PACK_ROOT = "/data/latent-basemap/runs/round-0173/queue/artifacts"
SOURCE_PROBE_SCHEMA = "round0173-prompted-language-probe-v1"
STAGING_SCHEMA = "round0168-prompted-diverse-u12-staging-v1"

DIMENSION = 768
TRAINING_ROWS = 12_474_331
SOURCE_CORPUS_ROWS = 49_500
SOURCE_QUERY_ROWS = 500

#: Every language keeps this many corpus rows after the repair.  The worst
#: language (kor_Hang) loses six rows, so equalizing at 49,494 costs at most
#: six additional clean rows per language and makes per-language FFR read
#: against one identical corpus size.
RETAINED_CORPUS_ROWS = 49_494
#: No query row is an overlap or a within-pack repeat, so the query ID sets are
#: sealed exactly as R0173 emitted them.
RETAINED_QUERY_ROWS = 500

#: The five exact stored-fp16 training-family overlaps, transcribed from the
#: table independently verified in accepted Review 0173.  These are registered
#: before the repair runs; observing a different set is a fail-closed abort.
REGISTERED_TRAINING_OVERLAPS: tuple[tuple[str, str, int, int, int], ...] = (
    ("arb_Arab", "corpus", 1691, 875069, 4949122),
    ("arb_Arab", "corpus", 28454, 1505153, 4611329),
    ("cmn_Hani", "corpus", 580, 849744, 5708181),
    ("cmn_Hani", "corpus", 888, 856357, 5653373),
    ("tha_Thai", "corpus", 40682, 1788247, 11586019),
)
#: R0173's audit reported 16 duplicate probe rows by fingerprint.  Resolved by
#: complete row bytes they are 16 families of exactly two members, every family
#: inside one language's corpus split.
REGISTERED_WITHIN_PACK_FAMILIES = 16
REGISTERED_WITHIN_PACK_DUPLICATE_ROWS = 16
REGISTERED_WITHIN_PACK_MAXIMUM_FAMILY = 2
#: The R0108 selectors draw from source rows above every language's U12
#: training budget, so exact source-row identity is expected to be empty.  A
#: nonzero count would mean index-level leakage that the fp16 identity missed.
REGISTERED_SOURCE_ROW_OVERLAPS = 0

HELD_OUT_LANGUAGE = "pol_Latn"


class Round0208Error(RuntimeError):
    """The registered prompted OOD repair contract or its inputs changed."""


def registered_overlap_keys() -> set[tuple[str, str, int, int, int]]:
    return set(REGISTERED_TRAINING_OVERLAPS)


def repair_plan(
    *,
    language: str,
    split: str,
    excluded_ordinals: Sequence[int],
    source_rows: int,
) -> list[int]:
    """Return the retained ordinals for one language/split.

    The rule is removal-only and order-preserving: drop every excluded ordinal,
    then keep the first ``RETAINED_CORPUS_ROWS`` (corpus) or
    ``RETAINED_QUERY_ROWS`` (queries) survivors in ascending ordinal order.  No
    row is replaced, re-embedded, reordered, or drawn from outside R0173.
    """
    if split == "corpus":
        expected_source = SOURCE_CORPUS_ROWS
        keep = RETAINED_CORPUS_ROWS
    elif split == "queries":
        expected_source = SOURCE_QUERY_ROWS
        keep = RETAINED_QUERY_ROWS
    else:
        raise Round0208Error(f"R0208 {language} split {split!r} is not registered")
    if source_rows != expected_source:
        raise Round0208Error(f"R0208 {language} {split} row count changed")
    excluded = set(int(value) for value in excluded_ordinals)
    if any(value < 0 or value >= expected_source for value in excluded):
        raise Round0208Error(f"R0208 {language} {split} exclusion is out of range")
    survivors = [ordinal for ordinal in range(expected_source) if ordinal not in excluded]
    if len(survivors) < keep:
        raise Round0208Error(
            f"R0208 {language} {split} keeps {len(survivors)} rows, below the "
            f"registered {keep}"
        )
    return survivors[:keep]


def validate_census(census: Mapping[str, Any]) -> None:
    """Fail closed unless the observed input census matches the registration."""
    if int(census.get("training_rows", -1)) != TRAINING_ROWS:
        raise Round0208Error("R0208 U12 training population changed")
    if int(census.get("probe_rows", -1)) != 20 * (
        SOURCE_CORPUS_ROWS + SOURCE_QUERY_ROWS
    ):
        raise Round0208Error("R0208 R0173 probe pack row count changed")
    observed = {
        (
            str(item["language"]),
            str(item["split"]),
            int(item["ordinal"]),
            int(item["source_row"]),
            int(item["training_compact_row"]),
        )
        for item in census.get("exact_training_family_overlaps") or ()
    }
    if observed != registered_overlap_keys():
        raise Round0208Error(
            "R0208 observed a different exact training-family overlap set than "
            "accepted Review 0173 registered"
        )
    if (
        int(census.get("within_pack_exact_families", -1))
        != REGISTERED_WITHIN_PACK_FAMILIES
        or int(census.get("within_pack_duplicate_rows", -1))
        != REGISTERED_WITHIN_PACK_DUPLICATE_ROWS
        or int(census.get("within_pack_maximum_family", -1))
        != REGISTERED_WITHIN_PACK_MAXIMUM_FAMILY
        or int(census.get("within_pack_cross_split_families", -1)) != 0
        or int(census.get("within_pack_cross_language_families", -1)) != 0
    ):
        raise Round0208Error("R0208 within-pack exact-family census changed")
    if (
        int(census.get("source_row_identity_overlaps", -1))
        != REGISTERED_SOURCE_ROW_OVERLAPS
    ):
        raise Round0208Error(
            "R0208 found exact source-row identity leakage that the stored-fp16 "
            "identity did not report"
        )


__all__ = [
    "CAPABILITY",
    "DIMENSION",
    "HELD_OUT_LANGUAGE",
    "PACK_SCHEMA",
    "REGISTERED_SOURCE_ROW_OVERLAPS",
    "REGISTERED_TRAINING_OVERLAPS",
    "REGISTERED_WITHIN_PACK_DUPLICATE_ROWS",
    "REGISTERED_WITHIN_PACK_FAMILIES",
    "REGISTERED_WITHIN_PACK_MAXIMUM_FAMILY",
    "RETAINED_CORPUS_ROWS",
    "RETAINED_QUERY_ROWS",
    "ROUND_ID",
    "Round0208Error",
    "SOURCE_CORPUS_ROWS",
    "SOURCE_PACK_ROOT",
    "SOURCE_PROBE_SCHEMA",
    "SOURCE_QUERY_ROWS",
    "SOURCE_ROUND_ID",
    "STAGING_SCHEMA",
    "TRAINING_ROWS",
    "registered_overlap_keys",
    "repair_plan",
    "validate_census",
]
