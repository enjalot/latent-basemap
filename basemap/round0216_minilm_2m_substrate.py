"""Frozen contract for the MiniLM 2M mixed substrate and its exact k15 graph.

Phase 1 of the MiniLM-100M v2 plan. Two things happen here and they are
deliberately in one round, because the graph is only meaningful for the exact
population the substrate froze:

1. Assemble a 2,000,000-row mixed universe at the owner-confirmed 100M
   composition shares, with full row provenance.
2. Build a **fresh exact** k15 fuzzy graph over it.

Everything about this round is a reaction to R0215, which measured the v1 150M
map and found (a) its PQ-derived graph recovers under half of each row's true
neighbours *everywhere*, and (b) its clumps are rows left with **zero** valid
edges after canonicalization, not duplicate mass. So:

* the graph is **exact**, never quantized, and its recall against brute-force
  truth is asserted rather than assumed;
* a **degree-zero tripwire** aborts the round if any retained row ends with no
  edges — the failure mode that produced v1's clumps, which no prior round
  checked for.

The corpus loading contract is R0025's, re-confirmed 2026-08-08: the three base
corpora are headerless raw little-endian float32 despite a `.npy` suffix, the
new code corpus is a real `.npy`, only complete 384-float rows count, and the
one 802-byte trailing fragment (fineweb shard 37) is ignored while its full
file hash is still bound.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ROUND_ID = "0216"
CAPABILITY = "minilm-mixed-2m-substrate-and-exact-k15-graph-v1"
SUBSTRATE_SCHEMA = "round0216-minilm-mixed-2m-substrate-v1"
GRAPH_SCHEMA = "round0216-minilm-mixed-2m-exact-k15-graph-v1"

ROWS = 2_000_000
DIMENSION = 384
GRAPH_K = 15
SELECTION_SEED = 216

#: Owner-confirmed 2026-08-08. Shares are exact row counts, not ratios, so the
#: 100M ladder can reuse the identical law by scaling this table.
COMPOSITION: tuple[tuple[str, int], ...] = (
    ("fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2", 800_000),
    ("RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2", 500_000),
    ("pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2", 500_000),
    ("starcoderdata-code-chunked-120-all-MiniLM-L6-v2", 200_000),
)
TARGET_SHARES = {name: n / ROWS for name, n in COMPOSITION}

#: R0025's loading contract.
RAW_FORMAT = "raw-headerless-float32-little-endian-with-.npy-suffix"
ROW_POLICY = "complete-384-float32-rows-only"
TRAILING_FRAGMENT_POLICY = (
    "ignore-incomplete-non-row-trailing-bytes-but-bind-full-source-file-sha256"
)
#: The single known ragged shard, catalogued by R0025.
KNOWN_TRAILING_FRAGMENTS = {
    "fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train/"
    "data-00037-of-00099.npy": {"bytes": 802, "full_rows": 912_356},
}

#: Shards excluded from every selection pool, with the measurement that
#: justifies each. R0025 treated shard 37 as merely *ragged* — 802 trailing
#: bytes over whole rows — but it is **damaged**: 24.89% of its rows are exactly
#: zero, against 0.00% in both immediate neighbours, and the damage grows toward
#: the tail (11.81% over the first 70%, 55.42% over the last 30%). That is a
#: partially-failed write, and its ~227,000 zero rows account for roughly 97% of
#: the 235,469 zero rows R0033 catalogued across the whole 150M population.
#: The whole shard is dropped rather than filtered: when a quarter of a shard
#: failed to write, its surviving non-zero rows are not trustworthy either, and
#: 912,356 rows out of a 93.9M corpus costs nothing.
EXCLUDED_SHARDS = {
    "fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train/"
    "data-00037-of-00099.npy": {
        "reason": "partially-failed write: 24.89% exactly-zero rows vs 0.00% in "
                  "shards 36 and 38; zero fraction rises to 55.42% in the tail",
        "measured_utc": "2026-08-08",
        "zero_rate": 0.2489,
        "rows_dropped": 912_356,
    },
}

#: Zero rows exist outside shard 37 too, at trace levels (17 in a 500k RedPajama
#: sample, 6 in a 500k pile sample). R0033 established the policy — exclude
#: them — and R0215 showed why it matters: a degenerate row cannot acquire
#: meaningful edges, and edgeless rows are what produced the v1 clumps.
ZERO_ROW_POLICY = "exclude exactly-zero and nonfinite rows from the selection pool"

#: Exact-graph qualification. These are floors on a *brute-force* comparison,
#: so unlike R0171's ANN qualification a miss here means the builder is wrong,
#: not that a parameter needs tuning.
RECALL_PROBE_ROWS = 4_096
RECALL_PROBE_SEED = 217
MEAN_RECALL_FLOOR = 0.999
P10_RECALL_FLOOR = 0.99
#: The R0215 tripwire. v1 carried 2,779,481 zero-degree rows (~1.85%) and that
#: is what produced its clumps. Zero is the only acceptable number.
MAX_ZERO_DEGREE_ROWS = 0


class Round0216Error(RuntimeError):
    """The registered 2M substrate or exact-graph contract changed."""


def resolve_shard_rows(*, relative_path: str, size_bytes: int) -> int:
    """Complete rows in a shard, honouring the trailing-fragment policy."""
    row_bytes = DIMENSION * 4
    full, remainder = divmod(int(size_bytes), row_bytes)
    if remainder:
        known = KNOWN_TRAILING_FRAGMENTS.get(relative_path)
        if known is None or int(known["bytes"]) != remainder:
            raise Round0216Error(
                f"unregistered trailing fragment in {relative_path}: "
                f"{remainder} bytes over {full} complete rows. R0025 catalogued "
                "exactly one; a new one means the corpus changed under us."
            )
        if int(known["full_rows"]) != full:
            raise Round0216Error(
                f"{relative_path} row count moved: R0025 recorded "
                f"{known['full_rows']}, this file gives {full}"
            )
    return full


def validate_composition(counts: Mapping[str, int]) -> dict[str, Any]:
    """Fail closed unless the assembled universe is exactly the registered mix."""
    total = sum(int(v) for v in counts.values())
    if total != ROWS:
        raise Round0216Error(f"substrate has {total} rows, registered {ROWS}")
    observed: dict[str, Any] = {}
    for name, want in COMPOSITION:
        got = int(counts.get(name, 0))
        if got != want:
            raise Round0216Error(
                f"{name}: assembled {got} rows, registered {want}"
            )
        observed[name] = {
            "rows": got,
            "share": got / ROWS,
            "registered_share": TARGET_SHARES[name],
        }
    return observed


def validate_graph(
    *, degrees: Mapping[str, Any], recall: Mapping[str, float], edges: int
) -> dict[str, Any]:
    """Assert exactness, then assert the R0215 tripwire."""
    mean = float(recall["mean_recall_at_k"])
    p10 = float(recall["p10_recall_at_k"])
    if mean < MEAN_RECALL_FLOOR or p10 < P10_RECALL_FLOOR:
        raise Round0216Error(
            f"exact-graph recall {mean:.6f}/{p10:.6f} is below the "
            f"{MEAN_RECALL_FLOOR}/{P10_RECALL_FLOOR} floors; an 'exact' builder "
            "that misses brute-force truth is broken, not undertuned"
        )
    zero = int(degrees["zero_degree_rows"])
    if zero > MAX_ZERO_DEGREE_ROWS:
        raise Round0216Error(
            f"{zero} rows have zero edges. R0215 showed this is exactly what "
            "produced the v1 150M clumps; the rung does not proceed with "
            "edgeless rows."
        )
    if int(edges) <= 0:
        raise Round0216Error("graph has no directed edges")
    return {
        "mean_recall_at_k": mean,
        "p10_recall_at_k": p10,
        "mean_recall_floor": MEAN_RECALL_FLOOR,
        "p10_recall_floor": P10_RECALL_FLOOR,
        "zero_degree_rows": zero,
        "zero_degree_tripwire": MAX_ZERO_DEGREE_ROWS,
        "directed_edges": int(edges),
        "exactness": "brute-force fp32 cosine top-k over the full substrate",
    }


__all__ = [
    "CAPABILITY",
    "COMPOSITION",
    "DIMENSION",
    "GRAPH_K",
    "GRAPH_SCHEMA",
    "EXCLUDED_SHARDS",
    "KNOWN_TRAILING_FRAGMENTS",
    "MAX_ZERO_DEGREE_ROWS",
    "MEAN_RECALL_FLOOR",
    "P10_RECALL_FLOOR",
    "RAW_FORMAT",
    "RECALL_PROBE_ROWS",
    "RECALL_PROBE_SEED",
    "ROUND_ID",
    "ROWS",
    "ROW_POLICY",
    "ZERO_ROW_POLICY",
    "Round0216Error",
    "SELECTION_SEED",
    "SUBSTRATE_SCHEMA",
    "TARGET_SHARES",
    "TRAILING_FRAGMENT_POLICY",
    "resolve_shard_rows",
    "validate_composition",
    "validate_graph",
]
