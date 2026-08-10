"""R0240 constants — build and qualify the 100,000,000-row k15 fuzzy graph.

R0238 assembled, nested, sealed and priced this rung and its build was refused
**a priori** by the wall guard: `~7.04 GPU-h` needed from a clean start against
a `6.0` cap, with the read term (`8,254.17 s`) the part that tipped it. The
owner raised the rung's cap to **9.0 GPU-h** on 2026-08-10 (registered in
`guides/plan-minilm-100m-v2.md`, "Rules earned in Phase 1"), because the
estimate was wrong rather than the measurement inconvenient: at 100M the
substrate cannot stay page-resident, so the 51-pass I/O term measures
`1.61-2.29 h` against a predicted `0.67-0.92 h`.

**This round inherits R0238's substrate, truth probe and reachability ceiling
BY HASH and re-assembles nothing.** A 100M re-assembly costs `153.6 GB` of
writes against ~330 GB free and about `0.41 GPU-h`, and would produce the same
bytes: the five ladder prefix hashes registered below were recomputed from
R0238's own file by an independent reviewer in one pass. Every one of them is
asserted before the first cell runs, and a mismatch raises.

**The science code is R0238's, imported read-only.** `experiments/round0240_
nodes.py` calls `experiments.round0238_nodes.run_ladder` and `run_qualify`
unchanged. That is deliberate: those two functions carry the registered checks
this round must not weaken - the R0215 in-and-out degree-zero tripwire, the
tie-aware/strict recall floors, the `min = 0.0` forensic, the worst-of-five
selection rule, the signal-free cooperative supervision, the read-only-memmap
precondition on every cuVS input, and the per-rung re-derivation with
`imbalance_tolerance_by_c` that review-0238-01/F3 and F4 recorded as still
owed. Re-typing 1,150 reviewed lines to relabel them would risk exactly the
kind of silent transcription defect this program has paid for twice. What this
round adds beside them is a sealed R0240 attestation carrying the headline
numbers and the inheritance proof, and a queue whose budget, deadlines and
carried prediction are this round's own.

**Disclosed consequence:** `build-ladder.json` and `qualified-graph.json`
produced under this round carry `"round_id": "0238"` and R0238's schema names,
because the implementing module stamps its own identity. They are R0240
artifacts by queue, by release commit and by output path; the attestation
beside them says so and binds them by sha256.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from basemap.round0238_rung5 import (
    CANDIDATE,
    DIMENSION,
    GRAPH_DEGREE,
    GRAPH_K,
    GUARD_IMBALANCE_MARGIN,
    INTERMEDIATE_GRAPH_DEGREE,
    MAX_ITERATIONS,
    MAX_ZERO_DEGREE_ROWS,
    NN_DESCENT_SETTING,
    PRIMARY_IMBALANCE_SEED,
    RECALL_MEAN_FLOOR,
    RECALL_P10_FLOOR,
    SPILL,
)


class Round0240Error(RuntimeError):
    """Any fail-closed condition this round registers."""


ROUND_ID = "0240"
ROWS = 100_000_000
PARENT_ROUND_ID = "0238"

#: Owner-raised on 2026-08-10 from 6.0. The rung needs ~7.04 GPU-h clean;
#: R0238's refusal at 6.0 was correct and is why that round closed at 2.20.
GPU_HOURS_CAP = 9.0
GPU_HOURS_CAP_NOTE = (
    "9.0 GPU-h, raised from 6.0 by the owner on 2026-08-10 and registered in "
    "guides/plan-minilm-100m-v2.md. R0238 priced the rung at ~7.04 GPU-h from "
    "a clean start (assemble 1,486.73 + truth 1,508.59 + reachability 345.79 + "
    "grid 780.61 + build 18,237.29 predicted + qualification 3,000 reserve = "
    "25,359.01 s) and refused the build a priori rather than breach 6.0. This "
    "round inherits the first three of those terms as sealed artifacts, so its "
    "own outstanding work is the grid, the build and the qualification: about "
    "22,018 s = 6.12 GPU-h predicted, inside 9.0 with 2.88 h of margin. If the "
    "build would still breach 9.0, the wall guard refuses it a priori again "
    "and this round reports the overage rather than trimming a registered "
    "check."
)

#: The build cell's own COOPERATIVE deadline. Not a signal: on expiry the
#: parent writes the cell's abort flag and waits for the child to unwind its
#: own CUDA context through the normal interpreter path (R0239).
BUILD_TIMEOUT_S = 26_000.0
BUILD_TIMEOUT_NOTE = (
    "26,000 s. Sized so it can only fire when the round is genuinely about to "
    "breach its 9.0 GPU-h cap, never as a routine bound on an extrapolation "
    "this program has already been wrong about by 3.4x: predicted build wall "
    "is 18,237 s and the cap leaves 32,400 - ~890 (grid) - 3,000 (qualify "
    "reserve) = 28,510 s. It is a cooperative flag, not a signalling bound on "
    "subprocess.run; nothing in this round can signal a process holding a CUDA "
    "context."
)

#: Runner soft deadlines are `max(2 x p90_wall_s, 900)`, and since R0240's
#: wiring the node RELAYS that flag to the build child. Both p90s are therefore
#: sized ABOVE the cell's own deadline so the runner never preempts it.
LADDER_P90_WALL_S = 16_000.0
QUALIFY_P90_WALL_S = 4_000.0
P90_WALL_NOTE = (
    "the runner's soft deadline is max(2 x p90_wall_s, 900 s) and, since the "
    "R0240 wiring of review-0239-02/F4, the node relays that cooperative flag "
    "to the child. ladder: 2 x 16,000 = 32,000 s against a node bounded at "
    "~890 + 26,000 = 26,890 s. qualify: 2 x 4,000 = 8,000 s against R0237's "
    "measured 634 s at half this rung. The runner's flag is therefore a "
    "backstop behind the cell's own deadline, never the first thing to fire."
)

#: What the ladder node holds back from the build's wall budget for the
#: qualification node. R0237 measured 634.0 s at 50,000,000 rows.
QUALIFY_RESERVE_S = 3_000.0

# --------------------------------------------------------------------------- #
# what is inherited, and the literals every one of it must reproduce
# --------------------------------------------------------------------------- #
R0238_QUEUE_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts"
)
INHERITED_SUBSTRATE_MANIFEST = (
    f"{R0238_QUEUE_ARTIFACTS}/"
    "minilm-mixed-100000k-nested-substrate-and-reserves-v1/substrate.json"
)
INHERITED_TRUTH_MANIFEST = (
    f"{R0238_QUEUE_ARTIFACTS}/"
    "minilm-mixed-100000k-uniform-probe-k15-truth-v1/probe-k15-truth.json"
)
INHERITED_REACHABILITY_MANIFEST = (
    f"{R0238_QUEUE_ARTIFACTS}/"
    "minilm-mixed-100000k-cluster-spill-c400-reachability-v1/reachability.json"
)

#: Recomputed from R0238's own bytes in one sequential pass by the independent
#: reviewer (review-0238-2026-08-09-01), and each of the first four also
#: reproduces the sealed value of the round that first drew it.
REGISTERED_LADDER_PREFIX_SHA256: dict[int, str] = {
    6_250_000: (
        "5d976ab6d895db45095967afd5ce7dd078a6242bc62edfff941c120fec473e36"
    ),
    12_500_000: (
        "bd004db8511c9e3ea44bbc1471f739cdcc5d78adb35f7cb53c96919638ec7ad5"
    ),
    25_000_000: (
        "466ef03904f86fdd7bb5b3b491a028be7a06dcf1c0856effdf0c7757adc7bc03"
    ),
    50_000_000: (
        "e7ccf848e0f42e7ac4efa84636b243c9fa75d8e8e0644c21973179cdde1002a9"
    ),
    100_000_000: (
        "f3f1b4b75d2612683a275c0b483fac948527d58e9f9836c165572fbe147ab645"
    ),
}
REGISTERED_ORDERED_SUBSTRATE_SHA256 = REGISTERED_LADDER_PREFIX_SHA256[ROWS]
REGISTERED_SUBSTRATE_FILE_SHA256 = (
    "c0b85842236daa7e8a1c6fd8b27032360894b5445389ca1fc5b2b0196124a63d"
)
REGISTERED_SUBSTRATE_FILE_BYTES = 153_600_000_128
REGISTERED_PROVENANCE_FILE_SHA256 = (
    "35bc0c65f365daba4edecde43553f40c5aa9a7f6c13d11854fa68fd240caa87c"
)
REGISTERED_COMPOSITION: tuple[tuple[str, int], ...] = (
    ("fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2", 40_000_000),
    ("RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2", 25_000_000),
    ("pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2", 25_000_000),
    ("starcoderdata-code-chunked-120-all-MiniLM-L6-v2", 10_000_000),
)
REGISTERED_SHARD_COUNTS: dict[str, int] = {
    "fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2": 98,
    "RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2": 150,
    "pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2": 177,
    "starcoderdata-code-chunked-120-all-MiniLM-L6-v2": 21,
}
REGISTERED_RESERVE_ROWS = 200_000
REGISTERED_COVERAGE = 1.0

#: R0238's sealed c = 400 partition ceiling at THIS rung, on THIS probe.
REGISTERED_REACHABILITY_CEILING_C400 = 0.9983062666666668
REGISTERED_REACHABILITY_ZERO_REACHABLE_ROWS = 1

REGISTERED_TRUTH_PROBE_ROWS = 500_000
REGISTERED_TRUTH_PROBE_SEED = 238_000

INHERITANCE_NOTE = (
    "R0238's substrate, uniform truth probe and c = 400 reachability ceiling "
    "are bound at their FULL sha256 signatures and re-earned by nothing. Every "
    "literal above is asserted against the sealed manifest before the first "
    "cell runs and a mismatch raises Round0240Error, so 'inherited' is a "
    "checked claim rather than a path reference. A 100M re-assembly would cost "
    "153.6 GB of writes and ~0.41 GPU-h to reproduce bytes an independent "
    "reviewer already recomputed hash-for-hash in one pass."
)

# --------------------------------------------------------------------------- #
# the adverse carry this round registers, and what it is measured against
# --------------------------------------------------------------------------- #
#: R0238's MEASURED worst-of-five at this rung and this c. It is this round's
#: prediction, not R0237's 50M value, because the doubling has already happened.
CARRIED_IMBALANCE_AT_C400 = 2.8204385
CARRIED_IMBALANCE_SOURCE = (
    "R0238's sealed five-seed grid at N = 100,000,000, c = 400, s = 8 "
    "(imbalance.json a06000bdf0ad878973e775c2158a6c9b8ea5969a75adaa4e24b68c0e"
    "f4851912), worst of five: capability "
    "minilm-mixed-cluster-spill-s8-imbalance-100m-c400-five-seed-v1, released "
    "by review-0238-2026-08-09-01"
)
R0238_MEASURED_IMBALANCE_BY_SEED: dict[int, float] = {
    226: 2.2244495,
    236: 2.8204385,
    1236: 2.2500385,
    2236: 2.696567,
    3236: 2.0973435,
}
#: The two sealed guard numbers behind R0238's realised tolerance. The
#: tolerance itself is COMPUTED from them here rather than transcribed, because
#: review-0238-01/F4 had to derive it by hand when R0238 published neither.
R0238_ADMISSIBLE_MAX_CLUSTER_ROWS = 9948339.67145981
R0238_GUARDED_MAX_CLUSTER_ROWS = 6570967.363268
#: R0237's 50M-derived prediction, kept only so this round can restate the
#: two-step drift 2.456543 -> 2.8204385 -> (measured here) in one place.
R0237_PREDICTION_IMBALANCE_AT_C400 = 2.456543
R0237_PREDICTION_TOLERANCE_AT_C400 = 0.738255

ADVERSE_CARRY_NOTE = (
    "R0238 measured worst-of-five imbalance at c = 400, N = 100,000,000 as "
    "2.8204385 against a prediction of 2.456543 carried from 50M: "
    "measured_exceeds_prediction TRUE, +14.81%, larger than both the +3.15% "
    "(mean) and +10.69% (primary seed) priors that round registered. "
    "review-0238-01/F5 adds two things R0238 did not state: the MEAN moved "
    "+7.70% as well, so the whole distribution shifted right rather than only "
    "the tail; and the seed ranking essentially INVERTED across the doubling "
    "(50M's worst seed 3236 is 100M's best; 100M's worst seed 236 was "
    "second-lowest at 50M), with the relative sample sd up 78%. A seed's "
    "imbalance at one N therefore carries no information about its imbalance "
    "at the next, which is the correct justification for the worst-of-n rule "
    "and the reason this round RE-MEASURES all five seeds on this substrate "
    "rather than reusing R0238's grid. c is re-derived from this round's own "
    "worst-of-five with GUARD_IMBALANCE_MARGIN = 1.164884 applied, and the "
    "round reports the re-derived c whether or not it is 400. The margin is "
    "NOT raised: review-0236-01/F3 found no path from seed spread to OOM, so "
    "the hazard a larger margin creates is a spurious refusal."
)

TOLERANCE_PUBLICATION_NOTE = (
    "tolerance_to_adverse_imbalance = admissible_max_cluster_rows / "
    "guarded_max_cluster_rows - 1, the definition every prior rung uses. "
    "R0238 said its cell was 'comfortably admissible' and never published the "
    "number; review-0238-01/F4 computed it as 51.3984%, down from the 73.8255% "
    "the pricing carried - more than half the tolerance consumed in one "
    "doubling. This round publishes the measured tolerance at its own "
    "re-derived c as a first-class field, from the ladder's own sealed guard "
    "inputs."
)

SELECTION_NOTE = (
    "c = 400 at s = 8 is confirmed by two independent reviews. It holds "
    "73.83% tolerance at 17.396 GiB predicted device, 0.787x of the fitted "
    "law's range, against c = 200's 3.59% at a 1.320x extrapolation - and on "
    "R0238's own measured drift c = 200 would have been INADMISSIBLE at this "
    "rung (2.061112 x 1.1481 = 2.366 gives a guarded max cluster of ~11.02M "
    "against an admissible 9.95M). Nothing above c = 400 has a measured "
    "reachability ceiling anywhere in this program. The candidate set is "
    "registered as (400,) a priori and c is nevertheless RE-DERIVED from this "
    "round's own measurement under the same margin, so the round can report a "
    "disagreement instead of assuming one cannot happen."
)

QUALIFICATION_NOTE = (
    "recall is scored against R0238's sealed builder-independent brute-force "
    "truth on a UNIFORM 500,000-row probe drawn at seed 238000 before the "
    "substrate existed - never a seed union and never hub-biased "
    "(review-0227-01 caught exactly that). The probe size is a registered "
    "budget decision: full exact truth over all 100,000,000 rows costs ~75 "
    "GPU-h. Its standard error on a mean near 0.9957 is ~4.3e-6 against "
    "0.0957 of slack to the 0.90 floor. The round reports strict AND "
    "tie-aware, p10, min, rows carrying loss and both decile series, and the "
    "R0215 degree-zero tripwire over ALL 100,000,000 rows in BOTH in- and "
    "out-degree, which has never been tested at this rung and is the check "
    "that catches the v1 failure mode. If min = 0.0 appears, the sealed "
    "zero-recall forensic adjudicates each row individually - duplicate-family "
    "fp32 artefacts are expected but are verified, not assumed."
)

COST_NOTE = (
    "realised cost is reported as peak device (over baseline), peak host "
    "ANONYMOUS bytes (never RSS - R0233 measured 7.93 GB anon against 41.1 GB "
    "RSS at the 6.25M rung), peak spill scratch, wall, and measured I/O: the "
    "architectural volume beside the child's own read_bytes/write_bytes and "
    "the /data block-device deltas, with mincore substrate residency before "
    "and after. review-0238-01/F2 blocked R0238's I/O restatement because "
    "those instruments lived only in a build path that never ran; here they "
    "run. The comparison is against R0238's measured 1.61-2.29 h for the "
    "51-pass term."
)

SAFETY_NOTE = (
    "every buffer handed to cuVS is a read-only np.memmap, asserted on the "
    "substrate view and on every intermediate the streaming builder assembles, "
    "and sealed as cuvs_inputs_asserted_memmap. No signal is delivered to any "
    "process on any path: the imbalance grid and the build cell are both "
    "supervised by basemap.gpu_child_supervision.run_gpu_child_cooperative "
    "(Popen + communicate() with no signalling bound, plus a cooperative flag "
    "file the child polls), the node relays the runner's ROUNDRUN_ABORT_FLAG "
    "onto the cell's flag, and the runner itself can no longer signal a node "
    "that declares gpu_required (workshop branch runner-signal-safety; "
    "proc.kill() is gone and an AST pin keeps it gone). experiments/check_signal_safety.py "
    "gates every file this round adds as well as basemap/*.py."
)

# --------------------------------------------------------------------------- #
# capability + schema names for what THIS round attests
# --------------------------------------------------------------------------- #
LADDER_CAPABILITY = "minilm-mixed-100000k-cluster-spill-build-ladder-v1"
GRAPH_CAPABILITY = "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1"
IMBALANCE_CAPABILITY = (
    "minilm-mixed-cluster-spill-s8-imbalance-100m-c400-five-seed-v2"
)
ATTEST_CAPABILITY = "minilm-mixed-100000k-rung-build-and-qualification-v1"
LADDER_ATTEST_SCHEMA = "round0240-100000k-build-ladder-attestation-v1"
QUALIFY_ATTEST_SCHEMA = "round0240-100000k-qualification-attestation-v1"
LADDER_ATTEST_FILE = "round0240-ladder-attestation.json"
QUALIFY_ATTEST_FILE = "round0240-qualification-attestation.json"

IMPLEMENTATION_NOTE = (
    "the ladder and qualification nodes are experiments.round0238_nodes."
    "run_ladder and run_qualify, imported read-only and called unchanged. "
    "Their sealed receipts therefore carry round_id 0238 and R0238 schema "
    "names; they are R0240 artifacts by queue, release commit and output path, "
    "and this attestation binds each of them by sha256 and states the "
    "provenance explicitly rather than leaving a reader to discover it."
)


def tolerance_to_adverse_imbalance(
    *, admissible_max_cluster_rows: float, guarded_max_cluster_rows: float
) -> float:
    """The program's tolerance statistic, in the one definition every rung uses."""
    guarded = float(guarded_max_cluster_rows)
    if guarded <= 0.0:
        raise Round0240Error("guarded max cluster rows must be positive")
    return float(admissible_max_cluster_rows) / guarded - 1.0


R0238_REALISED_TOLERANCE_AT_C400 = tolerance_to_adverse_imbalance(
    admissible_max_cluster_rows=R0238_ADMISSIBLE_MAX_CLUSTER_ROWS,
    guarded_max_cluster_rows=R0238_GUARDED_MAX_CLUSTER_ROWS,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Round0240Error(message)


def verify_inherited_substrate(sealed: Mapping[str, Any]) -> dict[str, Any]:
    """Assert every registered literal against R0238's sealed manifest.

    Fail-closed and cheap: it reads a 52 KB JSON, not 153.6 GB. The bytes
    themselves are bound by the queue manifest's full signature on this file
    and by `verify_signature` on the substrate array inside the node.
    """
    _require(
        str(sealed.get("round_id")) == PARENT_ROUND_ID,
        f"inherited substrate is round {sealed.get('round_id')}, not "
        f"{PARENT_ROUND_ID}",
    )
    _require(
        int(sealed.get("rows", -1)) == ROWS,
        f"inherited substrate has {sealed.get('rows')} rows, not {ROWS}",
    )
    _require(
        int(sealed.get("dimension", -1)) == DIMENSION,
        "inherited substrate dimension is not 384",
    )

    nesting = dict(sealed.get("nesting") or {})
    observed_prefixes = {
        int(rows): str(value)
        for rows, value in (
            nesting.get("ladder_prefix_ordered_sha256") or {}
        ).items()
    }
    for rows, digest in sorted(REGISTERED_LADDER_PREFIX_SHA256.items()):
        _require(
            observed_prefixes.get(rows) == digest,
            f"inherited ladder prefix at {rows:,} rows is "
            f"{observed_prefixes.get(rows)!r}, registered {digest!r}. STOP: "
            "the substrate this round binds is not the one R0238 sealed",
        )
    _require(
        str(sealed.get("ordered_substrate_sha256"))
        == REGISTERED_ORDERED_SUBSTRATE_SHA256,
        "inherited ordered_substrate_sha256 does not match the registered "
        "100,000,000-row digest",
    )
    _require(
        bool(nesting.get("byte_identical_prefix")) is True
        and bool(nesting.get("positional_prefix")) is True
        and int(nesting.get("parent_rows_missing_from_child", -1)) == 0,
        "inherited nesting is not a byte-identical positional prefix with zero "
        "parent rows missing",
    )

    substrate_signature = dict(sealed.get("substrate") or {})
    _require(
        str(substrate_signature.get("sha256")) == REGISTERED_SUBSTRATE_FILE_SHA256
        and int(substrate_signature.get("bytes", -1))
        == REGISTERED_SUBSTRATE_FILE_BYTES,
        "inherited substrate.f32.npy signature does not match the registered "
        "sha256/size",
    )
    _require(
        str((sealed.get("provenance") or {}).get("sha256"))
        == REGISTERED_PROVENANCE_FILE_SHA256,
        "inherited provenance.npy sha256 does not match the registered value",
    )

    composition = dict(sealed.get("composition") or {})
    for corpus, rows in REGISTERED_COMPOSITION:
        _require(
            int((composition.get(corpus) or {}).get("rows", -1)) == rows,
            f"inherited composition gives {corpus} "
            f"{(composition.get(corpus) or {}).get('rows')} rows, registered "
            f"{rows}",
        )
    _require(
        sum(int(v.get("rows", 0)) for v in composition.values()) == ROWS,
        "inherited composition does not sum to 100,000,000",
    )

    span = dict((sealed.get("selection") or {}).get("shard_span") or {})
    coverage: dict[str, dict[str, float]] = {}
    for corpus, shards in sorted(REGISTERED_SHARD_COUNTS.items()):
        entry = dict(span.get(corpus) or {})
        union = dict(entry.get("union") or {})
        increment = dict(entry.get("increment") or {})
        _require(
            float(union.get("coverage", -1.0)) == REGISTERED_COVERAGE
            and float(increment.get("coverage", -1.0)) == REGISTERED_COVERAGE,
            f"inherited shard coverage for {corpus} is union "
            f"{union.get('coverage')} / increment {increment.get('coverage')}, "
            f"registered {REGISTERED_COVERAGE} on both",
        )
        _require(
            int(union.get("shards_total", -1)) == shards
            and int(union.get("shards_touched", -1)) == shards,
            f"inherited shard count for {corpus} is "
            f"{union.get('shards_touched')}/{union.get('shards_total')}, "
            f"registered {shards}/{shards}",
        )
        coverage[corpus] = {
            "union": float(union["coverage"]),
            "increment": float(increment["coverage"]),
            "shards": float(shards),
        }

    reserve = dict(sealed.get("reserve") or {})
    disjointness = dict(reserve.get("disjointness") or {})
    _require(
        int(sealed.get("reserve_rows", -1)) == REGISTERED_RESERVE_ROWS,
        f"inherited reserve is {sealed.get('reserve_rows')} rows, registered "
        f"{REGISTERED_RESERVE_ROWS}",
    )
    _require(
        int(disjointness.get("global_intersection_rows", -1)) == 0,
        "inherited reserve is not globally disjoint from the training rows",
    )
    per_corpus = dict(disjointness.get("per_corpus") or {})
    for corpus, _rows in REGISTERED_COMPOSITION:
        _require(
            int((per_corpus.get(corpus) or {}).get("intersection_rows", -1)) == 0,
            f"inherited reserve intersects {corpus}'s training rows",
        )

    return {
        "verified": True,
        "note": INHERITANCE_NOTE,
        "round_id_of_source": PARENT_ROUND_ID,
        "rows": ROWS,
        "ladder_prefix_ordered_sha256": {
            str(rows): digest
            for rows, digest in sorted(REGISTERED_LADDER_PREFIX_SHA256.items())
        },
        "ordered_substrate_sha256": REGISTERED_ORDERED_SUBSTRATE_SHA256,
        "substrate_file_sha256": REGISTERED_SUBSTRATE_FILE_SHA256,
        "provenance_file_sha256": REGISTERED_PROVENANCE_FILE_SHA256,
        "composition": {corpus: rows for corpus, rows in REGISTERED_COMPOSITION},
        "shard_coverage": coverage,
        "reserve_rows": REGISTERED_RESERVE_ROWS,
        "reserve_disjoint_three_ways": True,
    }


def verify_inherited_truth(sealed: Mapping[str, Any]) -> dict[str, Any]:
    """Assert the inherited probe is the registered uniform draw at this rung."""
    _require(
        str(sealed.get("round_id")) == PARENT_ROUND_ID
        and int(sealed.get("rows", -1)) == ROWS,
        "inherited truth probe is not R0238's 100,000,000-row artifact",
    )
    _require(
        int(sealed.get("probe_rows", -1)) == REGISTERED_TRUTH_PROBE_ROWS
        and int(sealed.get("probe_seed", -1)) == REGISTERED_TRUTH_PROBE_SEED,
        "inherited truth probe is not the registered 500,000 rows at seed "
        "238000",
    )
    _require(
        int(sealed.get("k", -1)) == GRAPH_K,
        "inherited truth probe is not k = 15",
    )
    return {
        "verified": True,
        "probe_rows": REGISTERED_TRUTH_PROBE_ROWS,
        "probe_seed": REGISTERED_TRUTH_PROBE_SEED,
        "population": sealed.get("population"),
        "method": sealed.get("method"),
    }


def verify_inherited_reachability(sealed: Mapping[str, Any]) -> dict[str, Any]:
    """Assert the inherited c = 400 partition ceiling is R0238's sealed one."""
    _require(
        str(sealed.get("round_id")) == PARENT_ROUND_ID
        and int(sealed.get("rows", -1)) == ROWS,
        "inherited reachability is not R0238's 100,000,000-row artifact",
    )
    ceilings = {
        int(key): float(value)
        for key, value in (sealed.get("strict_ceiling_by_clusters") or {}).items()
    }
    _require(
        ceilings.get(400) == REGISTERED_REACHABILITY_CEILING_C400,
        f"inherited c = 400 ceiling is {ceilings.get(400)}, registered "
        f"{REGISTERED_REACHABILITY_CEILING_C400}",
    )
    zero_reachable = {
        int(cell["clusters"]): int(cell["rows_with_zero_reachable"])
        for cell in (sealed.get("cells") or [])
    }
    _require(
        zero_reachable.get(400) == REGISTERED_REACHABILITY_ZERO_REACHABLE_ROWS,
        f"inherited c = 400 zero-reachable rows is {zero_reachable.get(400)}, "
        f"registered {REGISTERED_REACHABILITY_ZERO_REACHABLE_ROWS}",
    )
    return {
        "verified": True,
        "strict_ceiling_c400": REGISTERED_REACHABILITY_CEILING_C400,
        "rows_with_zero_reachable_c400": (
            REGISTERED_REACHABILITY_ZERO_REACHABLE_ROWS
        ),
        "probe_rows": sealed.get("probe_rows"),
    }


def ladder_attestation(
    *, ladder: Mapping[str, Any], release_sha: str
) -> dict[str, Any]:
    """This round's own reading of the ladder receipt R0238's code sealed.

    Publishes the two things R0238 owed and did not pay: the measured
    tolerance at the selected `c`, and the carried-prediction comparison
    restated against the value that is now a MEASUREMENT rather than an
    extrapolation.
    """
    selection = dict(ladder.get("cluster_selection") or {})
    worst = {
        int(key): float(value)
        for key, value in (ladder.get("worst_seed_imbalance_at_this_rung") or {})
        .items()
    }
    primary = {
        int(key): float(value)
        for key, value in (
            ladder.get("primary_seed_imbalance_at_this_rung") or {}
        ).items()
    }
    selected = selection.get("selected_clusters")
    selected = None if selected is None else int(selected)
    candidates = {
        int(entry["clusters"]): entry
        for entry in (selection.get("candidates_considered") or [])
        if entry.get("clusters") is not None
    }
    admissible = None
    guarded = None
    if selected is not None and selected in candidates:
        admissible = candidates[selected].get("admissible_max_cluster_rows")
        guarded = candidates[selected].get("guarded_max_cluster_rows")
    if guarded is None:
        for cell in ladder.get("ladder") or []:
            guard = dict(cell.get("guard") or {})
            if int(cell.get("clusters") or -1) == selected:
                guarded = guard.get("guarded_max_cluster_rows")
                admissible = admissible or guard.get("admissible_max_cluster_rows")
    tolerance = (
        tolerance_to_adverse_imbalance(
            admissible_max_cluster_rows=float(admissible),
            guarded_max_cluster_rows=float(guarded),
        )
        if admissible is not None and guarded is not None else None
    )
    measured_worst = worst.get(400)
    per_seed = {
        str(int(cell["seed"])): float(cell["imbalance_max_over_mean"])
        for cell in (ladder.get("measured_imbalance") or {}).get("cells") or []
        if int(cell.get("clusters") or -1) == 400
        and int(cell.get("rows") or -1) == ROWS
    }
    movement = {
        seed: {
            "r0238": R0238_MEASURED_IMBALANCE_BY_SEED[int(seed)],
            "r0240": value,
            "relative_move": (
                value / R0238_MEASURED_IMBALANCE_BY_SEED[int(seed)] - 1.0
            ),
        }
        for seed, value in sorted(per_seed.items())
        if int(seed) in R0238_MEASURED_IMBALANCE_BY_SEED
    }
    return {
        "schema": LADDER_ATTEST_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": str(release_sha),
        "capability": ATTEST_CAPABILITY,
        "capabilities": [LADDER_CAPABILITY, IMBALANCE_CAPABILITY],
        "rows": ROWS,
        "spill": SPILL,
        "candidate": CANDIDATE,
        "nn_descent_setting": NN_DESCENT_SETTING,
        "nn_descent": {
            "graph_degree": GRAPH_DEGREE,
            "intermediate_graph_degree": INTERMEDIATE_GRAPH_DEGREE,
            "max_iterations": MAX_ITERATIONS,
        },
        "implementation_note": IMPLEMENTATION_NOTE,
        "selection_note": SELECTION_NOTE,
        "adverse_carry_note": ADVERSE_CARRY_NOTE,
        "tolerance_note": TOLERANCE_PUBLICATION_NOTE,
        "imbalance_margin": GUARD_IMBALANCE_MARGIN,
        "imbalance_margin_unchanged": GUARD_IMBALANCE_MARGIN == 1.164884,
        "selected_clusters": selected,
        "selected_clusters_registered_candidate": 400,
        "selected_differs_from_registered": (
            None if selected is None else bool(selected != 400)
        ),
        "worst_seed_imbalance_at_this_rung": {
            str(key): value for key, value in sorted(worst.items())
        },
        "primary_seed_imbalance_at_this_rung": {
            str(key): value for key, value in sorted(primary.items())
        },
        "per_seed_imbalance_at_c400": per_seed,
        "carried_prediction": {
            "source": CARRIED_IMBALANCE_SOURCE,
            "imbalance_at_c400": CARRIED_IMBALANCE_AT_C400,
            "realised_tolerance_at_c400": R0238_REALISED_TOLERANCE_AT_C400,
            "r0237_prediction_imbalance_at_c400": (
                R0237_PREDICTION_IMBALANCE_AT_C400
            ),
            "r0237_prediction_tolerance_at_c400": (
                R0237_PREDICTION_TOLERANCE_AT_C400
            ),
            "measured_worst_at_c400_this_round": measured_worst,
            "excess_over_carried_prediction": (
                None if measured_worst is None
                else measured_worst / CARRIED_IMBALANCE_AT_C400 - 1.0
            ),
            "measured_exceeds_carried_prediction": (
                None if measured_worst is None
                else bool(measured_worst > CARRIED_IMBALANCE_AT_C400)
            ),
            "per_seed_movement_from_r0238": movement,
        },
        "measured_tolerance_at_selected_c": tolerance,
        "measured_tolerance_percent": (
            None if tolerance is None else tolerance * 100.0
        ),
        "admissible_max_cluster_rows": admissible,
        "guarded_max_cluster_rows": guarded,
        "wall_budget": ladder.get("wall_budget"),
        "ladder_stopped_at": ladder.get("ladder_stopped_at"),
        "gpu_hours_cap": GPU_HOURS_CAP,
        "gpu_hours_cap_note": GPU_HOURS_CAP_NOTE,
        "build_timeout_s": BUILD_TIMEOUT_S,
        "build_timeout_note": BUILD_TIMEOUT_NOTE,
        "p90_wall_note": P90_WALL_NOTE,
        "safety_note": SAFETY_NOTE,
        "cost_note": COST_NOTE,
        "training_performed": False,
    }


def qualification_attestation(
    *, qualified: Mapping[str, Any], release_sha: str
) -> dict[str, Any]:
    """The headline qualification numbers, restated under this round's identity."""
    selected = dict(qualified.get("selected_graph") or {})
    strict = dict(selected.get("strict") or {})
    tie = dict(selected.get("tie_aware") or {})
    degrees = dict(qualified.get("degrees") or {})
    structural = dict(selected.get("structural") or {})
    in_zero = degrees.get("in_degree_zero_rows")
    out_zero = degrees.get("out_degree_zero_rows")
    return {
        "schema": QUALIFY_ATTEST_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": str(release_sha),
        "capability": ATTEST_CAPABILITY,
        "capabilities": [GRAPH_CAPABILITY],
        "rows": ROWS,
        "k": GRAPH_K,
        "spill": SPILL,
        "selected_clusters": qualified.get("selected_clusters"),
        "selected_cell": qualified.get("selected_cell"),
        "implementation_note": IMPLEMENTATION_NOTE,
        "qualification_note": QUALIFICATION_NOTE,
        "recall_population": qualified.get("recall_population"),
        "structural_population": qualified.get("structural_population"),
        "probe_rows": qualified.get("probe_rows"),
        "probe_seed": qualified.get("probe_seed"),
        "strict_recall_at_15": strict,
        "tie_aware_recall_at_15": tie,
        "density_decile_strict": selected.get("density_decile_strict"),
        "density_decile_tie_aware": selected.get("density_decile_tie_aware"),
        "rows_carrying_any_loss": selected.get("rows_carrying_any_loss"),
        "fraction_carrying_any_loss": selected.get("fraction_carrying_any_loss"),
        "missing_true_edges": selected.get("missing_true_edges"),
        "tie_aware_rows_at_zero": selected.get("tie_aware_rows_at_zero"),
        "zero_recall_forensic": selected.get("zero_recall_forensic"),
        "floors": {
            "tie_aware_mean": RECALL_MEAN_FLOOR,
            "tie_aware_p10": RECALL_P10_FLOOR,
            "zero_degree_rows_in_and_out": MAX_ZERO_DEGREE_ROWS,
        },
        "r0215_tripwire": {
            "population": degrees.get("population"),
            "in_degree_zero_rows": in_zero,
            "out_degree_zero_rows": out_zero,
            "in_degree_min": degrees.get("in_degree_min"),
            "out_degree_min": degrees.get("out_degree_min"),
            "raw_graph_zero_degree_rows": structural.get("zero_degree_rows"),
            "holds": (
                None if in_zero is None or out_zero is None
                else bool(int(in_zero) == 0 and int(out_zero) == 0)
            ),
            "note": (
                "R0215 traced the v1 150M map's clumps to rows with no valid "
                "canonical edges - twice the canonical degree-zero rate inside "
                "clumps. The tripwire fires on realised graph degree over ALL "
                "rows in BOTH directions and has never been evaluated at "
                "100,000,000 rows before this round."
            ),
        },
        "degrees": degrees,
        "directed_edges": qualified.get("directed_edges"),
        "edges_per_row": qualified.get("edges_per_row"),
        "fuzzy_weight_range": qualified.get("fuzzy_weight_range"),
        "builder_cosine_agreement": qualified.get("builder_cosine_agreement"),
        "reachability_at_this_rung": qualified.get("reachability_at_this_rung"),
        "measured_tolerance_at_this_rung": (
            (qualified.get("per_rung_derivation") or {})
            .get(str(ROWS), {})
            .get("selected_tolerance")
        ),
        "per_rung_derivation": qualified.get("per_rung_derivation"),
        "measured_doubling_50m_to_100m": (
            (qualified.get("imbalance_table") or {})
            .get("measured_doubling_50m_to_100m")
        ),
        "io_observed_regime": qualified.get("io_observed_regime"),
        "cost_note": COST_NOTE,
        "safety_note": SAFETY_NOTE,
        "graph": qualified.get("graph"),
        "neighbour_ids": qualified.get("neighbour_ids"),
        "training_performed": False,
        "gate_registered": False,
        "adoption_claimed": False,
        "map_quality_claimed": False,
        "scope_note": (
            "R0240 builds and qualifies the 100,000,000-row k15 fuzzy graph "
            "and does nothing else. It trains NO map, registers NO gate, and "
            "claims no dose, adoption, map-quality or registry result. Phase 3 "
            "consumes this graph; this round does not anticipate it."
        ),
    }


__all__ = [
    "ADVERSE_CARRY_NOTE",
    "ATTEST_CAPABILITY",
    "BUILD_TIMEOUT_S",
    "BUILD_TIMEOUT_NOTE",
    "CARRIED_IMBALANCE_AT_C400",
    "CARRIED_IMBALANCE_SOURCE",
    "COST_NOTE",
    "GPU_HOURS_CAP",
    "GPU_HOURS_CAP_NOTE",
    "GRAPH_CAPABILITY",
    "IMBALANCE_CAPABILITY",
    "IMPLEMENTATION_NOTE",
    "INHERITANCE_NOTE",
    "INHERITED_REACHABILITY_MANIFEST",
    "INHERITED_SUBSTRATE_MANIFEST",
    "INHERITED_TRUTH_MANIFEST",
    "LADDER_ATTEST_FILE",
    "LADDER_ATTEST_SCHEMA",
    "LADDER_CAPABILITY",
    "LADDER_P90_WALL_S",
    "P90_WALL_NOTE",
    "QUALIFICATION_NOTE",
    "QUALIFY_ATTEST_FILE",
    "QUALIFY_ATTEST_SCHEMA",
    "QUALIFY_P90_WALL_S",
    "QUALIFY_RESERVE_S",
    "R0237_PREDICTION_IMBALANCE_AT_C400",
    "R0237_PREDICTION_TOLERANCE_AT_C400",
    "R0238_ADMISSIBLE_MAX_CLUSTER_ROWS",
    "R0238_GUARDED_MAX_CLUSTER_ROWS",
    "R0238_MEASURED_IMBALANCE_BY_SEED",
    "R0238_REALISED_TOLERANCE_AT_C400",
    "REGISTERED_COMPOSITION",
    "REGISTERED_LADDER_PREFIX_SHA256",
    "REGISTERED_ORDERED_SUBSTRATE_SHA256",
    "REGISTERED_REACHABILITY_CEILING_C400",
    "REGISTERED_RESERVE_ROWS",
    "REGISTERED_SHARD_COUNTS",
    "REGISTERED_SUBSTRATE_FILE_SHA256",
    "REGISTERED_TRUTH_PROBE_ROWS",
    "REGISTERED_TRUTH_PROBE_SEED",
    "ROUND_ID",
    "ROWS",
    "Round0240Error",
    "SAFETY_NOTE",
    "SELECTION_NOTE",
    "TOLERANCE_PUBLICATION_NOTE",
    "ladder_attestation",
    "qualification_attestation",
    "tolerance_to_adverse_imbalance",
    "verify_inherited_reachability",
    "verify_inherited_substrate",
    "verify_inherited_truth",
]
