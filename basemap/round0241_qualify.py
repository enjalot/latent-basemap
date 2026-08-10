"""R0241 constants — qualify the 100,000,000-row k15 neighbour graph.

R0240 **built** the graph (`graph-k15-ids.i32.npy` `7fa70cad...` and
`graph-k15-cos.f32.npy` `474f14d2...`, `6,000,000,128` bytes each) and did not
qualify it: its single fused `qualify_100000k` node spent `5.094` GPU-h,
breached the round's cap and returned `1` without measuring recall, deciles,
the `min = 0.0` forensic or the R0215 degree-zero tripwire.

**The reason it failed is a design fault this round fixes by splitting, not by
spending more.** R0238's `run_qualify` computes candidate cosines for **all**
`100,000,000` rows because the fuzzy symmetrisation that follows it needs them.
But recall on a `500,000`-row probe needs only the probe's rows, and the R0215
degree-zero tripwire needs no substrate at all - it is a blocked scan of one
`6 GB` id array. review-0240-01 measured both nodes independently: the tripwire
at `21.4 s` and the full probe gather at `256.4 s`, against a fused node that
spent `18,338 s` and delivered nothing.

**No cost constant is carried from R0240's prose.** Its gather-amplification
table (`2.196x`, `4.276 h`) does not reproduce - review-0240-01 re-ran it and
got `6.176x` and `8.899 h`, and the constants live in an agent scratchpad with
no durable receipt, so the claim that the table is a reusable cost model is
blocked. Every prediction this round makes is therefore measured by the stage
that is about to pay it, on its own first units, and sealed in its own receipt.

review-0240-01 also stated the ordering diagnosis more precisely than R0240
did, and the sharper version is what this round acts on: **the node computed
the product's input at product scale (`100,000,000` rows of candidate cosines)
as a prerequisite of a science check that consumes `0.5%` of it, and persisted
nothing until after the last long stage.** Splitting fixes the first half;
persisting `_score_probe`'s output the moment it returns fixes the second, and
would have salvaged R0240 even unsplit. Both are registered here. No fuzzy
symmetrisation runs: that is a `~4.3 h` **product** step and it becomes its own
round.

**Everything is bound by hash and nothing is rebuilt.** The substrate, the
uniform truth probe and the `c = 400` reachability ceiling are R0238's, verified
through `basemap.round0240_rung5`'s registered literals; the graph arrays and
the build ladder receipt are R0240's, verified at their full signatures here.

**The science code is imported read-only.** `graph_validity` /
`_graph_validity_blocked`, `strict_containment_rows`, `tie_aware_rows`,
`summarize`, `zero_recall_forensic`, `truth_probe_query_rows`,
`rung_derivation`, `imbalance_tolerance`, `replicate_grid_table` and
`carry_distance` are called unchanged. No registered check is re-typed. What
this module adds is what did not exist: an **in-degree** measurement over the
raw directed graph, the exact symmetrised-isolation identity derived from it,
a self-calibrating wall guard with poll points at the stages R0240's node had
none, and the sampling-power arithmetic for a `0.5%` probe.

Nothing here trains a map, registers a gate, or claims an adoption.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from basemap.round0238_rung5 import (
    DENSITY_DECILES,
    DIMENSION,
    DUPLICATE_FAMILY_KTH_COSINE,
    FP32_TIE_NOISE_FLOOR,
    GRAPH_K,
    MAX_ZERO_DEGREE_ROWS,
    RECALL_MEAN_FLOOR,
    RECALL_P10_FLOOR,
    RECALL_POPULATION,
    SPILL,
    STRUCTURAL_POPULATION,
    TRUTH_PROBE_ROWS,
    TRUTH_PROBE_SEED,
    ZERO_RECALL_FORENSIC_NOTE,
)
from basemap.round0240_rung5 import (
    R0237_PREDICTION_IMBALANCE_AT_C400,
    R0238_ADMISSIBLE_MAX_CLUSTER_ROWS,
    R0238_GUARDED_MAX_CLUSTER_ROWS,
    R0238_MEASURED_IMBALANCE_BY_SEED,
    tolerance_to_adverse_imbalance,
)


class Round0241Error(RuntimeError):
    """Any fail-closed condition this round registers."""


ROUND_ID = "0241"
ROWS = 100_000_000
BUILD_ROUND_ID = "0240"
SUBSTRATE_ROUND_ID = "0238"

#: Owner-raised to 12.0 on 2026-08-10 (from 9.0, from 6.0). This round's own
#: proposal prices the work below 0.2 GPU-h, so the cap should never bind; if
#: the guard predicts it will, the round STOPS and reports rather than trimming
#: a registered check.
GPU_HOURS_CAP = 12.0
GPU_HOURS_CAP_NOTE = (
    "12.0 GPU-h, owner-raised on 2026-08-10 from 9.0 and before that from 6.0. "
    "R0240 spent 10.675 h against 9.0 - ladder 5.581 (succeeded, graph built) "
    "plus qualify 5.094 (aborted cooperatively at the soft deadline) - and "
    "reported the +1.675 h breach rather than trimming a check. This round "
    "rebuilds nothing and prices its two nodes at well under 0.2 GPU-h "
    "combined, so the cap is not the binding constraint on anything here. It "
    "is registered so every attempt's GPU time can be charged against it."
)

# --------------------------------------------------------------------------- #
# what is inherited, and bound by hash
# --------------------------------------------------------------------------- #
R0240_BUILD_DIR = (
    "/data/latent-basemap/runs/round-0240/queue/artifacts/"
    "minilm-mixed-100000k-cluster-spill-build-ladder-v1"
)
INHERITED_LADDER_RECEIPT = f"{R0240_BUILD_DIR}/build-ladder.json"
INHERITED_GRAPH_IDS = (
    f"{R0240_BUILD_DIR}/builds/r0238-n100000000-c400-s8/graph-k15-ids.i32.npy"
)
INHERITED_GRAPH_COS = (
    f"{R0240_BUILD_DIR}/builds/r0238-n100000000-c400-s8/graph-k15-cos.f32.npy"
)
INHERITED_BUILD_RECEIPT = (
    f"{R0240_BUILD_DIR}/builds/r0238-n100000000-c400-s8/build-receipt.json"
)
INHERITED_LADDER_ATTESTATION = (
    f"{R0240_BUILD_DIR}/round0240-ladder-attestation.json"
)

#: The graph R0240 built, at the signatures `roundreport` published in
#: result-0240's generated output table.
REGISTERED_GRAPH_IDS_SHA256 = (
    "7fa70cad7cf5bf204774bef542a09a56e0dfc9ab934a1dcb572147c1dec6939e"
)
REGISTERED_GRAPH_COS_SHA256 = (
    "474f14d20bcc01a251ac95a20045a973feb23eb2c9ab712137904d12418cec3e"
)
REGISTERED_GRAPH_ARRAY_BYTES = 6_000_000_128
REGISTERED_LADDER_RECEIPT_SHA256 = (
    "36b469531a9ec4131324593de5445587ea772f418821487c7c225ae944e40a63"
)
REGISTERED_SELECTED_CLUSTERS = 400
REGISTERED_SELECTED_CELL = "r0238-n100000000-c400-s8"

INHERITANCE_NOTE = (
    "R0240's graph arrays and build-ladder receipt, and R0238's substrate, "
    "uniform truth probe and c = 400 reachability ceiling, are bound at their "
    "FULL sha256 signatures and re-earned by nothing. Every literal is "
    "asserted inside the node before any measurement, and a mismatch raises "
    "Round0241Error and stops the round. Rebuilding the graph would cost "
    "5.58 GPU-h to reproduce bytes that already exist and are already sealed."
)

# --------------------------------------------------------------------------- #
# node A — the R0215 degree-zero tripwire, first, on the id array alone
# --------------------------------------------------------------------------- #
#: Row stripe for both structural passes. Matches R0238's GRAPH_SCAN_BLOCK so
#: the blocked reviewed scan and this round's own degree pass walk the array in
#: the same units and their totals are directly comparable.
TRIPWIRE_BLOCK = 5_000_000

#: WHICH GRAPH. Every degree statistic this round publishes is computed on the
#: RAW DIRECTED k15 graph `graph-k15-ids.i32.npy` (`7fa70cad...`). Nothing here
#: is computed on the fuzzy-symmetrised graph, which does not exist yet.
GRAPH_UNDER_TEST = "raw directed k15 neighbour graph (graph-k15-ids.i32.npy)"

DEGREE_SEMANTICS_NOTE = (
    "Every degree number in this artifact names the graph it is computed on, "
    "because review-0240-01/F1 showed the program had been reporting ONE "
    "number TWICE since at least R0237. Definitions, once: "
    "(a) RAW OUT-DEGREE - usable entries in a row of graph-k15-ids.i32.npy, "
    "where usable is R0220's definition: first occurrence of that id within "
    "the row, in range, not the row itself. It is exactly k = 15 for every row "
    "unless the builder emitted a self-loop, an intra-row duplicate or an "
    "out-of-range id, so a nonzero count of anything below k is a builder "
    "defect and that is precisely what it tests. "
    "(b) RAW IN-DEGREE - how many rows name this row as a neighbour. It is the "
    "informative direction, it is genuinely variable, and no round in this "
    "program had ever measured it before review-0240-01 did. "
    "(c) SYMMETRISED DEGREE - degree in A + A^T - A o A^T. In-degree and "
    "out-degree there are THE SAME NUMBER by construction, so reporting both "
    "as two checks is an algebraic identity and must never be done again."
)

VACUITY_NOTE = (
    "What this tripwire can and cannot fail, stated before it runs. On a raw "
    "k-NN graph the out-degree is k for every row by construction, so the "
    "out-degree half can only fail if the BUILDER emitted degenerate rows - "
    "self-loops, intra-row duplicates or out-of-range ids. That is a real "
    "failure mode (it is what would make a row edgeless here) and it has never "
    "been checked at this scale, but it is a narrow one and this round says so "
    "rather than letting a clean result read as a broader guarantee. The v1 "
    "defect R0215 diagnosed arose at a LATER stage: rows lost their edges at "
    "canonicalization/exclusion, not at build time (R0034 counted 2,779,481 "
    "such rows, ~1.85% of the v1 150M map, and R0215 showed clump rows carry "
    "twice the canonical degree-zero rate, 0.0365 against 0.0180). No "
    "canonicalization has been applied to this graph. So a clean tripwire here "
    "certifies the BUILDER's output and does NOT certify the rung against the "
    "v1 failure mode end to end; the tripwire must be re-run after "
    "canonicalization and after symmetrisation. This round refuses to claim "
    "otherwise."
)

TRIPWIRE_NOTE = (
    "R0215 made a degree-zero tripwire mandatory on every Phase-2 graph build "
    "and it has NEVER been run at 100,000,000 rows: R0238 built no graph and "
    "R0240's fused node raised inside the structural scan before reporting "
    "one. THIS NODE IS ITS FIRST EXECUTION AT THIS SCALE, on the raw directed "
    "graph, in both directions computed SEPARATELY and named separately. It "
    "needs no substrate and no CUDA context: it is two blocked passes over one "
    "6,000,000,128-byte int32 array, which review-0240-01 measured at 21.4 s "
    "including the 1.5e9-entry bincount."
)

SYMMETRISED_IDENTITY_NOTE = (
    "The symmetrised zero-degree count is reported here as a DERIVED quantity, "
    "computed from the raw graph without symmetrising, and it is one number "
    "rather than two. The symmetrised support is exactly the union of the "
    "directed adjacency and its transpose: UMAP's "
    "compute_membership_strengths assigns weight 0 to a self-loop and a "
    "strictly positive weight to every other entry, A + A^T - A o A^T at "
    "set_op_mix_ratio 1.0 is nonzero wherever either operand is, and "
    "eliminate_zeros drops only the self-loops. So row i is isolated in the "
    "symmetrised graph IFF it has no usable out-edge AND no usable in-edge in "
    "the raw graph. tests/test_round0241_cpu_smoke.py proves the identity "
    "against R0238's own _fuzzy_symmetrise_blocked on synthetic graphs that DO "
    "contain isolated rows, and on a case where a row has an in-edge only. "
    "The fuzzy round that follows must report symmetrised degree ONCE and say "
    "that its two directions are the same number."
)

#: review-0240-01/F1's independent measurement on the same sealed bytes. This
#: round reconciles against it and treats any disagreement as the finding.
REVIEWER_SOURCE = "review-0240-2026-08-10-01, F1"
REVIEWER_RAW_ZERO_IN_DEGREE_ROWS = 4_424_010
REVIEWER_RAW_IN_DEGREE_MAX = 44_530
REVIEWER_RAW_IN_DEGREE_MEAN = 15.0
REVIEWER_RAW_OUT_DEGREE_MIN = 15
REVIEWER_RAW_OUT_DEGREE_ZERO_ROWS = 0

HUBNESS_NOTE = (
    "4.42% of rows having no incoming edge is ordinary hubness for a "
    "high-dimensional k-NN graph, and every one of those rows still carries "
    "its own 15 out-edges, so after symmetrisation it has 15 edges and "
    "attractive force. It is therefore NOT the v1 failure mode, which was rows "
    "with no valid edges at all. What this round can settle from the graph "
    "alone is the count and the distribution; what it CANNOT settle is whether "
    "hub concentration distorts the embedding, because that is a property of "
    "the map and no map exists. This round reports the distribution and stops "
    "there rather than reassuring."
)

#: Reported, never used to relax a floor: `out + in` double-counts a mutual
#: pair, `max(out, in)` under-counts one, so the true symmetrised degree lies
#: between them. Only the zero test is exact, and the zero test is the tripwire.
SYMMETRISED_DEGREE_BOUND_NOTE = (
    "symmetrised degree of a row lies in [max(out_usable, in_usable), "
    "out_usable + in_usable]: the upper bound double-counts every mutual pair "
    "and the lower bound counts each pair once. The bounds coincide at zero, "
    "which is why the tripwire itself is exact while the degree distribution "
    "is reported as an interval."
)

# --------------------------------------------------------------------------- #
# node B — recall on the registered probe, and only the probe's rows
# --------------------------------------------------------------------------- #
#: Anchor rows per gather block. Small enough that a poll fires roughly every
#: second at R0240's measured gather rate, which is the whole point: R0240's
#: cooperative abort took 2 h 52 m to reach a safe point because
#: `_candidate_cosines` and `_score_probe` contain no poll at all.
PROBE_GATHER_BLOCK = 16_384

RECALL_NOTE = (
    "recall is scored on R0238's ALREADY-REGISTERED 500,000-row uniform probe "
    "at seed 238000, re-derived in-node from the seed alone through "
    "truth_probe_query_rows and asserted equal to the sealed query rows. It is "
    "not a fresh draw, it is not a seed union and it is not hub-biased - "
    "review-0227-01 caught exactly that construction. Registering the EXISTING "
    "probe also makes the measurement directly comparable to the c = 400 "
    "reachability ceiling 0.9983063, which was measured on the same rows. "
    "Candidate cosines are recomputed independently of the builder's "
    "accumulator (review-0216-01) for the PROBE'S ROWS ONLY - 500,000 anchors, "
    "not 100,000,000 - which is what makes this node minutes rather than hours."
)

CPU_ARITHMETIC_NOTE = (
    "the candidate-cosine recompute runs in CPU float32 numpy, not on the "
    "card. At 500,000 x 15 x 384 the arithmetic is 2.9 GFLOP and the node is "
    "entirely gather-bound, so a CUDA context would buy nothing and would "
    "reintroduce the one hazard this program has twice needed a reboot to "
    "recover from. Both nodes still declare gpu_required so they hold the "
    "exclusive GPU lease and so the runner never signals them; neither node "
    "creates a CUDA context, and the receipts say so."
)

# --------------------------------------------------------------------------- #
# the wall guard R0240 did not have
# --------------------------------------------------------------------------- #
WALL_GUARD_NOTE = (
    "R0240's own finding: a cap checked only at admission is not a bound. "
    "rounds/runner.py refuses to START a node once cumulative GPU wall reaches "
    "the cap, so `qualify_100000k` was admitted correctly at 5.58 h of 9.0 and "
    "then ran 5.09 h with nothing able to stop it on budget. Every stage here "
    "therefore carries a StageGuard: it measures the wall of its own first "
    "calibration units, predicts the whole stage from that measurement rather "
    "than from any inherited constant, and RAISES BEFORE the expensive part if "
    "the prediction exceeds the stage's budget. It also polls the runner's "
    "cooperative abort flag and its own deadline every unit, so the "
    "flag-to-unwind gap is one unit rather than the 2 h 52 m R0240 measured. "
    "No signal is involved on any path: the guard raises in-band and the "
    "interpreter unwinds normally."
)

#: Hard cooperative deadlines. Both are ~20x the measured cost of the stage they
#: bound, so they can only fire on a genuine pathology, and they are enforced by
#: an in-band raise at a poll point - never by a signal, and never by the
#: child-process wall bound CPython implements as a SIGKILL on expiry.
TRIPWIRE_DEADLINE_S = 1_800.0
RECALL_DEADLINE_S = 2_400.0
#: What node A holds back from its own budget for node B.
RECALL_RESERVE_S = 1_200.0
#: Predicted stage wall must leave this much slack inside the budget.
WALL_GUARD_MARGIN = 1.25
#: Units measured before the guard is allowed to predict.
CALIBRATION_UNITS = 1

SAFETY_NOTE = (
    "This round starts NO child process, creates NO CUDA context and hands "
    "cuVS NOTHING, so the read-only-memmap precondition is satisfied "
    "vacuously and is nevertheless asserted: every array either node opens is "
    "opened with np.load(mmap_mode='r') and the node raises unless it is a "
    "non-writeable read-only memmap. Neither file this round adds to the node "
    "path imports the child-process module, constructs a child, calls a "
    "process-signalling method, or bounds anything by a wall that signals on "
    "expiry. That is a property of the SOURCE, asserted token by token in "
    "tests/test_round0241_contract.py by reading the files, and it is not "
    "delegated to a detector: experiments/check_signal_safety.py is run for "
    "information only because it is unreleased, has a known waiver hole on a "
    "locally hoisted argument vector, and 2 of 18 planted evasions still "
    "evade it, so its exit code is NOT cited as evidence of anything. To stop "
    "this round, write <queue_root>/logs/<node>.abort and wait."
)

# --------------------------------------------------------------------------- #
# what the round explicitly does NOT do
# --------------------------------------------------------------------------- #
NO_SYMMETRISATION_NOTE = (
    "The fuzzy symmetrisation over all 100,000,000 rows is NOT run here. It "
    "costs ~4.3 h of gather plus ~30 GB of edge output and it is a PRODUCT "
    "step; qualification is a SCIENCE check. Ordering the product step first "
    "is what let a failed product step destroy a passing science check in "
    "R0240. It becomes its own registered round after this one passes. "
    "Consequently this round publishes no fuzzy weights, no edge list and no "
    "edges_per_row, and nothing downstream may read one into it."
)

SCOPE_NOTE = (
    "R0241 qualifies the 100,000,000-row k15 NEIGHBOUR graph and discharges "
    "three arithmetic debts. It builds nothing, symmetrises nothing, trains no "
    "map, registers no gate and claims no dose, adoption, map-quality or "
    "registry result."
)

# --------------------------------------------------------------------------- #
# capabilities
# --------------------------------------------------------------------------- #
TRIPWIRE_CAPABILITY = "minilm-mixed-100000k-k15-degree-zero-tripwire-v1"
RECALL_CAPABILITY = "minilm-mixed-100000k-k15-neighbour-graph-qualification-v1"
DERIVATION_CAPABILITY = "minilm-mixed-cluster-spill-per-rung-rederivation-100m-v1"
TRIPWIRE_SCHEMA = "round0241-100000k-degree-zero-tripwire-v1"
RECALL_SCHEMA = "round0241-100000k-graph-qualification-v1"
RECALL_EARLY_SCHEMA = "round0241-100000k-recall-first-write-v1"
TRIPWIRE_FILE = "degree-zero-tripwire.json"
RECALL_FILE = "graph-qualification.json"
RECALL_EARLY_FILE = "recall-first-write.json"

EARLY_PERSIST_NOTE = (
    "review-0240-01/F2: R0240 DID compute recall on this probe and then lost "
    "it. Its sealed traceback shows the raise happened inside "
    "_graph_validity_blocked, 44 lines AFTER _score_probe returned, and the "
    "node persisted nothing until the end - so a round that had the answer "
    "reported 'no recall was measured'. The reviewer's conclusion was that a "
    "single write before the structural scan would have salvaged it. This file "
    "is that write: the recall summary, its verdict and its sampling power are "
    "sealed the instant they exist, before the builder-agreement pass and "
    "before any derivation can fail. It is deliberately redundant with "
    "graph-qualification.json, which supersedes it when the node completes."
)

# --------------------------------------------------------------------------- #
# the three debts this round discharges, from sealed artifacts only
# --------------------------------------------------------------------------- #
DEBTS_NOTE = (
    "Three items have been owed for three rounds and cost zero GPU. "
    "(1) The per-rung re-derivation of c from imbalance MEASURED at 100M - "
    "review-0238-01/F3 found no per_rung_derivation key in R0238's ladder and "
    "R0240 did not publish one either. (2) The c = 400 per-seed movement "
    "table across 50M -> 100M -> 100M(re-measured). (3) The measured "
    "tolerance_to_adverse_imbalance, which R0238 never published, review-0238-"
    "01/F4 had to derive by hand as 51.3984%, and R0240 reported as 51.4516% "
    "in its result BODY but sealed in no artifact. All three are computed here "
    "from sealed, hash-bound receipts only - never from another round's prose - "
    "using the reviewed pure functions rung_derivation, imbalance_tolerance, "
    "carry_distance and replicate_grid_table imported unchanged."
)

#: The tolerance chain, in the one definition every rung uses. The first two
#: legs are arithmetic on R0240's registered guard inputs; the third is
#: computed in-node from R0240's own sealed ladder receipt.
TOLERANCE_CHAIN_NOTE = (
    "tolerance_to_adverse_imbalance = admissible_max_cluster_rows / "
    "guarded_max_cluster_rows - 1. Leg 1 is the value the 50M pricing carried "
    "(imbalance 2.456543); leg 2 is R0238's measured worst-of-five 2.8204385; "
    "leg 3 is R0240's own re-measured worst-of-five, read from its sealed "
    "cluster_selection. The chain shows what the 50M -> 100M doubling cost and "
    "that re-measurement did not erode it further."
)


def _positive(value: float, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise Round0241Error(f"{label} must be a finite positive number")
    return number


# --------------------------------------------------------------------------- #
# sampling power of a 0.5% uniform probe — arithmetic, stated honestly
# --------------------------------------------------------------------------- #
def sampling_uncertainty(
    *,
    mean: float,
    sd: float,
    probe_rows: int = TRUTH_PROBE_ROWS,
    population: int = ROWS,
) -> dict[str, Any]:
    """Standard error of a per-row mean estimated from a uniform sample.

    The probe is drawn WITHOUT replacement, so the finite-population correction
    `sqrt((N - n) / (N - 1))` applies. At `n = 500,000` of `N = 100,000,000` it
    is `0.9975`, i.e. it changes nothing: a 0.5% sample is statistically almost
    identical to sampling with replacement. Reporting it makes that explicit
    rather than leaving a reader to wonder whether it was neglected.
    """
    n = int(probe_rows)
    big_n = int(population)
    if n <= 1 or big_n <= n:
        raise Round0241Error("sampling uncertainty needs 1 < probe_rows < population")
    spread = float(sd)
    if not math.isfinite(spread) or spread < 0.0:
        raise Round0241Error("sampling uncertainty needs a finite non-negative sd")
    fpc = math.sqrt((big_n - n) / (big_n - 1))
    se_srs = spread / math.sqrt(n)
    se = se_srs * fpc
    return {
        "estimator": "per-row recall mean over a uniform without-replacement probe",
        "probe_rows": n,
        "population_rows": big_n,
        "sampling_fraction": n / big_n,
        "sample_sd": spread,
        "standard_error_srs": se_srs,
        "finite_population_correction": fpc,
        "standard_error": se,
        "ci95_half_width": 1.959963984540054 * se,
        "ci95_low": float(mean) - 1.959963984540054 * se,
        "ci95_high": float(mean) + 1.959963984540054 * se,
        "note": (
            "the standard error describes uncertainty in the POPULATION MEAN "
            "only. It says nothing about the tail, and nothing about where in "
            "the map any loss sits"
        ),
    }


def region_detection_power(
    *,
    region_rows: int,
    probe_rows: int = TRUTH_PROBE_ROWS,
    population: int = ROWS,
) -> dict[str, Any]:
    """Probability a uniform probe touches a localised region of `region_rows`.

    This is the question R0228's displacement work cares about: a contiguous
    patch of the map whose edges are bad. A uniform probe finds such a region
    only by landing in it, and the chance of that is governed by the region's
    SIZE, not by its severity.
    """
    n = int(probe_rows)
    big_n = int(population)
    m = int(region_rows)
    if m < 0 or m > big_n:
        raise Round0241Error("region_rows must lie in [0, population]")
    # P(no probe row lands in the region), hypergeometric, computed in logs.
    if m == 0:
        miss = 1.0
    elif m > big_n - n:
        miss = 0.0
    else:
        miss = math.exp(
            math.lgamma(big_n - m + 1) - math.lgamma(big_n - m - n + 1)
            - math.lgamma(big_n + 1) + math.lgamma(big_n - n + 1)
        )
    return {
        "region_rows": m,
        "region_fraction": m / big_n,
        "probe_rows": n,
        "expected_probe_rows_in_region": n * m / big_n,
        "probability_missed_entirely": miss,
        "probability_detected": 1.0 - miss,
    }


def smallest_detectable_region(
    *,
    confidence: float = 0.95,
    probe_rows: int = TRUTH_PROBE_ROWS,
    population: int = ROWS,
) -> dict[str, Any]:
    """Smallest region the probe hits at least once with probability `confidence`.

    Solved on the with-replacement approximation `1 - (1 - m/N)**n >= c` and
    then TIGHTENED by walking the exact hypergeometric, so the reported figure
    is the exact threshold rather than an approximation.
    """
    if not 0.0 < float(confidence) < 1.0:
        raise Round0241Error("confidence must lie strictly inside (0, 1)")
    n = int(probe_rows)
    big_n = int(population)
    approx = big_n * (1.0 - (1.0 - float(confidence)) ** (1.0 / n))
    candidate = max(1, int(math.floor(approx)) - 4)
    while candidate <= big_n:
        power = region_detection_power(
            region_rows=candidate, probe_rows=n, population=big_n
        )
        if power["probability_detected"] >= float(confidence):
            return {
                "confidence": float(confidence),
                "region_rows": candidate,
                "region_fraction": candidate / big_n,
                "probability_detected": power["probability_detected"],
                "approximation_used_as_seed": approx,
            }
        candidate += 1
    raise Round0241Error("no region size reaches the requested confidence")


def probe_power_statement(
    *,
    mean: float,
    sd: float,
    probe_rows: int = TRUTH_PROBE_ROWS,
    population: int = ROWS,
    region_sizes: Sequence[int] = (
        100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000,
    ),
) -> dict[str, Any]:
    """What a 0.5% uniform probe can and cannot establish, with the arithmetic.

    Written as a first-class artifact field because the honest limitation is
    part of the result, not a caveat bolted onto prose.
    """
    return {
        "uncertainty": sampling_uncertainty(
            mean=mean, sd=sd, probe_rows=probe_rows, population=population
        ),
        "detection_by_region_size": [
            region_detection_power(
                region_rows=int(size), probe_rows=probe_rows,
                population=population,
            )
            for size in region_sizes
        ],
        "smallest_region_detected_at_95": smallest_detectable_region(
            confidence=0.95, probe_rows=probe_rows, population=population
        ),
        "smallest_region_detected_at_99": smallest_detectable_region(
            confidence=0.99, probe_rows=probe_rows, population=population
        ),
        "can_establish": [
            "the population mean of per-row recall, to the standard error above",
            "the population fraction of rows carrying any loss, to a comparable "
            "standard error",
            "the presence of ANY defect affecting more than the "
            "smallest-region-detected-at-95 row count, wherever it sits, "
            "because uniformity guarantees the probe is not steered away from it",
            "the recall profile by density decile, since the deciles are cut on "
            "the probe's own exact 15th-best cosine and each carries ~50,000 rows",
        ],
        "cannot_establish": [
            "that NO localised region of bad edges exists: a region smaller "
            "than the 95% threshold above is more likely than not to be missed "
            "entirely, and a region small enough to be missed is by "
            "construction invisible in every statistic this node reports",
            "that a region which IS sampled is bad, unless it contributes "
            "enough probe rows to move a decile - a 1,000-row region "
            "contributes ~5 probe rows and disappears inside the tail",
            "anything about the SPATIAL arrangement of loss: the probe carries "
            "no map coordinates, so 'localised' is not even expressible in it. "
            "That is the failure mode R0228's displacement work cares about and "
            "it needs a spatially-aware instrument, not a bigger uniform probe",
            "the exact per-row truth for the 99.5% of rows never searched: full "
            "exact truth over all 100,000,000 rows costs ~75 GPU-h",
        ],
        "the_tripwire_is_not_a_sample": (
            "the R0215 degree-zero tripwire has NO sampling uncertainty. It is "
            "evaluated over all 100,000,000 rows in both directions, so a "
            "single edgeless row anywhere in the map is caught with "
            "probability 1. The sampling limits above bound the RECALL claim "
            "only, and the tripwire is the check that catches the v1 failure "
            "mode."
        ),
    }


# --------------------------------------------------------------------------- #
# the wall guard, self-calibrating and polled
# --------------------------------------------------------------------------- #
class StageGuard:
    """Predict a stage's wall from its OWN first units, then bound it.

    Three jobs, all of which R0240's qualification node lacked:

    1. **Predict before the expensive part.** After `calibration_units` the
       guard extrapolates and raises if the stage cannot fit its budget. A
       refusal costs the calibration and nothing else.
    2. **Poll every unit.** `poll` calls the injected abort check on every unit,
       so the runner's cooperative flag is observed within one unit instead of
       within one four-hour stage.
    3. **Bound the wall in-band.** On deadline the guard RAISES; it never
       signals, never spawns and never touches another process.

    Everything is injected (`clock`, `abort_check`) so the whole class is
    exercised on the CPU in tests with no I/O.
    """

    def __init__(
        self,
        *,
        label: str,
        units_total: int,
        budget_s: float,
        deadline_s: float,
        abort_check: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        calibration_units: int = CALIBRATION_UNITS,
        margin: float = WALL_GUARD_MARGIN,
    ) -> None:
        self.label = str(label)
        self.units_total = int(units_total)
        if self.units_total <= 0:
            raise Round0241Error(f"{label} needs at least one unit of work")
        self.budget_s = _positive(budget_s, label=f"{label} budget")
        self.deadline_s = _positive(deadline_s, label=f"{label} deadline")
        self.calibration_units = max(1, int(calibration_units))
        self.margin = _positive(margin, label=f"{label} margin")
        self._abort_check = abort_check
        self._clock = clock
        self._started = clock()
        self.units_done = 0
        self.prediction: dict[str, Any] | None = None

    @property
    def elapsed_s(self) -> float:
        return self._clock() - self._started

    def poll(self, where: str) -> None:
        """Observe the runner's abort request and this stage's own deadline."""
        if self._abort_check is not None:
            self._abort_check(f"{self.label}: {where}")
        elapsed = self.elapsed_s
        if elapsed > self.deadline_s:
            raise Round0241Error(
                f"{self.label} passed its cooperative deadline "
                f"({elapsed:.1f} s > {self.deadline_s:.1f} s) at {where}. "
                "Unwinding in-band; no signal was delivered and no child "
                "process exists to signal."
            )

    def unit_done(self, where: str) -> None:
        """Record one completed unit and, at calibration, decide whether to go on."""
        self.units_done += 1
        self.poll(where)
        if self.units_done == self.calibration_units:
            self.prediction = self.predict()
            if not self.prediction["fits"]:
                raise Round0241Error(
                    f"{self.label} wall guard REFUSES the remaining work: "
                    f"predicted {self.prediction['predicted_total_s']:.1f} s "
                    f"x margin {self.margin} = "
                    f"{self.prediction['predicted_with_margin_s']:.1f} s "
                    f"against a budget of {self.budget_s:.1f} s, measured on "
                    f"this stage's own first {self.units_done} unit(s). "
                    "Reported, not trimmed."
                )

    def predict(self) -> dict[str, Any]:
        """Extrapolate the stage from the units it has actually completed."""
        elapsed = self.elapsed_s
        if self.units_done <= 0:
            raise Round0241Error(f"{self.label} cannot predict before a unit")
        per_unit = elapsed / self.units_done
        total = per_unit * self.units_total
        with_margin = total * self.margin
        return {
            "label": self.label,
            "units_total": self.units_total,
            "units_measured": self.units_done,
            "measured_seconds": elapsed,
            "seconds_per_unit": per_unit,
            "predicted_total_s": total,
            "margin": self.margin,
            "predicted_with_margin_s": with_margin,
            "budget_s": self.budget_s,
            "deadline_s": self.deadline_s,
            "fits": bool(with_margin <= self.budget_s),
            "basis": (
                "this stage's OWN measured units, never an inherited constant "
                "and never a figure transcribed from another round's prose"
            ),
        }

    def receipt(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "units_total": self.units_total,
            "units_done": self.units_done,
            "wall_s": self.elapsed_s,
            "budget_s": self.budget_s,
            "deadline_s": self.deadline_s,
            "deadline_reached": False,
            "calibration_prediction": self.prediction,
            "poll_points_per_unit": 1,
            "note": WALL_GUARD_NOTE,
        }


# --------------------------------------------------------------------------- #
# the degree cross-check
# --------------------------------------------------------------------------- #
#: Fields the reviewed blocked scan and this round's own degree pass must agree
#: on exactly. They are computed by two independent code paths over the same
#: bytes, so a disagreement means one of them is wrong and the round stops.
CROSS_CHECKED_FIELDS: tuple[str, ...] = (
    "rows", "width", "out_of_range_entries", "rows_with_out_of_range",
    "self_loop_entries", "rows_with_self_loop", "duplicate_entries",
    "rows_with_duplicates", "min_usable_degree", "rows_below_k",
    "zero_degree_rows",
)


def cross_check_structural(
    *, reviewed: Mapping[str, Any], own: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless two independent structural passes agree exactly.

    `reviewed` is R0238's `_graph_validity_blocked`, which wraps R0220's
    `graph_validity` unchanged. `own` is this round's degree pass, which exists
    because in-degree needed a per-row accumulator the reviewed function does
    not build. Computing out-degree in the same loop is free, and comparing the
    two is the cheapest independent check available: the same 1.5e9 entries,
    two code paths, one answer.
    """
    disagreements = {
        field: {"reviewed": reviewed.get(field), "own": own.get(field)}
        for field in CROSS_CHECKED_FIELDS
        if int(reviewed.get(field, -1)) != int(own.get(field, -2))
    }
    if disagreements:
        raise Round0241Error(
            "R0241 structural cross-check FAILED: R0220's reviewed "
            f"graph_validity and this round's degree pass disagree on "
            f"{sorted(disagreements)}. {disagreements}"
        )
    return {
        "agree": True,
        "fields": list(CROSS_CHECKED_FIELDS),
        "reviewed_source": (
            "experiments.round0238_nodes._graph_validity_blocked, which wraps "
            "basemap.round0220_cuvs_qualification.graph_validity unchanged"
        ),
        "own_source": (
            "basemap.round0241_qualify's blocked degree pass, written because "
            "in-degree needs a per-row accumulator no reviewed function builds"
        ),
        "note": (
            "an in-node probe that shares the builder's accumulator is not "
            "evidence (review-0216-01); two passes that share nothing but the "
            "bytes are"
        ),
    }


def reconcile_in_degree(
    *,
    measured_zero_in_degree_rows: int,
    measured_in_degree_max: int,
    measured_in_degree_mean: float,
    measured_out_degree_min: int,
    measured_out_degree_zero_rows: int,
) -> dict[str, Any]:
    """This round's raw in-degree against review-0240-01's independent value.

    The reviewer measured the same sealed bytes with its own blocked
    `bincount` in `21.4 s`. Two independent measurements of the same file
    should agree exactly; if they do not, THAT is the finding, and it is
    reported as a disagreement rather than resolved in favour of either side.
    """
    agree = (
        int(measured_zero_in_degree_rows) == REVIEWER_RAW_ZERO_IN_DEGREE_ROWS
        and int(measured_in_degree_max) == REVIEWER_RAW_IN_DEGREE_MAX
        and int(measured_out_degree_zero_rows)
        == REVIEWER_RAW_OUT_DEGREE_ZERO_ROWS
    )
    return {
        "graph": GRAPH_UNDER_TEST,
        "reference": REVIEWER_SOURCE,
        "reference_is_independent": True,
        "zero_in_degree_rows": {
            "measured_here": int(measured_zero_in_degree_rows),
            "reviewer": REVIEWER_RAW_ZERO_IN_DEGREE_ROWS,
            "difference": int(measured_zero_in_degree_rows)
            - REVIEWER_RAW_ZERO_IN_DEGREE_ROWS,
            "fraction_of_rows": int(measured_zero_in_degree_rows) / ROWS,
        },
        "in_degree_max": {
            "measured_here": int(measured_in_degree_max),
            "reviewer": REVIEWER_RAW_IN_DEGREE_MAX,
        },
        "in_degree_mean": {
            "measured_here": float(measured_in_degree_mean),
            "reviewer": REVIEWER_RAW_IN_DEGREE_MEAN,
            "expected_exactly": float(GRAPH_K),
            "note": (
                "the mean raw in-degree is exactly k when every entry is in "
                "range, because the same 1.5e9 entries are counted from the "
                "other side; a mean below k is proof of out-of-range ids"
            ),
        },
        "out_degree_min": {
            "measured_here": int(measured_out_degree_min),
            "reviewer": REVIEWER_RAW_OUT_DEGREE_MIN,
        },
        "agree": bool(agree),
        "disagreement_is_the_finding": not bool(agree),
        "hubness_note": HUBNESS_NOTE,
    }


def tripwire_verdict(
    *,
    out_degree_zero_rows: int,
    in_degree_zero_rows: int,
    symmetrised_isolated_rows: int,
    raw_zero_degree_rows: int,
    max_zero_degree_rows: int = MAX_ZERO_DEGREE_ROWS,
) -> dict[str, Any]:
    """The R0215 tripwire, with every direction named by the graph it is on.

    review-0240-01/F1 established that "zero-degree `== 0` in both in-degree
    and out-degree" has been ONE number reported TWICE since at least R0237,
    because both were taken on the symmetrised graph where they are equal by
    construction. This function refuses that shape:

    * `out_degree_zero_rows` and `raw_graph_zero_degree_rows` are on the RAW
      directed graph and they GATE - a row with no usable out-edge would be
      edgeless after symmetrisation unless something points at it;
    * `in_degree_zero_rows` is on the RAW directed graph and is DESCRIPTIVE -
      `~4.42%` of rows in any high-dimensional k-NN graph are nobody's
      neighbour, and gating on it would fail every rung spuriously;
    * `symmetrised_isolated_rows` is the conjunction, which is the exact
      symmetrised zero-degree count and is ONE number, not two.
    """
    holds = (
        int(symmetrised_isolated_rows) <= int(max_zero_degree_rows)
        and int(out_degree_zero_rows) <= int(max_zero_degree_rows)
        and int(raw_zero_degree_rows) <= int(max_zero_degree_rows)
    )
    return {
        "population": STRUCTURAL_POPULATION,
        "graph": GRAPH_UNDER_TEST,
        "first_execution_at_this_scale": True,
        "raw_out_degree_zero_rows": int(out_degree_zero_rows),
        "raw_in_degree_zero_rows": int(in_degree_zero_rows),
        "raw_graph_zero_degree_rows": int(raw_zero_degree_rows),
        "symmetrised_isolated_rows_derived": int(symmetrised_isolated_rows),
        "max_zero_degree_rows": int(max_zero_degree_rows),
        "holds": bool(holds),
        "gating_quantities": [
            "raw_out_degree_zero_rows", "raw_graph_zero_degree_rows",
            "symmetrised_isolated_rows_derived",
        ],
        "descriptive_quantities": ["raw_in_degree_zero_rows"],
        "degree_semantics_note": DEGREE_SEMANTICS_NOTE,
        "vacuity_note": VACUITY_NOTE,
        "identity_note": SYMMETRISED_IDENTITY_NOTE,
        "hubness_note": HUBNESS_NOTE,
        "r0215_note": TRIPWIRE_NOTE,
    }


def recall_verdict(
    *,
    tie_aware: Mapping[str, Any],
    strict: Mapping[str, Any],
    mean_floor: float = RECALL_MEAN_FLOOR,
    p10_floor: float = RECALL_P10_FLOOR,
) -> dict[str, Any]:
    """The registered recall floors, evaluated and reported either way."""
    mean = float(tie_aware["mean"])
    p10 = float(tie_aware["p10"])
    return {
        "population": RECALL_POPULATION,
        "tie_aware_mean": mean,
        "tie_aware_p10": p10,
        "tie_aware_min": float(tie_aware["min"]),
        "strict_mean": float(strict["mean"]),
        "strict_p10": float(strict["p10"]),
        "strict_min": float(strict["min"]),
        "mean_floor": float(mean_floor),
        "p10_floor": float(p10_floor),
        "mean_clears": bool(mean >= float(mean_floor)),
        "p10_clears": bool(p10 >= float(p10_floor)),
        "holds": bool(mean >= float(mean_floor) and p10 >= float(p10_floor)),
        "note": RECALL_NOTE,
    }


# --------------------------------------------------------------------------- #
# the tolerance chain — arithmetic on sealed guard inputs
# --------------------------------------------------------------------------- #
def tolerance_chain(
    *,
    r0240_admissible_max_cluster_rows: float,
    r0240_guarded_max_cluster_rows: float,
    r0240_measured_imbalance: float,
    imbalance_margin: float,
    mean_cluster_rows: float,
) -> dict[str, Any]:
    """`73.8255% -> 51.3984% -> (R0240's own)`, computed rather than transcribed.

    Leg 1 re-derives the guarded max cluster the 50M pricing implied from the
    carried imbalance and the registered margin, so no percentage is copied
    from prose; legs 2 and 3 use the sealed guard inputs of R0238 and R0240.
    """
    margin = _positive(imbalance_margin, label="imbalance margin")
    mean_rows = _positive(mean_cluster_rows, label="mean cluster rows")
    admissible = _positive(
        r0240_admissible_max_cluster_rows, label="R0240 admissible max cluster"
    )
    leg1_guarded = R0237_PREDICTION_IMBALANCE_AT_C400 * margin * mean_rows
    legs = [
        {
            "leg": "carried from the 50M pricing",
            "imbalance": R0237_PREDICTION_IMBALANCE_AT_C400,
            "guarded_max_cluster_rows": leg1_guarded,
            "admissible_max_cluster_rows": admissible,
            "tolerance": tolerance_to_adverse_imbalance(
                admissible_max_cluster_rows=admissible,
                guarded_max_cluster_rows=leg1_guarded,
            ),
        },
        {
            "leg": "R0238 measured worst-of-five at this rung",
            "imbalance": max(R0238_MEASURED_IMBALANCE_BY_SEED.values()),
            "guarded_max_cluster_rows": R0238_GUARDED_MAX_CLUSTER_ROWS,
            "admissible_max_cluster_rows": R0238_ADMISSIBLE_MAX_CLUSTER_ROWS,
            "tolerance": tolerance_to_adverse_imbalance(
                admissible_max_cluster_rows=R0238_ADMISSIBLE_MAX_CLUSTER_ROWS,
                guarded_max_cluster_rows=R0238_GUARDED_MAX_CLUSTER_ROWS,
            ),
        },
        {
            "leg": "R0240 re-measured worst-of-five, from its sealed ladder",
            "imbalance": float(r0240_measured_imbalance),
            "guarded_max_cluster_rows": float(r0240_guarded_max_cluster_rows),
            "admissible_max_cluster_rows": admissible,
            "tolerance": tolerance_to_adverse_imbalance(
                admissible_max_cluster_rows=admissible,
                guarded_max_cluster_rows=float(r0240_guarded_max_cluster_rows),
            ),
        },
    ]
    for entry in legs:
        entry["tolerance_percent"] = float(entry["tolerance"]) * 100.0
    return {
        "definition": (
            "tolerance_to_adverse_imbalance = admissible_max_cluster_rows / "
            "guarded_max_cluster_rows - 1"
        ),
        "imbalance_margin_applied": margin,
        "mean_cluster_rows": mean_rows,
        "legs": legs,
        "consumed_in_the_50m_to_100m_doubling": (
            float(legs[0]["tolerance"]) - float(legs[1]["tolerance"])
        ),
        "moved_on_re_measurement": (
            float(legs[2]["tolerance"]) - float(legs[1]["tolerance"])
        ),
        "note": TOLERANCE_CHAIN_NOTE,
    }


def per_seed_movement(
    *,
    r0240_by_seed: Mapping[int, float],
    r0238_by_seed: Mapping[int, float] = R0238_MEASURED_IMBALANCE_BY_SEED,
    r0237_by_seed: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    """The `c = 400` per-seed movement table, owed since R0237.

    `r0237_by_seed` is the sealed 50M grid when the caller has it; every value
    comes from a hash-bound receipt and none from prose.
    """
    seeds = sorted(
        set(int(s) for s in r0240_by_seed)
        | set(int(s) for s in r0238_by_seed)
        | set(int(s) for s in (r0237_by_seed or {}))
    )
    rows = []
    for seed in seeds:
        at_50m = None if r0237_by_seed is None else r0237_by_seed.get(seed)
        at_100m = r0238_by_seed.get(seed)
        again = r0240_by_seed.get(seed)
        rows.append({
            "seed": int(seed),
            "r0237_50m": None if at_50m is None else float(at_50m),
            "r0238_100m": None if at_100m is None else float(at_100m),
            "r0240_100m_remeasured": None if again is None else float(again),
            "move_50m_to_100m": (
                None if at_50m in (None, 0) or at_100m is None
                else float(at_100m) / float(at_50m) - 1.0
            ),
            "move_on_remeasurement": (
                None if at_100m in (None, 0) or again is None
                else float(again) / float(at_100m) - 1.0
            ),
        })
    ranked_50m = [
        r["seed"] for r in sorted(
            (r for r in rows if r["r0237_50m"] is not None),
            key=lambda r: r["r0237_50m"],
        )
    ]
    ranked_100m = [
        r["seed"] for r in sorted(
            (r for r in rows if r["r0238_100m"] is not None),
            key=lambda r: r["r0238_100m"],
        )
    ]
    ranked_again = [
        r["seed"] for r in sorted(
            (r for r in rows if r["r0240_100m_remeasured"] is not None),
            key=lambda r: r["r0240_100m_remeasured"],
        )
    ]
    return {
        "clusters": REGISTERED_SELECTED_CLUSTERS,
        "spill": SPILL,
        "rows": rows,
        "ranking_ascending_50m": ranked_50m,
        "ranking_ascending_100m": ranked_100m,
        "ranking_ascending_100m_remeasured": ranked_again,
        "ranking_preserved_across_the_doubling": (
            None if not ranked_50m or not ranked_100m
            else bool(ranked_50m == ranked_100m)
        ),
        "ranking_preserved_on_remeasurement": (
            None if not ranked_100m or not ranked_again
            else bool(ranked_100m == ranked_again)
        ),
        "reading": (
            "review-0238-01/F5 found the seed ranking INVERTED across the "
            "50M -> 100M doubling, so a seed carries no information about its "
            "own imbalance at the next N. R0240 re-measured all five on the "
            "same bytes at the same N and reproduced them closely, which "
            "locates the instability in CHANGING N rather than in the "
            "measurement. This table publishes both movements side by side, "
            "which is the thing R0237, R0238 and R0240 each owed and none paid."
        ),
    }


__all__ = [
    "BUILD_ROUND_ID",
    "CALIBRATION_UNITS",
    "CPU_ARITHMETIC_NOTE",
    "CROSS_CHECKED_FIELDS",
    "DEBTS_NOTE",
    "DEGREE_SEMANTICS_NOTE",
    "DENSITY_DECILES",
    "DERIVATION_CAPABILITY",
    "DIMENSION",
    "DUPLICATE_FAMILY_KTH_COSINE",
    "FP32_TIE_NOISE_FLOOR",
    "GPU_HOURS_CAP",
    "GPU_HOURS_CAP_NOTE",
    "GRAPH_K",
    "GRAPH_UNDER_TEST",
    "HUBNESS_NOTE",
    "INHERITANCE_NOTE",
    "INHERITED_BUILD_RECEIPT",
    "INHERITED_GRAPH_COS",
    "INHERITED_GRAPH_IDS",
    "INHERITED_LADDER_ATTESTATION",
    "INHERITED_LADDER_RECEIPT",
    "MAX_ZERO_DEGREE_ROWS",
    "NO_SYMMETRISATION_NOTE",
    "PROBE_GATHER_BLOCK",
    "EARLY_PERSIST_NOTE",
    "RECALL_CAPABILITY",
    "RECALL_DEADLINE_S",
    "RECALL_EARLY_FILE",
    "RECALL_EARLY_SCHEMA",
    "RECALL_FILE",
    "RECALL_MEAN_FLOOR",
    "RECALL_NOTE",
    "RECALL_P10_FLOOR",
    "RECALL_POPULATION",
    "RECALL_RESERVE_S",
    "RECALL_SCHEMA",
    "REGISTERED_GRAPH_ARRAY_BYTES",
    "REGISTERED_GRAPH_COS_SHA256",
    "REGISTERED_GRAPH_IDS_SHA256",
    "REGISTERED_LADDER_RECEIPT_SHA256",
    "REGISTERED_SELECTED_CELL",
    "REGISTERED_SELECTED_CLUSTERS",
    "REVIEWER_RAW_IN_DEGREE_MAX",
    "REVIEWER_RAW_IN_DEGREE_MEAN",
    "REVIEWER_RAW_OUT_DEGREE_MIN",
    "REVIEWER_RAW_OUT_DEGREE_ZERO_ROWS",
    "REVIEWER_RAW_ZERO_IN_DEGREE_ROWS",
    "REVIEWER_SOURCE",
    "ROUND_ID",
    "ROWS",
    "Round0241Error",
    "SAFETY_NOTE",
    "SCOPE_NOTE",
    "SPILL",
    "STRUCTURAL_POPULATION",
    "SUBSTRATE_ROUND_ID",
    "SYMMETRISED_DEGREE_BOUND_NOTE",
    "SYMMETRISED_IDENTITY_NOTE",
    "StageGuard",
    "TOLERANCE_CHAIN_NOTE",
    "TRIPWIRE_BLOCK",
    "TRIPWIRE_CAPABILITY",
    "TRIPWIRE_DEADLINE_S",
    "TRIPWIRE_FILE",
    "TRIPWIRE_NOTE",
    "TRIPWIRE_SCHEMA",
    "TRUTH_PROBE_ROWS",
    "TRUTH_PROBE_SEED",
    "VACUITY_NOTE",
    "WALL_GUARD_MARGIN",
    "WALL_GUARD_NOTE",
    "ZERO_RECALL_FORENSIC_NOTE",
    "cross_check_structural",
    "per_seed_movement",
    "probe_power_statement",
    "recall_verdict",
    "reconcile_in_degree",
    "region_detection_power",
    "sampling_uncertainty",
    "smallest_detectable_region",
    "tolerance_chain",
    "tripwire_verdict",
]
