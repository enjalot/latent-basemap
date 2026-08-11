"""Execute R0244 — the four prerequisites review-0243-01 §8 put before a map.

Four nodes, none of which trains anything and none of which claims a map
result:

* `did_registration` — seals R0228's displacement DiD **inference rule** before
  any displacement exists, with its resolution table computed rather than
  claimed, and seals the row populations the rule names (they are computable
  from sealed vectors today, which turns the training round into a join). It
  computes NO displacement: R0228's statistic consumes trained `(n, 2)` map
  coordinates and this round trains no map. It also consumes R0238's per-row
  reachability vector `strict-c400.f64.npy` through the imported
  `loss_decomposition`, which is the discharge review-0243-01 §7 asked for —
  that vector was the round's most load-bearing input and was invisible in the
  Inputs table.
* `watchdog_100000k` — fixes the safety gap review-0243-01 §6 found and proves
  the fix. It runs a positive control that plants an over-budget allocation and
  fails closed unless the guard observes it, then re-runs R0243's fuzzy
  symmetrisation stage under the threaded instrument to measure the stage's
  TRUE anonymous peak against the `12,058,918,912` B the R0243 receipt carries.
* `sampler_100000k` — loads R0243's `30 GB` edge list as a sampling
  distribution over `875,131,479.5054033` total weight, draws from it with the
  two-level scheme a `100M` trainer must use, and checks the draw against the
  distribution with a mis-sampler as the positive control.
* `text_100000k` — reads the actual documents behind a sample of cluster
  `168`'s rows and their tie-forgiven substitutes, verifies every text against
  its substrate row by re-embedding, and publishes the cosine between the
  missed true neighbour and its substitute — the quantity tie tolerance does
  NOT certify.

Every registered check is IMPORTED, never re-typed: `verify_inheritance`,
`_readonly_memmap`, `_blocked_descending_sort`, `_fuzzy_symmetrise_blocked`,
`_check_runner_abort`, `_memmap_attestation`, R0242's `io_counters`,
`json_scrub`, `loss_decomposition`, R0228's `density_matched_control` and
R0229's `exact_displacement_permutation` / `holm_bonferroni`. R0242's
`_HostWatchdog` is SUBCLASSED, not edited.

No node in this module starts a child process, creates a CUDA context, hands
cuVS anything, or contains a signalling construct of any kind. The text node
re-embeds on the CPU deliberately, so that the round holds the GPU lease and
never uses the card.
"""
from __future__ import annotations

import gc
import glob
import math
import os
import shutil
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap import round0113_prompt_contrast as prompt_contract
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0216_minilm_2m_substrate import EXCLUDED_SHARDS
from basemap.round0227_low_c_contract import TIE_TOLERANCE
from basemap.round0238_rung5 import (
    COMPOSITION,
    DIMENSION,
    GRAPH_K,
    TRUTH_PROBE_ROWS,
    json_safe,
)
from basemap.round0241_qualify import GPU_HOURS_CAP_NOTE
from basemap.round0242_locality import (
    io_counters,
    json_scrub,
    loss_decomposition,
)
from basemap.round0244_guard import (
    FUZZY_STAGE_ANON_BUDGET_BYTES,
    GPU_HOURS_CAP,
    ROUND_ID,
    ROWS,
    Round0244Error,
    ThreadedHostWatchdog,
    WATCHDOG_NOTE,
    WATCHDOG_SAMPLE_INTERVAL_S,
    boundary_only_understatement,
    run_watchdog_positive_control,
)
from basemap.round0244_prereq import (
    NEAR_DUPLICATE_CATEGORIES,
    R0243_DIRECTED_EDGES,
    R0243_ENTRIES_AT_OR_ABOVE_ONE,
    R0243_TOTAL_WEIGHT,
    R0243_WEIGHT_MAX,
    R0243_WEIGHT_MIN,
    SAMPLER_BLOCK_EDGES,
    SAMPLER_DRAWS,
    SAMPLER_EPOCHS,
    SAMPLER_MAX_ANONYMOUS_BYTES,
    sampler_max_anonymous_bytes,
    SAMPLER_MIN_DRAWS_PER_S,
    SAMPLER_NOTE,
    SAMPLER_SEED,
    SAMPLER_WEIGHT_BINS,
    TEXT_BINDING_COSINE_FLOOR,
    TEXT_NOTE,
    TEXT_SAMPLE_PAIRS,
    TEXT_SAMPLE_SEED,
    classify_text_pair,
    did_populations,
    did_registration,
    did_requirement,
    excerpt,
    sampling_fidelity,
    two_level_weight_sample,
    uniform_sample_control,
    weight_block_profile,
)
from experiments.round0238_nodes import (
    _blocked_descending_sort,
    _check_runner_abort,
    _fuzzy_symmetrise_blocked,
)
from experiments.round0241_nodes import _readonly_memmap, verify_inheritance
from experiments.round0242_nodes import _memmap_attestation

DID_ACTION = "did_registration"
WATCHDOG_ACTION = "watchdog_100000k"
SAMPLER_ACTION = "sampler_100000k"
TEXT_ACTION = "text_100000k"

DID_CAPABILITY = "minilm-mixed-100000k-displacement-did-registration-v1"
WATCHDOG_CAPABILITY = "round0244-threaded-host-watchdog-v1"
SAMPLER_CAPABILITY = "minilm-mixed-100000k-k15-fuzzy-edge-sampler-v1"
TEXT_CAPABILITY = "minilm-mixed-100000k-cluster-168-text-forensics-v1"

DID_FILE = "did-registration.json"
WATCHDOG_FILE = "host-watchdog.json"
SAMPLER_FILE = "edge-sampler.json"
TEXT_FILE = "cluster-168-text.json"

DID_SCHEMA = "round0244-displacement-did-registration-v1"
WATCHDOG_SCHEMA = "round0244-threaded-host-watchdog-and-true-peak-v1"
SAMPLER_SCHEMA = "round0244-fuzzy-edge-sampling-distribution-v1"
TEXT_SCHEMA = "round0244-cluster-168-near-duplicate-forensics-v1"

#: R0243's sealed symmetrisation constants, bound so a re-run that does not
#: reproduce the stage cannot be published as a measurement OF that stage.
R0243_SIGMAS_MEAN = 0.05017701756424675
R0243_RHOS_MEAN = 0.2587219380916214
R0243_RECEIPT_ANONYMOUS_PEAK_BYTES = 12_058_918_912
R0243_RECEIPT_POLLS = 6

CHUNK_ROOT = "/data/chunks"
EMBEDDING_SUFFIX = "-all-MiniLM-L6-v2"
EMBEDDING_ROOT = "/data/embeddings"
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLUSTER_UNDER_TEST = 168
LABEL_BLOCK = 10_000_000

SCOPE_NOTE = (
    "R0244 trains nothing, registers no gate, adopts nothing and makes no map "
    "quality claim. It discharges the four instrument-side prerequisites "
    "review-0243-01 section 8 put in front of the first ladder map, and it "
    "registers - without running - the one instrument that can settle whether "
    "the residual displaces rows."
)
SAFETY_NOTE = (
    "every bulk input is a read-only np.memmap, re-measured on the live "
    "objects at seal time; nothing is handed to cuVS; no child process is "
    "started; no signal is delivered on any path; the host guard is polled by "
    "a sampling THREAD and enforces in band through the cooperative abort "
    "flag, never with a signal."
)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _bound_path(job: Mapping[str, Any], key: str, *, label: str) -> str:
    entry = job.get(key)
    if not isinstance(entry, Mapping):
        raise Round0244Error(f"R0244 job does not bind {label} ({key})")
    path = str(entry.get("canonical_path") or "")
    if not path or not os.path.exists(path):
        raise Round0244Error(f"R0244 {label} is absent at {path!r}")
    declared = int(entry.get("bytes", -1))
    actual = os.path.getsize(path)
    if declared >= 0 and declared != actual:
        raise Round0244Error(
            f"R0244 {label} is {actual} bytes, the manifest declared {declared}"
        )
    return path


def _receipt_envelope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "round_id": ROUND_ID,
        "release_sha": str(manifest["release_sha"]),
        "rows": ROWS,
        "k": GRAPH_K,
        "gpu_hours_cap": GPU_HOURS_CAP,
        "gpu_hours_cap_note": GPU_HOURS_CAP_NOTE,
        "scope_note": SCOPE_NOTE,
        "safety_note": SAFETY_NOTE,
        "training_performed": False,
        "gate_registered": False,
        "adoption_claimed": False,
        "map_quality_claimed": False,
        "displacement_measured": False,
        "cuvs_calls": 0,
        "cuda_context_created": False,
        "child_processes_launched": 0,
        "signal_delivered": False,
    }


def _seal(output: str, name: str, body: Mapping[str, Any]) -> None:
    atomic_write_new_json(
        os.path.join(output, name),
        prompt_contract.seal(json_safe(json_scrub(dict(body)))),
        immutable=True,
    )


# --------------------------------------------------------------------------- #
# node 1 — the DiD registration (Part B), and the R0238 reachability discharge
# --------------------------------------------------------------------------- #
def run_did(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0244Error("R0244 handler received another queue")
    started = time.monotonic()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0244 DiD")

    strict_recall = _readonly_memmap(
        _bound_path(job, "r0242_probe_strict_recall", label="strict recall"),
        label="R0244 strict recall", shape=(TRUTH_PROBE_ROWS,),
    )
    reachability = _readonly_memmap(
        _bound_path(job, "r0238_strict_reachability", label="reachability"),
        label="R0244 R0238 reachability", shape=(TRUTH_PROBE_ROWS,),
    )
    sealed_builder = _readonly_memmap(
        _bound_path(
            job, "r0242_probe_builder_missing_edges", label="builder missing"
        ),
        label="R0244 sealed builder-missing", shape=(TRUTH_PROBE_ROWS,),
    )
    tie_builder = _readonly_memmap(
        _bound_path(
            job, "r0243_probe_tie_aware_builder_missing_edges",
            label="tie-aware builder missing",
        ),
        label="R0244 tie-aware builder-missing", shape=(TRUTH_PROBE_ROWS,),
    )
    truth_cos = _readonly_memmap(
        _bound_path(job, "truth_cos", label="truth cosines"),
        label="R0244 truth cosines", shape=(TRUTH_PROBE_ROWS, GRAPH_K),
    )

    #: The discharge. R0238's per-row reachability vector is the input every
    #: decomposition in R0242 and R0243 consumed and that no Inputs table has
    #: ever shown. It is bound here as a first-class artifact AND used: the
    #: builder-missing vector is recomputed from it through the imported
    #: `loss_decomposition` and must reproduce R0242's sealed bytes exactly.
    decomposition = loss_decomposition(
        strict=np.asarray(strict_recall, dtype=np.float64),
        reachability=np.asarray(reachability, dtype=np.float64),
        k=GRAPH_K,
    )
    vectors = decomposition.pop("vectors")
    recomputed = np.asarray(vectors["builder_lost"], dtype=np.int64)
    sealed = np.asarray(sealed_builder, dtype=np.int64)
    disagreements = int(np.count_nonzero(recomputed != sealed))
    reachability_discharge = {
        "input": "R0238 strict-c400.f64.npy, the per-row reachability vector",
        "declared_in_the_inputs_table": True,
        "recomputed_builder_missing_total": int(recomputed.sum()),
        "sealed_builder_missing_total": int(sealed.sum()),
        "rows_disagreeing": disagreements,
        "agrees": bool(disagreements == 0),
        "why_this_matters": (
            "review-0243-01 section 7: this vector is the single most "
            "load-bearing input to R0242 and R0243 and appeared in no Inputs "
            "table - only the 7,937-byte reachability.json that references it. "
            "Declaring it is half the discharge; consuming it so a wrong file "
            "would fail the round is the other half."
        ),
    }
    if disagreements:
        raise Round0244Error(
            "R0244 STOP: R0238's reachability vector does not reproduce "
            f"R0242's sealed builder-missing vector on {disagreements} rows"
        )

    populations = did_populations(
        strict_builder_missing=sealed,
        tie_aware_builder_missing=np.asarray(tie_builder, dtype=np.int64),
        kth_cosine=np.asarray(truth_cos[:, GRAPH_K - 1], dtype=np.float64),
    )
    vector_dir = create_fresh_directory(
        os.path.join(output, "vectors"), label="R0244 DiD vectors"
    )
    saved = {
        "genuine_lost_rows": atomic_save_new_npy(
            os.path.join(vector_dir, "did-genuine-lost-rows.i64.npy"),
            np.asarray(populations["genuine"]["lost_sample"], dtype=np.int64),
            immutable=True,
        ),
        "genuine_control_rows": atomic_save_new_npy(
            os.path.join(vector_dir, "did-genuine-control-rows.i64.npy"),
            np.asarray(
                populations["genuine"]["control_sample"], dtype=np.int64
            ),
            immutable=True,
        ),
        "tie_forgiven_lost_rows": atomic_save_new_npy(
            os.path.join(vector_dir, "did-tie-forgiven-lost-rows.i64.npy"),
            np.asarray(
                populations["tie_forgiven"]["lost_sample"], dtype=np.int64
            ),
            immutable=True,
        ),
        "tie_forgiven_control_rows": atomic_save_new_npy(
            os.path.join(vector_dir, "did-tie-forgiven-control-rows.i64.npy"),
            np.asarray(
                populations["tie_forgiven"]["control_sample"], dtype=np.int64
            ),
            immutable=True,
        ),
        "placebo_a_rows": atomic_save_new_npy(
            os.path.join(vector_dir, "did-placebo-a-rows.i64.npy"),
            np.asarray(populations["placebo_a"], dtype=np.int64),
            immutable=True,
        ),
        "placebo_b_rows": atomic_save_new_npy(
            os.path.join(vector_dir, "did-placebo-b-rows.i64.npy"),
            np.asarray(populations["placebo_b"], dtype=np.int64),
            immutable=True,
        ),
    }
    registration = did_registration()
    requirement = did_requirement()

    body = dict(_receipt_envelope(manifest))
    body.update({
        "schema": DID_SCHEMA,
        "capabilities": [DID_CAPABILITY],
        "registration": registration,
        "requirement": requirement,
        "populations": {
            key: value for key, value in populations.items()
            if key not in ("genuine", "tie_forgiven", "placebo_a", "placebo_b")
        },
        "population_sizes": {
            "genuine_lost_sample": int(
                np.asarray(populations["genuine"]["lost_sample"]).size
            ),
            "genuine_control_sample": int(
                np.asarray(populations["genuine"]["control_sample"]).size
            ),
            "tie_forgiven_lost_sample": int(
                np.asarray(populations["tie_forgiven"]["lost_sample"]).size
            ),
            "tie_forgiven_control_sample": int(
                np.asarray(populations["tie_forgiven"]["control_sample"]).size
            ),
            "placebo_a": int(np.asarray(populations["placebo_a"]).size),
            "placebo_b": int(np.asarray(populations["placebo_b"]).size),
        },
        "genuine_decile_counts_lost": populations["genuine"][
            "decile_counts_lost"
        ],
        "genuine_decile_counts_control": populations["genuine"][
            "decile_counts_control"
        ],
        "tie_forgiven_decile_counts_lost": populations["tie_forgiven"][
            "decile_counts_lost"
        ],
        "tie_forgiven_decile_counts_control": populations["tie_forgiven"][
            "decile_counts_control"
        ],
        "r0238_reachability_discharge": reachability_discharge,
        "strict_decomposition": decomposition,
        "sealed_vectors": saved,
        "bulk_input_memmap_attestation": _memmap_attestation({
            "strict_recall": strict_recall,
            "reachability": reachability,
            "sealed_builder_missing": sealed_builder,
            "tie_aware_builder_missing": tie_builder,
            "truth_cosines": truth_cos,
        }),
        "performance": {"total_wall_s": time.monotonic() - started},
    })
    _seal(output, DID_FILE, body)


# --------------------------------------------------------------------------- #
# node 2 — the watchdog fix, its positive control, and the true peak
# --------------------------------------------------------------------------- #
def run_watchdog(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0244Error("R0244 handler received another queue")
    started = time.monotonic()
    inheritance = verify_inheritance(job)
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0244 watchdog"
    )

    #: 1. Prove the guard fires, before trusting it with a 42 GB stage.
    control = run_watchdog_positive_control(
        flag_path=os.path.join(output, "watchdog-control.abort")
    )

    #: 2. Re-run R0243's stage under the working instrument.
    watchdog = ThreadedHostWatchdog(
        anonymous_budget_bytes=FUZZY_STAGE_ANON_BUDGET_BYTES,
        abort_flag_path=os.environ.get("ROUNDRUN_ABORT_FLAG"),
        label="R0244 fuzzy symmetrisation re-measurement",
    )
    ids = _readonly_memmap(
        str(inheritance["graph"]["ids"]["canonical_path"]),
        label="R0244 graph ids", shape=(ROWS, GRAPH_K),
    )
    cosines = _readonly_memmap(
        str(inheritance["graph"]["cosines"]["canonical_path"]),
        label="R0244 builder cosines", shape=(ROWS, GRAPH_K),
    )
    io_before = io_counters()
    watchdog.start()
    try:
        sort_started = time.monotonic()
        ids_sorted, cos_sorted = _blocked_descending_sort(
            ids=ids, cosines=cosines, rows=ROWS
        )
        sort_wall = time.monotonic() - sort_started
        watchdog.poll("after the descending sort")

        dists = np.maximum((1.0 - cos_sorted).astype(np.float32), 0.0)
        if not np.isfinite(dists).all():
            raise Round0244Error("R0244 candidate distances are not finite")
        del cos_sorted
        gc.collect()
        watchdog.poll("after the distance transform")

        import umap.umap_ as umap_api

        fuzzy_started = time.monotonic()
        fuzzy = _fuzzy_symmetrise_blocked(
            knn_indices=ids_sorted, knn_dists=dists, rows=ROWS, k=GRAPH_K,
            umap_api=umap_api, out_dir=output,
        )
        fuzzy_wall = time.monotonic() - fuzzy_started
        del dists, ids_sorted
        gc.collect()
        watchdog.poll("after the fuzzy symmetrisation")

        directed_edges = int(fuzzy["directed_edges"])
        sigmas_mean = float(fuzzy["sigmas_mean"])
        rhos_mean = float(fuzzy["rhos_mean"])
        scratch = str(fuzzy["scratch"])
        for key in ("src", "dst", "weights"):
            fuzzy.pop(key, None)
        gc.collect()
        shutil.rmtree(scratch, ignore_errors=True)
    finally:
        watchdog.stop()
    io_after = io_counters()

    reproduction = {
        "directed_edges": directed_edges,
        "r0243_directed_edges": R0243_DIRECTED_EDGES,
        "directed_edges_agree": bool(directed_edges == R0243_DIRECTED_EDGES),
        "sigmas_mean": sigmas_mean,
        "r0243_sigmas_mean": R0243_SIGMAS_MEAN,
        "rhos_mean": rhos_mean,
        "r0243_rhos_mean": R0243_RHOS_MEAN,
        "sigmas_mean_agree": bool(sigmas_mean == R0243_SIGMAS_MEAN),
        "rhos_mean_agree": bool(rhos_mean == R0243_RHOS_MEAN),
        "why": (
            "a peak measured on a stage that is not R0243's stage measures "
            "nothing about R0243. The edge count and both UMAP moments are "
            "bit-level identities of the same computation on the same bytes."
        ),
    }
    if not (
        reproduction["directed_edges_agree"]
        and reproduction["sigmas_mean_agree"]
        and reproduction["rhos_mean_agree"]
    ):
        raise Round0244Error(
            "R0244 STOP: the re-run did not reproduce R0243's symmetrisation "
            f"({reproduction}); its anonymous peak is not R0243's peak"
        )

    guard = watchdog.receipt()
    comparison = boundary_only_understatement(guard)
    comparison.update({
        "r0243_receipt_anonymous_peak_bytes": (
            R0243_RECEIPT_ANONYMOUS_PEAK_BYTES
        ),
        "r0243_receipt_polls": R0243_RECEIPT_POLLS,
        "true_peak_over_r0243_receipt_multiple": (
            guard["thread_peak_anonymous_bytes"]
            / float(R0243_RECEIPT_ANONYMOUS_PEAK_BYTES)
        ),
        "true_peak_over_r0243_receipt_bytes": (
            int(guard["thread_peak_anonymous_bytes"])
            - R0243_RECEIPT_ANONYMOUS_PEAK_BYTES
        ),
    })

    body = dict(_receipt_envelope(manifest))
    body.update({
        "schema": WATCHDOG_SCHEMA,
        "capabilities": [WATCHDOG_CAPABILITY],
        "watchdog_note": WATCHDOG_NOTE,
        "sample_interval_s": WATCHDOG_SAMPLE_INTERVAL_S,
        "positive_control": control,
        "host_watchdog": guard,
        "true_peak_against_the_r0243_receipt": comparison,
        "stage_reproduction": reproduction,
        "stage_walls": {
            "descending_sort_s": sort_wall,
            "fuzzy_symmetrisation_s": fuzzy_wall,
        },
        "fuzzy": fuzzy,
        "io": {
            "read_bytes": int(io_after["read_bytes"]) - int(
                io_before["read_bytes"]
            ),
            "write_bytes": int(io_after["write_bytes"]) - int(
                io_before["write_bytes"]
            ),
        },
        "edges_republished": False,
        "edges_note": (
            "the symmetrised edge list is NOT rewritten: R0243 sealed it and "
            "review-0243-01 asked the next rung to prefer a recorded hash to a "
            "6 GB copy. The scratch this stage needs is removed before the "
            "receipt is sealed."
        ),
        "bulk_input_memmap_attestation": _memmap_attestation({
            "graph_ids": ids, "builder_cosines": cosines,
        }),
        "abort_policy": (
            "the sampling thread NEVER raises into the main thread and never "
            "signals. On a trip it writes the runner's cooperative abort flag, "
            "which every reviewed stripe loop already polls once per stripe, "
            "and poll() raises in band at the next call site."
        ),
        "performance": {"total_wall_s": time.monotonic() - started},
    })
    _seal(output, WATCHDOG_FILE, body)


# --------------------------------------------------------------------------- #
# node 3 — the edge list as a sampling distribution
# --------------------------------------------------------------------------- #
def run_sampler(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0244Error("R0244 handler received another queue")
    started = time.monotonic()
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0244 sampler"
    )
    header_path = _bound_path(job, "edges_header", label="edge header")
    header = np.load(header_path, allow_pickle=False)
    declared_edges = int(header["directed_edges"])
    header_check = {
        "n_nodes": int(header["n_nodes"]),
        "k": int(header["k"]),
        "directed_edges": declared_edges,
        "agrees_with_r0243": bool(
            int(header["n_nodes"]) == ROWS
            and int(header["k"]) == GRAPH_K
            and declared_edges == R0243_DIRECTED_EDGES
        ),
    }
    if not header_check["agrees_with_r0243"]:
        raise Round0244Error(
            f"R0244 edge header is not R0243's graph: {header_check}"
        )

    src = _readonly_memmap(
        _bound_path(job, "edges_src", label="edge sources"),
        label="R0244 edge sources", shape=(declared_edges,),
    )
    dst = _readonly_memmap(
        _bound_path(job, "edges_dst", label="edge targets"),
        label="R0244 edge targets", shape=(declared_edges,),
    )
    wts = _readonly_memmap(
        _bound_path(job, "edges_wts", label="edge weights"),
        label="R0244 edge weights", shape=(declared_edges,),
    )

    watchdog = ThreadedHostWatchdog(
        anonymous_budget_bytes=16 * (1 << 30),
        abort_flag_path=os.environ.get("ROUNDRUN_ABORT_FLAG"),
        label="R0244 edge sampler",
    )
    io_before = io_counters()
    watchdog.start()
    try:
        profile_started = time.monotonic()
        profile = weight_block_profile(
            wts, block=SAMPLER_BLOCK_EDGES, bins=SAMPLER_WEIGHT_BINS,
            epochs=SAMPLER_EPOCHS, abort_check=_check_runner_abort,
        )
        profile_wall = time.monotonic() - profile_started
        watchdog.poll("after the streaming weight profile")

        binding = {
            "total_weight": profile["total_weight"],
            "r0243_total_weight": R0243_TOTAL_WEIGHT,
            "total_weight_relative_gap": abs(
                profile["total_weight"] - R0243_TOTAL_WEIGHT
            ) / R0243_TOTAL_WEIGHT,
            "max_weight": profile["max_weight"],
            "r0243_max_weight": R0243_WEIGHT_MAX,
            "min_weight": profile["min_weight"],
            "r0243_min_weight": R0243_WEIGHT_MIN,
            "entries_at_or_above_one": profile["entries_at_or_above_one"],
            "r0243_entries_at_or_above_one": R0243_ENTRIES_AT_OR_ABOVE_ONE,
            "non_finite_entries": profile["non_finite_entries"],
            "non_positive_entries": profile["non_positive_entries"],
        }
        binding["is_r0243s_distribution"] = bool(
            binding["total_weight_relative_gap"] < 1e-12
            and profile["max_weight"] == R0243_WEIGHT_MAX
            and profile["min_weight"] == R0243_WEIGHT_MIN
            and profile["entries_at_or_above_one"]
            == R0243_ENTRIES_AT_OR_ABOVE_ONE
            and profile["non_finite_entries"] == 0
            and profile["non_positive_entries"] == 0
        )
        if not binding["is_r0243s_distribution"]:
            raise Round0244Error(
                f"R0244 loaded weights are not R0243's distribution: {binding}"
            )

        draw_started = time.monotonic()
        sample = two_level_weight_sample(
            wts, profile=profile, draws=int(job.get("draws") or SAMPLER_DRAWS),
            seed=SAMPLER_SEED, abort_check=_check_runner_abort,
        )
        draw_wall = time.monotonic() - draw_started
        watchdog.poll("after the two-level draw")

        fidelity = sampling_fidelity(profile=profile, sample=sample)
        control = uniform_sample_control(
            wts, profile=profile,
            draws=min(2_000_000, int(sample["draws"])),
        )
        watchdog.poll("after the fidelity check and its mis-sampler control")

        #: The other half of the trainer's contract: gather the endpoints of
        #: the drawn edges out of two 10 GB memmaps.
        gather_n = min(5_000_000, int(sample["draws"]))
        index = np.sort(
            np.asarray(sample["edge_index"][:gather_n], dtype=np.int64)
        )
        gather_started = time.monotonic()
        gathered_src = np.asarray(src[index], dtype=np.int64)
        gathered_dst = np.asarray(dst[index], dtype=np.int64)
        gather_wall = time.monotonic() - gather_started
        in_range = bool(
            gathered_src.min() >= 0 and gathered_src.max() < ROWS
            and gathered_dst.min() >= 0 and gathered_dst.max() < ROWS
        )
        self_loops = int(np.count_nonzero(gathered_src == gathered_dst))
        distinct_sources = int(np.unique(gathered_src).size)
        del gathered_src, gathered_dst, index
        gc.collect()
        watchdog.poll("after the endpoint gather")
        draws = int(sample["draws"])
        distinct_edges = int(sample["distinct_edges_drawn"])
        del sample
        gc.collect()
    finally:
        watchdog.stop()
    io_after = io_counters()

    guard = watchdog.receipt()
    peak = int(guard["thread_peak_anonymous_bytes"])
    draws_per_s = draws / draw_wall if draw_wall > 0 else math.inf
    verdict_arms = {
        "header_and_moments_bind_to_r0243": binding["is_r0243s_distribution"],
        "draw_matches_the_distribution": bool(fidelity["holds"]),
        "mis_sampler_is_rejected": bool(control["rejected"]),
        "endpoints_in_range": in_range,
        "no_self_loops": bool(self_loops == 0),
        "anonymous_peak_within_budget": bool(
            peak <= sampler_max_anonymous_bytes()
        ),
        "throughput_above_floor": bool(draws_per_s >= SAMPLER_MIN_DRAWS_PER_S),
    }

    body = dict(_receipt_envelope(manifest))
    body.update({
        "schema": SAMPLER_SCHEMA,
        "capabilities": [SAMPLER_CAPABILITY],
        "sampler_note": SAMPLER_NOTE,
        "header": header_check,
        "binding_to_r0243": binding,
        "weight_profile": {
            key: value for key, value in profile.items()
            if key not in ("block_sums", "bin_mass", "bin_counts")
        },
        "weight_bin_mass": [
            float(value) for value in np.asarray(profile["bin_mass"])
        ],
        "weight_bin_counts": [
            int(value) for value in np.asarray(profile["bin_counts"])
        ],
        "draw": {
            "draws": draws,
            "seed": SAMPLER_SEED,
            "distinct_edges_drawn": distinct_edges,
            "distinct_fraction": distinct_edges / float(draws),
            "wall_s": draw_wall,
            "draws_per_s": draws_per_s,
            "min_draws_per_s": SAMPLER_MIN_DRAWS_PER_S,
        },
        "fidelity": fidelity,
        "mis_sampler_control": control,
        "endpoint_gather": {
            "edges": gather_n,
            "wall_s": gather_wall,
            "edges_per_s": gather_n / gather_wall if gather_wall > 0 else None,
            "in_range": in_range,
            "self_loops": self_loops,
            "distinct_sources": distinct_sources,
        },
        "walls": {
            "profile_s": profile_wall,
            "draw_s": draw_wall,
            "gather_s": gather_wall,
        },
        "host_watchdog": guard,
        "anonymous_peak_bytes": peak,
        "anonymous_budget_bytes": SAMPLER_MAX_ANONYMOUS_BYTES,
        "io": {
            "read_bytes": int(io_after["read_bytes"]) - int(
                io_before["read_bytes"]
            ),
            "write_bytes": int(io_after["write_bytes"]) - int(
                io_before["write_bytes"]
            ),
        },
        "bulk_input_memmap_attestation": _memmap_attestation({
            "edges_src": src, "edges_dst": dst, "edges_wts": wts,
        }),
        "verdict_arms": verdict_arms,
        "loads_as_a_sampling_distribution": all(verdict_arms.values()),
        "performance": {"total_wall_s": time.monotonic() - started},
    })
    _seal(output, SAMPLER_FILE, body)
    if not all(verdict_arms.values()):
        raise Round0244Error(
            "R0244 STOP: the symmetrised edge list did not demonstrate as a "
            f"sampling distribution: {verdict_arms}"
        )


# --------------------------------------------------------------------------- #
# node 4 — cluster 168's text
# --------------------------------------------------------------------------- #
def _corpus_shards(corpus: str) -> list[str]:
    """The shard paths in the order the substrate draw indexed them.

    This is `experiments.round0233_nodes._shards`'s ordering — sorted, minus
    `.tmp.npy`, minus R0216's excluded fineweb shard 37 — reproduced here for
    paths only, because the row counts it also resolves require reading every
    shard header and this node needs none of them.
    """
    out: list[str] = []
    pattern = os.path.join(EMBEDDING_ROOT, corpus, "train", "*.npy")
    for path in sorted(glob.glob(pattern)):
        if path.endswith(".tmp.npy"):
            continue
        if os.path.relpath(path, EMBEDDING_ROOT) in EXCLUDED_SHARDS:
            continue
        out.append(path)
    if not out:
        raise Round0244Error(f"R0244 found no embedding shards for {corpus}")
    return out


def _chunk_parquet(corpus: str, shard_path: str, *, shard_index: int) -> str:
    """The chunk parquet holding the text behind an embedding shard.

    Three of the four corpora name their parquet exactly as their `.npy`
    (`data-00000-of-01987`), and the pile's parquet directory is a SUPERSET of
    its embedded shards, so name matching is the only correct rule there.
    `starcoderdata-code-chunked-120` uses a different scheme entirely
    (`000_000500000.parquet` against `data-00000-of-00020.npy`) with the same
    count and the same order, so the fallback is positional.

    A positional fallback is an assumption, so it is not left as one: the
    caller checks the parquet's row count against the shard's complete-row
    count, and every text this node publishes is separately re-embedded and
    required to reproduce its substrate row.
    """
    if not corpus.endswith(EMBEDDING_SUFFIX):
        raise Round0244Error(f"R0244 cannot map {corpus} to a chunk corpus")
    chunk_corpus = corpus[: -len(EMBEDDING_SUFFIX)]
    name = os.path.basename(shard_path)
    if not name.endswith(".npy"):
        raise Round0244Error(f"R0244 shard {name} is not a .npy")
    directory = os.path.join(CHUNK_ROOT, chunk_corpus, "train")
    by_name = os.path.join(directory, name[: -len(".npy")] + ".parquet")
    if os.path.exists(by_name):
        return by_name
    listing = sorted(glob.glob(os.path.join(directory, "*.parquet")))
    shards = _corpus_shards(corpus)
    if len(listing) == len(shards) and 0 <= int(shard_index) < len(listing):
        return listing[int(shard_index)]
    raise Round0244Error(
        f"R0244 cannot resolve a chunk parquet for {name}: no "
        f"{by_name!r}, and {directory} holds {len(listing)} parquets against "
        f"{len(shards)} embedded shards, so a positional fallback would be a "
        "guess"
    )


def _shard_rows(shard_path: str) -> int:
    """Complete `384`-float32 rows in an embedding shard, header or not."""
    with open(shard_path, "rb") as handle:
        real_npy = handle.read(6) == b"\x93NUMPY"
    if real_npy:
        return int(np.load(shard_path, mmap_mode="r").shape[0])
    return int(os.path.getsize(shard_path) // (DIMENSION * 4))


def _read_chunk_texts(
    path: str, rows: list[int], *, expect_rows: int
) -> dict[int, dict[str, Any]]:
    """Read only the row groups the requested rows fall in.

    Fail-closed on the shard-to-parquet correspondence: a parquet whose row
    count differs from its embedding shard's is not the text behind that
    shard, whatever its name says.
    """
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(path)
    if int(handle.metadata.num_rows) != int(expect_rows):
        raise Round0244Error(
            f"R0244 chunk parquet {path} has {handle.metadata.num_rows} rows "
            f"against {expect_rows} in its embedding shard; the row indices in "
            "R0238's provenance do not address this file"
        )
    names = set(handle.schema_arrow.names)
    columns = ["chunk_text"] + [
        name for name in ("chunk_index", "id", "url") if name in names
    ]
    offsets: list[int] = [0]
    for index in range(handle.metadata.num_row_groups):
        offsets.append(offsets[-1] + handle.metadata.row_group(index).num_rows)
    wanted: dict[int, list[int]] = {}
    for row in rows:
        group = int(np.searchsorted(offsets, row, side="right") - 1)
        if group < 0 or group >= handle.metadata.num_row_groups:
            raise Round0244Error(f"R0244 row {row} is outside {path}")
        wanted.setdefault(group, []).append(row)
    out: dict[int, dict[str, Any]] = {}
    for group, group_rows in wanted.items():
        table = handle.read_row_group(group, columns=columns)
        base = offsets[group]
        for row in group_rows:
            local = row - base
            record = {"parquet_row": int(row), "shard_rows": offsets[-1]}
            for name in columns:
                record[name] = table.column(name)[local].as_py()
            out[row] = record
    return out


def run_text(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0244Error("R0244 handler received another queue")
    started = time.monotonic()
    inheritance = verify_inheritance(job)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0244 text")

    substrate = _readonly_memmap(
        _bound_path(job, "substrate_array", label="substrate"),
        label="R0244 substrate", shape=(ROWS, DIMENSION),
    )
    provenance = np.load(
        _bound_path(job, "provenance", label="provenance"),
        mmap_mode="r", allow_pickle=False,
    )
    if provenance.shape != (ROWS,):
        raise Round0244Error("R0244 provenance is not one record per row")
    labels = _readonly_memmap(
        _bound_path(job, "r0242_primary_cluster", label="primary labels"),
        label="R0244 primary labels", shape=(ROWS,),
    )
    probe_rows = _readonly_memmap(
        _bound_path(job, "probe_query_rows", label="probe query rows"),
        label="R0244 probe query rows", shape=(TRUTH_PROBE_ROWS,),
    )
    probe_cluster = _readonly_memmap(
        _bound_path(job, "r0242_probe_cluster", label="probe labels"),
        label="R0244 probe labels", shape=(TRUTH_PROBE_ROWS,),
    )
    strict_builder = _readonly_memmap(
        _bound_path(
            job, "r0242_probe_builder_missing_edges", label="builder missing"
        ),
        label="R0244 strict builder-missing", shape=(TRUTH_PROBE_ROWS,),
    )
    tie_builder = _readonly_memmap(
        _bound_path(
            job, "r0243_probe_tie_aware_builder_missing_edges",
            label="tie-aware builder missing",
        ),
        label="R0244 tie-aware builder-missing", shape=(TRUTH_PROBE_ROWS,),
    )
    truth_ids = _readonly_memmap(
        _bound_path(job, "truth_ids", label="truth ids"),
        label="R0244 truth ids", shape=(TRUTH_PROBE_ROWS, GRAPH_K),
    )
    truth_cos = _readonly_memmap(
        _bound_path(job, "truth_cos", label="truth cosines"),
        label="R0244 truth cosines", shape=(TRUTH_PROBE_ROWS, GRAPH_K),
    )
    graph_ids = _readonly_memmap(
        str(inheritance["graph"]["ids"]["canonical_path"]),
        label="R0244 graph ids", shape=(ROWS, GRAPH_K),
    )
    graph_cos = _readonly_memmap(
        str(inheritance["graph"]["cosines"]["canonical_path"]),
        label="R0244 builder cosines", shape=(ROWS, GRAPH_K),
    )

    corpora = [name for name, _rows in COMPOSITION]
    shard_index = {
        index: _corpus_shards(name) for index, name in enumerate(corpora)
    }

    #: What cluster 168 IS, over all of its rows rather than its probe sample.
    member_blocks: list[np.ndarray] = []
    for begin in range(0, ROWS, LABEL_BLOCK):
        _check_runner_abort("R0244 cluster membership scan")
        end = min(begin + LABEL_BLOCK, ROWS)
        stripe = np.asarray(labels[begin:end])
        member_blocks.append(np.flatnonzero(stripe == CLUSTER_UNDER_TEST) + begin)
        del stripe
    members = np.concatenate(member_blocks).astype(np.int64)
    del member_blocks
    member_corpus = np.asarray(provenance["corpus"][members], dtype=np.int64)
    corpus_mix = {
        corpora[index]: int(np.count_nonzero(member_corpus == index))
        for index in range(len(corpora))
    }
    global_corpus_mix = {name: int(rows) for name, rows in COMPOSITION}
    del member_corpus

    #: The probe rows in cluster 168 whose strict loss was tie-forgiven.
    probe_labels = np.asarray(probe_cluster, dtype=np.int64)
    strict = np.asarray(strict_builder, dtype=np.int64)
    tie = np.asarray(tie_builder, dtype=np.int64)
    forgiven = np.flatnonzero(
        (probe_labels == CLUSTER_UNDER_TEST) & (strict > tie)
    )
    if forgiven.size == 0:
        raise Round0244Error(
            "R0244 found no tie-forgiven probe row in cluster 168; there is "
            "nothing to read"
        )
    rng = np.random.default_rng(TEXT_SAMPLE_SEED)
    take = min(int(TEXT_SAMPLE_PAIRS), int(forgiven.size))
    chosen = np.sort(rng.choice(forgiven, size=take, replace=False))

    #: Resolve one (query, missed true neighbour, tie substitute) triple per
    #: sampled probe row.
    triples: list[dict[str, Any]] = []
    for probe in chosen:
        _check_runner_abort("R0244 tie-substitute resolution")
        query_row = int(probe_rows[probe])
        truth = np.asarray(truth_ids[probe], dtype=np.int64)
        truth_c = np.asarray(truth_cos[probe], dtype=np.float64)
        built = np.asarray(graph_ids[query_row], dtype=np.int64)
        built_c = np.asarray(graph_cos[query_row], dtype=np.float64)
        missing = ~np.isin(truth, built)
        extra = ~np.isin(built, truth)
        if not missing.any() or not extra.any():
            continue
        best: tuple[float, int, int] | None = None
        for position in np.flatnonzero(missing):
            gaps = np.abs(built_c[extra] - truth_c[position])
            local = int(np.argmin(gaps))
            gap = float(gaps[local])
            if gap <= TIE_TOLERANCE and (best is None or gap < best[0]):
                best = (gap, int(truth[position]), int(built[extra][local]))
        if best is None:
            continue
        gap, missed_row, substitute_row = best
        triples.append({
            "probe_index": int(probe),
            "query_row": query_row,
            "missed_truth_row": missed_row,
            "tie_substitute_row": substitute_row,
            "cosine_gap_to_the_tie_threshold": gap,
            "tie_tolerance": float(TIE_TOLERANCE),
            "strict_builder_missing_edges": int(strict[probe]),
            "tie_aware_builder_missing_edges": int(tie[probe]),
        })
    if not triples:
        raise Round0244Error(
            "R0244 resolved no (missed truth, tie substitute) pair in the "
            "sampled cluster-168 rows"
        )

    #: Resolve every needed substrate row to its chunk text.
    needed = sorted({
        row for triple in triples
        for row in (
            triple["query_row"], triple["missed_truth_row"],
            triple["tie_substitute_row"],
        )
    })
    by_shard: dict[tuple[int, int], list[int]] = {}
    for row in needed:
        record = provenance[row]
        by_shard.setdefault(
            (int(record["corpus"]), int(record["shard"])), []
        ).append(row)
    texts: dict[int, dict[str, Any]] = {}
    for (corpus_index, shard), rows in sorted(by_shard.items()):
        _check_runner_abort("R0244 chunk text read")
        corpus = corpora[corpus_index]
        shards = shard_index[corpus_index]
        if shard >= len(shards):
            raise Round0244Error(
                f"R0244 provenance names shard {shard} of {corpus}, which has "
                f"{len(shards)}"
            )
        parquet = _chunk_parquet(corpus, shards[shard], shard_index=shard)
        local = [int(provenance[row]["row"]) for row in rows]
        records = _read_chunk_texts(
            parquet, local, expect_rows=_shard_rows(shards[shard])
        )
        for row, position in zip(rows, local):
            record = dict(records[position])
            record.update({
                "substrate_row": int(row),
                "corpus": corpus,
                "shard": int(shard),
                "shard_file": os.path.basename(parquet),
            })
            texts[int(row)] = record

    #: VERIFY the binding rather than assume it: re-embed each chunk with the
    #: model that produced the substrate and require it to reproduce the row.
    os.environ.setdefault("HF_HOME", "/data/hf")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(TEXT_MODEL, device="cpu")
    order = sorted(texts)
    encoded = np.asarray(
        model.encode(
            [str(texts[row]["chunk_text"]) for row in order],
            normalize_embeddings=True, batch_size=8, show_progress_bar=False,
        ),
        dtype=np.float64,
    )
    unit: dict[int, np.ndarray] = {}
    verification: list[dict[str, Any]] = []
    for position, row in enumerate(order):
        vector = np.asarray(substrate[row], dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        vector = vector / norm if norm > 0 else vector
        unit[row] = vector
        cosine = float(vector @ encoded[position])
        verification.append({
            "substrate_row": int(row),
            "corpus": texts[row]["corpus"],
            "shard_file": texts[row]["shard_file"],
            "parquet_row": int(texts[row]["parquet_row"]),
            "re_embedded_cosine": cosine,
            "binds": bool(cosine >= TEXT_BINDING_COSINE_FLOOR),
        })
    binding_holds = all(entry["binds"] for entry in verification)
    if not binding_holds:
        raise Round0244Error(
            "R0244 STOP: a chunk text does not reproduce its substrate row "
            "under the substrate's own model; the row-to-text mapping is not "
            f"what this node assumed: {verification}"
        )

    pairs: list[dict[str, Any]] = []
    for triple in triples:
        query = unit[triple["query_row"]]
        missed = unit[triple["missed_truth_row"]]
        substitute = unit[triple["tie_substitute_row"]]
        missed_text = str(texts[triple["missed_truth_row"]]["chunk_text"])
        substitute_text = str(
            texts[triple["tie_substitute_row"]]["chunk_text"]
        )
        classification = classify_text_pair(missed_text, substitute_text)
        missed_record = texts[triple["missed_truth_row"]]
        substitute_record = texts[triple["tie_substitute_row"]]
        pairs.append({
            **triple,
            "cosine_query_to_missed": float(query @ missed),
            "cosine_query_to_substitute": float(query @ substitute),
            "cosine_missed_to_substitute": float(missed @ substitute),
            "same_corpus": bool(
                missed_record["corpus"] == substitute_record["corpus"]
            ),
            "same_shard_file": bool(
                missed_record["shard_file"] == substitute_record["shard_file"]
            ),
            "same_source_document": bool(
                missed_record.get("id") is not None
                and missed_record.get("id") == substitute_record.get("id")
            ),
            "classification": classification,
            "query_excerpt": excerpt(
                str(texts[triple["query_row"]]["chunk_text"])
            ),
            "missed_truth_excerpt": excerpt(missed_text),
            "tie_substitute_excerpt": excerpt(substitute_text),
            "missed_corpus": missed_record["corpus"],
            "substitute_corpus": substitute_record["corpus"],
        })

    categories: dict[str, int] = {name: 0 for name, _text in NEAR_DUPLICATE_CATEGORIES}
    for pair in pairs:
        categories[pair["classification"]["category"]] += 1
    cosines = np.asarray(
        [pair["cosine_missed_to_substitute"] for pair in pairs],
        dtype=np.float64,
    )
    jaccards = np.asarray(
        [pair["classification"]["jaccard_char_5gram"] for pair in pairs],
        dtype=np.float64,
    )

    body = dict(_receipt_envelope(manifest))
    body.update({
        "schema": TEXT_SCHEMA,
        "capabilities": [TEXT_CAPABILITY],
        "text_note": TEXT_NOTE,
        "cluster": CLUSTER_UNDER_TEST,
        "cluster_rows": int(members.size),
        "cluster_corpus_mix": corpus_mix,
        "cluster_corpus_share": {
            name: (count / float(members.size)) for name, count in
            corpus_mix.items()
        },
        "substrate_corpus_share": {
            name: (rows / float(ROWS))
            for name, rows in global_corpus_mix.items()
        },
        "probe_rows_in_cluster": int(
            np.count_nonzero(probe_labels == CLUSTER_UNDER_TEST)
        ),
        "probe_rows_with_tie_forgiveness": int(forgiven.size),
        "sampled_probe_rows": int(take),
        "resolved_pairs": len(pairs),
        "sample_seed": TEXT_SAMPLE_SEED,
        "text_binding_cosine_floor": TEXT_BINDING_COSINE_FLOOR,
        "text_binding_verification": verification,
        "text_binding_holds": binding_holds,
        "minimum_re_embedded_cosine": float(
            min(entry["re_embedded_cosine"] for entry in verification)
        ),
        "near_duplicate_categories": [
            {"name": name, "definition": text}
            for name, text in NEAR_DUPLICATE_CATEGORIES
        ],
        "category_counts": categories,
        "cosine_missed_to_substitute": {
            "min": float(cosines.min()),
            "median": float(np.median(cosines)),
            "mean": float(cosines.mean()),
            "max": float(cosines.max()),
        },
        "jaccard_missed_to_substitute": {
            "min": float(jaccards.min()),
            "median": float(np.median(jaccards)),
            "mean": float(jaccards.mean()),
            "max": float(jaccards.max()),
        },
        "pairs": pairs,
        "bulk_input_memmap_attestation": _memmap_attestation({
            "substrate": substrate, "primary_labels": labels,
            "graph_ids": graph_ids, "builder_cosines": graph_cos,
            "truth_ids": truth_ids, "truth_cosines": truth_cos,
        }),
        "model": TEXT_MODEL,
        "model_device": "cpu",
        "performance": {"total_wall_s": time.monotonic() - started},
    })
    _seal(output, TEXT_FILE, body)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == DID_ACTION:
        run_did(active, job)
    elif action == WATCHDOG_ACTION:
        run_watchdog(active, job)
    elif action == SAMPLER_ACTION:
        run_sampler(active, job)
    elif action == TEXT_ACTION:
        run_text(active, job)
    else:
        raise Round0244Error(f"R{ROUND_ID} does not authorize action {action!r}")


__all__ = [
    "CLUSTER_UNDER_TEST",
    "DID_ACTION",
    "DID_CAPABILITY",
    "DID_FILE",
    "DID_SCHEMA",
    "SAMPLER_ACTION",
    "SAMPLER_CAPABILITY",
    "SAMPLER_FILE",
    "SAMPLER_SCHEMA",
    "TEXT_ACTION",
    "TEXT_CAPABILITY",
    "TEXT_FILE",
    "TEXT_SCHEMA",
    "WATCHDOG_ACTION",
    "WATCHDOG_CAPABILITY",
    "WATCHDOG_FILE",
    "WATCHDOG_SCHEMA",
    "run_did",
    "run_job",
    "run_sampler",
    "run_text",
    "run_watchdog",
]
