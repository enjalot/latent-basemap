"""Build R0246's queue. Never launches anything.

Three CPU-shaped nodes against bytes that already exist: R0245's sealed sampler
receipt (the basis for the registered coverage floor), R0243's `10 GB` weight
array (streamed once so the abort-read gap inside `two_level_weight_sample` can
be re-measured after the fix), and R0238's truth probe, substrate and R0240's
graph ids (for the tie-aware precision measurement). Nothing is rebuilt and no
edge list is republished.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0227_low_c_contract import TIE_TOLERANCE
from basemap.round0238_rung5 import GRAPH_K, TRUTH_PROBE_ROWS, TRUTH_PROBE_SEED
from basemap.round0240_rung5 import (
    INHERITED_SUBSTRATE_MANIFEST,
    INHERITED_TRUTH_MANIFEST,
)
from basemap.round0241_qualify import (
    INHERITED_GRAPH_IDS,
    REGISTERED_GRAPH_IDS_SHA256,
    REGISTERED_SELECTED_CLUSTERS,
)
from basemap.round0244_prereq import (
    R0238_PROVENANCE_SHA256,
    R0243_EDGES_HEADER_SHA256,
    R0243_EDGES_WTS_SHA256,
)
from basemap.round0245_guard import (
    MIN_BINDING_SLOPE_BASIS,
    MIN_BINDING_SLOPE_BYTES_PER_S,
    R0244_BUDGET_HEADROOM_BYTES,
    R0244_MEASURED_SLOPE_BYTES_PER_S,
)
from basemap.round0245_sampler import R0245_SAMPLER_DRAWS
from basemap.round0246_guard import (
    ATTEMPT_1_GAP_S,
    COVERAGE_GATE_MIN_EXPECTED_SAMPLES,
    GPU_HOURS_CAP,
    MIN_THREAD_SAMPLE_COVERAGE,
    POLL_GATE_NOTE,
    R0246_MAX_POLL_SPACING_S,
    ROUND_ID,
    ROWS,
    SAMPLER_LIVENESS_NOTE,
    registered_thresholds,
)
from basemap.round0246_tie import (
    TIE_AGGREGATE_ONLY_RULE,
    TIE_AWARE_CLAIM_LEDGER,
    TIE_PRECISION_ROWS,
    TIE_PRECISION_SEED,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
from experiments.round0246_nodes import (
    GUARD_ACTION,
    GUARD_CAPABILITY,
    NODE_ANON_BUDGET_BYTES,
    NODE_HEADROOM_BYTES,
    R0245_SEALED_MAX_ABORT_READ_GAP_S,
    SAMPLER_ACTION,
    SAMPLER_CAPABILITY,
    TIE_ACTION,
    TIE_CAPABILITY,
)

ROUND_ROOT = "/data/latent-basemap/runs/round-0246"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0246-2026-08-10.md")
HANDLER_MODULE = "experiments.round0246_nodes"

R0238_TRUTH_DIR = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-uniform-probe-k15-truth-v1"
)
R0243_FUZZY_DIR = (
    "/data/latent-basemap/runs/round-0243/queue/artifacts/"
    "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1"
)
R0245_SAMPLER_RECEIPT = (
    "/data/latent-basemap/runs/round-0245/queue-correction-1/artifacts/"
    "minilm-mixed-100000k-k15-sampler-power-and-limits-v1/sampler-power.json"
)
#: The receipt whose `24.46713631998864` s gap this round exists to close, and
#: whose healthy sampling coverage the registered floor is calibrated against.
R0245_SAMPLER_RECEIPT_SHA256 = (
    "283867075d837ba296700b88ce41fcc3fe6afad04f8693a0be432fde6c281ede"
)

SMALL_INPUTS: dict[str, str] = {
    "substrate_manifest": INHERITED_SUBSTRATE_MANIFEST,
    "truth_reference": INHERITED_TRUTH_MANIFEST,
    "edges_header": os.path.join(R0243_FUZZY_DIR, "edges-k15-fuzzy-header.npz"),
    "r0245_sampler_receipt": R0245_SAMPLER_RECEIPT,
}
VECTOR_INPUTS: dict[str, str] = {
    "probe_query_rows": os.path.join(R0238_TRUTH_DIR, "probe-query-rows.i64.npy"),
    "truth_ids": os.path.join(R0238_TRUTH_DIR, "truth-k15-ids.i32.npy"),
    "truth_cos": os.path.join(R0238_TRUTH_DIR, "truth-k15-cos.f32.npy"),
}
#: Bound at full sha256 here and by size in the node, but kept out of
#: `expected_inputs`: the v2.1 proportionate-verification rule.
BULK_INPUTS: dict[str, str] = {
    "graph_ids": INHERITED_GRAPH_IDS,
    "edges_wts": os.path.join(R0243_FUZZY_DIR, "edges-k15-fuzzy-wts.f32.npy"),
}
MANIFEST_INPUTS = ("substrate_array", "provenance")

REGISTERED_DIGESTS = (
    ("graph_ids", REGISTERED_GRAPH_IDS_SHA256),
    ("edges_header", R0243_EDGES_HEADER_SHA256),
    ("edges_wts", R0243_EDGES_WTS_SHA256),
    ("r0245_sampler_receipt", R0245_SAMPLER_RECEIPT_SHA256),
)


def _issued_round(release_sha: str) -> tuple[dict[str, Any], list[str]]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor",
         base_commit, release_sha],
        check=False,
    ).returncode == 0
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or not descendant
    ):
        raise RuntimeError("R0246 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0246 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError(
            f"R0246 release checkout is at {observed}, not {release_sha}"
        )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [os.path.join(RELEASE_ROOT, ".venv/bin/python"), "-m", "pytest", "-q",
         "tests/test_round0246_contract.py", "tests/test_round0246_cpu_smoke.py"],
        cwd=RELEASE_ROOT, env=environment, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0246 release CPU smoke failed:\n{completed.stdout[-4000:]}"
        )
    return {
        "release_sha": release_sha,
        "cuda_visible_devices": "",
        "command": (
            "pytest -q tests/test_round0246_contract.py "
            "tests/test_round0246_cpu_smoke.py"
        ),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "reaches": [
            "ALL THREE node entry paths executed end to end through run_job on "
            "tiny inputs - the defect class that killed R0216 (NameError), "
            "R0236 (arity) and R0242 attempt 1 (missing module)",
            "the A1-bis reviewer control: an OSError planted inside the "
            "sampling thread, the death recorded, the receipt reporting a dead "
            "sampler, and the liveness gate REFUSING the receipt",
            "the A3-bis reviewer control: ROUNDRUN_ABORT_FLAG pointed at an "
            "existing writable DIRECTORY, both the precondition and the "
            "watchdog constructor refusing, and the directory left intact",
            "the A2-bis reviewer control: attempt 1's exact "
            f"{ATTEMPT_1_GAP_S} s gap replayed at own-slope 0.0 and REFUSED, "
            "with the R0245 form shown passing the same gap; plus a one-poll "
            "and a zero-poll stage, both refused",
            "the novel attack battery: a starved sampler, a poll-inflated "
            "coverage figure, a silently returning sampling loop, an "
            "intermittent OSError under the consecutive tolerance, a FIFO / "
            "read-only file / character device / dangling symlink / symlink-to-"
            "directory flag path, a TOCTOU replacement caught at the node tail, "
            "and five ways of defeating the poll-spacing gate",
            "the polled two_level_weight_sample reproducing the unpolled one "
            "element by element - the assertion that the fix is inert",
            "the tie-aware aggregate-only gate permitting an aggregate use and "
            "REFUSING a small-integer one",
            "a source-level assertion that no file this round adds contains a "
            "signalling construct - read, not delegated to a detector",
        ],
    }


def prepare_round0246(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    prior_gpu_wall_s: float = 0.0,
    only: tuple[str, ...] = (),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0246 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)

    signatures: dict[str, dict[str, Any]] = {}
    for key, path in {**SMALL_INPUTS, **VECTOR_INPUTS, **BULK_INPUTS}.items():
        if not os.path.exists(path):
            raise RuntimeError(f"R0246 inherited input absent: {key} at {path}")
        signatures[key] = expected_input_signature(path)
    for key, digest in REGISTERED_DIGESTS:
        if signatures[key].get("sha256") != digest:
            raise RuntimeError(
                f"R0246 STOP: {key} hashes to {signatures[key].get('sha256')}, "
                f"registered {digest}"
            )
    with open(INHERITED_SUBSTRATE_MANIFEST, encoding="utf-8") as handle:
        substrate_manifest = json.load(handle)
    signatures["substrate_array"] = dict(substrate_manifest["substrate"])
    signatures["provenance"] = dict(substrate_manifest["provenance"])
    if signatures["provenance"].get("sha256") != R0238_PROVENANCE_SHA256:
        raise RuntimeError("R0246 STOP: R0238 provenance digest moved")
    for key in MANIFEST_INPUTS:
        path = str(signatures[key]["canonical_path"])
        if not os.path.exists(path):
            raise RuntimeError(f"R0246 inherited input absent: {key} at {path}")
        if os.path.getsize(path) != int(signatures[key]["bytes"]):
            raise RuntimeError(f"R0246 {key} is not its declared size")

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0246 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        smoke_path, _release_cpu_smoke(release_sha), immutable=True
    )
    atomic_write_new_json(
        os.path.join(preflight, "registered-thresholds.json"),
        {
            "registered_before_the_round_ran": registered_thresholds(),
            "node_anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "node_headroom_bytes": NODE_HEADROOM_BYTES,
            "registered_max_poll_spacing_s": R0246_MAX_POLL_SPACING_S,
            "r0245_sealed_widest_abort_read_gap_s": (
                R0245_SEALED_MAX_ABORT_READ_GAP_S
            ),
            "tie_aggregate_only_rule": TIE_AGGREGATE_ONLY_RULE,
            "note": POLL_GATE_NOTE,
        },
        immutable=True,
    )

    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
        *(signatures[key] for key in SMALL_INPUTS),
        *(signatures[key] for key in VECTOR_INPUTS),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    guard_dir = os.path.join(artifacts, GUARD_CAPABILITY)
    sampler_dir = os.path.join(artifacts, SAMPLER_CAPABILITY)
    tie_dir = os.path.join(artifacts, TIE_CAPABILITY)

    policy = {"gpu_required": True, "training_performed": False, "cpu_heavy": True}
    gpu_budget_remaining_s = float(GPU_HOURS_CAP * 3600.0 - prior_gpu_wall_s)

    jobs: list[dict[str, Any]] = [
        {
            "id": "guard_0246", "action": GUARD_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": [],
            "outputs": [guard_dir],
            "done_marker": os.path.join(artifacts, "guard_0246.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 600.0,
            "capability": GUARD_CAPABILITY,
            "r0245_sampler_receipt": signatures["r0245_sampler_receipt"],
            "anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "stage_budget_s": 1_800.0,
            "gpu_budget_remaining_s": gpu_budget_remaining_s,
            "node_policy": dict(policy),
        },
        {
            "id": "sampler_0246", "action": SAMPLER_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": ["guard_0246"],
            "outputs": [sampler_dir],
            "done_marker": os.path.join(artifacts, "sampler_0246.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 1_800.0,
            "capability": SAMPLER_CAPABILITY,
            "edges_header": signatures["edges_header"],
            "edges_wts": signatures["edges_wts"],
            "draws": R0245_SAMPLER_DRAWS,
            "anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "stage_budget_s": 5_400.0,
            "node_policy": dict(policy),
        },
        {
            "id": "tie_0246", "action": TIE_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": ["sampler_0246"],
            "outputs": [tie_dir],
            "done_marker": os.path.join(artifacts, "tie_0246.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 1_200.0,
            "capability": TIE_CAPABILITY,
            **{key: signatures[key] for key in VECTOR_INPUTS},
            "substrate_array": signatures["substrate_array"],
            "graph_ids": signatures["graph_ids"],
            "tie_rows": TIE_PRECISION_ROWS,
            "tie_seed": TIE_PRECISION_SEED,
            "anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "stage_budget_s": 3_600.0,
            "node_policy": dict(policy),
        },
    ]

    if only:
        keep = set(only)
        unknown = keep - {str(job["id"]) for job in jobs}
        if unknown:
            raise RuntimeError(f"R0246 has no such node(s): {sorted(unknown)}")
        jobs = [job for job in jobs if str(job["id"]) in keep]
        for job in jobs:
            job["deps"] = [dep for dep in job["deps"] if dep in keep]

    queue = _base_manifest(
        round_id=ROUND_ID, release_sha=release_sha, round_file=ROUND_FILE,
        queue_root=queue_root, gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu", gpu=True,
    )
    queue.update({
        "schema": "round0246-guard-closure-and-abort-poll-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-graph",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-100000k-nested-substrate-and-reserves-v1",
            "minilm-mixed-100000k-uniform-probe-k15-truth-v1",
            "minilm-mixed-100000k-cluster-spill-k15-neighbour-graph-v1",
            "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1",
            "minilm-mixed-100000k-k15-fuzzy-edge-sampler-v1",
            "minilm-mixed-100000k-k15-sampler-power-and-limits-v1",
            "round0244-threaded-host-watchdog-v1",
            "round0245-guard-closure-v1",
            "runner-signal-free-node-supervision-v1",
        ],
        "capabilities_produced": [
            GUARD_CAPABILITY, SAMPLER_CAPABILITY, TIE_CAPABILITY,
        ],
        "prior_attempt_gpu_wall_s": float(prior_gpu_wall_s),
        "gpu_budget_remaining_s": gpu_budget_remaining_s,
        "substrate_inherited": True,
        "graph_inherited": True,
        "inherited_artifacts": [
            "graph", "provenance", "r0243_edges", "r0245_sampler_receipt",
            "substrate", "truth",
        ],
        "inherited_from": [
            "/data/latent-basemap/runs/round-0238/queue",
            "/data/latent-basemap/runs/round-0240/queue",
            "/data/latent-basemap/runs/round-0243/queue",
            "/data/latent-basemap/runs/round-0245/queue-correction-1",
        ],
        "inheritance_note": (
            "Every byte this round reads already exists and is bound at its "
            "published digest. The only bulk reads are R0243's "
            "10,044,413,144-byte weight array, streamed once so the abort-read "
            "gap inside two_level_weight_sample can be re-measured after the "
            "fix, and a 20,000-row sample of R0238's substrate. Nothing is "
            "rebuilt, no edge list is republished, and no map is trained."
        ),
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            "total": sum(float(job["p90_wall_s"]) for job in jobs)
        },
        "scientific_contract": {
            "question": (
                "review-0245-01 accepted R0245 and then defeated all three of "
                "its guard fixes outside their own positive controls, and one "
                "of the three had been loosened after it caught its author. "
                "This round closes the three, closes the abort-read gap that "
                "blocks the first multi-hour training node, and decides what "
                "the tie-aware tolerance finding invalidates."
            ),
            "rows": ROWS,
            "k": GRAPH_K,
            "clusters": REGISTERED_SELECTED_CLUSTERS,
            "closure_1_the_oserror_arm": {
                "defect": (
                    "review-0245-01 A1-bis: _loop caught OSError FIRST and "
                    "passed, so a planted OSError left the thread alive and "
                    "sampling nothing with sampling_thread_alive true, poll() "
                    "quiet, sample_coverage 0.0500, and "
                    "_require_live_sampler() PASSING it."
                ),
                "closure": (
                    "the OSError arm is bounded at a registered "
                    "consecutive-failure count; ANY unrequested exit from the "
                    "sampling loop is a recorded death, including a return "
                    "that raises nothing; and require_live_sampler() gates on "
                    "a registered THREAD sample-coverage floor, computed on "
                    "the thread's own samples because the published "
                    "sample_coverage counts boundary poll() samples too."
                ),
                "min_thread_sample_coverage": MIN_THREAD_SAMPLE_COVERAGE,
                "coverage_gate_min_expected_samples": (
                    COVERAGE_GATE_MIN_EXPECTED_SAMPLES
                ),
                "positive_control": (
                    "the reviewer's own shape: OSError raised inside the "
                    "sampling thread from the second sample onward."
                ),
                "note": SAMPLER_LIVENESS_NOTE,
            },
            "closure_2_the_flag_path": {
                "defect": (
                    "review-0245-01 A3-bis: require_enforceable_abort_flag "
                    "validated os.path.dirname(path), so an existing writable "
                    "DIRECTORY at the flag path passed the precondition, armed "
                    "the watchdog, and left abort_flag_written false with "
                    "'[Errno 21] Is a directory'."
                ),
                "closure": (
                    "a probe file is created and removed AT THE PATH, both in "
                    "the node precondition and again at arm time; anything "
                    "that exists and is not a writable regular file is "
                    "refused; and require_abort_flag_landed() gates the node "
                    "tail on abort_flag_written, which is the only thing that "
                    "can catch a time-of-check/time-of-use replacement."
                ),
                "positive_control": (
                    "the reviewer's own shape: ROUNDRUN_ABORT_FLAG pointed at "
                    "an existing writable directory."
                ),
            },
            "closure_3_the_poll_spacing_gate": {
                "defect": (
                    "review-0245-01 D: R0245's correction rebound the gate to "
                    f"max(own slope, 1.0) B/s, and attempt 1's exact "
                    f"{ATTEMPT_1_GAP_S} s gap replayed at own-slope 0.0 PASSES "
                    "it (binding slope 1.0 B/s, permitted spacing "
                    "47,244,640,256 s). Zero and one tracker calls passed "
                    "trivially, and the intervals before the first read and "
                    "after the last were never measured."
                ),
                "closure": (
                    "the binding slope is max(own measured slope, "
                    f"{MIN_BINDING_SLOPE_BYTES_PER_S} B/s); the measurement is "
                    "anchored at stage start and stage end; a non-zero measured "
                    "spacing and at least ceil(wall / ceiling) + 1 reads are "
                    "required; the spacing is capped at the registered "
                    f"{R0246_MAX_POLL_SPACING_S} s regardless of declared "
                    "headroom; and the worst-case verdict is a hard gate for "
                    "any node declaring training_performed true."
                ),
                "min_binding_slope_bytes_per_s": MIN_BINDING_SLOPE_BYTES_PER_S,
                "min_binding_slope_basis": MIN_BINDING_SLOPE_BASIS,
                "registered_max_poll_spacing_s": R0246_MAX_POLL_SPACING_S,
                "registered_max_poll_spacing_basis": (
                    f"{R0244_BUDGET_HEADROOM_BYTES} B of R0244 sealed headroom "
                    f"over {R0244_MEASURED_SLOPE_BYTES_PER_S} B/s of R0244 "
                    "sealed measured slope"
                ),
                "positive_control": (
                    "the reviewer's own shape: attempt 1's exact "
                    f"{ATTEMPT_1_GAP_S} s gap replayed at own-slope 0.0, plus "
                    "a one-poll and a zero-poll stage."
                ),
                "note": POLL_GATE_NOTE,
            },
            "closure_4_the_blocker": {
                "defect": (
                    "review-0245-01 section C: the widest abort-read gap is "
                    f"{R0245_SEALED_MAX_ABORT_READ_GAP_S} s inside R0244's "
                    "registered two_level_weight_sample, whose abort_check "
                    "fired once per 128 blocks. Against "
                    f"{R0244_MEASURED_SLOPE_BYTES_PER_S} B/s that permits "
                    "287,929,172,523 B against "
                    f"{R0244_BUDGET_HEADROOM_BYTES} B of headroom - 6.09x over."
                ),
                "closure": (
                    "every loop in two_level_weight_sample polls once per "
                    "unit; the block-choice searchsorted, the stable order and "
                    "the weight gather are chunked; and the 40,000,000-element "
                    "np.unique is replaced by a per-group distinct count "
                    "inside the polled loop. The draw is unchanged and the "
                    "node fails closed unless it reproduces R0245's sealed "
                    "distinct count and fidelity statistics at the same seed."
                ),
                "no_multi_hour_training_node_may_run_until_this_passes": True,
            },
            "closure_5_the_tie_aware_precision_finding": {
                "defect": (
                    "review-0245-01 E: TIE_TOLERANCE sits at 1.68-1.86x its "
                    "own p99 noise floor and the measured maximum "
                    "disagreement exceeds the tolerance outright. Aggregate "
                    "figures stand; per-row uses do not."
                ),
                "closure": (
                    "the per-decision error rate is MEASURED - the same "
                    "candidates scored in float32 and float64 from the same "
                    "bytes, counting verdict flips - and propagated through a "
                    "ledger of every published claim in R0241 and R0243 that "
                    "rests on a tie-aware decision. The tolerance is NOT "
                    "raised; the aggregate-only rule is registered instead."
                ),
                "rule": TIE_AGGREGATE_ONLY_RULE,
                "tie_tolerance": float(TIE_TOLERANCE),
                "claims_in_the_ledger": len(TIE_AWARE_CLAIM_LEDGER),
                "tie_precision_rows": TIE_PRECISION_ROWS,
                "tie_precision_seed": TIE_PRECISION_SEED,
            },
            "truth_probe_rows": TRUTH_PROBE_ROWS,
            "truth_probe_seed": TRUTH_PROBE_SEED,
            "gpu_hours_cap": GPU_HOURS_CAP,
            "gpu_hours_cap_note": (
                "1.0 GPU-h. Every node is CPU-shaped: the round holds the GPU "
                "lease so no second workload can touch the card, and creates "
                "no CUDA context at all."
            ),
            "no_training": True,
            "no_gate_registered": True,
            "no_adoption_claimed": True,
            "no_map_quality_claimed": True,
            "no_displacement_measured": True,
            "no_atlas_claim": True,
            "no_build": True,
            "no_assembly": True,
        },
    })
    queue_path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(queue_path, queue, immutable=True)
    return queue_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    parser.add_argument("--prior-gpu-wall-s", type=float, default=0.0)
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args(argv)
    path = prepare_round0246(
        release_sha=args.release_sha, queue_root=args.queue_root,
        prior_gpu_wall_s=args.prior_gpu_wall_s, only=tuple(args.only),
    )
    print(json.dumps({"queue": path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
