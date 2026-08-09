#!/usr/bin/env python3
"""Prepare, but never launch, the R0238 queue — the nested 50M rung.

Five nodes: the high-`c` reachability ceiling on R0236's sealed 25M substrate
(first, because it answers the 100M question and depends on nothing this round
produces), assembly of the nested 50M substrate with its inherited reserve,
exact truth for the registered uniform probe, the five-seed imbalance grid plus
one `cluster-spill-nnd` cell with the block-layer and page-cache instruments,
and qualification with the 100M verdict. No training node exists by design.

Every inherited scientific input is declared at its FULL signature — R0236's
sealed substrate manifest, probe truth and build ladder, R0235's and R0233's
sealed build ladders, R0229's sealed nn-descent quality sweep, adopted-arm build
receipt and spill-reachability artifact — so that no device point, no imbalance
figure and no reachability reference in this round can come from anywhere but a
hash-bound artifact (review-0233-01 D3, upheld by review-0235-01 and 0236-01).

**Correction queues inherit the substrate; they do not rebuild it.** At
50,000,000 rows the substrate is `76.8 GB` and R0236 spent `~76 GB` re-earning
its 25M one twice before reclaiming the space. `--inherit-substrate <manifest>`
drops the `assemble_100000k` node and binds that manifest at its full signature,
so a re-run after a setup-class defect costs the failed node and nothing else.
Review-0233-01 D7 objects to releasing a capability that lives inside a failed
queue's tree; the round file registers that trade explicitly and states which
queue each released artifact came from.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0238_rung5 import (
    ADVERSE_DRIFT_NOTE,
    BUILD_TIMEOUT_S,
    CODE_POOL_EXTENSION,
    CODE_POOL_ROWS,
    CODE_POOL_SHARDS,
    DECLINED_DERISK_NOTE,
    POOL_EXTENSION_UNIFORMITY,
    PREDICTION_IMBALANCE_AT_C400,
    PREDICTION_TOLERANCE_AT_C400,
    COMPOSITION,
    C_BUILD_MIN,
    C_BUILD_MIN_NOTE,
    DIMENSION,
    GPU_HOURS_CAP,
    GRAPH_CAPABILITY,
    HUNDRED_M_CANDIDATES,
    HUNDRED_M_RULE,
    IMBALANCE_CAPABILITY,
    IMBALANCE_PROBE_CLUSTERS,
    IMBALANCE_PROBE_ROWS,
    IMBALANCE_REPLICATE_SEEDS,
    IO_CAPABILITY,
    IO_REGIME_NOTE,
    LADDER_CAPABILITY,
    MARGIN_NOTE,
    NN_DESCENT_SETTING,
    PARENT_ROWS,
    PRIMARY_IMBALANCE_SEED,
    REACHABILITY_CAPABILITY,
    REACHABILITY_CLUSTERS,
    REACHABILITY_CONCERN_FLOOR,
    REACHABILITY_NOTE,
    REACHABILITY_ROWS,
    RECALL_MEAN_FLOOR,
    RECALL_P10_FLOOR,
    RECALL_POPULATION,
    REPLICATE_NOTE,
    RESERVE_ROWS,
    ROUND_ID,
    ROWS,
    SELECTION_CANDIDATES,
    SELECTION_LAW,
    SPILL,
    STRUCTURAL_POPULATION,
    SUBSTRATE_CAPABILITY,
    TRUTH_AFFORDABILITY_NOTE,
    TRUTH_CAPABILITY,
    TRUTH_METHOD,
    TRUTH_PROBE_ROWS,
    TRUTH_PROBE_SEED,
    ZERO_RECALL_FORENSIC_NOTE,
)
from experiments.round0238_nodes import (
    ASSEMBLE_ACTION,
    LADDER_ACTION,
    QUALIFY_ACTION,
    REACHABILITY_ACTION,
    TRUTH_ACTION,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0238"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0238-2026-08-09.md")
EMB = "/data/embeddings"

#: Inherited, sealed, hash-bound scientific inputs.
R0237_ROOT = "/data/latent-basemap/runs/round-0237/queue/artifacts"
R0237_SUBSTRATE_MANIFEST = os.path.join(
    R0237_ROOT, "minilm-mixed-50000k-nested-substrate-and-reserves-v1",
    "substrate.json",
)
R0237_LADDER = os.path.join(
    R0237_ROOT, "minilm-mixed-50000k-cluster-spill-build-ladder-v1",
    "build-ladder.json",
)
R0237_REACHABILITY = os.path.join(
    R0237_ROOT, "minilm-mixed-25000k-cluster-spill-high-c-reachability-v1",
    "reachability.json",
)
R0236_ROOT = (
    "/data/latent-basemap/runs/round-0236/queue-correction-2/artifacts"
)
R0236_LADDER = os.path.join(
    R0236_ROOT, "minilm-mixed-25000k-cluster-spill-build-ladder-v1",
    "build-ladder.json",
)
R0235_LADDER = (
    "/data/latent-basemap/runs/round-0235/queue/artifacts/"
    "minilm-mixed-12500k-cluster-spill-build-ladder-v1/build-ladder.json"
)
R0233_LADDER = (
    "/data/latent-basemap/runs/round-0233/queue-correction-1/artifacts/"
    "minilm-mixed-6250k-cluster-spill-build-ladder-v1/build-ladder.json"
)
R0229_SWEEP = (
    "/data/latent-basemap/runs/round-0229/queue-correction-1/artifacts/"
    "minilm-mixed-2m-nnd-quality-sweep-v1/nnd-quality-sweep.json"
)
R0229_ARM = (
    "/data/latent-basemap/runs/round-0229/queue-phase2-correction-3/artifacts/"
    "spill-lifted-build/spill-lifted-build.json"
)
R0229_REACHABILITY = (
    "/data/latent-basemap/runs/round-0229/queue-correction-1/artifacts/"
    "minilm-mixed-2m-spill-reachability-v1/spill-reachability.json"
)

#: P90 wall per node at this rung, from R0237's measured walls scaled by the
#: doubling ratios its own cells established, plus the I/O term review-0237-01
#: priced for a substrate that no longer fits in page cache.
REACHABILITY_P90_WALL_S = 900.0
ASSEMBLE_P90_WALL_S = 5_400.0
TRUTH_P90_WALL_S = 2_400.0
LADDER_P90_WALL_S = 18_000.0
QUALIFY_P90_WALL_S = 3_000.0


def _issued_round(release_sha: str) -> tuple[dict[str, Any], list[str]]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor",
         base_commit, release_sha],
        check=False, timeout=10,
    ).returncode == 0
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or not descendant
    ):
        raise RuntimeError("R0238 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0238 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError(f"R0238 release checkout is at {observed}, not {release_sha}")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [os.path.join(RELEASE_ROOT, ".venv/bin/python"), "-m", "pytest", "-q",
         "tests/test_round0238_contract.py", "tests/test_round0238_cpu_smoke.py"],
        cwd=RELEASE_ROOT, env=environment, capture_output=True, text=True,
        timeout=900,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0238 release CPU smoke failed:\n{completed.stdout[-4000:]}"
        )
    return {
        "release_sha": release_sha,
        "cuda_visible_devices": "",
        "command": (
            "pytest -q tests/test_round0238_contract.py "
            "tests/test_round0238_cpu_smoke.py"
        ),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "reaches": [
            "nesting assertion on provenance keys (must raise when a parent row "
            "is absent and when the prefix is permuted)",
            "reserve disjointness assertion (must raise on one shared row)",
            "shard span assertion on a forced prefix (must raise)",
            "the registered truth probe (uniform, distinct, ascending, "
            "reproducible from the seed alone)",
            "the five-seed replicate table with its within-N spread and sd",
            "the tolerance-to-adverse-imbalance arithmetic against "
            "review-0236-01's published 25M / 50M / 100M figures",
            "the 100M verdict rule, including that it recommends c = 400 on "
            "review-0236-01's own numbers and refuses a c whose reachability "
            "is below the concern floor",
            "the reachability cell summary and its zero-reachable tripwire",
            "the zero-recall forensic (must verify a duplicate-family row and "
            "must NOT verify a genuine retrieval miss)",
            "the imbalance-grid call-site arity assertion (R0236's lost queue)",
            "the conjunctive swap rule and the continuous MemAvailable sampler",
            "the shared-card device-free admission (must refuse when the card "
            "holds a foreign allocation)",
            "device-law homogeneity filter (an igd 128 / it 20 point is refused)",
            "the imbalance margin reaching BOTH the guard and the rung derivation",
            "the delegated build path (must raise without a config capacity)",
            "memmap precondition for cuVS inputs (must raise on an ndarray)",
            "signal-free abort policy (must raise on SIGTERM)",
            "R0216 fuzzy law on a tiny memmapped substrate",
        ],
    }


def _source_size_manifest(queue_root: str) -> str:
    corpora: dict[str, Any] = {}
    for corpus, _rows in COMPOSITION:
        sizes: dict[str, int] = {}
        root = os.path.join(EMB, corpus, "train")
        for name in sorted(os.listdir(root)):
            if not name.endswith(".npy") or name.endswith(".tmp.npy"):
                continue
            path = os.path.join(root, name)
            sizes[os.path.relpath(path, EMB)] = os.path.getsize(path)
        if not sizes:
            raise RuntimeError(f"R0238 found no shards for {corpus}")
        corpora[corpus] = {"shard_sizes": sizes}
    path = os.path.join(queue_root, "source-size-manifest.json")
    atomic_write_new_json(
        path, {"schema": "round0238-source-size-manifest-v1", "corpora": corpora},
        immutable=True,
    )
    return path


#: The sealed manifest each node produces, under a queue's `artifacts/` dir.
INHERITABLE: dict[str, tuple[str, str]] = {
    "reachability": (REACHABILITY_CAPABILITY, "reachability.json"),
    "substrate": (SUBSTRATE_CAPABILITY, "substrate.json"),
    "truth": (TRUTH_CAPABILITY, "probe-k15-truth.json"),
    "ladder": (LADDER_CAPABILITY, "build-ladder.json"),
}
#: The node each inheritable artifact would otherwise be produced by.
INHERIT_NODE = {
    "reachability": "reachability_100000k",
    "substrate": "assemble_100000k",
    "truth": "truth_probe_100000k",
    "ladder": "ladder_100000k",
}


def _prior_gpu_wall(queue_root: str | None) -> float:
    """GPU wall every earlier attempt of THIS round has already charged.

    Read from the earlier queue's terminal receipt, so the wall-budget guard in
    the ladder node is sized against the round's real remaining cap rather than
    against a fresh 6.0 hours (review-0224: charge every attempt).
    """
    if not queue_root:
        return 0.0
    path = os.path.join(queue_root, "runner-terminal.json")
    if not os.path.exists(path):
        raise RuntimeError(f"R0238 --inherit-from: no terminal receipt at {path}")
    with open(path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    if str(receipt.get("round_id")) != ROUND_ID:
        raise RuntimeError("R0238 --inherit-from: terminal receipt is another round")
    return float(receipt.get("gpu_wall_s") or 0.0) + float(
        receipt.get("prior_attempt_gpu_wall_s") or 0.0
    )


def _discover_inheritable(queue_root: str) -> dict[str, dict[str, Any]]:
    """Sealed manifests an earlier attempt of THIS round already produced."""
    found: dict[str, dict[str, Any]] = {}
    for key, (capability, name) in INHERITABLE.items():
        path = os.path.join(queue_root, "artifacts", capability, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            sealed = json.load(handle)
        if str(sealed.get("round_id")) != ROUND_ID:
            raise RuntimeError(
                f"R0238 --inherit-from: {path} is round {sealed.get('round_id')}, "
                "not this round"
            )
        found[key] = expected_input_signature(path)
    if not found:
        raise RuntimeError(
            f"R0238 --inherit-from: no sealed R0238 manifest under {queue_root}"
        )
    return found


def prepare_round0238(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    inherit_from: str | None = None,
    include_qualify: bool = True,
) -> str:
    """Build the queue.

    `inherit_from` is an earlier attempt's queue root. Every sealed manifest it
    already produced is bound at its FULL signature and the node that would have
    produced it is dropped, so a correction after a setup-class defect costs the
    failed node and nothing else.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0238 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)

    inherited = {
        "parent_substrate": R0237_SUBSTRATE_MANIFEST,
        "r0237_ladder": R0237_LADDER,
        "r0237_reachability": R0237_REACHABILITY,
        "r0236_ladder": R0236_LADDER,
        "r0235_ladder": R0235_LADDER,
        "r0233_ladder": R0233_LADDER,
        "r0229_sweep": R0229_SWEEP,
        "r0229_arm": R0229_ARM,
        "r0229_reachability": R0229_REACHABILITY,
    }
    signatures = {}
    for key, path in inherited.items():
        if not os.path.exists(path):
            raise RuntimeError(f"R0238 inherited input absent: {key} at {path}")
        signatures[key] = expected_input_signature(path)

    reused: dict[str, dict[str, Any]] = (
        _discover_inheritable(inherit_from) if inherit_from else {}
    )
    prior_gpu_wall_s = _prior_gpu_wall(inherit_from)
    reused_substrate = reused.get("substrate")

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0238 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    manifest_path = _source_size_manifest(queue_root)

    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
        expected_input_signature(manifest_path),
        *signatures.values(),
        *reused.values(),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    cache_root = os.path.join(ROUND_ROOT, "child-cache")
    scratch_root = os.path.join(ROUND_ROOT, "spill-scratch")

    reach_dir = os.path.join(artifacts, REACHABILITY_CAPABILITY)
    reach_manifest = os.path.join(reach_dir, "reachability.json")
    substrate_dir = os.path.join(artifacts, SUBSTRATE_CAPABILITY)
    substrate_manifest = os.path.join(substrate_dir, "substrate.json")
    truth_dir = os.path.join(artifacts, TRUTH_CAPABILITY)
    truth_manifest = os.path.join(truth_dir, "probe-k15-truth.json")
    ladder_dir = os.path.join(artifacts, LADDER_CAPABILITY)
    ladder_manifest = os.path.join(ladder_dir, "build-ladder.json")
    graph_dir = os.path.join(artifacts, GRAPH_CAPABILITY)

    intra = lambda path: {"kind": "file", "canonical_path": path}  # noqa: E731
    policy = {"gpu_required": True, "training_performed": False, "cpu_heavy": False}

    #: Downstream nodes read the substrate either from this queue's own assemble
    #: node or, on a correction, from the sealed manifest of the attempt that
    #: already earned it. Both forms carry `canonical_path`; the inherited one
    #: additionally carries `sha256`, so it is bound MORE tightly, not less.
    substrate_reference = reused_substrate or intra(substrate_manifest)
    reach_reference = reused.get("reachability") or intra(reach_manifest)
    truth_reference = reused.get("truth") or intra(truth_manifest)
    ladder_reference = reused.get("ladder") or intra(ladder_manifest)
    downstream_deps = [] if reused_substrate else ["assemble_100000k"]

    def _deps(*names: str) -> list[str]:
        """Depend only on nodes this queue actually runs."""
        return [
            name for name in names
            if name not in {INHERIT_NODE[key] for key in reused}
        ]

    jobs: list[dict[str, Any]] = []
    reachability_job: dict[str, Any] | None = None
    if "reachability" not in reused:
        reachability_job = {
            "id": "reachability_100000k", "action": REACHABILITY_ACTION,
            "handler_module": "experiments.round0238_nodes",
            "handler_callable": "run_job",
            "deps": _deps("truth_probe_100000k"),
            "outputs": [reach_dir],
            "done_marker": os.path.join(
                artifacts, "reachability_100000k.done.json"
            ),
            "expected_inputs": expected_inputs,
            "p90_wall_s": REACHABILITY_P90_WALL_S,
            "capability": REACHABILITY_CAPABILITY,
            "substrate_manifest": substrate_reference,
            "truth_reference": truth_reference,
            "r0237_reachability": signatures["r0237_reachability"],
            "cache_root": cache_root,
            "node_policy": dict(policy),
        }
    if not reused_substrate:
        jobs.append({
            "id": "assemble_100000k", "action": ASSEMBLE_ACTION,
            "handler_module": "experiments.round0238_nodes",
            "handler_callable": "run_job", "deps": [],
            "outputs": [substrate_dir],
            "done_marker": os.path.join(artifacts, "assemble_100000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": ASSEMBLE_P90_WALL_S,
            "capability": SUBSTRATE_CAPABILITY,
            "source_size_manifest": expected_input_signature(manifest_path),
            "parent_substrate_manifest": signatures["parent_substrate"],
            "node_policy": {**policy, "cpu_heavy": True},
        })
    if "truth" not in reused:
        jobs.append({
            "id": "truth_probe_100000k", "action": TRUTH_ACTION,
            "handler_module": "experiments.round0238_nodes",
            "handler_callable": "run_job", "deps": list(downstream_deps),
            "outputs": [truth_dir],
            "done_marker": os.path.join(artifacts, "truth_probe_100000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": TRUTH_P90_WALL_S,
            "capability": TRUTH_CAPABILITY,
            "substrate_manifest": substrate_reference,
            "probe_rows": TRUTH_PROBE_ROWS,
            "probe_seed": TRUTH_PROBE_SEED,
            "node_policy": dict(policy),
        })
    if reachability_job is not None:
        jobs.append(reachability_job)
    if "ladder" not in reused:
        jobs.append({
            "id": "ladder_100000k", "action": LADDER_ACTION,
            "handler_module": "experiments.round0238_nodes",
            "handler_callable": "run_job", "deps": _deps("truth_probe_100000k"),
            "outputs": [ladder_dir],
            "done_marker": os.path.join(artifacts, "ladder_100000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": LADDER_P90_WALL_S,
            "capability": LADDER_CAPABILITY,
            "substrate_manifest": substrate_reference,
            "r0229_sweep": signatures["r0229_sweep"],
            "r0229_arm": signatures["r0229_arm"],
            "r0233_ladder": signatures["r0233_ladder"],
            "r0235_ladder": signatures["r0235_ladder"],
            "r0236_ladder": signatures["r0236_ladder"],
            "r0237_ladder": signatures["r0237_ladder"],
            "scratch_root": scratch_root,
            "cache_root": cache_root,
            "build_timeout_s": BUILD_TIMEOUT_S,
            "gpu_budget_remaining_s": float(
                GPU_HOURS_CAP * 3600.0 - prior_gpu_wall_s
            ),
            "qualify_reserve_s": QUALIFY_P90_WALL_S,
            "node_policy": dict(policy),
        })
    if include_qualify:
        jobs.append({
            "id": "qualify_100000k", "action": QUALIFY_ACTION,
            "handler_module": "experiments.round0238_nodes",
            "handler_callable": "run_job",
            "deps": _deps("ladder_100000k", "reachability_100000k"),
            "outputs": [graph_dir],
            "done_marker": os.path.join(artifacts, "qualify_100000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": QUALIFY_P90_WALL_S,
            "capability": GRAPH_CAPABILITY,
            "substrate_manifest": substrate_reference,
            "truth_reference": truth_reference,
            "ladder_reference": ladder_reference,
            "reachability_reference": reach_reference,
            "r0229_reachability": signatures["r0229_reachability"],
            "r0233_ladder": signatures["r0233_ladder"],
            "r0235_ladder": signatures["r0235_ladder"],
            "r0236_ladder": signatures["r0236_ladder"],
            "r0237_ladder": signatures["r0237_ladder"],
            "probe_rows": TRUTH_PROBE_ROWS,
            "probe_seed": TRUTH_PROBE_SEED,
            "node_policy": dict(policy),
        })

    queue = _base_manifest(
        round_id=ROUND_ID, release_sha=release_sha, round_file=ROUND_FILE,
        queue_root=queue_root, gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu", gpu=True,
    )
    queue.update({
        "schema": "round0238-100000k-nested-rung-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-graph",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-50000k-nested-substrate-and-reserves-v1",
            "minilm-mixed-50000k-cluster-spill-build-ladder-v1",
            "minilm-mixed-25000k-cluster-spill-high-c-reachability-v1",
            "minilm-mixed-cluster-spill-s8-imbalance-50m-five-seed-v1",
            "minilm-mixed-25000k-cluster-spill-build-ladder-v1",
            "minilm-mixed-cluster-spill-s8-imbalance-replicate-grid-v1",
        ],
        "capabilities_produced": [
            REACHABILITY_CAPABILITY, SUBSTRATE_CAPABILITY, TRUTH_CAPABILITY,
            LADDER_CAPABILITY, GRAPH_CAPABILITY, IMBALANCE_CAPABILITY,
            IO_CAPABILITY,
        ],
        "prior_attempt_gpu_wall_s": prior_gpu_wall_s,
        "gpu_budget_remaining_s": float(GPU_HOURS_CAP * 3600.0 - prior_gpu_wall_s),
        "inherited_artifacts": sorted(reused),
        "inherited_from": inherit_from,
        "qualify_included": bool(include_qualify),
        "qualify_omitted_reason": (
            None if include_qualify else
            "the wall-budget guard refuses the c = 400 build a priori inside "
            "the round's remaining GPU cap, so no graph exists to qualify. "
            "Registered in the round's 2026-08-09 addendum; running the node "
            "would raise rather than measure anything."
        ),
        "substrate_inherited": bool(reused_substrate),
        "substrate_reference": substrate_reference,
        "inheritance_note": (
            "review-0237-01 made this the STANDING rule and said to stop "
            "re-litigating it per round: a correction queue binds every sealed "
            "manifest the earlier attempt already produced, by full sha256, "
            "and drops the node that would have produced it. At 100,000,000 "
            "rows the substrate alone is 153.6 GB and the build cell alone is "
            "~3.9 GPU-h against a 6.0 cap, so recomputing either after a "
            "setup-class defect in a later node cannot fit. /data holds about "
            "437 GB and this round needs about 236 GB of it, so a correction "
            "queue must also RECLAIM the failed attempt's orphaned bulk "
            "outputs before it launches - review-0237-01 noted R0237 gained "
            "76.8 GB by inheriting and gave 18 GB back by duplicating the "
            "graph. review-0233-01 D7 prefers every released capability to "
            "come from a queue that terminated 'succeeded'; the round file "
            "registers this trade and the result states which queue each "
            "artifact came from."
        ),
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            "total": sum(
                float(job["p90_wall_s"]) for job in jobs
            )
        },
        "scientific_contract": {
            "question": (
                "can the FINAL rung be assembled so that it CONTAINS rung 4's "
                "50,000,000 training rows while still spanning every corpus "
                "(including a code corpus whose pool was extended under a "
                "registered selection-law change), does cluster-spill-nnd at "
                "s = 8 and c = 400 - a partition no graph in this program has "
                "ever been built at - reproduce brute-force truth on a "
                "registered uniform probe over 100,000,000 rows with no "
                "edgeless row in either direction, what does the first rung on "
                "the far side of the page-cache boundary actually cost in I/O, "
                "and does this round's own measured imbalance at 100,000,000 "
                "rows hold the 73.83% of tolerance the prediction promised "
                "against a 25M->50M drift that ran ADVERSE at this c?"
            ),
            "rows": ROWS,
            "parent_rows": PARENT_ROWS,
            "reserve_rows": RESERVE_ROWS,
            "reserve_source": (
                "inherited verbatim through R0237, R0236 and R0235 from R0233"
            ),
            "dimension": DIMENSION,
            "spill": SPILL,
            "nn_descent_setting": NN_DESCENT_SETTING,
            "selection_law": SELECTION_LAW,
            "selection_candidates": list(SELECTION_CANDIDATES),
            "c_build_min": C_BUILD_MIN,
            "c_build_min_note": C_BUILD_MIN_NOTE,
            "imbalance_probe_clusters": list(IMBALANCE_PROBE_CLUSTERS),
            "imbalance_probe_rows": list(IMBALANCE_PROBE_ROWS),
            "imbalance_replicate_seeds": list(IMBALANCE_REPLICATE_SEEDS),
            "primary_imbalance_seed": PRIMARY_IMBALANCE_SEED,
            "replicate_note": REPLICATE_NOTE,
            "imbalance_margin_note": MARGIN_NOTE,
            "reachability_rows": REACHABILITY_ROWS,
            "reachability_clusters": list(REACHABILITY_CLUSTERS),
            "reachability_concern_floor": REACHABILITY_CONCERN_FLOOR,
            "reachability_note": REACHABILITY_NOTE,
            "hundred_m_candidates": list(HUNDRED_M_CANDIDATES),
            "hundred_m_rule": HUNDRED_M_RULE,
            "code_pool_extension": CODE_POOL_EXTENSION,
            "code_pool_extension_uniformity": POOL_EXTENSION_UNIFORMITY,
            "code_pool_rows": CODE_POOL_ROWS,
            "code_pool_shards": CODE_POOL_SHARDS,
            "adverse_drift_note": ADVERSE_DRIFT_NOTE,
            "prediction_imbalance_at_c400_from_50m": (
                PREDICTION_IMBALANCE_AT_C400
            ),
            "prediction_tolerance_at_c400": PREDICTION_TOLERANCE_AT_C400,
            "declined_derisk_build": DECLINED_DERISK_NOTE,
            "recall_population": RECALL_POPULATION,
            "structural_population": STRUCTURAL_POPULATION,
            "truth_method": TRUTH_METHOD,
            "truth_probe_rows": TRUTH_PROBE_ROWS,
            "truth_probe_seed": TRUTH_PROBE_SEED,
            "truth_affordability": TRUTH_AFFORDABILITY_NOTE,
            "zero_recall_forensic_note": ZERO_RECALL_FORENSIC_NOTE,
            "io_regime_note": IO_REGIME_NOTE,
            "floors": {
                "tie_aware_mean": RECALL_MEAN_FLOOR,
                "tie_aware_p10": RECALL_P10_FLOOR,
                "zero_degree_rows_in_and_out": 0,
                "shard_coverage_per_corpus": 0.999,
                "nested_parent_rows_missing": 0,
                "reserve_training_intersection_rows": 0,
                "increment_reserve_intersection_rows": 0,
            },
            "safety": (
                "every buffer handed to cuVS is a read-only np.memmap, asserted "
                "on the substrate view and on every intermediate per-cluster "
                "spill file; the abort path is an in-band cooperative flag and "
                "no signal is ever delivered to a build process; the build path "
                "is R0233's reviewed script reached through R0235's reviewed "
                "one-constant rebind and R0236's checked delegation, with no "
                "new code inside the child; free device memory is read "
                "immediately before the cell and the cell is refused if the "
                "predicted charge does not fit what the shared card can give; "
                "any foreign compute process is recorded and never touched"
            ),
            "no_training": True,
            "no_gate_registered": True,
            "no_adoption_claimed": True,
            "no_map_quality_claimed": True,
            "builds_the_100m_rung": True,
        },
    })
    queue_path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(queue_path, queue, immutable=True)
    return queue_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    parser.add_argument(
        "--inherit-from", default=None,
        help=(
            "an earlier R0238 queue root; every sealed manifest it produced is "
            "bound by sha256 and its node is dropped"
        ),
    )
    parser.add_argument(
        "--no-qualify", action="store_true",
        help=(
            "omit the qualification node. Only correct when the wall-budget "
            "guard refuses the build a priori, so no graph exists to qualify; "
            "registered in the round file when used"
        ),
    )
    args = parser.parse_args(argv)
    path = prepare_round0238(
        release_sha=args.release_sha, queue_root=args.queue_root,
        inherit_from=args.inherit_from,
        include_qualify=not args.no_qualify,
    )
    print(json.dumps({"queue": path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
