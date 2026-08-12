#!/usr/bin/env python3
"""Prepare, but never launch, the R0259 queue.

Two nodes, in this order:

1. `controls_0259` (CPU) — ten positive controls, each planting a defect the
   **shipped** path accepts; the container controls; the rung applicability
   controls in both directions; the numpy pairwise self-check; and the
   five-incompatibility census, evaluated against R0243's sealed manifest.
   Cheap, and it runs first so a defect in a guard is known before the probe
   touches 30 GB.
2. `train100m_0259` (GPU) — `run_train_100m`, the 100M rung's registered entry,
   exercised at the real rung on a node holding a live CUDA context, with a
   registered per-batch bound.

**This queue registers nothing.** No floor, no estimator, no gate, no map, no
model. Every artifact is a measurement or a control.

**Nothing is trained.** A ~10 h fit is out of scope by mandate, and the entry
refuses an unbounded probe rather than becoming one by omission.
"""
from __future__ import annotations

import argparse
import glob
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
from basemap import round0113_prompt_contrast as prompt_contract
from basemap import round0259_hundred_m as rung
from basemap.round0247_registry import registry_fingerprint
from basemap.round0253_stop_hooks import (
    NOT_A_FAMILY_CELL,
    THE_INSTRUMENT_IS_DEFEATABLE,
    registered_ceiling_s,
)
from basemap.round0254_dispatch import (
    SCOPE_MODULES,
    assert_derived_entries_install,
    dispatch_census,
    entry_tuples,
    gate_census,
    scope_residual,
)
from basemap.round0258_graph_load import EDGE_ARRAYS, EDGE_HEADER_PATH, SCAN_CHUNK_ELEMENTS
from basemap.round0259_hundred_m import (
    RUNG_100M,
    RUNGS,
    SUBSTRATE_100M_PATH,
    assert_chunk_loops_poll_v2,
    assert_pairwise_rule,
    assert_rung_applicability_controls,
    assert_structural_defect_controls_v2,
    assert_substrate_dimension,
)
from experiments.round0259_nodes import (
    BATCH_SIZE,
    CONTROLS_ACTION,
    CONTROLS_CAPABILITY,
    MIN_MEM_AVAILABLE_BYTES,
    NODE_ANON_BUDGET_BYTES,
    POS_RATIO,
    PROBE_BATCHES_FILE_BACKED,
    PROBE_BATCHES_RESIDENT,
    PROBE_BATCHES_SUBSTRATE,
    SUM_PROOF_LEAVES,
    TRAIN_ACTION,
    TRAIN_CAPABILITY,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ID = "0259"
ROUND_ROOT = "/data/latent-basemap/runs/round-0259"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0259-2026-08-12.md")

R0243_FUZZY_MANIFEST = rung.R0243_FUZZY_MANIFEST

DISK_RESERVE_BYTES = 20 << 30

GPU_HOURS_CAP = 0.5
CONTROLS_P90_WALL_S = 600.0
#: The probe reads `30,133,239,432` B of members, materialises `20,088,826,032` B
#: of endpoints, builds a `20,088,826,032` B float64 CDF and runs
#: `1000 + 120 + 60` bounded batches. R0258 measured the same reads at `18`-`38` s
#: per repetition; `1800` s is generous against that and is a p90, not a deadline.
TRAIN_P90_WALL_S = 1_800.0


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
        raise RuntimeError("R0259 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0259 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _upstream_review_state(required: list[str]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    contingent: list[str] = []
    for round_id in required:
        reviews = []
        for path in sorted(
            glob.glob(os.path.join(LAB_ROOT, f"review-{round_id}-*.md"))
        ):
            frontmatter = _frontmatter(path)
            reviews.append({
                "file": os.path.basename(path),
                "status": frontmatter.get("status"),
                "sha256": expected_input_signature(path)["sha256"],
            })
        accepted = [item for item in reviews if item["status"] == "accepted"]
        state[round_id] = {
            "reviews_present": reviews, "accepted_reviews": len(accepted)
        }
        if not accepted:
            contingent.append(round_id)
    return {
        "required_reviews": list(required),
        "by_round": state,
        "rounds_without_an_accepted_review": contingent,
        "claims_contingent_on": contingent,
    }


def _free_bytes(path: str) -> int:
    stat = os.statvfs(path)
    return int(stat.f_bavail) * int(stat.f_frsize)


def _the_artifacts_exist() -> dict[str, Any]:
    """Every path this round reads, stat-ed against R0243's own sealed block."""
    sealed = prompt_contract.read_sealed(
        R0243_FUZZY_MANIFEST, label="R0243 sealed fuzzy graph"
    )
    if sealed.get("schema") != RUNGS[RUNG_100M]["schema"]:
        raise RuntimeError("R0259: R0243's sealed schema is not the 100M rung's")
    outputs = sealed["outputs"]
    present: dict[str, Any] = {}
    for key, name in (("edges_sources", "sources"),
                      ("edges_targets", "targets"),
                      ("edges_weights", "weights")):
        declared = outputs[key]
        spec = EDGE_ARRAYS[name]
        if declared["canonical_path"] != spec["path"]:
            raise RuntimeError(f"R0259 {name} path disagrees with R0243's seal")
        if declared["sha256"] != spec["sha256"]:
            raise RuntimeError(f"R0259 {name} sha256 disagrees with R0243's seal")
        size = os.path.getsize(spec["path"])
        if size != int(declared["bytes"]):
            raise RuntimeError(
                f"R0259 {name} is {size} B, sealed at {declared['bytes']} B"
            )
        present[name] = {
            "canonical_path": spec["path"], "bytes": size,
            "sha256": declared["sha256"], "kind": "file",
        }
    present["header"] = {
        "canonical_path": EDGE_HEADER_PATH,
        "bytes": os.path.getsize(EDGE_HEADER_PATH),
        "sha256": outputs["edges_header"]["sha256"],
        "kind": "file",
    }
    substrate = assert_substrate_dimension()
    return {
        "schema": "round0259-artifact-presence-v1",
        "arrays": present,
        "substrate": substrate,
        "r0243_manifest": expected_input_signature(R0243_FUZZY_MANIFEST),
    }


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0259 release checkout differs from requested release")
    basetemp = "/data/tmp/pytest-r0259-smoke"
    tmpdir = "/data/tmp/pytest-r0259-smoke-tmp"
    command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        f"--basetemp={basetemp}",
        "tests/test_round0259_contract.py", "tests/test_edgelist_smoke.py",
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "TMPDIR": tmpdir,
    })
    os.makedirs(tmpdir, exist_ok=True)
    #: **No `timeout=` anywhere in this file, deliberately.** CPython implements
    #: `subprocess.run(..., timeout=N)` as `Popen.kill()`, a hidden SIGKILL.
    completed = subprocess.run(
        command, cwd=RELEASE_ROOT, env=environment,
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0259 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return prompt_contract.seal({
        "schema": "round0259-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cuda_hidden": True,
        "returncode": completed.returncode,
        "basetemp": basetemp,
        "tmpdir": tmpdir,
        "stdout_tail": completed.stdout.strip().splitlines()[-5:],
    })


def prepare_round0259(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    only_nodes: tuple[str, ...] | None = None,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0259 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    review_state = _upstream_review_state(list(required_reviews))
    presence = _the_artifacts_exist()

    free = _free_bytes("/data")
    if free < DISK_RESERVE_BYTES:
        raise RuntimeError(
            f"R0259 needs {DISK_RESERVE_BYTES} B free on /data; {free} B available"
        )

    census = dispatch_census()
    guard = assert_derived_entries_install(SCOPE_MODULES, census)
    gates = gate_census(entry_tuples(guard["derived"]))
    residual = scope_residual(census, SCOPE_MODULES)
    structural = assert_structural_defect_controls_v2()
    pairwise = assert_pairwise_rule()
    chunk_audit = assert_chunk_loops_poll_v2()
    applicability = assert_rung_applicability_controls()

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0259 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    presence_path = os.path.join(preflight, "artifact-presence.json")
    atomic_write_new_json(presence_path, prompt_contract.seal(presence), immutable=True)

    shared_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
        expected_input_signature(presence_path),
        presence["r0243_manifest"],
    ])
    train_inputs = _dedupe(list(shared_inputs) + [
        presence["arrays"][name]
        for name in ("sources", "targets", "weights", "header")
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    jobs: list[dict[str, Any]] = []
    p90: dict[str, float] = {}

    controls_node = "controls_0259"
    jobs.append({
        "id": controls_node,
        "action": CONTROLS_ACTION,
        "handler_module": "experiments.round0259_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, CONTROLS_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{controls_node}.done.json"),
        "expected_inputs": list(shared_inputs),
        "scratch": "/data/tmp/round0259-controls",
        "p90_wall_s": CONTROLS_P90_WALL_S,
        "scope_modules": list(SCOPE_MODULES),
        "structural_plants_at_prepare": structural["defects_planted"],
        "numeric_plants_at_prepare": pairwise["numeric_controls"]["defects_planted"],
        "chunk_loop_audit_at_prepare": chunk_audit["functions_checked"],
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": False, "training_performed": False, "cpu_heavy": False
        },
    })
    p90[controls_node] = CONTROLS_P90_WALL_S

    train_node = "train100m_0259"
    jobs.append({
        "id": train_node,
        "action": TRAIN_ACTION,
        "handler_module": "experiments.round0259_nodes",
        "handler_callable": "run_job",
        "deps": [controls_node],
        "outputs": [os.path.join(artifacts, TRAIN_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{train_node}.done.json"),
        "expected_inputs": list(train_inputs),
        "graph_manifest": R0243_FUZZY_MANIFEST,
        "p90_wall_s": TRAIN_P90_WALL_S,
        "rung": RUNG_100M,
        "rows": RUNGS[RUNG_100M]["rows"],
        "k": RUNGS[RUNG_100M]["k"],
        "dimension": RUNGS[RUNG_100M]["dimension"],
        "directed_edges": 2_511_103_254,
        "batch_size": BATCH_SIZE,
        "pos_ratio": POS_RATIO,
        "probe_batches_resident": PROBE_BATCHES_RESIDENT,
        "probe_batches_file_backed": PROBE_BATCHES_FILE_BACKED,
        "probe_batches_substrate": PROBE_BATCHES_SUBSTRATE,
        "sum_proof_leaves": list(SUM_PROOF_LEAVES),
        "scan_chunk_elements": SCAN_CHUNK_ELEMENTS,
        "declared_anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
        "min_mem_available_bytes": MIN_MEM_AVAILABLE_BYTES,
        "substrate": SUBSTRATE_100M_PATH,
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": True, "training_performed": False, "cpu_heavy": True
        },
    })
    p90[train_node] = TRAIN_P90_WALL_S

    if only_nodes is not None:
        wanted = set(only_nodes)
        unknown = wanted - {job["id"] for job in jobs}
        if unknown:
            raise RuntimeError(f"R0259 has no node(s) {sorted(unknown)}")
        jobs = [job for job in jobs if job["id"] in wanted]
        for job in jobs:
            job["deps"] = [dep for dep in job["deps"] if dep in wanted]
        p90 = {key: value for key, value in p90.items() if key in wanted}
    p90["total"] = sum(value for key, value in p90.items() if key != "total")

    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0259-hundred-m-entry-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1",
            "round0258-100m-graph-load-and-edge-prep-intervals-v1",
        ],
        "capabilities_produced": [
            capability
            for job_id, capability in (
                (controls_node, CONTROLS_CAPABILITY),
                (train_node, TRAIN_CAPABILITY),
            )
            if only_nodes is None or job_id in set(only_nodes)
        ],
        "correction_of": (None if only_nodes is None else QUEUE_ROOT),
        "correction_nodes": (None if only_nodes is None else list(only_nodes)),
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": p90,
        "scientific_contract": {
            "question_a": (
                "review-0258-01 §H.1: round0113_nodes.run_train is a ~2M, k=50, "
                "768-d, R0113-schema trainer, so gating it does not gate a 100M "
                "train. Counted from load_graph's source, how many distinct "
                "mismatches block a 100M config, and does a loader branch plus a "
                "rung-specific entry let a 100M config actually load and enter "
                "training?"
            ),
            "question_b": (
                "review-0258-01 §E falsified R0258's named irreducible residue by "
                "replicating numpy's pairwise split n2 = n//2 - n//2 % 8, and "
                "could not verify that rule across numpy versions. Does the "
                "chunked sum reproduce np.sum BITWISE over the real "
                "2,511,103,254 float64 at leaves 2^24 / 2^20 / 2^14, and does a "
                "self-check refuse a numpy on which it does not?"
            ),
            "question_c": (
                "review-0258-01 §H.2/§H.3 measured the fit's per-batch endpoint "
                "gather at 0.2625x the ceiling file-backed and ~4,000x slower "
                "than resident, and priced the remedy at 10.2 s / 18.7 GiB. With "
                "the endpoints actually materialised, what is the per-batch "
                "producer interval at 100M -- draw, gather, pin, H2D and the "
                "on-device negative draw -- and how does it compare to the "
                f"registered ceiling of {registered_ceiling_s()} s?"
            ),
            "question_d": (
                "review-0258-01 §D.3 planted five defects against R0258's "
                "chunk_loop_polls and three passed. Does a strengthened guard "
                "refuse all of them while an honest install still passes, and is "
                "each plant verified to be ACCEPTED by the shipped R0258 guard so "
                "the refusal is a demonstrated behaviour change?"
            ),
            "population": (
                "R0243's sealed 100M k15 fuzzy graph -- 2,511,103,254 directed "
                "edges over 100,000,000 rows at k = 15, three .npy members of "
                "10,044,413,144 B each -- R0238's 153,600,000,128 B substrate, "
                "and the release's own source"
            ),
            "rows": RUNGS[RUNG_100M]["rows"],
            "k": RUNGS[RUNG_100M]["k"],
            "dimension": RUNGS[RUNG_100M]["dimension"],
            "directed_edges": 2_511_103_254,
            "registered_ceiling_s": registered_ceiling_s(),
            "repetitions": 1,
            "acceptance_rule": (
                "No numerical outcome makes this round a failure. The round FAILS "
                "if a polled stage is not bitwise identical to the shipped one, "
                "if a planted defect passes a guard, if the SHIPPED R0258 guard "
                "does not accept a structural plant, if a wrong-shape container "
                "does not fail loudly, if a 2M config enters the 100M entry, or "
                "if the bulk .npz path changes."
            ),
            "the_100m_train_is_not_in_scope": (
                "no fit runs. run_train_100m refuses an unbounded probe, and the "
                "device-resident feature gather a fit needs cannot be constructed "
                "at this rung at all: X is 100,000,000 x 384, which at fp16 is "
                "76,800,000,000 B against a 33,679,736,832 B card. That ratio is "
                "measured by the node, not asserted here."
            ),
            "artifact_presence": presence,
            "controls_at_prepare": {
                "structural_plants": structural["defects_planted"],
                "structural_plants_accepted_by_the_shipped_r0258_guard": structural[
                    "defects_accepted_by_the_shipped_r0258_guard"],
                "numeric_plants": pairwise["numeric_controls"]["defects_planted"],
                "pairwise_checks_run": pairwise["checks_run"],
                "numpy_version_observed": pairwise["numpy_version_observed"],
                "numpy_version_verified": pairwise["numpy_version_verified"],
                "rung_manifests_checked": applicability["manifests_checked"],
            },
            "install_and_gate_at_prepare": {
                "entry_count": int(gates["entries_audited"]),
                "entries_that_construct_a_gate": int(
                    gates["entries_that_construct_a_gate"]
                ),
                "entries_that_both_install_and_gate": int(
                    gates["entries_that_both_install_and_gate"]
                ),
                "scope_residual": residual,
            },
            "registers_nothing": NOT_A_FAMILY_CELL,
            "gate_registered": False,
            "floors_registered": 0,
            "registry_fingerprint": registry_fingerprint(),
            "registry_mutated": False,
            "guard_modules_edited": False,
            "science_modules_edited": (
                "Two pre-existing files change. (1) "
                "`basemap/pumap/parametric_umap/datasets/edge_list_dataset.py`: "
                "`load_edge_arrays` gains a branch that dispatches to the streamed "
                "100M members. The discriminator is CONTENT -- a directory holding "
                "the four member files, or an .npz whose member set is exactly the "
                "scalar header keys -- so a bulk .npz carrying `sources` is never "
                "claimed and every rung at or below 50M takes the identical path "
                "it always has. `bulk_npz_is_not_claimed()` is the positive "
                "control and it runs in the release CPU smoke. (2) "
                "`basemap/round0254_dispatch.py`: one line adding "
                "`experiments.round0259_nodes` to SCOPE_MODULES. No metric, "
                "neighbour set, ordering, rounding, threshold, treatment or digest "
                "changes; `basemap/artifact_identity.py`, `basemap/panel_v2.py`, "
                "the registry and every guard module are untouched."
            ),
            "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
            "upstream_review_state": review_state,
            "evaluation_performed": False,
            "production_or_publishing": False,
        },
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    parser.add_argument("--only-nodes", default=None)
    args = parser.parse_args(argv)
    only = (
        tuple(item.strip() for item in args.only_nodes.split(",") if item.strip())
        if args.only_nodes
        else None
    )
    print(json.dumps({
        "queue_manifest": prepare_round0259(
            release_sha=args.release_sha,
            queue_root=args.queue_root,
            only_nodes=only,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
