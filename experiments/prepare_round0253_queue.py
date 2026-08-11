#!/usr/bin/env python3
"""Prepare, but never launch, the R0253 queue.

Three nodes, in this order:

1. `hookinstall_0253` (CPU) — the stop hook, installed as the first statement of
   every registered ladder-train node entry, proved on the entry path the runner
   dispatches through rather than asserted.
2. `writepath_0253` (CPU) — the un-polled write path review-0252-01 §H measured
   at `144.50039366001147` s `= 57.55x` the ceiling and no census could see,
   instrumented and re-measured at four real sizes up to `153,600,000,000` B,
   plus the sibling audit.
3. `cudasetup_0253` (GPU) — the 100M-rung setup gap on a node holding a live
   CUDA context, which is the case that cannot be signalled.

**This queue registers nothing.** No floor, no estimator, no gate, no map. Every
artifact is a measurement.
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
from basemap.round0216_minilm_2m_substrate import CAPABILITY as R0216_CAPABILITY
from basemap.round0247_registry import registered_value, registry_fingerprint
from basemap.round0250_seed_extension_n16 import (
    GRAPH_CAPABILITY,
    GRAPH_K,
    GRAPH_SCHEMA,
    GRAPH_SOURCE_ROUND_ID,
    ROWS,
    SEALED_DIRECTED_EDGES,
    TEMPLATE_SEED,
)
from basemap.round0250_panel_n16 import DIMENSION
from basemap.round0252_stoppability import (
    HASH_CAPABILITY as R0252_HASH_CAPABILITY,
    TARGET_DIMENSION,
    TARGET_ROWS,
    TARGET_SUBSTRATE_BYTES,
)
from basemap.round0253_stop_hooks import (
    CUDA_CAPABILITY,
    HOOKED_MODULES,
    INSTALL_CAPABILITY,
    NOT_A_FAMILY_CELL,
    REGISTERED_ENTRIES,
    THE_INSTRUMENT_IS_DEFEATABLE,
    WRITE_CAPABILITY,
    assert_every_entry_installs,
)
from basemap.round0253_write_path import assert_write_loop_polls
from experiments.round0253_nodes import (
    CUDA_ACTION,
    INSTALL_ACTION,
    WRITE_ACTION,
    WRITE_LADDER_BYTES,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ID = "0253"
ROUND_ROOT = "/data/latent-basemap/runs/round-0253"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0253-2026-08-11.md")

R0216_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    f"{R0216_CAPABILITY}"
)
GRAPH_MANIFEST = os.path.join(R0216_ARTIFACTS, "substrate-graph.json")
R0252_HASH = (
    f"/data/latent-basemap/runs/round-0252/queue-attempt-2/artifacts/"
    f"{R0252_HASH_CAPABILITY}/chunked-integrity-hash.json"
)

#: Peak transient `/data`: one file at a time, deleted before the next.
PEAK_SCRATCH_BYTES = max(WRITE_LADDER_BYTES)
DISK_RESERVE_BYTES = 40 << 30

#: Registered cap. The only GPU node writes and hashes one 153.6 GB file and
#: gathers 200,000 rows, i.e. ~7 min; 1.0 leaves room for one setup-class retry
#: inside the mandate's 3.0 GPU-h ceiling.
GPU_HOURS_CAP = 1.0
INSTALL_P90_WALL_S = 900.0
WRITE_P90_WALL_S = 5_400.0
CUDA_P90_WALL_S = 2_400.0


def _issued_round(release_sha: str) -> tuple[dict[str, Any], list[str]]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor", base_commit, release_sha],
        check=False,
    ).returncode == 0
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or not descendant
    ):
        raise RuntimeError("R0253 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0253 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _upstream_review_state(required: list[str]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    contingent: list[str] = []
    for round_id in required:
        reviews = []
        for path in sorted(glob.glob(os.path.join(LAB_ROOT, f"review-{round_id}-*.md"))):
            frontmatter = _frontmatter(path)
            reviews.append({
                "file": os.path.basename(path),
                "status": frontmatter.get("status"),
                "sha256": expected_input_signature(path)["sha256"],
            })
        accepted = [item for item in reviews if item["status"] == "accepted"]
        state[round_id] = {"reviews_present": reviews, "accepted_reviews": len(accepted)}
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


def _disk_headroom() -> dict[str, Any]:
    """Refuse at prepare time if the largest rung will not fit with slack.

    R0252's first attempt died because three rungs were on `/data` at once. Here
    both scratch-writing nodes create one file, measure it and delete it before
    the next, so peak is the LARGEST rung, and the two nodes never overlap
    because the runner is sequential.
    """
    free = _free_bytes("/data")
    if free < PEAK_SCRATCH_BYTES + DISK_RESERVE_BYTES:
        raise RuntimeError(
            f"R0253 needs {PEAK_SCRATCH_BYTES + DISK_RESERVE_BYTES} B free on "
            f"/data for the 100M-scale rung; {free} B available"
        )
    return {
        "free_bytes_at_prepare": free,
        "peak_scratch_bytes": PEAK_SCRATCH_BYTES,
        "reserve_bytes": DISK_RESERVE_BYTES,
        "one_file_at_a_time": True,
        "the_peak_is_the_largest_rung_not_their_sum": (
            PEAK_SCRATCH_BYTES == max(WRITE_LADDER_BYTES)
            and PEAK_SCRATCH_BYTES < sum(WRITE_LADDER_BYTES)
        ),
        "scratch_is_on_data_not_root": True,
    }


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0253 release checkout differs from requested release")
    basetemp = "/data/tmp/pytest-r0253-smoke"
    tmpdir = "/data/tmp/pytest-r0253-smoke-tmp"
    command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        f"--basetemp={basetemp}", "tests/test_round0253_contract.py",
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
    #: `subprocess.run(..., timeout=N)` as `Popen.kill()`, a hidden SIGKILL, and
    #: `plan-minilm-100m-v2.md` makes purging it binding before any GPU round.
    completed = subprocess.run(
        command, cwd=RELEASE_ROOT, env=environment,
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0253 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return prompt_contract.seal({
        "schema": "round0253-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cuda_hidden": True,
        "returncode": completed.returncode,
        "basetemp": basetemp,
        "tmpdir": tmpdir,
        "stdout_tail": completed.stdout.strip().splitlines()[-5:],
    })


def _sealed_graph() -> tuple[dict[str, Any], dict[str, Any], int]:
    signature = expected_input_signature(GRAPH_MANIFEST)
    manifest = prompt_contract.read_sealed(
        GRAPH_MANIFEST, label="R0216 sealed substrate+graph receipt"
    )
    if (
        manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("round_id") != GRAPH_SOURCE_ROUND_ID
        or manifest.get("capability") != GRAPH_CAPABILITY
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or int(manifest.get("k", -1)) != GRAPH_K
    ):
        raise RuntimeError("R0253 sealed R0216 substrate+graph contract changed")
    edges = int(manifest.get("directed_edge_count", 0)) or int(
        (manifest.get("graph_checks") or {}).get("directed_edges", 0)
    )
    if edges != SEALED_DIRECTED_EDGES:
        raise RuntimeError(
            f"R0253 sealed graph reports {edges} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    return signature, manifest, edges


def _sealed_r0252_hash() -> dict[str, Any]:
    """R0252's sealed hash artifact — the source of the `write_wall_s` this round
    re-measures.  Bound by signature so the comparison has a provenance."""
    receipt = prompt_contract.read_sealed(R0252_HASH, label="R0252 sealed hash node")
    if receipt.get("round_id") != "0252":
        raise RuntimeError("R0252 sealed hash receipt contract changed")
    return expected_input_signature(R0252_HASH)


def prepare_round0253(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    only_nodes: tuple[str, ...] | None = None,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0253 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    graph_manifest_signature, graph_manifest, edges = _sealed_graph()
    substrate_signature = dict(graph_manifest["substrate"])
    graph_signature = dict(graph_manifest["graph"])
    provenance_signature = dict(graph_manifest["provenance"])
    r0252_hash = _sealed_r0252_hash()
    review_state = _upstream_review_state(list(required_reviews))
    headroom = _disk_headroom()
    install_audit = assert_every_entry_installs()
    write_guard = assert_write_loop_polls()

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0253 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)

    shared_inputs = _dedupe([
        round_signature,
        graph_manifest_signature,
        substrate_signature,
        graph_signature,
        provenance_signature,
        expected_input_signature(smoke_path),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    jobs: list[dict[str, Any]] = []
    p90: dict[str, float] = {}

    install_node = "hookinstall_0253"
    jobs.append({
        "id": install_node,
        "action": INSTALL_ACTION,
        "handler_module": "experiments.round0253_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, INSTALL_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{install_node}.done.json"),
        "expected_inputs": list(shared_inputs),
        "p90_wall_s": INSTALL_P90_WALL_S,
        "registered_entries": [
            {"module": module, "function": function}
            for module, function in REGISTERED_ENTRIES
        ],
        "hooked_modules": list(HOOKED_MODULES),
        "upstream_review_state": review_state,
        "node_policy": {"gpu_required": False, "training_performed": False, "cpu_heavy": False},
    })
    p90[install_node] = INSTALL_P90_WALL_S

    write_node = "writepath_0253"
    jobs.append({
        "id": write_node,
        "action": WRITE_ACTION,
        "handler_module": "experiments.round0253_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, WRITE_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{write_node}.done.json"),
        "expected_inputs": _dedupe([*shared_inputs, r0252_hash]),
        "p90_wall_s": WRITE_P90_WALL_S,
        "write_ladder_bytes": list(WRITE_LADDER_BYTES),
        "disk_headroom": headroom,
        "r0252_hash_receipt": r0252_hash,
        "upstream_review_state": review_state,
        "node_policy": {"gpu_required": False, "training_performed": False, "cpu_heavy": True},
    })
    p90[write_node] = WRITE_P90_WALL_S

    cuda_node = "cudasetup_0253"
    jobs.append({
        "id": cuda_node,
        "action": CUDA_ACTION,
        "handler_module": "experiments.round0253_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, CUDA_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{cuda_node}.done.json"),
        "expected_inputs": list(shared_inputs),
        "p90_wall_s": CUDA_P90_WALL_S,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": graph_manifest_signature,
        "template_seed": TEMPLATE_SEED,
        "disk_headroom": headroom,
        "upstream_review_state": review_state,
        "node_policy": {"gpu_required": True, "training_performed": False, "cpu_heavy": True},
    })
    p90[cuda_node] = CUDA_P90_WALL_S

    if only_nodes is not None:
        wanted = set(only_nodes)
        unknown = wanted - {job["id"] for job in jobs}
        if unknown:
            raise RuntimeError(f"R0253 has no node(s) {sorted(unknown)}")
        jobs = [job for job in jobs if job["id"] in wanted]
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
        "schema": "round0253-stop-hook-install-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [GRAPH_CAPABILITY, R0252_HASH_CAPABILITY],
        "capabilities_produced": [
            capability
            for job_id, capability in (
                (install_node, INSTALL_CAPABILITY),
                (write_node, WRITE_CAPABILITY),
                (cuda_node, CUDA_CAPABILITY),
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
                "review-0252-01 §A: `set_abort_poll` is called from exactly one "
                "file in the repository, R0252's own measurement node, so the fix "
                "R0252 measured is installed nowhere a ladder train runs. With the "
                f"install now the FIRST statement of all {len(REGISTERED_ENTRIES)} "
                "registered ladder-train node entries, does the entry path the "
                "runner dispatches through actually carry the hook -- proved by "
                "executing it, by reading the release's own AST, and by requiring "
                "the installed hook to raise out of a real "
                "`expected_input_signature` call?"
            ),
            "question_b": (
                "review-0252-01 §H: `_write_sized_file` has no abort read at all "
                "and the gate is not constructed until after it runs, so R0252's "
                "own cooperative stop took 93.135091 s = 37.09x the ceiling and "
                "the shipped node still pays 144.50039366001147 s = 57.55x on its "
                "top rung, none of which appeared in any census because code that "
                "never polls emits no gap. What is the widest abort-read gap of a "
                f"sized write at four real sizes up to {TARGET_SUBSTRATE_BYTES} B "
                "in an unpolled arm and a polled arm; how long does a stop planted "
                "mid-write take; and how many sibling write loops in the release "
                "are in the same class?"
            ),
            "question_c": (
                "roundreport's poll-coverage table reads `not instrumented` for "
                "every R0252 node because no node emits a covered span. With each "
                "node emitting `observed_span_s` -- the union in wall time of the "
                "intervals its poll record actually covers -- what fraction of "
                "each node is the census maximum computed over?"
            ),
            "question_d": (
                "review-0252-01 §J.3: the setup gap at the 100M rung on a node "
                "holding a live CUDA context is unmeasured, and that is the case "
                "that cannot be signalled. On such a node -- CUDA presence "
                "verified from /proc/self/maps, not from an environment variable "
                f"-- what are the write, hash, read-only memmap open, "
                "sampler-order gather and model-allocation walls at a real "
                f"{TARGET_ROWS} x {TARGET_DIMENSION} fp32 substrate size, and what "
                "is the widest gap between abort reads across all of them?"
            ),
            "population": "sealed R0216 2,000,000-row mixed MiniLM substrate (config only) plus synthetic 100M-scale files",
            "sealed_directed_edges": edges,
            "write_ladder_bytes": list(WRITE_LADDER_BYTES),
            "the_100m_figures_are_measured_not_projected": True,
            "target_rows": TARGET_ROWS,
            "target_dimension": TARGET_DIMENSION,
            "target_substrate_bytes": TARGET_SUBSTRATE_BYTES,
            "registered_entries": [
                f"{module}.{function}" for module, function in REGISTERED_ENTRIES
            ],
            "entry_install_audit_at_prepare": install_audit,
            "write_loop_poll_guard_at_prepare": write_guard,
            "hooked_modules": list(HOOKED_MODULES),
            "registered_max_poll_spacing_s": registered_value("r0246_max_poll_spacing_s"),
            "disk_headroom": headroom,
            "registers_nothing": NOT_A_FAMILY_CELL,
            "gate_registered": False,
            "floors_registered": 0,
            "registry_fingerprint": registry_fingerprint(),
            "registry_mutated": False,
            "guard_modules_edited": False,
            "science_modules_edited": (
                "None. `basemap/artifact_identity.py` and `basemap/panel_v2.py` "
                "are untouched by R0253 -- their R0252 hook is CALLED, not "
                "changed. The only edits to pre-existing files are one "
                "`install_stop_hooks(label=...)` line as the first statement of "
                "each registered node entry and the import it needs: 20 inserted "
                "lines, 0 deleted, across 6 experiments/*_nodes.py files. No "
                "metric, neighbour set, ordering, rounding, threshold, treatment "
                "or digest changes."
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
        "queue_manifest": prepare_round0253(
            release_sha=args.release_sha,
            queue_root=args.queue_root,
            only_nodes=only,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
