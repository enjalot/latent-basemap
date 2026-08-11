#!/usr/bin/env python3
"""Prepare, but never launch, the R0258 queue.

Two nodes, in this order:

1. `controls_0258` (CPU) — the five structural plants and the five numeric
   plants this round's guards must refuse, the chunk-loop audit over every
   shipped polled stage, and the dispatch-derived install/gate census after
   `round0113_nodes.run_train` and `round0238_nodes.run_assemble` were given
   gates. Cheap, and it runs first so a defect in the guards is known before the
   long node touches 90 GB of reads.
2. `graphload_0258` (GPU) — the measurement. R0243's real 100M fuzzy graph,
   three arms, five stages, `n = 3`, on a node holding a live CUDA context.

**This queue registers nothing.** No floor, no estimator, no gate, no map, no
model. Every artifact is a measurement.

**Nothing is trained.** The GPU node creates a CUDA context and holds it because
that is the state a terminal cannot stop; it runs no fit and produces no map.
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
from basemap.round0258_graph_load import (
    ARMS,
    ARM_UNPOLLED_CONTROL,
    DIRECTED_EDGES,
    EDGE_ARRAYS,
    EDGE_HEADER_PATH,
    K,
    READ_CHUNK_BYTES,
    ROWS,
    SCAN_CHUNK_ELEMENTS,
    SHIPPED_ARMS,
    STAGES,
    arm_schedule,
    assert_chunk_loops_poll,
    assert_structural_defect_controls,
)
from experiments.round0258_nodes import (
    CONTROLS_ACTION,
    CONTROLS_CAPABILITY,
    GRAPHLOAD_ACTION,
    GRAPHLOAD_CAPABILITY,
    MIN_MEM_AVAILABLE_BYTES,
    NODE_ANON_BUDGET_BYTES,
    REPETITIONS,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ID = "0258"
ROUND_ROOT = "/data/latent-basemap/runs/round-0258"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0258-2026-08-11.md")

R0243_FUZZY_MANIFEST = (
    "/data/latent-basemap/runs/round-0243/queue/artifacts/"
    "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1/fuzzy-graph.json"
)

#: This round writes NOTHING large. It reads the graph and allocates host
#: memory; the only files it creates are two small JSON artifacts. The disk
#: check exists so the round cannot start on a full volume, not because it
#: needs room.
DISK_RESERVE_BYTES = 20 << 30

GPU_HOURS_CAP = 2.0
CONTROLS_P90_WALL_S = 600.0
#: Nine repetitions. Each reads `30,133,239,432` B cold and runs five stages
#: over `2,511,103,254` elements. R0252's cold sequential read on this box is
#: `1.2263` GB/s, so a cold 30 GB load alone is `~24.6` s; the scans and the f64
#: CDF are the rest. `5400` s is `~600` s per repetition, which is generous
#: against that arithmetic and is a p90, not a deadline.
GRAPHLOAD_P90_WALL_S = 5_400.0


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
        raise RuntimeError("R0258 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0258 round must declare its required reviews")
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


def _the_artifact_exists() -> dict[str, Any]:
    """The premise R0253 got wrong, checked at prepare time from the filesystem.

    R0253 §D3 recorded that R0238's substrate and R0240's k15 graph "no longer
    exist on this box". Every path this round reads is stat-ed here and its size
    compared to R0243's sealed `outputs` block, so the round cannot be issued on
    a premise that is false in either direction.
    """
    sealed = prompt_contract.read_sealed(
        R0243_FUZZY_MANIFEST, label="R0243 sealed fuzzy graph"
    )
    if int(sealed.get("directed_edges", -1)) != DIRECTED_EDGES:
        raise RuntimeError("R0243 sealed fuzzy graph contract changed")
    outputs = sealed["outputs"]
    present: dict[str, Any] = {}
    for key, name in (("edges_sources", "sources"),
                      ("edges_targets", "targets"),
                      ("edges_weights", "weights")):
        declared = outputs[key]
        spec = EDGE_ARRAYS[name]
        if declared["canonical_path"] != spec["path"]:
            raise RuntimeError(f"R0258 {name} path disagrees with R0243's seal")
        if declared["sha256"] != spec["sha256"]:
            raise RuntimeError(f"R0258 {name} sha256 disagrees with R0243's seal")
        if not os.path.isfile(spec["path"]):
            raise RuntimeError(
                f"R0258 cannot measure the 100M graph: {spec['path']} is absent"
            )
        size = os.path.getsize(spec["path"])
        if size != int(declared["bytes"]) or size != spec["bytes"]:
            raise RuntimeError(
                f"R0258 {name} is {size} B, sealed at {declared['bytes']} B"
            )
        present[name] = {
            "canonical_path": spec["path"],
            "bytes": size,
            "sha256": declared["sha256"],
            "kind": "file",
        }
    header_size = os.path.getsize(EDGE_HEADER_PATH)
    present["header"] = {
        "canonical_path": EDGE_HEADER_PATH,
        "bytes": header_size,
        "sha256": outputs["edges_header"]["sha256"],
        "kind": "file",
    }
    return {
        "schema": "round0258-artifact-presence-v1",
        "arrays": present,
        "r0243_manifest": expected_input_signature(R0243_FUZZY_MANIFEST),
        "what_r0253_said": (
            "R0253 §D3: 'R0238's 100,000,000-row substrate and R0240's k15 graph "
            "no longer exist on this box, and rebuilding them is ~7 GPU-h "
            "against this round's 1.0 cap.' Every byte above is present at "
            "R0243's sealed signature and this check refuses to issue the round "
            "if that stops being true."
        ),
    }


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0258 release checkout differs from requested release")
    basetemp = "/data/tmp/pytest-r0258-smoke"
    tmpdir = "/data/tmp/pytest-r0258-smoke-tmp"
    command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        f"--basetemp={basetemp}", "tests/test_round0258_contract.py",
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
            f"R0258 release CPU smoke failed:\n{completed.stdout}\n"
            f"{completed.stderr}"
        )
    return prompt_contract.seal({
        "schema": "round0258-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cuda_hidden": True,
        "returncode": completed.returncode,
        "basetemp": basetemp,
        "tmpdir": tmpdir,
        "stdout_tail": completed.stdout.strip().splitlines()[-5:],
    })


def prepare_round0258(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    only_nodes: tuple[str, ...] | None = None,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0258 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    review_state = _upstream_review_state(list(required_reviews))
    presence = _the_artifact_exists()

    free = _free_bytes("/data")
    if free < DISK_RESERVE_BYTES:
        raise RuntimeError(
            f"R0258 needs {DISK_RESERVE_BYTES} B free on /data; {free} B available"
        )

    census = dispatch_census()
    guard = assert_derived_entries_install(SCOPE_MODULES, census)
    gates = gate_census(entry_tuples(guard["derived"]))
    residual = scope_residual(census, SCOPE_MODULES)
    structural = assert_structural_defect_controls()
    chunk_audit = assert_chunk_loops_poll()

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0258 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    presence_path = os.path.join(preflight, "artifact-presence.json")
    atomic_write_new_json(presence_path, prompt_contract.seal(presence),
                          immutable=True)

    shared_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
        expected_input_signature(presence_path),
        presence["r0243_manifest"],
    ])
    graph_inputs = _dedupe(list(shared_inputs) + [
        presence["arrays"][name]
        for name in ("sources", "targets", "weights", "header")
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    jobs: list[dict[str, Any]] = []
    p90: dict[str, float] = {}

    controls_node = "controls_0258"
    jobs.append({
        "id": controls_node,
        "action": CONTROLS_ACTION,
        "handler_module": "experiments.round0258_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, CONTROLS_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{controls_node}.done.json"),
        "expected_inputs": list(shared_inputs),
        "p90_wall_s": CONTROLS_P90_WALL_S,
        "scope_modules": list(SCOPE_MODULES),
        "structural_controls_at_prepare": structural["defects_planted"],
        "chunk_loop_audit_at_prepare": chunk_audit["functions_checked"],
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": False, "training_performed": False, "cpu_heavy": False
        },
    })
    p90[controls_node] = CONTROLS_P90_WALL_S

    graph_node = "graphload_0258"
    jobs.append({
        "id": graph_node,
        "action": GRAPHLOAD_ACTION,
        "handler_module": "experiments.round0258_nodes",
        "handler_callable": "run_job",
        "deps": [controls_node],
        "outputs": [os.path.join(artifacts, GRAPHLOAD_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{graph_node}.done.json"),
        "expected_inputs": list(graph_inputs),
        "p90_wall_s": GRAPHLOAD_P90_WALL_S,
        "rows": ROWS,
        "k": K,
        "directed_edges": DIRECTED_EDGES,
        "repetitions": REPETITIONS,
        "arms": list(ARMS),
        "shipped_arms": list(SHIPPED_ARMS),
        "control_arm": ARM_UNPOLLED_CONTROL,
        "stages": list(STAGES),
        "schedule": [
            {"arm": arm, "repetition": repetition}
            for arm, repetition in arm_schedule(ARMS, REPETITIONS)
        ],
        "read_chunk_bytes": READ_CHUNK_BYTES,
        "scan_chunk_elements": SCAN_CHUNK_ELEMENTS,
        "declared_anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
        "min_mem_available_bytes": MIN_MEM_AVAILABLE_BYTES,
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": True, "training_performed": False, "cpu_heavy": True
        },
    })
    p90[graph_node] = GRAPHLOAD_P90_WALL_S

    if only_nodes is not None:
        wanted = set(only_nodes)
        unknown = wanted - {job["id"] for job in jobs}
        if unknown:
            raise RuntimeError(f"R0258 has no node(s) {sorted(unknown)}")
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
        "schema": "round0258-100m-graph-load-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1",
        ],
        "capabilities_produced": [
            capability
            for job_id, capability in (
                (controls_node, CONTROLS_CAPABILITY),
                (graph_node, GRAPHLOAD_CAPABILITY),
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
                "R0253 §D3 refused to carry its 2M graph-load figure "
                "(3.742829583992716 s = 1.4906011026447623x the ceiling) up to "
                "100M, correctly, and recorded that the artifacts to measure it "
                "at 100M no longer existed. They exist. Over R0243's real "
                f"{DIRECTED_EDGES} directed edges, on a node holding a live CUDA "
                "context, what is the WIDEST interval in the graph load and the "
                "edge preparation, and how does it compare to the registered "
                f"ceiling of {registered_ceiling_s()} s?"
            ),
            "question_b": (
                "Each of the five stages is one numpy or np.load call in the "
                "release today, over 2.5 billion elements, with no abort read "
                "inside any of them. Chunking each of them so an abort read runs "
                "between chunks: does the widest interval fall below the ceiling, "
                "and is every polled stage's output BITWISE identical to the "
                "shipped one? Bitwise, not close -- the CDF divides every "
                "sampling draw, so an approximate CDF is a changed science path."
            ),
            "question_c": (
                "O_DIRECT is review-0254-01 §B.1's corrected best write arm at "
                "0.035147x. This path writes nothing, so the flush ladder does "
                "not apply; what carries over is the read side. Does an O_DIRECT "
                "read arm beat a buffered one on the widest interval, and what "
                "does it cost in wall?"
            ),
            "question_d": (
                "review-0254-01 §F: 25 of 35 dispatch-derived entries construct "
                "no gate, and the two that matter -- round0238_nodes.run_assemble "
                "and round0113_nodes.run_train -- are both in the 25, so their "
                "stop latency is SILENCE rather than a measured zero. With a gate "
                "wired into both, how many entries now both install and gate?"
            ),
            "population": (
                "R0243's sealed 100M k15 fuzzy graph -- "
                f"{DIRECTED_EDGES} directed edges over {ROWS} rows at k = {K}, "
                "three .npy members of 10,044,413,144 B each -- plus the "
                "release's own source and every queue.json on this box"
            ),
            "rows": ROWS,
            "directed_edges": DIRECTED_EDGES,
            "registered_ceiling_s": registered_ceiling_s(),
            "arms": list(ARMS),
            "control_arm": ARM_UNPOLLED_CONTROL,
            "repetitions": REPETITIONS,
            "acceptance_rule": (
                "No numerical outcome makes this round a failure. A widest "
                "interval under the ceiling, over it, or unchanged by polling are "
                "each findings to publish with their coverage. The round FAILS "
                "only if a polled stage is not bitwise identical to the shipped "
                "one, if a planted defect passes a guard, or if the graph on disk "
                "is not R0243's."
            ),
            "the_100m_train_is_not_in_scope": (
                "no fit runs. Measuring the load and the edge preparation does "
                "not require completing one, and a ~10 h train is not this "
                "round's business. The CUDA context exists to make the "
                "measurement the un-signallable case, not to train."
            ),
            "artifact_presence": presence,
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
                "Three pre-existing files change, and none of them changes a "
                "number. (1) `basemap/round0254_dispatch.py`: one line adding "
                "`experiments.round0258_nodes` to SCOPE_MODULES, so this round's "
                "own entries are audited by the same guard as everyone else's. "
                "(2) `experiments/round0113_nodes.py` and (3) "
                "`experiments/round0238_nodes.py`: each gains a `_node_gate` "
                "helper and, inside `run_train` / `run_assemble`, a "
                "CoverageLedger window, an AbortPollGate constructed BEFORE the "
                "first read, a PollRecorder, `wrapped(...)` calls at stage "
                "boundaries, `model.abort_poll = wrapped` around `fit()` in "
                "run_train, and four new receipt keys (`node`, `gap_report`, "
                "`enforcement_poll_spacing`, `poll_coverage`, "
                "`observed_span_s`). The gate verdicts are SCORED AND PUBLISHED, "
                "never raised on, so no acceptance rule moves. No metric, "
                "neighbour set, ordering, rounding, threshold, treatment or "
                "digest changes; `basemap/artifact_identity.py`, "
                "`basemap/panel_v2.py`, the registry and every guard module are "
                "untouched."
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
        "queue_manifest": prepare_round0258(
            release_sha=args.release_sha,
            queue_root=args.queue_root,
            only_nodes=only,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
