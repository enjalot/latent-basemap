"""Build R0262's queue: the int8 substrate, and the wired host-int8 pipeline.

Two nodes, both bounded, neither a fit.
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

from basemap import round0113_prompt_contrast as prompt_contract
from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0253_stop_hooks import registered_ceiling_s
from basemap.round0254_dispatch import (
    SCOPE_MODULES,
    assert_derived_entries_install,
    dispatch_census,
    entry_tuples,
    gate_census,
    scope_residual,
)
from basemap.round0258_graph_load import EDGE_ARRAYS, EDGE_HEADER_PATH
from basemap.round0259_hundred_m import (
    RUNG_100M,
    RUNGS,
    SUBSTRATE_100M_PATH,
    assert_pairwise_rule,
    assert_substrate_dimension,
)
from basemap.round0262_host_int8_adapter import (
    INT8_100M_DIR,
    INT8_100M_PATH,
    SCALES_100M_PATH,
)
from experiments.round0262_nodes import (
    BATCH_SIZE,
    DIMENSION,
    FEATURE_ROWS_PER_UPDATE,
    LR_HORIZON_100M,
    MIN_MEM_AVAILABLE_BYTES,
    MIN_MEM_AVAILABLE_QUANTISE_BYTES,
    NODE_ANON_BUDGET_BYTES,
    POS_RATIO,
    PROBE_UPDATES,
    QUANTISE_ACTION,
    QUANTISE_CAPABILITY,
    QUANTISE_ROWS_PER_CHUNK,
    R0243_FUZZY_MANIFEST,
    ROWS_100M,
    UPDATE_BUDGET_S,
    WIRED_ACTION,
    WIRED_CAPABILITY,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ID = "0262"
ROUND_ROOT = "/data/latent-basemap/runs/round-0262"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/data/latent-basemap/release/round-0262"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0262-2026-08-12.md")

#: The int8 artifact is `38,400,000,000` B plus `200,000,000` B of scales.
#: Reserve that plus room.
DISK_RESERVE_BYTES = 60 << 30

GPU_HOURS_CAP = 1.0
#: The encode streams `153,600,000,128` B at the ~2 GB/s this volume delivers
#: and writes `38,600,000,000` B; measured at `5.5` min for the read+encode+
#: fidelity path plus the write. `1500` s is a p90, not a deadline.
QUANTISE_P90_WALL_S = 1_500.0
#: Reads `38,600,000,000` B of int8 into anonymous RAM, opens three graph
#: members, builds a `20,088,826,032` B float64 CDF, runs `400` bounded updates
#: and a `600`-step fidelity warmup.
WIRED_P90_WALL_S = 1_800.0


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
        raise RuntimeError("R0262 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0262 round must declare its required reviews")
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
    sealed = prompt_contract.read_sealed(
        R0243_FUZZY_MANIFEST, label="R0243 sealed fuzzy graph"
    )
    if sealed.get("schema") != RUNGS[RUNG_100M]["schema"]:
        raise RuntimeError("R0262: R0243's sealed schema is not the 100M rung's")
    outputs = sealed["outputs"]
    present: dict[str, Any] = {}
    for key, name in (("edges_sources", "sources"),
                      ("edges_targets", "targets"),
                      ("edges_weights", "weights")):
        declared = outputs[key]
        spec = EDGE_ARRAYS[name]
        if declared["canonical_path"] != spec["path"]:
            raise RuntimeError(f"R0262 {name} path disagrees with R0243's seal")
        if declared["sha256"] != spec["sha256"]:
            raise RuntimeError(f"R0262 {name} sha256 disagrees with R0243's seal")
        size = os.path.getsize(spec["path"])
        if size != int(declared["bytes"]):
            raise RuntimeError(
                f"R0262 {name} is {size} B, sealed at {declared['bytes']} B")
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
    return {
        "schema": "round0262-artifact-presence-v1",
        "arrays": present,
        "substrate": assert_substrate_dimension(),
        "r0243_manifest": expected_input_signature(R0243_FUZZY_MANIFEST),
    }


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0262 release checkout differs from requested release")
    basetemp = "/data/tmp/pytest-r0262-smoke"
    tmpdir = "/data/tmp/pytest-r0262-smoke-tmp"
    command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        f"--basetemp={basetemp}",
        "tests/test_round0262_contract.py", "tests/test_edgelist_smoke.py",
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
            f"R0262 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return prompt_contract.seal({
        "schema": "round0262-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cuda_hidden": True,
        "returncode": completed.returncode,
        "basetemp": basetemp,
        "tmpdir": tmpdir,
        "stdout_tail": completed.stdout.strip().splitlines()[-5:],
    })


def prepare_round0262(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    only_nodes: tuple[str, ...] | None = None,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0262 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    review_state = _upstream_review_state(list(required_reviews))
    presence = _the_artifacts_exist()

    free = _free_bytes("/data")
    if free < DISK_RESERVE_BYTES:
        raise RuntimeError(
            f"R0262 needs {DISK_RESERVE_BYTES} B free on /data; {free} B available")

    # The install/gate census, with R0262's own entries in scope. Adding the
    # module grows the denominator: every derived `run_*` in it must install
    # `install_stop_hooks` as its first statement or this raises.
    census = dispatch_census()
    guard = assert_derived_entries_install(SCOPE_MODULES, census)
    gates = gate_census(entry_tuples(guard["derived"]))
    residual = scope_residual(census, SCOPE_MODULES)
    pairwise = assert_pairwise_rule()

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0262 queue")
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
    wired_inputs = _dedupe(list(shared_inputs) + [
        presence["arrays"][name]
        for name in ("sources", "targets", "weights", "header")
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    ensure_data_directory(os.path.dirname(INT8_100M_DIR))
    jobs: list[dict[str, Any]] = []
    p90: dict[str, float] = {}

    quantise_node = "quantise0262"
    jobs.append({
        "id": quantise_node,
        "action": QUANTISE_ACTION,
        "handler_module": "experiments.round0262_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, QUANTISE_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{quantise_node}.done.json"),
        "expected_inputs": list(shared_inputs),
        "scratch": "/data/tmp/round0262-quantise",
        "p90_wall_s": QUANTISE_P90_WALL_S,
        "substrate": SUBSTRATE_100M_PATH,
        "int8_path": INT8_100M_PATH,
        "scales_path": SCALES_100M_PATH,
        "rows": ROWS_100M,
        "dimension": DIMENSION,
        "rows_per_chunk": QUANTISE_ROWS_PER_CHUNK,
        "min_mem_available_bytes": MIN_MEM_AVAILABLE_QUANTISE_BYTES,
        "numeric_plants_at_prepare": pairwise["numeric_controls"]["defects_planted"],
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": False, "training_performed": False, "cpu_heavy": True
        },
    })
    p90[quantise_node] = QUANTISE_P90_WALL_S

    wired_node = "wired0262"
    jobs.append({
        "id": wired_node,
        "action": WIRED_ACTION,
        "handler_module": "experiments.round0262_nodes",
        "handler_callable": "run_job",
        "deps": [quantise_node],
        "outputs": [os.path.join(artifacts, WIRED_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{wired_node}.done.json"),
        "expected_inputs": list(wired_inputs),
        "graph_manifest": R0243_FUZZY_MANIFEST,
        "p90_wall_s": WIRED_P90_WALL_S,
        "rung": RUNG_100M,
        "rows": ROWS_100M,
        "k": RUNGS[RUNG_100M]["k"],
        "dimension": DIMENSION,
        "directed_edges": 2_511_103_254,
        "batch_size": BATCH_SIZE,
        "pos_ratio": POS_RATIO,
        "feature_rows_per_update": FEATURE_ROWS_PER_UPDATE,
        "probe_updates": PROBE_UPDATES,
        "lr_horizon": LR_HORIZON_100M,
        "update_budget_s": UPDATE_BUDGET_S,
        "declared_anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
        "min_mem_available_bytes": MIN_MEM_AVAILABLE_BYTES,
        "int8_path": INT8_100M_PATH,
        "scales_path": SCALES_100M_PATH,
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": True, "training_performed": False, "cpu_heavy": True
        },
    })
    p90[wired_node] = WIRED_P90_WALL_S

    if only_nodes is not None:
        wanted = set(only_nodes)
        unknown = wanted - {job["id"] for job in jobs}
        if unknown:
            raise RuntimeError(f"R0262 has no node(s) {sorted(unknown)}")
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
        "schema": "round0262-host-int8-wiring-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1",
            "round0259-100m-entry-and-per-batch-intervals-v1",
        ],
        "capabilities_produced": [
            capability
            for job_id, capability in (
                (quantise_node, QUANTISE_CAPABILITY),
                (wired_node, WIRED_CAPABILITY),
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
                "review-0259-01 §D: the 100M rung's only remaining blocker is "
                "that no host-resident-X path is wired to the weighted fuzzy "
                "edge sampler. Does the SHIPPED "
                "`ParametricUMAP._prepare_edge_list_training` now select a "
                "host-int8 pipeline that stamps `weighted_effective: true`, at "
                "the real 100M rung, against R0243's real graph?"
            ),
            "question_b": (
                "review-0259-01 §L: int8 fidelity on the R0238 substrate is "
                "unverified — its timing used a synthetic array. What is the "
                "quantisation error over EVERY one of the 100,000,000 rows, and "
                "how large is the resulting gradient perturbation relative to "
                "the stochastic gradient noise the registered recipe already "
                "trains through?"
            ),
            "question_c": (
                "review-0259-01 §C.3: a fit gathers 2 x batch_size feature rows "
                "per update and R0259 timed half of one. On the wired path, at "
                f"the registered {BATCH_SIZE}/{POS_RATIO} shape and a full "
                f"{FEATURE_ROWS_PER_UPDATE}-row update, what is the per-update "
                f"interval against the plan's {UPDATE_BUDGET_S} s budget and the "
                f"registered {LR_HORIZON_100M}-update horizon?"
            ),
            "question_d": (
                "Every stoppability measurement from R0250-R0258 was taken on a "
                "path that could not run this job. On the host-int8 100M entry "
                "itself, what is the widest interval between abort polls, with "
                "what coverage, against the registered ceiling of "
                f"{registered_ceiling_s()} s?"
            ),
            "question_e": (
                "review-0259-01 §F.2 and §G.2, and the mandate's three named "
                "defects: does `assert_pairwise_rule` run on THIS entry (R0259 "
                "recorded numpy 2.4.4 and never called it in its 100M node); "
                "which accounting sees a 38.6 GB host allocation; and is "
                "`HostStreamEdgeSampler`'s host-resident endpoint copy a no-op "
                "that an isinstance reader rule mis-classifies?"
            ),
            "install_and_gate_at_prepare": {
                "entry_count": guard["derived"]["entry_count"],
                "every_entry_installs_effectively": (
                    guard["audit"]["every_entry_installs_effectively"]
                ),
                "entries_that_construct_a_gate": gates["entries_that_construct_a_gate"],
                "entries_that_both_install_and_gate": (
                    gates["entries_that_both_install_and_gate"]
                ),
                "scope_residual": residual,
            },
            "numpy_pairwise_rule_at_prepare": {
                "numpy_version_observed": pairwise["numpy_version_observed"],
                "checks_run": pairwise["checks_run"],
                "every_check_bitwise_identical": pairwise["every_check_bitwise_identical"],
            },
            "acceptance": (
                "The round is answered when the shipped entry selects "
                "`host_int8_hybrid` with `weighted_effective: true` at the real "
                "100M rung, the per-update interval is measured at a full "
                f"{FEATURE_ROWS_PER_UPDATE}-row update, the int8 gradient "
                "perturbation is reported against the SGD noise floor with a "
                "determinism control, and the widest poll gap is reported with "
                "its coverage. No fit runs."
            ),
        },
    })

    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    parser.add_argument("--only-node", action="append", default=None)
    args = parser.parse_args(argv)
    path = prepare_round0262(
        release_sha=args.release_sha,
        queue_root=args.queue_root,
        only_nodes=tuple(args.only_node) if args.only_node else None,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
