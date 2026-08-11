#!/usr/bin/env python3
"""Prepare, but never launch, the R0257 queue.

Five nodes:

1-3. `train_seed42`, `train_seed43`, `train_seed44` (GPU) — one `6250k` rung map
     each, on R0233's sealed substrate and sealed cluster-spill k15 fuzzy graph,
     under R0217's treatment with only the rung, the graph and the seed moved.
4.   `panel_6250k` (GPU) — the rung's own purity centroids and high-D reference,
     then all three maps scored with the accepted `panel_v2` config.
5.   `judge_6250k` (CPU) — the SEALED `n = 29` `MAD_n` criteria applied to the
     three maps, with the family-purity guard, the judgement guards, and their
     positive controls.

The script builds all three cell configs **here** from R0217's own `train_config`,
proves each reproduces one shared `rung_invariant_sha256` equal to R0217's template
digest under the seed/graph/rung mask, proves the three full-config digests are
three distinct values, and stamps the shared digest into every job. It validates
the sealed gate artifact at prepare time as well as in the node, so a moved floor is
caught before a single GPU second is spent.

**No `timeout=` appears anywhere in this file, deliberately.** CPython implements
`subprocess.run(..., timeout=N)` as `Popen.kill()`, i.e. a hidden SIGKILL, which
`plan-minilm-100m-v2.md` makes binding to purge before any further GPU round.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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
from basemap.round0257_judgement import (
    GATED_METRICS,
    GATE_CAPABILITY,
    INDEPENDENCE_LIMITATION,
    PANEL_FALSE_ALARM_RATE,
    POWER_MATERIALITY,
    REGISTERED_FFR_FLOOR,
    REGISTERED_K1024_BAND,
    REGISTERED_K256_BAND,
    REGISTERED_N,
    validate_gate_artifact,
)
from basemap.round0257_rung_contract import (
    FAMILY_RULE,
    GATE_REGISTERABLE_HERE,
    GRAPH_CAPABILITY,
    PANEL_CAPABILITY,
    PIPELINE_IDENTITY_NOTE,
    PIPELINE_IDENTITY_STRINGS_NOT_MASKED,
    REGISTERED_SUCCESSFUL_UPDATES,
    REGISTERED_UPDATE_BOUND,
    ROUND_ID,
    RUNG_ROWS,
    RUNG_SLUG,
    SEALED_RUNG_DIRECTED_EDGES,
    SEEDS,
    SUBSTRATE_CAPABILITY,
    VERDICT_CAPABILITY,
    dose_view,
    map_capability,
    rung_cell_id,
    rung_cell_ids,
    rung_invariant_sha256,
    rung_train_config,
)
from basemap.round0257_rung_pipeline import (
    DEVICE_BUDGET_BYTES,
    HOST_ANON_BUDGET_BYTES,
    HOST_RSS_LIMIT_GIB,
    predict_rung_footprint,
)
from basemap.round0255_treatment import file_sha256
from experiments.round0257_nodes import JUDGE_ACTION, PANEL_ACTION, TRAIN_ACTION
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0257"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0257-2026-08-11.md")

#: R0233's sealed rung artifacts.
R0233_SUBSTRATE = (
    "/data/latent-basemap/runs/round-0233/queue/artifacts/"
    f"{SUBSTRATE_CAPABILITY}/substrate.json"
)
R0233_GRAPH = (
    "/data/latent-basemap/runs/round-0233/queue-correction-1/artifacts/"
    f"{GRAPH_CAPABILITY}/qualified-graph.json"
)
#: R0216's sealed 2M substrate+graph receipt, bound ONLY so R0217's template can be
#: reconstructed with real signatures. No 2M byte is trained on in this round.
R0216_GRAPH_MANIFEST = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    "minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate-graph.json"
)
#: R0256's sealed, repaired n = 29 gate. The criteria this round judges by.
R0256_GATE = (
    "/data/latent-basemap/runs/round-0256/queue-correction-1/artifacts/"
    f"{GATE_CAPABILITY}/minilm-calibrated-madn-floors-n29-repaired.json"
)

#: The round's registered cap. The expected spend is 1.94 GPU-h and the worst case
#: within the registered 70 upd/s performance floor is 3.24 GPU-h; the cap is
#: deliberately above both rather than tight to either.
GPU_HOURS_CAP = 6.0
#: 255,142 updates at the registered 70 upd/s floor is 3,645 s. p90 is set above
#: that so a cell that is merely slow is not misreported as a stall.
TRAIN_P90_WALL_S = 4_200.0
PANEL_P90_WALL_S = 3_600.0
JUDGE_P90_WALL_S = 600.0


def _issued_round(release_sha: str) -> tuple[dict[str, Any], list[str]]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        [
            "git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor",
            base_commit, release_sha,
        ],
        check=False,
        capture_output=True,
    )
    if descendant.returncode != 0:
        raise RuntimeError(
            f"R0257 release {release_sha} is not a descendant of the round's "
            f"base_commit {base_commit}"
        )
    if str(frontmatter.get("status")) != "issued":
        raise RuntimeError("R0257 round file is not issued")
    if float(frontmatter.get("gpu_hours_cap", 0.0)) != GPU_HOURS_CAP:
        raise RuntimeError(
            "R0257 queue cap does not match the issued round's gpu_hours_cap"
        )
    return frontmatter, _frontmatter_list(frontmatter, "required_reviews")


def _review_state(required: list[str]) -> dict[str, Any]:
    import glob
    import re

    state: dict[str, Any] = {}
    contingent: list[str] = []
    for round_id in required:
        reviews = []
        for path in sorted(glob.glob(os.path.join(LAB_ROOT, f"review-{round_id}-*.md"))):
            with open(path, "r", encoding="utf-8") as handle:
                head = handle.read(4096)
            match = re.search(r"^status:\s*\"?([a-z]+)\"?\s*$", head, re.M)
            reviews.append({
                "file": os.path.basename(path),
                "sha256": file_sha256(path),
                "status": match.group(1) if match else "unknown",
            })
        accepted = [item for item in reviews if item["status"] == "accepted"]
        state[round_id] = {
            "reviews_present": reviews,
            "accepted_reviews": len(accepted),
        }
        if not accepted:
            contingent.append(round_id)
    return {
        "required_reviews": list(required),
        "by_round": state,
        "rounds_without_an_accepted_review": contingent,
        "claims_contingent_on": contingent,
        "note": (
            "Review is post-hoc: it blocks the downstream claim, not the launch. "
            "R0255 and R0256 both carry PARTIAL reviews that registered the gate "
            "contingently, so every verdict this round publishes is contingent on "
            "those registrations standing."
        ),
    }


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0257 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--basetemp=/data/tmp/pytest-r0257-smoke",
        "tests/test_round0257_contract.py",
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "TMPDIR": "/data/tmp/pytest-r0257-smoke-tmp",
    })
    os.makedirs("/data/tmp/pytest-r0257-smoke-tmp", exist_ok=True)
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    receipt = prompt_contract.seal({
        "schema": "round0257-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cwd": RELEASE_ROOT,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
        "path_exercised": (
            "rung config construction for all three seeds against R0217's own "
            "template, the rung-invariant digest equality and its four negative "
            "cases, the ceil-derived horizon at the rung's sealed edge count, the "
            "rung pipeline subclass geometry assertion, the sealed gate validator "
            "and all six of its plants, the three behavioural judge controls, the "
            "family-purity guard and its four rung-map plants, the absence of "
            "subprocess timeouts in the round's own sources, and the entry path of "
            "all three node actions"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0257 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def prepare_round0257(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    only_nodes: tuple[str, ...] | None = None,
    completed_train_root: str | None = None,
) -> str:
    """Build the queue manifest.

    `completed_train_root` points the panel node at train receipts a PREVIOUS
    queue root already produced. A correction queue that re-runs only the panel
    and the judge must consume the trains that already ran -- retraining them
    would spend GPU hours to reproduce artifacts that are already sealed, and
    would charge the round for a setup defect twice.
    """
    frontmatter, required_reviews = _issued_round(release_sha)
    review_state = _review_state(required_reviews)

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0257 queue root")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))

    round_signature = expected_input_signature(ROUND_FILE)
    substrate_manifest_signature = expected_input_signature(R0233_SUBSTRATE)
    graph_manifest_signature = expected_input_signature(R0233_GRAPH)
    r0216_manifest_signature = expected_input_signature(R0216_GRAPH_MANIFEST)
    gate_signature = expected_input_signature(R0256_GATE)

    with open(R0233_SUBSTRATE, "rb") as handle:
        substrate_manifest = json.load(handle)
    with open(R0233_GRAPH, "rb") as handle:
        graph_manifest = json.load(handle)
    with open(R0216_GRAPH_MANIFEST, "rb") as handle:
        r0216_manifest = json.load(handle)
    with open(R0256_GATE, "rb") as handle:
        gate_artifact = json.load(handle)

    if int(substrate_manifest["rows"]) != RUNG_ROWS:
        raise RuntimeError("R0257 sealed rung substrate is not the registered rung")
    edges = int(graph_manifest["directed_edges"])
    if edges != SEALED_RUNG_DIRECTED_EDGES:
        raise RuntimeError(
            f"R0257 sealed rung graph reports {edges} directed edges, registered "
            f"{SEALED_RUNG_DIRECTED_EDGES}"
        )
    # The same key path the node's reader uses, asserted at prepare time so a
    # manifest whose qualification block is missing is a prepare-time refusal.
    tie_aware = graph_manifest["selected_graph"]["tie_aware"]
    if (
        int(tie_aware["n"]) != RUNG_ROWS
        or float(tie_aware["mean"]) < float(graph_manifest["floors"]["tie_aware_mean"])
        or float(tie_aware["p10"]) < float(graph_manifest["floors"]["tie_aware_p10"])
        or int(graph_manifest["degrees"]["zero_degree_rows"]) != 0
    ):
        raise RuntimeError("R0257 sealed rung graph does not clear its own floors")

    # The gate is validated HERE as well as in the node: a moved floor must be a
    # prepare-time refusal, not a discovery after three GPU-hours.
    gate = validate_gate_artifact(gate_artifact)

    substrate_signature = dict(substrate_manifest["substrate"])
    provenance_signature = dict(substrate_manifest["provenance"])
    graph_signature = dict(graph_manifest["graph"])
    r0217_template_signatures = {
        "substrate": dict(r0216_manifest["substrate"]),
        "graph": dict(r0216_manifest["graph"]),
        "graph_manifest": r0216_manifest_signature,
    }

    configs: dict[int, dict[str, Any]] = {}
    config_hashes: dict[str, str] = {}
    invariants: set[str] = set()
    for seed in SEEDS:
        config, config_sha, invariant = rung_train_config(
            seed=seed,
            rows=RUNG_ROWS,
            graph_edges=edges,
            substrate_signature=substrate_signature,
            graph_signature=graph_signature,
            graph_manifest_signature=graph_manifest_signature,
            r0217_substrate_signature=r0217_template_signatures["substrate"],
            r0217_graph_signature=r0217_template_signatures["graph"],
            r0217_graph_manifest_signature=r0217_template_signatures["graph_manifest"],
        )
        configs[seed] = config
        config_hashes[str(seed)] = config_sha
        invariants.add(invariant)
    if len(invariants) != 1:
        raise RuntimeError(
            f"R0257 the three rung cells carry {len(invariants)} invariant digests"
        )
    if len(set(config_hashes.values())) != len(SEEDS):
        raise RuntimeError("R0257 two rung cells have the same full-config digest")
    rung_invariant = sorted(invariants)[0]
    updates = int(configs[SEEDS[0]]["optimizer"]["successful_positive_lr_updates"])
    if updates != REGISTERED_SUCCESSFUL_UPDATES:
        raise RuntimeError("R0257 derived horizon is not the registered one")

    predictions = {str(seed): predict_rung_footprint(seed) for seed in SEEDS}

    smoke = _release_cpu_smoke(release_sha)
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, smoke, immutable=True)

    identity_path = os.path.join(preflight, "rung-config-identity.json")
    atomic_write_new_json(
        identity_path,
        prompt_contract.seal({
            "schema": "round0257-rung-config-identity-v1",
            "round_id": ROUND_ID,
            "release_sha": release_sha,
            "rung": RUNG_SLUG,
            "rows": RUNG_ROWS,
            "seeds": list(SEEDS),
            "cell_ids": list(rung_cell_ids()),
            "rung_invariant_sha256": rung_invariant,
            "rung_invariant_equals_r0217_template_under_the_mask": True,
            "pipeline_identity_strings_not_masked": dict(
                PIPELINE_IDENTITY_STRINGS_NOT_MASKED
            ),
            "pipeline_identity_note": PIPELINE_IDENTITY_NOTE,
            "config_sha256_by_seed": config_hashes,
            "distinct_config_digests": len(set(config_hashes.values())),
            "dose": dose_view(edges),
            "memory_predictions": predictions,
            "registry_fingerprint": registry_fingerprint(),
            "configs": {str(seed): configs[seed] for seed in SEEDS},
        }),
        immutable=True,
    )

    gate_preflight_path = os.path.join(preflight, "n29-gate-as-read.json")
    atomic_write_new_json(
        gate_preflight_path,
        prompt_contract.seal({
            "schema": "round0257-n29-gate-as-read-v1",
            "round_id": ROUND_ID,
            "release_sha": release_sha,
            "gate_artifact": gate_signature,
            "gate": gate,
            "validated_at_prepare_time": True,
            "family_rule": FAMILY_RULE,
            "power_materiality": POWER_MATERIALITY,
            "independence_limitation": INDEPENDENCE_LIMITATION,
            "this_round_writes_no_floor": (
                "R0257 has no code path that fits, refits, or registers a floor. "
                "Every criterion is read from the sealed artifact above and "
                "asserted equal to the values R0255 registered and R0256 "
                "republished bitwise unchanged."
            ),
        }),
        immutable=True,
    )

    shared_inputs = _dedupe([
        round_signature,
        substrate_manifest_signature,
        graph_manifest_signature,
        substrate_signature,
        provenance_signature,
        graph_signature,
        r0216_manifest_signature,
        expected_input_signature(smoke_path),
        expected_input_signature(identity_path),
        expected_input_signature(gate_preflight_path),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    jobs: list[dict[str, Any]] = []
    p90: dict[str, float] = {}
    panel_cells: list[dict[str, Any]] = []
    train_nodes: list[str] = []

    for seed in SEEDS:
        capability = map_capability(seed)
        node = f"train_seed{seed}"
        output = os.path.join(artifacts, capability)
        receipt_root = (
            os.path.join(completed_train_root, capability)
            if completed_train_root
            else output
        )
        jobs.append({
            "id": node,
            "action": TRAIN_ACTION,
            "handler_module": "experiments.round0257_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"{node}.done.json"),
            "expected_inputs": shared_inputs,
            "p90_wall_s": TRAIN_P90_WALL_S,
            "training_seed": int(seed),
            "capability": capability,
            "cell_id": rung_cell_id(seed),
            "substrate_manifest_signature": substrate_manifest_signature,
            "graph_manifest_signature": graph_manifest_signature,
            "r0217_template_signatures": r0217_template_signatures,
            "rung_invariant_sha256": rung_invariant,
            "registered_dose_bound": REGISTERED_UPDATE_BOUND,
            "memory_prediction": predictions[str(seed)],
            "node_policy": {
                "gpu_required": True,
                "training_performed": True,
                "cpu_heavy": False,
            },
        })
        p90[node] = TRAIN_P90_WALL_S
        train_nodes.append(node)
        panel_cells.append({
            "seed": int(seed),
            "capability": capability,
            "cell_id": rung_cell_id(seed),
            "train_receipt": {
                "kind": "file",
                "canonical_path": os.path.join(
                    receipt_root, f"{node}-train-receipt.json"
                ),
            },
        })

    panel_node = f"panel_{RUNG_SLUG}"
    panel_output = os.path.join(artifacts, PANEL_CAPABILITY)
    jobs.append({
        "id": panel_node,
        "action": PANEL_ACTION,
        "handler_module": "experiments.round0257_nodes",
        "handler_callable": "run_job",
        "deps": list(train_nodes),
        "outputs": [panel_output],
        "done_marker": os.path.join(artifacts, f"{panel_node}.done.json"),
        "expected_inputs": shared_inputs,
        "p90_wall_s": PANEL_P90_WALL_S,
        "substrate_manifest_signature": substrate_manifest_signature,
        "graph_manifest_signature": graph_manifest_signature,
        "cells": panel_cells,
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
            "cpu_heavy": False,
        },
    })
    p90[panel_node] = PANEL_P90_WALL_S

    judge_node = f"judge_{RUNG_SLUG}"
    jobs.append({
        "id": judge_node,
        "action": JUDGE_ACTION,
        "handler_module": "experiments.round0257_nodes",
        "handler_callable": "run_job",
        "deps": [panel_node],
        "outputs": [os.path.join(artifacts, VERDICT_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{judge_node}.done.json"),
        "expected_inputs": _dedupe([*shared_inputs, gate_signature]),
        "p90_wall_s": JUDGE_P90_WALL_S,
        "gate_artifact_signature": gate_signature,
        "panel_signature": {
            "kind": "file",
            "canonical_path": os.path.join(
                panel_output, f"{panel_node}-ladder-panel.json"
            ),
        },
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
            "cpu_heavy": False,
        },
    })
    p90[judge_node] = JUDGE_P90_WALL_S

    if only_nodes is not None:
        wanted = set(only_nodes)
        unknown = wanted - {job["id"] for job in jobs}
        if unknown:
            raise RuntimeError(f"R0257 correction queue names unknown nodes: {unknown}")
        jobs = [job for job in jobs if job["id"] in wanted]
        for job in jobs:
            if set(job["deps"]) - wanted:
                raise RuntimeError(
                    f"R0257 correction node {job['id']} depends on a node the "
                    "correction queue does not carry"
                )
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
        "schema": "round0257-ladder-rung-judged-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            SUBSTRATE_CAPABILITY,
            GRAPH_CAPABILITY,
            GATE_CAPABILITY,
        ],
        "capabilities_produced": [
            *(map_capability(seed) for seed in SEEDS),
            PANEL_CAPABILITY,
            VERDICT_CAPABILITY,
        ],
        "jobs": jobs,
        "p90_wall_s": p90,
        "registered": {
            "rung": RUNG_SLUG,
            "rows": RUNG_ROWS,
            "seeds": list(SEEDS),
            "cell_ids": list(rung_cell_ids()),
            "why_this_rung_and_this_seed_count": (
                "stated and justified in round-0257-2026-08-11.md BEFORE the run: "
                "no map has ever been trained above 2M in this program, the 6.25M "
                "graph is the only ladder rung qualified against exact truth over "
                "ALL its rows, ladder-6250k-h2048-seed42 is the cell id the shipped "
                "family-purity guard already names, and three seeds fit the cap "
                "with retry headroom while separating a rung failure from a seed "
                "failure at a 0.0341 panel false-alarm rate"
            ),
            "sealed_directed_edges": edges,
            "successful_positive_lr_updates": updates,
            "registered_update_bound": REGISTERED_UPDATE_BOUND,
            "dose": dose_view(edges),
            "rung_invariant_sha256": rung_invariant,
            "device_budget_bytes": DEVICE_BUDGET_BYTES,
            "host_anonymous_budget_bytes": HOST_ANON_BUDGET_BYTES,
            "host_rss_limit_gib": HOST_RSS_LIMIT_GIB,
            "guarded_on": "host ANONYMOUS bytes, never RSS",
            "gate_capability_consumed": GATE_CAPABILITY,
            "gated_metrics": list(GATED_METRICS),
            "registered_n": REGISTERED_N,
            "registered_ffr_floor": REGISTERED_FFR_FLOOR,
            "registered_k256_band": list(REGISTERED_K256_BAND),
            "registered_k1024_band": list(REGISTERED_K1024_BAND),
            "panel_false_alarm_rate": PANEL_FALSE_ALARM_RATE,
            "power_materiality": POWER_MATERIALITY,
            "independence_limitation": INDEPENDENCE_LIMITATION,
            "family_rule": FAMILY_RULE,
            "gate_registerable_here": GATE_REGISTERABLE_HERE,
            "no_floor_is_written_by_this_round": True,
            "a_failing_map_is_a_finding": (
                "Registered before the run: a rung map that fails is published "
                "failing, with its margin. No retrain, no other seed, no adjusted "
                "floor."
            ),
        },
    })
    manifest_path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(manifest_path, queue, immutable=True)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="prepare the R0257 queue")
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    parser.add_argument("--only-nodes", default="")
    parser.add_argument("--completed-train-root", default="")
    args = parser.parse_args(argv)
    only = tuple(item for item in args.only_nodes.split(",") if item) or None
    path = prepare_round0257(
        release_sha=args.release_sha,
        queue_root=args.queue_root,
        only_nodes=only,
        completed_train_root=args.completed_train_root or None,
    )
    print(json.dumps({"queue": path, "sha256": file_sha256(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
