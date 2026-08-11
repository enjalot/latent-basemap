#!/usr/bin/env python3
"""Prepare, but never launch, the R0252 queue.

Three nodes, in this order:

1. `hashpoll_0252` (CPU) — the integrity hash, unpolled and polled, at four real
   sizes up to and including `153,600,000,128` B, the exact size of a
   `100,000,000 x 384` fp32 substrate. R0251 projected there; this measures.
   It runs first because it costs no GPU and would invalidate the round's
   premise if it failed.
2. `panelpoll_0252` (GPU) — `score_panel` with and without the new hook, on
   R0218's archived seed-42 checkpoint, requiring both arms to agree byte for
   byte. This is the call `roundreport`'s census ranks as R0251's binding
   interval.
3. `tail_0252` (GPU, long) — the per-batch tail at `600,000` updates against
   R0251's `10,000`, with the trainer stop-latency control run first so a
   stoppability failure costs seconds rather than an hour of GPU.

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
from basemap.round0216_minilm_2m_substrate import CAPABILITY as R0216_CAPABILITY
from basemap.round0217_minilm_2m_seed_family import (
    TRAIN_SCHEMA as R0217_TRAIN_SCHEMA,
    capability_for_seed as r0217_capability_for_seed,
)
from basemap.round0247_registry import registered_value, registry_fingerprint
from basemap.round0250_panel_n16 import (
    DIMENSION,
    PANEL_CAPABILITY as R0250_PANEL_CAPABILITY,
    PANEL_CAPABILITY_N16,
    POOLED_SEEDS,
    REFERENCE_KEY,
)
from basemap.round0250_seed_extension_n16 import (
    GRAPH_CAPABILITY,
    GRAPH_K,
    GRAPH_SCHEMA,
    GRAPH_SOURCE_ROUND_ID,
    ROWS,
    SEALED_DIRECTED_EDGES,
    TEMPLATE_SEED,
)
from basemap.round0251_rescore import RESCORED_SEED
from basemap.round0251_trainer_setup import SETUP_CAPABILITY as R0251_SETUP_CAPABILITY
from basemap.round0252_stoppability import (
    HASH_CAPABILITY,
    INTEGRITY_GUARANTEE,
    PANEL_CAPABILITY,
    STOP_CONTROL_DELAY_S,
    STOP_CONTROL_UPDATES,
    TAIL_CAPABILITY,
    TAIL_RUNG_UPDATES,
    TARGET_DIMENSION,
    TARGET_ROWS,
    TARGET_SUBSTRATE_BYTES,
    declared_sites_match_the_release,
)
from experiments.round0252_nodes import HASH_ACTION, PANEL_ACTION, TAIL_ACTION
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ID = "0252"
ROUND_ROOT = "/data/latent-basemap/runs/round-0252"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0252-2026-08-11.md")

R0216_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    f"{R0216_CAPABILITY}"
)
GRAPH_MANIFEST = os.path.join(R0216_ARTIFACTS, "substrate-graph.json")
R0217_ARTIFACTS = "/data/latent-basemap/runs/round-0217/queue-correction-1/artifacts"
R0218_PANEL = (
    f"/data/latent-basemap/runs/round-0218/queue/artifacts/{R0250_PANEL_CAPABILITY}/"
    "seed-family-panel.json"
)
R0250_PANEL_N16 = (
    f"/data/latent-basemap/runs/round-0250/queue/artifacts/{PANEL_CAPABILITY_N16}/"
    "seed-family-panel-n16.json"
)
R0251_TRAINER_SETUP = (
    f"/data/latent-basemap/runs/round-0251/queue/artifacts/{R0251_SETUP_CAPABILITY}/"
    "trainer-setup-and-batch-tail.json"
)

#: The size ladder the hash node measures, in bytes. The last one is the real
#: target: `100,000,000 x 384 x 4`. Four points, spanning 50x, so the polled
#: gap's independence of file size is a fitted slope rather than an assertion.
SYNTHETIC_SIZES_BYTES: tuple[int, ...] = (
    12_288_000_128,
    49_152_000_128,
    TARGET_SUBSTRATE_BYTES,
)
#: Peak transient disk: one synthetic file at a time, deleted before the next.
PEAK_SCRATCH_BYTES = max(SYNTHETIC_SIZES_BYTES)

#: R0252's registered cap. The long rung is ~600,000 updates at R0251's measured
#: 113.9-114.5 upd/s, i.e. ~5,250 s = 1.46 h, plus ~60 s for the panel node and
#: no GPU at all for the hash node. 2.5 leaves room for one retry inside the
#: mandate's 3.0 GPU-h ceiling.
GPU_HOURS_CAP = 2.5
HASH_P90_WALL_S = 3_600.0
PANEL_P90_WALL_S = 1_200.0
TAIL_P90_WALL_S = 9_000.0


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
        raise RuntimeError("R0252 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0252 round must declare its required reviews")
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
    """Refuse to prepare a queue whose scratch file will not fit with slack.

    The hash node writes one `153,600,000,128` B file at a time and deletes it
    before the next. This checks that peak plus a 40 GiB reserve fits on `/data`
    at prepare time, so a full volume is a prepare-class refusal rather than a
    node failure halfway through a 143 GiB write.
    """
    reserve = 40 << 30
    free = _free_bytes("/data")
    if free < PEAK_SCRATCH_BYTES + reserve:
        raise RuntimeError(
            f"R0252 needs {PEAK_SCRATCH_BYTES + reserve} B free on /data for the "
            f"100M-scale hash rung; {free} B available"
        )
    return {
        "free_bytes_at_prepare": free,
        "peak_scratch_bytes": PEAK_SCRATCH_BYTES,
        "reserve_bytes": reserve,
        "one_file_at_a_time": True,
    }


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0252 release checkout differs from requested release")
    basetemp = "/data/tmp/pytest-r0252-smoke"
    tmpdir = "/data/tmp/pytest-r0252-smoke-tmp"
    command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        f"--basetemp={basetemp}", "tests/test_round0252_contract.py",
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "TMPDIR": tmpdir,
    })
    os.makedirs(tmpdir, exist_ok=True)
    started = time.monotonic()
    #: **No `timeout=` anywhere in this file, deliberately.** CPython implements
    #: `subprocess.run(..., timeout=N)` as `Popen.kill()`, i.e. a hidden SIGKILL,
    #: and `plan-minilm-100m-v2.md` makes purging that construct binding before
    #: any further GPU round. A contract test greps this file for it.
    completed = subprocess.run(
        command, cwd=RELEASE_ROOT, env=environment,
        capture_output=True, text=True, check=False,
    )
    wall = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0252 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return prompt_contract.seal({
        "schema": "round0252-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cuda_hidden": True,
        "returncode": completed.returncode,
        "wall_seconds": wall,
        "basetemp": basetemp,
        "tmpdir": tmpdir,
        "stdout_tail": completed.stdout.strip().splitlines()[-5:],
        "poll_sites": declared_sites_match_the_release(),
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
        raise RuntimeError("R0252 sealed R0216 substrate+graph contract changed")
    edges = int(manifest.get("directed_edge_count", 0)) or int(
        (manifest.get("graph_checks") or {}).get("directed_edges", 0)
    )
    if edges != SEALED_DIRECTED_EDGES:
        raise RuntimeError(
            f"R0252 sealed graph reports {edges} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    return signature, manifest, edges


def _sealed_seed42() -> dict[str, Any]:
    capability = r0217_capability_for_seed(RESCORED_SEED)
    path = os.path.join(R0217_ARTIFACTS, capability, "train-receipt.json")
    receipt = prompt_contract.read_sealed(path, label="R0217 seed-42 train receipt")
    if (
        receipt.get("schema") != R0217_TRAIN_SCHEMA
        or receipt.get("round_id") != "0217"
        or receipt.get("capability") != capability
        or int(receipt.get("training_seed", -1)) != RESCORED_SEED
    ):
        raise RuntimeError("R0217 seed-42 train receipt contract changed")
    model = dict(receipt["model"])
    if expected_input_signature(model["canonical_path"]) != model:
        raise RuntimeError("R0217 seed-42 checkpoint bytes changed")
    return {
        "capability": capability,
        "receipt": expected_input_signature(path),
        "model": model,
    }


def _sealed_panels() -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    panel = prompt_contract.read_sealed(R0218_PANEL, label="R0218 four-seed panel")
    if str(panel["high_d_reference_key"]) != REFERENCE_KEY:
        raise RuntimeError("R0218's reference identity is not the registered one")
    inputs.append(expected_input_signature(R0218_PANEL))
    reference = dict(panel["shared_high_d_reference"])
    if expected_input_signature(reference["canonical_path"]) != reference:
        raise RuntimeError("R0218's published high-D reference bytes changed")
    inputs.append(reference)
    for entry in dict(panel.get("centroids") or {}).values():
        centroid = dict(entry)
        if expected_input_signature(centroid["canonical_path"]) != centroid:
            raise RuntimeError("R0218 published centroid bytes changed")
        inputs.append(centroid)
    pooled = prompt_contract.read_sealed(
        R0250_PANEL_N16, label="R0250 sealed sixteen-cell panel"
    )
    if int(pooled.get("n", -1)) != len(POOLED_SEEDS) or str(
        pooled.get("high_d_reference_key")
    ) != REFERENCE_KEY:
        raise RuntimeError("R0250's sixteen-cell panel contract changed")
    inputs.append(expected_input_signature(R0250_PANEL_N16))
    return inputs


def _sealed_r0251_trainer_setup() -> dict[str, Any]:
    receipt = prompt_contract.read_sealed(
        R0251_TRAINER_SETUP, label="R0251 sealed trainer setup"
    )
    if receipt.get("round_id") != "0251" or receipt.get("capability") != R0251_SETUP_CAPABILITY:
        raise RuntimeError("R0251 sealed trainer-setup contract changed")
    if "tail_verdict" not in receipt or "tail_model" not in receipt:
        raise RuntimeError("R0251 trainer-setup receipt carries no tail model")
    return expected_input_signature(R0251_TRAINER_SETUP)


def prepare_round0252(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    only_nodes: tuple[str, ...] | None = None,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0252 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    graph_manifest_signature, graph_manifest, edges = _sealed_graph()
    substrate_signature = dict(graph_manifest["substrate"])
    graph_signature = dict(graph_manifest["graph"])
    provenance_signature = dict(graph_manifest["provenance"])
    seed42 = _sealed_seed42()
    panel_inputs = _sealed_panels()
    r0251_setup = _sealed_r0251_trainer_setup()
    review_state = _upstream_review_state(list(required_reviews))
    sites = declared_sites_match_the_release()
    headroom = _disk_headroom()

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0252 queue")
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

    hash_node = "hashpoll_0252"
    jobs.append({
        "id": hash_node,
        "action": HASH_ACTION,
        "handler_module": "experiments.round0252_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, HASH_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{hash_node}.done.json"),
        "expected_inputs": list(shared_inputs),
        "p90_wall_s": HASH_P90_WALL_S,
        "substrate_signature": substrate_signature,
        "synthetic_sizes_bytes": list(SYNTHETIC_SIZES_BYTES),
        "disk_headroom": headroom,
        "upstream_review_state": review_state,
        "node_policy": {"gpu_required": False, "training_performed": False, "cpu_heavy": True},
    })
    p90[hash_node] = HASH_P90_WALL_S

    panel_node = "panelpoll_0252"
    jobs.append({
        "id": panel_node,
        "action": PANEL_ACTION,
        "handler_module": "experiments.round0252_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, PANEL_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{panel_node}.done.json"),
        "expected_inputs": _dedupe([
            *shared_inputs, *panel_inputs, seed42["receipt"], seed42["model"],
        ]),
        "p90_wall_s": PANEL_P90_WALL_S,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": graph_manifest_signature,
        "panel_evidence": R0218_PANEL,
        "panel_n16": expected_input_signature(R0250_PANEL_N16),
        "rescored_seed": RESCORED_SEED,
        "rescored_capability": seed42["capability"],
        "panel_stop_delay_s": 0.5,
        "upstream_review_state": review_state,
        "node_policy": {"gpu_required": True, "training_performed": False, "cpu_heavy": False},
    })
    p90[panel_node] = PANEL_P90_WALL_S

    tail_node = "tail_0252"
    jobs.append({
        "id": tail_node,
        "action": TAIL_ACTION,
        "handler_module": "experiments.round0252_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, TAIL_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{tail_node}.done.json"),
        "expected_inputs": _dedupe([*shared_inputs, r0251_setup]),
        "p90_wall_s": TAIL_P90_WALL_S,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": graph_manifest_signature,
        "template_seed": TEMPLATE_SEED,
        "tail_rung_updates": TAIL_RUNG_UPDATES,
        "stop_control_updates": STOP_CONTROL_UPDATES,
        "trainer_stop_delay_s": STOP_CONTROL_DELAY_S,
        "r0251_trainer_setup": r0251_setup,
        "upstream_review_state": review_state,
        "node_policy": {"gpu_required": True, "training_performed": True, "cpu_heavy": False},
    })
    p90[tail_node] = TAIL_P90_WALL_S

    if only_nodes is not None:
        wanted = set(only_nodes)
        unknown = wanted - {job["id"] for job in jobs}
        if unknown:
            raise RuntimeError(f"R0252 has no node(s) {sorted(unknown)}")
        jobs = [job for job in jobs if job["id"] in wanted]
        for job in jobs:
            missing = set(job["deps"]) - wanted
            if missing:
                raise RuntimeError(
                    f"R0252 correction node {job['id']} depends on a node the "
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
        "schema": "round0252-stoppability-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            GRAPH_CAPABILITY,
            R0250_PANEL_CAPABILITY,
            PANEL_CAPABILITY_N16,
            R0251_SETUP_CAPABILITY,
            seed42["capability"],
        ],
        "capabilities_produced": [
            capability
            for job_id, capability in (
                (hash_node, HASH_CAPABILITY),
                (panel_node, PANEL_CAPABILITY),
                (tail_node, TAIL_CAPABILITY),
            )
            if only_nodes is None or job_id in set(only_nodes)
        ],
        "correction_of": (None if only_nodes is None else QUEUE_ROOT),
        "correction_nodes": (None if only_nodes is None else list(only_nodes)),
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": p90,
        "scientific_contract": {
            "question_a": (
                "with a cooperative-abort read between the chunks of "
                "basemap/artifact_identity.py's already-chunked hash loops, what "
                "is the widest abort-read gap of one integrity hash, measured "
                f"unpolled and polled at four real file sizes up to "
                f"{TARGET_SUBSTRATE_BYTES} B -- the exact size of a "
                f"{TARGET_ROWS} x {TARGET_DIMENSION} fp32 substrate -- against "
                f"the registered {registered_value('r0246_max_poll_spacing_s')} s "
                "ceiling; does the polled gap stop depending on file size; do "
                "both arms produce the identical digest and does it match the "
                "sealed one; and how long does a stop actually take when a flag "
                "is planted mid-hash?"
            ),
            "question_b": (
                "roundreport's abort-poll gap census ranks one score_panel call "
                "as R0251's binding interval at 1.980511222005589 s = "
                "0.7887487648242341x the ceiling. With ten abort reads added "
                "inside basemap/panel_v2 -- seven between score_panel's existing "
                "phases and three between _self_knn's existing bounded loop "
                "iterations -- which site holds the widest remaining interval, "
                "what is it against the ceiling, and does every published panel "
                "value stay byte-identical to the un-hooked arm and to R0218's "
                "sealed seed-42 cell?"
            ),
            "question_c": (
                "R0251 refused to publish a per-batch return level because its "
                "pre-registered threshold ladder moved it 86.69863492594678x "
                f"against an identification limit of 10.0. At {TAIL_RUNG_UPDATES} "
                "updates -- sixty times the batches, same estimator, same ladder, "
                "same bootstrap, same limit -- is the extreme-value fit identified, "
                "and what does the distribution-free bound become?"
            ),
            "population": "sealed R0216 2,000,000-row mixed MiniLM substrate",
            "sealed_directed_edges": edges,
            "rescored_seed": RESCORED_SEED,
            "pooled_seed_family": list(POOLED_SEEDS),
            "hash_size_ladder_bytes": [
                int(substrate_signature["bytes"]), *SYNTHETIC_SIZES_BYTES
            ],
            "the_100m_figure_is_measured_not_projected": True,
            "target_rows": TARGET_ROWS,
            "target_dimension": TARGET_DIMENSION,
            "target_substrate_bytes": TARGET_SUBSTRATE_BYTES,
            "integrity_guarantee": INTEGRITY_GUARANTEE,
            "declared_poll_sites": sites,
            "tail_rung_updates": TAIL_RUNG_UPDATES,
            "stop_control_updates": STOP_CONTROL_UPDATES,
            "stop_control_delay_s": STOP_CONTROL_DELAY_S,
            "registered_max_poll_spacing_s": registered_value("r0246_max_poll_spacing_s"),
            "disk_headroom": headroom,
            "registers_nothing": (
                "R0252 registers no floor, band, multiplier, estimator, gate or "
                "map. Every artifact is a measurement and no rung is a family cell."
            ),
            "gate_registered": False,
            "floors_registered": 0,
            "registry_fingerprint": registry_fingerprint(),
            "registry_mutated": False,
            "guard_modules_edited": False,
            "science_modules_edited": (
                "basemap/artifact_identity.py and basemap/panel_v2.py, additive "
                "only: a module-level `abort_poll` defaulting to None, a "
                "`set_abort_poll`, a `_poll_abort` reading it, site constants, and "
                "calls to `_poll_abort` between existing loop iterations and "
                "existing phases. No existing line is modified, no metric, "
                "neighbour set, ordering or rounding changes, and the panel node "
                "proves it by requiring the hooked and un-hooked arms to agree "
                "byte for byte."
            ),
            "upstream_review_state": review_state,
            "evaluation_performed": True,
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
        "queue_manifest": prepare_round0252(
            release_sha=args.release_sha,
            queue_root=args.queue_root,
            only_nodes=only,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
