#!/usr/bin/env python3
"""Prepare, but never launch, the R0266 queue — the map-level host-int8 fidelity cell.

Three nodes in one queue, in this order:

1. `train_minilm_fneg_2m_hostint8` (GPU) — ONE cell, seed 42, the EXACT R0265 pinned
   recipe plus the single delta `x_residency="host_int8"`, built from R0217's template
   via `round0266_int8_treatment` and proved by `assert_registered_int8_recipe`.
2. `score_minilm_fneg_2m_hostint8_panel` (GPU) — the int8 map scored on the SAME
   instruments R0265's panel uses (held-out FFR, purity k256/k1024, collapse, fog).
3. `register_hostint8_map_fidelity_gate` (CPU) — the map-level int8-fidelity gate:
   criterion (a) the int8 cell's five gated metrics INSIDE R0265's sealed family bands,
   and criterion (b) `|int8 - fp32_seed42| <= the family's 13-seed observed range`. Every
   band and every range is bound from a SEALED artifact by sha256 and read at gate time.

This builder REFUSES until the round file is issued (status: issued, base_commit an
ancestor of the release) and every bound input exists — including R0265's sealed family
floors and sealed n=13 panel. It builds and proves the int8 config HERE, seals the
import-source closure (R0265's closure + the R0266 int8 recipe module), and binds every
sealed cross-round input (R0216 graph, R0218 panel, held-out truth, R0265 floors, R0265
panel, the fp32 seed-42 sibling) by real sha256 + bytes computed from the file on disk.
"""
from __future__ import annotations

import argparse
import hashlib
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
from basemap.round0218_minilm_2m_panel import CAPABILITY as R0218_PANEL_CAPABILITY, CENTROID_KS
from basemap.round0247_registry import registry_fingerprint
from basemap.round0254_dispatch import (
    SCOPE_MODULES,
    assert_derived_entries_install,
    dispatch_census,
    entry_tuples,
    gate_census,
    scope_residual,
)
from basemap import round0266_int8_treatment as T
from basemap.round0266_int8_treatment import (
    CAPABILITY,
    CLOSURE_SCHEMA,
    ROUND_ID,
    SEED,
    TRAIN_CLOSURE_MODULES,
    assert_registered_int8_recipe,
    int8_train_config,
    runtime_closure_hashes,
)
from basemap import round0265_fneg_treatment as R0265
from basemap.round0217_minilm_2m_seed_family import (
    ROWS,
    SEALED_DIRECTED_EDGES,
    successful_updates_for_edges,
    seed_invariant_sha256,
)
from experiments.round0266_nodes import (
    GATE_ACTION,
    GATE_CAPABILITY,
    PANEL_ACTION,
    PANEL_CAPABILITY,
    TRAIN_ACTION,
)
import experiments.round0265_nodes as R0265N
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
# Reuse R0265's prepare data anchors verbatim (same substrate/graph/panel/held-out truth).
from experiments.prepare_round0265_queue import (
    GRAPH_MANIFEST,
    HELDOUT_PROBES,
    HELDOUT_TRUTH,
    R0218_PANEL,
    _sealed_graph,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0266"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0266-2026-08-15.md")

#: R0265's SEALED family floors (gate-only-1 re-seal) — criterion (a) bands.
R0265_FLOORS = (
    "/data/latent-basemap/runs/round-0265/queue-gate-only-1/artifacts/"
    "minilm-fneg-2m-family-floors-n13-v1/minilm-fneg-family-floors-n13.json"
)
#: R0265's SEALED n=13 panel (correction-3) — criterion (b) fp32 sibling + 13-seed range.
R0265_PANEL = (
    "/data/latent-basemap/runs/round-0265/queue-correction-3/artifacts/"
    "minilm-fneg-2m-family-panel-n13-v1/fneg-family-panel-n13.json"
)
#: R0265's seed-42 fp32 SIBLING train receipt (correction-3) — the paired-cell lineage
#: whose panel metrics criterion (b) reads from the sealed panel.
R0265_FP32_SIBLING = (
    "/data/latent-basemap/runs/round-0265/queue-correction-3/artifacts/"
    "minilm-mixed-2m-fneg-x4-md000-seed42-r0265-v1/train-receipt.json"
)

#: One x4-dose 2M train (~0.78 GPU-h) + a one-map panel; the cap sits above the ~1 GPU-h
#: estimate rather than tight to it.
GPU_HOURS_CAP = 2.0
TRAIN_P90_WALL_S = 5_400.0
PANEL_P90_WALL_S = 1_800.0
GATE_P90_WALL_S = 600.0


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
        raise RuntimeError(
            "R0266 round is not issued for this release. EXPECTED until the round file is "
            "issued and its base_commit is an ancestor of the release."
        )
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0266 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _upstream_review_state(required: list[str]) -> dict[str, Any]:
    import glob

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
        "note": (
            "Review is post-hoc: it blocks the downstream claim, not the launch. The "
            "map-level int8 fidelity verdict this round registers is "
            "registered-and-contingent until its required upstream reviews are accepted."
        ),
    }


def _signature(path: str, label: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise RuntimeError(f"R0266 bound input absent: {label} at {path}")
    return expected_input_signature(path)


def _treatment_closure_seal(release_sha: str) -> dict[str, Any]:
    """SHA-256 of every R0266 training-closure source (R0265's closure + the int8 module)."""
    observed = runtime_closure_hashes(TRAIN_CLOSURE_MODULES)
    files: dict[str, Any] = {}
    for name, entry in observed.items():
        relative = os.path.relpath(
            entry["path"], os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        files[name] = {
            "module": name,
            "path": relative,
            "bytes_at_release": entry["bytes"],
            "sha256_at_release": entry["sha256"],
        }
    return prompt_contract.seal({
        "schema": CLOSURE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "modules": list(TRAIN_CLOSURE_MODULES),
        "files": files,
        "how_to_read_this": (
            "R0266 is R0265's fneg recipe plus one field (x_residency=host_int8). This "
            "seals the R0266 training import closure: R0265's closure (the fneg-merged "
            "core.py commit 9794b4b included) PLUS the round0266 int8 recipe module. A "
            "node that ran different bytes for either would refuse."
        ),
    })


def prepare_round0266(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0266 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    review_state = _upstream_review_state(list(required_reviews))
    graph_manifest_signature, graph_manifest, edges = _sealed_graph()
    base_horizon = successful_updates_for_edges(edges)

    substrate_signature = dict(graph_manifest["substrate"])
    graph_signature = dict(graph_manifest["graph"])

    # Build and prove the int8 cell config here (needs no train artifacts).
    config, config_sha = int8_train_config(
        seed=SEED,
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        substrate_signature=substrate_signature,
        graph_edges=edges,
        rows=ROWS,
    )
    recipe = assert_registered_int8_recipe(config)
    cell_seed_invariant = seed_invariant_sha256(config)

    # Bind every sealed cross-round input by real sha256 + bytes.
    r0218_panel = _signature(R0218_PANEL, "R0218 frozen panel")
    heldout_probes = _signature(HELDOUT_PROBES, "R0233 held-out probes")
    heldout_truth = _signature(HELDOUT_TRUTH, "R0233 held-out truth top10")
    r0265_floors = _signature(R0265_FLOORS, "R0265 sealed family floors")
    r0265_panel = _signature(R0265_PANEL, "R0265 sealed n=13 panel")
    r0265_fp32_sibling = _signature(R0265_FP32_SIBLING, "R0265 seed-42 fp32 sibling receipt")

    # Cross-check the bound sealed R0265 artifacts really are what the gate expects, and
    # confirm the seed-42 fp32 sibling is present in the panel — so a swapped input is
    # caught at prepare, not only at gate time.
    floors_sealed = prompt_contract.read_sealed(
        r0265_floors["canonical_path"], label="R0265 sealed family floors"
    )
    if (
        floors_sealed.get("capability") != R0265N.GATE_CAPABILITY
        or floors_sealed.get("gate_registered") is not True
    ):
        raise RuntimeError("R0266 --r0265-floors is not the sealed R0265 family floors gate")
    panel_sealed = prompt_contract.read_sealed(
        r0265_panel["canonical_path"], label="R0265 sealed n=13 panel"
    )
    if (
        panel_sealed.get("capability") != R0265N.PANEL_CAPABILITY
        or int(panel_sealed.get("n", -1)) != R0265N.N_FAMILY
        or str(SEED) not in dict(panel_sealed.get("panel_metric_table") or {})
    ):
        raise RuntimeError("R0266 --r0265-panel is not the sealed n=13 panel with a seed-42 cell")

    census = dispatch_census()
    guard = assert_derived_entries_install(SCOPE_MODULES, census)
    gates = gate_census(entry_tuples(guard["derived"]))
    residual = scope_residual(census, SCOPE_MODULES)

    ensure_data_directory(ROUND_ROOT, label="R0266 round root")
    queue_root = create_fresh_directory(queue_root, label="R0266 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    closure_path = os.path.join(preflight, "treatment-source-closure.json")
    atomic_write_new_json(closure_path, _treatment_closure_seal(release_sha), immutable=True)
    identity_path = os.path.join(preflight, "hostint8-cell-identity.json")
    atomic_write_new_json(
        identity_path,
        prompt_contract.seal({
            "schema": "round0266-hostint8-cell-identity-v1",
            "round_id": ROUND_ID,
            "release_sha": release_sha,
            "sealed_directed_edges": edges,
            "base_horizon": base_horizon,
            "seed": SEED,
            "capability": CAPABILITY,
            "x_residency": T.X_RESIDENCY,
            "recipe": recipe,
            "cell_seed_invariant_sha256": cell_seed_invariant,
            "config": config,
            "config_sha256": config_sha,
            "base_recipe_schema": R0265.FNEG_RECIPE_SCHEMA,
            "registry_fingerprint": registry_fingerprint(),
        }),
        immutable=True,
    )
    closure_signature = expected_input_signature(closure_path)

    shared_inputs = _dedupe([
        round_signature,
        graph_manifest_signature,
        substrate_signature,
        graph_signature,
        expected_input_signature(identity_path),
        closure_signature,
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    p90: dict[str, float] = {}
    jobs: list[dict[str, Any]] = []

    # 1. the host-int8 train.
    train_node = TRAIN_ACTION
    train_output = os.path.join(artifacts, CAPABILITY)
    jobs.append({
        "id": train_node,
        "action": TRAIN_ACTION,
        "handler_module": "experiments.round0266_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [train_output],
        "done_marker": os.path.join(artifacts, f"{train_node}.done.json"),
        "expected_inputs": shared_inputs,
        "p90_wall_s": TRAIN_P90_WALL_S,
        "training_seed": int(SEED),
        "capability": CAPABILITY,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": graph_manifest_signature,
        "cell_seed_invariant_sha256": cell_seed_invariant,
        "base_horizon": base_horizon,
        "treatment_closure": closure_signature,
        "node_policy": {"gpu_required": True, "training_performed": True, "cpu_heavy": False},
    })
    p90[train_node] = TRAIN_P90_WALL_S

    # 2. the single-cell panel.
    panel_node = PANEL_ACTION
    panel_output = os.path.join(artifacts, PANEL_CAPABILITY)
    jobs.append({
        "id": panel_node,
        "action": PANEL_ACTION,
        "handler_module": "experiments.round0266_nodes",
        "handler_callable": "run_job",
        "deps": [train_node],
        "outputs": [panel_output],
        "done_marker": os.path.join(artifacts, f"{panel_node}.done.json"),
        "expected_inputs": _dedupe([*shared_inputs, r0218_panel, heldout_probes, heldout_truth]),
        "p90_wall_s": PANEL_P90_WALL_S,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": graph_manifest_signature,
        "panel_evidence": R0218_PANEL,
        "centroid_ks": list(CENTROID_KS),
        "heldout_probes": heldout_probes,
        "heldout_truth": heldout_truth,
        "cell": {
            "seed": int(SEED),
            "capability": CAPABILITY,
            "train_receipt": {"kind": "file", "canonical_path": os.path.join(train_output, "train-receipt.json")},
        },
        "gate_registerable_here": False,
        "upstream_review_state": review_state,
        "node_policy": {"gpu_required": True, "training_performed": False, "cpu_heavy": False},
    })
    p90[panel_node] = PANEL_P90_WALL_S

    # 3. the map-level int8-fidelity gate (CPU).
    gate_node = GATE_ACTION
    jobs.append({
        "id": gate_node,
        "action": GATE_ACTION,
        "handler_module": "experiments.round0266_nodes",
        "handler_callable": "run_job",
        "deps": [panel_node],
        "outputs": [os.path.join(artifacts, GATE_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{gate_node}.done.json"),
        "expected_inputs": _dedupe([
            *shared_inputs, r0265_floors, r0265_panel, r0265_fp32_sibling,
        ]),
        "p90_wall_s": GATE_P90_WALL_S,
        "int8_panel": {"kind": "file", "canonical_path": os.path.join(panel_output, "hostint8-cell-panel.json")},
        "r0265_floors": r0265_floors,
        "r0265_panel": r0265_panel,
        "r0265_fp32_sibling": r0265_fp32_sibling,
        "upstream_review_state": review_state,
        "node_policy": {"gpu_required": False, "training_performed": False, "cpu_heavy": True},
    })
    p90[gate_node] = GATE_P90_WALL_S
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
        "schema": "round0266-hostint8-map-fidelity-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            R0216_CAPABILITY,
            R0218_PANEL_CAPABILITY,
            R0265N.GATE_CAPABILITY,
            R0265N.PANEL_CAPABILITY,
        ],
        "capabilities_produced": [CAPABILITY, PANEL_CAPABILITY, GATE_CAPABILITY],
        "jobs": jobs,
        "p90_wall_s": p90,
        "scope_modules": list(SCOPE_MODULES),
        "stop_hook_install_guard": {
            "derived_entries": guard["derived"],
            "every_derived_entry_installs": guard["audit"]["every_entry_installs_effectively"],
            "gate_census": gates,
            "scope_residual": residual,
        },
        "registered": {
            "what_this_round_is": (
                "train ONE 2M cell (seed 42) of the promoted pinned fneg recipe with the "
                "single delta x_residency=host_int8, score it on R0265's instruments, and "
                "gate map-level int8 fidelity: (a) the cell's five gated metrics INSIDE "
                "R0265's sealed family bands, and (b) |int8 - fp32_seed42| per metric <= "
                "the family's 13-seed observed range. Every band/range is read from a "
                "sealed R0265 artifact bound by sha256 -- never a typed literal."
            ),
            "seed": int(SEED),
            "seed_paired_to": "R0265 seed-42 (the fp32 sibling)",
            "x_residency": T.X_RESIDENCY,
            "cell_seed_invariant_sha256": cell_seed_invariant,
            "base_horizon": base_horizon,
            "x4_horizon": int(config["optimizer"]["successful_positive_lr_updates"]),
            "sealed_directed_edges": edges,
            "rows": ROWS,
            "consumes_sealed_r0265_instruments": {
                "family_floors": R0265N.GATE_CAPABILITY,
                "n13_panel": R0265N.PANEL_CAPABILITY,
                "fp32_sibling_sha256": r0265_fp32_sibling["sha256"],
            },
            "r0262_map_level_obligation": (
                "review-0262-01 §H.1 demanded a MAP-LEVEL int8 fidelity cell before any "
                "long int8 train; R0262 proved fidelity only at the gradient level on a "
                "600-update proxy. This round IS that map-level evidence."
            ),
            "gate_registerable_here": True,
            "acceptance_rule": (
                "the round trains the int8 cell, scores it, and registers the fidelity "
                "gate. NO NUMERICAL OUTCOME makes it a failure: whether criteria (a) and "
                "(b) hold is a MEASUREMENT reported either way, contingent on review."
            ),
        },
    })
    manifest_path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(manifest_path, queue, immutable=True)
    return manifest_path


def file_sha256_manifest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="prepare the R0266 queue")
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=None)
    args = parser.parse_args(argv)
    path = prepare_round0266(release_sha=args.release_sha, queue_root=(args.queue_root or QUEUE_ROOT))
    print(json.dumps({"queue": path, "sha256": file_sha256_manifest(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
