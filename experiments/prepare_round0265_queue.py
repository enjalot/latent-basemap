#!/usr/bin/env python3
"""Prepare, but never launch, the R0265 queue — the largest round of the program.

Fifteen nodes in one queue, in this order:

1-13. `train_minilm_fneg_2m_seed{42..54}` (GPU) — the thirteen fneg cells, one map per
      seed, each built from R0217's template with the PINNED fneg recipe applied and
      proved by `round0265_fneg_treatment.assert_registered_recipe`.
14.   `score_minilm_fneg_2m_panel_n13` (GPU) — the thirteen maps scored on held-out FFR
      (R0233's sealed reserve truth), purity_fidelity_k256/k1024 (R0218's frozen panel
      reference), COLLAPSE and FOG, plus the seed-42 cross-check.
15.   `register_fneg_family_floors_n13` (CPU) — pool n=13, read R0234's sealed 95/95
      n=13 multiplier, fit the held-out FFR floor, CONVERT R0264's provisional
      collapse/fog bands to family-calibrated, and GATE on held-out FFR floor + R0263's
      two-sided k256 band + collapse floor + fail-closed fog ceiling.

The script builds all thirteen fneg configs HERE, proves they share one recipe digest
and are thirteen distinct configs, and seals the import-source closure (the fneg-merged
`core.py` first) so every training node refuses unless the bytes it imported are the
sealed ones. The panel and gate consume artifacts this queue has not produced yet, so
their references to them are intra-queue; the sealed cross-round inputs (R0216 graph,
R0218 panel, R0263 band, R0264 provisional bands, R0234 n=13, the held-out truth) are
bound by real sha256 + bytes computed from the file on disk at prepare time.

This builder REFUSES until the round file is issued and every bound input exists —
correct while R0264 is still a draft: R0265 cannot be prepared until its provisional
collapse/fog bands are sealed.
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
from basemap.round0265_fneg_treatment import (
    CANONICAL_SEED,
    CLOSURE_SCHEMA,
    DOSE_MULTIPLIER,
    FNEG_HORIZON,
    ROUND_ID,
    SEEDS,
    TRAIN_CLOSURE_MODULES,
    assert_family_shares_one_recipe,
    capability_for_seed,
    file_sha256,
    fneg_train_config,
    runtime_closure_hashes,
    validate_fneg_dose,
)
from basemap.round0221_minilm_2m_seed_extension import (
    SEALED_GRAPH_MANIFEST_SIGNATURE,
    SEALED_GRAPH_SIGNATURE,
    SEALED_SUBSTRATE_SIGNATURE,
)
from basemap.round0217_minilm_2m_seed_family import (
    ROWS,
    SEALED_DIRECTED_EDGES,
    successful_updates_for_edges,
)
from experiments.round0265_nodes import (
    GATE_ACTION,
    GATE_CAPABILITY,
    PANEL_ACTION,
    PANEL_CAPABILITY,
    R0234_N13_TWO_SIDED_KEY,
    TRAIN_ACTION,
    TRAIN_SCHEMA,
)
from experiments.round0263_nodes import GATE_CAPABILITY as R0263_GATE_CAPABILITY
from experiments.round0264_nodes import GATE_CAPABILITY as R0264_GATE_CAPABILITY
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0265"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0265-2026-08-14.md")

R0216_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    f"{R0216_CAPABILITY}"
)
GRAPH_MANIFEST = os.path.join(R0216_ARTIFACTS, "substrate-graph.json")
R0218_PANEL = (
    f"/data/latent-basemap/runs/round-0218/queue/artifacts/{R0218_PANEL_CAPABILITY}/"
    "seed-family-panel.json"
)
R0234_GATE = (
    "/data/latent-basemap/runs/round-0234/queue/artifacts/"
    "minilm-mixed-2m-calibrated-robust-floors-n13-v1/"
    "minilm-calibrated-robust-floors-n13.json"
)
R0263_GATE = (
    "/data/latent-basemap/runs/round-0263/queue/artifacts/"
    f"{R0263_GATE_CAPABILITY}/minilm-two-sided-k256-purity-band-n29.json"
)
R0264_GATE = (
    "/data/latent-basemap/runs/round-0264/queue-correction-1/artifacts/"
    f"{R0264_GATE_CAPABILITY}/minilm-collapse-floor-fog-ceiling-provisional-n29.json"
)
#: R0233's sealed 20k held-out reserve truth, copied into the round's own /data inputs
#: dir (owner places the files there), the probe embeddings and their exact cosine
#: top-10 substrate row indices. Bound by real sha256 + bytes below.
HELDOUT_TRUTH_DIR = "/data/latent-basemap/runs/round-0265/inputs/heldout-truth-2m"
HELDOUT_PROBES = os.path.join(HELDOUT_TRUTH_DIR, "probes.f32.npy")
HELDOUT_TRUTH = os.path.join(HELDOUT_TRUTH_DIR, "truth-top10.npy")

#: Thirteen fneg cells at x4 dose (~0.78 GPU-h each on R0255's x1 measurement scaled by
#: the x4 horizon) plus a thirteen-map panel; the registered cap sits above the ~11 h
#: estimate rather than tight to it.
GPU_HOURS_CAP = 12.0
TRAIN_P90_WALL_S = 7_200.0
PANEL_P90_WALL_S = 7_200.0
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
            "R0265 round is not issued for this release. EXPECTED until the round file "
            "is issued and R0264's provisional bands are sealed."
        )
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0265 round must declare its required reviews")
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
            "n=13 family floors this round registers are registered-and-contingent "
            "until R0263 and R0264 carry accepted reviews."
        ),
    }


def _signature(path: str, label: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise RuntimeError(f"R0265 bound input absent: {label} at {path}")
    return expected_input_signature(path)


def _sealed_graph() -> tuple[dict[str, Any], dict[str, Any], int]:
    signature = expected_input_signature(GRAPH_MANIFEST)
    manifest = prompt_contract.read_sealed(
        signature["canonical_path"], label="sealed R0216 substrate+graph receipt"
    )
    edges = int(manifest.get("directed_edge_count", 0)) or int(
        (manifest.get("graph_checks") or {}).get("directed_edges", 0)
    )
    if edges != SEALED_DIRECTED_EDGES:
        raise RuntimeError(
            f"R0265 sealed graph reports {edges} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    return signature, manifest, edges


def _treatment_closure_seal(release_sha: str) -> dict[str, Any]:
    """SHA-256 of every training-closure source at this release — the fneg-merged core."""
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
            "the fneg recipe is a NEW treatment, so there is no prior release to diff "
            "against; this seals the current bytes of the training import closure. The "
            "fneg-merged core.py (commit 9794b4b) is a member, so a node that ran a "
            "pre-merge core would refuse."
        ),
    })


def prepare_round0265(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0265 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    review_state = _upstream_review_state(list(required_reviews))
    graph_manifest_signature, graph_manifest, edges = _sealed_graph()
    base_horizon = successful_updates_for_edges(edges)
    dose = validate_fneg_dose(updates=DOSE_MULTIPLIER * base_horizon, edge_count=edges)

    substrate_signature = dict(graph_manifest["substrate"])
    graph_signature = dict(graph_manifest["graph"])

    # Build the thirteen fneg configs here and prove they are one recipe, seed aside.
    configs: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        config, _sha = fneg_train_config(
            seed=seed,
            graph_signature=graph_signature,
            graph_manifest_signature=graph_manifest_signature,
            substrate_signature=substrate_signature,
            graph_edges=edges,
            rows=ROWS,
        )
        configs[seed] = config
    family = assert_family_shares_one_recipe(configs)

    # Bind every sealed cross-round input by real sha256 + bytes.
    r0234_n13 = _signature(R0234_GATE, "R0234 sealed n=13 calibration")
    r0263_gate = _signature(R0263_GATE, "R0263 sealed two-sided k256 band")
    r0264_gate = _signature(R0264_GATE, "R0264 sealed provisional collapse/fog bands")
    r0218_panel = _signature(R0218_PANEL, "R0218 frozen panel")
    heldout_probes = _signature(HELDOUT_PROBES, "R0233 held-out probes")
    heldout_truth = _signature(HELDOUT_TRUTH, "R0233 held-out truth top10")

    census = dispatch_census()
    guard = assert_derived_entries_install(SCOPE_MODULES, census)
    gates = gate_census(entry_tuples(guard["derived"]))
    residual = scope_residual(census, SCOPE_MODULES)

    ensure_data_directory(ROUND_ROOT, label="R0265 round root")
    queue_root = create_fresh_directory(queue_root, label="R0265 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    closure_path = os.path.join(preflight, "treatment-source-closure.json")
    atomic_write_new_json(closure_path, _treatment_closure_seal(release_sha), immutable=True)
    family_path = os.path.join(preflight, "fneg-family-identity.json")
    atomic_write_new_json(
        family_path,
        prompt_contract.seal({
            "schema": "round0265-fneg-family-identity-v1",
            "round_id": ROUND_ID,
            "release_sha": release_sha,
            "sealed_directed_edges": edges,
            "base_horizon": base_horizon,
            "dose_registration": dose,
            "family": family,
            "seeds": list(SEEDS),
            "n_pooled": len(SEEDS),
            "registry_fingerprint": registry_fingerprint(),
            "configs": {str(seed): configs[seed] for seed in SEEDS},
        }),
        immutable=True,
    )
    closure_signature = expected_input_signature(closure_path)

    shared_inputs = _dedupe([
        round_signature,
        graph_manifest_signature,
        substrate_signature,
        graph_signature,
        expected_input_signature(family_path),
        closure_signature,
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    jobs: list[dict[str, Any]] = []
    p90: dict[str, float] = {}
    panel_cells: list[dict[str, Any]] = []
    train_nodes: list[str] = []

    for seed in SEEDS:
        capability = capability_for_seed(seed)
        node = f"train_minilm_fneg_2m_seed{seed}"
        output = os.path.join(artifacts, capability)
        jobs.append({
            "id": node,
            "action": TRAIN_ACTION,
            "handler_module": "experiments.round0265_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"{node}.done.json"),
            "expected_inputs": shared_inputs,
            "p90_wall_s": TRAIN_P90_WALL_S,
            "training_seed": int(seed),
            "capability": capability,
            "graph_manifest": GRAPH_MANIFEST,
            "graph_manifest_signature": graph_manifest_signature,
            "family_seed_invariant_sha256": family["seed_invariant_sha256"],
            "base_horizon": base_horizon,
            "treatment_closure": closure_signature,
            "node_policy": {"gpu_required": True, "training_performed": True, "cpu_heavy": False},
        })
        p90[node] = TRAIN_P90_WALL_S
        train_nodes.append(node)
        panel_cells.append({
            "seed": int(seed),
            "capability": capability,
            "train_receipt": {"kind": "file", "canonical_path": os.path.join(output, "train-receipt.json")},
        })

    panel_node = PANEL_ACTION
    panel_output = os.path.join(artifacts, PANEL_CAPABILITY)
    jobs.append({
        "id": panel_node,
        "action": PANEL_ACTION,
        "handler_module": "experiments.round0265_nodes",
        "handler_callable": "run_job",
        "deps": list(train_nodes),
        "outputs": [panel_output],
        "done_marker": os.path.join(artifacts, f"{panel_node}.done.json"),
        "expected_inputs": _dedupe([
            *shared_inputs, r0218_panel, heldout_probes, heldout_truth,
        ]),
        "p90_wall_s": PANEL_P90_WALL_S,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": graph_manifest_signature,
        "panel_evidence": R0218_PANEL,
        "centroid_ks": list(CENTROID_KS),
        "heldout_probes": heldout_probes,
        "heldout_truth": heldout_truth,
        "cells": panel_cells,
        "gate_registerable_here": False,
        "upstream_review_state": review_state,
        "node_policy": {"gpu_required": True, "training_performed": False, "cpu_heavy": False},
    })
    p90[panel_node] = PANEL_P90_WALL_S

    gate_node = GATE_ACTION
    jobs.append({
        "id": gate_node,
        "action": GATE_ACTION,
        "handler_module": "experiments.round0265_nodes",
        "handler_callable": "run_job",
        "deps": [panel_node],
        "outputs": [os.path.join(artifacts, GATE_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{gate_node}.done.json"),
        "expected_inputs": _dedupe([*shared_inputs, r0234_n13, r0263_gate, r0264_gate]),
        "p90_wall_s": GATE_P90_WALL_S,
        "panel_n13": {"kind": "file", "canonical_path": os.path.join(panel_output, "fneg-family-panel-n13.json")},
        "r0234_n13": r0234_n13,
        "r0263_gate": r0263_gate,
        "r0264_gate": r0264_gate,
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
        "schema": "round0265-fneg-family-floors-n13-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            R0216_CAPABILITY,
            R0218_PANEL_CAPABILITY,
            R0263_GATE_CAPABILITY,
            R0264_GATE_CAPABILITY,
            "minilm-mixed-2m-calibrated-robust-floors-n13-v1",
        ],
        "capabilities_produced": [
            *(capability_for_seed(seed) for seed in SEEDS),
            PANEL_CAPABILITY,
            GATE_CAPABILITY,
        ],
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
                "train thirteen fneg 2M cells (seeds 42-54) of the promoted pinned "
                "recipe (umap a=1.9328/b=0.7905, dose x4, fneg_weight=1.0/lo=0.1/hi=0.4), "
                "pool them at n=13, fit the held-out FFR floor and CONVERT R0264's "
                "provisional collapse/fog bands to family-calibrated on this fneg family, "
                "and gate on held-out FFR floor + R0263's two-sided k256 band + collapse "
                "floor + fail-closed fog ceiling. density_v2/v3 stay descriptive-only."
            ),
            "seeds": list(SEEDS),
            "n_pooled": len(SEEDS),
            "seed_invariant_sha256": family["seed_invariant_sha256"],
            "dose_multiplier": DOSE_MULTIPLIER,
            "base_horizon": base_horizon,
            "x4_horizon": FNEG_HORIZON,
            "sealed_directed_edges": edges,
            "rows": ROWS,
            "n13_multiplier_source": R0234_N13_TWO_SIDED_KEY,
            "consumes_registered_instruments": {
                "r0263_two_sided_k256_band": R0263_GATE_CAPABILITY,
                "r0264_provisional_collapse_fog": R0264_GATE_CAPABILITY,
            },
            "gate_registerable_here": True,
            "acceptance_rule": (
                "the round trains thirteen fneg cells, pools them, fits the floors and "
                "registers the gate. NO NUMERICAL OUTCOME makes it a failure: whether "
                "the cells clear their own family floors, whether the converted bands "
                "move the provisional ones, and whether seed-42 lands metric-close are "
                "all MEASUREMENTS reported either way, contingent on review."
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
    parser = argparse.ArgumentParser(description="prepare the R0265 queue")
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    args = parser.parse_args(argv)
    path = prepare_round0265(release_sha=args.release_sha, queue_root=args.queue_root)
    print(json.dumps({"queue": path, "sha256": file_sha256_manifest(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
