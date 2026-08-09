#!/usr/bin/env python3
"""Prepare, but never launch, the R0228 cluster-spill map queue.

Fifteen nodes in one queue: one `c = 4` cluster-spill build, three fuzzy-graph
nodes, nine train cells, one panel comparison, one geometry probe.

**Everything that can fail cheaply fails here.** R0223 lost `0.60` GPU-h because
a schema constant in a node that runs *after* three train cells was a plausible
name rather than the sealed one. This round runs nine train cells, so the same
mistake would cost `1.8` GPU-h. So this script opens and validates, before any
GPU work:

* R0216's sealed substrate + exact graph receipt;
* R0220's sealed exact-truth receipt and both truth arrays;
* R0227's sealed ladder and the two 2M builds it will reuse (`c = 8`, `c = 16`),
  including their child receipts and emitted neighbour-id arrays;
* R0218's sealed panel, R0222's sealed `n = 8` gate, **R0225's sealed tolerance
  gates** and **R0223's sealed cuVS comparison** — every schema, every key the
  nodes will read, and every coordinate array the geometry node will load;
* the config construction itself, against a hypothetical edge count, which must
  reproduce the cross-round treatment digest `c28cfd61...`.
"""
from __future__ import annotations

import argparse
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
    GRAPH_CAPABILITY as R0216_GRAPH_CAPABILITY,
    GRAPH_SCHEMA as R0216_GRAPH_SCHEMA,
    HIDDEN_DIMENSION,
)
from basemap.round0218_minilm_2m_panel import CAPABILITY as R0218_PANEL_CAPABILITY
from basemap.round0227_low_c_contract import (
    BUILD_SCHEMA as R0227_BUILD_SCHEMA,
    LADDER_SCHEMA as R0227_LADDER_SCHEMA,
    LOW_C_CAPABILITY as R0227_LADDER_CAPABILITY,
    guard_decision,
)
from basemap.round0228_low_c_map import (
    ADOPTION_CLAIMED,
    CELLS,
    CLUSTERS_BUILT_HERE,
    CLUSTERS_FROM_R0227,
    CLUSTER_COUNTS,
    CLUSTER_SPILL_BUILDER,
    COMPARISON_CAPABILITY,
    DENSITY_V2_STATUS,
    DIMENSION,
    EQUIVALENCE_CLAIMED,
    EVIDENCE_LIMITS,
    EXACT_FAMILY_SEEDS,
    GATED_METRICS,
    GATE_REGISTERABLE_HERE,
    GATE_RELEASE_CLAIMED,
    GEOMETRY_CAPABILITY,
    GRAPH_CAPABILITIES,
    GRAPH_K,
    HOST_RSS_LIMIT_GIB,
    IDENTITY_BOUND_NOTE,
    MAP_CAPABILITIES,
    R0216_SEALED_DIRECTED_EDGES,
    R0217_TREATMENT_INVARIANT_SHA256,
    R0222_GATE_SCHEMA,
    R0223_COMPARISON_SCHEMA,
    R0223_CUVS_SEEDS,
    R0225_GATE_SCHEMA,
    R0227_TIE_AWARE_RECALL_BY_C,
    RECALL_POPULATION,
    RECALL_POPULATION_NOTE,
    REGISTERED_UPDATE_BOUND,
    ROUND_ID,
    ROWS,
    SEEDS,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    TEMPLATE_SEED,
    USE_AMP,
    graph_capability,
    map_capability,
    train_config,
    treatment_invariant_sha256,
)
from basemap.round0228_geometry import SCATTER_SAMPLE_ROWS
from experiments.round0228_nodes import (
    CLUSTER_BUILD_ACTION,
    COMPARE_ACTION,
    FUZZY_ACTION,
    GEOMETRY_ACTION,
    TRAIN_ACTION,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0228"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0228-2026-08-08.md")

R0216_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    f"{R0216_CAPABILITY}"
)
R0216_GRAPH_MANIFEST = os.path.join(R0216_ARTIFACTS, "substrate-graph.json")
R0218_PANEL_EVIDENCE = (
    "/data/latent-basemap/runs/round-0218/queue/artifacts/"
    f"{R0218_PANEL_CAPABILITY}/seed-family-panel.json"
)
R0220_TRUTH_RECEIPT = (
    "/data/latent-basemap/runs/round-0220/queue-correction-1/artifacts/"
    "exact-k15-truth/truth-rebuild.json"
)
R0222_GATE_ARTIFACT = (
    "/data/latent-basemap/runs/round-0222/queue/artifacts/"
    "minilm-mixed-2m-quality-gates-n8-v1/minilm-quality-gates-n8.json"
)
R0223_COMPARISON_ARTIFACT = (
    "/data/latent-basemap/runs/round-0223/queue-correction-3/artifacts/"
    "minilm-mixed-2m-cuvs-graph-map-comparison-v1/cuvs-graph-map-comparison.json"
)
R0225_GATE_ARTIFACT = (
    "/data/latent-basemap/runs/round-0225/queue/artifacts/"
    "minilm-mixed-2m-tolerance-gates-n8-v1/minilm-tolerance-gates-n8.json"
)
R0227_LADDER_ROOT = (
    "/data/latent-basemap/runs/round-0227/queue/artifacts/"
    f"{R0227_LADDER_CAPABILITY}"
)
R0227_LADDER_ARTIFACT = os.path.join(R0227_LADDER_ROOT, "low-c-build-ladder.json")

CUVS_CACHE_ROOT = os.path.join(ROUND_ROOT, "child-cache")
SCRATCH_ROOT = "/data/latent-basemap/scratch/round-0228"

#: R0217 measured ~0.197 GPU-h per 2M train cell; nine cells is ~1.78 GPU-h. One
#: `c = 4` build (R0227 measured 2M cluster-spill builds at ~20-40 s), three
#: fuzzy-graph nodes, one nine-cell panel and one geometry probe add ~0.12.
#: The registered cap is the mandate's 3.0.
GPU_HOURS_CAP = 3.0
CLUSTER_BUILD_P90_WALL_S = 1_800.0
FUZZY_NODE_P90_WALL_S = 900.0
TRAIN_NODE_P90_WALL_S = 1_800.0
COMPARE_NODE_P90_WALL_S = 1_800.0
GEOMETRY_NODE_P90_WALL_S = 1_800.0

#: Used only to prove the config construction at prepare time.
CONSTRUCTION_PROBE_EDGES = 48_000_000


def _issued_round(release_sha: str) -> tuple[dict[str, Any], list[str]]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        [
            "git",
            "-C",
            RELEASE_ROOT,
            "merge-base",
            "--is-ancestor",
            base_commit,
            release_sha,
        ],
        check=False,
        timeout=10,
    ).returncode == 0
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or not descendant
    ):
        raise RuntimeError("R0228 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0228 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0228 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0228_low_c_map.py",
        "tests/test_round0228_cpu_smoke.py",
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    })
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    receipt = prompt_contract.seal({
        "schema": "round0228-release-cpu-smoke-v1",
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
            "R0217-template config construction with a cluster-spill graph "
            "swapped in, cross-round treatment-digest equality, ceil-derived "
            "dose, graph validation, published-map validation, the exact "
            "permutation tests and the trend test against known answers, the "
            "full comparison arithmetic against a synthetic family, the clump "
            "detector and the density-matched displacement battery, config "
            "seal, post-fit accounting, checkpoint publish, reload and a "
            "downstream panel call"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0228 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _sealed_r0216() -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(R0216_GRAPH_MANIFEST)
    manifest = prompt_contract.read_sealed(
        signature["canonical_path"], label="sealed R0216 substrate+graph receipt"
    )
    checks = manifest.get("graph_checks") or {}
    edges = int(manifest.get("directed_edge_count", 0)) or int(
        checks.get("directed_edges", 0)
    )
    if (
        manifest.get("schema") != R0216_GRAPH_SCHEMA
        or manifest.get("capability") != R0216_GRAPH_CAPABILITY
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or int(manifest.get("k", -1)) != GRAPH_K
        or int(checks.get("zero_degree_rows", -1)) != 0
        or edges != R0216_SEALED_DIRECTED_EDGES
    ):
        raise RuntimeError("R0228 sealed R0216 substrate+graph contract changed")
    return signature, manifest


def _sealed_truth() -> dict[str, Any]:
    signature = expected_input_signature(R0220_TRUTH_RECEIPT)
    truth = prompt_contract.read_sealed(
        R0220_TRUTH_RECEIPT, label="R0220 exact k15 truth receipt"
    )
    if (
        truth.get("round_id") != "0220"
        or int(truth.get("rows", -1)) != ROWS
        or int(truth.get("k", -1)) != GRAPH_K
        or not truth["probe"]["passed"]
    ):
        raise RuntimeError("R0228 sealed R0220 truth receipt contract changed")
    for key in ("ids", "cosines"):
        prompt_contract.verify_signature(
            truth["outputs"][key], label=f"R0220 truth {key}"
        )
    return signature


def _sealed_r0227_builds() -> dict[int, dict[str, Any]]:
    """R0227's sealed 2M neighbour ids for `c = 8` and `c = 16`, read-only."""
    ladder_signature = expected_input_signature(R0227_LADDER_ARTIFACT)
    ladder = prompt_contract.read_sealed(
        R0227_LADDER_ARTIFACT, label="R0227 sealed low-c build ladder"
    )
    if (
        ladder.get("schema") != R0227_LADDER_SCHEMA
        or ladder.get("round_id") != "0227"
        or ladder.get("candidate") != CLUSTER_SPILL_BUILDER
    ):
        raise RuntimeError("R0228 sealed R0227 ladder contract changed")
    fitted = {
        int(item["clusters"]): item
        for item in ladder.get("builds") or []
        if int(item.get("rows", -1)) == ROWS and item.get("fit")
    }
    out: dict[int, dict[str, Any]] = {}
    for clusters in CLUSTERS_FROM_R0227:
        if clusters not in fitted:
            raise RuntimeError(
                f"R0228 needs a fitted R0227 2M build at c={clusters}; the sealed "
                "ladder has none"
            )
        setting_id = str(fitted[clusters]["setting_id"])
        build_dir = os.path.join(R0227_LADDER_ROOT, "builds", setting_id)
        ids_path = os.path.join(build_dir, "graph-k15-ids.i32.npy")
        receipt_path = os.path.join(build_dir, "build-receipt.json")
        for path in (ids_path, receipt_path):
            if not os.path.exists(path):
                raise RuntimeError(f"R0228 R0227 artifact is absent: {path}")
        with open(receipt_path, encoding="utf-8") as handle:
            child = json.load(handle)
        if (
            child.get("schema") != R0227_BUILD_SCHEMA
            or int(child.get("clusters", -1)) != clusters
            or int(child.get("rows", -1)) != ROWS
            or int(child.get("k", -1)) != GRAPH_K
            or child.get("fit") is not True
            or child.get("graph_emitted") is not True
            or int(child.get("zero_degree_rows", -1)) != 0
            or int(child.get("rows_below_k", -1)) != 0
        ):
            raise RuntimeError(
                f"R0228 R0227 c={clusters} build receipt is not a fitted, complete "
                "graph"
            )
        out[clusters] = {
            "setting_id": setting_id,
            "ids_signature": expected_input_signature(ids_path),
            "receipt_signature": expected_input_signature(receipt_path),
            "ladder_signature": ladder_signature,
            "published_tie_aware_recall": float(
                R0227_TIE_AWARE_RECALL_BY_C[clusters]
            ),
            "provenance": {
                "source_round": "0227",
                "setting_id": setting_id,
                "reason": (
                    "cuVS nn-descent and the k-means seeding are not bit-"
                    "reproducible across runs, so rebuilding would break the link "
                    "to the recall level review-0227-01 verified. These are the "
                    "bytes that number describes."
                ),
            },
        }
    return out


def _sealed_downstream() -> dict[str, dict[str, Any]]:
    """Open every artifact a late node will read, and check every key it uses."""
    panel_signature = expected_input_signature(R0218_PANEL_EVIDENCE)
    panel = prompt_contract.read_sealed(
        R0218_PANEL_EVIDENCE, label="R0218 sealed panel"
    )
    gate_signature = expected_input_signature(R0222_GATE_ARTIFACT)
    gate = prompt_contract.read_sealed(
        R0222_GATE_ARTIFACT, label="R0222 sealed n=8 gate"
    )
    tolerance_signature = expected_input_signature(R0225_GATE_ARTIFACT)
    tolerance = prompt_contract.read_sealed(
        R0225_GATE_ARTIFACT, label="R0225 sealed tolerance gates"
    )
    cuvs_signature = expected_input_signature(R0223_COMPARISON_ARTIFACT)
    cuvs = prompt_contract.read_sealed(
        R0223_COMPARISON_ARTIFACT, label="R0223 sealed cuVS comparison"
    )
    if (
        gate.get("schema") != R0222_GATE_SCHEMA
        or gate.get("round_id") != "0222"
        or gate.get("gate_registered") is not True
        or {int(seed) for seed in gate["pooled_panel_metric_cells"]}
        != set(EXACT_FAMILY_SEEDS)
    ):
        raise RuntimeError("R0228 sealed R0222 gate contract changed")
    if (
        tolerance.get("schema") != R0225_GATE_SCHEMA
        or tolerance.get("round_id") != "0225"
        or tolerance.get("gate_registered") is not True
    ):
        raise RuntimeError("R0228 sealed R0225 tolerance gate contract changed")
    gates = dict(tolerance["gate"]["gates"])
    for metric in GATED_METRICS:
        entry = gates[metric]
        float(entry["one_sided_tolerance_95_95"]["floor"])
        float(entry["mean_minus_2sd"]["floor"])
    for metric in ("purity_fidelity_k256", "purity_fidelity_k1024"):
        band = gates[metric]["two_sided_log_ratio_95_95"]
        for key in (
            "k2",
            "log_lower",
            "log_upper",
            "log_ratio_mean",
            "log_ratio_sample_sd_ddof1",
            "ratio_lower",
            "ratio_upper",
        ):
            float(band[key])
    if (
        cuvs.get("schema") != R0223_COMPARISON_SCHEMA
        or cuvs.get("round_id") != "0223"
        or [int(value) for value in cuvs.get("seeds") or []] != list(R0223_CUVS_SEEDS)
        or {int(seed) for seed in cuvs["cuvs_panel_metric_cells"]}
        != set(R0223_CUVS_SEEDS)
    ):
        raise RuntimeError("R0228 sealed R0223 comparison contract changed")
    # Every coordinate array the geometry node will open must exist and bind now,
    # and every raw purity ratio the comparison node will read must be locatable.
    coordinate_signatures: dict[str, Any] = {}
    for seed in EXACT_FAMILY_SEEDS:
        if str(seed) in panel["cells"]:
            signature = dict(panel["cells"][str(seed)]["coordinates"])
            purity = panel["cells"][str(seed)]["panel"]["purity"]
        else:
            signature = dict(gate["new_cells"][str(seed)]["coordinates"])
            purity = gate["new_cells"][str(seed)]["panel"]["purity"]
        for granularity in ("k256", "k1024"):
            if float(purity[granularity]) <= 0.0:
                raise RuntimeError(
                    f"R0228 exact cell {seed} has a nonpositive {granularity} "
                    "purity ratio; the unfolded log scale is undefined"
                )
        prompt_contract.verify_signature(
            signature, label=f"exact-graph seed {seed} coordinates"
        )
        coordinate_signatures[f"exact-seed{seed}"] = signature
    for seed in R0223_CUVS_SEEDS:
        signature = dict(cuvs["cells"][str(seed)]["coordinates"])
        prompt_contract.verify_signature(
            signature, label=f"R0223 cuVS seed {seed} coordinates"
        )
        coordinate_signatures[f"r0223-cuvs-seed{seed}"] = signature
    return {
        "panel_signature": panel_signature,
        "gate_signature": gate_signature,
        "tolerance_signature": tolerance_signature,
        "cuvs_signature": cuvs_signature,
        "coordinate_signatures": coordinate_signatures,
    }


def _construction_proof(
    *,
    substrate_signature: dict[str, Any],
    r0216_graph_signature: dict[str, Any],
    r0216_manifest_signature: dict[str, Any],
) -> dict[str, Any]:
    """Prove, before any GPU work, that only the registered paths move."""
    probe_graph_signature = {
        "kind": "file",
        "canonical_path": "/data/latent-basemap/runs/round-0228/<pending>/edges-k15-fuzzy.npz",
        "bytes": 1,
        "sha256": "0" * 64,
    }
    probe_manifest_signature = {
        "kind": "file",
        "canonical_path": "/data/latent-basemap/runs/round-0228/<pending>/cluster-spill-graph.json",
        "bytes": 1,
        "sha256": "1" * 64,
    }
    invariants = set()
    capabilities = set()
    for clusters, seed in CELLS:
        config, _config_sha, invariant = train_config(
            clusters=clusters,
            seed=seed,
            graph_signature=probe_graph_signature,
            graph_manifest_signature=probe_manifest_signature,
            substrate_signature=substrate_signature,
            r0216_graph_signature=r0216_graph_signature,
            r0216_graph_manifest_signature=r0216_manifest_signature,
            graph_edges=CONSTRUCTION_PROBE_EDGES,
            rows=ROWS,
        )
        invariants.add(invariant)
        capabilities.add(str(config["capability"]))
    if invariants != {R0217_TREATMENT_INVARIANT_SHA256}:
        raise RuntimeError(
            "R0228 construction does not reproduce the cross-round treatment "
            f"digest: {sorted(invariants)} != {R0217_TREATMENT_INVARIANT_SHA256}"
        )
    if len(capabilities) != len(CELLS):
        raise RuntimeError("R0228 cells do not carry distinct capabilities")
    return {
        "schema": "round0228-treatment-construction-proof-v1",
        "round_id": ROUND_ID,
        "template_round": "0217",
        "template_seed": TEMPLATE_SEED,
        "treatment_invariant_sha256": R0217_TREATMENT_INVARIANT_SHA256,
        "cross_round_digest_note": (
            "R0217, R0221 and R0223 all carry this digest under the registered "
            "mask, and review-0223-01 reproduced it with its own independent "
            "masking implementation on all three"
        ),
        "cells": [{"clusters": c, "seed": s} for c, s in CELLS],
        "capabilities": sorted(capabilities),
        "construction_probe_edges": CONSTRUCTION_PROBE_EDGES,
        "construction_probe_note": (
            "a hypothetical edge count used only to prove the construction; the "
            "real horizon is derived inside each node from the sealed graph"
        ),
    }


#: The queue whose artifacts a `--geometry-only` correction binds. Its fourteen
#: completed nodes are sealed on disk; the geometry probe is the only one that
#: failed, and retraining nine cells to reproduce bytes that already exist would
#: burn ~1.5 GPU-h for nothing.
SOURCE_QUEUE_ROOT = QUEUE_ROOT
CORRECTION_QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue-correction-1")


def _sealed_source_artifacts() -> dict[str, Any]:
    """Bind the already-sealed R0228 comparison and graph receipts by hash."""
    from basemap.round0228_low_c_map import BUILD_SCHEMA, COMPARISON_SCHEMA

    artifacts = os.path.join(SOURCE_QUEUE_ROOT, "artifacts")
    comparison_path = os.path.join(
        artifacts,
        COMPARISON_CAPABILITY,
        "cluster-spill-graph-map-comparison.json",
    )
    if not os.path.exists(comparison_path):
        raise RuntimeError(
            "R0228 geometry correction needs the sealed comparison artifact from "
            f"the source queue; it is absent at {comparison_path}"
        )
    comparison = prompt_contract.read_sealed(
        comparison_path, label="R0228 sealed map comparison"
    )
    if (
        comparison.get("schema") != COMPARISON_SCHEMA
        or comparison.get("round_id") != ROUND_ID
        or len(comparison.get("cells") or {}) != len(CELLS)
    ):
        raise RuntimeError("R0228 sealed comparison contract changed")
    # Every coordinate array the geometry node will open, bound now.
    for key, cell in comparison["cells"].items():
        prompt_contract.verify_signature(
            dict(cell["coordinates"]), label=f"R0228 {key} coordinates"
        )
    graph_signatures: dict[str, Any] = {}
    for clusters in CLUSTER_COUNTS:
        path = os.path.join(
            artifacts, graph_capability(clusters), "cluster-spill-graph.json"
        )
        manifest = prompt_contract.read_sealed(
            path, label=f"R0228 sealed c={clusters} graph receipt"
        )
        if (
            manifest.get("schema") != BUILD_SCHEMA
            or int(manifest.get("clusters", -1)) != clusters
        ):
            raise RuntimeError(f"R0228 sealed c={clusters} graph contract changed")
        prompt_contract.verify_signature(
            dict(manifest["loss_arrays"]["lost_edges_per_row"]),
            label=f"R0228 c={clusters} lost-edge counts",
        )
        graph_signatures[str(clusters)] = expected_input_signature(path)
    return {
        "comparison_signature": expected_input_signature(comparison_path),
        "graph_signatures": graph_signatures,
    }


def prepare_round0228(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    geometry_only: bool = False,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0228 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    r0216_manifest_signature, r0216_manifest = _sealed_r0216()
    substrate_signature = dict(r0216_manifest["substrate"])
    provenance_signature = dict(r0216_manifest["provenance"])
    r0216_graph_signature = dict(r0216_manifest["graph"])
    truth_signature = _sealed_truth()
    r0227_builds = _sealed_r0227_builds()
    downstream = _sealed_downstream()
    proof = _construction_proof(
        substrate_signature=substrate_signature,
        r0216_graph_signature=r0216_graph_signature,
        r0216_manifest_signature=r0216_manifest_signature,
    )
    guards = {
        str(clusters): guard_decision(rows=ROWS, clusters=clusters)
        for clusters in CLUSTERS_BUILT_HERE
    }
    if not all(bool(value.get("allowed")) for value in guards.values()):
        raise RuntimeError(f"R0228 guard refuses a registered build cell: {guards}")
    source = _sealed_source_artifacts() if geometry_only else None

    ensure_data_directory(ROUND_ROOT)
    ensure_data_directory(CUVS_CACHE_ROOT)
    ensure_data_directory(SCRATCH_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0228 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    proof_path = os.path.join(preflight, "treatment-construction-proof.json")
    atomic_write_new_json(proof_path, prompt_contract.seal(proof), immutable=True)

    expected_inputs = _dedupe([
        round_signature,
        r0216_manifest_signature,
        substrate_signature,
        provenance_signature,
        r0216_graph_signature,
        truth_signature,
        downstream["panel_signature"],
        downstream["gate_signature"],
        downstream["tolerance_signature"],
        downstream["cuvs_signature"],
        *(entry["ids_signature"] for entry in r0227_builds.values()),
        *(entry["receipt_signature"] for entry in r0227_builds.values()),
        *(entry["ladder_signature"] for entry in r0227_builds.values()),
        *downstream["coordinate_signatures"].values(),
        expected_input_signature(smoke_path),
        expected_input_signature(proof_path),
        *(
            [source["comparison_signature"], *source["graph_signatures"].values()]
            if geometry_only
            else []
        ),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    jobs: list[dict[str, Any]] = []
    p90: dict[str, float] = {}

    build_nodes: dict[int, str] = {}
    for clusters in [] if geometry_only else CLUSTERS_BUILT_HERE:
        node = f"build_cluster_spill_c{clusters}_2m"
        build_nodes[clusters] = node
        jobs.append({
            "id": node,
            "action": CLUSTER_BUILD_ACTION,
            "handler_module": "experiments.round0228_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [os.path.join(artifacts, f"cluster-spill-build-c{clusters}")],
            "done_marker": os.path.join(artifacts, f"{node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": CLUSTER_BUILD_P90_WALL_S,
            "clusters": int(clusters),
            "substrate_signature": substrate_signature,
            "cuvs_cache_root": CUVS_CACHE_ROOT,
            "scratch_root": SCRATCH_ROOT,
            "guard": guards[str(clusters)],
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
                "cpu_heavy": True,
            },
        })
        p90[node] = CLUSTER_BUILD_P90_WALL_S

    fuzzy_nodes: dict[int, str] = {}
    graph_manifest_paths: dict[int, str] = {}
    for clusters in [] if geometry_only else CLUSTER_COUNTS:
        node = f"fuzzy_graph_c{clusters}"
        fuzzy_nodes[clusters] = node
        output = os.path.join(artifacts, graph_capability(clusters))
        graph_manifest_paths[clusters] = os.path.join(
            output, "cluster-spill-graph.json"
        )
        if clusters in CLUSTERS_BUILT_HERE:
            build_dir = os.path.join(
                artifacts,
                f"cluster-spill-build-c{clusters}",
                f"low-c-n{ROWS}-c{clusters}",
            )
            neighbour_reference = {
                "kind": "file",
                "canonical_path": os.path.join(build_dir, "graph-k15-ids.i32.npy"),
            }
            source_receipt = {
                "kind": "file",
                "canonical_path": os.path.join(build_dir, "build-receipt.json"),
            }
            provenance = {
                "source_round": ROUND_ID,
                "setting_id": f"low-c-n{ROWS}-c{clusters}",
                "reason": (
                    "R0227's 2M ladder cells were c = 64/32/16/8; c = 4 is the "
                    "configuration its own per-rung table selects and it was "
                    "never built at 2M, so this round builds it with R0227's "
                    "unmodified script"
                ),
            }
            deps = [build_nodes[clusters]]
        else:
            neighbour_reference = r0227_builds[clusters]["ids_signature"]
            source_receipt = r0227_builds[clusters]["receipt_signature"]
            provenance = r0227_builds[clusters]["provenance"]
            deps = []
        jobs.append({
            "id": node,
            "action": FUZZY_ACTION,
            "handler_module": "experiments.round0228_nodes",
            "handler_callable": "run_job",
            "deps": deps,
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"{node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": FUZZY_NODE_P90_WALL_S,
            "clusters": int(clusters),
            "capability": graph_capability(clusters),
            "graph_manifest_signature": r0216_manifest_signature,
            "truth_receipt_signature": truth_signature,
            "neighbour_ids_reference": neighbour_reference,
            "neighbour_ids_provenance": provenance,
            "source_build_receipt": source_receipt,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
                "cpu_heavy": False,
            },
        })
        p90[node] = FUZZY_NODE_P90_WALL_S

    train_nodes: list[str] = []
    for clusters, seed in [] if geometry_only else CELLS:
        capability = map_capability(clusters, seed)
        node = f"train_c{clusters}_seed{seed}"
        train_nodes.append(node)
        jobs.append({
            "id": node,
            "action": TRAIN_ACTION,
            "handler_module": "experiments.round0228_nodes",
            "handler_callable": "run_job",
            "deps": [fuzzy_nodes[clusters]],
            "outputs": [os.path.join(artifacts, capability)],
            "done_marker": os.path.join(artifacts, f"{node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": TRAIN_NODE_P90_WALL_S,
            "clusters": int(clusters),
            "training_seed": int(seed),
            "capability": capability,
            "graph_manifest_signature": {
                "kind": "file",
                "canonical_path": graph_manifest_paths[clusters],
            },
            "r0216_graph_signature": r0216_graph_signature,
            "r0216_graph_manifest_signature": r0216_manifest_signature,
            "treatment_invariant_sha256": R0217_TREATMENT_INVARIANT_SHA256,
            "registered_dose_bound": REGISTERED_UPDATE_BOUND,
            "node_policy": {
                "gpu_required": True,
                "training_performed": True,
                "cpu_heavy": False,
            },
        })
        p90[node] = TRAIN_NODE_P90_WALL_S

    compare_node = "compare_cluster_spill_panel"
    compare_output = os.path.join(artifacts, COMPARISON_CAPABILITY)
    compare_job = {
        "id": compare_node,
        "action": COMPARE_ACTION,
        "handler_module": "experiments.round0228_nodes",
        "handler_callable": "run_job",
        "deps": list(train_nodes),
        "outputs": [compare_output],
        "done_marker": os.path.join(artifacts, f"{compare_node}.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": COMPARE_NODE_P90_WALL_S,
        "capability": COMPARISON_CAPABILITY,
        "graph_manifest_signature": r0216_manifest_signature,
        "panel_evidence": R0218_PANEL_EVIDENCE,
        "r0222_gate_signature": downstream["gate_signature"],
        "r0225_gate_signature": downstream["tolerance_signature"],
        "r0223_comparison_signature": downstream["cuvs_signature"],
        "cells": [
            {
                "clusters": int(clusters),
                "seed": int(seed),
                "capability": map_capability(clusters, seed),
                "train_receipt": {
                    "kind": "file",
                    "canonical_path": os.path.join(
                        artifacts, map_capability(clusters, seed), "train-receipt.json"
                    ),
                },
            }
            for clusters, seed in CELLS
        ],
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
            "cpu_heavy": False,
        },
    }
    if not geometry_only:
        jobs.append(compare_job)
        p90[compare_node] = COMPARE_NODE_P90_WALL_S

    # A geometry-only correction binds the ALREADY-SEALED comparison and graph
    # receipts by their real hashes instead of by an intra-queue path, and has no
    # dependency to wait on. Nothing is retrained: the nine map artifacts, their
    # coordinates and the panel comparison are sealed on disk and are re-verified
    # here rather than reproduced.
    geometry_node = "probe_cluster_spill_geometry"
    if geometry_only:
        comparison_reference = dict(source["comparison_signature"])
        graph_manifest_references = {
            key: dict(value) for key, value in source["graph_signatures"].items()
        }
        geometry_deps: list[str] = []
    else:
        comparison_reference = {
            "kind": "file",
            "canonical_path": os.path.join(
                compare_output, "cluster-spill-graph-map-comparison.json"
            ),
        }
        graph_manifest_references = {
            str(clusters): {
                "kind": "file",
                "canonical_path": graph_manifest_paths[clusters],
            }
            for clusters in CLUSTER_COUNTS
        }
        geometry_deps = [compare_node]
    jobs.append({
        "id": geometry_node,
        "action": GEOMETRY_ACTION,
        "handler_module": "experiments.round0228_nodes",
        "handler_callable": "run_job",
        "deps": geometry_deps,
        "outputs": [os.path.join(artifacts, GEOMETRY_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{geometry_node}.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": GEOMETRY_NODE_P90_WALL_S,
        "capability": GEOMETRY_CAPABILITY,
        "graph_manifest_signature": r0216_manifest_signature,
        "truth_receipt_signature": truth_signature,
        "panel_evidence": R0218_PANEL_EVIDENCE,
        "r0222_gate_signature": downstream["gate_signature"],
        "r0223_comparison_signature": downstream["cuvs_signature"],
        "comparison_signature": comparison_reference,
        "graph_manifests": graph_manifest_references,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
            "cpu_heavy": True,
        },
    })
    p90[geometry_node] = GEOMETRY_NODE_P90_WALL_S
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
        "schema": "round0228-minilm-mixed-2m-cluster-spill-map-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            R0216_GRAPH_CAPABILITY,
            R0218_PANEL_CAPABILITY,
            "minilm-mixed-2m-quality-gates-n8-v1",
            "minilm-mixed-2m-tolerance-gates-n8-v1",
            "minilm-mixed-2m-cuvs-graph-map-comparison-v1",
            R0227_LADDER_CAPABILITY,
        ],
        "capabilities_produced": (
            [GEOMETRY_CAPABILITY]
            if geometry_only
            else [
                *GRAPH_CAPABILITIES,
                *MAP_CAPABILITIES,
                COMPARISON_CAPABILITY,
                GEOMETRY_CAPABILITY,
            ]
        ),
        "training_performed": not geometry_only,
        "jobs": jobs,
        "p90_gpu_seconds": p90,
        "scientific_contract": {
            "question": (
                "does a map trained on a cluster-spill-nnd graph differ from a "
                "map trained on exact truth, on the 2M substrate where exact "
                "truth exists?"
            ),
            "why_now": (
                "review-0227-01 established that recall is no longer a sufficient "
                "proxy: the residual loss is 99.6-99.9% mutual, regionally "
                "clustered and NOT the partition cut, so it cannot be bought back "
                "by lowering c, and R0215 showed regionally clustered edge "
                "absence is what produced the v1 150M clumps. Only training can "
                "answer it."
            ),
            "population": "sealed R0216 queue-correction-3 mixed MiniLM 2M substrate",
            "recall_population": RECALL_POPULATION,
            "recall_population_note": RECALL_POPULATION_NOTE,
            "graphs": {
                str(clusters): (
                    f"{CLUSTER_SPILL_BUILDER} at c={clusters}, symmetrised "
                    "through R0216's identical fuzzy law"
                )
                for clusters in CLUSTER_COUNTS
            },
            "clusters_built_here": list(CLUSTERS_BUILT_HERE),
            "clusters_reused_from_r0227": list(CLUSTERS_FROM_R0227),
            "lower_c_bracket_note": (
                "c = 4 is the structural floor of this builder (at spill s = 2, "
                "c = 2 partitions nothing), so the mandate's lower-c bracket does "
                "not exist inside cluster-spill-nnd. The no-partition limit is a "
                "monolithic build, and that arm is R0223's three cuVS cells at "
                "tie-aware 0.994164, which this round compares against."
            ),
            "control": (
                "R0222's sealed eight-cell exact-graph family (seeds 42-49) on "
                "the byte-identical R0218 high-D reference, plus R0223's three "
                "cuVS cells as the monolithic-graph arm"
            ),
            "only_treatment_vs_control": "the k-NN graph",
            "hidden_dimension": HIDDEN_DIMENSION,
            "input_dimension": DIMENSION,
            "precision": USE_AMP,
            "cluster_counts": list(CLUSTER_COUNTS),
            "seeds": list(SEEDS),
            "cells": len(CELLS),
            "exact_family_seeds": list(EXACT_FAMILY_SEEDS),
            "treatment_source_round": "0217",
            "treatment_template_seed": TEMPLATE_SEED,
            "treatment_invariant_sha256": R0217_TREATMENT_INVARIANT_SHA256,
            "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
            "dose_rule": (
                "ceil(1e6 * active_directed_edges / 603,086,368), derived in-node "
                "from each sealed graph's own edge count; a different edge count "
                "legitimately yields a different horizon and review-0223-01 "
                "verified that this is ceil quantisation, not a deviation"
            ),
            "registered_update_bound": REGISTERED_UPDATE_BOUND,
            "host_rss_limit_gib": HOST_RSS_LIMIT_GIB,
            "gated_metrics": list(GATED_METRICS),
            "density_v2_status": DENSITY_V2_STATUS,
            "identity_bound_note": IDENTITY_BOUND_NOTE,
            "geometry_battery": (
                "R0215's clump detector on all twenty coordinate arrays, plus a "
                "density-matched true-neighbour displacement statistic for the "
                f"rows that actually lost edges ({SCATTER_SAMPLE_ROWS} sampled "
                "per population), with the eight exact-graph maps as the null arm"
            ),
            "statistical_tests": (
                "exact permutation per configuration against the eight-cell exact "
                "family (C(11,3) = 165 relabellings), pooled (C(17,9) = 24,310), "
                "and an exact trend test in log2(c) over the nine candidate cells "
                "alone (1,680 label assignments)"
            ),
            "gate_registerable_here": GATE_REGISTERABLE_HERE,
            "gate_release_claimed": GATE_RELEASE_CLAIMED,
            "adoption_claimed": ADOPTION_CLAIMED,
            "equivalence_claimed": EQUIVALENCE_CLAIMED,
            "evidence_limits": EVIDENCE_LIMITS,
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
    parser.add_argument("--queue-root", default=None)
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help=(
            "rebuild ONLY the geometry probe, bound to the source queue's "
            "already-sealed comparison and graph receipts; retrains nothing"
        ),
    )
    args = parser.parse_args(argv)
    queue_root = args.queue_root or (
        CORRECTION_QUEUE_ROOT if args.geometry_only else QUEUE_ROOT
    )
    print(json.dumps({
        "queue_manifest": prepare_round0228(
            release_sha=args.release_sha,
            queue_root=queue_root,
            geometry_only=args.geometry_only,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
