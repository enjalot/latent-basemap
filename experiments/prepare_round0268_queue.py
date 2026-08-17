#!/usr/bin/env python3
"""Prepare, but never launch, the R0268 queue — the 100M ×2 host-int8 FLAGSHIP.

Five nodes in one queue, in this order:

1-3. `train_minilm_fneg_100m_x2_hostint8` × 3 (GPU, seeds 42/43/44, SEQUENTIAL) — the
     promoted fneg recipe at N=100M, dose ×2, on the host-int8 X path R0266/R0267 validated,
     built from R0265's template retargeted to the sealed R0238 100M substrate + R0243 k15
     graph and proved by `round0268_int8_treatment.assert_registered_100m_int8_recipe`. X is
     R0262's PRE-SEALED 100M int8 substrate loaded WHOLE (full-file digest bind). A FRESH
     train: all three seeds train from scratch (no salvage, no bind).
4.   `score_minilm_fneg_100m_x2_panel` (GPU) — the three maps scored on R0265's instruments:
     held-out FFR via the out-of-substrate reserve-projection instrument (disc=int(ROWS·
     0.001)=100000), collapse, fog (all on the FULL 100M coordinates), plus DESCRIPTIVE-only
     purity on the R0238 first-2M prefix + the lineage check (100M-prefix != R0216-c3).
5.   `register_fneg_100m_x2_seedmean_gate` (CPU) — the pre-registered 100M gate: the
     SEED-MEAN collapse inside P1's ×2 asymptote band (SAME as 50M) widened by z·σ_fam/√n,
     plus per-seed backstops on collapse/fog/FFR. Every band/floor/σ_fam/P1-edge is bound by
     sha256 and read/recomputed at gate time.

This builder REFUSES until the round file is issued (status: issued, base_commit an ancestor
of the release) and every bound input exists — including the sealed R0238 100M substrate +
R0243 graph manifests, the R0238 held-out reserve + reserve-query-rows, the sealed 100M
reserve-neighbour truth (BUILD IT FIRST with `build_round0268_reserve_truth.py`), R0262's
sealed 100M int8 substrate + its identity manifest, R0218's frozen panel (for the R0216-c3
lineage reference + descriptive centroid granularities), R0265's sealed family floors + n=13
panel, and the sealed P1 analysis-v2 result.
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
from basemap import round0268_int8_treatment as T
from basemap.round0268_int8_treatment import (
    CLOSURE_SCHEMA,
    ROUND_ID,
    ROWS,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    TRAIN_CLOSURE_MODULES,
    assert_registered_100m_int8_recipe,
    capability_for_seed,
    fneg_seed_invariant_sha256,
    int8_100m_train_config,
    runtime_closure_hashes,
)
from basemap.round0217_minilm_2m_seed_family import successful_updates_for_edges
from experiments.round0268_nodes import (
    GATE_ACTION,
    GATE_CAPABILITY,
    PANEL_ACTION,
    PANEL_CAPABILITY,
    PANEL_SCHEMA,
    TRAIN_ACTION,
    TRAIN_SCHEMA,
)
import experiments.round0265_nodes as R0265N
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
# Reuse R0265/R0266's data anchors for the 2M frozen panel + the R0265 sealed instruments.
from experiments.prepare_round0265_queue import R0218_PANEL
from experiments.prepare_round0266_queue import (
    R0265_FLOORS,
    R0265_PANEL,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0268"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
#: NOTE: the round file must be ISSUED before this builder will run. Flag for the owner.
ROUND_FILE = os.path.join(LAB_ROOT, "round-0268-2026-08-17.md")

#: The sealed R0238 100M substrate manifest (the nested-prefix ladder's carve).
R0238_SUBSTRATE_MANIFEST = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-nested-substrate-and-reserves-v1/substrate.json"
)
#: The sealed R0243 100M k15 fuzzy graph manifest (streamed-member layout).
R0243_GRAPH_MANIFEST = (
    "/data/latent-basemap/runs/round-0243/queue/artifacts/"
    "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1/fuzzy-graph.json"
)
#: The sealed R0238 200k held-out reserve embeddings + the 2000 held-out probe rows INTO it.
#: The reserve-projection FFR instrument projects reserve.f32[reserve-query-rows] through
#: each map; these are the SAME rows the sealed reserve-neighbour truth was built for.
R0238_RESERVE = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-nested-substrate-and-reserves-v1/reserve.f32.npy"
)
R0238_RESERVE_QUERY_ROWS = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-nested-substrate-and-reserves-v1/reserve-query-rows.i64.npy"
)
#: The sealed R0268 reserve-neighbour truth: the exact-cosine top-10 100M substrate
#: neighbours of reserve.f32[reserve-query-rows] (indices INTO the 100M substrate). BUILD
#: THIS FIRST with build_round0268_reserve_truth.py; this builder refuses until it exists.
R0268_RESERVE_NEIGHBOUR_TRUTH = (
    "/data/latent-basemap/runs/round-0268/ffr/reserve-truth-100m/truth-top10.npy"
)
#: The frozen P1 analysis-v2 result — the ×2 collapse asymptote band (plain JSON).
P1_ASYMPTOTE = "/data/latent-basemap/sandbox/logs/analysis_v2_result.json"


#: Three 100M ×2 host-int8 trains (~24 GPU-h each) + a three-map 100M panel + the CPU gate;
#: the cap sits above the ~72 GPU-h train estimate + panel/gate (plan §4 costs).
GPU_HOURS_CAP = 80.0
TRAIN_P90_WALL_S = 86_400.0   # 24 h/seed (measured host-int8 0.0102 s/update × 8.33M ≈ 23.6 h)
PANEL_P90_WALL_S = 18_000.0   # 3 × full-100M transform + scoring
GATE_P90_WALL_S = 600.0


def _issued_round(release_sha: str) -> tuple[dict[str, Any], list[str]]:
    if not os.path.exists(ROUND_FILE):
        raise RuntimeError(
            f"R0268 round file absent: {ROUND_FILE}. EXPECTED until the owner ISSUES the "
            "round-0268 round file (status: issued, base_commit an ancestor of the release)."
        )
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
            "R0268 round is not issued for this release. EXPECTED until the round file is "
            "issued and its base_commit is an ancestor of the release."
        )
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0268 round must declare its required reviews")
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
            "Review is post-hoc: it blocks the downstream claim, not the launch. The 100M "
            "PASS/FAIL this round registers is registered-and-contingent until its required "
            "upstream reviews are accepted."
        ),
    }


def _signature(path: str, label: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise RuntimeError(f"R0268 bound input absent: {label} at {path}")
    return expected_input_signature(path)


def _treatment_closure_seal(release_sha: str) -> dict[str, Any]:
    """SHA-256 of every R0268 training-closure source (R0266's closure + the 100M module)."""
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
            "R0268 is R0265's fneg recipe at 100M ×2 on R0266's host-int8 path. This seals "
            "the R0268 training import closure: R0266's closure (the fneg-merged core + the "
            "int8 routing module) PLUS the round0268 100M ×2 recipe module. A node that ran "
            "different bytes for any of them would refuse."
        ),
    })


def _sealed_graph_binding() -> dict[str, Any]:
    """Read + validate the sealed R0243 fuzzy-graph manifest; return the bindings prepare
    needs: the manifest signature, the four streamed-member signatures, a graph provenance
    signature (edges dir + header sha) for the config, and the directed edge count."""
    graph_manifest_signature = _signature(R0243_GRAPH_MANIFEST, "R0243 100M graph manifest")
    graph_manifest = prompt_contract.read_sealed(
        graph_manifest_signature["canonical_path"], label="R0243 100M graph manifest"
    )
    capabilities = graph_manifest.get("capabilities") or []
    tripwire = graph_manifest.get("post_canonical_tripwire") or {}
    sym = graph_manifest.get("symmetrised_degree") or {}
    if (
        str(graph_manifest.get("round_id")) != T.R0243_ROUND_ID
        or T.R0243_GRAPH_CAPABILITY not in capabilities
        or int(graph_manifest.get("rows", -1)) != ROWS
        or int(graph_manifest.get("k", -1)) != 15
        or int(graph_manifest.get("directed_edges", -1)) != SEALED_DIRECTED_EDGES
    ):
        raise RuntimeError("R0268 --graph manifest is not the sealed R0243 100M k15 graph")
    if int(tripwire.get("zero_degree_rows", -1)) != 0 or int(sym.get("zero_degree_rows", -1)) != 0:
        raise RuntimeError("R0268 --graph manifest lacks the R0243 zero-degree tripwire (== 0)")
    outputs = graph_manifest.get("outputs") or {}
    member_signatures = {}
    for name in ("edges_header", "edges_sources", "edges_targets", "edges_weights"):
        sig = dict(outputs[name])
        if not os.path.exists(sig.get("canonical_path", "")):
            raise RuntimeError(f"R0268 sealed R0243 graph member absent: {name}")
        member_signatures[name] = sig
    header_path = str(member_signatures["edges_header"]["canonical_path"])
    edges_dir = os.path.dirname(header_path)
    graph_signature = {
        "kind": "file",
        "canonical_path": edges_dir,
        "sha256": str(member_signatures["edges_header"]["sha256"]),
    }
    return {
        "manifest_signature": graph_manifest_signature,
        "member_signatures": member_signatures,
        "graph_signature": graph_signature,
        "edges_dir": edges_dir,
        "directed_edges": int(graph_manifest["directed_edges"]),
    }


def prepare_round0268(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0268 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    review_state = _upstream_review_state(list(required_reviews))

    # Bind the sealed R0238 100M substrate manifest, read its inner signatures.
    substrate_manifest_signature = _signature(R0238_SUBSTRATE_MANIFEST, "R0238 100M substrate manifest")
    substrate_manifest = prompt_contract.read_sealed(
        substrate_manifest_signature["canonical_path"], label="R0238 100M substrate manifest"
    )
    if (
        str(substrate_manifest.get("round_id")) != T.R0238_ROUND_ID
        or substrate_manifest.get("capability") != T.R0238_SUBSTRATE_CAPABILITY
        or int(substrate_manifest.get("rows", -1)) != ROWS
        or str(substrate_manifest.get("ordered_substrate_sha256")) != T.R0238_SUBSTRATE_ORDERED_SHA256
    ):
        raise RuntimeError("R0268 --substrate manifest is not the sealed R0238 100M substrate")
    substrate_signature = dict(substrate_manifest["substrate"])

    # Bind the sealed R0243 100M graph (streamed-member layout).
    graph = _sealed_graph_binding()
    graph_manifest_signature = graph["manifest_signature"]
    graph_signature = graph["graph_signature"]
    edges = graph["directed_edges"]
    base_horizon = successful_updates_for_edges(edges)

    # Build and prove the three ×2 cell configs HERE (needs no train artifacts). All three
    # share ONE masked seed-invariant digest (the recipe outside the seed).
    invariants: set[str] = set()
    per_seed_config_sha: dict[str, str] = {}
    recipe = None
    for seed in SEEDS:
        config, config_sha = int8_100m_train_config(
            seed=seed,
            graph_signature=graph_signature,
            graph_manifest_signature=graph_manifest_signature,
            substrate_signature=substrate_signature,
            graph_edges=edges,
            rows=ROWS,
        )
        recipe = assert_registered_100m_int8_recipe(config)
        invariants.add(fneg_seed_invariant_sha256(config))
        per_seed_config_sha[str(seed)] = config_sha
    if len(invariants) != 1:
        raise RuntimeError("R0268 three cells do not share one masked recipe digest")
    cell_seed_invariant = sorted(invariants)[0]

    # Bind every sealed cross-round input by real sha256 + bytes.
    r0218_panel = _signature(R0218_PANEL, "R0218 frozen panel (R0216-c3 lineage reference)")
    reserve = _signature(R0238_RESERVE, "R0238 100M held-out reserve")
    reserve_query_rows = _signature(R0238_RESERVE_QUERY_ROWS, "R0238 reserve query rows")
    reserve_truth = _signature(R0268_RESERVE_NEIGHBOUR_TRUTH, "R0268 100M reserve-neighbour truth")
    r0265_floors = _signature(R0265_FLOORS, "R0265 sealed family floors")
    r0265_panel = _signature(R0265_PANEL, "R0265 sealed n=13 panel")
    p1_asymptote = _signature(P1_ASYMPTOTE, "P1 analysis-v2 ×2 asymptote band")

    # Cross-check the bound sealed instruments at prepare (not only at gate time).
    floors_sealed = prompt_contract.read_sealed(r0265_floors["canonical_path"], label="R0265 floors")
    if floors_sealed.get("capability") != R0265N.GATE_CAPABILITY or floors_sealed.get("gate_registered") is not True:
        raise RuntimeError("R0268 --r0265-floors is not the sealed R0265 family floors gate")
    panel_sealed = prompt_contract.read_sealed(r0265_panel["canonical_path"], label="R0265 panel")
    if panel_sealed.get("capability") != R0265N.PANEL_CAPABILITY or int(panel_sealed.get("n", -1)) != R0265N.N_FAMILY:
        raise RuntimeError("R0268 --r0265-panel is not the sealed n=13 panel")
    with open(p1_asymptote["canonical_path"], encoding="utf-8") as handle:
        p1_json = json.load(handle)
    if "yinf_x2" not in dict(p1_json.get("bands") or {}) or p1_json.get("verdict") != "GO":
        raise RuntimeError("R0268 --p1-asymptote is not the frozen GO analysis-v2 result")
    # The R0218 panel must carry the R0216-c3 2M ordered reference for the lineage check.
    r0218_sealed = prompt_contract.read_sealed(r0218_panel["canonical_path"], label="R0218 panel")
    r0216_c3_reference = str((r0218_sealed.get("lineage") or {}).get("ordered_substrate_sha256") or "")
    if len(r0216_c3_reference) != 64:
        raise RuntimeError("R0268 --r0218-panel carries no R0216-c3 2M ordered reference (lineage)")

    census = dispatch_census()
    guard = assert_derived_entries_install(SCOPE_MODULES, census)
    gates = gate_census(entry_tuples(guard["derived"]))
    residual = scope_residual(census, SCOPE_MODULES)

    ensure_data_directory(ROUND_ROOT, label="R0268 round root")
    queue_root = create_fresh_directory(queue_root, label="R0268 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    closure_path = os.path.join(preflight, "treatment-source-closure.json")
    atomic_write_new_json(closure_path, _treatment_closure_seal(release_sha), immutable=True)
    identity_path = os.path.join(preflight, "fneg-100m-x2-cell-identity.json")
    atomic_write_new_json(
        identity_path,
        prompt_contract.seal({
            "schema": "round0268-fneg-100m-x2-cell-identity-v1",
            "round_id": ROUND_ID,
            "release_sha": release_sha,
            "sealed_directed_edges": edges,
            "base_horizon": base_horizon,
            "x2_horizon": int(T.DOSE_MULTIPLIER * base_horizon),
            "seeds": list(SEEDS),
            "x_residency": T.X_RESIDENCY,
            "dose_multiplier": T.DOSE_MULTIPLIER,
            "rows": ROWS,
            "recipe": recipe,
            "cell_seed_invariant_sha256": cell_seed_invariant,
            "per_seed_config_sha256": per_seed_config_sha,
            "registry_fingerprint": registry_fingerprint(),
        }),
        immutable=True,
    )
    closure_signature = expected_input_signature(closure_path)

    # The PRE-SEALED int8 substrate (host-int8 fix proven at 50M): R0268 LOADS R0262's sealed
    # 100M int8 substrate WHOLE (no prefix slice). Bind R0262's substrate.i8 +
    # substrate-scales.f16 as a sealed input via a FULL-FILE LAW manifest that pins the
    # parent files (path + size) and — the load-bearing content binding — the whole-file
    # digests. Also bind the R0262 identity manifest. Verify sizes here; the loader re-hashes
    # the whole file (38.4 GB streamed) before the liveness watchdog starts.
    r0262_identity = _signature(T.R0262_IDENTITY_MANIFEST, "R0262 int8 identity manifest")
    r0262_identity_sealed = prompt_contract.read_sealed(
        r0262_identity["canonical_path"], label="R0262 int8 identity manifest"
    )
    if r0262_identity_sealed.get("schema") != T.R0262_IDENTITY_SCHEMA:
        raise RuntimeError("R0268 --r0262-identity is not the sealed R0262 int8 identity manifest")
    quantise = r0262_identity_sealed.get("quantise") or {}
    if (
        int(quantise.get("rows", -1)) != T.R0262_ROWS
        or int(quantise.get("int8_bytes", -1)) != T.R0262_I8_BYTES
        or int(quantise.get("scales_bytes", -1)) != T.R0262_SCALES_BYTES
    ):
        raise RuntimeError("R0268 R0262 int8 identity manifest byte accounting changed")
    for label, parent_path, parent_bytes in (
        ("R0262 100M int8 substrate", T.R0262_I8_PATH, T.R0262_I8_BYTES),
        ("R0262 100M int8 scales", T.R0262_SCALES_PATH, T.R0262_SCALES_BYTES),
    ):
        if not os.path.exists(parent_path):
            raise RuntimeError(f"R0268 bound int8 parent absent: {label} at {parent_path}")
        observed_bytes = int(os.stat(parent_path).st_size)
        if observed_bytes != int(parent_bytes):
            raise RuntimeError(
                f"R0268 bound int8 parent size mismatch: {label} at {parent_path} is "
                f"{observed_bytes} bytes, expected {int(parent_bytes)}"
            )
    int8_substrate_manifest_path = os.path.join(preflight, "int8-full-file-substrate-manifest.json")
    atomic_write_new_json(
        int8_substrate_manifest_path,
        prompt_contract.seal(T.int8_full_substrate_manifest_body(release_sha=release_sha)),
        immutable=True,
    )
    int8_substrate_manifest_signature = expected_input_signature(int8_substrate_manifest_path)

    shared_inputs = _dedupe([
        round_signature,
        graph_manifest_signature,
        substrate_manifest_signature,
        substrate_signature,
        graph_signature,
        expected_input_signature(identity_path),
        int8_substrate_manifest_signature,
        r0262_identity,
        closure_signature,
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    p90: dict[str, float] = {}
    jobs: list[dict[str, Any]] = []

    # 1-3. the host-int8 trains — all three seeds train from scratch (fresh round).
    train_outputs: dict[int, str] = {}
    train_ids: list[str] = []
    for seed in SEEDS:
        capability = capability_for_seed(seed)
        train_node = f"{TRAIN_ACTION}_seed{seed}"
        train_output = os.path.join(artifacts, capability)
        train_outputs[seed] = train_output
        train_ids.append(train_node)
        jobs.append({
            "id": train_node,
            "action": TRAIN_ACTION,
            "handler_module": "experiments.round0268_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [train_output],
            "done_marker": os.path.join(artifacts, f"{train_node}.done.json"),
            "expected_inputs": shared_inputs,
            "p90_wall_s": TRAIN_P90_WALL_S,
            "training_seed": int(seed),
            "capability": capability,
            "graph_manifest_signature": graph_manifest_signature,
            "substrate_manifest_signature": substrate_manifest_signature,
            "int8_substrate_manifest_signature": int8_substrate_manifest_signature,
            "cell_seed_invariant_sha256": cell_seed_invariant,
            "base_horizon": base_horizon,
            "treatment_closure": closure_signature,
            "node_policy": {"gpu_required": True, "training_performed": True, "cpu_heavy": False},
        })
        p90[train_node] = TRAIN_P90_WALL_S

    # 4. the three-cell panel (fresh trains only — one train dep per seed).
    panel_cells = [
        {
            "seed": int(seed),
            "capability": capability_for_seed(seed),
            "train_receipt": {
                "kind": "file",
                "canonical_path": os.path.join(train_outputs[seed], "train-receipt.json"),
            },
        }
        for seed in SEEDS
    ]
    panel_node = PANEL_ACTION
    panel_output = os.path.join(artifacts, PANEL_CAPABILITY)
    jobs.append({
        "id": panel_node,
        "action": PANEL_ACTION,
        "handler_module": "experiments.round0268_nodes",
        "handler_callable": "run_job",
        "deps": list(train_ids),
        "outputs": [panel_output],
        "done_marker": os.path.join(artifacts, f"{panel_node}.done.json"),
        "expected_inputs": _dedupe([
            *shared_inputs, r0218_panel, reserve, reserve_query_rows, reserve_truth,
        ]),
        "p90_wall_s": PANEL_P90_WALL_S,
        "graph_manifest_signature": graph_manifest_signature,
        "substrate_manifest_signature": substrate_manifest_signature,
        # The R0218 panel is bound for the R0216-c3 lineage reference + descriptive centroid
        # granularities (a config list, NOT R0218's centroid arrays; the centroids are re-fit
        # on the R0238 prefix inline).
        "panel_evidence": r0218_panel,
        "centroid_ks": list(CENTROID_KS),
        "heldout_reserve": reserve,
        "reserve_query_rows": reserve_query_rows,
        "reserve_truth": reserve_truth,
        "cells": panel_cells,
        "gate_registerable_here": False,
        "upstream_review_state": review_state,
        "node_policy": {"gpu_required": True, "training_performed": False, "cpu_heavy": False},
    })
    p90[panel_node] = PANEL_P90_WALL_S

    # 5. the seed-mean gate (CPU).
    gate_node = GATE_ACTION
    jobs.append({
        "id": gate_node,
        "action": GATE_ACTION,
        "handler_module": "experiments.round0268_nodes",
        "handler_callable": "run_job",
        "deps": [panel_node],
        "outputs": [os.path.join(artifacts, GATE_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{gate_node}.done.json"),
        "expected_inputs": _dedupe([
            *shared_inputs, r0265_floors, r0265_panel, p1_asymptote,
        ]),
        "p90_wall_s": GATE_P90_WALL_S,
        "panel": {"kind": "file", "canonical_path": os.path.join(panel_output, "fneg-100m-x2-panel.json")},
        "r0265_floors": r0265_floors,
        "r0265_panel": r0265_panel,
        "p1_asymptote": p1_asymptote,
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
        "schema": "round0268-fneg-100m-x2-seedmean-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            T.R0238_SUBSTRATE_CAPABILITY,
            T.R0243_GRAPH_CAPABILITY,
            T.R0262_INT8_CAPABILITY,
            R0218_PANEL_CAPABILITY,
            R0265N.GATE_CAPABILITY,
            R0265N.PANEL_CAPABILITY,
        ],
        "capabilities_produced": [*T.CAPABILITIES, PANEL_CAPABILITY, GATE_CAPABILITY],
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
                "train THREE FRESH 100M cells (seeds 42/43/44, SEQUENTIAL) of the promoted "
                "fneg recipe at dose ×2 on the host-int8 X path (R0266/R0267-validated), "
                "score them on R0265's instruments, and register the pre-registered 100M "
                "gate: the SEED-MEAN collapse inside P1's ×2 asymptote band (SAME as 50M) "
                "widened by 1.96·σ_fam/√3, plus per-seed backstops on collapse/fog/FFR. Every "
                "band/floor/σ_fam/P1-edge is read from a sealed artifact bound by sha256 — "
                "never a literal. Purity is descriptive-only (lineage: 100M-prefix != "
                "R0216-c3). A FRESH train: no salvage, no bind."
            ),
            "seeds": list(SEEDS),
            "trained_seeds": list(SEEDS),
            "x_residency": T.X_RESIDENCY,
            "dose_multiplier": T.DOSE_MULTIPLIER,
            "cell_seed_invariant_sha256": cell_seed_invariant,
            "base_horizon": base_horizon,
            "x2_horizon": int(T.DOSE_MULTIPLIER * base_horizon),
            "sealed_directed_edges": edges,
            "rows": ROWS,
            "dose_is_derived_from_the_sealed_edge_count": (
                f"successful_positive_lr_updates = {T.DOSE_MULTIPLIER} * "
                f"successful_updates_for_edges({edges}) = {T.DOSE_MULTIPLIER} * "
                f"{base_horizon} = {int(T.DOSE_MULTIPLIER * base_horizon)}"
            ),
            "consumes_sealed_100m_inputs": {
                "substrate": T.R0238_SUBSTRATE_CAPABILITY,
                "graph": T.R0243_GRAPH_CAPABILITY,
                "held_out_reserve_sha256": reserve["sha256"],
                "reserve_neighbour_truth_sha256": reserve_truth["sha256"],
            },
            "x_is_a_pre_sealed_int8_full_file": {
                "why": (
                    "the 100M host-int8 flagship LOADS R0262's sealed 100M int8 substrate "
                    "WHOLE (no prefix slice; at 100M the file IS the substrate) instead of "
                    "encoding fp32->int8 on the fly at train time; the multi-minute on-the-fly "
                    "encode blocked the node liveness watchdog. The design-fix path R0267 "
                    "proved at 50M, here at full 100M."
                ),
                "int8_substrate_manifest": T.INT8_SUBSTRATE_CAPABILITY,
                "int8_substrate_manifest_sha256": int8_substrate_manifest_signature["sha256"],
                "parent_artifact": T.R0262_INT8_CAPABILITY,
                "parent_identity_manifest_sha256": r0262_identity["sha256"],
                "full_file_law": T.full_file_law_block(),
            },
            "lineage_check": {
                "rule": "the 100M substrate's first-2M ordered hash != R0216-c3's sealed 2M "
                        "reference -> purity is DESCRIPTIVE (built on the 100M prefix, no "
                        "cross-lineage claim)",
                "r0216_c3_2m_reference_sha256": r0216_c3_reference,
                "expected": "non_match",
            },
            "consumes_sealed_gate_instruments": {
                "family_floors": R0265N.GATE_CAPABILITY,
                "n13_panel_for_sigma_fam": R0265N.PANEL_CAPABILITY,
                "p1_x2_asymptote_band_sha256": p1_asymptote["sha256"],
            },
            "host_rss_limit_gib": {
                "train": 115.0,
                "panel": 115.0,
                "basis": "analytic (50M measured 75.66 GiB scaled to 100M ~104 + 11 margin); "
                         "panel refined from the dry-run later",
            },
            "gate_registerable_here": True,
            "acceptance_rule": (
                "the round trains the three cells, scores them, and registers the gate. NO "
                "NUMERICAL OUTCOME makes it a failure: the 100M PASS/FAIL is a MEASUREMENT "
                "reported either way; a FAIL/AMBIGUOUS returns to the owner."
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
    parser = argparse.ArgumentParser(description="prepare the R0268 queue")
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=None)
    args = parser.parse_args(argv)
    path = prepare_round0268(
        release_sha=args.release_sha,
        queue_root=(args.queue_root or QUEUE_ROOT),
    )
    print(json.dumps({"queue": path, "sha256": file_sha256_manifest(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
