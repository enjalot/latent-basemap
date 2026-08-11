#!/usr/bin/env python3
"""Prepare, but never launch, the R0255 queue.

Sixteen nodes in one queue, in this order:

1-13. `train_minilm_mixed_2m_seed{58..70}` (GPU) — the thirteen cells the owner's
      2026-08-11 ruling asks for, taking the exact-graph 2M family from `n = 16` to
      `n = 29`.
14.   `replay_control_seed42` (GPU) — R0217's own canonical cell retrained on this
      release. **Not a family cell.**
15.   `score_minilm_mixed_2m_panel_n29` (GPU) — the thirteen new maps and the replay
      control on R0218's frozen panel, pooled with R0250's sealed sixteen.
16.   `register_calibrated_madn_floors_n29` (CPU) — `MAD_n` at `n = 29` by owner
      ruling, with the poolability test, attainability and detection power beside
      every floor, the independence control, and the JOINT criteria.

The script builds all thirteen cell configs **here**, from R0217's own
`train_config`, proves each reproduces R0217's *published* seed-invariant digest,
proves each reconstructs R0217's canonical config **byte for byte** once the nine
seed-bearing paths are restored, proves the twenty-nine full-config digests are
twenty-nine distinct values, and stamps the shared digest into every job.

It also seals the **treatment source closure**: for every module in
`round0255_treatment.TRAIN_CLOSURE_MODULES`, the SHA-256 of the file at R0217's
release commit `8ae96e16` and at this release, so every training node can refuse
unless the bytes it imported are the sealed ones and a reviewer can see exactly which
of them moved since the family was created.

The panel and gate nodes consume artifacts this queue has not produced yet, so their
references to them are **intra-queue**: a canonical path with no hash, resolved
inside the node.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
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
    train_config as r0217_train_config,
)
from basemap.round0221_minilm_2m_seed_extension import (
    TRAIN_SCHEMA as R0221_TRAIN_SCHEMA,
    capability_for_seed as r0221_capability_for_seed,
)
from basemap.round0230_minilm_2m_seed_extension_n13 import (
    TRAIN_SCHEMA as R0230_TRAIN_SCHEMA,
    capability_for_seed as r0230_capability_for_seed,
)
from basemap.round0247_registry import registry_fingerprint
from basemap.round0250_seed_extension_n16 import (
    TRAIN_SCHEMA as R0250_TRAIN_SCHEMA,
    capability_for_seed as r0250_capability_for_seed,
)
from basemap.round0255_gate_n29 import (
    GATED_METRICS,
    GATE_CAPABILITY,
    IDENTITY_BOUND_AT_N,
    JOINT_CRITERIA_RULE,
    N_EXACT,
    N_HELD_OUT,
    OWNER_RULING,
    OWNER_RULING_ESTIMATOR,
    RETAINED_FAMILY_SOURCES,
    SELECTION_RULE,
)
from basemap.round0255_panel_n29 import (
    ANCHOR_CORPUS_COUNTS,
    CENTROID_KS,
    CORPUS_SLUGS,
    DENSITY_V2_STATUS,
    HI_D_AGREEMENT,
    PANEL_CAPABILITY,
    PANEL_CAPABILITY_N16,
    PANEL_CAPABILITY_N29,
    PANEL_METRICS,
    PANEL_SCHEMA_N16,
    POOLED_CELL_SOURCES,
    REFERENCE_CONTENT_SHA256,
    REFERENCE_KEY,
)
from basemap.round0255_seed_extension_n29 import (
    CAPABILITIES,
    DEVICE_BUDGET_BYTES,
    DIMENSION,
    GATE_REGISTERABLE_HERE,
    GRAPH_CAPABILITY,
    GRAPH_K,
    GRAPH_SCHEMA,
    GRAPH_SOURCE_ROUND_ID,
    HOST_ANON_BUDGET_BYTES,
    HOST_RSS_LIMIT_GIB,
    IDENTITY_BOUND_AT_N29,
    MEMORY_POLICY,
    OWNER_RULING_N,
    POOLED_SEEDS,
    R0217_SEED_INVARIANT_SHA256,
    R0250_POOLED_SEEDS,
    REGISTERED_SUCCESSFUL_UPDATES,
    REGISTERED_UPDATE_BOUND,
    REPLAY_CONTROL_CAPABILITY,
    REPLAY_CONTROL_SEED,
    ROUND_ID,
    ROWS,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    STANDING_MINIMUM_N,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    TEMPLATE_SEED,
    USE_AMP,
    assert_extension_differs_only_by_seed,
    assert_reconstructs_r0217_template,
    capability_for_seed,
    predict_cell_footprint,
    replay_control_config,
    successful_updates_for_edges,
    train_config,
    validate_registered_dose,
)
from basemap.round0255_treatment import (
    CLOSURE_SCHEMA,
    R0217_RELEASE_SHA,
    TRAIN_CLOSURE_MODULES,
    file_sha256,
    runtime_closure_hashes,
)
from experiments.round0255_nodes import GATE_ACTION, PANEL_ACTION, TRAIN_ACTION
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0255"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0255-2026-08-11.md")

R0216_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    f"{R0216_CAPABILITY}"
)
GRAPH_MANIFEST = os.path.join(R0216_ARTIFACTS, "substrate-graph.json")
R0217_ARTIFACTS = "/data/latent-basemap/runs/round-0217/queue-correction-1/artifacts"
R0218_ARTIFACTS = (
    f"/data/latent-basemap/runs/round-0218/queue/artifacts/{PANEL_CAPABILITY}"
)
R0218_PANEL = os.path.join(R0218_ARTIFACTS, "seed-family-panel.json")
R0221_ARTIFACTS = "/data/latent-basemap/runs/round-0221/queue/artifacts"
R0223_COMPARISON = (
    "/data/latent-basemap/runs/round-0223/queue-correction-3/artifacts/"
    "minilm-mixed-2m-cuvs-graph-map-comparison-v1/cuvs-graph-map-comparison.json"
)
R0225_GATE = (
    "/data/latent-basemap/runs/round-0225/queue/artifacts/"
    "minilm-mixed-2m-tolerance-gates-n8-v1/minilm-tolerance-gates-n8.json"
)
R0228_COMPARISON = (
    "/data/latent-basemap/runs/round-0228/queue/artifacts/"
    "minilm-mixed-2m-cluster-spill-graph-map-comparison-v1/"
    "cluster-spill-graph-map-comparison.json"
)
R0230_ARTIFACTS = "/data/latent-basemap/runs/round-0230/queue/artifacts"
R0231_GATE = (
    "/data/latent-basemap/runs/round-0231/queue/artifacts/"
    "minilm-mixed-2m-robust-floors-n13-v1/minilm-robust-floors-n13.json"
)
R0234_GATE = (
    "/data/latent-basemap/runs/round-0234/queue/artifacts/"
    "minilm-mixed-2m-calibrated-robust-floors-n13-v1/"
    "minilm-calibrated-robust-floors-n13.json"
)
R0250_ARTIFACTS = "/data/latent-basemap/runs/round-0250/queue/artifacts"
R0250_PANEL = os.path.join(
    R0250_ARTIFACTS, PANEL_CAPABILITY_N16, "seed-family-panel-n16.json"
)
R0250_GATE = os.path.join(
    R0250_ARTIFACTS,
    "minilm-mixed-2m-calibrated-robust-floors-n16-v1",
    "minilm-calibrated-robust-floors-n16.json",
)

#: R0250 measured 0.19553-0.19625 GPU-h per cell under this identical treatment, so
#: fourteen cells is ~2.75 GPU-h; the n=16 panel node scored three maps and this one
#: scores fourteen. The registered cap is the round's 5.0 h, deliberately above the
#: estimate rather than tight to it.
GPU_HOURS_CAP = 5.0
TRAIN_P90_WALL_S = 1_800.0
PANEL_P90_WALL_S = 3_600.0
GATE_P90_WALL_S = 5_400.0


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
        raise RuntimeError("R0255 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0255 round must declare its required reviews")
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
            "The n=29 floors this round registers are registered-and-contingent "
            "until the rounds above carry accepted reviews."
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
        raise RuntimeError("R0255 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--basetemp=/data/tmp/pytest-r0255-smoke",
        "tests/test_round0255_contract.py",
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "TMPDIR": "/data/tmp/pytest-r0255-smoke-tmp",
    })
    os.makedirs("/data/tmp/pytest-r0255-smoke-tmp", exist_ok=True)
    started = time.monotonic()
    #: **No `timeout=` anywhere in this file, deliberately.** CPython implements
    #: `subprocess.run(..., timeout=N)` as `Popen.kill()`, i.e. a hidden SIGKILL,
    #: and `plan-minilm-100m-v2.md` makes purging that construct binding before any
    #: further GPU round.
    completed = subprocess.run(
        command,
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    receipt = prompt_contract.seal({
        "schema": "round0255-release-cpu-smoke-v1",
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
            "R0217-template config construction for seeds 58-70 and for the replay "
            "control, seed-invariant digest equality, byte-for-byte reconstruction "
            "of R0217's canonical config, the predictive memory guard including its "
            "refusal branch, the registered ceil-derived dose assertion, both "
            "guards' five planted defects each and their honest-input acceptance, "
            "the n=29 calibration entry, the owner-ruling registration, the "
            "poolability statistic, the attainability/power table, the independence "
            "control including its not-inert arm, the joint-criteria construction "
            "and AND, and the entry path of every one of the three node actions"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0255 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _treatment_closure_seal(release_sha: str) -> dict[str, Any]:
    """SHA-256 of every training-closure source at R0217's release and at this one.

    The old side is read with `git show` from the release checkout, so it is the
    committed bytes of `8ae96e16` rather than anything reconstructed. The new side
    is the file on disk, and every training node re-hashes it at import time and
    refuses unless it matches.
    """
    observed = runtime_closure_hashes(TRAIN_CLOSURE_MODULES)
    files: dict[str, Any] = {}
    changed: list[str] = []
    absent: list[str] = []
    diffs: dict[str, Any] = {}
    for name, entry in observed.items():
        relative = os.path.relpath(entry["path"], os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        old = subprocess.run(
            ["git", "-C", RELEASE_ROOT, "show", f"{R0217_RELEASE_SHA}:{relative}"],
            check=False,
            capture_output=True,
        )
        if old.returncode == 0:
            old_sha = hashlib.sha256(old.stdout).hexdigest()
            old_bytes = len(old.stdout)
        else:
            old_sha = None
            old_bytes = None
            absent.append(name)
        if old_sha is not None and old_sha != entry["sha256"]:
            changed.append(name)
            stat = subprocess.run(
                [
                    "git", "-C", RELEASE_ROOT, "diff", "--numstat",
                    R0217_RELEASE_SHA, release_sha, "--", relative,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            diffs[name] = stat.stdout.strip()
        files[name] = {
            "module": name,
            "path": relative,
            "bytes_at_release": entry["bytes"],
            "sha256_at_release": entry["sha256"],
            "bytes_at_r0217": old_bytes,
            "sha256_at_r0217": old_sha,
            "present_at_r0217": old_sha is not None,
            "changed_since_r0217": old_sha is not None and old_sha != entry["sha256"],
        }
    return prompt_contract.seal({
        "schema": CLOSURE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "r0217_release_sha": R0217_RELEASE_SHA,
        "modules": list(TRAIN_CLOSURE_MODULES),
        "files": files,
        "modules_changed_since_r0217": changed,
        "modules_absent_at_r0217": absent,
        "numstat_by_changed_module": diffs,
        "how_to_read_this": (
            "A module absent at R0217 is a CONTRACT module a later extension added "
            "(R0221's and R0230's), not a change to what R0217's own cells ran. A "
            "changed module is a real delta and is named: this round does NOT claim "
            "the closure is byte-identical to R0217's, it claims the bytes that ran "
            "are the bytes sealed here and publishes exactly which of them moved. "
            "The positive evidence that the treatment is unchanged is the seed-42 "
            "replay control."
        ),
    })


def _sealed_graph() -> tuple[dict[str, Any], dict[str, Any], int]:
    signature = expected_input_signature(GRAPH_MANIFEST)
    manifest = prompt_contract.read_sealed(
        signature["canonical_path"], label="sealed R0216 substrate+graph receipt"
    )
    checks = manifest.get("graph_checks") or {}
    if (
        manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("round_id") != GRAPH_SOURCE_ROUND_ID
        or manifest.get("capability") != GRAPH_CAPABILITY
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or int(manifest.get("k", -1)) != GRAPH_K
        or manifest.get("training_performed") is not False
    ):
        raise RuntimeError("R0255 sealed R0216 substrate+graph contract changed")
    if int(checks.get("zero_degree_rows", -1)) != 0:
        raise RuntimeError("R0255 requires a graph with zero degree-zero rows")
    edges = int(manifest.get("directed_edge_count", 0)) or int(
        checks.get("directed_edges", 0)
    )
    if edges != SEALED_DIRECTED_EDGES:
        raise RuntimeError(
            f"R0255 sealed graph reports {edges} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    return signature, manifest, edges


def _prior_family(
    manifest_signature: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Read the sixteen sealed cells R0255 extends: R0217's ... R0250's."""
    invariants: set[str] = set()
    config_hashes: dict[str, str] = {}
    model_hashes: dict[str, str] = {}
    receipts: dict[str, Any] = {}
    sources = (
        [(seed, "0217", R0217_ARTIFACTS, R0217_TRAIN_SCHEMA, r0217_capability_for_seed)
         for seed in (42, 43, 44, 45)]
        + [(seed, "0221", R0221_ARTIFACTS, R0221_TRAIN_SCHEMA, r0221_capability_for_seed)
           for seed in (46, 47, 48, 49)]
        + [(seed, "0230", R0230_ARTIFACTS, R0230_TRAIN_SCHEMA, r0230_capability_for_seed)
           for seed in (50, 51, 52, 53, 54)]
        + [(seed, "0250", R0250_ARTIFACTS, R0250_TRAIN_SCHEMA, r0250_capability_for_seed)
           for seed in (55, 56, 57)]
    )
    for seed, round_id, root, schema, capability_fn in sources:
        capability = capability_fn(seed)
        receipt_path = os.path.join(root, capability, "train-receipt.json")
        receipt = prompt_contract.read_sealed(
            receipt_path, label=f"R{round_id} seed-{seed} train receipt"
        )
        train_checks = receipt.get("train_checks") or {}
        if (
            receipt.get("schema") != schema
            or receipt.get("round_id") != round_id
            or receipt.get("capability") != capability
            or int(receipt.get("training_seed", -1)) != seed
            or int(receipt.get("directed_edges", -1)) != SEALED_DIRECTED_EDGES
            or receipt.get("training_performed") is not True
            or not train_checks
            or not all(bool(value) for value in train_checks.values())
        ):
            raise RuntimeError(f"R{round_id} seed-{seed} train receipt contract changed")
        if (
            dict(receipt.get("substrate") or {}) != dict(manifest["substrate"])
            or dict(receipt.get("graph_manifest") or {}) != manifest_signature
        ):
            raise RuntimeError(
                f"R{round_id} seed-{seed} was not trained on the substrate R0255 "
                "extends"
            )
        if int(receipt.get("optimizer_updates", -1)) != REGISTERED_SUCCESSFUL_UPDATES:
            raise RuntimeError(
                f"R{round_id} seed-{seed} horizon is not the registered "
                f"{REGISTERED_SUCCESSFUL_UPDATES}"
            )
        invariants.add(str(receipt["seed_invariant_sha256"]))
        config_hashes[str(seed)] = str(receipt["production_config_sha256"])
        model_hashes[str(seed)] = str(receipt["model"]["sha256"])
        if seed == REPLAY_CONTROL_SEED:
            receipts["replay_target"] = {
                "path": receipt_path,
                "signature": expected_input_signature(receipt_path),
                "production_config_sha256": config_hashes[str(seed)],
                "model_sha256": model_hashes[str(seed)],
            }
    if invariants != {R0217_SEED_INVARIANT_SHA256}:
        raise RuntimeError(
            "the sixteen prior cells do not carry one published seed-invariant "
            f"digest: {sorted(invariants)}"
        )
    if len(set(model_hashes.values())) != len(R0250_POOLED_SEEDS):
        raise RuntimeError("the sixteen prior cells contain a duplicated checkpoint")
    return {
        "seed_invariant_sha256": R0217_SEED_INVARIANT_SHA256,
        "config_sha256_by_seed": config_hashes,
        "model_sha256_by_seed": model_hashes,
        **receipts,
    }


def _sealed_panels() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str]:
    """R0218's frozen panel bytes and R0250's sealed sixteen-cell pooled table."""
    signature = expected_input_signature(R0218_PANEL)
    panel = prompt_contract.read_sealed(
        R0218_PANEL, label="R0218 MiniLM 2M four-seed panel"
    )
    reference = dict(panel["shared_high_d_reference"])
    if expected_input_signature(reference["canonical_path"]) != reference:
        raise RuntimeError(
            "R0218's published high-D reference bytes changed; the twenty-nine "
            "cells would not be poolable"
        )
    if (
        str(panel["high_d_reference_key"]) != REFERENCE_KEY
        or str(panel["high_d_reference_content_sha256"]) != REFERENCE_CONTENT_SHA256
        or dict(panel["anchor_corpus_counts"]) != dict(ANCHOR_CORPUS_COUNTS)
    ):
        raise RuntimeError(
            "R0218's reference identity is not the registered one; STOP — the "
            "twenty-nine cells are not poolable"
        )
    for seed in panel["seeds"]:
        numerators = panel["cells"][str(seed)]["panel"]["purity_numerators"]
        for key, expected in HI_D_AGREEMENT.items():
            if float(numerators[key]["hi_D_agreement"]) != float(expected):
                raise RuntimeError(
                    f"R0218 seed-{seed} hi-D agreement {key} is not {expected}"
                )
    seed42_coordinates = str(
        panel["cells"][str(REPLAY_CONTROL_SEED)]["coordinates_ordered_sha256"]
    )
    inputs = [signature, reference]
    declared = dict(panel.get("centroids") or {})
    if set(declared) != {str(k) for k in CENTROID_KS}:
        raise RuntimeError("R0218 centroid vocabularies changed")
    for k in CENTROID_KS:
        centroid = dict(declared[str(k)])
        if expected_input_signature(centroid["canonical_path"]) != centroid:
            raise RuntimeError(f"R0218 published centroids k{k} bytes changed")
        inputs.append(centroid)

    n16_signature = expected_input_signature(R0250_PANEL)
    n16 = prompt_contract.read_sealed(
        R0250_PANEL, label="R0250 sealed sixteen-cell panel"
    )
    if (
        n16.get("schema") != PANEL_SCHEMA_N16
        or n16.get("round_id") != "0250"
        or int(n16.get("n", -1)) != len(R0250_POOLED_SEEDS)
        or str(n16.get("high_d_reference_key")) != REFERENCE_KEY
        or n16.get("gate_registerable_here") is not False
    ):
        raise RuntimeError("R0250's sixteen-cell panel receipt contract changed")
    inputs.append(n16_signature)
    return panel, n16_signature, inputs, seed42_coordinates


def _sealed_gate_inputs() -> list[dict[str, Any]]:
    """The six sealed artifacts the joint criteria and the held-out cells read."""
    signatures = []
    for path, label in (
        (R0223_COMPARISON, "R0223 sealed cuVS comparison"),
        (R0225_GATE, "R0225 sealed n=8 tolerance gate"),
        (R0228_COMPARISON, "R0228 sealed cluster-spill comparison"),
        (R0231_GATE, "R0231 sealed n=13 robust gate"),
        (R0234_GATE, "R0234 sealed n=13 calibrated gate"),
        (R0250_GATE, "R0250 sealed n=16 calibrated gate"),
    ):
        prompt_contract.read_sealed(path, label=label)
        signatures.append(expected_input_signature(path))
    return signatures


def prepare_round0255(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    only_nodes: tuple[str, ...] | None = None,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0255 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    graph_manifest_signature, graph_manifest, edges = _sealed_graph()
    updates = successful_updates_for_edges(edges)
    if updates > REGISTERED_UPDATE_BOUND:
        raise RuntimeError(
            f"R0255 derived horizon {updates} exceeds the registered bound "
            f"{REGISTERED_UPDATE_BOUND}"
        )
    dose = validate_registered_dose(updates=updates, edge_count=edges)
    prior = _prior_family(graph_manifest_signature, graph_manifest)
    panel, panel_n16_signature, panel_inputs, seed42_coordinates = _sealed_panels()
    gate_inputs = _sealed_gate_inputs()
    review_state = _upstream_review_state(list(required_reviews))

    substrate_signature = dict(graph_manifest["substrate"])
    graph_signature = dict(graph_manifest["graph"])
    provenance_signature = dict(graph_manifest["provenance"])

    template, _template_sha = r0217_train_config(
        seed=TEMPLATE_SEED,
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        substrate_signature=substrate_signature,
        graph_edges=edges,
        rows=ROWS,
    )
    configs: dict[int, dict[str, Any]] = {}
    reconstructions: dict[str, Any] = {}
    for seed in SEEDS:
        config, _sha = train_config(
            seed=seed,
            graph_signature=graph_signature,
            graph_manifest_signature=graph_manifest_signature,
            substrate_signature=substrate_signature,
            graph_edges=edges,
            rows=ROWS,
        )
        configs[seed] = config
        reconstructions[str(seed)] = assert_reconstructs_r0217_template(config, template)
    family = assert_extension_differs_only_by_seed(
        configs, expected_seed_invariant=prior["seed_invariant_sha256"]
    )
    if not family["matches_r0217_published_seed_invariant"]:
        raise RuntimeError(
            "R0255 seed-invariant digest does not match R0217's published value"
        )
    pooled_config_hashes = {
        **prior["config_sha256_by_seed"],
        **family["per_seed_config_sha256"],
    }
    if len(set(pooled_config_hashes.values())) != len(POOLED_SEEDS):
        raise RuntimeError(
            "R0255 cell configs collide with the prior sixteen: the twenty-nine "
            "cells must be twenty-nine distinct configs sharing one seed-invariant "
            "digest"
        )

    replay_config, replay_config_sha = replay_control_config(
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        substrate_signature=substrate_signature,
        graph_edges=edges,
        rows=ROWS,
    )
    if replay_config_sha != prior["replay_target"]["production_config_sha256"]:
        raise RuntimeError(
            "R0255 replay control config digest does not equal R0217's sealed "
            f"seed-{REPLAY_CONTROL_SEED} production-config digest"
        )

    predictions = {str(seed): predict_cell_footprint(seed) for seed in SEEDS}
    predictions[REPLAY_CONTROL_CAPABILITY] = predict_cell_footprint(
        REPLAY_CONTROL_SEED, replay_control=True
    )
    refused = sorted(
        key for key, item in predictions.items() if item["refused_a_priori"]
    )

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0255 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    closure_path = os.path.join(preflight, "treatment-source-closure.json")
    atomic_write_new_json(
        closure_path, _treatment_closure_seal(release_sha), immutable=True
    )
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    family_path = os.path.join(preflight, "seed-extension-n29-identity.json")
    atomic_write_new_json(
        family_path,
        prompt_contract.seal({
            "schema": "round0255-seed-extension-n29-config-identity-v1",
            "round_id": ROUND_ID,
            "release_sha": release_sha,
            "sealed_directed_edges": edges,
            "successful_positive_lr_updates": updates,
            "dose_registration": dose,
            "prior_family": prior,
            "family": family,
            "byte_for_byte_reconstruction_of_r0217": reconstructions,
            "replay_control": {
                "seed": REPLAY_CONTROL_SEED,
                "capability": REPLAY_CONTROL_CAPABILITY,
                "is_a_family_cell": False,
                "production_config_sha256": replay_config_sha,
                "equals_r0217_sealed_production_config_sha256": True,
                "r0218_seed42_coordinates_ordered_sha256": seed42_coordinates,
            },
            "pooled_seed_family": list(POOLED_SEEDS),
            "n_pooled": len(POOLED_SEEDS),
            "owner_ruling_n": OWNER_RULING_N,
            "standing_minimum_n": STANDING_MINIMUM_N,
            "identity_bound_at_n_pooled": IDENTITY_BOUND_AT_N29,
            "pooled_config_sha256_by_seed": pooled_config_hashes,
            "memory_predictions": predictions,
            "refused_a_priori": refused,
            "memory_policy": MEMORY_POLICY,
            "registry_fingerprint": registry_fingerprint(),
            "configs": {str(seed): configs[seed] for seed in SEEDS},
            "replay_control_config": replay_config,
        }),
        immutable=True,
    )

    closure_signature = expected_input_signature(closure_path)
    shared_inputs = _dedupe([
        round_signature,
        graph_manifest_signature,
        substrate_signature,
        graph_signature,
        provenance_signature,
        expected_input_signature(smoke_path),
        expected_input_signature(family_path),
        closure_signature,
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    jobs: list[dict[str, Any]] = []
    p90: dict[str, float] = {}

    panel_cells: list[dict[str, Any]] = []
    train_nodes: list[str] = []
    for seed in SEEDS:
        if predictions[str(seed)]["refused_a_priori"]:
            continue
        capability = capability_for_seed(seed)
        node = f"train_minilm_mixed_2m_seed{seed}"
        output = os.path.join(artifacts, capability)
        jobs.append({
            "id": node,
            "action": TRAIN_ACTION,
            "handler_module": "experiments.round0255_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"{node}.done.json"),
            "expected_inputs": shared_inputs,
            "p90_wall_s": TRAIN_P90_WALL_S,
            "training_seed": int(seed),
            "capability": capability,
            "is_replay_control": False,
            "graph_manifest": GRAPH_MANIFEST,
            "graph_manifest_signature": graph_manifest_signature,
            "family_seed_invariant_sha256": family["seed_invariant_sha256"],
            "registered_dose_bound": REGISTERED_UPDATE_BOUND,
            "memory_prediction": predictions[str(seed)],
            "treatment_closure": closure_signature,
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
            "is_replay_control": False,
            "train_receipt": {
                "kind": "file",
                "canonical_path": os.path.join(output, "train-receipt.json"),
            },
        })
    if len(train_nodes) != len(SEEDS):
        raise RuntimeError("R0255 refused a cell a priori; the family would be short")

    replay_node = "replay_control_seed42"
    replay_output = os.path.join(artifacts, REPLAY_CONTROL_CAPABILITY)
    jobs.append({
        "id": replay_node,
        "action": TRAIN_ACTION,
        "handler_module": "experiments.round0255_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [replay_output],
        "done_marker": os.path.join(artifacts, f"{replay_node}.done.json"),
        "expected_inputs": _dedupe([
            *shared_inputs, prior["replay_target"]["signature"]
        ]),
        "p90_wall_s": TRAIN_P90_WALL_S,
        "training_seed": REPLAY_CONTROL_SEED,
        "capability": REPLAY_CONTROL_CAPABILITY,
        "is_replay_control": True,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": graph_manifest_signature,
        "family_seed_invariant_sha256": family["seed_invariant_sha256"],
        "registered_dose_bound": REGISTERED_UPDATE_BOUND,
        "memory_prediction": predictions[REPLAY_CONTROL_CAPABILITY],
        "treatment_closure": closure_signature,
        "r0217_receipt": prior["replay_target"]["signature"],
        "node_policy": {
            "gpu_required": True,
            "training_performed": True,
            "cpu_heavy": False,
        },
    })
    p90[replay_node] = TRAIN_P90_WALL_S
    panel_cells.append({
        "seed": REPLAY_CONTROL_SEED,
        "capability": REPLAY_CONTROL_CAPABILITY,
        "is_replay_control": True,
        "train_receipt": {
            "kind": "file",
            "canonical_path": os.path.join(replay_output, "train-receipt.json"),
        },
    })

    panel_node = "score_minilm_mixed_2m_panel_n29"
    panel_output = os.path.join(artifacts, PANEL_CAPABILITY_N29)
    jobs.append({
        "id": panel_node,
        "action": PANEL_ACTION,
        "handler_module": "experiments.round0255_nodes",
        "handler_callable": "run_job",
        "deps": list(train_nodes) + [replay_node],
        "outputs": [panel_output],
        "done_marker": os.path.join(artifacts, f"{panel_node}.done.json"),
        "expected_inputs": _dedupe([*shared_inputs, *panel_inputs]),
        "p90_wall_s": PANEL_P90_WALL_S,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": graph_manifest_signature,
        "panel_evidence": R0218_PANEL,
        "panel_n16_signature": panel_n16_signature,
        "prior_model_sha256_by_seed": prior["model_sha256_by_seed"],
        "r0218_seed42_coordinates_ordered_sha256": seed42_coordinates,
        "cells": panel_cells,
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
            "cpu_heavy": False,
        },
    })
    p90[panel_node] = PANEL_P90_WALL_S

    gate_node = "register_calibrated_madn_floors_n29"
    jobs.append({
        "id": gate_node,
        "action": GATE_ACTION,
        "handler_module": "experiments.round0255_nodes",
        "handler_callable": "run_job",
        "deps": [panel_node],
        "outputs": [os.path.join(artifacts, GATE_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{gate_node}.done.json"),
        "expected_inputs": _dedupe([*shared_inputs, *gate_inputs]),
        "p90_wall_s": GATE_P90_WALL_S,
        "panel_n29": {
            "kind": "file",
            "canonical_path": os.path.join(panel_output, "seed-family-panel-n29.json"),
        },
        "r0223_comparison": gate_inputs[0],
        "r0225_gate": gate_inputs[1],
        "r0228_comparison": gate_inputs[2],
        "r0231_gate": gate_inputs[3],
        "r0234_gate": gate_inputs[4],
        "r0250_gate": gate_inputs[5],
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
            "cpu_heavy": True,
        },
    })
    p90[gate_node] = GATE_P90_WALL_S

    if only_nodes is not None:
        wanted = set(only_nodes)
        unknown = wanted - {job["id"] for job in jobs}
        if unknown:
            raise RuntimeError(f"R0255 correction queue names unknown nodes: {unknown}")
        jobs = [job for job in jobs if job["id"] in wanted]
        for job in jobs:
            if set(job["deps"]) - wanted:
                raise RuntimeError(
                    f"R0255 correction node {job['id']} depends on a node the "
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
        "schema": "round0255-n29-madn-gate-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            GRAPH_CAPABILITY,
            PANEL_CAPABILITY,
            PANEL_CAPABILITY_N16,
            "minilm-mixed-2m-cuvs-graph-map-comparison-v1",
            "minilm-mixed-2m-tolerance-gates-n8-v1",
            "minilm-mixed-2m-cluster-spill-graph-map-comparison-v1",
            "minilm-mixed-2m-robust-floors-n13-v1",
            "minilm-mixed-2m-calibrated-robust-floors-n13-v1",
            "minilm-mixed-2m-calibrated-robust-floors-n16-v1",
            *(r0217_capability_for_seed(seed) for seed in (42, 43, 44, 45)),
            *(r0221_capability_for_seed(seed) for seed in (46, 47, 48, 49)),
            *(r0230_capability_for_seed(seed) for seed in (50, 51, 52, 53, 54)),
            *(r0250_capability_for_seed(seed) for seed in (55, 56, 57)),
        ],
        "capabilities_produced": [
            *CAPABILITIES,
            REPLAY_CONTROL_CAPABILITY,
            PANEL_CAPABILITY_N29,
            GATE_CAPABILITY,
        ],
        "jobs": jobs,
        "p90_wall_s": p90,
        "registered": {
            "owner_ruling": OWNER_RULING,
            "owner_ruling_estimator": OWNER_RULING_ESTIMATOR,
            "owner_ruling_n": OWNER_RULING_N,
            "seeds": list(SEEDS),
            "pooled_seed_family": list(POOLED_SEEDS),
            "n_pooled": len(POOLED_SEEDS),
            "n_exact": N_EXACT,
            "n_held_out": N_HELD_OUT,
            "identity_bound_at_n": IDENTITY_BOUND_AT_N,
            "standing_minimum_n": STANDING_MINIMUM_N,
            "successful_positive_lr_updates": updates,
            "registered_update_bound": REGISTERED_UPDATE_BOUND,
            "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
            "achieved_positive_draws_per_edge": dose[
                "achieved_positive_draws_per_edge"
            ],
            "sealed_directed_edges": edges,
            "rows": ROWS,
            "dimension": DIMENSION,
            "use_amp": USE_AMP,
            "device_budget_bytes": DEVICE_BUDGET_BYTES,
            "host_anonymous_budget_bytes": HOST_ANON_BUDGET_BYTES,
            "host_rss_limit_gib": HOST_RSS_LIMIT_GIB,
            "gated_metrics": list(GATED_METRICS),
            "density_v2_status": DENSITY_V2_STATUS,
            "panel_metrics": list(PANEL_METRICS),
            "corpus_slugs": list(CORPUS_SLUGS),
            "pooled_cell_sources": dict(POOLED_CELL_SOURCES),
            "selection_rule_is_reported_not_applied": SELECTION_RULE,
            "joint_criteria_rule": JOINT_CRITERIA_RULE,
            "retained_family_sources": [dict(item) for item in RETAINED_FAMILY_SOURCES],
            "gate_registerable_here": True,
            "gate_registerable_in_train_or_panel_nodes": GATE_REGISTERABLE_HERE,
            "replay_control": {
                "seed": REPLAY_CONTROL_SEED,
                "capability": REPLAY_CONTROL_CAPABILITY,
                "is_a_family_cell": False,
                "why": (
                    "R0251 proved the map-side SCORER stable by re-scoring an "
                    "archived checkpoint; review-0251 noted the release diff was "
                    "stronger evidence. This retrains R0217's own cell on this "
                    "release, which is the train-side half neither covered. It is "
                    "never pooled: the family-purity guard refuses a family that "
                    "contains it, and ships a positive control that plants exactly "
                    "that."
                ),
            },
        },
    })
    manifest_path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(manifest_path, queue, immutable=True)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="prepare the R0255 queue")
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    parser.add_argument("--only-nodes", default="")
    args = parser.parse_args(argv)
    only = tuple(item for item in args.only_nodes.split(",") if item) or None
    path = prepare_round0255(
        release_sha=args.release_sha, queue_root=args.queue_root, only_nodes=only
    )
    print(json.dumps({"queue": path, "sha256": file_sha256(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
