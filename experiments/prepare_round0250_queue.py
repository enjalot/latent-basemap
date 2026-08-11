#!/usr/bin/env python3
"""Prepare, but never launch, the R0250 queue.

Six nodes in one queue, in this order:

1. `trainloop_0250` (GPU) — the trainer's own inner loops, two arms on a short
   rung, scored by a real `AbortPollGate` against the registered ceiling.
2. `blocksize_0250` (CPU) — the `truthcos_0247` block-size resolution.
3-5. `train_minilm_mixed_2m_seed{55,56,57}` (GPU) — the three cells that take the
   exact-graph family from `n = 13` to the standing minimum `n = 16`.
6. `score_minilm_mixed_2m_panel_n16` (GPU) — the three new maps on R0218's frozen
   panel, pooled with R0230's thirteen.
7. `register_calibrated_robust_floors_n16` (CPU) — the gate, derived at 16, under
   the JOINT criteria.

The script builds all three cell configs **here**, from R0217's own `train_config`,
proves each reproduces R0217's *published* seed-invariant digest, proves each
reconstructs R0217's canonical config **byte for byte** once the nine seed-bearing
paths are restored, proves the sixteen full-config digests are sixteen distinct
values, and stamps the shared digest into every job so each node re-derives it and
refuses to train if its own config drifted. It runs the predictive memory guard for
every cell at prepare time and seals the prediction, including `refused_a_priori`.

The panel and gate nodes consume artifacts this queue has not produced yet, so
their references to them are **intra-queue**: a canonical path with no hash,
resolved inside the node. That is R0229's fix.
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
    HIDDEN_DIMENSION,
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
from basemap.round0238_rung5 import GRAPH_K as PROBE_GRAPH_K, TRUTH_PROBE_ROWS
from basemap.round0247_registry import registry_fingerprint
from basemap.round0250_blocksize import (
    ADAPTIVE_TARGET_FRACTION,
    BLOCKSIZE_CAPABILITY,
    BLOCK_SIZES_PROBED,
    PER_ROW_COST_STABILITY_LIMIT,
)
from basemap.round0250_gate_n16 import (
    GATED_METRICS,
    GATE_CAPABILITY,
    IDENTITY_BOUND_AT_N,
    JOINT_CRITERIA_RULE,
    N_EXACT,
    N_HELD_OUT,
    RETAINED_FAMILY_SOURCES,
    SELECTION_RULE,
)
from basemap.round0250_panel_n16 import (
    ANCHOR_CORPUS_COUNTS,
    CENTROID_KS,
    CORPUS_SLUGS,
    DENSITY_V2_STATUS,
    HI_D_AGREEMENT,
    PANEL_CAPABILITY,
    PANEL_CAPABILITY_N13,
    PANEL_CAPABILITY_N16,
    PANEL_METRICS,
    PANEL_SCHEMA_N13,
    POOLED_CELL_SOURCES,
    REFERENCE_CONTENT_SHA256,
    REFERENCE_KEY,
)
from basemap.round0250_seed_extension_n16 import (
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
    IDENTITY_BOUND_AT_N16,
    MEMORY_POLICY,
    POOLED_SEEDS,
    R0217_SEED_INVARIANT_SHA256,
    R0230_POOLED_SEEDS,
    REGISTERED_ACHIEVED_DRAWS_PER_EDGE,
    REGISTERED_SUCCESSFUL_UPDATES,
    REGISTERED_UPDATE_BOUND,
    ROUND_ID,
    ROWS,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    STANDING_MINIMUM_N,
    SWAP_GROWTH_ABORT_BYTES,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    TEMPLATE_SEED,
    USE_AMP,
    assert_extension_differs_only_by_seed,
    assert_reconstructs_r0217_template,
    capability_for_seed,
    predict_cell_footprint,
    successful_updates_for_edges,
    train_config,
    validate_registered_dose,
)
from basemap.round0250_trainer_loops import (
    PER_BATCH_HOOK,
    PROJECTION_TARGET_HOURS,
    SHORT_HORIZON_UPDATES,
    TRAINER_LOOP_CAPABILITY,
)
from experiments.round0250_nodes import (
    BLOCKSIZE_ACTION,
    GATE_ACTION,
    PANEL_ACTION,
    TRAINLOOP_ACTION,
    TRAIN_ACTION,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0250"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0250-2026-08-11.md")

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
R0230_PANEL = os.path.join(
    R0230_ARTIFACTS, PANEL_CAPABILITY_N13, "seed-family-panel-n13.json"
)
R0231_GATE = (
    "/data/latent-basemap/runs/round-0231/queue/artifacts/"
    "minilm-mixed-2m-robust-floors-n13-v1/minilm-robust-floors-n13.json"
)
R0234_GATE = (
    "/data/latent-basemap/runs/round-0234/queue/artifacts/"
    "minilm-mixed-2m-calibrated-robust-floors-n13-v1/"
    "minilm-calibrated-robust-floors-n13.json"
)
R0238_PROBE = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-uniform-probe-k15-truth-v1"
)
R0238_SUBSTRATE = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-nested-substrate-and-reserves-v1/substrate.f32.npy"
)
R0247_TRUTHCOS = (
    "/data/latent-basemap/runs/round-0247/queue/artifacts/"
    "minilm-mixed-100000k-uniform-probe-k15-truth-cos-f64-v1/"
    "truth-cosine-precision.json"
)

#: The 100M nested substrate's geometry, as R0238 sealed it.
SUBSTRATE_ROWS = 100_000_000
SUBSTRATE_DIMENSION = DIMENSION

#: Rows walked per block-size arm. Five arms (four cold + one warm) plus the
#: adaptive probe touch `6 * ARM_ROWS * 15` substrate rows in total, which is a
#: small fraction of the probe and a few GB of random reads.
BLOCK_ARM_ROWS = 60_000

#: R0230 measured 0.19765-0.19825 GPU-h per cell under this identical treatment,
#: so three cells is ~0.60 GPU-h; the panel node scored five cells in 22.8 s and
#: this one scores three; the trainer-loop node runs two short fits of
#: SHORT_HORIZON_UPDATES updates each. The registered cap is the round's 2.0 h,
#: deliberately above the estimate rather than tight to it.
GPU_HOURS_CAP = 2.0
TRAINLOOP_P90_WALL_S = 1_800.0
BLOCKSIZE_P90_WALL_S = 2_400.0
TRAIN_P90_WALL_S = 1_800.0
PANEL_P90_WALL_S = 1_800.0
GATE_P90_WALL_S = 1_800.0


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
        raise RuntimeError("R0250 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0250 round must declare its required reviews")
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
            "The n=16 floors this round registers are registered-and-contingent "
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
        raise RuntimeError("R0250 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--basetemp=/data/tmp/pytest-r0250-smoke",
        "tests/test_round0250_contract.py",
        "tests/test_round0250_cpu_smoke.py",
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "TMPDIR": "/data/tmp/pytest-r0250-smoke-tmp",
    })
    os.makedirs("/data/tmp/pytest-r0250-smoke-tmp", exist_ok=True)
    started = time.monotonic()
    #: **No `timeout=` anywhere in this file, deliberately.** CPython implements
    #: `subprocess.run(..., timeout=N)` as `Popen.kill()`, i.e. a hidden SIGKILL,
    #: and `plan-minilm-100m-v2.md` makes purging that construct binding before
    #: any further GPU round. The hazard is not hypothetical for a pytest child:
    #: result-0249's addendum measured 36 CUDA-family mappings in an idle pytest
    #: process that had merely imported torch, while the card read 2 MiB. A
    #: contract test greps this file for the construct.
    completed = subprocess.run(
        command,
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    receipt = prompt_contract.seal({
        "schema": "round0250-release-cpu-smoke-v1",
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
            "R0217-template config construction for seeds 55-57, seed-invariant "
            "digest equality, byte-for-byte reconstruction of R0217's canonical "
            "config, the predictive memory guard including its refusal branch, the "
            "registered ceil-derived dose assertion, the short-rung config diff "
            "guard, the per-batch poll installer including its restore, the "
            "ceiling and projection arithmetic, the block-size resolution rule, "
            "the n=16 calibration and selection, the joint-criteria construction "
            "and AND, the falsifiability statement, and the entry path of every "
            "one of the five node actions"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0250 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


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
        raise RuntimeError("R0250 sealed R0216 substrate+graph contract changed")
    if int(checks.get("zero_degree_rows", -1)) != 0:
        raise RuntimeError("R0250 requires a graph with zero degree-zero rows")
    edges = int(manifest.get("directed_edge_count", 0)) or int(
        checks.get("directed_edges", 0)
    )
    if edges != SEALED_DIRECTED_EDGES:
        raise RuntimeError(
            f"R0250 sealed graph reports {edges} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    return signature, manifest, edges


def _prior_family(
    manifest_signature: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Read the thirteen sealed cells R0250 extends: R0217's, R0221's, R0230's."""
    invariants: set[str] = set()
    config_hashes: dict[str, str] = {}
    model_hashes: dict[str, str] = {}
    sources = (
        [(seed, "0217", R0217_ARTIFACTS, R0217_TRAIN_SCHEMA, r0217_capability_for_seed)
         for seed in (42, 43, 44, 45)]
        + [(seed, "0221", R0221_ARTIFACTS, R0221_TRAIN_SCHEMA, r0221_capability_for_seed)
           for seed in (46, 47, 48, 49)]
        + [(seed, "0230", R0230_ARTIFACTS, R0230_TRAIN_SCHEMA, r0230_capability_for_seed)
           for seed in (50, 51, 52, 53, 54)]
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
                f"R{round_id} seed-{seed} was not trained on the substrate R0250 "
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
    if invariants != {R0217_SEED_INVARIANT_SHA256}:
        raise RuntimeError(
            "the thirteen prior cells do not carry one published seed-invariant "
            f"digest: {sorted(invariants)}"
        )
    if len(set(model_hashes.values())) != len(R0230_POOLED_SEEDS):
        raise RuntimeError("the thirteen prior cells contain a duplicated checkpoint")
    return {
        "seed_invariant_sha256": R0217_SEED_INVARIANT_SHA256,
        "config_sha256_by_seed": config_hashes,
        "model_sha256_by_seed": model_hashes,
    }


def _sealed_panels() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """R0218's frozen panel bytes and R0230's sealed thirteen-cell pooled table."""
    signature = expected_input_signature(R0218_PANEL)
    panel = prompt_contract.read_sealed(
        R0218_PANEL, label="R0218 MiniLM 2M four-seed panel"
    )
    reference = dict(panel["shared_high_d_reference"])
    if expected_input_signature(reference["canonical_path"]) != reference:
        raise RuntimeError(
            "R0218's published high-D reference bytes changed; the sixteen cells "
            "would not be poolable"
        )
    if (
        str(panel["high_d_reference_key"]) != REFERENCE_KEY
        or str(panel["high_d_reference_content_sha256"]) != REFERENCE_CONTENT_SHA256
        or dict(panel["anchor_corpus_counts"]) != dict(ANCHOR_CORPUS_COUNTS)
    ):
        raise RuntimeError(
            "R0218's reference identity is not the registered one; STOP — the "
            "sixteen cells are not poolable"
        )
    for seed in panel["seeds"]:
        numerators = panel["cells"][str(seed)]["panel"]["purity_numerators"]
        for key, expected in HI_D_AGREEMENT.items():
            if float(numerators[key]["hi_D_agreement"]) != float(expected):
                raise RuntimeError(
                    f"R0218 seed-{seed} hi-D agreement {key} is not {expected}"
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

    n13_signature = expected_input_signature(R0230_PANEL)
    n13 = prompt_contract.read_sealed(
        R0230_PANEL, label="R0230 sealed thirteen-cell panel"
    )
    if (
        n13.get("schema") != PANEL_SCHEMA_N13
        or n13.get("round_id") != "0230"
        or int(n13.get("n", -1)) != len(R0230_POOLED_SEEDS)
        or str(n13.get("high_d_reference_key")) != REFERENCE_KEY
        or n13.get("gate_registerable_here") is not False
    ):
        raise RuntimeError("R0230's thirteen-cell panel receipt contract changed")
    inputs.append(n13_signature)
    return panel, n13_signature, inputs


def _sealed_gate_inputs() -> list[dict[str, Any]]:
    """The four sealed artifacts the joint criteria and the held-out cells read."""
    signatures = []
    for path, label in (
        (R0223_COMPARISON, "R0223 sealed cuVS comparison"),
        (R0225_GATE, "R0225 sealed n=8 tolerance gate"),
        (R0228_COMPARISON, "R0228 sealed cluster-spill comparison"),
        (R0231_GATE, "R0231 sealed n=13 robust gate"),
        (R0234_GATE, "R0234 sealed n=13 calibrated gate"),
    ):
        prompt_contract.read_sealed(path, label=label)
        signatures.append(expected_input_signature(path))
    return signatures


def prepare_round0250(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0250 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    graph_manifest_signature, graph_manifest, edges = _sealed_graph()
    updates = successful_updates_for_edges(edges)
    if updates > REGISTERED_UPDATE_BOUND:
        raise RuntimeError(
            f"R0250 derived horizon {updates} exceeds the registered bound "
            f"{REGISTERED_UPDATE_BOUND}"
        )
    dose = validate_registered_dose(updates=updates, edge_count=edges)
    prior = _prior_family(graph_manifest_signature, graph_manifest)
    panel, panel_n13_signature, panel_inputs = _sealed_panels()
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
            "R0250 seed-invariant digest does not match R0217's published value"
        )
    pooled_config_hashes = {
        **prior["config_sha256_by_seed"],
        **family["per_seed_config_sha256"],
    }
    if len(set(pooled_config_hashes.values())) != len(POOLED_SEEDS):
        raise RuntimeError(
            "R0250 cell configs collide with the prior thirteen: the sixteen cells "
            "must be sixteen distinct configs sharing one seed-invariant digest"
        )

    predictions = {str(seed): predict_cell_footprint(seed) for seed in SEEDS}
    refused = sorted(
        int(seed) for seed, item in predictions.items() if item["refused_a_priori"]
    )

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0250 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    family_path = os.path.join(preflight, "seed-extension-n16-identity.json")
    atomic_write_new_json(
        family_path,
        prompt_contract.seal({
            "schema": "round0250-seed-extension-n16-config-identity-v1",
            "round_id": ROUND_ID,
            "release_sha": release_sha,
            "sealed_directed_edges": edges,
            "successful_positive_lr_updates": updates,
            "dose_registration": dose,
            "prior_family": prior,
            "family": family,
            "byte_for_byte_reconstruction_of_r0217": reconstructions,
            "pooled_seed_family": list(POOLED_SEEDS),
            "n_pooled": len(POOLED_SEEDS),
            "standing_minimum_n": STANDING_MINIMUM_N,
            "identity_bound_at_n_pooled": IDENTITY_BOUND_AT_N16,
            "pooled_config_sha256_by_seed": pooled_config_hashes,
            "memory_predictions": predictions,
            "refused_a_priori": refused,
            "memory_policy": MEMORY_POLICY,
            "registry_fingerprint": registry_fingerprint(),
            "configs": {str(seed): configs[seed] for seed in SEEDS},
        }),
        immutable=True,
    )

    probe_inputs = [
        expected_input_signature(os.path.join(R0238_PROBE, "truth-k15-ids.i32.npy")),
        expected_input_signature(os.path.join(R0238_PROBE, "probe-query-rows.i64.npy")),
        expected_input_signature(R0238_SUBSTRATE),
        expected_input_signature(R0247_TRUTHCOS),
    ]
    shared_inputs = _dedupe([
        round_signature,
        graph_manifest_signature,
        substrate_signature,
        graph_signature,
        provenance_signature,
        expected_input_signature(smoke_path),
        expected_input_signature(family_path),
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
            "handler_module": "experiments.round0250_nodes",
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
            "train_receipt": {
                "kind": "file",
                "canonical_path": os.path.join(output, "train-receipt.json"),
            },
        })
    if not train_nodes:
        raise RuntimeError("R0250 refused every cell a priori; nothing to run")

    panel_node = "score_minilm_mixed_2m_panel_n16"
    panel_output = os.path.join(artifacts, PANEL_CAPABILITY_N16)
    jobs.append({
        "id": panel_node,
        "action": PANEL_ACTION,
        "handler_module": "experiments.round0250_nodes",
        "handler_callable": "run_job",
        "deps": list(train_nodes),
        "outputs": [panel_output],
        "done_marker": os.path.join(artifacts, f"{panel_node}.done.json"),
        "expected_inputs": _dedupe([*shared_inputs, *panel_inputs]),
        "p90_wall_s": PANEL_P90_WALL_S,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": graph_manifest_signature,
        "panel_evidence": R0218_PANEL,
        "panel_n13_signature": panel_n13_signature,
        "prior_model_sha256_by_seed": prior["model_sha256_by_seed"],
        "cells": panel_cells,
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
            "cpu_heavy": False,
        },
    })
    p90[panel_node] = PANEL_P90_WALL_S

    gate_node = "register_calibrated_robust_floors_n16"
    jobs.append({
        "id": gate_node,
        "action": GATE_ACTION,
        "handler_module": "experiments.round0250_nodes",
        "handler_callable": "run_job",
        "deps": [panel_node],
        "outputs": [os.path.join(artifacts, GATE_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{gate_node}.done.json"),
        "expected_inputs": _dedupe([*shared_inputs, *gate_inputs]),
        "p90_wall_s": GATE_P90_WALL_S,
        "panel_n16": {
            "kind": "file",
            "canonical_path": os.path.join(panel_output, "seed-family-panel-n16.json"),
        },
        "r0223_comparison": gate_inputs[0],
        "r0225_gate": gate_inputs[1],
        "r0228_comparison": gate_inputs[2],
        "r0231_gate": gate_inputs[3],
        "r0234_gate": gate_inputs[4],
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
            "cpu_heavy": True,
        },
    })
    p90[gate_node] = GATE_P90_WALL_S

    blocksize_node = "blocksize_0250"
    jobs.append({
        "id": blocksize_node,
        "action": BLOCKSIZE_ACTION,
        "handler_module": "experiments.round0250_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, BLOCKSIZE_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{blocksize_node}.done.json"),
        "expected_inputs": _dedupe([*shared_inputs, *probe_inputs]),
        "p90_wall_s": BLOCKSIZE_P90_WALL_S,
        "truth_ids": probe_inputs[0],
        "probe_query_rows": probe_inputs[1],
        "substrate_array": probe_inputs[2],
        "r0247_truthcos": probe_inputs[3],
        "truth_probe_rows": TRUTH_PROBE_ROWS,
        "graph_k": PROBE_GRAPH_K,
        "substrate_rows": SUBSTRATE_ROWS,
        "substrate_dimension": SUBSTRATE_DIMENSION,
        "arm_rows": BLOCK_ARM_ROWS,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
            "cpu_heavy": True,
        },
    })
    p90[blocksize_node] = BLOCKSIZE_P90_WALL_S

    trainloop_node = "trainloop_0250"
    jobs.append({
        "id": trainloop_node,
        "action": TRAINLOOP_ACTION,
        "handler_module": "experiments.round0250_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, TRAINER_LOOP_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{trainloop_node}.done.json"),
        "expected_inputs": shared_inputs,
        "p90_wall_s": TRAINLOOP_P90_WALL_S,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": graph_manifest_signature,
        "short_horizon_updates": SHORT_HORIZON_UPDATES,
        "node_policy": {
            "gpu_required": True,
            "training_performed": True,
            "cpu_heavy": False,
        },
    })
    p90[trainloop_node] = TRAINLOOP_P90_WALL_S

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
        "schema": "round0250-trainer-loops-and-n16-gate-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            GRAPH_CAPABILITY,
            PANEL_CAPABILITY,
            PANEL_CAPABILITY_N13,
            "minilm-mixed-2m-cuvs-graph-map-comparison-v1",
            "minilm-mixed-2m-tolerance-gates-n8-v1",
            "minilm-mixed-2m-cluster-spill-graph-map-comparison-v1",
            "minilm-mixed-2m-robust-floors-n13-v1",
            "minilm-mixed-2m-calibrated-robust-floors-n13-v1",
            "minilm-mixed-100000k-uniform-probe-k15-truth-cos-f64-v1",
            *(r0217_capability_for_seed(seed) for seed in (42, 43, 44, 45)),
            *(r0221_capability_for_seed(seed) for seed in (46, 47, 48, 49)),
            *(r0230_capability_for_seed(seed) for seed in (50, 51, 52, 53, 54)),
        ],
        "capabilities_produced": [
            TRAINER_LOOP_CAPABILITY,
            BLOCKSIZE_CAPABILITY,
            *CAPABILITIES,
            PANEL_CAPABILITY_N16,
            GATE_CAPABILITY,
        ],
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": p90,
        "scientific_contract": {
            "question_a": (
                "what is the training path's widest cooperative-abort-read gap and "
                "its allocation slope, measured on a short rung, and does it satisfy "
                "the registered max_poll_spacing ceiling of "
                "2.5109531834854018 s -- the ceiling being R0244's sealed budget "
                "headroom over R0244's sealed measured slope? And, given the "
                "measured per-row cost of truthcos_0247's gather, does its "
                "unregistered block size need registering?"
            ),
            "question_b": (
                "does extending the exact-graph MiniLM 2M family from thirteen "
                "seeds to the standing minimum sixteen, under R0217's treatment "
                "with the seed as the only free variable, and re-deriving the "
                "calibrated robust multiplier at that n, produce a gate that "
                "delivers 95% coverage, that a defining cell can fail, and whose "
                "JOINT criteria with R0225/R0231/R0234 a map must clear?"
            ),
            "population": "sealed R0216 2,000,000-row mixed MiniLM substrate",
            "graph": (
                "sealed R0216 exact k15 fuzzy graph (recall 1.000000, 0 "
                "zero-degree rows)"
            ),
            "sealed_directed_edges": edges,
            "hidden_dimension": HIDDEN_DIMENSION,
            "input_dimension": DIMENSION,
            "precision": USE_AMP,
            "seeds": list(SEEDS),
            "cells": len(SEEDS),
            "pooled_seed_family": list(POOLED_SEEDS),
            "n_pooled": len(POOLED_SEEDS),
            "standing_minimum_n": STANDING_MINIMUM_N,
            "identity_bound_at_n_pooled": IDENTITY_BOUND_AT_N16,
            "identity_bound_note": (
                "max|x - xbar|/s <= (n-1)/sqrt(n) = 3.75 at n = 16, against "
                "3.3282011773513750 at n = 13. It is the operative test for the "
                "mean - k*s families; for the registered median - k*MAD_n family "
                "R0234's rank-slack bound is +inf and a defining cell can fail at "
                "any multiplier."
            ),
            "treatment_source_round": "0217",
            "treatment_template_seed": TEMPLATE_SEED,
            "capabilities_by_seed": {
                str(seed): capability_for_seed(seed) for seed in SEEDS
            },
            "family_seed_invariant_sha256": family["seed_invariant_sha256"],
            "r0217_published_seed_invariant_sha256": R0217_SEED_INVARIANT_SHA256,
            "per_seed_config_sha256": family["per_seed_config_sha256"],
            "masked_config_identity": family["masked_config_identity"],
            "byte_for_byte_reconstruction_of_r0217": reconstructions,
            "pooled_config_sha256_by_seed": pooled_config_hashes,
            "prior_model_sha256_by_seed": prior["model_sha256_by_seed"],
            "only_treatment_between_cells": "the training seed",
            "gate_registerable_here": GATE_REGISTERABLE_HERE,
            "gate_registered_by_the_final_node": True,
            "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
            "successful_positive_lr_updates": updates,
            "registered_successful_positive_lr_updates": REGISTERED_SUCCESSFUL_UPDATES,
            "achieved_positive_draws_per_edge": dose[
                "achieved_positive_draws_per_edge"
            ],
            "registered_achieved_positive_draws_per_edge": (
                REGISTERED_ACHIEVED_DRAWS_PER_EDGE
            ),
            "dose_quantum_draws_per_edge": dose["dose_quantum_draws_per_edge"],
            "dose_rule": dose["dose_rule"],
            "registered_update_bound": REGISTERED_UPDATE_BOUND,
            "host_rss_limit_gib": HOST_RSS_LIMIT_GIB,
            "device_budget_bytes": DEVICE_BUDGET_BYTES,
            "host_anonymous_budget_bytes": HOST_ANON_BUDGET_BYTES,
            "swap_growth_abort_bytes": SWAP_GROWTH_ABORT_BYTES,
            "memory_policy": MEMORY_POLICY,
            "memory_predictions": predictions,
            "refused_a_priori": refused,
            "full_population_transform_rows": ROWS,
            "panel_config_source": "accepted R0113 panel_config()",
            "panel_metrics": list(PANEL_METRICS),
            "gated_metrics": list(GATED_METRICS),
            "corpus_ffr_slices": list(CORPUS_SLUGS),
            "shared_high_d_reference": dict(panel["shared_high_d_reference"]),
            "reference_source_round": "0218",
            "reference_must_be_byte_identical_to_r0218": True,
            "reference_key": REFERENCE_KEY,
            "reference_content_sha256": REFERENCE_CONTENT_SHA256,
            "hi_d_agreement_required": dict(HI_D_AGREEMENT),
            "anchor_corpus_counts": dict(ANCHOR_CORPUS_COUNTS),
            "prior_cells_read_not_rescored": dict(POOLED_CELL_SOURCES),
            "density_v2_status": DENSITY_V2_STATUS,
            "selection_rule": SELECTION_RULE,
            "joint_criteria_rule": JOINT_CRITERIA_RULE,
            "retained_family_sources": [dict(item) for item in RETAINED_FAMILY_SOURCES],
            "held_out_cells_expected": N_HELD_OUT,
            "exact_cells_expected": N_EXACT,
            "identity_bound_at_gate_n": IDENTITY_BOUND_AT_N,
            "trainer_loop_short_horizon_updates": SHORT_HORIZON_UPDATES,
            "trainer_loop_projection_target_hours": PROJECTION_TARGET_HOURS,
            "trainer_loop_per_batch_hook": dict(PER_BATCH_HOOK),
            "block_sizes_probed": list(BLOCK_SIZES_PROBED),
            "block_arm_rows": BLOCK_ARM_ROWS,
            "per_row_cost_stability_limit": PER_ROW_COST_STABILITY_LIMIT,
            "adaptive_target_fraction": ADAPTIVE_TARGET_FRACTION,
            "registry_fingerprint": registry_fingerprint(),
            "registry_mutated": False,
            "guard_modules_edited": False,
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
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0250(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
