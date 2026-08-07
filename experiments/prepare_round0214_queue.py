#!/usr/bin/env python3
"""Prepare, but never launch, the R0214 seed-43 diverse panel readout."""
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

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0105_search import GROUPS
from basemap.round0108_evaluation import IN_MIX_LANGUAGES, POLISH
from basemap.round0160_prompted_seed_family import CAPABILITY as FAMILY_CAPABILITY
from basemap.round0161_prompted_gate_registration import CAPABILITY as GATE_CAPABILITY
from basemap.round0168_prompted_diverse_staging import (
    CAPABILITY as STAGING_CAPABILITY,
)
from basemap.round0209_prompted_diverse_graph import CAPABILITY as GRAPH_CAPABILITY
from basemap.round0210_prompted_diverse_low_dose import successful_updates_for_edges
from basemap.round0211_prompted_diverse_panel import (
    PACK_CAPABILITY,
    PACK_CORPUS_ROWS,
    PACK_QUERY_ROWS,
    PACK_ROWS,
    PACK_SCHEMA,
    ROWS,
)
from basemap.round0214_seed43_panel import CAPABILITY, ROUND_ID, SEED
from basemap.round0212_prompted_diverse_seed43 import (
    CAPABILITY as MODEL_CAPABILITY,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0169_queue import (
    FAMILY_PATH,
    GATES_PATH,
    GROUP_IDS_PATH,
    RAW_R0132_OOD,
    STAGING_MANIFEST,
    _accepted_bundle,
    _read_sealed,
)
from experiments.prepare_round0210_queue import GRAPH_MANIFEST


LANGUAGES = (*IN_MIX_LANGUAGES, POLISH)
ROUND_ROOT = "/data/latent-basemap/runs/round-0214"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0214-2026-08-07.md")
R0168_REVIEW = os.path.join(LAB_ROOT, "review-0168-2026-08-03-01.md")
OOD_PACK_PATH = (
    "/data/latent-basemap/runs/round-0208/queue/artifacts/"
    "jina-prompted-u12-ood-probe-pack-v2/probe-pack.json"
)
# R0210's first queue reached a registered terminal `failed` state before any
# CUDA training child started and published no model; the map was trained by its
# dated `queue-correction-1` relaunch.
TRAIN_OUTPUT = (
    "/data/latent-basemap/runs/round-0212/queue-cap-corrected/artifacts/"
    "seed43-low-dose-train"
)
R0211_EVALUATION = (
    "/data/latent-basemap/runs/round-0211/queue-correction-2/artifacts/"
    "jina-prompted-diverse-u12-evaluation-panel-v1/scale-evaluation.json"
)
R0173_PACK_ROOT = "/data/latent-basemap/runs/round-0173/queue/artifacts"
#: Populated by `_reproducing_review_bundle` when an accepted review's bound
#: result prose no longer matches the live file. Stamped into the queue so the
#: result and its review must state the divergence rather than inherit it.
RESULT_PROSE_DIVERGENCE: list[str] = []
GPU_HOURS_CAP = 2.5


def _reproducing_review_bundle(round_id: str) -> list[dict[str, Any]]:
    """Bind the accepted review of `round_id` whose document hashes reproduce.

    `_accepted_bundle` requires exactly one accepted review per round and that
    both its `round_sha256` and `result_sha256` match the live files. Both
    assumptions can fail legitimately: the runner watcher re-published
    result-0210 to apply that review's *own* corrected GPU charge, which
    stranded the review's binding, and recovery attempts can leave more than one
    accepted review behind.

    The two bindings are not equally load-bearing, so they are treated
    differently:

    * **round file** — append-only pre-registration. A mismatch means the
      registration moved under the review. Never accepted; raises.
    * **result file** — prose. R0214 does not consume prose; it consumes the
      sealed model, production config, and train receipt, and this prepare
      re-verifies each of those by hash and cross-checks the receipt against the
      sealed graph's edge count and dose rule. A mismatch is recorded in
      `RESULT_PROSE_DIVERGENCE`, stamped into the queue's scientific contract,
      and must be stated in this round's result — not silently inherited.
    """
    candidates = sorted(
        glob.glob(os.path.join(LAB_ROOT, f"review-{round_id}-*.md"))
    )
    accepted = [
        path for path in candidates if _frontmatter(path).get("status") == "accepted"
    ]
    if not accepted:
        raise RuntimeError(f"R0214 requires an accepted Review {round_id}; found 0")
    prose_divergence: list[str] = []
    round_mismatch: list[str] = []
    for path in accepted:
        frontmatter = _frontmatter(path)
        if frontmatter.get("round_id") != round_id:
            continue
        issued = expected_input_signature(
            os.path.join(LAB_ROOT, str(frontmatter.get("round") or ""))
        )
        result = expected_input_signature(
            os.path.join(LAB_ROOT, str(frontmatter.get("result") or ""))
        )
        if issued["sha256"] != frontmatter.get("round_sha256"):
            # The issued round is append-only; a mismatch here means the
            # pre-registration itself moved under the review. Never accept that.
            round_mismatch.append(os.path.basename(path))
            continue
        if result["sha256"] != frontmatter.get("result_sha256"):
            # The result *prose* was re-published after this review. That is a
            # reporting divergence, not a scientific one: what R0214 consumes is
            # the sealed model, production config, and train receipt, and this
            # prepare re-verifies every one of those by hash and cross-checks the
            # receipt against the sealed graph's edge count and dose rule. Record
            # the divergence so the result and its review can state it, and keep
            # going rather than blocking the panel on prose bytes.
            prose_divergence.append(
                f"{os.path.basename(path)} binds result sha256 "
                f"{frontmatter.get('result_sha256')} but the live result is "
                f"{result['sha256']}"
            )
        RESULT_PROSE_DIVERGENCE.extend(prose_divergence)
        return [issued, result, expected_input_signature(path)]
    raise RuntimeError(
        f"R0214 rejects every accepted Review {round_id}: "
        f"round-file binding mismatch in {round_mismatch or 'none'}; an issued "
        "round file is append-only and must reproduce exactly"
    )


def _issued_round(release_sha: str) -> dict[str, Any]:
    """Accept the issued round at its base commit or any descendant release.

    The R0173 ancestor test rather than exact equality: an issued round file is
    append-only, so its `base_commit` cannot be rewritten when a setup-class
    correction elsewhere in the campaign advances the shared release.
    """
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor", base_commit, release_sha],
        check=False,
        timeout=10,
    ).returncode == 0
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or not descendant
    ):
        raise RuntimeError("R0214 round is not issued for this release")
    return expected_input_signature(ROUND_FILE)


def _pack_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    signature = expected_input_signature(OOD_PACK_PATH)
    pack = _read_sealed(signature, label="accepted R0208 OOD pack v2")
    shape = pack.get("shape") or {}
    audit = pack.get("audit") or {}
    if (
        pack.get("schema") != PACK_SCHEMA
        or pack.get("round_id") != "0208"
        or pack.get("capabilities") != [PACK_CAPABILITY]
        or audit.get("passed_after_repair") is not True
        or int(audit.get("source_row_identity_overlaps", -1)) != 0
        or int(shape.get("pack_rows", -1)) != PACK_ROWS
        or int(shape.get("corpus_rows_per_language", -1)) != PACK_CORPUS_ROWS
        or int(shape.get("query_rows_per_language", -1)) != PACK_QUERY_ROWS
        or tuple(shape.get("languages") or ()) != LANGUAGES
    ):
        raise RuntimeError("R0214 accepted R0208 OOD pack v2 changed")
    signatures: list[dict[str, Any]] = [signature]
    language_outputs: dict[str, str] = {}
    for language in LANGUAGES:
        entry = (pack.get("languages") or {})[language]
        for value in (entry["source_arrays"] or {}).values():
            observed = expected_input_signature(str(value["canonical_path"]))
            if observed != dict(value):
                raise RuntimeError(f"R0214 {language} pack v2 source bytes changed")
            signatures.append(observed)
        for split in ("corpus", "queries"):
            for key in ("ordinals", "source_rows"):
                value = entry["retained"][split][key]
                observed = expected_input_signature(str(value["canonical_path"]))
                if observed != dict(value):
                    raise RuntimeError(
                        f"R0214 {language} {split} retained {key} bytes changed"
                    )
                signatures.append(observed)
        receipt = dict(entry["receipt"])
        observed = expected_input_signature(str(receipt["canonical_path"]))
        if observed != receipt:
            raise RuntimeError(f"R0214 {language} R0173 receipt bytes changed")
        signatures.append(observed)
        language_outputs[language] = os.path.dirname(str(receipt["canonical_path"]))
        if language_outputs[language] != os.path.join(
            R0173_PACK_ROOT, f"prompted-{language}"
        ):
            raise RuntimeError(f"R0214 {language} probe path left the R0173 pack")
    return signature, signatures, language_outputs


def _matched_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    family_signature = expected_input_signature(FAMILY_PATH)
    family = _read_sealed(family_signature, label="accepted R0160 family")
    gate_signature = expected_input_signature(GATES_PATH)
    gates = _read_sealed(gate_signature, label="accepted R0161 gates")
    if (
        family.get("capability") != FAMILY_CAPABILITY
        or gates.get("capability") != GATE_CAPABILITY
        or gates.get("registered") is not True
        or gates.get("family_evidence") != family_signature
    ):
        raise RuntimeError("R0214 prompted family/gate lineage changed")
    seed42 = family["cells"]["seed42"]
    accepted_score = _read_sealed(seed42["native_score"], label="accepted seed42 score")
    accepted_query = _read_sealed(
        accepted_score["query_reserve"], label="accepted R0113 query reserve"
    )
    accepted_selection = _read_sealed(
        accepted_score["query_selection"], label="accepted seed42 query selection"
    )
    inputs = [
        family_signature,
        gate_signature,
        dict(family["lineage"]["assembly"]),
        dict(family["lineage"]["document_compact"]),
        dict(family["shared_prompted_reference"]),
        *[dict(value) for value in family["centroids"].values()],
        dict(seed42["native_score"]),
        dict(accepted_score["train_receipt"]),
        dict(accepted_score["combined_query_truth"]),
        dict(accepted_score["query_reserve"]),
        dict(accepted_query["outputs"]["document"]),
        dict(accepted_score["query_selection"]),
        dict(accepted_selection["positions"]),
    ]
    return family_signature, gate_signature, inputs


def prepare_round0211(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0214 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    dependencies = [
        *_accepted_bundle("0132"),
        *_accepted_bundle("0160"),
        *_accepted_bundle("0161"),
        *_accepted_bundle("0168", review_path=R0168_REVIEW),
        *_reproducing_review_bundle("0208"),
        *_reproducing_review_bundle("0209"),
        *_reproducing_review_bundle("0210"),
        *_reproducing_review_bundle("0211"),
        *_reproducing_review_bundle("0212"),
    ]
    staging_signature = expected_input_signature(STAGING_MANIFEST)
    staging = _read_sealed(staging_signature, label="accepted R0168 staging")
    if int(staging.get("rows", -1)) != ROWS:
        raise RuntimeError("R0214 accepted staging contract changed")
    staging_inputs = [
        staging_signature,
        dict(staging["host_fp16"]),
        dict(staging["population"]["mapping"]),
        dict(staging["duplicate_control"]["arrays"]),
    ]
    graph_signature = expected_input_signature(GRAPH_MANIFEST)
    graph = _read_sealed(graph_signature, label="sealed R0209 graph")
    edges = int(graph["directed_edge_count"])
    graph_inputs = [
        graph_signature,
        dict(graph["graph"]),
        dict(graph["high_d_reference"]),
        *[dict(value) for value in (graph["centroids"] or {}).values()],
    ]
    train_receipt_path = os.path.join(TRAIN_OUTPUT, "train-receipt.json")
    train_signature = expected_input_signature(train_receipt_path)
    train = _read_sealed(train_signature, label="sealed R0210 train receipt")
    updates = successful_updates_for_edges(edges)
    if (
        train.get("round_id") != "0210"
        or int(train.get("optimizer_updates", -1)) != updates
        or train.get("graph_manifest") != graph_signature
    ):
        raise RuntimeError("R0214 sealed R0210 train receipt does not match the graph")
    train_inputs = [train_signature, dict(train["model"]), dict(train["production_config"])]

    pack_signature, pack_inputs, language_outputs = _pack_inputs()
    family_signature, gate_signature, matched_inputs = _matched_inputs()
    group_signature = expected_input_signature(GROUP_IDS_PATH)
    group_ids = np.load(GROUP_IDS_PATH, mmap_mode="r", allow_pickle=False)
    if (
        group_ids.shape != (ROWS,)
        or group_ids.dtype != np.uint8
        or set(np.unique(group_ids).tolist()) != set(range(len(GROUPS)))
    ):
        raise RuntimeError("R0214 accepted R0132 group IDs changed")
    raw_signature = expected_input_signature(RAW_R0132_OOD)

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0214 GPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    expected_inputs = _dedupe([
        round_signature,
        *dependencies,
        *staging_inputs,
        *graph_inputs,
        *train_inputs,
        *pack_inputs,
        *matched_inputs,
        group_signature,
        raw_signature,
        expected_input_signature(R0211_EVALUATION),
    ])
    output = os.path.join(artifacts, CAPABILITY)
    job = {
        "id": "evaluate_prompted_diverse_u12_seed43",
        "action": "evaluate_prompted_diverse_u12_seed43",
        "handler_module": "experiments.round0214_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "evaluation.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 6_600.0,
        "staging_manifest": staging_signature,
        "graph_manifest": GRAPH_MANIFEST,
        "train_output": TRAIN_OUTPUT,
        "family_evidence": family_signature,
        "gate_registration": gate_signature,
        "group_ids": group_signature,
        "raw_r0132_ood": raw_signature,
        "ood_pack": pack_signature,
        "r0211_evaluation": expected_input_signature(R0211_EVALUATION),
        "language_outputs": language_outputs,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
            "cpu_heavy": True,
        },
    }
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
        "schema": "round0214-prompted-diverse-u12-seed43-panel-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-evaluation",
        "required_reviews": ["0132", "0160", "0161", "0168", "0208", "0209", "0210", "0211", "0212"],
        "capability_dependencies": [
            STAGING_CAPABILITY,
            GRAPH_CAPABILITY,
            MODEL_CAPABILITY,
            PACK_CAPABILITY,
            FAMILY_CAPABILITY,
            GATE_CAPABILITY,
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "p90_gpu_seconds": {
            "evaluate_prompted_diverse_u12_seed43": 6_600.0,
            "total": 6_600.0,
        },
        "scientific_contract": {
            "primary_registered_readout": "descriptive two-cell measurement",
            "decisive_quality_gate_registered": False,
            "seed": SEED,
            "paired_seed": 42,
            "gate_registerable_here": False,
            "retention_reference": "accepted prompted 2M seed-42 matched panel",
            "all_quality_cells_role": "descriptive, not decisive",
            "density_v2_role": "diagnostic-only, transcribed",
            "projection_ffr_role": "diagnostic-only",
            "per_language_ffr_reported": True,
            "ood_reserve": "sealed R0208 pack v2 retained ordinals only",
            "ood_pack_rows": PACK_ROWS,
            "held_out_language": POLISH,
            "atlas_quality_claim_available": False,
            "training_performed": False,
            "production_or_publishing": False,
            "upstream_result_prose_divergence": list(RESULT_PROSE_DIVERGENCE),
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
        "queue_manifest": prepare_round0211(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
