#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only R0219 MiniLM gate registration.

Zero GPU: `gpu_hours_cap` is `0.0` and the single node declares
`gpu_required: false`. The script binds R0218's terminal queue, terminal receipt
and sealed panel artifact by hash, and records the exact upstream review state at
preparation time rather than asserting one — R0218's independent review is
post-hoc and may not exist yet, and the protocol lets an already-registered
experiment run while blocking only the downstream *claim*.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0218_minilm_2m_panel import (
    CAPABILITY as PANEL_CAPABILITY,
    EVALUATION_SCHEMA as PANEL_SCHEMA,
    SEEDS,
)
from basemap.round0219_minilm_2m_gate_registration import (
    CAPABILITY,
    EXCLUDED_METRICS,
    FORMULA,
    GATE_METRICS,
    MULTIPLIER,
    PRECISION_NOTE,
    ROUND_ID,
    SD_DDOF,
)
from experiments.round0219_nodes import ACTION
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0219"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0219-2026-08-08.md")
R0218_QUEUE = "/data/latent-basemap/runs/round-0218/queue/queue.json"
R0218_TERMINAL = "/data/latent-basemap/runs/round-0218/queue/runner-terminal.json"
R0218_PANEL = (
    "/data/latent-basemap/runs/round-0218/queue/artifacts/"
    f"{PANEL_CAPABILITY}/seed-family-panel.json"
)


def _issued_round(release_sha: str) -> tuple[dict[str, Any], list[str]]:
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
        raise RuntimeError("R0219 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if reviews != ["0218"]:
        raise RuntimeError("R0219 must declare required_reviews: ['0218']")
    return expected_input_signature(ROUND_FILE), reviews


def _upstream_review_state() -> dict[str, Any]:
    """Record, rather than assume, the state of R0218's independent review."""
    reviews = []
    for path in sorted(glob.glob(os.path.join(LAB_ROOT, "review-0218-*.md"))):
        frontmatter = _frontmatter(path)
        reviews.append({
            "file": os.path.basename(path),
            "status": frontmatter.get("status"),
            "sha256": expected_input_signature(path)["sha256"],
        })
    accepted = [item for item in reviews if item["status"] == "accepted"]
    return {
        "round_id": "0218",
        "reviews_present": reviews,
        "accepted_reviews": len(accepted),
        "gate_release_contingent_on_review_0218": not accepted,
        "note": (
            "The slim protocol allows an already-registered experiment to run "
            "before its upstream review lands; it blocks the downstream claim. "
            "If no accepted review-0218 exists here, the floors this round "
            "computes are registered but not released until one does."
        ),
    }


def _terminal_r0218() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    queue_signature = expected_input_signature(R0218_QUEUE)
    terminal_signature = expected_input_signature(R0218_TERMINAL)
    with open(R0218_TERMINAL, encoding="utf-8") as handle:
        terminal = json.load(handle)
    if (
        terminal.get("round_id") != "0218"
        or terminal.get("verdict") != "succeeded"
        or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
        or terminal.get("queue_manifest_sha256_at_finish") != queue_signature["sha256"]
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("boundary_problems") != []
    ):
        raise RuntimeError("R0218 terminal premise changed")
    panel_signature = expected_input_signature(R0218_PANEL)
    panel = prompt_contract.read_sealed(
        R0218_PANEL, label="R0218 MiniLM 2M four-seed panel"
    )
    if (
        panel.get("schema") != PANEL_SCHEMA
        or panel.get("round_id") != "0218"
        or panel.get("capabilities") != [PANEL_CAPABILITY]
        or panel.get("seeds") != list(SEEDS)
        or panel.get("gate_registered") is not False
        or panel.get("gate_registerable_here") is not False
    ):
        raise RuntimeError("R0218 panel artifact contract changed")
    results = sorted(glob.glob(os.path.join(LAB_ROOT, "result-0218-*.md")))
    complete = [
        path for path in results if _frontmatter(path).get("status") == "complete"
    ]
    if len(complete) != 1:
        raise RuntimeError("R0219 requires exactly one complete Result 0218")
    return panel_signature, [
        queue_signature,
        terminal_signature,
        expected_input_signature(complete[0]),
    ]


def prepare_round0219(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0219 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    panel_signature, lineage = _terminal_r0218()
    review_state = _upstream_review_state()
    expected_inputs = _dedupe([round_signature, *lineage, panel_signature])

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0219 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    node = "register_minilm_quality_gates"
    job = {
        "id": node,
        "action": ACTION,
        "handler_module": "experiments.round0219_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, CAPABILITY)],
        "done_marker": os.path.join(artifacts, "minilm-quality-gates.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 30.0,
        "panel_evidence": panel_signature["canonical_path"],
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
            "cpu_heavy": False,
        },
    }
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.0,
        execution_authority="autonomous-cpu",
        gpu=False,
    )
    queue.update({
        "schema": "round0219-minilm-mixed-2m-quality-gate-registration-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [PANEL_CAPABILITY],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "question": (
                "what binding quality floors follow from the R0218 four-cell "
                "MiniLM mixed-2M panel?"
            ),
            "formula": FORMULA,
            "sample_standard_deviation_ddof": SD_DDOF,
            "multiplier": MULTIPLIER,
            "n": len(SEEDS),
            "precision_note": PRECISION_NOTE,
            "gate_metrics": list(GATE_METRICS),
            "excluded_metrics": dict(EXCLUDED_METRICS),
            "density_v2_role": "diagnostic-only, transcribed",
            "per_corpus_ffr_role": "descriptive; not a registered floor",
            "higher_is_better": True,
            "r0161_prompted_floors_unchanged": True,
            "r0193_mixed_english_floors_unchanged": True,
            "upstream_review_state": review_state,
            "no_training": True,
            "no_evaluation": True,
            "gpu_hours": 0.0,
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
        "queue_manifest": prepare_round0219(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
