"""Build the R0256 queue — one CPU node that repairs R0255's gate evidence.

No training, no GPU, no new cells. The node reads R0255's sealed gate artifact,
R0255's sealed twenty-nine-cell panel and the two sealed held-out families, and
re-derives everything R0255 got wrong *about* its floors while requiring every
floor and band to come back bitwise identical.

`--preflight-only` on the runner checks the four bound inputs exist at their
declared sizes and hashes before anything runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, ensure_data_directory
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0234_calibrated_floors import (
    DESCRIPTIVE_METRICS,
    GATED_METRICS,
    TOLERANCE_CONFIDENCE,
    TOLERANCE_CONTENT,
)
from basemap.round0255_gate_n29 import (
    GATE_CAPABILITY as R0255_GATE_CAPABILITY,
    IDENTITY_BOUND_AT_N,
    N_EXACT,
    N_HELD_OUT,
    OWNER_RULING,
    OWNER_RULING_ESTIMATOR,
)
from basemap.round0255_panel_n29 import PANEL_CAPABILITY_N29
from basemap.round0256_gate_repair import (
    GATE_CAPABILITY,
    REGISTERED_ESTIMATOR_NAMES,
    ROUND_ID,
    SUPERSEDES_CAPABILITY,
)

from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
from experiments.round0256_nodes import REPAIR_ACTION


ROUND_ROOT = "/data/latent-basemap/runs/round-0256"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0256-2026-08-11.md")

R0223_COMPARISON = (
    "/data/latent-basemap/runs/round-0223/queue-correction-3/artifacts/"
    "minilm-mixed-2m-cuvs-graph-map-comparison-v1/cuvs-graph-map-comparison.json"
)
R0228_COMPARISON = (
    "/data/latent-basemap/runs/round-0228/queue/artifacts/"
    "minilm-mixed-2m-cluster-spill-graph-map-comparison-v1/"
    "cluster-spill-graph-map-comparison.json"
)
R0255_PANEL = (
    f"/data/latent-basemap/runs/round-0255/queue/artifacts/{PANEL_CAPABILITY_N29}/"
    "seed-family-panel-n29.json"
)
R0255_GATE = (
    f"/data/latent-basemap/runs/round-0255/queue/artifacts/{R0255_GATE_CAPABILITY}/"
    "minilm-calibrated-madn-floors-n29.json"
)

#: CPU-only. Review-0255 measured the restricted calibration at 11.5 s; the rest
#: of the node is reading four sealed JSONs and fitting order statistics.
REPAIR_P90_WALL_S = 90.0
GPU_HOURS_CAP = 0.05


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _issued_round(release_sha: str) -> tuple[dict, list[str]]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    import subprocess

    descendant = subprocess.run(
        [
            "git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor",
            base_commit, release_sha,
        ],
        check=False,
    ).returncode == 0
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or not descendant
    ):
        raise RuntimeError("R0256 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0256 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _upstream_review_state(required: list[str]) -> dict:
    import glob

    state: dict = {}
    contingent: list[str] = []
    for round_id in required:
        reviews = []
        for path in sorted(
            glob.glob(os.path.join(LAB_ROOT, f"review-{round_id}-*.md"))
        ):
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
            "R0255's review is `partial` and blocked four claims; this round is "
            "the response to it, so the repaired evidence is "
            "registered-and-contingent until R0256 itself carries an accepted "
            "review."
        ),
    }


def _sealed_inputs() -> list[dict]:
    signatures = []
    for path, label in (
        (R0223_COMPARISON, "R0223 sealed cuVS comparison"),
        (R0228_COMPARISON, "R0228 sealed cluster-spill comparison"),
        (R0255_PANEL, "R0255 sealed twenty-nine-cell panel"),
        (R0255_GATE, "R0255 sealed n=29 calibrated MAD_n gate"),
    ):
        prompt_contract.read_sealed(path, label=label)
        signatures.append(expected_input_signature(path))
    return signatures


def prepare_round0256(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0256 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    review_state = _upstream_review_state(list(required_reviews))
    inputs = _sealed_inputs()
    r0223, r0228, panel, gate = inputs

    ensure_data_directory(queue_root, label="R0256 queue root")
    artifacts = os.path.join(queue_root, "artifacts")
    ensure_data_directory(artifacts, label="R0256 artifacts")

    node = REPAIR_ACTION
    jobs = [{
        "id": node,
        "action": REPAIR_ACTION,
        "handler_module": "experiments.round0256_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, GATE_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{node}.done.json"),
        "expected_inputs": _dedupe([round_signature, *inputs]),
        "p90_wall_s": REPAIR_P90_WALL_S,
        "r0223_comparison": r0223,
        "r0228_comparison": r0228,
        "panel_n29": panel,
        "r0255_gate": gate,
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
            "cpu_heavy": True,
        },
    }]

    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-cpu",
        gpu=False,
    )
    queue.update({
        "schema": "round0256-n29-gate-evidence-repair-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-analysis",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            PANEL_CAPABILITY_N29,
            R0255_GATE_CAPABILITY,
            "minilm-mixed-2m-cuvs-graph-map-comparison-v1",
            "minilm-mixed-2m-cluster-spill-graph-map-comparison-v1",
        ],
        "capabilities_produced": [GATE_CAPABILITY],
        "jobs": jobs,
        "p90_wall_s": {node: REPAIR_P90_WALL_S, "total": REPAIR_P90_WALL_S},
        "registered": {
            "what_this_round_is": (
                "a repair of the EVIDENCE published beside R0255's registered "
                "n = 29 MAD_n floors. It changes no floor value and trains no cell."
            ),
            "owner_ruling": OWNER_RULING,
            "owner_ruling_estimator": OWNER_RULING_ESTIMATOR,
            "estimators_calibrated": list(REGISTERED_ESTIMATOR_NAMES),
            "n_exact": N_EXACT,
            "n_held_out": N_HELD_OUT,
            "identity_bound_at_n": IDENTITY_BOUND_AT_N,
            "tolerance_content": TOLERANCE_CONTENT,
            "tolerance_confidence": TOLERANCE_CONFIDENCE,
            "gated_metrics": list(GATED_METRICS),
            "descriptive_metrics": list(DESCRIPTIVE_METRICS),
            "supersedes_capability": SUPERSEDES_CAPABILITY,
            "acceptance_rule": (
                "every execution check holds, including: no floor and no band moves "
                "bitwise against R0255's registered values; the restricted "
                "calibration reproduces R0255's multipliers, power and false-fail "
                "rates bitwise; every arm of the independence control passes its "
                "perturbation into the fit; every planted builder that selects a "
                "held-out cell makes the arms report false while the honest builder "
                "still reports independent; every planted registry defect is "
                "refused; and the two-sided band power is strictly below the "
                "one-sided floor power."
            ),
            "what_would_falsify_the_independence_claim": (
                "any floor or band moving under arm 1 or arm 2. If that happens the "
                "round publishes it as a finding about the gate, not as a defect in "
                "the control. The claim is being MEASURED here, not assumed."
            ),
            "gpu_hours_expected": 0.0,
            "gpu_required_anywhere": False,
        },
    })
    manifest_path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(manifest_path, queue, immutable=True)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="prepare the R0256 queue")
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    args = parser.parse_args(argv)
    path = prepare_round0256(
        release_sha=args.release_sha, queue_root=args.queue_root
    )
    print(json.dumps({"queue": path, "sha256": file_sha256(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
