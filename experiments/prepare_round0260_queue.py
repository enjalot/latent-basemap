"""Build the R0260 queue — register the one-sided purity floors, then re-adjudicate.

Two CPU nodes and one dependency edge, and the edge is the point. The
registration node is bound to the sealed twenty-nine-cell 2M family panel and the
sealed R0256 gate, and to **nothing from R0257**; the re-adjudication node depends
on it, binds its output by hash at read time, and refuses to run unless the
registration was sealed first. The runner enforces the ordering and both
directions are recorded in receipts rather than in prose.

No training, no GPU, no new cells, no new seeds.

`--preflight-only` on the runner checks every bound input exists at its declared
size and hash before anything runs.
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
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0234_calibrated_floors import (
    DESCRIPTIVE_METRICS,
    GATED_METRICS,
    TOLERANCE_CONFIDENCE,
    TOLERANCE_CONTENT,
)
from basemap.round0254_dispatch import (
    SCOPE_MODULES,
    assert_derived_entries_install,
    dispatch_census,
    entry_tuples,
    gate_census,
    scope_residual,
)
from basemap.round0255_gate_n29 import IDENTITY_BOUND_AT_N, N_EXACT
from basemap.round0255_panel_n29 import PANEL_CAPABILITY_N29
from basemap.round0256_gate_repair import (
    GATE_CAPABILITY as R0256_GATE_CAPABILITY,
    REGISTERED_ESTIMATOR_NAMES,
)
from basemap.round0257_rung_contract import rung_cell_ids
from basemap.round0260_one_sided_purity import (
    GATE_CAPABILITY,
    NO_TUNING_STATEMENT,
    OWNER_RULING,
    OWNER_RULING_DATE,
    OWNER_RULING_SECTION,
    ROUND_ID,
    SUPERSEDED_PURITY_CRITERIA_IN,
    WHY_ONE_SIDED,
)

from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
from experiments.round0260_nodes import (
    READJUDICATE_ACTION,
    READJUDICATION_CAPABILITY,
    REGISTER_ACTION,
    REGISTRATION_ARTIFACT,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0260"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0260-2026-08-12.md")

R0255_PANEL = (
    f"/data/latent-basemap/runs/round-0255/queue/artifacts/{PANEL_CAPABILITY_N29}/"
    "seed-family-panel-n29.json"
)
R0256_GATE = (
    "/data/latent-basemap/runs/round-0256/queue-correction-1/artifacts/"
    f"{R0256_GATE_CAPABILITY}/minilm-calibrated-madn-floors-n29-repaired.json"
)
R0257_PANEL = (
    "/data/latent-basemap/runs/round-0257/queue-correction-2/artifacts/"
    "minilm-mixed-6250k-ladder-panel-v1/panel_6250k-ladder-panel.json"
)
R0257_VERDICT = (
    "/data/latent-basemap/runs/round-0257/queue-correction-2/artifacts/"
    "minilm-mixed-6250k-rung-gate-verdict-v1/judge_6250k-rung-verdict.json"
)

#: CPU-only. The restricted n=29 calibration cost R0256 about 11.5 s; everything
#: else is reading four sealed JSONs and fitting order statistics over 29 cells.
REGISTER_P90_WALL_S = 90.0
READJUDICATE_P90_WALL_S = 45.0
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
        raise RuntimeError("R0260 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0260 round must declare its required reviews")
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
            "This round executes a direct owner ruling issued in response to "
            "review-0257-01, so the re-registration is registered-and-contingent "
            "until R0260 itself carries an accepted review."
        ),
    }


def _sealed(path: str, label: str) -> dict:
    prompt_contract.read_sealed(path, label=label)
    return expected_input_signature(path)


def prepare_round0260(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0260 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    review_state = _upstream_review_state(list(required_reviews))

    panel_n29 = _sealed(R0255_PANEL, "R0255 sealed twenty-nine-cell panel")
    r0256_gate = _sealed(R0256_GATE, "R0256 sealed repaired n=29 gate")
    r0257_panel = _sealed(R0257_PANEL, "R0257 sealed rung panel")
    r0257_verdict = _sealed(R0257_VERDICT, "R0257 sealed rung verdict")

    census = dispatch_census()
    guard = assert_derived_entries_install(SCOPE_MODULES, census)
    gates = gate_census(entry_tuples(guard["derived"]))
    residual = scope_residual(census, SCOPE_MODULES)

    ensure_data_directory(ROUND_ROOT, label="R0260 round root")
    queue_root = create_fresh_directory(queue_root, label="R0260 queue")
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts"), label="R0260 artifacts"
    )

    register_node = REGISTER_ACTION
    register_output = os.path.join(artifacts, GATE_CAPABILITY)
    register_done = os.path.join(artifacts, f"{register_node}.done.json")

    #: The registration node's declared inputs: the sealed 2M family and the
    #: sealed R0256 gate. NOTHING from R0257. The node reads this list back out
    #: of its own job and refuses if a rung artifact appears in it.
    register_inputs = _dedupe([round_signature, panel_n29, r0256_gate])

    readjudicate_node = READJUDICATE_ACTION
    jobs = [
        {
            "id": register_node,
            "action": REGISTER_ACTION,
            "handler_module": "experiments.round0260_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [register_output],
            "done_marker": register_done,
            "expected_inputs": register_inputs,
            "p90_wall_s": REGISTER_P90_WALL_S,
            "panel_n29": panel_n29,
            "r0256_gate": r0256_gate,
            "scope_modules": list(SCOPE_MODULES),
            "upstream_review_state": review_state,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
                "cpu_heavy": True,
            },
        },
        {
            "id": readjudicate_node,
            "action": READJUDICATE_ACTION,
            "handler_module": "experiments.round0260_nodes",
            "handler_callable": "run_job",
            "deps": [register_node],
            "outputs": [os.path.join(artifacts, READJUDICATION_CAPABILITY)],
            "done_marker": os.path.join(artifacts, f"{readjudicate_node}.done.json"),
            "expected_inputs": _dedupe([round_signature, r0257_panel, r0257_verdict]),
            "p90_wall_s": READJUDICATE_P90_WALL_S,
            "registration": {
                "kind": "file",
                "canonical_path": os.path.join(register_output, REGISTRATION_ARTIFACT),
                "produced_by": register_node,
            },
            "registration_done_marker": register_done,
            "r0257_panel": r0257_panel,
            "r0257_verdict": r0257_verdict,
            "scope_modules": list(SCOPE_MODULES),
            "upstream_review_state": review_state,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
                "cpu_heavy": False,
            },
        },
    ]

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
        "schema": "round0260-one-sided-purity-registration-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-analysis",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            PANEL_CAPABILITY_N29,
            R0256_GATE_CAPABILITY,
            "minilm-mixed-6250k-ladder-panel-v1",
            "minilm-mixed-6250k-rung-gate-verdict-v1",
        ],
        "capabilities_produced": [GATE_CAPABILITY, READJUDICATION_CAPABILITY],
        "jobs": jobs,
        "p90_wall_s": {
            register_node: REGISTER_P90_WALL_S,
            readjudicate_node: READJUDICATE_P90_WALL_S,
            "total": REGISTER_P90_WALL_S + READJUDICATE_P90_WALL_S,
        },
        "scope_modules": list(SCOPE_MODULES),
        "stop_hook_install_guard": {
            "derived_entries": guard["derived"],
            "every_derived_entry_installs": guard["holds"],
            "gate_census": gates,
            "scope_residual": residual,
        },
        "registered": {
            "what_this_round_is": (
                "the owner's 2026-08-12 ruling, executed: the two purity criteria "
                "are re-registered as ONE-SIDED floors failing under-separation "
                "only, recalibrated from the sealed n = 29 family, and registered "
                "BEFORE R0257's three maps are re-adjudicated under them."
            ),
            "owner_ruling": OWNER_RULING,
            "owner_ruling_date": OWNER_RULING_DATE,
            "owner_ruling_section": OWNER_RULING_SECTION,
            "why_one_sided": WHY_ONE_SIDED,
            "no_tuning_statement": NO_TUNING_STATEMENT,
            "estimators_calibrated": list(REGISTERED_ESTIMATOR_NAMES),
            "n_exact": N_EXACT,
            "identity_bound_at_n": IDENTITY_BOUND_AT_N,
            "tolerance_content": TOLERANCE_CONTENT,
            "tolerance_confidence": TOLERANCE_CONFIDENCE,
            "gated_metrics": list(GATED_METRICS),
            "descriptive_metrics": list(DESCRIPTIVE_METRICS),
            "supersedes_purity_criteria_in": SUPERSEDED_PURITY_CRITERIA_IN,
            "maps_re_adjudicated": list(rung_cell_ids()),
            "the_registration_node_binds_no_rung_artifact": True,
            "how_that_is_enforced": (
                "the registration node reads its own `expected_inputs` back out of "
                "the manifest and raises if any declared path matches a rung "
                "marker. The list above is that node's whole input set."
            ),
            "acceptance_rule": (
                "every execution check holds on both nodes. Specifically: the "
                "registration node binds no rung artifact and says so from its own "
                "manifest; the restricted calibration, the retired two-sided bands "
                "and the ffr floor all reproduce bitwise from the sealed family; "
                "the multiplier sits below the identity bound and every defining "
                "cell can fail; every planted registry, family and sidedness "
                "defect lands as registered; and the re-adjudication node refuses "
                "unless the registration artifact was sealed before it started. "
                "NO NUMERICAL OUTCOME MAKES THIS ROUND A FAILURE: three passing "
                "maps, three failing maps and a split are each findings to publish "
                "with margins, power and false-alarm rate."
            ),
            "a_failing_map_is_a_finding": (
                "If a re-adjudicated map still fails, it is published failing, "
                "with its margin. No retrain, no other seed, no adjusted floor. "
                "The owner ruling's 'do not tune anything to preserve the "
                "published pass' applies here with full force, in both directions."
            ),
            "gpu_hours_expected": 0.0,
            "gpu_required_anywhere": False,
        },
    })
    manifest_path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(manifest_path, queue, immutable=True)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="prepare the R0260 queue")
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    args = parser.parse_args(argv)
    path = prepare_round0260(release_sha=args.release_sha, queue_root=args.queue_root)
    print(json.dumps({"queue": path, "sha256": file_sha256(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
