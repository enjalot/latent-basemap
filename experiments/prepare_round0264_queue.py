"""Build the R0264 queue (DRAFT) — register the collapse floor + fail-closed fog
ceiling + density_v3-descriptive, then compile the legacy-vs-healthy separation record.

**This is a DRAFT builder.** The round file
(`latent-labs/basemap-100m/round-0264-2026-08-13.md`) is `status: draft` and waits on
P1's dose x N decomposition (the `N_RULE_PENDING_P1_DECOMPOSITION` placeholder) before
it can be issued. `_issued_round` requires `status: issued`, so this builder REFUSES to
produce a queue until the round is issued — which is correct: a draft round has no
queue. It is authored now so that issuing is the only remaining step.

Two CPU nodes, one dependency edge. Node 1 (register) is bound to the round file, the
sealed R0234 n=13 and R0255 n=29 calibration JSONs (for the bitwise validation-gate and
k1 reproductions) and the collapse/fog program's own calibration + measurements records
(the healthy/legacy/md035 family source and an optional cross-check). Node 2
(separation) depends on it, binds its output by hash at read time, and refuses to run
unless the registration sealed first.

No training, no GPU, no new cells, no new seeds. Every job is `cpu_heavy` /
`gpu_required: False` and `CUDA_VISIBLE_DEVICES` is forced empty in the child
environment.
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
from basemap.round0254_dispatch import (
    SCOPE_MODULES,
    assert_derived_entries_install,
    dispatch_census,
    entry_tuples,
    gate_census,
    scope_residual,
)
from basemap.round0255_gate_n29 import N_EXACT

from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
from experiments.round0264_nodes import (
    GATE_CAPABILITY,
    N_RULE_PENDING,
    PROVISIONAL_STATUS,
    REGISTER_ACTION,
    REGISTRATION_ARTIFACT,
    ROUND_ID,
    SEALED,
    SEPARATION_ACTION,
    SEPARATION_ARTIFACT,
    SEPARATION_CAPABILITY,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0264"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0264-2026-08-13.md")

#: Every bound input's path, resolved from the node's SEALED provenance table so the
#: builder and the node agree on exactly one string per input.
R0234_N13 = SEALED["r0234_n13_calibration"]["path"]
R0255_N29 = SEALED["r0255_n29_calibration"]["path"]
COLLAPSE_FOG_CALIBRATION = SEALED["collapse_fog_calibration"]["path"]
COLLAPSE_FOG_MEASUREMENTS = SEALED["collapse_fog_measurements"]["path"]

#: CPU-only. Node 1's cost is three Gaussian-null calibrations (n=13/29 at 4M families,
#: n=7 at 1M) plus order statistics over seven cells; node 2 reads two JSONs and applies
#: two judges over 29 + 7 + 2 cells.
REGISTER_P90_WALL_S = 90.0
SEPARATION_P90_WALL_S = 45.0
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
        raise RuntimeError(
            "R0264 round is not issued for this release. This is EXPECTED while the round "
            f"is status: draft: it stays draft until the N-adjustment rule "
            f"({N_RULE_PENDING}) is filled from P1's dose x N decomposition and the round "
            "is issued. A draft round has no queue."
        )
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0264 round must declare its required reviews")
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
            "Review is post-hoc: it blocks the downstream claim, not the launch. This "
            "round registers PROVISIONAL bands and is itself registered-and-contingent "
            "until R0264 carries an accepted review AND the N-adjustment rule is filled."
        ),
    }


def _signature(path: str, label: str) -> dict:
    """Real sha256 + bytes from the file on disk, at prepare time.

    These inputs (the R0234/R0255 calibration JSONs and the collapse/fog program's
    records) are plain JSON, not seal-enveloped, so they are signed with
    `expected_input_signature` rather than read through `read_sealed`. A byte that
    changed on disk changes the manifest.
    """
    if not os.path.exists(path):
        raise RuntimeError(f"R0264 bound input absent: {label} at {path}")
    return expected_input_signature(path)


def prepare_round0264(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0264 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    review_state = _upstream_review_state(list(required_reviews))

    # Every bound input's sha256 + bytes are computed from the file on disk HERE.
    r0234_n13 = _signature(R0234_N13, "R0234 sealed n=13 calibration")
    r0255_n29 = _signature(R0255_N29, "R0255 sealed n=29 calibration")
    collapse_fog_calibration = _signature(
        COLLAPSE_FOG_CALIBRATION, "collapse/fog program calibration record"
    )
    collapse_fog_measurements = _signature(
        COLLAPSE_FOG_MEASUREMENTS, "collapse/fog program measurements record"
    )

    census = dispatch_census()
    guard = assert_derived_entries_install(SCOPE_MODULES, census)
    gates = gate_census(entry_tuples(guard["derived"]))
    residual = scope_residual(census, SCOPE_MODULES)

    ensure_data_directory(ROUND_ROOT, label="R0264 round root")
    queue_root = create_fresh_directory(queue_root, label="R0264 queue")
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts"), label="R0264 artifacts"
    )

    register_node = REGISTER_ACTION
    register_output = os.path.join(artifacts, GATE_CAPABILITY)
    register_done = os.path.join(artifacts, f"{register_node}.done.json")

    #: Node 1's declared inputs: the round file, the two sealed calibration JSONs (the
    #: bitwise anchors) and the collapse/fog program's calibration + measurements
    #: records (the family source + optional cross-check). No coordinates are bound: the
    #: family statistics are the node's sealed constants.
    register_inputs = _dedupe([
        round_signature, r0234_n13, r0255_n29,
        collapse_fog_calibration, collapse_fog_measurements,
    ])

    separation_node = SEPARATION_ACTION
    separation_output = os.path.join(artifacts, SEPARATION_CAPABILITY)
    jobs = [
        {
            "id": register_node,
            "action": REGISTER_ACTION,
            "handler_module": "experiments.round0264_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [register_output],
            "done_marker": register_done,
            "expected_inputs": register_inputs,
            "p90_wall_s": REGISTER_P90_WALL_S,
            "r0234_n13": r0234_n13,
            "r0255_n29": r0255_n29,
            "collapse_fog_calibration": collapse_fog_calibration,
            "collapse_fog_measurements": collapse_fog_measurements,
            "scope_modules": list(SCOPE_MODULES),
            "upstream_review_state": review_state,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
                "cpu_heavy": True,
            },
        },
        {
            "id": separation_node,
            "action": SEPARATION_ACTION,
            "handler_module": "experiments.round0264_nodes",
            "handler_callable": "run_job",
            "deps": [register_node],
            "outputs": [separation_output],
            "done_marker": os.path.join(artifacts, f"{separation_node}.done.json"),
            "expected_inputs": _dedupe([round_signature, collapse_fog_measurements]),
            "p90_wall_s": SEPARATION_P90_WALL_S,
            "registration": {
                "kind": "file",
                "canonical_path": os.path.join(register_output, REGISTRATION_ARTIFACT),
                "produced_by": register_node,
            },
            "registration_done_marker": register_done,
            "collapse_fog_measurements": collapse_fog_measurements,
            "scope_modules": list(SCOPE_MODULES),
            "upstream_review_state": review_state,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
                "cpu_heavy": True,
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
    # CPU round that must not touch the card: force the child device mask empty.
    queue["child_environment"]["CUDA_VISIBLE_DEVICES"] = ""
    queue.update({
        "schema": "round0264-collapse-floor-fog-ceiling-registration-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-analysis",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [],
        "capabilities_produced": [GATE_CAPABILITY, SEPARATION_CAPABILITY],
        "jobs": jobs,
        "p90_wall_s": {
            register_node: REGISTER_P90_WALL_S,
            separation_node: SEPARATION_P90_WALL_S,
            "total": REGISTER_P90_WALL_S + SEPARATION_P90_WALL_S,
        },
        "scope_modules": list(SCOPE_MODULES),
        "stop_hook_install_guard": {
            "derived_entries": guard["derived"],
            "every_derived_entry_installs": guard["audit"][
                "every_entry_installs_effectively"
            ],
            "gate_census": gates,
            "scope_residual": residual,
        },
        "registered": {
            "what_this_round_is": (
                "register three things from the CPU-only collapse/fog map-quality "
                "program: (a) COLLAPSE as a one-sided lower floor on r10/R*sqrt(N); (b) "
                "FOG as a one-sided upper ceiling on low_density_mass_fraction that FAILS "
                "CLOSED on degeneracy (resolution_levels < 1 => NOT_MEASURABLE, never a "
                "pass); (c) density_v3 as descriptive_only_never_a_gate. The bands are "
                "PROVISIONAL, fitted to the HEALTHY family (seven core min_dist=0.00 umap "
                "arms); the legacy n=29 family is the collapsed fingerprint only."
            ),
            "provisional_status": PROVISIONAL_STATUS,
            "n_exact": N_EXACT,
            "n_adjustment_rule": N_RULE_PENDING,
            "n_adjustment_rule_is_pending": True,
            "capabilities_produced": [GATE_CAPABILITY, SEPARATION_CAPABILITY],
            "the_registration_node_binds_no_coordinates": True,
            "how_the_fog_fail_closed_is_proven": (
                "node 1 measures a planted degenerate map through the real map_fog "
                "binning (resolution_levels < 1), feeds it to the fog judge which must "
                "REFUSE it (NOT_MEASURABLE), and shows the same map PASSES a naive judge "
                "that skips the degeneracy check — so the guard is proven to change the "
                "outcome, not merely to be present. The sealed md035 arm (peak bin 99) is "
                "refused too, and an honest non-degenerate map still passes."
            ),
            "acceptance_rule": (
                "every execution check holds on both nodes: the validation gate (n=13) "
                "and k1 (n=29) reproduce bitwise; the provisional bands reproduce the "
                "published values and the collapse floor separates the two modes (legacy "
                "max < floor < healthy min); the registry gates collapse+fog only with "
                "density_v3 absent; every planted registry defect is refused; the fog "
                "gate fails closed on a planted degenerate map AND the guard is shown to "
                "change the outcome; and the separation node refuses unless the "
                "registration sealed first. NO NUMERICAL OUTCOME MAKES THIS ROUND A "
                "FAILURE."
            ),
            "gpu_hours_expected": 0.0,
            "gpu_required_anywhere": False,
            "draft_note": (
                f"status: draft — held open by {N_RULE_PENDING}. This builder refuses "
                "until the round is issued."
            ),
        },
    })
    manifest_path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(manifest_path, queue, immutable=True)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="prepare the R0264 queue (DRAFT)")
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    args = parser.parse_args(argv)
    path = prepare_round0264(release_sha=args.release_sha, queue_root=args.queue_root)
    print(json.dumps({"queue": path, "sha256": file_sha256(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
