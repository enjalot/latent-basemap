"""Build R0248's queue. Never launches anything.

Two CPU-shaped nodes that read no bulk array at all. The only inputs are the
round file and the release CPU smoke, so there is nothing to rehash and nothing
to inherit: R0248's subject is the guard layer and the process boundary, not a
measurement over the substrate.

The manifest carries the new `external_memory_limit` block, derived from the
registry rather than typed, so `roundrun` can place every node under a cgroup v2
`memory.max` with `memory.swap.max=0` without importing anything from basemap.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0238_rung5 import GRAPH_K
from basemap.round0247_registry import (
    GPU_HOURS_CAP,
    REGISTERED_REGISTRY_SHA256,
    REGISTERED_SAFETY_PARAMETERS,
    ROWS,
    SAFETY_PARAMETER_CLASS_NOTE,
    registry_fingerprint,
    safety_parameter_inventory,
)
from basemap.round0248_external import (
    CONTROL_MEMORY_MAX_BYTES,
    EXTERNAL_BOUND_NOTE,
    cgroup_v2_memory_available,
    external_memory_limit_declaration,
)
from basemap.round0248_guard import GUARD_SCOPE_NOTE
from basemap.round0248_inventory import derive_inventory
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
from experiments.round0248_nodes import (
    EXTERNAL_ACTION,
    EXTERNAL_CAPABILITY,
    GAPGUARD_ACTION,
    GAPGUARD_CAPABILITY,
    NODE_ANON_BUDGET_BYTES,
)

ROUND_ID = "0248"
ROUND_ROOT = "/data/latent-basemap/runs/round-0248"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0248-2026-08-11.md")
HANDLER_MODULE = "experiments.round0248_nodes"


def _issued_round(release_sha: str) -> tuple[dict[str, Any], list[str]]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor",
         base_commit, release_sha],
        check=False,
    ).returncode == 0
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or not descendant
    ):
        raise RuntimeError("R0248 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0248 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError(
            f"R0248 release checkout is at {observed}, not {release_sha}"
        )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [os.path.join(RELEASE_ROOT, ".venv/bin/python"), "-m", "pytest", "-q",
         "tests/test_round0248_contract.py", "tests/test_round0248_cpu_smoke.py"],
        cwd=RELEASE_ROOT, env=environment, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0248 release CPU smoke failed:\n{completed.stdout[-4000:]}"
        )
    return {
        "release_sha": release_sha,
        "cuda_visible_devices": "",
        "command": (
            "pytest -q tests/test_round0248_contract.py "
            "tests/test_round0248_cpu_smoke.py"
        ),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "reaches": [
            "BOTH node entry paths executed end to end through run_job - the "
            "defect class that killed R0216 (NameError), R0236 (arity) and "
            "R0242 attempt 1 (missing module)",
            "the mechanically derived inventory over every guarded module, "
            "failing closed on a gate comparison that reads a bare "
            "module-level mirror of a registered bound",
            "review-0247-01 A.3's two assignments to "
            "WATCHDOG_MAX_OBSERVATION_GAP_S and its mean, with the 5.0 s "
            "coverage receipt still REFUSED",
            "review-0247-01 A.3's assignment to SAMPLER_MAX_ANONYMOUS_BYTES, "
            "with an over-budget peak still failing the verdict arm",
            "review-0247-01 A.4's one-call marker on a no-op, with the gate "
            "AND require_enforcement_evidence still refusing it",
            "review-0247-01 A.6's replay=True gate, with the declaration "
            "recorded, the waived arms NAMED, and sealing refused",
            "the EXTERNAL cgroup memory bound: a plain CPU allocator killed by "
            "the kernel under memory.max with the in-process guard disabled, a "
            "second child surviving the same limit, and the escape battery in "
            "both scope modes",
        ],
    }


def prepare_round0248(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    prior_gpu_wall_s: float = 0.0,
    only: tuple[str, ...] = (),
    external_control_mode: str = "root-scope",
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0248 release SHA must be one full commit")
    if registry_fingerprint() != REGISTERED_REGISTRY_SHA256:
        raise RuntimeError(
            "R0248 STOP: the safety-parameter registry fingerprint "
            f"{registry_fingerprint()} does not match the pinned "
            f"{REGISTERED_REGISTRY_SHA256}"
        )
    inventory = derive_inventory(repo_root=RELEASE_ROOT)
    if not inventory["holds"]:
        raise RuntimeError(
            "R0248 STOP: the mechanically derived inventory does not hold at "
            f"the release: {inventory['gate_constants']['defects']} "
            f"{inventory['arm_waivers']['defects']}"
        )
    availability = cgroup_v2_memory_available()
    if not availability["holds"]:
        raise RuntimeError(
            f"R0248 STOP: this box cannot place a cgroup memory limit and the "
            f"external bound is the round's subject: {availability}"
        )
    round_signature, required_reviews = _issued_round(release_sha)

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0248 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        smoke_path, _release_cpu_smoke(release_sha), immutable=True
    )
    atomic_write_new_json(
        os.path.join(preflight, "registered-safety-parameters.json"),
        {
            "registered_before_the_round_ran": safety_parameter_inventory(),
            "mechanically_derived_inventory": inventory,
            "cgroup_v2_availability": availability,
            "external_memory_limit": external_memory_limit_declaration(),
            "node_anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "registry_fingerprint": registry_fingerprint(),
            "registered_registry_sha256": REGISTERED_REGISTRY_SHA256,
            "class_note": SAFETY_PARAMETER_CLASS_NOTE,
            "guard_scope_note": GUARD_SCOPE_NOTE,
        },
        immutable=True,
    )

    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    gapguard_dir = os.path.join(artifacts, GAPGUARD_CAPABILITY)
    external_dir = os.path.join(artifacts, EXTERNAL_CAPABILITY)

    policy = {"gpu_required": False, "training_performed": False,
              "cpu_heavy": False}
    gpu_budget_remaining_s = float(GPU_HOURS_CAP * 3600.0 - prior_gpu_wall_s)

    jobs: list[dict[str, Any]] = [
        {
            "id": "gapguard_0248", "action": GAPGUARD_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": [],
            "outputs": [gapguard_dir],
            "done_marker": os.path.join(artifacts, "gapguard_0248.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 300.0,
            "capability": GAPGUARD_CAPABILITY,
            "anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "stage_budget_s": 1_800.0,
            "gpu_budget_remaining_s": gpu_budget_remaining_s,
            "node_policy": dict(policy),
        },
        {
            "id": "external_0248", "action": EXTERNAL_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": ["gapguard_0248"],
            "outputs": [external_dir],
            "done_marker": os.path.join(artifacts, "external_0248.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 300.0,
            "capability": EXTERNAL_CAPABILITY,
            "anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "stage_budget_s": 1_800.0,
            "external_control_mode": external_control_mode,
            "external_control_max_bytes": CONTROL_MEMORY_MAX_BYTES,
            "node_policy": dict(policy),
        },
    ]

    if only:
        keep = set(only)
        unknown = keep - {str(job["id"]) for job in jobs}
        if unknown:
            raise RuntimeError(f"R0248 has no such node(s): {sorted(unknown)}")
        jobs = [job for job in jobs if str(job["id"]) in keep]
        for job in jobs:
            job["deps"] = [dep for dep in job["deps"] if dep in keep]

    queue = _base_manifest(
        round_id=ROUND_ID, release_sha=release_sha, round_file=ROUND_FILE,
        queue_root=queue_root, gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-cpu", gpu=False,
    )
    queue.update({
        "schema": "round0248-comparison-site-and-external-bound-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-guard",
        "required_reviews": list(required_reviews),
        #: THE NEW BLOCK. The runner reads this and places every node under a
        #: transient systemd scope with memory.max and memory.swap.max=0. The
        #: numbers are derived from the registry by
        #: `external_memory_limit_declaration()`, never typed.
        "external_memory_limit": external_memory_limit_declaration(),
        "capability_dependencies": [
            "round0244-threaded-host-watchdog-v1",
            "round0245-guard-closure-v1",
            "round0246-guard-closure-v1",
            "round0247-safety-parameter-registry-v1",
            "runner-signal-free-node-supervision-v1",
        ],
        "capabilities_produced": [GAPGUARD_CAPABILITY, EXTERNAL_CAPABILITY],
        "prior_attempt_gpu_wall_s": float(prior_gpu_wall_s),
        "gpu_budget_remaining_s": gpu_budget_remaining_s,
        "substrate_inherited": False,
        "graph_inherited": False,
        "inherited_artifacts": [],
        "inheritance_note": (
            "R0248 reads no bulk array at all. Its subject is the guard layer "
            "and the process boundary, so its only inputs are the round file "
            "and the release CPU smoke."
        ),
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {"total": 0.0},
        "scientific_contract": {
            "question": (
                "review-0247-01 defeated R0247 five ways in an hour, the third "
                "consecutive round defeated in the same shape: the guard "
                "verifies one object and the decision reads another. R0248 "
                "splits the five into two kinds - four bounds that never "
                "entered the decision, which are bugs and are fixed with "
                "controls in the reviewer's shape, and three that require the "
                "node to deliberately cheat, which no in-process Python guard "
                "can reach. For those, the memory bound moves OUT of the "
                "process onto a cgroup v2 memory.max."
            ),
            "rows": ROWS,
            "k": GRAPH_K,
            "closure_1_the_two_observation_gap_bounds": {
                "defect": (
                    "review-0247-01 A.3: round0246_guard:278 and :289 compared "
                    "against module globals. Two assignments and "
                    "review-0246-01 A's 5.0 s receipt passed, while "
                    "registered_watchdog_max_observation_gap_s in the same "
                    "receipt still read 2.5109531834854018."
                ),
                "closure": (
                    "both comparisons call registered_value() and resolve the "
                    "registry at the moment of the comparison; the receipt "
                    "publishes the value read AT THE COMPARISON SITE."
                ),
                "positive_control": (
                    "the reviewer's two assignments, with the 5.0 s receipt "
                    "still refused"
                ),
            },
            "closure_2_the_sampler_anonymous_ceiling": {
                "defect": (
                    "review-0247-01 A.3: sampler_max_anonymous_bytes was "
                    "registered as parameter 14 and still enforced as a bare "
                    "literal at round0244_prereq:112, compared as an imported "
                    "module global in three node verdict arms."
                ),
                "closure": (
                    "the literal is a mirror; all three arms call "
                    "sampler_max_anonymous_bytes(), which reads the registry."
                ),
                "positive_control": (
                    "SAMPLER_MAX_ANONYMOUS_BYTES = 1 << 50, with an "
                    "over-budget peak still failing the arm"
                ),
            },
            "closure_3_the_runtime_abort_reader_allowlist": {
                "defect": (
                    "review-0247-01 A.4: registered_abort_reader is exported "
                    "and one call made a no-op pass require() AND "
                    "require_enforcement_evidence."
                ),
                "closure": (
                    "sanction is by qualified name against the source-level "
                    "set; the marker sanctions nothing, its application is a "
                    "recorded violation, and every verdict names WHICH "
                    "mechanism sanctioned the reader (review-0247-01 H5)."
                ),
                "positive_control": (
                    "the decorator call and a raw setattr, both refused, with "
                    "a name-registered reader still passing"
                ),
            },
            "closure_4_replay_and_the_derived_inventory": {
                "defect": (
                    "review-0247-01 A.6: replay is a self-declared bool that "
                    "waives two arms of require() and was not in the "
                    "inventory - the shape R0247 retired training_performed "
                    "for. Four rounds have shipped a hand-written inventory "
                    "and four reviews have found an omission."
                ),
                "closure": (
                    "replay is registered; the declaration is recorded, the "
                    "waived arms are NAMED in the verdict, and sealing "
                    "refuses. The inventory itself is now DERIVED from the "
                    "source with ast: every module-level constant a gate "
                    "compares against, and every arm a declaration can waive."
                ),
                "positive_control": (
                    "a replay=True gate refused at the sealing boundary, and "
                    "the derivation finding both waivers; the derivation also "
                    "found FIVE further bare-registered comparisons that no "
                    "review had named"
                ),
                "parameters_registered": len(REGISTERED_SAFETY_PARAMETERS),
                "gate_constant_comparisons_derived": inventory[
                    "gate_constants"
                ]["comparisons_against_module_level_constants"],
                "arm_waivers_derived": inventory["arm_waivers"][
                    "arms_waivable_by_a_declaration"
                ],
            },
            "closure_5_the_external_memory_bound": {
                "defect": (
                    "three defeats - rebinding the registry mapping, mutating "
                    "the dict behind the MappingProxyType, and scripting the "
                    "module clock - require the node to deliberately cheat, "
                    "and are the same category as the fabricated receipt R0247 "
                    "published as unclosable. No in-process Python guard can "
                    "constrain the process it runs in."
                ),
                "closure": (
                    "the runner places every node under a cgroup v2 "
                    "memory.max with memory.swap.max=0 through a transient "
                    "systemd scope. It is not a Python object: no rebind, no "
                    "gc.get_referents(), no clock to script."
                ),
                "positive_control": (
                    "a plain CPU allocator asking for 8x its limit is killed "
                    "by the kernel with the in-process guard disabled, "
                    "journalctl -k names the scope as the oom_memcg, and a "
                    "second child under the same limit survives"
                ),
                "what_it_does_not_cover": (
                    "poll spacing, abort latency, GPU memory and receipt "
                    "honesty. Only independent recomputation by a reviewer "
                    "who did not run the round covers fabrication."
                ),
                "external_memory_limit": external_memory_limit_declaration(),
                "note": EXTERNAL_BOUND_NOTE,
            },
            "gpu_hours_cap": GPU_HOURS_CAP,
            "gpu_hours_cap_note": (
                "0.0 GPU-h expected. Both nodes are CPU-shaped and neither "
                "creates a CUDA context; the queue does not take the GPU "
                "lease. The cgroup limit is deliberately never applied to a "
                "process holding a CUDA context in this round, because a "
                "kernel OOM kill of a CUDA holder is the wedge this machine "
                "has been rebooted for twice."
            ),
            "no_training": True,
            "no_gate_registered": True,
            "no_adoption_claimed": True,
            "no_map_quality_claimed": True,
            "no_displacement_measured": True,
            "no_atlas_claim": True,
            "no_build": True,
            "no_assembly": True,
        },
    })
    queue_path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(queue_path, queue, immutable=True)
    return queue_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    parser.add_argument("--prior-gpu-wall-s", type=float, default=0.0)
    parser.add_argument("--external-control-mode", default="root-scope")
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args(argv)
    path = prepare_round0248(
        release_sha=args.release_sha, queue_root=args.queue_root,
        prior_gpu_wall_s=args.prior_gpu_wall_s, only=tuple(args.only),
        external_control_mode=args.external_control_mode,
    )
    print(json.dumps({"queue": path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
