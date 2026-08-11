"""Build R0249's queue. Never launches anything.

Two CPU-shaped nodes that read no bulk array at all. The only inputs are the
round file and the release CPU smoke, so there is nothing to rehash and nothing
to inherit: R0249's subject is the guard layer and the process boundary, not a
measurement over the substrate.

The manifest's `external_memory_limit` block now declares `memory.high` in
`root-scope`, derived from the registry rather than typed, and the preparation
REFUSES to build a queue whose mode this box cannot place — there is no
downgrade to a bound the node could lift.
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
from basemap.round0248_inventory import (
    derive_inventory,
    discover_registry_regime_modules,
    discover_round_modules,
)
from basemap.round0249_external import (
    CONTROL_MEMORY_HIGH_BYTES,
    DEFAULT_EXTERNAL_MEMORY_MODE,
    MEMORY_HIGH_NOTE,
    external_memory_limit_declaration,
    external_memory_mode_availability,
    require_external_memory_mode,
)
from basemap.round0249_guard import ACCEPTED_RESIDUAL_RISKS, GUARD_SCOPE_NOTE
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
from experiments.round0249_nodes import (
    EXTERNAL_ACTION,
    EXTERNAL_CAPABILITY,
    GUARDFIX_ACTION,
    GUARDFIX_CAPABILITY,
    NODE_ANON_BUDGET_BYTES,
)

ROUND_ID = "0249"
ROUND_ROOT = "/data/latent-basemap/runs/round-0249"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0249-2026-08-11.md")
HANDLER_MODULE = "experiments.round0249_nodes"


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
        raise RuntimeError("R0249 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0249 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError(
            f"R0249 release checkout is at {observed}, not {release_sha}"
        )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [os.path.join(RELEASE_ROOT, ".venv/bin/python"), "-m", "pytest", "-q",
         "tests/test_round0249_contract.py", "tests/test_round0249_cpu_smoke.py"],
        cwd=RELEASE_ROOT, env=environment, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0249 release CPU smoke failed:\n{completed.stdout[-4000:]}"
        )
    return {
        "release_sha": release_sha,
        "cuda_visible_devices": "",
        "command": (
            "pytest -q tests/test_round0249_contract.py "
            "tests/test_round0249_cpu_smoke.py"
        ),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "reaches": [
            "BOTH node entry paths executed end to end through run_job - the "
            "defect class that killed R0216 (NameError), R0236 (arity) and "
            "R0242 attempt 1 (missing module)",
            "review-0248-01 §B's two assignments verbatim, both now raising "
            "AttributeError, with the seal still refused",
            "a verdict whose disclosure fields have ALL been flipped, still "
            "refused on no_sealing_blocking_override_was_attempted alone",
            "the clamp control routing `replay` through record_declaration - "
            "the function its call sites actually use",
            "the discovered module scope over every round0*.py in the release, "
            "with a planted bare registered symbol OUTSIDE the registry regime "
            "still caught",
            "the memory.high throttle control: a plain CPU allocator asking "
            "for 8x its limit is throttled and SURVIVES, with the kernel's "
            "memory.events high counter as the evidence, and the identical "
            "allocator under memory.max killed for the contrast",
            "the escape battery refusing all five attempts in root-scope, and "
            "the fail-closed control refusing an unplaceable mode rather than "
            "downgrading it",
        ],
    }


def prepare_round0249(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    prior_gpu_wall_s: float = 0.0,
    only: tuple[str, ...] = (),
    external_control_mode: str = DEFAULT_EXTERNAL_MEMORY_MODE,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0249 release SHA must be one full commit")
    if registry_fingerprint() != REGISTERED_REGISTRY_SHA256:
        raise RuntimeError(
            "R0249 STOP: the safety-parameter registry fingerprint "
            f"{registry_fingerprint()} does not match the pinned "
            f"{REGISTERED_REGISTRY_SHA256}"
        )
    inventory = derive_inventory(repo_root=RELEASE_ROOT)
    if not inventory["holds"]:
        raise RuntimeError(
            "R0249 STOP: the mechanically derived inventory does not hold at "
            f"the release: {inventory['gate_constants']['defects']} "
            f"{inventory['arm_waivers']['defects']}"
        )
    #: fail closed here too: a queue that declares a mode this box cannot place
    #: must not be built, rather than built and quietly downgraded at run time.
    mode_state = require_external_memory_mode(external_control_mode)
    round_signature, required_reviews = _issued_round(release_sha)

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0249 queue")
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
            "round_modules_discovered": len(
                discover_round_modules(repo_root=RELEASE_ROOT)
            ),
            "registry_regime_modules": list(
                discover_registry_regime_modules(repo_root=RELEASE_ROOT)
            ),
            "external_memory_mode": mode_state,
            "external_memory_limit": external_memory_limit_declaration(
                mode=external_control_mode
            ),
            "node_anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "registry_fingerprint": registry_fingerprint(),
            "registered_registry_sha256": REGISTERED_REGISTRY_SHA256,
            "class_note": SAFETY_PARAMETER_CLASS_NOTE,
            "guard_scope_note": GUARD_SCOPE_NOTE,
            "accepted_residual_risks": [
                dict(row) for row in ACCEPTED_RESIDUAL_RISKS
            ],
        },
        immutable=True,
    )

    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    guardfix_dir = os.path.join(artifacts, GUARDFIX_CAPABILITY)
    external_dir = os.path.join(artifacts, EXTERNAL_CAPABILITY)

    policy = {"gpu_required": False, "training_performed": False,
              "cpu_heavy": False}
    gpu_budget_remaining_s = float(GPU_HOURS_CAP * 3600.0 - prior_gpu_wall_s)

    jobs: list[dict[str, Any]] = [
        {
            "id": "guardfix_0249", "action": GUARDFIX_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": [],
            "outputs": [guardfix_dir],
            "done_marker": os.path.join(artifacts, "guardfix_0249.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 300.0,
            "capability": GUARDFIX_CAPABILITY,
            "anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "stage_budget_s": 1_800.0,
            "gpu_budget_remaining_s": gpu_budget_remaining_s,
            "node_policy": dict(policy),
        },
        {
            "id": "external_0249", "action": EXTERNAL_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": ["guardfix_0249"],
            "outputs": [external_dir],
            "done_marker": os.path.join(artifacts, "external_0249.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 300.0,
            "capability": EXTERNAL_CAPABILITY,
            "anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "stage_budget_s": 1_800.0,
            "external_control_mode": external_control_mode,
            "external_control_high_bytes": CONTROL_MEMORY_HIGH_BYTES,
            "node_policy": dict(policy),
        },
    ]

    if only:
        keep = set(only)
        unknown = keep - {str(job["id"]) for job in jobs}
        if unknown:
            raise RuntimeError(f"R0249 has no such node(s): {sorted(unknown)}")
        jobs = [job for job in jobs if str(job["id"]) in keep]
        for job in jobs:
            job["deps"] = [dep for dep in job["deps"] if dep in keep]

    queue = _base_manifest(
        round_id=ROUND_ID, release_sha=release_sha, round_file=ROUND_FILE,
        queue_root=queue_root, gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-cpu", gpu=False,
    )
    queue.update({
        "schema": "round0249-disclosure-enforcement-and-memory-high-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-guard",
        "required_reviews": list(required_reviews),
        #: R0249: memory.high in root-scope. The runner reads this block and
        #: places every node under a transient systemd scope; it does NOT fall
        #: back to a weaker mode if root-scope cannot be placed.
        "external_memory_limit": external_memory_limit_declaration(
            mode=external_control_mode
        ),
        "capability_dependencies": [
            "round0244-threaded-host-watchdog-v1",
            "round0245-guard-closure-v1",
            "round0246-guard-closure-v1",
            "round0247-safety-parameter-registry-v1",
            "round0248-external-cgroup-memory-bound-v1",
            "runner-signal-free-node-supervision-v1",
        ],
        "capabilities_produced": [GUARDFIX_CAPABILITY, EXTERNAL_CAPABILITY],
        "prior_attempt_gpu_wall_s": float(prior_gpu_wall_s),
        "gpu_budget_remaining_s": gpu_budget_remaining_s,
        "substrate_inherited": False,
        "graph_inherited": False,
        "inherited_artifacts": [],
        "inheritance_note": (
            "R0249 reads no bulk array at all. Its subject is the guard layer "
            "and the process boundary, so its only inputs are the round file "
            "and the release CPU smoke."
        ),
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {"total": 0.0},
        "scientific_contract": {
            "question": (
                "review-0248-01 is the fourth consecutive review to defeat "
                "this guard family, and it separated what is left into two "
                "kinds. Two defects are real and mechanical: four disclosure "
                "fields that R0247 had already established should be "
                "read-only, and one enforcement classification that was false "
                "three ways and switched off the arm that would have caught "
                "them. Everything else it holds open requires the node to "
                "DELIBERATELY CHEAT. R0249 fixes the two, replaces the last "
                "hand-written module list with discovery, moves the external "
                "bound from a kill onto back-pressure, and stops."
            ),
            "rows": ROWS,
            "k": GRAPH_K,
            "closure_1_the_disclosure_fields": {
                "defect": (
                    "review-0248-01 §B: AbortPollTracker.replay and "
                    ".clock_is_the_registered_monotonic_clock were plain "
                    "instance attributes. Two assignments after construction "
                    "flipped replay_only to false, emptied "
                    "gate_arms_waived_by_declaration, and "
                    "require_enforcement_evidence() SEALED a replayed, "
                    "scripted-clock gate as enforcement evidence."
                ),
                "closure": (
                    "all four - those two, inner_is_a_registered_abort_reader, "
                    "and the safety_overrides tuple the new sealing arm reads "
                    "- are read-only properties, the treatment R0247 gave "
                    "max_poll_spacing_s on this same class."
                ),
                "positive_control": (
                    "the reviewer's two statements verbatim, each raising "
                    "AttributeError, with the seal still refused"
                ),
            },
            "closure_2_the_enforcement_classification": {
                "defect": (
                    "review-0248-01 §B finding 3: `replay` was registered "
                    "enforcement='clamped', which is false three ways - its "
                    "only call site does not clamp, run_clamp_controls proved "
                    "a clamp through a function no call site uses, and "
                    "weakening_overrides() filters on 'refused' so the "
                    "weakening record never reached "
                    "no_weakening_safety_override_was_attempted, which "
                    "reported True on a replayed gate."
                ),
                "closure": (
                    "a third enforcement class, 'declared': the value stands "
                    "(a replay receipt still says it is a replay), no clamp is "
                    "claimed, and the weakening record reaches "
                    "sealing_refused_overrides() so "
                    "require_enforcement_evidence() refuses WITHOUT reading a "
                    "disclosure attribute. Registering it 'refused' instead "
                    "would have made require() refuse every replay and turned "
                    "nine meaningful gate_refused_it arms vacuous."
                ),
                "positive_control": (
                    "a verdict with every disclosure field flipped, still "
                    "refused on no_sealing_blocking_override_was_attempted "
                    "alone; plus the non-vacuity control showing require() "
                    "still passes a replay inside the ceiling and still "
                    "refuses R0245 attempt-1's gap on the spacing arm"
                ),
            },
            "closure_3_the_discovered_module_scope": {
                "defect": (
                    "review-0248-01 §D.3: GUARDED_MODULES was a second "
                    "hand-written list - the artifact the mechanical inventory "
                    "existed to eliminate - and one file outside it, "
                    "experiments/round0242_nodes.py:246, compared a bare "
                    "WATCHDOG_ANON_BYTES against a second literal copy of a "
                    "registered number."
                ),
                "closure": (
                    "two discovered scopes. Every round0*.py under basemap/ "
                    "and experiments/ is scanned for bare registered symbols, "
                    "a judgement the registry makes with no human input; the "
                    "subset whose source names round0247_registry is "
                    "additionally required to have every compared constant "
                    "triaged. round0242_nodes:246 reads the registry now."
                ),
                "positive_control": (
                    "the R0242 shape planted in a module that does NOT import "
                    "the registry - the exact position the real defect was in "
                    "- and still caught"
                ),
                "round_modules_discovered": len(
                    discover_round_modules(repo_root=RELEASE_ROOT)
                ),
                "registry_regime_modules_discovered": len(
                    discover_registry_regime_modules(repo_root=RELEASE_ROOT)
                ),
                "gate_constant_comparisons_derived": inventory[
                    "gate_constants"
                ]["comparisons_against_module_level_constants"],
                "arm_waivers_derived": inventory["arm_waivers"][
                    "arms_waivable_by_a_declaration"
                ],
                "parameters_registered": len(REGISTERED_SAFETY_PARAMETERS),
            },
            "closure_4_the_external_bound_throttles_instead_of_killing": {
                "defect": (
                    "review-0248-01 §E.5: memory.max charges page cache, which "
                    "this deliberately memmap-heavy workload produces by the "
                    "terabyte while anonymous memory stays near 4 GB, so 'the "
                    "limit sits above the in-process budget, therefore the "
                    "cooperative trip fires first' was an assumption about "
                    "memory COMPOSITION rather than arithmetic. And a "
                    "memory.max breach OOM-kills, which against a CUDA holder "
                    "is the wedge this box has been rebooted for twice. "
                    "§E.6: a user-scope node lifts its own limit in two writes."
                ),
                "closure": (
                    "memory.high with memory.swap.max=0 and NO memory.max, in "
                    "root-scope by default, with no silent downgrade."
                ),
                "positive_control": (
                    "a plain CPU allocator asking for 8x its memory.high is "
                    "THROTTLED and survives - kernel memory.events high "
                    "counter rising, oom and oom_kill both zero, its own "
                    "allocation rate collapsing by orders of magnitude - while "
                    "the identical allocator under memory.max is killed"
                ),
                "what_it_costs": (
                    "a hard stop. memory.high only makes continuing "
                    "expensive, so the thing that STOPS a runaway is now the "
                    "cooperative in-process guard, with the kernel as brake "
                    "rather than executioner."
                ),
                "external_memory_limit": external_memory_limit_declaration(
                    mode=external_control_mode
                ),
                "note": MEMORY_HIGH_NOTE,
            },
            "accepted_residual_risks": [
                dict(row) for row in ACCEPTED_RESIDUAL_RISKS
            ],
            "gpu_hours_cap": GPU_HOURS_CAP,
            "gpu_hours_cap_note": (
                "0.0 GPU-h expected. Both nodes are CPU-shaped and neither "
                "creates a CUDA context; the queue does not take the GPU "
                "lease. The one memory.max child in the round is the contrast "
                "arm's plain bytearray allocator, which imports no array "
                "library at all."
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
    parser.add_argument(
        "--external-control-mode", default=DEFAULT_EXTERNAL_MEMORY_MODE)
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args(argv)
    availability = external_memory_mode_availability(args.external_control_mode)
    if not availability["available"]:
        raise SystemExit(
            f"R0249 STOP: {json.dumps(availability)}. R0249 does not downgrade "
            "an unplaceable external memory mode."
        )
    path = prepare_round0249(
        release_sha=args.release_sha, queue_root=args.queue_root,
        prior_gpu_wall_s=args.prior_gpu_wall_s, only=tuple(args.only),
        external_control_mode=args.external_control_mode,
    )
    print(json.dumps({"queue": path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
