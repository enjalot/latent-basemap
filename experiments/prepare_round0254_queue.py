#!/usr/bin/env python3
"""Prepare, but never launch, the R0254 queue.

Two nodes, both CPU, in this order:

1. `dispatch_0254` — the entry list derived from the runner's own dispatch
   record, an install audit that four planted shapes cannot get past, and the
   install-and-gate count.  Cheap; runs first so a defect in it is known before
   the long node starts.
2. `writeback_0254` — six flush disciplines at the `49,152,000,000` B rung, five
   repetitions each, round-robin with a rotating start, plus one unpolled
   control and a stop planted mid-write.

**This queue registers nothing.**  No floor, no estimator, no gate, no map, no
model.  Every artifact is a measurement.

**No GPU node.**  review-0253-01 §K.1 put the write stall's distribution first
because it "is the only item that can invalidate everything downstream", and the
100M graph load is explicitly out of scope this round.  `CUDA_VISIBLE_DEVICES` is
empty and each node reads `/proc/self/maps` to say so from evidence rather than
from the environment variable.
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
from basemap.round0247_registry import registered_value, registry_fingerprint
from basemap.round0253_stop_hooks import (
    HOOKED_MODULES,
    NOT_A_FAMILY_CELL,
    THE_INSTRUMENT_IS_DEFEATABLE,
)
from basemap.round0254_dispatch import (
    DISPATCH_CAPABILITY,
    QUEUE_GLOB,
    SCOPE_MODULES,
    WRITEBACK_CAPABILITY,
    assert_derived_entries_install,
    dispatch_census,
    gate_census,
    entry_tuples,
    scope_residual,
)
from basemap.round0254_writeback import (
    ARM_UNPOLLED_CONTROL,
    SHIPPED_ARMS,
    STALL_SIZE_BYTES,
    WRITE_BLOCK_BYTES,
    arm_schedule,
    assert_write_loop_polls,
    dirty_page_settings,
)
from experiments.round0254_nodes import (
    DISPATCH_ACTION,
    REPETITIONS,
    WRITEBACK_ACTION,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ID = "0254"
ROUND_ROOT = "/data/latent-basemap/runs/round-0254"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0254-2026-08-11.md")

R0253_WRITE = (
    "/data/latent-basemap/runs/round-0253/queue/artifacts/"
    "round0253-polled-write-path-v1/writepath_0253-polled-write-path.json"
)

#: Peak transient `/data`: one 49 GB file at a time, unlinked before the next.
PEAK_SCRATCH_BYTES = STALL_SIZE_BYTES
DISK_RESERVE_BYTES = 40 << 30

#: Registered cap.  Both nodes are CPU; the cap exists because `roundrun`
#: charges wall against it regardless, and the mandate's ceiling is 3.0.
GPU_HOURS_CAP = 2.0
DISPATCH_P90_WALL_S = 900.0
#: 31 writes of 49 GB. R0253 measured `60.59` s for one at this size and
#: review-0253-01 measured `37.7`--`45.4` s; `O_DIRECT` and the 32 MiB cadence
#: are the arms that could be slower. 5400 s is ~`174` s per write.
WRITEBACK_P90_WALL_S = 5_400.0


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
        raise RuntimeError("R0254 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0254 round must declare its required reviews")
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
        state[round_id] = {"reviews_present": reviews, "accepted_reviews": len(accepted)}
        if not accepted:
            contingent.append(round_id)
    return {
        "required_reviews": list(required),
        "by_round": state,
        "rounds_without_an_accepted_review": contingent,
        "claims_contingent_on": contingent,
    }


def _free_bytes(path: str) -> int:
    stat = os.statvfs(path)
    return int(stat.f_bavail) * int(stat.f_frsize)


def _disk_headroom() -> dict[str, Any]:
    free = _free_bytes("/data")
    if free < PEAK_SCRATCH_BYTES + DISK_RESERVE_BYTES:
        raise RuntimeError(
            f"R0254 needs {PEAK_SCRATCH_BYTES + DISK_RESERVE_BYTES} B free on "
            f"/data for the stall rung; {free} B available"
        )
    return {
        "free_bytes_at_prepare": free,
        "peak_scratch_bytes": PEAK_SCRATCH_BYTES,
        "reserve_bytes": DISK_RESERVE_BYTES,
        "one_file_at_a_time": True,
        "writes_planned": len(arm_schedule(SHIPPED_ARMS, REPETITIONS)) + 3,
        "total_bytes_written_across_the_node": (
            (len(arm_schedule(SHIPPED_ARMS, REPETITIONS)) + 3) * STALL_SIZE_BYTES
        ),
        "the_peak_is_one_file_not_their_sum": True,
        "scratch_is_on_data_not_root": True,
    }


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0254 release checkout differs from requested release")
    basetemp = "/data/tmp/pytest-r0254-smoke"
    tmpdir = "/data/tmp/pytest-r0254-smoke-tmp"
    command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        f"--basetemp={basetemp}", "tests/test_round0254_contract.py",
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "TMPDIR": tmpdir,
    })
    os.makedirs(tmpdir, exist_ok=True)
    #: **No `timeout=` anywhere in this file, deliberately.** CPython implements
    #: `subprocess.run(..., timeout=N)` as `Popen.kill()`, a hidden SIGKILL, and
    #: `plan-minilm-100m-v2.md` makes purging it binding before any GPU round.
    completed = subprocess.run(
        command, cwd=RELEASE_ROOT, env=environment,
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0254 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return prompt_contract.seal({
        "schema": "round0254-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cuda_hidden": True,
        "returncode": completed.returncode,
        "basetemp": basetemp,
        "tmpdir": tmpdir,
        "stdout_tail": completed.stdout.strip().splitlines()[-5:],
    })


def _sealed_r0253_write() -> dict[str, Any]:
    """R0253's sealed write-path artifact — the `1.108830746147159x` this round
    is trying to explain.  Bound by signature so the comparison has provenance."""
    receipt = prompt_contract.read_sealed(R0253_WRITE, label="R0253 sealed write path")
    if receipt.get("round_id") != "0253":
        raise RuntimeError("R0253 sealed write-path receipt contract changed")
    return expected_input_signature(R0253_WRITE)


def prepare_round0254(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    only_nodes: tuple[str, ...] | None = None,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0254 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    r0253_write = _sealed_r0253_write()
    review_state = _upstream_review_state(list(required_reviews))
    headroom = _disk_headroom()

    census = dispatch_census()
    guard = assert_derived_entries_install(SCOPE_MODULES, census)
    gates = gate_census(entry_tuples(guard["derived"]))
    residual = scope_residual(census, SCOPE_MODULES)
    write_guard = assert_write_loop_polls()

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0254 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)

    shared_inputs = _dedupe([
        round_signature,
        r0253_write,
        expected_input_signature(smoke_path),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    jobs: list[dict[str, Any]] = []
    p90: dict[str, float] = {}

    dispatch_node = "dispatch_0254"
    jobs.append({
        "id": dispatch_node,
        "action": DISPATCH_ACTION,
        "handler_module": "experiments.round0254_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, DISPATCH_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{dispatch_node}.done.json"),
        "expected_inputs": list(shared_inputs),
        "p90_wall_s": DISPATCH_P90_WALL_S,
        "scope_modules": list(SCOPE_MODULES),
        "derived_entries_at_prepare": [
            f"{row['module']}.{row['function']}"
            for row in guard["derived"]["entries"]
        ],
        "upstream_review_state": review_state,
        "node_policy": {"gpu_required": False, "training_performed": False, "cpu_heavy": False},
    })
    p90[dispatch_node] = DISPATCH_P90_WALL_S

    writeback_node = "writeback_0254"
    jobs.append({
        "id": writeback_node,
        "action": WRITEBACK_ACTION,
        "handler_module": "experiments.round0254_nodes",
        "handler_callable": "run_job",
        "deps": [dispatch_node],
        "outputs": [os.path.join(artifacts, WRITEBACK_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{writeback_node}.done.json"),
        "expected_inputs": list(shared_inputs),
        "p90_wall_s": WRITEBACK_P90_WALL_S,
        "stall_size_bytes": STALL_SIZE_BYTES,
        "repetitions": REPETITIONS,
        "shipped_arms": list(SHIPPED_ARMS),
        "control_arm": ARM_UNPOLLED_CONTROL,
        "schedule": [
            {"repetition": repetition, "arm": arm}
            for repetition, arm in arm_schedule(SHIPPED_ARMS, REPETITIONS)
        ],
        "disk_headroom": headroom,
        "upstream_review_state": review_state,
        "node_policy": {"gpu_required": False, "training_performed": False, "cpu_heavy": True},
    })
    p90[writeback_node] = WRITEBACK_P90_WALL_S

    if only_nodes is not None:
        wanted = set(only_nodes)
        unknown = wanted - {job["id"] for job in jobs}
        if unknown:
            raise RuntimeError(f"R0254 has no node(s) {sorted(unknown)}")
        jobs = [job for job in jobs if job["id"] in wanted]
        for job in jobs:
            job["deps"] = [dep for dep in job["deps"] if dep in wanted]
        p90 = {key: value for key, value in p90.items() if key in wanted}
    p90["total"] = sum(value for key, value in p90.items() if key != "total")

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
        "schema": "round0254-write-stall-and-dispatch-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [],
        "capabilities_produced": [
            capability
            for job_id, capability in (
                (dispatch_node, DISPATCH_CAPABILITY),
                (writeback_node, WRITEBACK_CAPABILITY),
            )
            if only_nodes is None or job_id in set(only_nodes)
        ],
        "correction_of": (None if only_nodes is None else QUEUE_ROOT),
        "correction_nodes": (None if only_nodes is None else list(only_nodes)),
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": p90,
        "scientific_contract": {
            "question_a": (
                "review-0253-01 §D falsified R0253's one prescriptive statement: "
                "an 8 MiB write unit moves the median 11x and the p99 ~700x and "
                "the maximum not at all, so the binding interval is a device "
                "writeback event and not a property of the write unit. Holding "
                f"the unit at {WRITE_BLOCK_BYTES} B and writing {STALL_SIZE_BYTES} "
                f"B under {len(SHIPPED_ARMS)} flush disciplines "
                f"({', '.join(SHIPPED_ARMS)}), {REPETITIONS} repetitions each, "
                "round-robin with a rotating start: what is the DISTRIBUTION of "
                "the per-write interval under each, and does any of them keep "
                "every observed interval under the registered ceiling? If none "
                "does, that is the answer and the ceiling is what has to move."
            ),
            "question_b": (
                "review-0253-01 §E: the closing fsync + posix_fadvise + close on a "
                "49 GB file ranked SECOND in R0253's own generated census at "
                "0.962761x and appears nowhere in R0253's result. Timed as its own "
                "named interval in every arm, how large is it and does any flush "
                "discipline shrink it?"
            ),
            "question_c": (
                "review-0253-01 §A.2: across every queue.json on this box, 0 of "
                "206 dispatched handlers were among R0253's 19 registered entries, "
                "because the runner resolves run_job and run_job was excluded from "
                "R0253's census by name. Derived instead from the dispatch record "
                "-- every (handler_module, handler_callable) pair the runner has "
                "resolved, closed transitively over the module-level run_* "
                "functions they delegate to -- how many entries are there, and how "
                "many carry an install that is unconditional, unshadowed at module "
                "and function scope, and resolves at runtime to the release "
                "function?"
            ),
            "question_d": (
                "review-0253-01 §A.3 planted `if False:`, a module-level shadow "
                "and a deferred lambda against R0253's own AST auditor and all "
                "three passed. Does the replacement refuse all three, plus a "
                "try-guarded install and a function-local shadow, while still "
                "accepting an honest one -- each control written to a real "
                "importable module and passed to the SHIPPED auditor rather than "
                "to a copy of its logic (review-0253-01 §I)?"
            ),
            "question_e": (
                "review-0253-01 §F: 17 of R0253's 19 entries construct no "
                "AbortPollGate at any depth, so they emit no gap series and their "
                "stop latency is silence rather than a measured zero. Of the "
                "dispatch-derived entries, how many both install AND construct a "
                "gate?"
            ),
            "population": (
                "synthetic 49,152,000,000 B files on /data, plus the release's own "
                "source and every queue.json on this box"
            ),
            "stall_size_bytes": STALL_SIZE_BYTES,
            "write_block_bytes": WRITE_BLOCK_BYTES,
            "repetitions_per_arm": REPETITIONS,
            "shipped_arms": list(SHIPPED_ARMS),
            "control_arm": ARM_UNPOLLED_CONTROL,
            "the_smaller_write_unit_remedy_is_not_retried": (
                "review-0253-01 §D falsified it and plan-minilm-100m-v2.md records "
                "it. Retrying it would spend the disk budget re-deriving a known "
                "negative."
            ),
            "vm_dirty_settings_are_read_not_tuned": dirty_page_settings(),
            "scope_modules": list(SCOPE_MODULES),
            "dispatch_census_at_prepare": {
                "queue_glob": QUEUE_GLOB,
                "queue_manifests_scanned": census["queue_manifests_scanned"],
                "distinct_dispatched_handlers": census["distinct_dispatched_handlers"],
                "distinct_dispatched_modules": census["distinct_dispatched_modules"],
                "dispatched_callables_by_name": census["dispatched_callables_by_name"],
            },
            "derived_entry_count_at_prepare": guard["derived"]["entry_count"],
            "effective_install_audit_at_prepare": {
                "entries_audited": guard["audit"]["entries_audited"],
                "entries_with_an_effective_install":
                    guard["audit"]["entries_with_an_effective_install"],
                "every_entry_installs_effectively":
                    guard["audit"]["every_entry_installs_effectively"],
            },
            "install_and_gate_census_at_prepare": {
                "entries_audited": gates["entries_audited"],
                "entries_that_install_effectively": gates["entries_that_install_effectively"],
                "entries_that_construct_a_gate": gates["entries_that_construct_a_gate"],
                "entries_that_both_install_and_gate":
                    gates["entries_that_both_install_and_gate"],
            },
            "scope_residual_at_prepare": residual,
            "write_loop_poll_guard_at_prepare": write_guard,
            "hooked_modules": list(HOOKED_MODULES),
            "registered_max_poll_spacing_s": registered_value("r0246_max_poll_spacing_s"),
            "disk_headroom": headroom,
            "no_gpu_node": (
                "Both nodes declare gpu_required: false and the child environment "
                "sets CUDA_VISIBLE_DEVICES to empty. The 100M graph load (~7 "
                "GPU-h) is explicitly out of scope this round and remains "
                "unmeasured and unpolled. Each node reads /proc/self/maps and "
                "publishes the result, because the environment variable is not "
                "evidence (plan-minilm-100m-v2.md, 2026-08-11)."
            ),
            "registers_nothing": NOT_A_FAMILY_CELL,
            "gate_registered": False,
            "floors_registered": 0,
            "registry_fingerprint": registry_fingerprint(),
            "registry_mutated": False,
            "guard_modules_edited": False,
            "science_modules_edited": (
                "None. Two kinds of edit touch pre-existing files, and "
                "`git diff --numstat 7e2a8fe..HEAD` is the source of both counts. "
                "(1) Thirteen `install_stop_hooks(label=...)` lines, each the "
                "first statement of a function, across seven "
                "`experiments/*_nodes.py` files -- `round0113_nodes` 4, "
                "`round0250_nodes` 4, and one `run_job` each in `round0218_nodes`, "
                "`round0230_nodes`, `round0238_nodes`, `round0240_nodes` and "
                "`round0253_nodes`: 13 insertions, 0 deletions. (2) "
                "`basemap/round0253_write_path.py`, an audit path the mandate "
                "explicitly permits editing, gains an optional `source=` on "
                "`write_loop_polls` / `assert_write_loop_polls` so its positive "
                "control can call the guard instead of re-implementing it "
                "(review-0253-01 §I): 17 insertions, 6 deletions. Total across "
                "pre-existing files: 30 insertions, 6 deletions, 8 files. No "
                "metric, neighbour set, ordering, rounding, threshold, treatment "
                "or digest changes; `basemap/artifact_identity.py` and "
                "`basemap/panel_v2.py` are untouched, and no guard, registry or "
                "watchdog module appears in the diff."
            ),
            "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
            "upstream_review_state": review_state,
            "evaluation_performed": False,
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
    parser.add_argument("--only-nodes", default=None)
    args = parser.parse_args(argv)
    only = (
        tuple(item.strip() for item in args.only_nodes.split(",") if item.strip())
        if args.only_nodes
        else None
    )
    print(json.dumps({
        "queue_manifest": prepare_round0254(
            release_sha=args.release_sha,
            queue_root=args.queue_root,
            only_nodes=only,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
