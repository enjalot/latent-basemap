#!/usr/bin/env python3
"""R0252 node handlers — make a node stoppable at the calls that actually bind.

* `hashpoll_0252` (CPU) — §A. The integrity hash, unpolled and polled, at four
  sizes including a **real 153,600,000,128-byte file**, which is the exact size
  of the 100M x 384 fp32 substrate this program is heading for. R0251 published
  a projection there; this node publishes a measurement. Plus a stop-latency
  control that plants a flag mid-hash and times the stop.
* `panelpoll_0252` (GPU) — §B. `roundreport`'s census puts R0251's binding
  interval in one `score_panel` call at `0.7887487648242341x` the ceiling. This
  node rescores R0218's archived seed-42 checkpoint with and without the new
  `basemap/panel_v2` hook, requires the two arms to agree byte-for-byte with each
  other and with R0218's sealed values, and times a stop mid-score.
* `tail_0252` (GPU) — §C. The per-batch tail at `600,000` updates, sixty times
  R0251's rung, fitted with R0251's unchanged estimator and its pre-registered
  `10.0` identification limit; plus a stop-latency control on the training loop.

Every node scores a real `AbortPollGate` and publishes the score whether it
holds or refuses, on R0250's reporting path: a refusal here IS the measurement.
No node writes the runner's abort flag; every stop-latency control writes its own
flag path, so measuring stoppability can never stop the queue.
"""
from __future__ import annotations

import gc
import json
import os
import random
import resource
import shutil
import statistics
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap import artifact_identity
from basemap import panel_v2 as panel_v2_module
from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0217_minilm_2m_seed_family import BATCH_SIZE
from basemap.round0238_rung5 import json_safe
from basemap.round0242_locality import io_counters, json_scrub
from basemap.round0250_panel_n16 import (
    DIMENSION,
    PANEL_METRICS,
    POOLED_SEEDS,
    REFERENCE_MISMATCH_MESSAGE,
    ROWS,
    corpus_ffr_view,
    panel_execution_ok,
    panel_metric_view,
    raw_purity_ratios,
)
from basemap.round0250_trainer_loops import short_rung_config
from basemap.round0251_rescore import (
    RESCORED_SEED,
    Round0251RescoreError,
    compare_rescore,
    sealed_seed42_cell,
)
from basemap.round0251_trainer_setup import PollRecorder, phase_report
from basemap.round0252_stoppability import (
    HASH_CAPABILITY,
    HASH_SCHEMA,
    INTEGRITY_GUARANTEE,
    NOT_A_FAMILY_CELL,
    PANEL_CAPABILITY,
    PANEL_SCHEMA,
    STOP_CONTROL_DELAY_S,
    STOP_CONTROL_UPDATES,
    TAIL_CAPABILITY,
    TAIL_RUNG_UPDATES,
    TAIL_SCHEMA,
    TARGET_DIMENSION,
    TARGET_ROWS,
    TARGET_SUBSTRATE_BYTES,
    THE_INSTRUMENT_IS_DEFEATABLE,
    Round0252Error,
    declared_sites_match_the_release,
    gap_reduction,
    gap_report,
    measure_stop_latency,
    prior_rung_from_artifact,
    size_law,
    tail_identification,
)
from experiments import round0218_nodes
from experiments.round0113_nodes import _new_model
from experiments.round0230_nodes import _open_substrate, _sealed_graph
from experiments.round0250_nodes import (
    _bound_path,
    _load_centroids,
    _sealed_panel_evidence,
)
from experiments.round0251_nodes import (
    _authenticate_seed42,
    _guard_tail_reported,
    _node_gate,
    _node_guard,
    _receipt_envelope as _r0251_envelope,
    _run_trainer_arm,
    _score_gate_without_raising,
    _seal,
    _start_node,
)


ROUND_ID = "0252"

HASH_ACTION = "round0252_hash_abort_poll"
PANEL_ACTION = "round0252_panel_abort_poll"
TAIL_ACTION = "round0252_long_rung_tail"
ACTIONS: tuple[str, ...] = (HASH_ACTION, PANEL_ACTION, TAIL_ACTION)

ARM_UNPOLLED = "unpolled"
ARM_POLLED = "polled"

#: Scratch for the synthetic 100M-scale file. On `/data`, never `/tmp`; created
#: fresh and removed by the node whether it succeeds or fails.
SCRATCH_ROOT = "/data/tmp/round0252-hash-scratch"

#: 8 MiB, the chunk size `basemap/artifact_identity.py` already used before this
#: round. Named here so the receipt states the unit the gap is bounded by.
HASH_CHUNK_BYTES = 8 << 20

#: Written in blocks of this size so host anonymous memory stays flat while a
#: 143 GiB file is created.
WRITE_BLOCK_BYTES = 256 << 20
#: `fsync` + `POSIX_FADV_DONTNEED` after this much, so the page cache does not
#: absorb the file being written and make the later "cold" read a lie.
WRITE_FLUSH_EVERY_BYTES = 2 << 30


class Round0252NodeError(RuntimeError):
    """The registered R0252 node contract changed."""


def _receipt_envelope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(_r0251_envelope(manifest))
    body["round_id"] = ROUND_ID
    return body


def _evict(path: str) -> dict[str, Any]:
    """Ask the kernel to drop this file's page cache, by ITS OWN path.

    review-0251-01 credited R0251's disclosure that its eviction call was handed
    the 6,679-byte graph *manifest* rather than the 580,136,932-byte edge npz, so
    "cold" meant "asked to be cold" about the wrong file entirely. Here the file
    under measurement is the file evicted.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return {
        "path": path,
        "bytes": os.path.getsize(path),
        "method": "posix_fadvise(POSIX_FADV_DONTNEED) on the measured file itself",
        "note": (
            "POSIX_FADV_DONTNEED is advisory: the kernel may retain pages another "
            "mapping still references, so 'cold' means 'asked to be cold'. A "
            "warmer cache makes the read faster, so an un-evicted arm UNDER-states "
            "the unpolled interval and over-states the margin."
        ),
    }


# --------------------------------------------------------------------------- #
# A. the integrity hash
# --------------------------------------------------------------------------- #


def _write_sized_file(path: str, total_bytes: int, *, seed: int) -> dict[str, Any]:
    """Create a real, fully-allocated file of exactly `total_bytes`.

    Not sparse and not `fallocate`d: an unwritten extent reads back as zeros
    without touching the device, which would make the hash arm faster than any
    real substrate and bias this round's headline in the reassuring direction.
    Every block is written from a pseudorandom buffer.
    """
    rng = np.random.default_rng(seed)
    block = rng.integers(0, 256, size=WRITE_BLOCK_BYTES, dtype=np.uint8).tobytes()
    started = time.monotonic()
    written = 0
    since_flush = 0
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        while written < total_bytes:
            take = min(WRITE_BLOCK_BYTES, total_bytes - written)
            view = block if take == WRITE_BLOCK_BYTES else block[:take]
            offset = 0
            while offset < take:
                offset += os.write(fd, view[offset:])
            written += take
            since_flush += take
            if since_flush >= WRITE_FLUSH_EVERY_BYTES:
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                since_flush = 0
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    observed = os.path.getsize(path)
    if observed != total_bytes:
        raise Round0252NodeError(
            f"R0252 wrote {observed} B to {path}, asked for {total_bytes}"
        )
    return {
        "path": path,
        "bytes": observed,
        "write_wall_s": time.monotonic() - started,
        "sparse": False,
        "allocated_blocks_512b": os.stat(path).st_blocks,
        "fully_allocated": os.stat(path).st_blocks * 512 >= observed,
    }


def _hash_arm(*, arm: str, path: str, label: str, evict: bool) -> dict[str, Any]:
    """One `expected_input_signature` call, with or without the chunk hook."""
    guard = _node_guard(f"{label} {arm}")
    gate = _node_gate(f"{label} {arm}", training_performed=False)
    eviction = _evict(path) if evict else {"path": path, "method": "none"}
    with guard:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0252 hash stage entered")
        previous = (
            artifact_identity.set_abort_poll(recorder) if arm == ARM_POLLED else None
        )
        started = time.monotonic()
        try:
            signature = expected_input_signature(path)
        finally:
            if arm == ARM_POLLED:
                artifact_identity.set_abort_poll(previous)
        wall = time.monotonic() - started
        recorder("R0252 file hashed")
        gate.finish(f"R0252 {arm} hash stage end")
    tail = _guard_tail_reported(guard, label=f"{label} {arm}")
    scored = _score_gate_without_raising(gate, tail, label=f"{label} {arm}")
    report = gap_report(recorder.records, arm=arm)
    size = int(signature["bytes"])
    return {
        "arm": arm,
        "path": path,
        "bytes": size,
        "sha256": str(signature["sha256"]),
        "hash_wall_s": wall,
        "throughput_bytes_per_s": (size / wall) if wall > 0 else None,
        "chunk_bytes": HASH_CHUNK_BYTES,
        "chunks": -(-size // HASH_CHUNK_BYTES),
        "abort_reads_inside_the_hash": max(0, len(recorder.records) - 1),
        "page_cache": eviction,
        "gap_report": report,
        "enforcement_poll_spacing": scored,
        "guard_tail": tail,
    }


def run_hash(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0252NodeError("R0252 hash handler received another queue")
    started = time.monotonic()
    label = "R0252 chunked integrity hash"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0252 hash")
    sites = declared_sites_match_the_release()

    substrate_path = str(dict(job["substrate_signature"])["canonical_path"])
    declared_substrate_sha256 = str(dict(job["substrate_signature"])["sha256"])
    synthetic_sizes = [int(value) for value in job["synthetic_sizes_bytes"]]
    if TARGET_SUBSTRATE_BYTES not in synthetic_sizes:
        raise Round0252NodeError(
            "R0252 hash node must measure the 100M substrate size itself, not "
            f"project to it: {TARGET_SUBSTRATE_BYTES} not in {synthetic_sizes}"
        )

    io_before = io_counters()
    scratch = create_fresh_directory(SCRATCH_ROOT, label="R0252 hash scratch")
    ladder: list[dict[str, Any]] = []
    stop_controls: list[dict[str, Any]] = []
    try:
        # rung 0: the real sealed R0216 substrate, whose declared digest is the
        # ground truth this node's polled arm has to reproduce exactly.
        rungs: list[tuple[str, str, dict[str, Any] | None]] = [
            ("r0216_substrate_3g", substrate_path, None)
        ]
        for size in synthetic_sizes:
            path = os.path.join(scratch, f"synthetic-{size}.bin")
            creation = _write_sized_file(path, size, seed=20260811 + (size % 9973))
            rungs.append((f"synthetic_{size}", path, creation))

        for rung_id, path, creation in rungs:
            unpolled = _hash_arm(arm=ARM_UNPOLLED, path=path, label=label, evict=True)
            polled = _hash_arm(arm=ARM_POLLED, path=path, label=label, evict=True)
            if unpolled["sha256"] != polled["sha256"]:
                raise Round0252NodeError(
                    f"R0252 {rung_id}: the polled hash disagrees with the unpolled "
                    f"one ({polled['sha256']} != {unpolled['sha256']})"
                )
            reduction = gap_reduction(
                before=unpolled["gap_report"], after=polled["gap_report"]
            )
            entry = {
                "rung": rung_id,
                "bytes": int(unpolled["bytes"]),
                "creation": creation,
                "is_the_real_sealed_substrate": creation is None,
                "arms": {ARM_UNPOLLED: unpolled, ARM_POLLED: polled},
                "the_two_arms_agree_on_the_digest": True,
                "reduction": reduction,
            }
            if creation is None:
                entry["matches_the_declared_sealed_digest"] = bool(
                    polled["sha256"] == declared_substrate_sha256
                    and unpolled["sha256"] == declared_substrate_sha256
                )
                if not entry["matches_the_declared_sealed_digest"]:
                    raise Round0252NodeError(
                        "R0252 substrate digest does not match the sealed one"
                    )
            ladder.append(entry)

            if int(unpolled["bytes"]) == TARGET_SUBSTRATE_BYTES:
                # The control that matters: a stop requested in the middle of the
                # 100M-scale hash, the single call R0251 projected at ~26x the
                # ceiling.
                def run_hash_under_poll(poll, _path=path):
                    previous = artifact_identity.set_abort_poll(poll)
                    try:
                        expected_input_signature(_path)
                    finally:
                        artifact_identity.set_abort_poll(previous)

                _evict(path)
                stop_controls.append(measure_stop_latency(
                    label="integrity hash of the 100M-scale substrate",
                    flag_path=os.path.join(scratch, "stop-control.abort"),
                    delay_s=STOP_CONTROL_DELAY_S,
                    run=run_hash_under_poll,
                ))
            if creation is not None:
                os.unlink(path)
    finally:
        if os.path.isdir(scratch):
            for name in os.listdir(scratch):
                try:
                    os.unlink(os.path.join(scratch, name))
                except OSError:
                    pass
            shutil.rmtree(scratch, ignore_errors=True)
    io_after = io_counters()

    polled_points = [
        {"bytes": entry["bytes"], "widest_gap_s": entry["arms"][ARM_POLLED]["gap_report"]["widest_gap_s"]}
        for entry in ladder
    ]
    unpolled_points = [
        {"bytes": entry["bytes"], "widest_gap_s": entry["arms"][ARM_UNPOLLED]["gap_report"]["widest_gap_s"]}
        for entry in ladder
    ]
    at_target = next(
        entry for entry in ladder if entry["bytes"] == TARGET_SUBSTRATE_BYTES
    )
    verdict = {
        "the_hash_is_now_polled_between_chunks": True,
        "chunk_bytes": HASH_CHUNK_BYTES,
        "measured_at_the_100m_substrate_size": True,
        "this_is_a_measurement_not_a_projection": (
            "R0251 published 65.03201434087654 s = 25.89933367479475x the ceiling "
            "as a projection from a 3,072,000,128 B substrate. This node hashed a "
            f"real {TARGET_SUBSTRATE_BYTES} B file -- the exact size of a "
            f"{TARGET_ROWS} x {TARGET_DIMENSION} fp32 substrate -- and reports the "
            "measured interval in both arms. The file's CONTENT is synthetic; "
            "sha256 throughput does not depend on content, and the file sits on "
            "the same volume, filesystem and CPU the real substrate will."
        ),
        "at_the_100m_size": {
            "unpolled_widest_gap_s": at_target["arms"][ARM_UNPOLLED]["gap_report"]["widest_gap_s"],
            "unpolled_widest_gap_over_the_ceiling": at_target["arms"][ARM_UNPOLLED]["gap_report"]["widest_gap_over_the_ceiling"],
            "polled_widest_gap_s": at_target["arms"][ARM_POLLED]["gap_report"]["widest_gap_s"],
            "polled_widest_gap_over_the_ceiling": at_target["arms"][ARM_POLLED]["gap_report"]["widest_gap_over_the_ceiling"],
            "reduction": at_target["reduction"],
        },
        "polled_size_law": size_law(polled_points),
        "unpolled_size_law": size_law(unpolled_points),
        "stop_latency": stop_controls,
        "every_rung_agrees_on_its_digest": True,
        "the_sealed_substrate_digest_reproduces": True,
    }

    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": HASH_SCHEMA,
        "capability": HASH_CAPABILITY,
        "capabilities": [HASH_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "declared_poll_sites": sites,
        "integrity_guarantee": INTEGRITY_GUARANTEE,
        "declared_substrate_sha256": declared_substrate_sha256,
        "ladder": ladder,
        "verdict": verdict,
        "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
        "training_performed": False,
        "gate_registered": False,
        "map_decision_made": False,
        "map_quality_claimed": False,
        "published_a_map": False,
        "is_a_family_cell": False,
        "not_a_family_cell_note": NOT_A_FAMILY_CELL,
        "io": {
            "read_bytes": int(io_after["read_bytes"] - io_before["read_bytes"]),
            "write_bytes": int(io_after["write_bytes"] - io_before["write_bytes"]),
        },
        "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
        "performance": {"total_wall_s": time.monotonic() - started},
    })
    _seal(output, "chunked-integrity-hash.json", body)
    print(json.dumps({
        "capability": HASH_CAPABILITY,
        "polled_at_100m_over_ceiling": verdict["at_the_100m_size"][
            "polled_widest_gap_over_the_ceiling"
        ],
        "unpolled_at_100m_over_ceiling": verdict["at_the_100m_size"][
            "unpolled_widest_gap_over_the_ceiling"
        ],
    }))


# --------------------------------------------------------------------------- #
# B. the scorer
# --------------------------------------------------------------------------- #


def run_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch
    from basemap.panel_v2 import (
        hiD_reference_key,
        load_hiD_reference,
        reset_process_cuda_peak,
        sample_anchors,
        score_panel,
    )
    from basemap.pumap.parametric_umap import ParametricUMAP

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0252NodeError("R0252 panel handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0252NodeError("R0252 score_panel measurement requires CUDA")
    started = time.monotonic()
    label = "R0252 score_panel abort poll"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0252 panel")
    sites = declared_sites_match_the_release()

    sealed = round0218_nodes._sealed_substrate(job)
    source = round0218_nodes._open_substrate(sealed)
    corpus_of_row = round0218_nodes._corpus_of_row(sealed)
    panel_evidence, panel_signature = _sealed_panel_evidence(job)
    pooled = prompt_contract.read_sealed(
        _bound_path(job, "panel_n16", label="R0250 sealed sixteen-cell panel"),
        label="R0250 sealed sixteen-cell panel",
    )
    target = sealed_seed42_cell(panel_evidence, pooled_panel=pooled)
    model_path = _authenticate_seed42(
        dict(panel_evidence["cells"][str(RESCORED_SEED)]), sealed
    )

    reset_process_cuda_peak()
    cfg = prompt_contract.panel_config()
    centroids, centroid_signatures = _load_centroids(panel_evidence)
    data_identity = {
        "kind": "ordered_array",
        "shape": [ROWS, DIMENSION],
        "dtype": np.dtype("<f4").str,
        "sha256": sealed["ordered_substrate_sha256"],
    }
    reference_identity = {
        "data_identity": data_identity,
        "convention": dict(round0218_nodes.REFERENCE_CONVENTION),
    }
    reference_signature = dict(panel_evidence["shared_high_d_reference"])
    reference_path = prompt_contract.verify_signature(
        reference_signature, label="R0218 shared high-D reference"
    )
    if expected_input_signature(reference_path) != reference_signature:
        raise Round0251RescoreError(f"{REFERENCE_MISMATCH_MESSAGE} file signature drift")
    reference = load_hiD_reference(reference_path)
    anchors = sample_anchors(ROWS, cfg)
    if not np.array_equal(
        np.asarray(anchors, dtype=np.int64),
        np.asarray(reference["anchor_ids"], dtype=np.int64),
    ):
        raise Round0251RescoreError(f"{REFERENCE_MISMATCH_MESSAGE} anchor drift")
    rederived_key, _parts = hiD_reference_key(
        source, anchors, cfg, centroids, kf=int(reference["kf"]), **reference_identity
    )
    if str(rederived_key) != str(reference["key"]):
        raise Round0251RescoreError(f"{REFERENCE_MISMATCH_MESSAGE} re-derived key drift")
    anchor_labels = round0218_nodes._anchor_corpus_labels(corpus_of_row, anchors)

    model = ParametricUMAP.load(model_path, device="cuda")
    coordinates = np.asarray(model.transform(source, batch_size=8192), dtype=np.float32)
    if coordinates.shape != (ROWS, 2) or not np.isfinite(coordinates).all():
        raise Round0252NodeError("R0252 seed-42 transform is not a finite 2M x 2 array")
    coordinates_path = os.path.join(output, f"coordinates-seed{RESCORED_SEED}.npy")
    atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
    coordinates_signature = expected_input_signature(coordinates_path)
    observed_coordinates_sha256 = ordered_array_sha256(coordinates)
    del model
    torch.cuda.empty_cache()
    gc.collect()

    def _score(provenance_note: str):
        return score_panel(
            source,
            coordinates,
            config=cfg,
            centroids_by_k=centroids,
            hiD_reference=reference,
            reference_identity=reference_identity,
            ffr_group_labels=anchor_labels,
            scale_admission=None,
            provenance={
                "round_id": ROUND_ID,
                "seed": RESCORED_SEED,
                "capability": str(target["capability"]),
                "universe": "R0216-queue-correction-3-minilm-mixed-2m",
                "substrate": dict(sealed["substrate_signature"]),
                "provenance_array": dict(sealed["provenance_signature"]),
                "coordinates": coordinates_signature,
                "shared_high_d_reference": reference_signature,
                "reference_source_round": "0218",
                "rescore_of_round": str(target["source_round"]),
                "arm": provenance_note,
            },
        )

    arms: dict[str, Any] = {}
    panels: dict[str, Any] = {}
    for arm in (ARM_UNPOLLED, ARM_POLLED):
        guard = _node_guard(f"{label} {arm}")
        gate = _node_gate(f"{label} {arm}", training_performed=False)
        with guard:
            gate.start()
            recorder = PollRecorder(gate=gate, clock=time.monotonic)
            recorder.anchor("R0252 panel stage entered")
            previous = (
                panel_v2_module.set_abort_poll(recorder) if arm == ARM_POLLED else None
            )
            call_started = time.monotonic()
            try:
                panels[arm] = _score(arm)
            finally:
                if arm == ARM_POLLED:
                    panel_v2_module.set_abort_poll(previous)
            call_wall = time.monotonic() - call_started
            recorder("R0252 panel scored")
            gate.finish(f"R0252 {arm} panel stage end")
        tail = _guard_tail_reported(guard, label=f"{label} {arm}")
        scored = _score_gate_without_raising(gate, tail, label=f"{label} {arm}")
        if not panel_execution_ok(panels[arm]):
            raise Round0252NodeError(f"R0252 {arm} rescored panel is collapsed")
        if not bool(panels[arm]["provenance"]["hiD_reference_reused"]):
            raise Round0251RescoreError(
                f"{REFERENCE_MISMATCH_MESSAGE} seed-42 recomputed the reference"
            )
        arms[arm] = {
            "arm": arm,
            "score_panel_wall_s": call_wall,
            "abort_reads_inside_score_panel": max(0, len(recorder.records) - 1),
            "gap_report": gap_report(recorder.records, arm=arm),
            "enforcement_poll_spacing": scored,
            "guard_tail": tail,
        }

    reduction = gap_reduction(
        before=arms[ARM_UNPOLLED]["gap_report"], after=arms[ARM_POLLED]["gap_report"]
    )

    # The science-path proof: the two arms must agree with each other on every
    # published value, and both must agree with R0218's sealed cell.
    comparisons = {
        arm: compare_rescore(
            sealed=target,
            observed_panel_metrics=panel_metric_view(panels[arm]),
            observed_ratios=raw_purity_ratios(panels[arm]),
            observed_hi_d_agreement={
                key: float(panels[arm]["purity_numerators"][key]["hi_D_agreement"])
                for key in ("k256", "k1024")
            },
            observed_corpus_ffr=corpus_ffr_view(panels[arm]),
            observed_coordinates_sha256=observed_coordinates_sha256,
        )
        for arm in (ARM_UNPOLLED, ARM_POLLED)
    }
    arm_agreement = {
        "panel_metrics_identical": (
            panel_metric_view(panels[ARM_UNPOLLED]) == panel_metric_view(panels[ARM_POLLED])
        ),
        "raw_ratios_identical": (
            raw_purity_ratios(panels[ARM_UNPOLLED]) == raw_purity_ratios(panels[ARM_POLLED])
        ),
        "corpus_ffr_identical": (
            corpus_ffr_view(panels[ARM_UNPOLLED]) == corpus_ffr_view(panels[ARM_POLLED])
        ),
        "purity_numerators_identical": (
            panels[ARM_UNPOLLED]["purity_numerators"] == panels[ARM_POLLED]["purity_numerators"]
        ),
        "guards_identical": panels[ARM_UNPOLLED]["guards"] == panels[ARM_POLLED]["guards"],
    }
    if not all(arm_agreement.values()):
        raise Round0252NodeError(
            f"R0252 the abort-poll hook moved a panel value: {arm_agreement}"
        )
    execution_checks = {
        "the_hook_changed_no_panel_value": all(arm_agreement.values()),
        "both_arms_reproduce_r0218": all(
            int(comparisons[arm]["values_drifted"]) == 0
            and int(comparisons[arm]["values_compared"]) > 0
            for arm in comparisons
        ),
        "the_polled_arm_read_the_flag_inside_score_panel": (
            arms[ARM_POLLED]["abort_reads_inside_score_panel"] > 0
        ),
        "the_unpolled_arm_read_nothing_inside_score_panel": (
            arms[ARM_UNPOLLED]["abort_reads_inside_score_panel"] == 0
        ),
        "the_transform_covered_every_row": int(coordinates.shape[0]) == ROWS,
    }
    if not all(execution_checks.values()):
        raise Round0252NodeError(f"R0252 panel execution checks failed: {execution_checks}")

    def run_score_under_poll(poll):
        previous = panel_v2_module.set_abort_poll(poll)
        try:
            _score("stop_latency_control")
        finally:
            panel_v2_module.set_abort_poll(previous)

    stop = measure_stop_latency(
        label="score_panel on the sealed 2M substrate",
        flag_path=os.path.join(output, "stop-control.abort"),
        delay_s=float(job.get("panel_stop_delay_s", 0.5)),
        run=run_score_under_poll,
    )

    verdict = {
        "the_scorer_now_reads_the_abort_flag": True,
        "declared_poll_sites": list(panel_v2_module.ABORT_POLL_SITES),
        "widest_gap_unpolled_s": arms[ARM_UNPOLLED]["gap_report"]["widest_gap_s"],
        "widest_gap_unpolled_over_the_ceiling": arms[ARM_UNPOLLED]["gap_report"]["widest_gap_over_the_ceiling"],
        "widest_gap_polled_s": arms[ARM_POLLED]["gap_report"]["widest_gap_s"],
        "widest_gap_polled_over_the_ceiling": arms[ARM_POLLED]["gap_report"]["widest_gap_over_the_ceiling"],
        "reduction": reduction,
        "what_was_inside_score_panel": arms[ARM_POLLED]["gap_report"]["gaps_by_site"],
        "the_widest_site_inside_score_panel": arms[ARM_POLLED]["gap_report"]["widest_gap_after"],
        "stop_latency": stop,
        "the_science_path_is_unchanged": arm_agreement,
    }

    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": PANEL_SCHEMA,
        "capability": PANEL_CAPABILITY,
        "capabilities": [PANEL_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "declared_poll_sites": sites,
        "rescored_seed": RESCORED_SEED,
        "r0218_sealed_cell": target,
        "rescored_panel": panels[ARM_POLLED],
        "rescored_coordinates": coordinates_signature,
        "rescored_coordinates_ordered_sha256": observed_coordinates_sha256,
        "comparisons": comparisons,
        "arms": arms,
        "arm_agreement": arm_agreement,
        "reduction": reduction,
        "stop_latency": stop,
        "verdict": verdict,
        "execution_checks": execution_checks,
        "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
        "sources": {
            "r0218_panel": panel_signature,
            "r0250_panel_n16": expected_input_signature(
                _bound_path(job, "panel_n16", label="R0250 panel")
            ),
            "centroids": centroid_signatures,
            "shared_high_d_reference": reference_signature,
        },
        "training_performed": False,
        "gate_registered": False,
        "map_decision_made": False,
        "map_quality_claimed": False,
        "published_a_map": False,
        "is_a_family_cell": False,
        "not_a_family_cell_note": NOT_A_FAMILY_CELL,
        "performance": {
            "total_wall_s": time.monotonic() - started,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
        },
    })
    _seal(output, "score-panel-abort-poll.json", body)
    print(json.dumps({
        "capability": PANEL_CAPABILITY,
        "widest_gap_polled_over_ceiling": verdict["widest_gap_polled_over_the_ceiling"],
        "widest_gap_unpolled_over_ceiling": verdict["widest_gap_unpolled_over_the_ceiling"],
        "stop_latency_s": stop["stop_latency_s"],
    }))


# --------------------------------------------------------------------------- #
# C. the tail
# --------------------------------------------------------------------------- #


def _control_fit(*, config: Mapping[str, Any], job: Mapping[str, Any], updates: int,
                 flag_path: str, delay_s: float) -> dict[str, Any]:
    """A short fit interrupted by a flag file, timed end to end."""
    import torch

    from basemap.round0217_minilm_2m_pipeline import (
        MiniLMHostFp32EndpointArray,
        MiniLMMixedTrainingInput,
    )

    graph = _sealed_graph(job)
    source, substrate_signature = _open_substrate(graph)
    seed = int(config["seed"])
    dataset = MiniLMHostFp32EndpointArray(
        source, source_signature=substrate_signature, buffer_rows=BATCH_SIZE
    )
    wrapper = MiniLMMixedTrainingInput(dataset, graph, seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _new_model(config)
    model._max_train_steps = updates
    model._bench_warmup = max(updates // 4, 1)
    model._perf_profile = True
    model._perf_floor = config["execution"]["minimum_train_upd_s"]
    model._perf_warn_rate = config["execution"]["warning_train_upd_s"]
    model._perf_subfloor_patience = 2
    model._abort_on_first_nonfinite = True

    def run(poll):
        model.abort_poll = poll
        try:
            model.fit(
                wrapper,
                low_memory=True,
                verbose=False,
                n_processes=6,
                random_state=seed,
                resample_negatives=False,
                precomputed_edges_path=graph["signature"]["canonical_path"],
                use_wandb=False,
            )
        finally:
            model.abort_poll = None

    outcome = measure_stop_latency(
        label="ParametricUMAP.fit train loop",
        flag_path=flag_path,
        delay_s=float(delay_s),
        run=run,
    )
    outcome["updates_offered"] = int(updates)
    outcome["updates_completed_before_the_stop"] = int(
        dict(getattr(model, "_train_stats", {}) or {}).get(
            "optimizer_steps_succeeded", -1
        )
    )
    del model, wrapper, dataset, source, graph
    torch.cuda.empty_cache()
    gc.collect()
    return outcome


def run_tail(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    from basemap.round0217_minilm_2m_seed_family import train_config as r0217_train_config

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0252NodeError("R0252 tail handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0252NodeError("R0252 long-rung tail measurement requires CUDA")
    started = time.monotonic()
    label = "R0252 long-rung batch tail"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0252 tail")
    sites = declared_sites_match_the_release()

    template_graph = _sealed_graph(job)
    template_source, template_signature = _open_substrate(template_graph)
    template, _sha = r0217_train_config(
        seed=int(job["template_seed"]),
        graph_signature=template_graph["signature"],
        graph_manifest_signature=template_graph["manifest_signature"],
        substrate_signature=template_signature,
        graph_edges=template_graph["directed_edges"],
        rows=ROWS,
    )
    del template_graph, template_source
    gc.collect()

    tail_updates = int(job.get("tail_rung_updates", TAIL_RUNG_UPDATES))
    control_updates = int(job.get("stop_control_updates", STOP_CONTROL_UPDATES))
    tail_config, tail_sha, tail_identity = short_rung_config(template, updates=tail_updates)
    control_config, control_sha, control_identity = short_rung_config(
        template, updates=control_updates
    )
    atomic_write_new_json(
        os.path.join(output, "long-rung-configs.json"),
        {
            "round_id": ROUND_ID,
            "capability": TAIL_CAPABILITY,
            "tail_rung": {"config": tail_config, "sha256": tail_sha, "identity": tail_identity},
            "stop_control_rung": {
                "config": control_config, "sha256": control_sha, "identity": control_identity
            },
        },
        immutable=True,
    )

    io_before = io_counters()
    # The stop-latency control runs FIRST, so a stoppability failure costs
    # seconds rather than an hour and a half of GPU.
    stop = _control_fit(
        config=control_config,
        job=job,
        updates=control_updates,
        flag_path=os.path.join(output, "stop-control.abort"),
        delay_s=float(job.get("trainer_stop_delay_s", STOP_CONTROL_DELAY_S)),
    )
    if not bool(stop["the_work_stopped_cooperatively"]):
        raise Round0252NodeError(
            "R0252 trainer stop-latency control did not stop the fit; the long "
            "rung is not launched"
        )

    guard = _node_guard(f"{label} tail")
    gate = _node_gate(f"{label} tail", training_performed=True)
    with guard:
        gate.start()
        arm = _run_trainer_arm(
            arm="tail", config=tail_config, job=job, updates=tail_updates,
            gate=gate, evict=False,
        )
        guard.poll("R0252 tail fit complete")
        gate.finish("R0252 tail stage end")
    guard_tail = _guard_tail_reported(guard, label=f"{label} tail")
    scored = _score_gate_without_raising(gate, guard_tail, label=f"{label} tail")
    io_after = io_counters()

    prior = prior_rung_from_artifact(
        prompt_contract.read_sealed(
            _bound_path(job, "r0251_trainer_setup", label="R0251 sealed trainer setup"),
            label="R0251 sealed trainer setup",
        )
    )
    identification = tail_identification(
        arm["batch_gaps"], arm_wall_s=float(arm["stage_wall_s"]), prior_rung=prior
    )
    gaps = np.asarray(arm["batch_gaps"], dtype=np.float64)
    np.save(os.path.join(output, "tail-batch-gaps.npy"), gaps)

    verdict = {
        "rung_updates": tail_updates,
        "batch_multiple_over_r0251": tail_updates / max(1, int(prior["batches"])),
        "steady_updates_per_s": arm["steady_updates_per_s"],
        "observed_max_batch_gap_s": float(gaps.max()),
        "observed_median_batch_gap_s": float(np.median(gaps)),
        "widest_gap_across_both_phases_s": arm["phase_report"]["widest_gap_across_both_phases_s"],
        "widest_gap_across_both_phases_over_the_ceiling": arm["phase_report"][
            "widest_gap_across_both_phases_over_the_ceiling"
        ],
        "the_binding_phase": arm["phase_report"]["the_binding_phase"],
        "tail": identification,
        "stop_latency": stop,
        "the_instrument_is_r0251s_unchanged": (
            "the arm, the recorder, the phase split and the tail estimator are "
            "R0251's own functions, imported rather than re-typed, so the only "
            "thing this rung changes is the number of batches. The setup site "
            "names still read 'R0251 node ...' because they ARE R0251's constants."
        ),
    }

    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": TAIL_SCHEMA,
        "capability": TAIL_CAPABILITY,
        "capabilities": [TAIL_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "declared_poll_sites": sites,
        "long_rungs": {"tail": tail_identity, "stop_control": control_identity},
        "long_rung_config_sha256": {"tail": tail_sha, "stop_control": control_sha},
        "arm": {key: value for key, value in arm.items() if key != "batch_gaps"},
        "batch_gap_series_summary": {
            "count": int(gaps.size),
            "min_s": float(gaps.min()),
            "median_s": float(np.median(gaps)),
            "max_s": float(gaps.max()),
            "mean_s": float(gaps.mean()),
        },
        "enforcement_poll_spacing": {"tail": scored},
        "guard_tails": {"tail": guard_tail},
        "prior_rung": prior,
        "tail_identification": identification,
        "stop_latency": stop,
        "verdict": verdict,
        "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
        "training_performed": True,
        "gate_registered": False,
        "map_decision_made": False,
        "map_quality_claimed": False,
        "published_a_map": False,
        "is_a_family_cell": False,
        "not_a_family_cell_note": NOT_A_FAMILY_CELL,
        "io": {
            "read_bytes": int(io_after["read_bytes"] - io_before["read_bytes"]),
            "write_bytes": int(io_after["write_bytes"] - io_before["write_bytes"]),
        },
        "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
        "performance": {
            "total_wall_s": time.monotonic() - started,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
        },
    })
    _seal(output, "long-rung-batch-tail.json", body)
    print(json.dumps({
        "capability": TAIL_CAPABILITY,
        "batches": int(gaps.size),
        "tail_identified": identification["tail_verdict"]["the_extreme_value_fit_is_identified"],
        "threshold_ladder_spread": identification["tail_verdict"][
            "threshold_ladder_return_level_spread"
        ],
        "stop_latency_s": stop["stop_latency_s"],
    }))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == HASH_ACTION:
        run_hash(active, job)
        return
    if action == PANEL_ACTION:
        run_panel(active, job)
        return
    if action == TAIL_ACTION:
        run_tail(active, job)
        return
    raise Round0252NodeError(f"R0252 has no handler for action {action!r}")
