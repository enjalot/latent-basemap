#!/usr/bin/env python3
"""Prepare, but never launch, the R0267 queue — the 50M ×2 host-int8 staging rung.

Five nodes in one queue, in this order:

1-3. `train_minilm_fneg_50m_x2_hostint8` × 3 (GPU, seeds 42/43/44) — the promoted fneg
     recipe at N=50M, dose ×2, on the host-int8 X path P5 (R0266) validated, built from
     R0265's template retargeted to the sealed R0237 50M substrate + k15 graph and proved
     by `round0267_int8_treatment.assert_registered_50m_int8_recipe`.
4.   `score_minilm_fneg_50m_x2_panel` (GPU) — the three maps scored on R0265's instruments
     (held-out FFR against the sealed R0237 exact k15 truth, purity k256/k1024 against
     R0218's frozen reference, collapse, fog).
5.   `register_fneg_50m_x2_seedmean_gate` (CPU) — the pre-registered 50M gate: the SEED-MEAN
     collapse inside P1's ×2 asymptote band widened by z·σ_fam/√n, plus per-seed backstops.
     Every band/floor/σ_fam/P1-edge is bound by sha256 and read/recomputed at gate time.

This builder REFUSES until the round file is issued (status: issued, base_commit an
ancestor of the release) and every bound input exists — including the sealed R0237 50M
substrate + graph manifests, the R0237 reserve + exact truth, R0218's frozen panel, R0265's
sealed family floors + n=13 panel, and the sealed P1 analysis-v2 result. It builds and
proves the three ×2 configs HERE, seals the import-source closure (R0266's + the R0267 50M
recipe module), and binds every sealed cross-round input by real sha256 + bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0218_minilm_2m_panel import CAPABILITY as R0218_PANEL_CAPABILITY, CENTROID_KS
from basemap.round0247_registry import registry_fingerprint
from basemap.round0254_dispatch import (
    SCOPE_MODULES,
    assert_derived_entries_install,
    dispatch_census,
    entry_tuples,
    gate_census,
    scope_residual,
)
from basemap import round0267_int8_treatment as T
from basemap.round0267_int8_treatment import (
    CANONICAL_SEED,
    CLOSURE_SCHEMA,
    ROUND_ID,
    ROWS,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    TRAIN_CLOSURE_MODULES,
    assert_registered_50m_int8_recipe,
    capability_for_seed,
    fneg_seed_invariant_sha256,
    int8_50m_train_config,
    runtime_closure_hashes,
)
from basemap.round0217_minilm_2m_seed_family import successful_updates_for_edges
from experiments.round0267_nodes import (
    GATE_ACTION,
    GATE_CAPABILITY,
    PANEL_ACTION,
    PANEL_CAPABILITY,
    SALVAGE_REASON,
    SALVAGE_SEED,
    SALVAGE_SEED42_COORDINATES_SHA256,
    SALVAGE_SOURCE_RUN,
    TRAIN_ACTION,
)
import experiments.round0265_nodes as R0265N
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
# Reuse R0265/R0266's data anchors for the 2M frozen panel + the R0265 sealed instruments.
from experiments.prepare_round0265_queue import R0218_PANEL
from experiments.prepare_round0266_queue import (
    R0265_FLOORS,
    R0265_PANEL,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0267"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0267-2026-08-15.md")

#: The sealed R0237 50M substrate + graph manifests (the nested-prefix ladder's carve).
R0237_SUBSTRATE_MANIFEST = (
    "/data/latent-basemap/runs/round-0237/queue/artifacts/"
    "minilm-mixed-50000k-nested-substrate-and-reserves-v1/substrate.json"
)
R0237_GRAPH_MANIFEST = (
    "/data/latent-basemap/runs/round-0237/queue-correction-1/artifacts/"
    "minilm-mixed-50000k-cluster-spill-k15-fuzzy-graph-v1/qualified-graph.json"
)
#: The sealed R0237 50M exact k15 truth (uniform 1M probe rows into the substrate + ids).
R0237_TRUTH_QUERY_ROWS = (
    "/data/latent-basemap/runs/round-0237/queue/artifacts/"
    "minilm-mixed-50000k-uniform-probe-k15-truth-v1/probe-query-rows.i64.npy"
)
R0237_TRUTH_IDS = (
    "/data/latent-basemap/runs/round-0237/queue/artifacts/"
    "minilm-mixed-50000k-uniform-probe-k15-truth-v1/truth-k15-ids.i32.npy"
)
#: The sealed R0237 200k held-out reserve embeddings (held-out reserve lineage).
R0237_RESERVE = (
    "/data/latent-basemap/runs/round-0237/queue/artifacts/"
    "minilm-mixed-50000k-nested-substrate-and-reserves-v1/reserve.f32.npy"
)
#: The frozen P1 analysis-v2 result — the ×2 collapse asymptote band (plain JSON).
P1_ASYMPTOTE = "/data/latent-basemap/sandbox/logs/analysis_v2_result.json"

# --------------------------------------------------------------------------- #
# SALVAGE (correction-4): bind seed42's clean correction-3 artifacts as a
# completed cell instead of re-training the 11.9 GPU-h.
#
# seed42 (50M ×2 host-int8) trained the FULL ×2 horizon cleanly in correction-3
# (4,162,228 updates, 0 AMP-skips, 0 nonfinite — proven by the train log) and
# saved its map, then the node died on the too-tight 60 GiB post-hoc RSS
# assertion (peak 75.81 GiB). The delegate ruled: raise the limit, BIND seed42's
# saved artifacts as a completed cell, re-run only seeds 43/44. seed42 has NO
# train-receipt (the RSS raise preempted it), so the salvage records the clean
# full-horizon train from the log's step-accounting line in lieu of a receipt.
# These are the SEALED correction-3 digests, verified at bind time.
# --------------------------------------------------------------------------- #
SALVAGE_EXPECTED_DIGESTS = {
    "coordinates.npy": SALVAGE_SEED42_COORDINATES_SHA256,
    "model.pt": "6151293536f80a9a2a728a6f62dc0af388d7aba6d1c3cc14675576db18a5a065",
    "admission.json": "53e7778805647a37616847667864f29d120cff56502a8608adc9353f25241b48",
    "production-config.json": (
        "eebd4d8c483c1780b199fb6d45dbd5ba85853e402f64a809346ee18e7ee41544"
    ),
}
SALVAGE_EXPECTED_LOG_SHA256 = (
    "c310d487ba0f16c28027706b3cfa757c0da00fef7023390ae84eefbedffeda56"
)
#: The stable PREFIX that locates the step-accounting line — no literal update count, so
#: the dose is checked by DERIVATION (validate_dose_x2), never by a magic number in a marker.
SALVAGE_STEP_ACCOUNTING_MARKER = "Step accounting (lr_horizon):"
#: Parses the clean-train counters out of the located line (succeeded / AMP-skips / nonfinite).
SALVAGE_STEP_ACCOUNTING_RE = re.compile(
    r"succeeded\s+(?P<succeeded>\d+),\s*AMP-skips\s+(?P<amp_skips>\d+),\s*"
    r"nonfinite\s+loss/grad\s+(?P<nonfinite_loss>\d+)/(?P<nonfinite_grad>\d+)"
)


def _extract_clean_train_step_line(log_path: str) -> tuple[str, int]:
    """The last step-accounting line from the (streamed) train log + its succeeded count.

    Locates the line by a stable prefix (no literal update count), parses its counters, and
    asserts a CLEAN full-horizon train: 0 AMP-skips and 0 nonfinite loss/grad. The succeeded
    count is returned so the caller can check the ×2 dose equality by derivation
    (validate_dose_x2), not against a marker literal. The 600 MB log is scanned line-by-line,
    never materialised.
    """
    found: str | None = None
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if SALVAGE_STEP_ACCOUNTING_MARKER in line:
                found = line.strip()
    if not found:
        raise RuntimeError(
            f"R0267 salvage: no step-accounting line in {log_path} "
            f"(expected prefix {SALVAGE_STEP_ACCOUNTING_MARKER!r})"
        )
    match = SALVAGE_STEP_ACCOUNTING_RE.search(found)
    if match is None:
        raise RuntimeError(
            f"R0267 salvage: step-accounting line has no succeeded/AMP-skips/nonfinite "
            f"counters: {found!r}"
        )
    amp_skips = int(match.group("amp_skips"))
    nonfinite_loss = int(match.group("nonfinite_loss"))
    nonfinite_grad = int(match.group("nonfinite_grad"))
    if amp_skips != 0 or nonfinite_loss != 0 or nonfinite_grad != 0:
        raise RuntimeError(
            f"R0267 salvage: step-accounting line is not a clean train "
            f"(AMP-skips={amp_skips}, nonfinite loss/grad={nonfinite_loss}/{nonfinite_grad}): "
            f"{found!r}"
        )
    return found, int(match.group("succeeded"))


def bind_salvaged_seed42_cell(
    *,
    artifact_dir: str,
    log_path: str,
    expected_cell_invariant: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Bind seed42's sealed correction-3 artifacts as a completed cell (verify digests).

    Returns ``(cell_record, salvaged_invariant)``. Every artifact's file digest is verified
    against the sealed correction-3 pins (mismatch -> raise), the clean-train step-accounting
    line is extracted from the log as the train evidence, and the recipe invariant is read
    from the bound production-config.json (NOT a fabricated train-receipt). If
    ``expected_cell_invariant`` is given, the salvaged invariant must equal it — proving the
    salvaged cell is treatment-identical to the freshly-built 43/44 cells.
    """
    capability = T.capability_for_seed(SALVAGE_SEED)
    signatures: dict[str, dict[str, Any]] = {}
    for name, expected in SALVAGE_EXPECTED_DIGESTS.items():
        sig = _signature(os.path.join(artifact_dir, name), f"R0267 salvaged seed42 {name}")
        if sig["sha256"] != expected:
            raise RuntimeError(
                f"R0267 salvage digest mismatch for {name}: {sig['sha256']} != {expected}"
            )
        signatures[name] = sig
    log_sig = _signature(log_path, "R0267 salvaged seed42 train log")
    if log_sig["sha256"] != SALVAGE_EXPECTED_LOG_SHA256:
        raise RuntimeError(
            f"R0267 salvage log digest mismatch: {log_sig['sha256']} != "
            f"{SALVAGE_EXPECTED_LOG_SHA256}"
        )
    step_line, succeeded_updates = _extract_clean_train_step_line(log_sig["canonical_path"])
    # Requirement (b): the ×2 dose is a CHECKED EQUALITY against a DERIVED target, not the
    # literal 4,162,228 — validate_dose_x2 raises unless the log's succeeded count equals
    # 2 * successful_updates_for_edges(SEALED_DIRECTED_EDGES) under the quantisation bound.
    dose_receipt = T.validate_dose_x2(
        updates=succeeded_updates, edge_count=SEALED_DIRECTED_EDGES
    )

    with open(signatures["production-config.json"]["canonical_path"], encoding="utf-8") as handle:
        prod_config = json.load(handle)
    salvaged_invariant = str(prod_config.get("seed_invariant_sha256") or "")
    if not salvaged_invariant:
        raise RuntimeError("R0267 salvage: production-config.json carries no seed_invariant_sha256")
    if str(prod_config.get("round_id")) != ROUND_ID or int(prod_config.get("seed", -1)) != SALVAGE_SEED:
        raise RuntimeError("R0267 salvage: production-config.json is not the seed42 R0267 cell")
    if expected_cell_invariant is not None and salvaged_invariant != expected_cell_invariant:
        raise RuntimeError(
            "R0267 salvage: seed42's saved config is NOT treatment-identical to the "
            f"correction-4 cells: {salvaged_invariant} != {expected_cell_invariant}"
        )

    cell_record = {
        "seed": SALVAGE_SEED,
        "capability": capability,
        "salvaged": True,
        "salvage": {
            "salvaged": True,
            "source_run": SALVAGE_SOURCE_RUN,
            "reason": SALVAGE_REASON,
            "train_evidence": {
                "train_log": log_sig,
                "train_log_sha256": log_sig["sha256"],
                "step_accounting_line": step_line,
                "succeeded_updates": succeeded_updates,
                "dose_receipt": dose_receipt,
                "dose_checked_equality": (
                    f"succeeded {succeeded_updates} == {T.DOSE_MULTIPLIER} * "
                    f"successful_updates_for_edges({SEALED_DIRECTED_EDGES}) = "
                    f"{T.DOSE_MULTIPLIER} * {dose_receipt['base_successful_updates']} = "
                    f"{dose_receipt['successful_updates']} (validate_dose_x2, quantisation-bounded)"
                ),
                "in_lieu_of": (
                    "there is no seed42 train-receipt: the too-tight RSS assertion raised "
                    "AFTER the clean full-horizon train saved the map, preempting the "
                    "receipt. The step-accounting line + the derived dose equality are the "
                    "clean-train evidence."
                ),
            },
            "digests": {name: sig["sha256"] for name, sig in signatures.items()},
            "seed_invariant_sha256": salvaged_invariant,
            "production_config_substrate_sha256": (
                str(((prod_config.get("config") or {}).get("input") or {}).get("substrate_sha256") or "")
            ),
        },
        "coordinates": signatures["coordinates.npy"],
        "model": signatures["model.pt"],
        "admission": signatures["admission.json"],
        "production_config": signatures["production-config.json"],
        "train_log": log_sig,
    }
    return cell_record, salvaged_invariant

#: Three 50M ×2 host-int8 trains (~5 GPU-h each) + a three-map 50M panel + the CPU gate;
#: the cap sits above the ~15 GPU-h train estimate + panel/reference (plan costs).
GPU_HOURS_CAP = 24.0
TRAIN_P90_WALL_S = 21_600.0
PANEL_P90_WALL_S = 7_200.0
GATE_P90_WALL_S = 600.0


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
        raise RuntimeError(
            "R0267 round is not issued for this release. EXPECTED until the round file is "
            "issued and its base_commit is an ancestor of the release."
        )
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0267 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _upstream_review_state(required: list[str]) -> dict[str, Any]:
    import glob

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
        "note": (
            "Review is post-hoc: it blocks the downstream claim, not the launch. The 50M "
            "PASS/FAIL this round registers is registered-and-contingent until its required "
            "upstream reviews are accepted; the 100M ×2 commit is gated on a 50M pass."
        ),
    }


def _signature(path: str, label: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise RuntimeError(f"R0267 bound input absent: {label} at {path}")
    return expected_input_signature(path)


def _treatment_closure_seal(release_sha: str) -> dict[str, Any]:
    """SHA-256 of every R0267 training-closure source (R0266's closure + the 50M module)."""
    observed = runtime_closure_hashes(TRAIN_CLOSURE_MODULES)
    files: dict[str, Any] = {}
    for name, entry in observed.items():
        relative = os.path.relpath(
            entry["path"], os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        files[name] = {
            "module": name,
            "path": relative,
            "bytes_at_release": entry["bytes"],
            "sha256_at_release": entry["sha256"],
        }
    return prompt_contract.seal({
        "schema": CLOSURE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "modules": list(TRAIN_CLOSURE_MODULES),
        "files": files,
        "how_to_read_this": (
            "R0267 is R0265's fneg recipe at 50M ×2 on R0266's host-int8 path. This seals "
            "the R0267 training import closure: R0266's closure (the fneg-merged core + the "
            "int8 routing module) PLUS the round0267 50M ×2 recipe module. A node that ran "
            "different bytes for any of them would refuse."
        ),
    })


def prepare_round0267(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    salvage: Mapping[str, Any] | None = None,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0267 release SHA must be one full commit")
    if salvage is not None and int(salvage.get("seed", -1)) != SALVAGE_SEED:
        raise ValueError(f"R0267 salvage supports only seed {SALVAGE_SEED}")
    round_signature, required_reviews = _issued_round(release_sha)
    review_state = _upstream_review_state(list(required_reviews))

    # Bind the sealed R0237 50M substrate + graph manifests, read their inner signatures.
    substrate_manifest_signature = _signature(R0237_SUBSTRATE_MANIFEST, "R0237 50M substrate manifest")
    graph_manifest_signature = _signature(R0237_GRAPH_MANIFEST, "R0237 50M graph manifest")
    substrate_manifest = prompt_contract.read_sealed(
        substrate_manifest_signature["canonical_path"], label="R0237 50M substrate manifest"
    )
    graph_manifest = prompt_contract.read_sealed(
        graph_manifest_signature["canonical_path"], label="R0237 50M graph manifest"
    )
    if (
        substrate_manifest.get("capability") != T.R0237_SUBSTRATE_CAPABILITY
        or int(substrate_manifest.get("rows", -1)) != ROWS
        or str(substrate_manifest.get("ordered_substrate_sha256")) != T.R0237_SUBSTRATE_ORDERED_SHA256
    ):
        raise RuntimeError("R0267 --substrate manifest is not the sealed R0237 50M substrate")
    if (
        graph_manifest.get("capability") != T.R0237_GRAPH_CAPABILITY
        or int(graph_manifest.get("rows", -1)) != ROWS
        or int(graph_manifest.get("directed_edges", -1)) != SEALED_DIRECTED_EDGES
    ):
        raise RuntimeError("R0267 --graph manifest is not the sealed R0237 50M k15 graph")
    substrate_signature = dict(substrate_manifest["substrate"])
    graph_signature = dict(graph_manifest["graph"])
    edges = int(graph_manifest["directed_edges"])
    base_horizon = successful_updates_for_edges(edges)

    # Build and prove the three ×2 cell configs HERE (needs no train artifacts). All three
    # share ONE masked seed-invariant digest (the recipe outside the seed).
    invariants: set[str] = set()
    per_seed_config_sha: dict[str, str] = {}
    recipe = None
    for seed in SEEDS:
        config, config_sha = int8_50m_train_config(
            seed=seed,
            graph_signature=graph_signature,
            graph_manifest_signature=graph_manifest_signature,
            substrate_signature=substrate_signature,
            graph_edges=edges,
            rows=ROWS,
        )
        recipe = assert_registered_50m_int8_recipe(config)
        invariants.add(fneg_seed_invariant_sha256(config))
        per_seed_config_sha[str(seed)] = config_sha
    if len(invariants) != 1:
        raise RuntimeError("R0267 three cells do not share one masked recipe digest")
    cell_seed_invariant = sorted(invariants)[0]

    # SALVAGE mode: bind seed42's sealed correction-3 artifacts as a completed cell (verify
    # its digests + prove treatment-identity to the freshly-built cell invariant) and omit
    # its train node; only seeds 43/44 train. Otherwise all three seeds train.
    salvaged_cell: dict[str, Any] | None = None
    if salvage is not None:
        salvaged_cell, _salvaged_invariant = bind_salvaged_seed42_cell(
            artifact_dir=str(salvage["artifact_dir"]),
            log_path=str(salvage["log_path"]),
            expected_cell_invariant=cell_seed_invariant,
        )
        train_seeds = [seed for seed in SEEDS if int(seed) != SALVAGE_SEED]
    else:
        train_seeds = list(SEEDS)

    # Bind every sealed cross-round input by real sha256 + bytes.
    r0218_panel = _signature(R0218_PANEL, "R0218 frozen panel")
    truth_query_rows = _signature(R0237_TRUTH_QUERY_ROWS, "R0237 50M truth query rows")
    truth_ids = _signature(R0237_TRUTH_IDS, "R0237 50M truth ids")
    reserve = _signature(R0237_RESERVE, "R0237 50M held-out reserve")
    r0265_floors = _signature(R0265_FLOORS, "R0265 sealed family floors")
    r0265_panel = _signature(R0265_PANEL, "R0265 sealed n=13 panel")
    p1_asymptote = _signature(P1_ASYMPTOTE, "P1 analysis-v2 ×2 asymptote band")

    # Cross-check the bound sealed R0265 instruments + the P1 band at prepare, not only at
    # gate time, so a swapped input is caught early.
    floors_sealed = prompt_contract.read_sealed(r0265_floors["canonical_path"], label="R0265 floors")
    if floors_sealed.get("capability") != R0265N.GATE_CAPABILITY or floors_sealed.get("gate_registered") is not True:
        raise RuntimeError("R0267 --r0265-floors is not the sealed R0265 family floors gate")
    panel_sealed = prompt_contract.read_sealed(r0265_panel["canonical_path"], label="R0265 panel")
    if panel_sealed.get("capability") != R0265N.PANEL_CAPABILITY or int(panel_sealed.get("n", -1)) != R0265N.N_FAMILY:
        raise RuntimeError("R0267 --r0265-panel is not the sealed n=13 panel")
    with open(p1_asymptote["canonical_path"], encoding="utf-8") as handle:
        p1_json = json.load(handle)
    if "yinf_x2" not in dict(p1_json.get("bands") or {}) or p1_json.get("verdict") != "GO":
        raise RuntimeError("R0267 --p1-asymptote is not the frozen GO analysis-v2 result")

    census = dispatch_census()
    guard = assert_derived_entries_install(SCOPE_MODULES, census)
    gates = gate_census(entry_tuples(guard["derived"]))
    residual = scope_residual(census, SCOPE_MODULES)

    ensure_data_directory(ROUND_ROOT, label="R0267 round root")
    queue_root = create_fresh_directory(queue_root, label="R0267 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    closure_path = os.path.join(preflight, "treatment-source-closure.json")
    atomic_write_new_json(closure_path, _treatment_closure_seal(release_sha), immutable=True)
    identity_path = os.path.join(preflight, "fneg-50m-x2-cell-identity.json")
    atomic_write_new_json(
        identity_path,
        prompt_contract.seal({
            "schema": "round0267-fneg-50m-x2-cell-identity-v1",
            "round_id": ROUND_ID,
            "release_sha": release_sha,
            "sealed_directed_edges": edges,
            "base_horizon": base_horizon,
            "x2_horizon": int(T.DOSE_MULTIPLIER * base_horizon),
            "seeds": list(SEEDS),
            "x_residency": T.X_RESIDENCY,
            "dose_multiplier": T.DOSE_MULTIPLIER,
            "rows": ROWS,
            "recipe": recipe,
            "cell_seed_invariant_sha256": cell_seed_invariant,
            "per_seed_config_sha256": per_seed_config_sha,
            "registry_fingerprint": registry_fingerprint(),
        }),
        immutable=True,
    )
    closure_signature = expected_input_signature(closure_path)

    # The PRE-SEALED int8 substrate (the delegate-approved fix for the 50M host-int8
    # setup failures): R0267 LOADS R0262's sealed 100M host-int8 substrate's first-50M
    # nested prefix instead of encoding fp32->int8 on the fly at train time (the
    # multi-minute encode blocked the liveness watchdog). Bind R0262's substrate.i8 +
    # substrate-scales.f16 as a sealed input by recording a SLICE LAW manifest that
    # pins the parent files (path + size) and — the load-bearing content binding — the
    # 50M bytes actually trained on by prefix_i8_sha256 / prefix_scales_sha256. The
    # per-row int8 encoder makes that prefix byte-identical to the on-the-fly encode
    # (offline receipt, 0 mismatches), so the LOAD is map-identical.
    for label, parent_path, parent_bytes in (
        ("R0262 100M int8 substrate", T.R0262_PARENT_I8_PATH, T.R0262_PARENT_I8_BYTES),
        ("R0262 100M int8 scales", T.R0262_PARENT_SCALES_PATH, T.R0262_PARENT_SCALES_BYTES),
    ):
        if not os.path.exists(parent_path):
            raise RuntimeError(f"R0267 bound int8 parent absent: {label} at {parent_path}")
        observed_bytes = int(os.stat(parent_path).st_size)
        if observed_bytes != int(parent_bytes):
            raise RuntimeError(
                f"R0267 bound int8 parent size mismatch: {label} at {parent_path} is "
                f"{observed_bytes} bytes, expected {int(parent_bytes)}"
            )
    int8_substrate_manifest_path = os.path.join(preflight, "int8-slice-substrate-manifest.json")
    atomic_write_new_json(
        int8_substrate_manifest_path,
        prompt_contract.seal(T.int8_slice_substrate_manifest_body(release_sha=release_sha)),
        immutable=True,
    )
    int8_substrate_manifest_signature = expected_input_signature(int8_substrate_manifest_path)

    shared_inputs = _dedupe([
        round_signature,
        graph_manifest_signature,
        substrate_manifest_signature,
        substrate_signature,
        graph_signature,
        expected_input_signature(identity_path),
        int8_substrate_manifest_signature,
        closure_signature,
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    p90: dict[str, float] = {}
    jobs: list[dict[str, Any]] = []

    # 1-3. the host-int8 trains. In salvage mode seed42 is bound (correction-3), not
    # trained, so only seeds 43/44 train; otherwise all three seeds (42/43/44) train.
    train_outputs: dict[int, str] = {}
    train_ids: list[str] = []
    for seed in train_seeds:
        capability = capability_for_seed(seed)
        train_node = f"{TRAIN_ACTION}_seed{seed}"
        train_output = os.path.join(artifacts, capability)
        train_outputs[seed] = train_output
        train_ids.append(train_node)
        jobs.append({
            "id": train_node,
            "action": TRAIN_ACTION,
            "handler_module": "experiments.round0267_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [train_output],
            "done_marker": os.path.join(artifacts, f"{train_node}.done.json"),
            "expected_inputs": shared_inputs,
            "p90_wall_s": TRAIN_P90_WALL_S,
            "training_seed": int(seed),
            "capability": capability,
            "graph_manifest_signature": graph_manifest_signature,
            "substrate_manifest_signature": substrate_manifest_signature,
            "int8_substrate_manifest_signature": int8_substrate_manifest_signature,
            "cell_seed_invariant_sha256": cell_seed_invariant,
            "base_horizon": base_horizon,
            "treatment_closure": closure_signature,
            "node_policy": {"gpu_required": True, "training_performed": True, "cpu_heavy": False},
        })
        p90[train_node] = TRAIN_P90_WALL_S

    # 4. the three-cell panel. seed42 is either a fresh train (train_outputs) or the bound
    # salvaged correction-3 cell; 43/44 are always fresh correction-4 trains.
    panel_cells: list[dict[str, Any]] = []
    for seed in SEEDS:
        if salvaged_cell is not None and int(seed) == SALVAGE_SEED:
            panel_cells.append(salvaged_cell)
        else:
            panel_cells.append({
                "seed": int(seed),
                "capability": capability_for_seed(seed),
                "salvaged": False,
                "train_receipt": {"kind": "file", "canonical_path": os.path.join(train_outputs[seed], "train-receipt.json")},
            })
    # In salvage mode the panel also binds seed42's saved artifacts as gate-verified inputs.
    salvage_inputs: list[dict[str, Any]] = []
    if salvaged_cell is not None:
        salvage_inputs = [
            salvaged_cell["coordinates"],
            salvaged_cell["model"],
            salvaged_cell["admission"],
            salvaged_cell["production_config"],
            salvaged_cell["train_log"],
        ]

    panel_node = PANEL_ACTION
    panel_output = os.path.join(artifacts, PANEL_CAPABILITY)
    jobs.append({
        "id": panel_node,
        "action": PANEL_ACTION,
        "handler_module": "experiments.round0267_nodes",
        "handler_callable": "run_job",
        "deps": list(train_ids),
        "outputs": [panel_output],
        "done_marker": os.path.join(artifacts, f"{panel_node}.done.json"),
        "expected_inputs": _dedupe([
            *shared_inputs, r0218_panel, truth_query_rows, truth_ids, reserve, *salvage_inputs,
        ]),
        "p90_wall_s": PANEL_P90_WALL_S,
        "graph_manifest_signature": graph_manifest_signature,
        "substrate_manifest_signature": substrate_manifest_signature,
        "panel_evidence": R0218_PANEL,
        "centroid_ks": list(CENTROID_KS),
        "truth_query_rows": truth_query_rows,
        "truth_ids": truth_ids,
        "heldout_reserve": reserve,
        "cells": panel_cells,
        "gate_registerable_here": False,
        "upstream_review_state": review_state,
        "node_policy": {"gpu_required": True, "training_performed": False, "cpu_heavy": False},
    })
    p90[panel_node] = PANEL_P90_WALL_S

    # 5. the seed-mean gate (CPU).
    gate_node = GATE_ACTION
    jobs.append({
        "id": gate_node,
        "action": GATE_ACTION,
        "handler_module": "experiments.round0267_nodes",
        "handler_callable": "run_job",
        "deps": [panel_node],
        "outputs": [os.path.join(artifacts, GATE_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{gate_node}.done.json"),
        "expected_inputs": _dedupe([
            *shared_inputs, r0265_floors, r0265_panel, p1_asymptote,
        ]),
        "p90_wall_s": GATE_P90_WALL_S,
        "panel": {"kind": "file", "canonical_path": os.path.join(panel_output, "fneg-50m-x2-panel.json")},
        "r0265_floors": r0265_floors,
        "r0265_panel": r0265_panel,
        "p1_asymptote": p1_asymptote,
        "upstream_review_state": review_state,
        "node_policy": {"gpu_required": False, "training_performed": False, "cpu_heavy": True},
    })
    p90[gate_node] = GATE_P90_WALL_S
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
        "schema": "round0267-fneg-50m-x2-seedmean-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            T.R0237_SUBSTRATE_CAPABILITY,
            T.R0237_GRAPH_CAPABILITY,
            R0218_PANEL_CAPABILITY,
            R0265N.GATE_CAPABILITY,
            R0265N.PANEL_CAPABILITY,
        ],
        "capabilities_produced": [*T.CAPABILITIES, PANEL_CAPABILITY, GATE_CAPABILITY],
        "jobs": jobs,
        "p90_wall_s": p90,
        "scope_modules": list(SCOPE_MODULES),
        "stop_hook_install_guard": {
            "derived_entries": guard["derived"],
            "every_derived_entry_installs": guard["audit"]["every_entry_installs_effectively"],
            "gate_census": gates,
            "scope_residual": residual,
        },
        "registered": {
            "what_this_round_is": (
                "train THREE 50M cells (seeds 42/43/44) of the promoted fneg recipe at dose "
                "×2 on the host-int8 X path (R0266-validated), score them on R0265's "
                "instruments, and register the pre-registered 50M gate: the SEED-MEAN "
                "collapse inside P1's ×2 asymptote band widened by 1.96·σ_fam/√3, plus "
                "per-seed backstops on collapse/fog/FFR/purity. Every band/floor/σ_fam/"
                "P1-edge is read from a sealed artifact bound by sha256 — never a literal."
            ),
            "seeds": list(SEEDS),
            "trained_seeds": list(train_seeds),
            "salvaged_seed": (SALVAGE_SEED if salvaged_cell is not None else None),
            "salvage": (
                {
                    "salvaged": True,
                    "seed": SALVAGE_SEED,
                    "source_run": SALVAGE_SOURCE_RUN,
                    "reason": SALVAGE_REASON,
                    "why": (
                        "seed42 trained the FULL ×2 horizon cleanly in correction-3 "
                        "(4,162,228 updates, 0 AMP-skips, 0 nonfinite) and saved its map, "
                        "then the node died on the too-tight 60 GiB post-hoc RSS assertion "
                        "(peak 75.81 GiB). Its saved artifacts are BOUND as a completed cell "
                        "(digests verified) rather than re-training the 11.9 GPU-h; only "
                        "seeds 43/44 re-train. seed42 is treatment-identical to 43/44 "
                        "(same masked-config invariant); the raised RSS limit is an "
                        "execution-resource field, not a treatment field."
                    ),
                    "train_evidence": salvaged_cell["salvage"]["train_evidence"],
                    "digests": salvaged_cell["salvage"]["digests"],
                    "seed_invariant_sha256": salvaged_cell["salvage"]["seed_invariant_sha256"],
                }
                if salvaged_cell is not None
                else None
            ),
            "x_residency": T.X_RESIDENCY,
            "dose_multiplier": T.DOSE_MULTIPLIER,
            "cell_seed_invariant_sha256": cell_seed_invariant,
            "base_horizon": base_horizon,
            "x2_horizon": int(T.DOSE_MULTIPLIER * base_horizon),
            "sealed_directed_edges": edges,
            "rows": ROWS,
            "dose_is_derived_from_the_sealed_edge_count": (
                f"successful_positive_lr_updates = {T.DOSE_MULTIPLIER} * "
                f"successful_updates_for_edges({edges}) = {T.DOSE_MULTIPLIER} * "
                f"{base_horizon} = {int(T.DOSE_MULTIPLIER * base_horizon)}"
            ),
            "consumes_sealed_r0237_inputs": {
                "substrate": T.R0237_SUBSTRATE_CAPABILITY,
                "graph": T.R0237_GRAPH_CAPABILITY,
                "exact_truth": "minilm-mixed-50000k-uniform-probe-k15-truth-v1",
                "held_out_reserve_sha256": reserve["sha256"],
            },
            "x_is_a_pre_sealed_int8_nested_prefix": {
                "why": (
                    "the 50M host-int8 rung LOADS R0262's sealed 100M int8 substrate's "
                    "first-50M-row nested prefix (offset 0) instead of encoding fp32->int8 "
                    "on the fly at train time; the multi-minute on-the-fly encode blocked "
                    "the node liveness watchdog. The per-row int8 encoder makes the prefix "
                    "byte-identical to the encode (offline receipt: 0 mismatches vs the "
                    "sealed 50M fp32 prefix), so the LOAD is MAP-IDENTICAL."
                ),
                "int8_substrate_manifest": T.INT8_SLICE_SUBSTRATE_CAPABILITY,
                "int8_substrate_manifest_sha256": int8_substrate_manifest_signature["sha256"],
                "parent_artifact": T.R0262_INT8_CAPABILITY,
                "slice_law": T.slice_law_block(),
            },
            "consumes_sealed_gate_instruments": {
                "family_floors": R0265N.GATE_CAPABILITY,
                "n13_panel_for_sigma_fam": R0265N.PANEL_CAPABILITY,
                "p1_x2_asymptote_band_sha256": p1_asymptote["sha256"],
            },
            "gate_registerable_here": True,
            "acceptance_rule": (
                "the round trains the three cells, scores them, and registers the gate. NO "
                "NUMERICAL OUTCOME makes it a failure: the 50M PASS/FAIL is a MEASUREMENT "
                "reported either way, contingent on review; the 100M ×2 commit is gated on "
                "a 50M pass."
            ),
        },
    })
    manifest_path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(manifest_path, queue, immutable=True)
    return manifest_path


def file_sha256_manifest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="prepare the R0267 queue")
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=None)
    parser.add_argument(
        "--salvage-seed", type=int, default=None,
        help=f"bind this seed's saved artifacts as a completed cell instead of training it "
             f"(only seed {SALVAGE_SEED} is supported)",
    )
    parser.add_argument(
        "--salvage-from", default=None,
        help="the correction-3 artifact directory holding the salvaged seed's "
             "coordinates.npy/model.pt/admission.json/production-config.json",
    )
    parser.add_argument(
        "--salvage-log", default=None,
        help="the salvaged seed's train log (the clean full-horizon step-accounting proof, "
             "in lieu of the train-receipt the RSS raise preempted)",
    )
    args = parser.parse_args(argv)
    salvage: dict[str, Any] | None = None
    if args.salvage_seed is not None or args.salvage_from is not None or args.salvage_log is not None:
        if args.salvage_seed is None or not args.salvage_from or not args.salvage_log:
            parser.error("--salvage-seed, --salvage-from and --salvage-log must be given together")
        salvage = {
            "seed": int(args.salvage_seed),
            "artifact_dir": args.salvage_from,
            "log_path": args.salvage_log,
        }
    path = prepare_round0267(
        release_sha=args.release_sha,
        queue_root=(args.queue_root or QUEUE_ROOT),
        salvage=salvage,
    )
    print(json.dumps({"queue": path, "sha256": file_sha256_manifest(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
