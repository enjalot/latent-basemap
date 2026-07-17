"""Real scorer performance gates and the fresh synthetic 4x regression."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (canonical_json, expected_input_signature,
                                       path_signature, sha256_bytes, sha256_file)
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.panel_v2 import (FORMULA_VERSION, PanelV2Config, hiD_reference_key,
                              load_embeddings, load_hiD_reference,
                              load_query_truth, ordered_array_sha256,
                              query_truth_key, sample_anchors)
from basemap.release_preflight import validate_release_preflight_receipt
from basemap.round0005_staging import MAP_EXPECTATIONS
from experiments.compare_panel_cache import (
    compare as compare_persisted_scalars,
    extract_persisted_scientific_scalars,
    load_fixture, run_actual_scorer,
)
from experiments.score_complete_panel import validate_query_truth_consumers

SCALE_ROWS = 8_000_000
PERFORMANCE_CERTIFICATE_SCHEMA = "round0005_performance_certificate.v3"
SCALE_POLICY_SCHEMA = "round0005_scale_policy.v3"


def _certificate(body: dict) -> dict:
    return {"body": body, "sha256": sha256_bytes(canonical_json(body))}


def _valid_signature(signature) -> bool:
    if not isinstance(signature, dict):
        return False
    try:
        return expected_input_signature(signature["canonical_path"]) == signature
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _valid_path_signature(signature) -> bool:
    """Revalidate the richer symlink-aware signatures in release receipts."""
    if not isinstance(signature, dict) or not isinstance(signature.get("path"), str):
        return False
    try:
        return path_signature(signature["path"]) == signature
    except (OSError, TypeError, ValueError):
        return False


def _valid_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(char in "0123456789abcdef" for char in value))


def validate_regression_certificate(report: dict) -> dict:
    if not isinstance(report, dict):
        return {"passed": False, "checks": {"report_object": False}}
    certificate = report.get("certificate")
    signatures = report.get("measured_outputs")
    ratio = report.get("measured_total_slowdown")
    baseline = report.get("baseline")
    slowed = report.get("slowed")
    body = certificate.get("body") if isinstance(certificate, dict) else None
    expected_report_fields = {
        "schema", "passed", "allows_scale_launch", "checks", "wall_limit_s",
        "measured_total_slowdown", "minimum_total_slowdown",
        "slowdown_injection_factor", "baseline", "slowed", "fixture_signature",
        "measured_outputs", "certificate",
    }
    expected_body_fields = {
        "schema", "fixture_signature", "measured_outputs", "baseline_wall_s",
        "slowed_wall_s", "measured_total_slowdown", "minimum_total_slowdown",
        "slowdown_injection_factor", "checks",
    }

    def numeric(value) -> bool:
        return (isinstance(value, (int, float)) and not isinstance(value, bool) and
                math.isfinite(float(value)))

    baseline_wall = baseline.get("wall_s") if isinstance(baseline, dict) else None
    slowed_wall = slowed.get("wall_s") if isinstance(slowed, dict) else None
    factor = report.get("slowdown_injection_factor")
    recomputed_ratio = (float(slowed_wall) / float(baseline_wall)
                        if numeric(baseline_wall) and float(baseline_wall) > 0 and
                        numeric(slowed_wall) else None)
    expected_checks = {
        "baseline_real_scorer_passes": (
            numeric(baseline_wall) and numeric(report.get("wall_limit_s")) and
            float(baseline_wall) <= float(report["wall_limit_s"])),
        "slowed_real_scorer_rejected": (
            numeric(slowed_wall) and numeric(report.get("wall_limit_s")) and
            not float(slowed_wall) < float(report["wall_limit_s"])),
        "measured_total_slowdown_at_least_4": (
            recomputed_ratio is not None and recomputed_ratio >= 4.0),
        "scientific_scalars_unchanged": (
            isinstance(baseline, dict) and isinstance(slowed, dict) and
            baseline.get("persisted_scalars") == slowed.get("persisted_scalars")),
        "both_truths_built_once": (
            isinstance(baseline, dict) and isinstance(slowed, dict) and
            baseline.get("query_truth_cache", {}).get("build_count") == 1 and
            slowed.get("query_truth_cache", {}).get("build_count") == 1),
    }
    output_payloads_current = False
    if (isinstance(signatures, dict) and set(signatures) == {"baseline", "slowed"} and
            all(_valid_signature(value) for value in signatures.values())):
        try:
            with open(signatures["baseline"]["canonical_path"], encoding="utf-8") as handle:
                persisted_baseline = json.load(handle)
            with open(signatures["slowed"]["canonical_path"], encoding="utf-8") as handle:
                persisted_slowed = json.load(handle)
            output_payloads_current = (persisted_baseline == baseline and
                                       persisted_slowed == slowed)
        except (OSError, ValueError, json.JSONDecodeError):
            output_payloads_current = False
    consumer_contracts = False
    if isinstance(baseline, dict) and isinstance(slowed, dict):
        consumer_contracts = all(
            validate_query_truth_consumers(
                item.get("query_truth_cache") or {}, ["synthetic-regression"],
                k_hit=10, expected_builds=1)["passed"]
            for item in (baseline, slowed))
    checks = {
        "exact_report_fields": set(report) == expected_report_fields,
        "schema": report.get("schema") == "round0005_real_scorer_4x_regression.v3",
        "passed_matches_checks": (
            report.get("passed") is True and report.get("checks") == expected_checks and
            all(expected_checks.values())),
        "does_not_itself_release_scale": report.get("allows_scale_launch") is False,
        "measured_total_slowdown_at_least_4": (
            numeric(ratio) and recomputed_ratio is not None and float(ratio) == recomputed_ratio and
            float(ratio) >= 4.0 and report.get("minimum_total_slowdown") == 4.0),
        "wall_budget_recomputed": (
            numeric(baseline_wall) and numeric(report.get("wall_limit_s")) and
            float(report["wall_limit_s"]) == float(baseline_wall) * 4.0),
        "injection_is_real_and_at_least_4x": (
            numeric(factor) and float(factor) >= 4.0 and
            isinstance(baseline, dict) and isinstance(slowed, dict) and
            numeric(baseline.get("phase_delay_s")) and
            numeric(slowed.get("phase_delay_s")) and
            float(baseline["phase_delay_s"]) > 0 and
            float(slowed["phase_delay_s"]) ==
            float(baseline["phase_delay_s"]) * float(factor)),
        "certificate_content_bound": (
            isinstance(certificate, dict) and set(certificate) == {"body", "sha256"} and
            isinstance(body, dict) and set(body) == expected_body_fields and
            certificate.get("sha256") == sha256_bytes(canonical_json(body)) and
            body == {
                "schema": "round0005_real_scorer_4x_regression_body.v1",
                "fixture_signature": report.get("fixture_signature"),
                "measured_outputs": signatures,
                "baseline_wall_s": baseline_wall,
                "slowed_wall_s": slowed_wall,
                "measured_total_slowdown": ratio,
                "minimum_total_slowdown": report.get("minimum_total_slowdown"),
                "slowdown_injection_factor": factor,
                "checks": expected_checks,
            }),
        "fixture_signature_current": _valid_signature(report.get("fixture_signature")),
        "real_scorer_payloads": (
            isinstance(baseline, dict) and isinstance(slowed, dict) and
            baseline.get("schema") == "round0005_actual_query_score.v1" and
            slowed.get("schema") == "round0005_actual_query_score.v1" and
            baseline.get("cache_enabled") is False and
            slowed.get("cache_enabled") is False and
            baseline.get("fixture_payload_sha256") ==
            slowed.get("fixture_payload_sha256")),
        "exact_query_truth_consumers": consumer_contracts,
        "measured_output_payloads_current": output_payloads_current,
        "measured_output_signatures_current": (
            isinstance(signatures, dict) and set(signatures) == {"baseline", "slowed"} and
            all(_valid_signature(value) for value in signatures.values())),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _sealed_reference_identities(context: dict) -> tuple[dict, dict, dict]:
    seal_path = context["paths"]["testbed_seal"]
    with open(seal_path, encoding="utf-8") as handle:
        seal = json.load(handle)
    train = seal["train"]
    train_file = train["embedding_signature"]
    data_identity = {
        "kind": "ordered_shards",
        "shape": list(train["array"]["shape"]),
        "dtype": np.dtype(train["array"]["dtype"]).str,
        "shards": [{
            "position": 0,
            "name": os.path.basename(train_file["canonical_path"]),
            "bytes": train_file["bytes"], "sha256": train_file["sha256"],
        }],
    }
    centroid_identities = {}
    for name, count in (("k256", 256), ("k1024", 1024)):
        value = seal["centroids"][name]
        signature = value["signature"]
        if expected_input_signature(signature["canonical_path"]) != signature:
            raise ValueError("sealed private-reference centroid changed")
        array = np.load(signature["canonical_path"], mmap_mode="r")
        if list(array.shape) != value["array"]["shape"] or int(array.shape[0]) != count:
            raise ValueError("sealed private-reference centroid shape changed")
        centroid_identities[str(count)] = {
            "shape": [int(item) for item in array.shape],
            "dtype": array.dtype.str,
            "sha256": ordered_array_sha256(array),
        }
    query_corpus = context["query"]["manifest"]["corpus"]
    if (query_corpus.get("testbed") != seal["testbed_root"] or
            query_corpus.get("ordered_train_embeddings") !=
            seal["train"]["root_signature"] or
            query_corpus.get("ordered_train_ids") !=
            seal["sample_indices"]["signature"] or
            query_corpus.get("n_train") != seal["contract"]["train_rows"]):
        raise ValueError("query corpus identity differs from the sealed testbed")
    return data_identity, centroid_identities, query_corpus


def _load_private_reference_receipt(panel: dict, reference_keys: set,
                                    *, context: dict | None = None) -> dict | None:
    signature = panel.get("hiD_reference_receipt")
    if not _valid_signature(signature):
        return None
    try:
        with open(signature["canonical_path"], encoding="utf-8") as handle:
            receipt = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    required = {
        "schema", "reference", "identity_key", "content_sha256", "key_parts",
        "built_and_reloaded_in_same_scorer", "pre_gate_reference_consumed",
    }
    valid = (
        isinstance(receipt, dict) and set(receipt) == required and
        receipt.get("schema") == "round0005_private_hiD_reference_receipt.v1" and
        receipt.get("identity_key") in reference_keys and len(reference_keys) == 1 and
        receipt.get("content_sha256") == panel.get("hiD_reference_content_sha256") and
        _valid_sha256(receipt.get("content_sha256")) and
        isinstance(receipt.get("key_parts"), dict) and receipt["key_parts"] and
        receipt.get("built_and_reloaded_in_same_scorer") is True and
        receipt.get("pre_gate_reference_consumed") is False and
        _valid_signature(receipt.get("reference")))
    if not valid:
        return None
    # A JSON receipt is not evidence that an NPZ exists or contains the key it
    # claims.  Strictly load the archive and revalidate every identity part.
    try:
        reopened = load_hiD_reference(
            receipt["reference"]["canonical_path"],
            expected_key=receipt["identity_key"],
            expected_key_parts=receipt["key_parts"])
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None
    if (reopened.get("key") != receipt["identity_key"] or
            reopened.get("key_parts") != receipt["key_parts"] or
            reopened.get("content_sha256") != receipt["content_sha256"]):
        return None
    if context is not None:
        try:
            data_identity, centroid_identities, _query_corpus = \
                _sealed_reference_identities(context)
            cfg = PanelV2Config(frac=0.001, n_anchors=10_000,
                                corpus_chunk=500_000)
            anchors = sample_anchors(2_000_000, cfg)
            expected_key, expected_parts = hiD_reference_key(
                None, anchors, cfg, None,
                kf=2_000, data_identity=data_identity,
                centroid_identities=centroid_identities)
        except Exception:
            return None
        if (receipt["identity_key"] != expected_key or
                receipt["key_parts"] != expected_parts):
            return None
    return receipt


def _validate_private_reference_receipt(panel: dict, reference_keys: set) -> bool:
    return _load_private_reference_receipt(panel, reference_keys) is not None


def _valid_process_cuda_peak(value: dict, *, reported_maximum) -> bool:
    required = {"schema", "available", "allocated_bytes", "reserved_bytes",
                "allocated_gib", "reserved_gib", "maximum_gib"}
    if (not isinstance(value, dict) or set(value) != required or
            value.get("schema") != "process_cuda_peak.v1" or
            value.get("available") is not True):
        return False
    allocated = value.get("allocated_bytes")
    reserved = value.get("reserved_bytes")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0
           for item in (allocated, reserved)):
        return False
    gib = float(1024 ** 3)
    allocated_gib = round(allocated / gib, 6)
    reserved_gib = round(reserved / gib, 6)
    maximum_gib = max(allocated_gib, reserved_gib)
    return (value.get("allocated_gib") == allocated_gib and
            value.get("reserved_gib") == reserved_gib and
            value.get("maximum_gib") == maximum_gib and
            reported_maximum == maximum_gib)


def evaluate_panel(panel: dict, *, wall_max_s: float = 120.0,
                   peak_max_gb: float = 26.0, expected_builds: int = 1,
                   panel_signature: dict | None = None,
                   regression_report: dict | None = None,
                   regression_signature: dict | None = None) -> dict:
    """Evaluate only measured fields emitted by the real nine-map scorer."""
    wall = panel.get("total_wall_s")
    peak = panel.get("peak_gpu_gb")
    telemetry = panel.get("query_truth_cache") or {}
    builds = telemetry.get("build_count")
    runs = panel.get("runs") or {}
    formula = panel.get("formula_version")
    real_runs = True
    for run in runs.values() if isinstance(runs, dict) else ():
        full = run.get("panel_full") if isinstance(run, dict) else None
        real_runs = real_runs and isinstance(full, dict)
        if not isinstance(full, dict):
            continue
        real_runs = real_runs and full.get("schema") == "panel_v2"
        real_runs = real_runs and full.get("formula_version") == formula
        provenance = full.get("provenance") or {}
        real_runs = real_runs and provenance.get("scorer") == "complete_panel"
        for leaf, full_key in (
                ("ffr", "ffr"), ("recall@k", "recall@k"),
                ("density", "density"), ("proj_ffr", None)):
            if full_key is not None:
                real_runs = real_runs and run.get(leaf) == full.get(full_key)
        real_runs = real_runs and run.get("proj_ffr") is not None
        real_runs = real_runs and run.get("proj_knn_regressor_ffr") is not None
        real_runs = real_runs and run.get("proj_random_floor_ffr") is not None
    measured_wall = (isinstance(wall, (int, float)) and not isinstance(wall, bool) and
                     math.isfinite(float(wall)) and float(wall) >= 0)
    measured_peak = (isinstance(peak, (int, float)) and not isinstance(peak, bool) and
                     math.isfinite(float(peak)) and float(peak) >= 0)
    process_peak = panel.get("process_cuda_peak") or {}
    measured_process_peak = _valid_process_cuda_peak(
        process_peak, reported_maximum=peak)
    consumer_contract = validate_query_truth_consumers(
        telemetry, MAP_EXPECTATIONS, k_hit=10, expected_builds=expected_builds)
    reference_keys = {run.get("hiD_reference_key") for run in runs.values()
                      if isinstance(run, dict)} if isinstance(runs, dict) else set()
    base_checks = {
        "wall": measured_wall and float(wall) <= wall_max_s,
        "peak_gpu": measured_peak and float(peak) <= peak_max_gb,
        "process_global_allocated_and_reserved_peak": measured_process_peak,
        "highd_build_count": builds == expected_builds,
        "one_maximum_k15_truth_shared": consumer_contract["passed"],
        "nine_maps": isinstance(runs, dict) and set(runs) == set(MAP_EXPECTATIONS),
        "real_scorer_schema": isinstance(formula, str) and real_runs,
        "one_private_panel_reference": (
            len(reference_keys) == 1 and None not in reference_keys and
            panel.get("pre_gate_reference_consumed") is False and
            _valid_sha256(panel.get("hiD_reference_content_sha256")) and
            _validate_private_reference_receipt(panel, reference_keys)),
    }
    regression_validation = (validate_regression_certificate(regression_report)
                             if regression_report is not None else
                             {"passed": False, "checks": {"missing": False}})
    binding_checks = {
        "panel_signature_current": _valid_signature(panel_signature),
        "regression_signature_current": _valid_signature(regression_signature),
        "regression_certificate": regression_validation["passed"],
    }
    passed = all(base_checks.values())
    allows_scale = passed and all(binding_checks.values())
    limits = {"wall_s": wall_max_s, "peak_gpu_gb": peak_max_gb,
              "highd_build_count": expected_builds, "map_count": 9,
              "minimum_measured_total_slowdown": 4.0}
    body = {
        "schema": "round0005_scale_performance_certificate_body.v1",
        "round_id": "0005",
        "panel_input": panel_signature,
        "regression_input": regression_signature,
        "base_checks": base_checks,
        "binding_checks": binding_checks,
        "observed": {"wall_s": wall, "peak_gpu_gb": peak,
                     "highd_build_count": builds,
                     "map_count": len(panel.get("runs") or {})},
        "limits": limits,
    }
    return {
        "schema": "round0005_measured_panel_performance_gate.v3",
        "passed": passed,
        "allows_scale_launch": allows_scale,
        "checks": {**base_checks, **binding_checks},
        "query_truth_consumer_contract": consumer_contract,
        "regression_validation": regression_validation,
        "observed": {"wall_s": wall, "peak_gpu_gb": peak,
                     "highd_build_count": builds,
                     "map_count": len(panel.get("runs") or {})},
        "limits": limits,
        "certificate": _certificate(body),
    }


def run_synthetic_regression(*, fixture_path: str, out_root: str,
                             baseline_phase_delay_s: float = 0.01,
                             slowdown_injection_factor: float = 16.0) -> dict:
    """Execute the real scorer twice and require >=4x *measured total* slowdown."""
    if baseline_phase_delay_s <= 0:
        raise ValueError("synthetic regression needs a positive real phase delay")
    if slowdown_injection_factor < 4.0:
        raise ValueError("slowdown injection factor must be at least 4")
    fixture = load_fixture(fixture_path)
    out_root = create_fresh_directory(out_root, label="synthetic performance output root")
    baseline = run_actual_scorer(
        fixture, cache_enabled=False, cache_dir=None,
        phase_delay_s=baseline_phase_delay_s, label="synthetic-regression")
    slowed = run_actual_scorer(
        fixture, cache_enabled=False, cache_dir=None,
        phase_delay_s=baseline_phase_delay_s * slowdown_injection_factor,
        label="synthetic-regression")
    baseline_wall = float(baseline["wall_s"])
    slowed_wall = float(slowed["wall_s"])
    measured_ratio = slowed_wall / baseline_wall if baseline_wall > 0 else float("inf")
    # A fresh measured baseline is the budget.  The injected run must exceed it
    # by at least four in total process wall, including all unchanged overhead.
    wall_limit = baseline_wall * 4.0
    baseline_passed = baseline_wall <= wall_limit
    slowed_passed = slowed_wall < wall_limit
    same_scalars = baseline["persisted_scalars"] == slowed["persisted_scalars"]
    checks = {
        "baseline_real_scorer_passes": baseline_passed,
        "slowed_real_scorer_rejected": not slowed_passed,
        "measured_total_slowdown_at_least_4": measured_ratio >= 4.0,
        "scientific_scalars_unchanged": same_scalars,
        "both_truths_built_once": (
            baseline["query_truth_cache"]["build_count"] == 1 and
            slowed["query_truth_cache"]["build_count"] == 1),
    }
    baseline_path = os.path.join(out_root, "baseline.json")
    slowed_path = os.path.join(out_root, "slowed.json")
    atomic_write_new_json(baseline_path, baseline, immutable=True)
    atomic_write_new_json(slowed_path, slowed, immutable=True)
    output_signatures = {
        "baseline": expected_input_signature(baseline_path),
        "slowed": expected_input_signature(slowed_path),
    }
    body = {
        "schema": "round0005_real_scorer_4x_regression_body.v1",
        "fixture_signature": fixture["signature"],
        "measured_outputs": output_signatures,
        "baseline_wall_s": baseline_wall,
        "slowed_wall_s": slowed_wall,
        "measured_total_slowdown": measured_ratio,
        "minimum_total_slowdown": 4.0,
        "slowdown_injection_factor": float(slowdown_injection_factor),
        "checks": checks,
    }
    report = {
        "schema": "round0005_real_scorer_4x_regression.v3",
        "passed": all(checks.values()),
        # This artifact certifies that a regression is rejected; it is
        # deliberately not a passing scale-performance certificate.
        "allows_scale_launch": False,
        "checks": checks,
        "wall_limit_s": wall_limit,
        "measured_total_slowdown": measured_ratio,
        "minimum_total_slowdown": 4.0,
        "slowdown_injection_factor": float(slowdown_injection_factor),
        "baseline": baseline,
        "slowed": slowed,
        "fixture_signature": fixture["signature"],
        "measured_outputs": output_signatures,
        "certificate": _certificate(body),
    }
    atomic_write_new_json(os.path.join(out_root, "regression.json"), report, immutable=True)
    return report


def derive_scale_rows(embedding_path: str, *, dimensions: int,
                      loaded_matrix=None) -> dict:
    """Derive scientific row count from the actual ordered embedding input."""
    if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0:
        raise ValueError("scale-row dimensions must be a positive integer")
    matrix = loaded_matrix if loaded_matrix is not None else load_embeddings(
        embedding_path, dim=dimensions)
    if len(matrix.shape) != 2 or int(matrix.shape[1]) != dimensions:
        raise ValueError("scale-row embedding dimensions mismatch")
    signature = expected_input_signature(embedding_path)
    body = {
        "schema": "round0005_scale_row_derivation.v1",
        "embedding_input": signature,
        "scientific_rows": int(len(matrix)),
        "dimensions": int(dimensions),
        "derivation": "ordered embedding matrix shape",
    }
    return {**body, "certificate_sha256": sha256_bytes(canonical_json(body))}


def validate_scale_rows(value: dict, *, scientific_rows: int) -> dict:
    required = {"schema", "embedding_input", "scientific_rows", "dimensions",
                "derivation", "certificate_sha256"}
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeError("scale-row derivation fields are invalid")
    body = {key: value[key] for key in required - {"certificate_sha256"}}
    if (not isinstance(scientific_rows, int) or isinstance(scientific_rows, bool) or
            scientific_rows <= 0 or
            value.get("schema") != "round0005_scale_row_derivation.v1" or
            value.get("derivation") != "ordered embedding matrix shape" or
            value.get("scientific_rows") != scientific_rows or
            not isinstance(value.get("dimensions"), int) or
            isinstance(value.get("dimensions"), bool) or value["dimensions"] <= 0 or
            value.get("certificate_sha256") != sha256_bytes(canonical_json(body)) or
            not _valid_signature(value.get("embedding_input"))):
        raise RuntimeError("scale-row derivation/content certificate mismatch")
    # Do not trust a caller-supplied shape or a recomputed self-hash.  Reopen the
    # exact signed input and derive its actual matrix shape again at admission.
    matrix = load_embeddings(
        value["embedding_input"]["canonical_path"], dim=value["dimensions"])
    if (len(matrix.shape) != 2 or int(matrix.shape[1]) != value["dimensions"] or
            int(len(matrix)) != scientific_rows):
        raise RuntimeError("scale-row derivation disagrees with actual embedding shape")
    return value


def _valid_release_sha(value) -> bool:
    return (isinstance(value, str) and len(value) == 40 and
            all(char in "0123456789abcdef" for char in value))


def require_current_release_sha(release_sha: str, *, repo_root: str | None = None) -> str:
    if not _valid_release_sha(release_sha):
        raise RuntimeError("release SHA must be exact lowercase 40-hex")
    root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    current = subprocess.check_output(
        ["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    if current != release_sha:
        raise RuntimeError(f"release SHA mismatch: checkout={current} certificate={release_sha}")
    return current


def scale_input_identity_sha256(row_derivation: dict, *, scientific_rows: int) -> str:
    validate_scale_rows(row_derivation, scientific_rows=scientific_rows)
    return sha256_bytes(canonical_json(row_derivation))


def measured_report_identity(*, panel: dict, regression: dict,
                             panel_signature: dict, regression_signature: dict,
                             measured_gate: dict) -> dict:
    """Bind report bytes, reported rows, and the scorer's ordered corpus identity."""
    if not _valid_signature(panel_signature) or not _valid_signature(regression_signature):
        raise ValueError("measured report signatures are stale")
    reference_keys = {run.get("hiD_reference_key") for run in (panel.get("runs") or {}).values()
                      if isinstance(run, dict)}
    receipt = _load_private_reference_receipt(panel, reference_keys)
    if receipt is None:
        raise ValueError("measured panel lacks a current private-reference receipt")
    ordered_data = receipt.get("key_parts", {}).get("data")
    shape = ordered_data.get("shape") if isinstance(ordered_data, dict) else None
    reported_rows = panel.get("n")
    if (not isinstance(reported_rows, int) or isinstance(reported_rows, bool) or
            reported_rows <= 0 or not isinstance(shape, list) or len(shape) != 2 or
            shape[0] != reported_rows):
        raise ValueError("measured panel reported rows disagree with ordered corpus identity")
    body = {
        "schema": "round0005_measured_report_identity.v1",
        "panel_report": panel_signature,
        "regression_report": regression_signature,
        "reported_corpus_rows": reported_rows,
        "ordered_corpus_identity": ordered_data,
        "private_reference_identity_key": receipt["identity_key"],
        "private_reference_content_sha256": receipt["content_sha256"],
        "measured_gate_certificate": measured_gate.get("certificate"),
        "measured_gate_observed": measured_gate.get("observed"),
        "measured_gate_limits": measured_gate.get("limits"),
        "measured_total_slowdown": regression.get("measured_total_slowdown"),
    }
    return {"body": body, "sha256": sha256_bytes(canonical_json(body))}


SCALE_EVIDENCE_FIELDS = {
    "panel_report", "regression_report", "benchmark_corpus",
    "private_reference_archive", "private_reference_receipt", "scale_input",
    "release_preflight_receipt", "environment_manifest", "environment_freeze",
    "python_executable",
}
SCALE_CERTIFICATE_FIELDS = {
    "schema", "passed", "allows_scale_launch", "release_sha", "scientific_rows",
    "row_derivation", "limits", "evidence", "environment_binding", "measured_gate",
    "measured_report_identity_sha256", "measured_total_slowdown", "identity_sha256",
}
CANONICAL_SCALE_LIMITS = {
    "wall_max_s": 120.0,
    "peak_max_gb": 26.0,
    "expected_builds": 1,
    "minimum_measured_total_slowdown": 4.0,
    "required_map_count": 9,
}


def _load_signed_json(signature: dict, *, label: str) -> dict:
    if not _valid_signature(signature):
        raise ValueError(f"{label} signature is stale or malformed")
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _release_environment_binding(release: dict) -> dict:
    return {
        "schema": "round0005_scale_environment_binding.v1",
        "release_preflight_identity_sha256": release["identity_sha256"],
        "implementation_commits": release["implementation_commits"],
        "pushed_ref": release["pushed_ref"],
        "pushed_ref_tip": release["pushed_ref_tip"],
        "environment_identity_sha": release["environment_identity_sha"],
        "environment_freeze_sha": release["environment_freeze_sha"],
        "run_checkout": release["run_checkout"],
    }


def _reopen_scale_evidence(*, evidence: dict, row_derivation: dict,
                           scientific_rows: int, release_sha: str,
                           limits: dict) -> dict:
    """Reopen and recompute every fact that authorizes a scale launch."""
    if not isinstance(evidence, dict) or set(evidence) != SCALE_EVIDENCE_FIELDS:
        raise ValueError("scale certificate evidence fields are not exact")
    if limits != CANONICAL_SCALE_LIMITS:
        raise ValueError("scale certificate limits are not the canonical limits")
    release_path_fields = {
        "environment_manifest", "environment_freeze", "python_executable"}
    for label, signature in evidence.items():
        valid = (_valid_path_signature(signature) if label in release_path_fields
                 else _valid_signature(signature))
        if not valid:
            raise ValueError(f"scale evidence changed: {label}")

    panel = _load_signed_json(evidence["panel_report"], label="measured panel")
    regression = _load_signed_json(
        evidence["regression_report"], label="measured regression")
    fixture = load_fixture(evidence["benchmark_corpus"]["canonical_path"])
    if (fixture["signature"] != evidence["benchmark_corpus"] or
            regression.get("fixture_signature") != evidence["benchmark_corpus"]):
        raise ValueError("regression benchmark corpus binding changed")

    reference_keys = {
        run.get("hiD_reference_key") for run in (panel.get("runs") or {}).values()
        if isinstance(run, dict)
    }
    reference_receipt = _load_private_reference_receipt(panel, reference_keys)
    if reference_receipt is None:
        raise ValueError("panel private reference archive/receipt is invalid")
    if (panel.get("hiD_reference_receipt") != evidence["private_reference_receipt"] or
            reference_receipt["reference"] != evidence["private_reference_archive"]):
        raise ValueError("certificate private reference evidence differs from the panel")

    validate_scale_rows(row_derivation, scientific_rows=scientific_rows)
    if row_derivation["embedding_input"] != evidence["scale_input"]:
        raise ValueError("certificate scale input differs from reopened row derivation")

    release_path = evidence["release_preflight_receipt"]["canonical_path"]
    release_record = _json_object(release_path, label="release preflight receipt")
    release = validate_release_preflight_receipt(
        release_path,
        expected_identity_sha256=release_record.get("identity_sha256"),
        expected_signature=evidence["release_preflight_receipt"])
    if release["release_sha"] != release_sha:
        raise ValueError("release preflight receipt differs from scale release")
    expected_release_evidence = {
        "release_preflight_receipt": evidence["release_preflight_receipt"],
        "environment_manifest": release["environment_manifest"],
        "environment_freeze": release["environment_freeze"],
        "python_executable": release["python_executable"],
    }
    for label in ("environment_manifest", "environment_freeze", "python_executable"):
        if evidence[label] != expected_release_evidence[label]:
            raise ValueError(f"release-bound {label} differs from certificate evidence")

    regression_validation = validate_regression_certificate(regression)
    if not regression_validation["passed"]:
        raise ValueError(
            f"measured regression certificate failed: {regression_validation['checks']}")
    baseline_wall = regression["baseline"]["wall_s"]
    slowed_wall = regression["slowed"]["wall_s"]
    ratio = float(slowed_wall) / float(baseline_wall)
    if not math.isfinite(ratio) or ratio < 4.0:
        raise ValueError("reopened measured total slowdown is below 4x")

    measured_gate = evaluate_panel(
        panel, wall_max_s=limits["wall_max_s"],
        peak_max_gb=limits["peak_max_gb"],
        expected_builds=limits["expected_builds"],
        panel_signature=evidence["panel_report"], regression_report=regression,
        regression_signature=evidence["regression_report"])
    if measured_gate.get("allows_scale_launch") is not True:
        raise ValueError(f"reopened measured panel blocks scale: {measured_gate['checks']}")
    report_identity = measured_report_identity(
        panel=panel, regression=regression,
        panel_signature=evidence["panel_report"],
        regression_signature=evidence["regression_report"],
        measured_gate=measured_gate)
    return {
        "measured_gate": measured_gate,
        "measured_report_identity_sha256": report_identity["sha256"],
        "measured_total_slowdown": ratio,
        "environment_binding": _release_environment_binding(release),
    }


def issue_scale_performance_certificate(*, panel_path: str, regression_path: str,
                                        release_preflight_receipt: str,
                                        row_derivation: dict) -> dict:
    """Issue replayable evidence; no caller-supplied hash or slowdown is trusted."""
    panel_signature = expected_input_signature(panel_path)
    regression_signature = expected_input_signature(regression_path)
    panel = _load_signed_json(panel_signature, label="measured panel")
    regression = _load_signed_json(regression_signature, label="measured regression")
    scientific_rows = (row_derivation.get("scientific_rows")
                       if isinstance(row_derivation, dict) else None)
    if (not isinstance(scientific_rows, int) or isinstance(scientific_rows, bool) or
            scientific_rows < SCALE_ROWS):
        raise ValueError("scale input row derivation must reopen at least 8,000,000 rows")
    validate_scale_rows(row_derivation, scientific_rows=scientific_rows)
    reference_keys = {
        run.get("hiD_reference_key") for run in (panel.get("runs") or {}).values()
        if isinstance(run, dict)
    }
    reference_receipt = _load_private_reference_receipt(panel, reference_keys)
    if reference_receipt is None:
        raise ValueError("measured panel lacks a valid private NPZ reference")
    release_signature = expected_input_signature(release_preflight_receipt)
    release_record = _json_object(
        release_signature["canonical_path"], label="release preflight receipt")
    release = validate_release_preflight_receipt(
        release_signature["canonical_path"],
        expected_identity_sha256=release_record.get("identity_sha256"),
        expected_signature=release_signature)
    evidence = {
        "panel_report": panel_signature,
        "regression_report": regression_signature,
        "benchmark_corpus": regression.get("fixture_signature"),
        "private_reference_archive": reference_receipt["reference"],
        "private_reference_receipt": panel["hiD_reference_receipt"],
        "scale_input": row_derivation["embedding_input"],
        "release_preflight_receipt": release_signature,
        "environment_manifest": release["environment_manifest"],
        "environment_freeze": release["environment_freeze"],
        "python_executable": release["python_executable"],
    }
    derived = _reopen_scale_evidence(
        evidence=evidence, row_derivation=row_derivation,
        scientific_rows=scientific_rows, release_sha=release["release_sha"],
        limits=CANONICAL_SCALE_LIMITS)
    body = {
        "schema": PERFORMANCE_CERTIFICATE_SCHEMA,
        "passed": True,
        "allows_scale_launch": True,
        "release_sha": release["release_sha"],
        "scientific_rows": scientific_rows,
        "row_derivation": row_derivation,
        "limits": CANONICAL_SCALE_LIMITS,
        "evidence": evidence,
        **derived,
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def validate_scale_performance_certificate(
        report: dict, *, expected_release_sha: str, scientific_rows: int,
        row_derivation: dict) -> dict:
    checks = {
        "exact_fields": isinstance(report, dict) and set(report) == SCALE_CERTIFICATE_FIELDS,
        "schema": isinstance(report, dict) and
            report.get("schema") == PERFORMANCE_CERTIFICATE_SCHEMA,
        "passed": isinstance(report, dict) and report.get("passed") is True,
        "allows_scale_launch": isinstance(report, dict) and
            report.get("allows_scale_launch") is True,
        "release_sha": (_valid_release_sha(expected_release_sha) and
                         isinstance(report, dict) and
                         report.get("release_sha") == expected_release_sha),
        "scientific_rows": (isinstance(scientific_rows, int) and
                            not isinstance(scientific_rows, bool) and
                            scientific_rows >= SCALE_ROWS and isinstance(report, dict) and
                            report.get("scientific_rows") == scientific_rows),
        "row_derivation": isinstance(report, dict) and
            report.get("row_derivation") == row_derivation,
        "evidence_reopened": False,
        "derived_values_exact": False,
        "content_identity": False,
    }
    error = None
    if checks["exact_fields"]:
        try:
            derived = _reopen_scale_evidence(
                evidence=report["evidence"], row_derivation=report["row_derivation"],
                scientific_rows=report["scientific_rows"],
                release_sha=report["release_sha"], limits=report["limits"])
            checks["evidence_reopened"] = True
            checks["derived_values_exact"] = all(
                report.get(key) == value for key, value in derived.items())
            body = {key: report[key] for key in SCALE_CERTIFICATE_FIELDS
                    if key != "identity_sha256"}
            checks["content_identity"] = (
                _valid_sha256(report.get("identity_sha256")) and
                report["identity_sha256"] == sha256_bytes(canonical_json(body)))
        except (KeyError, OSError, RuntimeError, TypeError, ValueError,
                json.JSONDecodeError) as exc:
            error = str(exc)
    return {"passed": all(checks.values()), "checks": checks, "error": error}


# The v2 implementation above remains only as a parser/negative-test reference.
# Production v3 certificates are derived exclusively from a completed genuine
# controller run; a standalone panel/regression pair can never enter this path.
_PRODUCTION_EVIDENCE_FIELDS = {
    "queue_manifest", "controller_terminal", "release_preflight_receipt",
    "gate_preparation_receipt", "maps_seal", "testbed_seal", "scale_input",
    "environment_manifest", "environment_freeze", "python_executable", "jobs",
}
_PRODUCTION_CERTIFICATE_FIELDS = {
    "schema", "passed", "allows_scale_launch", "release_sha",
    "scientific_rows", "row_derivation", "limits", "evidence",
    "environment_binding", "controller_identity", "benchmark_identity",
    "measured_gate", "measured_total_slowdown", "identity_sha256",
}
_CERTIFYING_JOB_IDS = (
    "fresh_uncached_2m", "cached_nine_map", "scalar_equivalence",
    "synthetic_4x_regression",
)


def _json_object(path: str, *, label: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _role_signatures(manifest: dict) -> dict[str, dict]:
    roles = {}
    for value in manifest.get("program_inputs", []):
        if not isinstance(value, dict) or set(value) != {"role", "signature"}:
            raise ValueError("queue program input roles are malformed")
        roles[value["role"]] = value["signature"]
    return roles


def _valid_watchdog_verdict(value: dict, *, job: str,
                            controller_claim_sha256: str) -> bool:
    if not isinstance(value, dict) or value.get("job") != job:
        return False
    identity = value.get("identity_sha256")
    body = {key: value[key] for key in value if key != "identity_sha256"}
    return (value.get("schema") == "round0005_watchdog_verdict.v1" and
            value.get("controller_claim_sha256") == controller_claim_sha256 and
            value.get("status") == "clean" and value.get("error") is None and
            _valid_sha256(identity) and
            identity == sha256_bytes(canonical_json(body)))


_RUNTIME_GPU_SNAPSHOT_FIELDS = {
    "schema", "at", "job", "root_pid", "lease_owner", "process_tree",
    "known_process_identities", "allowed_services", "gpu_snapshot",
    "unknown_gpu_processes", "job_gpu_processes", "service_gpu_processes",
    "memory_accounting", "errors",
}
_RUNTIME_GPU_ACCOUNTING_FIELDS = {
    "job_allocated_accounted_mb", "job_reserved_cap_mb",
    "service_allocated_accounted_mb", "service_reserved_cap_mb",
    "cumulative_allocated_accounted_mb", "cumulative_reserved_cap_mb",
    "device_total_mb",
}
_GPU_SNAPSHOT_FIELDS = {
    "at", "gpu", "compute_apps", "compute_app_records", "compute_pids",
    "free_mb", "used_mb", "total_mb", "n_co_tenants", "observer",
    "gpu_uuid", "gpu_name", "gpu_driver",
}
_PROCESS_RECORD_FIELDS = {"pid", "ppid", "proc_starttime_ticks", "cmdline_sha256"}


def _valid_captured_process_record(value: dict) -> bool:
    return (
        isinstance(value, dict) and set(value) == _PROCESS_RECORD_FIELDS and
        isinstance(value.get("pid"), int) and not isinstance(value["pid"], bool) and
        value["pid"] > 0 and isinstance(value.get("ppid"), int) and
        not isinstance(value["ppid"], bool) and value["ppid"] >= 0 and
        isinstance(value.get("proc_starttime_ticks"), int) and
        not isinstance(value["proc_starttime_ticks"], bool) and
        value["proc_starttime_ticks"] > 0 and
        _valid_sha256(value.get("cmdline_sha256")))


def _validate_runtime_gpu_snapshot(snapshot: dict, *, raw_job: dict,
                                   manifest: dict, label: str) -> float:
    """Recompute the controller/watchdog accounting relationships exactly."""
    accounting = snapshot.get("memory_accounting") if isinstance(snapshot, dict) else None
    gpu = snapshot.get("gpu_snapshot") if isinstance(snapshot, dict) else None
    expected_cap = raw_job.get("node_policy", {}).get("gpu_memory_cap_mb")
    if (not isinstance(snapshot, dict) or set(snapshot) != _RUNTIME_GPU_SNAPSHOT_FIELDS or
            snapshot.get("schema") != "round0005_runtime_gpu_telemetry.v2" or
            snapshot.get("job") != raw_job.get("id") or
            not isinstance(snapshot.get("root_pid"), int) or
            isinstance(snapshot.get("root_pid"), bool) or snapshot["root_pid"] <= 0 or
            not isinstance(snapshot.get("lease_owner"), dict) or
            not isinstance(snapshot.get("process_tree"), list) or
            not snapshot["process_tree"] or
            not isinstance(snapshot.get("known_process_identities"), list) or
            not isinstance(snapshot.get("allowed_services"), dict) or
            not isinstance(snapshot.get("job_gpu_processes"), list) or
            not isinstance(snapshot.get("service_gpu_processes"), list) or
            snapshot.get("unknown_gpu_processes") != [] or
            snapshot.get("errors") != [] or
            not isinstance(accounting, dict) or
            set(accounting) != _RUNTIME_GPU_ACCOUNTING_FIELDS or
            not isinstance(gpu, dict) or
            set(gpu) != _GPU_SNAPSHOT_FIELDS or
            not isinstance(gpu.get("compute_app_records"), list)):
        raise ValueError(f"{label} runtime GPU telemetry is incomplete/failing")
    tree = snapshot["process_tree"]
    known = snapshot["known_process_identities"]
    if (any(not _valid_captured_process_record(value) for value in [*tree, *known]) or
            len({value["pid"] for value in tree}) != len(tree) or
            len({value["pid"] for value in known}) != len(known) or
            snapshot["root_pid"] not in {value["pid"] for value in tree}):
        raise ValueError(f"{label} captured process identities are incomplete")
    tree_by_pid = {value["pid"]: value for value in tree}
    known_by_pid = {value["pid"]: value for value in known}
    if any(known_by_pid.get(pid) != value for pid, value in tree_by_pid.items()):
        raise ValueError(f"{label} process tree is not authenticated by captured identities")
    rooted = {snapshot["root_pid"]}
    changed = True
    while changed:
        changed = False
        for value in tree:
            if value["pid"] not in rooted and value["ppid"] in rooted:
                rooted.add(value["pid"]); changed = True
    if rooted != set(tree_by_pid):
        raise ValueError(f"{label} process tree contains a row outside the launched child")

    with open(manifest["environment_manifest"], encoding="utf-8") as handle:
        sealed_environment = json.load(handle)
    if (gpu.get("observer") != expected_input_signature("/usr/bin/nvidia-smi") or
            gpu.get("gpu_uuid") != sealed_environment.get("gpu_uuid") or
            gpu.get("gpu_name") != sealed_environment.get("gpu_name") or
            gpu.get("gpu_driver") != sealed_environment.get("gpu_driver") or
            manifest["child_environment"].get("CUDA_VISIBLE_DEVICES") !=
            gpu.get("gpu_uuid")):
        raise ValueError(f"{label} GPU observation is not bound to the sealed device")
    compute_records = gpu["compute_app_records"]
    if (any(not isinstance(value, dict) or set(value) != {
            "gpu_uuid", "pid", "used_memory_mb"} or
            value.get("gpu_uuid") != gpu["gpu_uuid"] or
            not isinstance(value.get("pid"), int) or isinstance(value["pid"], bool) or
            value["pid"] <= 0 or not isinstance(value.get("used_memory_mb"), (int, float)) or
            isinstance(value["used_memory_mb"], bool) or
            not math.isfinite(float(value["used_memory_mb"])) or
            float(value["used_memory_mb"]) < 0 for value in compute_records) or
            len({value["pid"] for value in compute_records}) != len(compute_records) or
            gpu.get("compute_pids") != [value["pid"] for value in compute_records] or
            gpu.get("n_co_tenants") != len(compute_records)):
        raise ValueError(f"{label} GPU process rows are malformed or device-ambiguous")
    service_by_pid = {value["pid"]: value for value in manifest["allowed_processes"]}
    if set(service_by_pid) & set(tree_by_pid):
        raise ValueError(f"{label} service PID aliases the launched child tree")
    allowed = snapshot["allowed_services"]
    if (not isinstance(allowed, dict) or set(allowed) != {
            "expected", "observed", "gpu_snapshot"} or
            allowed.get("expected") != manifest["allowed_processes"] or
            allowed.get("observed") != manifest["allowed_processes"] or
            allowed.get("gpu_snapshot") != gpu):
        raise ValueError(f"{label} allowed service process identities are unauthenticated")
    expected_job = [value for value in compute_records if value["pid"] in tree_by_pid]
    expected_services = [value for value in compute_records if value["pid"] in service_by_pid]
    expected_unknown = [value for value in compute_records
                        if value["pid"] not in tree_by_pid and
                        value["pid"] not in service_by_pid]
    if (snapshot["job_gpu_processes"] != expected_job or
            snapshot["service_gpu_processes"] != expected_services or
            snapshot["unknown_gpu_processes"] != expected_unknown or expected_unknown):
        raise ValueError(f"{label} GPU process allocation does not match authenticated PIDs")
    numeric = {}
    for key in _RUNTIME_GPU_ACCOUNTING_FIELDS:
        value = accounting[key]
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                not math.isfinite(float(value)) or float(value) < 0):
            raise ValueError(f"{label} runtime GPU accounting leaf {key} is invalid")
        numeric[key] = float(value)
    if (not isinstance(expected_cap, int) or isinstance(expected_cap, bool) or
            numeric["job_reserved_cap_mb"] != float(expected_cap) or
            numeric["job_allocated_accounted_mb"] > float(expected_cap) or
            numeric["cumulative_allocated_accounted_mb"] !=
            numeric["job_allocated_accounted_mb"] +
            numeric["service_allocated_accounted_mb"] or
            numeric["cumulative_reserved_cap_mb"] !=
            numeric["job_reserved_cap_mb"] + numeric["service_reserved_cap_mb"] or
            numeric["device_total_mb"] <= 0 or
            numeric["cumulative_allocated_accounted_mb"] >
            numeric["device_total_mb"] or
            numeric["cumulative_reserved_cap_mb"] > numeric["device_total_mb"] or
            float(gpu.get("total_mb", -1)) != numeric["device_total_mb"]):
        raise ValueError(f"{label} runtime GPU accounting does not recompute")
    recomputed_job = sum(float(value["used_memory_mb"]) for value in expected_job)
    recomputed_service = sum(float(value["used_memory_mb"]) for value in expected_services)
    if (numeric["job_allocated_accounted_mb"] != recomputed_job or
            numeric["service_allocated_accounted_mb"] != recomputed_service or
            (numeric["job_allocated_accounted_mb"] > 0 and not expected_job)):
        raise ValueError(f"{label} claimed job GPU allocation lacks process evidence")
    return numeric["job_allocated_accounted_mb"]


def _registry_signature(registry: dict, path: str) -> dict:
    canonical = os.path.realpath(path)
    value = registry.get(canonical) if isinstance(registry, dict) else None
    if (not isinstance(value, dict) or
            value.get("signature") != expected_input_signature(canonical)):
        raise ValueError(f"controller cumulative registry omits/stales {canonical}")
    return value["signature"]


def _validate_controller_job(*, raw_job: dict, evidence: dict, registry: dict,
                             manifest_sha: str, controller_id: str,
                             manifest: dict,
                             controller_claim: dict) -> dict:
    fields = {"controller_job", "done_marker", "log", "outputs"}
    if not isinstance(evidence, dict) or set(evidence) != fields:
        raise ValueError(f"controller evidence fields mismatch for {raw_job['id']}")
    expected_paths = {
        "controller_job": raw_job["manifest"],
        "done_marker": raw_job["done_marker"], "log": raw_job["log"],
    }
    for label, path in expected_paths.items():
        if evidence[label] != expected_input_signature(path):
            raise ValueError(f"{raw_job['id']} {label} signature/path changed")
        _registry_signature(registry, path)
    expected_outputs = [expected_input_signature(path) for path in raw_job["outputs"]]
    if evidence["outputs"] != expected_outputs:
        raise ValueError(f"{raw_job['id']} output signatures/order changed")
    for path in raw_job["outputs"]:
        _registry_signature(registry, path)

    controller = _json_object(raw_job["manifest"], label="controller job receipt")
    done = _json_object(raw_job["done_marker"], label="controller done marker")
    controller_fields = {
        "schema", "controller_id", "fixture_only", "job", "manifest_sha256",
        "controller_claim_sha256", "child_capability", "child_capability_ack",
        "runtime_contract", "record", "post_child_integrity", "watchdog_clean",
        "finished",
    }
    done_fields = {
        "schema", "status", "fixture_only", "job", "manifest_sha256",
        "controller_claim_sha256",
        "runtime_contract_sha256", "output_signatures", "post_child_integrity",
        "finished", "watchdog_verdict_sha256",
    }
    record = controller.get("record") if isinstance(controller, dict) else None
    capability = controller.get("child_capability") if isinstance(controller, dict) else None
    acknowledgement = (controller.get("child_capability_ack")
                       if isinstance(controller, dict) else None)
    claim_sha = controller_claim["identity_sha256"]
    output_map = {path: expected_input_signature(path) for path in raw_job["outputs"]}
    if (set(controller) != controller_fields or
            controller.get("schema") != "round0005_controller_job.v3" or
            controller.get("fixture_only") is not False or
            controller.get("controller_id") != controller_id or
            controller.get("job") != raw_job["id"] or
            controller.get("manifest_sha256") != manifest_sha or
            controller.get("controller_claim_sha256") != claim_sha or
            not isinstance(capability, dict) or
            capability.get("controller_claim") != controller_claim or
            capability.get("controller_id") != controller_id or
            capability.get("node_id") != raw_job["id"] or
            acknowledgement != {
                "schema": "round0005_child_capability_ack.v1",
                "launch_nonce": capability.get("launch_nonce"),
                "pid": capability.get("child_pid"),
                "capability_sha256": sha256_bytes(canonical_json(capability)),
            } or
            controller.get("runtime_contract") != raw_job or
            controller.get("post_child_integrity") is not True or
            controller.get("watchdog_clean") is not True or
            not isinstance(record, dict) or record.get("status") != "ok" or
            record.get("child_capability_ack") != acknowledgement or
            record.get("exit_code") != 0 or
            record.get("output_signatures") != output_map or
            not _valid_watchdog_verdict(record.get("watchdog_verdict"),
                                        job=raw_job["id"],
                                        controller_claim_sha256=claim_sha)):
        raise ValueError(f"{raw_job['id']} controller receipt is not genuine/successful")
    watchdog_sha = record["watchdog_verdict"]["identity_sha256"]
    controller_gpu_mb = _validate_runtime_gpu_snapshot(
        record.get("gpu_post"), raw_job=raw_job, manifest=manifest,
        label=f"{raw_job['id']} controller-final")
    watchdog_snapshot = record["watchdog_verdict"].get("snapshot")
    watchdog_gpu_mb = (0.0 if watchdog_snapshot is None else
                       _validate_runtime_gpu_snapshot(
                           watchdog_snapshot, raw_job=raw_job, manifest=manifest,
                           label=f"{raw_job['id']} watchdog-final"))
    if (set(done) != done_fields or
            done.get("schema") != "round0005_controller_done.v3" or
            done.get("fixture_only") is not False or done.get("status") != "ok" or
            done.get("job") != raw_job["id"] or
            done.get("manifest_sha256") != manifest_sha or
            done.get("controller_claim_sha256") != claim_sha or
            done.get("runtime_contract_sha256") !=
            sha256_bytes(canonical_json(raw_job)) or
            done.get("output_signatures") != output_map or
            done.get("post_child_integrity") is not True or
            done.get("watchdog_verdict_sha256") != watchdog_sha):
        raise ValueError(f"{raw_job['id']} done marker is forged/incomplete")
    return {
        "controller": controller, "done": done, "outputs": output_map,
        "controller_final_gpu_mb": controller_gpu_mb,
        "watchdog_final_gpu_mb": watchdog_gpu_mb,
    }


def _complete_scalar_leaves(report: dict, *, label: str) -> dict:
    leaves = extract_persisted_scientific_scalars(report)
    if not leaves:
        raise ValueError(f"{label} has no persisted scientific scalar leaves")
    invalid = []
    for key, value in leaves.items():
        if value is None:
            invalid.append(key)
        elif isinstance(value, float) and not math.isfinite(value):
            invalid.append(key)
        elif not isinstance(value, (str, bool, int, float)):
            invalid.append(key)
    if invalid:
        raise ValueError(f"{label} has missing/null/nonfinite scalar leaves: {invalid[:8]}")
    return leaves


_COMPLETE_REPORT_FIELDS = {
    "testbed", "n", "n_holdout", "n_holdout_unique",
    "held_disjoint_from_train", "held_hash", "query_artifact",
    "sample_indices_hash", "frac", "n_anchors", "seed", "scorer_commit",
    "scorer_dirty", "formula_version", "private_outputs",
    "hiD_reference_path", "hiD_reference_key",
    "hiD_reference_content_sha256", "hiD_reference_receipt",
    "pre_gate_reference_consumed", "runs", "query_truth_peak", "started",
    "finished", "total_wall_s", "query_truth_cache", "query_truth_contract",
    "process_cuda_peak", "peak_gpu_allocated_gb", "peak_gpu_reserved_gb",
    "peak_gpu_gb", "performance_gate",
}
_COMPLETE_RUN_FIELDS = {
    "run_dir", "wall_s", "no_model_reference", "ffr", "recall@k",
    "purity_k256", "purity_k1024", "density", "proj_ffr",
    "proj_recall@k", "proj_knn_regressor_ffr", "proj_random_floor_ffr",
    "proj_beats_knn", "proj_margin_over_knn", "coordinate_alignment",
    "hiD_reference_key", "hiD_reference_reused", "panel_full",
}

_QUERY_TRUTH_TELEMETRY_FIELDS = {
    "enabled", "key", "path", "maximum_k", "build_count",
    "disk_load_count", "consumer_count", "hit_count", "consumers",
    "build_wall_s",
}


def _validate_query_truth_evidence(report: dict, *, context: dict,
                                   cache_enabled: bool,
                                   expected_cache_path: str | None,
                                   label: str) -> dict | None:
    telemetry = report.get("query_truth_cache")
    if not isinstance(telemetry, dict) or set(telemetry) != _QUERY_TRUTH_TELEMETRY_FIELDS:
        raise ValueError(f"{label} query-truth telemetry fields are not exact")
    _data_identity, _centroids, corpus_identity = _sealed_reference_identities(context)
    query = context["query"]
    query_identity = {
        "artifact_identity_sha256": query["identity_sha256"],
        "ordered_query_ids_sha256": query["manifest"]["ordered_ids_sha256"],
        "ordered_query_embeddings_sha256":
            query["manifest"]["ordered_embeddings_sha256"],
    }
    cfg = PanelV2Config(frac=0.001, n_anchors=10_000, corpus_chunk=500_000)
    expected_key, expected_parts = query_truth_key(
        corpus_identity=corpus_identity, query_identity=query_identity, cfg=cfg,
        k=15, corpus_cardinality=2_000_000, query_rows=20_000,
        dimensions=768, candidate_compute_backend="cuda")
    consumer_count = telemetry.get("consumer_count")
    expected_npz = (os.path.join(expected_cache_path, f"{expected_key}.npz")
                    if cache_enabled else None)
    build_wall = telemetry.get("build_wall_s")
    if (telemetry.get("enabled") is not cache_enabled or
            telemetry.get("key") != expected_key or
            telemetry.get("path") != expected_npz or
            telemetry.get("maximum_k") != 15 or
            telemetry.get("build_count") != 1 or
            telemetry.get("disk_load_count") != 0 or
            not isinstance(consumer_count, int) or isinstance(consumer_count, bool) or
            consumer_count != len(telemetry.get("consumers") or []) or
            telemetry.get("hit_count") != consumer_count - 1 or
            not isinstance(build_wall, (int, float)) or isinstance(build_wall, bool) or
            not math.isfinite(float(build_wall)) or float(build_wall) < 0):
        raise ValueError(f"{label} query-truth build/load/hit relationships are invalid")
    if not cache_enabled:
        if expected_cache_path is not None:
            raise ValueError(f"{label} disabled query cache has a reported root")
        return None
    if (not isinstance(expected_cache_path, str) or
            os.path.realpath(expected_cache_path) != expected_cache_path or
            os.path.islink(expected_cache_path) or
            not os.path.isdir(expected_cache_path) or
            os.listdir(expected_cache_path) != [f"{expected_key}.npz"] or
            os.path.islink(expected_npz)):
        raise ValueError(f"{label} query-truth cache directory is empty/wrong/ambiguous")
    reopened = load_query_truth(
        expected_npz, expected_key=expected_key, expected_key_parts=expected_parts,
        expected_candidate_compute_backend="cuda")
    if (reopened.get("path") != expected_npz or
            reopened.get("build_wall_s") != build_wall or
            reopened.get("query_rows") != 20_000 or
            reopened.get("corpus_cardinality") != 2_000_000 or
            reopened.get("k") != 15):
        raise ValueError(f"{label} query-truth NPZ differs from reported telemetry")
    return reopened
_PANEL_FULL_FIELDS = {
    "schema", "formula_version", "n", "n_dims_hi", "n_dims_lo", "frac",
    "k_hit", "k_frac", "k_density", "anchor_seed", "n_anchors",
    "anchor_hash", "ffr", "recall@k", "n_ffr_anchors", "purity",
    "purity_numerators", "purity_exactness", "centroid_hashes",
    "n_purity_anchors", "density", "n_density_anchors", "provenance",
    "guards",
}


def _validate_complete_report_structure(report: dict, *, context: dict,
                                        cache_enabled: bool,
                                        expected_report_path: str,
                                        expected_cache_path: str | None,
                                        label: str) -> None:
    if set(report) != _COMPLETE_REPORT_FIELDS:
        raise ValueError(f"{label} top-level fields are incomplete or unexpected")
    expected_private = {
        "report": expected_report_path,
        "hiD_reference": os.path.join(os.path.dirname(expected_report_path),
                                       "hiD-reference.npz"),
        "hiD_reference_receipt": os.path.join(
            os.path.dirname(expected_report_path), "hiD-reference-receipt.json"),
        "query_truth_cache": expected_cache_path,
    }
    query = report.get("query_artifact")
    expected_query = context["query"]
    if (report.get("n_holdout") != 20_000 or
            report.get("n_holdout_unique") != 20_000 or
            report.get("held_disjoint_from_train") is not True or
            not _valid_sha256(report.get("held_hash")) or
            not isinstance(query, dict) or
            query.get("manifest_path") != expected_query.get("manifest_path") or
            query.get("manifest_sha256") != expected_query.get("manifest_sha256") or
            query.get("identity_sha256") != expected_query.get("identity_sha256") or
            not _valid_sha256(report.get("sample_indices_hash")) or
            report.get("frac") != 0.001 or report.get("n_anchors") != 10_000 or
            report.get("seed") != 123 or
            report.get("formula_version") != FORMULA_VERSION or
            report.get("scorer_commit") != context["release_sha"][:12] or
            report.get("scorer_dirty") is not False or
            report.get("private_outputs") != expected_private or
            report.get("hiD_reference_path") != expected_private["hiD_reference"] or
            report.get("pre_gate_reference_consumed") is not False):
        raise ValueError(f"{label} corpus/query/source/private-output identity is incomplete")

    expected_purity = {"k256", "k1024"}
    for map_label in sorted(MAP_EXPECTATIONS):
        run = report["runs"].get(map_label)
        panel = run.get("panel_full") if isinstance(run, dict) else None
        provenance = panel.get("provenance") if isinstance(panel, dict) else None
        guards = panel.get("guards") if isinstance(panel, dict) else None
        coordinate = run.get("coordinate_alignment") if isinstance(run, dict) else None
        if (not isinstance(run, dict) or set(run) != _COMPLETE_RUN_FIELDS or
                not isinstance(panel, dict) or set(panel) != _PANEL_FULL_FIELDS or
                panel.get("schema") != "panel_v2" or
                panel.get("formula_version") != report["formula_version"] or
                panel.get("n") != 2_000_000 or panel.get("n_dims_hi") != 768 or
                panel.get("n_dims_lo") != 2 or panel.get("frac") != 0.001 or
                panel.get("k_hit") != 10 or panel.get("k_frac") != 2_000 or
                panel.get("k_density") != 15 or panel.get("n_anchors") != 10_000 or
                panel.get("n_ffr_anchors") != 10_000 or
                panel.get("n_purity_anchors") != 10_000 or
                panel.get("n_density_anchors") != 10_000 or
                set(panel.get("purity") or {}) != expected_purity or
                set(panel.get("purity_numerators") or {}) != expected_purity or
                set(panel.get("centroid_hashes") or {}) != expected_purity or
                any(not isinstance(value, dict) or
                    set(value) != {"hi_D_agreement", "map_agreement"}
                    for value in (panel.get("purity_numerators") or {}).values()) or
                not isinstance(guards, dict) or set(guards) != {
                    "coords_finite", "coords_collapsed", "emb_finite",
                    "emb_norm_mean", "emb_zero_rows", "hit_guard", "density_guard"} or
                any(not isinstance(guards.get(key), dict) or
                    set(guards[key]) != {"boundary_min_gap", "overselect"} or
                    guards[key].get("overselect") != 8
                    for key in ("hit_guard", "density_guard")) or
                guards.get("coords_finite") is not True or
                guards.get("coords_collapsed") is not False or
                guards.get("emb_finite") is not True or
                guards.get("emb_zero_rows") != 0 or
                not isinstance(guards.get("emb_norm_mean"), (int, float)) or
                isinstance(guards.get("emb_norm_mean"), bool) or
                abs(float(guards["emb_norm_mean"]) - 1.0) > 0.01 or
                not isinstance(provenance, dict) or
                provenance.get("hiD_reference_key") != run.get("hiD_reference_key") or
                provenance.get("hiD_reference_reused") is not True or
                run.get("hiD_reference_reused") is not True or
                run.get("no_model_reference") is not False or
                run.get("ffr") != panel.get("ffr") or
                run.get("recall@k") != panel.get("recall@k") or
                run.get("density") != panel.get("density") or
                run.get("purity_k256") != panel["purity"].get("k256") or
                run.get("purity_k1024") != panel["purity"].get("k1024") or
                not isinstance(coordinate, dict) or set(coordinate) != {
                    "policy", "gathered_rows_sha256"} or
                coordinate.get("policy") !=
                "semantic IDs gathered into canonical corpus order" or
                not _valid_sha256(coordinate.get("gathered_rows_sha256"))):
            raise ValueError(f"{label} map {map_label} scientific structure is incomplete")

    query_peak = report.get("query_truth_peak") or {}
    process_peak = report.get("process_cuda_peak") or {}
    if (not _valid_process_cuda_peak(
            query_peak, reported_maximum=query_peak.get("maximum_gib")) or
            report.get("peak_gpu_allocated_gb") != process_peak.get("allocated_gib") or
            report.get("peak_gpu_reserved_gb") != process_peak.get("reserved_gib") or
            query_peak.get("maximum_gib", 0) > process_peak.get("maximum_gib", -1)):
        raise ValueError(f"{label} process/query CUDA telemetry is inconsistent")

    consumers = validate_query_truth_consumers(
        report.get("query_truth_cache") or {}, sorted(MAP_EXPECTATIONS),
        k_hit=10, expected_builds=1)
    if report.get("query_truth_contract") != consumers or consumers["passed"] is not True:
        raise ValueError(f"{label} query-truth consumer telemetry does not recompute")
    _validate_query_truth_evidence(
        report, context=context, cache_enabled=cache_enabled,
        expected_cache_path=expected_cache_path, label=label)
    gate = report.get("performance_gate")
    expected_gate_checks = {
        "highd_build_count": True, "exact_query_consumers": True,
        "exact_round0005_map_labels": True, "wall": True, "peak_gpu": True,
    }
    if (not isinstance(gate, dict) or set(gate) != {
            "passed", "checks", "wall_max_s", "peak_gpu_max_gb",
            "expected_highd_builds"} or gate.get("passed") is not True or
            gate.get("checks") != expected_gate_checks or
            (expected_cache_path is None) != (not cache_enabled)):
        raise ValueError(f"{label} measured performance gate is incomplete")


def _controller_wall_matches(report_wall: float, controller: dict, *,
                             component_wall: float | None = None) -> bool:
    seconds = controller.get("record", {}).get("seconds")
    measured = report_wall if component_wall is None else component_wall
    if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or
            not math.isfinite(float(seconds)) or float(seconds) <= 0 or
            not math.isfinite(float(measured)) or float(measured) < 0):
        return False
    # Controller time begins before the child handshake and ends after output
    # closure, so it must cover scorer time with only bounded orchestration slack.
    slack = float(seconds) - float(measured)
    return slack >= -1.0 and slack <= max(60.0, float(seconds) * 0.25)


def _sha1_16(path: str) -> str:
    import hashlib
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _validate_complete_panel_report(report: dict, *, context: dict,
                                    cache_enabled: bool,
                                    expected_wall_limit: float | None,
                                    expected_report_path: str,
                                    expected_cache_path: str | None,
                                    label: str) -> dict:
    labels = sorted(MAP_EXPECTATIONS)
    runs = report.get("runs")
    if (report.get("n") != 2_000_000 or report.get("testbed") != context["testbed"] or
            not isinstance(runs, dict) or list(runs) != labels):
        raise ValueError(f"{label} is not the exact ordered 2M nine-map benchmark")
    _validate_complete_report_structure(
        report, context=context, cache_enabled=cache_enabled,
        expected_report_path=expected_report_path,
        expected_cache_path=expected_cache_path, label=label)
    map_identities = {}
    for map_label in labels:
        run = runs.get(map_label)
        panel = run.get("panel_full") if isinstance(run, dict) else None
        map_root = context["maps"][map_label]
        coords = os.path.join(map_root, "coords.parquet")
        coords_sha = _sha1_16(coords)
        provenance = panel.get("provenance") if isinstance(panel, dict) else None
        required_summary = {
            "ffr", "recall@k", "purity_k256", "purity_k1024", "density",
            "proj_ffr", "proj_recall@k", "proj_knn_regressor_ffr",
            "proj_random_floor_ffr", "proj_beats_knn", "proj_margin_over_knn",
        }
        if (not isinstance(run, dict) or not required_summary.issubset(run) or
                any(run[key] is None for key in required_summary) or
                run.get("run_dir") != os.path.basename(map_root) or
                not isinstance(panel, dict) or panel.get("schema") != "panel_v2" or
                panel.get("n") != 2_000_000 or not isinstance(provenance, dict) or
                provenance.get("scorer") != "complete_panel" or
                provenance.get("run") != os.path.basename(map_root) or
                provenance.get("coords_sha") != coords_sha or
                provenance.get("hiD_reference_reused") is not True):
            raise ValueError(f"{label} map {map_label} is incomplete or not map-bound")
        map_identities[map_label] = {
            "map_root": expected_input_signature(map_root),
            "coords": expected_input_signature(coords), "coords_sha1_16": coords_sha,
        }
    telemetry = report.get("query_truth_cache")
    consumers = validate_query_truth_consumers(
        telemetry or {}, labels, k_hit=10, expected_builds=1)
    gate = report.get("performance_gate")
    private = report.get("private_outputs")
    reference_keys = {runs[name].get("hiD_reference_key") for name in labels}
    if (not consumers["passed"] or not isinstance(gate, dict) or
            gate.get("passed") is not True or
            gate.get("expected_highd_builds") != 1 or
            gate.get("wall_max_s") != expected_wall_limit or
            gate.get("peak_gpu_max_gb") != (26.0 if expected_wall_limit else None) or
            not isinstance(gate.get("checks"), dict) or
            not all(gate["checks"].values()) or
            not isinstance(private, dict) or
            private.get("query_truth_cache") != expected_cache_path or
            (expected_cache_path is None) != (not cache_enabled) or
            len(reference_keys) != 1 or None in reference_keys or
            _load_private_reference_receipt(
                report, reference_keys, context=context) is None or
            not _valid_process_cuda_peak(
                report.get("process_cuda_peak") or {},
                reported_maximum=report.get("peak_gpu_gb"))):
        raise ValueError(f"{label} telemetry/cache/reference evidence is incomplete")
    wall = report.get("total_wall_s")
    if (not isinstance(wall, (int, float)) or isinstance(wall, bool) or
            not math.isfinite(float(wall)) or float(wall) < 0 or
            (expected_wall_limit is not None and float(wall) > expected_wall_limit)):
        raise ValueError(f"{label} wall measurement is invalid")
    leaves = _complete_scalar_leaves(report, label=label)
    return {"map_identities": map_identities, "scalar_leaves": leaves,
            "wall_s": float(wall), "peak_gpu_gb": report["peak_gpu_gb"],
            "telemetry": telemetry}


def _validate_terminal_gate(*, terminal: dict, manifest: dict,
                            manifest_path: str, registry: dict) -> dict:
    receipt_path = terminal.get("terminal_gate_receipt")
    checkpoint_path = terminal.get("terminal_comprehensive_checkpoint")
    _registry_signature(registry, receipt_path)
    _registry_signature(registry, checkpoint_path)
    receipt = _json_object(receipt_path, label="terminal gate receipt")
    identity = receipt.get("identity_sha256")
    body = {key: receipt[key] for key in receipt if key != "identity_sha256"}
    if (receipt.get("schema") != "round0005_integrity_receipt.v3" or
            receipt.get("phase") != "terminal-gate-boundary" or
            receipt.get("status") != "matched" or receipt.get("error") is not None or
            receipt.get("original_manifest_path") != os.path.realpath(manifest_path) or
            receipt.get("original_manifest_sha256") != sha256_file(manifest_path) or
            receipt.get("current_manifest_sha256") != sha256_file(manifest_path) or
            receipt.get("expected") != receipt.get("observed") or
            identity != sha256_bytes(canonical_json(body))):
        raise ValueError("terminal gate receipt is forged/stale")
    checkpoint = _json_object(checkpoint_path, label="terminal comprehensive checkpoint")
    stored_gate = checkpoint.get("terminal_gate_receipt")
    if (checkpoint.get("event") != "terminal-comprehensive" or
            checkpoint.get("integrity_match") is not True or
            not isinstance(stored_gate, dict) or
            stored_gate.get("receipt_path") != receipt_path):
        raise ValueError("terminal comprehensive checkpoint is incomplete")
    from basemap.roundwatch_gate import RoundwatchGateAuthority
    current_gate = RoundwatchGateAuthority().check(
        manifest=manifest, manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path))
    stable = {"instance_id", "gate_id", "authority", "round_event",
              "gate_prepared_event", "control_event"}
    recorded_identity = stored_gate.get("gate", {}).get("event_identity", {})
    current_identity = current_gate.get("event_identity", {})
    if ({key: recorded_identity.get(key) for key in stable} !=
            {key: current_identity.get(key) for key in stable}):
        raise ValueError("terminal/current Roundwatch control identity changed")
    return current_identity


def _validate_controller_checkpoint_journal(*, terminal: dict, manifest: dict,
                                            registry: dict) -> dict:
    from basemap.round0005_program import ROUND0005_NODES

    root = os.path.realpath(manifest["controller_checkpoints_dir"])
    names = sorted(os.listdir(root))
    paths = [os.path.join(root, name) for name in names]
    if (not names or any(os.path.dirname(path) != root for path in paths) or
            set(paths) != {path for path in registry if os.path.dirname(path) == root}):
        raise ValueError("controller checkpoint journal is absent or not cumulative")
    records = []
    claim = terminal.get("controller_claim")
    claim_sha = terminal.get("controller_claim_sha256")
    for sequence, path in enumerate(paths):
        _registry_signature(registry, path)
        record = _json_object(path, label="controller checkpoint")
        event = record.get("event")
        if (record.get("schema") != "round0005_controller_checkpoint.v1" or
                record.get("sequence") != sequence or
                not isinstance(event, str) or
                os.path.basename(path) != f"{sequence:06d}-{event}.json" or
                record.get("controller_id") != terminal["controller_id"] or
                record.get("controller_claim_sha256") != claim_sha):
            raise ValueError("controller checkpoint sequence/identity is forged")
        records.append(record)
    if any(value["event"] in {"exception", "runtime-violation"}
           for value in records):
        raise ValueError("controller violation checkpoint forbids scale certification")

    expected_nodes = [node.node_id for node in ROUND0005_NODES]
    core = [value for value in records if value["event"] != "gpu-telemetry"]
    expected_core = ["admission", "lease-acquired"]
    for _node in expected_nodes:
        expected_core.extend(("boundary", "launch", "completion",
                              "cumulative-registry"))
    expected_core.extend(("terminal-comprehensive", "terminal"))
    if [value["event"] for value in core] != expected_core:
        raise ValueError("controller checkpoint event DAG is incomplete or reordered")
    terminal_jobs = {value["name"]: value for value in terminal["jobs"]}
    if (core[0].get("controller_claim") != claim or
            claim.get("construction_receipt") != expected_input_signature(
                core[0].get("construction_receipt")) or
            core[0].get("controller_entry_gate_sha256") !=
            claim.get("entry_gate_sha256") or
            sha256_bytes(canonical_json(core[0].get("controller_entry_gate"))) !=
            claim.get("entry_gate_sha256")):
        raise ValueError("controller admission checkpoint omits the issued claim")
    cursor = 2
    for node_id in expected_nodes:
        boundary, launch, completion, cumulative = core[cursor:cursor + 4]
        cursor += 4
        if any(value.get("job") != node_id
               for value in (boundary, launch, completion)) or \
                cumulative.get("completed_job") != node_id or \
                completion.get("status") != "ok":
            raise ValueError(f"controller checkpoint job chain is invalid for {node_id}")
        gate_receipt = boundary.get("gate_receipt")
        ack = launch.get("child_capability_ack")
        job_record = terminal_jobs[node_id]
        controller_job = _json_object(
            next(job["manifest"] for job in manifest["jobs"] if job["id"] == node_id),
            label=f"{node_id} controller claim binding")
        capability = controller_job.get("child_capability")
        if (not isinstance(gate_receipt, dict) or
                gate_receipt.get("comprehensive_integrity", {}).get(
                    "integrity_match") is not True or
                _registry_signature(registry, gate_receipt.get("receipt_path")) is None or
                not isinstance(ack, dict) or set(ack) != {
                    "schema", "launch_nonce", "pid", "capability_sha256"} or
                ack.get("schema") != "round0005_child_capability_ack.v1" or
                ack.get("pid") != job_record.get("child_pid") or
                not _valid_sha256(ack.get("capability_sha256")) or
                ack != controller_job.get("child_capability_ack") or
                not isinstance(capability, dict) or
                capability.get("controller_claim") != claim or
                ack.get("capability_sha256") !=
                sha256_bytes(canonical_json(capability)) or
                not isinstance(launch.get("launch_gate_receipt"), dict)):
            raise ValueError(f"controller launch capability/gate is invalid for {node_id}")
        completed = cumulative.get("completed_jobs")
        if (not isinstance(completed, list) or node_id not in completed or
                not _valid_sha256(cumulative.get("registry_sha256"))):
            raise ValueError(f"controller cumulative checkpoint is invalid for {node_id}")
    raw_by_id = {value["id"]: value for value in manifest["jobs"]}
    telemetry_counts = {node_id: 0 for node_id in expected_nodes}
    telemetry_peaks = {node_id: 0.0 for node_id in expected_nodes}
    for telemetry in (value for value in records if value["event"] == "gpu-telemetry"):
        snapshot = telemetry.get("snapshot")
        if (telemetry.get("job") not in expected_nodes or
                not isinstance(snapshot, dict) or
                snapshot.get("job") != telemetry.get("job") or
                snapshot.get("errors") not in (None, [])):
            raise ValueError("controller GPU telemetry checkpoint is malformed/failing")
        node_id = telemetry["job"]
        measured = _validate_runtime_gpu_snapshot(
            snapshot, raw_job=raw_by_id[node_id], manifest=manifest,
            label=f"{node_id} controller journal")
        telemetry_counts[node_id] += 1
        telemetry_peaks[node_id] = max(telemetry_peaks[node_id], measured)
    if any(telemetry_counts[node_id] <= 0 for node_id in expected_nodes):
        raise ValueError("controller GPU telemetry journal omits a canonical node")
    if (core[-2].get("integrity_match") is not True or
            core[-1].get("terminal_verdict") != "passed"):
        raise ValueError("controller terminal checkpoint is not successful")
    return {
        "checkpoint_count": len(records),
        "journal_sha256": sha256_bytes(canonical_json([
            registry[path]["signature"] for path in paths])),
        "events": [value["event"] for value in records],
        "gpu_sample_counts": telemetry_counts,
        "gpu_observed_peak_mb": telemetry_peaks,
    }


def _reopen_production_scale_evidence(*, evidence: dict, row_derivation: dict,
                                      scientific_rows: int,
                                      release_sha: str) -> dict:
    if not isinstance(evidence, dict) or set(evidence) != _PRODUCTION_EVIDENCE_FIELDS:
        raise ValueError("production scale evidence fields are not exact")
    if evidence.get("scale_input") != row_derivation.get("embedding_input"):
        raise ValueError("scale input differs from its reopened row derivation")
    validate_scale_rows(row_derivation, scientific_rows=scientific_rows)

    for label in ("queue_manifest", "controller_terminal",
                  "release_preflight_receipt", "gate_preparation_receipt",
                  "maps_seal", "testbed_seal"):
        if not _valid_signature(evidence.get(label)):
            raise ValueError(f"production scale evidence changed: {label}")
    for label in ("environment_manifest", "environment_freeze", "python_executable"):
        if not _valid_path_signature(evidence.get(label)):
            raise ValueError(f"release environment evidence changed: {label}")

    manifest_path = evidence["queue_manifest"]["canonical_path"]
    manifest = _json_object(manifest_path, label="queue manifest")
    from basemap.gate_preparation import validate_gate_preparation_receipt
    from basemap.queue_admission import ROUND0005_QUEUE_FIELDS
    from basemap.round0005_program import (ROUND0005_NODES,
                                           validate_exact_program)
    from basemap.round0005_staging import (validate_round0005_testbed_seal,
                                           validate_staged_map_seal)
    if (set(manifest) != ROUND0005_QUEUE_FIELDS or
            manifest.get("execution_authority") != "planner-gpu" or
            manifest.get("required_reviews") != [] or
            manifest.get("release_sha") != release_sha):
        raise ValueError("completed queue is not the exact production Round 0005 manifest")
    context = validate_exact_program(
        manifest, manifest_path=manifest_path, repo_root=manifest["repo_root"])
    context = {**context, "release_sha": release_sha}
    validate_gate_preparation_receipt(
        manifest["gate_preparation_receipt"], manifest_path=manifest_path,
        manifest=manifest)
    roles = _role_signatures(manifest)
    if (evidence["gate_preparation_receipt"] !=
            expected_input_signature(manifest["gate_preparation_receipt"]) or
            evidence["maps_seal"] != roles.get("maps_seal") or
            evidence["testbed_seal"] != roles.get("testbed_seal")):
        raise ValueError("certificate seals/gate receipt differ from the queue")
    testbed = validate_round0005_testbed_seal(
        roles["testbed_seal"]["canonical_path"], require_round0005=True)
    maps = validate_staged_map_seal(
        roles["maps_seal"]["canonical_path"],
        expected_testbed_seal=roles["testbed_seal"]["canonical_path"],
        require_round0005=True)
    if maps.get("expected_rows") != 2_000_000 or len(maps.get("maps", [])) != 9:
        raise ValueError("benchmark seals are not exact 2M/nine-map evidence")

    release = validate_release_preflight_receipt(
        roles["release_preflight_receipt"]["canonical_path"],
        expected_identity_sha256=manifest["release_preflight_identity"],
        expected_signature=roles["release_preflight_receipt"])
    if (evidence["release_preflight_receipt"] !=
            roles["release_preflight_receipt"] or
            release.get("release_sha") != release_sha or
            evidence["environment_manifest"] != release["environment_manifest"] or
            evidence["environment_freeze"] != release["environment_freeze"] or
            evidence["python_executable"] != release["python_executable"]):
        raise ValueError("release/environment evidence differs from live remote authority")

    terminal_path = evidence["controller_terminal"]["canonical_path"]
    if terminal_path != manifest["controller_terminal_summary"]:
        raise ValueError("controller terminal path differs from the queue")
    terminal = _json_object(terminal_path, label="controller terminal summary")
    expected_nodes = [node.node_id for node in ROUND0005_NODES]
    if (terminal.get("schema") != "round0005_controller_terminal.v3" or
            terminal.get("fixture_only") is not False or
            terminal.get("terminal_verdict") != "passed" or
            terminal.get("queue_manifest_path") != manifest_path or
            terminal.get("queue_manifest_sha256") != sha256_file(manifest_path) or
            terminal.get("queue_release_sha") != release_sha or
            terminal.get("controller_claim_sha256") !=
            (terminal.get("controller_claim") or {}).get("identity_sha256") or
            terminal.get("required_jobs") != expected_nodes or
            terminal.get("completed_jobs") != expected_nodes or
            [value.get("name") for value in terminal.get("jobs", [])
             if isinstance(value, dict)] != expected_nodes or
            any(value.get("status") != "ok" for value in terminal.get("jobs", []))):
        raise ValueError("controller terminal is not a genuine complete production run")
    controller_claim = terminal["controller_claim"]
    claim_body = {key: controller_claim[key] for key in controller_claim
                  if key != "identity_sha256"}
    process_claim = controller_claim.get("controller_process")
    with open(manifest["environment_manifest"], encoding="utf-8") as handle:
        sealed_environment = json.load(handle)
    expected_python = os.path.realpath(os.path.join(
        sealed_environment["venv_path"], "bin", "python"))
    if (controller_claim.get("schema") != "round0005_queue_controller_claim.v1" or
            controller_claim.get("fixture_only") is not False or
            controller_claim.get("controller_id") != terminal["controller_id"] or
            controller_claim.get("controller_pid") != terminal["controller_pid"] or
            controller_claim.get("controller_starttime_ticks") !=
            terminal["controller_starttime_ticks"] or
            controller_claim.get("manifest") != evidence["queue_manifest"] or
            controller_claim.get("ordered_job_ids") != expected_nodes or
            controller_claim.get("identity_sha256") !=
            sha256_bytes(canonical_json(claim_body)) or
            not isinstance(process_claim, dict) or
            process_claim.get("pid") != terminal["controller_pid"] or
            process_claim.get("proc_starttime_ticks") !=
            terminal["controller_starttime_ticks"] or
            len(process_claim.get("argv") or []) != 4 or
            os.path.realpath(process_claim["argv"][0]) != expected_python or
            process_claim["argv"][1:] != [
                "-m", "basemap.run_controller", manifest_path]):
        raise ValueError("controller terminal QueueAdmission claim is forged/incomplete")
    registry = terminal.get("cumulative_registry")
    if (not isinstance(registry, dict) or terminal.get("cumulative_registry_sha256") !=
            sha256_bytes(canonical_json(registry))):
        raise ValueError("controller terminal cumulative registry is incomplete")
    from basemap.run_controller import _verify_cumulative_registry
    _verify_cumulative_registry(registry)
    if any(os.path.basename(path).startswith("watchdog-") for path in registry):
        raise ValueError("watchdog emergency evidence forbids scale certification")
    checkpoint_journal = _validate_controller_checkpoint_journal(
        terminal=terminal, manifest=manifest, registry=registry)

    jobs_evidence = evidence.get("jobs")
    if (not isinstance(jobs_evidence, dict) or
            list(jobs_evidence) != list(_CERTIFYING_JOB_IDS)):
        raise ValueError("certificate lacks exact ordered controller evidence")
    raw_by_id = {job["id"]: job for job in manifest["jobs"]}
    validated_jobs = {}
    for node_id in _CERTIFYING_JOB_IDS:
        validated_jobs[node_id] = _validate_controller_job(
            raw_job=raw_by_id[node_id], evidence=jobs_evidence[node_id],
            registry=registry, manifest_sha=sha256_file(manifest_path),
            controller_id=terminal["controller_id"], manifest=manifest,
            controller_claim=controller_claim)

    uncached_job = raw_by_id["fresh_uncached_2m"]
    cached_job = raw_by_id["cached_nine_map"]
    scalar_job = raw_by_id["scalar_equivalence"]
    regression_job = raw_by_id["synthetic_4x_regression"]
    uncached = _json_object(uncached_job["outputs"][0], label="uncached panel")
    cached = _json_object(cached_job["outputs"][0], label="cached panel")
    uncached_measure = _validate_complete_panel_report(
        uncached, context=context, cache_enabled=False, expected_wall_limit=720.0,
        expected_report_path=uncached_job["outputs"][0],
        expected_cache_path=None, label="uncached controller benchmark")
    cached_measure = _validate_complete_panel_report(
        cached, context=context, cache_enabled=True, expected_wall_limit=120.0,
        expected_report_path=cached_job["outputs"][0],
        expected_cache_path=cached_job["outputs"][3],
        label="cached controller benchmark")
    if (not _controller_wall_matches(
            uncached_measure["wall_s"],
            validated_jobs["fresh_uncached_2m"]["controller"]) or
            not _controller_wall_matches(
                cached_measure["wall_s"],
                validated_jobs["cached_nine_map"]["controller"])):
        raise ValueError("panel wall measurements differ from controller runtime")
    direct_equivalence = compare_persisted_scalars(
        uncached, cached, label="controller-uncached-vs-cached")
    if direct_equivalence.get("passed") is not True:
        raise ValueError("uncached/cached persisted scalars are not exactly equivalent")

    equivalence = _json_object(
        scalar_job["outputs"][0], label="scalar equivalence report")
    self_contained = equivalence.get("self_contained")
    if (equivalence.get("schema") !=
            "round0005_real_input_cache_scalar_equivalence.v3" or
            equivalence.get("passed") is not True or
            equivalence.get("differences") != {} or
            not isinstance(equivalence.get("checks"), dict) or
            not all(equivalence["checks"].values()) or
            not isinstance(self_contained, dict) or
            self_contained.get("consumed_sibling_job_outputs") is not False):
        raise ValueError("scalar-equivalence controller output is incomplete/failing")
    scalar_uncached = _load_signed_json(
        self_contained["uncached_report"], label="scalar uncached panel")
    scalar_cached = _load_signed_json(
        self_contained["cached_report"], label="scalar cached panel")
    recomputed = compare_persisted_scalars(
        scalar_uncached, scalar_cached, label="real-input-complete-panel")
    expected_equivalence = {
        **recomputed,
        "schema": "round0005_real_input_cache_scalar_equivalence.v3",
        "self_contained": self_contained,
    }
    if equivalence != expected_equivalence:
        raise ValueError("scalar equivalence report does not recompute exactly")
    _validate_complete_panel_report(
        scalar_uncached, context=context, cache_enabled=False,
        expected_wall_limit=None,
        expected_report_path=self_contained["uncached_report"]["canonical_path"],
        expected_cache_path=None,
        label="scalar-equivalence uncached benchmark")
    scalar_cache_path = self_contained["output_contract"]["cached"][
        "query_truth_cache"]
    _validate_complete_panel_report(
        scalar_cached, context=context, cache_enabled=True,
        expected_wall_limit=None,
        expected_report_path=self_contained["cached_report"]["canonical_path"],
        expected_cache_path=scalar_cache_path,
        label="scalar-equivalence cached benchmark")
    scalar_component_wall = (float(scalar_uncached["total_wall_s"]) +
                             float(scalar_cached["total_wall_s"]))
    if not _controller_wall_matches(
            scalar_component_wall,
            validated_jobs["scalar_equivalence"]["controller"],
            component_wall=scalar_component_wall):
        raise ValueError("scalar-equivalence scorer walls differ from controller runtime")

    regression = _json_object(
        regression_job["outputs"][-1], label="4x regression report")
    regression_validation = validate_regression_certificate(regression)
    if not regression_validation["passed"]:
        raise ValueError("controller regression output does not recompute/persist at >=4x")
    baseline_wall = float(regression["baseline"]["wall_s"])
    slowed_wall = float(regression["slowed"]["wall_s"])
    slowdown = slowed_wall / baseline_wall
    if not math.isfinite(slowdown) or slowdown < 4.0:
        raise ValueError("reopened measured regression slowdown is below 4x")
    if not _controller_wall_matches(
            baseline_wall + slowed_wall,
            validated_jobs["synthetic_4x_regression"]["controller"],
            component_wall=baseline_wall + slowed_wall):
        raise ValueError("regression measured walls differ from controller runtime")

    gate_identity = _validate_terminal_gate(
        terminal=terminal, manifest=manifest, manifest_path=manifest_path,
        registry=registry)
    measured_gate = {
        "schema": "round0005_recomputed_2m_nine_map_gate.v1",
        "passed": True, "benchmark_rows": 2_000_000,
        "map_count": 9, "cached_wall_s": cached_measure["wall_s"],
        "cached_peak_gpu_gb": cached_measure["peak_gpu_gb"],
        "cached_query_truth_builds": cached_measure["telemetry"]["build_count"],
        "cached_controller_gpu_samples": checkpoint_journal[
            "gpu_sample_counts"]["cached_nine_map"],
        "cached_controller_observed_peak_mb": max(
            checkpoint_journal["gpu_observed_peak_mb"]["cached_nine_map"],
            validated_jobs["cached_nine_map"]["controller_final_gpu_mb"],
            validated_jobs["cached_nine_map"]["watchdog_final_gpu_mb"]),
        "scalar_equivalence_sha256": sha256_bytes(canonical_json(
            equivalence)), "regression_slowdown": slowdown,
    }
    controller_identity = {
        "terminal": evidence["controller_terminal"],
        "controller_id": terminal["controller_id"],
        "controller_claim_sha256": controller_claim["identity_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "cumulative_registry_sha256": terminal["cumulative_registry_sha256"],
        "checkpoint_journal": checkpoint_journal,
        "gate_identity": gate_identity,
    }
    benchmark_identity = {
        "testbed_seal": evidence["testbed_seal"],
        "testbed_identity_sha256": testbed["identity_sha256"],
        "maps_seal": evidence["maps_seal"],
        "maps_identity_sha256": maps["identity_sha256"],
        "map_identities": cached_measure["map_identities"],
        "uncached_scalar_sha256": sha256_bytes(canonical_json(
            uncached_measure["scalar_leaves"])),
        "cached_scalar_sha256": sha256_bytes(canonical_json(
            cached_measure["scalar_leaves"])),
    }
    return {
        "environment_binding": _release_environment_binding(release),
        "controller_identity": controller_identity,
        "benchmark_identity": benchmark_identity,
        "measured_gate": measured_gate,
        "measured_total_slowdown": slowdown,
    }


def issue_scale_performance_certificate(
        *, queue_manifest_path: str | None = None,
        controller_terminal_summary: str | None = None,
        row_derivation: dict,
        panel_path: str | None = None, regression_path: str | None = None,
        release_preflight_receipt: str | None = None) -> dict:
    """Issue v3 only from an exact completed production controller program."""
    if any(value is not None for value in
           (panel_path, regression_path, release_preflight_receipt)):
        raise ValueError(
            "standalone panel/regression evidence is non-production and cannot issue a "
            "scale certificate")
    if not queue_manifest_path or not controller_terminal_summary:
        raise ValueError("scale certification requires queue manifest and terminal summary")
    scientific_rows = (row_derivation.get("scientific_rows")
                       if isinstance(row_derivation, dict) else None)
    if (not isinstance(scientific_rows, int) or isinstance(scientific_rows, bool) or
            scientific_rows < SCALE_ROWS):
        raise ValueError("scale input must reopen at least 8,000,000 rows")
    validate_scale_rows(row_derivation, scientific_rows=scientific_rows)
    manifest_signature = expected_input_signature(queue_manifest_path)
    terminal_signature = expected_input_signature(controller_terminal_summary)
    manifest = _json_object(queue_manifest_path, label="queue manifest")
    roles = _role_signatures(manifest)
    release = validate_release_preflight_receipt(
        roles["release_preflight_receipt"]["canonical_path"],
        expected_identity_sha256=manifest["release_preflight_identity"],
        expected_signature=roles["release_preflight_receipt"])
    jobs = {}
    by_id = {job["id"]: job for job in manifest.get("jobs", [])}
    for node_id in _CERTIFYING_JOB_IDS:
        raw = by_id[node_id]
        jobs[node_id] = {
            "controller_job": expected_input_signature(raw["manifest"]),
            "done_marker": expected_input_signature(raw["done_marker"]),
            "log": expected_input_signature(raw["log"]),
            "outputs": [expected_input_signature(path) for path in raw["outputs"]],
        }
    evidence = {
        "queue_manifest": manifest_signature,
        "controller_terminal": terminal_signature,
        "release_preflight_receipt": roles["release_preflight_receipt"],
        "gate_preparation_receipt": expected_input_signature(
            manifest["gate_preparation_receipt"]),
        "maps_seal": roles["maps_seal"], "testbed_seal": roles["testbed_seal"],
        "scale_input": row_derivation["embedding_input"],
        "environment_manifest": release["environment_manifest"],
        "environment_freeze": release["environment_freeze"],
        "python_executable": release["python_executable"], "jobs": jobs,
    }
    derived = _reopen_production_scale_evidence(
        evidence=evidence, row_derivation=row_derivation,
        scientific_rows=scientific_rows, release_sha=release["release_sha"])
    body = {
        "schema": PERFORMANCE_CERTIFICATE_SCHEMA, "passed": True,
        "allows_scale_launch": True, "release_sha": release["release_sha"],
        "scientific_rows": scientific_rows, "row_derivation": row_derivation,
        "limits": CANONICAL_SCALE_LIMITS, "evidence": evidence, **derived,
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def validate_scale_performance_certificate(
        report: dict, *, expected_release_sha: str, scientific_rows: int,
        row_derivation: dict) -> dict:
    checks = {
        "exact_fields": isinstance(report, dict) and
            set(report) == _PRODUCTION_CERTIFICATE_FIELDS,
        "schema": isinstance(report, dict) and
            report.get("schema") == PERFORMANCE_CERTIFICATE_SCHEMA,
        "passed": isinstance(report, dict) and report.get("passed") is True,
        "allows_scale_launch": isinstance(report, dict) and
            report.get("allows_scale_launch") is True,
        "release_sha": isinstance(report, dict) and
            report.get("release_sha") == expected_release_sha,
        "scientific_rows": (isinstance(scientific_rows, int) and
                            scientific_rows >= SCALE_ROWS and isinstance(report, dict) and
                            report.get("scientific_rows") == scientific_rows),
        "row_derivation": isinstance(report, dict) and
            report.get("row_derivation") == row_derivation,
        "limits": isinstance(report, dict) and
            report.get("limits") == CANONICAL_SCALE_LIMITS,
        "evidence_reopened": False, "derived_values_exact": False,
        "content_identity": False,
    }
    error = None
    if checks["exact_fields"]:
        try:
            derived = _reopen_production_scale_evidence(
                evidence=report["evidence"], row_derivation=row_derivation,
                scientific_rows=scientific_rows, release_sha=expected_release_sha)
            checks["evidence_reopened"] = True
            checks["derived_values_exact"] = all(
                report.get(key) == value for key, value in derived.items())
            body = {key: report[key] for key in _PRODUCTION_CERTIFICATE_FIELDS
                    if key != "identity_sha256"}
            checks["content_identity"] = (
                _valid_sha256(report.get("identity_sha256")) and
                report["identity_sha256"] == sha256_bytes(canonical_json(body)))
        except (KeyError, OSError, RuntimeError, TypeError, ValueError,
                json.JSONDecodeError) as exc:
            error = str(exc)
    return {"passed": all(checks.values()), "checks": checks, "error": error}


def validate_scale_policy(policy: dict, *, report_path: str, report: dict) -> dict:
    required = {"schema", "release_sha", "scientific_rows", "row_derivation",
                "certificate"}
    certificate = policy.get("certificate") if isinstance(policy, dict) else None
    current = expected_input_signature(report_path)
    expected_certificate = {**current, "schema": PERFORMANCE_CERTIFICATE_SCHEMA}
    try:
        policy_input_identity = scale_input_identity_sha256(
            policy.get("row_derivation"), scientific_rows=policy.get("scientific_rows"))
        report_input_identity = scale_input_identity_sha256(
            report.get("row_derivation"), scientific_rows=report.get("scientific_rows"))
    except (OSError, RuntimeError, TypeError, ValueError):
        policy_input_identity = None
        report_input_identity = None
    checks = {
        "exact_fields": isinstance(policy, dict) and set(policy) == required,
        "schema": isinstance(policy, dict) and policy.get("schema") == SCALE_POLICY_SCHEMA,
        "certificate_current": certificate == expected_certificate,
        "release_bound": isinstance(policy, dict) and
            policy.get("release_sha") == report.get("release_sha"),
        "rows_bound": isinstance(policy, dict) and
            policy.get("scientific_rows") == report.get("scientific_rows"),
        "input_bound": (policy_input_identity is not None and
                          policy_input_identity == report_input_identity),
    }
    return {"passed": all(checks.values()), "checks": checks}


def require_scale_performance_gate(report_path: str | None, *, scientific_rows: int,
                                   row_derivation: dict | None = None,
                                   release_sha: str | None = None,
                                   scale_policy: dict | None = None) -> dict | None:
    """Hard-block every 8M/30M launcher without current content-bound evidence."""
    if scientific_rows < SCALE_ROWS:
        if report_path is not None:
            raise RuntimeError("below-scale launch cannot carry a performance certificate")
        return None
    if not report_path:
        raise RuntimeError(
            f"{scientific_rows:,}-row launcher requires a signed pre-gate performance certificate")
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    if row_derivation is None:
        raise RuntimeError(
            f"{scientific_rows:,}-row launcher requires actual-input row derivation")
    if release_sha is None:
        raise RuntimeError(
            f"{scientific_rows:,}-row launcher requires its exact release SHA")
    validation = validate_scale_performance_certificate(
        report, expected_release_sha=release_sha, scientific_rows=scientific_rows,
        row_derivation=row_derivation)
    if not validation["passed"]:
        raise RuntimeError(
            f"performance certificate blocks {scientific_rows:,}-row launcher: "
            f"{report_path}: {validation['checks']}")
    if scale_policy is not None:
        policy_validation = validate_scale_policy(
            scale_policy, report_path=report_path, report=report)
        if not policy_validation["passed"]:
            raise RuntimeError(
                f"performance scale policy blocks {scientific_rows:,}-row launcher: "
                f"{policy_validation['checks']}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture")
    mode.add_argument("--queue-manifest")
    parser.add_argument("--out-root")
    parser.add_argument("--out")
    parser.add_argument("--baseline-phase-delay", type=float, default=0.01)
    parser.add_argument("--slowdown-injection-factor", type=float, default=16.0)
    parser.add_argument("--controller-terminal",
                        help="exact completed production controller terminal summary")
    parser.add_argument("--scale-row-derivation",
                        help="JSON derivation for the exact >=8M embedding input being released")
    args = parser.parse_args(argv)
    if args.fixture:
        from basemap.run_controller import require_round0005_child_admission
        require_round0005_child_admission(
            "experiments/round0005_performance_gate.py")
        if not args.out_root or args.out:
            parser.error("--fixture requires --out-root and does not accept --out")
        report = run_synthetic_regression(
            fixture_path=args.fixture, out_root=args.out_root,
            baseline_phase_delay_s=args.baseline_phase_delay,
            slowdown_injection_factor=args.slowdown_injection_factor)
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError(
                "post-terminal performance certificate issuance must be CUDA-hidden")
        if not args.out or args.out_root:
            parser.error("--queue-manifest requires --out and does not accept --out-root")
        if not args.controller_terminal or not args.scale_row_derivation:
            parser.error(
                "--queue-manifest requires --controller-terminal and "
                "--scale-row-derivation")
        with open(args.scale_row_derivation, encoding="utf-8") as handle:
            row_derivation = json.load(handle)
        report = issue_scale_performance_certificate(
            queue_manifest_path=args.queue_manifest,
            controller_terminal_summary=args.controller_terminal,
            row_derivation=row_derivation)
        reopened = validate_scale_performance_certificate(
            report, expected_release_sha=report["release_sha"],
            scientific_rows=report["scientific_rows"],
            row_derivation=row_derivation)
        if reopened.get("passed") is not True:
            raise RuntimeError(
                f"new production certificate did not immediately reopen: {reopened}")
        atomic_write_new_json(args.out, report, immutable=True)
    print(json.dumps({"schema": report["schema"], "passed": report["passed"]}, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
