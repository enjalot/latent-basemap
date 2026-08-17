"""Slim-runner >=8M score_panel scale-performance admission (additive to v1).

The v1 (`experiments.round0005_performance_gate`) production-controller path is left
completely untouched.  This module adds the SECOND accepted form of `score_panel`
`scale_admission` used by the slim-runner rounds (first consumer: R0267's 50M panel),
plus the tightly-scoped cert-production bootstrap marker.

It certifies an OPERATIONAL property only — that the scorer's wall / peak-host-RSS /
peak-VRAM behaviour at the target scale does not blow up — bound to this round's OWN
substrate under this exact release.  It changes NO scientific gate value.  See the
pre-registered design `latent-labs/basemap-100m/design-slim-8m-admission-2026-08-17.md`.

Evidence (equivalent-or-better than the retired-v1 2M jina golden testbed proxy):
  (i)   a measured `score_panel` pass at target scale on the round's own substrate under
        the round's `release_sha`, with sealed wall / peak-host-RSS / peak-VRAM receipts
        each <= a pre-registered budget, and `row_derivation` matching X;
  (ii)  the existing >=4x synthetic regression machinery (`run_synthetic_regression`
        + `validate_regression_certificate`), reused unchanged;
  (iii) release-preflight binding via `basemap.release_preflight`, release_sha equality;
  (iv)  the assembled cert is sealed by sha256 and its bytes bound as a digest-verified
        queue input (same signature helpers the round already uses).
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from .artifact_identity import canonical_json, expected_input_signature, sha256_bytes
from . import round0113_prompt_contrast as prompt_contract
from .panel_v2 import _scale_matrix_origin_matches, _valid_sha256
from .release_preflight import validate_release_preflight_receipt
from experiments.round0005_performance_gate import (
    _valid_release_sha,
    validate_regression_certificate,
    validate_scale_rows,
)

#: The lowest scale this admission form governs (same boundary the guard enforces).
SLIM_SCALE_ROWS = 8_000_000

SLIM_SCALE_CERT_SCHEMA = "round0267_slim_scale_performance_certificate.v1"

#: The two additive `scale_admission["kind"]` markers this module recognises.
SLIM_SCALE_CERT_KIND = "slim-scale-cert"
SLIM_CERT_PRODUCTION_KIND = "slim-cert-production"

#: The stamp the production-mode pass writes into its output payload, and the ONLY value
#: the consumption side (gate node / resultcheck) refuses.  A scientific panel never
#: carries it.
SLIM_CERT_PRODUCTION_PURPOSE = "slim-scale-cert-production"

#: Pre-registered budgets for the 50M rung (declared BEFORE the measurement so pass/fail
#: is mechanical).  peak VRAM == the node's DEVICE_BUDGET_BYTES (30 GiB); peak host RSS ==
#: the node's HOST_RSS_LIMIT_GIB (100 GiB); wall == the blowup-catching 1800 s ceiling.
SLIM_WALL_MAX_S = 1800.0
SLIM_PEAK_VRAM_MAX_BYTES = 30 * (1 << 30)
SLIM_PEAK_HOST_RSS_MAX_GIB = 100.0
SLIM_BUDGETS = {
    "wall_max_s": SLIM_WALL_MAX_S,
    "peak_vram_max_bytes": SLIM_PEAK_VRAM_MAX_BYTES,
    "peak_host_rss_max_gib": SLIM_PEAK_HOST_RSS_MAX_GIB,
}

SLIM_ADMISSION_FIELDS = {
    "kind", "release_sha", "scientific_rows", "row_derivation", "certificate",
}
SLIM_PRODUCTION_MARKER_FIELDS = {"kind", "release_sha", "row_derivation", "budgets"}
SLIM_CERT_FIELDS = {
    "schema", "kind", "release_sha", "scientific_rows", "row_derivation", "budgets",
    "measurement", "regression", "release_preflight", "identity_sha256",
}


class SlimScaleAdmissionError(RuntimeError):
    """A slim-runner >=8M scale-performance admission was rejected."""


def _numeric(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(float(value)))


def build_slim_scale_admission(*, release_sha: str, scientific_rows: int,
                               row_derivation: Mapping[str, Any],
                               certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble the slim (kind='slim-scale-cert') `scale_admission` the guard consumes."""
    return {
        "kind": SLIM_SCALE_CERT_KIND,
        "release_sha": str(release_sha),
        "scientific_rows": int(scientific_rows),
        "row_derivation": dict(row_derivation),
        "certificate": dict(certificate),
    }


def slim_cert_production_marker(*, release_sha: str, row_derivation: Mapping[str, Any],
                               budgets: Mapping[str, Any] = SLIM_BUDGETS) -> dict[str, Any]:
    """Assemble the tightly-scoped cert-production `scale_admission` marker."""
    return {
        "kind": SLIM_CERT_PRODUCTION_KIND,
        "release_sha": str(release_sha),
        "row_derivation": dict(row_derivation),
        "budgets": dict(budgets),
    }


def _require_row_derivation_matches(X, row_derivation: Mapping[str, Any], *,
                                    scientific_rows: int) -> None:
    """(i) The scale matrix IS the reopened signed embedding input (not a copy)."""
    validate_scale_rows(dict(row_derivation), scientific_rows=scientific_rows)
    if not _scale_matrix_origin_matches(X, dict(row_derivation)):
        raise SlimScaleAdmissionError(
            "slim scale matrix is not the reopened signed embedding input")


def permit_slim_cert_production(scale_admission: Mapping[str, Any], *, X,
                                scientific_rows: int) -> dict[str, Any]:
    """Permit ONE >=8M measurement pass for the explicit production marker, and cause
    `score_panel` to stamp its output ``purpose='slim-scale-cert-production'``.

    This is the ONLY >=8M-without-cert path.  It carries the target release + substrate
    identity + declared budgets, binds `row_derivation` to the actual matrix, and returns
    a record whose ``purpose`` tells `score_panel` to stamp the output — an output the
    consumption side then REFUSES.  It produces perf receipts and nothing else.
    """
    if (not isinstance(scale_admission, Mapping) or
            set(scale_admission) != SLIM_PRODUCTION_MARKER_FIELDS or
            scale_admission.get("kind") != SLIM_CERT_PRODUCTION_KIND):
        raise SlimScaleAdmissionError(
            "slim cert-production marker fields are not exact")
    release_sha = scale_admission["release_sha"]
    if not _valid_release_sha(release_sha):
        raise SlimScaleAdmissionError(
            "slim cert-production marker release_sha must be lowercase 40-hex")
    if scientific_rows < SLIM_SCALE_ROWS:
        raise SlimScaleAdmissionError(
            "slim cert-production marker is only for >=8,000,000-row passes")
    if dict(scale_admission["budgets"]) != SLIM_BUDGETS:
        raise SlimScaleAdmissionError(
            "slim cert-production marker budgets are not the pre-registered budgets")
    _require_row_derivation_matches(
        X, scale_admission["row_derivation"], scientific_rows=scientific_rows)
    body = {
        "schema": SLIM_SCALE_CERT_SCHEMA,
        "kind": SLIM_CERT_PRODUCTION_KIND,
        "purpose": SLIM_CERT_PRODUCTION_PURPOSE,
        "release_sha": str(release_sha),
        "scientific_rows": int(scientific_rows),
        "row_derivation": dict(scale_admission["row_derivation"]),
        "budgets": dict(SLIM_BUDGETS),
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _reopen_measurement(measurement: Mapping[str, Any], *, budgets: Mapping[str, Any],
                        checks: dict[str, bool]) -> None:
    """(i) Reopen the sealed production-mode panel output and check wall/RSS/VRAM."""
    panel_signature = measurement.get("panel_output") if isinstance(
        measurement, Mapping) else None
    checks["measurement_panel_signature"] = False
    checks["measurement_is_production_pass"] = False
    checks["wall_within_budget"] = False
    checks["peak_host_rss_within_budget"] = False
    checks["peak_vram_within_budget"] = False
    if not isinstance(measurement, Mapping) or set(measurement) != {"panel_output"}:
        return
    try:
        panel_path = prompt_contract.verify_signature(
            panel_signature, label="slim measurement panel output")
    except Exception:
        return
    checks["measurement_panel_signature"] = True
    try:
        with open(panel_path, encoding="utf-8") as handle:
            panel = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return
    if not isinstance(panel, dict):
        return
    checks["measurement_is_production_pass"] = (
        panel.get("purpose") == SLIM_CERT_PRODUCTION_PURPOSE)
    provenance = panel.get("provenance") if isinstance(panel, dict) else None
    if not isinstance(provenance, dict):
        return
    wall = provenance.get("wall_s")
    rss = provenance.get("peak_rss_gb")
    cuda_peak = provenance.get("cuda_peak")
    checks["wall_within_budget"] = (
        _numeric(wall) and float(wall) <= float(budgets["wall_max_s"]))
    checks["peak_host_rss_within_budget"] = (
        _numeric(rss) and float(rss) <= float(budgets["peak_host_rss_max_gib"]))
    if isinstance(cuda_peak, dict) and cuda_peak.get("available") is True:
        allocated = cuda_peak.get("allocated_bytes")
        reserved = cuda_peak.get("reserved_bytes")
        if _numeric(allocated) and _numeric(reserved):
            peak_vram_bytes = max(int(allocated), int(reserved))
            checks["peak_vram_within_budget"] = (
                peak_vram_bytes <= int(budgets["peak_vram_max_bytes"]))


def validate_slim_scale_admission(scale_admission: Mapping[str, Any], *, X,
                                  scientific_rows: int) -> dict[str, Any]:
    """Validate a slim (kind='slim-scale-cert') admission against its sealed cert.

    Returns the reopened certificate record on success; raises
    ``SlimScaleAdmissionError`` (with the failing checks) otherwise.
    """
    if (not isinstance(scale_admission, Mapping) or
            set(scale_admission) != SLIM_ADMISSION_FIELDS or
            scale_admission.get("kind") != SLIM_SCALE_CERT_KIND):
        raise SlimScaleAdmissionError("slim scale admission fields are not exact")
    if scientific_rows < SLIM_SCALE_ROWS:
        raise SlimScaleAdmissionError(
            "slim scale admission is only for >=8,000,000-row passes")

    checks: dict[str, bool] = {}
    error: str | None = None

    # (iv) the cert is a digest-bound queue input: verify its file bytes + inner seal.
    cert: dict[str, Any] | None = None
    try:
        cert_path = prompt_contract.verify_signature(
            scale_admission["certificate"], label="slim scale-performance certificate")
        with open(cert_path, encoding="utf-8") as handle:
            loaded = json.load(handle)
        checks["certificate_digest_bound"] = True
        if isinstance(loaded, dict):
            body = {key: loaded[key] for key in loaded if key != "identity_sha256"}
            checks["certificate_sealed"] = (
                _valid_sha256(loaded.get("identity_sha256")) and
                loaded.get("identity_sha256") == sha256_bytes(canonical_json(body)))
            if checks["certificate_sealed"]:
                cert = loaded
        else:
            checks["certificate_sealed"] = False
    except Exception as exc:  # noqa: BLE001 - fail closed, record the cause
        checks["certificate_digest_bound"] = False
        checks["certificate_sealed"] = False
        error = str(exc)

    if cert is not None:
        checks["exact_cert_fields"] = set(cert) == SLIM_CERT_FIELDS
        checks["schema"] = cert.get("schema") == SLIM_SCALE_CERT_SCHEMA
        checks["kind"] = cert.get("kind") == SLIM_SCALE_CERT_KIND
        # (iii)/(a) release_sha equality: admission == cert == valid 40-hex.
        checks["release_sha_valid"] = _valid_release_sha(scale_admission["release_sha"])
        checks["release_sha_matches_cert"] = (
            scale_admission["release_sha"] == cert.get("release_sha"))
        checks["scientific_rows_match"] = (
            scale_admission["scientific_rows"] == scientific_rows and
            cert.get("scientific_rows") == scientific_rows)
        checks["row_derivation_matches_cert"] = (
            scale_admission["row_derivation"] == cert.get("row_derivation"))
        checks["budgets_are_pre_registered"] = cert.get("budgets") == SLIM_BUDGETS
        # (i) row derivation IS the reopened signed input at the target scale.
        try:
            _require_row_derivation_matches(
                X, cert.get("row_derivation") or {}, scientific_rows=scientific_rows)
            checks["row_derivation_is_reopened_input"] = True
        except Exception as exc:  # noqa: BLE001 - fail closed
            checks["row_derivation_is_reopened_input"] = False
            error = error or str(exc)
        # (i) the measured pass came in <= every pre-registered budget.
        _reopen_measurement(
            cert.get("measurement") or {}, budgets=SLIM_BUDGETS, checks=checks)
        # (ii) the >=4x synthetic regression, via the existing machinery.
        try:
            regression_path = prompt_contract.verify_signature(
                cert.get("regression"), label="slim >=4x regression report")
            with open(regression_path, encoding="utf-8") as handle:
                regression = json.load(handle)
            checks["regression_digest_bound"] = True
            checks["regression_certificate_passes"] = bool(
                validate_regression_certificate(regression).get("passed"))
        except Exception as exc:  # noqa: BLE001 - fail closed
            checks["regression_digest_bound"] = False
            checks["regression_certificate_passes"] = False
            error = error or str(exc)
        # (iii) release-preflight binding, via the existing production validator.
        try:
            release_binding = cert.get("release_preflight")
            if (not isinstance(release_binding, Mapping) or
                    set(release_binding) != {"receipt", "identity_sha256"}):
                raise SlimScaleAdmissionError(
                    "slim cert release_preflight binding fields are not exact")
            receipt_signature = release_binding["receipt"]
            release = validate_release_preflight_receipt(
                receipt_signature["canonical_path"],
                expected_identity_sha256=release_binding["identity_sha256"],
                expected_signature=receipt_signature)
            checks["release_preflight_reopened"] = True
            checks["release_preflight_release_sha"] = (
                release.get("release_sha") == cert.get("release_sha"))
        except Exception as exc:  # noqa: BLE001 - fail closed
            checks["release_preflight_reopened"] = False
            checks["release_preflight_release_sha"] = False
            error = error or str(exc)

    if not all(checks.values()):
        raise SlimScaleAdmissionError(
            f"slim scale-performance admission rejected: checks={checks} error={error}")
    return cert


def assert_not_slim_cert_production_panel(panel: Mapping[str, Any], *, label: str,
                                          error_cls: type[BaseException] = RuntimeError
                                          ) -> None:
    """Consumption-side ENFORCEMENT: refuse a panel that carries the production stamp.

    A production-mode pass exists only to produce perf receipts; its output must never be
    scored as a scientific panel.  Any panel artifact carrying
    ``purpose='slim-scale-cert-production'`` is refused here (raised, not asserted).
    """
    if isinstance(panel, Mapping) and panel.get("purpose") == SLIM_CERT_PRODUCTION_PURPOSE:
        raise error_cls(
            f"{label}: refusing a panel stamped "
            f"purpose={SLIM_CERT_PRODUCTION_PURPOSE!r} (a cert-production perf pass, "
            f"not a scientific panel)")
