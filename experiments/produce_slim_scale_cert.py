#!/usr/bin/env python3
"""Produce (but do not consume) the slim >=8M scale-performance certificate.

This is the tightly-scoped bootstrap for the slim-runner scale-performance admission
(design `latent-labs/basemap-100m/design-slim-8m-admission-2026-08-17.md`). Given the
round's `release_sha` and the sealed fp32 substrate, it:

  1. opens the substrate memmap and derives `row_derivation`
     (`round0005_performance_gate.derive_scale_rows`);
  2. invokes `score_panel(...)` ONCE in production mode
     (`scale_admission={"kind": "slim-cert-production", ...}`) — the ONLY >=8M pass that
     runs without a finished cert — capturing wall / peak-host-RSS / peak-VRAM. Its output
     is STAMPED `purpose="slim-scale-cert-production"` and is refused by every consuming
     (scientific) node;
  3. runs the existing >=4x synthetic regression (`run_synthetic_regression`);
  4. binds an already-issued release-preflight receipt for this release
     (`validate_release_preflight_receipt`);
  5. assembles + sha256-seals the slim cert and writes everything under the run tree.

The measurement pass (step 2) is a real 50M GPU pass — DO NOT run it from tests. The
assembly logic (`assemble_slim_scale_cert`) is unit-tested with a small (<8M) fixture and
monkeypatched inner validators; the GPU pass is run by the owner after review.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, ensure_data_directory
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.release_preflight import validate_release_preflight_receipt
from basemap.slim_scale_admission import (
    SLIM_BUDGETS,
    SLIM_CERT_PRODUCTION_PURPOSE,
    SLIM_SCALE_CERT_KIND,
    SLIM_SCALE_CERT_SCHEMA,
    SLIM_SCALE_ROWS,
    _reopen_measurement,
    build_slim_scale_admission,
    slim_cert_production_marker,
    validate_slim_scale_admission,
)
from experiments.round0005_performance_gate import (
    derive_scale_rows,
    run_synthetic_regression,
    validate_regression_certificate,
)

#: Where R0267's slim cert artifacts land (bound by prepare_round0267_queue afterward).
DEFAULT_CERT_ROOT = "/data/latent-basemap/runs/round-0267/slim-scale-cert"
DIMENSION = 384


def assemble_slim_scale_cert(
    *,
    release_sha: str,
    scientific_rows: int,
    row_derivation,
    panel_output_path: str,
    regression_path: str,
    release_preflight_receipt_path: str,
    release_preflight_identity_sha256: str,
    out_path: str,
) -> tuple[str, dict]:
    """Assemble + sha256-seal the slim scale-performance certificate.

    Validates each piece of evidence up-front (the measured pass is a production pass within
    every pre-registered budget, the >=4x regression passes, the release-preflight receipt
    reopens and binds this release), then seals and writes the cert. Returns
    ``(out_path, cert)``. Raises on any failing evidence.
    """
    panel_signature = expected_input_signature(panel_output_path)
    regression_signature = expected_input_signature(regression_path)
    release_signature = expected_input_signature(release_preflight_receipt_path)

    # (i) the measured pass IS a production pass within every pre-registered budget.
    measurement_checks: dict[str, bool] = {}
    _reopen_measurement(
        {"panel_output": panel_signature}, budgets=SLIM_BUDGETS, checks=measurement_checks)
    if not all(measurement_checks.values()):
        raise RuntimeError(
            f"slim measurement pass does not satisfy the pre-registered budgets / stamp: "
            f"{measurement_checks}")

    # (ii) the >=4x synthetic regression passes (reused machinery).
    with open(regression_path, encoding="utf-8") as handle:
        regression = json.load(handle)
    regression_validation = validate_regression_certificate(regression)
    if not regression_validation.get("passed"):
        raise RuntimeError(
            f"slim >=4x regression certificate failed: {regression_validation.get('checks')}")

    # (iii) the release-preflight receipt reopens and binds THIS release.
    release = validate_release_preflight_receipt(
        release_signature["canonical_path"],
        expected_identity_sha256=release_preflight_identity_sha256,
        expected_signature=release_signature)
    if release.get("release_sha") != release_sha:
        raise RuntimeError(
            "slim cert release-preflight receipt release_sha differs from the cert release")

    body = {
        "schema": SLIM_SCALE_CERT_SCHEMA,
        "kind": SLIM_SCALE_CERT_KIND,
        "release_sha": str(release_sha),
        "scientific_rows": int(scientific_rows),
        "row_derivation": dict(row_derivation),
        "budgets": dict(SLIM_BUDGETS),
        "measurement": {"panel_output": panel_signature},
        "regression": regression_signature,
        "release_preflight": {
            "receipt": release_signature,
            "identity_sha256": str(release_preflight_identity_sha256),
        },
    }
    cert = prompt_contract.seal(body)
    atomic_write_new_json(out_path, cert, immutable=True)
    return out_path, cert


def _load_reference_and_centroids(panel_evidence_path: str, centroid_ks):
    """Load the R0218 frozen high-D reference + purity centroids (as run_panel does)."""
    from basemap.panel_v2 import load_hiD_reference
    from experiments.round0265_nodes import _load_centroids

    panel_evidence = prompt_contract.read_sealed(
        panel_evidence_path, label="R0218 MiniLM frozen panel reference")
    centroids, centroid_signatures = _load_centroids(
        panel_evidence, [int(k) for k in centroid_ks])
    reference_signature = dict(panel_evidence["shared_high_d_reference"])
    reference_path = prompt_contract.verify_signature(
        reference_signature, label="R0218 shared high-D reference")
    reference = load_hiD_reference(reference_path)
    return panel_evidence, reference, reference_signature, centroids, centroid_signatures


def run_measurement_pass(
    *,
    release_sha: str,
    substrate_path: str,
    coordinates_path: str,
    panel_evidence_path: str,
    ordered_substrate_sha256: str,
    centroid_ks,
    out_dir: str,
) -> tuple[str, dict]:
    """Run ONE >=8M `score_panel` measurement pass in production mode (GPU).

    DO NOT call from tests: this is a real ~50M GPU pass. Returns
    ``(panel_output_path, row_derivation)``. The panel output is stamped
    ``purpose='slim-scale-cert-production'`` and is refused by every scientific consumer.
    """
    import numpy as np
    import experiments.round0218_nodes as round0218_nodes
    from basemap.panel_v2 import reset_process_cuda_peak, score_panel

    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise RuntimeError("slim cert-production measurement pass requires CUDA")

    source = np.load(substrate_path, mmap_mode="r", allow_pickle=False)
    if source.dtype != np.float32 or source.ndim != 2 or int(source.shape[1]) != DIMENSION:
        raise RuntimeError("slim measurement substrate geometry changed")
    rows = int(source.shape[0])
    if rows < SLIM_SCALE_ROWS:
        raise RuntimeError("slim measurement pass is only for >=8,000,000-row substrates")
    coordinates = np.asarray(
        np.load(coordinates_path, allow_pickle=False), dtype=np.float32)
    if coordinates.shape != (rows, 2) or not np.isfinite(coordinates).all():
        raise RuntimeError("slim measurement coordinates are not a finite map of the substrate")

    row_derivation = derive_scale_rows(
        substrate_path, dimensions=DIMENSION, loaded_matrix=source)
    scale_admission = slim_cert_production_marker(
        release_sha=release_sha, row_derivation=row_derivation)

    cfg = prompt_contract.panel_config()
    (_panel_evidence, reference, _reference_signature, centroids,
     _centroid_signatures) = _load_reference_and_centroids(panel_evidence_path, centroid_ks)
    reference_identity = {
        "data_identity": {
            "kind": "ordered_array",
            "shape": [rows, DIMENSION],
            "dtype": np.dtype("<f4").str,
            "sha256": str(ordered_substrate_sha256),
        },
        "convention": dict(round0218_nodes.REFERENCE_CONVENTION),
    }

    reset_process_cuda_peak()
    panel = score_panel(
        source,
        coordinates,
        config=cfg,
        centroids_by_k=centroids,
        hiD_reference=reference,
        reference_identity=reference_identity,
        scale_admission=scale_admission,
        provenance={
            "purpose": SLIM_CERT_PRODUCTION_PURPOSE,
            "release_sha": release_sha,
            "measurement": "slim >=8M scale-performance measurement pass",
        },
    )
    if panel.get("purpose") != SLIM_CERT_PRODUCTION_PURPOSE:
        raise RuntimeError("slim measurement pass was not stamped as a cert-production pass")

    out_dir = ensure_data_directory(out_dir, label="slim scale-cert measurement output")
    panel_output_path = os.path.join(out_dir, "measurement-panel.json")
    atomic_write_new_json(panel_output_path, _json_safe(panel), immutable=True)
    return panel_output_path, row_derivation


def _json_safe(value):
    from basemap.round0238_rung5 import json_safe
    from basemap.round0242_locality import json_scrub

    return json_safe(json_scrub(dict(value)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="produce the slim >=8M scale-performance cert")
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--substrate", required=True,
                        help="the sealed fp32 50M substrate .npy the round scores")
    parser.add_argument("--coordinates", required=True,
                        help="a 50M x 2 map of the substrate to exercise the scorer on")
    parser.add_argument("--panel-evidence", required=True,
                        help="the sealed R0218 MiniLM frozen 2M panel (reference + centroids)")
    parser.add_argument("--ordered-substrate-sha256", required=True,
                        help="the sealed ordered substrate sha256 (from the R0237 manifest)")
    parser.add_argument("--centroid-ks", default="256,1024",
                        help="comma-separated purity centroid granularities (default 256,1024)")
    parser.add_argument("--regression-fixture", required=True,
                        help="the round0005 scorer fixture for the >=4x synthetic regression")
    parser.add_argument("--release-preflight-receipt", required=True,
                        help="an already-issued release-preflight receipt for this release")
    parser.add_argument("--out-root", default=DEFAULT_CERT_ROOT)
    args = parser.parse_args(argv)

    out_root = ensure_data_directory(args.out_root, label="slim scale-cert root")
    centroid_ks = [int(part) for part in args.centroid_ks.split(",") if part.strip()]

    panel_output_path, row_derivation = run_measurement_pass(
        release_sha=args.release_sha,
        substrate_path=os.path.realpath(args.substrate),
        coordinates_path=os.path.realpath(args.coordinates),
        panel_evidence_path=os.path.realpath(args.panel_evidence),
        ordered_substrate_sha256=args.ordered_substrate_sha256,
        centroid_ks=centroid_ks,
        out_dir=os.path.join(out_root, "measurement"),
    )
    regression = run_synthetic_regression(
        fixture_path=os.path.realpath(args.regression_fixture),
        out_root=os.path.join(out_root, "regression"))
    regression_path = os.path.join(out_root, "regression", "regression.json")

    with open(os.path.realpath(args.release_preflight_receipt), encoding="utf-8") as handle:
        receipt_record = json.load(handle)
    release_identity = receipt_record.get("identity_sha256")

    cert_path = os.path.join(out_root, "slim-scale-cert.json")
    scientific_rows = int(row_derivation["scientific_rows"])
    _out, cert = assemble_slim_scale_cert(
        release_sha=args.release_sha,
        scientific_rows=scientific_rows,
        row_derivation=row_derivation,
        panel_output_path=panel_output_path,
        regression_path=regression_path,
        release_preflight_receipt_path=os.path.realpath(args.release_preflight_receipt),
        release_preflight_identity_sha256=release_identity,
        out_path=cert_path,
    )

    # Immediately reopen the sealed cert exactly as the guard will (fail closed here).
    import numpy as np

    source = np.load(os.path.realpath(args.substrate), mmap_mode="r", allow_pickle=False)
    admission = build_slim_scale_admission(
        release_sha=args.release_sha,
        scientific_rows=scientific_rows,
        row_derivation=row_derivation,
        certificate=expected_input_signature(cert_path),
    )
    validate_slim_scale_admission(admission, X=source, scientific_rows=scientific_rows)

    print(json.dumps({
        "purpose": SLIM_CERT_PRODUCTION_PURPOSE,
        "slim_scale_cert": cert_path,
        "identity_sha256": cert["identity_sha256"],
        "scientific_rows": scientific_rows,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
