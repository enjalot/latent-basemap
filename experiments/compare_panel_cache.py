"""Run real cache-off/cache-on held-out scoring and compare persisted scalars."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (expected_input_signature, ordered_array_sha256,
                                       sha256_bytes, canonical_json)
from basemap.output_safety import (atomic_write_new_json, create_fresh_directory,
                                   require_empty_directory)
from basemap.panel_v2 import PanelV2Config, QueryTruthCache
from experiments.score_complete_panel import (parse_run_pairs, score_query_bundle,
                                               validate_query_truth_consumers)

FIXTURE_SCHEMA = "round0005_scorer_fixture.v1"


def _inherited_lease_pass_fds() -> tuple[int, ...]:
    """Preserve only a valid controller-owned GPU lease descriptor for children."""
    raw = os.environ.get("BASEMAP_GPU_LEASE_FD")
    if raw is None:
        return ()
    try:
        fd = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("BASEMAP_GPU_LEASE_FD is not an integer descriptor") from exc
    if fd < 0:
        raise RuntimeError("BASEMAP_GPU_LEASE_FD is negative")
    try:
        os.fstat(fd)
    except OSError as exc:
        raise RuntimeError("BASEMAP_GPU_LEASE_FD is stale/closed") from exc
    return (fd,)


def complete_panel_child_output_contract(out_root: str, cache_mode: str) -> dict:
    """Enumerate the scorer siblings created inside one private child root."""
    if cache_mode not in {"on", "off"}:
        raise ValueError(f"invalid query-cache mode: {cache_mode}")
    root = os.path.realpath(out_root)
    return {
        "report": os.path.join(root, "report.json"),
        "hiD_reference": os.path.join(root, "hiD-reference.npz"),
        "hiD_reference_receipt": os.path.join(root, "hiD-reference-receipt.json"),
        "child_process_receipt": os.path.join(root, "child-process.json"),
        "query_truth_cache": (os.path.join(root, "query-truth-cache")
                              if cache_mode == "on" else None),
    }


def complete_panel_child_argv(*, python: str, script: str, runs: list[str],
                              testbed: str, query_artifact: str,
                              query_expectation: str, out_root: str,
                              cache_mode: str, dim: int, frac: float,
                              n_anchors: int, seed: int,
                              require_nine_maps: bool) -> list[str]:
    argv = [
        python, script, "--runs", *runs, "--testbed", testbed,
        "--query-artifact", query_artifact,
        "--query-expectation", query_expectation,
        "--query-cache-mode", cache_mode, "--expected-highd-builds", "1",
        "--dim", str(dim), "--frac", str(frac), "--n-anchors", str(n_anchors),
        "--seed", str(seed), "--out-root", out_root,
    ]
    if require_nine_maps:
        argv.append("--require-round0005-nine-maps")
    return argv


def scalar_equivalence_descendant_argvs(parent_argv: list[str], *, repo_root: str) \
        -> list[list[str]]:
    """Derive the only two scorer descendants from the hashed parent argv."""
    def one(flag: str) -> str:
        if parent_argv.count(flag) != 1:
            raise RuntimeError(f"scalar-equivalence parent argv lacks unique {flag}")
        position = parent_argv.index(flag)
        if position + 1 >= len(parent_argv):
            raise RuntimeError(f"scalar-equivalence parent argv has no value for {flag}")
        return parent_argv[position + 1]

    runs_at = parent_argv.index("--runs") + 1
    runs = []
    while runs_at < len(parent_argv) and not parent_argv[runs_at].startswith("--"):
        runs.append(parent_argv[runs_at]); runs_at += 1
    if not runs:
        raise RuntimeError("scalar-equivalence parent argv has no signed map list")
    script = os.path.join(repo_root, "experiments", "score_complete_panel.py")
    common = {
        "python": parent_argv[0], "script": script, "runs": runs,
        "testbed": one("--testbed"), "query_artifact": one("--query-artifact"),
        "query_expectation": one("--query-expectation"),
        "dim": int(one("--dim")), "frac": float(one("--frac")),
        "n_anchors": int(one("--n-anchors")), "seed": int(one("--seed")),
        "require_nine_maps": "--require-round0005-nine-maps" in parent_argv,
    }
    parent_root = one("--out-root")
    return [
        complete_panel_child_argv(
            **common, out_root=os.path.join(parent_root, "uncached"), cache_mode="off"),
        complete_panel_child_argv(
            **common, out_root=os.path.join(parent_root, "cached"), cache_mode="on"),
    ]


def load_fixture(path: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        schema = str(archive["schema"])
        if schema != FIXTURE_SCHEMA:
            raise ValueError(f"scorer fixture schema must be {FIXTURE_SCHEMA}")
        arrays = {name: np.array(archive[name], copy=True) for name in ("X", "Z", "Xq", "Zq")}
        meta = json.loads(str(archive["meta"]))
        payload_sha = str(archive["payload_sha256"])
    observed_sha = sha256_bytes(canonical_json({
        "meta": meta,
        "arrays": {name: ordered_array_sha256(value) for name, value in arrays.items()},
    }))
    if observed_sha != payload_sha:
        raise ValueError("scorer fixture payload SHA-256 mismatch")
    X, Z, Xq, Zq = (arrays[name] for name in ("X", "Z", "Xq", "Zq"))
    if any(value.dtype != np.dtype("float32") or not np.isfinite(value).all()
           for value in (X, Z, Xq, Zq)):
        raise ValueError("scorer fixture arrays must be finite float32")
    if len(X) != len(Z) or len(Xq) != len(Zq) or X.shape[1] != Xq.shape[1] or \
            Z.shape[1] != Zq.shape[1]:
        raise ValueError("scorer fixture shapes do not align")
    return {**arrays, "meta": meta, "payload_sha256": payload_sha,
            "signature": expected_input_signature(path)}


def run_actual_scorer(fixture: dict, *, cache_enabled: bool, cache_dir: str | None,
                      phase_delay_s: float = 0.0, label: str = "fixture-map") -> dict:
    cfg = PanelV2Config(**fixture["meta"]["panel_config"])
    cache = QueryTruthCache(cache_dir=cache_dir, enabled=cache_enabled)
    started = time.monotonic()
    cache.get_or_build(
        fixture["Xq"], fixture["X"], cfg=cfg,
        corpus_identity={
            "ordered_embeddings_sha256": ordered_array_sha256(fixture["X"]),
            "shape": list(fixture["X"].shape), "dtype": fixture["X"].dtype.name,
        },
        query_identity={
            "ordered_embeddings_sha256": ordered_array_sha256(fixture["Xq"]),
            "shape": list(fixture["Xq"].shape), "dtype": fixture["Xq"].dtype.name,
        },
        k=15)
    scalars = score_query_bundle(
        X=fixture["X"], Z=fixture["Z"], Xq=fixture["Xq"], Zq=fixture["Zq"],
        cfg=cfg, truth_cache=cache, label=label,
        random_seed=int(fixture["meta"]["seed"]), phase_delay_s=phase_delay_s)
    return {
        "schema": "round0005_actual_query_score.v1",
        "cache_enabled": bool(cache_enabled),
        "fixture_payload_sha256": fixture["payload_sha256"],
        "persisted_scalars": scalars,
        "query_truth_cache": cache.telemetry(),
        "wall_s": round(time.monotonic() - started, 6),
        "phase_delay_s": float(phase_delay_s),
    }


def _flatten_scalars(value, *, prefix="") -> dict:
    """Flatten persisted scientific scalar leaves without runtime/provenance noise."""
    out = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else key
            out.update(_flatten_scalars(value[key], prefix=child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            out.update(_flatten_scalars(item, prefix=child))
    elif value is None or isinstance(value, (str, bool, int, float)):
        out[prefix] = value
    return out


def extract_persisted_scientific_scalars(report: dict) -> dict:
    """Extract every persisted scientific leaf from a complete scorer report.

    This deliberately includes panel metric metadata, numerators, guard outcomes,
    and projection decisions.  It excludes only runtime/provenance/reference/cache
    telemetry, whose values are expected to differ between cache modes.
    """
    runs = report.get("runs")
    if not isinstance(runs, dict) or not runs:
        scalars = report.get("persisted_scalars")
        if not isinstance(scalars, dict):
            raise ValueError("cache equivalence input is not a complete/fixture scorer report")
        return _flatten_scalars(scalars)
    extracted = {}
    excluded_run_fields = {
        "run_dir", "wall_s", "coordinate_alignment", "hiD_reference_key",
        "hiD_reference_reused", "no_model_reference",
    }
    for label, run in sorted(runs.items()):
        if not isinstance(run, dict):
            raise ValueError(f"scorer run {label} is not an object")
        public = {key: value for key, value in run.items()
                  if key not in excluded_run_fields and key != "panel_full"}
        full = run.get("panel_full")
        if not isinstance(full, dict):
            raise ValueError(f"scorer run {label} has no complete panel payload")
        scientific_full = {key: value for key, value in full.items()
                           if key != "provenance"}
        extracted.update(_flatten_scalars(public, prefix=f"runs.{label}.summary"))
        extracted.update(_flatten_scalars(scientific_full, prefix=f"runs.{label}.panel_full"))
    return extracted


def compare(uncached: dict, cached: dict, *, label: str = "complete-panel") -> dict:
    left = extract_persisted_scientific_scalars(uncached)
    right = extract_persisted_scientific_scalars(cached)
    differences = {}
    for key in sorted(set(left) | set(right)):
        left_present = key in left
        right_present = key in right
        if not left_present or not right_present or left[key] != right[key]:
            differences[key] = {
                "uncached_present": left_present,
                "cached_present": right_present,
                "uncached": left[key] if left_present else None,
                "cached": right[key] if right_present else None,
            }
    telemetry = cached.get("query_truth_cache") or {}
    uncached_telemetry = uncached.get("query_truth_cache") or {}
    run_labels = sorted((cached.get("runs") or {}).keys())
    expected_consumers = (len(run_labels) * 4 if run_labels else 4)
    checks = {
        "persisted_scalars_identical": not differences,
        "uncached_real_build_once": uncached_telemetry.get("build_count") == 1,
        "cached_real_build_once": telemetry.get("build_count") == 1,
        "maximum_k_15": telemetry.get("maximum_k") == 15,
        "cached_truth_shared": telemetry.get("consumer_count") == expected_consumers,
        "same_run_labels": set(uncached.get("runs") or {}) == set(cached.get("runs") or {}),
    }
    if run_labels:
        checks["uncached_consumer_contract"] = validate_query_truth_consumers(
            uncached_telemetry, run_labels, k_hit=10, expected_builds=1)["passed"]
        checks["cached_consumer_contract"] = validate_query_truth_consumers(
            telemetry, run_labels, k_hit=10, expected_builds=1)["passed"]
    return {
        "schema": "round0005_cache_scalar_equivalence.v2",
        "passed": all(checks.values()),
        "label": label,
        "checks": checks,
        "uncached_scalars": left,
        "cached_scalars": right,
        "differences": differences,
        "uncached_query_truth_telemetry": uncached_telemetry,
        "cached_query_truth_telemetry": telemetry,
    }


def _run_complete_panel_child(*, runs: list[str], testbed: str, query_artifact: str,
                              query_expectation: str, out_root: str, cache_mode: str,
                              dim: int, frac: float, n_anchors: int, seed: int,
                              require_nine_maps: bool) -> dict:
    # Validate before launching; ``score_complete_panel`` independently repeats
    # this check so a duplicate label cannot be collapsed in either layer.
    parse_run_pairs(runs)
    argv = complete_panel_child_argv(
        python=sys.executable,
        script=os.path.join(os.path.dirname(__file__), "score_complete_panel.py"),
        runs=runs, testbed=testbed, query_artifact=query_artifact,
        query_expectation=query_expectation, out_root=out_root,
        cache_mode=cache_mode, dim=dim, frac=frac, n_anchors=n_anchors,
        seed=seed, require_nine_maps=require_nine_maps)
    contract = complete_panel_child_output_contract(out_root, cache_mode)
    from basemap.run_controller import run_round0005_admitted_descendant
    pass_fds = _inherited_lease_pass_fds()
    if len(pass_fds) != 1:
        raise RuntimeError("scalar-equivalence child lacks its exact inherited lease")
    proc = run_round0005_admitted_descendant(
        argv, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    atomic_write_new_json(
        contract["child_process_receipt"],
        {"argv": argv, "returncode": proc.returncode,
         "stdout": proc.stdout, "stderr": proc.stderr,
         "inherited_lease_fd": (pass_fds[0] if pass_fds else None),
         "output_contract": contract}, immutable=True)
    if proc.returncode:
        raise RuntimeError(
            f"complete-panel cache-{cache_mode} child failed ({proc.returncode}): "
            f"{proc.stderr[-2000:]}")
    missing = [path for key, path in contract.items()
               if key != "query_truth_cache" and
               not os.path.isfile(path)]
    cache_path = contract["query_truth_cache"]
    if cache_path is not None and not os.path.isdir(cache_path):
        missing.append(cache_path)
    if missing:
        raise RuntimeError(f"complete-panel child omitted declared outputs: {missing}")
    with open(contract["report"], encoding="utf-8") as handle:
        return json.load(handle)


def run_real_input_equivalence(*, runs: list[str], testbed: str, query_artifact: str,
                               query_expectation: str, out_root: str, dim: int = 768,
                               frac: float = 0.001, n_anchors: int = 10_000,
                               seed: int = 123, require_nine_maps: bool = False) -> dict:
    """Execute both modes from original signed inputs; consume no sibling job output."""
    out_root = require_empty_directory(out_root, label="real cache equivalence private parent")
    if not out_root.startswith("/data/"):
        raise ValueError("real cache equivalence output parent must be under /data")
    uncached_root = create_fresh_directory(
        os.path.join(out_root, "uncached"), label="uncached scorer output")
    uncached = _run_complete_panel_child(
        runs=runs, testbed=testbed, query_artifact=query_artifact,
        query_expectation=query_expectation, out_root=uncached_root, cache_mode="off",
        dim=dim, frac=frac, n_anchors=n_anchors, seed=seed,
        require_nine_maps=require_nine_maps)
    cached_root = create_fresh_directory(
        os.path.join(out_root, "cached"), label="cached scorer output")
    cached = _run_complete_panel_child(
        runs=runs, testbed=testbed, query_artifact=query_artifact,
        query_expectation=query_expectation, out_root=cached_root, cache_mode="on",
        dim=dim, frac=frac, n_anchors=n_anchors, seed=seed,
        require_nine_maps=require_nine_maps)
    report = compare(uncached, cached, label="real-input-complete-panel")
    report["schema"] = "round0005_real_input_cache_scalar_equivalence.v3"
    report["self_contained"] = {
        "consumed_sibling_job_outputs": False,
        "uncached_report": expected_input_signature(os.path.join(uncached_root, "report.json")),
        "cached_report": expected_input_signature(os.path.join(cached_root, "report.json")),
        "uncached_private_reference": expected_input_signature(
            os.path.join(uncached_root, "hiD-reference.npz")),
        "cached_private_reference": expected_input_signature(
            os.path.join(cached_root, "hiD-reference.npz")),
        "output_contract": {
            "uncached": complete_panel_child_output_contract(uncached_root, "off"),
            "cached": complete_panel_child_output_contract(cached_root, "on"),
            "equivalence": os.path.join(out_root, "equivalence.json"),
        },
    }
    atomic_write_new_json(os.path.join(out_root, "equivalence.json"), report, immutable=True)
    return report


def run_equivalence(*, fixture_path: str, out_root: str) -> dict:
    fixture = load_fixture(fixture_path)
    out_root = create_fresh_directory(out_root, label="cache equivalence output root")
    uncached = run_actual_scorer(fixture, cache_enabled=False, cache_dir=None)
    cached = run_actual_scorer(
        fixture, cache_enabled=True, cache_dir=os.path.join(out_root, "truth-cache"))
    uncached_path = os.path.join(out_root, "uncached.json")
    cached_path = os.path.join(out_root, "cached.json")
    atomic_write_new_json(uncached_path, uncached, immutable=True)
    atomic_write_new_json(cached_path, cached, immutable=True)
    with open(uncached_path, encoding="utf-8") as handle:
        persisted_uncached = json.load(handle)
    with open(cached_path, encoding="utf-8") as handle:
        persisted_cached = json.load(handle)
    report = compare(persisted_uncached, persisted_cached)
    report["compared_persisted_reports"] = {
        "uncached": expected_input_signature(uncached_path),
        "cached": expected_input_signature(cached_path),
    }
    atomic_write_new_json(os.path.join(out_root, "equivalence.json"), report, immutable=True)
    return report


def main(argv=None) -> int:
    from basemap.run_controller import require_round0005_child_admission
    require_round0005_child_admission("experiments/compare_panel_cache.py")
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture")
    mode.add_argument("--runs", nargs="+")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--testbed")
    parser.add_argument("--query-artifact")
    parser.add_argument("--query-expectation")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--frac", type=float, default=0.001)
    parser.add_argument("--n-anchors", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--require-round0005-nine-maps", action="store_true")
    args = parser.parse_args(argv)
    if args.fixture:
        if any((args.testbed, args.query_artifact, args.query_expectation)):
            parser.error("fixture mode does not accept real scorer inputs")
        report = run_equivalence(fixture_path=args.fixture, out_root=args.out_root)
    else:
        if not all((args.testbed, args.query_artifact, args.query_expectation)):
            parser.error("real-input mode requires --testbed, --query-artifact, and --query-expectation")
        report = run_real_input_equivalence(
            runs=args.runs, testbed=args.testbed, query_artifact=args.query_artifact,
            query_expectation=args.query_expectation, out_root=args.out_root,
            dim=args.dim, frac=args.frac, n_anchors=args.n_anchors, seed=args.seed,
            require_nine_maps=args.require_round0005_nine_maps)
    print(json.dumps({"passed": report["passed"], "checks": report["checks"]}, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
