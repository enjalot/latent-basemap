"""Fail-closed experiment-design and producer/consumer contract validation."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .artifact_identity import path_signature, sha256_bytes, canonical_json

SCHEMA = "basemap_experiment_contract.v1"


def _load_structured(path: str):
    with open(path, encoding="utf-8") as handle:
        if path.endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(handle)
        return json.load(handle)


def _flatten(value, prefix="") -> dict:
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(child, name))
        return out
    if isinstance(value, list):
        out = {}
        for index, child in enumerate(value):
            name = f"{prefix}[{index}]"
            out.update(_flatten(child, name))
        return out
    return {prefix: value}


def config_differences(left: dict, right: dict) -> set[str]:
    a, b = _flatten(left), _flatten(right)
    missing = object()
    return {key for key in set(a) | set(b) if a.get(key, missing) != b.get(key, missing)}


def _load_ids(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        values = np.load(path, allow_pickle=False)
    elif path.endswith(".npz"):
        payload = np.load(path, allow_pickle=False)
        keys = list(payload.keys())
        if len(keys) != 1:
            raise ValueError(f"ID npz must contain exactly one array: {path}: {keys}")
        values = payload[keys[0]]
    else:
        payload = _load_structured(path)
        values = payload.get("ids") if isinstance(payload, dict) else payload
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.integer) or values.ndim != 1:
        raise ValueError(f"IDs must be a one-dimensional integer array: {path}")
    values = values.astype(np.int64, copy=False)
    if values.size and values.min() < 0:
        raise ValueError(f"negative ID in {path}")
    if len(np.unique(values)) != len(values):
        raise ValueError(f"duplicate IDs in {path}")
    return values


def ids_identity(values: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return sha256_bytes(canonical_json({"dtype": "int64", "length": len(values)}) +
                        values.tobytes())


def _path(base: str, value: str) -> str:
    return value if os.path.isabs(value) else os.path.realpath(os.path.join(base, value))


def validate_contract(contract: dict, *, contract_path: str = ".") -> dict:
    """Validate a contract and return a content-bound, readable report.

    No CUDA libraries are imported here.  The caller can therefore run this at
    queue admission before any model or scorer allocation.
    """
    if contract.get("schema") != SCHEMA:
        raise ValueError(f"contract schema must be {SCHEMA!r}")
    base = os.path.dirname(os.path.realpath(contract_path))
    errors, checks = [], {}

    jobs = contract.get("jobs") or []
    artifacts = contract.get("artifacts") or []
    job_by_id = {job.get("id"): job for job in jobs}
    artifact_by_id = {artifact.get("id"): artifact for artifact in artifacts}
    if len(job_by_id) != len(jobs) or None in job_by_id:
        errors.append("job IDs must be present and unique")
    if len(artifact_by_id) != len(artifacts) or None in artifact_by_id:
        errors.append("artifact IDs must be present and unique")

    produced_by, consumed_by = {}, {key: [] for key in artifact_by_id}
    for job in jobs:
        for artifact_id in job.get("produces", []):
            if artifact_id not in artifact_by_id:
                errors.append(f"job {job.get('id')} produces undeclared artifact {artifact_id}")
            elif artifact_id in produced_by:
                errors.append(f"artifact {artifact_id} has multiple producers")
            else:
                produced_by[artifact_id] = job.get("id")
        for artifact_id in job.get("consumes", []):
            if artifact_id not in artifact_by_id:
                errors.append(f"job {job.get('id')} consumes undeclared artifact {artifact_id}")
            else:
                consumed_by[artifact_id].append(job.get("id"))

    artifact_sigs = {}
    for artifact_id, artifact in artifact_by_id.items():
        producer = artifact.get("producer")
        consumers = artifact.get("consumers") or []
        if producer and produced_by.get(artifact_id) != producer:
            errors.append(f"artifact {artifact_id} producer edge mismatch")
        if sorted(consumers) != sorted(consumed_by.get(artifact_id, [])):
            errors.append(f"artifact {artifact_id} consumer edge mismatch: declared={consumers} "
                          f"actual={consumed_by.get(artifact_id, [])}")
        if artifact.get("scientific", True) and not artifact.get("terminal") and not consumers:
            errors.append(f"produced-but-unconsumed scientific artifact {artifact_id}")
        if artifact.get("kind") == "query" and not consumers:
            errors.append(f"produced-but-unconsumed query artifact {artifact_id}")
        if artifact.get("path"):
            try:
                artifact_sigs[artifact_id] = path_signature(_path(base, artifact["path"]))
            except Exception as exc:
                errors.append(f"artifact {artifact_id} identity failed: {exc}")
    checks["producer_consumer_paths"] = not any("artifact" in error or "produces" in error or
                                                 "consumes" in error for error in errors)

    arms = {arm.get("id"): arm for arm in contract.get("arms", [])}
    if len(arms) != len(contract.get("arms", [])) or None in arms:
        errors.append("arm IDs must be present and unique")
    configs = {}
    arm_ids = {}
    for arm_id, arm in arms.items():
        try:
            configs[arm_id] = _load_structured(_path(base, arm["config"]))
            arm_ids[arm_id] = _load_ids(_path(base, arm["row_ids"]))
        except Exception as exc:
            errors.append(f"arm {arm_id} load failed: {exc}")

    for comparison in contract.get("comparisons", []):
        ids = comparison.get("arms") or []
        missing = [arm_id for arm_id in ids if arm_id not in arms]
        if missing:
            errors.append(f"comparison {comparison.get('id')} has unknown arms {missing}")
            continue
        allowed = set(comparison.get("allowed_config_differences") or [])
        first = ids[0]
        for other in ids[1:]:
            if first in configs and other in configs:
                diff = config_differences(configs[first], configs[other])
                extra = sorted(diff - allowed)
                if extra:
                    errors.append(f"comparison {comparison.get('id')} config diff outside allow-list: "
                                  f"{first} vs {other}: {extra}")
            for field in comparison.get("equal_arm_fields", ["seed", "updates"]):
                if arms[first].get(field) != arms[other].get(field):
                    errors.append(f"comparison {comparison.get('id')} {field} parity failed: "
                                  f"{first}={arms[first].get(field)!r}, "
                                  f"{other}={arms[other].get(field)!r}")
            if comparison.get("same_ordered_rows", True) and first in arm_ids and other in arm_ids:
                if not np.array_equal(arm_ids[first], arm_ids[other]):
                    errors.append(f"comparison {comparison.get('id')} row correspondence failed: "
                                  f"{first} != {other}")
        # A treatment must actually differ in at least one declared field.
        if len(ids) > 1 and first in configs:
            observed = set()
            for other in ids[1:]:
                if other in configs:
                    observed |= config_differences(configs[first], configs[other])
            required = set(comparison.get("required_config_differences") or [])
            if not required.issubset(observed):
                errors.append(f"comparison {comparison.get('id')} missing required treatment "
                              f"differences {sorted(required - observed)}")

    checks["config_seed_update_row_parity"] = not any(
        token in error for error in errors for token in ("comparison", "arm "))

    correspondence = contract.get("teacher_student_correspondence")
    if correspondence:
        try:
            teacher = _load_ids(_path(base, correspondence["teacher_ids"]))
            student = _load_ids(_path(base, correspondence["student_old_ids"]))
            if not np.array_equal(teacher, student):
                errors.append("teacher/student ordered row correspondence failed")
        except Exception as exc:
            errors.append(f"teacher/student correspondence failed: {exc}")
    checks["teacher_student_correspondence"] = not any("teacher/student" in e for e in errors)

    cohorts = {}
    for cohort in contract.get("cohorts", []):
        try:
            cohorts[cohort["id"]] = _load_ids(_path(base, cohort["ids"]))
        except Exception as exc:
            errors.append(f"cohort {cohort.get('id')} failed: {exc}")
    cohort_names = sorted(cohorts)
    for index, left in enumerate(cohort_names):
        for right in cohort_names[index + 1:]:
            overlap = np.intersect1d(cohorts[left], cohorts[right], assume_unique=True)
            if len(overlap):
                errors.append(f"cohort overlap {left}/{right}: {len(overlap)} IDs")
    if contract.get("cohort_universe"):
        try:
            universe = _load_ids(_path(base, contract["cohort_universe"]))
            union = np.sort(np.concatenate(list(cohorts.values()))) if cohorts else np.array([], np.int64)
            if not np.array_equal(np.sort(universe), union):
                errors.append("cohorts are not disjoint/exhaustive for the declared universe")
        except Exception as exc:
            errors.append(f"cohort universe failed: {exc}")
    checks["cohorts_disjoint_exhaustive"] = not any("cohort" in e for e in errors)

    landmarks = contract.get("landmarks")
    if landmarks:
        try:
            active = _load_ids(_path(base, landmarks["ids"]))
            old = _load_ids(_path(base, landmarks["old_ids"]))
            exact = int(landmarks["exact_count"])
            dose = float(landmarks["dose_of_old"])
            if len(active) != exact:
                errors.append(f"landmark count {len(active)} != {exact}")
            if not np.isclose(len(active) / len(old), dose, rtol=0, atol=1e-12):
                errors.append(f"landmark dose {len(active) / len(old)} != {dose}")
            if not set(active.tolist()).issubset(set(old.tolist())):
                errors.append("landmarks are not a subset of old IDs")
        except Exception as exc:
            errors.append(f"landmark validation failed: {exc}")
    checks["landmark_dose"] = not any("landmark" in e for e in errors)

    for convention in contract.get("convention_checks", []):
        artifact = artifact_by_id.get(convention.get("artifact"))
        if artifact is None:
            errors.append(f"convention check references unknown artifact {convention.get('artifact')}")
            continue
        metadata = artifact.get("metadata") or {}
        for field, expected in (convention.get("equals") or {}).items():
            if metadata.get(field) != expected:
                errors.append(f"artifact {artifact['id']} convention {field}={metadata.get(field)!r} "
                              f"!= {expected!r}")
    checks["embedding_query_conventions"] = not any("convention" in e for e in errors)

    return {
        "schema": "basemap_experiment_contract_report.v1",
        "experiment": contract.get("experiment"),
        "passed": not errors,
        "checks": checks,
        "errors": errors,
        "artifact_signatures": artifact_sigs,
        "row_identities": {arm_id: ids_identity(values) for arm_id, values in arm_ids.items()},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate a basemap experiment contract")
    parser.add_argument("contract")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    contract = _load_structured(args.contract)
    report = validate_contract(contract, contract_path=args.contract)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"passed": report["passed"], "errors": report["errors"]}, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
