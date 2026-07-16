"""Queue release and immutable run-checkout preflight."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess

from .artifact_identity import (git_checkout_state, is_ancestor, path_signature,
                                sha256_bytes)

FULL_SHA = re.compile(r"[0-9a-f]{40}")
HASH64 = re.compile(r"[0-9a-f]{64}")
CACHE_KEYS = ("XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "TRITON_CACHE_DIR",
              "PYTHONPYCACHEPREFIX")


def _git(repo: str, *args: str) -> str:
    proc = subprocess.run(["git", "-C", repo, *args], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _canonical_freeze_sha(path: str) -> str:
    lines = sorted(line.strip() for line in open(path, encoding="utf-8") if line.strip())
    return sha256_bytes(("".join(line + "\n" for line in lines)).encode("utf-8"))


def verify_release(*, integration_repo: str, release_sha: str,
                   implementation_commits: list[str], pushed_ref: str,
                   run_checkout: str, environment_manifest: str,
                   cache_environment: dict[str, str]) -> dict:
    errors = []
    if not FULL_SHA.fullmatch(release_sha):
        errors.append("release_sha is not a full lowercase 40-character SHA")
    for commit in implementation_commits:
        if not FULL_SHA.fullmatch(commit):
            errors.append(f"implementation commit is not a full SHA: {commit}")
        elif FULL_SHA.fullmatch(release_sha) and not is_ancestor(integration_repo, commit, release_sha):
            errors.append(f"implementation commit {commit} is not an ancestor of {release_sha}")
    try:
        pushed_tip = _git(integration_repo, "rev-parse", pushed_ref)
        if FULL_SHA.fullmatch(release_sha) and not is_ancestor(integration_repo, release_sha, pushed_tip):
            errors.append(f"release {release_sha} is not reachable from pushed ref {pushed_ref}")
    except Exception as exc:
        pushed_tip = None; errors.append(str(exc))

    try:
        checkout = git_checkout_state(run_checkout)
        if checkout["head"] != release_sha:
            errors.append("run checkout HEAD does not equal release SHA")
        if not checkout["detached"]:
            errors.append("run checkout is not detached")
        if not checkout["clean"]:
            errors.append(f"run checkout is dirty: {checkout['porcelain']}")
    except Exception as exc:
        checkout = None; errors.append(f"run checkout verification failed: {exc}")

    try:
        with open(environment_manifest, encoding="utf-8") as handle:
            environment = json.load(handle)
        freeze_path = os.path.realpath(environment["freeze_file"])
        freeze_sha = _canonical_freeze_sha(freeze_path)
        if freeze_sha != environment.get("freeze_sha256"):
            errors.append(f"environment freeze changed: {freeze_sha} != "
                          f"{environment.get('freeze_sha256')}")
        if not HASH64.fullmatch(str(environment.get("identity_sha256", ""))):
            errors.append("environment identity_sha256 is missing or malformed")
        venv = os.path.realpath(environment["venv_path"])
        if not os.path.isdir(venv):
            errors.append(f"resolved execution venv is missing: {venv}")
    except Exception as exc:
        environment = None; freeze_sha = None; venv = None
        errors.append(f"environment manifest verification failed: {exc}")

    if str(cache_environment.get("PYTHONDONTWRITEBYTECODE")) != "1":
        errors.append("PYTHONDONTWRITEBYTECODE must equal 1")
    for key in CACHE_KEYS:
        value = cache_environment.get(key)
        if not value or not os.path.realpath(value).startswith("/data/"):
            errors.append(f"{key} must resolve under /data")

    return {
        "schema": "queue_release_preflight.v1",
        "passed": not errors,
        "errors": errors,
        "release_sha": release_sha,
        "implementation_commits": implementation_commits,
        "pushed_ref": pushed_ref,
        "pushed_ref_tip": pushed_tip,
        "run_checkout": checkout,
        "environment_manifest": (path_signature(environment_manifest)
                                 if os.path.isfile(environment_manifest) else None),
        "environment_freeze_sha": freeze_sha,
        "environment_identity_sha": (environment.get("identity_sha256")
                                     if environment else None),
        "resolved_venv": venv,
        "cache_environment": cache_environment,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration-repo", required=True)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--implementation-commit", action="append", required=True)
    parser.add_argument("--pushed-ref", required=True)
    parser.add_argument("--run-checkout", required=True)
    parser.add_argument("--environment-manifest", required=True)
    parser.add_argument("--cache-environment", required=True,
                        help="JSON file containing the cache environment")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    with open(args.cache_environment, encoding="utf-8") as handle:
        cache = json.load(handle)
    report = verify_release(integration_repo=args.integration_repo, release_sha=args.release_sha,
                            implementation_commits=args.implementation_commit,
                            pushed_ref=args.pushed_ref, run_checkout=args.run_checkout,
                            environment_manifest=args.environment_manifest,
                            cache_environment=cache)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"passed": report["passed"], "errors": report["errors"]}, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
