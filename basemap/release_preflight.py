"""Replayable release preflight for the immutable Round 0005 queue.

The queue never treats a clean detached checkout as proof that a release was
published.  A preflight receipt records the exact remote-tracking ref tip and
all implementation commits, and every admission boundary recomputes the report
from Git and the sealed environment.  Any changed ref tip, checkout, freeze,
venv executable, or receipt byte therefore closes the gate.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

from .artifact_identity import (canonical_json, expected_input_signature,
                                git_checkout_state, path_signature, sha256_bytes)
from .output_safety import atomic_write_new_json

FULL_SHA = re.compile(r"[0-9a-f]{40}")
HASH64 = re.compile(r"[0-9a-f]{64}")
CACHE_KEYS = ("XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "TRITON_CACHE_DIR",
              "PYTHONPYCACHEPREFIX", "NUMBA_CACHE_DIR", "MPLCONFIGDIR")
RELEASE_PREFLIGHT_SCHEMA = "round0005_release_preflight.v3"
REMOTE_OBSERVATION_SCHEMA = "round0005_remote_observation.v1"
GIT_EXECUTABLE = "/usr/bin/git"
REMOTE_OBSERVATION_MAX_AGE_S = 24 * 60 * 60


def _git_environment() -> dict[str, str]:
    return {
        "HOME": "/home/enjalot", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
    }


def _git(repo: str, *args: str) -> str:
    proc = subprocess.run([GIT_EXECUTABLE, "-C", repo, *args], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          env=_git_environment(), close_fds=True)
    if proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _is_ancestor(repo: str, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        [GIT_EXECUTABLE, "-C", repo, "merge-base", "--is-ancestor",
         ancestor, descendant], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, env=_git_environment(), close_fds=True).returncode == 0


def _remote_parts(pushed_ref: str) -> tuple[str, str]:
    match = re.fullmatch(r"refs/remotes/([^/]+)/(.+)", str(pushed_ref))
    if not match or match.group(2).startswith("/") or ".." in match.group(2).split("/"):
        raise ValueError(
            "pushed_ref must be an exact refs/remotes/<remote>/<branch> ref")
    return match.group(1), f"refs/heads/{match.group(2)}"


class _ProductionRemoteObserver:
    """Non-configurable production observer for one exact server ref."""

    fixture_only = False

    def observe(self, *, repo: str, pushed_ref: str) -> dict[str, Any]:
        remote_name, server_ref = _remote_parts(pushed_ref)
        remote_url = _git(repo, "remote", "get-url", remote_name)
        if not remote_url or "\n" in remote_url:
            raise RuntimeError("configured release remote URL is absent or ambiguous")
        proc = subprocess.run(
            [GIT_EXECUTABLE, "-C", repo, "ls-remote", "--refs", remote_url,
             server_ref], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=_git_environment(), close_fds=True, timeout=35)
        if proc.returncode:
            raise RuntimeError(
                f"remote observation failed closed for {remote_name}: {proc.stderr.strip()}")
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise RuntimeError(
                f"remote observation is absent/ambiguous for {server_ref}: {lines!r}")
        fields = lines[0].split("\t")
        if len(fields) != 2 or fields[1] != server_ref or not FULL_SHA.fullmatch(fields[0]):
            raise RuntimeError(f"remote observation response is malformed: {lines[0]!r}")
        if _git(repo, "remote", "get-url", remote_name) != remote_url:
            raise RuntimeError("configured release remote URL changed during observation")
        body = {
            "schema": REMOTE_OBSERVATION_SCHEMA,
            "observer": "production-git-ls-remote",
            "git_executable": expected_input_signature(GIT_EXECUTABLE),
            "integration_repo": os.path.realpath(repo),
            "remote_name": remote_name,
            "remote_url": remote_url,
            "tracking_ref": pushed_ref,
            "server_ref": server_ref,
            "server_tip": fields[0],
            "response_line": lines[0],
            "observed_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        }
        return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


_FIXTURE_REMOTE_OBSERVER_CAPABILITY = object()


class _FixtureRemoteObserver:
    """Explicit local-ref observer for isolated tests; never production-valid."""

    fixture_only = True

    def __init__(self, *, capability):
        if capability is not _FIXTURE_REMOTE_OBSERVER_CAPABILITY:
            raise RuntimeError("fixture remote observer capability is invalid")

    def observe(self, *, repo: str, pushed_ref: str) -> dict[str, Any]:
        remote_name, server_ref = _remote_parts(pushed_ref)
        tip = _git(repo, "rev-parse", "--verify", pushed_ref)
        body = {
            "schema": REMOTE_OBSERVATION_SCHEMA,
            "observer": "fixture-local-ref-only",
            "git_executable": expected_input_signature(GIT_EXECUTABLE),
            "integration_repo": os.path.realpath(repo),
            "remote_name": remote_name, "remote_url": "fixture-only://local-ref",
            "tracking_ref": pushed_ref, "server_ref": server_ref,
            "server_tip": tip, "response_line": f"{tip}\t{server_ref}",
            "observed_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        }
        return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _new_fixture_remote_observer() -> _FixtureRemoteObserver:
    return _FixtureRemoteObserver(capability=_FIXTURE_REMOTE_OBSERVER_CAPABILITY)


def _canonical_freeze_sha(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        lines = sorted(line.strip() for line in handle if line.strip())
    return sha256_bytes(("".join(line + "\n" for line in lines)).encode("utf-8"))


def _identity_body(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report[key] for key in sorted(report) if key != "identity_sha256"}


def _canonical_existing_path(path: str, *, label: str, directory: bool = False) -> str:
    if not isinstance(path, str) or not os.path.isabs(path):
        raise ValueError(f"{label} must be an absolute path")
    absolute = os.path.abspath(path)
    canonical = os.path.realpath(absolute)
    if canonical != absolute:
        raise ValueError(f"{label} must not traverse a symlink: {path}")
    predicate = os.path.isdir if directory else os.path.isfile
    if not predicate(canonical):
        raise FileNotFoundError(f"{label} is missing: {canonical}")
    return canonical


def _verify_release_with_observer(*, integration_repo: str, release_sha: str,
                                  implementation_commits: list[str], pushed_ref: str,
                                  run_checkout: str, environment_manifest: str,
                                  cache_environment: dict[str, str], observer) -> dict[str, Any]:
    """Recompute every fact needed to call a release pushed and executable.

    The server observation carries its own authenticated observation time.  Live
    replays compare every stable remote/release fact while the original time and
    report identity remain externally sealed by the queue manifest.
    ``pushed_ref`` must be an explicit remote-tracking ref, not a local branch
    that could make an unpushed commit appear released.
    """
    errors: list[str] = []
    integration = os.path.realpath(integration_repo)
    checkout_root = os.path.realpath(run_checkout)
    env_path = os.path.realpath(environment_manifest)
    commits = list(implementation_commits)

    if not FULL_SHA.fullmatch(str(release_sha)):
        errors.append("release_sha is not a full lowercase 40-character SHA")
    if not commits:
        errors.append("at least one implementation commit is required")
    if len(commits) != len(set(commits)):
        errors.append("implementation commits must be unique and ordered")
    for commit in commits:
        if not FULL_SHA.fullmatch(str(commit)):
            errors.append(f"implementation commit is not a full SHA: {commit}")
        elif FULL_SHA.fullmatch(str(release_sha)):
            try:
                if not _is_ancestor(integration, commit, release_sha):
                    errors.append(
                        f"implementation commit {commit} is not an ancestor of {release_sha}")
            except Exception as exc:
                errors.append(f"implementation ancestry check failed for {commit}: {exc}")

    remote_ref = False
    remote_name = server_ref = remote_url = None
    observation = None
    try:
        remote_name, server_ref = _remote_parts(pushed_ref)
        remote_ref = True
        observation = observer.observe(repo=integration, pushed_ref=pushed_ref)
        remote_url = observation["remote_url"]
        if observer.fixture_only is False and observation.get("observer") != \
                "production-git-ls-remote":
            raise RuntimeError("production release path received a fixture observation")
    except Exception as exc:
        errors.append(f"server remote observation failed: {exc}")
    try:
        pushed_tip = _git(integration, "rev-parse", "--verify", pushed_ref)
        if not FULL_SHA.fullmatch(pushed_tip):
            errors.append("pushed ref did not resolve to one full commit SHA")
        elif observation is not None and pushed_tip != observation["server_tip"]:
            errors.append(
                f"local tracking ref {pushed_ref} differs from observed server tip")
        if (observation is not None and FULL_SHA.fullmatch(str(release_sha)) and
                not _is_ancestor(integration, release_sha, observation["server_tip"])):
            errors.append(
                f"release {release_sha} is not reachable from pushed ref server evidence "
                f"{server_ref}")
    except Exception as exc:
        pushed_tip = None
        errors.append(str(exc))

    try:
        integration_head = _git(integration, "rev-parse", "HEAD")
        integration_git_dir = _git(integration, "rev-parse", "--absolute-git-dir")
    except Exception as exc:
        integration_head = None
        integration_git_dir = None
        errors.append(f"integration repository verification failed: {exc}")

    try:
        checkout = git_checkout_state(checkout_root)
        if checkout["head"] != release_sha:
            errors.append("run checkout HEAD does not equal release SHA")
        if not checkout["detached"]:
            errors.append("run checkout is not detached")
        if not checkout["clean"]:
            errors.append(f"run checkout is dirty: {checkout['porcelain']}")
    except Exception as exc:
        checkout = None
        errors.append(f"run checkout verification failed: {exc}")

    environment = None
    freeze_sha = None
    freeze_signature = None
    venv = None
    python_entry_signature = None
    python_signature = None
    environment_signature = None
    try:
        env_path = _canonical_existing_path(
            environment_manifest, label="environment manifest")
        with open(env_path, encoding="utf-8") as handle:
            environment = json.load(handle)
        if not isinstance(environment, dict):
            raise ValueError("environment manifest must be a JSON object")
        freeze_path = _canonical_existing_path(
            environment["freeze_file"], label="environment freeze")
        freeze_sha = _canonical_freeze_sha(freeze_path)
        if freeze_sha != environment.get("freeze_sha256"):
            errors.append(f"environment freeze changed: {freeze_sha} != "
                          f"{environment.get('freeze_sha256')}")
        if not HASH64.fullmatch(str(environment.get("identity_sha256", ""))):
            errors.append("environment identity_sha256 is missing or malformed")
        venv = _canonical_existing_path(
            environment["venv_path"], label="execution venv", directory=True)
        python_entry = os.path.join(venv, "bin", "python")
        if not os.path.lexists(python_entry):
            raise FileNotFoundError(f"execution venv Python is missing: {python_entry}")
        python_entry_signature = path_signature(python_entry)
        python = _canonical_existing_path(
            os.path.realpath(python_entry), label="resolved execution venv Python")
        environment_signature = path_signature(env_path)
        freeze_signature = path_signature(freeze_path)
        python_signature = path_signature(python)
    except Exception as exc:
        errors.append(f"environment manifest verification failed: {exc}")

    cache = dict(cache_environment) if isinstance(cache_environment, dict) else {}
    expected_cache_fields = {"PYTHONDONTWRITEBYTECODE", *CACHE_KEYS}
    if set(cache) != expected_cache_fields:
        errors.append(
            f"cache environment fields mismatch: expected={sorted(expected_cache_fields)} "
            f"observed={sorted(cache)}")
    if str(cache.get("PYTHONDONTWRITEBYTECODE")) != "1":
        errors.append("PYTHONDONTWRITEBYTECODE must equal 1")
    for key in CACHE_KEYS:
        value = cache.get(key)
        if (not isinstance(value, str) or not os.path.isabs(value) or
                os.path.commonpath(["/data", os.path.abspath(value)]) != "/data"):
            errors.append(f"{key} must be an absolute path contained by /data")

    report: dict[str, Any] = {
        "schema": RELEASE_PREFLIGHT_SCHEMA,
        "passed": not errors,
        "errors": errors,
        "integration_repo": integration,
        "integration_git_dir": integration_git_dir,
        "integration_head": integration_head,
        "release_sha": release_sha,
        "implementation_commits": commits,
        "pushed_ref": pushed_ref,
        "pushed_ref_tip": pushed_tip,
        "pushed_ref_is_remote_tracking": bool(remote_ref),
        "remote_name": remote_name,
        "remote_url": remote_url,
        "server_ref": server_ref,
        "server_tip": observation.get("server_tip") if observation else None,
        "remote_observation": observation,
        "run_checkout_path": checkout_root,
        "run_checkout": checkout,
        "environment_manifest_path": env_path,
        "environment_manifest": environment_signature,
        "environment_freeze": freeze_signature,
        "environment_freeze_sha": freeze_sha,
        "environment_identity_sha": (environment.get("identity_sha256")
                                     if isinstance(environment, dict) else None),
        "resolved_venv": venv,
        "python_invocation_path": (python_entry if python_entry_signature is not None else None),
        "venv_python_entry": python_entry_signature,
        "python_executable": python_signature,
        "cache_environment": cache,
    }
    report["identity_sha256"] = sha256_bytes(canonical_json(_identity_body(report)))
    return report


def verify_release(*, integration_repo: str, release_sha: str,
                   implementation_commits: list[str], pushed_ref: str,
                   run_checkout: str, environment_manifest: str,
                   cache_environment: dict[str, str]) -> dict[str, Any]:
    """Production release verification with a mandatory live server observation."""
    return _verify_release_with_observer(
        integration_repo=integration_repo, release_sha=release_sha,
        implementation_commits=implementation_commits, pushed_ref=pushed_ref,
        run_checkout=run_checkout, environment_manifest=environment_manifest,
        cache_environment=cache_environment, observer=_ProductionRemoteObserver())


def _verify_release_fixture_only(**kwargs) -> dict[str, Any]:
    return _verify_release_with_observer(
        **kwargs, observer=_new_fixture_remote_observer())


def issue_release_preflight_receipt(path: str, **kwargs) -> dict[str, Any]:
    """Verify and atomically publish one immutable replayable receipt."""
    report = verify_release(**kwargs)
    if not report["passed"]:
        raise RuntimeError("release preflight rejected: " + "; ".join(report["errors"]))
    atomic_write_new_json(path, report, immutable=True)
    return report


def _issue_release_preflight_receipt_fixture_only(
        path: str, **kwargs) -> dict[str, Any]:
    """Publish explicit local-ref-only evidence for isolated fixture tests."""
    report = _verify_release_fixture_only(**kwargs)
    if not report["passed"]:
        raise RuntimeError("fixture release preflight rejected: " +
                           "; ".join(report["errors"]))
    atomic_write_new_json(path, report, immutable=True)
    return report


def _validate_remote_observation(value: dict[str, Any], *, fixture_only: bool) -> None:
    fields = {
        "schema", "observer", "git_executable", "integration_repo", "remote_name",
        "remote_url", "tracking_ref", "server_ref", "server_tip", "response_line",
        "observed_at", "identity_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise RuntimeError("release remote observation fields are incomplete")
    body = {key: value[key] for key in value if key != "identity_sha256"}
    expected_observer = ("fixture-local-ref-only" if fixture_only else
                         "production-git-ls-remote")
    try:
        observed_at = datetime.fromisoformat(value["observed_at"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("release remote observation time is invalid") from exc
    age = time.time() - observed_at.timestamp()
    if (observed_at.tzinfo is None or age < -60 or
            age > REMOTE_OBSERVATION_MAX_AGE_S):
        raise RuntimeError("release remote observation is stale or from the future")
    if (value["schema"] != REMOTE_OBSERVATION_SCHEMA or
            value["observer"] != expected_observer or
            expected_input_signature(GIT_EXECUTABLE) != value["git_executable"] or
            not FULL_SHA.fullmatch(str(value["server_tip"])) or
            value["response_line"] !=
            f"{value['server_tip']}\t{value['server_ref']}" or
            value["identity_sha256"] != sha256_bytes(canonical_json(body))):
        raise RuntimeError("release remote observation identity is forged")


def _stable_release_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items()
            if key not in {"identity_sha256", "remote_observation"}}


def release_reports_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare two fresh observations while ignoring only observation time/hash."""
    if _stable_release_report(left) != _stable_release_report(right):
        return False
    left_observation = dict(left.get("remote_observation") or {})
    right_observation = dict(right.get("remote_observation") or {})
    for value in (left_observation, right_observation):
        value.pop("observed_at", None)
        value.pop("identity_sha256", None)
    return left_observation == right_observation


def _validate_release_preflight_receipt(
        path: str, *, fixture_only: bool,
        expected_identity_sha256: str | None = None,
        expected_signature: dict[str, Any] | None = None) -> dict[str, Any]:
    """Reopen a receipt, validate its external binding, and replay live state.

    A receipt's own content hash cannot authenticate a same-owner rewrite: an
    attacker could change a field and recompute that hash.  Production callers
    therefore must supply both values sealed into the gate-hashed queue
    manifest: the report identity and the complete no-follow file signature.
    The explicit fixture-only lane remains usable without that production trust
    root for isolated release-observer tests.
    """
    canonical = _canonical_existing_path(path, label="release preflight receipt")
    stat_result = os.stat(canonical, follow_symlinks=False)
    if stat_result.st_nlink != 1 or stat_result.st_mode & 0o222:
        raise RuntimeError("release preflight receipt changed after publication")
    if not fixture_only:
        if (not isinstance(expected_identity_sha256, str) or
                not HASH64.fullmatch(expected_identity_sha256) or
                not isinstance(expected_signature, dict)):
            raise RuntimeError(
                "production release validation requires the externally sealed "
                "receipt identity and file signature")
        observed_signature = expected_input_signature(canonical)
        if (expected_signature.get("canonical_path") != canonical or
                observed_signature != expected_signature):
            raise RuntimeError(
                "release preflight receipt differs from its gate-hashed signature")
    with open(canonical, encoding="utf-8") as handle:
        recorded = json.load(handle)
    if not isinstance(recorded, dict) or recorded.get("schema") != RELEASE_PREFLIGHT_SCHEMA:
        raise ValueError("release preflight receipt schema is missing or unsupported")
    identity = recorded.get("identity_sha256")
    if (not isinstance(identity, str) or not HASH64.fullmatch(identity) or
            sha256_bytes(canonical_json(_identity_body(recorded))) != identity):
        raise ValueError("release preflight receipt identity is invalid")
    if not fixture_only and identity != expected_identity_sha256:
        raise RuntimeError(
            "release preflight receipt differs from its gate-hashed identity")
    if recorded.get("passed") is not True or recorded.get("errors") != []:
        raise RuntimeError("release preflight receipt did not pass")
    _validate_remote_observation(
        recorded.get("remote_observation"), fixture_only=fixture_only)
    verify = _verify_release_fixture_only if fixture_only else verify_release
    current = verify(
        integration_repo=recorded["integration_repo"],
        release_sha=recorded["release_sha"],
        implementation_commits=recorded["implementation_commits"],
        pushed_ref=recorded["pushed_ref"],
        run_checkout=recorded["run_checkout_path"],
        environment_manifest=recorded["environment_manifest_path"],
        cache_environment=recorded["cache_environment"],
    )
    if not release_reports_equivalent(current, recorded):
        raise RuntimeError(
            "release preflight changed after publication: "
            f"expected={recorded!r} observed={current!r}")
    return recorded


def validate_release_preflight_receipt(
        path: str, *, expected_identity_sha256: str,
        expected_signature: dict[str, Any]) -> dict[str, Any]:
    """Reopen externally sealed production evidence and observe the server again."""
    return _validate_release_preflight_receipt(
        path, fixture_only=False,
        expected_identity_sha256=expected_identity_sha256,
        expected_signature=expected_signature)


def _validate_release_preflight_receipt_fixture_only(path: str) -> dict[str, Any]:
    return _validate_release_preflight_receipt(path, fixture_only=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration-repo", required=True)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--implementation-commit", action="append", required=True)
    parser.add_argument("--pushed-ref", required=True)
    parser.add_argument("--run-checkout", required=True)
    parser.add_argument("--environment-manifest", required=True)
    parser.add_argument("--cache-environment", required=True,
                        help="JSON file containing the exact cache environment")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    with open(args.cache_environment, encoding="utf-8") as handle:
        cache = json.load(handle)
    report = issue_release_preflight_receipt(
        args.out, integration_repo=args.integration_repo, release_sha=args.release_sha,
        implementation_commits=args.implementation_commit, pushed_ref=args.pushed_ref,
        run_checkout=args.run_checkout, environment_manifest=args.environment_manifest,
        cache_environment=cache)
    print(json.dumps({"passed": report["passed"],
                      "identity_sha256": report["identity_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
